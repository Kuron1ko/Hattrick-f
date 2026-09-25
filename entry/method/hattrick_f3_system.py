from __future__ import annotations

"""Complete Hattrick-f3 method definition.

Hattrick-f3 starts from a six-objective Hattrick checkpoint and uses a
front-loaded cosine schedule to anneal from the complete six-objective update
to the flow-only ordered update.  All method behavior is local to this module
and :mod:`method.hattrick_system`; no research runner is imported.
"""

import math
from argparse import ArgumentParser, Namespace

import torch

from method import hattrick_system as base
from utils.robust_proj_utils import assign_gradients_and_step


METHOD_NAME = "Hattrick-f3"
FLOW_OBJECTIVE_NAMES = ("Fh", "Fhm", "Fhml")
FLOW_OBJECTIVE_INDICES = (0, 2, 4)
ZERO_TOLERANCE = 1e-12


def register_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--anneal-epochs", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--low-budget", type=float, default=0.03)
    parser.add_argument("--phase-a-epoch", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")


def anneal_alpha(epoch: int, anneal_epochs: int) -> float:
    if anneal_epochs < 2:
        raise ValueError("anneal_epochs must be at least 2")
    if epoch >= anneal_epochs:
        return 0.0
    progress = float(epoch - 1) / float(anneal_epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def ordered_projection_from_raw(
    raw_gradients: list[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    basis: list[torch.Tensor] = []
    projected: list[torch.Tensor] = []
    for raw in raw_gradients:
        work = raw.to(dtype=torch.float64)
        raw_norm = torch.linalg.vector_norm(work)
        if not torch.isfinite(raw_norm).item():
            raise RuntimeError("Non-finite raw Hattrick-f3 gradient")
        if float(raw_norm.item()) <= ZERO_TOLERANCE:
            projected.append(torch.zeros_like(raw))
            continue
        for _ in range(2):
            for basis_vector in basis:
                work = work - torch.dot(work, basis_vector) * basis_vector
        projected_norm = torch.linalg.vector_norm(work)
        if not torch.isfinite(projected_norm).item():
            raise RuntimeError("Non-finite projected Hattrick-f3 gradient")
        if float(projected_norm.item()) <= ZERO_TOLERANCE:
            projected.append(torch.zeros_like(raw))
            continue
        basis.append(work / projected_norm)
        projected.append(work.to(dtype=raw.dtype))
    final = torch.stack(projected, dim=0).sum(dim=0)
    if not torch.isfinite(final).all().item():
        raise RuntimeError("Non-finite flow-only Hattrick-f3 update")
    return projected, final


def gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left.double()) * torch.linalg.vector_norm(
        right.double()
    )
    if float(denominator.item()) <= 1e-20:
        return 0.0
    return float(torch.dot(left.double(), right.double()).item() / denominator.item())


def train_epoch(
    *,
    model,
    props,
    dataset,
    loader,
    optimizer,
    alpha: float,
) -> dict[str, float]:
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    props.research_return_policy = False
    path_masks = base.move_dataset_static(dataset, props.device)
    totals: dict[str, float] = {}
    first_probe: dict[str, float] = {}
    count = 0

    for batch_index, inputs in enumerate(loader):
        values = base.unpack_to_device(inputs, props)
        losses, reported = base.build_objectives(
            model, props, dataset, values, path_masks
        )
        if any(not torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError("Non-finite Hattrick-f3 objective")

        six = base.ordered_project_gradients(model, losses)
        flow_raw = [six.raw_gradients[index] for index in FLOW_OBJECTIVE_INDICES]
        flow_projected, flow_update = ordered_projection_from_raw(flow_raw)
        final_update = float(alpha) * six.final_gradient + (
            1.0 - float(alpha)
        ) * flow_update
        if not torch.isfinite(final_update).all().item():
            raise RuntimeError("Non-finite annealed Hattrick-f3 update")

        if batch_index == 0:
            first_probe = base.projection_diagnostics(six, base.OBJECTIVE_NAMES)
            for name, gradient in zip(FLOW_OBJECTIVE_NAMES, flow_projected):
                first_probe[f"flow_projected_gradient_norm_{name}"] = float(
                    torch.linalg.vector_norm(gradient).item()
                )

        assign_gradients_and_step(
            model,
            final_update,
            optimizer,
            six.parameter_shapes,
        )
        diagnostics = {
            f"reported_{name}": float(value)
            for name, value in zip(base.OBJECTIVE_NAMES, reported)
        }
        diagnostics.update(
            {
                "anneal_alpha": float(alpha),
                "six_update_norm": float(
                    torch.linalg.vector_norm(six.final_gradient).item()
                ),
                "flow_update_norm": float(torch.linalg.vector_norm(flow_update).item()),
                "annealed_update_norm": float(
                    torch.linalg.vector_norm(final_update).item()
                ),
                "six_vs_flow_update_cosine": gradient_cosine(
                    six.final_gradient, flow_update
                ),
            }
        )
        for name, value in diagnostics.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1

    props.research_return_admitted = False
    result = {name: value / max(count, 1) for name, value in totals.items()}
    result.update(first_probe)
    result["train_batches"] = count
    return result


def run(args: Namespace) -> None:
    args.method = METHOD_NAME
    if args.anneal_epochs < 2 or args.anneal_epochs > args.epochs:
        raise SystemExit("--anneal-epochs must be between 2 and --epochs")
    if args.low_budget < 0:
        raise SystemExit("--low-budget must be non-negative")
    base.run_training(args)

