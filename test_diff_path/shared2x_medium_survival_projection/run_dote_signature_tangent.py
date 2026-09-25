from __future__ import annotations

"""Non-locking Fh-tangent DOTE residual-signature training direction.

Unlike inserting a seventh ordered objective, this runner does not add the
signature direction to the Gram-Schmidt protection basis.  The six paper
objectives are projected exactly as before.  The signature gradient is merely
projected into Fh's first-order tangent space and added as an advisory update;
therefore later Fhm/Uhm gradients retain their full native subspace.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import run_dote_signature_experiment as signature
from utils.robust_proj_utils import flatten_grads


THIS_DIR = Path(__file__).resolve().parent
base = signature.base
native_build_objectives = signature.native_build_objectives
NATIVE_NAMES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
ADVISORY_WEIGHT = 0.05


def project_fh_tangent(vector: torch.Tensor, fh: torch.Tensor) -> torch.Tensor:
    work = vector.to(dtype=torch.float64)
    basis = fh.to(dtype=torch.float64)
    denominator = torch.dot(basis, basis)
    if float(denominator.item()) <= 1e-20:
        return vector
    for _ in range(2):
        work = work - torch.dot(work, basis) / denominator * basis
    return work.to(dtype=vector.dtype)


def train_epoch(model, props, dataset, loader, optimizer) -> dict[str, float]:
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    path_masks = base.shared.base.move_dataset_static(dataset, props.device)
    totals = {f"reported_{name}": 0.0 for name in NATIVE_NAMES}
    totals["reported_Sd"] = 0.0
    first_probe: dict[str, float] = {}
    count = 0
    for batch_index, inputs in enumerate(loader):
        values = base.shared.unpack_to_device(inputs, props)
        native_losses, native_reported = native_build_objectives(
            model, props, dataset, values, path_masks
        )
        signature.AUX_WEIGHT = 1.0
        advisory_loss, advisory_diagnostics = signature.signature_loss(
            model, props, dataset, values, path_masks
        )
        if any(not torch.isfinite(loss).item() for loss in native_losses):
            raise RuntimeError("Non-finite six-objective loss")
        if not torch.isfinite(advisory_loss).item():
            raise RuntimeError("Non-finite signature advisory loss")

        advisory_raw, advisory_shapes = flatten_grads(
            model,
            advisory_loss,
            retain_graph=False,
            zero_grad=True,
        )
        result = base.ordered_project_gradients(model, native_losses)
        if advisory_shapes != result.parameter_shapes:
            raise RuntimeError("Advisory/base parameter shapes differ")
        advisory_tangent = project_fh_tangent(
            advisory_raw.to(dtype=result.final_gradient.dtype),
            result.raw_gradients[0],
        )
        final_gradient = result.final_gradient + ADVISORY_WEIGHT * advisory_tangent
        if batch_index == 0:
            first_probe = base.projection_diagnostics(result, NATIVE_NAMES)
            first_probe.update(
                {
                    "raw_gradient_norm_Sd": float(
                        torch.linalg.vector_norm(advisory_raw).item()
                    ),
                    "fh_tangent_gradient_norm_Sd": float(
                        torch.linalg.vector_norm(advisory_tangent).item()
                    ),
                    "weighted_fh_tangent_gradient_norm_Sd": float(
                        ADVISORY_WEIGHT
                        * torch.linalg.vector_norm(advisory_tangent).item()
                    ),
                    "fh_dot_weighted_Sd": float(
                        torch.dot(
                            result.raw_gradients[0].to(dtype=torch.float64),
                            (ADVISORY_WEIGHT * advisory_tangent).to(dtype=torch.float64),
                        ).item()
                    ),
                }
            )
        base.assign_gradients_and_step(
            model,
            final_gradient,
            optimizer,
            result.parameter_shapes,
        )
        for name, value in zip(NATIVE_NAMES, native_reported):
            totals[f"reported_{name}"] += float(value)
        totals["reported_Sd"] += float(advisory_diagnostics["reported_Sd"])
        count += 1
    props.research_return_admitted = False
    means = {name: value / max(count, 1) for name, value in totals.items()}
    means.update(first_probe)
    means["train_batches"] = count
    return means


def source_hashes() -> dict[str, str]:
    paths = {
        "run_dote_signature_tangent.py": Path(__file__).resolve(),
        "run_dote_signature_experiment.py": THIS_DIR / "run_dote_signature_experiment.py",
        "persistent_runner.py": signature.PERSISTENT_RUNNER,
        "dote_runner.py": signature.DOTE_RUNNER,
        "dote_checkpoint": signature.DOTE_CHECKPOINT,
        "ordered_projection.py": signature.TEST_DIR / "shared2x_full_objectives" / "ordered_projection.py",
    }
    hashes = {name: base.sha256(path) for name, path in paths.items()}
    settings = json.dumps(
        {
            "advisory_weight": ADVISORY_WEIGHT,
            "native_objectives": NATIVE_NAMES,
            "signature_is_projection_basis": False,
            "signature_projected_against": "Fh only",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["dote_signature_tangent/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


def configure(weight: float) -> None:
    global ADVISORY_WEIGHT
    ADVISORY_WEIGHT = float(weight)
    tag = f"w{ADVISORY_WEIGHT:g}".replace(".", "p")
    base.OBJECTIVE_NAMES = NATIVE_NAMES
    base.build_objectives = native_build_objectives
    base.train_epoch = train_epoch
    base.source_hashes = source_hashes
    base.OUTPUT_ROOT = THIS_DIR / "artifacts_dote_signature_tangent" / tag


def main() -> None:
    parser = argparse.ArgumentParser(description="Fh-tangent DOTE signature advisory")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--weight", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configure(args.weight)
    run_dir = base.run_one(args.level, args.seed, force=args.force)
    method = {
        "method": "non-locking Fh-tangent DOTE Medium residual signature",
        "native_projection_objectives": list(NATIVE_NAMES),
        "signature_in_projection_basis": False,
        "signature_projected_against": "Fh only",
        "advisory_weight": ADVISORY_WEIGHT,
        "path_policy_distillation": False,
        "teacher_used_at_inference": False,
        "strict_esm": True,
        "source_sha256": source_hashes(),
    }
    base.write_json(run_dir / "method.json", method)
    print(json.dumps(method, indent=2), flush=True)


if __name__ == "__main__":
    main()
