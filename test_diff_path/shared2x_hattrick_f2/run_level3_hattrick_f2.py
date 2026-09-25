from __future__ import annotations

"""Hattrick-f2: cosine-annealed MLU continuation for Level-3, 2x GEANT.

The starting checkpoint is the original Hattrick trained with the complete
six-objective order Fh -> Uh -> Fhm -> Uhm -> Fhml -> Uhml.  During the f2
continuation, each update is

    g_f2 = alpha * g_six + (1 - alpha) * g_flow,

where g_six is the exact six-objective ordered-projection update, g_flow is the
exact Fh -> Fhm -> Fhml ordered-projection update, and alpha follows a cosine
schedule from 1 to 0.  Blending the projected updates is intentional: merely
scaling an MLU loss would not smoothly remove its priority-basis effect from
the ordered projection.

Inference is unchanged and strict.  Routing policies see ESM predictions
only; actual traffic is used afterwards for training loss and admission.
"""

import argparse
import copy
import importlib.util
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
INFRA_PATH = (
    TEST_DIR / "shared2x_hattrick_f1" / "run_level3_hattrick_f1.py"
)
OUTPUT_DIR = THIS_DIR / "artifacts" / "level3_seed490"

_infra_spec = importlib.util.spec_from_file_location(
    "hattrick_f2_level3_infrastructure", INFRA_PATH
)
if _infra_spec is None or _infra_spec.loader is None:
    raise RuntimeError(f"cannot import Level-3 infrastructure: {INFRA_PATH}")
infra = importlib.util.module_from_spec(_infra_spec)
sys.modules[_infra_spec.name] = infra
_infra_spec.loader.exec_module(infra)

from utils.training_utils import loss_mf, loss_mlu  # noqa: E402


base = infra.base
SEED = infra.SEED
TRAIN_START = infra.TRAIN_START
TRAIN_STOP = infra.TRAIN_STOP
VALIDATION_BLOCKS = infra.VALIDATION_BLOCKS
INDEPENDENT_START = infra.INDEPENDENT_START
INDEPENDENT_STOP = infra.INDEPENDENT_STOP
HOLDOUT_WINDOWS = infra.HOLDOUT_WINDOWS
PHASE_A_PATH = infra.PHASE_A_PATH
HATTRICK_F_PATH = infra.HATTRICK_F_PATH
OBJECTIVE_NAMES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
FLOW_OBJECTIVE_NAMES = ("Fh", "Fhm", "Fhml")
FLOW_OBJECTIVE_INDICES = (0, 2, 4)
ZERO_TOLERANCE = 1e-12


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True)


def anneal_alpha(epoch: int, epochs: int) -> float:
    """Cosine continuation: epoch 1 is six-loss; final epoch is flow-only."""
    if epochs < 2:
        raise ValueError("Hattrick-f2 needs at least two epochs")
    progress = float(epoch - 1) / float(epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_objectives(model, props, dataset, values, path_masks):
    components = base.components(model, props, dataset, values, path_masks)
    loss_fh, value_fh = loss_mf(
        components["admitted_high"], components["opt1_mf"].detach()
    )
    loss_uh, value_uh = loss_mlu(
        components["edges_high"], values[8].detach()
    )
    loss_fhm, value_fhm = loss_mf(
        components["admitted_high"] + components["admitted_medium"],
        components["opt2_mf"].detach(),
    )
    loss_uhm, value_uhm = loss_mlu(
        components["edges_high_medium"], values[9].detach()
    )
    loss_fhml, value_fhml = loss_mf(
        components["all_traffic"], components["opt3_mf"].detach()
    )
    loss_uhml, value_uhml = loss_mlu(
        components["edges_all"], values[10].detach()
    )
    return (
        (loss_fh, loss_uh, loss_fhm, loss_uhm, loss_fhml, loss_uhml),
        (value_fh, value_uh, value_fhm, value_uhm, value_fhml, value_uhml),
    )


def ordered_projection_from_raw(
    raw_gradients: list[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Apply the repository's two-pass ordered projection to cached gradients."""
    basis: list[torch.Tensor] = []
    projected: list[torch.Tensor] = []
    for raw in raw_gradients:
        work = raw.to(dtype=torch.float64)
        raw_norm = torch.linalg.vector_norm(work)
        if not torch.isfinite(raw_norm).item():
            raise RuntimeError("non-finite raw Hattrick-f2 gradient")
        if float(raw_norm.item()) <= ZERO_TOLERANCE:
            projected.append(torch.zeros_like(raw))
            continue
        for _ in range(2):
            for basis_vector in basis:
                work = work - torch.dot(work, basis_vector) * basis_vector
        projected_norm = torch.linalg.vector_norm(work)
        if not torch.isfinite(projected_norm).item():
            raise RuntimeError("non-finite projected Hattrick-f2 gradient")
        if float(projected_norm.item()) <= ZERO_TOLERANCE:
            projected.append(torch.zeros_like(raw))
            continue
        basis.append(work / projected_norm)
        projected.append(work.to(dtype=raw.dtype))
    final = torch.stack(projected, dim=0).sum(dim=0)
    if not torch.isfinite(final).all().item():
        raise RuntimeError("non-finite flow-only Hattrick-f2 update")
    return projected, final


def gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left.double()) * torch.linalg.vector_norm(
        right.double()
    )
    if float(denominator.item()) <= 1e-20:
        return 0.0
    return float(torch.dot(left.double(), right.double()).item() / denominator.item())


def train_epoch(
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
    path_masks = base.hf_core.move_static(dataset, props.device)
    totals: dict[str, float] = {}
    first_probe: dict[str, float] = {}
    batch_count = 0
    for batch_index, inputs in enumerate(loader):
        values = base.hf_core.shared.unpack_to_device(inputs, props)
        losses, reported = build_objectives(
            model, props, dataset, values, path_masks
        )
        if any(not torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError("non-finite Hattrick-f2 objective")

        six = base.hf_core.ordered_project_gradients(model, losses)
        flow_raw = [six.raw_gradients[index] for index in FLOW_OBJECTIVE_INDICES]
        flow_projected, flow_update = ordered_projection_from_raw(flow_raw)
        final_update = float(alpha) * six.final_gradient + (
            1.0 - float(alpha)
        ) * flow_update
        if not torch.isfinite(final_update).all().item():
            raise RuntimeError("non-finite annealed Hattrick-f2 update")

        if batch_index == 0:
            first_probe = base.hf_core.projection_diagnostics(
                six, OBJECTIVE_NAMES
            )
            for name, gradient in zip(FLOW_OBJECTIVE_NAMES, flow_projected):
                first_probe[f"flow_projected_gradient_norm_{name}"] = float(
                    torch.linalg.vector_norm(gradient).item()
                )

        base.hf_core.assign_gradients_and_step(
            model,
            final_update,
            optimizer,
            six.parameter_shapes,
        )
        diagnostics = {
            f"reported_{name}": float(value)
            for name, value in zip(OBJECTIVE_NAMES, reported)
        }
        diagnostics.update(
            {
                "anneal_alpha": float(alpha),
                "six_update_norm": float(
                    torch.linalg.vector_norm(six.final_gradient).item()
                ),
                "flow_update_norm": float(
                    torch.linalg.vector_norm(flow_update).item()
                ),
                "annealed_update_norm": float(
                    torch.linalg.vector_norm(final_update).item()
                ),
                "six_vs_flow_update_cosine": gradient_cosine(
                    six.final_gradient, flow_update
                ),
            }
        )
        for key, value in diagnostics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batch_count += 1
    props.research_return_admitted = False
    result = {
        key: value / max(batch_count, 1) for key, value in totals.items()
    }
    result.update(first_probe)
    result["train_batches"] = batch_count
    return result


def train_f2(
    props,
    phase_a_payload,
    train_dataset,
    validation_datasets,
    independent_dataset,
    baseline_summaries,
    epochs: int,
):
    directory = OUTPUT_DIR / "hattrick_f2"
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = base.hf_core.Hattrick(props).to(
        device=props.device, dtype=props.dtype
    )
    model.load_state_dict(infra.checkpoint_state(phase_a_payload))
    optimizer = base.hf_core.ADAMOptimizer(model.parameters(), lr=float(props.lr))

    initial = infra.evaluate_blocks(
        model, props, validation_datasets, directory, 0
    )
    best_rank = infra.candidate_rank(initial, baseline_summaries)
    best_payload = {
        "epoch": 0,
        "rank": best_rank,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "method": "hattrick_f2",
    }
    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        set_seed(SEED + epoch * 104729)
        alpha = anneal_alpha(epoch, epochs)
        loader = base.hf_core.shared.data_loader(
            train_dataset, props.batch_size, True, SEED + epoch * 1009
        )
        training = train_epoch(
            model, props, train_dataset, loader, optimizer, alpha
        )
        evaluations = infra.evaluate_blocks(
            model, props, validation_datasets, directory, epoch
        )
        rank = infra.candidate_rank(evaluations, baseline_summaries)
        blocks = []
        for (rows, summary, _diagnostics), baseline_summary in zip(
            evaluations, baseline_summaries
        ):
            current = base.summary_index(summary)
            reference = base.summary_index(baseline_summary)
            blocks.append(
                {
                    "high_mean": current["High"]["norm_fulfill_mean"],
                    "high_min": base.high_min(rows),
                    "medium_mean": current["Medium"]["norm_fulfill_mean"],
                    "medium_gain": current["Medium"]["norm_fulfill_mean"]
                    - reference["Medium"]["norm_fulfill_mean"],
                    "low_mean": current["Low"]["norm_fulfill_mean"],
                }
            )
        row = {
            "epoch": epoch,
            "anneal_alpha": alpha,
            "eligible": rank[0],
            "minimum_medium_gain": rank[1],
            "a_high_mean": blocks[0]["high_mean"],
            "a_high_min": blocks[0]["high_min"],
            "a_medium_mean": blocks[0]["medium_mean"],
            "a_medium_gain": blocks[0]["medium_gain"],
            "a_low_mean": blocks[0]["low_mean"],
            "b_high_mean": blocks[1]["high_mean"],
            "b_high_min": blocks[1]["high_min"],
            "b_medium_mean": blocks[1]["medium_mean"],
            "b_medium_gain": blocks[1]["medium_gain"],
            "b_low_mean": blocks[1]["low_mean"],
            **training,
        }
        history.append(row)
        infra.write_csv(directory / "train_history.csv", history)
        if rank[0] and rank[1] > 0.0 and rank > best_rank:
            best_rank = rank
            best_payload = {
                "epoch": epoch,
                "rank": rank,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "method": "hattrick_f2",
            }
        print(
            f"[Hattrick-f2] epoch={epoch}/{epochs} alpha={alpha:.4f} "
            f"eligible={rank[0]} "
            f"Hmin=({blocks[0]['high_min']:.6f},"
            f"{blocks[1]['high_min']:.6f}) "
            f"dM=({blocks[0]['medium_gain']:+.6f},"
            f"{blocks[1]['medium_gain']:+.6f})",
            flush=True,
        )

    torch.save(best_payload, directory / "best_model.pt")
    model.load_state_dict(best_payload["model_state_dict"])
    eval_rows, eval_summary, eval_diagnostics = base.evaluate_and_save(
        model,
        props,
        independent_dataset,
        INDEPENDENT_START,
        directory,
        "independent_evaluation",
    )
    complete = {
        "best_epoch": int(best_payload["epoch"]),
        "best_rank": list(best_rank),
        "runtime_seconds": time.perf_counter() - started,
        "independent": {
            "classes": eval_summary,
            "diagnostics": eval_diagnostics,
        },
    }
    infra.write_json(directory / "complete.json", complete)
    return model, {"rows": eval_rows, "summary": eval_summary, **complete}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hattrick-f2 Level-3 cosine-annealed MLU continuation"
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.epochs < 2:
        parser.error("--epochs must be at least 2 for an annealing schedule")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    props = base.hf_core.shared.build_props(3, device)
    if "load2x" not in str(props.topo):
        raise RuntimeError(f"refusing non-2x topology: {props.topo}")
    train_dataset = base.hf_core.DM_Dataset_within_Cluster(
        props, 0, TRAIN_START, TRAIN_STOP
    )
    validation_datasets = [
        base.hf_core.DM_Dataset_within_Cluster(props, 0, start, stop)
        for start, stop in VALIDATION_BLOCKS
    ]
    independent_dataset = base.hf_core.DM_Dataset_within_Cluster(
        props, 0, INDEPENDENT_START, INDEPENDENT_STOP
    )

    phase_a_payload = torch.load(
        PHASE_A_PATH, map_location=device, weights_only=False
    )
    phase_a_model = base.hf_core.Hattrick(props).to(
        device=device, dtype=props.dtype
    )
    phase_a_model.load_state_dict(infra.checkpoint_state(phase_a_payload))
    hattrick_f_payload = torch.load(
        HATTRICK_F_PATH, map_location=device, weights_only=False
    )
    hattrick_f_model = base.hf_core.Hattrick(props).to(
        device=device, dtype=props.dtype
    )
    hattrick_f_model.load_state_dict(infra.checkpoint_state(hattrick_f_payload))

    baseline_summaries = []
    for block_index, ((start, _stop), dataset) in enumerate(
        zip(VALIDATION_BLOCKS, validation_datasets)
    ):
        _rows, summary, _diagnostics = base.evaluate_and_save(
            phase_a_model,
            props,
            dataset,
            start,
            OUTPUT_DIR,
            f"phase_a_validation_{chr(97 + block_index)}",
        )
        baseline_summaries.append(summary)
    phase_rows, phase_summary, phase_diagnostics = base.evaluate_and_save(
        phase_a_model,
        props,
        independent_dataset,
        INDEPENDENT_START,
        OUTPUT_DIR,
        "phase_a_independent",
    )
    f_rows, f_summary, f_diagnostics = base.evaluate_and_save(
        hattrick_f_model,
        props,
        independent_dataset,
        INDEPENDENT_START,
        OUTPUT_DIR,
        "hattrick_f_independent",
    )
    phase_result = {
        "rows": phase_rows,
        "summary": phase_summary,
        "diagnostics": phase_diagnostics,
    }
    f_result = {
        "rows": f_rows,
        "summary": f_summary,
        "diagnostics": f_diagnostics,
    }

    f2_model, f2_result = train_f2(
        props,
        phase_a_payload,
        train_dataset,
        validation_datasets,
        independent_dataset,
        baseline_summaries,
        args.epochs,
    )
    holdout_rows, holdout_summary, maximum_capacity_ratio = (
        infra.strict_temporal_holdout(
            {
                "phase_a": phase_a_model,
                "hattrick_f": hattrick_f_model,
                "hattrick_f2": f2_model,
            },
            props,
            device,
        )
    )
    infra.write_csv(OUTPUT_DIR / "holdout_snapshots.csv", holdout_rows)
    infra.write_csv(OUTPUT_DIR / "holdout_summary.csv", holdout_summary)
    report = {
        "experiment": "Hattrick-f2 Level-3 2x strict-ESM validation",
        "exploratory": True,
        "seed": SEED,
        "device": str(device),
        "cpu_threads": 1,
        "topology": str(props.topo),
        "load_factor": 2.0,
        "splits": {
            "train": [TRAIN_START, TRAIN_STOP],
            "validation_blocks": [list(value) for value in VALIDATION_BLOCKS],
            "independent": [INDEPENDENT_START, INDEPENDENT_STOP],
            "holdout": {
                key: list(value) for key, value in HOLDOUT_WINDOWS.items()
            },
        },
        "epochs": args.epochs,
        "method": {
            "name": "Hattrick-f2",
            "phase_a_objectives": list(OBJECTIVE_NAMES),
            "released_objectives": list(FLOW_OBJECTIVE_NAMES),
            "annealed_quantity": "ordered projected update",
            "update_formula": "alpha*g_six + (1-alpha)*g_flow",
            "schedule": "0.5*(1+cos(pi*(epoch-1)/(epochs-1)))",
            "alpha_start": 1.0,
            "alpha_end": 0.0,
            "optimizer_reset": True,
            "all_parameters_trainable": True,
            "strict_esm_inference": True,
        },
        "phase_a_checkpoint": str(PHASE_A_PATH.resolve()),
        "phase_a_sha256": infra.sha256(PHASE_A_PATH),
        "hattrick_f_checkpoint": str(HATTRICK_F_PATH.resolve()),
        "hattrick_f_sha256": infra.sha256(HATTRICK_F_PATH),
        "best_epoch": int(f2_result["best_epoch"]),
        "independent_vs_phase_a": {
            "hattrick_f": infra.paired(phase_result, f_result),
            "hattrick_f2": infra.paired(phase_result, f2_result),
        },
        "strict_esm_holdout": holdout_summary,
        "maximum_post_admission_capacity_ratio": maximum_capacity_ratio,
        "capacity_passes": maximum_capacity_ratio <= 1.0001,
        "runtime_seconds": float(f2_result["runtime_seconds"]),
    }
    infra.write_json(OUTPUT_DIR / "report.json", report)
    print(OUTPUT_DIR.resolve(), flush=True)


if __name__ == "__main__":
    main()
