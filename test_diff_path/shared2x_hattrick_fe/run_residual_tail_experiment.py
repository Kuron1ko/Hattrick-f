from __future__ import annotations

"""2x-only residual-tail continuation for Hattrick-f.

The architecture and strict-ESM inference path are unchanged.  During the
fulfilment-only continuation, one empirical High underprediction template is
rotated into each batch.  Checkpoints must pass the High/Low safety gate and
improve Medium on both halves of the validation window.
"""

import argparse
import copy
import csv
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
BASE_PATH = THIS_DIR / "run_micro_experiment.py"
_spec = importlib.util.spec_from_file_location("hattrick_fe_base", BASE_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot import {BASE_PATH}")
base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = base
_spec.loader.exec_module(base)


OUTPUT_DIR = THIS_DIR / "artifacts" / "residual_tail_seed490"
SEED = 490
TRAIN_START = 0
TRAIN_STOP = 100
VALIDATION_BLOCKS = ((350, 375), (375, 400))
EVAL_START = 400
EVAL_STOP = 500
TEMPLATE_QUANTILES = (0.80, 0.90, 0.975)
TEMPLATE_SUPPORTS = (0.20, 0.15, 0.10)
MAX_RESIDUAL_FACTOR = 1.20
LOW_BUDGET = 0.03
CLASSES = ("High", "Medium", "Low")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def build_residual_templates(dataset, props):
    """Return per-OD positive actual/ESM tail ratios from 2x train only."""
    observed = []
    loader = base.hf_core.shared.data_loader(dataset, props.batch_size, False, SEED)
    for inputs in loader:
        actual = inputs[2].to(dtype=torch.float32)
        predicted = inputs[3].to(dtype=torch.float32)
        ratio = torch.ones_like(actual)
        valid = (predicted > 1e-6) & (actual > predicted)
        ratio[valid] = actual[valid] / predicted[valid]
        observed.append(ratio.clamp(1.0, MAX_RESIDUAL_FACTOR))
    ratios = torch.cat(observed, dim=0)
    dense_templates = [
        torch.quantile(ratios, q, dim=0) for q in TEMPLATE_QUANTILES
    ]
    templates = []
    for dense, support in zip(dense_templates, TEMPLATE_SUPPORTS):
        flattened = dense.reshape(-1)
        count = max(1, int(np.ceil(flattened.numel() * support)))
        selected = torch.topk(flattened, count, largest=True).indices
        sparse = torch.ones_like(flattened)
        sparse[selected] = flattened[selected]
        templates.append(sparse.reshape_as(dense))
    diagnostics = []
    for q, support, template in zip(
        TEMPLATE_QUANTILES, TEMPLATE_SUPPORTS, templates
    ):
        diagnostics.append(
            {
                "quantile": q,
                "target_support": support,
                "mean_factor": float(template.mean().item()),
                "p90_factor": float(torch.quantile(template, 0.90).item()),
                "max_factor": float(template.max().item()),
                "active_fraction": float((template > 1.001).float().mean().item()),
            }
        )
    return templates, diagnostics


def residual_stress_values(values: tuple, template: torch.Tensor) -> tuple:
    """Exacerbate ESM underprediction without changing actual 2x traffic."""
    stressed = list(values)
    factor = template.to(device=values[3].device, dtype=values[3].dtype)
    while factor.ndim < values[3].ndim:
        factor = factor.unsqueeze(0)
    stressed[3] = values[3] / factor.clamp_min(1.0)
    return tuple(stressed)


def train_epoch(
    model,
    teacher,
    props,
    dataset,
    loader,
    optimizer,
    templates: list[torch.Tensor],
    epoch: int,
) -> dict[str, float]:
    model.train()
    teacher.eval()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    props.research_return_policy = False
    path_masks = base.hf_core.move_static(dataset, props.device)
    totals: dict[str, float] = {}
    first_probe: dict[str, float] = {}
    template_counts = [0 for _ in templates]
    count = 0
    for batch_index, inputs in enumerate(loader):
        values = base.hf_core.shared.unpack_to_device(inputs, props)
        template_index = (epoch - 1 + batch_index) % len(templates)
        template_counts[template_index] += 1
        stressed_values = residual_stress_values(values, templates[template_index])

        nominal = base.components(model, props, dataset, values, path_masks)
        stressed = base.components(model, props, dataset, stressed_values, path_masks)
        with torch.no_grad():
            teacher_nominal = base.components(
                teacher, props, dataset, values, path_masks
            )
            teacher_stressed = base.components(
                teacher, props, dataset, stressed_values, path_masks
            )

        fulfillment, diagnostics = base.fulfillment_losses(nominal, stressed)
        high_guard, high_diag = base.envelope_guard(
            [nominal["edges_high"], stressed["edges_high"]],
            [teacher_nominal["edges_high"], teacher_stressed["edges_high"]],
            base.ENVELOPE_SLACKS["high"],
        )
        hm_guard, hm_diag = base.envelope_guard(
            [nominal["edges_high_medium"], stressed["edges_high_medium"]],
            [
                teacher_nominal["edges_high_medium"],
                teacher_stressed["edges_high_medium"],
            ],
            base.ENVELOPE_SLACKS["high_medium"],
        )
        all_guard, all_diag = base.envelope_guard(
            [nominal["edges_all"], stressed["edges_all"]],
            [teacher_nominal["edges_all"], teacher_stressed["edges_all"]],
            base.ENVELOPE_SLACKS["all"],
        )
        losses = (
            fulfillment[0],
            high_guard,
            fulfillment[1],
            hm_guard,
            fulfillment[2],
            all_guard,
        )
        objective_names = ("Fh", "Bh", "Fhm", "Bhm", "Fhml", "Ball")
        for prefix, values_ in (
            ("high_envelope", high_diag),
            ("hm_envelope", hm_diag),
            ("all_envelope", all_diag),
        ):
            diagnostics.update(
                {f"{prefix}_{key}": value for key, value in values_.items()}
            )
        if any(not torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError("non-finite residual-tail objective")
        projected = base.hf_core.ordered_project_gradients(model, losses)
        if batch_index == 0:
            first_probe = base.hf_core.projection_diagnostics(
                projected, objective_names
            )
        base.hf_core.assign_gradients_and_step(
            model,
            projected.final_gradient,
            optimizer,
            projected.parameter_shapes,
        )
        for key, value in diagnostics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
    props.research_return_admitted = False
    result = {key: value / max(count, 1) for key, value in totals.items()}
    result.update(first_probe)
    result["train_batches"] = count
    for index, used in enumerate(template_counts):
        result[f"template_{index}_batches"] = used
    return result


def block_rank(
    evaluations: list[tuple[list[dict], list[dict]]],
    baselines: list[list[dict]],
) -> tuple:
    medium_gains = []
    medium_means = []
    medium_p10 = []
    medium_p1 = []
    high_means = []
    low_margins = []
    eligible = True
    for (rows, summary), baseline_summary in zip(evaluations, baselines):
        current = base.summary_index(summary)
        reference = base.summary_index(baseline_summary)
        low_margin = float(current["Low"]["norm_fulfill_mean"]) - float(
            reference["Low"]["norm_fulfill_mean"]
        )
        eligible = eligible and (
            base.high_min(rows) >= 0.995 - 1e-4
            and low_margin >= -LOW_BUDGET
            and max(float(row["max_admitted_capacity_ratio"]) for row in summary)
            <= 1.0001
            and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
        )
        medium_gains.append(
            float(current["Medium"]["norm_fulfill_mean"])
            - float(reference["Medium"]["norm_fulfill_mean"])
        )
        medium_means.append(float(current["Medium"]["norm_fulfill_mean"]))
        medium_p10.append(float(current["Medium"]["norm_fulfill_p10"]))
        medium_p1.append(float(current["Medium"]["norm_fulfill_p1"]))
        high_means.append(float(current["High"]["norm_fulfill_mean"]))
        low_margins.append(low_margin)
    # A candidate must improve Medium on both blocks to outrank epoch zero.
    return (
        int(eligible),
        min(medium_gains),
        float(np.mean(medium_means)),
        min(medium_p10),
        min(medium_p1),
        min(high_means),
        min(low_margins),
    )


def evaluate_blocks(model, props, datasets, directory: Path, epoch: int):
    evaluations = []
    diagnostics = []
    for block_index, ((start, _stop), dataset) in enumerate(
        zip(VALIDATION_BLOCKS, datasets)
    ):
        rows, summary, diag = base.evaluate_and_save(
            model,
            props,
            dataset,
            start,
            directory,
            f"validation_{chr(97 + block_index)}_epoch_{epoch:03d}",
        )
        evaluations.append((rows, summary))
        diagnostics.append(diag)
    return evaluations, diagnostics


def train_candidate(
    props,
    phase_a: dict,
    train_dataset,
    validation_datasets,
    eval_dataset,
    baseline_blocks,
    templates,
    epochs: int,
):
    directory = OUTPUT_DIR / "hattrick_fer"
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = base.hf_core.Hattrick(props).to(
        device=props.device, dtype=props.dtype
    )
    model.load_state_dict(phase_a["model_state_dict"])
    teacher = base.hf_core.Hattrick(props).to(
        device=props.device, dtype=props.dtype
    )
    teacher.load_state_dict(phase_a["model_state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = base.hf_core.ADAMOptimizer(model.parameters(), lr=props.lr)

    initial, _ = evaluate_blocks(model, props, validation_datasets, directory, 0)
    best_rank = block_rank(initial, baseline_blocks)
    best_payload = {
        "epoch": 0,
        "rank": best_rank,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "method": "hattrick_fer",
    }
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        set_seed(SEED + epoch * 104729)
        loader = base.hf_core.shared.data_loader(
            train_dataset, props.batch_size, True, SEED + epoch * 1009
        )
        train_metrics = train_epoch(
            model,
            teacher,
            props,
            train_dataset,
            loader,
            optimizer,
            templates,
            epoch,
        )
        evaluations, val_diags = evaluate_blocks(
            model, props, validation_datasets, directory, epoch
        )
        rank = block_rank(evaluations, baseline_blocks)
        block_rows = []
        for (rows, summary), baseline_summary in zip(evaluations, baseline_blocks):
            current = base.summary_index(summary)
            reference = base.summary_index(baseline_summary)
            block_rows.append(
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
            "eligible": int(rank[0]),
            "minimum_medium_gain": rank[1],
            "a_high_mean": block_rows[0]["high_mean"],
            "a_high_min": block_rows[0]["high_min"],
            "a_medium_mean": block_rows[0]["medium_mean"],
            "a_medium_gain": block_rows[0]["medium_gain"],
            "a_low_mean": block_rows[0]["low_mean"],
            "b_high_mean": block_rows[1]["high_mean"],
            "b_high_min": block_rows[1]["high_min"],
            "b_medium_mean": block_rows[1]["medium_mean"],
            "b_medium_gain": block_rows[1]["medium_gain"],
            "b_low_mean": block_rows[1]["low_mean"],
            **{f"a_{key}": value for key, value in val_diags[0].items()},
            **{f"b_{key}": value for key, value in val_diags[1].items()},
            **train_metrics,
        }
        history.append(row)
        write_csv(directory / "train_history.csv", history)
        if rank > best_rank:
            best_rank = rank
            best_payload = {
                "epoch": epoch,
                "rank": rank,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "method": "hattrick_fer",
            }
        print(
            f"[hattrick_fer] epoch={epoch}/{epochs} eligible={rank[0]} "
            f"Hmin=({block_rows[0]['high_min']:.6f},"
            f"{block_rows[1]['high_min']:.6f}) "
            f"dM=({block_rows[0]['medium_gain']:+.6f},"
            f"{block_rows[1]['medium_gain']:+.6f})",
            flush=True,
        )

    torch.save(best_payload, directory / "best_model.pt")
    model.load_state_dict(best_payload["model_state_dict"])
    eval_rows, eval_summary, eval_diag = base.evaluate_and_save(
        model, props, eval_dataset, EVAL_START, directory, "evaluation"
    )
    complete = {
        "method": "hattrick_fer",
        "best_epoch": int(best_payload["epoch"]),
        "best_rank": list(best_rank),
        "runtime_seconds": time.perf_counter() - started,
        "evaluation": {"classes": eval_summary, "diagnostics": eval_diag},
    }
    write_json(directory / "complete.json", complete)
    return model, {"rows": eval_rows, "summary": eval_summary, **complete}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    props = base.hf_core.shared.build_props(4, device)
    if "load2x" not in str(props.topo):
        raise RuntimeError(f"refusing non-2x topology: {props.topo}")
    train_dataset = base.hf_core.DM_Dataset_within_Cluster(
        props, 0, TRAIN_START, TRAIN_STOP
    )
    validation_datasets = [
        base.hf_core.DM_Dataset_within_Cluster(props, 0, start, stop)
        for start, stop in VALIDATION_BLOCKS
    ]
    eval_dataset = base.hf_core.DM_Dataset_within_Cluster(
        props, 0, EVAL_START, EVAL_STOP
    )
    phase_a = torch.load(base.PHASE_A_PATH, map_location=device, weights_only=False)
    phase_a_model = base.hf_core.Hattrick(props).to(
        device=device, dtype=props.dtype
    )
    phase_a_model.load_state_dict(phase_a["model_state_dict"])

    baseline_blocks = []
    for block_index, ((start, _stop), dataset) in enumerate(
        zip(VALIDATION_BLOCKS, validation_datasets)
    ):
        _rows, summary, _diag = base.evaluate_and_save(
            phase_a_model,
            props,
            dataset,
            start,
            OUTPUT_DIR,
            f"phase_a_validation_{chr(97 + block_index)}",
        )
        baseline_blocks.append(summary)
    baseline_eval_rows, baseline_eval_summary, baseline_eval_diag = (
        base.evaluate_and_save(
            phase_a_model,
            props,
            eval_dataset,
            EVAL_START,
            OUTPUT_DIR,
            "phase_a_evaluation",
        )
    )
    baseline_result = {
        "rows": baseline_eval_rows,
        "summary": baseline_eval_summary,
        "diagnostics": baseline_eval_diag,
    }

    templates, template_diagnostics = build_residual_templates(train_dataset, props)
    print(json.dumps({"residual_templates": template_diagnostics}), flush=True)
    candidate_model, candidate_result = train_candidate(
        props,
        phase_a,
        train_dataset,
        validation_datasets,
        eval_dataset,
        baseline_blocks,
        templates,
        args.epochs,
    )
    far_rows, far_capacity_ratio = base.far_evaluate(
        {"phase_a": phase_a_model, "hattrick_fer": candidate_model},
        props,
        device,
    )
    write_csv(OUTPUT_DIR / "far_summary.csv", far_rows)
    report = {
        "experiment": "Hattrick-fER 2x residual-tail micro validation",
        "exploratory": True,
        "seed": SEED,
        "device": str(device),
        "cpu_threads": 1,
        "topology": str(props.topo),
        "load_factor": 2.0,
        "splits": {
            "train": [TRAIN_START, TRAIN_STOP],
            "validation_blocks": [list(value) for value in VALIDATION_BLOCKS],
            "evaluation": [EVAL_START, EVAL_STOP],
            "far_strict_esm": [base.FAR_START, base.FAR_STOP],
        },
        "epochs": args.epochs,
        "residual_templates": {
            "quantiles": TEMPLATE_QUANTILES,
            "supports": TEMPLATE_SUPPORTS,
            "maximum_factor": MAX_RESIDUAL_FACTOR,
            "direction": "High ESM divided by empirical positive actual/ESM factor; actual 2x traffic unchanged",
            "diagnostics": template_diagnostics,
        },
        "envelope_relative_slacks": base.ENVELOPE_SLACKS,
        "objective_order": ["Fh", "Bh", "Fhm", "Bhm", "Fhml", "Ball"],
        "checkpoint_rule": "High/Low/capacity safe and Medium gain positive on both validation halves",
        "best_epoch": int(candidate_result["best_epoch"]),
        "evaluation_vs_phase_a": base.paired_comparison(
            baseline_result, candidate_result
        ),
        "far_strict_esm": far_rows,
        "far_maximum_post_admission_capacity_ratio": far_capacity_ratio,
        "far_capacity_passes": far_capacity_ratio <= 1.0001,
        "runtime_seconds": float(candidate_result["runtime_seconds"]),
        "phase_a_checkpoint": str(base.PHASE_A_PATH.resolve()),
        "phase_a_checkpoint_sha256": base.sha256(base.PHASE_A_PATH),
    }
    write_json(OUTPUT_DIR / "report.json", report)
    print(OUTPUT_DIR.resolve(), flush=True)


if __name__ == "__main__":
    main()
