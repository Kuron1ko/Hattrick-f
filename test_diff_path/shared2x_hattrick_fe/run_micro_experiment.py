from __future__ import annotations

"""Resource-light matched test of the Hattrick-f MLU-envelope continuation.

This experiment is intentionally isolated from formal artifacts.  It compares
an ordinary fulfillment-only continuation with Hattrick-fE from the same
Phase-A checkpoint, using the same 100 training snapshots and epoch order.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
HF_DIR = TEST_DIR / "shared2x_hattrick_f"
VALIDATION_DIR = TEST_DIR / "hattrick_f_final_validation"
PHASE_A_DIR = (
    VALIDATION_DIR / "artifacts" / "phase_a_current" / "seed_490"
)
PHASE_A_PATH = PHASE_A_DIR / "best_model.pt"
OUTPUT_DIR = THIS_DIR / "artifacts" / "micro_seed490"

SEED = 490
TRAIN_START = 0
TRAIN_STOP = 100
VAL_START = 350
VAL_STOP = 400
EVAL_START = 400
EVAL_STOP = 500
FAR_START = 9000
FAR_STOP = 9500
CLASSES = ("High", "Medium", "Low")
ENVELOPE_SLACKS = {"high": 0.02, "high_medium": 0.05, "all": 0.08}
LOW_BUDGET = 0.03

for path in (HF_DIR, ROOT, VALIDATION_DIR):
    value = str(path.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)

import evaluate_holdout as holdout  # noqa: E402
from utils.training_utils import loss_mf  # noqa: E402

_hf_spec = importlib.util.spec_from_file_location(
    "hattrick_fe_hf_core", HF_DIR / "run_experiment.py"
)
if _hf_spec is None or _hf_spec.loader is None:
    raise RuntimeError("cannot import Hattrick-f core")
hf_core = importlib.util.module_from_spec(_hf_spec)
sys.modules[_hf_spec.name] = hf_core
_hf_spec.loader.exec_module(hf_core)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def components(model, props, dataset, values, path_masks) -> dict[str, torch.Tensor]:
    output, _ = hf_core.shared.model_forward(
        model, props, dataset, values, path_masks
    )
    (
        edges_high,
        edges_high_medium,
        edges_all,
        _edges_high_final,
        _edges_high_medium_final,
        all_traffic,
        admitted_high,
        admitted_medium,
        _admitted_low,
    ) = output
    return {
        "edges_high": edges_high,
        "edges_high_medium": edges_high_medium,
        "edges_all": edges_all,
        "all_traffic": all_traffic,
        "admitted_high": admitted_high,
        "admitted_medium": admitted_medium,
        "opt1_mf": values[11],
        "opt2_mf": values[12],
        "opt3_mf": values[13],
    }


def high_stress_values(values: tuple, scale: float) -> tuple:
    stressed = list(values)
    stressed[3] = values[3] * float(scale)
    return tuple(stressed)


def fulfillment_losses(
    nominal: dict[str, torch.Tensor],
    stressed: dict[str, torch.Tensor] | None,
) -> tuple[tuple[torch.Tensor, ...], dict[str, float]]:
    sources = [nominal] if stressed is None else [nominal, stressed]
    fh_losses = []
    fhm_losses = []
    fhml_losses = []
    reported = {"Fh": [], "Fhm": [], "Fhml": []}
    for item in sources:
        loss_fh, value_fh = loss_mf(
            item["admitted_high"], item["opt1_mf"].detach()
        )
        loss_fhm, value_fhm = loss_mf(
            item["admitted_high"] + item["admitted_medium"],
            item["opt2_mf"].detach(),
        )
        loss_fhml, value_fhml = loss_mf(
            item["all_traffic"], item["opt3_mf"].detach()
        )
        fh_losses.append(loss_fh)
        fhm_losses.append(loss_fhm)
        fhml_losses.append(loss_fhml)
        reported["Fh"].append(float(value_fh))
        reported["Fhm"].append(float(value_fhm))
        reported["Fhml"].append(float(value_fhml))
    count = float(len(sources))
    losses = (
        sum(fh_losses) / count,
        sum(fhm_losses) / count,
        sum(fhml_losses) / count,
    )
    diagnostics = {
        f"reported_{name}": float(np.mean(values_))
        for name, values_ in reported.items()
    }
    return losses, diagnostics


def envelope_guard(
    candidates: list[torch.Tensor],
    teachers: list[torch.Tensor],
    slack: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    ratios = []
    for candidate, teacher in zip(candidates, teachers):
        candidate_mlu = candidate.reshape(candidate.shape[0], -1).amax(dim=1)
        teacher_mlu = (
            teacher.detach().reshape(teacher.shape[0], -1).amax(dim=1)
        ).clamp_min(1e-12)
        ratios.append(candidate_mlu / teacher_mlu)
    ratio = torch.cat(ratios)
    violation = torch.relu(ratio - (1.0 + float(slack)))
    return violation.square().mean(), {
        "ratio_mean": float(ratio.detach().mean().item()),
        "ratio_max": float(ratio.detach().max().item()),
        "active_fraction": float(
            (violation.detach() > 0).to(dtype=torch.float32).mean().item()
        ),
        "violation_mean": float(violation.detach().mean().item()),
    }


def train_epoch(
    model,
    teacher,
    props,
    dataset,
    loader,
    optimizer,
    method: str,
    stress_scale: float,
) -> dict[str, float]:
    model.train()
    teacher.eval()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    props.research_return_policy = False
    path_masks = hf_core.move_static(dataset, props.device)
    totals: dict[str, float] = {}
    first_probe: dict[str, float] = {}
    count = 0
    for batch_index, inputs in enumerate(loader):
        values = hf_core.shared.unpack_to_device(inputs, props)
        nominal = components(model, props, dataset, values, path_masks)
        if method == "release_control":
            losses, diagnostics = fulfillment_losses(nominal, None)
            objective_names = ("Fh", "Fhm", "Fhml")
        else:
            stressed_values = high_stress_values(values, stress_scale)
            stressed = components(
                model, props, dataset, stressed_values, path_masks
            )
            with torch.no_grad():
                teacher_nominal = components(
                    teacher, props, dataset, values, path_masks
                )
                teacher_stressed = components(
                    teacher, props, dataset, stressed_values, path_masks
                )
            fulfillment, diagnostics = fulfillment_losses(nominal, stressed)
            high_guard, high_diag = envelope_guard(
                [nominal["edges_high"], stressed["edges_high"]],
                [teacher_nominal["edges_high"], teacher_stressed["edges_high"]],
                ENVELOPE_SLACKS["high"],
            )
            hm_guard, hm_diag = envelope_guard(
                [
                    nominal["edges_high_medium"],
                    stressed["edges_high_medium"],
                ],
                [
                    teacher_nominal["edges_high_medium"],
                    teacher_stressed["edges_high_medium"],
                ],
                ENVELOPE_SLACKS["high_medium"],
            )
            all_guard, all_diag = envelope_guard(
                [nominal["edges_all"], stressed["edges_all"]],
                [teacher_nominal["edges_all"], teacher_stressed["edges_all"]],
                ENVELOPE_SLACKS["all"],
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
            raise RuntimeError("non-finite continuation objective")
        projected = hf_core.ordered_project_gradients(model, losses)
        if batch_index == 0:
            first_probe = hf_core.projection_diagnostics(
                projected, objective_names
            )
        hf_core.assign_gradients_and_step(
            model,
            projected.final_gradient,
            optimizer,
            projected.parameter_shapes,
        )
        for key, value in diagnostics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
    props.research_return_admitted = False
    output = {key: value / max(count, 1) for key, value in totals.items()}
    output.update(first_probe)
    output["train_batches"] = count
    return output


def estimate_stress_scale(dataset, props) -> float:
    ratios = []
    loader = hf_core.shared.data_loader(dataset, props.batch_size, False, SEED)
    for inputs in loader:
        actual = inputs[2].reshape(inputs[2].shape[0], -1).sum(dim=1)
        predicted = inputs[3].reshape(inputs[3].shape[0], -1).sum(dim=1)
        ratios.extend((actual / predicted.clamp_min(1e-12)).cpu().numpy().tolist())
    positive_tail = np.maximum(np.asarray(ratios, dtype=np.float64), 1.0)
    return float(np.clip(np.quantile(positive_tail, 0.90), 1.05, 1.25))


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def high_min(rows: list[dict]) -> float:
    return min(
        float(row["norm_fulfill"])
        for row in rows
        if row["class"] == "High"
    )


def checkpoint_rank(
    rows: list[dict], summary: list[dict], baseline_summary: list[dict]
) -> tuple:
    current = summary_index(summary)
    baseline = summary_index(baseline_summary)
    eligible = (
        high_min(rows) >= 0.995 - 1e-4
        and float(current["Low"]["norm_fulfill_mean"])
        >= float(baseline["Low"]["norm_fulfill_mean"]) - LOW_BUDGET
        and max(float(row["max_admitted_capacity_ratio"]) for row in summary)
        <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
    )
    return (
        int(eligible),
        float(current["Medium"]["norm_fulfill_mean"]),
        float(current["Medium"]["norm_fulfill_p10"]),
        float(current["Medium"]["norm_fulfill_p1"]),
        float(current["High"]["norm_fulfill_mean"]),
        float(current["Low"]["norm_fulfill_mean"]),
    )


def evaluate_and_save(model, props, dataset, start: int, directory: Path, name: str):
    rows, summary, diagnostics = hf_core.evaluate(
        model, props, dataset, start
    )
    write_csv(directory / f"{name}_metrics.csv", rows)
    write_json(
        directory / f"{name}_summary.json",
        {"classes": summary, "diagnostics": diagnostics},
    )
    return rows, summary, diagnostics


def train_method(
    method: str,
    props,
    phase_a: dict,
    train_dataset,
    val_dataset,
    eval_dataset,
    baseline_val_summary: list[dict],
    stress_scale: float,
    epochs: int,
) -> tuple[torch.nn.Module, dict]:
    directory = OUTPUT_DIR / method
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = hf_core.Hattrick(props).to(device=props.device, dtype=props.dtype)
    model.load_state_dict(phase_a["model_state_dict"])
    teacher = hf_core.Hattrick(props).to(device=props.device, dtype=props.dtype)
    teacher.load_state_dict(phase_a["model_state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = hf_core.ADAMOptimizer(model.parameters(), lr=props.lr)

    baseline_rows, baseline_summary, _ = evaluate_and_save(
        model, props, val_dataset, VAL_START, directory, "validation_epoch_000"
    )
    best_rank = checkpoint_rank(
        baseline_rows, baseline_summary, baseline_val_summary
    )
    best_payload = {
        "epoch": 0,
        "rank": best_rank,
        "model_state_dict": model.state_dict(),
        "method": method,
    }
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        set_seed(SEED + epoch * 104729)
        loader = hf_core.shared.data_loader(
            train_dataset, props.batch_size, True, SEED + epoch * 1009
        )
        train_metrics = train_epoch(
            model,
            teacher,
            props,
            train_dataset,
            loader,
            optimizer,
            method,
            stress_scale,
        )
        val_rows, val_summary, val_diag = evaluate_and_save(
            model,
            props,
            val_dataset,
            VAL_START,
            directory,
            f"validation_epoch_{epoch:03d}",
        )
        rank = checkpoint_rank(val_rows, val_summary, baseline_val_summary)
        indexed = summary_index(val_summary)
        row = {
            "epoch": epoch,
            "eligible": int(rank[0]),
            "high_mean": indexed["High"]["norm_fulfill_mean"],
            "high_p1": indexed["High"]["norm_fulfill_p1"],
            "high_p10": indexed["High"]["norm_fulfill_p10"],
            "high_min": high_min(val_rows),
            "medium_mean": indexed["Medium"]["norm_fulfill_mean"],
            "medium_p1": indexed["Medium"]["norm_fulfill_p1"],
            "medium_p10": indexed["Medium"]["norm_fulfill_p10"],
            "low_mean": indexed["Low"]["norm_fulfill_mean"],
            **val_diag,
            **train_metrics,
        }
        history.append(row)
        write_csv(directory / "train_history.csv", history)
        if rank > best_rank:
            best_rank = rank
            best_payload = {
                "epoch": epoch,
                "rank": rank,
                "model_state_dict": model.state_dict(),
                "method": method,
            }
        print(
            f"[{method}] epoch={epoch}/{epochs} eligible={rank[0]} "
            f"H={row['high_mean']:.6f} Hmin={row['high_min']:.6f} "
            f"M={row['medium_mean']:.6f} L={row['low_mean']:.6f}",
            flush=True,
        )

    torch.save(best_payload, directory / "best_model.pt")
    model.load_state_dict(best_payload["model_state_dict"])
    eval_rows, eval_summary, eval_diag = evaluate_and_save(
        model, props, eval_dataset, EVAL_START, directory, "evaluation"
    )
    complete = {
        "method": method,
        "best_epoch": int(best_payload["epoch"]),
        "best_rank": list(best_rank),
        "runtime_seconds": time.perf_counter() - started,
        "evaluation": {"classes": eval_summary, "diagnostics": eval_diag},
    }
    write_json(directory / "complete.json", complete)
    return model, {"rows": eval_rows, "summary": eval_summary, **complete}


def far_evaluate(models: dict[str, torch.nn.Module], props, device: torch.device):
    holdout.LOAD_FACTOR = 2.0
    # The micro-training props point at the 500-snapshot priomask dataset.
    # The frozen far protocol instead uses the long GEANT trace (t9001--t9500).
    # Match evaluate_far_1x_2x.py and switch only the dataset locator; the
    # topology tensors and model architecture are unchanged.
    training_topology = props.topo
    props.topo = holdout.TOPOLOGY
    runtime = load_module("hattrick_fe_far_runtime", holdout.RUNTIME_SOURCE)
    snapshot_module = load_module(
        "hattrick_fe_far_snapshot", ROOT / "utils" / "snapshot_utils.py"
    )
    cluster_module = load_module(
        "hattrick_fe_far_cluster", ROOT / "utils" / "cluster_utils.py"
    )
    try:
        dataset = holdout.RawHoldoutDataset(
            props, FAR_START, FAR_STOP, snapshot_module, cluster_module
        )
    finally:
        props.topo = training_topology
    static = holdout.move_static(dataset, device)
    values: dict[tuple[str, str], list[float]] = {
        (method, class_name): []
        for method in models
        for class_name in CLASSES
    }
    maximum_ratio = 0.0
    for batch in dataset.batches(8):
        node_features = batch["node_features"].to(device)
        capacities = batch["capacities"].to(device)
        actual = tuple(value.to(device) for value in batch["actual"])
        predicted = tuple(value.to(device) for value in batch["predicted"])
        for method, model in models.items():
            policy = holdout.generate_prediction_only_policy(
                model,
                props,
                static,
                node_features,
                capacities,
                predicted,
            )
            admitted, cumulative_admitted, _ = holdout.sequential_actual_admission(
                model, props, static, policy, actual, capacities
            )
            for class_index, class_name in enumerate(CLASSES):
                demand = (
                    actual[class_index].reshape(len(batch["indices"]), -1).sum(dim=1)
                    / float(holdout.PATHS_PER_OD)
                )
                admitted_total = admitted[class_index].sum(dim=1)
                fulfill = admitted_total / demand.clamp_min(1e-12)
                values[(method, class_name)].extend(
                    fulfill.detach().cpu().numpy().tolist()
                )
                ratio = holdout.link_ratio(
                    cumulative_admitted[class_index], static.pte, capacities
                ).amax(dim=1)
                maximum_ratio = max(maximum_ratio, float(ratio.max().item()))
    rows = []
    for (method, class_name), selected in values.items():
        array = np.asarray(selected, dtype=np.float64)
        rows.append(
            {
                "method": method,
                "class": class_name,
                "n": int(array.size),
                "mean": float(array.mean()),
                "p1": float(np.percentile(array, 1)),
                "p10": float(np.percentile(array, 10)),
                "min": float(array.min()),
            }
        )
    return rows, maximum_ratio


def paired_comparison(reference: dict, candidate: dict) -> dict:
    left = summary_index(reference["summary"])
    right = summary_index(candidate["summary"])
    return {
        class_name: {
            "reference_mean": float(left[class_name]["norm_fulfill_mean"]),
            "candidate_mean": float(right[class_name]["norm_fulfill_mean"]),
            "delta": float(
                right[class_name]["norm_fulfill_mean"]
                - left[class_name]["norm_fulfill_mean"]
            ),
            "reference_normalized_mlu": float(
                left[class_name]["normalized_mlu_mean"]
            ),
            "candidate_normalized_mlu": float(
                right[class_name]["normalized_mlu_mean"]
            ),
        }
        for class_name in CLASSES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--resume-evaluation",
        action="store_true",
        help="load the already selected checkpoints and resume final evaluation",
    )
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

    props = hf_core.shared.build_props(4, device)
    train_dataset = hf_core.DM_Dataset_within_Cluster(
        props, 0, TRAIN_START, TRAIN_STOP
    )
    val_dataset = hf_core.DM_Dataset_within_Cluster(props, 0, VAL_START, VAL_STOP)
    eval_dataset = hf_core.DM_Dataset_within_Cluster(
        props, 0, EVAL_START, EVAL_STOP
    )
    phase_a = torch.load(PHASE_A_PATH, map_location=device, weights_only=False)
    baseline_model = hf_core.Hattrick(props).to(
        device=device, dtype=props.dtype
    )
    baseline_model.load_state_dict(phase_a["model_state_dict"])
    baseline_val_rows, baseline_val_summary, _ = evaluate_and_save(
        baseline_model,
        props,
        val_dataset,
        VAL_START,
        OUTPUT_DIR,
        "phase_a_validation",
    )
    baseline_eval_rows, baseline_eval_summary, baseline_eval_diag = evaluate_and_save(
        baseline_model,
        props,
        eval_dataset,
        EVAL_START,
        OUTPUT_DIR,
        "phase_a_evaluation",
    )
    baseline_result = {
        "rows": baseline_eval_rows,
        "summary": baseline_eval_summary,
        "diagnostics": baseline_eval_diag,
    }
    stress_scale = estimate_stress_scale(train_dataset, props)

    models = {"phase_a": baseline_model}
    results = {"phase_a": baseline_result}
    for method in ("release_control", "hattrick_fe"):
        if args.resume_evaluation:
            directory = OUTPUT_DIR / method
            checkpoint = torch.load(
                directory / "best_model.pt", map_location=device, weights_only=False
            )
            model = hf_core.Hattrick(props).to(device=device, dtype=props.dtype)
            model.load_state_dict(checkpoint["model_state_dict"])
            complete = json.loads(
                (directory / "complete.json").read_text(encoding="utf-8")
            )
            result = {
                "summary": complete["evaluation"]["classes"],
                **complete,
            }
        else:
            model, result = train_method(
                method,
                props,
                phase_a,
                train_dataset,
                val_dataset,
                eval_dataset,
                baseline_val_summary,
                stress_scale,
                args.epochs,
            )
        models[method] = model
        results[method] = result

    far_rows, far_capacity_ratio = far_evaluate(models, props, device)
    write_csv(OUTPUT_DIR / "far_summary.csv", far_rows)
    report = {
        "experiment": "Hattrick-fE resource-isolated matched micro validation",
        "exploratory": True,
        "seed": SEED,
        "device": str(device),
        "cpu_threads": 1,
        "splits": {
            "train": [TRAIN_START, TRAIN_STOP],
            "validation_selection_only": [VAL_START, VAL_STOP],
            "evaluation": [EVAL_START, EVAL_STOP],
            "far_strict_esm": [FAR_START, FAR_STOP],
        },
        "epochs": args.epochs,
        "stress": {
            "class": "High",
            "scale": stress_scale,
            "derivation": "90th percentile of positive actual/ESM aggregate High ratio on micro-train, clipped to [1.05,1.25]",
        },
        "envelope_relative_slacks": ENVELOPE_SLACKS,
        "objective_order": ["Fh", "Bh", "Fhm", "Bhm", "Fhml", "Ball"],
        "phase_a_checkpoint": str(PHASE_A_PATH.resolve()),
        "phase_a_checkpoint_sha256": sha256(PHASE_A_PATH),
        "evaluation_vs_phase_a": {
            method: paired_comparison(baseline_result, results[method])
            for method in ("release_control", "hattrick_fe")
        },
        "evaluation_hattrick_fe_vs_release_control": paired_comparison(
            results["release_control"], results["hattrick_fe"]
        ),
        "far_strict_esm": far_rows,
        "far_maximum_post_admission_capacity_ratio": far_capacity_ratio,
        "far_capacity_passes": far_capacity_ratio <= 1.0001,
        "best_epochs": {
            method: int(results[method]["best_epoch"])
            for method in ("release_control", "hattrick_fe")
        },
        "runtime_seconds": {
            method: float(results[method]["runtime_seconds"])
            for method in ("release_control", "hattrick_fe")
        },
    }
    write_json(OUTPUT_DIR / "report.json", report)
    print(OUTPUT_DIR.resolve(), flush=True)


if __name__ == "__main__":
    main()
