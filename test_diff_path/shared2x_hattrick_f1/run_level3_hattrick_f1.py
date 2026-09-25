from __future__ import annotations

"""Hattrick-f1 Level-3 experiment on 2x GEANT traffic.

Phase A is the original six-objective Level-3 checkpoint.  Phase f1 removes
MLU minimization and moves the cumulative Medium/total-fulfilment gradients in
the local null space of the worst actual High fulfilment gradients.  A linear
High-CVaR corrector is activated when the training tail loses its headroom.

Inference is unchanged and strict: policies see ESM predictions only.  Actual
traffic is introduced afterwards, solely by sequential admission/evaluation.
"""

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
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
FE_DIR = TEST_DIR / "shared2x_hattrick_fe"
HF_DIR = TEST_DIR / "shared2x_hattrick_f"
VALIDATION_DIR = TEST_DIR / "hattrick_f_final_validation"
BASE_PATH = FE_DIR / "run_micro_experiment.py"

PHASE_A_PATH = (
    TEST_DIR
    / "shared2x_order_epsilon"
    / "artifacts"
    / "phase_a"
    / "level3_validation_only"
    / "seed_490"
    / "best_model.pt"
)
HATTRICK_F_PATH = (
    HF_DIR
    / "artifacts"
    / "level3_validation_only"
    / "fh_release"
    / "seed_490"
    / "best_model.pt"
)
OUTPUT_DIR = THIS_DIR / "artifacts" / "level3_seed490"

SEED = 490
TRAIN_START = 0
TRAIN_STOP = 350
VALIDATION_BLOCKS = ((350, 375), (375, 400))
INDEPENDENT_START = 400
INDEPENDENT_STOP = 500
HOLDOUT_WINDOWS = {"near": (500, 1000), "far": (9000, 9500)}
CLASSES = ("High", "Medium", "Low")
PATHS_PER_OD = 8
TRAIN_HIGH_TAIL_RELATIVE_BUDGET = 0.001
VALIDATION_HIGH_FLOOR = 0.995
VALIDATION_HIGH_TOLERANCE = 1e-4
LOW_BUDGET = 0.03
WORST_HIGH_SAMPLES = 2
GRADIENT_CLIP_NORM = 25.0
PREDICTOR_CORRECTOR_NORM_RATIO = 0.5
LEARNING_RATE_SCALE = 0.25


_base_spec = importlib.util.spec_from_file_location("hattrick_f1_base", BASE_PATH)
if _base_spec is None or _base_spec.loader is None:
    raise RuntimeError(f"cannot import {BASE_PATH}")
base = importlib.util.module_from_spec(_base_spec)
sys.modules[_base_spec.name] = base
_base_spec.loader.exec_module(base)


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


def checkpoint_state(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return value
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise KeyError("checkpoint has no model state dictionary")


def flat_gradient(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    flattened = torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    ).to(dtype=torch.float32)
    norm = torch.linalg.vector_norm(flattened)
    if not torch.isfinite(norm).item():
        raise RuntimeError("non-finite Hattrick-f1 gradient")
    if float(norm.item()) > GRADIENT_CLIP_NORM:
        flattened = flattened * (GRADIENT_CLIP_NORM / norm)
    return flattened


def orthonormal_basis(vectors: list[torch.Tensor]) -> list[torch.Tensor]:
    basis: list[torch.Tensor] = []
    for vector in vectors:
        work = vector.to(dtype=torch.float64)
        for _ in range(2):
            for existing in basis:
                work = work - torch.dot(work, existing) * existing
        norm = torch.linalg.vector_norm(work)
        if float(norm.item()) > 1e-10:
            basis.append(work / norm)
    return basis


def project_to_tangent(
    vector: torch.Tensor, basis: list[torch.Tensor]
) -> torch.Tensor:
    work = vector.to(dtype=torch.float64)
    for _ in range(2):
        for existing in basis:
            work = work - torch.dot(work, existing) * existing
    return work.to(dtype=vector.dtype)


def high_ratios(
    admitted_high: torch.Tensor, opt1_mf: torch.Tensor
) -> torch.Tensor:
    admitted = admitted_high.reshape(admitted_high.shape[0], -1).sum(dim=1)
    optimum = opt1_mf.reshape(opt1_mf.shape[0], -1).sum(dim=1).clamp_min(1e-12)
    return admitted / optimum


def train_epoch(model, teacher, props, dataset, loader, optimizer) -> dict[str, float]:
    model.train()
    teacher.eval()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    props.research_return_policy = False
    path_masks = base.hf_core.move_static(dataset, props.device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    totals: dict[str, float] = {}
    batch_count = 0
    for inputs in loader:
        values = base.hf_core.shared.unpack_to_device(inputs, props)
        output = base.components(model, props, dataset, values, path_masks)
        with torch.no_grad():
            teacher_output = base.components(
                teacher, props, dataset, values, path_masks
            )
        ratios = high_ratios(output["admitted_high"], output["opt1_mf"])
        teacher_ratios = high_ratios(
            teacher_output["admitted_high"], teacher_output["opt1_mf"]
        )
        tail_count = min(WORST_HIGH_SAMPLES, int(ratios.numel()))
        worst_indices = torch.topk(ratios.detach(), tail_count, largest=False).indices

        high_rows = [
            flat_gradient(-ratios[index], parameters, retain_graph=True)
            for index in worst_indices
        ]
        high_basis = orthonormal_basis(high_rows)
        tail_mean = ratios[worst_indices].mean()
        teacher_tail_mean = teacher_ratios[worst_indices].mean()
        corrector_target = (
            teacher_tail_mean.detach() - TRAIN_HIGH_TAIL_RELATIVE_BUDGET
        )
        corrector_loss = torch.relu(corrector_target - tail_mean)
        corrector_gradient = flat_gradient(
            corrector_loss, parameters, retain_graph=True
        )

        loss_fhm, reported_fhm = base.loss_mf(
            output["admitted_high"] + output["admitted_medium"],
            output["opt2_mf"].detach(),
        )
        loss_fhml, reported_fhml = base.loss_mf(
            output["all_traffic"], output["opt3_mf"].detach()
        )
        ordered = base.hf_core.ordered_project_gradients(
            model, (loss_fhm, loss_fhml)
        )
        tangent_parts = [
            project_to_tangent(gradient, high_basis)
            for gradient in ordered.projected_gradients
        ]
        predictor = torch.stack(tangent_parts, dim=0).sum(dim=0)
        predictor_norm_before = torch.linalg.vector_norm(predictor)
        corrector_norm = torch.linalg.vector_norm(corrector_gradient)
        corrector_active = float(corrector_loss.detach().item() > 0.0)
        if corrector_active and float(corrector_norm.item()) > 1e-12:
            maximum_predictor = (
                PREDICTOR_CORRECTOR_NORM_RATIO * corrector_norm
            )
            if predictor_norm_before > maximum_predictor:
                predictor = predictor * (
                    maximum_predictor / predictor_norm_before.clamp_min(1e-12)
                )
        final_gradient = predictor + corrector_gradient
        final_norm = torch.linalg.vector_norm(final_gradient)
        if final_norm > GRADIENT_CLIP_NORM:
            final_gradient = final_gradient * (GRADIENT_CLIP_NORM / final_norm)
            final_norm = torch.linalg.vector_norm(final_gradient)
        if not torch.isfinite(final_gradient).all().item():
            raise RuntimeError("non-finite Hattrick-f1 final gradient")

        maximum_tangent_cosine = 0.0
        predictor_norm = torch.linalg.vector_norm(predictor)
        if float(predictor_norm.item()) > 1e-12:
            maximum_tangent_cosine = max(
                (
                    abs(float(torch.dot(predictor.double(), item).item()))
                    / float(predictor_norm.item())
                )
                for item in high_basis
            ) if high_basis else 0.0

        base.hf_core.assign_gradients_and_step(
            model,
            final_gradient,
            optimizer,
            ordered.parameter_shapes,
        )
        diagnostics = {
            "reported_Fhm": float(reported_fhm),
            "reported_Fhml": float(reported_fhml),
            "train_high_tail_mean": float(tail_mean.detach().item()),
            "teacher_high_tail_mean": float(teacher_tail_mean.detach().item()),
            "high_tail_delta_vs_teacher": float(
                (tail_mean.detach() - teacher_tail_mean.detach()).item()
            ),
            "train_high_min": float(ratios.detach().min().item()),
            "corrector_active": corrector_active,
            "corrector_loss": float(corrector_loss.detach().item()),
            "corrector_gradient_norm": float(corrector_norm.item()),
            "predictor_gradient_norm_before_cap": float(
                predictor_norm_before.item()
            ),
            "predictor_gradient_norm": float(predictor_norm.item()),
            "final_gradient_norm": float(final_norm.item()),
            "high_tangent_rank": float(len(high_basis)),
            "maximum_tangent_cosine": maximum_tangent_cosine,
        }
        for key, value in diagnostics.items():
            totals[key] = totals.get(key, 0.0) + value
        batch_count += 1
    props.research_return_admitted = False
    result = {key: value / max(batch_count, 1) for key, value in totals.items()}
    result["train_batches"] = batch_count
    return result


def evaluate_blocks(model, props, datasets, directory: Path, epoch: int):
    evaluations = []
    for block_index, ((start, _stop), dataset) in enumerate(
        zip(VALIDATION_BLOCKS, datasets)
    ):
        rows, summary, diagnostics = base.evaluate_and_save(
            model,
            props,
            dataset,
            start,
            directory,
            f"validation_{chr(97 + block_index)}_epoch_{epoch:03d}",
        )
        evaluations.append((rows, summary, diagnostics))
    return evaluations


def candidate_rank(evaluations, baseline_summaries) -> tuple:
    medium_gains = []
    medium_p10 = []
    medium_p1 = []
    high_means = []
    low_margins = []
    eligible = True
    for (rows, summary, _diag), baseline_summary in zip(
        evaluations, baseline_summaries
    ):
        current = base.summary_index(summary)
        reference = base.summary_index(baseline_summary)
        low_margin = float(current["Low"]["norm_fulfill_mean"]) - float(
            reference["Low"]["norm_fulfill_mean"]
        )
        eligible = eligible and (
            base.high_min(rows)
            >= VALIDATION_HIGH_FLOOR - VALIDATION_HIGH_TOLERANCE
            and low_margin >= -LOW_BUDGET
            and max(float(row["max_admitted_capacity_ratio"]) for row in summary)
            <= 1.0001
            and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
        )
        medium_gains.append(
            float(current["Medium"]["norm_fulfill_mean"])
            - float(reference["Medium"]["norm_fulfill_mean"])
        )
        medium_p10.append(float(current["Medium"]["norm_fulfill_p10"]))
        medium_p1.append(float(current["Medium"]["norm_fulfill_p1"]))
        high_means.append(float(current["High"]["norm_fulfill_mean"]))
        low_margins.append(low_margin)
    return (
        int(eligible),
        min(medium_gains),
        float(np.mean(medium_gains)),
        min(medium_p10),
        min(medium_p1),
        min(high_means),
        min(low_margins),
    )


def train_f1(
    props,
    phase_a_payload,
    train_dataset,
    validation_datasets,
    independent_dataset,
    baseline_summaries,
    epochs: int,
):
    directory = OUTPUT_DIR / "hattrick_f1"
    directory.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = base.hf_core.Hattrick(props).to(device=props.device, dtype=props.dtype)
    model.load_state_dict(checkpoint_state(phase_a_payload))
    teacher = base.hf_core.Hattrick(props).to(
        device=props.device, dtype=props.dtype
    )
    teacher.load_state_dict(checkpoint_state(phase_a_payload))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = base.hf_core.ADAMOptimizer(
        model.parameters(), lr=float(props.lr) * LEARNING_RATE_SCALE
    )

    initial = evaluate_blocks(model, props, validation_datasets, directory, 0)
    best_rank = candidate_rank(initial, baseline_summaries)
    best_payload = {
        "epoch": 0,
        "rank": best_rank,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "method": "hattrick_f1",
    }
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        set_seed(SEED + epoch * 104729)
        loader = base.hf_core.shared.data_loader(
            train_dataset, props.batch_size, True, SEED + epoch * 1009
        )
        training = train_epoch(
            model, teacher, props, train_dataset, loader, optimizer
        )
        evaluations = evaluate_blocks(
            model, props, validation_datasets, directory, epoch
        )
        rank = candidate_rank(evaluations, baseline_summaries)
        blocks = []
        for (rows, summary, _diag), baseline_summary in zip(
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
        write_csv(directory / "train_history.csv", history)
        if rank[0] and rank[1] > 0.0 and rank > best_rank:
            best_rank = rank
            best_payload = {
                "epoch": epoch,
                "rank": rank,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "method": "hattrick_f1",
            }
        print(
            f"[Hattrick-f1] epoch={epoch}/{epochs} eligible={rank[0]} "
            f"Hmin=({blocks[0]['high_min']:.6f},{blocks[1]['high_min']:.6f}) "
            f"dM=({blocks[0]['medium_gain']:+.6f},"
            f"{blocks[1]['medium_gain']:+.6f}) "
            f"repair={training['corrector_active']:.3f}",
            flush=True,
        )

    torch.save(best_payload, directory / "best_model.pt")
    model.load_state_dict(best_payload["model_state_dict"])
    eval_rows, eval_summary, eval_diag = base.evaluate_and_save(
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
        "independent": {"classes": eval_summary, "diagnostics": eval_diag},
    }
    write_json(directory / "complete.json", complete)
    return model, {"rows": eval_rows, "summary": eval_summary, **complete}


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def strict_temporal_holdout(models: dict[str, torch.nn.Module], props, device):
    holdout = base.holdout
    holdout.LOAD_FACTOR = 2.0
    snapshot_module = load_module(
        "hattrick_f1_snapshot", ROOT / "utils" / "snapshot_utils.py"
    )
    cluster_module = load_module(
        "hattrick_f1_cluster", ROOT / "utils" / "cluster_utils.py"
    )
    training_topology = props.topo
    props.topo = holdout.TOPOLOGY
    rows = []
    maximum_capacity_ratio = 0.0
    try:
        for window, (start, stop) in HOLDOUT_WINDOWS.items():
            dataset = holdout.RawHoldoutDataset(
                props, start, stop, snapshot_module, cluster_module
            )
            static = holdout.move_static(dataset, device)
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
                    admitted, cumulative_admitted, _ = (
                        holdout.sequential_actual_admission(
                            model, props, static, policy, actual, capacities
                        )
                    )
                    for class_index, class_name in enumerate(CLASSES):
                        batch_size = len(batch["indices"])
                        demand = (
                            actual[class_index].reshape(batch_size, -1).sum(dim=1)
                            / float(PATHS_PER_OD)
                        )
                        admitted_total = admitted[class_index].sum(dim=1)
                        fulfill = admitted_total / demand.clamp_min(1e-12)
                        capacity_ratio = holdout.link_ratio(
                            cumulative_admitted[class_index], static.pte, capacities
                        ).amax(dim=1)
                        maximum_capacity_ratio = max(
                            maximum_capacity_ratio,
                            float(capacity_ratio.max().item()),
                        )
                        if float(fulfill.max().item()) > 1.0001:
                            raise RuntimeError("raw High/Medium/Low fulfilment exceeds one")
                        if float(capacity_ratio.max().item()) > 1.0001:
                            raise RuntimeError("sequential admission exceeded capacity")
                        for local, snapshot in enumerate(batch["indices"]):
                            rows.append(
                                {
                                    "window": window,
                                    "snapshot": int(snapshot),
                                    "method": method,
                                    "class": class_name,
                                    "fulfill_ratio": float(fulfill[local].item()),
                                    "demand": float(demand[local].item()),
                                    "admitted": float(admitted_total[local].item()),
                                    "capacity_ratio": float(
                                        capacity_ratio[local].item()
                                    ),
                                }
                            )
            print(f"[strict-ESM] completed {window} {start}:{stop}", flush=True)
    finally:
        props.topo = training_topology

    summary = []
    for window in HOLDOUT_WINDOWS:
        for method in models:
            for class_name in CLASSES:
                selected = np.asarray(
                    [
                        row["fulfill_ratio"]
                        for row in rows
                        if row["window"] == window
                        and row["method"] == method
                        and row["class"] == class_name
                    ],
                    dtype=np.float64,
                )
                summary.append(
                    {
                        "window": window,
                        "method": method,
                        "class": class_name,
                        "n": int(selected.size),
                        "mean": float(selected.mean()),
                        "p1": float(np.percentile(selected, 1)),
                        "p10": float(np.percentile(selected, 10)),
                        "min": float(selected.min()),
                    }
                )
    return rows, summary, maximum_capacity_ratio


def paired(reference: dict, candidate: dict) -> dict:
    left = base.summary_index(reference["summary"])
    right = base.summary_index(candidate["summary"])
    return {
        class_name: {
            "reference_mean": float(left[class_name]["norm_fulfill_mean"]),
            "candidate_mean": float(right[class_name]["norm_fulfill_mean"]),
            "delta": float(
                right[class_name]["norm_fulfill_mean"]
                - left[class_name]["norm_fulfill_mean"]
            ),
            "reference_p1": float(left[class_name]["norm_fulfill_p1"]),
            "candidate_p1": float(right[class_name]["norm_fulfill_p1"]),
            "reference_p10": float(left[class_name]["norm_fulfill_p10"]),
            "candidate_p10": float(right[class_name]["norm_fulfill_p10"]),
        }
        for class_name in CLASSES
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
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
    phase_a_model.load_state_dict(checkpoint_state(phase_a_payload))
    hattrick_f_payload = torch.load(
        HATTRICK_F_PATH, map_location=device, weights_only=False
    )
    hattrick_f_model = base.hf_core.Hattrick(props).to(
        device=device, dtype=props.dtype
    )
    hattrick_f_model.load_state_dict(checkpoint_state(hattrick_f_payload))

    baseline_summaries = []
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
        baseline_summaries.append(summary)
    phase_rows, phase_summary, phase_diag = base.evaluate_and_save(
        phase_a_model,
        props,
        independent_dataset,
        INDEPENDENT_START,
        OUTPUT_DIR,
        "phase_a_independent",
    )
    hf_rows, hf_summary, hf_diag = base.evaluate_and_save(
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
        "diagnostics": phase_diag,
    }
    hf_result = {"rows": hf_rows, "summary": hf_summary, "diagnostics": hf_diag}

    f1_model, f1_result = train_f1(
        props,
        phase_a_payload,
        train_dataset,
        validation_datasets,
        independent_dataset,
        baseline_summaries,
        args.epochs,
    )
    holdout_rows, holdout_summary, maximum_capacity_ratio = strict_temporal_holdout(
        {
            "phase_a": phase_a_model,
            "hattrick_f": hattrick_f_model,
            "hattrick_f1": f1_model,
        },
        props,
        device,
    )
    write_csv(OUTPUT_DIR / "holdout_snapshots.csv", holdout_rows)
    write_csv(OUTPUT_DIR / "holdout_summary.csv", holdout_summary)
    report = {
        "experiment": "Hattrick-f1 Level-3 2x strict-ESM validation",
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
            "holdout": {key: list(value) for key, value in HOLDOUT_WINDOWS.items()},
        },
        "epochs": args.epochs,
        "method": {
            "predictor_objectives": ["Fhm", "Fhml"],
            "high_tangent_rows": WORST_HIGH_SAMPLES,
            "train_high_tail_relative_budget": TRAIN_HIGH_TAIL_RELATIVE_BUDGET,
            "predictor_corrector_norm_ratio": PREDICTOR_CORRECTOR_NORM_RATIO,
            "learning_rate_scale": LEARNING_RATE_SCALE,
            "mlu_in_phase_f1": False,
            "strict_esm_inference": True,
        },
        "phase_a_checkpoint": str(PHASE_A_PATH.resolve()),
        "phase_a_sha256": sha256(PHASE_A_PATH),
        "hattrick_f_checkpoint": str(HATTRICK_F_PATH.resolve()),
        "hattrick_f_sha256": sha256(HATTRICK_F_PATH),
        "best_epoch": int(f1_result["best_epoch"]),
        "independent_vs_phase_a": {
            "hattrick_f": paired(phase_result, hf_result),
            "hattrick_f1": paired(phase_result, f1_result),
        },
        "strict_esm_holdout": holdout_summary,
        "maximum_post_admission_capacity_ratio": maximum_capacity_ratio,
        "capacity_passes": maximum_capacity_ratio <= 1.0001,
        "runtime_seconds": float(f1_result["runtime_seconds"]),
    }
    write_json(OUTPUT_DIR / "report.json", report)
    print(OUTPUT_DIR.resolve(), flush=True)


if __name__ == "__main__":
    main()
