from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
ROUND1_DIR = THIS_DIR.parent
TEST_DIR = ROUND1_DIR.parent
ROOT = TEST_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(ROUND1_DIR))

import run_hattrick_strict2x_research as base
import run_experiment as round1
from frameworks.hattrick_system import Hattrick
from hybrid_model import FrozenHighEnsemble, clear_transformer_caches, module_state_sha256
from penalty import PENALTIES, OrderPenalty, order_hinge
from utils.AdamOptimizer import ADAMOptimizer
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster
from utils.robust_proj_utils import assign_gradients_and_step, project_gradients_one_optimizer_robust
from utils.training_utils import loss_mf, loss_mlu


TOPOLOGY = "geant_priomask500_shared_load2x_train"
OUTPUT_ROOT = THIS_DIR / "artifacts"
WARMSTART_ROOT = ROUND1_DIR / "artifacts" / "level2_proxy" / "baseline"
PHASE_A_SOURCE_DESCRIPTION = "round-1 unchanged-Hattrick matched baseline checkpoint"
LAMBDA_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
HIGH_GATE = 0.98
HIGH_GATE_CONSECUTIVE_EPOCHS = 2
CALIBRATION_BATCHES = 8
GRAD_CLIP_NORM = 25.0
HIGH_INVARIANCE_TOLERANCE = 1e-5

LEVELS = {
    1: {
        "label": "level1_correctness",
        "train": (0, 32),
        "validation": (32, 40),
        "evaluation": (32, 40),
        "phase_b_epochs": 2,
        "selection_eligible": False,
    },
    2: {
        "label": "level2_proxy",
        "train": (0, 160),
        "validation": (160, 200),
        "evaluation": (200, 250),
        "phase_b_epochs": 6,
        "selection_eligible": True,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def build_props(level: int, device: torch.device):
    props = round1.build_props(level, device)
    props.research_return_admitted = False
    props.research_return_policy = False
    return props


def source_hashes() -> dict[str, str]:
    return {
        "round2/run_round2.py": sha256(Path(__file__).resolve()),
        "round2/hybrid_model.py": sha256(THIS_DIR / "hybrid_model.py"),
        "round2/penalty.py": sha256(THIS_DIR / "penalty.py"),
        "frameworks/hattrick_system.py": sha256(ROOT / "frameworks" / "hattrick_system.py"),
        "utils/robust_proj_utils.py": sha256(ROOT / "utils" / "robust_proj_utils.py"),
    }


def warmstart_path(seed: int) -> Path:
    return WARMSTART_ROOT / f"seed_{seed}" / "best_model.pt"


def warmstart_gate_audit(seed: int) -> dict:
    checkpoint_path = warmstart_path(seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No phase-A Hattrick checkpoint for seed {seed}: {checkpoint_path}"
        )
    run_dir = checkpoint_path.parent
    history = read_csv(run_dir / "train_history.csv")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_epoch = int(checkpoint["epoch"])
    through_checkpoint = [row for row in history if int(row["epoch"]) <= checkpoint_epoch]
    if len(through_checkpoint) < HIGH_GATE_CONSECUTIVE_EPOCHS:
        raise RuntimeError("Phase-A history is too short to audit the High gate")
    trailing = through_checkpoint[-HIGH_GATE_CONSECUTIVE_EPOCHS:]
    trailing_high = [float(row["high_norm_mean"]) for row in trailing]
    passed = all(value >= HIGH_GATE for value in trailing_high)
    if not passed:
        raise RuntimeError(
            f"High gate failed for seed {seed}: trailing validation means={trailing_high}"
        )
    summary_path = run_dir / f"validation_epoch_{checkpoint_epoch:03d}_summary.json"
    validation = json.loads(summary_path.read_text(encoding="utf-8"))
    validation_high = next(
        row for row in validation["classes"] if row["class"] == "High"
    )
    return {
        "passed": True,
        "threshold": HIGH_GATE,
        "consecutive_epochs": HIGH_GATE_CONSECUTIVE_EPOCHS,
        "checkpoint_epoch": checkpoint_epoch,
        "trailing_validation_high_norm_mean": trailing_high,
        "selected_validation_high_norm_mean": float(validation_high["norm_fulfill_mean"]),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
    }


def load_frozen_ensemble(props, seed: int, device: torch.device) -> FrozenHighEnsemble:
    checkpoint = torch.load(warmstart_path(seed), map_location=device, weights_only=False)
    student = Hattrick(props).to(device=device, dtype=props.dtype)
    student.load_state_dict(checkpoint["model_state_dict"])
    teacher = copy.deepcopy(student)
    ensemble = FrozenHighEnsemble(teacher=teacher, student=student)
    ensemble.to(device=device, dtype=props.dtype)
    ensemble.assert_teacher_immutable()
    return ensemble


def model_forward(model, props, dataset, values, path_masks):
    return base.model_forward(model, props, dataset, values, path_masks)


def training_objectives(
    model: FrozenHighEnsemble,
    props,
    dataset,
    values,
    path_masks,
    penalty_kind: str,
) -> tuple[tuple[torch.Tensor, ...], tuple[float, ...], OrderPenalty]:
    opt2 = values[9]
    opt3 = values[10]
    opt1_mf = values[11]
    opt2_mf = values[12]
    opt3_mf = values[13]
    output, _ = model_forward(model, props, dataset, values, path_masks)
    (
        _edges_high,
        edges_high_medium,
        edges_all,
        _edges_high_final,
        _edges_high_medium_final,
        all_traffic,
        _admitted_high,
        admitted_medium,
        admitted_low,
    ) = output
    # Slot 1 is intentionally zero: High is protected by architecture, not an
    # optimization compromise.  Keeping a graph-connected zero lets the
    # repository's ordered projector retain its four-objective interface.
    loss1 = admitted_medium.sum() * 0.0
    loss2, value2 = loss_mlu(edges_high_medium, opt2)
    loss3, value3 = loss_mf(all_traffic, opt3_mf)
    loss4, value4 = loss_mlu(edges_all, opt3)
    order = order_hinge(
        admitted_medium,
        admitted_low,
        opt2_mf - opt1_mf,
        opt3_mf - opt2_mf,
        kind=penalty_kind,
    )
    return (loss1, loss2, loss3, loss4), (0.0, value2, value3, value4), order


def flattened_gradient(model: torch.nn.Module, loss: torch.Tensor) -> torch.Tensor:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
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
    if torch.isfinite(norm).item() and float(norm.item()) > GRAD_CLIP_NORM:
        flattened = flattened * (GRAD_CLIP_NORM / float(norm.item()))
    return flattened


def project_away(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(reference)
    if float(norm.item()) <= 1e-12:
        return vector
    return vector - torch.dot(vector, reference) / (torch.dot(reference, reference) + 1e-9) * reference


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float((torch.dot(left, right) / denominator).item())


def gradient_probe(model, losses, order_loss, actual_lambda: float) -> dict[str, float]:
    _loss1, loss2, loss3, loss4 = losses
    g2 = flattened_gradient(model.student, loss2)
    g3 = flattened_gradient(model.student, loss3)
    g4 = flattened_gradient(model.student, loss4)
    go = flattened_gradient(model.student, order_loss)
    projected_flow = project_away(g3, g2)
    projected_order = project_away(go, g2)
    combined = project_away(g3 + actual_lambda * go, g2)
    return {
        "gradient_norm_high_medium": float(torch.linalg.vector_norm(g2).item()),
        "gradient_norm_total_flow": float(torch.linalg.vector_norm(g3).item()),
        "gradient_norm_all_mlu": float(torch.linalg.vector_norm(g4).item()),
        "gradient_norm_order": float(torch.linalg.vector_norm(go).item()),
        "projected_gradient_norm_total_flow": float(torch.linalg.vector_norm(projected_flow).item()),
        "projected_gradient_norm_order": float(torch.linalg.vector_norm(projected_order).item()),
        "projected_gradient_norm_combined": float(torch.linalg.vector_norm(combined).item()),
        "cosine_order_high_medium": cosine(go, g2),
        "cosine_order_total_flow": cosine(go, g3),
        "cosine_projected_order_total_flow": cosine(projected_order, projected_flow),
    }


def calibration_path(penalty_kind: str) -> Path:
    label = "full_bidirectional_hinge" if penalty_kind == "full_hinge" else penalty_kind
    return OUTPUT_ROOT / "level0_calibration" / label / "calibration.json"


def run_calibration(penalty_kind: str, force: bool = False) -> Path:
    output = calibration_path(penalty_kind)
    if output.exists() and not force:
        print(f"[skip] calibration exists: {output}", flush=True)
        return output
    seed = 490
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = build_props(2, device)
    gate = warmstart_gate_audit(seed)
    model = load_frozen_ensemble(props, seed, device).train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    dataset = DM_Dataset_within_Cluster(
        props, 0, 0, CALIBRATION_BATCHES * props.batch_size
    )
    path_masks = base.move_dataset_static(dataset, device)
    loader = round1.data_loader(dataset, props.batch_size, False, seed)
    rows = []
    for batch_index, inputs in enumerate(loader):
        values = base.unpack_to_device(inputs, props)
        losses, _reported, order = training_objectives(
            model, props, dataset, values, path_masks, penalty_kind
        )
        probe = gradient_probe(model, losses, order.loss, 1.0)
        denominator = probe["projected_gradient_norm_order"]
        ratio = probe["projected_gradient_norm_total_flow"] / (denominator + 1e-12)
        if not math.isfinite(ratio) or denominator <= 1e-12:
            raise RuntimeError(f"Unusable order gradient on calibration batch {batch_index}")
        rows.append(
            {
                "batch": batch_index,
                "lambda_ratio": ratio,
                "order_loss": float(order.loss.detach().item()),
                "active_fraction": float(order.active_fraction.detach().item()),
                "medium_norm_mean": float(order.medium_norm.detach().mean().item()),
                "low_norm_mean": float(order.low_norm.detach().mean().item()),
                "low_gradient_scale_mean": float(
                    order.low_gradient_scale.detach().mean().item()
                ),
                **probe,
            }
        )
    lambda_reference = float(np.median([row["lambda_ratio"] for row in rows]))
    result = {
        "mechanism": "freeze High policy after validation gate; order hinge keeps Medium and Low gradients",
        "penalty": penalty_kind,
        "seed": seed,
        "gate": gate,
        "batches": len(rows),
        "lambda_reference": lambda_reference,
        "multipliers": list(LAMBDA_MULTIPLIERS),
        "actual_lambdas": {
            multiplier_label(value): lambda_reference * value
            for value in LAMBDA_MULTIPLIERS
        },
        "rows": rows,
        "teacher_state_sha256": model.teacher_state_sha256,
        "source_sha256": source_hashes(),
    }
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
    return output


def multiplier_label(value: float) -> str:
    return "multiplier_" + format(float(value), ".12g").replace(".", "p")


def load_lambda(penalty_kind: str, multiplier: float) -> tuple[float, dict | None]:
    if multiplier == 0.0:
        return 0.0, None
    if not calibration_path(penalty_kind).exists():
        run_calibration(penalty_kind)
    calibration = json.loads(calibration_path(penalty_kind).read_text(encoding="utf-8"))
    return float(calibration["lambda_reference"]) * multiplier, calibration


def evaluation_diagnostics(rows: list[dict]) -> dict[str, float]:
    return round1.evaluation_diagnostics(rows)


def evaluate(model, props, dataset, start_index):
    clear_transformer_caches(model)
    rows, summary = base.evaluate(model, props, dataset, start_index)
    return rows, summary, evaluation_diagnostics(rows)


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def checkpoint_rank(summary: list[dict], diagnostics: dict) -> tuple:
    indexed = summary_index(summary)
    feasible = (
        float(indexed["High"]["norm_fulfill_mean"]) >= HIGH_GATE
        and max(float(row["max_admitted_capacity_ratio"]) for row in summary) <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
    )
    return (
        int(feasible),
        float(indexed["Medium"]["norm_fulfill_p10"]),
        float(indexed["Medium"]["norm_fulfill_p1"]),
        -float(diagnostics["inversion_positive_gap_mean"]),
    )


def train_epoch(
    model, props, dataset, loader, optimizer, penalty_kind: str, actual_lambda: float
) -> dict:
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    path_masks = base.move_dataset_static(dataset, props.device)
    expected_teacher_hash = model.teacher_state_sha256
    totals: dict[str, float] = {}
    count = 0
    first_probe: dict[str, float] = {}
    for batch_index, inputs in enumerate(loader):
        values = base.unpack_to_device(inputs, props)
        losses, reported, order = training_objectives(
            model, props, dataset, values, path_masks, penalty_kind
        )
        loss1, loss2, loss3, loss4 = losses
        combined_loss3 = loss3 if actual_lambda == 0.0 else loss3 + actual_lambda * order.loss
        objectives = (loss1, loss2, combined_loss3, loss4)
        if any(not torch.isfinite(loss).item() for loss in objectives):
            raise RuntimeError("Non-finite phase-B objective")
        if batch_index == 0:
            first_probe = gradient_probe(model, losses, order.loss, actual_lambda)
        final_grads, shapes, *_ = project_gradients_one_optimizer_robust(
            model.student, loss1, loss2, combined_loss3, loss4, optimizer
        )
        if not torch.isfinite(final_grads).all().item():
            raise RuntimeError("Non-finite phase-B projected gradient")
        assign_gradients_and_step(model.student, final_grads, optimizer, shapes)
        batch_metrics = {
            "reported_high_medium_mlu": float(reported[1]),
            "reported_total_flow": float(reported[2]),
            "reported_all_mlu": float(reported[3]),
            "order_loss": float(order.loss.detach().item()),
            "order_active_fraction": float(order.active_fraction.detach().item()),
            "train_medium_norm_mean": float(order.medium_norm.detach().mean().item()),
            "train_low_norm_mean": float(order.low_norm.detach().mean().item()),
            "train_raw_gap_mean": float(order.raw_gap.detach().mean().item()),
            "low_gradient_scale_mean": float(
                order.low_gradient_scale.detach().mean().item()
            ),
        }
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    props.research_return_admitted = False
    model.assert_teacher_immutable()
    if model.teacher_state_sha256 != expected_teacher_hash:
        raise RuntimeError("Teacher identity changed during training")
    result = {key: value / max(count, 1) for key, value in totals.items()}
    result.update(first_probe)
    result["train_batches"] = count
    result["teacher_immutable"] = 1
    return result


def run_directory(level: int, penalty_kind: str, multiplier: float, seed: int) -> Path:
    level_root = OUTPUT_ROOT / LEVELS[level]["label"]
    if multiplier == 0.0 or penalty_kind == "full_hinge":
        return level_root / multiplier_label(multiplier) / f"seed_{seed}"
    return level_root / penalty_kind / multiplier_label(multiplier) / f"seed_{seed}"


def _safe_remove(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise RuntimeError(f"Refusing unsafe removal: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def run_one(
    level: int, penalty_kind: str, multiplier: float, seed: int, force: bool = False
) -> Path:
    spec = LEVELS[level]
    actual_lambda, calibration = load_lambda(penalty_kind, multiplier)
    run_dir = run_directory(level, penalty_kind, multiplier, seed)
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not force:
        print(f"[skip] complete {run_dir}", flush=True)
        return run_dir
    if force:
        _safe_remove(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = build_props(level, device)
    gate = warmstart_gate_audit(seed)
    train_start, train_end = spec["train"]
    val_start, val_end = spec["validation"]
    eval_start, eval_end = spec["evaluation"]
    config = {
        "level": level,
        "label": spec["label"],
        "seed": seed,
        "topology": TOPOLOGY,
        "paths_per_pair": 8,
        "phase_a": {
            "source": PHASE_A_SOURCE_DESCRIPTION,
            "gate": gate,
        },
        "phase_b": {
            "epochs": spec["phase_b_epochs"],
            "train": [train_start, train_end],
            "validation": [val_start, val_end],
            "evaluation": [eval_start, eval_end],
            "high_policy": "frozen teacher, structurally excluded from student updates",
            "medium_low_policy": "trainable student",
            "penalty_kind": penalty_kind,
            "penalty": (
                "top-25%-mean(ReLU(N_low - N_mid)^2)"
                if penalty_kind == "tail_flow_balanced_squared"
                else "mean(ReLU(N_low - N_mid))"
            ) + "; neither Medium nor Low detached",
            "low_gradient_policy": (
                "unscaled full gradient"
                if penalty_kind == "full_hinge"
                else "nonzero Low gradient scaled by oracle_low/oracle_medium so raw-flow gradients are balanced"
            ),
            "lambda_multiplier": multiplier,
            "lambda_reference": None if calibration is None else calibration["lambda_reference"],
            "actual_lambda": actual_lambda,
            "ordered_objectives": [
                "zero High slot (High is hard-frozen)",
                "High+Medium pre-admission MLU",
                "total admitted flow + lambda * full bidirectional hinge",
                "all-class pre-admission MLU",
            ],
        },
        "selection_eligible": spec["selection_eligible"],
        "level1_data_note": (
            "correctness only: warm-start checkpoint has seen the Level-2 train/validation window"
            if level == 1
            else None
        ),
        "source_sha256": source_hashes(),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
    }
    write_json(run_dir / "config.json", config)

    train_dataset = DM_Dataset_within_Cluster(props, 0, train_start, train_end)
    val_dataset = DM_Dataset_within_Cluster(props, 0, val_start, val_end)
    eval_dataset = DM_Dataset_within_Cluster(props, 0, eval_start, eval_end)
    for dataset, expected in (
        (train_dataset, train_end - 1),
        (val_dataset, val_end - 1),
        (eval_dataset, eval_end - 1),
    ):
        if int(dataset.max_source_index_read) != expected:
            raise RuntimeError("Split-safe reader audit failed")

    model = load_frozen_ensemble(props, seed, device)
    optimizer = ADAMOptimizer(model.student.parameters(), lr=props.lr)
    initial_teacher_hash = model.teacher_state_sha256
    initial_rows, initial_summary, initial_diagnostics = evaluate(
        model, props, eval_dataset, eval_start
    )
    write_csv(run_dir / "initial_evaluation_metrics.csv", initial_rows)
    write_json(
        run_dir / "initial_evaluation_summary.json",
        {"classes": initial_summary, "diagnostics": initial_diagnostics},
    )

    history: list[dict] = read_csv(run_dir / "train_history.csv")
    best_rank = None
    best_epoch = None
    best_path = run_dir / "best_model.pt"
    final_path = run_dir / "final_model.pt"
    start_epoch = 1
    if final_path.exists() and not force:
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_path.exists():
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            best_rank = tuple(best_checkpoint["rank"])
            best_epoch = int(best_checkpoint["epoch"])
        print(f"[resume] {run_dir} at phase-B epoch {start_epoch}", flush=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, int(spec["phase_b_epochs"]) + 1):
        loader = round1.data_loader(
            train_dataset, props.batch_size, True, seed + epoch * 1009
        )
        train_metrics = train_epoch(
            model,
            props,
            train_dataset,
            loader,
            optimizer,
            penalty_kind,
            actual_lambda,
        )
        validation_rows, validation_summary, validation_diagnostics = evaluate(
            model, props, val_dataset, val_start
        )
        write_csv(run_dir / f"validation_epoch_{epoch:03d}_metrics.csv", validation_rows)
        write_json(
            run_dir / f"validation_epoch_{epoch:03d}_summary.json",
            {"classes": validation_summary, "diagnostics": validation_diagnostics},
        )
        indexed = summary_index(validation_summary)
        rank = checkpoint_rank(validation_summary, validation_diagnostics)
        row = {
            "epoch": epoch,
            "actual_lambda": actual_lambda,
            "high_norm_mean": indexed["High"]["norm_fulfill_mean"],
            "medium_norm_mean": indexed["Medium"]["norm_fulfill_mean"],
            "medium_norm_p1": indexed["Medium"]["norm_fulfill_p1"],
            "medium_norm_p10": indexed["Medium"]["norm_fulfill_p10"],
            "low_norm_mean": indexed["Low"]["norm_fulfill_mean"],
            "low_fulfill_mean": indexed["Low"]["fulfill_ratio_mean"],
            "max_disabled_flow": max(item["max_disabled_flow"] for item in validation_summary),
            "max_post_admission_mlu": max(
                item["max_admitted_capacity_ratio"] for item in validation_summary
            ),
            **validation_diagnostics,
            **train_metrics,
        }
        history = [item for item in history if int(item["epoch"]) < epoch]
        history.append(row)
        write_csv(run_dir / "train_history.csv", history)
        payload = {
            "epoch": epoch,
            "rank": rank,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "teacher_state_sha256": model.teacher_state_sha256,
        }
        torch.save(payload, final_path)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            torch.save(payload, best_path)
        print(
            f"[round2 {spec['label']}] multiplier={multiplier:g} seed={seed} "
            f"epoch={epoch}/{spec['phase_b_epochs']} high={row['high_norm_mean']:.4f} "
            f"mid_p10={row['medium_norm_p10']:.4f} low={row['low_norm_mean']:.4f} "
            f"inv={row['inversion_positive_gap_mean']:.4f} best={best_epoch}",
            flush=True,
        )

    evaluations = []
    for name, path in (("final", final_path), ("best", best_path)):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.assert_teacher_immutable()
        rows, summary, diagnostics = evaluate(model, props, eval_dataset, eval_start)
        write_csv(run_dir / f"{name}_evaluation_metrics.csv", rows)
        write_json(
            run_dir / f"{name}_evaluation_summary.json",
            {"classes": summary, "diagnostics": diagnostics},
        )
        initial_high = {
            int(row["snapshot"]): float(row["admitted_traffic"])
            for row in initial_rows
            if row["class"] == "High"
        }
        high_delta = max(
            abs(float(row["admitted_traffic"]) - initial_high[int(row["snapshot"])])
            for row in rows
            if row["class"] == "High"
        )
        evaluations.append(
            {
                "checkpoint": name,
                "epoch": int(checkpoint["epoch"]),
                "classes": summary,
                "diagnostics": diagnostics,
                "max_high_admitted_delta_from_frozen_initial": high_delta,
            }
        )

    if model.teacher_state_sha256 != initial_teacher_hash:
        raise RuntimeError("Teacher hash differs from the phase-B initial teacher")
    if (
        max(item["max_high_admitted_delta_from_frozen_initial"] for item in evaluations)
        > HIGH_INVARIANCE_TOLERANCE
    ):
        raise RuntimeError("High admitted traffic changed despite hard freeze")
    complete = {
        "best_epoch": best_epoch,
        "best_rank": best_rank,
        "actual_lambda": actual_lambda,
        "teacher_state_sha256": initial_teacher_hash,
        "teacher_immutable": True,
        "high_admission_invariance_tolerance": HIGH_INVARIANCE_TOLERANCE,
        "runtime_seconds": time.perf_counter() - started,
        "evaluations": evaluations,
    }
    write_json(complete_path, complete)
    print(f"[ok] complete {run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Round 2: freeze High after its validation gate, then train full M/L hinge"
    )
    parser.add_argument("--level", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--penalty", choices=PENALTIES, default="full_hinge")
    parser.add_argument("--lambda-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.level == 0:
        if args.seed != 490:
            raise SystemExit("Level 0 is calibrated once at seed 490")
        run_calibration(args.penalty, force=args.force)
        return
    if args.lambda_multiplier < 0:
        raise SystemExit("--lambda-multiplier must be nonnegative")
    if args.lambda_multiplier != 0 and args.lambda_multiplier not in LAMBDA_MULTIPLIERS:
        raise SystemExit(f"Multiplier must be 0 or one of {LAMBDA_MULTIPLIERS}")
    run_one(
        args.level,
        args.penalty,
        args.lambda_multiplier,
        args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
