from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))

import run_hattrick_strict2x_research as base
from frameworks.hattrick_system import Hattrick
from penalties import PENALTIES, PenaltyResult, priority_order_penalty
from utils.AdamOptimizer import ADAMOptimizer
from utils.args_parser import parse_args
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster, custom_collate
from utils.robust_proj_utils import assign_gradients_and_step, project_gradients_one_optimizer_robust
from utils.training_utils import loss_mf, loss_mlu


TOPOLOGY = "geant_priomask500_shared_load2x_train"
K = 8
CLASSES = ("High", "Medium", "Low")
OUTPUT_ROOT = THIS_DIR / "artifacts"
CALIBRATION_BATCHES = 8
LAMBDA_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
GRAD_CLIP_NORM = 25.0

LEVELS = {
    1: {
        "label": "level1_correctness",
        "train": (0, 32),
        "validation": (32, 40),
        "evaluation": (32, 40),
        "epochs": 2,
    },
    2: {
        "label": "level2_proxy",
        "train": (0, 160),
        "validation": (160, 200),
        "evaluation": (200, 250),
        "epochs": 12,
    },
    3: {
        "label": "level3_validation_only",
        "train": (0, 350),
        "validation": (350, 400),
        "evaluation": (350, 400),
        "epochs": 30,
    },
    4: {
        "label": "level4_confirmation",
        "train": (0, 350),
        "validation": (350, 400),
        "evaluation": (400, 500),
        "epochs": 60,
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
    spec = LEVELS.get(level, LEVELS[1])
    train_start, train_end = spec["train"]
    val_start, val_end = spec["validation"]
    props = parse_args(
        [
            "--topo", TOPOLOGY,
            "--mode", "train",
            "--epochs", str(spec["epochs"]),
            "--batch_size", "8",
            "--num_paths_per_pair", str(K),
            "--num_transformer_layers", "3",
            "--num_gnn_layers", "3",
            "--num_mlp1_hidden_layers", "2",
            "--num_mlp2_hidden_layers", "2",
            "--rau1", "3",
            "--rau2", "3",
            "--rau3", "3",
            "--train_clusters", "0",
            "--train_start_indices", str(train_start),
            "--train_end_indices", str(train_end),
            "--val_clusters", "0",
            "--val_start_indices", str(val_start),
            "--val_end_indices", str(val_end),
            "--pred", "1",
            "--dynamic", "0",
            "--lr", "0.0005",
            "--pred_type", "esm",
            "--initial_training", "1",
            "--violation", "1",
            "--path_mask", "0",
        ]
    )
    props.device = device
    props.dtype = torch.float32
    props.research_return_admitted = False
    props.research_return_policy = False
    return props


def data_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
        generator=generator,
    )


def multiplier_label(value: float) -> str:
    compact = format(float(value), ".12g").replace("-", "m").replace(".", "p")
    return f"multiplier_{compact}"


def run_directory(level: int, penalty: str, multiplier: float, seed: int) -> Path:
    level_dir = OUTPUT_ROOT / LEVELS[level]["label"]
    if float(multiplier) == 0.0:
        return level_dir / "baseline" / f"seed_{seed}"
    return level_dir / penalty / multiplier_label(multiplier) / f"seed_{seed}"


def calibration_path(penalty: str) -> Path:
    return OUTPUT_ROOT / "level0_calibration" / penalty / "calibration.json"


def source_hashes() -> dict[str, str]:
    return {
        "shared2x_order_regularizer/run_experiment.py": sha256(Path(__file__).resolve()),
        "shared2x_order_regularizer/penalties.py": sha256(THIS_DIR / "penalties.py"),
        "frameworks/hattrick_system.py": sha256(ROOT / "frameworks" / "hattrick_system.py"),
        "utils/training_utils.py": sha256(ROOT / "utils" / "training_utils.py"),
        "utils/robust_proj_utils.py": sha256(ROOT / "utils" / "robust_proj_utils.py"),
        "utils/build_dataset_within_cluster.py": sha256(ROOT / "utils" / "build_dataset_within_cluster.py"),
    }


def unpack_to_device(inputs, props):
    return base.unpack_to_device(inputs, props)


def model_forward(model, props, dataset, values, path_masks):
    return base.model_forward(model, props, dataset, values, path_masks)


def training_objectives(model, props, dataset, values, path_masks, penalty: str):
    (
        _node_features,
        _capacities,
        _tm1,
        _tm1_pred,
        _tm2,
        _tm2_pred,
        _tm3,
        _tm3_pred,
        opt1,
        opt2,
        opt3,
        _opt1_mf,
        opt2_mf,
        opt3_mf,
        _snapshots,
    ) = values
    output, _ = model_forward(model, props, dataset, values, path_masks)
    (
        edges_util_high,
        edges_util_high_medium,
        edges_util_all,
        _edges_util_high_final,
        _edges_util_high_medium_final,
        all_traffic,
        _admitted_high,
        admitted_medium,
        admitted_low,
    ) = output
    loss1, value1 = loss_mlu(edges_util_high, opt1)
    loss2, value2 = loss_mlu(edges_util_high_medium, opt2)
    loss3, value3 = loss_mf(all_traffic, opt3_mf)
    loss4, value4 = loss_mlu(edges_util_all, opt3)
    penalty_result = priority_order_penalty(
        admitted_medium,
        admitted_low,
        opt2_mf - values[11],
        opt3_mf - opt2_mf,
        penalty,
    )
    return (loss1, loss2, loss3, loss4), (value1, value2, value3, value4), penalty_result


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


def project_one(vector: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    denominator = torch.dot(reference, reference) + 1e-9
    return vector - (torch.dot(reference, vector) / denominator) * reference


def project_against_first_two(
    vector: torch.Tensor, first: torch.Tensor, second: torch.Tensor
) -> torch.Tensor:
    valid = []
    if float(torch.linalg.vector_norm(first).item()) > 1e-12:
        valid.append(first)
    second_projected = project_one(second, first) if valid else second
    if float(torch.linalg.vector_norm(second_projected).item()) > 1e-12:
        valid.append(second_projected)
    if not valid:
        return vector
    matrix = torch.stack(valid, dim=1).to(dtype=torch.float64)
    q, r = torch.linalg.qr(matrix, mode="reduced")
    independent = torch.abs(torch.diag(r)) > 1e-8
    q = q[:, independent]
    projected = vector.to(dtype=torch.float64) - q @ (q.t() @ vector.to(dtype=torch.float64))
    return projected.to(dtype=vector.dtype)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float((torch.dot(left, right) / denominator).item())


def gradient_diagnostics(
    model: torch.nn.Module,
    losses: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    penalty_loss: torch.Tensor,
    actual_lambda: float,
) -> dict:
    loss1, loss2, loss3, loss4 = losses
    gradients = [flattened_gradient(model, loss) for loss in losses]
    regularizer_gradient = flattened_gradient(model, penalty_loss)
    combined_third_gradient = flattened_gradient(model, loss3 + actual_lambda * penalty_loss)
    projected_base = project_against_first_two(gradients[2], gradients[0], gradients[1])
    projected_regularizer = project_against_first_two(
        regularizer_gradient, gradients[0], gradients[1]
    )
    projected_combined = project_against_first_two(
        combined_third_gradient, gradients[0], gradients[1]
    )
    names = ("high_mlu", "high_medium_mlu", "total_flow", "all_mlu")
    result = {
        f"gradient_norm_{name}": float(torch.linalg.vector_norm(gradient).item())
        for name, gradient in zip(names, gradients)
    }
    result.update(
        {
            "gradient_norm_order": float(torch.linalg.vector_norm(regularizer_gradient).item()),
            "projected_gradient_norm_total_flow": float(torch.linalg.vector_norm(projected_base).item()),
            "projected_gradient_norm_order": float(torch.linalg.vector_norm(projected_regularizer).item()),
            "projected_gradient_norm_combined_third": float(torch.linalg.vector_norm(projected_combined).item()),
            "cosine_order_high": cosine(regularizer_gradient, gradients[0]),
            "cosine_order_high_medium": cosine(regularizer_gradient, gradients[1]),
            "cosine_order_total_flow": cosine(regularizer_gradient, gradients[2]),
            "cosine_projected_order_total_flow": cosine(projected_regularizer, projected_base),
        }
    )
    return result


def run_calibration(penalty: str, force: bool = False) -> Path:
    output = calibration_path(penalty)
    if output.exists() and not force:
        print(f"[skip] calibration exists: {output}", flush=True)
        return output
    set_seed(490)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = build_props(1, device)
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    dataset = DM_Dataset_within_Cluster(props, 0, 0, CALIBRATION_BATCHES * props.batch_size)
    path_masks = base.move_dataset_static(dataset, device)
    model = Hattrick(props).to(device=device, dtype=props.dtype).train()
    loader = data_loader(dataset, props.batch_size, False, 490)
    rows = []
    for batch_index, inputs in enumerate(loader):
        values = unpack_to_device(inputs, props)
        losses, _reported, penalty_result = training_objectives(
            model, props, dataset, values, path_masks, penalty
        )
        diagnostics = gradient_diagnostics(model, losses, penalty_result.loss, 1.0)
        numerator = diagnostics["projected_gradient_norm_total_flow"]
        denominator = diagnostics["projected_gradient_norm_order"]
        ratio = numerator / (denominator + 1e-12)
        if not math.isfinite(ratio) or denominator <= 1e-12:
            raise RuntimeError(
                f"Penalty {penalty} has no usable projected gradient on calibration batch {batch_index}"
            )
        rows.append(
            {
                "batch": batch_index,
                "lambda_ratio": ratio,
                "penalty": float(penalty_result.loss.detach().item()),
                "active_fraction": float(penalty_result.active_fraction.detach().item()),
                "raw_gap_mean": float(penalty_result.raw_gap.detach().mean().item()),
                "positive_gap_mean": float(penalty_result.positive_gap.detach().mean().item()),
                **diagnostics,
            }
        )
    if len(rows) != CALIBRATION_BATCHES:
        raise RuntimeError(f"Expected {CALIBRATION_BATCHES} calibration batches, got {len(rows)}")
    lambda_reference = float(np.median([float(row["lambda_ratio"]) for row in rows]))
    result = {
        "penalty": penalty,
        "seed": 490,
        "topology": TOPOLOGY,
        "batches": CALIBRATION_BATCHES,
        "batch_size": props.batch_size,
        "lambda_reference": lambda_reference,
        "multipliers": list(LAMBDA_MULTIPLIERS),
        "actual_lambdas": {
            multiplier_label(multiplier): lambda_reference * multiplier
            for multiplier in LAMBDA_MULTIPLIERS
        },
        "rows": rows,
        "source_sha256": source_hashes(),
    }
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)
    return output


def load_lambda(penalty: str, multiplier: float) -> tuple[float, dict | None]:
    if float(multiplier) == 0.0:
        return 0.0, None
    path = calibration_path(penalty)
    if not path.exists():
        run_calibration(penalty)
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration["penalty"] != penalty or int(calibration["seed"]) != 490:
        raise RuntimeError(f"Calibration identity mismatch: {path}")
    return float(calibration["lambda_reference"]) * float(multiplier), calibration


def penalty_stats(result: PenaltyResult) -> dict[str, float]:
    raw_gap = result.raw_gap.detach().to(dtype=torch.float32)
    return {
        "order_penalty": float(result.loss.detach().item()),
        "order_active_fraction": float(result.active_fraction.detach().item()),
        "order_raw_gap_mean": float(raw_gap.mean().item()),
        "order_raw_gap_p90": float(torch.quantile(raw_gap, 0.9).item()),
        "train_medium_norm_mean": float(result.medium_norm.detach().mean().item()),
        "train_low_norm_mean": float(result.low_norm.detach().mean().item()),
    }


def train_epoch(
    model,
    props,
    dataset,
    loader,
    optimizer,
    penalty: str,
    actual_lambda: float,
) -> dict:
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    path_masks = base.move_dataset_static(dataset, props.device)
    totals: dict[str, float] = {}
    count = 0
    probe: dict[str, float] = {}
    for batch_index, inputs in enumerate(loader):
        values = unpack_to_device(inputs, props)
        losses, reported, result = training_objectives(
            model, props, dataset, values, path_masks, penalty
        )
        loss1, loss2, loss3, loss4 = losses
        # Preserve the exact unchanged-Hattrick graph for the matched baseline.
        # A syntactic ``loss3 + 0 * regularizer`` can perturb signed near-zero
        # components enough to change Adam's first update.
        combined_loss3 = loss3 if actual_lambda == 0.0 else loss3 + actual_lambda * result.loss
        all_losses = (loss1, loss2, combined_loss3, loss4)
        if any(not torch.isfinite(loss).item() for loss in all_losses):
            raise RuntimeError("Training produced a non-finite objective")
        # Keep the zero-weight matched baseline byte-for-byte on the original
        # backward path; the extra diagnostic backward traversals are reserved
        # for actual regularized candidates.
        if batch_index == 0 and actual_lambda != 0.0:
            probe = gradient_diagnostics(model, losses, result.loss, actual_lambda)
        final_grads, shapes, *_ = project_gradients_one_optimizer_robust(
            model, loss1, loss2, combined_loss3, loss4, optimizer
        )
        if not torch.isfinite(final_grads).all().item():
            raise RuntimeError("Training produced non-finite projected gradients")
        assign_gradients_and_step(model, final_grads, optimizer, shapes)

        current = {
            "reported_high_mlu": float(reported[0]),
            "reported_high_medium_mlu": float(reported[1]),
            "reported_total_flow": float(reported[2]),
            "reported_all_mlu": float(reported[3]),
            **penalty_stats(result),
        }
        for key, value in current.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
    props.research_return_admitted = False
    means = {key: value / max(count, 1) for key, value in totals.items()}
    means.update(probe)
    means["train_batches"] = count
    return means


def train_epoch_dispatch(
    epoch: int,
    total_epochs: int,
    model,
    props,
    dataset,
    loader,
    optimizer,
    penalty: str,
    actual_lambda: float,
) -> dict:
    if actual_lambda == 0.0:
        # The matched baseline deliberately calls the repository's unchanged
        # training function rather than merely constructing an algebraically
        # equivalent zero-weight graph.
        props.mode = "train"
        props.sim_mf_mlu = 0
        props.research_return_admitted = False
        base.unchanged_train(
            epoch - 1,
            total_epochs,
            model,
            props,
            [dataset],
            [loader],
            [optimizer],
        )
        return {"baseline_original_training_path": 1}
    return train_epoch(
        model,
        props,
        dataset,
        loader,
        optimizer,
        penalty,
        actual_lambda,
    )


def evaluation_diagnostics(rows: list[dict]) -> dict[str, float]:
    by_key = {(int(row["snapshot"]), row["class"]): row for row in rows}
    snapshots = sorted({int(row["snapshot"]) for row in rows})
    gaps = np.asarray(
        [
            float(by_key[(snapshot, "Low")]["norm_fulfill"])
            - float(by_key[(snapshot, "Medium")]["norm_fulfill"])
            for snapshot in snapshots
        ],
        dtype=np.float64,
    )
    return {
        "inversion_raw_gap_mean": float(gaps.mean()),
        "inversion_positive_gap_mean": float(np.maximum(gaps, 0.0).mean()),
        "inversion_violation_fraction": float((gaps > 0).mean()),
        "inversion_gap_p90": float(np.percentile(gaps, 90)),
        "inversion_gap_max": float(gaps.max()),
    }


def evaluate(model, props, dataset, start_index: int) -> tuple[list[dict], list[dict], dict]:
    rows, summary = base.evaluate(model, props, dataset, start_index)
    return rows, summary, evaluation_diagnostics(rows)


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def checkpoint_rank(summary: list[dict], diagnostics: dict) -> tuple:
    indexed = summary_index(summary)
    high = indexed["High"]
    medium = indexed["Medium"]
    feasible = (
        float(high["norm_fulfill_mean"]) >= 0.98
        and max(float(row["max_admitted_capacity_ratio"]) for row in summary) <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
    )
    if feasible:
        return (
            1,
            float(medium["norm_fulfill_p10"]),
            float(medium["norm_fulfill_p1"]),
            -float(diagnostics["inversion_positive_gap_mean"]),
        )
    return (
        0,
        float(high["norm_fulfill_mean"]),
        float(high["norm_fulfill_p10"]),
        float(medium["norm_fulfill_p10"]),
    )


def _safe_remove_run_dir(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise RuntimeError(f"Refusing to remove unsafe path: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def _critical_config(config: dict) -> tuple:
    return (
        config["level"],
        config["penalty"],
        float(config["lambda_multiplier"]),
        float(config["actual_lambda"]),
        config["seed"],
        config["topology"],
        tuple(config["train"]),
        tuple(config["validation"]),
        tuple(config["evaluation"]),
        config["epochs"],
    )


def run_one(level: int, penalty: str, multiplier: float, seed: int, force: bool = False) -> Path:
    if level not in LEVELS:
        raise ValueError(f"Training level must be one of {tuple(LEVELS)}")
    spec = LEVELS[level]
    actual_lambda, calibration = load_lambda(penalty, multiplier)
    effective_penalty = "none" if float(multiplier) == 0.0 else penalty
    run_dir = run_directory(level, penalty, multiplier, seed)
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not force:
        print(f"[skip] complete {run_dir}", flush=True)
        return run_dir
    if force:
        _safe_remove_run_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    props = build_props(level, device)
    train_start, train_end = spec["train"]
    val_start, val_end = spec["validation"]
    eval_start, eval_end = spec["evaluation"]
    config = {
        "level": level,
        "label": spec["label"],
        "penalty": effective_penalty,
        "penalty_implementation": penalty,
        "lambda_multiplier": float(multiplier),
        "lambda_reference": None if calibration is None else float(calibration["lambda_reference"]),
        "actual_lambda": actual_lambda,
        "seed": seed,
        "topology": TOPOLOGY,
        "paths_per_pair": K,
        "shared_paths": True,
        "train": [train_start, train_end],
        "validation": [val_start, val_end],
        "evaluation": [eval_start, eval_end],
        "evaluation_is_previously_viewed_confirmation": level == 4,
        "epochs": spec["epochs"],
        "batch_size": props.batch_size,
        "learning_rate": props.lr,
        "checkpoint_rule": "among checkpoints with High NormFulFill mean >=0.98, exact post-admission MLU <=1.0001, and disabled flow <=1e-8, maximize validation Medium NormFulFill P10 then P1 then minimize positive inversion gap; otherwise use a High-first diagnostic fallback",
        "objectives": [
            "High maximum-across-stages MLU",
            "High+Medium maximum-across-stages MLU",
            "total admitted flow + lambda * priority-order regularizer",
            "all-class MLU",
        ],
        "source_sha256": source_hashes(),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "data_access": {
            "max_source_index_read": max(train_end, val_end, eval_end) - 1,
            "reader_contract": "DM_Dataset_within_Cluster uses np.loadtxt skiprows/max_rows for every split",
        },
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        previous_config = json.loads(config_path.read_text(encoding="utf-8"))
        if _critical_config(previous_config) != _critical_config(config):
            raise RuntimeError(f"Existing run config does not match requested run: {run_dir}")
    else:
        write_json(config_path, config)

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

    model = Hattrick(props).to(device=device, dtype=props.dtype)
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    history_path = run_dir / "train_history.csv"
    history = read_csv(history_path)
    start_epoch = 1
    best_rank = None
    best_epoch = None
    final_model_path = run_dir / "final_model.pt"
    best_model_path = run_dir / "best_model.pt"
    if final_model_path.exists() and not force:
        checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
        if _critical_config(checkpoint["config"]) != _critical_config(config):
            raise RuntimeError("Resume checkpoint config mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_model_path.exists():
            best_checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
            best_rank = tuple(best_checkpoint["rank"])
            best_epoch = int(best_checkpoint["epoch"])
        print(f"[resume] {run_dir} at epoch {start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, int(spec["epochs"]) + 1):
        loader = data_loader(train_dataset, props.batch_size, True, seed + epoch * 1009)
        train_metrics = train_epoch_dispatch(
            epoch,
            int(spec["epochs"]),
            model,
            props,
            train_dataset,
            loader,
            optimizer,
            penalty,
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
        rank = checkpoint_rank(validation_summary, validation_diagnostics)
        indexed = summary_index(validation_summary)
        history_row = {
            "epoch": epoch,
            "actual_lambda": actual_lambda,
            "high_norm_mean": indexed["High"]["norm_fulfill_mean"],
            "medium_norm_mean": indexed["Medium"]["norm_fulfill_mean"],
            "medium_norm_p1": indexed["Medium"]["norm_fulfill_p1"],
            "medium_norm_p10": indexed["Medium"]["norm_fulfill_p10"],
            "low_norm_mean": indexed["Low"]["norm_fulfill_mean"],
            "low_fulfill_mean": indexed["Low"]["fulfill_ratio_mean"],
            "max_disabled_flow": max(row["max_disabled_flow"] for row in validation_summary),
            "max_post_admission_mlu": max(
                row["max_admitted_capacity_ratio"] for row in validation_summary
            ),
            "checkpoint_feasible": rank[0],
            **validation_diagnostics,
            **train_metrics,
        }
        history = [row for row in history if int(row["epoch"]) < epoch]
        history.append(history_row)
        write_csv(history_path, history)
        checkpoint_payload = {
            "epoch": epoch,
            "rank": rank,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }
        torch.save(checkpoint_payload, final_model_path)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            torch.save(checkpoint_payload, best_model_path)
        print(
            f"[{spec['label']}] penalty={effective_penalty} multiplier={multiplier:g} "
            f"seed={seed} epoch={epoch}/{spec['epochs']} high={history_row['high_norm_mean']:.4f} "
            f"mid_p10={history_row['medium_norm_p10']:.4f} low={history_row['low_norm_mean']:.4f} "
            f"inv={history_row['inversion_positive_gap_mean']:.4f} best={best_epoch}",
            flush=True,
        )

    evaluations = []
    final_reference_rows = None
    for checkpoint_name, path in (("final", final_model_path), ("best", best_model_path)):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        rows, summary, diagnostics = evaluate(model, props, eval_dataset, eval_start)
        for row in rows:
            row.update(
                {
                    "checkpoint": checkpoint_name,
                    "penalty": effective_penalty,
                    "lambda_multiplier": float(multiplier),
                    "actual_lambda": actual_lambda,
                    "seed": seed,
                    "level": level,
                }
            )
        write_csv(run_dir / f"{checkpoint_name}_evaluation_metrics.csv", rows)
        write_json(
            run_dir / f"{checkpoint_name}_evaluation_summary.json",
            {"classes": summary, "diagnostics": diagnostics},
        )
        evaluations.append(
            {
                "checkpoint": checkpoint_name,
                "epoch": int(checkpoint["epoch"]),
                "classes": summary,
                "diagnostics": diagnostics,
            }
        )
        if checkpoint_name == "final":
            final_reference_rows = rows

    recovery_checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
    recovery_model = Hattrick(props).to(device=device, dtype=props.dtype)
    recovery_optimizer = ADAMOptimizer(recovery_model.parameters(), lr=props.lr)
    recovery_model.load_state_dict(recovery_checkpoint["model_state_dict"])
    recovery_optimizer.load_state_dict(recovery_checkpoint["optimizer_state_dict"])
    recovery_rows, _, _ = evaluate(recovery_model, props, eval_dataset, eval_start)
    numeric_fields = (
        "admitted_traffic",
        "demand",
        "fulfill_ratio",
        "oracle_admitted_traffic",
        "norm_fulfill",
        "raw_mlu",
        "oracle_mlu",
        "normalized_mlu",
        "disabled_flow",
        "admitted_capacity_ratio",
    )
    max_recovery_delta = max(
        abs(float(expected[field]) - float(actual[field]))
        for expected, actual in zip(final_reference_rows, recovery_rows)
        for field in numeric_fields
    )
    recovery = {
        "checkpoint_epoch": int(recovery_checkpoint["epoch"]),
        "model_and_optimizer_loaded": True,
        "optimizer_parameter_states": len(recovery_optimizer.state),
        "evaluation_rows": len(recovery_rows),
        "max_numeric_delta_from_saved_final_evaluation": max_recovery_delta,
        # CUDA replay of the same checkpoint can differ by a few float32 ULPs
        # in aggregate link-load metrics.  Keep the observed delta and use a
        # tolerance far below any reporting precision or feasibility guard.
        "numeric_tolerance": 1e-5,
        "passes": len(recovery_rows) == len(final_reference_rows) and max_recovery_delta <= 1e-5,
    }
    write_json(run_dir / "checkpoint_recovery.json", recovery)
    if not recovery["passes"]:
        raise RuntimeError(f"Checkpoint recovery mismatch: {recovery}")

    complete = {
        "best_epoch": best_epoch,
        "best_rank": best_rank,
        "selection_status": "guard_feasible" if best_rank and best_rank[0] else "high_first_fallback",
        "actual_lambda": actual_lambda,
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "checkpoint_recovery": recovery,
        "evaluations": evaluations,
    }
    write_json(complete_path, complete)
    print(f"[ok] complete {run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared-2x Hattrick priority-order regularizer")
    parser.add_argument("--level", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--penalty", choices=PENALTIES, required=True)
    parser.add_argument("--lambda-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.level == 0:
        if args.seed != 490:
            raise SystemExit("Level 0 calibration is frozen to seed 490")
        run_calibration(args.penalty, force=args.force)
        return
    if args.lambda_multiplier < 0:
        raise SystemExit("--lambda-multiplier must be nonnegative")
    if args.lambda_multiplier != 0 and args.lambda_multiplier not in LAMBDA_MULTIPLIERS:
        raise SystemExit(f"Multiplier must be one of 0 or {LAMBDA_MULTIPLIERS}")
    run_one(args.level, args.penalty, args.lambda_multiplier, args.seed, force=args.force)


if __name__ == "__main__":
    main()
