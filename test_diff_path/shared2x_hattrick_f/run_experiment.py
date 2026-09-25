from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
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
ORDER_DIR = TEST_DIR / "shared2x_order_epsilon"
FULL_DIR = TEST_DIR / "shared2x_full_objectives"
SHARED_DIR = TEST_DIR / "shared2x_order_regularizer"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(ORDER_DIR))
sys.path.insert(0, str(FULL_DIR))
sys.path.insert(0, str(SHARED_DIR))

from frameworks.hattrick_system import Hattrick  # noqa: E402
from ordered_projection import ordered_project_gradients, projection_diagnostics  # noqa: E402
from utils.AdamOptimizer import ADAMOptimizer  # noqa: E402
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster  # noqa: E402
from utils.robust_proj_utils import assign_gradients_and_step  # noqa: E402
from utils.training_utils import loss_mf  # noqa: E402

# Import the already-audited strict-ESM shared-path runtime without copying its
# simulator or evaluation code.
import run_experiment as shared  # noqa: E402


LEVEL = 3
SEED = 490
EPOCHS = 15
HIGH_TRAIN_FLOOR = 0.998
HIGH_HARD_FLOOR = 0.995
HIGH_HARD_TOLERANCE = 1e-4
LOW_MEAN_BUDGET = 0.02
GUARDED_OBJECTIVE_NAMES = ("HighFloorGuard", "HighMLUGuard", "Fhm", "Fhml")
FH_RELEASE_OBJECTIVE_NAMES = ("Fh", "Fhm", "Fhml")
OUTPUT_ROOT = THIS_DIR / "artifacts" / "level3_validation_only"
PHASE_A_PATH = (
    ORDER_DIR
    / "artifacts"
    / "phase_a"
    / "level3_validation_only"
    / "seed_490"
    / "best_model.pt"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


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


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def slack_label(slack: float | None) -> str:
    return "released" if slack is None else f"guard_{slack:.3f}".replace(".", "p")


def source_hashes() -> dict[str, str]:
    files = {
        "run_experiment.py": Path(__file__).resolve(),
        "ordered_projection.py": FULL_DIR / "ordered_projection.py",
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "strict_esm_runtime.py": SHARED_DIR / "run_experiment.py",
    }
    return {name: sha256(path) for name, path in files.items()}


def move_static(dataset, device: torch.device):
    return shared.base.move_dataset_static(dataset, device)


def forward_components(model, props, dataset, values, path_masks) -> dict[str, torch.Tensor]:
    output, _ = shared.model_forward(model, props, dataset, values, path_masks)
    (
        edges_high,
        _edges_high_medium,
        _edges_all,
        _edges_high_final,
        _edges_high_medium_final,
        all_traffic,
        admitted_high,
        admitted_medium,
        _admitted_low,
    ) = output
    return {
        "edges_high": edges_high,
        "all_traffic": all_traffic,
        "admitted_high": admitted_high,
        "admitted_medium": admitted_medium,
        "opt1_mf": values[11],
        "opt2_mf": values[12],
        "opt3_mf": values[13],
    }


def normalized_high(admitted_high: torch.Tensor, oracle_high: torch.Tensor) -> torch.Tensor:
    admitted = admitted_high.reshape(admitted_high.shape[0], -1).sum(dim=1)
    oracle = oracle_high.detach().reshape(-1).clamp_min(1e-12)
    return admitted / oracle


def high_floor_guard(
    admitted_high: torch.Tensor,
    oracle_high: torch.Tensor,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = normalized_high(admitted_high, oracle_high)
    violation = torch.relu(float(floor) - normalized)
    # Linear hinge gives the repair direction a useful magnitude.  Once every
    # sample is safe its gradient is exactly zero and is skipped by projection.
    return violation.mean(), normalized, violation


def high_mlu_guard(
    edges_high: torch.Tensor,
    teacher_edges_high: torch.Tensor,
    slack: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate = edges_high.reshape(edges_high.shape[0], -1).amax(dim=1)
    teacher = (
        teacher_edges_high.detach()
        .reshape(teacher_edges_high.shape[0], -1)
        .amax(dim=1)
        .clamp_min(1e-12)
    )
    if slack is None:
        # Connected zero: ordered projection observes a true inactive objective
        # without special-casing the rest of the training code.
        zero = candidate.sum() * 0.0
        return zero, candidate / teacher, torch.zeros_like(candidate)
    ratio = candidate / teacher
    violation = torch.relu(ratio - (1.0 + float(slack)))
    return violation.mean(), ratio, violation


def build_objectives(
    components: dict[str, torch.Tensor],
    teacher_edges_high: torch.Tensor,
    floor: float,
    mlu_slack: float | None,
    keep_fh: bool,
):
    if keep_fh:
        loss_fh, value_fh = loss_mf(
            components["admitted_high"], components["opt1_mf"].detach()
        )
        loss_fhm, value_fhm = loss_mf(
            components["admitted_high"] + components["admitted_medium"],
            components["opt2_mf"].detach(),
        )
        loss_fhml, value_fhml = loss_mf(
            components["all_traffic"], components["opt3_mf"].detach()
        )
        return (
            (loss_fh, loss_fhm, loss_fhml),
            {
                "reported_Fh": float(value_fh),
                "reported_Fhm": float(value_fhm),
                "reported_Fhml": float(value_fhml),
            },
        )
    high_guard, high_norm, high_violation = high_floor_guard(
        components["admitted_high"], components["opt1_mf"], floor
    )
    mlu_guard, high_mlu_ratio, mlu_violation = high_mlu_guard(
        components["edges_high"], teacher_edges_high, mlu_slack
    )
    loss_fhm, value_fhm = loss_mf(
        components["admitted_high"] + components["admitted_medium"],
        components["opt2_mf"].detach(),
    )
    loss_fhml, value_fhml = loss_mf(
        components["all_traffic"], components["opt3_mf"].detach()
    )
    losses = (high_guard, mlu_guard, loss_fhm, loss_fhml)
    diagnostics = {
        "reported_high_norm": float(high_norm.detach().mean().item()),
        "reported_high_floor_active_fraction": float(
            (high_violation.detach() > 0).to(dtype=torch.float32).mean().item()
        ),
        "reported_high_floor_violation_mean": float(high_violation.detach().mean().item()),
        "reported_high_mlu_ratio": float(high_mlu_ratio.detach().mean().item()),
        "reported_high_mlu_guard_active_fraction": float(
            (mlu_violation.detach() > 0).to(dtype=torch.float32).mean().item()
        ),
        "reported_high_mlu_violation_mean": float(mlu_violation.detach().mean().item()),
        "reported_Fhm": float(value_fhm),
        "reported_Fhml": float(value_fhml),
    }
    return losses, diagnostics


def train_epoch(
    model,
    teacher,
    props,
    dataset,
    loader,
    optimizer,
    floor: float,
    mlu_slack: float | None,
    keep_fh: bool,
) -> dict[str, float]:
    objective_names = FH_RELEASE_OBJECTIVE_NAMES if keep_fh else GUARDED_OBJECTIVE_NAMES
    model.train()
    teacher.eval()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    props.research_return_policy = False
    path_masks = move_static(dataset, props.device)
    totals: dict[str, float] = {}
    first_probe: dict[str, float] = {}
    count = 0
    for batch_index, inputs in enumerate(loader):
        values = shared.unpack_to_device(inputs, props)
        candidate = forward_components(model, props, dataset, values, path_masks)
        if mlu_slack is None:
            # Full release does not need the frozen reference forward pass.
            teacher_edges_high = candidate["edges_high"].detach()
        else:
            with torch.no_grad():
                teacher_components = forward_components(
                    teacher, props, dataset, values, path_masks
                )
            teacher_edges_high = teacher_components["edges_high"]
        losses, diagnostics = build_objectives(
            candidate,
            teacher_edges_high,
            floor,
            mlu_slack,
            keep_fh,
        )
        if any(not torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError("non-finite Hattrick-f objective")
        projected = ordered_project_gradients(model, losses)
        if batch_index == 0:
            first_probe = projection_diagnostics(projected, objective_names)
        assign_gradients_and_step(
            model, projected.final_gradient, optimizer, projected.parameter_shapes
        )
        for name, value in diagnostics.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1
    props.research_return_admitted = False
    means = {name: value / max(count, 1) for name, value in totals.items()}
    means.update(first_probe)
    means["train_batches"] = count
    return means


def class_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def evaluate(model, props, dataset, start: int):
    return shared.evaluate(model, props, dataset, start)


def checkpoint_rank(
    rows: list[dict],
    summary: list[dict],
    baseline_summary: list[dict],
) -> tuple:
    indexed = class_index(summary)
    baseline = class_index(baseline_summary)
    high_values = np.asarray(
        [float(row["norm_fulfill"]) for row in rows if row["class"] == "High"]
    )
    high_safe = bool((high_values >= HIGH_HARD_FLOOR - HIGH_HARD_TOLERANCE).all())
    low_safe = (
        float(indexed["Low"]["norm_fulfill_mean"])
        >= float(baseline["Low"]["norm_fulfill_mean"]) - LOW_MEAN_BUDGET
    )
    simulator_safe = (
        max(float(row["max_admitted_capacity_ratio"]) for row in summary) <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
    )
    return (
        int(high_safe and low_safe and simulator_safe),
        float(indexed["Medium"]["norm_fulfill_mean"]),
        float(indexed["Medium"]["norm_fulfill_p10"]),
        float(indexed["Medium"]["norm_fulfill_p1"]),
        float(indexed["High"]["norm_fulfill_mean"]),
        float(indexed["Low"]["norm_fulfill_mean"]),
    )


def capture_high_policy(model, props, dataset, start: int) -> dict[int, np.ndarray]:
    model.eval()
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_admitted = False
    props.research_return_policy = True
    path_masks = move_static(dataset, props.device)
    loader = shared.data_loader(dataset, 1, False, 0)
    captured: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for local_index, inputs in enumerate(loader):
            values = shared.unpack_to_device(inputs, props)
            policies, _ = shared.model_forward(model, props, dataset, values, path_masks)
            captured[start + local_index] = (
                policies[0].reshape(-1).detach().cpu().numpy().astype(np.float64)
            )
    props.research_return_policy = False
    return captured


def policy_shape_metrics(policy: np.ndarray, k: int = 8) -> dict[str, float]:
    weights = policy.reshape(-1, k)
    totals = weights.sum(axis=1)
    active = totals > 1e-12
    probs = np.zeros_like(weights)
    probs[active] = weights[active] / totals[active, None]
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1) / math.log(k)
    return {
        "normalized_entropy": float(entropy[active].mean()),
        "top1_share": float(probs[active].max(axis=1).mean()),
        "effective_paths": float(np.exp(entropy[active] * math.log(k)).mean()),
    }


def route_metrics(
    initial: dict[int, np.ndarray], final: dict[int, np.ndarray], k: int = 8
) -> tuple[list[dict], dict[str, float]]:
    rows: list[dict] = []
    for snapshot in sorted(initial):
        left = initial[snapshot].reshape(-1, k)
        right = final[snapshot].reshape(-1, k)
        left_sum = left.sum(axis=1)
        right_sum = right.sum(axis=1)
        active = (left_sum > 1e-12) & (right_sum > 1e-12)
        left_prob = left[active] / left_sum[active, None]
        right_prob = right[active] / right_sum[active, None]
        tv = 0.5 * np.abs(right_prob - left_prob).sum(axis=1)
        weight = left_sum[active]
        weighted_tv = float(np.average(tv, weights=weight))
        argmax_change = float(np.mean(left_prob.argmax(1) != right_prob.argmax(1)))
        left_shape = policy_shape_metrics(initial[snapshot], k)
        right_shape = policy_shape_metrics(final[snapshot], k)
        rows.append(
            {
                "snapshot": snapshot,
                "demand_weighted_tv": weighted_tv,
                "argmax_change_fraction": argmax_change,
                "initial_entropy": left_shape["normalized_entropy"],
                "final_entropy": right_shape["normalized_entropy"],
                "initial_top1_share": left_shape["top1_share"],
                "final_top1_share": right_shape["top1_share"],
            }
        )
    return rows, {
        "demand_weighted_tv_mean": float(np.mean([row["demand_weighted_tv"] for row in rows])),
        "argmax_change_fraction_mean": float(
            np.mean([row["argmax_change_fraction"] for row in rows])
        ),
        "initial_entropy_mean": float(np.mean([row["initial_entropy"] for row in rows])),
        "final_entropy_mean": float(np.mean([row["final_entropy"] for row in rows])),
        "entropy_delta": float(
            np.mean([row["final_entropy"] - row["initial_entropy"] for row in rows])
        ),
        "initial_top1_share_mean": float(
            np.mean([row["initial_top1_share"] for row in rows])
        ),
        "final_top1_share_mean": float(
            np.mean([row["final_top1_share"] for row in rows])
        ),
    }


def load_baseline_summary() -> tuple[list[dict], list[dict], dict]:
    checkpoint = torch.load(PHASE_A_PATH, map_location="cpu", weights_only=False)
    epoch = int(checkpoint["epoch"])
    summary_path = PHASE_A_PATH.parent / f"validation_epoch_{epoch:03d}_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics_path = PHASE_A_PATH.parent / f"validation_epoch_{epoch:03d}_metrics.csv"
    rows = read_csv(metrics_path)
    return rows, payload["classes"], payload["diagnostics"]


def run_one(
    mlu_slack: float | None, *, epochs: int, force: bool, keep_fh: bool
) -> Path:
    if not PHASE_A_PATH.exists():
        raise FileNotFoundError(f"six-loss Phase-A checkpoint missing: {PHASE_A_PATH}")
    label = "fh_release" if keep_fh else slack_label(mlu_slack)
    run_dir = OUTPUT_ROOT / label / f"seed_{SEED}"
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not force:
        print(f"[skip] {run_dir}", flush=True)
        return run_dir
    if force and run_dir.exists():
        # Keep destructive scope explicit and below the experiment artifact root.
        import shutil

        resolved = run_dir.resolve()
        if OUTPUT_ROOT.resolve() not in resolved.parents:
            raise RuntimeError(f"unsafe removal target: {resolved}")
        shutil.rmtree(resolved)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    props = shared.build_props(LEVEL, device)
    spec = shared.LEVELS[LEVEL]
    train_start, train_end = spec["train"]
    val_start, val_end = spec["validation"]
    train_dataset = DM_Dataset_within_Cluster(props, 0, train_start, train_end)
    val_dataset = DM_Dataset_within_Cluster(props, 0, val_start, val_end)
    if int(train_dataset.max_source_index_read) != train_end - 1:
        raise RuntimeError("train split audit failed")
    if int(val_dataset.max_source_index_read) != val_end - 1:
        raise RuntimeError("validation split audit failed")

    phase_a = torch.load(PHASE_A_PATH, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(phase_a["model_state_dict"])
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    # Intentional reset: old Adam moments encode the removed MLU objectives.
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    baseline_rows, baseline_summary, baseline_diagnostics = load_baseline_summary()
    config = {
        "method": "Hattrick-f",
        "level": LEVEL,
        "seed": SEED,
        "strict_inference": "ESM prediction is the only traffic input to routing policy",
        "training_labels": "actual traffic is used only by differentiable admission/loss",
        "train": [train_start, train_end],
        "validation": [val_start, val_end],
        "epochs": epochs,
        "learning_rate": props.lr,
        "batch_size": props.batch_size,
        "phase_a": {
            "checkpoint": str(PHASE_A_PATH.resolve()),
            "sha256": sha256(PHASE_A_PATH),
            "objectives": ["Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml"],
        },
        "phase_f": {
            "optimizer_reset": True,
            "ordered_objectives": list(
                FH_RELEASE_OBJECTIVE_NAMES if keep_fh else GUARDED_OBJECTIVE_NAMES
            ),
            "persistent_fh_maximization": keep_fh,
            "high_training_floor": HIGH_TRAIN_FLOOR,
            "high_hard_validation_floor": HIGH_HARD_FLOOR,
            "high_mlu_teacher_relative_slack": mlu_slack,
            "persistent_mlu_minimization": False,
            "all_parameters_trainable": True,
        },
        "selection": "hard High/Low/simulator guard, then Medium mean/P10/P1",
        "source_sha256": source_hashes(),
    }
    write_json(run_dir / "config.json", config)

    initial_policy = capture_high_policy(model, props, val_dataset, val_start)
    history: list[dict] = []
    best_rank = None
    best_epoch = None
    best_path = run_dir / "best_model.pt"
    final_path = run_dir / "final_model.pt"
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loader = shared.data_loader(
            train_dataset, props.batch_size, True, SEED + epoch * 1009
        )
        train_metrics = train_epoch(
            model,
            teacher,
            props,
            train_dataset,
            loader,
            optimizer,
            HIGH_TRAIN_FLOOR,
            mlu_slack,
            keep_fh,
        )
        rows, summary, diagnostics = evaluate(model, props, val_dataset, val_start)
        write_csv(run_dir / f"validation_epoch_{epoch:03d}_metrics.csv", rows)
        write_json(
            run_dir / f"validation_epoch_{epoch:03d}_summary.json",
            {"classes": summary, "diagnostics": diagnostics},
        )
        indexed = class_index(summary)
        high_values = [
            float(row["norm_fulfill"]) for row in rows if row["class"] == "High"
        ]
        rank = checkpoint_rank(rows, summary, baseline_summary)
        row = {
            "epoch": epoch,
            "eligible": int(rank[0]),
            "high_norm_mean": indexed["High"]["norm_fulfill_mean"],
            "high_norm_p10": indexed["High"]["norm_fulfill_p10"],
            "high_norm_min": min(high_values),
            "medium_norm_mean": indexed["Medium"]["norm_fulfill_mean"],
            "medium_norm_p10": indexed["Medium"]["norm_fulfill_p10"],
            "medium_norm_p1": indexed["Medium"]["norm_fulfill_p1"],
            "low_norm_mean": indexed["Low"]["norm_fulfill_mean"],
            **diagnostics,
            **train_metrics,
        }
        history.append(row)
        write_csv(run_dir / "train_history.csv", history)
        payload = {
            "epoch": epoch,
            "rank": rank,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }
        torch.save(payload, final_path)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            torch.save(payload, best_path)
        print(
            f"[Hattrick-f {label}] epoch={epoch}/{epochs} eligible={rank[0]} "
            f"H={row['high_norm_mean']:.6f} Hmin={row['high_norm_min']:.6f} "
            f"M={row['medium_norm_mean']:.6f} L={row['low_norm_mean']:.6f} "
            f"best={best_epoch}",
            flush=True,
        )

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    best_rows, best_summary, best_diagnostics = evaluate(
        model, props, val_dataset, val_start
    )
    final_policy = capture_high_policy(model, props, val_dataset, val_start)
    route_rows, route_summary = route_metrics(initial_policy, final_policy)
    write_csv(run_dir / "best_metrics.csv", best_rows)
    write_json(
        run_dir / "best_summary.json",
        {"classes": best_summary, "diagnostics": best_diagnostics},
    )
    write_csv(run_dir / "route_reconstruction.csv", route_rows)
    indexed_base = class_index(baseline_summary)
    indexed_best = class_index(best_summary)
    comparison = {
        cls: {
            "baseline_norm_mean": float(indexed_base[cls]["norm_fulfill_mean"]),
            "hattrick_f_norm_mean": float(indexed_best[cls]["norm_fulfill_mean"]),
            "delta": float(
                indexed_best[cls]["norm_fulfill_mean"]
                - indexed_base[cls]["norm_fulfill_mean"]
            ),
        }
        for cls in ("High", "Medium", "Low")
    }
    complete = {
        "status": "COMPLETE",
        "best_epoch": int(best["epoch"]),
        "best_rank": list(best["rank"]),
        "baseline_diagnostics": baseline_diagnostics,
        "comparison": comparison,
        "route_reconstruction": route_summary,
        "runtime_seconds": time.perf_counter() - started,
        "best_checkpoint_sha256": sha256(best_path),
    }
    write_json(complete_path, complete)
    print(json.dumps(complete, indent=2), flush=True)
    return run_dir


def parse_slack(value: str) -> float | None:
    if value.lower() in {"none", "released", "off"}:
        return None
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("MLU slack must be non-negative")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Level-3 validation of Hattrick-f MLU release"
    )
    parser.add_argument(
        "--mlu-slack",
        nargs="+",
        type=parse_slack,
        default=[None, 0.01, 0.03],
        help="teacher-relative one-sided High MLU slack; use 'none' for full release",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--keep-fh",
        action="store_true",
        help="keep continuous Fh priority and remove only all MLU objectives",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.keep_fh and len(args.mlu_slack) != 1:
        parser.error("--keep-fh takes exactly one (ignored) --mlu-slack value")
    for slack in args.mlu_slack:
        run_one(slack, epochs=args.epochs, force=args.force, keep_fh=args.keep_fh)


if __name__ == "__main__":
    main()
