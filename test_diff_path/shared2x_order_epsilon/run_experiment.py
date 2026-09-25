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
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
FULL_DIR = TEST_DIR / "shared2x_full_objectives"
SHARED_DIR = TEST_DIR / "shared2x_order_regularizer"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(FULL_DIR))

from frameworks.hattrick_system import Hattrick  # noqa: E402
from lp_oracle import run_level0, shared  # noqa: E402
from objectives import (  # noqa: E402
    APPROACHES,
    EPSILONS,
    CONTROL_NAMES,
    build_objectives,
    validate_objective_names,
)
from ordered_projection import (  # noqa: E402
    ordered_project_gradients,
    projection_diagnostics,
)
from utils.AdamOptimizer import ADAMOptimizer  # noqa: E402
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster  # noqa: E402
from utils.robust_proj_utils import assign_gradients_and_step  # noqa: E402


TOPOLOGY = "geant_priomask500_shared_load2x_train"
K = 8
OUTPUT_ROOT = THIS_DIR / "artifacts"
PHASE_A_GATE_MEAN = 0.9965
PHASE_A_GATE_P10 = 0.995
LOW_FULFILL_BUDGET = 0.02
EPSILON_HARD_TOL = 1e-4

LEVELS = {
    1: {
        "label": "level1_correctness",
        "train": (0, 32),
        "validation": (32, 40),
        "evaluation": (32, 40),
        "phase_a_epochs": 2,
        "phase_b_epochs": 2,
        "selection_eligible": False,
    },
    2: {
        "label": "level2_proxy",
        "train": (0, 160),
        "validation": (160, 200),
        "evaluation": (200, 250),
        "phase_a_epochs": 12,
        "phase_b_epochs": 6,
        "selection_eligible": True,
    },
    3: {
        "label": "level3_validation_only",
        "train": (0, 350),
        "validation": (350, 400),
        "evaluation": (350, 400),
        "phase_a_epochs": 30,
        "phase_b_epochs": 15,
        "selection_eligible": True,
    },
    4: {
        "label": "level4_confirmation",
        "train": (0, 350),
        "validation": (350, 400),
        "evaluation": (400, 500),
        "phase_a_epochs": 60,
        "phase_b_epochs": 30,
        "selection_eligible": True,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "objectives.py": THIS_DIR / "objectives.py",
        "lp_oracle.py": THIS_DIR / "lp_oracle.py",
        "ordered_projection.py": FULL_DIR / "ordered_projection.py",
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "shared2x_order_regularizer/run_experiment.py": SHARED_DIR / "run_experiment.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def epsilon_label(epsilon: float | None) -> str:
    if epsilon is None:
        return "none"
    return format(float(epsilon), ".12g").replace(".", "p")


def phase_a_directory(level: int, seed: int) -> Path:
    return OUTPUT_ROOT / "phase_a" / LEVELS[level]["label"] / f"seed_{seed}"


def run_directory(level: int, approach: str, epsilon: float | None, seed: int) -> Path:
    root = OUTPUT_ROOT / "phase_b" / LEVELS[level]["label"] / approach
    if approach == "epsilon":
        root = root / f"epsilon_{epsilon_label(epsilon)}"
    return root / f"seed_{seed}"


def safe_remove(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"refusing unsafe removal: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def build_props(level: int, device: torch.device):
    props = shared.build_props(level, device)
    props.research_return_admitted = False
    props.research_return_policy = False
    return props


def datasets_for_level(props, level: int):
    spec = LEVELS[level]
    train = DM_Dataset_within_Cluster(props, 0, *spec["train"])
    validation = DM_Dataset_within_Cluster(props, 0, *spec["validation"])
    evaluation = DM_Dataset_within_Cluster(props, 0, *spec["evaluation"])
    for dataset, split in (
        (train, spec["train"]),
        (validation, spec["validation"]),
        (evaluation, spec["evaluation"]),
    ):
        if int(dataset.max_source_index_read) != split[1] - 1:
            raise RuntimeError("split-safe reader audit failed")
    return train, validation, evaluation


def forward_components(model, props, dataset, values, path_masks):
    output, _ = shared.model_forward(model, props, dataset, values, path_masks)
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
        "admitted_high": admitted_high,
        "admitted_medium": admitted_medium,
        "all_traffic": all_traffic,
        "opt1": values[8],
        "opt2": values[9],
        "opt3": values[10],
        "opt1_mf": values[11],
        "opt2_mf": values[12],
        "opt3_mf": values[13],
    }


def train_epoch(model, props, dataset, loader, optimizer, approach: str, epsilon: float | None):
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    props.research_return_policy = False
    path_masks = shared.base.move_dataset_static(dataset, props.device)
    totals: dict[str, float] = {}
    first_probe: dict[str, float] = {}
    count = 0
    for batch_index, inputs in enumerate(loader):
        values = shared.unpack_to_device(inputs, props)
        losses, reported, names, guard = build_objectives(
            approach,
            epsilon,
            **forward_components(model, props, dataset, values, path_masks),
        )
        validate_objective_names(approach, names)
        if any(not torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError("non-finite ordered objective")
        projected = ordered_project_gradients(model, losses)
        if batch_index == 0:
            first_probe = projection_diagnostics(projected, names)
        assign_gradients_and_step(
            model, projected.final_gradient, optimizer, projected.parameter_shapes
        )
        for name, value in zip(names, reported):
            totals[f"reported_{name}"] = totals.get(f"reported_{name}", 0.0) + float(value)
        if guard is not None:
            for name, value in (
                ("high_guard_loss", guard.loss.detach()),
                ("high_guard_active_fraction", guard.active_fraction.detach()),
                ("high_guard_slack_mean", guard.slack.detach().mean()),
                ("high_guard_slack_min", guard.slack.detach().min()),
            ):
                totals[name] = totals.get(name, 0.0) + float(value.item())
        count += 1
    props.research_return_admitted = False
    result = {key: value / max(count, 1) for key, value in totals.items()}
    result.update(first_probe)
    result["train_batches"] = count
    return result


def evaluate(model, props, dataset, start: int):
    rows, summary, diagnostics = shared.evaluate(model, props, dataset, start)
    high_values = [float(row["norm_fulfill"]) for row in rows if row["class"] == "High"]
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "high_norm_min": min(high_values),
            "high_norm_violation_fraction_eps_0p0025": float(
                np.mean(np.asarray(high_values) < 1 - 0.0025 - EPSILON_HARD_TOL)
            ),
            "high_norm_violation_fraction_eps_0p005": float(
                np.mean(np.asarray(high_values) < 1 - 0.005 - EPSILON_HARD_TOL)
            ),
            "high_norm_violation_fraction_eps_0p01": float(
                np.mean(np.asarray(high_values) < 1 - 0.01 - EPSILON_HARD_TOL)
            ),
        }
    )
    return rows, summary, diagnostics


def class_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def phase_a_rank(summary: list[dict]) -> tuple:
    indexed = class_index(summary)
    high = indexed["High"]
    safe = (
        max(float(row["max_admitted_capacity_ratio"]) for row in summary) <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
    )
    gate = (
        float(high["norm_fulfill_mean"]) >= PHASE_A_GATE_MEAN
        and float(high["norm_fulfill_p10"]) >= PHASE_A_GATE_P10
    )
    return (
        int(safe and gate),
        float(high["norm_fulfill_mean"]),
        float(high["norm_fulfill_p10"]),
        float(high["norm_fulfill_p1"]),
        float(indexed["Medium"]["norm_fulfill_mean"]),
    )


def phase_a_config(level: int, seed: int) -> dict:
    spec = LEVELS[level]
    return {
        "level": level,
        "label": spec["label"],
        "seed": seed,
        "topology": TOPOLOGY,
        "paths_per_pair": K,
        "load_factor": 2.0,
        "shared_paths": True,
        "train": list(spec["train"]),
        "validation": list(spec["validation"]),
        "evaluation": list(spec["evaluation"]),
        "phase_a_epochs": spec["phase_a_epochs"],
        "objectives": list(CONTROL_NAMES),
        "gate": {
            "high_norm_mean": PHASE_A_GATE_MEAN,
            "high_norm_p10": PHASE_A_GATE_P10,
        },
        "source_sha256": source_hashes(),
    }


def audit_phase_a_gate(level: int, seed: int, run_dir: Path) -> dict:
    """Re-audit an immutable Phase-A checkpoint against the current gate.

    Keeping this separate from Phase-A ``complete.json`` preserves the original
    0.9975 NO-GO while allowing an explicitly authorized gate revision.
    """
    checkpoint_path = run_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = int(checkpoint["epoch"])
    summary_path = run_dir / f"validation_epoch_{epoch:03d}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    high = next(row for row in summary["classes"] if row["class"] == "High")
    mean_value = float(high["norm_fulfill_mean"])
    p10_value = float(high["norm_fulfill_p10"])
    audit = {
        "status": "PASS" if mean_value >= PHASE_A_GATE_MEAN and p10_value >= PHASE_A_GATE_P10 else "FAIL",
        "passed": mean_value >= PHASE_A_GATE_MEAN and p10_value >= PHASE_A_GATE_P10,
        "checkpoint_epoch": epoch,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "high_norm_mean": mean_value,
        "high_norm_p10": p10_value,
        "mean_threshold": PHASE_A_GATE_MEAN,
        "p10_threshold": PHASE_A_GATE_P10,
        "mean_margin": mean_value - PHASE_A_GATE_MEAN,
        "p10_margin": p10_value - PHASE_A_GATE_P10,
        "policy_change": "user-authorized Mean gate relaxation from 0.9975 to 0.9965; P10 unchanged",
        "original_phase_a_complete": str((run_dir / "complete.json").resolve()),
    }
    audit_path = OUTPUT_ROOT / "gate_audits" / LEVELS[level]["label"] / f"seed_{seed}.json"
    write_json(audit_path, audit)
    return audit


def ensure_phase_a(level: int, seed: int, *, force: bool = False) -> Path:
    run_dir = phase_a_directory(level, seed)
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not force:
        return run_dir
    if force:
        safe_remove(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    props = build_props(level, device)
    train_dataset, validation_dataset, _ = datasets_for_level(props, level)
    config = phase_a_config(level, seed)
    write_json(run_dir / "config.json", config)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    final_path = run_dir / "final_model.pt"
    best_path = run_dir / "best_model.pt"
    history = read_csv(run_dir / "train_history.csv")
    start_epoch = 1
    best_rank = None
    best_epoch = None
    if final_path.exists() and not force:
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            best_rank = tuple(best["rank"])
            best_epoch = int(best["epoch"])
    started = time.perf_counter()
    for epoch in range(start_epoch, int(LEVELS[level]["phase_a_epochs"]) + 1):
        loader = shared.data_loader(
            train_dataset, props.batch_size, True, seed + epoch * 1009
        )
        metrics = train_epoch(
            model, props, train_dataset, loader, optimizer, "control", None
        )
        rows, summary, diagnostics = evaluate(
            model, props, validation_dataset, LEVELS[level]["validation"][0]
        )
        write_csv(run_dir / f"validation_epoch_{epoch:03d}_metrics.csv", rows)
        write_json(
            run_dir / f"validation_epoch_{epoch:03d}_summary.json",
            {"classes": summary, "diagnostics": diagnostics},
        )
        indexed = class_index(summary)
        rank = phase_a_rank(summary)
        history = [row for row in history if int(row["epoch"]) < epoch]
        history.append(
            {
                "epoch": epoch,
                "high_norm_mean": indexed["High"]["norm_fulfill_mean"],
                "high_norm_p1": indexed["High"]["norm_fulfill_p1"],
                "high_norm_p10": indexed["High"]["norm_fulfill_p10"],
                "medium_norm_mean": indexed["Medium"]["norm_fulfill_mean"],
                "medium_norm_p1": indexed["Medium"]["norm_fulfill_p1"],
                "medium_norm_p10": indexed["Medium"]["norm_fulfill_p10"],
                "max_post_admission_mlu": max(
                    float(row["max_admitted_capacity_ratio"]) for row in summary
                ),
                "max_disabled_flow": max(float(row["max_disabled_flow"]) for row in summary),
                **diagnostics,
                **metrics,
            }
        )
        write_csv(run_dir / "train_history.csv", history)
        payload = {
            "epoch": epoch,
            "rank": rank,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": rng_state(),
            "config": config,
        }
        torch.save(payload, final_path)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            torch.save(payload, best_path)
        print(
            f"[phase-a {LEVELS[level]['label']}] seed={seed} "
            f"epoch={epoch}/{LEVELS[level]['phase_a_epochs']} "
            f"high={indexed['High']['norm_fulfill_mean']:.4f}/"
            f"{indexed['High']['norm_fulfill_p10']:.4f} "
            f"mid={indexed['Medium']['norm_fulfill_mean']:.4f} best={best_epoch}",
            flush=True,
        )
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    gate_passed = bool(int(best["rank"][0]))
    write_json(
        complete_path,
        {
            "status": "PASS" if gate_passed else "GATE_FAILED",
            "gate_passed": gate_passed,
            "best_epoch": int(best["epoch"]),
            "best_rank": list(best["rank"]),
            "runtime_seconds_this_invocation": time.perf_counter() - started,
            "best_checkpoint_sha256": sha256(best_path),
            "final_checkpoint_sha256": sha256(final_path),
        },
    )
    return run_dir


def high_gate_stats(rows: list[dict], epsilon: float) -> dict[str, float | bool]:
    values = np.asarray(
        [float(row["norm_fulfill"]) for row in rows if row["class"] == "High"],
        dtype=np.float64,
    )
    threshold = 1.0 - epsilon - EPSILON_HARD_TOL
    return {
        "epsilon_threshold": threshold,
        "high_norm_min": float(values.min()),
        "high_epsilon_violation_fraction": float((values < threshold).mean()),
        "high_epsilon_all_slices_pass": bool((values >= threshold).all()),
    }


def checkpoint_rank(
    approach: str,
    epsilon: float | None,
    rows: list[dict],
    summary: list[dict],
    initial_summary: list[dict],
) -> tuple:
    indexed = class_index(summary)
    initial = class_index(initial_summary)
    safe = (
        max(float(row["max_admitted_capacity_ratio"]) for row in summary) <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
        and float(indexed["Low"]["fulfill_ratio_mean"])
        >= float(initial["Low"]["fulfill_ratio_mean"]) - LOW_FULFILL_BUDGET
    )
    if approach == "epsilon":
        assert epsilon is not None
        high_stats = high_gate_stats(rows, epsilon)
        high_safe = bool(high_stats["high_epsilon_all_slices_pass"])
    elif approach == "swap":
        high_values = [
            float(row["norm_fulfill"]) for row in rows if row["class"] == "High"
        ]
        high_safe = (
            float(indexed["High"]["norm_fulfill_mean"])
            >= float(initial["High"]["norm_fulfill_mean"]) - 0.005
            and float(indexed["High"]["norm_fulfill_p10"])
            >= float(initial["High"]["norm_fulfill_p10"]) - 0.005
            and min(high_values) >= 0.98
        )
    else:
        high_safe = float(indexed["High"]["norm_fulfill_mean"]) >= 0.98
    return (
        int(safe and high_safe),
        float(indexed["Medium"]["norm_fulfill_p10"]),
        float(indexed["Medium"]["norm_fulfill_p1"]),
        float(indexed["Medium"]["norm_fulfill_mean"]),
        float(indexed["Low"]["fulfill_ratio_mean"]),
    )


def capture_high_state(model, props, dataset, start_index: int) -> dict[int, dict[str, np.ndarray]]:
    model.eval()
    props.mode = "test"
    path_masks = shared.base.move_dataset_static(dataset, props.device)
    loader = shared.data_loader(dataset, 1, False, 0)
    captured: dict[int, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        for local_index, inputs in enumerate(loader):
            values = shared.unpack_to_device(inputs, props)
            props.research_return_policy = True
            props.sim_mf_mlu = 0
            policies, _ = shared.model_forward(
                model, props, dataset, values, path_masks
            )
            props.research_return_policy = False
            props.sim_mf_mlu = 1
            admitted, capacities = shared.model_forward(
                model, props, dataset, values, path_masks
            )
            props.sim_mf_mlu = 0
            high_policy = policies[0].reshape(-1).detach().cpu().numpy().astype(np.float64)
            high_admitted = admitted[0].reshape(1, -1).to(dtype=torch.float32)
            link = torch.sparse.mm(
                dataset.pte.to(dtype=torch.float32).t(), high_admitted.t()
            ).reshape(-1)
            captured[start_index + local_index] = {
                "policy": high_policy,
                "admitted": high_admitted.reshape(-1).cpu().numpy().astype(np.float64),
                "link": link.cpu().numpy().astype(np.float64),
                "capacity": capacities.reshape(-1).cpu().numpy().astype(np.float64),
            }
    props.research_return_policy = False
    return captured


def route_change_artifacts(
    output_dir: Path,
    initial: dict[int, dict[str, np.ndarray]],
    final: dict[int, dict[str, np.ndarray]],
    initial_rows: list[dict],
    final_rows: list[dict],
) -> dict:
    initial_medium = {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in initial_rows
        if row["class"] == "Medium"
    }
    final_medium = {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in final_rows
        if row["class"] == "Medium"
    }
    snapshot_rows = []
    path_rows = []
    link_rows = []
    edge_index = None
    for snapshot in sorted(initial):
        before = initial[snapshot]
        after = final[snapshot]
        policy_l1 = float(np.abs(after["policy"] - before["policy"]).sum())
        before_argmax = before["policy"].reshape(-1, K).argmax(axis=1)
        after_argmax = after["policy"].reshape(-1, K).argmax(axis=1)
        argmax_changes = float(np.mean(before_argmax != after_argmax))
        admitted_delta = float(after["admitted"].sum() - before["admitted"].sum())
        link_l1 = float(np.abs(after["link"] - before["link"]).sum())
        medium_delta = final_medium[snapshot] - initial_medium[snapshot]
        snapshot_rows.append(
            {
                "snapshot": snapshot,
                "high_policy_l1": policy_l1,
                "high_main_path_change_fraction": argmax_changes,
                "high_admitted_delta": admitted_delta,
                "high_link_load_l1": link_l1,
                "medium_norm_delta": medium_delta,
            }
        )
        for path_id, (left, right) in enumerate(zip(before["policy"], after["policy"])):
            path_rows.append(
                {
                    "snapshot": snapshot,
                    "path_id": path_id,
                    "od_id": path_id // K,
                    "within_od_path": path_id % K,
                    "initial_high_policy": float(left),
                    "final_high_policy": float(right),
                    "delta": float(right - left),
                }
            )
        for link_id, (left, right, capacity) in enumerate(
            zip(before["link"], after["link"], before["capacity"])
        ):
            link_rows.append(
                {
                    "snapshot": snapshot,
                    "link_id": link_id,
                    "capacity": float(capacity),
                    "initial_high_admitted_load": float(left),
                    "final_high_admitted_load": float(right),
                    "delta": float(right - left),
                }
            )
    write_csv(output_dir / "route_snapshot_metrics.csv", snapshot_rows)
    write_csv(output_dir / "high_path_policy_changes.csv", path_rows)
    write_csv(output_dir / "high_link_load_changes.csv", link_rows)
    l1 = np.asarray([row["high_policy_l1"] for row in snapshot_rows])
    medium = np.asarray([row["medium_norm_delta"] for row in snapshot_rows])
    correlation = 0.0
    if len(l1) > 1 and np.std(l1) > 1e-12 and np.std(medium) > 1e-12:
        correlation = float(np.corrcoef(l1, medium)[0, 1])
    return {
        "high_policy_l1_mean": float(l1.mean()),
        "high_main_path_change_fraction_mean": float(
            np.mean([row["high_main_path_change_fraction"] for row in snapshot_rows])
        ),
        "high_admitted_delta_abs_max": float(
            max(abs(row["high_admitted_delta"]) for row in snapshot_rows)
        ),
        "high_link_load_l1_mean": float(
            np.mean([row["high_link_load_l1"] for row in snapshot_rows])
        ),
        "corr_high_policy_l1_vs_medium_norm_delta": correlation,
    }


def phase_b_config(
    level: int,
    approach: str,
    epsilon: float | None,
    seed: int,
    phase_a_path: Path,
) -> dict:
    spec = LEVELS[level]
    names = {
        "control": ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml"),
        "swap": ("Fh", "Fhm", "Uh", "Uhm", "Fhml", "Uhml"),
        "epsilon": ("HighGuard", "Fm", "Uhm", "Fhml", "Uhml", "Uh"),
    }[approach]
    return {
        "level": level,
        "label": spec["label"],
        "approach": approach,
        "epsilon": epsilon,
        "seed": seed,
        "topology": TOPOLOGY,
        "paths_per_pair": K,
        "train": list(spec["train"]),
        "validation": list(spec["validation"]),
        "evaluation": list(spec["evaluation"]),
        "evaluation_is_previously_viewed_confirmation": level == 4,
        "phase_b_epochs": spec["phase_b_epochs"],
        "all_parameters_trainable": True,
        "ordered_objectives": list(names),
        "epsilon_definition": (
            "per-snapshot admitted High / detached oracle High; train at 1-epsilon/2, hard gate at 1-epsilon-1e-4"
            if approach == "epsilon"
            else None
        ),
        "checkpoint_low_policy": (
            "The original Low budget remains a validation checkpoint-selection guard. "
            "It is diagnostic rather than a final rejection gate; cross-class impact is checked afterward "
            "using absolute admitted traffic versus matched control."
        ),
        "checkpoint_rank": (
            "High/Low/capacity/mask feasibility, then Medium P10/P1/Mean, then Low tie-breaker"
        ),
        "phase_a_checkpoint": str(phase_a_path.resolve()),
        "phase_a_checkpoint_sha256": sha256(phase_a_path),
        "source_sha256": source_hashes(),
    }


def run_phase_b(
    level: int,
    approach: str,
    epsilon: float | None,
    seed: int,
    *,
    force: bool = False,
) -> Path:
    phase_a_dir = ensure_phase_a(level, seed, force=False)
    phase_a_complete = json.loads((phase_a_dir / "complete.json").read_text(encoding="utf-8"))
    gate_audit = audit_phase_a_gate(level, seed, phase_a_dir)
    if not gate_audit["passed"] and level != 1:
        refusal = OUTPUT_ROOT / "refusals" / LEVELS[level]["label"] / f"seed_{seed}.json"
        write_json(
            refusal,
            {
                "status": "PHASE_A_GATE_FAILED",
                "phase_b_started": False,
                "phase_a_complete": phase_a_complete,
                "current_gate_audit": gate_audit,
            },
        )
        raise RuntimeError(f"Phase-A High gate failed; Phase B refused: {refusal}")
    phase_a_path = phase_a_dir / "best_model.pt"
    run_dir = run_directory(level, approach, epsilon, seed)
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not force:
        return run_dir
    if force:
        safe_remove(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = build_props(level, device)
    train_dataset, validation_dataset, evaluation_dataset = datasets_for_level(props, level)
    phase_a = torch.load(phase_a_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(phase_a["model_state_dict"])
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Phase B must not freeze any model parameter")
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    optimizer.load_state_dict(copy.deepcopy(phase_a["optimizer_state_dict"]))
    restore_rng_state(phase_a["rng_state"])
    config = phase_b_config(level, approach, epsilon, seed, phase_a_path)
    config["phase_a_gate_audit"] = gate_audit
    write_json(run_dir / "config.json", config)
    initial_parameter_hash = state_sha256(model.state_dict())
    initial_optimizer_repr_hash = hashlib.sha256(
        repr(optimizer.state_dict()).encode("utf-8")
    ).hexdigest()
    initial_val_rows, initial_val_summary, initial_val_diagnostics = evaluate(
        model, props, validation_dataset, LEVELS[level]["validation"][0]
    )
    initial_rows, initial_summary, initial_diagnostics = evaluate(
        model, props, evaluation_dataset, LEVELS[level]["evaluation"][0]
    )
    write_csv(run_dir / "initial_evaluation_metrics.csv", initial_rows)
    write_json(
        run_dir / "initial_evaluation_summary.json",
        {"classes": initial_summary, "diagnostics": initial_diagnostics},
    )
    initial_high_state = capture_high_state(
        model, props, evaluation_dataset, LEVELS[level]["evaluation"][0]
    )
    history = read_csv(run_dir / "train_history.csv")
    best_rank = None
    best_epoch = None
    start_epoch = 1
    final_path = run_dir / "final_model.pt"
    best_path = run_dir / "best_model.pt"
    if final_path.exists() and not force:
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            best_rank = tuple(best["rank"])
            best_epoch = int(best["epoch"])
    started = time.perf_counter()
    for epoch in range(start_epoch, int(LEVELS[level]["phase_b_epochs"]) + 1):
        loader = shared.data_loader(
            train_dataset, props.batch_size, True, seed + 100_000 + epoch * 1009
        )
        train_metrics = train_epoch(
            model, props, train_dataset, loader, optimizer, approach, epsilon
        )
        val_rows, val_summary, val_diagnostics = evaluate(
            model, props, validation_dataset, LEVELS[level]["validation"][0]
        )
        write_csv(run_dir / f"validation_epoch_{epoch:03d}_metrics.csv", val_rows)
        write_json(
            run_dir / f"validation_epoch_{epoch:03d}_summary.json",
            {"classes": val_summary, "diagnostics": val_diagnostics},
        )
        indexed = class_index(val_summary)
        rank = checkpoint_rank(
            approach, epsilon, val_rows, val_summary, initial_val_summary
        )
        row = {
            "epoch": epoch,
            "checkpoint_feasible": int(rank[0]),
            "high_norm_mean": indexed["High"]["norm_fulfill_mean"],
            "high_norm_p1": indexed["High"]["norm_fulfill_p1"],
            "high_norm_p10": indexed["High"]["norm_fulfill_p10"],
            "high_norm_min": val_diagnostics["high_norm_min"],
            "medium_norm_mean": indexed["Medium"]["norm_fulfill_mean"],
            "medium_norm_p1": indexed["Medium"]["norm_fulfill_p1"],
            "medium_norm_p10": indexed["Medium"]["norm_fulfill_p10"],
            "low_norm_mean": indexed["Low"]["norm_fulfill_mean"],
            "low_fulfill_mean": indexed["Low"]["fulfill_ratio_mean"],
            "legacy_low_guard_pass": (
                float(indexed["Low"]["fulfill_ratio_mean"])
                >= float(class_index(initial_val_summary)["Low"]["fulfill_ratio_mean"])
                - LOW_FULFILL_BUDGET
            ),
            "max_post_admission_mlu": max(
                float(item["max_admitted_capacity_ratio"]) for item in val_summary
            ),
            "max_disabled_flow": max(
                float(item["max_disabled_flow"]) for item in val_summary
            ),
            **val_diagnostics,
            **train_metrics,
        }
        if approach == "epsilon":
            row.update(high_gate_stats(val_rows, float(epsilon)))
        history = [item for item in history if int(item["epoch"]) < epoch]
        history.append(row)
        write_csv(run_dir / "train_history.csv", history)
        payload = {
            "epoch": epoch,
            "rank": rank,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": rng_state(),
            "config": config,
            "phase_a_parameter_sha256": initial_parameter_hash,
        }
        torch.save(payload, final_path)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            torch.save(payload, best_path)
        print(
            f"[phase-b {LEVELS[level]['label']}] {approach} "
            f"eps={epsilon} seed={seed} epoch={epoch}/"
            f"{LEVELS[level]['phase_b_epochs']} high={row['high_norm_mean']:.4f} "
            f"mid={row['medium_norm_mean']:.4f}/p10={row['medium_norm_p10']:.4f} "
            f"feasible={bool(rank[0])} best={best_epoch}",
            flush=True,
        )
    evaluations = []
    recovery_reference = None
    for checkpoint_name, checkpoint_path in (("final", final_path), ("best", best_path)):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        rows, summary, diagnostics = evaluate(
            model, props, evaluation_dataset, LEVELS[level]["evaluation"][0]
        )
        write_csv(run_dir / f"{checkpoint_name}_evaluation_metrics.csv", rows)
        write_json(
            run_dir / f"{checkpoint_name}_evaluation_summary.json",
            {"classes": summary, "diagnostics": diagnostics},
        )
        gate = high_gate_stats(rows, float(epsilon)) if approach == "epsilon" else None
        final_high_state = capture_high_state(
            model, props, evaluation_dataset, LEVELS[level]["evaluation"][0]
        )
        route = route_change_artifacts(
            run_dir / checkpoint_name,
            initial_high_state,
            final_high_state,
            initial_rows,
            rows,
        )
        rank = checkpoint_rank(
            approach, epsilon, rows, summary, initial_summary
        )
        evaluations.append(
            {
                "checkpoint": checkpoint_name,
                "epoch": int(checkpoint["epoch"]),
                "eligible": bool(rank[0]),
                "classes": summary,
                "diagnostics": diagnostics,
                "epsilon_gate": gate,
                "route_change": route,
            }
        )
        if checkpoint_name == "final":
            recovery_reference = rows
    recovery_checkpoint = torch.load(final_path, map_location=device, weights_only=False)
    recovery_model = Hattrick(props).to(device=device, dtype=props.dtype)
    recovery_model.load_state_dict(recovery_checkpoint["model_state_dict"])
    recovery_rows, _, _ = evaluate(
        recovery_model, props, evaluation_dataset, LEVELS[level]["evaluation"][0]
    )
    numeric = (
        "admitted_traffic",
        "fulfill_ratio",
        "norm_fulfill",
        "raw_mlu",
        "normalized_mlu",
        "disabled_flow",
        "admitted_capacity_ratio",
    )
    recovery_delta = max(
        abs(float(left[field]) - float(right[field]))
        for left, right in zip(recovery_reference, recovery_rows)
        for field in numeric
    )
    if recovery_delta > 1e-5:
        raise RuntimeError(f"checkpoint recovery mismatch: {recovery_delta}")
    complete = {
        "status": "COMPLETE",
        "best_epoch": best_epoch,
        "best_rank": list(best_rank),
        "phase_a_parameter_sha256": initial_parameter_hash,
        "phase_a_optimizer_repr_sha256": initial_optimizer_repr_hash,
        "all_parameters_trainable": all(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "checkpoint_recovery_max_delta": recovery_delta,
        "evaluations": evaluations,
        "artifact_sha256": {
            "best_model.pt": sha256(best_path),
            "final_model.pt": sha256(final_path),
            "train_history.csv": sha256(run_dir / "train_history.csv"),
        },
    }
    write_json(complete_path, complete)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared-2x objective order and per-snapshot epsilon experiment"
    )
    parser.add_argument("--level", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--approach", choices=APPROACHES, required=True)
    parser.add_argument("--epsilon", type=float, choices=EPSILONS)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.approach == "epsilon" and args.epsilon is None:
        parser.error("--approach epsilon requires --epsilon")
    if args.approach != "epsilon" and args.epsilon is not None:
        parser.error("--epsilon is only valid with --approach epsilon")
    if args.level == 0:
        if args.seed != 490 or args.approach != "control" or args.epsilon is not None:
            parser.error("Level 0 runs once as --approach control --seed 490")
        result = run_level0(OUTPUT_ROOT / "level0_lp", force=args.force)
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.level >= 2:
        level0 = OUTPUT_ROOT / "level0_lp" / "complete.json"
        if not level0.exists():
            parser.error("run Level 0 before neural scaling")
        result = json.loads(level0.read_text(encoding="utf-8"))
        if not result["neural_scaling_authorized"]:
            parser.error("Level 0 is NO_GO; neural scaling is not authorized")
    ensure_phase_a(args.level, args.seed, force=args.force)
    path = run_phase_b(
        args.level,
        args.approach,
        args.epsilon,
        args.seed,
        force=args.force,
    )
    print(f"[ok] {path}", flush=True)


if __name__ == "__main__":
    main()
