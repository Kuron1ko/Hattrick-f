from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
PIMA_DIR = TEST_DIR / "shared2x_medium_adapter"
for item in (str(ROOT), str(TEST_DIR), str(PIMA_DIR), str(THIS_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

runtime_spec = importlib.util.spec_from_file_location(
    "shared2x_sandwich_runtime", PIMA_DIR / "run_experiment.py"
)
if runtime_spec is None or runtime_spec.loader is None:
    raise RuntimeError("Unable to load the shared frozen-policy runtime")
runtime = importlib.util.module_from_spec(runtime_spec)
sys.modules[runtime_spec.name] = runtime
runtime_spec.loader.exec_module(runtime)

from planner import SandwichMediumPlanner, od_incidence


K = 8
OUTPUT_ROOT = THIS_DIR / "artifacts"
LEVELS = {
    1: {"label": "level1_eight_samples", "range": (200, 208)},
    2: {"label": "level2_proxy", "range": (200, 250)},
    3: {"label": "level3_validation", "range": (350, 400)},
    4: {"label": "level4_confirmation", "range": (400, 500)},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_remove(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise RuntimeError(f"Unsafe removal target: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def build_planner(cache: runtime.PolicyCache) -> SandwichMediumPlanner:
    pte = cache.dataset.pte.coalesce().cpu()
    indices = pte.indices().numpy()
    values = pte.values().numpy().astype(np.float64, copy=False)
    path_count, edge_count = pte.shape
    links_by_paths = sparse.coo_matrix(
        (values, (indices[1], indices[0])), shape=(edge_count, path_count)
    ).tocsr()
    if path_count % K:
        raise RuntimeError("Path count is not divisible by K")
    return SandwichMediumPlanner(
        links_by_paths, od_incidence(path_count // K, K)
    )


def baseline_admitted(model, props, cache: runtime.PolicyCache) -> tuple[torch.Tensor, ...]:
    buckets: list[list[torch.Tensor]] = [[], [], []]
    for offset in range(0, len(cache), 8):
        stop = min(offset + 8, len(cache))
        indices = torch.arange(offset, stop, device=props.device)
        batch = runtime.select_cache(cache, indices)
        admitted, _ = runtime.simulate_admission(
            model, props, cache.dataset, batch, None
        )
        for class_index in range(3):
            buckets[class_index].append(admitted[class_index].detach().cpu())
    return tuple(torch.cat(items, dim=0) for items in buckets)


def rows_from_admitted(
    cache: runtime.PolicyCache,
    admitted: tuple[np.ndarray, np.ndarray, np.ndarray],
    planner: SandwichMediumPlanner,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    masks = None
    if cache.path_masks is not None:
        masks = [
            cache.path_masks[index].reshape(-1).detach().cpu().numpy().astype(bool)
            for index in range(3)
        ]
    tms = [value.detach().cpu().numpy() for value in cache.tms]
    capacities = cache.capacities.detach().cpu().numpy().astype(np.float64)
    oracle_flows = [value.detach().cpu().numpy() for value in cache.oracle_flows]
    oracle_mlus = [value.detach().cpu().numpy() for value in cache.oracle_mlus]
    for sample in range(len(cache)):
        cumulative = np.zeros(planner.num_paths, dtype=np.float64)
        for class_index, class_name in enumerate(runtime.CLASSES):
            flow = admitted[class_index][sample]
            cumulative += flow
            load = planner.links_by_paths @ cumulative
            ratio = load / np.maximum(capacities[sample], 1e-12)
            raw_mlu = float(ratio.max(initial=0.0))
            total = float(flow.sum())
            od_demand = tms[class_index][sample].reshape(planner.num_pairs, K)[:, 0]
            demand = float(od_demand.sum())
            oracle = max(float(oracle_flows[class_index][sample]), 1e-9)
            disabled = 0.0
            if masks is not None and (~masks[class_index]).any():
                disabled = float(np.abs(flow[~masks[class_index]]).max(initial=0.0))
            rows.append(
                {
                    "snapshot": cache.source_start + sample,
                    "class": class_name,
                    "admitted_traffic": total,
                    "demand": demand,
                    "fulfill_ratio": total / max(demand, 1e-9),
                    "oracle_admitted_traffic": oracle,
                    "norm_fulfill": total / oracle,
                    "raw_mlu": raw_mlu,
                    "oracle_mlu": float(oracle_mlus[class_index][sample]),
                    "normalized_mlu": raw_mlu
                    / max(float(oracle_mlus[class_index][sample]), 1e-9),
                    "disabled_flow": disabled,
                    "admitted_capacity_ratio": raw_mlu,
                }
            )
    if any(
        not math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key != "class"
    ):
        raise RuntimeError("Evaluation produced NaN or Inf")
    return rows, runtime.summarize_rows(rows)


def run_one(
    level: int, backbone_seed: int, low_reservation: str, force: bool
) -> Path:
    spec = LEVELS[level]
    run_dir = (
        OUTPUT_ROOT
        / spec["label"]
        / f"low_{low_reservation}"
        / f"backbone_{backbone_seed}"
    )
    if (run_dir / "complete.json").exists() and not force:
        print(f"[skip] {run_dir}", flush=True)
        return run_dir
    if force:
        safe_remove(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint_path = runtime.load_backbone(4, backbone_seed, props, device)
    start, end = spec["range"]
    cache = runtime.build_policy_cache(model, props, start, end, batch_size=8)
    baseline_rows, baseline_summary = runtime.evaluate_cache(model, props, cache, None)
    base = baseline_admitted(model, props, cache)
    base_np = tuple(value.numpy().astype(np.float64) for value in base)
    planner = build_planner(cache)
    medium_mask = None
    if cache.path_masks is not None:
        medium_mask = (
            cache.path_masks[1].reshape(-1).detach().cpu().numpy().astype(bool)
        )
    low_mask = None
    if cache.path_masks is not None:
        low_mask = (
            cache.path_masks[2].reshape(-1).detach().cpu().numpy().astype(bool)
        )

    optimized_medium: list[np.ndarray] = []
    optimized_low: list[np.ndarray] = []
    diagnostics: list[dict] = []
    started = time.perf_counter()
    for sample in range(len(cache)):
        demand = (
            cache.tms[1][sample]
            .reshape(planner.num_pairs, K)[:, 0]
            .detach().cpu().numpy().astype(np.float64)
        )
        if low_reservation == "footprint":
            plan = planner.solve(
                cache.capacities[sample].detach().cpu().numpy(),
                base_np[0][sample],
                base_np[2][sample],
                demand,
                medium_mask,
            )
            medium_flow = plan.path_flow
            low_flow = base_np[2][sample]
            medium_total = plan.total
            low_total = float(low_flow.sum())
            plan_status = plan.status
        elif low_reservation == "od_entitlement":
            low_demand = (
                cache.tms[2][sample]
                .reshape(planner.num_pairs, K)[:, 0]
                .detach().cpu().numpy().astype(np.float64)
            )
            low_floor = base_np[2][sample].reshape(planner.num_pairs, K).sum(axis=1)
            plan = planner.solve_with_low_entitlements(
                cache.capacities[sample].detach().cpu().numpy(),
                base_np[0][sample],
                demand,
                low_demand,
                low_floor,
                medium_mask,
                low_mask,
            )
            medium_flow = plan.medium_flow
            low_flow = plan.low_flow
            medium_total = plan.medium_total
            low_total = plan.low_total
            plan_status = plan.status
        else:
            raise ValueError(f"Unknown Low reservation: {low_reservation}")
        baseline_medium_total = float(base_np[1][sample].sum())
        if medium_total < baseline_medium_total - 1e-7:
            raise RuntimeError(
                f"Medium dominance invariant failed at snapshot {start + sample}"
            )
        optimized_medium.append(medium_flow)
        optimized_low.append(low_flow)
        diagnostics.append(
            {
                "snapshot": start + sample,
                "baseline_medium": baseline_medium_total,
                "optimized_medium": medium_total,
                "absolute_gain": medium_total - baseline_medium_total,
                "baseline_low": float(base_np[2][sample].sum()),
                "optimized_low": low_total,
                "low_total_gap": low_total - float(base_np[2][sample].sum()),
                "lp_status": plan_status,
            }
        )
    elapsed = time.perf_counter() - started
    for row in diagnostics:
        row["planning_seconds_per_snapshot"] = elapsed / max(len(cache), 1)

    candidate_admitted = (
        base_np[0],
        np.stack(optimized_medium, axis=0),
        np.stack(optimized_low, axis=0),
    )
    candidate_rows, candidate_summary = rows_from_admitted(
        cache, candidate_admitted, planner
    )
    gaps = runtime.metric_gaps(candidate_summary, baseline_summary)
    high = runtime.summary_index(candidate_summary)["High"]
    feasible = (
        float(high["norm_fulfill_mean"]) >= 0.995
        and abs(gaps["high_mean_gap"]) <= 1e-6
        and abs(gaps["low_mean_gap"]) <= 1e-6
        and abs(gaps["low_p10_gap"]) <= 1e-6
        and gaps["medium_mean_gap"] >= -1e-8
        and gaps["medium_p1_gap"] >= -1e-8
        and gaps["medium_p10_gap"] >= -1e-8
        and max(float(row["max_admitted_capacity_ratio"]) for row in candidate_summary)
        <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in candidate_summary) <= 1e-8
    )
    config = {
        "method": "Low-Reserved Sandwich Max-Flow (LR-SMF)",
        "level": level,
        "range": [start, end],
        "load_factor": 2.0,
        "low_reservation": low_reservation,
        "backbone_checkpoint": str(checkpoint_path),
        "backbone_sha256": sha256(checkpoint_path),
        "inference_contract": "The admission layer sees realized class demands and capacities, as the original sequential admission simulator does.",
        "invariants": (
            [
                "High admitted path flow is copied exactly from Hattrick",
                "Low admitted path flow and its link footprint are reserved exactly",
                "Only Medium is re-optimized inside the remaining feasible capacity",
            ]
            if low_reservation == "footprint"
            else [
                "High admitted path flow is copied exactly from Hattrick",
                "Every Low OD keeps at least its baseline admitted amount",
                "Medium and Low may reroute jointly inside residual capacity",
            ]
        ),
        "source_sha256": {
            "planner.py": sha256(THIS_DIR / "planner.py"),
            "run_experiment.py": sha256(Path(__file__).resolve()),
        },
    }
    runtime.write_json(run_dir / "config.json", config)
    runtime.write_csv(run_dir / "baseline_metrics.csv", baseline_rows)
    runtime.write_csv(run_dir / "candidate_metrics.csv", candidate_rows)
    runtime.write_csv(run_dir / "planning_diagnostics.csv", diagnostics)
    runtime.write_json(
        run_dir / "summary.json",
        {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "gaps": gaps,
            "feasible": feasible,
            "planning_seconds_per_snapshot": elapsed / max(len(cache), 1),
        },
    )
    complete = {"status": "complete", "feasible": feasible, "gaps": gaps}
    runtime.write_json(run_dir / "complete.json", complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=tuple(LEVELS), required=True)
    parser.add_argument("--backbone-seed", type=int, default=490)
    parser.add_argument(
        "--low-reservation",
        choices=("footprint", "od_entitlement"),
        default="od_entitlement",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_one(args.level, args.backbone_seed, args.low_reservation, args.force)


if __name__ == "__main__":
    main()
