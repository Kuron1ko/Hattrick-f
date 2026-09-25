from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import replace
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
    "shared2x_causal_lp_runtime", PIMA_DIR / "run_experiment.py"
)
if runtime_spec is None or runtime_spec.loader is None:
    raise RuntimeError("Unable to load shared frozen-policy runtime")
runtime = importlib.util.module_from_spec(runtime_spec)
sys.modules[runtime_spec.name] = runtime
runtime_spec.loader.exec_module(runtime)

from planner import CausalResidualAdmissionPlanner, od_incidence


K = 8
OUTPUT_ROOT = THIS_DIR / "artifacts"
LEVELS = {
    1: {"label": "level1_eight_samples", "range": (200, 208), "backbone_level": 2},
    2: {"label": "level2_proxy", "range": (200, 250), "backbone_level": 2},
    3: {"label": "level3_validation", "range": (350, 400), "backbone_level": 4},
    4: {"label": "level4_confirmation", "range": (400, 500), "backbone_level": 4},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def planner_from_cache(cache: runtime.PolicyCache) -> CausalResidualAdmissionPlanner:
    pte = cache.dataset.pte.coalesce().cpu()
    indices = pte.indices().numpy()
    values = pte.values().numpy().astype(np.float64, copy=False)
    path_count, edge_count = pte.shape
    links_by_paths = sparse.coo_matrix(
        (values, (indices[1], indices[0])), shape=(edge_count, path_count)
    ).tocsr()
    if path_count % K:
        raise RuntimeError("Path count is not divisible by K")
    return CausalResidualAdmissionPlanner(
        links_by_paths, od_incidence(path_count // K, K)
    )


def predicted_baseline_admission(model, props, cache: runtime.PolicyCache) -> tuple[torch.Tensor, ...]:
    admitted_buckets = [[], [], []]
    for offset in range(0, len(cache), 8):
        stop = min(offset + 8, len(cache))
        indices = torch.arange(offset, stop, device=props.device)
        batch = runtime.select_cache(cache, indices)
        batch["tms"] = tuple(value.index_select(0, indices) for value in cache.predicted_tms)
        admitted, _ = runtime.simulate_admission(model, props, cache.dataset, batch, None)
        for class_index in range(3):
            admitted_buckets[class_index].append(admitted[class_index].detach())
    return tuple(torch.cat(items, dim=0) for items in admitted_buckets)


def enabled_masks(cache: runtime.PolicyCache) -> tuple[np.ndarray | None, np.ndarray | None]:
    if cache.path_masks is None:
        return None, None
    return (
        cache.path_masks[1].reshape(-1).detach().cpu().numpy().astype(bool),
        cache.path_masks[2].reshape(-1).detach().cpu().numpy().astype(bool),
    )


def build_planned_cache(
    model,
    props,
    cache: runtime.PolicyCache,
    low_entitlement_factor: float,
    protection_mode: str,
    lp_blend: float,
    planner_mode: str,
) -> tuple[runtime.PolicyCache, list[dict]]:
    planner = planner_from_cache(cache)
    predicted_admitted = predicted_baseline_admission(model, props, cache)
    medium_mask, low_mask = enabled_masks(cache)
    medium_policies: list[torch.Tensor] = []
    low_policies: list[torch.Tensor] = []
    diagnostics: list[dict] = []
    started = time.perf_counter()
    for sample in range(len(cache)):
        capacity = cache.capacities[sample].detach().cpu().numpy().astype(np.float64)
        high_flow = predicted_admitted[0][sample].detach().cpu().numpy().astype(np.float64)
        high_load = planner.links_by_paths @ high_flow
        residual = np.maximum(capacity - high_load, 0.0)
        predicted_medium = (
            cache.predicted_tms[1][sample].reshape(planner.num_pairs, K)[:, 0]
            .detach().cpu().numpy().astype(np.float64)
        )
        predicted_low = (
            cache.predicted_tms[2][sample].reshape(planner.num_pairs, K)[:, 0]
            .detach().cpu().numpy().astype(np.float64)
        )
        baseline_medium_flow = predicted_admitted[1][sample].detach().cpu().numpy().astype(np.float64)
        baseline_low_flow = predicted_admitted[2][sample].detach().cpu().numpy().astype(np.float64)
        baseline_low_load = planner.links_by_paths @ baseline_low_flow
        if protection_mode == "footprint":
            residual = np.maximum(
                residual - float(low_entitlement_factor) * baseline_low_load, 0.0
            )
            planned_low_demand = np.zeros_like(predicted_low)
            low_floor = np.zeros_like(predicted_low)
        elif protection_mode == "entitlement":
            planned_low_demand = predicted_low
            low_floor = (
                baseline_low_flow.reshape(planner.num_pairs, K).sum(axis=1)
                * float(low_entitlement_factor)
            )
        else:
            raise ValueError(f"Unknown protection mode: {protection_mode}")
        base_medium_policy = (
            cache.policies[1][sample].reshape(-1).detach().cpu().numpy().astype(np.float64)
        )
        if planner_mode == "flow":
            result = planner.solve(
                residual,
                predicted_medium,
                planned_low_demand,
                low_floor,
                medium_enabled=medium_mask,
                low_enabled=low_mask,
            )
            medium_denominator = np.repeat(predicted_medium, K)
            medium_policy = np.divide(
                result.medium_flow,
                medium_denominator,
                out=np.zeros_like(result.medium_flow),
                where=medium_denominator > 1e-12,
            )
            planned_medium_total = result.medium_total
            planned_low_total = result.low_total
            lp_status = result.status
        elif planner_mode == "gate":
            gate_result = planner.solve_medium_gates(
                residual, predicted_medium, base_medium_policy
            )
            gate = np.divide(
                gate_result.admitted_by_od,
                predicted_medium,
                out=np.zeros_like(predicted_medium),
                where=predicted_medium > 1e-12,
            )
            medium_policy = base_medium_policy * np.repeat(gate, K)
            planned_medium_total = gate_result.admitted_total
            planned_low_total = float(baseline_low_flow.sum())
            lp_status = gate_result.status
        else:
            raise ValueError(f"Unknown planner mode: {planner_mode}")
        medium_policy = (
            (1.0 - float(lp_blend)) * base_medium_policy
            + float(lp_blend) * medium_policy
        )
        if protection_mode == "footprint":
            low_policy = cache.policies[2][sample].reshape(-1).detach().cpu().numpy().astype(np.float64)
        else:
            low_denominator = np.repeat(predicted_low, K)
            low_policy = np.divide(
                result.low_flow,
                low_denominator,
                out=np.zeros_like(result.low_flow),
                where=low_denominator > 1e-12,
            )
        medium_policies.append(
            torch.as_tensor(medium_policy, device=props.device, dtype=props.dtype).reshape(-1, 1)
        )
        low_policies.append(
            torch.as_tensor(low_policy, device=props.device, dtype=props.dtype).reshape(-1, 1)
        )
        diagnostics.append(
            {
                "snapshot": cache.source_start + sample,
                "predicted_baseline_medium": float(baseline_medium_flow.sum()),
                "predicted_planned_medium": planned_medium_total,
                "predicted_medium_gain": planned_medium_total - float(baseline_medium_flow.sum()),
                "predicted_baseline_low": float(baseline_low_flow.sum()),
                "predicted_planned_low": (
                    float(baseline_low_flow.sum()) if protection_mode == "footprint" else planned_low_total
                ),
                "low_entitlement_total": float(low_floor.sum()),
                "medium_policy_mass_max": float(
                    medium_policy.reshape(planner.num_pairs, K).sum(axis=1).max()
                ),
                "low_policy_mass_max": float(
                    low_policy.reshape(planner.num_pairs, K).sum(axis=1).max()
                ),
                "lp_status": lp_status,
            }
        )
    elapsed = time.perf_counter() - started
    for row in diagnostics:
        row["mean_planning_seconds_per_snapshot"] = elapsed / max(len(cache), 1)
    planned = replace(
        cache,
        policies=(
            cache.policies[0],
            torch.stack(medium_policies, dim=0),
            torch.stack(low_policies, dim=0),
        ),
    )
    return planned, diagnostics


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def gaps(candidate: list[dict], baseline: list[dict]) -> dict[str, float]:
    c = summary_index(candidate)
    b = summary_index(baseline)
    return {
        f"{class_name.lower()}_{metric}_gap": float(c[class_name][f"norm_fulfill_{metric}"])
        - float(b[class_name][f"norm_fulfill_{metric}"])
        for class_name in runtime.CLASSES
        for metric in ("mean", "p1", "p10")
    }


def safe_remove(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise RuntimeError(f"Unsafe removal target: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def run_one(
    level: int,
    backbone_seed: int,
    low_entitlement_factor: float,
    protection_mode: str,
    lp_blend: float,
    planner_mode: str,
    force: bool,
) -> Path:
    spec = LEVELS[level]
    factor_label = format(float(low_entitlement_factor), ".4g").replace(".", "p")
    blend_label = format(float(lp_blend), ".4g").replace(".", "p")
    run_dir = (
        OUTPUT_ROOT
        / spec["label"]
        / protection_mode
        / planner_mode
        / f"low_floor_{factor_label}"
        / f"lp_blend_{blend_label}"
        / f"backbone_{backbone_seed}"
    )
    if (run_dir / "complete.json").exists() and not force:
        print(f"[skip] {run_dir}", flush=True)
        return run_dir
    if force:
        safe_remove(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_level = int(spec["backbone_level"])
    props = runtime.build_props(backbone_level, device)
    model, checkpoint_path = runtime.load_backbone(backbone_level, backbone_seed, props, device)
    start, end = spec["range"]
    cache = runtime.build_policy_cache(model, props, start, end, batch_size=8)
    baseline_rows, baseline_summary = runtime.evaluate_cache(model, props, cache, None)
    planned_cache, diagnostics = build_planned_cache(
        model, props, cache, low_entitlement_factor, protection_mode, lp_blend, planner_mode
    )
    candidate_rows, candidate_summary = runtime.evaluate_cache(model, props, planned_cache, None)
    metric_gaps = gaps(candidate_summary, baseline_summary)
    high_mean = float(summary_index(candidate_summary)["High"]["norm_fulfill_mean"])
    feasible = (
        high_mean >= min(0.995, float(summary_index(baseline_summary)["High"]["norm_fulfill_mean"]) - 1e-6)
        and abs(metric_gaps["high_mean_gap"]) <= 1e-6
        and metric_gaps["low_mean_gap"] >= -0.003
        and metric_gaps["low_p10_gap"] >= -0.01
        and max(float(row["max_admitted_capacity_ratio"]) for row in candidate_summary) <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in candidate_summary) <= 1e-8
    )
    config = {
        "method": "Causal Residual Admission LP (CRAL)",
        "level": level,
        "range": [start, end],
        "backbone_level": backbone_level,
        "backbone_checkpoint": str(checkpoint_path),
        "backbone_sha256": sha256(checkpoint_path),
        "low_entitlement_factor": float(low_entitlement_factor),
        "protection_mode": protection_mode,
        "lp_blend": float(lp_blend),
        "planner_mode": planner_mode,
        "inference_contract": "LP sees predicted traffic, capacities, candidate paths, and frozen backbone policies only; actual traffic is used only by evaluation replay",
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
            "gaps": metric_gaps,
            "feasible": feasible,
            "predicted_medium_gain_mean": float(
                np.mean([row["predicted_medium_gain"] for row in diagnostics])
            ),
            "planning_seconds_per_snapshot": float(
                diagnostics[0]["mean_planning_seconds_per_snapshot"]
            ),
        },
    )
    complete = {"status": "complete", "feasible": feasible, "gaps": metric_gaps}
    runtime.write_json(run_dir / "complete.json", complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=tuple(LEVELS), required=True)
    parser.add_argument("--backbone-seed", type=int, default=490)
    parser.add_argument("--low-entitlement-factor", type=float, default=1.0)
    parser.add_argument(
        "--protection-mode", choices=("footprint", "entitlement"), default="footprint"
    )
    parser.add_argument("--lp-blend", type=float, default=1.0)
    parser.add_argument("--planner-mode", choices=("gate", "flow"), default="gate")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_one(
        args.level,
        args.backbone_seed,
        args.low_entitlement_factor,
        args.protection_mode,
        args.lp_blend,
        args.planner_mode,
        args.force,
    )


if __name__ == "__main__":
    main()
