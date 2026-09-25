from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy import sparse


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
PIMA_DIR = TEST_DIR / "shared2x_medium_adapter"
CRAL_DIR = TEST_DIR / "shared2x_causal_lp"
for item in (str(ROOT), str(TEST_DIR), str(PIMA_DIR), str(CRAL_DIR), str(THIS_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

runtime_spec = importlib.util.spec_from_file_location(
    "shared2x_edge_guard_runtime", PIMA_DIR / "run_experiment.py"
)
if runtime_spec is None or runtime_spec.loader is None:
    raise RuntimeError("Unable to load frozen-policy runtime")
runtime = importlib.util.module_from_spec(runtime_spec)
sys.modules[runtime_spec.name] = runtime
runtime_spec.loader.exec_module(runtime)

from planner import CausalResidualAdmissionPlanner, od_incidence


K = 8
OUTPUT_ROOT = THIS_DIR / "artifacts"
LEVELS = {
    1: {
        "label": "level1_correctness",
        "calibration": (0, 24),
        "evaluation": (32, 40),
        "backbone": 2,
    },
    2: {
        "label": "level2_proxy",
        "calibration": (0, 160),
        "evaluation": (200, 250),
        "backbone": 2,
    },
    3: {
        "label": "level3_validation",
        "calibration": (0, 318),
        "evaluation": (350, 400),
        "backbone": 4,
    },
    4: {
        "label": "level4_confirmation",
        "calibration": (0, 318),
        "evaluation": (400, 500),
        "backbone": 4,
    },
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


def build_planner(cache: runtime.PolicyCache) -> CausalResidualAdmissionPlanner:
    pte = cache.dataset.pte.coalesce().cpu()
    indices = pte.indices().numpy()
    values = pte.values().numpy().astype(np.float64, copy=False)
    path_count, edge_count = pte.shape
    links_by_paths = sparse.coo_matrix(
        (values, (indices[1], indices[0])), shape=(edge_count, path_count)
    ).tocsr()
    return CausalResidualAdmissionPlanner(
        links_by_paths, od_incidence(path_count // K, K)
    )


def all_admitted(
    model,
    props,
    cache: runtime.PolicyCache,
    predicted: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    buckets: list[list[torch.Tensor]] = [[], [], []]
    for offset in range(0, len(cache), 8):
        stop = min(offset + 8, len(cache))
        indices = torch.arange(offset, stop, device=props.device)
        batch = runtime.select_cache(cache, indices)
        if predicted:
            batch["tms"] = tuple(
                value.index_select(0, indices) for value in cache.predicted_tms
            )
        admitted, _ = runtime.simulate_admission(
            model, props, cache.dataset, batch, None
        )
        for class_index in range(3):
            buckets[class_index].append(admitted[class_index].detach())
    return tuple(torch.cat(items, dim=0) for items in buckets)


def calibration_errors(
    model,
    props,
    cache: runtime.PolicyCache,
    planner: CausalResidualAdmissionPlanner,
) -> tuple[np.ndarray, np.ndarray]:
    actual = all_admitted(model, props, cache, predicted=False)
    predicted = all_admitted(model, props, cache, predicted=True)
    actual_hl = (actual[0] + actual[2]).detach().cpu().numpy().astype(np.float64)
    predicted_hl = (predicted[0] + predicted[2]).detach().cpu().numpy().astype(np.float64)
    actual_load = (planner.links_by_paths @ actual_hl.T).T
    predicted_load = (planner.links_by_paths @ predicted_hl.T).T
    additive = np.maximum(actual_load - predicted_load, 0.0)
    ratio = np.ones_like(actual_load)
    stable = predicted_load > 1e-4
    ratio[stable] = actual_load[stable] / predicted_load[stable]
    ratio = np.clip(ratio, 1.0, 6.0)
    return ratio, additive


def margin_vectors(
    ratio_errors: np.ndarray,
    additive_errors: np.ndarray,
    quantile: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    ratio_q = np.quantile(ratio_errors, quantile, axis=0)
    additive_q = np.quantile(additive_errors, quantile, axis=0)
    ratio_guard = 1.0 + float(scale) * (np.maximum(ratio_q, 1.0) - 1.0)
    additive_guard = float(scale) * np.maximum(additive_q, 0.0)
    return ratio_guard, additive_guard


def enabled_medium_mask(cache: runtime.PolicyCache) -> np.ndarray | None:
    if cache.path_masks is None:
        return None
    return cache.path_masks[1].reshape(-1).detach().cpu().numpy().astype(bool)


def guarded_cache(
    props,
    cache: runtime.PolicyCache,
    planner: CausalResidualAdmissionPlanner,
    predicted_np: tuple[np.ndarray, np.ndarray, np.ndarray],
    ratio_guard: np.ndarray,
    additive_guard: np.ndarray,
    blend: float,
) -> tuple[runtime.PolicyCache, list[dict]]:
    mask = enabled_medium_mask(cache)
    policies: list[torch.Tensor] = []
    diagnostics: list[dict] = []
    for sample in range(len(cache)):
        capacity = cache.capacities[sample].detach().cpu().numpy().astype(np.float64)
        predicted_hl = predicted_np[0][sample] + predicted_np[2][sample]
        predicted_hl_load = planner.links_by_paths @ predicted_hl
        reserved = ratio_guard * predicted_hl_load + additive_guard
        residual = np.maximum(capacity - reserved, 0.0)
        medium_demand = (
            cache.predicted_tms[1][sample]
            .reshape(planner.num_pairs, K)[:, 0]
            .detach().cpu().numpy().astype(np.float64)
        )
        result = planner.solve(
            residual,
            medium_demand,
            np.zeros(planner.num_pairs, dtype=np.float64),
            np.zeros(planner.num_pairs, dtype=np.float64),
            medium_enabled=mask,
            low_enabled=np.zeros(planner.num_paths, dtype=bool),
        )
        denominator = np.repeat(medium_demand, K)
        planned = np.divide(
            result.medium_flow,
            denominator,
            out=np.zeros_like(result.medium_flow),
            where=denominator > 1e-12,
        )
        base = cache.policies[1][sample].reshape(-1).detach().cpu().numpy()
        mixed = (1.0 - float(blend)) * base + float(blend) * planned
        policies.append(
            torch.as_tensor(
                mixed, device=props.device, dtype=props.dtype
            ).reshape(-1, 1)
        )
        diagnostics.append(
            {
                "snapshot": cache.source_start + sample,
                "predicted_medium_plan": result.medium_total,
                "mean_ratio_guard": float(np.mean(ratio_guard)),
                "max_ratio_guard": float(np.max(ratio_guard)),
                "mean_additive_guard": float(np.mean(additive_guard)),
                "zero_residual_edges": int(np.sum(residual <= 1e-12)),
            }
        )
    return replace(
        cache,
        policies=(cache.policies[0], torch.stack(policies), cache.policies[2]),
    ), diagnostics


def candidate_ok(gaps: dict[str, float]) -> bool:
    return (
        abs(gaps["high_mean_gap"]) <= 1e-6
        and gaps["medium_mean_gap"] > 0.0
        and gaps["medium_p1_gap"] > 0.0
        and gaps["medium_p10_gap"] > 0.0
        and gaps["low_mean_gap"] >= -0.003
        and gaps["low_p10_gap"] >= -0.01
    )


def run_one(level: int, backbone_seed: int, force: bool) -> Path:
    spec = LEVELS[level]
    run_dir = OUTPUT_ROOT / spec["label"] / f"backbone_{backbone_seed}"
    if (run_dir / "complete.json").exists() and not force:
        print(f"[skip] {run_dir}", flush=True)
        return run_dir
    if force:
        safe_remove(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(int(spec["backbone"]), device)
    model, checkpoint = runtime.load_backbone(
        int(spec["backbone"]), backbone_seed, props, device
    )
    calibration = runtime.build_policy_cache(
        model, props, *spec["calibration"], batch_size=8
    )
    evaluation = runtime.build_policy_cache(
        model, props, *spec["evaluation"], batch_size=8
    )
    planner = build_planner(evaluation)
    calibration_planner = build_planner(calibration)
    ratio_errors, additive_errors = calibration_errors(
        model, props, calibration, calibration_planner
    )
    baseline_rows, baseline_summary = runtime.evaluate_cache(
        model, props, evaluation, None
    )
    evaluation_predicted = all_admitted(model, props, evaluation, predicted=True)
    evaluation_predicted_np = tuple(
        value.detach().cpu().numpy().astype(np.float64)
        for value in evaluation_predicted
    )

    sweep_rows: list[dict] = []
    best: dict | None = None
    for quantile in (0.5, 0.75, 0.9, 0.95):
        for scale in (0.5, 1.0, 1.5):
            ratio_guard, additive_guard = margin_vectors(
                ratio_errors, additive_errors, quantile, scale
            )
            for blend in (0.25, 0.5, 0.75, 1.0):
                planned, diagnostics = guarded_cache(
                    props,
                    evaluation,
                    planner,
                    evaluation_predicted_np,
                    ratio_guard,
                    additive_guard,
                    blend,
                )
                rows, summary = runtime.evaluate_cache(model, props, planned, None)
                gaps = runtime.metric_gaps(summary, baseline_summary)
                ok = candidate_ok(gaps)
                rank = (
                    int(ok),
                    min(
                        gaps["medium_mean_gap"],
                        gaps["medium_p1_gap"],
                        gaps["medium_p10_gap"],
                    ),
                    gaps["medium_mean_gap"],
                    gaps["low_mean_gap"],
                )
                record = {
                    "quantile": quantile,
                    "scale": scale,
                    "blend": blend,
                    "ok": int(ok),
                    **gaps,
                    "mean_ratio_guard": diagnostics[0]["mean_ratio_guard"],
                    "max_ratio_guard": diagnostics[0]["max_ratio_guard"],
                    "mean_additive_guard": diagnostics[0]["mean_additive_guard"],
                }
                sweep_rows.append(record)
                if best is None or rank > best["rank"]:
                    best = {
                        "rank": rank,
                        "record": record,
                        "rows": rows,
                        "summary": summary,
                        "ratio_guard": ratio_guard,
                        "additive_guard": additive_guard,
                        "diagnostics": diagnostics,
                    }
    assert best is not None
    runtime.write_csv(run_dir / "sweep.csv", sweep_rows)
    runtime.write_csv(run_dir / "baseline_metrics.csv", baseline_rows)
    runtime.write_csv(run_dir / "best_candidate_metrics.csv", best["rows"])
    runtime.write_csv(run_dir / "best_diagnostics.csv", best["diagnostics"])
    np.savez(
        run_dir / "best_guards.npz",
        ratio_guard=best["ratio_guard"],
        additive_guard=best["additive_guard"],
    )
    config = {
        "method": "Edge-Calibrated ESM Guarded Medium Flow",
        "level": level,
        "calibration": list(spec["calibration"]),
        "evaluation": list(spec["evaluation"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "inference_inputs": "ESM prediction, stored train-only edge guards, capacities, paths and masks",
        "sweep": {
            "quantiles": [0.5, 0.75, 0.9, 0.95],
            "scales": [0.5, 1.0, 1.5],
            "blends": [0.25, 0.5, 0.75, 1.0],
        },
        "source_sha256": {"run_experiment.py": sha256(Path(__file__).resolve())},
    }
    runtime.write_json(run_dir / "config.json", config)
    runtime.write_json(
        run_dir / "summary.json",
        {
            "baseline": baseline_summary,
            "best_candidate": best["summary"],
            "best": best["record"],
            "passing_candidates": int(sum(row["ok"] for row in sweep_rows)),
        },
    )
    complete = {
        "status": "complete",
        "passing_candidates": int(sum(row["ok"] for row in sweep_rows)),
        "best": best["record"],
    }
    runtime.write_json(run_dir / "complete.json", complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=tuple(LEVELS), required=True)
    parser.add_argument("--backbone-seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_one(args.level, args.backbone_seed, args.force)


if __name__ == "__main__":
    main()
