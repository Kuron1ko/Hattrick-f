from __future__ import annotations

import argparse
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
from scipy.optimize import linprog


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
RUNTIME_DIR = TEST_DIR / "shared2x_medium_adapter"
CAUSAL_DIR = TEST_DIR / "shared2x_causal_lp"
for item in (str(ROOT), str(TEST_DIR), str(RUNTIME_DIR), str(CAUSAL_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

spec = importlib.util.spec_from_file_location(
    "shared2x_scenario_robust_runtime", RUNTIME_DIR / "run_experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load frozen-policy runtime")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)

from planner import od_incidence


K = 8
OUTPUT_ROOT = THIS_DIR / "artifacts"
LEVELS = {
    1: {"label": "level1_eight_samples", "range": (200, 208), "backbone_level": 2},
    2: {"label": "level2_proxy", "range": (200, 250), "backbone_level": 2},
    3: {"label": "level3_validation", "range": (350, 400), "backbone_level": 4},
    4: {"label": "level4_confirmation", "range": (400, 500), "backbone_level": 4},
}


def planner_matrices(cache: runtime.PolicyCache) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    pte = cache.dataset.pte.coalesce().cpu()
    indices = pte.indices().numpy()
    values = pte.values().numpy().astype(np.float64, copy=False)
    path_count, edge_count = pte.shape
    links_by_paths = sparse.coo_matrix(
        (values, (indices[1], indices[0])), shape=(edge_count, path_count)
    ).tocsr()
    return links_by_paths, od_incidence(path_count // K, K)


def od_values(tensor: torch.Tensor) -> np.ndarray:
    values = tensor.detach().cpu().numpy().squeeze(-1)
    return values.reshape(values.shape[0], -1, K)[:, :, 0].astype(np.float64)


def normalized_shape_features(cache: runtime.PolicyCache) -> np.ndarray:
    parts = []
    totals = []
    for predicted in cache.predicted_tms:
        values = od_values(predicted)
        total = values.sum(axis=1, keepdims=True)
        parts.append(values / np.maximum(total, 1e-9))
        totals.append(np.log1p(total))
    return np.concatenate(parts + totals, axis=1)


def scenario_demands(
    history: runtime.PolicyCache,
    evaluation: runtime.PolicyCache,
    sample: int,
    scenario_count: int,
    neighbor_mode: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    # At time t, snapshots earlier than t have already been observed and can
    # causally refresh the error-scenario library.  The current actual TM is
    # never included.
    history_features = np.concatenate(
        [normalized_shape_features(history), normalized_shape_features(evaluation)[:sample]],
        axis=0,
    )
    current_features = normalized_shape_features(evaluation)[sample]
    if neighbor_mode == "knn":
        feature_scale = np.std(history_features, axis=0) + 1e-5
        distance = np.mean(((history_features - current_features) / feature_scale) ** 2, axis=1)
        selected = np.argsort(distance)[:scenario_count]
    elif neighbor_mode == "recent":
        selected = np.arange(max(0, len(history) - scenario_count), len(history))
    else:
        raise ValueError(f"Unknown neighbor mode: {neighbor_mode}")

    scenarios: list[np.ndarray] = []
    for class_index in range(3):
        history_actual = np.concatenate(
            [od_values(history.tms[class_index]), od_values(evaluation.tms[class_index])[:sample]],
            axis=0,
        )
        history_predicted = np.concatenate(
            [
                od_values(history.predicted_tms[class_index]),
                od_values(evaluation.predicted_tms[class_index])[:sample],
            ],
            axis=0,
        )
        current = od_values(evaluation.predicted_tms[class_index])[sample]
        corrected = [current]
        for index in selected:
            scale = float(current.sum()) / max(float(history_predicted[index].sum()), 1e-9)
            residual = (history_actual[index] - history_predicted[index]) * scale
            corrected.append(np.maximum(current + residual, 0.0))
        scenarios.append(np.stack(corrected, axis=0))
    return scenarios, selected


def expand_od(values: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    expanded = np.repeat(values, K, axis=1)
    return torch.as_tensor(expanded, device=device, dtype=dtype).unsqueeze(-1)


def baseline_scenario_flows(
    model,
    props,
    cache: runtime.PolicyCache,
    sample: int,
    scenarios: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    count = scenarios[0].shape[0]
    policies = tuple(value[sample : sample + 1].expand(count, -1, -1) for value in cache.policies)
    tms = tuple(expand_od(value, props.device, props.dtype) for value in scenarios)
    capacities = cache.capacities[sample : sample + 1].expand(count, -1)
    batch = {
        "policies": policies,
        "tms": tms,
        "capacities": capacities,
        "oracle_flows": tuple(torch.ones(count, device=props.device, dtype=props.dtype) for _ in range(3)),
    }
    with torch.no_grad():
        admitted, _ = runtime.simulate_admission(model, props, cache.dataset, batch, None)
    return (
        admitted[0].detach().cpu().numpy().astype(np.float64),
        admitted[2].detach().cpu().numpy().astype(np.float64),
    )


def solve_robust_medium_policy(
    links_by_paths: sparse.csr_matrix,
    od_by_paths: sparse.csr_matrix,
    capacities: np.ndarray,
    scenarios: list[np.ndarray],
    high_flows: np.ndarray,
    low_flows: np.ndarray,
    low_factor: float,
    enabled: np.ndarray | None,
) -> np.ndarray:
    medium_scenarios = scenarios[1]
    matrices = []
    rhs = []
    for index in range(medium_scenarios.shape[0]):
        high_load = links_by_paths @ high_flows[index]
        low_load = links_by_paths @ low_flows[index]
        residual = np.maximum(capacities - high_load - float(low_factor) * low_load, 0.0)
        demand_per_path = np.repeat(medium_scenarios[index], K)
        matrices.append(links_by_paths @ sparse.diags(demand_per_path))
        rhs.append(residual)
    matrices.append(od_by_paths)
    rhs.append(np.ones(od_by_paths.shape[0], dtype=np.float64))
    nominal = np.repeat(medium_scenarios[0], K)
    if enabled is None:
        bounds = [(0.0, 1.0)] * links_by_paths.shape[1]
    else:
        bounds = [(0.0, 1.0) if flag else (0.0, 0.0) for flag in enabled]
    result = linprog(
        -nominal,
        A_ub=sparse.vstack(matrices, format="csr"),
        b_ub=np.concatenate(rhs),
        bounds=bounds,
        method="highs",
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"Scenario-robust LP failed ({result.status}): {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def planned_cache(
    model,
    props,
    history: runtime.PolicyCache,
    evaluation: runtime.PolicyCache,
    scenario_count: int,
    neighbor_mode: str,
    low_factor: float,
    blend: float,
) -> tuple[runtime.PolicyCache, list[dict]]:
    links_by_paths, od_by_paths = planner_matrices(evaluation)
    enabled = None
    if evaluation.path_masks is not None:
        enabled = evaluation.path_masks[1].reshape(-1).detach().cpu().numpy().astype(bool)
    policies = []
    diagnostics = []
    started = time.perf_counter()
    for sample in range(len(evaluation)):
        scenarios, selected = scenario_demands(
            history, evaluation, sample, scenario_count, neighbor_mode
        )
        high_flows, low_flows = baseline_scenario_flows(
            model, props, evaluation, sample, scenarios
        )
        robust = solve_robust_medium_policy(
            links_by_paths,
            od_by_paths,
            evaluation.capacities[sample].detach().cpu().numpy().astype(np.float64),
            scenarios,
            high_flows,
            low_flows,
            low_factor,
            enabled,
        )
        base = evaluation.policies[1][sample].reshape(-1).detach().cpu().numpy().astype(np.float64)
        mixed = (1.0 - float(blend)) * base + float(blend) * robust
        policies.append(
            torch.as_tensor(mixed, device=props.device, dtype=props.dtype).reshape(-1, 1)
        )
        diagnostics.append(
            {
                "snapshot": evaluation.source_start + sample,
                "neighbors": " ".join(str(int(index)) for index in selected),
                "robust_nominal_policy_traffic": float(
                    np.dot(np.repeat(scenarios[1][0], K), robust)
                ),
                "base_policy_mass_max": float(base.reshape(-1, K).sum(axis=1).max()),
                "robust_policy_mass_max": float(robust.reshape(-1, K).sum(axis=1).max()),
            }
        )
    elapsed = time.perf_counter() - started
    for row in diagnostics:
        row["planning_seconds_per_snapshot"] = elapsed / max(len(evaluation), 1)
    return replace(
        evaluation,
        policies=(
            evaluation.policies[0],
            torch.stack(policies, dim=0),
            evaluation.policies[2],
        ),
    ), diagnostics


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def metric_gaps(candidate: list[dict], baseline: list[dict]) -> dict[str, float]:
    c = summary_index(candidate)
    b = summary_index(baseline)
    return {
        f"{name.lower()}_{metric}_gap": float(c[name][f"norm_fulfill_{metric}"])
        - float(b[name][f"norm_fulfill_{metric}"])
        for name in runtime.CLASSES
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
    scenario_count: int,
    neighbor_mode: str,
    low_factor: float,
    blend: float,
    force: bool,
) -> Path:
    setting = LEVELS[level]
    start, end = setting["range"]
    run_dir = (
        OUTPUT_ROOT
        / setting["label"]
        / neighbor_mode
        / f"scenarios_{scenario_count}"
        / f"low_{format(low_factor, '.3g').replace('.', 'p')}"
        / f"blend_{format(blend, '.3g').replace('.', 'p')}"
        / f"backbone_{backbone_seed}"
    )
    if (run_dir / "complete.json").exists() and not force:
        return run_dir
    if force:
        safe_remove(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(int(setting["backbone_level"]), device)
    model, checkpoint = runtime.load_backbone(
        int(setting["backbone_level"]), backbone_seed, props, device
    )
    history = runtime.build_policy_cache(model, props, 0, start, batch_size=16)
    evaluation = runtime.build_policy_cache(model, props, start, end, batch_size=16)
    baseline_rows, baseline_summary = runtime.evaluate_cache(model, props, evaluation, None)
    candidate_cache, diagnostics = planned_cache(
        model,
        props,
        history,
        evaluation,
        scenario_count,
        neighbor_mode,
        low_factor,
        blend,
    )
    candidate_rows, candidate_summary = runtime.evaluate_cache(
        model, props, candidate_cache, None
    )
    gaps = metric_gaps(candidate_summary, baseline_summary)
    feasible = (
        summary_index(candidate_summary)["High"]["norm_fulfill_mean"] >= 0.995
        and abs(gaps["high_mean_gap"]) <= 1e-6
        and gaps["medium_mean_gap"] > 0.0
        and gaps["medium_p1_gap"] > 0.0
        and gaps["medium_p10_gap"] > 0.0
        and gaps["low_mean_gap"] >= -0.003
        and gaps["low_p10_gap"] >= -0.01
    )
    config = {
        "method": "K-nearest ESM error-scenario robust Medium routing",
        "level": level,
        "range": [start, end],
        "history_range": [0, start],
        "backbone_checkpoint": str(checkpoint),
        "scenario_count": scenario_count,
        "neighbor_mode": neighbor_mode,
        "low_factor": low_factor,
        "blend": blend,
        "inference_contract": "current policy sees current ESM prediction plus historical prediction errors only; current actual TM is replay/evaluation only",
    }
    runtime.write_json(run_dir / "config.json", config)
    runtime.write_csv(run_dir / "baseline_metrics.csv", baseline_rows)
    runtime.write_csv(run_dir / "candidate_metrics.csv", candidate_rows)
    runtime.write_csv(run_dir / "diagnostics.csv", diagnostics)
    result = {
        "status": "complete",
        "feasible": feasible,
        "gaps": gaps,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "planning_seconds_per_snapshot": diagnostics[0]["planning_seconds_per_snapshot"],
    }
    runtime.write_json(run_dir / "summary.json", result)
    runtime.write_json(run_dir / "complete.json", {key: result[key] for key in ("status", "feasible", "gaps")})
    print(json.dumps({key: result[key] for key in ("status", "feasible", "gaps")}), flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=tuple(LEVELS), required=True)
    parser.add_argument("--backbone-seed", type=int, default=490)
    parser.add_argument("--scenario-count", type=int, default=4)
    parser.add_argument("--neighbor-mode", choices=("knn", "recent"), default="knn")
    parser.add_argument("--low-factor", type=float, default=1.0)
    parser.add_argument("--blend", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_one(
        args.level,
        args.backbone_seed,
        args.scenario_count,
        args.neighbor_mode,
        args.low_factor,
        args.blend,
        args.force,
    )


if __name__ == "__main__":
    main()
