from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog


THIS_DIR = Path(__file__).resolve().parent
CORE_PATH = THIS_DIR / "probe_high_optimal_face.py"
spec = importlib.util.spec_from_file_location("shared2x_hofr_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load HOFR core")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
runtime = core.runtime
method = core.method
K = core.K
CALIBRATION = (0, 350)
EXPLORATION = (350, 358)
VALIDATION = (358, 400)


class ThreePolicyAdapter:
    def __init__(self, path_count: int):
        self.path_count = int(path_count)

    def adapt_batch(self, policies: list[torch.Tensor], batch: dict) -> list[torch.Tensor]:
        features = batch["path_features"]
        for index in range(3):
            start = index * self.path_count
            policies[index] = features[:, start : start + self.path_count].unsqueeze(-1)
        return policies


def fit_affine_calibrator(cache) -> list[dict[str, np.ndarray]]:
    calibrators = []
    for actual, predicted in zip(cache.tms, cache.predicted_tms):
        x = predicted[:, :, 0].reshape(len(cache), -1, K)[:, :, 0].cpu().numpy().astype(np.float64)
        y = actual[:, :, 0].reshape(len(cache), -1, K)[:, :, 0].cpu().numpy().astype(np.float64)
        x_mean = x.mean(axis=0)
        y_mean = y.mean(axis=0)
        centered = x - x_mean
        variance = np.square(centered).sum(axis=0)
        slope = (centered * (y - y_mean)).sum(axis=0) / np.maximum(variance, 1e-9)
        slope = np.clip(slope, 0.0, 2.0)
        intercept = y_mean - slope * x_mean
        fitted = np.maximum(x * slope + intercept, 0.0)
        residual = y - fitted
        calibrators.append(
            {
                "slope": slope,
                "intercept": intercept,
                "residual_std": residual.std(axis=0, ddof=1),
                "train_mae_raw": np.mean(np.abs(y - x), axis=0),
                "train_mae_calibrated": np.mean(np.abs(residual), axis=0),
            }
        )
    return calibrators


def calibrated_demand(
    repeated_prediction: np.ndarray,
    calibrator: dict[str, np.ndarray],
    blend: float,
    uncertainty: float,
) -> np.ndarray:
    raw = repeated_prediction.reshape(-1, K)[:, 0]
    calibrated = np.maximum(
        raw * calibrator["slope"] + calibrator["intercept"], 0.0
    )
    estimate = (1.0 - blend) * raw + blend * calibrated
    estimate = estimate + float(uncertainty) * calibrator["residual_std"]
    return np.maximum(estimate, 0.0)


def normalized_anchor(policy: np.ndarray) -> np.ndarray:
    values = -np.log(np.clip(policy, 1e-12, None))
    positive = values[np.isfinite(values) & (values > 1e-12)]
    scale = float(np.median(positive)) if positive.size else 1.0
    return values / max(scale, 1e-12)


def normalized_policy_from_flow(flow: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    grouped = flow.reshape(-1, K)
    totals = grouped.sum(axis=1, keepdims=True)
    policy = np.divide(
        grouped, totals, out=np.zeros_like(grouped), where=totals > 1e-12
    )
    missing = totals.squeeze(-1) <= 1e-12
    if np.any(missing):
        base = fallback.reshape(-1, K)[missing]
        base_sum = base.sum(axis=1, keepdims=True)
        policy[missing] = np.divide(
            base, base_sum, out=np.zeros_like(base), where=base_sum > 1e-12
        )
    return policy.reshape(-1)


def solve_lp(c, constraints, bounds):
    result = linprog(
        c,
        A_ub=constraints,
        b_ub=bounds,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result


def joint_optimize_one(
    edge_path: sparse.csr_matrix,
    od_path: sparse.csr_matrix,
    policies: tuple[np.ndarray, np.ndarray, np.ndarray],
    repeated_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    capacity: np.ndarray,
    calibrators,
    config: tuple[float, float, float],
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    blend, uncertainty, medium_slack_fraction = config
    demands = [
        calibrated_demand(prediction, calibration, blend, uncertainty)
        for prediction, calibration in zip(repeated_predictions, calibrators)
    ]
    valid_indices = [np.flatnonzero(policy > 0.0) for policy in policies]
    edge = [edge_path[:, indices] for indices in valid_indices]
    od = [od_path[:, indices] for indices in valid_indices]
    anchors = [normalized_anchor(policy)[indices] for policy, indices in zip(policies, valid_indices)]

    primary_constraints = sparse.vstack([edge[0], od[0]], format="csr")
    primary_bounds = np.concatenate([capacity, demands[0]])
    primary = solve_lp(
        -np.ones(len(valid_indices[0]), dtype=np.float64),
        primary_constraints,
        primary_bounds,
    )
    high_optimum = float(-primary.fun)
    high_tolerance = max(1e-6, 1e-5 * high_optimum)

    high_count, medium_count = len(valid_indices[0]), len(valid_indices[1])
    edge_hm = sparse.hstack([edge[0], edge[1]], format="csr")
    od_h = sparse.hstack(
        [od[0], sparse.csr_matrix((od[0].shape[0], medium_count))], format="csr"
    )
    od_m = sparse.hstack(
        [sparse.csr_matrix((od[1].shape[0], high_count)), od[1]], format="csr"
    )
    high_floor_hm = sparse.csr_matrix(
        np.concatenate([-np.ones(high_count), np.zeros(medium_count)])[None, :]
    )
    hm_constraints = sparse.vstack([edge_hm, od_h, od_m, high_floor_hm], format="csr")
    hm_bounds = np.concatenate(
        [capacity, demands[0], demands[1], [-(high_optimum - high_tolerance)]]
    )
    medium_objective = np.concatenate(
        [np.zeros(high_count), -np.ones(medium_count)]
    )
    secondary = solve_lp(medium_objective, hm_constraints, hm_bounds)
    medium_optimum = float(secondary.x[high_count:].sum())
    medium_target = medium_optimum * (1.0 - float(medium_slack_fraction))
    medium_tolerance = max(1e-6, 1e-5 * medium_target)

    counts = [len(indices) for indices in valid_indices]
    total_count = sum(counts)
    edge_hml = sparse.hstack(edge, format="csr")
    od_blocks = []
    for class_index in range(3):
        blocks = []
        for other in range(3):
            if class_index == other:
                blocks.append(od[class_index])
            else:
                blocks.append(sparse.csr_matrix((od[class_index].shape[0], counts[other])))
        od_blocks.append(sparse.hstack(blocks, format="csr"))
    high_floor = sparse.csr_matrix(
        np.concatenate([-np.ones(counts[0]), np.zeros(counts[1] + counts[2])])[None, :]
    )
    medium_floor = sparse.csr_matrix(
        np.concatenate(
            [np.zeros(counts[0]), -np.ones(counts[1]), np.zeros(counts[2])]
        )[None, :]
    )
    hml_constraints = sparse.vstack(
        [edge_hml, *od_blocks, high_floor, medium_floor], format="csr"
    )
    hml_bounds = np.concatenate(
        [
            capacity,
            demands[0], demands[1], demands[2],
            [-(high_optimum - high_tolerance), -(medium_target - medium_tolerance)],
        ]
    )
    low_objective = np.concatenate(
        [np.zeros(counts[0] + counts[1]), -np.ones(counts[2])]
    )
    tertiary = solve_lp(low_objective, hml_constraints, hml_bounds)
    low_optimum = float(tertiary.x[counts[0] + counts[1] :].sum())
    low_tolerance = max(1e-6, 1e-5 * low_optimum)

    low_floor = sparse.csr_matrix(
        np.concatenate([np.zeros(counts[0] + counts[1]), -np.ones(counts[2])])[None, :]
    )
    anchor_constraints = sparse.vstack([hml_constraints, low_floor], format="csr")
    anchor_bounds = np.concatenate([hml_bounds, [-(low_optimum - low_tolerance)]])
    anchor_objective = np.concatenate(anchors)
    anchored = solve_lp(anchor_objective, anchor_constraints, anchor_bounds)

    flows = []
    offset = 0
    for count in counts:
        full = np.zeros(edge_path.shape[1], dtype=np.float64)
        full[valid_indices[len(flows)]] = anchored.x[offset : offset + count]
        flows.append(full)
        offset += count
    output = tuple(
        normalized_policy_from_flow(flow, policy)
        for flow, policy in zip(flows, policies)
    )
    estimated_edge_load = sum(
        np.asarray(edge_path @ (policy * np.repeat(demand, K))).reshape(-1)
        for policy, demand in zip(output, demands)
    )
    return output, {
        "high_optimum": high_optimum,
        "medium_optimum": medium_optimum,
        "medium_target_after_slack": medium_target,
        "low_optimum": low_optimum,
        "normalized_policy_capacity_ratio": float(
            np.max(estimated_edge_load / np.maximum(capacity, 1e-9))
        ),
        "high_path_l1": float(np.mean(np.abs(output[0] - policies[0]))),
        "medium_path_l1": float(np.mean(np.abs(output[1] - policies[1]))),
        "low_path_l1": float(np.mean(np.abs(output[2] - policies[2]))),
    }


def build_joint_cache(cache, calibrators, config):
    edge_path = core.scipy_path_edge(cache.dataset)
    od_path = core.od_path_matrix(edge_path.shape[1])
    outputs = [[], [], []]
    diagnostics = []
    started = time.perf_counter()
    for index in range(len(cache)):
        policies = tuple(
            value[index].squeeze(-1).cpu().numpy().astype(np.float64)
            for value in cache.policies
        )
        predicted = tuple(
            value[index].squeeze(-1).cpu().numpy().astype(np.float64)
            for value in cache.predicted_tms
        )
        capacity = cache.capacities[index].cpu().numpy().astype(np.float64)
        result, record = joint_optimize_one(
            edge_path, od_path, policies, predicted, capacity, calibrators, config
        )
        for class_index in range(3):
            outputs[class_index].append(
                torch.as_tensor(
                    result[class_index],
                    device=cache.capacities.device,
                    dtype=cache.policies[class_index].dtype,
                )
            )
        diagnostics.append(record)
    features = torch.cat(
        [torch.stack(value, dim=0) for value in outputs], dim=1
    )
    return replace(cache, path_features=features), {
        "seconds": time.perf_counter() - started,
        "high_path_l1_mean": float(np.mean([row["high_path_l1"] for row in diagnostics])),
        "medium_path_l1_mean": float(np.mean([row["medium_path_l1"] for row in diagnostics])),
        "low_path_l1_mean": float(np.mean([row["low_path_l1"] for row in diagnostics])),
        "max_normalized_policy_capacity_ratio": float(
            max(row["normalized_policy_capacity_ratio"] for row in diagnostics)
        ),
        "high_optimum_mean": float(np.mean([row["high_optimum"] for row in diagnostics])),
        "medium_optimum_mean": float(np.mean([row["medium_optimum"] for row in diagnostics])),
        "low_optimum_mean": float(np.mean([row["low_optimum"] for row in diagnostics])),
    }


def evaluate_config(model, props, cache, calibrators, baseline, config):
    candidate_cache, diagnostics = build_joint_cache(cache, calibrators, config)
    adapter = ThreePolicyAdapter(int(cache.policies[0].shape[1]))
    rows, summary = runtime.evaluate_cache(
        model, props, candidate_cache, adapter, batch_size=16
    )
    delta = method.gaps(summary, baseline)
    feasible = (
        method.summary_index(summary)["High"]["norm_fulfill_mean"] >= 0.995
        and delta["Low.norm_fulfill_mean"] >= -0.005
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )
    score = (
        delta["Medium.norm_fulfill_mean"]
        + 0.35 * delta["Medium.norm_fulfill_p1"]
        + 0.35 * delta["Medium.norm_fulfill_p10"]
    )
    return {
        "config": {
            "affine_calibration_blend": config[0],
            "residual_uncertainty_std": config[1],
            "medium_slack_fraction": config[2],
        },
        "feasible": bool(feasible),
        "medium_score": float(score),
        "metrics": method.compact(summary),
        "delta": delta,
        "diagnostics": diagnostics,
        "rows": rows,
        "candidate_cache": candidate_cache,
    }


def rank(result):
    delta = result["delta"]
    return (
        int(result["feasible"]),
        float(result["medium_score"]),
        float(delta["Low.norm_fulfill_mean"]),
        float(delta["High.norm_fulfill_mean"]),
    )


def public(result):
    return {key: value for key, value in result.items() if key not in ("rows", "candidate_cache")}


def mechanism_audit(cache, calibrators, config):
    one = core.probe.slice_cache(cache, 0, 1) if hasattr(core, "probe") else replace(
        cache,
        policies=tuple(value[:1].clone() for value in cache.policies),
        tms=tuple(value[:1].clone() for value in cache.tms),
        predicted_tms=tuple(value[:1].clone() for value in cache.predicted_tms),
        capacities=cache.capacities[:1].clone(),
        oracle_flows=tuple(value[:1].clone() for value in cache.oracle_flows),
        oracle_mlus=tuple(value[:1].clone() for value in cache.oracle_mlus),
    )
    original, original_diag = build_joint_cache(one, calibrators, config)
    medium = one.predicted_tms[1].reshape(1, -1, K).roll(37, dims=1).reshape_as(
        one.predicted_tms[1]
    )
    altered_input = replace(
        one,
        predicted_tms=(one.predicted_tms[0], medium, one.predicted_tms[2]),
    )
    altered, altered_diag = build_joint_cache(altered_input, calibrators, config)
    actual_changed = replace(
        one,
        tms=tuple(torch.randn_like(value) * 1000.0 for value in one.tms),
    )
    actual_only, _ = build_joint_cache(actual_changed, calibrators, config)
    path_count = int(one.policies[0].shape[1])
    return {
        "high_policy_l1_after_medium_change": float(
            torch.mean(
                torch.abs(
                    original.path_features[:, :path_count]
                    - altered.path_features[:, :path_count]
                )
            ).item()
        ),
        "high_optimum_original": original_diag["high_optimum_mean"],
        "high_optimum_changed_medium": altered_diag["high_optimum_mean"],
        "high_optimum_absolute_difference": abs(
            original_diag["high_optimum_mean"] - altered_diag["high_optimum_mean"]
        ),
        "actual_tm_independence_max_abs_delta": float(
            torch.max(torch.abs(original.path_features - actual_only.path_features)).item()
        ),
    }


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    calibration_cache = core.build_esm_only_cache(
        model, props, *CALIBRATION, batch_size=64
    )
    calibrators = fit_affine_calibrator(calibration_cache)
    calibration_summary = {
        name: {
            "raw_mae_mean": float(value["train_mae_raw"].mean()),
            "calibrated_mae_mean": float(value["train_mae_calibrated"].mean()),
        }
        for name, value in zip(("High", "Medium", "Low"), calibrators)
    }

    exploration_cache = core.build_esm_only_cache(
        model, props, *EXPLORATION, batch_size=16
    )
    exploration_rows, exploration_baseline = runtime.evaluate_cache(
        model, props, exploration_cache, None, batch_size=16
    )
    configs = [
        (1.0, uncertainty, medium_slack)
        for uncertainty in (0.25, 0.50, 0.75)
        for medium_slack in (0.0, 0.005, 0.010)
    ]
    exploration = []
    for config in configs:
        result = evaluate_config(
            model, props, exploration_cache, calibrators, exploration_baseline, config
        )
        exploration.append(result)
        print(json.dumps(public(result), ensure_ascii=False), flush=True)
    winner = max(exploration, key=rank)
    frozen = (
        winner["config"]["affine_calibration_blend"],
        winner["config"]["residual_uncertainty_std"],
        winner["config"]["medium_slack_fraction"],
    )

    validation_cache = core.build_esm_only_cache(
        model, props, *VALIDATION, batch_size=32
    )
    validation_rows, validation_baseline = runtime.evaluate_cache(
        model, props, validation_cache, None, batch_size=32
    )
    validation = evaluate_config(
        model, props, validation_cache, calibrators, validation_baseline, frozen
    )
    payload = {
        "method": "joint HOFR probe with train-only affine ESM calibration",
        "scope": "exact lexicographic layer probe; frozen Hattrick supplies ML path prior",
        "traffic_scale": "2x",
        "input_contract": {
            "offline_calibration": "actual and ESM TMs from 0-349",
            "online_policy": "ESM prediction only",
            "actual_validation_tm": "admission replay only",
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.sha256(checkpoint),
        "selection_protocol": {
            "calibration": list(CALIBRATION),
            "exploration": list(EXPLORATION),
            "validation": list(VALIDATION),
            "final_test_400_500_touched": False,
        },
        "calibration": calibration_summary,
        "exploration_baseline": method.compact(exploration_baseline),
        "exploration": [public(value) for value in exploration],
        "frozen_config": winner["config"],
        "validation_baseline": method.compact(validation_baseline),
        "validation": public(validation),
        "mechanism_audit": mechanism_audit(exploration_cache, calibrators, frozen),
    }
    runtime.write_json(THIS_DIR / "hofr_joint_probe.json", payload)
    runtime.write_csv(THIS_DIR / "joint_validation_baseline_rows.csv", validation_rows)
    runtime.write_csv(THIS_DIR / "joint_validation_candidate_rows.csv", validation["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
