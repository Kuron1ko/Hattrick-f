from __future__ import annotations

import importlib.util
import json
import math
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
RUNTIME_PATH = TEST_DIR / "shared2x_medium_adapter" / "run_experiment.py"
METHOD_PATH = TEST_DIR / "shared2x_active_set_router" / "probe_esm_self_correction.py"
K = 8
EXPLORATION = (350, 358)
VALIDATION = (358, 400)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runtime = load_module("shared2x_hofr_runtime", RUNTIME_PATH)
method = load_module("shared2x_hofr_metrics", METHOD_PATH)


class HighPolicyAdapter:
    def adapt_batch(self, policies: list[torch.Tensor], batch: dict) -> list[torch.Tensor]:
        policies[0] = batch["path_features"].unsqueeze(-1)
        return policies


def build_esm_only_cache(model, props, start: int, stop: int, batch_size: int):
    """Build policies with ESM predictions in every online-TM input slot.

    Actual TMs are retained only in cache.tms for the final admission replay.
    """
    dataset = runtime.DM_Dataset_within_Cluster(props, 0, start, stop)
    path_masks = runtime.shared.base.move_dataset_static(dataset, props.device)
    keys = (
        "p0", "p1", "p2", "tm0", "tm1", "tm2", "ptm0", "ptm1", "ptm2", "cap",
        "of0", "of1", "of2", "om0", "om1", "om2",
    )
    buckets = {key: [] for key in keys}
    loader = runtime.shared.data_loader(dataset, batch_size, False, 0)
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    with torch.no_grad():
        for inputs in loader:
            actual_values = list(runtime.shared.unpack_to_device(inputs, props))
            policy_values = list(actual_values)
            for actual_index, predicted_index in ((2, 3), (4, 5), (6, 7)):
                policy_values[actual_index] = actual_values[predicted_index]
            policies = runtime.cached_policy_forward(
                model, props, dataset, tuple(policy_values), path_masks
            )
            batch = int(actual_values[2].shape[0])
            capacities = actual_values[1]
            if capacities.shape[0] == 1 and batch > 1:
                capacities = capacities.expand(batch, -1)
            for index in range(3):
                buckets[f"p{index}"].append(policies[index].detach())
            buckets["tm0"].append(actual_values[2].detach())
            buckets["tm1"].append(actual_values[4].detach())
            buckets["tm2"].append(actual_values[6].detach())
            buckets["ptm0"].append(actual_values[3].detach())
            buckets["ptm1"].append(actual_values[5].detach())
            buckets["ptm2"].append(actual_values[7].detach())
            buckets["cap"].append(capacities.detach())
            buckets["of0"].append(actual_values[11].reshape(-1).detach())
            buckets["of1"].append((actual_values[12] - actual_values[11]).reshape(-1).detach())
            buckets["of2"].append((actual_values[13] - actual_values[12]).reshape(-1).detach())
            buckets["om0"].append(actual_values[8].reshape(-1).detach())
            buckets["om1"].append(actual_values[9].reshape(-1).detach())
            buckets["om2"].append(actual_values[10].reshape(-1).detach())
    props.research_return_policy = False
    joined = {key: torch.cat(value, dim=0) for key, value in buckets.items()}
    return runtime.PolicyCache(
        dataset=dataset,
        path_masks=path_masks,
        source_start=start,
        policies=(joined["p0"], joined["p1"], joined["p2"]),
        tms=(joined["tm0"], joined["tm1"], joined["tm2"]),
        predicted_tms=(joined["ptm0"], joined["ptm1"], joined["ptm2"]),
        capacities=joined["cap"],
        oracle_flows=(joined["of0"], joined["of1"], joined["of2"]),
        oracle_mlus=(joined["om0"], joined["om1"], joined["om2"]),
    )


def scipy_path_edge(dataset) -> sparse.csr_matrix:
    pte = dataset.pte.coalesce().to(device="cpu", dtype=torch.float64)
    indices = pte.indices().numpy()
    values = pte.values().numpy()
    # dataset.pte is [path, edge]; optimization constraints need [edge, path].
    return sparse.csr_matrix(
        (values, (indices[1], indices[0])),
        shape=(int(pte.shape[1]), int(pte.shape[0])),
    )


def od_path_matrix(path_count: int) -> sparse.csr_matrix:
    if path_count % K:
        raise RuntimeError(f"Path count {path_count} is not divisible by K={K}")
    od_count = path_count // K
    return sparse.csr_matrix(
        (
            np.ones(path_count, dtype=np.float64),
            (np.repeat(np.arange(od_count), K), np.arange(path_count)),
        ),
        shape=(od_count, path_count),
    )


def normalize_cost(values: np.ndarray) -> np.ndarray:
    positive = values[values > 1e-12]
    scale = float(np.median(positive)) if positive.size else 1.0
    return values / max(scale, 1e-12)


def optimize_one(
    edge_path: sparse.csr_matrix,
    od_path: sparse.csr_matrix,
    base_policies: tuple[np.ndarray, np.ndarray, np.ndarray],
    predicted: tuple[np.ndarray, np.ndarray, np.ndarray],
    capacity: np.ndarray,
    margin: float,
    low_price_weight: float,
    anchor_weight: float,
) -> tuple[np.ndarray, dict]:
    path_count = edge_path.shape[1]
    base_high, base_medium, base_low = base_policies
    high_tm, medium_tm, low_tm = predicted
    valid = base_high > 0.0
    valid_indices = np.flatnonzero(valid)
    if not valid_indices.size:
        raise RuntimeError("No valid High paths")
    edge_valid = edge_path[:, valid_indices]
    od_valid = od_path[:, valid_indices]
    demand_od = high_tm.reshape(-1, K)[:, 0]
    effective_capacity = np.maximum(capacity * (1.0 - margin), 1e-9)
    constraints = sparse.vstack([edge_valid, od_valid], format="csr")
    bounds = np.concatenate([effective_capacity, demand_od])

    primary = linprog(
        -np.ones(valid_indices.size, dtype=np.float64),
        A_ub=constraints,
        b_ub=bounds,
        bounds=(0.0, None),
        method="highs",
    )
    if not primary.success:
        raise RuntimeError(f"High primary LP failed: {primary.message}")
    high_optimum = float(-primary.fun)

    base_high_flow = base_high * high_tm
    base_medium_flow = base_medium * medium_tm
    base_low_flow = base_low * low_tm
    base_high_load = np.asarray(edge_path @ base_high_flow).reshape(-1)
    medium_load = np.asarray(edge_path @ base_medium_flow).reshape(-1)
    low_load = np.asarray(edge_path @ base_low_flow).reshape(-1)
    residual_after_high = np.maximum(capacity - base_high_load, 0.03 * capacity)
    residual_after_medium = np.maximum(
        capacity - base_high_load - medium_load, 0.03 * capacity
    )
    medium_shadow = medium_load / np.square(residual_after_high + 1e-9)
    low_shadow = low_load / np.square(residual_after_medium + 1e-9)
    edge_price = normalize_cost(medium_shadow) + float(low_price_weight) * normalize_cost(low_shadow)
    path_price = np.asarray(edge_path.T @ edge_price).reshape(-1)
    path_price = normalize_cost(path_price)
    network_anchor = -np.log(np.clip(base_high, 1e-12, None))
    objective = path_price + float(anchor_weight) * normalize_cost(network_anchor)

    tolerance = max(1e-7, high_optimum * 1e-7)
    secondary_constraints = sparse.vstack(
        [constraints, -np.ones((1, valid_indices.size), dtype=np.float64)],
        format="csr",
    )
    secondary_bounds = np.concatenate([bounds, [-(high_optimum - tolerance)]])
    secondary = linprog(
        objective[valid_indices],
        A_ub=secondary_constraints,
        b_ub=secondary_bounds,
        bounds=(0.0, None),
        method="highs",
    )
    if not secondary.success:
        raise RuntimeError(f"High secondary LP failed: {secondary.message}")
    flow = np.zeros(path_count, dtype=np.float64)
    flow[valid_indices] = secondary.x
    grouped_flow = flow.reshape(-1, K)
    accepted_by_od = grouped_flow.sum(axis=1, keepdims=True)
    grouped_policy = np.divide(
        grouped_flow,
        accepted_by_od,
        out=np.zeros_like(grouped_flow),
        where=accepted_by_od > 1e-12,
    )
    missing = accepted_by_od.squeeze(-1) <= 1e-12
    if np.any(missing):
        fallback = base_high.reshape(-1, K)[missing]
        fallback_sum = fallback.sum(axis=1, keepdims=True)
        grouped_policy[missing] = np.divide(
            fallback,
            fallback_sum,
            out=np.zeros_like(fallback),
            where=fallback_sum > 1e-12,
        )
    policy = grouped_policy.reshape(-1)
    group_sum = grouped_policy.sum(axis=1)
    predicted_edge_load = np.asarray(edge_path @ flow).reshape(-1)
    return policy, {
        "high_optimum": high_optimum,
        "secondary_total": float(flow.sum()),
        "max_effective_capacity_ratio": float(
            np.max(predicted_edge_load / effective_capacity)
        ),
        "max_group_policy_sum": float(group_sum.max()),
        "mean_group_policy_sum": float(group_sum.mean()),
        "path_l1_from_network": float(np.mean(np.abs(policy - base_high))),
        "changed_path_fraction": float(np.mean(np.abs(policy - base_high) > 1e-5)),
    }


def build_hofr_cache(cache, config: tuple[float, float, float]):
    margin, low_price_weight, anchor_weight = config
    edge_path = scipy_path_edge(cache.dataset)
    od_path = od_path_matrix(edge_path.shape[1])
    output = []
    diagnostics = []
    started = time.perf_counter()
    for index in range(len(cache)):
        policies = tuple(
            value[index].squeeze(-1).detach().cpu().numpy().astype(np.float64)
            for value in cache.policies
        )
        predicted = tuple(
            value[index].squeeze(-1).detach().cpu().numpy().astype(np.float64)
            for value in cache.predicted_tms
        )
        capacity = cache.capacities[index].detach().cpu().numpy().astype(np.float64)
        policy, record = optimize_one(
            edge_path, od_path, policies, predicted, capacity,
            margin, low_price_weight, anchor_weight,
        )
        output.append(torch.as_tensor(policy, device=cache.capacities.device, dtype=cache.policies[0].dtype))
        diagnostics.append(record)
    features = torch.stack(output, dim=0)
    return replace(cache, path_features=features), {
        "seconds": time.perf_counter() - started,
        "path_l1_mean": float(np.mean([row["path_l1_from_network"] for row in diagnostics])),
        "changed_path_fraction_mean": float(
            np.mean([row["changed_path_fraction"] for row in diagnostics])
        ),
        "max_effective_capacity_ratio": float(
            max(row["max_effective_capacity_ratio"] for row in diagnostics)
        ),
        "high_optimum_mean": float(np.mean([row["high_optimum"] for row in diagnostics])),
    }


def predicted_summary(model, props, cache, high_policy: torch.Tensor) -> dict:
    policies = [high_policy.unsqueeze(-1), cache.policies[1], cache.policies[2]]
    values = method.predicted_fulfillment(model, props, cache, policies)
    return {
        name: float(values[:, index].mean().item())
        for index, name in enumerate(("High", "Medium", "Low"))
    }


def config_result(model, props, cache, baseline_summary, config):
    candidate_cache, diagnostics = build_hofr_cache(cache, config)
    adapter = HighPolicyAdapter()
    rows, summary = runtime.evaluate_cache(
        model, props, candidate_cache, adapter, batch_size=16
    )
    delta = method.gaps(summary, baseline_summary)
    predicted_base = predicted_summary(
        model, props, cache, cache.policies[0].squeeze(-1)
    )
    predicted_candidate = predicted_summary(
        model, props, cache, candidate_cache.path_features
    )
    feasible = (
        method.summary_index(summary)["High"]["norm_fulfill_mean"] >= 0.995
        and delta["High.norm_fulfill_mean"] >= -0.0002
        and delta["Low.norm_fulfill_mean"] >= -0.005
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )
    medium_score = (
        delta["Medium.norm_fulfill_mean"]
        + 0.35 * delta["Medium.norm_fulfill_p1"]
        + 0.35 * delta["Medium.norm_fulfill_p10"]
    )
    return {
        "config": {
            "robust_capacity_margin": config[0],
            "low_shadow_weight": config[1],
            "network_anchor_weight": config[2],
        },
        "feasible": bool(feasible),
        "medium_score": float(medium_score),
        "metrics": method.compact(summary),
        "delta": delta,
        "predicted_fulfillment_base": predicted_base,
        "predicted_fulfillment_candidate": predicted_candidate,
        "diagnostics": diagnostics,
        "rows": rows,
        "candidate_cache": candidate_cache,
    }


def rank(result: dict) -> tuple:
    delta = result["delta"]
    return (
        int(result["feasible"]),
        float(result["medium_score"]),
        float(delta["Low.norm_fulfill_mean"]),
        float(delta["High.norm_fulfill_mean"]),
    )


def sensitivity_audit(cache, config) -> dict:
    one = replace(
        cache,
        policies=tuple(value[:1].clone() for value in cache.policies),
        tms=tuple(value[:1].clone() for value in cache.tms),
        predicted_tms=tuple(value[:1].clone() for value in cache.predicted_tms),
        capacities=cache.capacities[:1].clone(),
        oracle_flows=tuple(value[:1].clone() for value in cache.oracle_flows),
        oracle_mlus=tuple(value[:1].clone() for value in cache.oracle_mlus),
    )
    original, original_diag = build_hofr_cache(one, config)
    medium = one.predicted_tms[1].reshape(1, -1, K).roll(shifts=37, dims=1).reshape_as(
        one.predicted_tms[1]
    )
    changed_medium = replace(
        one,
        predicted_tms=(one.predicted_tms[0], medium, one.predicted_tms[2]),
    )
    altered, altered_diag = build_hofr_cache(changed_medium, config)
    actual_changed = replace(
        one,
        tms=tuple(torch.randn_like(value) * 1000.0 for value in one.tms),
    )
    actual_only, _ = build_hofr_cache(actual_changed, config)
    return {
        "high_policy_l1_after_medium_change": float(
            torch.mean(torch.abs(original.path_features - altered.path_features)).item()
        ),
        "high_total_flow_target_original": original_diag["high_optimum_mean"],
        "high_total_flow_target_changed_medium": altered_diag["high_optimum_mean"],
        "target_absolute_difference": abs(
            original_diag["high_optimum_mean"] - altered_diag["high_optimum_mean"]
        ),
        "actual_tm_independence_max_abs_delta": float(
            torch.max(torch.abs(original.path_features - actual_only.path_features)).item()
        ),
    }


def without_internal(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in ("rows", "candidate_cache")}


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    exploration_cache = build_esm_only_cache(model, props, *EXPLORATION, batch_size=16)
    exploration_rows, exploration_baseline = runtime.evaluate_cache(
        model, props, exploration_cache, None, batch_size=16
    )

    configurations = [
        (margin, low_weight, anchor)
        for margin in (0.0, 0.03, 0.07)
        for low_weight in (0.25, 1.0)
        for anchor in (0.02, 0.10)
    ]
    exploration = []
    for config in configurations:
        result = config_result(
            model, props, exploration_cache, exploration_baseline, config
        )
        exploration.append(result)
        print(json.dumps(without_internal(result), ensure_ascii=False), flush=True)
    winner = max(exploration, key=rank)
    frozen_config = (
        winner["config"]["robust_capacity_margin"],
        winner["config"]["low_shadow_weight"],
        winner["config"]["network_anchor_weight"],
    )

    validation_cache = build_esm_only_cache(model, props, *VALIDATION, batch_size=32)
    validation_rows, validation_baseline = runtime.evaluate_cache(
        model, props, validation_cache, None, batch_size=32
    )
    validation = config_result(
        model, props, validation_cache, validation_baseline, frozen_config
    )
    audit = sensitivity_audit(exploration_cache, frozen_config)
    payload = {
        "method": "HOFR core probe: ESM-conditioned High optimal-face reselection",
        "scope": (
            "core-mechanism test using the frozen Hattrick network as the ML policy prior; "
            "not a newly trained HOFR model"
        ),
        "traffic_scale": "2x",
        "input_contract": {
            "network_and_optimizer": "ESM-predicted TMs only",
            "actual_tms": "sequential-admission evaluation only",
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.sha256(checkpoint),
        "selection_protocol": {
            "exploration": list(EXPLORATION),
            "validation": list(VALIDATION),
            "final_test_400_500_touched": False,
        },
        "baseline_exploration": method.compact(exploration_baseline),
        "exploration": [without_internal(value) for value in exploration],
        "frozen_config": winner["config"],
        "validation_baseline": method.compact(validation_baseline),
        "validation": without_internal(validation),
        "mechanism_audit": audit,
    }
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    runtime.write_json(THIS_DIR / "hofr_core_probe.json", payload)
    runtime.write_csv(THIS_DIR / "validation_baseline_rows.csv", validation_rows)
    runtime.write_csv(THIS_DIR / "validation_hofr_rows.csv", validation["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
