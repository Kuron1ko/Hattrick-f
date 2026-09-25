from __future__ import annotations

"""Read-only diagnosis of the frozen persistent core on snapshots 350--399.

This script never constructs snapshots 400--499.  Actual traffic is used only
for retrospective admission/upper-bound diagnosis; every model policy and the
counterfactual residual-LP policy use ESM predictions.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "test_diff_path"
BASE = TEST_DIR / "shared2x_sparse_path_cross_attention_persistent"
CORE_RUNNER = BASE / "run_experiment.py"
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
STRICT_RUNNER = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "evaluate_strict_esm_sequential.py"
)
GUARD_RUNNER = BASE / "train_low_guard.py"
CORE_CHECKPOINT = (
    BASE / "level4_selection_validation_only" / "selected_checkpoint.pt"
)
NATIVE_CHECKPOINT = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "final_model.pt"
)
GUARD_CHECKPOINT = (
    BASE
    / "artifacts"
    / "development_stronger_low_guard_300_349"
    / "development_winner.pt"
)
ROWS_DIR = BASE / "artifacts" / "low_guard_v2_joint_validation"
OUTPUT = (
    TEST_DIR
    / "analysis"
    / "artifacts"
    / "persistent_joint350_medium_diagnosis.json"
)
K = 8

for item in (str(ROOT), str(TEST_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

from frameworks.hattrick_system import Hattrick


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pte_info(cache):
    pte = cache.dataset.pte.coalesce()
    indices = pte.indices()
    return pte, indices[0], indices[1], pte.values()


def simulate(runtime, model, props, cache, policies, demands=None, batch_size=10):
    demands = cache.tms if demands is None else demands
    admitted_chunks = [[], [], []]
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            stop = min(start + batch_size, len(cache))
            current_policies = [value[start:stop] for value in policies]
            current_demands = [value[start:stop] for value in demands]
            capacities = cache.capacities[start:stop]
            ratios = model.simulate(
                current_policies,
                current_demands,
                capacities,
                pte_info(cache),
                stop - start,
                props,
                rate_cap=props.rate_cap,
            )[:3]
            for class_index, (ratio, demand) in enumerate(zip(ratios, current_demands)):
                admitted_chunks[class_index].append(
                    ratio.reshape(stop - start, -1) * demand.squeeze(-1)
                )
    return tuple(torch.cat(chunks, dim=0) for chunks in admitted_chunks)


def edge_load(cache, flow):
    return torch.sparse.mm(
        cache.dataset.pte.to(dtype=torch.float32).t(), flow.to(dtype=torch.float32).t()
    ).t()


def normalized(cache, admitted, class_index):
    return admitted[class_index].sum(dim=1) / cache.oracle_flows[class_index].clamp_min(1e-9)


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p1": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def policy_snapshot_metrics(policy, demand):
    grouped = policy.squeeze(-1).reshape(len(policy), -1, K).clamp_min(0.0)
    mass = grouped.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probability = grouped / mass
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    effective = entropy.exp()
    main_share = probability.max(dim=-1).values
    hhi = probability.square().sum(dim=-1)
    od_demand = demand.squeeze(-1).reshape(len(policy), -1, K)[:, :, 0]
    weights = od_demand / od_demand.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return {
        "effective_paths": (effective * weights).sum(dim=1),
        "main_share": (main_share * weights).sum(dim=1),
        "hhi": (hhi * weights).sum(dim=1),
    }


def policy_difference(left, right, demand):
    left = left.squeeze(-1).reshape(len(left), -1, K).clamp_min(0.0)
    right = right.squeeze(-1).reshape(len(right), -1, K).clamp_min(0.0)
    left = left / left.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    right = right / right.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = demand.squeeze(-1).reshape(len(left), -1, K)[:, :, 0]
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    tv = 0.5 * (left - right).abs().sum(dim=-1)
    changed = (left.argmax(dim=-1) != right.argmax(dim=-1)).to(weights.dtype)
    return {
        "tv": (tv * weights).sum(dim=1),
        "argmax_changed": (changed * weights).sum(dim=1),
    }


def scipy_pte(cache):
    pte = cache.dataset.pte.coalesce().cpu()
    indices = pte.indices().numpy()
    values = pte.values().numpy().astype(np.float64, copy=False)
    return coo_matrix(
        (values, (indices[0], indices[1])), shape=tuple(pte.shape)
    ).tocsr()


def solve_class(pte, demand_per_path, enabled, residual, return_policy=False):
    demand_per_path = np.asarray(demand_per_path, dtype=np.float64).reshape(-1)
    enabled = np.asarray(enabled, dtype=bool).reshape(-1)
    residual = np.maximum(np.asarray(residual, dtype=np.float64).reshape(-1), 0.0)
    active = np.flatnonzero(enabled)
    pair_count = demand_per_path.size // K
    active_demand = demand_per_path[active]
    edge_constraints = pte[active, :].T.multiply(active_demand)
    pair_rows = active // K
    pair_constraints = coo_matrix(
        (
            np.ones(active.size, dtype=np.float64),
            (pair_rows, np.arange(active.size)),
        ),
        shape=(pair_count, active.size),
    ).tocsr()
    result = linprog(
        -active_demand,
        A_ub=vstack((edge_constraints, pair_constraints), format="csr"),
        b_ub=np.concatenate((residual, np.ones(pair_count))),
        bounds=(0.0, 1.0),
        method="highs",
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"LP failure {result.status}: {result.message}")
    if not return_policy:
        return float(-result.fun)
    policy = np.zeros(demand_per_path.size, dtype=np.float32)
    policy[active] = np.clip(result.x, 0.0, 1.0).astype(np.float32)
    return float(-result.fun), policy


def load_rows(path):
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values_by_snapshot(rows, class_name):
    return np.asarray(
        [
            float(row["norm_fulfill"])
            for row in rows
            if row["class"] == class_name
        ],
        dtype=np.float64,
    )


def main():
    torch.manual_seed(20260825)
    np.random.seed(20260825)
    model_module = load("joint350_medium_model", CORE_RUNNER)
    edge = load("joint350_medium_edge", EDGE_RUNNER)
    strict = load("joint350_medium_strict", STRICT_RUNNER)
    guard_module = load("joint350_medium_guard", GUARD_RUNNER)
    runtime = edge.runtime
    runtime.set_seed(20260825)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)

    core_payload = torch.load(CORE_CHECKPOINT, map_location=device, weights_only=False)
    native_payload = torch.load(NATIVE_CHECKPOINT, map_location=device, weights_only=False)
    core = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    core.load_state_dict(core_payload["model_state_dict"], strict=True)
    core.eval()
    native = Hattrick(props).to(device=device, dtype=props.dtype)
    native.load_state_dict(native_payload["model_state_dict"], strict=True)
    native.eval()
    core_cache, core_audit = strict.build_strict_policy_cache(
        runtime, core, props, 350, 400, 25, actual_input_mode="zero"
    )
    native_cache, native_audit = strict.build_strict_policy_cache(
        runtime, native, props, 350, 400, 25, actual_input_mode="zero"
    )

    zeros = torch.zeros_like(core_cache.policies[2])
    combinations = {
        "nativeH_nativeM": (
            native_cache.policies[0], native_cache.policies[1], zeros
        ),
        "coreH_nativeM": (
            core_cache.policies[0], native_cache.policies[1], zeros
        ),
        "nativeH_coreM": (
            native_cache.policies[0], core_cache.policies[1], zeros
        ),
        "coreH_coreM": (core_cache.policies[0], core_cache.policies[1], zeros),
    }
    admitted = {
        name: simulate(runtime, core, props, core_cache, policies)
        for name, policies in combinations.items()
    }
    normalized_medium = {
        name: normalized(core_cache, value, 1).detach().cpu().numpy()
        for name, value in admitted.items()
    }
    nn = normalized_medium["nativeH_nativeM"]
    cn = normalized_medium["coreH_nativeM"]
    nc = normalized_medium["nativeH_coreM"]
    cc = normalized_medium["coreH_coreM"]
    factorial = {
        "four_cells": {name: describe(value) for name, value in normalized_medium.items()},
        "high_effect_given_native_medium": describe(cn - nn),
        "medium_effect_given_native_high": describe(nc - nn),
        "high_effect_given_core_medium": describe(cc - nc),
        "medium_effect_given_core_high": describe(cc - cn),
        "interaction": describe(cc - cn - nc + nn),
    }

    policy_metrics = {}
    path_series = {}
    for class_index, class_name in enumerate(("High", "Medium")):
        class_metrics = {}
        for name, cache in (("native", native_cache), ("core", core_cache)):
            current = policy_snapshot_metrics(
                cache.policies[class_index], cache.predicted_tms[class_index]
            )
            class_metrics[name] = {
                key: describe(value.detach().cpu().numpy())
                for key, value in current.items()
            }
            for key, value in current.items():
                path_series[f"{class_name}_{name}_{key}"] = value.detach().cpu().numpy()
        difference = policy_difference(
            core_cache.policies[class_index],
            native_cache.policies[class_index],
            core_cache.predicted_tms[class_index],
        )
        class_metrics["core_vs_native"] = {
            key: describe(value.detach().cpu().numpy())
            for key, value in difference.items()
        }
        for key, value in difference.items():
            path_series[f"{class_name}_difference_{key}"] = value.detach().cpu().numpy()
        policy_metrics[class_name] = class_metrics

    pte = scipy_pte(core_cache)
    # GEANT's static reader returns ``None`` when no explicit per-class mask is
    # needed.  Invalid padded paths are nevertheless exact zeros after the
    # model's masked softmax, so the union of positive policy support recovers
    # the admissible variables without opening a disabled route.
    if core_cache.path_masks is None:
        enabled_medium = (
            core_cache.policies[1].abs().sum(dim=0).squeeze(-1) > 0
        ).detach().cpu().numpy()
        enabled_low = (
            core_cache.policies[2].abs().sum(dim=0).squeeze(-1) > 0
        ).detach().cpu().numpy()
    else:
        enabled_medium = (
            core_cache.path_masks[1].reshape(-1).detach().cpu().numpy().astype(bool)
        )
        enabled_low = (
            core_cache.path_masks[2].reshape(-1).detach().cpu().numpy().astype(bool)
        )
    capacities = core_cache.capacities.detach().cpu().numpy()

    core_high_load = edge_load(core_cache, admitted["coreH_coreM"][0]).detach().cpu().numpy()
    native_high_load = edge_load(core_cache, admitted["nativeH_nativeM"][0]).detach().cpu().numpy()
    core_hm_load = (
        edge_load(core_cache, admitted["coreH_coreM"][0])
        + edge_load(core_cache, admitted["coreH_coreM"][1])
    ).detach().cpu().numpy()
    native_hm_load = (
        edge_load(core_cache, admitted["nativeH_nativeM"][0])
        + edge_load(core_cache, admitted["nativeH_nativeM"][1])
    ).detach().cpu().numpy()

    actual_medium = core_cache.tms[1].squeeze(-1).detach().cpu().numpy()
    actual_low = core_cache.tms[2].squeeze(-1).detach().cpu().numpy()
    oracle_medium = core_cache.oracle_flows[1].detach().cpu().numpy()
    oracle_low = core_cache.oracle_flows[2].detach().cpu().numpy()
    actual_fixed_high = {"core": [], "native": []}
    actual_fixed_upstream_low = {"core": [], "native": []}
    for index in range(len(core_cache)):
        for name, high_load in (("core", core_high_load), ("native", native_high_load)):
            optimum = solve_class(
                pte,
                actual_medium[index],
                enabled_medium,
                capacities[index] - high_load[index],
            )
            actual_fixed_high[name].append(optimum / max(oracle_medium[index], 1e-9))
        for name, hm_load in (("core", core_hm_load), ("native", native_hm_load)):
            optimum = solve_class(
                pte,
                actual_low[index],
                enabled_low,
                capacities[index] - hm_load[index],
            )
            actual_fixed_upstream_low[name].append(optimum / max(oracle_low[index], 1e-9))

    # Strict-prediction residual LP: preserve the core High policy, replace only
    # Medium with the ESM-demand optimum, and score it later on actual traffic.
    predicted_high_only = simulate(
        runtime,
        core,
        props,
        core_cache,
        (core_cache.policies[0], zeros, zeros),
        demands=(
            core_cache.predicted_tms[0],
            torch.zeros_like(core_cache.predicted_tms[1]),
            torch.zeros_like(core_cache.predicted_tms[2]),
        ),
    )[0]
    predicted_high_load = edge_load(core_cache, predicted_high_only).detach().cpu().numpy()
    predicted_medium = core_cache.predicted_tms[1].squeeze(-1).detach().cpu().numpy()
    lp_medium = []
    lp_predicted_objective = []
    for index in range(len(core_cache)):
        objective, policy = solve_class(
            pte,
            predicted_medium[index],
            enabled_medium,
            capacities[index] - predicted_high_load[index],
            return_policy=True,
        )
        lp_medium.append(policy)
        lp_predicted_objective.append(objective)
    lp_medium = torch.from_numpy(np.stack(lp_medium)).to(
        device=device, dtype=props.dtype
    ).unsqueeze(-1)
    strict_lp_admitted = simulate(
        runtime,
        core,
        props,
        core_cache,
        (core_cache.policies[0], lp_medium, zeros),
    )
    strict_lp_norm = normalized(core_cache, strict_lp_admitted, 1).detach().cpu().numpy()

    # Diagnose the selected Low head, especially its validation tail.
    guard_payload = torch.load(GUARD_CHECKPOINT, map_location=device, weights_only=False)
    guard = guard_module.LowGuard(
        int(guard_payload["edge_count"]),
        int(guard_payload["feature_count"]),
        float(guard_payload["max_toll"]),
    ).to(device)
    guard.load_state_dict(guard_payload["state_dict"], strict=True)
    guard.eval()
    guard_features = edge.edge_features(core_cache)
    guarded_cache, guard_tolls = guard_module.build_candidate_cache(
        guard, core_cache, guard_features, batch_size=25
    )

    core_rows = load_rows(ROWS_DIR / "core_rows.csv")
    native_rows = load_rows(ROWS_DIR / "native_rows.csv")
    guarded_rows = load_rows(ROWS_DIR / "guarded_rows.csv")
    medium_gain = values_by_snapshot(core_rows, "Medium") - values_by_snapshot(
        native_rows, "Medium"
    )
    high_gain = values_by_snapshot(core_rows, "High") - values_by_snapshot(
        native_rows, "High"
    )
    low_core_gain = values_by_snapshot(core_rows, "Low") - values_by_snapshot(
        native_rows, "Low"
    )
    low_guard_gain = values_by_snapshot(guarded_rows, "Low") - values_by_snapshot(
        native_rows, "Low"
    )
    guard_difference = policy_difference(
        guarded_cache.path_features.unsqueeze(-1),
        core_cache.policies[2],
        core_cache.predicted_tms[2],
    )

    prediction_error = {}
    error_series = {}
    for class_index, class_name in enumerate(("High", "Medium", "Low")):
        actual = (
            core_cache.tms[class_index]
            .squeeze(-1)
            .reshape(len(core_cache), -1, K)[:, :, 0]
        )
        predicted = (
            core_cache.predicted_tms[class_index]
            .squeeze(-1)
            .reshape(len(core_cache), -1, K)[:, :, 0]
        )
        wmape = (predicted - actual).abs().sum(dim=1) / actual.sum(dim=1).clamp_min(1e-9)
        bias = (predicted - actual).sum(dim=1) / actual.sum(dim=1).clamp_min(1e-9)
        prediction_error[class_name] = {
            "wmape": describe(wmape.detach().cpu().numpy()),
            "bias": describe(bias.detach().cpu().numpy()),
        }
        error_series[f"{class_name}_wmape"] = wmape.detach().cpu().numpy()
        error_series[f"{class_name}_bias"] = bias.detach().cpu().numpy()

    core_medium_norm = values_by_snapshot(core_rows, "Medium")
    native_medium_norm = values_by_snapshot(native_rows, "Medium")
    core_low_norm = values_by_snapshot(core_rows, "Low")
    native_low_norm = values_by_snapshot(native_rows, "Low")
    guarded_low_norm = values_by_snapshot(guarded_rows, "Low")
    fixed_high_core = np.asarray(actual_fixed_high["core"])
    fixed_high_native = np.asarray(actual_fixed_high["native"])
    fixed_low_core = np.asarray(actual_fixed_upstream_low["core"])
    fixed_low_native = np.asarray(actual_fixed_upstream_low["native"])

    residual_diagnosis = {
        "actual_medium_upper_bound_with_fixed_high": {
            "core_high": describe(fixed_high_core),
            "native_high": describe(fixed_high_native),
            "core_minus_native": describe(fixed_high_core - fixed_high_native),
            "core_stage2_execution_gap": describe(fixed_high_core - core_medium_norm),
            "native_stage2_execution_gap": describe(fixed_high_native - native_medium_norm),
        },
        "strict_prediction_medium_lp_with_core_high": {
            "actual_norm_fulfill": describe(strict_lp_norm),
            "gain_over_core_stage2": describe(strict_lp_norm - core_medium_norm),
            "gain_over_native": describe(strict_lp_norm - native_medium_norm),
            "predicted_objective": describe(lp_predicted_objective),
        },
        "actual_low_upper_bound_with_fixed_high_medium": {
            "core_upstream": describe(fixed_low_core),
            "native_upstream": describe(fixed_low_native),
            "core_minus_native": describe(fixed_low_core - fixed_low_native),
            "guard_execution_gap": describe(fixed_low_core - guarded_low_norm),
            "core_execution_gap": describe(fixed_low_core - core_low_norm),
        },
    }

    correlations = {
        "medium_gain_vs_high_gain": correlation(medium_gain, high_gain),
        "medium_gain_vs_high_policy_tv": correlation(
            medium_gain, path_series["High_difference_tv"]
        ),
        "medium_gain_vs_high_effective_path_change": correlation(
            medium_gain,
            path_series["High_core_effective_paths"]
            - path_series["High_native_effective_paths"],
        ),
        "medium_gain_vs_fixed_high_upper_bound_gain": correlation(
            medium_gain, fixed_high_core - fixed_high_native
        ),
        "medium_gain_vs_core_stage2_gap": correlation(
            medium_gain, fixed_high_core - core_medium_norm
        ),
        "medium_gain_vs_medium_esm_wmape": correlation(
            medium_gain, error_series["Medium_wmape"]
        ),
        "guarded_low_gain_vs_core_upstream_upper_bound_gain": correlation(
            low_guard_gain, fixed_low_core - fixed_low_native
        ),
        "guarded_low_gain_vs_low_esm_wmape": correlation(
            low_guard_gain, error_series["Low_wmape"]
        ),
        "guarded_low_gain_vs_medium_gain": correlation(low_guard_gain, medium_gain),
        "core_low_gain_vs_high_gain": correlation(low_core_gain, high_gain),
    }

    snapshots = []
    for index in range(len(core_cache)):
        snapshots.append(
            {
                "snapshot": 350 + index,
                "medium_gain": float(medium_gain[index]),
                "high_gain": float(high_gain[index]),
                "low_core_gain": float(low_core_gain[index]),
                "low_guard_gain": float(low_guard_gain[index]),
                "high_policy_tv": float(path_series["High_difference_tv"][index]),
                "high_effective_paths_core": float(
                    path_series["High_core_effective_paths"][index]
                ),
                "high_effective_paths_native": float(
                    path_series["High_native_effective_paths"][index]
                ),
                "medium_fixed_high_upper_core": float(fixed_high_core[index]),
                "medium_fixed_high_upper_native": float(fixed_high_native[index]),
                "medium_stage2_gap_core": float(
                    fixed_high_core[index] - core_medium_norm[index]
                ),
                "strict_lp_medium_norm": float(strict_lp_norm[index]),
                "low_fixed_upstream_upper_core": float(fixed_low_core[index]),
                "low_fixed_upstream_upper_native": float(fixed_low_native[index]),
                "low_guard_gap_to_upper": float(
                    fixed_low_core[index] - guarded_low_norm[index]
                ),
                "medium_esm_wmape": float(error_series["Medium_wmape"][index]),
                "low_esm_wmape": float(error_series["Low_wmape"][index]),
            }
        )

    report = {
        "scope": {
            "snapshots": [350, 400],
            "forbidden_snapshots_constructed": False,
            "policy_inputs": "strict literal-zero actual TM; ESM predictions only",
            "actual_tm_role": "retrospective admission and LP upper-bound diagnosis only",
            "device": str(device),
            "core_input_audit": core_audit,
            "native_input_audit": native_audit,
        },
        "factorial_high_vs_medium_policy": factorial,
        "policy_concentration": policy_metrics,
        "prediction_error": prediction_error,
        "residual_capacity_diagnosis": residual_diagnosis,
        "low_guard": {
            "policy_tv_vs_core": describe(
                guard_difference["tv"].detach().cpu().numpy()
            ),
            "argmax_change_vs_core": describe(
                guard_difference["argmax_changed"].detach().cpu().numpy()
            ),
            "toll_abs": describe(guard_tolls.abs().detach().cpu().numpy().reshape(-1)),
        },
        "correlations": correlations,
        "all_snapshot_diagnostics": snapshots,
        "worst_medium_snapshots": sorted(
            snapshots, key=lambda row: row["medium_gain"]
        )[:10],
        "best_medium_snapshots": sorted(
            snapshots, key=lambda row: row["medium_gain"], reverse=True
        )[:10],
        "worst_guarded_low_snapshots": sorted(
            snapshots, key=lambda row: row["low_guard_gain"]
        )[:12],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
