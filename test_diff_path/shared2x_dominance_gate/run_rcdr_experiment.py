from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR / "probe_esm_dominance_gate.py"
spec = importlib.util.spec_from_file_location("rcdr_runtime", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)
runtime = gate.runtime


CALIBRATION = (0, 318)
SMALL = (350, 358)
VALIDATION = (358, 400)
FINAL = (400, 500)
QUANTILES = (None, 0.80, 0.90, 0.95)
ALPHA_MULTIPLIERS = (1.0, 0.9, 0.75)
MIN_APPLIED_ALPHAS = (0.0, 0.005, 0.01, 0.02, 0.05)


def pte_info(cache):
    pte = cache.dataset.pte.coalesce()
    indices = pte.indices()
    return pte, indices[0], indices[1], pte.values()


def simulate(model, props, cache, policies, tms):
    return model.simulate(
        policies,
        tms,
        cache.capacities,
        pte_info(cache),
        len(cache),
        props,
        rate_cap=props.rate_cap,
    )[:3]


def edge_loads(cache, ratios, tms):
    pte_t = cache.dataset.pte.coalesce().transpose(0, 1).coalesce()
    loads = []
    for ratio, tm in zip(ratios, tms):
        flow = ratio.reshape(len(cache), -1) * tm.squeeze(-1)
        loads.append(
            torch.sparse.mm(pte_t, flow.transpose(0, 1)).transpose(0, 1)
        )
    return loads


def fit_residual_quantiles(cache):
    result = {}
    for class_index, (actual, predicted) in enumerate(
        zip(cache.tms, cache.predicted_tms)
    ):
        residual = actual - predicted
        result[class_index] = {
            q: (
                torch.quantile(residual, 1.0 - q, dim=0),
                torch.quantile(residual, q, dim=0),
            )
            for q in QUANTILES
            if q is not None
        }
    return result


def scenario_tms(one, residual_quantiles, quantile):
    predicted = tuple(value.clone() for value in one.predicted_tms)
    if quantile is None:
        return [predicted]
    lower = tuple(
        torch.clamp(
            one.predicted_tms[index] + residual_quantiles[index][quantile][0],
            min=0.0,
        )
        for index in range(3)
    )
    upper = tuple(
        torch.clamp(
            one.predicted_tms[index] + residual_quantiles[index][quantile][1],
            min=0.0,
        )
        for index in range(3)
    )
    return [predicted, lower, upper]


def slice_cache(cache, index):
    return gate.slice_cache(cache, index, index + 1)


def certified_alpha_one(
    model,
    props,
    one,
    proposed_medium,
    residual_quantiles,
    quantile,
):
    base_policies = list(one.policies)
    full_policies = [
        one.policies[0],
        proposed_medium.unsqueeze(0).unsqueeze(-1),
        one.policies[2],
    ]
    alpha = 1.0
    scenario_diagnostics = []
    for tms in scenario_tms(one, residual_quantiles, quantile):
        with torch.no_grad():
            base_ratios = simulate(model, props, one, base_policies, tms)
            full_ratios = simulate(model, props, one, full_policies, tms)
            base_load = edge_loads(one, base_ratios, tms)
            full_load = edge_loads(one, full_ratios, tms)
        base_hm = base_load[0][0] + base_load[1][0]
        full_hm = full_load[0][0] + full_load[1][0]
        low_reserve = base_load[2][0]
        capacity = one.capacities[0]
        headroom = torch.clamp(capacity - base_hm - low_reserve, min=0.0)
        increase = full_hm - base_hm
        positive = increase > 1e-8
        if torch.any(positive):
            edge_alpha = torch.clamp(headroom[positive] / increase[positive], 0.0, 1.0)
            scenario_alpha = float(torch.min(edge_alpha).item())
        else:
            scenario_alpha = 1.0
        alpha = min(alpha, scenario_alpha)
        scenario_diagnostics.append(
            {
                "scenario_alpha": scenario_alpha,
                "minimum_headroom": float(headroom.min().item()),
                "positive_edge_count": int(positive.sum().item()),
            }
        )
    return float(np.clip(alpha, 0.0, 1.0)), scenario_diagnostics


def build_projected_cache(
    model,
    props,
    cache,
    proposed,
    residual_quantiles,
    quantile,
    alpha_multiplier,
    min_applied_alpha,
):
    path_count = int(cache.policies[0].shape[1])
    base_medium = cache.policies[1].squeeze(-1)
    base_low = cache.policies[2].squeeze(-1)
    proposed_medium = proposed.path_features[:, :path_count]
    mixed_medium = []
    diagnostics = []
    for index in range(len(cache)):
        one = slice_cache(cache, index)
        certified, scenarios = certified_alpha_one(
            model,
            props,
            one,
            proposed_medium[index],
            residual_quantiles,
            quantile,
        )
        alpha = float(certified * alpha_multiplier)
        if alpha < float(min_applied_alpha):
            alpha = 0.0
        mixed_medium.append(
            (1.0 - alpha) * base_medium[index] + alpha * proposed_medium[index]
        )
        diagnostics.append(
            {
                "snapshot": int(cache.source_start + index),
                "certified_alpha": certified,
                "applied_alpha": alpha,
                "scenarios": scenarios,
            }
        )
    features = torch.cat(
        [torch.stack(mixed_medium, dim=0), base_low], dim=1
    )
    return replace(cache, path_features=features), diagnostics


def canonicalize(baseline_rows, candidate_rows, diagnostics):
    baseline = {
        (int(row["snapshot"]), row["class"]): row for row in baseline_rows
    }
    alpha = {int(row["snapshot"]): row["applied_alpha"] for row in diagnostics}
    for row in candidate_rows:
        snapshot = int(row["snapshot"])
        if row["class"] == "High" or alpha[snapshot] <= 1e-12:
            row.update(baseline[(snapshot, row["class"])])


def evaluate_config(
    model,
    props,
    cache,
    baseline_rows,
    proposed,
    residual_quantiles,
    quantile,
    alpha_multiplier,
    min_applied_alpha,
):
    projected, projection_rows = build_projected_cache(
        model,
        props,
        cache,
        proposed,
        residual_quantiles,
        quantile,
        alpha_multiplier,
        min_applied_alpha,
    )
    adapter = gate.sar.CorrectedPolicyAdapter(int(cache.policies[0].shape[1]))
    candidate_rows, candidate_summary = runtime.evaluate_cache(
        model, props, projected, adapter, batch_size=32
    )
    canonicalize(baseline_rows, candidate_rows, projection_rows)
    accepted = [row["applied_alpha"] > 1e-12 for row in projection_rows]
    return {
        "residual_quantile": quantile,
        "alpha_multiplier": float(alpha_multiplier),
        "min_applied_alpha": float(min_applied_alpha),
        "summary": gate.sar.compact(candidate_summary),
        "diagnostics": gate.diagnostics(baseline_rows, candidate_rows, accepted),
        "alpha_mean": float(np.mean([row["applied_alpha"] for row in projection_rows])),
        "alpha_positive_count": int(sum(accepted)),
        "projection_rows": projection_rows,
        "candidate_rows": candidate_rows,
    }


def public(row):
    return {
        key: value
        for key, value in row.items()
        if key not in ("candidate_rows", "projection_rows")
    }


def rank(row):
    diag = row["diagnostics"]
    return (
        int(diag["all_three_empirical_cdfs_noninferior"]),
        diag["classes"]["Medium"]["mean_gain"],
        diag["classes"]["Low"]["mean_gain"],
    )


def prepare(model, props, bounds):
    cache = runtime.build_policy_cache(model, props, *bounds, batch_size=32)
    baseline_rows, baseline_summary = runtime.evaluate_cache(
        model, props, cache, None, batch_size=32
    )
    proposed, _ = gate.build_proposals_and_proxy(model, props, cache)
    return cache, baseline_rows, baseline_summary, proposed


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    calibration = runtime.build_policy_cache(
        model, props, *CALIBRATION, batch_size=64
    )
    residual_quantiles = fit_residual_quantiles(calibration)

    small = prepare(model, props, SMALL)
    small_results = [
        evaluate_config(
            model,
            props,
            small[0],
            small[1],
            small[3],
            residual_quantiles,
            quantile,
            multiplier,
            min_alpha,
        )
        for quantile in (0.80,)
        for multiplier in ALPHA_MULTIPLIERS
        for min_alpha in MIN_APPLIED_ALPHAS
    ]
    small_feasible = [
        row
        for row in small_results
        if row["diagnostics"]["all_three_empirical_cdfs_noninferior"]
    ]
    validation = prepare(model, props, VALIDATION)
    validation_results = [
        evaluate_config(
            model,
            props,
            validation[0],
            validation[1],
            validation[3],
            residual_quantiles,
            row["residual_quantile"],
            row["alpha_multiplier"],
            row["min_applied_alpha"],
        )
        for row in small_feasible
    ]
    validation_feasible = [
        row
        for row in validation_results
        if row["diagnostics"]["all_three_empirical_cdfs_noninferior"]
    ]
    winner = max(validation_feasible, key=rank) if validation_feasible else None

    final = None
    final_result = None
    if winner is not None:
        final = prepare(model, props, FINAL)
        final_result = evaluate_config(
            model,
            props,
            final[0],
            final[1],
            final[3],
            residual_quantiles,
            winner["residual_quantile"],
            winner["alpha_multiplier"],
            winner["min_applied_alpha"],
        )
    payload = {
        "method": "RCDR residual-capacity dominance routing",
        "protocol": {
            "residual_calibration": list(CALIBRATION),
            "small_selection": list(SMALL),
            "independent_validation": list(VALIDATION),
            "final_confirmation": list(FINAL),
            "actual_tm_online": False,
        },
        "checkpoint": str(checkpoint),
        "small_baseline": gate.sar.compact(small[2]),
        "small_results": [public(row) for row in small_results],
        "small_feasible_count": len(small_feasible),
        "validation_baseline": gate.sar.compact(validation[2]),
        "validation_results": [public(row) for row in validation_results],
        "validation_feasible_count": len(validation_feasible),
        "winner": public(winner) if winner else None,
        "final_baseline": gate.sar.compact(final[2]) if final else None,
        "final": public(final_result) if final_result else None,
    }
    runtime.write_json(THIS_DIR / "rcdr_small_validation.json", payload)
    runtime.write_csv(
        THIS_DIR / "rcdr_validation_baseline_rows.csv", validation[1]
    )
    runtime.write_csv(
        THIS_DIR / "rcdr_validation_candidate_rows.csv",
        winner["candidate_rows"] if winner else validation[1],
    )
    runtime.write_json(
        THIS_DIR / "rcdr_frozen_config.json",
        {
            "residual_quantile": winner["residual_quantile"] if winner else None,
            "alpha_multiplier": winner["alpha_multiplier"] if winner else None,
            "min_applied_alpha": winner["min_applied_alpha"] if winner else None,
            "validation_passed": winner is not None,
        },
    )
    if final is not None and final_result is not None:
        runtime.write_csv(THIS_DIR / "rcdr_final_baseline_rows.csv", final[1])
        runtime.write_csv(
            THIS_DIR / "rcdr_final_candidate_rows.csv",
            final_result["candidate_rows"],
        )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
