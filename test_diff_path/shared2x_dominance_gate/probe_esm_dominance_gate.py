from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
ACTIVE_DIR = THIS_DIR.parent / "shared2x_active_set_router"
MODULE_PATH = ACTIVE_DIR / "probe_esm_self_correction.py"
spec = importlib.util.spec_from_file_location("dominance_gate_esm_sar", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
sar = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sar
spec.loader.exec_module(sar)
runtime = sar.runtime


FINAL_CONFIG = (24, 0.06, 0.25, 0.01)
EXPLORATION = (350, 358)
VALIDATION = (358, 400)
MEDIUM_MARGINS = (0.0, 0.002, 0.005, 0.01)
LOW_MARGINS = (-0.005, 0.0, 0.002, 0.005, 0.01, 0.02)


def slice_cache(cache, start: int, stop: int):
    def sliced(values):
        return tuple(value[start:stop].clone() for value in values)

    features = None
    if cache.path_features is not None:
        features = cache.path_features[start:stop].clone()
    return replace(
        cache,
        source_start=cache.source_start + start,
        policies=sliced(cache.policies),
        tms=sliced(cache.tms),
        predicted_tms=sliced(cache.predicted_tms),
        capacities=cache.capacities[start:stop].clone(),
        oracle_flows=sliced(cache.oracle_flows),
        oracle_mlus=sliced(cache.oracle_mlus),
        path_features=features,
    )


def build_proposals_and_proxy(model, props, cache):
    proposals = []
    proxy_rows = []
    path_count = int(cache.policies[0].shape[1])
    for index in range(len(cache)):
        one = slice_cache(cache, index, index + 1)
        corrected = sar.correct_cache(model, props, one, *FINAL_CONFIG)
        proposals.append(corrected.path_features)
        with torch.no_grad():
            base_values = sar.predicted_fulfillment(
                model, props, one, list(one.policies)
            )[0]
            candidate_policies = [
                one.policies[0],
                corrected.path_features[:, :path_count].unsqueeze(-1),
                corrected.path_features[:, path_count:].unsqueeze(-1),
            ]
            candidate_values = sar.predicted_fulfillment(
                model, props, one, candidate_policies
            )[0]
            pte_t = one.dataset.pte.coalesce().transpose(0, 1).coalesce()
            base_edge_loads = []
            candidate_edge_loads = []
            for class_index in range(3):
                predicted_tm = one.predicted_tms[class_index].squeeze(-1)
                base_flow = one.policies[class_index].squeeze(-1) * predicted_tm
                candidate_policy = candidate_policies[class_index].squeeze(-1)
                candidate_flow = candidate_policy * predicted_tm
                base_edge_loads.append(
                    torch.sparse.mm(pte_t, base_flow.transpose(0, 1))
                    .transpose(0, 1)[0]
                )
                candidate_edge_loads.append(
                    torch.sparse.mm(pte_t, candidate_flow.transpose(0, 1))
                    .transpose(0, 1)[0]
                )
            capacity = one.capacities[0].clamp_min(1e-9)
            base_h = base_edge_loads[0] / capacity
            base_hm = (base_edge_loads[0] + base_edge_loads[1]) / capacity
            base_all = sum(base_edge_loads) / capacity
            candidate_hm = (
                candidate_edge_loads[0] + candidate_edge_loads[1]
            ) / capacity
            candidate_all = sum(candidate_edge_loads) / capacity
            policy_features = {}
            for class_name, base_policy, candidate_policy in (
                ("medium", one.policies[1].squeeze(-1), candidate_policies[1].squeeze(-1)),
                ("low", one.policies[2].squeeze(-1), candidate_policies[2].squeeze(-1)),
            ):
                absolute = torch.abs(candidate_policy - base_policy)
                grouped_base = base_policy.reshape(1, -1, 8)
                grouped_candidate = candidate_policy.reshape(1, -1, 8)
                kl = (
                    grouped_candidate
                    * torch.log(
                        (grouped_candidate + 1e-12) / (grouped_base + 1e-12)
                    )
                ).sum(dim=-1).mean()
                policy_features[f"policy_{class_name}_l1"] = float(absolute.mean().item())
                policy_features[f"policy_{class_name}_max"] = float(absolute.max().item())
                policy_features[f"policy_{class_name}_kl"] = float(kl.item())
            traffic_features = {}
            for class_name, predicted_tm in zip(
                ("high", "medium", "low"), one.predicted_tms
            ):
                grouped = predicted_tm[:, ::8, 0]
                mean = grouped.mean().clamp_min(1e-9)
                traffic_features[f"demand_{class_name}_total"] = float(grouped.sum().item())
                traffic_features[f"demand_{class_name}_cv"] = float(
                    (grouped.std(unbiased=False) / mean).item()
                )
        proxy_rows.append(
            {
                "snapshot": int(one.source_start),
                "baseline_predicted_high": float(base_values[0].item()),
                "baseline_predicted_medium": float(base_values[1].item()),
                "baseline_predicted_low": float(base_values[2].item()),
                "candidate_predicted_medium": float(candidate_values[1].item()),
                "candidate_predicted_low": float(candidate_values[2].item()),
                "predicted_medium_gain": float(
                    candidate_values[1].item() - base_values[1].item()
                ),
                "predicted_low_gain": float(
                    candidate_values[2].item() - base_values[2].item()
                ),
                "base_high_util_max": float(base_h.max().item()),
                "base_hm_util_max": float(base_hm.max().item()),
                "base_all_util_max": float(base_all.max().item()),
                "base_hm_util_p90": float(torch.quantile(base_hm, 0.9).item()),
                "base_all_util_p90": float(torch.quantile(base_all, 0.9).item()),
                "candidate_hm_util_max": float(candidate_hm.max().item()),
                "candidate_all_util_max": float(candidate_all.max().item()),
                "candidate_hm_util_p90": float(torch.quantile(candidate_hm, 0.9).item()),
                "candidate_all_util_p90": float(torch.quantile(candidate_all, 0.9).item()),
                **traffic_features,
                **policy_features,
            }
        )
    return replace(cache, path_features=torch.cat(proposals, dim=0)), proxy_rows


def gated_cache(cache, proposed, proxy_rows, medium_margin: float, low_margin: float):
    features = proposed.path_features.clone()
    path_count = int(cache.policies[0].shape[1])
    accepted = []
    for index, proxy in enumerate(proxy_rows):
        take = (
            proxy["predicted_medium_gain"] >= float(medium_margin)
            and proxy["predicted_low_gain"] >= float(low_margin)
        )
        accepted.append(bool(take))
        if not take:
            features[index, :path_count] = cache.policies[1][index, :, 0]
            features[index, path_count:] = cache.policies[2][index, :, 0]
    return replace(cache, path_features=features), accepted


def row_map(rows):
    return {
        (int(row["snapshot"]), str(row["class"])): float(row["norm_fulfill"])
        for row in rows
    }


def ecdf_max_violation(baseline_values, candidate_values):
    baseline = np.sort(np.asarray(baseline_values, dtype=np.float64))
    candidate = np.sort(np.asarray(candidate_values, dtype=np.float64))
    grid = np.unique(np.concatenate([baseline, candidate]))
    baseline_cdf = np.searchsorted(baseline, grid, side="right") / len(baseline)
    candidate_cdf = np.searchsorted(candidate, grid, side="right") / len(candidate)
    return float(np.max(candidate_cdf - baseline_cdf))


def diagnostics(baseline_rows, candidate_rows, accepted):
    baseline = row_map(baseline_rows)
    candidate = row_map(candidate_rows)
    snapshots = sorted({key[0] for key in baseline})
    result = {
        "accepted": int(sum(accepted)),
        "total": int(len(accepted)),
        "classes": {},
    }
    for class_name in ("High", "Medium", "Low"):
        base_values = [baseline[(snapshot, class_name)] for snapshot in snapshots]
        cand_values = [candidate[(snapshot, class_name)] for snapshot in snapshots]
        deltas = np.asarray(cand_values) - np.asarray(base_values)
        result["classes"][class_name] = {
            "mean_gain": float(np.mean(deltas)),
            "paired_negative_count": int(np.sum(deltas < -1e-7)),
            "paired_min_gain": float(np.min(deltas)),
            "ecdf_max_upward_violation": ecdf_max_violation(base_values, cand_values),
        }
    result["all_three_empirical_cdfs_noninferior"] = all(
        row["ecdf_max_upward_violation"] <= 1e-12
        for row in result["classes"].values()
    )
    result["all_three_paired_noninferior"] = all(
        row["paired_negative_count"] == 0 for row in result["classes"].values()
    )
    return result


def evaluate(model, props, cache, baseline_rows, proposed, proxy_rows, margins):
    candidate_cache, accepted = gated_cache(cache, proposed, proxy_rows, *margins)
    adapter = sar.CorrectedPolicyAdapter(int(cache.policies[0].shape[1]))
    candidate_rows, candidate_summary = runtime.evaluate_cache(
        model, props, candidate_cache, adapter, batch_size=32
    )
    return {
        "medium_margin": float(margins[0]),
        "low_margin": float(margins[1]),
        "summary": sar.compact(candidate_summary),
        "diagnostics": diagnostics(baseline_rows, candidate_rows, accepted),
        "rows": candidate_rows,
    }


def public(result):
    return {key: value for key, value in result.items() if key != "rows"}


def rank(result):
    diag = result["diagnostics"]
    medium = diag["classes"]["Medium"]
    low = diag["classes"]["Low"]
    return (
        int(diag["all_three_paired_noninferior"]),
        int(diag["all_three_empirical_cdfs_noninferior"]),
        -medium["paired_negative_count"] - low["paired_negative_count"],
        medium["mean_gain"],
        low["mean_gain"],
    )


def prepare_split(model, props, bounds):
    cache = runtime.build_policy_cache(model, props, *bounds, batch_size=32)
    baseline_rows, baseline_summary = runtime.evaluate_cache(
        model, props, cache, None, batch_size=32
    )
    proposed, proxy_rows = build_proposals_and_proxy(model, props, cache)
    return cache, baseline_rows, baseline_summary, proposed, proxy_rows


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(3, device)
    model, checkpoint = runtime.load_backbone(3, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    exp = prepare_split(model, props, EXPLORATION)
    exploration_results = [
        evaluate(model, props, exp[0], exp[1], exp[3], exp[4], (medium, low))
        for medium in MEDIUM_MARGINS
        for low in LOW_MARGINS
    ]
    winner = max(exploration_results, key=rank)

    val = prepare_split(model, props, VALIDATION)
    validation = evaluate(
        model,
        props,
        val[0],
        val[1],
        val[3],
        val[4],
        (winner["medium_margin"], winner["low_margin"]),
    )
    payload = {
        "method": "ESM proposal with per-snapshot dominance abstention gate",
        "idea": (
            "Use ESM-SAR as a high-gain proposal, but fall back to Hattrick unless "
            "the ESM simulator predicts nonnegative Medium and Low gains with margins."
        ),
        "input_contract": {
            "online": "ESM-predicted TMs, Hattrick policies, capacities",
            "actual_tm": "evaluation only",
            "high_policy": "identical to Hattrick by construction",
        },
        "checkpoint": str(checkpoint),
        "exploration_range": list(EXPLORATION),
        "validation_range": list(VALIDATION),
        "proposal_config": {
            "steps": FINAL_CONFIG[0],
            "learning_rate": FINAL_CONFIG[1],
            "low_weight": FINAL_CONFIG[2],
            "anchor_weight": FINAL_CONFIG[3],
        },
        "exploration_baseline": sar.compact(exp[2]),
        "exploration_candidates": [public(row) for row in exploration_results],
        "winner": public(winner),
        "validation_baseline": sar.compact(val[2]),
        "validation": public(validation),
        "next_step": (
            "Replace the two fixed margins by a tiny conformal lower-bound critic "
            "trained on 0-349, then keep the same abstention rule."
        ),
    }
    runtime.write_json(THIS_DIR / "probe_result.json", payload)
    runtime.write_csv(THIS_DIR / "validation_baseline_rows.csv", val[1])
    runtime.write_csv(THIS_DIR / "validation_candidate_rows.csv", validation["rows"])
    runtime.write_csv(THIS_DIR / "validation_proxy_rows.csv", val[4])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
