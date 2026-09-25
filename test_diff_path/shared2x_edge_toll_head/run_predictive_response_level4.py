from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
METHOD_PATH = THIS_DIR / "run_causal_esm_low.py"
for item in (str(THIS_DIR), str(THIS_DIR.parent), str(THIS_DIR.parent.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)
spec = importlib.util.spec_from_file_location("predictive_response_runtime", METHOD_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {METHOD_PATH}")
method = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = method
spec.loader.exec_module(method)
edge = method.edge
risk = method.risk
runtime = method.runtime


def admitted_totals(model, props, cache, policies, tms, capacities):
    ratios = model.simulate(
        policies,
        list(tms),
        capacities,
        edge.pte_info(cache),
        int(capacities.shape[0]),
        props,
        rate_cap=props.rate_cap,
    )[:3]
    return torch.stack(
        [
            (ratio * demand.squeeze(-1)).sum(dim=1)
            for ratio, demand in zip(ratios, tms)
        ],
        dim=1,
    )


def expand_sample(values, sample, count):
    return values[sample : sample + 1].expand(count, *values.shape[1:])


def plan_cache(
    model,
    props,
    head,
    cache,
    ordinary_features,
    corrected_features,
    corrected_tms,
    medium_scales,
    low_scales,
    response_penalty,
):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    medium_outputs = []
    low_outputs = []
    selected_medium = []
    selected_low = []
    with torch.no_grad():
        for sample in range(len(cache)):
            base = [cache.policies[c][sample : sample + 1].squeeze(-1) for c in (1, 2)]
            _, ordinary_tolls, _ = head(
                ordinary_features[sample : sample + 1], base, pte
            )
            _, corrected_tolls, _ = head(
                corrected_features[sample : sample + 1], base, pte
            )

            m_count = len(medium_scales)
            m_base = base[0].expand(m_count, -1)
            m_toll = ordinary_tolls[:, :, 0].expand(m_count, -1) * medium_scales.reshape(-1, 1)
            m_policies = risk.route_from_tolls(m_base, m_toll, pte)
            predicted_tms = tuple(
                expand_sample(value, sample, m_count) for value in cache.predicted_tms
            )
            policies = [
                expand_sample(cache.policies[0], sample, m_count),
                m_policies.unsqueeze(-1),
                expand_sample(cache.policies[2], sample, m_count),
            ]
            totals = admitted_totals(
                model,
                props,
                cache,
                policies,
                predicted_tms,
                expand_sample(cache.capacities, sample, m_count),
            )
            medium_demand = predicted_tms[1].squeeze(-1).sum(dim=1).clamp_min(1e-9)
            medium_score = totals[:, 1] / medium_demand - float(response_penalty) * (
                medium_scales - 1.0
            ).square()
            medium_index = int(torch.argmax(medium_score).item())
            medium_policy = m_policies[medium_index : medium_index + 1]

            l_count = len(low_scales)
            l_base = base[1].expand(l_count, -1)
            l_toll = corrected_tolls[:, :, 1].expand(l_count, -1) * low_scales.reshape(-1, 1)
            l_policies = risk.route_from_tolls(l_base, l_toll, pte)
            causal_tms = tuple(
                expand_sample(value, sample, l_count) for value in corrected_tms
            )
            policies = [
                expand_sample(cache.policies[0], sample, l_count),
                medium_policy.unsqueeze(-1).expand(l_count, -1, -1),
                l_policies.unsqueeze(-1),
            ]
            totals = admitted_totals(
                model,
                props,
                cache,
                policies,
                causal_tms,
                expand_sample(cache.capacities, sample, l_count),
            )
            low_demand = causal_tms[2].squeeze(-1).sum(dim=1).clamp_min(1e-9)
            low_score = totals[:, 2] / low_demand - float(response_penalty) * (
                low_scales - 1.0
            ).square()
            low_index = int(torch.argmax(low_score).item())
            medium_outputs.append(medium_policy.squeeze(0))
            low_outputs.append(l_policies[low_index])
            selected_medium.append(float(medium_scales[medium_index].item()))
            selected_low.append(float(low_scales[low_index].item()))
    return (
        replace(
            cache,
            path_features=torch.cat(
                [torch.stack(medium_outputs), torch.stack(low_outputs)], dim=1
            ),
        ),
        selected_medium,
        selected_low,
    )


def evaluate(model, props, cache, planned, baseline_rows):
    adapter = edge.TwoPolicyAdapter(int(cache.policies[0].shape[1]))
    rows, summary = runtime.evaluate_cache(model, props, planned, adapter, batch_size=32)
    baseline_by_key = {(int(r["snapshot"]), r["class"]): r for r in baseline_rows}
    for row in rows:
        if row["class"] == "High":
            row.update(baseline_by_key[(int(row["snapshot"]), "High")])
    return {
        "summary": runtime.summary_index(summary),
        "diagnostics": edge.diagnostics(baseline_rows, rows),
        "rows": rows,
    }


def run_window(model, props, head, history, cache, penalty):
    baseline_rows, baseline_summary = runtime.evaluate_cache(
        model, props, cache, None, batch_size=32
    )
    corrected_tms = method.causal_predictions(history, cache, 1.0, 0.95, 8)
    corrected_cache = replace(cache, predicted_tms=corrected_tms)
    medium_scales = torch.linspace(0.25, 1.75, 13, device=props.device)
    low_scales = torch.linspace(0.25, 1.75, 13, device=props.device)
    planned, selected_medium, selected_low = plan_cache(
        model,
        props,
        head,
        cache,
        edge.edge_features(cache),
        edge.edge_features(corrected_cache),
        corrected_tms,
        medium_scales,
        low_scales,
        penalty,
    )
    result = evaluate(model, props, cache, planned, baseline_rows)
    result["selected_medium_scale_mean"] = float(np.mean(selected_medium))
    result["selected_low_scale_mean"] = float(np.mean(selected_low))
    return result, baseline_rows, baseline_summary


def main():
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    saved = torch.load(THIS_DIR / "best_edge_toll_head.pt", map_location=device)
    head = edge.EdgeTollHead(saved["edge_count"], saved["feature_count"]).to(device)
    head.load_state_dict(saved["state_dict"])
    head.eval()

    dev_history = runtime.build_policy_cache(model, props, 0, edge.SAFETY[0], batch_size=64)
    development = runtime.build_policy_cache(
        model, props, edge.SAFETY[0], edge.VALIDATION[1], batch_size=64
    )
    dev_candidates = []
    for penalty in (0.0, 0.001, 0.003, 0.01):
        result, _, _ = run_window(
            model, props, head, dev_history, development, penalty
        )
        result["response_penalty"] = penalty
        dev_candidates.append(result)
    winner = max(
        dev_candidates,
        key=lambda x: (
            -max(
                x["diagnostics"]["Medium"]["ecdf_max_upward_violation"],
                x["diagnostics"]["Low"]["ecdf_max_upward_violation"],
            ),
            x["diagnostics"]["Medium"]["mean_gain"],
            x["diagnostics"]["Low"]["mean_gain"],
        ),
    )

    final_history = runtime.build_policy_cache(model, props, 0, edge.FINAL[0], batch_size=64)
    final = runtime.build_policy_cache(model, props, *edge.FINAL, batch_size=64)
    final_result, final_rows, final_summary = run_window(
        model,
        props,
        head,
        final_history,
        final,
        winner["response_penalty"],
    )
    payload = {
        "method": "ESM self-consistent per-snapshot response selection",
        "current_actual_tm_used": False,
        "development_range": [edge.SAFETY[0], edge.VALIDATION[1]],
        "level4_range": list(edge.FINAL),
        "checkpoint": str(checkpoint),
        "development_candidates": [
            {k: v for k, v in x.items() if k != "rows"} for x in dev_candidates
        ],
        "development_winner": {
            k: v for k, v in winner.items() if k != "rows"
        },
        "level4_baseline": runtime.summary_index(final_summary),
        "level4_candidate": {
            k: v for k, v in final_result.items() if k != "rows"
        },
    }
    runtime.write_json(THIS_DIR / "predictive_response_level4.json", payload)
    runtime.write_csv(THIS_DIR / "predictive_level4_baseline_rows.csv", final_rows)
    runtime.write_csv(THIS_DIR / "predictive_level4_candidate_rows.csv", final_result["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
