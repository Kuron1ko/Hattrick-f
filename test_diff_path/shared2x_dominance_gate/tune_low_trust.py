from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR / "probe_esm_dominance_gate.py"
spec = importlib.util.spec_from_file_location("tune_low_trust", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)
runtime = gate.runtime


RANGE = (318, 400)
SAFETY = (318, 350)
VALIDATION = (350, 400)
MEDIUM_THRESHOLD = 0.008
LOW_THRESHOLD = 0.005
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)


def subset(rows, bounds):
    return [row for row in rows if bounds[0] <= int(row["snapshot"]) < bounds[1]]


def canonicalize(baseline_rows, candidate_rows, decisions):
    baseline = {
        (int(row["snapshot"]), row["class"]): row for row in baseline_rows
    }
    accepted = {RANGE[0] + index: value for index, value in enumerate(decisions)}
    for row in candidate_rows:
        snapshot = int(row["snapshot"])
        if row["class"] == "High" or not accepted[snapshot]:
            row.update(baseline[(snapshot, row["class"])])


def evaluate(model, props, cache, baseline_rows, proposed, proxy_rows, alpha):
    decisions = [
        row["predicted_medium_gain"] >= MEDIUM_THRESHOLD
        and row["predicted_low_gain"] >= LOW_THRESHOLD
        for row in proxy_rows
    ]
    features = proposed.path_features.clone()
    path_count = int(cache.policies[0].shape[1])
    base_medium = cache.policies[1].squeeze(-1)
    base_low = cache.policies[2].squeeze(-1)
    proposed_low = proposed.path_features[:, path_count:]
    features[:, path_count:] = (
        (1.0 - float(alpha)) * base_low + float(alpha) * proposed_low
    )
    for index, accepted in enumerate(decisions):
        if not accepted:
            features[index, :path_count] = base_medium[index]
            features[index, path_count:] = base_low[index]
    candidate_cache = replace(cache, path_features=features)
    adapter = gate.sar.CorrectedPolicyAdapter(path_count)
    candidate_rows, _ = runtime.evaluate_cache(
        model, props, candidate_cache, adapter, batch_size=32
    )
    canonicalize(baseline_rows, candidate_rows, decisions)
    split_results = {}
    for name, bounds in (("safety", SAFETY), ("validation", VALIDATION)):
        split_baseline = subset(baseline_rows, bounds)
        split_candidate = subset(candidate_rows, bounds)
        split_decisions = decisions[bounds[0] - RANGE[0] : bounds[1] - RANGE[0]]
        split_results[name] = gate.diagnostics(
            split_baseline, split_candidate, split_decisions
        )
    return {"low_trust_alpha": float(alpha), **split_results}


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache = runtime.build_policy_cache(model, props, *RANGE, batch_size=32)
    baseline_rows, _ = runtime.evaluate_cache(model, props, cache, None, batch_size=32)
    proposed, proxy_rows = gate.build_proposals_and_proxy(model, props, cache)
    results = [
        evaluate(model, props, cache, baseline_rows, proposed, proxy_rows, alpha)
        for alpha in ALPHAS
    ]
    feasible = [
        row for row in results
        if row["safety"]["all_three_empirical_cdfs_noninferior"]
        and row["validation"]["all_three_empirical_cdfs_noninferior"]
    ]
    feasible.sort(
        key=lambda row: (
            row["validation"]["classes"]["Low"]["mean_gain"],
            row["safety"]["classes"]["Low"]["mean_gain"],
            row["low_trust_alpha"],
        ),
        reverse=True,
    )
    payload = {
        "method": "Low-only trust region after direct CDF gate",
        "range": list(RANGE),
        "medium_threshold": MEDIUM_THRESHOLD,
        "low_threshold": LOW_THRESHOLD,
        "checkpoint": str(checkpoint),
        "results": results,
        "winner": feasible[0] if feasible else None,
    }
    runtime.write_json(THIS_DIR / "low_trust_selection_level4.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
