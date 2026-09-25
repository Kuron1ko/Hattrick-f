from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR / "probe_esm_dominance_gate.py"
spec = importlib.util.spec_from_file_location("final_direct_cdf_gate", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)
runtime = gate.runtime


FINAL_RANGE = (400, 500)
MEDIUM_THRESHOLD = 0.008
LOW_THRESHOLD = 0.005
LOW_TRUST_ALPHA = 0.9


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cache = runtime.build_policy_cache(model, props, *FINAL_RANGE, batch_size=32)
    baseline_rows, baseline_summary = runtime.evaluate_cache(
        model, props, cache, None, batch_size=32
    )
    started = time.perf_counter()
    proposed, proxy_rows = gate.build_proposals_and_proxy(model, props, cache)
    decisions = [
        row["predicted_medium_gain"] >= MEDIUM_THRESHOLD
        and row["predicted_low_gain"] >= LOW_THRESHOLD
        for row in proxy_rows
    ]
    features = proposed.path_features.clone()
    path_count = int(cache.policies[0].shape[1])
    base_low = cache.policies[2].squeeze(-1)
    features[:, path_count:] = (
        (1.0 - LOW_TRUST_ALPHA) * base_low
        + LOW_TRUST_ALPHA * proposed.path_features[:, path_count:]
    )
    for index, accepted in enumerate(decisions):
        if not accepted:
            features[index, :path_count] = cache.policies[1][index, :, 0]
            features[index, path_count:] = cache.policies[2][index, :, 0]
    candidate_cache = replace(cache, path_features=features)
    adapter = gate.sar.CorrectedPolicyAdapter(path_count)
    candidate_rows, candidate_summary = runtime.evaluate_cache(
        model, props, candidate_cache, adapter, batch_size=32
    )
    elapsed = time.perf_counter() - started

    # Rejected snapshots use byte-identical Hattrick policies; copy their replay
    # metrics, and always copy High because its policy is structurally bypassed.
    baseline_by_key = {
        (int(row["snapshot"]), row["class"]): row for row in baseline_rows
    }
    decision_by_snapshot = {
        int(row["snapshot"]): accepted for row, accepted in zip(proxy_rows, decisions)
    }
    for row in candidate_rows:
        snapshot = int(row["snapshot"])
        if row["class"] == "High" or not decision_by_snapshot[snapshot]:
            baseline_row = baseline_by_key[(snapshot, row["class"])]
            row.update(baseline_row)
    diag = gate.diagnostics(baseline_rows, candidate_rows, decisions)
    payload = {
        "method": "direct CDF-calibrated ESM gain gate over ESM-SAR proposals",
        "traffic_scale": "2x",
        "protocol": {
            "offline_memory": [0, 318],
            "safety_selection": [318, 350],
            "independent_validation": [350, 400],
            "level4_confirmation": list(FINAL_RANGE),
            "actual_tm_online": False,
        },
        "frozen_gate": {
            "medium_predicted_gain_threshold": MEDIUM_THRESHOLD,
            "low_predicted_gain_threshold": LOW_THRESHOLD,
            "low_trust_alpha": LOW_TRUST_ALPHA,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.sha256(checkpoint),
        "seconds_100_snapshots": elapsed,
        "baseline": gate.sar.compact(baseline_summary),
        "candidate_raw_summary": gate.sar.compact(candidate_summary),
        "dominance_diagnostics": diag,
    }
    runtime.write_json(THIS_DIR / "final_direct_cdf_gate_400_500.json", payload)
    runtime.write_csv(THIS_DIR / "final_direct_cdf_baseline_rows.csv", baseline_rows)
    runtime.write_csv(THIS_DIR / "final_direct_cdf_candidate_rows.csv", candidate_rows)
    runtime.write_csv(THIS_DIR / "final_direct_cdf_proxy_rows.csv", proxy_rows)
    runtime.write_csv(
        THIS_DIR / "final_direct_cdf_decisions.csv",
        [
            {"snapshot": row["snapshot"], "accept": int(accepted)}
            for row, accepted in zip(proxy_rows, decisions)
        ],
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
