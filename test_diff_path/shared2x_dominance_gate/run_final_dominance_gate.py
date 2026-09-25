from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR / "probe_esm_dominance_gate.py"
spec = importlib.util.spec_from_file_location("final_dominance_gate", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)
runtime = gate.runtime


FINAL_RANGE = (400, 500)
FROZEN_MARGINS = (0.002, 0.02)


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
    candidate_cache, accepted = gate.gated_cache(
        cache, proposed, proxy_rows, *FROZEN_MARGINS
    )
    adapter = gate.sar.CorrectedPolicyAdapter(int(cache.policies[0].shape[1]))
    candidate_rows, candidate_summary = runtime.evaluate_cache(
        model, props, candidate_cache, adapter, batch_size=32
    )
    elapsed = time.perf_counter() - started
    diag = gate.diagnostics(baseline_rows, candidate_rows, accepted)
    payload = {
        "method": "ESM proposal with per-snapshot dominance abstention gate",
        "traffic_scale": "2x",
        "selection_protocol": {
            "margin_search": list(gate.EXPLORATION),
            "independent_validation": list(gate.VALIDATION),
            "level4_confirmation": list(FINAL_RANGE),
            "frozen_medium_margin": FROZEN_MARGINS[0],
            "frozen_low_margin": FROZEN_MARGINS[1],
        },
        "input_contract": {
            "online": "ESM-predicted TMs, Hattrick policies, capacities",
            "actual_tm": "sequential-admission evaluation only",
            "high_policy": "identical to Hattrick by construction",
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.sha256(checkpoint),
        "seconds_100_snapshots": elapsed,
        "baseline": gate.sar.compact(baseline_summary),
        "candidate": gate.sar.compact(candidate_summary),
        "delta": gate.sar.gaps(candidate_summary, baseline_summary),
        "dominance_diagnostics": diag,
    }
    runtime.write_json(THIS_DIR / "final_result_400_500.json", payload)
    runtime.write_csv(THIS_DIR / "final_baseline_rows_400_500.csv", baseline_rows)
    runtime.write_csv(THIS_DIR / "final_candidate_rows_400_500.csv", candidate_rows)
    runtime.write_csv(THIS_DIR / "final_proxy_rows_400_500.csv", proxy_rows)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
