from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR / "probe_esm_self_correction.py"
spec = importlib.util.spec_from_file_location("final_esm_self_correction", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load ESM self-correction implementation")
method = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = method
spec.loader.exec_module(method)
runtime = method.runtime

FINAL_CONFIG = (24, 0.06, 0.25, 0.01)


def slice_cache(cache, start: int, stop: int):
    def sliced(values):
        return tuple(value[start:stop].clone() for value in values)

    return replace(
        cache,
        source_start=cache.source_start + start,
        policies=sliced(cache.policies),
        tms=sliced(cache.tms),
        predicted_tms=sliced(cache.predicted_tms),
        capacities=cache.capacities[start:stop].clone(),
        oracle_flows=sliced(cache.oracle_flows),
        oracle_mlus=sliced(cache.oracle_mlus),
        path_features=None,
    )


def correct_strictly_per_snapshot(model, props, cache):
    corrected = []
    for index in range(len(cache)):
        one = method.correct_cache(
            model, props, slice_cache(cache, index, index + 1), *FINAL_CONFIG
        )
        corrected.append(one.path_features)
    return replace(cache, path_features=torch.cat(corrected, dim=0))


def main() -> None:
    torch.manual_seed(20260820)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache = runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
    baseline_rows, baseline_summary = runtime.evaluate_cache(
        model, props, cache, None, batch_size=32
    )

    started = time.perf_counter()
    corrected = correct_strictly_per_snapshot(model, props, cache)
    correction_seconds = time.perf_counter() - started
    adapter = method.CorrectedPolicyAdapter(int(cache.policies[0].shape[1]))
    final_rows, final_summary = runtime.evaluate_cache(
        model, props, corrected, adapter, batch_size=32
    )

    # Quantify the numerical difference caused by vectorizing the nonsmooth
    # bottleneck simulator. The reported candidate always uses singleton mode.
    audit_full = method.correct_cache(model, props, slice_cache(cache, 0, 4), *FINAL_CONFIG)
    singleton_policies = []
    for index in range(4):
        one = method.correct_cache(
            model, props, slice_cache(cache, index, index + 1), *FINAL_CONFIG
        )
        singleton_policies.append(one.path_features)
    audit_single = torch.cat(singleton_policies, dim=0)
    batch_independence_max_abs_delta = float(
        (audit_full.path_features - audit_single).abs().max().item()
    )

    payload = {
        "method": "per-snapshot ESM sequential-admission self-correction",
        "final_config": {
            "steps": FINAL_CONFIG[0],
            "learning_rate": FINAL_CONFIG[1],
            "low_weight": FINAL_CONFIG[2],
            "anchor_weight": FINAL_CONFIG[3],
        },
        "selection_protocol": {
            "exploration": [350, 358],
            "validation": [358, 400],
            "final_untouched_test": [400, 500],
        },
        "input_contract": {
            "policy_correction": "ESM-predicted High/Medium/Low TMs only",
            "actual_TM": "sequential-admission evaluation only",
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.sha256(checkpoint),
        "correction_seconds_for_100_snapshots": correction_seconds,
        "deployment_mode": "strict one-snapshot-at-a-time correction",
        "batch_independence_max_abs_delta": batch_independence_max_abs_delta,
        "baseline": method.compact(baseline_summary),
        "candidate": method.compact(final_summary),
        "delta": method.gaps(final_summary, baseline_summary),
        "constraints_passed": method.feasible(
            final_summary, method.gaps(final_summary, baseline_summary)
        ),
    }
    output = THIS_DIR / "final_esm_self_correction_400_500.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime.write_csv(THIS_DIR / "final_baseline_rows_400_500.csv", baseline_rows)
    runtime.write_csv(THIS_DIR / "final_candidate_rows_400_500.csv", final_rows)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
