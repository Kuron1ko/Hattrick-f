from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TRAIN_PATH = THIS_DIR / "train_safety_memory_gate.py"
spec = importlib.util.spec_from_file_location("final_safety_memory_gate", TRAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {TRAIN_PATH}")
memory = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = memory
spec.loader.exec_module(memory)
gate = memory.gate
runtime = gate.runtime


FINAL_RANGE = (400, 500)
FEATURE_KEYS = (
    "baseline_predicted_high",
    "baseline_predicted_medium",
    "baseline_predicted_low",
    "predicted_medium_gain",
    "predicted_low_gain",
)
K = 8
RADIUS = 0.825049872229665
LOWER_QUANTILE = 0.1
MARGIN = 0.0


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    memory_features = read_rows(THIS_DIR / "memory_features_0_400.csv")
    memory_baseline = read_rows(THIS_DIR / "memory_baseline_rows_0_400.csv")
    memory_candidate = read_rows(THIS_DIR / "memory_candidate_rows_0_400.csv")
    labels = memory.gain_labels(memory_baseline, memory_candidate)
    training_rows = [
        row for row in memory_features if int(row["snapshot"]) < memory.TRAIN_RANGE[1]
    ]
    train_x_raw = np.asarray(
        [[float(row[key]) for key in FEATURE_KEYS] for row in training_rows],
        dtype=np.float64,
    )
    train_y = np.asarray(
        [
            [
                labels[int(row["snapshot"])]["Medium"],
                labels[int(row["snapshot"])]["Low"],
            ]
            for row in training_rows
        ],
        dtype=np.float64,
    )
    mean = train_x_raw.mean(axis=0)
    std = train_x_raw.std(axis=0)
    std[std < 1e-8] = 1.0
    train_x = (train_x_raw - mean) / std

    cache = runtime.build_policy_cache(model, props, *FINAL_RANGE, batch_size=32)
    baseline_rows, baseline_summary = runtime.evaluate_cache(
        model, props, cache, None, batch_size=32
    )
    started = time.perf_counter()
    proposed, proxy_rows = gate.build_proposals_and_proxy(model, props, cache)
    query_x_raw = np.asarray(
        [[float(row[key]) for key in FEATURE_KEYS] for row in proxy_rows],
        dtype=np.float64,
    )
    query_x = (query_x_raw - mean) / std
    decisions, bounds = memory.predict_acceptance(
        train_x,
        train_y,
        query_x,
        K,
        LOWER_QUANTILE,
        RADIUS,
        MARGIN,
    )
    features = proposed.path_features.clone()
    path_count = int(cache.policies[0].shape[1])
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

    # High is structurally identical to Hattrick. Canonicalize sub-ULP replay
    # differences so the empirical CDF audit reflects the actual policy identity.
    baseline_by_key = gate.row_map(baseline_rows)
    for row in candidate_rows:
        if row["class"] == "High":
            row["norm_fulfill"] = baseline_by_key[(int(row["snapshot"]), "High")]
    diag = gate.diagnostics(baseline_rows, candidate_rows, decisions)
    decision_rows = [
        {
            "snapshot": int(row["snapshot"]),
            "accept": int(accepted),
            "medium_neighbor_lower": float(bound[0]),
            "low_neighbor_lower": float(bound[1]),
        }
        for row, accepted, bound in zip(proxy_rows, decisions, bounds)
    ]
    payload = {
        "method": "counterfactual safety-memory gate over ESM-SAR proposals",
        "traffic_scale": "2x",
        "protocol": {
            "memory_train": list(memory.TRAIN_RANGE),
            "safety_selection": list(memory.SAFETY_RANGE),
            "independent_validation": list(memory.VALIDATION_RANGE),
            "level4_confirmation": list(FINAL_RANGE),
            "actual_tm_online": False,
        },
        "memory_gate": {
            "feature_keys": list(FEATURE_KEYS),
            "k": K,
            "radius": RADIUS,
            "lower_quantile": LOWER_QUANTILE,
            "margin": MARGIN,
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.sha256(checkpoint),
        "seconds_100_snapshots": elapsed,
        "baseline": gate.sar.compact(baseline_summary),
        "candidate_raw_summary": gate.sar.compact(candidate_summary),
        "delta_raw_summary": gate.sar.gaps(candidate_summary, baseline_summary),
        "dominance_diagnostics": diag,
    }
    runtime.write_json(THIS_DIR / "final_safety_memory_result_400_500.json", payload)
    runtime.write_csv(THIS_DIR / "final_safety_memory_baseline_rows.csv", baseline_rows)
    runtime.write_csv(THIS_DIR / "final_safety_memory_candidate_rows.csv", candidate_rows)
    runtime.write_csv(THIS_DIR / "final_safety_memory_decisions.csv", decision_rows)
    runtime.write_csv(THIS_DIR / "final_safety_memory_proxy_rows.csv", proxy_rows)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
