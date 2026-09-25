from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TRAIN_PATH = THIS_DIR / "train_safety_memory_gate.py"
spec = importlib.util.spec_from_file_location("direct_cdf_gate", TRAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {TRAIN_PATH}")
memory = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = memory
spec.loader.exec_module(memory)


MEDIUM_THRESHOLDS = (-1.0, 0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03)
LOW_THRESHOLDS = (-1.0, -0.01, -0.005, 0.0, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select(rows, bounds, medium_threshold, low_threshold):
    return {
        int(row["snapshot"]): (
            float(row["predicted_medium_gain"]) >= medium_threshold
            and float(row["predicted_low_gain"]) >= low_threshold
        )
        for row in rows
        if bounds[0] <= int(row["snapshot"]) < bounds[1]
    }


def main() -> None:
    features = read_rows(THIS_DIR / "memory_features_level4_0_400.csv")
    baseline = read_rows(THIS_DIR / "memory_baseline_rows_level4_0_400.csv")
    candidate = read_rows(THIS_DIR / "memory_candidate_rows_level4_0_400.csv")
    configs = []
    for medium in MEDIUM_THRESHOLDS:
        for low in LOW_THRESHOLDS:
            safety_decisions = select(features, memory.SAFETY_RANGE, medium, low)
            safety = memory.hybrid_diagnostics(baseline, candidate, safety_decisions)
            if not safety["all_three_empirical_cdfs_noninferior"] or safety["accepted"] == 0:
                continue
            validation_decisions = select(features, memory.VALIDATION_RANGE, medium, low)
            validation = memory.hybrid_diagnostics(baseline, candidate, validation_decisions)
            configs.append({
                "medium_threshold": medium,
                "low_threshold": low,
                "safety": safety,
                "validation": validation,
            })
    feasible = [
        row for row in configs
        if row["validation"]["all_three_empirical_cdfs_noninferior"]
        and row["validation"]["accepted"] > 0
    ]
    feasible.sort(
        key=lambda row: (
            row["validation"]["classes"]["Medium"]["mean_gain"],
            row["validation"]["classes"]["Low"]["mean_gain"],
        ),
        reverse=True,
    )
    payload = {
        "method": "direct CDF-calibrated ESM gain gate",
        "safety_feasible_count": len(configs),
        "validation_feasible_count": len(feasible),
        "winner": feasible[0] if feasible else None,
        "top10": feasible[:10],
    }
    memory.runtime.write_json(THIS_DIR / "direct_cdf_gate_selection_level4.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
