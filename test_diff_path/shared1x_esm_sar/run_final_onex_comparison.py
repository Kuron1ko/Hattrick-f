from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TUNER_PATH = THIS_DIR / "tune_onex_congestion_gate.py"
spec = importlib.util.spec_from_file_location("shared1x_final_runtime", TUNER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load 1x ESM-SAR tuner")
tuner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tuner
spec.loader.exec_module(tuner)
probe = tuner.probe
method = tuner.method
runtime = tuner.runtime


FROZEN_THRESHOLD = 0.875
TEST_RANGE = (400, 500)


def read_values(rows: list[dict], class_name: str) -> np.ndarray:
    return np.asarray(
        [float(row["norm_fulfill"]) for row in rows if row["class"] == class_name],
        dtype=np.float64,
    )


def paired_bootstrap(base: np.ndarray, candidate: np.ndarray, seed: int) -> dict:
    difference = candidate - base
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(20000, len(difference)))
    samples = difference[indices].mean(axis=1)
    return {
        "mean_delta": float(difference.mean()),
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "improved_fraction": float(np.mean(difference > 1e-7)),
        "degraded_fraction": float(np.mean(difference < -1e-7)),
    }


def main() -> None:
    torch.manual_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props = probe.load_onex(device)
    cache = runtime.build_policy_cache(model, props, *TEST_RANGE, batch_size=32)
    baseline_rows, baseline = runtime.evaluate_cache(model, props, cache, None, batch_size=32)

    started = time.perf_counter()
    fully_corrected = probe.correct_strict(model, props, cache)
    predicted = tuner.predicted_medium_fulfillment(model, props, cache)
    gated, active = tuner.gated_cache(
        cache, fully_corrected, predicted, FROZEN_THRESHOLD
    )
    correction_seconds = time.perf_counter() - started
    adapter = method.CorrectedPolicyAdapter(int(cache.policies[0].shape[1]))
    candidate_rows, candidate = runtime.evaluate_cache(
        model, props, gated, adapter, batch_size=32
    )
    delta = method.gaps(candidate, baseline)
    statistics = {
        class_name: paired_bootstrap(
            read_values(baseline_rows, class_name),
            read_values(candidate_rows, class_name),
            20260821 + index,
        )
        for index, class_name in enumerate(("High", "Medium", "Low"))
    }
    payload = {
        "method": "1x congestion-gated ESM-SAR",
        "interpretation": "frozen Hattrick plus a validation-selected inference correction; Hattrick weights are unchanged",
        "topology": probe.TOPOLOGY,
        "source_model": str(probe.MODEL_PATH),
        "source_model_sha256": runtime.sha256(probe.MODEL_PATH),
        "selection_protocol": {
            "hattrick_train": [0, 350],
            "gate_validation": [350, 400],
            "untouched_test": list(TEST_RANGE),
        },
        "frozen_config": {
            "steps": probe.TRANSFER_CONFIG[0],
            "learning_rate": probe.TRANSFER_CONFIG[1],
            "low_weight": probe.TRANSFER_CONFIG[2],
            "anchor_weight": probe.TRANSFER_CONFIG[3],
            "predicted_medium_trigger_threshold": FROZEN_THRESHOLD,
        },
        "input_contract": "current ESM predictions generate policy; current actual TMs are evaluation only",
        "test_trigger": {
            "active_count": int(active.sum().item()),
            "active_fraction": float(active.float().mean().item()),
            "predicted_medium_min": float(predicted.min().item()),
            "predicted_medium_mean": float(predicted.mean().item()),
            "predicted_medium_max": float(predicted.max().item()),
        },
        "correction_seconds_100_snapshots": correction_seconds,
        "baseline_original_hattrick": method.compact(baseline),
        "candidate_onex_esm_sar": method.compact(candidate),
        "delta": delta,
        "paired_bootstrap_20000": statistics,
    }
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    (THIS_DIR / "final_onex_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    runtime.write_csv(THIS_DIR / "final_onex_hattrick_rows.csv", baseline_rows)
    runtime.write_csv(THIS_DIR / "final_onex_esm_sar_rows.csv", candidate_rows)
    (THIS_DIR / "onex_esm_sar_model.json").write_text(
        json.dumps(
            {
                "source_model": str(probe.MODEL_PATH),
                "source_model_sha256": runtime.sha256(probe.MODEL_PATH),
                "topology": probe.TOPOLOGY,
                "config": payload["frozen_config"],
                "input_contract": payload["input_contract"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
