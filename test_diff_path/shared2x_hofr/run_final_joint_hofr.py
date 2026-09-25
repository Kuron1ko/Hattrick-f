from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR / "probe_joint_optimal_face.py"
spec = importlib.util.spec_from_file_location("shared2x_hofr_final", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load joint HOFR implementation")
joint = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = joint
spec.loader.exec_module(joint)
runtime = joint.runtime
method = joint.method


FINAL_CONFIG = (1.0, 0.75, 0.0)
FINAL_RANGE = (400, 500)


def class_values(rows: list[dict], class_name: str) -> np.ndarray:
    ordered = sorted(
        (
            (int(row["snapshot"]), float(row["norm_fulfill"]))
            for row in rows
            if row["class"] == class_name
        ),
        key=lambda item: item[0],
    )
    return np.asarray([value for _snapshot, value in ordered], dtype=np.float64)


def bootstrap(base: np.ndarray, candidate: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(base), size=(20000, len(base)))
    base_samples = base[indices]
    candidate_samples = candidate[indices]
    samples = {
        "mean": candidate_samples.mean(axis=1) - base_samples.mean(axis=1),
        "p1": np.quantile(candidate_samples, 0.01, axis=1)
        - np.quantile(base_samples, 0.01, axis=1),
        "p10": np.quantile(candidate_samples, 0.10, axis=1)
        - np.quantile(base_samples, 0.10, axis=1),
    }
    points = {
        "mean": float(candidate.mean() - base.mean()),
        "p1": float(np.quantile(candidate, 0.01) - np.quantile(base, 0.01)),
        "p10": float(np.quantile(candidate, 0.10) - np.quantile(base, 0.10)),
    }
    return {
        metric: {
            "delta": points[metric],
            "ci95": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
        for metric, values in samples.items()
    }


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    calibration_cache = joint.core.build_esm_only_cache(
        model, props, *joint.CALIBRATION, batch_size=64
    )
    calibrators = joint.fit_affine_calibrator(calibration_cache)
    final_cache = joint.core.build_esm_only_cache(
        model, props, *FINAL_RANGE, batch_size=32
    )
    baseline_rows, baseline = runtime.evaluate_cache(
        model, props, final_cache, None, batch_size=32
    )
    started = time.perf_counter()
    candidate = joint.evaluate_config(
        model, props, final_cache, calibrators, baseline, FINAL_CONFIG
    )
    elapsed = time.perf_counter() - started
    statistics = {
        class_name: bootstrap(
            class_values(baseline_rows, class_name),
            class_values(candidate["rows"], class_name),
            20260821 + class_index,
        )
        for class_index, class_name in enumerate(("High", "Medium", "Low"))
    }
    compact_candidate = joint.public(candidate)
    payload = {
        "method": "joint HOFR with train-only affine ESM calibration",
        "scope": (
            "frozen Hattrick supplies the learned path prior; exact lexicographic LP "
            "reselects High/Medium/Low jointly"
        ),
        "traffic_scale": "2x",
        "input_contract": {
            "offline_calibration": "actual and ESM TMs from 0-349",
            "online_policy": "ESM predictions only",
            "actual_test_tm": "sequential-admission evaluation only",
        },
        "selection_protocol": {
            "calibration": list(joint.CALIBRATION),
            "exploration": list(joint.EXPLORATION),
            "validation": list(joint.VALIDATION),
            "untouched_final_for_this_method": list(FINAL_RANGE),
        },
        "frozen_config": {
            "affine_calibration_blend": FINAL_CONFIG[0],
            "residual_uncertainty_std": FINAL_CONFIG[1],
            "medium_slack_fraction": FINAL_CONFIG[2],
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.sha256(checkpoint),
        "seconds_100_snapshots": elapsed,
        "baseline": method.compact(baseline),
        "candidate": compact_candidate["metrics"],
        "delta": compact_candidate["delta"],
        "diagnostics": compact_candidate["diagnostics"],
        "paired_bootstrap_20000": statistics,
        "target_checks": {
            "high_mean_at_least_0_995": (
                method.summary_index(
                    [
                        {"class": name, **values}
                        for name, values in compact_candidate["metrics"].items()
                    ]
                )["High"]["norm_fulfill_mean"]
                >= 0.995
            ),
            "low_mean_significant_decline": statistics["Low"]["mean"]["ci95"][1] < 0.0,
            "medium_mean_improved": compact_candidate["delta"]["Medium.norm_fulfill_mean"] > 0.0,
            "medium_p1_improved": compact_candidate["delta"]["Medium.norm_fulfill_p1"] > 0.0,
            "medium_p10_improved": compact_candidate["delta"]["Medium.norm_fulfill_p10"] > 0.0,
        },
    }
    runtime.write_json(THIS_DIR / "final_joint_hofr_400_500.json", payload)
    runtime.write_csv(THIS_DIR / "final_baseline_rows_400_500.csv", baseline_rows)
    runtime.write_csv(THIS_DIR / "final_joint_hofr_rows_400_500.csv", candidate["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
