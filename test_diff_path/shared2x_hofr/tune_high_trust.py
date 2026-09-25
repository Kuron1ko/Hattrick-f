from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR / "probe_joint_optimal_face.py"
spec = importlib.util.spec_from_file_location("shared2x_hofr_high_trust", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load joint HOFR")
joint = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = joint
spec.loader.exec_module(joint)
runtime = joint.runtime
method = joint.method


HOFR_CONFIG = (1.0, 0.75, 0.0)
ALPHAS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0)


def evaluate_alpha(model, props, cache, hofr_cache, baseline, alpha: float):
    path_count = int(cache.policies[0].shape[1])
    base_high = cache.policies[0].squeeze(-1)
    hofr_high = hofr_cache.path_features[:, :path_count]
    mixed_high = (1.0 - float(alpha)) * base_high + float(alpha) * hofr_high
    features = hofr_cache.path_features.clone()
    features[:, :path_count] = mixed_high
    mixed = replace(hofr_cache, path_features=features)
    adapter = joint.ThreePolicyAdapter(path_count)
    rows, summary = runtime.evaluate_cache(model, props, mixed, adapter, batch_size=32)
    delta = method.gaps(summary, baseline)
    indexed = method.summary_index(summary)
    feasible = (
        indexed["High"]["norm_fulfill_mean"] >= 0.995
        and delta["Medium.norm_fulfill_mean"] >= -0.0005
        and delta["Medium.norm_fulfill_p1"] >= 0.0
        and delta["Medium.norm_fulfill_p10"] >= 0.0
        and delta["Low.norm_fulfill_mean"] >= -0.003
        and delta["Low.norm_fulfill_p1"] >= -0.020
        and delta["Low.norm_fulfill_p10"] >= -0.005
    )
    medium_score = (
        delta["Medium.norm_fulfill_mean"]
        + 0.35 * delta["Medium.norm_fulfill_p1"]
        + 0.35 * delta["Medium.norm_fulfill_p10"]
    )
    return {
        "high_trust_alpha": float(alpha),
        "feasible": bool(feasible),
        "medium_score": float(medium_score),
        "metrics": method.compact(summary),
        "delta": delta,
        "high_policy_l1_from_hattrick": float(
            torch.mean(torch.abs(mixed_high - base_high)).item()
        ),
        "rows": rows,
        "cache": mixed,
    }


def rank(result):
    high = result["metrics"]["High"]
    return (
        int(result["feasible"]),
        float(high["norm_fulfill_mean"]),
        float(high["norm_fulfill_p1"]),
        float(result["medium_score"]),
    )


def public(result):
    return {key: value for key, value in result.items() if key not in ("rows", "cache")}


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    calibration_cache = joint.core.build_esm_only_cache(
        model, props, *joint.CALIBRATION, batch_size=64
    )
    calibrators = joint.fit_affine_calibrator(calibration_cache)
    validation_cache = joint.core.build_esm_only_cache(
        model, props, *joint.VALIDATION, batch_size=32
    )
    baseline_rows, baseline = runtime.evaluate_cache(
        model, props, validation_cache, None, batch_size=32
    )
    hofr_cache, diagnostics = joint.build_joint_cache(
        validation_cache, calibrators, HOFR_CONFIG
    )
    results = []
    for alpha in ALPHAS:
        result = evaluate_alpha(
            model, props, validation_cache, hofr_cache, baseline, alpha
        )
        results.append(result)
        print(json.dumps(public(result), ensure_ascii=False), flush=True)
    winner = max(results, key=rank)
    payload = {
        "method": "HOFR High-only policy trust region",
        "range": list(joint.VALIDATION),
        "hofr_config": {
            "affine_calibration_blend": HOFR_CONFIG[0],
            "residual_uncertainty_std": HOFR_CONFIG[1],
            "medium_slack_fraction": HOFR_CONFIG[2],
        },
        "selection_rule": (
            "Medium Mean/P1/P10 >= -0.0005/0/0; Low Mean/P1/P10 >= -0.003/-0.020/-0.005; "
            "then maximize High Mean, High P1, Medium composite"
        ),
        "baseline": method.compact(baseline),
        "hofr_diagnostics": diagnostics,
        "candidates": [public(result) for result in results],
        "winner": public(winner),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.sha256(checkpoint),
        "final_400_500_touched_during_selection": False,
    }
    runtime.write_json(THIS_DIR / "hofr_high_trust_validation.json", payload)
    runtime.write_csv(THIS_DIR / "high_trust_validation_baseline_rows.csv", baseline_rows)
    runtime.write_csv(THIS_DIR / "high_trust_validation_winner_rows.csv", winner["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
