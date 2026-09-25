from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


joint = load_module(
    "shared2x_hofr_high_trust_final_joint", THIS_DIR / "probe_joint_optimal_face.py"
)
statistics_module = load_module(
    "shared2x_hofr_high_trust_statistics", THIS_DIR / "run_final_joint_hofr.py"
)
runtime = joint.runtime
method = joint.method


HOFR_CONFIG = (1.0, 0.75, 0.0)
HIGH_TRUST_ALPHA = 0.75
FINAL_RANGE = (400, 500)


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    calibration_cache = joint.core.build_esm_only_cache(
        model, props, *joint.CALIBRATION, batch_size=64
    )
    calibrators = joint.fit_affine_calibrator(calibration_cache)
    cache = joint.core.build_esm_only_cache(
        model, props, *FINAL_RANGE, batch_size=32
    )
    baseline_rows, baseline = runtime.evaluate_cache(
        model, props, cache, None, batch_size=32
    )

    started = time.perf_counter()
    hofr_cache, diagnostics = joint.build_joint_cache(
        cache, calibrators, HOFR_CONFIG
    )
    path_count = int(cache.policies[0].shape[1])
    base_high = cache.policies[0].squeeze(-1)
    hofr_high = hofr_cache.path_features[:, :path_count]
    mixed_high = (
        (1.0 - HIGH_TRUST_ALPHA) * base_high
        + HIGH_TRUST_ALPHA * hofr_high
    )
    features = hofr_cache.path_features.clone()
    features[:, :path_count] = mixed_high
    candidate_cache = replace(hofr_cache, path_features=features)
    adapter = joint.ThreePolicyAdapter(path_count)
    candidate_rows, candidate = runtime.evaluate_cache(
        model, props, candidate_cache, adapter, batch_size=32
    )
    elapsed = time.perf_counter() - started
    delta = method.gaps(candidate, baseline)
    statistics = {
        class_name: statistics_module.bootstrap(
            statistics_module.class_values(baseline_rows, class_name),
            statistics_module.class_values(candidate_rows, class_name),
            20260831 + index,
        )
        for index, class_name in enumerate(("High", "Medium", "Low"))
    }
    previous_path = THIS_DIR / "final_joint_hofr_400_500.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous_candidate = previous["candidate"]
    new_compact = method.compact(candidate)
    payload = {
        "method": "joint HOFR with High-only policy trust region",
        "traffic_scale": "2x",
        "protocol_note": (
            "alpha was selected on 358-399 without using 400-499 in the trust-region search; "
            "however 400-499 had already been evaluated by the preceding alpha=1 HOFR experiment, "
            "so this is a follow-up confirmation rather than a fresh untouched test"
        ),
        "input_contract": {
            "offline_calibration": "actual and ESM TMs from 0-349",
            "online_policy": "ESM predictions only",
            "actual_test_tm": "sequential-admission evaluation only",
        },
        "selection": {
            "range": list(joint.VALIDATION),
            "high_trust_alpha": HIGH_TRUST_ALPHA,
            "hofr_config": {
                "affine_calibration_blend": HOFR_CONFIG[0],
                "residual_uncertainty_std": HOFR_CONFIG[1],
                "medium_slack_fraction": HOFR_CONFIG[2],
            },
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.sha256(checkpoint),
        "seconds_100_snapshots": elapsed,
        "baseline": method.compact(baseline),
        "candidate": new_compact,
        "delta_vs_hattrick": delta,
        "delta_vs_previous_alpha_1_hofr": {
            f"{class_name}.{metric}": float(
                new_compact[class_name][metric] - previous_candidate[class_name][metric]
            )
            for class_name in ("High", "Medium", "Low")
            for metric in (
                "norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10"
            )
        },
        "diagnostics": {
            **diagnostics,
            "high_policy_l1_from_hattrick": float(
                torch.mean(torch.abs(mixed_high - base_high)).item()
            ),
        },
        "paired_bootstrap_20000": statistics,
        "target_checks": {
            "high_mean_at_least_0_995": new_compact["High"]["norm_fulfill_mean"] >= 0.995,
            "low_mean_significant_decline": statistics["Low"]["mean"]["ci95"][1] < 0.0,
            "medium_mean_improved": delta["Medium.norm_fulfill_mean"] > 0.0,
            "medium_p1_improved": delta["Medium.norm_fulfill_p1"] > 0.0,
            "medium_p10_improved": delta["Medium.norm_fulfill_p10"] > 0.0,
        },
    }
    runtime.write_json(THIS_DIR / "final_high_trust_400_500.json", payload)
    runtime.write_csv(THIS_DIR / "final_high_trust_rows_400_500.csv", candidate_rows)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
