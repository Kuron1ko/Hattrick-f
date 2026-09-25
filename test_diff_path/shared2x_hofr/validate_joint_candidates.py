from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR / "probe_joint_optimal_face.py"
spec = importlib.util.spec_from_file_location("shared2x_hofr_joint_validation", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load joint HOFR probe")
joint = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = joint
spec.loader.exec_module(joint)
runtime = joint.runtime
method = joint.method


CONFIGS = (
    (1.0, 0.50, 0.0),
    (1.0, 0.50, 0.005),
    (1.0, 0.75, 0.0),
)


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
    results = []
    for config in CONFIGS:
        result = joint.evaluate_config(
            model, props, validation_cache, calibrators, baseline, config
        )
        results.append(result)
        print(json.dumps(joint.public(result), ensure_ascii=False), flush=True)
    winner = max(results, key=joint.rank)
    payload = {
        "method": "joint HOFR conservative-candidate validation",
        "range": list(joint.VALIDATION),
        "final_test_400_500_touched": False,
        "baseline": method.compact(baseline),
        "candidates": [joint.public(result) for result in results],
        "winner": joint.public(winner),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.sha256(checkpoint),
    }
    runtime.write_json(THIS_DIR / "hofr_joint_validation_candidates.json", payload)
    runtime.write_csv(THIS_DIR / "joint_validation_winner_rows.csv", winner["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
