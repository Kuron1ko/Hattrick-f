from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cal = load_module("calibrated_online_cal", HERE / "analyze_tm_calibrators.py")
base = cal.base
CALIBRATORS = ("raw", "affine", "ridge_0.01", "ridge_0.1", "knn_8")
CONFIGS = (
    (16, 0.08, 0.05, 0.01),
    (24, 0.06, 0.05, 0.01),
    (32, 0.06, 0.05, 0.01),
    (24, 0.06, 0.10, 0.01),
    (24, 0.06, 0.15, 0.01),
    (24, 0.06, 0.20, 0.01),
)


def predict(name, train_x, train_y, test_x):
    if name == "raw":
        return test_x
    if name == "affine":
        return cal.per_feature_affine(train_x, train_y, test_x)
    if name.startswith("ridge_"):
        return cal.ridge_log(train_x, train_y, test_x, float(name.split("_")[1]))
    if name.startswith("knn_"):
        return cal.knn_residual(train_x, train_y, test_x, int(name.split("_")[1]))
    raise KeyError(name)


def rebuilt_tms(cache, prediction, widths):
    pieces = np.split(prediction, np.cumsum(widths)[:-1], axis=1)
    return tuple(
        torch.as_tensor(value, device=cache.policies[0].device, dtype=cache.policies[0].dtype)
        .repeat_interleave(8, dim=1)
        .unsqueeze(-1)
        for value in pieces
    )


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "purpose": "strict-ESM training-only TM calibration plus fixed-High refinement",
        "strict_esm": True,
        "actual_tm_used_for_policy": False,
        "loads": {},
    }
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        train_cache = base.runtime.build_policy_cache(model, props, 0, 318, batch_size=32)
        train_x, train_y, widths = cal.arrays(train_cache)
        load_report = {}
        for split, bounds in {"development": (318, 400), "evaluation": (400, 500)}.items():
            cache = base.runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            _, baseline = base.trainer.evaluate(model, props, cache)
            test_x, _, _ = cal.arrays(cache)
            trials = []
            for calibrator in CALIBRATORS:
                prediction = predict(calibrator, train_x, train_y, test_x)
                planning_cache = replace(
                    cache, predicted_tms=rebuilt_tms(cache, prediction, widths)
                )
                for config in CONFIGS:
                    candidate = base.correction.correct_cache(
                        model, props, planning_cache, *config
                    )
                    candidate = replace(cache, path_features=candidate.path_features)
                    _, summary = base.trainer.evaluate(model, props, cache, candidate)
                    value = {
                        "calibrator": calibrator,
                        "config": list(config),
                        "summary": base.compact(summary),
                        "delta": base.delta(summary, baseline),
                    }
                    trials.append(value)
                    d = value["delta"]
                    print(
                        f"[{load}x/{split}] {calibrator} s={config[0]} l={config[2]:g} "
                        f"M={d['Medium.norm_fulfill_mean']:+.5f}/"
                        f"{d['Medium.norm_fulfill_p1']:+.5f}/"
                        f"{d['Medium.norm_fulfill_p10']:+.5f} "
                        f"L={d['Low.norm_fulfill_mean']:+.5f}",
                        flush=True,
                    )
            load_report[split] = {
                "baseline": base.compact(baseline),
                "trials": trials,
            }
        report["loads"][str(load)] = load_report
    path = HERE / "calibrated_online.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
