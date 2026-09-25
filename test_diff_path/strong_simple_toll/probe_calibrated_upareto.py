from __future__ import annotations

from dataclasses import replace
import importlib.util
import itertools
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


universal = load_module("calibrated_upareto_universal", HERE / "probe_universal_num.py")
across = load_module("calibrated_upareto_across", HERE / "probe_universal_across_loads.py")
cal = load_module("calibrated_upareto_cal", HERE / "analyze_tm_calibrators.py")
base = universal.base

CALIBRATORS = ("scale", "affine", "bias", "ridge_0.1")
BLENDS = (0.25, 0.50, 0.75, 1.00)
CONFIGS = tuple(
    (f"relative_softmin_{tau:g}", steps, 0.08, anchor)
    for tau, steps, anchor in itertools.product(
        (0.003, 0.005), (16, 24), (0.01, 0.02)
    )
)
MARGINS = (0.0, 0.0001)
METRICS = ("norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10")


def predict(name, train_x, train_y, test_x):
    if name == "scale":
        return cal.per_feature_scale(train_x, train_y, test_x)
    if name == "affine":
        return cal.per_feature_affine(train_x, train_y, test_x)
    if name == "bias":
        return np.maximum(test_x + np.mean(train_y - train_x, axis=0), 0.0)
    if name.startswith("ridge_"):
        return cal.ridge_log(train_x, train_y, test_x, float(name.split("_")[1]))
    raise KeyError(name)


def rebuilt_tms(cache, prediction, widths):
    pieces = np.split(prediction, np.cumsum(widths)[:-1], axis=1)
    return tuple(
        torch.as_tensor(
            value,
            device=cache.policies[0].device,
            dtype=cache.policies[0].dtype,
        )
        .repeat_interleave(8, dim=1)
        .unsqueeze(-1)
        for value in pieces
    )


def calibrated_cache(cache, calibrated, raw, widths, blend):
    prediction = (1.0 - blend) * raw + blend * calibrated
    return replace(cache, predicted_tms=rebuilt_tms(cache, prediction, widths))


def metric_floor(delta, class_name):
    return min(float(delta[f"{class_name}.{metric}"]) for metric in METRICS)


def score(delta):
    high_floor = metric_floor(delta, "High")
    high_mean = float(delta["High.norm_fulfill_mean"])
    secondary_floor = min(metric_floor(delta, "Medium"), metric_floor(delta, "Low"))
    return 2.0 * high_floor + high_mean + 0.10 * secondary_floor


def safe(delta, tolerance=-0.003):
    return all(
        float(delta[f"{class_name}.{metric}"]) >= tolerance
        for class_name in ("Medium", "Low")
        for metric in METRICS
    )


def evaluate(model, props, original_cache, planning_cache, baseline, spec):
    config = tuple(spec["config"])
    raw_candidate = universal.correct(model, props, planning_cache, config)
    guarded, active = across.pareto_guard(
        model, props, planning_cache, raw_candidate, spec["margin"]
    )
    candidate = replace(
        original_cache,
        policies=guarded.policies,
        path_features=guarded.path_features,
    )
    _, summary = base.runtime.evaluate_cache(
        model, props, candidate, None, batch_size=16
    )
    delta = base.delta(summary, baseline)
    return {
        **spec,
        "active": int(active.sum()),
        "summary": base.compact(summary),
        "delta": delta,
        "safe_secondary": safe(delta),
        "score": score(delta),
    }


def print_trial(stage, value):
    d = value["delta"]
    print(
        f"[{stage}] {value['calibrator']} b={value['blend']:.2f} "
        f"{value['config'][0]} s={value['config'][1]} a={value['config'][3]:g} "
        f"g={value['margin']:g} active={value['active']} safe={value['safe_secondary']} "
        f"H={d['High.norm_fulfill_mean']:+.5f}/"
        f"{d['High.norm_fulfill_p1']:+.5f}/"
        f"{d['High.norm_fulfill_p10']:+.5f} "
        f"M={d['Medium.norm_fulfill_mean']:+.5f}/"
        f"{d['Medium.norm_fulfill_p1']:+.5f}/"
        f"{d['Medium.norm_fulfill_p10']:+.5f} "
        f"L={d['Low.norm_fulfill_mean']:+.5f}/"
        f"{d['Low.norm_fulfill_p1']:+.5f}/"
        f"{d['Low.norm_fulfill_p10']:+.5f}",
        flush=True,
    )


def run_stage(model, props, cache, train_x, train_y, widths, specs, stage):
    _, baseline = base.trainer.evaluate(model, props, cache)
    raw, _, _ = cal.arrays(cache)
    calibrated = {
        name: predict(name, train_x, train_y, raw) for name in CALIBRATORS
    }
    planning = {
        (name, blend): calibrated_cache(cache, calibrated[name], raw, widths, blend)
        for name in CALIBRATORS
        for blend in BLENDS
    }
    values = []
    for spec in specs:
        value = evaluate(
            model,
            props,
            cache,
            planning[(spec["calibrator"], spec["blend"])],
            baseline,
            spec,
        )
        values.append(value)
        print_trial(stage, value)
    return baseline, values


def ranking_key(value):
    d = value["delta"]
    high_positive = metric_floor(d, "High") > 0.0
    return (value["safe_secondary"], high_positive, value["score"])


def strip_result(value):
    return {
        key: value[key]
        for key in ("calibrator", "blend", "config", "margin")
    }


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props, _ = base.screen.load_backbone(3, device)
    train_cache = base.runtime.build_policy_cache(
        model, props, 0, 318, batch_size=32
    )
    train_x, train_y, widths = cal.arrays(train_cache)

    all_specs = [
        {
            "calibrator": calibrator,
            "blend": blend,
            "config": list(config),
            "margin": margin,
        }
        for calibrator, blend, config, margin in itertools.product(
            CALIBRATORS, BLENDS, CONFIGS, MARGINS
        )
    ]
    small_cache = base.runtime.build_policy_cache(
        model, props, 318, 334, batch_size=16
    )
    small_baseline, small = run_stage(
        model, props, small_cache, train_x, train_y, widths, all_specs, "small"
    )
    small.sort(key=ranking_key, reverse=True)
    validation_specs = [strip_result(value) for value in small[:20]]

    validation_cache = base.runtime.build_policy_cache(
        model, props, 334, 400, batch_size=32
    )
    validation_baseline, validation = run_stage(
        model,
        props,
        validation_cache,
        train_x,
        train_y,
        widths,
        validation_specs,
        "validation",
    )
    validation.sort(key=ranking_key, reverse=True)
    evaluation_specs = [strip_result(value) for value in validation[:6]]

    evaluation_cache = base.runtime.build_policy_cache(
        model, props, 400, 500, batch_size=32
    )
    evaluation_baseline, evaluation = run_stage(
        model,
        props,
        evaluation_cache,
        train_x,
        train_y,
        widths,
        evaluation_specs,
        "evaluation",
    )
    evaluation.sort(key=ranking_key, reverse=True)
    report = {
        "method": "historically calibrated ESM plus class-symmetric U-Pareto",
        "strict_esm_online": True,
        "current_actual_tm_used_online": False,
        "training_bounds": [0, 318],
        "small_bounds": [318, 334],
        "validation_bounds": [334, 400],
        "evaluation_bounds": [400, 500],
        "small_baseline": base.compact(small_baseline),
        "small_top": small[:30],
        "validation_baseline": base.compact(validation_baseline),
        "validation": validation,
        "frozen_specs": evaluation_specs,
        "evaluation_baseline": base.compact(evaluation_baseline),
        "evaluation": evaluation,
    }
    path = HERE / "calibrated_upareto.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
