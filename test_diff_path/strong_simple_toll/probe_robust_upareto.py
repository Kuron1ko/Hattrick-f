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


common = load_module("robust_upareto_common", HERE / "probe_calibrated_upareto.py")
base, cal = common.base, common.cal
universal, across = common.universal, common.across
QUANTILES = (0.60, 0.75, 0.90)
KINDS = ("additive", "multiplicative")
BLENDS = (0.25, 0.50, 0.75, 1.00)
CONFIGS = (
    ("relative_softmin_0.002", 24, 0.08, 0.01),
    ("relative_softmin_0.003", 24, 0.08, 0.01),
    ("relative_softmin_0.003", 32, 0.08, 0.02),
)
MARGINS = (0.0, 0.0001)
METRICS = common.METRICS


def robust_prediction(kind, quantile, train_x, train_y, test_x):
    if kind == "additive":
        underprediction = np.maximum(train_y - train_x, 0.0)
        margin = np.quantile(underprediction, quantile, axis=0)
        return test_x + margin
    if kind == "multiplicative":
        ratio = train_y / np.maximum(train_x, 1e-6)
        factor = np.maximum(np.quantile(ratio, quantile, axis=0), 1.0)
        return test_x * factor
    raise KeyError(kind)


def planning_cache(cache, raw, robust, widths, blend):
    prediction = (1.0 - blend) * raw + blend * robust
    return replace(
        cache,
        predicted_tms=common.rebuilt_tms(cache, prediction, widths),
    )


def evaluate(model, props, cache, planning, baseline, spec):
    raw_candidate = universal.correct(
        model, props, planning, tuple(spec["config"])
    )
    guarded, active = across.pareto_guard(
        model, props, planning, raw_candidate, spec["margin"]
    )
    candidate = replace(
        cache,
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
        "safe_secondary": common.safe(delta),
        "score": common.score(delta),
    }


def show(stage, value):
    d = value["delta"]
    print(
        f"[{stage}] {value['kind']} q={value['quantile']:.2f} "
        f"b={value['blend']:.2f} {value['config'][0]} s={value['config'][1]} "
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


def rank(value):
    d = value["delta"]
    high_floor = common.metric_floor(d, "High")
    return (
        value["safe_secondary"],
        high_floor > 0.0,
        1.5 * high_floor + float(d["High.norm_fulfill_mean"]),
    )


def spec_only(value):
    return {
        key: value[key]
        for key in ("kind", "quantile", "blend", "config", "margin")
    }


def run_stage(model, props, cache, train_x, train_y, widths, specs, stage):
    _, baseline = base.trainer.evaluate(model, props, cache)
    raw, _, _ = cal.arrays(cache)
    robust = {
        (kind, quantile): robust_prediction(
            kind, quantile, train_x, train_y, raw
        )
        for kind in KINDS
        for quantile in QUANTILES
    }
    plans = {
        (kind, quantile, blend): planning_cache(
            cache, raw, robust[(kind, quantile)], widths, blend
        )
        for kind in KINDS
        for quantile in QUANTILES
        for blend in BLENDS
    }
    values = []
    for spec in specs:
        value = evaluate(
            model,
            props,
            cache,
            plans[(spec["kind"], spec["quantile"], spec["blend"])],
            baseline,
            spec,
        )
        values.append(value)
        show(stage, value)
    return baseline, values


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props, _ = base.screen.load_backbone(3, device)
    train_cache = base.runtime.build_policy_cache(
        model, props, 0, 318, batch_size=32
    )
    train_x, train_y, widths = cal.arrays(train_cache)
    specs = [
        {
            "kind": kind,
            "quantile": quantile,
            "blend": blend,
            "config": list(config),
            "margin": margin,
        }
        for kind, quantile, blend, config, margin in itertools.product(
            KINDS, QUANTILES, BLENDS, CONFIGS, MARGINS
        )
    ]
    small_cache = base.runtime.build_policy_cache(
        model, props, 318, 334, batch_size=16
    )
    small_baseline, small = run_stage(
        model, props, small_cache, train_x, train_y, widths, specs, "small"
    )
    small.sort(key=rank, reverse=True)
    validation_specs = [spec_only(value) for value in small[:18]]
    validation_cache = base.runtime.build_policy_cache(
        model, props, 334, 400, batch_size=32
    )
    validation_baseline, validation = run_stage(
        model, props, validation_cache, train_x, train_y, widths,
        validation_specs, "validation"
    )
    validation.sort(key=rank, reverse=True)
    frozen = [spec_only(value) for value in validation[:6]]
    evaluation_cache = base.runtime.build_policy_cache(
        model, props, 400, 500, batch_size=32
    )
    evaluation_baseline, evaluation = run_stage(
        model, props, evaluation_cache, train_x, train_y, widths,
        frozen, "evaluation"
    )
    evaluation.sort(key=rank, reverse=True)
    report = {
        "method": "one-sided residual-quantile robust ESM plus U-Pareto",
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
        "frozen_specs": frozen,
        "evaluation_baseline": base.compact(evaluation_baseline),
        "evaluation": evaluation,
    }
    path = HERE / "robust_upareto.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
