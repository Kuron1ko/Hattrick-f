from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import sys

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


refine = load_module("robust_priority_push_base", HERE / "probe_robust_priority_refine.py")
priority, base, cal = refine.priority, refine.base, refine.cal
KINDS = ("multiplicative",)
QUANTILES = (0.55, 0.65)
BLENDS = (0.40, 0.50)
HIGH_WEIGHTS = (0.90, 1.00, 1.20)
STEPS = (40, 48)
SECONDARY_BUDGETS = (0.020, 0.040, 0.080)


def run_stage(model, props, cache, train_x, train_y, widths, specs, stage):
    _, baseline = base.trainer.evaluate(model, props, cache)
    raw_tm, _, _ = cal.arrays(cache)
    corrected = {
        (kind, quantile): priority.robust.robust_prediction(
            kind, quantile, train_x, train_y, raw_tm
        )
        for kind in KINDS for quantile in QUANTILES
    }
    plans = {
        (kind, quantile, blend): priority.robust.planning_cache(
            cache, raw_tm, corrected[(kind, quantile)], widths, blend
        )
        for kind in KINDS for quantile in QUANTILES for blend in BLENDS
    }
    values = []
    for spec in specs:
        value = priority.evaluate(
            model, props, cache,
            plans[(spec["kind"], spec["quantile"], spec["blend"])],
            baseline, spec
        )
        values.append(value)
        priority.show(stage, value)
    return baseline, values


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props, _ = base.screen.load_backbone(3, device)
    train_cache = base.runtime.build_policy_cache(model, props, 0, 318, batch_size=32)
    train_x, train_y, widths = cal.arrays(train_cache)
    specs = [
        {
            "kind": kind,
            "quantile": quantile,
            "blend": blend,
            "high_weight": high_weight,
            "steps": steps,
            "secondary_budget": budget,
        }
        for kind, quantile, blend, high_weight, steps, budget in itertools.product(
            KINDS, QUANTILES, BLENDS, HIGH_WEIGHTS, STEPS, SECONDARY_BUDGETS
        )
    ]
    small_cache = base.runtime.build_policy_cache(model, props, 318, 334, batch_size=16)
    small_baseline, small = run_stage(
        model, props, small_cache, train_x, train_y, widths, specs, "small"
    )
    small.sort(key=priority.rank, reverse=True)
    validation_specs = [priority.spec_only(value) for value in small[:18]]
    validation_cache = base.runtime.build_policy_cache(model, props, 334, 400, batch_size=32)
    validation_baseline, validation = run_stage(
        model, props, validation_cache, train_x, train_y, widths,
        validation_specs, "validation"
    )
    validation.sort(key=priority.rank, reverse=True)
    frozen = [priority.spec_only(value) for value in validation[:6]]
    evaluation_cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
    evaluation_baseline, evaluation = run_stage(
        model, props, evaluation_cache, train_x, train_y, widths,
        frozen, "evaluation"
    )
    evaluation.sort(key=priority.rank, reverse=True)
    report = {
        "method": "final high-weight push for robust lexicographic routing",
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
    path = HERE / "robust_priority_push.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
