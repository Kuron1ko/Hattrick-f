from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

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


priority = load_module("final_robust_priority_base", HERE / "probe_robust_priority.py")
base, cal = priority.base, priority.cal

QUANTILE = 0.55
BLEND = 0.50
HIGH_WEIGHT = 0.88
STEPS = 40
SECONDARY_BUDGET = 0.02
TRAIN_BOUNDS = (0, 318)


def fit_factors(train_cache):
    predicted, actual, widths = cal.arrays(train_cache)
    ratio = actual / np.maximum(predicted, 1e-6)
    factor = np.maximum(np.quantile(ratio, QUANTILE, axis=0), 1.0)
    pieces = np.split(factor, np.cumsum(widths)[:-1])
    return tuple(
        torch.as_tensor(
            value,
            device=train_cache.policies[0].device,
            dtype=train_cache.policies[0].dtype,
        )
        .repeat_interleave(8)
        .reshape(1, -1, 1)
        for value in pieces
    )


def construct(model, props, cache, factors):
    planning = replace(
        cache,
        predicted_tms=tuple(
            predicted * (1.0 + BLEND * (factor - 1.0))
            for predicted, factor in zip(cache.predicted_tms, factors)
        ),
    )
    raw = priority.priority_correct(
        model, props, planning, HIGH_WEIGHT, STEPS
    )
    guarded, active = priority.priority_guard(
        model, props, planning, raw, SECONDARY_BUDGET
    )
    return replace(
        cache,
        policies=guarded.policies,
        path_features=guarded.path_features,
    ), active


def compact_rows(rows):
    return [
        {
            "snapshot": int(row["snapshot"]),
            "class": str(row["class"]),
            "norm_fulfill": float(row["norm_fulfill"]),
        }
        for row in rows
    ]


def max_policy_diff(left, right):
    return max(
        float((a - b).abs().max())
        for a, b in zip(left.policies, right.policies)
    )


def slice_cache(cache, count):
    return replace(
        cache,
        policies=tuple(value[:count] for value in cache.policies),
        tms=tuple(value[:count] for value in cache.tms),
        predicted_tms=tuple(value[:count] for value in cache.predicted_tms),
        capacities=cache.capacities[:count],
        oracle_flows=tuple(value[:count] for value in cache.oracle_flows),
        oracle_mlus=tuple(value[:count] for value in cache.oracle_mlus),
        path_features=cache.path_features[:count] if cache.path_features is not None else None,
    )


def information_audit(model, props, cache, factors):
    single = slice_cache(cache, 1)
    original, original_active = construct(model, props, single, factors)
    counterfactual, counterfactual_active = construct(
        model,
        props,
        replace(single, tms=tuple(torch.zeros_like(value) for value in single.tms)),
        factors,
    )
    return {
        "single_zero_actual_policy_max_abs_diff": max_policy_diff(
            original, counterfactual
        ),
        "single_zero_actual_activation_mismatch": int(
            (original_active != counterfactual_active).sum()
        ),
        "current_actual_tm_used_online": False,
    }


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(model, props, cache, factors, repetitions):
    construct(model, props, cache, factors)
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        construct(model, props, cache, factors)
    synchronize()
    elapsed = (time.perf_counter() - started) / repetitions
    return {
        "batch_size": len(cache),
        "repetitions": repetitions,
        "ms_per_batch": 1000.0 * elapsed,
        "ms_per_snapshot": 1000.0 * elapsed / len(cache),
    }


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "method": "RQP: residual-quantile robust lexicographic routing",
        "strict_esm_online": True,
        "current_actual_tm_used_online": False,
        "same_configuration_for_all_loads": True,
        "training_bounds": list(TRAIN_BOUNDS),
        "config": {
            "residual_ratio_quantile": QUANTILE,
            "robust_blend": BLEND,
            "high_priority_gain_weight": HIGH_WEIGHT,
            "optimization_steps": STEPS,
            "secondary_predicted_budget": SECONDARY_BUDGET,
        },
        "loads": {},
    }
    for load in (1, 2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        train_cache = base.runtime.build_policy_cache(
            model, props, *TRAIN_BOUNDS, batch_size=32
        )
        factors = fit_factors(train_cache)
        cache = base.runtime.build_policy_cache(
            model, props, 400, 500, batch_size=32
        )
        baseline_rows, baseline_summary = base.trainer.evaluate(
            model, props, cache
        )
        candidate, active = construct(model, props, cache, factors)
        candidate_rows, candidate_summary = base.runtime.evaluate_cache(
            model, props, candidate, None, batch_size=16
        )
        delta = base.delta(candidate_summary, baseline_summary)
        report["loads"][str(load)] = {
            "learned_residual_factors": int(sum(value.numel() // 8 for value in factors)),
            "active": int(active.sum()),
            "evaluation": {
                "baseline": base.compact(baseline_summary),
                "candidate": base.compact(candidate_summary),
                "delta": delta,
                "bootstrap": base.screen.method.large.paired_bootstrap(
                    baseline_rows, candidate_rows
                ),
                "rows": {
                    "baseline": compact_rows(baseline_rows),
                    "candidate": compact_rows(candidate_rows),
                },
            },
            "information_audit": information_audit(
                model, props, cache, factors
            ),
            "benchmark": {
                "batch_100": benchmark(model, props, cache, factors, 5),
                "batch_1": benchmark(
                    model, props, slice_cache(cache, 1), factors, 3
                ),
            },
        }
        print(
            f"[{load}x] active={int(active.sum())} "
            f"H={delta['High.norm_fulfill_mean']:+.6f}/"
            f"{delta['High.norm_fulfill_p1']:+.6f}/"
            f"{delta['High.norm_fulfill_p10']:+.6f} "
            f"M={delta['Medium.norm_fulfill_mean']:+.6f}/"
            f"{delta['Medium.norm_fulfill_p1']:+.6f}/"
            f"{delta['Medium.norm_fulfill_p10']:+.6f} "
            f"L={delta['Low.norm_fulfill_mean']:+.6f}/"
            f"{delta['Low.norm_fulfill_p1']:+.6f}/"
            f"{delta['Low.norm_fulfill_p10']:+.6f}",
            flush=True,
        )
    path = HERE / "final_robust_priority.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
