from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

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


across = load_module("final_universal_across", HERE / "probe_universal_across_loads.py")
universal, base = across.universal, across.base
CONFIG = ("relative_softmin_0.005", 16, 0.08, 0.02)
PARETO_MARGIN = 0.0001


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def construct(model, props, cache):
    raw = universal.correct(model, props, cache, CONFIG)
    return across.pareto_guard(
        model, props, cache, raw, PARETO_MARGIN
    )


def compact_rows(rows):
    return [
        {
            "snapshot": int(row["snapshot"]),
            "class": str(row["class"]),
            "norm_fulfill": float(row["norm_fulfill"]),
        }
        for row in rows
    ]


def slice_cache(cache, count):
    return replace(
        cache,
        policies=tuple(value[:count] for value in cache.policies),
        tms=tuple(value[:count] for value in cache.tms),
        predicted_tms=tuple(value[:count] for value in cache.predicted_tms),
        capacities=cache.capacities[:count],
        oracle_flows=tuple(value[:count] for value in cache.oracle_flows),
        oracle_mlus=tuple(value[:count] for value in cache.oracle_mlus),
        path_features=(
            cache.path_features[:count] if cache.path_features is not None else None
        ),
    )


def max_policy_diff(left, right):
    return max(
        float((a - b).abs().max())
        for a, b in zip(left.policies, right.policies)
    )


def information_audit(model, props, cache):
    single = slice_cache(cache, 1)
    original, original_active = construct(model, props, single)
    repeated, repeated_active = construct(model, props, single)
    counterfactual, counterfactual_active = construct(
        model,
        props,
        replace(single, tms=tuple(torch.zeros_like(value) for value in single.tms)),
    )
    return {
        "single_repeat_policy_max_abs_diff": max_policy_diff(original, repeated),
        "single_zero_actual_policy_max_abs_diff": max_policy_diff(
            original, counterfactual
        ),
        "single_repeat_activation_mismatch": int(
            (original_active != repeated_active).sum()
        ),
        "single_zero_actual_activation_mismatch": int(
            (original_active != counterfactual_active).sum()
        ),
        "current_actual_tm_used_online": False,
    }


def benchmark(model, props, cache, repetitions):
    for _ in range(2):
        construct(model, props, cache)
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        construct(model, props, cache)
    synchronize()
    seconds = (time.perf_counter() - started) / repetitions
    return {
        "batch_size": len(cache),
        "repetitions": repetitions,
        "ms_per_batch": 1000.0 * seconds,
        "ms_per_snapshot": 1000.0 * seconds / len(cache),
    }


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "method": "U-Pareto: class-symmetric max-min relative-gain routing",
        "objective": "smooth minimum over High/Medium/Low predicted fulfillment gains relative to Hattrick",
        "strict_esm_inference": True,
        "same_configuration_for_all_loads": True,
        "learned_parameters": 0,
        "config": list(CONFIG),
        "pareto_margin": PARETO_MARGIN,
        "loads": {},
    }
    for load in (1, 2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        cache = base.runtime.build_policy_cache(
            model, props, 400, 500, batch_size=32
        )
        baseline_rows, baseline_summary = base.trainer.evaluate(
            model, props, cache
        )
        candidate, active = construct(model, props, cache)
        candidate_rows, candidate_summary = base.runtime.evaluate_cache(
            model, props, candidate, None, batch_size=16
        )
        delta = base.delta(candidate_summary, baseline_summary)
        report["loads"][str(load)] = {
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
            "information_audit": information_audit(model, props, cache),
            "benchmark": {
                "batch_100": benchmark(model, props, cache, 20),
                "batch_1": benchmark(
                    model, props, slice_cache(cache, 1), 30
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
    path = HERE / "final_universal_pareto.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
