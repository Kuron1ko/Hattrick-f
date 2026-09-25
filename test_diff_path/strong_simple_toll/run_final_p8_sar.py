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


base = load_module("p8_sar_base", HERE / "run_tradeoff_screen.py")
CONFIG = (8, 0.08, 0.241, 0.01)
GATE = 0.70
SPLITS = {
    "safety": (318, 350),
    "validation": (350, 400),
    "evaluation": (400, 500),
}


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def construct_policy(model, props, cache):
    active = base.gate(cache)
    if not bool(active.any().item()):
        return replace(cache, path_features=base.base_path_features(cache)), active
    candidate = base.correction.correct_cache(model, props, cache, *CONFIG)
    return base.apply_fallback(cache, candidate, active), active


def compact_rows(rows):
    return [
        {
            "snapshot": int(row["snapshot"]),
            "class": str(row["class"]),
            "norm_fulfill": float(row["norm_fulfill"]),
        }
        for row in rows
    ]


def evaluate(model, props, cache):
    baseline_rows, baseline_summary = base.trainer.evaluate(model, props, cache)
    started = time.perf_counter()
    candidate, active = construct_policy(model, props, cache)
    synchronize()
    overlay_seconds = time.perf_counter() - started
    if not bool(active.any().item()):
        rows = [dict(row) for row in baseline_rows]
        summary = [dict(row) for row in baseline_summary]
    else:
        rows, summary = base.trainer.evaluate(model, props, cache, candidate)
    return {
        "baseline": base.compact(baseline_summary),
        "candidate": base.compact(summary),
        "delta": base.delta(summary, baseline_summary),
        "bootstrap": base.screen.method.large.paired_bootstrap(
            baseline_rows, rows
        ),
        "active": int(active.sum().item()),
        "sample_count": len(cache),
        "overlay_ms_per_snapshot": 1000.0 * overlay_seconds / len(cache),
        "rows": {
            "baseline": compact_rows(baseline_rows),
            "candidate": compact_rows(rows),
        },
    }


def information_audit(model, props, cache):
    original, original_active = construct_policy(model, props, cache)
    counterfactual_cache = replace(
        cache, tms=tuple(torch.zeros_like(value) for value in cache.tms)
    )
    counterfactual, counterfactual_active = construct_policy(
        model, props, counterfactual_cache
    )
    return {
        "activation_mismatches_after_zero_actual": int(
            (original_active != counterfactual_active).sum().item()
        ),
        "policy_max_abs_diff_after_zero_actual": float(
            (original.path_features - counterfactual.path_features).abs().max().item()
        ),
    }


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


def benchmark(model, props, cache, repetitions):
    for _ in range(3):
        construct_policy(model, props, cache)
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        construct_policy(model, props, cache)
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
    base.GATE = GATE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "method": "P8-SAR: gated eight-step strict-ESM admission refinement",
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "auxiliary_learned_parameters": 0,
        "parameters": {
            "steps": CONFIG[0],
            "learning_rate": CONFIG[1],
            "low_weight": CONFIG[2],
            "reverse_kl_anchor": CONFIG[3],
            "activation_threshold": GATE,
            "high_policy": "exact frozen Hattrick policy",
        },
        "objective": "predicted Medium fulfillment + 0.241 * predicted Low fulfillment - 0.01 * reverse-KL anchor",
        "splits": {key: list(value) for key, value in SPLITS.items()},
        "device": str(device),
        "loads": {},
    }
    for load in (1, 2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        load_report = {}
        caches = {}
        for split, bounds in SPLITS.items():
            cache = base.runtime.build_policy_cache(
                model, props, *bounds, batch_size=32
            )
            caches[split] = cache
            load_report[split] = evaluate(model, props, cache)
            d = load_report[split]["delta"]
            print(
                f"[{load}x/{split}] active={load_report[split]['active']}/{len(cache)} "
                f"M={d['Medium.norm_fulfill_mean']:+.6f} "
                f"L={d['Low.norm_fulfill_mean']:+.6f}",
                flush=True,
            )
        evaluation_cache = caches["evaluation"]
        load_report["information_audit"] = information_audit(
            model, props, evaluation_cache
        )
        load_report["benchmark"] = {
            "batch_100": benchmark(model, props, evaluation_cache, 20),
            "batch_1": benchmark(
                model, props, slice_cache(evaluation_cache, 1), 100
            ),
        }
        report["loads"][str(load)] = load_report
    path = HERE / "final_p8_sar_level4.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
