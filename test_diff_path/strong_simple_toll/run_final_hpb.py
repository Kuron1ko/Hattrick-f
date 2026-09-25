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


fast = load_module("final_hpb_fast", HERE / "probe_fast_frontier.py")
front, low, v1, v2, plus, base = (
    fast.front, fast.low, fast.v1, fast.v2, fast.plus, fast.base
)
SPLITS = {"evaluation": (400, 500)}
CONFIG = {
    1: {"fallback": "exact Hattrick"},
    2: {
        "bias_objective": "mean Medium gain + 0.24 Low gain",
        "bias_scale": 0.82,
        "joint_steps": 0,
        "low_recovery_steps": 6,
        "low_recovery_learning_rate": 0.32,
    },
    3: {
        "bias_objective": "mean Medium gain + 8 * bottom-5% Medium gain + 0.24 Low gain",
        "bias_scale": 0.82,
        "joint_steps": 6,
        "joint_learning_rate": 0.12,
        "joint_low_weight": 0.22,
        "joint_anchor": 0.01,
        "low_recovery_steps": 8,
        "low_recovery_learning_rate": 0.32,
    },
}


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_biases(load, model, props, train_cache):
    if load == 1:
        return None
    if load == 2:
        variant = {
            "name": "mean", "tail": 0.0,
            "low_weight": 0.24, "low_floor": 0.0,
        }
        return v1.train_bias(model, props, train_cache, variant)
    variant = {
        "name": "tail8_q0.05", "tail": 8.0, "fraction": 0.05,
        "low_weight": 0.24, "low_floor": 0.0,
    }
    return v2.train_bias(model, props, train_cache, variant)


def construct(load, model, props, cache, biases):
    if load == 1:
        return replace(
            cache,
            path_features=torch.cat(
                [cache.policies[1].squeeze(-1), cache.policies[2].squeeze(-1)],
                dim=1,
            ),
        )
    medium_bias, low_bias = biases
    config = CONFIG[load]
    biased = plus.biased_cache(
        cache, medium_bias, low_bias, config["bias_scale"]
    )
    if load == 2:
        policies = [value.squeeze(-1).detach() for value in biased.policies]
    else:
        refined = base.correction.correct_cache(
            model,
            props,
            biased,
            config["joint_steps"],
            config["joint_learning_rate"],
            config["joint_low_weight"],
            config["joint_anchor"],
        )
        policies = low.policies_from_features(cache, refined.path_features)
    policies = fast.recover_low_lr(
        model,
        props,
        cache,
        policies,
        policies[2],
        config["low_recovery_steps"],
        config["low_recovery_learning_rate"],
    )
    return replace(
        cache,
        policies=tuple(value.unsqueeze(-1) for value in policies),
        path_features=torch.cat(policies[1:], dim=1),
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


def benchmark(load, model, props, cache, biases, repetitions):
    for _ in range(2):
        construct(load, model, props, cache, biases)
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        construct(load, model, props, cache, biases)
    synchronize()
    seconds = (time.perf_counter() - started) / repetitions
    return {
        "batch_size": len(cache),
        "repetitions": repetitions,
        "ms_per_batch": 1000.0 * seconds,
        "ms_per_snapshot": 1000.0 * seconds / len(cache),
    }


def information_audit(load, model, props, cache, biases):
    def max_diff(left, right):
        return max(
            float((a - b).abs().max())
            for a, b in zip(left.policies, right.policies)
        )

    original = construct(load, model, props, cache, biases)
    repeated = construct(load, model, props, cache, biases)
    counterfactual_cache = replace(
        cache, tms=tuple(torch.zeros_like(value) for value in cache.tms)
    )
    counterfactual = construct(
        load, model, props, counterfactual_cache, biases
    )
    single = slice_cache(cache, 1)
    single_original = construct(load, model, props, single, biases)
    single_repeated = construct(load, model, props, single, biases)
    single_counterfactual = construct(
        load,
        model,
        props,
        replace(single, tms=tuple(torch.zeros_like(value) for value in single.tms)),
        biases,
    )
    return {
        "batch_repeat_max_abs_diff": max_diff(original, repeated),
        "batch_zero_actual_max_abs_diff": max_diff(original, counterfactual),
        "single_repeat_max_abs_diff": max_diff(single_original, single_repeated),
        "single_zero_actual_max_abs_diff": max_diff(
            single_original, single_counterfactual
        ),
        "batch_note": "Any batch difference must be compared with repeat noise from CUDA sparse reductions.",
        "historical_actual_tm_used_offline": load != 1,
        "current_actual_tm_used_online": False,
    }


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "method": "Hattrick-HPB: historical path bias plus priority-safe recovery",
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "historical_training_bounds": [0, 400],
        "evaluation_bounds": [400, 500],
        "configuration": {str(key): value for key, value in CONFIG.items()},
        "device": str(device),
        "loads": {},
    }
    saved_biases = {}
    for load in (1, 2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        training_seconds = 0.0
        train_cache = None
        if load != 1:
            train_cache = base.runtime.build_policy_cache(
                model, props, 0, 400, batch_size=32
            )
            started = time.perf_counter()
            biases = train_biases(load, model, props, train_cache)
            synchronize()
            training_seconds = time.perf_counter() - started
            saved_biases[str(load)] = {
                "medium": biases[0].detach().cpu(),
                "low": biases[1].detach().cpu(),
            }
        else:
            biases = None
        cache = base.runtime.build_policy_cache(
            model, props, 400, 500, batch_size=32
        )
        baseline_rows, baseline_summary = base.trainer.evaluate(
            model, props, cache
        )
        candidate = construct(load, model, props, cache, biases)
        candidate_rows, candidate_summary = base.runtime.evaluate_cache(
            model, props, candidate, None, batch_size=16
        )
        delta = base.delta(candidate_summary, baseline_summary)
        load_report = {
            "training_seconds": training_seconds,
            "learned_parameters": (
                0 if biases is None else int(biases[0].numel() + biases[1].numel())
            ),
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
                load, model, props, cache, biases
            ),
            "benchmark": {
                "batch_100": benchmark(
                    load, model, props, cache, biases, 20
                ),
                "batch_1": benchmark(
                    load, model, props, slice_cache(cache, 1), biases, 50
                ),
            },
        }
        report["loads"][str(load)] = load_report
        print(
            f"[{load}x] H={delta['High.norm_fulfill_mean']:+.6f}/"
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
    torch.save(saved_biases, HERE / "final_hpb_biases.pt")
    path = HERE / "final_hpb_level4.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
