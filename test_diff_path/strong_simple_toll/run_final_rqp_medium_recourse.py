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


recourse = load_module(
    "final_rqp_medium_recourse_base", HERE / "probe_rqp_medium_recourse.py"
)
final, base = recourse.final, recourse.base

SPEC = {
    "steps": 12,
    "high_budget": 0.0002,
    "low_budget": 0.002,
    "anchor": 0.0005,
}
EVALUATION_BOUNDS = (400, 500)


def max_policy_diff(left, right):
    return max(
        float((a - b).abs().max())
        for a, b in zip(left.policies, right.policies)
    )


def information_audit(model, props, cache, factors):
    single = final.slice_cache(cache, 1)
    original, _, original_active = recourse.construct(
        model, props, single, factors, SPEC
    )
    counterfactual, _, counterfactual_active = recourse.construct(
        model,
        props,
        replace(
            single,
            tms=tuple(torch.zeros_like(value) for value in single.tms),
        ),
        factors,
        SPEC,
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
    recourse.construct(model, props, cache, factors, SPEC)
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        recourse.construct(model, props, cache, factors, SPEC)
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
        "method": "RQP-R: short Medium recourse in a High trust region",
        "strict_esm_online": True,
        "current_actual_tm_used_online": False,
        "same_configuration_for_all_loads": True,
        "training_bounds": list(final.TRAIN_BOUNDS),
        "selection_bounds": {
            "small": [318, 334],
            "validation": [334, 400],
        },
        "evaluation_bounds": list(EVALUATION_BOUNDS),
        "config": {
            "rqp_residual_ratio_quantile": final.QUANTILE,
            "rqp_robust_blend": final.BLEND,
            "rqp_high_weight": final.HIGH_WEIGHT,
            "rqp_steps": final.STEPS,
            "recourse_steps": SPEC["steps"],
            "recourse_high_budget": SPEC["high_budget"],
            "recourse_low_budget": SPEC["low_budget"],
            "recourse_kl_anchor": SPEC["anchor"],
        },
        "loads": {},
    }
    for load in (1, 2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        train_cache = base.runtime.build_policy_cache(
            model, props, *final.TRAIN_BOUNDS, batch_size=32
        )
        factors = final.fit_factors(train_cache)
        cache = base.runtime.build_policy_cache(
            model, props, *EVALUATION_BOUNDS, batch_size=32
        )
        baseline_rows, baseline_summary = base.trainer.evaluate(
            model, props, cache
        )
        candidate, rqp_active, recourse_active = recourse.construct(
            model, props, cache, factors, SPEC
        )
        candidate_rows, candidate_summary = base.runtime.evaluate_cache(
            model, props, candidate, None, batch_size=16
        )
        delta = base.delta(candidate_summary, baseline_summary)
        report["loads"][str(load)] = {
            "learned_residual_factors": int(
                sum(value.numel() // 8 for value in factors)
            ),
            "rqp_active": int(rqp_active.sum()),
            "recourse_active": int(recourse_active.sum()),
            "evaluation": {
                "baseline": base.compact(baseline_summary),
                "candidate": base.compact(candidate_summary),
                "delta": delta,
                "bootstrap": base.screen.method.large.paired_bootstrap(
                    baseline_rows, candidate_rows
                ),
                "rows": {
                    "baseline": final.compact_rows(baseline_rows),
                    "candidate": final.compact_rows(candidate_rows),
                },
            },
            "information_audit": information_audit(
                model, props, cache, factors
            ),
            "benchmark": {
                "batch_100": benchmark(model, props, cache, factors, 5),
                "batch_1": benchmark(
                    model,
                    props,
                    final.slice_cache(cache, 1),
                    factors,
                    3,
                ),
            },
        }
        print(
            f"[{load}x] active={int(rqp_active.sum())}/"
            f"{int(recourse_active.sum())} "
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
    path = HERE / "final_rqp_medium_recourse.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
