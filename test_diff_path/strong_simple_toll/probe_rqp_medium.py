from __future__ import annotations

from dataclasses import replace
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


final = load_module("rqp_medium_final_base", HERE / "run_final_robust_priority.py")
priority, base = final.priority, final.base

TAU = priority.TAU
LR = priority.LR
ANCHOR = priority.ANCHOR
LOW_FLOOR = -0.02
STEPS = 40


def medium_correct(model, props, cache, high_weight, medium_weight):
    original = [value.squeeze(-1).detach() for value in cache.policies]
    with torch.no_grad():
        baseline = base.correction.predicted_fulfillment(
            model, props, cache, list(cache.policies)
        ).detach()
    logits = [
        torch.nn.Parameter(torch.log(value.clamp_min(1e-12)))
        for value in original
    ]
    optimizer = torch.optim.Adam(logits, lr=LR)
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        routed = [
            base.correction.routed_policy(value, old)
            for value, old in zip(logits, original)
        ]
        fulfill = base.correction.predicted_fulfillment(
            model,
            props,
            cache,
            [value.unsqueeze(-1) for value in routed],
        )
        gain = fulfill - baseline
        soft_min = -TAU * torch.logsumexp(-gain / TAU, dim=1)
        anchor = sum(
            priority.universal.reverse_kl(value, old)
            for value, old in zip(routed, original)
        )
        objective = (
            soft_min
            + float(high_weight) * gain[:, 0]
            + float(medium_weight) * gain[:, 1]
            + 0.05 * gain.mean(dim=1)
        ).sum() - ANCHOR * anchor
        (-objective).backward()
        optimizer.step()
    with torch.no_grad():
        policies = tuple(
            base.correction.routed_policy(value, old).unsqueeze(-1)
            for value, old in zip(logits, original)
        )
    return replace(
        cache,
        policies=policies,
        path_features=torch.cat(
            [policies[1].squeeze(-1), policies[2].squeeze(-1)], dim=1
        ),
    )


def guard(
    model,
    props,
    baseline,
    candidate,
    medium_floor,
    low_floor=LOW_FLOOR,
    high_floor=0.0001,
    high_slack_fraction=0.0,
):
    with torch.no_grad():
        old = base.correction.predicted_fulfillment(
            model, props, baseline, list(baseline.policies)
        )
        new = base.correction.predicted_fulfillment(
            model, props, candidate, list(candidate.policies)
        )
        gain = new - old
        required_high = torch.clamp_min(
            float(high_slack_fraction) * (1.0 - old[:, 0]),
            float(high_floor),
        )
        active = (
            (gain[:, 0] >= required_high)
            & (gain[:, 1] >= medium_floor)
            & (gain[:, 2] >= low_floor)
        )
        policies = tuple(
            torch.where(active[:, None, None], new_policy, old_policy)
            for new_policy, old_policy in zip(candidate.policies, baseline.policies)
        )
    return replace(
        baseline,
        policies=policies,
        path_features=torch.cat(
            [policies[1].squeeze(-1), policies[2].squeeze(-1)], dim=1
        ),
    ), active


def construct(model, props, cache, factors, spec):
    planning = replace(
        cache,
        predicted_tms=tuple(
            predicted * (1.0 + final.BLEND * (factor - 1.0))
            for predicted, factor in zip(cache.predicted_tms, factors)
        ),
    )
    raw = medium_correct(
        model,
        props,
        planning,
        spec["high_weight"],
        spec["medium_weight"],
    )
    guarded, active = guard(
        model,
        props,
        planning,
        raw,
        spec["medium_floor"],
        spec.get("low_floor", LOW_FLOOR),
        spec.get("high_floor", 0.0001),
        spec.get("high_slack_fraction", 0.0),
    )
    return replace(
        cache,
        policies=guarded.policies,
        path_features=guarded.path_features,
    ), active


def evaluate(model, props, cache, factors, baseline_summary, spec):
    candidate, active = construct(model, props, cache, factors, spec)
    _, summary = base.runtime.evaluate_cache(
        model, props, candidate, None, batch_size=16
    )
    return {
        **spec,
        "active": int(active.sum()),
        "summary": base.compact(summary),
        "delta": base.delta(summary, baseline_summary),
    }


def show(stage, load, value):
    d = value["delta"]
    print(
        f"[{stage} {load}x] wh={value['high_weight']:.2f} "
        f"wm={value['medium_weight']:.2f} mf={value['medium_floor']:+.4f} "
        f"active={value['active']} "
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


def score(pair):
    by_load = {value["load"]: value for value in pair}
    d2 = by_load[2]["delta"]
    d3 = by_load[3]["delta"]
    high_floor = min(
        d3["High.norm_fulfill_mean"],
        d3["High.norm_fulfill_p1"],
        d3["High.norm_fulfill_p10"],
    )
    secondary_floor = min(
        d2["Low.norm_fulfill_p1"],
        d2["Low.norm_fulfill_p10"],
        d3["Low.norm_fulfill_p1"],
        d3["Low.norm_fulfill_p10"],
    )
    medium_score = sum(
        d[f"Medium.{metric}"]
        for d in (d2, d3)
        for metric in (
            "norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10"
        )
    )
    return (
        high_floor >= 0.005,
        secondary_floor >= -0.005,
        medium_score,
        high_floor,
    )


def run_bounds(loads, bounds, specs, stage):
    grouped = {tuple(sorted(spec.items())): [] for spec in specs}
    baselines = {}
    for load in loads:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, props, _ = base.screen.load_backbone(load, device)
        train_cache = base.runtime.build_policy_cache(
            model, props, *final.TRAIN_BOUNDS, batch_size=32
        )
        factors = final.fit_factors(train_cache)
        cache = base.runtime.build_policy_cache(
            model, props, *bounds, batch_size=32
        )
        _, baseline = base.trainer.evaluate(model, props, cache)
        baselines[str(load)] = base.compact(baseline)
        for spec in specs:
            value = evaluate(model, props, cache, factors, baseline, spec)
            value["load"] = load
            grouped[tuple(sorted(spec.items()))].append(value)
            show(stage, load, value)
    ranked = sorted(
        ({"spec": dict(key), "loads": values, "score": score(values)}
         for key, values in grouped.items()),
        key=lambda value: value["score"],
        reverse=True,
    )
    return baselines, ranked


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    specs = [
        {
            "high_weight": high_weight,
            "medium_weight": medium_weight,
            "medium_floor": medium_floor,
        }
        for high_weight, medium_weight, medium_floor in itertools.product(
            (0.84, 0.88, 0.92),
            (0.00, 0.10, 0.20, 0.30),
            (-0.02, 0.0),
        )
    ]
    small_baselines, small = run_bounds(
        (2, 3), (318, 334), specs, "small"
    )
    frozen = [value["spec"] for value in small[:6]]
    validation_baselines, validation = run_bounds(
        (2, 3), (334, 400), frozen, "validation"
    )
    path = HERE / "rqp_medium_probe.json"
    path.write_text(
        json.dumps(
            {
                "method": "RQP with one Medium marginal-gain term",
                "strict_esm_online": True,
                "current_actual_tm_used_online": False,
                "training_bounds": list(final.TRAIN_BOUNDS),
                "small_bounds": [318, 334],
                "validation_bounds": [334, 400],
                "small_baselines": small_baselines,
                "small_ranked": small,
                "frozen_specs": frozen,
                "validation_baselines": validation_baselines,
                "validation_ranked": validation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"BEST={validation[0]['spec']} score={validation[0]['score']}")
    print(path, flush=True)


if __name__ == "__main__":
    main()
