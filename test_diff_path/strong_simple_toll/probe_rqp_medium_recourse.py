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


final = load_module("rqp_recourse_final_base", HERE / "run_final_robust_priority.py")
priority, base = final.priority, final.base

LR = 0.05
ANCHOR = 0.005
HIGH_PENALTY = 2.0
LOW_PENALTY = 0.5


def recourse_correct(model, props, reference, steps, low_budget, anchor_weight):
    original = [value.squeeze(-1).detach() for value in reference.policies]
    with torch.no_grad():
        baseline = base.correction.predicted_fulfillment(
            model, props, reference, list(reference.policies)
        ).detach()
    logits = [
        torch.nn.Parameter(torch.log(value.clamp_min(1e-12)))
        for value in original
    ]
    optimizer = torch.optim.Adam(logits, lr=LR)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        routed = [
            base.correction.routed_policy(value, old)
            for value, old in zip(logits, original)
        ]
        fulfill = base.correction.predicted_fulfillment(
            model,
            props,
            reference,
            [value.unsqueeze(-1) for value in routed],
        )
        gain = fulfill - baseline
        anchor = sum(
            priority.universal.reverse_kl(value, old)
            for value, old in zip(routed, original)
        )
        objective = (
            gain[:, 1]
            + 0.10 * gain[:, 2]
            - HIGH_PENALTY * torch.relu(-gain[:, 0])
            - LOW_PENALTY * torch.relu(-float(low_budget) - gain[:, 2])
        ).sum() - float(anchor_weight) * anchor
        (-objective).backward()
        optimizer.step()
    with torch.no_grad():
        policies = tuple(
            base.correction.routed_policy(value, old).unsqueeze(-1)
            for value, old in zip(logits, original)
        )
    return replace(
        reference,
        policies=policies,
        path_features=torch.cat(
            [policies[1].squeeze(-1), policies[2].squeeze(-1)], dim=1
        ),
    )


def construct(model, props, cache, factors, spec):
    planning = replace(
        cache,
        predicted_tms=tuple(
            predicted * (1.0 + final.BLEND * (factor - 1.0))
            for predicted, factor in zip(cache.predicted_tms, factors)
        ),
    )
    rqp_raw = priority.priority_correct(
        model, props, planning, final.HIGH_WEIGHT, final.STEPS
    )
    rqp_guarded, rqp_active = priority.priority_guard(
        model,
        props,
        planning,
        rqp_raw,
        final.SECONDARY_BUDGET,
    )
    reference = replace(
        planning,
        policies=rqp_guarded.policies,
        path_features=rqp_guarded.path_features,
    )
    recourse_raw = recourse_correct(
        model,
        props,
        reference,
        spec["steps"],
        spec["low_budget"],
        spec.get("anchor", ANCHOR),
    )
    with torch.no_grad():
        old = base.correction.predicted_fulfillment(
            model, props, reference, list(reference.policies)
        )
        new = base.correction.predicted_fulfillment(
            model, props, recourse_raw, list(recourse_raw.policies)
        )
        gain = new - old
        recourse_active = (
            rqp_active
            & (gain[:, 0] >= -float(spec["high_budget"]))
            & (gain[:, 1] >= 0.0001)
            & (gain[:, 2] >= -float(spec["low_budget"]))
        )
        policies = tuple(
            torch.where(recourse_active[:, None, None], new_policy, old_policy)
            for new_policy, old_policy in zip(
                recourse_raw.policies, reference.policies
            )
        )
    candidate = replace(
        cache,
        policies=policies,
        path_features=torch.cat(
            [policies[1].squeeze(-1), policies[2].squeeze(-1)], dim=1
        ),
    )
    return candidate, rqp_active, recourse_active


def evaluate(model, props, cache, factors, baseline_summary, spec):
    candidate, rqp_active, recourse_active = construct(
        model, props, cache, factors, spec
    )
    _, summary = base.runtime.evaluate_cache(
        model, props, candidate, None, batch_size=16
    )
    return {
        **spec,
        "rqp_active": int(rqp_active.sum()),
        "recourse_active": int(recourse_active.sum()),
        "summary": base.compact(summary),
        "delta": base.delta(summary, baseline_summary),
    }


def show(stage, load, value):
    d = value["delta"]
    print(
        f"[{stage} {load}x] s={value['steps']} "
        f"hb={value['high_budget']:.4f} lb={value['low_budget']:.3f} "
        f"a={value.get('anchor', ANCHOR):.4f} "
        f"active={value['rqp_active']}/{value['recourse_active']} "
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


def rank(group):
    by_load = {value["load"]: value for value in group}
    d2 = by_load[2]["delta"]
    d3 = by_load[3]["delta"]
    high_floor = min(
        d3["High.norm_fulfill_mean"],
        d3["High.norm_fulfill_p1"],
        d3["High.norm_fulfill_p10"],
    )
    low_floor = min(
        d2["Low.norm_fulfill_mean"],
        d2["Low.norm_fulfill_p1"],
        d2["Low.norm_fulfill_p10"],
        d3["Low.norm_fulfill_mean"],
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
    return high_floor >= 0.011, low_floor >= -0.005, medium_score, high_floor


def run_bounds(bounds, specs, stage):
    grouped = {tuple(sorted(spec.items())): [] for spec in specs}
    baselines = {}
    for load in (2, 3):
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
        (
            {"spec": dict(key), "loads": values, "score": rank(values)}
            for key, values in grouped.items()
        ),
        key=lambda value: value["score"],
        reverse=True,
    )
    return baselines, ranked


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    specs = [
        {
            "steps": steps,
            "high_budget": 0.0002,
            "low_budget": 0.002,
            "anchor": anchor,
        }
        for steps, anchor in itertools.product(
            (8, 12, 16, 24),
            (0.0, 0.0005, 0.001),
        )
    ]
    small_baselines, small = run_bounds((318, 334), specs, "small")
    frozen = [value["spec"] for value in small[:4]]
    validation_baselines, validation = run_bounds(
        (334, 400), frozen, "validation"
    )
    path = HERE / "rqp_medium_recourse.json"
    path.write_text(
        json.dumps(
            {
                "method": "RQP plus short Medium recourse in a High trust region",
                "strict_esm_online": True,
                "current_actual_tm_used_online": False,
                "training_bounds": [0, 318],
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
    print(
        f"BEST={validation[0]['spec']} score={validation[0]['score']}",
        flush=True,
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
