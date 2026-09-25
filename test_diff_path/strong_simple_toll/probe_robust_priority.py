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


robust = load_module("robust_priority_base", HERE / "probe_robust_upareto.py")
common = robust.common
base, cal = common.base, common.cal
universal = common.universal

KINDS = ("additive", "multiplicative")
QUANTILES = (0.60, 0.75)
BLENDS = (0.50, 0.75)
HIGH_WEIGHTS = (0.25, 0.50, 1.00)
STEPS = (24, 32)
SECONDARY_BUDGETS = (0.001, 0.003)
TAU = 0.002
LR = 0.08
ANCHOR = 0.01
METRICS = common.METRICS


def priority_correct(model, props, cache, high_weight, steps):
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
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        routed = [
            base.correction.routed_policy(value, old)
            for value, old in zip(logits, original)
        ]
        fulfill = base.correction.predicted_fulfillment(
            model, props, cache,
            [value.unsqueeze(-1) for value in routed],
        )
        gain = fulfill - baseline
        soft_min = -TAU * torch.logsumexp(-gain / TAU, dim=1)
        anchor = sum(
            universal.reverse_kl(value, old)
            for value, old in zip(routed, original)
        )
        objective = (
            soft_min
            + float(high_weight) * gain[:, 0]
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


def priority_guard(model, props, baseline, candidate, secondary_budget):
    with torch.no_grad():
        old = base.correction.predicted_fulfillment(
            model, props, baseline, list(baseline.policies)
        )
        new = base.correction.predicted_fulfillment(
            model, props, candidate, list(candidate.policies)
        )
        gain = new - old
        active = (
            (gain[:, 0] >= 0.0001)
            & (gain[:, 1] >= -secondary_budget)
            & (gain[:, 2] >= -secondary_budget)
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


def evaluate(model, props, cache, planning, baseline, spec):
    raw = priority_correct(
        model, props, planning, spec["high_weight"], spec["steps"]
    )
    guarded, active = priority_guard(
        model, props, planning, raw, spec["secondary_budget"]
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
    }


def rank(value):
    d = value["delta"]
    high_floor = common.metric_floor(d, "High")
    secondary_floor = min(
        common.metric_floor(d, "Medium"), common.metric_floor(d, "Low")
    )
    return (
        value["safe_secondary"],
        high_floor > 0.0,
        high_floor + float(d["High.norm_fulfill_mean"])
        + 0.05 * secondary_floor,
    )


def show(stage, value):
    d = value["delta"]
    print(
        f"[{stage}] {value['kind']} q={value['quantile']:.2f} "
        f"b={value['blend']:.2f} wh={value['high_weight']:.2f} "
        f"s={value['steps']} eps={value['secondary_budget']:.3f} "
        f"active={value['active']} safe={value['safe_secondary']} "
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


def spec_only(value):
    return {
        key: value[key]
        for key in (
            "kind", "quantile", "blend", "high_weight", "steps",
            "secondary_budget"
        )
    }


def run_stage(model, props, cache, train_x, train_y, widths, specs, stage):
    _, baseline = base.trainer.evaluate(model, props, cache)
    raw_tm, _, _ = cal.arrays(cache)
    corrected = {
        (kind, quantile): robust.robust_prediction(
            kind, quantile, train_x, train_y, raw_tm
        )
        for kind in KINDS for quantile in QUANTILES
    }
    plans = {
        (kind, quantile, blend): robust.planning_cache(
            cache, raw_tm, corrected[(kind, quantile)], widths, blend
        )
        for kind in KINDS for quantile in QUANTILES for blend in BLENDS
    }
    values = []
    for spec in specs:
        value = evaluate(
            model, props, cache,
            plans[(spec["kind"], spec["quantile"], spec["blend"])],
            baseline, spec
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
            "high_weight": high_weight,
            "steps": steps,
            "secondary_budget": budget,
        }
        for kind, quantile, blend, high_weight, steps, budget in itertools.product(
            KINDS, QUANTILES, BLENDS, HIGH_WEIGHTS, STEPS, SECONDARY_BUDGETS
        )
    ]
    small_cache = base.runtime.build_policy_cache(
        model, props, 318, 334, batch_size=16
    )
    small_baseline, small = run_stage(
        model, props, small_cache, train_x, train_y, widths, specs, "small"
    )
    small.sort(key=rank, reverse=True)
    validation_specs = [spec_only(value) for value in small[:20]]
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
        "method": "robust lexicographic relative-gain routing",
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
    path = HERE / "robust_priority.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
