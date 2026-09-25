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


final = load_module("rqp_slack_weight_final_base", HERE / "run_final_robust_priority.py")
priority, base = final.priority, final.base

TEMPERATURE = 0.02
HIGH_BASE = 0.88
HIGH_TRANSFER = 0.04
MEDIUM_MAX = 0.10


def adaptive_correct(
    model,
    props,
    cache,
    threshold,
    anchor_weight=priority.ANCHOR,
):
    original = [value.squeeze(-1).detach() for value in cache.policies]
    with torch.no_grad():
        baseline = base.correction.predicted_fulfillment(
            model, props, cache, list(cache.policies)
        ).detach()
        medium_gate = torch.sigmoid(
            (baseline[:, 0] - float(threshold)) / TEMPERATURE
        )
        high_weight = HIGH_BASE - HIGH_TRANSFER * medium_gate
        medium_weight = MEDIUM_MAX * medium_gate
    logits = [
        torch.nn.Parameter(torch.log(value.clamp_min(1e-12)))
        for value in original
    ]
    optimizer = torch.optim.Adam(logits, lr=priority.LR)
    for _ in range(final.STEPS):
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
        soft_min = -priority.TAU * torch.logsumexp(
            -gain / priority.TAU, dim=1
        )
        anchor = sum(
            priority.universal.reverse_kl(value, old)
            for value, old in zip(routed, original)
        )
        objective = (
            soft_min
            + high_weight * gain[:, 0]
            + medium_weight * gain[:, 1]
            + 0.05 * gain.mean(dim=1)
        ).sum() - float(anchor_weight) * anchor
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
    ), medium_gate


def construct(model, props, cache, factors, spec):
    planning = replace(
        cache,
        predicted_tms=tuple(
            predicted * (1.0 + final.BLEND * (factor - 1.0))
            for predicted, factor in zip(cache.predicted_tms, factors)
        ),
    )
    raw, medium_gate = adaptive_correct(
        model,
        props,
        planning,
        spec["threshold"],
        spec.get("anchor_weight", priority.ANCHOR),
    )
    with torch.no_grad():
        old = base.correction.predicted_fulfillment(
            model, props, planning, list(planning.policies)
        )
        new = base.correction.predicted_fulfillment(
            model, props, raw, list(raw.policies)
        )
        gain = new - old
        active = (
            (gain[:, 0] >= 0.0001)
            & (gain[:, 1] >= -0.02)
            & (gain[:, 2] >= float(spec["low_floor"]))
        )
        policies = tuple(
            torch.where(active[:, None, None], new_policy, old_policy)
            for new_policy, old_policy in zip(raw.policies, planning.policies)
        )
    return replace(
        cache,
        policies=policies,
        path_features=torch.cat(
            [policies[1].squeeze(-1), policies[2].squeeze(-1)], dim=1
        ),
    ), active, medium_gate


def evaluate(model, props, cache, factors, baseline_summary, spec):
    candidate, active, gate = construct(model, props, cache, factors, spec)
    _, summary = base.runtime.evaluate_cache(
        model, props, candidate, None, batch_size=16
    )
    return {
        **spec,
        "active": int(active.sum()),
        "medium_gate_mean": float(gate.mean()),
        "medium_gate_p10": float(torch.quantile(gate, 0.10)),
        "medium_gate_p90": float(torch.quantile(gate, 0.90)),
        "summary": base.compact(summary),
        "delta": base.delta(summary, baseline_summary),
    }


def show(stage, load, value):
    d = value["delta"]
    print(
        f"[{stage} {load}x] t={value['threshold']:.2f} "
        f"lf={value['low_floor']:+.3f} gate={value['medium_gate_mean']:.3f} "
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


def score(group):
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
            {"spec": dict(key), "loads": values, "score": score(values)}
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
        {"threshold": threshold, "low_floor": low_floor}
        for threshold, low_floor in itertools.product(
            (0.75, 0.80, 0.85, 0.90, 0.95, 0.98),
            (-0.02, -0.005),
        )
    ]
    small_baselines, small = run_bounds((318, 334), specs, "slack-small")
    frozen = [value["spec"] for value in small[:6]]
    validation_baselines, validation = run_bounds(
        (334, 400), frozen, "slack-validation"
    )
    path = HERE / "rqp_slack_weight.json"
    path.write_text(
        json.dumps(
            {
                "method": "single-stage ESM High-slack adaptive RQP",
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
    print(f"BEST={validation[0]['spec']} score={validation[0]['score']}")
    print(path, flush=True)


if __name__ == "__main__":
    main()
