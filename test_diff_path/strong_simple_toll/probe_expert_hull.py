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


base = load_module("expert_hull_base", HERE / "probe_all_class_pareto.py")
MARGINS = (0.0, 0.002, 0.005)


def train_expert(model, props, cache, weights):
    original = [value.squeeze(-1).detach() for value in cache.policies]
    logits = [
        torch.nn.Parameter(torch.log(value.clamp_min(1e-12)))
        for value in original
    ]
    optimizer = torch.optim.Adam(logits, lr=0.06)
    weight = torch.tensor(weights, device=original[0].device)
    for _ in range(32):
        optimizer.zero_grad(set_to_none=True)
        routed = [
            base.base.correction.routed_policy(value, old)
            for value, old in zip(logits, original)
        ]
        fulfill = base.base.correction.predicted_fulfillment(
            model,
            props,
            cache,
            [value.unsqueeze(-1) for value in routed],
        )
        anchor = sum(
            base.reverse_kl(value, old)
            for value, old in zip(routed, original)
        )
        objective = (fulfill * weight[None, :]).sum() - 0.005 * anchor
        (-objective).backward()
        optimizer.step()
    with torch.no_grad():
        routed = [
            base.base.correction.routed_policy(value, old)
            for value, old in zip(logits, original)
        ]
    return routed


def simplex_grid(parts=10):
    for values in itertools.product(range(parts + 1), repeat=3):
        used = sum(values)
        if used <= parts:
            yield (parts - used, *values)


def choose_hull(model, props, cache, experts, margin):
    # experts: base, High, Medium, Low; policies exclude trailing singleton dim.
    with torch.no_grad():
        baseline = base.base.correction.predicted_fulfillment(
            model, props, cache, list(cache.policies)
        )
        best_medium = baseline[:, 1].clone()
        chosen = [value.squeeze(-1).detach().clone() for value in cache.policies]
        chosen_weights = torch.zeros(
            (len(baseline), 4), device=baseline.device, dtype=baseline.dtype
        )
        chosen_weights[:, 0] = 1.0
        for integer_weights in simplex_grid(10):
            weight = torch.tensor(
                integer_weights, device=baseline.device, dtype=baseline.dtype
            ) / 10.0
            policies = [
                sum(weight[index] * experts[index][cls] for index in range(4))
                for cls in range(3)
            ]
            fulfill = base.base.correction.predicted_fulfillment(
                model,
                props,
                cache,
                [value.unsqueeze(-1) for value in policies],
            )
            feasible = (
                (fulfill[:, 0] >= baseline[:, 0] + margin)
                & (fulfill[:, 2] >= baseline[:, 2] + margin)
                & (fulfill[:, 1] > best_medium)
            )
            best_medium = torch.where(feasible, fulfill[:, 1], best_medium)
            chosen_weights = torch.where(
                feasible[:, None], weight[None, :].expand_as(chosen_weights), chosen_weights
            )
            for cls in range(3):
                chosen[cls][feasible] = policies[cls][feasible]
    return replace(
        cache,
        policies=tuple(value.unsqueeze(-1) for value in chosen),
        path_features=torch.cat([chosen[1], chosen[2]], dim=1),
    ), chosen_weights


def main():
    base.base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "purpose": "actual-TM convex hull of four simple routing experts",
        "deployable": False,
        "loads": {},
    }
    for load in (2, 3):
        model, props, _ = base.base.screen.load_backbone(load, device)
        cache = base.base.runtime.build_policy_cache(
            model, props, 400, 500, batch_size=32
        )
        _, baseline_summary = base.base.trainer.evaluate(model, props, cache)
        oracle = replace(cache, predicted_tms=cache.tms)
        experts = [[value.squeeze(-1).detach() for value in cache.policies]]
        experts.append(train_expert(model, props, oracle, (1.0, 0.0, 0.0)))
        experts.append(train_expert(model, props, oracle, (0.0, 1.0, 0.0)))
        experts.append(train_expert(model, props, oracle, (0.0, 0.0, 1.0)))
        trials = []
        for margin in MARGINS:
            selected, weights = choose_hull(model, props, oracle, experts, margin)
            candidate = replace(
                cache,
                policies=selected.policies,
                path_features=selected.path_features,
            )
            _, summary = base.evaluate_all(model, props, candidate)
            value = {
                "margin": margin,
                "changed_fraction": float((weights[:, 0] < 1).float().mean()),
                "mean_weights_base_high_medium_low": [
                    float(value) for value in weights.mean(dim=0)
                ],
                "summary": base.base.compact(summary),
                "delta": base.base.delta(summary, baseline_summary),
            }
            trials.append(value)
            d = value["delta"]
            print(
                f"[{load}x] margin={margin:g} changed={value['changed_fraction']:.2f} "
                f"H={d['High.norm_fulfill_mean']:+.6f}/"
                f"{d['High.norm_fulfill_p1']:+.6f}/"
                f"{d['High.norm_fulfill_p10']:+.6f} "
                f"M={d['Medium.norm_fulfill_mean']:+.6f}/"
                f"{d['Medium.norm_fulfill_p1']:+.6f}/"
                f"{d['Medium.norm_fulfill_p10']:+.6f} "
                f"L={d['Low.norm_fulfill_mean']:+.6f}/"
                f"{d['Low.norm_fulfill_p1']:+.6f}/"
                f"{d['Low.norm_fulfill_p10']:+.6f}",
                flush=True,
            )
        report["loads"][str(load)] = {
            "baseline": base.base.compact(baseline_summary),
            "trials": trials,
        }
    path = HERE / "expert_hull.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
