from __future__ import annotations

from dataclasses import replace
import importlib.util
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


base = load_module("all_class_pareto_base", HERE / "run_tradeoff_screen.py")
HIGH_WEIGHTS = (1.0, 2.0, 4.0, 8.0, 16.0)
LOW_WEIGHTS = (0.05, 0.10, 0.20, 0.30)


def reverse_kl(current, original):
    current_grouped = current.reshape(len(current), -1, 8)
    original_grouped = original.reshape(len(original), -1, 8)
    return (
        current_grouped
        * torch.log((current_grouped + 1e-12) / (original_grouped + 1e-12))
    ).sum(dim=-1).mean(dim=1).sum()


def correct_all(model, props, cache, high_weight, low_weight):
    base_policies = [value.squeeze(-1).detach() for value in cache.policies]
    logits = [
        torch.nn.Parameter(torch.log(value.clamp_min(1e-12)))
        for value in base_policies
    ]
    optimizer = torch.optim.Adam(logits, lr=0.06)
    for _ in range(24):
        optimizer.zero_grad(set_to_none=True)
        routed = [
            base.correction.routed_policy(value, original)
            for value, original in zip(logits, base_policies)
        ]
        fulfill = base.correction.predicted_fulfillment(
            model,
            props,
            cache,
            [value.unsqueeze(-1) for value in routed],
        )
        anchor = sum(
            reverse_kl(value, original)
            for value, original in zip(routed, base_policies)
        )
        objective = (
            high_weight * fulfill[:, 0].sum()
            + fulfill[:, 1].sum()
            + low_weight * fulfill[:, 2].sum()
            - 0.01 * anchor
        )
        (-objective).backward()
        optimizer.step()
    with torch.no_grad():
        routed = [
            base.correction.routed_policy(value, original)
            for value, original in zip(logits, base_policies)
        ]
    return replace(
        cache,
        policies=tuple(value.unsqueeze(-1) for value in routed),
        path_features=torch.cat([routed[1], routed[2]], dim=1),
    )


class AllClassAdapter:
    def adapt_batch(self, policies, batch):
        # The cache's three policy tensors already contain the candidate.
        return policies


def evaluate_all(model, props, cache):
    return base.runtime.evaluate_cache(
        model, props, cache, AllClassAdapter(), batch_size=16
    )


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "purpose": "all-class actual-TM Pareto feasibility diagnostic",
        "deployable": False,
        "loads": {},
    }
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
        _, baseline = base.trainer.evaluate(model, props, cache)
        oracle_cache = replace(cache, predicted_tms=cache.tms)
        trials = []
        for high_weight in HIGH_WEIGHTS:
            for low_weight in LOW_WEIGHTS:
                candidate = correct_all(
                    model, props, oracle_cache, high_weight, low_weight
                )
                # Restore actual-TM cache fields while retaining all three policies.
                candidate = replace(
                    cache,
                    policies=candidate.policies,
                    path_features=candidate.path_features,
                )
                _, summary = evaluate_all(model, props, candidate)
                value = {
                    "high_weight": high_weight,
                    "low_weight": low_weight,
                    "summary": base.compact(summary),
                    "delta": base.delta(summary, baseline),
                }
                trials.append(value)
                d = value["delta"]
                print(
                    f"[{load}x] h={high_weight:g} l={low_weight:g} "
                    f"H={d['High.norm_fulfill_mean']:+.6f}/"
                    f"{d['High.norm_fulfill_p1']:+.6f}/"
                    f"{d['High.norm_fulfill_p10']:+.6f} "
                    f"M={d['Medium.norm_fulfill_mean']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p1']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p10']:+.6f} "
                    f"L={d['Low.norm_fulfill_mean']:+.6f}",
                    flush=True,
                )
        report["loads"][str(load)] = {
            "baseline": base.compact(baseline),
            "trials": trials,
        }
    path = HERE / "all_class_pareto.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
