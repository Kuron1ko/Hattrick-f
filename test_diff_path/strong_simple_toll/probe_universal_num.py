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


base = load_module("universal_num_base", HERE / "run_tradeoff_screen.py")
CONFIGS = (
    ("proportional_fair", 8, 0.04, 0.02),
    ("proportional_fair", 16, 0.04, 0.02),
    ("relative_softmin_0.01", 8, 0.04, 0.02),
    ("relative_softmin_0.01", 16, 0.04, 0.02),
    ("relative_softmin_0.02", 8, 0.04, 0.02),
    ("relative_softmin_0.02", 16, 0.04, 0.02),
    ("relative_softmin_0.05", 16, 0.04, 0.02),
    ("relative_softmin_0.02", 16, 0.08, 0.02),
)


def reverse_kl(current, original):
    current_grouped = current.reshape(len(current), -1, 8)
    original_grouped = original.reshape(len(original), -1, 8)
    return (
        current_grouped
        * torch.log((current_grouped + 1e-12) / (original_grouped + 1e-12))
    ).sum(dim=-1).mean()


def utility(name, fulfill, baseline):
    if name == "proportional_fair":
        return torch.log(fulfill.clamp_min(1e-6)).sum(dim=1).sum()
    if name.startswith("relative_softmin_"):
        tau = float(name.rsplit("_", 1)[1])
        gain = fulfill - baseline
        soft_min = -tau * torch.logsumexp(-gain / tau, dim=1)
        # The small mean term distinguishes equally fair directions.
        return (soft_min + 0.10 * gain.mean(dim=1)).sum()
    raise KeyError(name)


def correct(model, props, cache, config):
    name, steps, learning_rate, anchor_weight = config
    original = [value.squeeze(-1).detach() for value in cache.policies]
    with torch.no_grad():
        baseline = base.correction.predicted_fulfillment(
            model, props, cache, list(cache.policies)
        ).detach()
    logits = [
        torch.nn.Parameter(torch.log(value.clamp_min(1e-12)))
        for value in original
    ]
    optimizer = torch.optim.Adam(logits, lr=learning_rate)
    for _ in range(steps):
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
        anchor = sum(reverse_kl(value, old) for value, old in zip(routed, original))
        objective = utility(name, fulfill, baseline) - anchor_weight * anchor
        (-objective).backward()
        optimizer.step()
    with torch.no_grad():
        routed = [
            base.correction.routed_policy(value, old)
            for value, old in zip(logits, original)
        ]
    return replace(
        cache,
        policies=tuple(value.unsqueeze(-1) for value in routed),
        path_features=torch.cat(routed[1:], dim=1),
    )


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props, _ = base.screen.load_backbone(3, device)
    report = {
        "method_family": "class-symmetric network utility maximization",
        "strict_esm": True,
        "splits": {},
    }
    for split, bounds in {"small": (350, 366), "validation": (350, 400), "evaluation": (400, 500)}.items():
        cache = base.runtime.build_policy_cache(model, props, *bounds, batch_size=32)
        _, baseline = base.trainer.evaluate(model, props, cache)
        predicted_baseline = base.correction.predicted_fulfillment(
            model, props, cache, list(cache.policies)
        ).detach()
        trials = []
        for config in CONFIGS:
            candidate = correct(model, props, cache, config)
            _, summary = base.runtime.evaluate_cache(
                model, props, candidate, None, batch_size=16
            )
            value = {
                "config": list(config),
                "summary": base.compact(summary),
                "delta": base.delta(summary, baseline),
            }
            trials.append(value)
            d = value["delta"]
            print(
                f"[{split}] {config[0]} s={config[1]} lr={config[2]:g} "
                f"H={d['High.norm_fulfill_mean']:+.5f}/"
                f"{d['High.norm_fulfill_p1']:+.5f}/"
                f"{d['High.norm_fulfill_p10']:+.5f} "
                f"M={d['Medium.norm_fulfill_mean']:+.5f}/"
                f"{d['Medium.norm_fulfill_p1']:+.5f}/"
                f"{d['Medium.norm_fulfill_p10']:+.5f} "
                f"L={d['Low.norm_fulfill_mean']:+.5f}",
                flush=True,
            )
        report["splits"][split] = {
            "bounds": list(bounds),
            "predicted_baseline_mean": [
                float(value) for value in predicted_baseline.mean(dim=0)
            ],
            "baseline": base.compact(baseline),
            "trials": trials,
        }
    path = HERE / "universal_num.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
