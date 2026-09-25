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


v1 = load_module("low_recovery_v1", HERE / "probe_global_path_bias.py")
v2 = load_module("low_recovery_v2", HERE / "probe_global_path_bias_v2.py")
plus = load_module("low_recovery_plus", HERE / "probe_bias_plus_online.py")
base = v1.base
STEPS = (0, 4, 8, 16, 32)


def policies_from_features(cache, path_features):
    path_count = cache.policies[0].shape[1]
    return [
        cache.policies[0].squeeze(-1).detach(),
        path_features[:, :path_count].detach(),
        path_features[:, path_count:].detach(),
    ]


def recover_low(model, props, cache, fixed, low_start, steps):
    if steps == 0:
        routed = [fixed[0], fixed[1], low_start]
    else:
        logits = torch.nn.Parameter(torch.log(low_start.clamp_min(1e-12)))
        optimizer = torch.optim.Adam([logits], lr=0.08)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            low = base.correction.routed_policy(logits, low_start)
            policies = [fixed[0], fixed[1], low]
            fulfill = base.correction.predicted_fulfillment(
                model,
                props,
                cache,
                [value.unsqueeze(-1) for value in policies],
            )
            anchor = v1.reverse_kl(low, low_start)
            objective = fulfill[:, 2].sum() - 0.01 * anchor
            (-objective).backward()
            optimizer.step()
        with torch.no_grad():
            low = base.correction.routed_policy(logits, low_start)
        routed = [fixed[0], fixed[1], low]
    return replace(
        cache,
        policies=tuple(value.unsqueeze(-1) for value in routed),
        path_features=torch.cat(routed[1:], dim=1),
    )


def build_target(load, model, props, train_cache, cache):
    if load == 2:
        variant = {"name": "mean", "tail": 0.0, "low_weight": 0.24, "low_floor": 0.0}
        medium_bias, low_bias = v1.train_bias(model, props, train_cache, variant)
        biased = plus.biased_cache(cache, medium_bias, low_bias, 1.0)
        target = biased
    else:
        variant = {
            "name": "tail8_q0.05",
            "tail": 8.0,
            "fraction": 0.05,
            "low_weight": 0.24,
            "low_floor": 0.0,
        }
        medium_bias, low_bias = v2.train_bias(model, props, train_cache, variant)
        biased = plus.biased_cache(cache, medium_bias, low_bias, 0.9)
        refined = base.correction.correct_cache(
            model, props, biased, 8, 0.08, 0.20, 0.01
        )
        target = replace(biased, path_features=refined.path_features)
    fixed = policies_from_features(cache, target.path_features)
    starts = {
        "original": cache.policies[2].squeeze(-1).detach(),
        "biased": biased.policies[2].squeeze(-1).detach(),
        "candidate": fixed[2],
    }
    return fixed, starts


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "method": "priority-safe Low-only residual recovery",
        "strict_esm_inference": True,
        "loads": {},
    }
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        train_cache = base.runtime.build_policy_cache(model, props, 0, 400, batch_size=32)
        cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
        _, baseline = base.trainer.evaluate(model, props, cache)
        fixed, starts = build_target(load, model, props, train_cache, cache)
        trials = []
        for start_name, low_start in starts.items():
            for steps in STEPS:
                candidate = recover_low(
                    model, props, cache, fixed, low_start, steps
                )
                _, summary = base.runtime.evaluate_cache(
                    model, props, candidate, None, batch_size=16
                )
                value = {
                    "start": start_name,
                    "steps": steps,
                    "summary": base.compact(summary),
                    "delta": base.delta(summary, baseline),
                }
                trials.append(value)
                d = value["delta"]
                print(
                    f"[{load}x] {start_name} s={steps} "
                    f"M={d['Medium.norm_fulfill_mean']:+.5f}/"
                    f"{d['Medium.norm_fulfill_p1']:+.5f}/"
                    f"{d['Medium.norm_fulfill_p10']:+.5f} "
                    f"L={d['Low.norm_fulfill_mean']:+.5f}/"
                    f"{d['Low.norm_fulfill_p1']:+.5f}/"
                    f"{d['Low.norm_fulfill_p10']:+.5f}",
                    flush=True,
                )
        report["loads"][str(load)] = {"trials": trials}
    path = HERE / "low_recovery.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
