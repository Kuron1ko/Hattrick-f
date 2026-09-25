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


base_v1 = load_module("global_path_bias_v2_base", HERE / "probe_global_path_bias.py")
base = base_v1.base
VARIANTS = tuple(
    {
        "name": f"tail{tail:g}_q{fraction:g}",
        "tail": tail,
        "fraction": fraction,
        "low_weight": 0.24,
        "low_floor": 0.0,
    }
    for fraction in (0.05, 0.10, 0.20)
    for tail in (2.0, 4.0, 8.0)
) + (
    {
        "name": "tail4_q0.1_low05",
        "tail": 4.0,
        "fraction": 0.10,
        "low_weight": 0.24,
        "low_floor": 0.5,
    },
)


def train_bias(model, props, cache, variant):
    oracle = replace(cache, predicted_tms=cache.tms)
    original = [value.squeeze(-1).detach() for value in cache.policies]
    medium_bias = torch.nn.Parameter(torch.zeros_like(original[1][:1]))
    low_bias = torch.nn.Parameter(torch.zeros_like(original[2][:1]))
    with torch.no_grad():
        baseline = base.correction.predicted_fulfillment(
            model, props, oracle, list(oracle.policies)
        ).detach()
    optimizer = torch.optim.Adam([medium_bias, low_bias], lr=0.04)
    for _ in range(128):
        optimizer.zero_grad(set_to_none=True)
        routed = [original[0]]
        for old, bias in zip(original[1:], (medium_bias, low_bias)):
            logits = torch.log(old.clamp_min(1e-12)) + bias.expand_as(old)
            routed.append(base.correction.routed_policy(logits, old))
        fulfill = base.correction.predicted_fulfillment(
            model,
            props,
            oracle,
            [value.unsqueeze(-1) for value in routed],
        )
        gain = fulfill - baseline
        bottom_count = max(1, round(len(cache) * float(variant["fraction"])))
        bottom_medium = torch.topk(
            gain[:, 1], k=bottom_count, largest=False
        ).values.mean()
        low_shortfall = torch.relu(-gain[:, 2]).mean()
        anchor = base_v1.reverse_kl(routed[1], original[1]) + base_v1.reverse_kl(
            routed[2], original[2]
        )
        objective = (
            gain[:, 1].mean()
            + float(variant["tail"]) * bottom_medium
            + float(variant["low_weight"]) * gain[:, 2].mean()
            - float(variant["low_floor"]) * low_shortfall
            - 0.002 * anchor
        )
        (-objective).backward()
        optimizer.step()
    return medium_bias.detach(), low_bias.detach()


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "method": "global path bias tail-objective screen",
        "strict_esm_inference": True,
        "training_bounds": [0, 350],
        "loads": {},
    }
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        train_cache = base.runtime.build_policy_cache(model, props, 0, 350, batch_size=32)
        caches = {
            "validation": base.runtime.build_policy_cache(model, props, 350, 400, batch_size=32),
            "evaluation": base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32),
        }
        baselines = {
            key: base.trainer.evaluate(model, props, value)[1]
            for key, value in caches.items()
        }
        trials = []
        for variant in VARIANTS:
            medium_bias, low_bias = train_bias(model, props, train_cache, variant)
            value = {"variant": variant, "splits": {}}
            for split, cache in caches.items():
                candidate = base_v1.apply_bias(cache, medium_bias, low_bias)
                _, summary = base.trainer.evaluate(model, props, cache, candidate)
                value["splits"][split] = {
                    "summary": base.compact(summary),
                    "delta": base.delta(summary, baselines[split]),
                }
                d = value["splits"][split]["delta"]
                print(
                    f"[{load}x/{split}] {variant['name']} "
                    f"M={d['Medium.norm_fulfill_mean']:+.5f}/"
                    f"{d['Medium.norm_fulfill_p1']:+.5f}/"
                    f"{d['Medium.norm_fulfill_p10']:+.5f} "
                    f"L={d['Low.norm_fulfill_mean']:+.5f}",
                    flush=True,
                )
            trials.append(value)
        report["loads"][str(load)] = {"trials": trials}
    path = HERE / "global_path_bias_v2.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
