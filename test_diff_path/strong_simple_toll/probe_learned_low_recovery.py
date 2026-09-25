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


low = load_module("learned_low_recovery_low", HERE / "probe_low_recovery.py")
v1, v2, plus, base = low.v1, low.v2, low.plus, low.base
TAIL_WEIGHTS = (0.0, 0.5, 1.0, 2.0, 4.0)


def final_policies(model, props, cache, medium_bias, low_bias):
    biased = plus.biased_cache(cache, medium_bias, low_bias, 0.9)
    refined = base.correction.correct_cache(
        model, props, biased, 8, 0.08, 0.20, 0.01
    )
    fixed = low.policies_from_features(cache, refined.path_features)
    return fixed


def train_low_bias(model, props, cache, fixed, start, tail_weight):
    oracle = replace(cache, predicted_tms=cache.tms)
    bias = torch.nn.Parameter(torch.zeros_like(start[:1]))
    with torch.no_grad():
        base_fulfill = base.correction.predicted_fulfillment(
            model,
            props,
            oracle,
            [fixed[0].unsqueeze(-1), fixed[1].unsqueeze(-1), start.unsqueeze(-1)],
        )[:, 2].detach()
    optimizer = torch.optim.Adam([bias], lr=0.05)
    for _ in range(128):
        optimizer.zero_grad(set_to_none=True)
        logits = torch.log(start.clamp_min(1e-12)) + bias.expand_as(start)
        routed = base.correction.routed_policy(logits, start)
        fulfill = base.correction.predicted_fulfillment(
            model,
            props,
            oracle,
            [fixed[0].unsqueeze(-1), fixed[1].unsqueeze(-1), routed.unsqueeze(-1)],
        )[:, 2]
        gain = fulfill - base_fulfill
        bottom = torch.topk(gain, k=max(1, len(cache) // 10), largest=False).values.mean()
        anchor = v1.reverse_kl(routed, start)
        objective = gain.mean() + tail_weight * bottom - 0.002 * anchor
        (-objective).backward()
        optimizer.step()
    return bias.detach()


def apply_low_bias(cache, fixed, start, bias):
    logits = torch.log(start.clamp_min(1e-12)) + bias.expand_as(start)
    routed_low = base.correction.routed_policy(logits, start)
    policies = [fixed[0], fixed[1], routed_low]
    return replace(
        cache,
        policies=tuple(value.unsqueeze(-1) for value in policies),
        path_features=torch.cat(policies[1:], dim=1),
    )


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props, _ = base.screen.load_backbone(3, device)
    train_cache = base.runtime.build_policy_cache(model, props, 0, 400, batch_size=32)
    cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
    _, baseline = base.trainer.evaluate(model, props, cache)
    variant = {
        "name": "tail8_q0.05", "tail": 8.0, "fraction": 0.05,
        "low_weight": 0.24, "low_floor": 0.0,
    }
    medium_bias, low_bias = v2.train_bias(model, props, train_cache, variant)
    train_fixed = final_policies(model, props, train_cache, medium_bias, low_bias)
    eval_fixed = final_policies(model, props, cache, medium_bias, low_bias)
    train_starts = {
        "original": train_cache.policies[2].squeeze(-1).detach(),
        "candidate": train_fixed[2],
    }
    eval_starts = {
        "original": cache.policies[2].squeeze(-1).detach(),
        "candidate": eval_fixed[2],
    }
    report = {
        "method": "historically learned priority-safe Low-only path bias",
        "strict_esm_inference": True,
        "trials": [],
    }
    for start_name in ("original", "candidate"):
        for tail_weight in TAIL_WEIGHTS:
            bias = train_low_bias(
                model, props, train_cache, train_fixed,
                train_starts[start_name], tail_weight,
            )
            candidate = apply_low_bias(
                cache, eval_fixed, eval_starts[start_name], bias
            )
            _, summary = base.runtime.evaluate_cache(
                model, props, candidate, None, batch_size=16
            )
            value = {
                "start": start_name,
                "tail_weight": tail_weight,
                "summary": base.compact(summary),
                "delta": base.delta(summary, baseline),
            }
            report["trials"].append(value)
            d = value["delta"]
            print(
                f"{start_name} tail={tail_weight:g} "
                f"M={d['Medium.norm_fulfill_mean']:+.5f}/"
                f"{d['Medium.norm_fulfill_p1']:+.5f}/"
                f"{d['Medium.norm_fulfill_p10']:+.5f} "
                f"L={d['Low.norm_fulfill_mean']:+.5f}/"
                f"{d['Low.norm_fulfill_p1']:+.5f}/"
                f"{d['Low.norm_fulfill_p10']:+.5f}",
                flush=True,
            )
    path = HERE / "learned_low_recovery.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
