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


front = load_module("fast_frontier_base", HERE / "probe_feasible_frontier.py")
low, v1, v2, plus, base = front.low, front.v1, front.v2, front.plus, front.base


def recover_low_lr(model, props, cache, fixed, low_start, steps, learning_rate):
    logits = torch.nn.Parameter(torch.log(low_start.clamp_min(1e-12)))
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        routed_low = base.correction.routed_policy(logits, low_start)
        fulfill = base.correction.predicted_fulfillment(
            model,
            props,
            cache,
            [fixed[0].unsqueeze(-1), fixed[1].unsqueeze(-1), routed_low.unsqueeze(-1)],
        )
        anchor = v1.reverse_kl(routed_low, low_start)
        objective = fulfill[:, 2].sum() - 0.01 * anchor
        (-objective).backward()
        optimizer.step()
    with torch.no_grad():
        routed_low = base.correction.routed_policy(logits, low_start)
    return [fixed[0], fixed[1], routed_low]


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"strict_esm_inference": True, "loads": {}}
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        train_cache = base.runtime.build_policy_cache(model, props, 0, 400, batch_size=32)
        cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
        _, baseline = base.trainer.evaluate(model, props, cache)
        feasible = []
        if load == 2:
            variant = {"name": "mean", "tail": 0.0, "low_weight": 0.24, "low_floor": 0.0}
            medium_bias, low_bias = v1.train_bias(model, props, train_cache, variant)
            biased = plus.biased_cache(cache, medium_bias, low_bias, 0.82)
            policies = [value.squeeze(-1).detach() for value in biased.policies]
            _, _, delta = front.evaluate_policies(model, props, cache, policies, baseline)
            feasible.append((delta["Low.norm_fulfill_mean"], {"scale": 0.82}, policies))
        else:
            variant = {
                "name": "tail8_q0.05", "tail": 8.0, "fraction": 0.05,
                "low_weight": 0.24, "low_floor": 0.0,
            }
            medium_bias, low_bias = v2.train_bias(model, props, train_cache, variant)
            for scale in (0.76, 0.78, 0.80, 0.82, 0.84):
                biased = plus.biased_cache(cache, medium_bias, low_bias, scale)
                for steps in (4, 6, 8):
                    for learning_rate in (0.08, 0.10, 0.12, 0.16):
                        for low_weight in (0.20, 0.22, 0.24):
                            refined = base.correction.correct_cache(
                                model, props, biased, steps, learning_rate,
                                low_weight, 0.01,
                            )
                            policies = low.policies_from_features(cache, refined.path_features)
                            _, _, delta = front.evaluate_policies(
                                model, props, cache, policies, baseline
                            )
                            config = {
                                "scale": scale, "joint_steps": steps,
                                "joint_learning_rate": learning_rate,
                                "joint_low_weight": low_weight,
                            }
                            if front.medium_feasible(delta):
                                feasible.append(
                                    (delta["Low.norm_fulfill_mean"], config, policies)
                                )
        feasible.sort(key=lambda value: value[0], reverse=True)
        recovered = []
        for _, config, policies in feasible[:10]:
            for recovery_steps in (2, 4, 6, 8):
                for recovery_lr in (0.08, 0.16, 0.24, 0.32):
                    routed = recover_low_lr(
                        model, props, cache, policies, policies[2],
                        recovery_steps, recovery_lr,
                    )
                    _, _, delta = front.evaluate_policies(
                        model, props, cache, routed, baseline
                    )
                    value = {
                        "config": config,
                        "recovery_steps": recovery_steps,
                        "recovery_learning_rate": recovery_lr,
                        "delta": delta,
                    }
                    recovered.append(value)
        recovered.sort(
            key=lambda value: value["delta"]["Low.norm_fulfill_mean"],
            reverse=True,
        )
        for value in recovered[:12]:
            d = value["delta"]
            print(
                f"[{load}x] {value['config']} "
                f"low={value['recovery_steps']}@{value['recovery_learning_rate']:g} "
                f"M={d['Medium.norm_fulfill_mean']:+.5f}/"
                f"{d['Medium.norm_fulfill_p1']:+.5f}/"
                f"{d['Medium.norm_fulfill_p10']:+.5f} "
                f"L={d['Low.norm_fulfill_mean']:+.5f}/"
                f"{d['Low.norm_fulfill_p1']:+.5f}/"
                f"{d['Low.norm_fulfill_p10']:+.5f}",
                flush=True,
            )
        report["loads"][str(load)] = {
            "feasible_count": len(feasible),
            "best_recovered": recovered[:30],
        }
    path = HERE / "fast_frontier.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
