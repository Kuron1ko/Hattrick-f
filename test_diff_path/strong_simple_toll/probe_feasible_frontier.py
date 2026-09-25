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


low = load_module("feasible_frontier_low", HERE / "probe_low_recovery.py")
v1, v2, plus, base = low.v1, low.v2, low.plus, low.base


def evaluate_policies(model, props, cache, policies, baseline):
    candidate = replace(
        cache,
        policies=tuple(value.unsqueeze(-1) for value in policies),
        path_features=torch.cat(policies[1:], dim=1),
    )
    _, summary = base.runtime.evaluate_cache(
        model, props, candidate, None, batch_size=16
    )
    return candidate, summary, base.delta(summary, baseline)


def medium_feasible(delta):
    return all(
        delta[f"Medium.{metric}"] >= 0.02
        for metric in ("norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10")
    )


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
        raw_trials = []
        candidates = []
        if load == 2:
            variant = {"name": "mean", "tail": 0.0, "low_weight": 0.24, "low_floor": 0.0}
            medium_bias, low_bias = v1.train_bias(model, props, train_cache, variant)
            for integer_scale in range(76, 101, 2):
                scale = integer_scale / 100
                biased = plus.biased_cache(cache, medium_bias, low_bias, scale)
                policies = [value.squeeze(-1).detach() for value in biased.policies]
                candidate, summary, delta = evaluate_policies(
                    model, props, cache, policies, baseline
                )
                raw_trials.append({"scale": scale, "delta": delta})
                if medium_feasible(delta):
                    candidates.append((delta["Low.norm_fulfill_mean"], {"scale": scale}, candidate, policies))
        else:
            variant = {
                "name": "tail8_q0.05", "tail": 8.0, "fraction": 0.05,
                "low_weight": 0.24, "low_floor": 0.0,
            }
            medium_bias, low_bias = v2.train_bias(model, props, train_cache, variant)
            for integer_scale in range(78, 97, 2):
                scale = integer_scale / 100
                biased = plus.biased_cache(cache, medium_bias, low_bias, scale)
                for steps in (6, 7, 8, 9, 10):
                    for low_weight in (0.18, 0.20, 0.22, 0.24):
                        refined = base.correction.correct_cache(
                            model, props, biased, steps, 0.08, low_weight, 0.01
                        )
                        policies = low.policies_from_features(cache, refined.path_features)
                        candidate, summary, delta = evaluate_policies(
                            model, props, cache, policies, baseline
                        )
                        config = {
                            "scale": scale, "steps": steps,
                            "low_weight": low_weight,
                        }
                        raw_trials.append({"config": config, "delta": delta})
                        if medium_feasible(delta):
                            candidates.append((delta["Low.norm_fulfill_mean"], config, candidate, policies))
        candidates.sort(key=lambda value: value[0], reverse=True)
        recovered = []
        for _, config, candidate, policies in candidates[:8]:
            for steps in (8, 16, 32, 64):
                recovery = low.recover_low(
                    model, props, cache, policies,
                    policies[2], steps,
                )
                _, summary = base.runtime.evaluate_cache(
                    model, props, recovery, None, batch_size=16
                )
                delta = base.delta(summary, baseline)
                recovered.append(
                    {"config": config, "recovery_steps": steps, "delta": delta}
                )
                print(
                    f"[{load}x] cfg={config} recover={steps} "
                    f"M={delta['Medium.norm_fulfill_mean']:+.5f}/"
                    f"{delta['Medium.norm_fulfill_p1']:+.5f}/"
                    f"{delta['Medium.norm_fulfill_p10']:+.5f} "
                    f"L={delta['Low.norm_fulfill_mean']:+.5f}/"
                    f"{delta['Low.norm_fulfill_p1']:+.5f}/"
                    f"{delta['Low.norm_fulfill_p10']:+.5f}",
                    flush=True,
                )
        report["loads"][str(load)] = {
            "raw_trials": raw_trials,
            "feasible_count": len(candidates),
            "recovered": recovered,
        }
    path = HERE / "feasible_frontier.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
