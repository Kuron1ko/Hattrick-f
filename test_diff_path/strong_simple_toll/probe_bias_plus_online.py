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


v1 = load_module("bias_plus_online_v1", HERE / "probe_global_path_bias.py")
v2 = load_module("bias_plus_online_v2", HERE / "probe_global_path_bias_v2.py")
base = v1.base
VARIANTS = (
    {"name": "tail8_q0.05", "tail": 8.0, "fraction": 0.05, "low_weight": 0.24, "low_floor": 0.0},
    {"name": "tail16_q0.1", "tail": 16.0, "fraction": 0.10, "low_weight": 0.24, "low_floor": 0.0},
)
SCALES = (0.90, 1.00, 1.05, 1.10)
ONLINE = tuple(
    (steps, 0.08, low_weight, 0.01)
    for steps in (1, 2, 4, 8)
    for low_weight in (0.20, 0.24, 0.30)
)


def biased_cache(cache, medium_bias, low_bias, scale):
    original = [value.squeeze(-1).detach() for value in cache.policies]
    routed = [original[0]]
    for old, bias in zip(original[1:], (medium_bias, low_bias)):
        logits = torch.log(old.clamp_min(1e-12)) + scale * bias.expand_as(old)
        routed.append(base.correction.routed_policy(logits, old))
    return replace(
        cache,
        policies=tuple(value.unsqueeze(-1) for value in routed),
        path_features=torch.cat(routed[1:], dim=1),
    )


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load = 3
    model, props, _ = base.screen.load_backbone(load, device)
    train_cache = base.runtime.build_policy_cache(model, props, 0, 400, batch_size=32)
    cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
    _, baseline = base.trainer.evaluate(model, props, cache)
    report = {
        "method": "historical global path bias plus at most eight strict-ESM steps",
        "strict_esm_inference": True,
        "actual_tm_used_for_policy": False,
        "trials": [],
    }
    for variant in VARIANTS:
        medium_bias, low_bias = v2.train_bias(model, props, train_cache, variant)
        for scale in SCALES:
            start = biased_cache(cache, medium_bias, low_bias, scale)
            for config in ONLINE:
                candidate = base.correction.correct_cache(model, props, start, *config)
                candidate = replace(cache, path_features=candidate.path_features)
                _, summary = base.trainer.evaluate(model, props, cache, candidate)
                value = {
                    "variant": variant,
                    "scale": scale,
                    "online": list(config),
                    "summary": base.compact(summary),
                    "delta": base.delta(summary, baseline),
                }
                report["trials"].append(value)
                d = value["delta"]
                feasible = all(
                    d[f"Medium.{metric}"] >= 0.02
                    for metric in (
                        "norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10"
                    )
                )
                if feasible or config[0] in (2, 8):
                    print(
                        f"{variant['name']} a={scale:g} s={config[0]} l={config[2]:g} "
                        f"ok={feasible} M={d['Medium.norm_fulfill_mean']:+.5f}/"
                        f"{d['Medium.norm_fulfill_p1']:+.5f}/"
                        f"{d['Medium.norm_fulfill_p10']:+.5f} "
                        f"L={d['Low.norm_fulfill_mean']:+.5f}",
                        flush=True,
                    )
    path = HERE / "bias_plus_online.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
