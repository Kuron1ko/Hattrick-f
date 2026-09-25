from __future__ import annotations

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


v1 = load_module("final_bias_scale_v1", HERE / "probe_global_path_bias.py")
v2 = load_module("final_bias_scale_v2", HERE / "probe_global_path_bias_v2.py")
base = v1.base
SCALES = (0.50, 0.75, 1.00, 1.10, 1.20, 1.40, 1.60)
LOAD_VARIANTS = {
    2: (
        {"name": "mean", "tail": 0.0, "low_weight": 0.24, "low_floor": 0.0},
        {"name": "tail05", "tail": 0.5, "low_weight": 0.24, "low_floor": 0.0},
    ),
    3: (
        {"name": "tail8_q0.05", "tail": 8.0, "fraction": 0.05, "low_weight": 0.24, "low_floor": 0.0},
        {"name": "tail16_q0.05", "tail": 16.0, "fraction": 0.05, "low_weight": 0.24, "low_floor": 0.0},
        {"name": "tail16_q0.1", "tail": 16.0, "fraction": 0.10, "low_weight": 0.24, "low_floor": 0.0},
    ),
}


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "method": "historical global path bias with one scalar strength",
        "strict_esm_inference": True,
        "training_bounds": [0, 400],
        "evaluation_bounds": [400, 500],
        "loads": {},
    }
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        train_cache = base.runtime.build_policy_cache(model, props, 0, 400, batch_size=32)
        cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
        _, baseline = base.trainer.evaluate(model, props, cache)
        trials = []
        for variant in LOAD_VARIANTS[load]:
            if load == 2:
                medium_bias, low_bias = v1.train_bias(model, props, train_cache, variant)
            else:
                medium_bias, low_bias = v2.train_bias(model, props, train_cache, variant)
            for scale in SCALES:
                candidate = v1.apply_bias(
                    cache, scale * medium_bias, scale * low_bias
                )
                _, summary = base.trainer.evaluate(model, props, cache, candidate)
                value = {
                    "variant": variant,
                    "scale": scale,
                    "summary": base.compact(summary),
                    "delta": base.delta(summary, baseline),
                }
                trials.append(value)
                d = value["delta"]
                feasible = all(
                    d[f"Medium.{metric}"] >= 0.02
                    for metric in (
                        "norm_fulfill_mean",
                        "norm_fulfill_p1",
                        "norm_fulfill_p10",
                    )
                )
                print(
                    f"[{load}x] {variant['name']} scale={scale:g} ok={feasible} "
                    f"M={d['Medium.norm_fulfill_mean']:+.5f}/"
                    f"{d['Medium.norm_fulfill_p1']:+.5f}/"
                    f"{d['Medium.norm_fulfill_p10']:+.5f} "
                    f"L={d['Low.norm_fulfill_mean']:+.5f}/"
                    f"{d['Low.norm_fulfill_p1']:+.5f}/"
                    f"{d['Low.norm_fulfill_p10']:+.5f}",
                    flush=True,
                )
        report["loads"][str(load)] = {
            "baseline": base.compact(baseline),
            "trials": trials,
        }
    path = HERE / "final_bias_scale_probe.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
