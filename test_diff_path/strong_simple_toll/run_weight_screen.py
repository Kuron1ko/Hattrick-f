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


base = load_module("weight_screen_base", HERE / "run_tradeoff_screen.py")
WEIGHTS = (0.242, 0.244, 0.246, 0.248, 0.250, 0.252, 0.254)
LEARNING_RATES = (0.08,)
SPLITS = {"safety": (318, 350), "validation": (350, 400)}


def main():
    base.runtime.set_seed(20260823)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "method": "single 8-step strict-ESM correction",
        "strict_esm": True,
        "actual_tm_used_for_policy": False,
        "loads": {},
    }
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        split_report = {}
        for split, bounds in SPLITS.items():
            cache = base.runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            _, baseline = base.trainer.evaluate(model, props, cache)
            rows = []
            for lr in LEARNING_RATES:
                for weight in WEIGHTS:
                    config = (8, lr, weight, 0.01)
                    value = base.public(
                        base.run_online(model, props, cache, baseline, config)
                    )
                    rows.append(value)
                    d = value["delta"]
                    print(
                        f"{load}x/{split} lr={lr:.2f} w={weight:.2f} "
                        f"M={d['Medium.norm_fulfill_mean']:+.6f}/"
                        f"{d['Medium.norm_fulfill_p1']:+.6f}/"
                        f"{d['Medium.norm_fulfill_p10']:+.6f} "
                        f"L={d['Low.norm_fulfill_mean']:+.6f}/"
                        f"{d['Low.norm_fulfill_p1']:+.6f}/"
                        f"{d['Low.norm_fulfill_p10']:+.6f}",
                        flush=True,
                    )
            split_report[split] = rows
        report["loads"][str(load)] = split_report
    path = HERE / "weight_screen.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
