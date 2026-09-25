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


base = load_module("knn_gate_base", HERE / "run_tradeoff_screen.py")
THRESHOLDS = (0.66, 0.68, 0.70)
CONFIG = (2, 32, 1.0, 1.0)
SPLITS = {"safety": (318, 350), "validation": (350, 400)}


def main():
    base.runtime.set_seed(20260823)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "strict_esm": True,
        "actual_tm_used_for_policy": False,
        "thresholds": list(THRESHOLDS),
        "config": list(CONFIG),
        "loads": {},
    }
    for load in (1, 2, 3):
        model, props, library = base.screen.load_backbone(load, device)
        split_values = {}
        for split, bounds in SPLITS.items():
            cache = base.runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            _, baseline = base.trainer.evaluate(model, props, cache)
            trials = []
            for threshold in THRESHOLDS:
                base.GATE = threshold
                value = base.public(
                    base.run_knn(model, props, cache, baseline, library, CONFIG)
                )
                value["threshold"] = threshold
                trials.append(value)
                d = value["delta"]
                print(
                    f"{load}x/{split} t={threshold:.2f} active={value['active']}/{len(cache)} "
                    f"M={d['Medium.norm_fulfill_mean']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p1']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p10']:+.6f} "
                    f"L={d['Low.norm_fulfill_mean']:+.6f}/"
                    f"{d['Low.norm_fulfill_p1']:+.6f}/"
                    f"{d['Low.norm_fulfill_p10']:+.6f}",
                    flush=True,
                )
            split_values[split] = trials
        report["loads"][str(load)] = split_values
    path = HERE / "knn_gate_screen.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
