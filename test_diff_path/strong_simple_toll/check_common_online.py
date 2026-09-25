from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("common_online_base", HERE / "run_tradeoff_screen.py")
configs = [
    (4, 0.06, 0.10, 0.01),
    (4, 0.08, 0.10, 0.01),
    (6, 0.06, 0.10, 0.01),
    (6, 0.08, 0.10, 0.01),
    (8, 0.06, 0.10, 0.01),
    (8, 0.08, 0.10, 0.01),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        for split, bounds in base.SPLITS.items():
            cache = base.runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            _, baseline = base.trainer.evaluate(model, props, cache)
            for config in configs:
                result = base.run_online(model, props, cache, baseline, config)
                d = result["delta"]
                print(
                    f"{load}x {split} {config} "
                    f"M={d['Medium.norm_fulfill_mean']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p1']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p10']:+.6f} "
                    f"L={d['Low.norm_fulfill_mean']:+.6f}/"
                    f"{d['Low.norm_fulfill_p1']:+.6f}/"
                    f"{d['Low.norm_fulfill_p10']:+.6f} "
                    f"ms={result['overlay_ms_per_snapshot']:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
