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


base = load_module("level4_probe_base", HERE / "run_tradeoff_screen.py")
CONFIGS = {
    "online_aggressive": (8, 0.08, 0.10, 0.01),
    "online_balanced": (8, 0.06, 0.25, 0.01),
    "online_pivot": (8, 0.08, 0.241, 0.01),
}
KNN_CONFIG = (2, 32, 1.0, 1.0)


def compact_rows(rows):
    return [
        {
            "snapshot": int(row["snapshot"]),
            "class": str(row["class"]),
            "norm_fulfill": float(row["norm_fulfill"]),
        }
        for row in rows
    ]


def main():
    base.runtime.set_seed(20260823)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "strict_esm": True,
        "actual_tm_used_for_policy": False,
        "evaluation": [400, 500],
        "loads": {},
    }
    for load in (2, 3):
        model, props, library = base.screen.load_backbone(load, device)
        cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
        baseline_rows, baseline_summary = base.trainer.evaluate(model, props, cache)
        methods = {}
        for name, config in CONFIGS.items():
            result = base.run_online(model, props, cache, baseline_summary, config)
            methods[name] = {
                **base.public(result),
                "rows": compact_rows(result["rows"]),
            }
        result = base.run_knn(
            model, props, cache, baseline_summary, library, KNN_CONFIG
        )
        methods["knn"] = {
            **base.public(result),
            "rows": compact_rows(result["rows"]),
        }
        report["loads"][str(load)] = {
            "baseline": base.compact(baseline_summary),
            "baseline_rows": compact_rows(baseline_rows),
            "methods": methods,
        }
        for name, value in methods.items():
            d = value["delta"]
            print(
                f"[{load}x/{name}] M={d['Medium.norm_fulfill_mean']:+.6f}/"
                f"{d['Medium.norm_fulfill_p1']:+.6f}/"
                f"{d['Medium.norm_fulfill_p10']:+.6f} "
                f"L={d['Low.norm_fulfill_mean']:+.6f}/"
                f"{d['Low.norm_fulfill_p1']:+.6f}/"
                f"{d['Low.norm_fulfill_p10']:+.6f} "
                f"ms={value['overlay_ms_per_snapshot']:.3f}",
                flush=True,
            )
    path = HERE / "level4_probe.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
