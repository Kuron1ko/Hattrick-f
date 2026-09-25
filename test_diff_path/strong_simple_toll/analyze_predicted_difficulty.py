from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("difficulty_base", HERE / "run_tradeoff_screen.py")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"loads": {}}
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        load_rows = {}
        for split, bounds in {
            "development": (318, 400),
            "evaluation": (400, 500),
        }.items():
            cache = base.runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            policies = list(cache.policies)
            predicted = base.correction.predicted_fulfillment(
                model, props, cache, policies
            ).detach().cpu().numpy()
            rows, _ = base.trainer.evaluate(model, props, cache)
            actual_medium = np.asarray(
                [row["norm_fulfill"] for row in rows if row["class"] == "Medium"]
            )
            pm = predicted[:, 1]
            value = {
                "predicted_medium_quantiles": np.quantile(
                    pm, [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
                ).tolist(),
                "actual_medium_quantiles": np.quantile(
                    actual_medium, [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
                ).tolist(),
                "correlation": float(np.corrcoef(pm, actual_medium)[0, 1]),
            }
            load_rows[split] = value
            print(load, split, value, flush=True)
        report["loads"][str(load)] = load_rows
    (HERE / "predicted_difficulty.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
