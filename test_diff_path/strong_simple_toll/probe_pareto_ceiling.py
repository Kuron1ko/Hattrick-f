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


base = load_module("pareto_ceiling_base", HERE / "run_tradeoff_screen.py")
WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.24, 0.30, 0.40, 0.60, 1.00)


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "purpose": "fixed-High Pareto feasibility diagnostic",
        "actual_branch_deployable": False,
        "loads": {},
    }
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
        _, baseline = base.trainer.evaluate(model, props, cache)
        oracle_cache = replace(cache, predicted_tms=cache.tms)
        trials = []
        for weight in WEIGHTS:
            config = (24, 0.06, weight, 0.01)
            candidate = base.correction.correct_cache(
                model, props, oracle_cache, *config
            )
            candidate = replace(cache, path_features=candidate.path_features)
            _, summary = base.trainer.evaluate(model, props, cache, candidate)
            value = {
                "weight": weight,
                "summary": base.compact(summary),
                "delta": base.delta(summary, baseline),
            }
            trials.append(value)
            d = value["delta"]
            print(
                f"[{load}x] w={weight:.2f} "
                f"M={d['Medium.norm_fulfill_mean']:+.6f}/"
                f"{d['Medium.norm_fulfill_p1']:+.6f}/"
                f"{d['Medium.norm_fulfill_p10']:+.6f} "
                f"L={d['Low.norm_fulfill_mean']:+.6f}/"
                f"{d['Low.norm_fulfill_p1']:+.6f}/"
                f"{d['Low.norm_fulfill_p10']:+.6f}",
                flush=True,
            )
        report["loads"][str(load)] = {
            "baseline": base.compact(baseline),
            "trials": trials,
        }
    path = HERE / "pareto_ceiling.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
