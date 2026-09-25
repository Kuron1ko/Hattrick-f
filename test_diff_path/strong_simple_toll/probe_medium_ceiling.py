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


base = load_module("medium_ceiling_base", HERE / "run_tradeoff_screen.py")
CONFIGS = (
    (8, 0.08, 0.0, 0.0),
    (16, 0.08, 0.0, 0.0),
    (24, 0.06, 0.0, 0.0),
    (32, 0.06, 0.0, 0.0),
)


def evaluate(model, props, cache, baseline, optimization_cache, config):
    candidate = base.correction.correct_cache(model, props, optimization_cache, *config)
    # Policies are applied to the original cache; optimization_cache differs only
    # in which TM tuple is exposed through predicted_tms.
    candidate = replace(cache, path_features=candidate.path_features)
    _, summary = base.trainer.evaluate(model, props, cache, candidate)
    return {
        "config": list(config),
        "summary": base.compact(summary),
        "delta": base.delta(summary, baseline),
    }


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "purpose": "diagnostic ceiling only; actual-TM branch is not deployable",
        "loads": {},
    }
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
        _, baseline = base.trainer.evaluate(model, props, cache)
        oracle_cache = replace(cache, predicted_tms=cache.tms)
        rows = {"strict_esm": [], "oracle_actual": []}
        for config in CONFIGS:
            for name, optimization_cache in (
                ("strict_esm", cache),
                ("oracle_actual", oracle_cache),
            ):
                value = evaluate(
                    model, props, cache, baseline, optimization_cache, config
                )
                rows[name].append(value)
                d = value["delta"]
                print(
                    f"[{load}x/{name}] steps={config[0]} "
                    f"M={d['Medium.norm_fulfill_mean']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p1']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p10']:+.6f} "
                    f"L={d['Low.norm_fulfill_mean']:+.6f}",
                    flush=True,
                )
        report["loads"][str(load)] = {
            "baseline": base.compact(baseline),
            "trials": rows,
        }
    path = HERE / "medium_ceiling.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
