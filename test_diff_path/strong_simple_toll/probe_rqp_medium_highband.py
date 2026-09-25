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


probe = load_module("rqp_medium_highband_base", HERE / "probe_rqp_medium.py")
base = probe.base


def score(value):
    by_load = {item["load"]: item for item in value["loads"]}
    d2 = by_load[2]["delta"]
    d3 = by_load[3]["delta"]
    high_floor = min(
        d3["High.norm_fulfill_mean"],
        d3["High.norm_fulfill_p1"],
        d3["High.norm_fulfill_p10"],
    )
    low_floor = min(
        d2["Low.norm_fulfill_mean"],
        d2["Low.norm_fulfill_p1"],
        d2["Low.norm_fulfill_p10"],
        d3["Low.norm_fulfill_mean"],
        d3["Low.norm_fulfill_p1"],
        d3["Low.norm_fulfill_p10"],
    )
    medium_score = sum(
        d[f"Medium.{metric}"]
        for d in (d2, d3)
        for metric in (
            "norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10"
        )
    )
    return high_floor >= 0.012, low_floor >= -0.005, medium_score, high_floor


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    specs = [
        {
            "high_weight": high_weight,
            "medium_weight": medium_weight,
            "medium_floor": -0.02,
            "low_floor": -0.005,
        }
        for high_weight in (0.96, 1.00, 1.10, 1.20)
        for medium_weight in (0.05, 0.10, 0.15)
    ]
    small_baselines, small = probe.run_bounds(
        (2, 3), (318, 334), specs, "highband-small"
    )
    small.sort(key=score, reverse=True)
    frozen = [value["spec"] for value in small[:6]]
    validation_baselines, validation = probe.run_bounds(
        (2, 3), (334, 400), frozen, "highband-validation"
    )
    validation.sort(key=score, reverse=True)
    for value in validation:
        value["constrained_score"] = score(value)
    path = HERE / "rqp_medium_highband.json"
    path.write_text(
        json.dumps(
            {
                "method": "high-band refinement of the one-stage RQP-M objective",
                "strict_esm_online": True,
                "current_actual_tm_used_online": False,
                "training_bounds": [0, 318],
                "small_bounds": [318, 334],
                "validation_bounds": [334, 400],
                "small_baselines": small_baselines,
                "small_ranked": small,
                "frozen_specs": frozen,
                "validation_baselines": validation_baselines,
                "validation_ranked": validation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"BEST={validation[0]['spec']} "
        f"score={validation[0]['constrained_score']}",
        flush=True,
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
