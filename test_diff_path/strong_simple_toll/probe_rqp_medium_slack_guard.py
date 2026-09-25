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


probe = load_module("rqp_medium_slack_guard_base", HERE / "probe_rqp_medium.py")
base = probe.base


def constrained_score(value):
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
    return (
        high_floor >= 0.0115,
        low_floor >= -0.005,
        medium_score,
        high_floor,
    )


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    specs = [
        {
            "high_weight": high_weight,
            "medium_weight": 0.10,
            "medium_floor": -0.02,
            "low_floor": -0.005,
            "high_slack_fraction": slack_fraction,
        }
        for high_weight in (0.84, 0.88)
        for slack_fraction in (0.005, 0.010, 0.020, 0.030, 0.050)
    ]
    baselines, ranked = probe.run_bounds(
        (2, 3), (334, 400), specs, "slack-validation"
    )
    ranked.sort(key=constrained_score, reverse=True)
    for value in ranked:
        value["constrained_score"] = constrained_score(value)
    path = HERE / "rqp_medium_slack_guard.json"
    path.write_text(
        json.dumps(
            {
                "method": "RQP-M with load-adaptive High-slack guard",
                "strict_esm_online": True,
                "current_actual_tm_used_online": False,
                "training_bounds": [0, 318],
                "validation_bounds": [334, 400],
                "baselines": baselines,
                "ranked": ranked,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"BEST={ranked[0]['spec']} score={ranked[0]['constrained_score']}",
        flush=True,
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
