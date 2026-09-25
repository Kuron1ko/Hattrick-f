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


universal = load_module("universal_across_base", HERE / "probe_universal_num.py")
base = universal.base
CONFIGS = (
    ("relative_softmin_0.005", 16, 0.08, 0.02),
    ("relative_softmin_0.005", 24, 0.08, 0.02),
)
MARGINS = (None, 0.0, 0.0001, 0.0005)


def pareto_guard(model, props, cache, candidate, margin):
    if margin is None:
        return candidate, torch.ones(len(cache), dtype=torch.bool, device=cache.policies[0].device)
    with torch.no_grad():
        baseline_fulfill = base.correction.predicted_fulfillment(
            model, props, cache, list(cache.policies)
        )
        candidate_fulfill = base.correction.predicted_fulfillment(
            model, props, candidate, list(candidate.policies)
        )
        active = ((candidate_fulfill - baseline_fulfill) >= margin).all(dim=1)
        policies = [
            torch.where(active[:, None, None], new, old)
            for new, old in zip(candidate.policies, cache.policies)
        ]
    return replace(
        cache,
        policies=tuple(policies),
        path_features=torch.cat(
            [policies[1].squeeze(-1), policies[2].squeeze(-1)], dim=1
        ),
    ), active


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "method": "one class-symmetric max-min relative-gain objective",
        "strict_esm": True,
        "same_configuration_for_all_loads": True,
        "loads": {},
    }
    for load in (1, 2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
        _, baseline = base.trainer.evaluate(model, props, cache)
        trials = []
        for config in CONFIGS:
            raw = universal.correct(model, props, cache, config)
            for margin in MARGINS:
                candidate, active = pareto_guard(
                    model, props, cache, raw, margin
                )
                _, summary = base.runtime.evaluate_cache(
                    model, props, candidate, None, batch_size=16
                )
                value = {
                    "config": list(config),
                    "pareto_margin": margin,
                    "active": int(active.sum()),
                    "summary": base.compact(summary),
                    "delta": base.delta(summary, baseline),
                }
                trials.append(value)
                d = value["delta"]
                print(
                    f"[{load}x] s={config[1]} margin={margin} active={value['active']} "
                    f"H={d['High.norm_fulfill_mean']:+.5f}/"
                    f"{d['High.norm_fulfill_p1']:+.5f}/"
                    f"{d['High.norm_fulfill_p10']:+.5f} "
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
    path = HERE / "universal_across_loads.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
