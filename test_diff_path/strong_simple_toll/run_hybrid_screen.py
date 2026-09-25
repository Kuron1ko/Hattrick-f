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


base = load_module("hybrid_screen_base", HERE / "run_tradeoff_screen.py")
screen = base.screen
trainer = base.trainer
runtime = base.runtime

DISCOVERY = {"a": (318, 334), "b": (350, 366)}
FULL = {"safety": (318, 350), "validation": (350, 400)}
CONFIGS = [(8, 0.08, 0.10, low_gain) for low_gain in (1.2, 1.6, 2.0, 3.0, 4.0)]


def run_candidate(
    model, props, cache, baseline, library, prototypes, config, public_only=True
):
    steps, lr, low_weight, low_gain = config
    active = base.gate(cache)
    online = base.correction.correct_cache(
        model, props, cache, steps, lr, low_weight, 0.01
    )
    features = trainer.edge_features(cache)
    tolls = screen.predict(library, prototypes, features).clone()
    tolls[:, 0].zero_()
    tolls[:, 1].mul_(low_gain)
    stable_low = trainer.route_with_tolls(cache, tolls, 1.0)
    path_count = int(cache.policies[0].shape[1])
    candidate = replace(
        cache,
        path_features=torch.cat(
            [
                online.path_features[:, :path_count],
                stable_low.path_features[:, path_count:],
            ],
            dim=1,
        ),
    )
    candidate = base.apply_fallback(cache, candidate, active)
    value = base.evaluate_cache(model, props, cache, candidate, baseline)
    value.update({"config": list(config), "active": int(active.sum().item())})
    return base.public(value) if public_only else value


def feasible(rows):
    return all(
        row["delta"]["Medium.norm_fulfill_mean"] > 0
        and row["delta"]["Medium.norm_fulfill_p1"] > -0.01
        and row["delta"]["Medium.norm_fulfill_p10"] > -0.005
        and row["delta"]["Low.norm_fulfill_mean"] >= -0.012
        and row["delta"]["Low.norm_fulfill_p1"] >= -0.01
        and row["delta"]["Low.norm_fulfill_p10"] >= -0.02
        for row in rows
    )


def main():
    runtime.set_seed(20260823)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contexts = {}
    for load in (2, 3):
        model, props, library = screen.load_backbone(load, device)
        prototypes = screen.fit_prototypes(library, 3)
        caches = {
            name: runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            for name, bounds in DISCOVERY.items()
        }
        baselines = {}
        for name, cache in caches.items():
            _, baselines[name] = trainer.evaluate(model, props, cache)
        contexts[load] = (model, props, library, prototypes, caches, baselines)

    trials = []
    for config in CONFIGS:
        all_rows = []
        load_values = {}
        for load, context in contexts.items():
            model, props, library, prototypes, caches, baselines = context
            rows = {
                name: run_candidate(
                    model,
                    props,
                    caches[name],
                    baselines[name],
                    library,
                    prototypes,
                    config,
                )
                for name in DISCOVERY
            }
            load_values[str(load)] = rows
            all_rows.extend(rows.values())
        robust_medium = min(
            row["delta"]["Medium.norm_fulfill_mean"] for row in all_rows
        )
        mean_medium = sum(
            row["delta"]["Medium.norm_fulfill_mean"] for row in all_rows
        ) / len(all_rows)
        robust_low = min(
            row["delta"]["Low.norm_fulfill_mean"] for row in all_rows
        )
        trials.append(
            {
                "config": list(config),
                "feasible": feasible(all_rows),
                "score": robust_medium + 0.25 * mean_medium + 0.10 * robust_low,
                "loads": load_values,
            }
        )
    selected = sorted(
        (row for row in trials if row["feasible"]),
        key=lambda row: row["score"],
        reverse=True,
    )[:4]
    full_results = []
    for row in selected:
        config = tuple(row["config"])
        load_values = {}
        for load, context in contexts.items():
            model, props, library, prototypes, _, _ = context
            values = {}
            for name, bounds in FULL.items():
                cache = runtime.build_policy_cache(model, props, *bounds, batch_size=32)
                _, baseline = trainer.evaluate(model, props, cache)
                values[name] = run_candidate(
                    model, props, cache, baseline, library, prototypes, config
                )
            load_values[str(load)] = values
        full_results.append({"config": list(config), "loads": load_values})
        print(f"config={config}", flush=True)
        for load, values in load_values.items():
            for split, value in values.items():
                d = value["delta"]
                print(
                    f"  {load}x/{split} M={d['Medium.norm_fulfill_mean']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p1']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p10']:+.6f} "
                    f"L={d['Low.norm_fulfill_mean']:+.6f}/"
                    f"{d['Low.norm_fulfill_p1']:+.6f}/"
                    f"{d['Low.norm_fulfill_p10']:+.6f}",
                    flush=True,
                )
    report = {
        "method": "strict-ESM Medium short correction plus stable Low toll",
        "strict_esm": True,
        "actual_tm_used_for_policy": False,
        "trials": trials,
        "selected": selected,
        "full_results": full_results,
    }
    path = HERE / "hybrid_screen.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
