from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tradeoff = load_module("online_grid_base", HERE / "run_tradeoff_screen.py")
screen = tradeoff.screen
trainer = tradeoff.trainer
runtime = tradeoff.runtime

DISCOVERY = {"a": (318, 334), "b": (350, 366)}
FULL = {"safety": (318, 350), "validation": (350, 400)}
CONFIGS = [
    (steps, learning_rate, low_weight, 0.01)
    for steps in (2, 4, 6, 8)
    for learning_rate in (0.04, 0.06, 0.08)
    for low_weight in (0.01, 0.05, 0.10, 0.25)
]


def score(rows):
    medium = [row["delta"]["Medium.norm_fulfill_mean"] for row in rows]
    low = [row["delta"]["Low.norm_fulfill_mean"] for row in rows]
    low_tail = [
        min(
            row["delta"]["Low.norm_fulfill_p1"],
            row["delta"]["Low.norm_fulfill_p10"],
        )
        for row in rows
    ]
    feasible = min(low) >= -0.003 and min(low_tail) >= -0.01
    # Robust-first selection: reward the weaker discovery block.
    value = min(medium) + 0.25 * sum(medium) / len(medium)
    return feasible, value


def main():
    runtime.set_seed(20260823)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "strict_esm": True,
        "actual_tm_used_for_policy": False,
        "device": str(device),
        "discovery": DISCOVERY,
        "full": FULL,
        "loads": {},
    }
    for load_factor in (2, 3):
        model, props, _ = screen.load_backbone(load_factor, device)
        caches = {
            key: runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            for key, bounds in DISCOVERY.items()
        }
        baselines = {}
        for key, cache in caches.items():
            _, baselines[key] = trainer.evaluate(model, props, cache)
        trials = []
        for config in CONFIGS:
            rows = [
                tradeoff.public(
                    tradeoff.run_online(model, props, caches[key], baselines[key], config)
                )
                for key in DISCOVERY
            ]
            feasible, value = score(rows)
            trials.append(
                {"config": list(config), "feasible": feasible, "score": value, "splits": rows}
            )
        ranked = sorted(
            (row for row in trials if row["feasible"]),
            key=lambda row: row["score"],
            reverse=True,
        )
        selected = ranked[:4]
        full_results = []
        full_caches = {
            key: runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            for key, bounds in FULL.items()
        }
        full_baselines = {}
        for key, cache in full_caches.items():
            _, full_baselines[key] = trainer.evaluate(model, props, cache)
        for row in selected:
            config = tuple(row["config"])
            values = {
                key: tradeoff.public(
                    tradeoff.run_online(
                        model, props, full_caches[key], full_baselines[key], config
                    )
                )
                for key in FULL
            }
            full_results.append({"config": list(config), "splits": values})
            print(
                f"[{load_factor}x] {config} "
                + " ".join(
                    f"{key}:M={value['delta']['Medium.norm_fulfill_mean']:+.5f},"
                    f"L={value['delta']['Low.norm_fulfill_mean']:+.5f}"
                    for key, value in values.items()
                ),
                flush=True,
            )
        report["loads"][str(load_factor)] = {
            "top_discovery": selected,
            "full_results": full_results,
        }
    path = HERE / "online_grid.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
