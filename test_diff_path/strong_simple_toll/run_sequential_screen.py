from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

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


base = load_module("sequential_screen_base", HERE / "run_tradeoff_screen.py")
screen = base.screen
trainer = base.trainer
runtime = base.runtime
correction = base.correction

DISCOVERY = {"a": (318, 334), "b": (350, 366)}
FULL = {"safety": (318, 350), "validation": (350, 400)}
CONFIGS = [
    (medium_steps, medium_lr, low_steps, low_lr, 0.01)
    for medium_steps in (4, 6, 8)
    for medium_lr in (0.06, 0.08)
    for low_steps in (2, 4, 6)
    for low_lr in (0.06,)
]


def reverse_kl(current, original):
    current_grouped = current.reshape(len(current), -1, 8)
    original_grouped = original.reshape(len(original), -1, 8)
    return (
        current_grouped
        * torch.log((current_grouped + 1e-12) / (original_grouped + 1e-12))
    ).sum(dim=-1).mean(dim=1).sum()


def correct_sequential(model, props, cache, config):
    medium_steps, medium_lr, low_steps, low_lr, anchor_weight = config
    high, base_medium, base_low = [value.squeeze(-1).detach() for value in cache.policies]

    medium_logits = torch.nn.Parameter(torch.log(base_medium.clamp_min(1e-12)))
    medium_optimizer = torch.optim.Adam([medium_logits], lr=float(medium_lr))
    for _ in range(int(medium_steps)):
        medium_optimizer.zero_grad(set_to_none=True)
        medium = correction.routed_policy(medium_logits, base_medium)
        fulfill = correction.predicted_fulfillment(
            model,
            props,
            cache,
            [high.unsqueeze(-1), medium.unsqueeze(-1), base_low.unsqueeze(-1)],
        )
        objective = fulfill[:, 1].sum() - float(anchor_weight) * reverse_kl(
            medium, base_medium
        )
        (-objective).backward()
        medium_optimizer.step()
    with torch.no_grad():
        medium = correction.routed_policy(medium_logits, base_medium).detach()

    low_logits = torch.nn.Parameter(torch.log(base_low.clamp_min(1e-12)))
    low_optimizer = torch.optim.Adam([low_logits], lr=float(low_lr))
    for _ in range(int(low_steps)):
        low_optimizer.zero_grad(set_to_none=True)
        low = correction.routed_policy(low_logits, base_low)
        fulfill = correction.predicted_fulfillment(
            model,
            props,
            cache,
            [high.unsqueeze(-1), medium.unsqueeze(-1), low.unsqueeze(-1)],
        )
        objective = fulfill[:, 2].sum() - float(anchor_weight) * reverse_kl(low, base_low)
        (-objective).backward()
        low_optimizer.step()
    with torch.no_grad():
        low = correction.routed_policy(low_logits, base_low).detach()
    return replace(cache, path_features=torch.cat([medium, low], dim=1))


def run_candidate(model, props, cache, baseline, config):
    active = base.gate(cache)
    started = time.perf_counter()
    candidate = correct_sequential(model, props, cache, config)
    candidate = base.apply_fallback(cache, candidate, active)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    value = base.evaluate_cache(model, props, cache, candidate, baseline)
    value.update(
        {
            "config": list(config),
            "active": int(active.sum().item()),
            "overlay_ms_per_snapshot": 1000.0 * seconds / len(cache),
        }
    )
    return base.public(value)


def feasible(rows):
    return all(
        row["delta"]["Medium.norm_fulfill_mean"] > 0
        and row["delta"]["Medium.norm_fulfill_p1"] > -0.002
        and row["delta"]["Medium.norm_fulfill_p10"] > -0.002
        and row["delta"]["Low.norm_fulfill_mean"] >= -0.003
        and row["delta"]["Low.norm_fulfill_p1"] >= -0.01
        and row["delta"]["Low.norm_fulfill_p10"] >= -0.01
        for row in rows
    )


def main():
    runtime.set_seed(20260823)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contexts = {}
    for load in (2, 3):
        model, props, _ = screen.load_backbone(load, device)
        caches = {
            name: runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            for name, bounds in DISCOVERY.items()
        }
        baselines = {}
        for name, cache in caches.items():
            _, baselines[name] = trainer.evaluate(model, props, cache)
        contexts[load] = (model, props, caches, baselines)

    trials = []
    for config in CONFIGS:
        loads = {}
        all_rows = []
        for load, (model, props, caches, baselines) in contexts.items():
            rows = {
                name: run_candidate(model, props, caches[name], baselines[name], config)
                for name in DISCOVERY
            }
            loads[str(load)] = rows
            all_rows.extend(rows.values())
        valid = feasible(all_rows)
        robust_medium = min(
            row["delta"]["Medium.norm_fulfill_mean"] for row in all_rows
        )
        mean_medium = sum(
            row["delta"]["Medium.norm_fulfill_mean"] for row in all_rows
        ) / len(all_rows)
        trials.append(
            {
                "config": list(config),
                "feasible": valid,
                "score": robust_medium + 0.25 * mean_medium,
                "loads": loads,
            }
        )

    selected = sorted(
        (row for row in trials if row["feasible"]),
        key=lambda row: row["score"],
        reverse=True,
    )[:4]
    full_results = []
    for selected_row in selected:
        config = tuple(selected_row["config"])
        load_values = {}
        for load in (2, 3):
            model, props, _, _ = contexts[load]
            split_values = {}
            for name, bounds in FULL.items():
                cache = runtime.build_policy_cache(model, props, *bounds, batch_size=32)
                _, baseline_summary = trainer.evaluate(model, props, cache)
                split_values[name] = run_candidate(
                    model, props, cache, baseline_summary, config
                )
            load_values[str(load)] = split_values
        full_results.append({"config": list(config), "loads": load_values})
        print(f"config={config}", flush=True)
        for load, values in load_values.items():
            for split, value in values.items():
                d = value["delta"]
                print(
                    f"  {load}x/{split} M={d['Medium.norm_fulfill_mean']:+.6f} "
                    f"L={d['Low.norm_fulfill_mean']:+.6f} "
                    f"ms={value['overlay_ms_per_snapshot']:.3f}",
                    flush=True,
                )
    report = {
        "method": "strict-ESM priority-decoupled sequential correction",
        "actual_tm_used_for_policy": False,
        "discovery": DISCOVERY,
        "full": FULL,
        "trials": trials,
        "selected": selected,
        "full_results": full_results,
    }
    path = HERE / "sequential_screen.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
