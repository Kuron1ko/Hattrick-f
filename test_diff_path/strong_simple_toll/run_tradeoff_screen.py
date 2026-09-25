from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "simple_prototype_toll"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


screen = load_module("strong_simple_base", BASE / "run_screen.py")
trainer = screen.trainer
runtime = screen.runtime
correction = screen.method.trainer.probe.correction

SPLITS = {"safety": (318, 350), "validation": (350, 400)}
KNN_CONFIGS = [
    (1, 8, 0.75, 0.50),
    (1, 16, 1.00, 0.75),
    (2, 16, 0.75, 0.50),
    (2, 32, 1.00, 1.00),
    (2, 32, 1.25, 1.00),
    (4, 32, 1.00, 1.00),
]
ONLINE_CONFIGS = [
    (1, 0.06, 0.25, 0.01),
    (2, 0.06, 0.25, 0.01),
    (4, 0.06, 0.25, 0.01),
    (8, 0.06, 0.25, 0.01),
    (12, 0.06, 0.25, 0.01),
    (24, 0.06, 0.25, 0.01),
]
GATE = 0.70


def compact(summary):
    return trainer.probe.compact(summary)


def delta(summary, baseline):
    return trainer.probe.gaps(summary, baseline)


def gate(cache) -> torch.Tensor:
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    capacities = cache.capacities.to(dtype=torch.float32).clamp_min(1e-9)
    total_path_flow = sum(
        policy.squeeze(-1).to(dtype=torch.float32)
        * demand.squeeze(-1).to(dtype=torch.float32)
        for policy, demand in zip(cache.policies, cache.predicted_tms)
    )
    total = torch.sparse.mm(pte.t(), total_path_flow.t()).t() / capacities
    return total.mean(dim=1) >= GATE


def base_path_features(cache):
    return torch.cat(
        [cache.policies[1].squeeze(-1), cache.policies[2].squeeze(-1)], dim=1
    )


def apply_fallback(cache, candidate, active):
    return replace(
        candidate,
        path_features=torch.where(
            active[:, None], candidate.path_features, base_path_features(cache)
        ),
    )


def evaluate_cache(model, props, cache, candidate, baseline):
    rows, summary = trainer.evaluate(model, props, cache, candidate)
    return {"summary": compact(summary), "delta": delta(summary, baseline), "rows": rows}


def run_knn(model, props, cache, baseline, library, config):
    medium_k, low_k, medium_gain, low_gain = config
    active = gate(cache)
    started = time.perf_counter()
    features = trainer.edge_features(cache)
    tolls, _ = screen.method.predict(library, features, medium_k, low_k)
    tolls[:, 0].mul_(medium_gain)
    tolls[:, 1].mul_(low_gain)
    candidate = trainer.route_with_tolls(cache, tolls, 1.0)
    candidate = apply_fallback(cache, candidate, active)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    overlay_seconds = time.perf_counter() - started
    result = evaluate_cache(model, props, cache, candidate, baseline)
    result.update(
        {
            "config": list(config),
            "active": int(active.sum().item()),
            "overlay_ms_per_snapshot": 1000.0 * overlay_seconds / len(cache),
        }
    )
    return result


def run_online(model, props, cache, baseline, config):
    active = gate(cache)
    started = time.perf_counter()
    candidate = correction.correct_cache(model, props, cache, *config)
    candidate = apply_fallback(cache, candidate, active)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    overlay_seconds = time.perf_counter() - started
    result = evaluate_cache(model, props, cache, candidate, baseline)
    result.update(
        {
            "config": list(config),
            "active": int(active.sum().item()),
            "overlay_ms_per_snapshot": 1000.0 * overlay_seconds / len(cache),
        }
    )
    return result


def public(value):
    return {key: item for key, item in value.items() if key != "rows"}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    runtime.set_seed(20260823)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "strict_esm": True,
        "actual_tm_used_for_policy": False,
        "gate": GATE,
        "device": str(device),
        "loads": {},
    }
    for load_factor in (2, 3):
        print(f"[{load_factor}x] build", flush=True)
        model, props, library = screen.load_backbone(load_factor, device)
        load_report = {"splits": {}}
        for split, bounds in SPLITS.items():
            cache = runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            _, baseline = trainer.evaluate(model, props, cache)
            knn_rows = []
            for config in KNN_CONFIGS:
                value = run_knn(model, props, cache, baseline, library, config)
                knn_rows.append(public(value))
                d = value["delta"]
                print(
                    f"[{load_factor}x/{split}] knn={config} "
                    f"dM={d['Medium.norm_fulfill_mean']:+.5f} "
                    f"dL={d['Low.norm_fulfill_mean']:+.5f} "
                    f"ms={value['overlay_ms_per_snapshot']:.3f}",
                    flush=True,
                )
            online_rows = []
            for config in ONLINE_CONFIGS:
                value = run_online(model, props, cache, baseline, config)
                online_rows.append(public(value))
                d = value["delta"]
                print(
                    f"[{load_factor}x/{split}] online={config[0]} "
                    f"dM={d['Medium.norm_fulfill_mean']:+.5f} "
                    f"dL={d['Low.norm_fulfill_mean']:+.5f} "
                    f"ms={value['overlay_ms_per_snapshot']:.3f}",
                    flush=True,
                )
            load_report["splits"][split] = {
                "baseline": compact(baseline),
                "knn": knn_rows,
                "online": online_rows,
            }
        report["loads"][str(load_factor)] = load_report
    path = HERE / "tradeoff_screen.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
