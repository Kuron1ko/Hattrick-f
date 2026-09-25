from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
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


screen = load_module("block_prototype_screen", HERE / "run_screen.py")
trainer = screen.trainer
runtime = screen.runtime
BLOCKS = ((318, 338), (338, 358), (358, 378), (378, 400))
CANDIDATES = (
    {"name": "P3", "prototypes": 3, "medium_gain": 0.20, "low_gain": 0.40},
    {"name": "P4", "prototypes": 4, "medium_gain": 0.125, "low_gain": 0.25},
    {"name": "P8", "prototypes": 8, "medium_gain": 0.25, "low_gain": 0.25},
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def paired_mean(rows_a: list[dict], rows_b: list[dict], seed: int) -> dict:
    baseline = {(row["snapshot"], row["class"]): row["norm_fulfill"] for row in rows_a}
    grouped: dict[str, list[float]] = {name: [] for name in ("High", "Medium", "Low")}
    for row in rows_b:
        key = (row["snapshot"], row["class"])
        grouped[row["class"]].append(float(row["norm_fulfill"] - baseline[key]))
    rng = np.random.default_rng(seed)
    result = {}
    for class_name, values in grouped.items():
        vector = np.asarray(values, dtype=np.float64)
        indices = rng.integers(0, len(vector), size=(4000, len(vector)))
        means = vector[indices].mean(axis=1)
        result[class_name] = {
            "mean": float(vector.mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "positive_fraction": float((vector > 0).mean()),
        }
    return result


def evaluate(model, props, cache, library, prototypes, config, seed):
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    query = trainer.edge_features(cache)
    tolls = screen.predict(library, prototypes, query).clone()
    tolls[:, 0].mul_(config["medium_gain"])
    tolls[:, 1].mul_(config["low_gain"])
    routed = trainer.route_with_tolls(cache, tolls, 1.0)
    candidate_rows, summary = trainer.evaluate(model, props, cache, routed)
    return {
        "summary": trainer.probe.compact(summary),
        "delta": trainer.probe.gaps(summary, baseline_summary),
        "paired_mean": paired_mean(baseline_rows, candidate_rows, seed),
    }


def main() -> None:
    torch.manual_seed(20260828)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    payload = {
        "method": "blocked temporal robustness for compressed edge-toll prototypes",
        "strict_esm_inference": True,
        "actual_tm_used_for_policy": False,
        "evaluation_400_500_used": False,
        "blocks": [list(value) for value in BLOCKS],
        "candidates": list(CANDIDATES),
        "loads": {},
    }
    for load_factor in (2, 3):
        model, props, library = screen.load_backbone(load_factor, device)
        fitted = {
            config["name"]: screen.fit_prototypes(library, config["prototypes"])
            for config in CANDIDATES
        }
        load_results = {}
        for config in CANDIDATES:
            block_results = []
            for index, bounds in enumerate(BLOCKS):
                cache = runtime.build_policy_cache(model, props, *bounds, batch_size=32)
                result = evaluate(
                    model,
                    props,
                    cache,
                    library,
                    fitted[config["name"]],
                    config,
                    seed=20260828 + load_factor * 100 + index,
                )
                result["range"] = list(bounds)
                block_results.append(result)
                delta = result["delta"]
                print(
                    f"[{load_factor}x/{config['name']}/{bounds[0]}:{bounds[1]}] "
                    f"dM={delta['Medium.norm_fulfill_mean']:+.6f} "
                    f"dL={delta['Low.norm_fulfill_mean']:+.6f}",
                    flush=True,
                )
            load_results[config["name"]] = block_results
        payload["loads"][str(load_factor)] = load_results
        del model, props, library, fitted
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload["seconds"] = time.perf_counter() - started
    write_json(HERE / "block_robustness_report.json", payload)
    print(json.dumps({"artifact": str(HERE / 'block_robustness_report.json'), "seconds": payload["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
