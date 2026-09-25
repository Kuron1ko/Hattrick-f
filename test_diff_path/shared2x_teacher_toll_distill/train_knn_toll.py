from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_module("knn_toll_trainer", "train_linear_toll.py")
large = load_module("knn_toll_large", "run_large_linear_toll.py")
probe = trainer.probe
runtime = trainer.runtime


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def fit_library(features: torch.Tensor, targets: torch.Tensor) -> dict:
    # Channel-wise normalization preserves edge identity but avoids domination
    # by feature channels with larger numerical scales.
    mean = features.mean(dim=(0, 1), keepdim=True)
    std = features.std(dim=(0, 1), keepdim=True).clamp_min(1e-3)
    return {
        "mean": mean,
        "std": std,
        "features": ((features - mean) / std).detach(),
        "targets": targets.detach(),
    }


def predict(library: dict, query: torch.Tensor, neighbors: int):
    normalized = (query - library["mean"]) / library["std"]
    # Mean squared distance over the complete topology state.
    distance = (
        normalized[:, None] - library["features"][None]
    ).square().mean(dim=(2, 3))
    values, indices = torch.topk(
        distance, k=min(int(neighbors), distance.shape[1]), largest=False, dim=1
    )
    selected = library["targets"][indices]
    tolls = selected.mean(dim=1)
    return tolls, {
        "nearest_distance_mean": float(values[:, 0].mean().item()),
        "kth_distance_mean": float(values[:, -1].mean().item()),
        "toll_abs_mean": float(tolls.abs().mean().item()),
        "toll_abs_max": float(tolls.abs().max().item()),
    }


def evaluate(model, props, cache, library: dict, neighbors: int):
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    features = trainer.edge_features(cache)
    tolls, retrieval = predict(library, features, neighbors)
    candidate = trainer.route_with_tolls(cache, tolls, 1.0)
    rows, summary = trainer.evaluate(model, props, cache, candidate)
    result = {
        "neighbors": neighbors,
        "summary": probe.compact(summary),
        "baseline": probe.compact(baseline_summary),
        "delta": probe.gaps(summary, baseline_summary),
        "bootstrap": large.paired_bootstrap(baseline_rows, rows),
        "retrieval": retrieval,
    }
    return result


def safe(result: dict) -> bool:
    delta = result["delta"]
    return (
        result["summary"]["High"]["norm_fulfill_mean"] >= 0.995
        and delta["Medium.norm_fulfill_mean"] > 0.0
        and delta["Medium.norm_fulfill_p1"] > 0.0
        and delta["Medium.norm_fulfill_p10"] > 0.0
        and delta["Low.norm_fulfill_mean"] >= -0.003
        and delta["Low.norm_fulfill_p1"] >= -0.01
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )


def medium_score(result: dict) -> float:
    delta = result["delta"]
    return min(
        delta["Medium.norm_fulfill_mean"],
        delta["Medium.norm_fulfill_p1"],
        delta["Medium.norm_fulfill_p10"],
    )


def cpu_library(library: dict) -> dict:
    return {key: value.detach().cpu() for key, value in library.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = (
        {"train": (0, 128), "safety": (128, 160), "validation": (160, 200), "evaluation": (200, 250)}
        if args.level == 2
        else {"train": (0, 318), "safety": (318, 350), "validation": (350, 400), "evaluation": (400, 500)}
    )
    props = runtime.build_props(args.level, device)
    model, checkpoint = runtime.load_backbone(args.level, 490, props, device)
    train_cache = runtime.build_policy_cache(
        model, props, *splits["train"], batch_size=32
    )
    train_features = trainer.edge_features(train_cache)
    train_targets = trainer.teacher_tolls(model, props, train_cache)
    library = fit_library(train_features, train_targets)
    safety_cache = runtime.build_policy_cache(
        model, props, *splits["safety"], batch_size=32
    )

    if args.level == 2:
        safety_trials = [
            evaluate(model, props, safety_cache, library, neighbors)
            for neighbors in (1, 2, 4, 8, 16, 32)
        ]
    else:
        small_report = json.loads(
            (HERE / "artifacts" / "knn_toll_level2" / "report.json").read_text(
                encoding="utf-8"
            )
        )
        safety_trials = [
            evaluate(
                model,
                props,
                safety_cache,
                library,
                int(small_report["selected_neighbors"]),
            )
        ]
    for result in safety_trials:
        print(
            f"k={result['neighbors']} safe={safe(result)} "
            f"M={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
            f"L={result['delta']['Low.norm_fulfill_mean']:+.6f}",
            flush=True,
        )
    eligible = [result for result in safety_trials if safe(result)]
    selected = max(eligible, key=medium_score) if eligible else None
    validation = None
    evaluation = None
    if selected:
        validation_cache = runtime.build_policy_cache(
            model, props, *splits["validation"], batch_size=32
        )
        validation = evaluate(
            model, props, validation_cache, library, selected["neighbors"]
        )
    if validation is not None and safe(validation):
        evaluation_cache = runtime.build_policy_cache(
            model, props, *splits["evaluation"], batch_size=32
        )
        evaluation = evaluate(
            model, props, evaluation_cache, library, selected["neighbors"]
        )

    artifact_dir = HERE / "artifacts" / f"knn_toll_level{args.level}"
    payload = {
        "method": "convex-hull local case retrieval for teacher edge tolls",
        "level": args.level,
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "historical_actual_tm_used_for_training": False,
        "architecture": {
            "stored_cases": len(train_cache),
            "state_features": int(train_features.shape[1] * train_features.shape[2]),
            "outputs": "Medium and Low toll per directed edge",
            "high_policy": "exact Hattrick bypass",
            "bounded_output": "uniform average of historical teacher tolls",
        },
        "checkpoint": str(checkpoint),
        "splits": splits,
        "safety_trials": safety_trials,
        "selected_neighbors": selected["neighbors"] if selected else None,
        "validation": validation,
        "evaluation": evaluation,
        "pass": bool(evaluation and safe(evaluation)),
        "seconds": time.perf_counter() - started,
    }
    write_json(artifact_dir / "report.json", payload)
    torch.save(
        {
            "library": cpu_library(library),
            "neighbors": selected["neighbors"] if selected else None,
            "checkpoint": str(checkpoint),
        },
        artifact_dir / "model.pt",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
