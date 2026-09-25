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


trainer = load_module("asymmetric_knn_trainer", "train_linear_toll.py")
large = load_module("asymmetric_knn_large", "run_large_linear_toll.py")
knn = load_module("asymmetric_knn_base", "train_knn_toll.py")
probe = trainer.probe
runtime = trainer.runtime


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def predict(library: dict, query: torch.Tensor, medium_k: int, low_k: int):
    normalized = (query - library["mean"]) / library["std"]
    distance = (
        normalized[:, None] - library["features"][None]
    ).square().mean(dim=(2, 3))
    maximum = min(max(int(medium_k), int(low_k)), distance.shape[1])
    values, indices = torch.topk(distance, k=maximum, largest=False, dim=1)
    targets = library["targets"][indices]
    medium = targets[:, : min(int(medium_k), maximum), 0].mean(dim=1)
    low = targets[:, : min(int(low_k), maximum), 1].mean(dim=1)
    tolls = torch.stack([medium, low], dim=1)
    return tolls, {
        "nearest_distance_mean": float(values[:, 0].mean().item()),
        "farthest_used_distance_mean": float(values[:, -1].mean().item()),
        "toll_abs_mean": float(tolls.abs().mean().item()),
        "toll_abs_max": float(tolls.abs().max().item()),
    }


def evaluate(model, props, cache, library, medium_k: int, low_k: int):
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    tolls, retrieval = predict(
        library, trainer.edge_features(cache), medium_k, low_k
    )
    candidate = trainer.route_with_tolls(cache, tolls, 1.0)
    rows, summary = trainer.evaluate(model, props, cache, candidate)
    return {
        "medium_neighbors": medium_k,
        "low_neighbors": low_k,
        "summary": probe.compact(summary),
        "baseline": probe.compact(baseline_summary),
        "delta": probe.gaps(summary, baseline_summary),
        "bootstrap": large.paired_bootstrap(baseline_rows, rows),
        "retrieval": retrieval,
    }


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


def low_score(result: dict) -> float:
    delta = result["delta"]
    return min(
        delta["Low.norm_fulfill_mean"],
        delta["Low.norm_fulfill_p1"],
        delta["Low.norm_fulfill_p10"],
    )


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
    features = trainer.edge_features(train_cache)
    targets = trainer.teacher_tolls(model, props, train_cache)
    library = knn.fit_library(features, targets)
    safety_cache = runtime.build_policy_cache(
        model, props, *splits["safety"], batch_size=32
    )
    medium_k = 2
    if args.level == 2:
        low_choices = (2, 4, 8, 16, 32, 64)
    else:
        small = json.loads(
            (HERE / "artifacts" / "asymmetric_knn_level2" / "report.json").read_text(
                encoding="utf-8"
            )
        )
        low_choices = (int(small["selected_low_neighbors"]),)
    safety_trials = [
        evaluate(model, props, safety_cache, library, medium_k, low_k)
        for low_k in low_choices
    ]
    for result in safety_trials:
        print(
            f"kM={medium_k} kL={result['low_neighbors']} safe={safe(result)} "
            f"M={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
            f"L={result['delta']['Low.norm_fulfill_mean']:+.6f}",
            flush=True,
        )
    eligible = [result for result in safety_trials if safe(result)]
    selected = max(eligible, key=low_score) if eligible else None
    validation = None
    evaluation = None
    if selected:
        validation_cache = runtime.build_policy_cache(
            model, props, *splits["validation"], batch_size=32
        )
        validation = evaluate(
            model,
            props,
            validation_cache,
            library,
            medium_k,
            selected["low_neighbors"],
        )
    if validation is not None and safe(validation):
        evaluation_cache = runtime.build_policy_cache(
            model, props, *splits["evaluation"], batch_size=32
        )
        evaluation = evaluate(
            model,
            props,
            evaluation_cache,
            library,
            medium_k,
            selected["low_neighbors"],
        )

    artifact_dir = HERE / "artifacts" / f"asymmetric_knn_level{args.level}"
    payload = {
        "method": "priority-asymmetric local case retrieval for edge tolls",
        "level": args.level,
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "historical_actual_tm_used_for_training": False,
        "principle": "small neighborhood preserves Medium specificity; larger Low neighborhood reduces downstream toll variance",
        "checkpoint": str(checkpoint),
        "splits": splits,
        "medium_neighbors": medium_k,
        "safety_trials": safety_trials,
        "selected_low_neighbors": selected["low_neighbors"] if selected else None,
        "validation": validation,
        "evaluation": evaluation,
        "pass": bool(evaluation and safe(evaluation)),
        "seconds": time.perf_counter() - started,
    }
    write_json(artifact_dir / "report.json", payload)
    torch.save(
        {
            "library": knn.cpu_library(library),
            "medium_neighbors": medium_k,
            "low_neighbors": selected["low_neighbors"] if selected else None,
            "checkpoint": str(checkpoint),
        },
        artifact_dir / "model.pt",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
