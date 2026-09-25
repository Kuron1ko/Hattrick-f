from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "run_experiment.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("shared1x_dual_gate_base", SOURCE)
method = base.method
trainer = base.trainer
runtime = base.runtime


def route_dual(
    cache,
    tolls: torch.Tensor,
    medium_gate: torch.Tensor,
    low_gate: torch.Tensor,
):
    gated = tolls.clone()
    gated[:, 0] = gated[:, 0] * medium_gate[:, None]
    gated[:, 1] = gated[:, 1] * low_gate[:, None]
    return trainer.route_with_tolls(cache, gated, 1.0)


def class_delta(
    baseline_rows: list[dict], candidate_rows: list[dict], class_name: str
) -> torch.Tensor:
    baseline = base.index_rows(baseline_rows)
    candidate = base.index_rows(candidate_rows)
    snapshots = sorted(snapshot for snapshot, cls in baseline if cls == class_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(
        [
            candidate[(snapshot, class_name)]["norm_fulfill"]
            - baseline[(snapshot, class_name)]["norm_fulfill"]
            for snapshot in snapshots
        ],
        dtype=torch.float32,
        device=device,
    )


def stats(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean().item()),
        "positive_fraction": float(
            (values > 0).to(dtype=torch.float32).mean().item()
        ),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def learn_dual_advantages(model, props, cache, features, library):
    tolls, _ = base.retrieve_tolls(library, features, exclude_self=True)
    zeros = torch.zeros(len(cache), device=features.device)
    ones = torch.ones(len(cache), device=features.device)
    baseline_rows, _ = trainer.evaluate(model, props, cache)
    medium_rows, _ = trainer.evaluate(
        model, props, cache, route_dual(cache, tolls, ones, zeros)
    )
    low_rows, _ = trainer.evaluate(
        model, props, cache, route_dual(cache, tolls, zeros, ones)
    )
    medium = class_delta(baseline_rows, medium_rows, "Medium")
    low = class_delta(baseline_rows, low_rows, "Low")
    return medium, low, {"Medium": stats(medium), "Low": stats(low)}


def make_gate(
    mode: dict,
    distance: torch.Tensor,
    advantages: torch.Tensor,
    count: int,
    device: torch.device,
):
    if mode["kind"] == "off":
        return torch.zeros(count, device=device), {"gate_fraction": 0.0}
    if mode["kind"] == "all":
        return torch.ones(count, device=device), {"gate_fraction": 1.0}
    return base.advantage_gate(distance, advantages, int(mode["neighbors"]))


def strict_pareto(result: dict) -> bool:
    delta = result["delta"]
    return (
        result["summary"]["High"]["norm_fulfill_mean"] >= 0.995
        and all(
            delta[f"{class_name}.norm_fulfill_{metric}"] >= -1e-6
            for class_name in ("Medium", "Low")
            for metric in ("mean", "p1", "p10")
        )
        and (
            delta["Medium.norm_fulfill_mean"] > 1e-6
            or delta["Low.norm_fulfill_mean"] > 1e-6
        )
    )


def rank(result: dict) -> tuple[float, ...]:
    delta = result["delta"]
    medium = [
        delta[f"Medium.norm_fulfill_{metric}"]
        for metric in ("mean", "p1", "p10")
    ]
    low = [
        delta[f"Low.norm_fulfill_{metric}"]
        for metric in ("mean", "p1", "p10")
    ]
    return min(medium + low), sum(medium), sum(low)


def evaluate_mode(
    model,
    props,
    cache,
    features,
    library,
    medium_advantages,
    low_advantages,
    medium_mode: dict,
    low_mode: dict,
):
    tolls, distance = base.retrieve_tolls(library, features)
    medium_gate, medium_stats = make_gate(
        medium_mode, distance, medium_advantages, len(cache), features.device
    )
    low_gate, low_stats = make_gate(
        low_mode, distance, low_advantages, len(cache), features.device
    )
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    candidate_rows, candidate_summary = trainer.evaluate(
        model,
        props,
        cache,
        route_dual(cache, tolls, medium_gate, low_gate),
    )
    result = {
        "medium_mode": medium_mode,
        "low_mode": low_mode,
        "medium_gate": medium_stats,
        "low_gate": low_stats,
        "summary": trainer.probe.compact(candidate_summary),
        "baseline": trainer.probe.compact(baseline_summary),
        "delta": trainer.probe.gaps(candidate_summary, baseline_summary),
        "bootstrap": method.large.paired_bootstrap(
            baseline_rows, candidate_rows
        ),
    }
    result["strict_pareto"] = strict_pareto(result)
    return result, baseline_rows, candidate_rows


def main() -> None:
    runtime.set_seed(20260825)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props = base.load_backbone(device)

    medium_modes = [
        {"kind": "off"},
        {"kind": "advantage_knn", "neighbors": 4},
        {"kind": "advantage_knn", "neighbors": 8},
        {"kind": "advantage_knn", "neighbors": 16},
        {"kind": "advantage_knn", "neighbors": 32},
        {"kind": "all"},
    ]
    low_modes = [
        {"kind": "off"},
        {"kind": "advantage_knn", "neighbors": 4},
        {"kind": "advantage_knn", "neighbors": 8},
        {"kind": "advantage_knn", "neighbors": 16},
        {"kind": "advantage_knn", "neighbors": 32},
        {"kind": "all"},
    ]

    print("[dual Level-2] building LOO action-advantage labels", flush=True)
    train, train_features, library = base.build_library(
        model, props, base.SMALL_SPLITS["train"]
    )
    medium_advantages, low_advantages, advantage_stats = learn_dual_advantages(
        model, props, train, train_features, library
    )
    safety = runtime.build_policy_cache(
        model, props, *base.SMALL_SPLITS["safety"], batch_size=32
    )
    safety_features = trainer.edge_features(safety)
    trials = []
    for medium_mode in medium_modes:
        for low_mode in low_modes:
            result, _, _ = evaluate_mode(
                model,
                props,
                safety,
                safety_features,
                library,
                medium_advantages,
                low_advantages,
                medium_mode,
                low_mode,
            )
            trials.append(result)
    eligible = [result for result in trials if result["strict_pareto"]]
    selected = max(eligible, key=rank) if eligible else None
    print(
        f"[dual Level-2] eligible={len(eligible)} selected="
        f"{None if selected is None else (selected['medium_mode'], selected['low_mode'])}",
        flush=True,
    )

    small_validation = None
    small_evaluation = None
    if selected:
        for split_name in ("validation", "evaluation"):
            cache = runtime.build_policy_cache(
                model, props, *base.SMALL_SPLITS[split_name], batch_size=32
            )
            result, _, _ = evaluate_mode(
                model,
                props,
                cache,
                trainer.edge_features(cache),
                library,
                medium_advantages,
                low_advantages,
                selected["medium_mode"],
                selected["low_mode"],
            )
            result["range"] = list(base.SMALL_SPLITS[split_name])
            if split_name == "validation":
                small_validation = result
            else:
                small_evaluation = result
            print(
                f"[dual Level-2 {split_name}] pareto={result['strict_pareto']} "
                f"dM={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
                f"dL={result['delta']['Low.norm_fulfill_mean']:+.6f}",
                flush=True,
            )

    large = None
    if selected and small_validation and small_validation["strict_pareto"]:
        print("[dual Level-4] retraining with frozen gate modes", flush=True)
        train, train_features, library = base.build_library(
            model, props, base.LARGE_SPLITS["train"]
        )
        medium_advantages, low_advantages, large_advantage_stats = (
            learn_dual_advantages(model, props, train, train_features, library)
        )
        results = {}
        final_baseline = []
        final_candidate = []
        for split_name in ("safety", "validation", "evaluation"):
            cache = runtime.build_policy_cache(
                model, props, *base.LARGE_SPLITS[split_name], batch_size=32
            )
            result, baseline_rows, candidate_rows = evaluate_mode(
                model,
                props,
                cache,
                trainer.edge_features(cache),
                library,
                medium_advantages,
                low_advantages,
                selected["medium_mode"],
                selected["low_mode"],
            )
            result["range"] = list(base.LARGE_SPLITS[split_name])
            results[split_name] = result
            if split_name == "evaluation":
                final_baseline, final_candidate = baseline_rows, candidate_rows
            print(
                f"[dual Level-4 {split_name}] pareto={result['strict_pareto']} "
                f"gM={result['medium_gate']['gate_fraction']:.3f} "
                f"gL={result['low_gate']['gate_fraction']:.3f} "
                f"dM={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
                f"dL={result['delta']['Low.norm_fulfill_mean']:+.6f}",
                flush=True,
            )
        large = {
            "advantage_stats": large_advantage_stats,
            "safety": results["safety"],
            "validation": results["validation"],
            "evaluation": results["evaluation"],
            "development_strict_pareto": bool(
                results["safety"]["strict_pareto"]
                and results["validation"]["strict_pareto"]
            ),
            "final_strict_pareto": bool(results["evaluation"]["strict_pareto"]),
            "rows": {
                "baseline": base.compact_rows(final_baseline),
                "candidate": base.compact_rows(final_candidate),
            },
        }
        torch.save(
            {
                "library": method.knn.cpu_library(library),
                "medium_advantages": medium_advantages.detach().cpu(),
                "low_advantages": low_advantages.detach().cpu(),
                "medium_mode": selected["medium_mode"],
                "low_mode": selected["low_mode"],
                "checkpoint": str(base.CHECKPOINT),
            },
            HERE / "dual_gate_model.pt",
        )

    payload = {
        "method": "strict-ESM dual advantage-gated local case retrieval",
        "hypothesis": "Separate offline contextual gates can reject harmful Medium and Low toll actions without observing current actual traffic.",
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "offline_actual_tm_use": "training-split action-advantage labels and evaluation only",
        "small": {
            "splits": {key: list(value) for key, value in base.SMALL_SPLITS.items()},
            "advantage_stats": advantage_stats,
            "trials": trials,
            "selected": None
            if selected is None
            else {
                "medium_mode": selected["medium_mode"],
                "low_mode": selected["low_mode"],
            },
            "validation": small_validation,
            "evaluation": small_evaluation,
        },
        "large": large,
        "seconds": time.perf_counter() - started,
    }
    base.write_json(HERE / "dual_gate_report.json", payload)
    print(
        json.dumps(
            {
                "selected": payload["small"]["selected"],
                "large_summary": None
                if large is None
                else {
                    "development_strict_pareto": large[
                        "development_strict_pareto"
                    ],
                    "final_strict_pareto": large["final_strict_pareto"],
                    "evaluation": large["evaluation"],
                },
                "seconds": payload["seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
