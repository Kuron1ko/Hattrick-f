from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
ROOT = TEST_DIR.parent
SOURCE = TEST_DIR / "shared2x_teacher_toll_distill" / "train_asymmetric_knn_toll.py"
TOPOLOGY = "geant_priomask500_shared"
CHECKPOINT = ROOT / "hattrick_geant_priomask500_shared_8sp.pkl"
MEDIUM_NEIGHBORS = 2
LOW_NEIGHBORS = 32
SMALL_SPLITS = {
    "train": (0, 128),
    "safety": (128, 160),
    "validation": (160, 200),
    "evaluation": (200, 250),
}
LARGE_SPLITS = {
    "train": (0, 318),
    "safety": (318, 350),
    "validation": (350, 400),
    "evaluation": (400, 500),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


method = load_module("shared1x_advantage_gate_method", SOURCE)
trainer = method.trainer
runtime = method.runtime


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_backbone(device: torch.device):
    runtime.shared.TOPOLOGY = TOPOLOGY
    props = runtime.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    model = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model = model.to(device=device, dtype=props.dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    trainer.runtime = runtime
    trainer.probe.runtime = runtime
    method.runtime = runtime
    return model, props


def distances(library: dict, query: torch.Tensor) -> torch.Tensor:
    normalized = (query - library["mean"]) / library["std"]
    return (normalized[:, None] - library["features"][None]).square().mean(
        dim=(2, 3)
    )


def retrieve_tolls(
    library: dict,
    query: torch.Tensor,
    exclude_self: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    distance = distances(library, query)
    if exclude_self:
        if distance.shape[0] != distance.shape[1]:
            raise ValueError("Leave-one-out retrieval requires aligned queries")
        distance = distance.clone()
        distance.fill_diagonal_(float("inf"))
    maximum = min(LOW_NEIGHBORS, distance.shape[1] - int(exclude_self))
    _, indices = torch.topk(distance, k=maximum, largest=False, dim=1)
    targets = library["targets"][indices]
    medium = targets[:, : min(MEDIUM_NEIGHBORS, maximum), 0].mean(dim=1)
    low = targets[:, :maximum, 1].mean(dim=1)
    return torch.stack([medium, low], dim=1), distance


def advantage_gate(
    distance: torch.Tensor,
    advantages: torch.Tensor,
    neighbors: int,
    exclude_self: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    if exclude_self:
        distance = distance.clone()
        distance.fill_diagonal_(float("inf"))
    k = min(int(neighbors), distance.shape[1] - int(exclude_self))
    values, indices = torch.topk(distance, k=k, largest=False, dim=1)
    local = advantages[indices]
    mean = local.mean(dim=1)
    standard_error = local.std(dim=1, unbiased=False) / (float(k) ** 0.5)
    # A one-standard-error lower confidence bound is a conservative offline
    # contextual-policy gate. It refuses the Medium action unless similar
    # historical states consistently benefited from it.
    score = mean - standard_error
    gate = (score > 0.0).to(dtype=torch.float32)
    return gate, {
        "gate_fraction": float(gate.mean().item()),
        "score_mean": float(score.mean().item()),
        "score_min": float(score.min().item()),
        "score_max": float(score.max().item()),
        "nearest_distance_mean": float(values[:, 0].mean().item()),
    }


def route(cache, tolls: torch.Tensor, medium_gate: torch.Tensor):
    gated = tolls.clone()
    gated[:, 0] = gated[:, 0] * medium_gate[:, None]
    # Low remains fully active. It is admitted after High and Medium, so this
    # cannot change their outcomes in the sequential simulator.
    return trainer.route_with_tolls(cache, gated, 1.0)


def index_rows(rows: list[dict]) -> dict[tuple[int, str], dict]:
    return {(int(row["snapshot"]), str(row["class"])): row for row in rows}


def medium_advantages(baseline_rows: list[dict], candidate_rows: list[dict]) -> torch.Tensor:
    baseline = index_rows(baseline_rows)
    candidate = index_rows(candidate_rows)
    snapshots = sorted(snapshot for snapshot, cls in baseline if cls == "Medium")
    return torch.tensor(
        [
            float(candidate[(snapshot, "Medium")]["norm_fulfill"])
            - float(baseline[(snapshot, "Medium")]["norm_fulfill"])
            for snapshot in snapshots
        ],
        dtype=torch.float32,
        device=next(iter(candidate_rows))["norm_fulfill"].device
        if torch.is_tensor(next(iter(candidate_rows))["norm_fulfill"])
        else None,
    )


def compact_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "snapshot": int(row["snapshot"]),
            "class": str(row["class"]),
            "norm_fulfill": float(row["norm_fulfill"]),
        }
        for row in rows
    ]


def evaluate(
    model,
    props,
    cache,
    tolls: torch.Tensor,
    medium_gate: torch.Tensor,
    gate_stats: dict[str, float],
) -> tuple[dict, list[dict], list[dict]]:
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    candidate_cache = route(cache, tolls, medium_gate)
    candidate_rows, candidate_summary = trainer.evaluate(
        model, props, cache, candidate_cache
    )
    delta = trainer.probe.gaps(candidate_summary, baseline_summary)
    result = {
        "summary": trainer.probe.compact(candidate_summary),
        "baseline": trainer.probe.compact(baseline_summary),
        "delta": delta,
        "bootstrap": method.large.paired_bootstrap(baseline_rows, candidate_rows),
        "gate": gate_stats,
    }
    result["pareto_safe"] = pareto_safe(result)
    return result, baseline_rows, candidate_rows


def pareto_safe(result: dict) -> bool:
    delta = result["delta"]
    return (
        result["summary"]["High"]["norm_fulfill_mean"] >= 0.995
        and delta["Medium.norm_fulfill_mean"] >= -1e-6
        and delta["Medium.norm_fulfill_p1"] >= -1e-6
        and delta["Medium.norm_fulfill_p10"] >= -1e-6
        # Retain the original task's "no significant Low decline" budget on
        # development data. Confirmation still reports exact signed changes.
        and delta["Low.norm_fulfill_mean"] >= -0.003
        and delta["Low.norm_fulfill_p1"] >= -0.01
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )


def score(result: dict) -> tuple[float, ...]:
    delta = result["delta"]
    return (
        min(
            delta["Medium.norm_fulfill_mean"],
            delta["Medium.norm_fulfill_p1"],
            delta["Medium.norm_fulfill_p10"],
        ),
        delta["Medium.norm_fulfill_mean"],
        delta["Low.norm_fulfill_mean"],
    )


def build_library(model, props, split: tuple[int, int]):
    cache = runtime.build_policy_cache(model, props, *split, batch_size=32)
    features = trainer.edge_features(cache)
    targets = trainer.teacher_tolls(model, props, cache)
    return cache, features, method.knn.fit_library(features, targets)


def learn_advantages(model, props, cache, features, library):
    tolls, _ = retrieve_tolls(library, features, exclude_self=True)
    ones = torch.ones(len(cache), device=features.device)
    baseline_rows, _ = trainer.evaluate(model, props, cache)
    candidate_rows, _ = trainer.evaluate(model, props, cache, route(cache, tolls, ones))
    baseline = index_rows(baseline_rows)
    candidate = index_rows(candidate_rows)
    values = [
        candidate[(snapshot, "Medium")]["norm_fulfill"]
        - baseline[(snapshot, "Medium")]["norm_fulfill"]
        for snapshot in range(cache.source_start, cache.source_start + len(cache))
    ]
    tensor = torch.tensor(values, dtype=torch.float32, device=features.device)
    return tensor, {
        "mean": float(tensor.mean().item()),
        "positive_fraction": float((tensor > 0).to(torch.float32).mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def evaluate_modes(
    model,
    props,
    cache,
    features,
    library,
    advantages,
    modes: list[dict],
):
    tolls, distance = retrieve_tolls(library, features)
    results = []
    for mode in modes:
        if mode["kind"] == "medium_off":
            gate = torch.zeros(len(cache), device=features.device)
            gate_stats = {"gate_fraction": 0.0}
        elif mode["kind"] == "medium_all":
            gate = torch.ones(len(cache), device=features.device)
            gate_stats = {"gate_fraction": 1.0}
        else:
            gate, gate_stats = advantage_gate(
                distance, advantages, int(mode["neighbors"])
            )
        result, baseline_rows, candidate_rows = evaluate(
            model, props, cache, tolls, gate, gate_stats
        )
        result["mode"] = mode
        results.append((result, baseline_rows, candidate_rows))
    return results


def main() -> None:
    runtime.set_seed(20260824)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props = load_backbone(device)

    print("[Level-2] building strict-ESM library and LOO advantages", flush=True)
    small_train, small_features, small_library = build_library(
        model, props, SMALL_SPLITS["train"]
    )
    small_advantages, small_advantage_stats = learn_advantages(
        model, props, small_train, small_features, small_library
    )
    modes = [
        {"kind": "medium_off"},
        {"kind": "advantage_knn", "neighbors": 4},
        {"kind": "advantage_knn", "neighbors": 8},
        {"kind": "advantage_knn", "neighbors": 16},
        {"kind": "advantage_knn", "neighbors": 32},
        {"kind": "medium_all"},
    ]
    small_safety = runtime.build_policy_cache(
        model, props, *SMALL_SPLITS["safety"], batch_size=32
    )
    small_safety_features = trainer.edge_features(small_safety)
    safety_trials_raw = evaluate_modes(
        model,
        props,
        small_safety,
        small_safety_features,
        small_library,
        small_advantages,
        modes,
    )
    safety_trials = [item[0] for item in safety_trials_raw]
    for result in safety_trials:
        print(
            f"[Level-2 safety] {result['mode']} safe={result['pareto_safe']} "
            f"gate={result['gate']['gate_fraction']:.3f} "
            f"dM={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
            f"dL={result['delta']['Low.norm_fulfill_mean']:+.6f}",
            flush=True,
        )
    eligible = [result for result in safety_trials if result["pareto_safe"]]
    selected = max(eligible, key=score) if eligible else None
    small_validation_result = None
    small_evaluation_result = None
    if selected:
        for split_name in ("validation", "evaluation"):
            cache = runtime.build_policy_cache(
                model, props, *SMALL_SPLITS[split_name], batch_size=32
            )
            feature = trainer.edge_features(cache)
            result = evaluate_modes(
                model,
                props,
                cache,
                feature,
                small_library,
                small_advantages,
                [selected["mode"]],
            )[0][0]
            result["range"] = list(SMALL_SPLITS[split_name])
            if split_name == "validation":
                small_validation_result = result
            else:
                small_evaluation_result = result
            print(
                f"[Level-2 {split_name}] safe={result['pareto_safe']} "
                f"dM={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
                f"dL={result['delta']['Low.norm_fulfill_mean']:+.6f}",
                flush=True,
            )

    large_payload = None
    if selected and small_validation_result and small_validation_result["pareto_safe"]:
        print("[Level-4] retraining library with frozen gate rule", flush=True)
        large_train, large_features, large_library = build_library(
            model, props, LARGE_SPLITS["train"]
        )
        if selected["mode"]["kind"] == "advantage_knn":
            large_advantages, large_advantage_stats = learn_advantages(
                model, props, large_train, large_features, large_library
            )
        else:
            large_advantages = torch.zeros(
                len(large_train), device=large_features.device
            )
            large_advantage_stats = None
        large_results = {}
        final_baseline_rows = None
        final_candidate_rows = None
        for split_name in ("safety", "validation", "evaluation"):
            cache = runtime.build_policy_cache(
                model, props, *LARGE_SPLITS[split_name], batch_size=32
            )
            feature = trainer.edge_features(cache)
            result, baseline_rows, candidate_rows = evaluate_modes(
                model,
                props,
                cache,
                feature,
                large_library,
                large_advantages,
                [selected["mode"]],
            )[0]
            result["range"] = list(LARGE_SPLITS[split_name])
            large_results[split_name] = result
            if split_name == "evaluation":
                final_baseline_rows = baseline_rows
                final_candidate_rows = candidate_rows
            print(
                f"[Level-4 {split_name}] safe={result['pareto_safe']} "
                f"gate={result['gate']['gate_fraction']:.3f} "
                f"dM={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
                f"dL={result['delta']['Low.norm_fulfill_mean']:+.6f}",
                flush=True,
            )
        large_payload = {
            "advantage_stats": large_advantage_stats,
            "safety": large_results["safety"],
            "validation": large_results["validation"],
            "evaluation": large_results["evaluation"],
            "development_gate_pass": bool(
                large_results["safety"]["pareto_safe"]
                and large_results["validation"]["pareto_safe"]
            ),
            "final_gate_pass": bool(large_results["evaluation"]["pareto_safe"]),
            "rows": {
                "baseline": compact_rows(final_baseline_rows or []),
                "candidate": compact_rows(final_candidate_rows or []),
            },
        }
        torch.save(
            {
                "library": method.knn.cpu_library(large_library),
                "selected_mode": selected["mode"],
                "advantages": large_advantages.detach().cpu(),
                "checkpoint": str(CHECKPOINT),
            },
            HERE / "model.pt",
        )

    payload = {
        "method": "strict-ESM advantage-gated local case retrieval",
        "hypothesis": "At 1x, Medium tolls should be refused unless similar predicted states show positive offline advantage; Low tolls remain active because Low is admitted last.",
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "offline_actual_tm_use": "training-split advantage labels and evaluation only",
        "checkpoint": str(CHECKPOINT),
        "fixed_toll_retrieval": {
            "medium_neighbors": MEDIUM_NEIGHBORS,
            "low_neighbors": LOW_NEIGHBORS,
            "scale": 1.0,
        },
        "small": {
            "splits": {key: list(value) for key, value in SMALL_SPLITS.items()},
            "loo_advantage_stats": small_advantage_stats,
            "safety_trials": safety_trials,
            "selected_mode": selected["mode"] if selected else None,
            "validation": small_validation_result,
            "evaluation": small_evaluation_result,
        },
        "large": large_payload,
        "seconds": time.perf_counter() - started,
    }
    write_json(HERE / "report.json", payload)
    print(json.dumps({"selected": payload["small"]["selected_mode"], "large": large_payload, "seconds": payload["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
