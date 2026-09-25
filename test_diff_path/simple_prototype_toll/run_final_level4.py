from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

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


screen = load_module("final_prototype_screen", HERE / "run_screen.py")
trainer = screen.trainer
runtime = screen.runtime
PROTOTYPE_COUNT = 3
MEDIUM_GAIN = 0.20
LOW_GAIN = 0.40
ACTIVATION_THRESHOLD = 0.70
SPLITS = {
    "train": (0, 318),
    "safety": (318, 350),
    "validation": (350, 400),
    "evaluation": (400, 500),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def exact_fallback_route(cache, tolls: torch.Tensor, active: torch.Tensor):
    routed = trainer.route_with_tolls(cache, tolls, 1.0)
    base = torch.cat(
        [cache.policies[1].squeeze(-1), cache.policies[2].squeeze(-1)], dim=1
    )
    path_features = torch.where(active[:, None], routed.path_features, base)
    return replace(routed, path_features=path_features)


def base_path_features(cache) -> torch.Tensor:
    return torch.cat(
        [cache.policies[1].squeeze(-1), cache.policies[2].squeeze(-1)], dim=1
    )


def strict_esm_gate_and_features(cache):
    """Compute the scalar gate first; construct the full state only if active."""
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    capacities = cache.capacities.to(dtype=torch.float32).clamp_min(1e-9)
    path_flows = [
        policy.squeeze(-1).to(dtype=torch.float32)
        * demand.squeeze(-1).to(dtype=torch.float32)
        for policy, demand in zip(cache.policies, cache.predicted_tms)
    ]
    total = torch.sparse.mm(
        pte.t(), (path_flows[0] + path_flows[1] + path_flows[2]).t()
    ).t() / capacities
    total_mean = total.mean(dim=1)
    active = total_mean >= ACTIVATION_THRESHOLD
    if not bool(active.any().item()):
        return total_mean, active, None
    high = torch.sparse.mm(pte.t(), path_flows[0].t()).t() / capacities
    medium = torch.sparse.mm(pte.t(), path_flows[1].t()).t() / capacities
    # Reuse the gate projection so the active path needs three sparse products,
    # equal to the original feature extractor rather than four.
    low = total - high - medium
    features = torch.stack(
        [
            high,
            medium,
            low,
            high + medium,
            total,
            torch.relu(1.0 - high),
            torch.relu(1.0 - high - medium),
        ],
        dim=-1,
    ).detach()
    return total_mean, active, features


def construct_policy(cache, library: dict, prototypes: dict):
    total_mean, active, features = strict_esm_gate_and_features(cache)
    if features is None:
        return replace(cache, path_features=base_path_features(cache)), {
            "features": None,
            "tolls": torch.zeros(
                len(cache), 2, int(cache.capacities.shape[1]),
                device=cache.capacities.device,
                dtype=torch.float32,
            ),
            "active": active,
            "total_mean": total_mean,
        }
    tolls = screen.predict(library, prototypes, features).clone()
    tolls[:, 0].mul_(MEDIUM_GAIN)
    tolls[:, 1].mul_(LOW_GAIN)
    tolls.mul_(active[:, None, None])
    return exact_fallback_route(cache, tolls, active), {
        "features": features,
        "tolls": tolls,
        "active": active,
        "total_mean": total_mean,
    }


def compact_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "snapshot": int(row["snapshot"]),
            "class": str(row["class"]),
            "norm_fulfill": float(row["norm_fulfill"]),
        }
        for row in rows
    ]


def evaluate(model, props, cache, library, prototypes):
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    candidate_cache, policy = construct_policy(cache, library, prototypes)
    if not bool(policy["active"].any().item()):
        # The deployment rule returns Hattrick directly in the inactive regime;
        # do not pass an identical tensor through a second adapter/simulator path.
        candidate_rows = [dict(row) for row in baseline_rows]
        candidate_summary = [dict(row) for row in baseline_summary]
    else:
        candidate_rows, candidate_summary = trainer.evaluate(
            model, props, cache, candidate_cache
        )
    delta = trainer.probe.gaps(candidate_summary, baseline_summary)
    return {
        "baseline": trainer.probe.compact(baseline_summary),
        "summary": trainer.probe.compact(candidate_summary),
        "delta": delta,
        "bootstrap": screen.method.large.paired_bootstrap(
            baseline_rows, candidate_rows
        ),
        "active_count": int(policy["active"].sum().item()),
        "sample_count": len(cache),
        "total_mean_range": [
            float(policy["total_mean"].min().item()),
            float(policy["total_mean"].max().item()),
        ],
        "rows": {
            "baseline": compact_rows(baseline_rows),
            "candidate": compact_rows(candidate_rows),
        },
    }


def information_audit(cache, library, prototypes) -> dict:
    candidate, policy = construct_policy(cache, library, prototypes)
    counterfactual = replace(
        cache,
        tms=tuple(torch.zeros_like(value) for value in cache.tms),
    )
    cf_candidate, cf_policy = construct_policy(counterfactual, library, prototypes)
    if policy["features"] is None and cf_policy["features"] is None:
        feature_difference = 0.0
    elif policy["features"] is None or cf_policy["features"] is None:
        feature_difference = float("inf")
    else:
        feature_difference = float(
            (policy["features"] - cf_policy["features"]).abs().max().item()
        )
    return {
        "feature_max_abs_diff_after_zero_actual": feature_difference,
        "toll_max_abs_diff_after_zero_actual": float(
            (policy["tolls"] - cf_policy["tolls"]).abs().max().item()
        ),
        "activation_mismatch_after_zero_actual": int(
            (policy["active"] != cf_policy["active"]).sum().item()
        ),
        "policy_max_abs_diff_after_zero_actual": float(
            (candidate.path_features - cf_candidate.path_features).abs().max().item()
        ),
    }


def slice_cache(cache, count: int):
    return replace(
        cache,
        policies=tuple(value[:count] for value in cache.policies),
        tms=tuple(value[:count] for value in cache.tms),
        predicted_tms=tuple(value[:count] for value in cache.predicted_tms),
        capacities=cache.capacities[:count],
        oracle_flows=tuple(value[:count] for value in cache.oracle_flows),
        oracle_mlus=tuple(value[:count] for value in cache.oracle_mlus),
        path_features=(cache.path_features[:count] if cache.path_features is not None else None),
    )


def benchmark_overlay_once(cache, library, prototypes, repetitions: int) -> dict:
    # Warm up kernels and allocations before timing.
    for _ in range(10):
        construct_policy(cache, library, prototypes)
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        construct_policy(cache, library, prototypes)
    synchronize()
    prototype_seconds = (time.perf_counter() - started) / repetitions

    for _ in range(10):
        features = trainer.edge_features(cache)
        tolls, _ = screen.method.predict(library, features, 2, 32)
        trainer.route_with_tolls(cache, tolls, 1.0)
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        features = trainer.edge_features(cache)
        tolls, _ = screen.method.predict(library, features, 2, 32)
        trainer.route_with_tolls(cache, tolls, 1.0)
    synchronize()
    retrieval_seconds = (time.perf_counter() - started) / repetitions
    return {
        "batch_size": len(cache),
        "repetitions": repetitions,
        "prototype_overlay_ms_per_batch": 1000.0 * prototype_seconds,
        "prototype_overlay_ms_per_snapshot": 1000.0
        * prototype_seconds
        / len(cache),
        "full_318_case_retrieval_ms_per_batch": 1000.0 * retrieval_seconds,
        "full_318_case_retrieval_ms_per_snapshot": 1000.0
        * retrieval_seconds
        / len(cache),
        "overlay_speedup_over_318_case_retrieval": retrieval_seconds
        / prototype_seconds,
    }


def benchmark_overlay(cache, library, prototypes) -> dict:
    return {
        "batch_100": benchmark_overlay_once(cache, library, prototypes, 100),
        "batch_1": benchmark_overlay_once(
            slice_cache(cache, 1), library, prototypes, 500
        ),
    }


def strict_gate(load_factor: int, result: dict) -> bool:
    delta = result["delta"]
    if load_factor == 1:
        return max(abs(value) for value in delta.values()) <= 1e-7
    return (
        abs(delta["High.norm_fulfill_mean"]) <= 1e-6
        and delta["Medium.norm_fulfill_mean"] > 0.0
        and delta["Medium.norm_fulfill_p1"] > 0.0
        and delta["Medium.norm_fulfill_p10"] > 0.0
        and delta["Low.norm_fulfill_mean"] >= -0.003
        and delta["Low.norm_fulfill_p1"] >= -0.01
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )


def main() -> None:
    runtime.set_seed(20260828)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    payload = {
        "method": "three-prototype strict-ESM congestion fallback tolls",
        "short_name": "P3-CFT",
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "historical_actual_tm_used_for_prototype_training": False,
        "offline_teacher": "24-step strict-ESM sequential-admission correction projected to edge tolls",
        "selection_disclosure": (
            "P=3, gains and threshold were frozen on snapshots 318-399. "
            "Snapshots 400-499 were not queried by this method before this run; "
            "the broader thread has previously evaluated other methods there."
        ),
        "parameters": {
            "prototype_count": PROTOTYPE_COUNT,
            "medium_gain": MEDIUM_GAIN,
            "low_gain": LOW_GAIN,
            "activation": "mean strict-ESM total edge utilization >= threshold",
            "activation_threshold": ACTIVATION_THRESHOLD,
            "high_policy": "exact Hattrick bypass",
            "inactive_policy": "bit-exact Hattrick fallback",
        },
        "splits": {key: list(value) for key, value in SPLITS.items()},
        "device": str(device),
        "loads": {},
    }
    artifact_model = {"method": payload["method"], "parameters": payload["parameters"], "loads": {}}
    for load_factor in (1, 2, 3):
        load_started = time.perf_counter()
        model, props, library = screen.load_backbone(load_factor, device)
        prototypes = screen.fit_prototypes(library, PROTOTYPE_COUNT)
        load_results = {}
        caches = {}
        cache_build_seconds = {}
        for split_name in ("safety", "validation", "evaluation"):
            split_started = time.perf_counter()
            caches[split_name] = runtime.build_policy_cache(
                model, props, *SPLITS[split_name], batch_size=32
            )
            synchronize()
            cache_build_seconds[split_name] = time.perf_counter() - split_started
            load_results[split_name] = evaluate(
                model, props, caches[split_name], library, prototypes
            )
            load_results[split_name]["gate_pass"] = strict_gate(
                load_factor, load_results[split_name]
            )
            delta = load_results[split_name]["delta"]
            print(
                f"[{load_factor}x/{split_name}] "
                f"active={load_results[split_name]['active_count']}/"
                f"{load_results[split_name]['sample_count']} "
                f"gate={load_results[split_name]['gate_pass']} "
                f"dM={delta['Medium.norm_fulfill_mean']:+.6f} "
                f"dL={delta['Low.norm_fulfill_mean']:+.6f}",
                flush=True,
            )
        evaluation_cache = caches["evaluation"]
        model_float_count = (
            int(prototypes["centroids"].numel())
            + int(prototypes["targets"].numel())
            + int(library["mean"].numel())
            + int(library["std"].numel())
        )
        full_library_float_count = sum(
            int(library[key].numel())
            for key in ("mean", "std", "features", "targets")
        )
        payload["loads"][str(load_factor)] = {
            "topology": screen.LOADS[load_factor]["topology"],
            "checkpoint": str(screen.LOADS[load_factor]["checkpoint"]),
            "cluster_sizes": prototypes["cluster_sizes"],
            "model_float_count": model_float_count,
            "model_bytes_float32": 4 * model_float_count,
            "full_library_bytes_float32": 4 * full_library_float_count,
            "compression_ratio": full_library_float_count / model_float_count,
            "cache_build_seconds": cache_build_seconds,
            "safety": load_results["safety"],
            "validation": load_results["validation"],
            "evaluation": load_results["evaluation"],
            "information_audit": information_audit(
                evaluation_cache, library, prototypes
            ),
            "timing": benchmark_overlay(
                evaluation_cache, library, prototypes
            ),
            "development_gate_pass": bool(
                load_results["safety"]["gate_pass"]
                and load_results["validation"]["gate_pass"]
            ),
            "final_gate_pass": bool(load_results["evaluation"]["gate_pass"]),
            "seconds": time.perf_counter() - load_started,
        }
        artifact_model["loads"][str(load_factor)] = {
            "topology": screen.LOADS[load_factor]["topology"],
            "checkpoint": str(screen.LOADS[load_factor]["checkpoint"]),
            "feature_mean": library["mean"].detach().cpu(),
            "feature_std": library["std"].detach().cpu(),
            "centroids": prototypes["centroids"].detach().cpu(),
            "targets": prototypes["targets"].detach().cpu(),
            "cluster_sizes": prototypes["cluster_sizes"],
        }
        del model, props, library, prototypes, caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload["all_development_gates_pass"] = all(
        value["development_gate_pass"] for value in payload["loads"].values()
    )
    payload["all_final_gates_pass"] = all(
        value["final_gate_pass"] for value in payload["loads"].values()
    )
    payload["seconds"] = time.perf_counter() - started
    write_json(HERE / "final_level4_report.json", payload)
    torch.save(artifact_model, HERE / "p3_cft_model.pt")
    print(
        json.dumps(
            {
                "artifact": str(HERE / "final_level4_report.json"),
                "all_development_gates_pass": payload["all_development_gates_pass"],
                "all_final_gates_pass": payload["all_final_gates_pass"],
                "seconds": payload["seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
