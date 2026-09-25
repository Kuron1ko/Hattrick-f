from __future__ import annotations

import argparse
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
CONFIGS = {
    1: {
        "label": "1x",
        "topology": "geant_priomask500_shared",
        "checkpoint": ROOT / "hattrick_geant_priomask500_shared_8sp.pkl",
    },
    3: {
        "label": "3x",
        "topology": "geant_priomask500_shared_load3x_train",
        "checkpoint": ROOT / "hattrick_geant_priomask500_shared_load3x_train_8sp.pkl",
    },
}
SPLITS = {
    "train": (0, 318),
    "safety": (318, 350),
    "validation": (350, 400),
    "evaluation": (400, 500),
}
MEDIUM_NEIGHBORS = 2
LOW_NEIGHBORS = 32


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


method = load_module("multi_asymmetric_knn_method", SOURCE)
trainer = method.trainer
runtime = method.runtime


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_backbone(load_factor: int, device: torch.device):
    config = CONFIGS[load_factor]
    checkpoint = config["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    runtime.shared.TOPOLOGY = config["topology"]
    props = runtime.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    model = torch.load(checkpoint, map_location=device, weights_only=False)
    model = model.to(device=device, dtype=props.dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    # The reused utilities contain their own module globals. Point every
    # evaluation entry point at the load-specific runtime selected above.
    trainer.runtime = runtime
    trainer.probe.runtime = runtime
    method.runtime = runtime
    return model, props, checkpoint


def compact_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "snapshot": int(row["snapshot"]),
            "class": str(row["class"]),
            "norm_fulfill": float(row["norm_fulfill"]),
        }
        for row in rows
    ]


def evaluate_split(model, props, cache, library: dict) -> tuple[dict, list[dict], list[dict]]:
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    tolls, retrieval = method.predict(
        library,
        trainer.edge_features(cache),
        MEDIUM_NEIGHBORS,
        LOW_NEIGHBORS,
    )
    candidate_cache = trainer.route_with_tolls(cache, tolls, 1.0)
    candidate_rows, candidate_summary = trainer.evaluate(
        model, props, cache, candidate_cache
    )
    result = {
        "summary": trainer.probe.compact(candidate_summary),
        "baseline": trainer.probe.compact(baseline_summary),
        "delta": trainer.probe.gaps(candidate_summary, baseline_summary),
        "bootstrap": method.large.paired_bootstrap(baseline_rows, candidate_rows),
        "retrieval": retrieval,
    }
    return result, baseline_rows, candidate_rows


def strict_gate(result: dict) -> bool:
    return method.safe(
        {
            "summary": result["summary"],
            "delta": result["delta"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-factor", type=int, choices=tuple(CONFIGS), required=True)
    args = parser.parse_args()
    runtime.set_seed(20260823 + args.load_factor)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props, checkpoint = load_backbone(args.load_factor, device)

    print(f"[{args.load_factor}x] building 0:318 strict-ESM case library", flush=True)
    train_cache = runtime.build_policy_cache(
        model, props, *SPLITS["train"], batch_size=32
    )
    train_features = trainer.edge_features(train_cache)
    train_targets = trainer.teacher_tolls(model, props, train_cache)
    library = method.knn.fit_library(train_features, train_targets)

    split_results: dict[str, dict] = {}
    final_baseline_rows: list[dict] = []
    final_candidate_rows: list[dict] = []
    for split_name in ("safety", "validation", "evaluation"):
        split_range = SPLITS[split_name]
        print(
            f"[{args.load_factor}x] evaluating {split_name} "
            f"{split_range[0]}:{split_range[1]}",
            flush=True,
        )
        cache = runtime.build_policy_cache(
            model, props, *split_range, batch_size=32
        )
        result, baseline_rows, candidate_rows = evaluate_split(
            model, props, cache, library
        )
        result["range"] = list(split_range)
        result["strict_gate"] = strict_gate(result)
        split_results[split_name] = result
        if split_name == "evaluation":
            final_baseline_rows = baseline_rows
            final_candidate_rows = candidate_rows
        print(
            f"[{args.load_factor}x] {split_name}: gate={result['strict_gate']} "
            f"High={result['summary']['High']['norm_fulfill_mean']:.6f} "
            f"dM={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
            f"dL={result['delta']['Low.norm_fulfill_mean']:+.6f}",
            flush=True,
        )

    # Policy construction depends only on predicted traffic. This local audit
    # verifies that altering the evaluation-only actual TM after caching cannot
    # change the retrieved tolls or the routed policy.
    evaluation_cache = runtime.build_policy_cache(
        model, props, *SPLITS["evaluation"], batch_size=32
    )
    features = trainer.edge_features(evaluation_cache)
    tolls, _ = method.predict(
        library, features, MEDIUM_NEIGHBORS, LOW_NEIGHBORS
    )
    policy = trainer.route_with_tolls(evaluation_cache, tolls, 1.0).path_features
    from dataclasses import replace

    counterfactual = replace(
        evaluation_cache,
        tms=tuple(torch.zeros_like(value) for value in evaluation_cache.tms),
    )
    counterfactual_features = trainer.edge_features(counterfactual)
    counterfactual_tolls, _ = method.predict(
        library, counterfactual_features, MEDIUM_NEIGHBORS, LOW_NEIGHBORS
    )
    counterfactual_policy = trainer.route_with_tolls(
        counterfactual, counterfactual_tolls, 1.0
    ).path_features

    artifact_dir = HERE / "artifacts" / f"{args.load_factor}x_level4"
    final = split_results["evaluation"]
    payload = {
        "method": "priority-asymmetric local case retrieval for edge tolls",
        "load_factor": args.load_factor,
        "topology": CONFIGS[args.load_factor]["topology"],
        "checkpoint": str(checkpoint),
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "historical_actual_tm_used_for_training": False,
        "transfer_rule": {
            "frozen_from_2x": True,
            "medium_neighbors": MEDIUM_NEIGHBORS,
            "low_neighbors": LOW_NEIGHBORS,
            "toll_scale": 1.0,
            "teacher_steps": int(trainer.probe.TEACHER_CONFIG[0]),
            "projection": "target-flow weighted log-policy ridge, ridge=0.01",
        },
        "splits": {key: list(value) for key, value in SPLITS.items()},
        "safety": split_results["safety"],
        "validation": split_results["validation"],
        "evaluation": final,
        "development_gate_pass": bool(
            split_results["safety"]["strict_gate"]
            and split_results["validation"]["strict_gate"]
        ),
        "final_gate_pass": bool(final["strict_gate"]),
        "information_audit": {
            "counterfactual_feature_max_abs_diff": float(
                (features - counterfactual_features).abs().max().item()
            ),
            "counterfactual_toll_max_abs_diff": float(
                (tolls - counterfactual_tolls).abs().max().item()
            ),
            "counterfactual_policy_max_abs_diff": float(
                (policy - counterfactual_policy).abs().max().item()
            ),
        },
        "seconds": time.perf_counter() - started,
        "rows": {
            "baseline": compact_rows(final_baseline_rows),
            "candidate": compact_rows(final_candidate_rows),
        },
    }
    write_json(artifact_dir / "report.json", payload)
    torch.save(
        {
            "library": method.knn.cpu_library(library),
            "medium_neighbors": MEDIUM_NEIGHBORS,
            "low_neighbors": LOW_NEIGHBORS,
            "checkpoint": str(checkpoint),
            "topology": CONFIGS[args.load_factor]["topology"],
        },
        artifact_dir / "model.pt",
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact_dir),
                "development_gate_pass": payload["development_gate_pass"],
                "final_gate_pass": payload["final_gate_pass"],
                "evaluation": final,
                "seconds": payload["seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
