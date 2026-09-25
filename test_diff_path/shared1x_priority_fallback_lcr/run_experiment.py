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
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


method = load_module("shared1x_priority_fallback_method", SOURCE)
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
    return model, props


def route_low_only(cache, tolls: torch.Tensor):
    priority_selective = tolls.clone()
    priority_selective[:, 0].zero_()
    return trainer.route_with_tolls(cache, priority_selective, 1.0)


def compact_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "snapshot": int(row["snapshot"]),
            "class": str(row["class"]),
            "norm_fulfill": float(row["norm_fulfill"]),
        }
        for row in rows
    ]


def evaluate(model, props, cache, library):
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    tolls, retrieval = method.predict(
        library,
        trainer.edge_features(cache),
        MEDIUM_NEIGHBORS,
        LOW_NEIGHBORS,
    )
    candidate_rows, candidate_summary = trainer.evaluate(
        model,
        props,
        cache,
        route_low_only(cache, tolls),
    )
    result = {
        "baseline": trainer.probe.compact(baseline_summary),
        "summary": trainer.probe.compact(candidate_summary),
        "delta": trainer.probe.gaps(candidate_summary, baseline_summary),
        "bootstrap": method.large.paired_bootstrap(
            baseline_rows, candidate_rows
        ),
        "retrieval": retrieval,
    }
    result["safe"] = safe(result)
    return result, baseline_rows, candidate_rows


def safe(result: dict) -> bool:
    delta = result["delta"]
    return (
        result["summary"]["High"]["norm_fulfill_mean"] >= 0.995
        and abs(delta["High.norm_fulfill_mean"]) <= 1e-6
        and abs(delta["Medium.norm_fulfill_mean"]) <= 1e-6
        and abs(delta["Medium.norm_fulfill_p1"]) <= 1e-6
        and abs(delta["Medium.norm_fulfill_p10"]) <= 1e-6
        and delta["Low.norm_fulfill_mean"] >= -0.003
        and delta["Low.norm_fulfill_p1"] >= -0.01
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )


def main() -> None:
    runtime.set_seed(20260827)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props = load_backbone(device)
    train_cache = runtime.build_policy_cache(
        model, props, *SPLITS["train"], batch_size=32
    )
    train_features = trainer.edge_features(train_cache)
    train_targets = trainer.teacher_tolls(model, props, train_cache)
    library = method.knn.fit_library(train_features, train_targets)

    results = {}
    final_baseline_rows = []
    final_candidate_rows = []
    for split_name in ("safety", "validation", "evaluation"):
        cache = runtime.build_policy_cache(
            model, props, *SPLITS[split_name], batch_size=32
        )
        result, baseline_rows, candidate_rows = evaluate(
            model, props, cache, library
        )
        result["range"] = list(SPLITS[split_name])
        results[split_name] = result
        if split_name == "evaluation":
            final_baseline_rows = baseline_rows
            final_candidate_rows = candidate_rows
        print(
            f"[{split_name}] safe={result['safe']} "
            f"dM={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
            f"dL={result['delta']['Low.norm_fulfill_mean']:+.6f}",
            flush=True,
        )

    payload = {
        "method": "1x priority-fallback local case retrieval",
        "principle": "At the near-saturation ceiling, preserve Hattrick High and Medium exactly and apply learned local edge tolls only to Low.",
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "historical_actual_tm_used_for_toll_training": False,
        "small_experiment_selection": {
            "source": str(
                TEST_DIR / "shared1x_advantage_gate_lcr" / "report.json"
            ),
            "rule": "Medium advantage below 1e-5 is treated as no reliable signal; choose the simpler exact bypass.",
            "selected": "High and Medium exact bypass; Low kNN edge toll active",
        },
        "checkpoint": str(CHECKPOINT),
        "splits": {key: list(value) for key, value in SPLITS.items()},
        "architecture": {
            "High": "exact Hattrick bypass",
            "Medium": "exact Hattrick bypass",
            "Low": "strict-ESM local retrieval of projected teacher edge tolls",
            "medium_neighbors_unused": MEDIUM_NEIGHBORS,
            "low_neighbors": LOW_NEIGHBORS,
            "toll_scale": 1.0,
        },
        "safety": results["safety"],
        "validation": results["validation"],
        "evaluation": results["evaluation"],
        "development_gate_pass": bool(
            results["safety"]["safe"] and results["validation"]["safe"]
        ),
        "final_gate_pass": bool(results["evaluation"]["safe"]),
        "rows": {
            "baseline": compact_rows(final_baseline_rows),
            "candidate": compact_rows(final_candidate_rows),
        },
        "seconds": time.perf_counter() - started,
    }
    write_json(HERE / "report.json", payload)
    torch.save(
        {
            "library": method.knn.cpu_library(library),
            "active_priority": "Low",
            "low_neighbors": LOW_NEIGHBORS,
            "toll_scale": 1.0,
            "checkpoint": str(CHECKPOINT),
        },
        HERE / "model.pt",
    )
    print(
        json.dumps(
            {
                "development_gate_pass": payload["development_gate_pass"],
                "final_gate_pass": payload["final_gate_pass"],
                "evaluation": payload["evaluation"],
                "seconds": payload["seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
