from __future__ import annotations

"""Development-only stronger Low guard sweep; hard-limited to snapshots 0-349.

The persistent Level-4 core is frozen.  High and Medium policy tensors are
never regenerated or replaced by the guard.  This sweep deliberately uses
300-349 as development evidence after the first formal guard was rejected;
350+ remains unopened here.
"""

import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
CORE = THIS_DIR / "level4_selection_validation_only" / "selected_checkpoint.pt"
NATIVE = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "final_model.pt"
)
MODEL_RUNNER = THIS_DIR / "run_experiment.py"
GUARD_RUNNER = THIS_DIR / "train_low_guard.py"
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
STRICT_RUNNER = (
    TEST_DIR / "shared2x_sparse_path_cross_attention" / "evaluate_strict_esm_sequential.py"
)
OUTPUT_DIR = THIS_DIR / "artifacts" / "development_stronger_low_guard_300_349"
SETTINGS = ((2.0, 1.0), (2.0, 3.0), (2.0, 10.0), (3.0, 3.0))
SCALES = (0.25, 0.5, 0.75, 1.0)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def low_values(rows):
    return {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in rows
        if row["class"] == "Low"
    }


def paired_audit(reference_rows, candidate_rows):
    reference = low_values(reference_rows)
    candidate = low_values(candidate_rows)
    if set(reference) != set(candidate):
        raise RuntimeError("Paired Low rows do not cover the same snapshots")
    delta = [candidate[index] - reference[index] for index in sorted(reference)]
    mean = statistics.fmean(delta)
    se = statistics.stdev(delta) / math.sqrt(len(delta))
    # df=49, matching the frozen Level-4 guard audit.
    two_sided = 2.009575
    one_sided = 1.676551
    return {
        "n": len(delta),
        "mean_delta": mean,
        "standard_error": se,
        "two_sided_95_lower": mean - two_sided * se,
        "two_sided_95_upper": mean + two_sided * se,
        "one_sided_95_lower": mean - one_sided * se,
        "wins": sum(value > 0 for value in delta),
        "ties": sum(value == 0 for value in delta),
        "losses": sum(value < 0 for value in delta),
        "noninferiority_margin": 0.005,
        "noninferiority_pass": bool(mean - one_sided * se >= -0.005),
        "no_detected_significant_decline_pass": bool(mean + two_sided * se >= 0.0),
        "pass": bool(
            mean - one_sided * se >= -0.005 and mean + two_sided * se >= 0.0
        ),
    }


def state_cpu(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def main():
    torch.manual_seed(20260824)
    model_module = load("strong_guard_model", MODEL_RUNNER)
    guard = load("strong_guard_library", GUARD_RUNNER)
    edge = load("strong_guard_edge", EDGE_RUNNER)
    strict = load("strong_guard_strict", STRICT_RUNNER)
    runtime = edge.runtime
    runtime.set_seed(20260824)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)

    payload = torch.load(CORE, map_location=device, weights_only=False)
    model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    caches = {}
    features = {}
    rows = {}
    audits = {}
    for name, bounds, batch_size in (
        ("train", (0, 250), 32),
        ("safety", (250, 300), 25),
        ("development", (300, 350), 25),
    ):
        cache, audit = strict.build_strict_policy_cache(
            runtime, model, props, *bounds, batch_size, actual_input_mode="zero"
        )
        caches[name] = cache
        audits[name] = audit
        features[name] = edge.edge_features(cache)
        rows[name], _ = runtime.evaluate_cache(
            model, props, cache, None, batch_size=batch_size
        )

    native_payload = torch.load(NATIVE, map_location=device, weights_only=False)
    native_model = runtime.Hattrick(props).to(device=device, dtype=props.dtype)
    native_model.load_state_dict(native_payload["model_state_dict"], strict=True)
    native_model.eval()
    native_cache, native_audit = strict.build_strict_policy_cache(
        runtime, native_model, props, 300, 350, 25, actual_input_mode="zero"
    )
    native_rows, native_summary = runtime.evaluate_cache(
        native_model, props, native_cache, None, batch_size=25
    )

    train_baseline = edge.baseline_normalized(model, props, caches["train"])
    candidates = []
    started = time.perf_counter()
    for setting_index, (max_toll, safety_weight) in enumerate(SETTINGS):
        trained, history = guard.train_one(
            edge,
            model,
            props,
            caches["train"],
            features["train"],
            train_baseline,
            caches["safety"],
            features["safety"],
            rows["safety"],
            max_toll,
            safety_weight,
            20262001 + setting_index,
        )
        for saved in trained:
            for scale in SCALES:
                head = guard.LowGuard(
                    int(caches["train"].capacities.shape[1]),
                    int(features["train"].shape[-1]),
                    float(max_toll) * float(scale),
                ).to(device)
                head.load_state_dict(saved["state"], strict=True)
                head.eval()
                development = guard.evaluate(
                    runtime,
                    edge,
                    model,
                    props,
                    head,
                    caches["development"],
                    features["development"],
                    rows["development"],
                )
                candidates.append(
                    {
                        "setting_index": setting_index,
                        "max_toll": max_toll,
                        "safety_weight": safety_weight,
                        "epoch": int(saved["epoch"]),
                        "scale": scale,
                        "effective_max_toll": float(max_toll) * float(scale),
                        "safety": guard.public(saved),
                        "development": guard.public(development),
                        "development_vs_native": edge.diagnostics(
                            native_rows, development["rows"]
                        ),
                        "paired_vs_native": paired_audit(
                            native_rows, development["rows"]
                        ),
                        "state": state_cpu(saved["state"]),
                    }
                )

    def rank(value):
        low = value["development"]["diagnostics"]["Low"]
        native = value["paired_vs_native"]
        return (
            int(native["pass"]),
            float(low["mean_gain"]),
            float(low["p1_gain"]),
            float(low["p10_gain"]),
            -float(value["development"]["toll_abs_mean"]),
        )

    winner = max(candidates, key=rank)
    checkpoint = OUTPUT_DIR / "development_winner.pt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": winner["state"],
            "edge_count": int(caches["train"].capacities.shape[1]),
            "feature_count": int(features["train"].shape[-1]),
            "max_toll": winner["effective_max_toll"],
            "development_selection": {
                key: value for key, value in winner.items() if key != "state"
            },
            "test_data_read": False,
        },
        checkpoint,
    )
    public_candidates = [
        {key: value for key, value in item.items() if key != "state"}
        for item in candidates
    ]
    report = {
        "status": "development-only; no snapshot >=350 constructed",
        "splits": {"train": [0, 250], "safety": [250, 300], "development": [300, 350]},
        "settings": [list(item) for item in SETTINGS],
        "scales": list(SCALES),
        "strict_esm_audits": {**audits, "native_development": native_audit},
        "development_core_low": edge.runtime.summary_index(
            runtime.evaluate_cache(model, props, caches["development"], None, 25)[1]
        )["Low"],
        "development_native_low": runtime.summary_index(native_summary)["Low"],
        "development_core_vs_native": edge.diagnostics(native_rows, rows["development"])["Low"],
        "winner": {key: value for key, value in winner.items() if key != "state"},
        "candidate_count": len(public_candidates),
        "candidates": public_candidates,
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint.resolve()),
        "test_data_read": False,
    }
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    concise = {
        "status": report["status"],
        "candidate_count": report["candidate_count"],
        "core_low": report["development_core_low"],
        "native_low": report["development_native_low"],
        "winner": report["winner"],
        "runtime_seconds": report["runtime_seconds"],
    }
    print(json.dumps(concise, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
