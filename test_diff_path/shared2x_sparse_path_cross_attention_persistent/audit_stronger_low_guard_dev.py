from __future__ import annotations

"""Post-selection development audit for the stronger Low-only guard.

All constructed datasets end at snapshot 349.  The rolling evaluations are a
robustness diagnostic for the already chosen hyperparameters, not fresh
holdouts: the hyperparameters were chosen after inspecting 300-349.
"""

import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import replace
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
WINNER = (
    THIS_DIR
    / "artifacts"
    / "development_stronger_low_guard_300_349"
    / "development_winner.pt"
)
TRAIN_SOURCE = THIS_DIR / "develop_stronger_low_guard_300_349.py"
TRAIN_REPORT = (
    THIS_DIR
    / "artifacts"
    / "development_stronger_low_guard_300_349"
    / "report.json"
)
MODEL_RUNNER = THIS_DIR / "run_experiment.py"
GUARD_RUNNER = THIS_DIR / "train_low_guard.py"
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
STRICT_RUNNER = (
    TEST_DIR / "shared2x_sparse_path_cross_attention" / "evaluate_strict_esm_sequential.py"
)
OUTPUT = (
    THIS_DIR
    / "artifacts"
    / "development_stronger_low_guard_300_349"
    / "postselection_oof_and_cpu_audit.json"
)

FIXED_MAX_TOLL = 2.0
FIXED_SAFETY_WEIGHT = 3.0
FIXED_EPOCH = 60
FIXED_SEED = 20262002
FOLDS = (
    ("fold_150_199", (0, 100), (100, 150), (150, 200)),
    ("fold_200_249", (0, 150), (150, 200), (200, 250)),
    ("fold_250_299", (0, 200), (200, 250), (250, 300)),
    ("fold_300_349", (0, 250), (250, 300), (300, 350)),
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise RuntimeError("Paired Low rows differ")
    delta = [candidate[index] - reference[index] for index in sorted(reference)]
    mean = statistics.fmean(delta)
    se = statistics.stdev(delta) / math.sqrt(len(delta))
    # Conservative normal critical value for aggregate n=200; fold-level
    # reports use the same threshold to avoid claiming exact small-n t tests.
    critical = 1.96
    one_sided = 1.645
    return {
        "n": len(delta),
        "mean_delta": mean,
        "standard_error": se,
        "two_sided_95_lower": mean - critical * se,
        "two_sided_95_upper": mean + critical * se,
        "one_sided_95_lower": mean - one_sided * se,
        "wins": sum(value > 0 for value in delta),
        "ties": sum(value == 0 for value in delta),
        "losses": sum(value < 0 for value in delta),
        "noninferiority_margin": 0.005,
        "noninferiority_pass": bool(mean - one_sided * se >= -0.005),
        "no_detected_significant_decline_pass": bool(mean + critical * se >= 0.0),
        "pass": bool(
            mean - one_sided * se >= -0.005 and mean + critical * se >= 0.0
        ),
    }


def build(runtime, strict, model, props, bounds, batch_size=25, mode="zero"):
    if bounds[1] > 350:
        raise RuntimeError("Audit is code-limited to snapshots 0-349")
    cache, audit = strict.build_strict_policy_cache(
        runtime, model, props, *bounds, batch_size, actual_input_mode=mode
    )
    return cache, audit


def cpu_counterfactual(runtime, strict, edge, guard, model_module, core_payload, winner):
    props = runtime.build_props(4, torch.device("cpu"))
    model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device="cpu", dtype=props.dtype
    )
    model.load_state_dict(core_payload["model_state_dict"], strict=True)
    model.eval()
    head = guard.LowGuard(
        int(winner["edge_count"]),
        int(winner["feature_count"]),
        float(winner["max_toll"]),
    ).to("cpu")
    head.load_state_dict(winner["state_dict"], strict=True)
    head.eval()

    zero, zero_audit = build(runtime, strict, model, props, (346, 350), 4, "zero")
    permuted, permuted_audit = build(
        runtime, strict, model, props, (346, 350), 4, "permute"
    )
    policy_difference = strict.policy_difference(zero.policies, permuted.policies)
    zero_features = edge.edge_features(zero)
    permuted_features = edge.edge_features(permuted)
    with torch.no_grad():
        zero_tolls = head(zero_features)
        permuted_tolls = head(permuted_features)
        pte = zero.dataset.pte.coalesce().to(dtype=torch.float32)
        zero_low = guard.route_low(
            zero.policies[2].squeeze(-1), zero_tolls, pte
        )
        permuted_low = guard.route_low(
            permuted.policies[2].squeeze(-1), permuted_tolls, pte
        )
    hybrid_policies = (
        zero.policies[0],
        zero.policies[1],
        zero_low.unsqueeze(-1),
    )
    return {
        "window": [346, 350],
        "device": "cpu",
        "zero_input_audit": zero_audit,
        "permuted_input_audit": permuted_audit,
        "backbone_policy_difference": policy_difference,
        "edge_features_torch_equal": bool(
            torch.equal(zero_features, permuted_features)
        ),
        "edge_features_max_abs_delta": float(
            (zero_features - permuted_features).abs().max().item()
        ),
        "guard_tolls_torch_equal": bool(torch.equal(zero_tolls, permuted_tolls)),
        "guard_tolls_max_abs_delta": float(
            (zero_tolls - permuted_tolls).abs().max().item()
        ),
        "guarded_low_policy_torch_equal": bool(torch.equal(zero_low, permuted_low)),
        "guarded_low_policy_max_abs_delta": float(
            (zero_low - permuted_low).abs().max().item()
        ),
        "hybrid_upstream_policy_torch_equal": {
            "High": bool(torch.equal(hybrid_policies[0], zero.policies[0])),
            "Medium": bool(torch.equal(hybrid_policies[1], zero.policies[1])),
        },
        "pass": bool(
            all(value["torch_equal"] for value in policy_difference.values())
            and torch.equal(zero_features, permuted_features)
            and torch.equal(zero_tolls, permuted_tolls)
            and torch.equal(zero_low, permuted_low)
            and torch.equal(hybrid_policies[0], zero.policies[0])
            and torch.equal(hybrid_policies[1], zero.policies[1])
        ),
    }


def main():
    torch.manual_seed(20260824)
    model_module = load("strong_oof_model", MODEL_RUNNER)
    guard = load("strong_oof_guard", GUARD_RUNNER)
    edge = load("strong_oof_edge", EDGE_RUNNER)
    strict = load("strong_oof_strict", STRICT_RUNNER)
    runtime = edge.runtime
    runtime.set_seed(20260824)
    winner = torch.load(WINNER, map_location="cpu", weights_only=False)
    train_report = json.loads(TRAIN_REPORT.read_text(encoding="utf-8"))
    chosen = train_report["winner"]
    expected = (
        float(chosen["max_toll"]),
        float(chosen["safety_weight"]),
        int(chosen["epoch"]),
        float(chosen["scale"]),
    )
    if expected != (FIXED_MAX_TOLL, FIXED_SAFETY_WEIGHT, FIXED_EPOCH, 1.0):
        raise RuntimeError(f"Development winner changed: {expected}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    core_payload = torch.load(CORE, map_location=device, weights_only=False)
    model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    model.load_state_dict(core_payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    native_payload = torch.load(NATIVE, map_location=device, weights_only=False)
    native_model = runtime.Hattrick(props).to(device=device, dtype=props.dtype)
    native_model.load_state_dict(native_payload["model_state_dict"], strict=True)
    native_model.eval()

    fold_reports = []
    aggregate_core_rows = []
    aggregate_candidate_rows = []
    aggregate_native_rows = []
    started = time.perf_counter()
    for fold_name, train_bounds, safety_bounds, evaluation_bounds in FOLDS:
        train_cache, train_audit = build(
            runtime, strict, model, props, train_bounds, 25, "zero"
        )
        safety_cache, safety_audit = build(
            runtime, strict, model, props, safety_bounds, 25, "zero"
        )
        evaluation_cache, evaluation_audit = build(
            runtime, strict, model, props, evaluation_bounds, 25, "zero"
        )
        train_features = edge.edge_features(train_cache)
        safety_features = edge.edge_features(safety_cache)
        evaluation_features = edge.edge_features(evaluation_cache)
        safety_rows, _ = runtime.evaluate_cache(
            model, props, safety_cache, None, batch_size=25
        )
        evaluation_rows, _ = runtime.evaluate_cache(
            model, props, evaluation_cache, None, batch_size=25
        )
        train_baseline = edge.baseline_normalized(model, props, train_cache)
        saved, history = guard.train_one(
            edge,
            model,
            props,
            train_cache,
            train_features,
            train_baseline,
            safety_cache,
            safety_features,
            safety_rows,
            FIXED_MAX_TOLL,
            FIXED_SAFETY_WEIGHT,
            FIXED_SEED,
        )
        selected = next(item for item in saved if int(item["epoch"]) == FIXED_EPOCH)
        head = guard.LowGuard(
            int(train_cache.capacities.shape[1]),
            int(train_features.shape[-1]),
            FIXED_MAX_TOLL,
        ).to(device)
        head.load_state_dict(selected["state"], strict=True)
        head.eval()
        evaluation = guard.evaluate(
            runtime,
            edge,
            model,
            props,
            head,
            evaluation_cache,
            evaluation_features,
            evaluation_rows,
        )
        candidate_cache, _ = guard.build_candidate_cache(
            head, evaluation_cache, evaluation_features, batch_size=25
        )
        probe = runtime.select_cache(
            candidate_cache,
            torch.arange(len(candidate_cache), device=device),
        )
        before = list(probe["policies"])
        after = guard.LowOnlyAdapter().adapt_batch(list(before), probe)
        full_exact = {
            "High": bool(torch.equal(before[0], after[0])),
            "Medium": bool(torch.equal(before[1], after[1])),
        }
        native_cache, native_audit = build(
            runtime, strict, native_model, props, evaluation_bounds, 25, "zero"
        )
        native_rows, _ = runtime.evaluate_cache(
            native_model, props, native_cache, None, batch_size=25
        )
        fold_reports.append(
            {
                "name": fold_name,
                "train": list(train_bounds),
                "safety_diagnostic": list(safety_bounds),
                "out_of_fit_evaluation": list(evaluation_bounds),
                "fixed_hyperparameters": {
                    "max_toll": FIXED_MAX_TOLL,
                    "safety_weight": FIXED_SAFETY_WEIGHT,
                    "epoch": FIXED_EPOCH,
                    "seed": FIXED_SEED,
                },
                "strict_esm_audits": {
                    "train": train_audit,
                    "safety": safety_audit,
                    "evaluation": evaluation_audit,
                    "native_evaluation": native_audit,
                },
                "candidate_vs_core": guard.public(evaluation),
                "candidate_vs_native": edge.diagnostics(
                    native_rows, evaluation["rows"]
                ),
                "paired_vs_native": paired_audit(native_rows, evaluation["rows"]),
                "full_upstream_policy_torch_equal": full_exact,
            }
        )
        aggregate_core_rows.extend(evaluation_rows)
        aggregate_candidate_rows.extend(evaluation["rows"])
        aggregate_native_rows.extend(native_rows)

    cpu_audit = cpu_counterfactual(
        runtime,
        strict,
        edge,
        guard,
        model_module,
        torch.load(CORE, map_location="cpu", weights_only=False),
        winner,
    )
    report = {
        "status": "post-selection development audit; no snapshot >=350 constructed",
        "selection_disclosure": {
            "winner_fixed_before_300_349_was_opened": False,
            "role_of_300_349": "development model selection, not holdout",
            "rolling_oof_role": (
                "post-selection robustness diagnostic with fixed hyperparameters; "
                "not a new independent confirmation"
            ),
        },
        "fixed_hyperparameters": {
            "max_toll": FIXED_MAX_TOLL,
            "safety_weight": FIXED_SAFETY_WEIGHT,
            "epoch": FIXED_EPOCH,
            "seed": FIXED_SEED,
        },
        "folds": fold_reports,
        "aggregate_evaluation_window": [150, 350],
        "aggregate_candidate_vs_core": edge.diagnostics(
            aggregate_core_rows, aggregate_candidate_rows
        ),
        "aggregate_candidate_vs_native": edge.diagnostics(
            aggregate_native_rows, aggregate_candidate_rows
        ),
        "aggregate_paired_vs_native": paired_audit(
            aggregate_native_rows, aggregate_candidate_rows
        ),
        "strict_cpu_zero_vs_permute": cpu_audit,
        "artifact_hashes": {
            "winner_checkpoint": sha256(WINNER),
            "training_source": sha256(TRAIN_SOURCE),
            "training_report": sha256(TRAIN_REPORT),
            "core_checkpoint": sha256(CORE),
            "native_checkpoint": sha256(NATIVE),
            "audit_source": sha256(Path(__file__).resolve()),
        },
        "runtime_seconds": time.perf_counter() - started,
        "test_data_read": False,
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    concise = {
        "status": report["status"],
        "fold_low_vs_core": [
            item["candidate_vs_core"]["diagnostics"]["Low"]
            for item in fold_reports
        ],
        "fold_low_vs_native": [
            item["candidate_vs_native"]["Low"] for item in fold_reports
        ],
        "aggregate_candidate_vs_core": report["aggregate_candidate_vs_core"]["Low"],
        "aggregate_candidate_vs_native": report["aggregate_candidate_vs_native"]["Low"],
        "aggregate_paired_vs_native": report["aggregate_paired_vs_native"],
        "cpu_audit_pass": cpu_audit["pass"],
        "runtime_seconds": report["runtime_seconds"],
    }
    print(json.dumps(concise, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
