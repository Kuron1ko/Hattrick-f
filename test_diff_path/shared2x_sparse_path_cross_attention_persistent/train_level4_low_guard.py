from __future__ import annotations

"""Train and freeze the predeclared Level-4 prediction-only Low guard.

This program is deliberately unable to construct the 400-499 test window.  It
selects the guard on 250-299, gates it on 300-349, freezes the head, and only
then computes guarded 350-399 metrics. Integrity hashes and the already-frozen
core-selection manifest may be read earlier, but their metric values never enter
the head optimizer or safety rank.
"""

import hashlib
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
GUARD_LIBRARY = THIS_DIR / "train_low_guard.py"
MODEL_RUNNER = THIS_DIR / "run_experiment.py"
LEVEL4_RUNNER = THIS_DIR / "run_level4_validation_only.py"
SELECTION_DIR = THIS_DIR / "level4_selection_validation_only"
SELECTION_WATCHER = SELECTION_DIR / "watch_and_select.py"
SELECTION_POLICY = SELECTION_DIR / "selection_policy.json"
BASE_WATCHER = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "checkpoint_selection_validation_only"
    / "watch_and_select.py"
)
FROZEN_CORE_MANIFEST = SELECTION_DIR / "manifest_frozen_epoch_060.json"
SELECTED_CORE = SELECTION_DIR / "selected_checkpoint.pt"
PROTOCOL = THIS_DIR / "level4_low_guard_protocol.json"
PRECOMMIT = THIS_DIR / "level4_precommit_manifest.json"
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
STRICT_EVALUATOR = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "evaluate_strict_esm_sequential.py"
)
NATIVE_VALIDATION = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "validation_epoch_060_metrics.csv"
)
NATIVE_CONFIG = NATIVE_VALIDATION.parent / "config.json"
NATIVE_CHECKPOINT = NATIVE_VALIDATION.parent / "final_model.pt"
OUTPUT_DIR = THIS_DIR / "artifacts" / "level4_low_guard_frozen"

EXPECTED_HASHES = {
    GUARD_LIBRARY: "e399cb7a3fd66d11f5379ffa6e906c7c8d4322c6a165f80abb59a8ca507f4565",
    MODEL_RUNNER: "c4061d15495d4d5ecfdf4e3ca7dd77fa0a64a732bfb0c1c1d0ceefb61a1df14f",
    LEVEL4_RUNNER: "dbdd45cee3d05dd8f1d8ba9695b09647ce31f650cdbcd3e562c6896d7cde7fba",
    SELECTION_WATCHER: "6ef9548a3368615125f03cb38ed32ba87dfb129ce074dd516a53d2af7f642f58",
    SELECTION_POLICY: "36026468706c73562c4ac61c812ca5054494f42c313fd71332f897525ea948bf",
    BASE_WATCHER: "c0e25d46ff0aedd9b1a30f66465e225db1f00815848812fc39d82e555a7d1197",
    PROTOCOL: "ee75d96b19a8ef9a95d51cd470ddf977d93684ef5fd7a5813e2113fbfd4946b6",
    EDGE_RUNNER: "839c6a8b2a091553fe817abedc97016b35e498980a4eb2d8f936cb5cbae4c866",
    STRICT_EVALUATOR: "4afdbd4fb2ccc4d8b6bd77297cf0dc24485f34b922719283f31121f1f741f98b",
    NATIVE_VALIDATION: "225f8809e26cf11e144589e447981a88f48fd07c7dce8777445dff3a80b9602d",
    NATIVE_CONFIG: "9e1c728c579abf78951a8f8d5fc3c564e34423559539bdfc72251e8ab7458324",
    NATIVE_CHECKPOINT: "394e0fe592491f70836b92a9a17543ea53d3fe1fbfc5b3f156bef0e4c3a78622",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_static_inputs() -> dict[str, str]:
    result = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {path}: {actual} != {expected}")
        result[str(path.resolve())] = actual
    return result


def verify_precommit() -> tuple[dict, str]:
    precommit = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    if precommit.get("test_data_read") is not False:
        raise RuntimeError("Precommit does not declare a test-blind state")
    items = precommit.get("files", [])
    declared = {Path(item["path"]).resolve() for item in items}
    required = {
        Path(__file__).resolve(),
        PROTOCOL.resolve(),
        LEVEL4_RUNNER.resolve(),
        SELECTION_WATCHER.resolve(),
        SELECTION_POLICY.resolve(),
        MODEL_RUNNER.resolve(),
        GUARD_LIBRARY.resolve(),
        STRICT_EVALUATOR.resolve(),
        EDGE_RUNNER.resolve(),
        BASE_WATCHER.resolve(),
    }
    missing = required - declared
    if missing:
        raise RuntimeError(f"Precommit omitted required files: {sorted(map(str, missing))}")
    for item in items:
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Precommitted source changed: {path}")
    return precommit, sha256(PRECOMMIT)


def read_frozen_core() -> tuple[dict, str]:
    manifest = json.loads(FROZEN_CORE_MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("frozen") or manifest.get("test_data_read") is not False:
        raise RuntimeError("Core manifest is not a test-blind frozen selection")
    if int(manifest.get("latest_history_epoch", -1)) != 60:
        raise RuntimeError("Core manifest did not freeze at epoch 60")
    selected = manifest.get("selection", {}).get("selected")
    copied = manifest.get("selected_checkpoint_copy")
    if selected is None or copied is None:
        raise RuntimeError("Core manifest has no selected checkpoint")
    if manifest.get("policy", {}).get("forbidden_test_window") != [400, 500]:
        raise RuntimeError("Core selection policy has an unexpected test window")
    selection = manifest.get("selection", {})
    if selection.get("core_low_used_for_selection") is not False:
        raise RuntimeError("Core Low unexpectedly influenced checkpoint selection")
    if selection.get("mandatory_downstream_low_guard") is not True:
        raise RuntimeError("Core manifest does not require the downstream guard")
    if selected.get("hard_gate_pass") is not True:
        raise RuntimeError("Selected core did not pass its declared hard gate")
    watcher = manifest.get("watcher_source_sha256", {})
    expected_watcher = {
        "watch_and_select.py": sha256(SELECTION_WATCHER),
        "base_watch_and_select.py": sha256(BASE_WATCHER),
        "selection_policy.json": sha256(SELECTION_POLICY),
    }
    for name, expected in expected_watcher.items():
        if watcher.get(name) != expected:
            raise RuntimeError(f"Frozen watcher source mismatch: {name}")
    config_record = manifest.get("source_run_config")
    if not isinstance(config_record, dict):
        raise RuntimeError("Core manifest did not freeze its run config")
    config_path = Path(config_record["path"])
    if sha256(config_path) != config_record["sha256"]:
        raise RuntimeError("Core run config changed after selection")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("train") != [0, 350]
        or config.get("validation") != [350, 400]
        or config.get("evaluation") != [350, 400]
    ):
        raise RuntimeError("Frozen core config does not use validation-only Level-4")
    run_directory = Path(manifest["source_run_directory"])
    history_path = run_directory / "train_history.csv"
    if sha256(history_path) != manifest["source_train_history_sha256"]:
        raise RuntimeError("Core train history changed after freeze")
    selected_epoch = int(selected["epoch"])
    selected_summary = run_directory / f"validation_epoch_{selected_epoch:03d}_summary.json"
    selected_metrics = run_directory / f"validation_epoch_{selected_epoch:03d}_metrics.csv"
    if sha256(selected_summary) != selected["validation_summary_sha256"]:
        raise RuntimeError("Selected validation summary changed after freeze")
    if sha256(selected_metrics) != selected["validation_metrics_sha256"]:
        raise RuntimeError("Selected validation rows changed after freeze")
    if Path(copied["path"]).resolve() != SELECTED_CORE.resolve():
        raise RuntimeError("Core checkpoint path does not match the frozen manifest")
    actual = sha256(SELECTED_CORE)
    if actual != copied["sha256"]:
        raise RuntimeError("Core checkpoint SHA256 does not match the frozen manifest")
    if int(copied["epoch"]) != int(selected["epoch"]):
        raise RuntimeError("Selected core epoch is inconsistent")
    if actual != selected["checkpoint"]["sha256"]:
        raise RuntimeError("Selected core copy differs from its immutable archive")
    return manifest, actual


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def low_by_snapshot(rows: list[dict]) -> dict[int, float]:
    values = {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in rows
        if row["class"] == "Low"
    }
    if set(values) != set(range(350, 400)):
        raise RuntimeError("Expected exactly validation snapshots 350-399")
    return values


def paired_native_audit(native_rows: list[dict], candidate_rows: list[dict]) -> dict:
    native = low_by_snapshot(native_rows)
    candidate = low_by_snapshot(candidate_rows)
    differences = [candidate[index] - native[index] for index in sorted(native)]
    mean = statistics.fmean(differences)
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    two_sided_critical = 2.009575  # df=49
    one_sided_critical = 1.676551  # df=49
    lower = mean - two_sided_critical * standard_error
    upper = mean + two_sided_critical * standard_error
    noninferiority_lower = mean - one_sided_critical * standard_error
    return {
        "n": len(differences),
        "mean_delta": mean,
        "standard_error": standard_error,
        "two_sided_95_lower": lower,
        "two_sided_95_upper": upper,
        "one_sided_95_lower": noninferiority_lower,
        "wins": sum(value > 0.0 for value in differences),
        "ties": sum(value == 0.0 for value in differences),
        "losses": sum(value < 0.0 for value in differences),
        "noninferiority_margin": 0.005,
        "one_sided_95_lower_min": -0.005,
        "two_sided_95_upper_min": 0.0,
        "noninferiority_pass": bool(noninferiority_lower >= -0.005),
        "no_detected_significant_decline_pass": bool(upper >= 0.0),
        "pass": bool(noninferiority_lower >= -0.005 and upper >= 0.0),
    }


def reference_row_audit(native_rows: list[dict], candidate_rows: list[dict]) -> dict:
    fields = ("demand", "oracle_admitted_traffic", "oracle_mlu")
    native = {
        (int(row["snapshot"]), row["class"]): row for row in native_rows
    }
    candidate = {
        (int(row["snapshot"]), row["class"]): row for row in candidate_rows
    }
    expected = {
        (snapshot, class_name)
        for snapshot in range(350, 400)
        for class_name in ("High", "Medium", "Low")
    }
    if set(native) != expected or set(candidate) != expected:
        raise RuntimeError("Native/candidate validation row coverage differs")
    maximum = max(
        abs(float(native[key][field]) - float(candidate[key][field]))
        for key in expected
        for field in fields
    )
    return {
        "snapshots": [350, 400],
        "row_count": len(expected),
        "fields": list(fields),
        "max_abs_delta": maximum,
        "pass": bool(maximum <= 1e-5),
    }


def strict_cpu_counterfactual_audit(
    runtime,
    strict,
    edge,
    guard,
    model_module,
    payload,
    head_state,
    max_toll,
) -> dict:
    cpu = torch.device("cpu")
    props = runtime.build_props(4, cpu)
    model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=cpu, dtype=props.dtype
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    zero, zero_audit = strict.build_strict_policy_cache(
        runtime, model, props, 300, 304, 4, actual_input_mode="zero"
    )
    permuted, permuted_audit = strict.build_strict_policy_cache(
        runtime, model, props, 300, 304, 4, actual_input_mode="permute"
    )
    policy = strict.policy_difference(zero.policies, permuted.policies)
    zero_features = edge.edge_features(zero)
    permuted_features = edge.edge_features(permuted)
    head = guard.LowGuard(
        int(zero.capacities.shape[1]), int(zero_features.shape[-1]), max_toll
    ).to(cpu)
    head.load_state_dict(head_state, strict=True)
    head.eval()
    zero_candidate, zero_tolls = guard.build_candidate_cache(
        head, zero, zero_features, batch_size=4
    )
    permuted_candidate, permuted_tolls = guard.build_candidate_cache(
        head, permuted, permuted_features, batch_size=4
    )
    result = {
        "window": [300, 304],
        "device": "cpu",
        "zero_input_audit": zero_audit,
        "permuted_input_audit": permuted_audit,
        "backbone_policy_difference": policy,
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
        "guarded_low_policy_torch_equal": bool(
            torch.equal(zero_candidate.path_features, permuted_candidate.path_features)
        ),
        "guarded_low_policy_max_abs_delta": float(
            (zero_candidate.path_features - permuted_candidate.path_features)
            .abs()
            .max()
            .item()
        ),
    }
    result["pass"] = bool(
        all(item["torch_equal"] for item in policy.values())
        and result["edge_features_torch_equal"]
        and result["guard_tolls_torch_equal"]
        and result["guarded_low_policy_torch_equal"]
    )
    return result


def full_upstream_policy_audit(guard, head, cache, features) -> dict:
    candidate, _ = guard.build_candidate_cache(head, cache, features, batch_size=25)
    # LowOnlyAdapter replaces only policy index 2 with candidate.path_features.
    # The first two tensors remain the exact objects stored in the frozen cache.
    before = cache.policies
    after = (before[0], before[1], candidate.path_features.unsqueeze(-1))
    return {
        class_name: {
            "torch_equal": bool(torch.equal(before[index], after[index])),
            "same_storage": bool(
                before[index].untyped_storage().data_ptr()
                == after[index].untyped_storage().data_ptr()
            ),
            "max_abs_delta": float(
                (before[index] - after[index]).abs().max().item()
            ),
        }
        for index, class_name in enumerate(("High", "Medium"))
    }


def main() -> None:
    static_hashes = verify_static_inputs()
    precommit, precommit_sha = verify_precommit()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["splits"] != {
        "guard_train": [0, 250],
        "guard_safety_selection": [250, 300],
        "guard_internal_gate_once": [300, 350],
        "joint_core_composite_validation_once": [350, 400],
        "forbidden_test_until_composite_freeze": [400, 500],
    }:
        raise RuntimeError("Low-guard split protocol changed")
    core_manifest, core_sha = read_frozen_core()
    recorded_precommit = core_manifest.get("watcher_source_sha256", {}).get(
        "level4_precommit_manifest.json"
    )
    if recorded_precommit != precommit_sha:
        raise RuntimeError("Core watcher and guard trainer saw different precommits")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing guard run: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    guard = load_module("persistent_level4_guard_library", GUARD_LIBRARY)
    model_module = load_module("persistent_level4_guard_model", MODEL_RUNNER)
    edge = load_module("persistent_level4_guard_edge", EDGE_RUNNER)
    strict = load_module("persistent_level4_guard_strict", STRICT_EVALUATOR)
    runtime = edge.runtime
    runtime.set_seed(20260824)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    payload = torch.load(SELECTED_CORE, map_location=device, weights_only=False)
    expected_epoch = int(core_manifest["selection"]["selected"]["epoch"])
    if int(payload.get("epoch", -1)) != expected_epoch:
        raise RuntimeError("Selected core payload epoch changed")
    if payload.get("config", {}).get("evaluation") != [350, 400]:
        raise RuntimeError("Selected core was not produced by validation-only Level-4")
    model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    load_result = model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    caches = {}
    features = {}
    rows = {}
    summaries = {}
    information_audits = {}
    for name, bounds, batch_size in (
        ("train", (0, 250), 32),
        ("safety", (250, 300), 25),
        ("internal_gate", (300, 350), 25),
    ):
        cache, audit = strict.build_strict_policy_cache(
            runtime,
            model,
            props,
            *bounds,
            batch_size,
            actual_input_mode="zero",
        )
        caches[name] = cache
        information_audits[name] = audit
        features[name] = edge.edge_features(cache)
        rows[name], summaries[name] = runtime.evaluate_cache(
            model, props, cache, None, batch_size=batch_size
        )

    started = time.perf_counter()
    train_baseline = edge.baseline_normalized(model, props, caches["train"])
    all_candidates = []
    histories = []
    settings = ((0.75, 10.0), (1.25, 30.0), (2.0, 60.0))
    for index, (max_toll, safety_weight) in enumerate(settings):
        candidates, history = guard.train_one(
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
            20261001 + index,
        )
        all_candidates.extend(candidates)
        histories.append(
            {
                "max_toll": max_toll,
                "safety_weight": safety_weight,
                "history": history,
            }
        )

    winner = max(all_candidates, key=guard.rank)
    head = guard.LowGuard(
        int(caches["train"].capacities.shape[1]),
        int(features["train"].shape[-1]),
        winner["max_toll"],
    ).to(device)
    head.load_state_dict(winner["state"], strict=True)
    head.eval()

    internal_gate = guard.evaluate(
        runtime,
        edge,
        model,
        props,
        head,
        caches["internal_gate"],
        features["internal_gate"],
        rows["internal_gate"],
    )
    internal_policy_audit = full_upstream_policy_audit(
        guard, head, caches["internal_gate"], features["internal_gate"]
    )
    internal_low = internal_gate["diagnostics"]["Low"]
    internal_max_capacity = max(
        float(row["admitted_capacity_ratio"]) for row in internal_gate["rows"]
    )
    internal_max_disabled = max(
        float(row["disabled_flow"]) for row in internal_gate["rows"]
    )
    internal_passed = bool(
        internal_low["mean_gain"] >= -1e-6
        and internal_low["p1_gain"] >= -1e-6
        and internal_low["p10_gain"] >= -1e-6
        and internal_low["paired_negative_count"] == 0
        and internal_low["ecdf_max_upward_violation"] <= 1e-12
        and all(
            value["torch_equal"] and value["same_storage"]
            for value in internal_policy_audit.values()
        )
        and internal_max_capacity <= 1.000001
        and internal_max_disabled <= 1e-8
    )
    strict_cpu_audit = strict_cpu_counterfactual_audit(
        runtime,
        strict,
        edge,
        guard,
        model_module,
        payload,
        winner["state"],
        winner["max_toll"],
    )
    internal_passed = bool(internal_passed and strict_cpu_audit["pass"])

    # Freeze the selected head before guarded 350-399 metrics are computed.
    checkpoint_path = OUTPUT_DIR / "frozen_low_guard_prevalidation.pt"
    torch.save(
        {
            "state_dict": winner["state"],
            "edge_count": int(caches["train"].capacities.shape[1]),
            "feature_count": int(features["train"].shape[-1]),
            "max_toll": winner["max_toll"],
            "selection": guard.public(winner),
            "internal_gate": guard.public(internal_gate),
            "internal_gate_passed": internal_passed,
            "backbone_sha256": core_sha,
            "protocol_sha256": EXPECTED_HASHES[PROTOCOL],
            "test_data_read": False,
            "guarded_joint_validation_metrics_computed": False,
        },
        checkpoint_path,
    )
    prevalidation_manifest = {
        "frozen": True,
        "internal_gate_passed": internal_passed,
        "guarded_joint_validation_metrics_computed": False,
        "test_data_read": False,
        "precommit": {
            "sha256": precommit_sha,
            "declared_file_count": len(precommit.get("files", [])),
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
        },
        "core_checkpoint_sha256": core_sha,
        "protocol_sha256": sha256(PROTOCOL),
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "internal_gate": guard.public(internal_gate),
        "internal_max_admitted_capacity_ratio": internal_max_capacity,
        "internal_max_disabled_flow": internal_max_disabled,
        "strict_cpu_counterfactual_audit": strict_cpu_audit,
    }
    runtime.write_json(
        OUTPUT_DIR / "head_frozen_before_joint_validation.json",
        prevalidation_manifest,
    )
    if not internal_passed:
        rejection = {
            "status": "rejected before joint validation",
            "test_data_read": False,
            "guarded_joint_validation_metrics_computed": False,
            "internal_gate": guard.public(internal_gate),
            "internal_max_admitted_capacity_ratio": internal_max_capacity,
            "internal_max_disabled_flow": internal_max_disabled,
            "strict_cpu_counterfactual_audit": strict_cpu_audit,
            "checkpoint_sha256": sha256(checkpoint_path),
        }
        runtime.write_json(OUTPUT_DIR / "rejected_internal_gate.json", rejection)
        print(json.dumps(rejection, ensure_ascii=False), flush=True)
        return

    # First and only guard use of 350-399; the head SHA above can no longer change.
    validation_cache, validation_audit = strict.build_strict_policy_cache(
        runtime,
        model,
        props,
        350,
        400,
        25,
        actual_input_mode="zero",
    )
    caches["validation"] = validation_cache
    information_audits["validation"] = validation_audit
    features["validation"] = edge.edge_features(validation_cache)
    rows["validation"], summaries["validation"] = runtime.evaluate_cache(
        model, props, validation_cache, None, batch_size=25
    )
    validation = guard.evaluate(
        runtime,
        edge,
        model,
        props,
        head,
        validation_cache,
        features["validation"],
        rows["validation"],
    )
    validation_policy_audit = full_upstream_policy_audit(
        guard, head, validation_cache, features["validation"]
    )
    native_rows = read_csv_rows(NATIVE_VALIDATION)
    native_diagnostics = edge.diagnostics(native_rows, validation["rows"])
    native_paired = paired_native_audit(native_rows, validation["rows"])
    native_reference_audit = reference_row_audit(native_rows, validation["rows"])
    core_gain = validation["diagnostics"]["Low"]
    high_summary = validation["summary"]["High"]
    native_low = native_diagnostics["Low"]
    native_medium = native_diagnostics["Medium"]
    max_capacity = max(
        float(row["admitted_capacity_ratio"]) for row in validation["rows"]
    )
    max_disabled = max(float(row["disabled_flow"]) for row in validation["rows"])
    acceptance_checks = {
        "guard_vs_core_low_mean": bool(core_gain["mean_gain"] >= 0.0),
        "guard_vs_core_low_p1": bool(core_gain["p1_gain"] >= 0.0),
        "guard_vs_core_low_p10": bool(core_gain["p10_gain"] >= 0.0),
        "guard_vs_core_no_negative_snapshot": bool(
            core_gain["paired_negative_count"] == 0
        ),
        "guard_vs_core_low_ecdf": bool(
            core_gain["ecdf_max_upward_violation"] <= 1e-12
        ),
        "native_low_paired_noninferiority": bool(native_paired["pass"]),
        "native_low_p1": bool(native_low["p1_gain"] >= -0.005),
        "native_low_p10": bool(native_low["p10_gain"] >= -0.005),
        "native_medium_mean": bool(native_medium["mean_gain"] >= 0.005),
        "native_medium_p1": bool(native_medium["p1_gain"] >= 0.0),
        "native_medium_p10": bool(native_medium["p10_gain"] >= 0.0),
        "high_mean": bool(high_summary["norm_fulfill_mean"] >= 0.995),
        "high_p1": bool(high_summary["norm_fulfill_p1"] >= 0.985),
        "high_p10": bool(high_summary["norm_fulfill_p10"] >= 0.99),
        "core_watcher_hard_gate": bool(
            core_manifest["selection"]["selected"]["hard_gate_pass"]
        ),
        "full_upstream_policy_exact": bool(
            all(
            value["torch_equal"] and value["same_storage"]
            for value in validation_policy_audit.values()
            )
        ),
        "probe_upstream_policy_exact": bool(
            all(validation["upstream_policy_torch_equal"].values())
        ),
        "native_reference_rows_match": bool(native_reference_audit["pass"]),
        "capacity": bool(max_capacity <= 1.000001),
        "disabled_flow": bool(max_disabled <= 1e-8),
    }
    accepted = bool(all(acceptance_checks.values()))
    runtime.write_csv(OUTPUT_DIR / "validation_core_rows.csv", rows["validation"])
    runtime.write_csv(
        OUTPUT_DIR / "validation_guarded_rows.csv", validation["rows"]
    )
    report = {
        "method": "prediction-only Low guard after frozen persistent Level-4 core",
        "status": "accepted" if accepted else "rejected",
        "protocol": protocol,
        "test_data_read": False,
        "precommit_sha256": precommit_sha,
        "static_source_sha256": static_hashes,
        "core_manifest_sha256": sha256(FROZEN_CORE_MANIFEST),
        "core_checkpoint_sha256": core_sha,
        "core_selected_epoch": expected_epoch,
        "state_load": {
            "missing": list(load_result.missing_keys),
            "unexpected": list(load_result.unexpected_keys),
        },
        "information_audits": information_audits,
        "safety_baseline": runtime.summary_index(summaries["safety"]),
        "safety_candidates": [guard.public(value) for value in all_candidates],
        "safety_winner": guard.public(winner),
        "internal_gate_core": runtime.summary_index(summaries["internal_gate"]),
        "internal_gate_guarded": guard.public(internal_gate),
        "internal_gate_passed_before_joint_validation": internal_passed,
        "internal_max_admitted_capacity_ratio": internal_max_capacity,
        "internal_max_disabled_flow": internal_max_disabled,
        "internal_upstream_policy_full_audit": internal_policy_audit,
        "strict_cpu_counterfactual_audit": strict_cpu_audit,
        "validation_core": runtime.summary_index(summaries["validation"]),
        "validation_guarded": guard.public(validation),
        "validation_upstream_policy_full_audit": validation_policy_audit,
        "validation_guarded_vs_native": native_diagnostics,
        "validation_guarded_vs_native_low_paired": native_paired,
        "validation_native_reference_row_audit": native_reference_audit,
        "validation_max_admitted_capacity_ratio": max_capacity,
        "validation_max_disabled_flow": max_disabled,
        "acceptance_checks": acceptance_checks,
        "accepted_before_test": accepted,
        "histories": histories,
        "runtime": {"device": str(device), "seconds": time.perf_counter() - started},
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
        },
        "validation_rows": {
            "core_sha256": sha256(OUTPUT_DIR / "validation_core_rows.csv"),
            "guarded_sha256": sha256(OUTPUT_DIR / "validation_guarded_rows.csv"),
            "native_sha256": sha256(NATIVE_VALIDATION),
        },
    }
    report_path = OUTPUT_DIR / "report.json"
    runtime.write_json(report_path, report)
    composite = {
        "schema": 1,
        "frozen": True,
        "accepted": accepted,
        "test_data_read": False,
        "forbidden_test_window": [400, 500],
        "core_manifest": {
            "path": str(FROZEN_CORE_MANIFEST.resolve()),
            "sha256": sha256(FROZEN_CORE_MANIFEST),
        },
        "core_checkpoint": {
            "path": str(SELECTED_CORE.resolve()),
            "sha256": core_sha,
            "epoch": expected_epoch,
        },
        "low_guard_checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
        },
        "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
        "precommit": {
            "path": str(PRECOMMIT.resolve()),
            "sha256": precommit_sha,
        },
        "training_report": {
            "path": str(report_path.resolve()),
            "sha256": sha256(report_path),
        },
        "validation_rows": {
            "core": {
                "path": str((OUTPUT_DIR / "validation_core_rows.csv").resolve()),
                "sha256": sha256(OUTPUT_DIR / "validation_core_rows.csv"),
            },
            "guarded": {
                "path": str((OUTPUT_DIR / "validation_guarded_rows.csv").resolve()),
                "sha256": sha256(OUTPUT_DIR / "validation_guarded_rows.csv"),
            },
            "native": {
                "path": str(NATIVE_VALIDATION.resolve()),
                "sha256": sha256(NATIVE_VALIDATION),
            },
        },
        "source_sha256": {
            **static_hashes,
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        },
        "test_open_authorization": bool(accepted),
    }
    runtime.write_json(OUTPUT_DIR / "composite_frozen_manifest.json", composite)
    print(json.dumps(composite, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
