from __future__ import annotations

"""One-shot joint validation for the frozen stronger Low-only repair.

The program has no optimizer and no candidate loop.  It verifies a separate
precommit, records access before constructing snapshots 350-399, evaluates one
fixed core/head pair under literal-zero actual-TM policy inputs, and freezes an
accept/reject manifest.  It cannot construct snapshots 400-499.
"""

import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
PERSISTENT_DIR = THIS_DIR.parent
TEST_DIR = PERSISTENT_DIR.parent
ROOT = TEST_DIR.parent
PROTOCOL = THIS_DIR / "protocol.json"
PRECOMMIT = THIS_DIR / "precommit_manifest.json"
AUTHORIZATION = THIS_DIR / "authorization_record.json"
ORIGINAL_PRECOMMIT = PERSISTENT_DIR / "level4_precommit_manifest.json"
CORE_RUNNER = PERSISTENT_DIR / "run_experiment.py"
CORE = PERSISTENT_DIR / "level4_selection_validation_only" / "selected_checkpoint.pt"
CORE_MANIFEST = (
    PERSISTENT_DIR
    / "level4_selection_validation_only"
    / "manifest_frozen_epoch_060.json"
)
HEAD = (
    PERSISTENT_DIR
    / "artifacts"
    / "development_stronger_low_guard_300_349"
    / "development_winner.pt"
)
DEVELOPMENT_REPORT = (
    PERSISTENT_DIR
    / "artifacts"
    / "development_stronger_low_guard_300_349"
    / "report.json"
)
DEVELOPMENT_SOURCE = PERSISTENT_DIR / "develop_stronger_low_guard_300_349.py"
POSTSELECTION_AUDIT = (
    PERSISTENT_DIR
    / "artifacts"
    / "development_stronger_low_guard_300_349"
    / "postselection_oof_and_cpu_audit.json"
)
V1_REJECTION = (
    PERSISTENT_DIR
    / "artifacts"
    / "level4_low_guard_frozen"
    / "rejected_internal_gate.json"
)
GUARD_LIBRARY = PERSISTENT_DIR / "train_low_guard.py"
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
STRICT = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "evaluate_strict_esm_sequential.py"
)
NATIVE = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "final_model.pt"
)
OUTPUT_DIR = PERSISTENT_DIR / "artifacts" / "low_guard_v2_joint_validation"

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


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_receipt_exclusive(path: Path, value) -> None:
    """Create the one-shot access lock atomically; an existing lock is fatal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise RuntimeError("Unable to write the one-shot access receipt")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_precommit(authorized_sha256: str) -> dict:
    if sha256(PRECOMMIT) != authorized_sha256:
        raise RuntimeError("Precommit does not match the external authorization")
    manifest = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    if manifest.get("persistent_composite_test_data_read") is not False:
        raise RuntimeError("The frozen persistent composite already read its test window")
    if manifest.get("low_guard_v2_joint_guarded_metrics_read") is not False:
        raise RuntimeError("Precommit was written after LowGuard-v2 joint evaluation")
    if manifest.get("core_checkpoint_selected_on_350_399") is not True:
        raise RuntimeError("Precommit omits the known core-validation reuse")
    if manifest.get("core_medium_metrics_previously_known") is not True:
        raise RuntimeError("Precommit omits the known Medium-validation reuse")
    if manifest.get("unrelated_abandoned_architectures_previously_used_400_499") is not True:
        raise RuntimeError("Precommit omits prior benchmark reuse")
    for item in manifest.get("files", []):
        path = Path(item["path"])
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Precommitted input changed: {path}")
    original = json.loads(ORIGINAL_PRECOMMIT.read_text(encoding="utf-8"))
    if original.get("test_data_read") is not False:
        raise RuntimeError("Original Level-4 precommit is not test blind")
    for item in original.get("files", []):
        path = Path(item["path"])
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Original frozen Level-4 source changed: {path}")
    declared = {Path(item["path"]).resolve() for item in manifest["files"]}
    required = {
        Path(__file__).resolve(),
        PROTOCOL.resolve(),
        CORE.resolve(),
        CORE_MANIFEST.resolve(),
        HEAD.resolve(),
        DEVELOPMENT_REPORT.resolve(),
        DEVELOPMENT_SOURCE.resolve(),
        POSTSELECTION_AUDIT.resolve(),
        ORIGINAL_PRECOMMIT.resolve(),
        V1_REJECTION.resolve(),
        CORE_RUNNER.resolve(),
        GUARD_LIBRARY.resolve(),
        EDGE_RUNNER.resolve(),
        STRICT.resolve(),
        NATIVE.resolve(),
    }
    missing = required - declared
    if missing:
        raise RuntimeError(f"Precommit omitted inputs: {sorted(map(str, missing))}")
    return manifest


def verify_authorization(precommit_sha: str) -> tuple[dict, str]:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    expected = {
        "authorized": True,
        "precommit_sha256": precommit_sha,
        "validator_sha256": sha256(Path(__file__).resolve()),
        "protocol_sha256": sha256(PROTOCOL),
        "core_sha256": sha256(CORE),
        "head_sha256": sha256(HEAD),
        "validation_window": [350, 400],
        "forbidden_test_window": [400, 500],
        "core_checkpoint_selected_on_350_399": True,
        "core_medium_metrics_previously_known": True,
        "low_guard_v2_head_trained_selected_or_scaled_on_350_399": False,
        "low_guard_v2_guarded_metrics_computed_on_350_399_before_authorization": False,
        "validation_window_role": "one-shot LowGuard-v2 head/composite check; not independent Medium confirmation",
        "persistent_composite_evaluated_on_400_499": False,
        "unrelated_abandoned_architectures_previously_used_400_499": True,
        "future_400_499_role": "adaptive confirmation of this exact frozen composite, not a globally untouched research holdout",
    }
    if authorization != expected:
        raise RuntimeError("External one-shot authorization does not match frozen inputs")
    return authorization, sha256(AUTHORIZATION)


def verify_protocol(protocol: dict) -> None:
    expected = {
        "fixed_candidate": {
            "core_epoch": 41,
            "guard_max_toll": 2.0,
            "guard_safety_weight": 3.0,
            "guard_epoch": 60,
            "scale": 1.0,
        },
        "validation_evidence_scope": {
            "core_checkpoint_selected_on_350_399": True,
            "core_medium_metrics_previously_known": True,
            "low_guard_v2_head_trained_selected_or_scaled_on_350_399": False,
            "low_guard_v2_guarded_metrics_computed_on_350_399_before_authorization": False,
            "role": "one-shot Low-head/composite constraint check; not an independent confirmation of the frozen core's Medium result",
        },
        "benchmark_reuse_disclosure": {
            "unrelated_abandoned_architectures_previously_used_400_499": True,
            "persistent_core_plus_low_guard_v2_previously_evaluated_on_400_499": False,
            "future_400_499_role": "adaptive confirmation of this exact frozen composite, not a globally untouched research holdout",
        },
        "primary_objective": {
            "mean_gain_min": 0.005,
            "p1_gain_min": 0.0,
            "p10_gain_min": 0.0,
        },
        "high_gates": {
            "norm_mean_min": 0.995,
            "norm_p1_min": 0.985,
            "norm_p10_min": 0.99,
        },
        "low_constraint": {
            "guard_vs_core_mean_gain_min": 0.0,
            "guard_vs_core_paired_bootstrap_95_lower_min": 0.0,
            "native_noninferiority_margin": 0.005,
            "native_paired_bootstrap_95_lower_min": -0.005,
            "native_p1_gain_min": -0.005,
            "native_p10_gain_min": -0.005,
            "bootstrap_seed": 20260824,
            "bootstrap_samples": 100000,
        },
        "feasibility": {
            "max_admitted_capacity_ratio": 1.000001,
            "max_disabled_flow": 1e-8,
            "high_medium_policy_torch_equal": True,
        },
    }
    for section, values in expected.items():
        actual = protocol.get(section, {})
        for key, value in values.items():
            if actual.get(key) != value:
                raise RuntimeError(f"Protocol mismatch: {section}.{key}")


def low_map(rows) -> dict[int, float]:
    values = {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in rows
        if row["class"] == "Low"
    }
    if set(values) != set(range(350, 400)):
        raise RuntimeError("Expected Low rows for exactly snapshots 350-399")
    return values


def paired_low(reference_rows, candidate_rows, seed: int, samples: int) -> dict:
    reference = low_map(reference_rows)
    candidate = low_map(candidate_rows)
    delta = np.asarray(
        [candidate[index] - reference[index] for index in sorted(reference)],
        dtype=np.float64,
    )
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / math.sqrt(len(delta)))
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=np.float64)
    chunk = 10000
    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        boot[start:stop] = delta[indices].mean(axis=1)
    return {
        "n": len(delta),
        "mean_delta": mean,
        "standard_error": se,
        "student_two_sided_95": [
            mean - 2.009575 * se,
            mean + 2.009575 * se,
        ],
        "student_one_sided_95_lower": mean - 1.676551 * se,
        "bootstrap_seed": seed,
        "bootstrap_samples": samples,
        "bootstrap_two_sided_95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "wins": int((delta > 1e-7).sum()),
        "ties_within_1e_7": int((np.abs(delta) <= 1e-7).sum()),
        "losses": int((delta < -1e-7).sum()),
        "minimum": float(delta.min()),
        "maximum": float(delta.max()),
    }


def reference_audit(native_rows, candidate_rows) -> dict:
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
    fields = ("demand", "oracle_admitted_traffic", "oracle_mlu")
    if set(native) != expected or set(candidate) != expected:
        raise RuntimeError("Reference/candidate row coverage mismatch")
    maximum = max(
        abs(float(native[key][field]) - float(candidate[key][field]))
        for key in expected
        for field in fields
    )
    return {"fields": list(fields), "max_abs_delta": maximum, "pass": maximum <= 1e-5}


def cpu_counterfactual_audit(
    runtime, model_module, guard, edge, strict, core_payload, head_payload
) -> dict:
    device = torch.device("cpu")
    props = runtime.build_props(4, device)
    model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    model.load_state_dict(core_payload["model_state_dict"], strict=True)
    model.eval()
    zero, zero_input = strict.build_strict_policy_cache(
        runtime, model, props, 350, 354, 4, actual_input_mode="zero"
    )
    permuted, permuted_input = strict.build_strict_policy_cache(
        runtime, model, props, 350, 354, 4, actual_input_mode="permute"
    )
    zero_features = edge.edge_features(zero)
    permuted_features = edge.edge_features(permuted)
    head = guard.LowGuard(
        int(head_payload["edge_count"]),
        int(head_payload["feature_count"]),
        float(head_payload["max_toll"]),
    ).to(device)
    head.load_state_dict(head_payload["state_dict"], strict=True)
    head.eval()
    zero_candidate, zero_toll = guard.build_candidate_cache(
        head, zero, zero_features, batch_size=4
    )
    permuted_candidate, permuted_toll = guard.build_candidate_cache(
        head, permuted, permuted_features, batch_size=4
    )
    policy_equal = {
        class_name: bool(torch.equal(left, right))
        for class_name, left, right in zip(
            ("High", "Medium", "Low"), zero.policies, permuted.policies
        )
    }
    result = {
        "zero_input": zero_input,
        "permuted_input": permuted_input,
        "core_policy_torch_equal": policy_equal,
        "edge_features_torch_equal": bool(
            torch.equal(zero_features, permuted_features)
        ),
        "edge_tolls_torch_equal": bool(torch.equal(zero_toll, permuted_toll)),
        "final_low_policy_torch_equal": bool(
            torch.equal(zero_candidate.path_features, permuted_candidate.path_features)
        ),
    }
    result["pass"] = bool(
        all(policy_equal.values())
        and result["edge_features_torch_equal"]
        and result["edge_tolls_torch_equal"]
        and result["final_low_policy_torch_equal"]
        and zero_input["model_actual_input_abs_max"] == 0.0
        and zero_input["model_actual_input_nonzero_count"] == 0
    )
    return result


def main() -> None:
    torch.manual_seed(20260824)
    np.random.seed(20260824)
    torch.use_deterministic_algorithms(True)
    # Authenticate the precommit bytes before parsing their self-declared paths.
    # This prevents an unauthorized manifest from inducing arbitrary file reads.
    precommit_sha = sha256(PRECOMMIT)
    authorization, authorization_sha = verify_authorization(precommit_sha)
    precommit = verify_precommit(precommit_sha)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    verify_protocol(protocol)
    if protocol["fixed_candidate"]["scale"] != 1.0:
        raise RuntimeError("Protocol candidate is not the full fixed guard")
    if protocol["splits"]["joint_validation_once"] != [350, 400]:
        raise RuntimeError("Unexpected validation window")
    if protocol["splits"]["forbidden_test"] != [400, 500]:
        raise RuntimeError("Unexpected test window")
    if OUTPUT_DIR.exists():
        raise RuntimeError("A prior joint-validation attempt already exists")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    receipt = {
        "status": "joint-validation access committed before dataset construction",
        "window": [350, 400],
        "precommit_sha256": precommit_sha,
        "authorization_sha256": authorization_sha,
        "core_sha256": sha256(CORE),
        "head_sha256": sha256(HEAD),
        "core_checkpoint_selected_on_350_399": True,
        "core_medium_metrics_previously_known": True,
        "low_guard_v2_guarded_metrics_computed_before_this_receipt": False,
        "validation_window_role": "one-shot LowGuard-v2 head/composite check; not independent Medium confirmation",
        "persistent_composite_evaluated_on_400_499": False,
        "unrelated_abandoned_architectures_previously_used_400_499": True,
    }
    receipt_path = OUTPUT_DIR / "access_receipt_before_350_399.json"
    write_receipt_exclusive(receipt_path, receipt)
    receipt_sha = sha256(receipt_path)

    for item in (str(ROOT), str(TEST_DIR)):
        if item not in sys.path:
            sys.path.insert(0, item)
    from frameworks.hattrick_system import Hattrick

    model_module = load("low_guard_v2_model", CORE_RUNNER)
    guard = load("low_guard_v2_library", GUARD_LIBRARY)
    edge = load("low_guard_v2_edge", EDGE_RUNNER)
    strict = load("low_guard_v2_strict", STRICT)
    runtime = edge.runtime
    runtime.set_seed(20260824)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    core_payload = torch.load(CORE, map_location=device, weights_only=False)
    head_payload = torch.load(HEAD, map_location=device, weights_only=False)
    if int(core_payload.get("epoch", -1)) != 41:
        raise RuntimeError("Frozen core payload is not epoch 41")
    if head_payload.get("test_data_read") is not False:
        raise RuntimeError("Low guard checkpoint is not test blind")
    selection = head_payload.get("development_selection", {})
    expected_selection = {
        "max_toll": 2.0,
        "safety_weight": 3.0,
        "epoch": 60,
        "scale": 1.0,
        "effective_max_toll": 2.0,
    }
    for key, value in expected_selection.items():
        if selection.get(key) != value:
            raise RuntimeError(f"Frozen Low guard selection mismatch: {key}")
    if float(head_payload.get("max_toll", -1.0)) != 2.0:
        raise RuntimeError("Frozen Low guard max_toll mismatch")
    postselection = json.loads(POSTSELECTION_AUDIT.read_text(encoding="utf-8"))
    if postselection.get("artifact_hashes", {}).get("core_checkpoint") != sha256(CORE):
        raise RuntimeError("Development audit was bound to a different core")
    if postselection.get("artifact_hashes", {}).get("winner_checkpoint") != sha256(HEAD):
        raise RuntimeError("Development audit was bound to a different Low guard")

    core = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    core.load_state_dict(core_payload["model_state_dict"], strict=True)
    core.eval()
    native_payload = torch.load(NATIVE, map_location=device, weights_only=False)
    native = Hattrick(props).to(device=device, dtype=props.dtype)
    native.load_state_dict(native_payload["model_state_dict"], strict=True)
    native.eval()
    head = guard.LowGuard(
        int(head_payload["edge_count"]),
        int(head_payload["feature_count"]),
        float(head_payload["max_toll"]),
    ).to(device)
    head.load_state_dict(head_payload["state_dict"], strict=True)
    head.eval()

    core_cache, core_input_audit = strict.build_strict_policy_cache(
        runtime, core, props, 350, 400, 25, actual_input_mode="zero"
    )
    native_cache, native_input_audit = strict.build_strict_policy_cache(
        runtime, native, props, 350, 400, 25, actual_input_mode="zero"
    )
    core_rows, core_summary = runtime.evaluate_cache(
        core, props, core_cache, None, batch_size=25
    )
    native_rows, native_summary = runtime.evaluate_cache(
        native, props, native_cache, None, batch_size=25
    )
    features = edge.edge_features(core_cache)
    candidate_cache, _candidate_tolls = guard.build_candidate_cache(
        head, core_cache, features, batch_size=25
    )
    all_indices = torch.arange(len(core_cache), device=props.device)
    candidate_batch = runtime.select_cache(candidate_cache, all_indices)
    before_adapter = list(candidate_batch["policies"])
    after_adapter = guard.LowOnlyAdapter().adapt_batch(
        list(before_adapter), candidate_batch
    )
    full_upstream_policy_audit = {
        class_name: {
            "torch_equal": bool(torch.equal(before_adapter[index], after_adapter[index])),
            "same_storage": bool(
                before_adapter[index].untyped_storage().data_ptr()
                == after_adapter[index].untyped_storage().data_ptr()
            ),
            "max_abs_delta": float(
                (before_adapter[index] - after_adapter[index]).abs().max().item()
            ),
        }
        for index, class_name in enumerate(("High", "Medium"))
    }
    guarded = guard.evaluate(
        runtime, edge, core, props, head, core_cache, features, core_rows
    )
    candidate_rows = guarded["rows"]
    candidate_summary = guarded["summary"]
    candidate_vs_native = edge.diagnostics(native_rows, candidate_rows)
    core_vs_native = edge.diagnostics(native_rows, core_rows)
    paired_vs_core = paired_low(core_rows, candidate_rows, 20260824, 100000)
    paired_vs_native = paired_low(native_rows, candidate_rows, 20260824, 100000)
    reference = reference_audit(native_rows, candidate_rows)
    strict_cpu = cpu_counterfactual_audit(
        runtime, model_module, guard, edge, strict, core_payload, head_payload
    )

    indexed = candidate_summary
    high = indexed["High"]
    medium_gain = candidate_vs_native["Medium"]
    low_gain = candidate_vs_native["Low"]
    core_low_gain = guarded["diagnostics"]["Low"]
    max_capacity = max(float(row["admitted_capacity_ratio"]) for row in candidate_rows)
    max_disabled = max(float(row["disabled_flow"]) for row in candidate_rows)
    checks = {
        "high_mean_at_least_0.995": high["norm_fulfill_mean"] >= 0.995,
        "high_p1_at_least_0.985": high["norm_fulfill_p1"] >= 0.985,
        "high_p10_at_least_0.990": high["norm_fulfill_p10"] >= 0.990,
        "medium_mean_gain_at_least_0.005": medium_gain["mean_gain"] >= 0.005,
        "medium_p1_gain_nonnegative": medium_gain["p1_gain"] >= 0.0,
        "medium_p10_gain_nonnegative": medium_gain["p10_gain"] >= 0.0,
        "guard_low_mean_gain_nonnegative": core_low_gain["mean_gain"] >= 0.0,
        "guard_low_bootstrap_lower_nonnegative": paired_vs_core["bootstrap_two_sided_95"][0] >= 0.0,
        "native_low_no_significant_decline": paired_vs_native["bootstrap_two_sided_95"][1] >= 0.0,
        "native_low_margin_0.005": paired_vs_native["bootstrap_two_sided_95"][0] >= -0.005,
        "native_low_p1_gain_at_least_minus_0.005": low_gain["p1_gain"] >= -0.005,
        "native_low_p10_gain_at_least_minus_0.005": low_gain["p10_gain"] >= -0.005,
        "high_policy_exact": bool(
            full_upstream_policy_audit["High"]["torch_equal"]
            and full_upstream_policy_audit["High"]["same_storage"]
        ),
        "medium_policy_exact": bool(
            full_upstream_policy_audit["Medium"]["torch_equal"]
            and full_upstream_policy_audit["Medium"]["same_storage"]
        ),
        "strict_cpu_counterfactual": bool(strict_cpu["pass"]),
        "reference_rows_match": bool(reference["pass"]),
        "capacity": max_capacity <= 1.000001,
        "disabled_flow": max_disabled <= 1e-8,
    }
    accepted = bool(all(checks.values()))
    runtime.write_csv(OUTPUT_DIR / "core_rows.csv", core_rows)
    runtime.write_csv(OUTPUT_DIR / "guarded_rows.csv", candidate_rows)
    runtime.write_csv(OUTPUT_DIR / "native_rows.csv", native_rows)
    report = {
        "status": "accepted" if accepted else "rejected",
        "accepted_before_test": accepted,
        "persistent_composite_test_data_read": False,
        "benchmark_reuse_disclosure": {
            "core_checkpoint_selected_on_350_399": True,
            "low_guard_v2_head_first_applied_to_350_399_in_this_run": True,
            "unrelated_abandoned_architectures_previously_used_400_499": True,
        },
        "window": [350, 400],
        "precommit_sha256": precommit_sha,
        "authorization": {
            "path": str(AUTHORIZATION.resolve()),
            "sha256": authorization_sha,
        },
        "access_receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": receipt_sha,
        },
        "protocol": protocol,
        "input_audits": {"core": core_input_audit, "native": native_input_audit},
        "strict_cpu_counterfactual_audit": strict_cpu,
        "full_350_399_upstream_policy_audit": full_upstream_policy_audit,
        "core": runtime.summary_index(core_summary),
        "native": runtime.summary_index(native_summary),
        "guarded": candidate_summary,
        "core_vs_native": core_vs_native,
        "guarded_vs_core": guarded["diagnostics"],
        "guarded_vs_native": candidate_vs_native,
        "paired_low_vs_core": paired_vs_core,
        "paired_low_vs_native": paired_vs_native,
        "reference_audit": reference,
        "max_admitted_capacity_ratio": max_capacity,
        "max_disabled_flow": max_disabled,
        "acceptance_checks": checks,
        "diagnostic_only": {
            "low_ecdf_upward_vs_core": core_low_gain["ecdf_max_upward_violation"],
            "low_ecdf_upward_vs_native": low_gain["ecdf_max_upward_violation"],
            "low_negative_snapshots_vs_native": low_gain["paired_negative_count"],
        },
        "source_sha256": {
            "validator": sha256(Path(__file__).resolve()),
            "protocol": sha256(PROTOCOL),
            "precommit": precommit_sha,
            "core": sha256(CORE),
            "head": sha256(HEAD),
        },
    }
    result_path = OUTPUT_DIR / "result.json"
    write_json(result_path, report)
    composite = {
        "schema": 2,
        "frozen": True,
        "accepted": accepted,
        "test_open_authorization": accepted,
        "persistent_composite_test_data_read": False,
        "unrelated_abandoned_architectures_previously_used_400_499": True,
        "forbidden_test_window": [400, 500],
        "core": {"path": str(CORE.resolve()), "sha256": sha256(CORE)},
        "low_guard": {"path": str(HEAD.resolve()), "sha256": sha256(HEAD), "scale": 1.0},
        "protocol": {"path": str(PROTOCOL.resolve()), "sha256": sha256(PROTOCOL)},
        "precommit": {"path": str(PRECOMMIT.resolve()), "sha256": precommit_sha},
        "authorization": {"path": str(AUTHORIZATION.resolve()), "sha256": authorization_sha},
        "access_receipt": {"path": str(receipt_path.resolve()), "sha256": receipt_sha},
        "validation_report": {"path": str(result_path.resolve()), "sha256": sha256(result_path)},
        "validation_rows": {
            name: {
                "path": str((OUTPUT_DIR / f"{name}_rows.csv").resolve()),
                "sha256": sha256(OUTPUT_DIR / f"{name}_rows.csv"),
            }
            for name in ("core", "guarded", "native")
        },
    }
    write_json(OUTPUT_DIR / "composite_frozen_manifest.json", composite)
    print(json.dumps({
        "status": report["status"],
        "High": candidate_summary["High"],
        "Medium": candidate_summary["Medium"],
        "Low": candidate_summary["Low"],
        "guarded_vs_native": candidate_vs_native,
        "paired_low_vs_native": paired_vs_native,
        "checks": checks,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
