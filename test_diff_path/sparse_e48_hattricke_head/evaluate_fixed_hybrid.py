from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
SPARSE_DIR = TEST_DIR / "shared2x_sparse_path_cross_attention"
EDGE_DIR = TEST_DIR / "shared2x_edge_toll_head"
STRICT_EVALUATOR = SPARSE_DIR / "evaluate_strict_esm_sequential.py"
SPARSE_RUNNER = SPARSE_DIR / "run_experiment.py"
EDGE_RUNNER = EDGE_DIR / "run_experiment.py"
SELECTION_MANIFEST = (
    SPARSE_DIR
    / "checkpoint_selection_validation_only"
    / "manifest_frozen_epoch_060.json"
)
SPARSE_CHECKPOINT = (
    SPARSE_DIR
    / "checkpoint_selection_validation_only"
    / "archive"
    / "epoch_048.pt"
)
EDGE_CHECKPOINT = EDGE_DIR / "best_edge_toll_head.pt"
HATTRICK_E_ROWS = EDGE_DIR / "strict_esm_2x_candidate_rows.csv"
OUTPUT_DIR = ROOT / "output" / "comparisons" / "sparse_e48_hattricke_head_strict2x"

# These labels are deliberately module-level so a second, hard-locked evaluator
# can reuse the exact same replay/audit implementation while identifying a
# separately trained head and its experiment-design status truthfully.
METHOD_NAME = (
    "sparse cross-attention e48 High + frozen Hattrick-e Medium/Low edge-toll"
)
METHOD_DESCRIPTION = (
    "Fixed sparse-e48 High backbone plus frozen Hattrick-e Medium/Low "
    "edge-toll head; strict ESM sequential admission"
)
HEAD_LABEL = "frozen Hattrick-e head"
EXPERIMENT_DIRECTION_TEST_INFORMED = False
EXPERIMENT_DESIGN_CAVEAT = None

EXPECTED_HASHES = {
    SPARSE_CHECKPOINT: "55dfbe1506148099c838f3e5e08ca871f280877e56748ae3ea5af574c0a3acad",
    SELECTION_MANIFEST: "a481a9ff750a00a5ab8abc5a76a66e8db05c82c0cecca7d1f807437282e09d6d",
    EDGE_CHECKPOINT: "00cdaf25b1fd08e8c5f5f084b1b10e04c3dfbfef58ae7386aac500208461753a",
    SPARSE_RUNNER: "3b40dff9801bc71e6606888d8a690adbe2e695b52fd1a4710f6bb97b71b19f56",
    EDGE_RUNNER: "839c6a8b2a091553fe817abedc97016b35e498980a4eb2d8f936cb5cbae4c866",
}
CLASSES = ("High", "Medium", "Low")
NUMERIC_ROW_FIELDS = (
    "admitted_traffic",
    "demand",
    "fulfill_ratio",
    "oracle_admitted_traffic",
    "norm_fulfill",
    "raw_mlu",
    "oracle_mlu",
    "normalized_mlu",
    "disabled_flow",
    "admitted_capacity_ratio",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
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


def verify_frozen_inputs() -> dict:
    result = {}
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen input changed: {path}; expected {expected}, got {actual}"
            )
        result[str(path.resolve())] = actual

    manifest = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    selected = manifest["selection"]["selected"]
    if manifest.get("test_data_read") is not False:
        raise RuntimeError("Sparse selection manifest does not exclude test data")
    if int(selected["epoch"]) != 48:
        raise RuntimeError("Sparse validation-only selection is not epoch 48")
    if selected["checkpoint"]["sha256"] != EXPECTED_HASHES[SPARSE_CHECKPOINT]:
        raise RuntimeError("Sparse selected-checkpoint hash differs from frozen input")
    return {
        "hashes": result,
        "sparse_selection": {
            "frozen": manifest["frozen"],
            "test_data_read": manifest["test_data_read"],
            "selection_tier": manifest["selection"]["selection_tier"],
            "epoch": selected["epoch"],
            "checkpoint_sha256": selected["checkpoint"]["sha256"],
            "validation_window": manifest["policy"]["allowed_candidate_window"],
            "forbidden_test_window": manifest["policy"]["forbidden_test_window"],
        },
        "checkpoint_discovery_or_ranking_in_hybrid": False,
        "test_metrics_used_to_choose_backbone_or_head": False,
        "experiment_direction_test_informed": EXPERIMENT_DIRECTION_TEST_INFORMED,
        "experiment_design_caveat": EXPERIMENT_DESIGN_CAVEAT,
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
        path_features=None,
    )


def validate_rows(rows: list[dict], start: int, end: int) -> None:
    expected_keys = {
        (snapshot, class_name)
        for snapshot in range(start, end)
        for class_name in CLASSES
    }
    actual_keys = {(int(row["snapshot"]), row["class"]) for row in rows}
    if len(rows) != len(expected_keys) or actual_keys != expected_keys:
        raise RuntimeError("Evaluation row coverage is incomplete or duplicated")
    for row in rows:
        for field in NUMERIC_ROW_FIELDS:
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"Non-finite {field} in evaluation rows")


def rows_by_key(rows: list[dict]) -> dict[tuple[int, str], dict]:
    return {(int(row["snapshot"]), row["class"]): row for row in rows}


def high_exact_audit(baseline_rows: list[dict], hybrid_rows: list[dict]) -> dict:
    baseline = rows_by_key(baseline_rows)
    hybrid = rows_by_key(hybrid_rows)
    fields = NUMERIC_ROW_FIELDS
    differences = {
        field: max(
            abs(
                float(baseline[(snapshot, "High")][field])
                - float(hybrid[(snapshot, "High")][field])
            )
            for snapshot in range(400, 500)
        )
        for field in fields
    }
    exact = all(value == 0.0 for value in differences.values())
    if not exact:
        raise RuntimeError(f"Hybrid changed High evaluation rows: {differences}")
    return {"all_numeric_fields_exact": exact, "max_abs_delta": differences}


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def summary_delta(candidate: list[dict], reference: list[dict]) -> dict:
    candidate_index = summary_index(candidate)
    reference_index = summary_index(reference)
    result = {}
    for class_name in CLASSES:
        result[class_name] = {
            suffix: float(candidate_index[class_name][f"norm_fulfill_{suffix}"])
            - float(reference_index[class_name][f"norm_fulfill_{suffix}"])
            for suffix in ("mean", "p1", "p10")
        }
    return result


def read_external_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for row in rows:
        converted.append(
            {
                key: (
                    value
                    if key == "class"
                    else int(value)
                    if key == "snapshot"
                    else float(value)
                )
                for key, value in row.items()
            }
        )
    return converted


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=METHOD_DESCRIPTION
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--policy-batch-size", type=int, default=20)
    parser.add_argument("--admission-batch-size", type=int, default=20)
    parser.add_argument("--audit-snapshots", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not 1 <= args.audit_snapshots <= 100:
        raise ValueError("audit-snapshots must be in [1,100]")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_audit = verify_frozen_inputs()
    strict = load_module("sparse_e48_hybrid_strict_runtime", STRICT_EVALUATOR)
    edge = load_module("sparse_e48_hybrid_edge_runtime", EDGE_RUNNER)
    spc = load_module("sparse_e48_hybrid_model_runtime", SPARSE_RUNNER)
    runtime = edge.runtime

    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)
    props = runtime.build_props(4, device)

    sparse_payload = torch.load(
        SPARSE_CHECKPOINT, map_location=device, weights_only=False
    )
    if int(sparse_payload.get("epoch", -1)) != 48:
        raise RuntimeError("Frozen sparse checkpoint payload is not epoch 48")
    sparse_state = sparse_payload["model_state_dict"]
    model = spc.SparsePathCrossAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    sparse_load = model.load_state_dict(sparse_state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    head_payload = torch.load(EDGE_CHECKPOINT, map_location=device, weights_only=False)
    if int(head_payload["feature_count"]) != 7 or int(head_payload["edge_count"]) != 72:
        raise RuntimeError(f"{HEAD_LABEL} contract is not 72 edges x 7 features")
    if "test_data_read" in head_payload and head_payload["test_data_read"] is not False:
        raise RuntimeError(f"{HEAD_LABEL} reports test data use during selection")
    if (
        "backbone_sha256" in head_payload
        and head_payload["backbone_sha256"] != EXPECTED_HASHES[SPARSE_CHECKPOINT]
    ):
        raise RuntimeError(f"{HEAD_LABEL} was not trained against sparse epoch 48")
    head = edge.EdgeTollHead(
        int(head_payload["edge_count"]), int(head_payload["feature_count"])
    ).to(device)
    head_load = head.load_state_dict(head_payload["state_dict"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    started = time.perf_counter()
    # The policy forward receives literal-zero actual TMs. Real TMs are stored
    # offline in cache.tms and first used by sequential admission below.
    cache, zero_policy_audit = strict.build_strict_policy_cache(
        runtime,
        model,
        props,
        400,
        500,
        args.policy_batch_size,
        actual_input_mode="zero",
    )
    causal_features = edge.edge_features(cache)
    if tuple(causal_features.shape) != (100, 72, 7):
        raise RuntimeError(f"Unexpected causal feature shape {causal_features.shape}")
    if not torch.isfinite(causal_features).all().item():
        raise RuntimeError("Non-finite causal edge features")

    # Independent feature-formula audit: channel 0 and residual channels must
    # be recomputed from sparse-e48 High, not inherited from old Hattrick-e.
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    capacities = cache.capacities.to(dtype=torch.float32).clamp_min(1e-9)
    predicted = [value.squeeze(-1).to(torch.float32) for value in cache.predicted_tms]
    policies = [value.squeeze(-1).to(torch.float32) for value in cache.policies]
    loads = [
        torch.sparse.mm(pte.t(), (policy * demand).t()).t() / capacities
        for policy, demand in zip(policies, predicted)
    ]
    high_load, medium_load, low_load = loads
    manual_features = torch.stack(
        [
            high_load,
            medium_load,
            low_load,
            high_load + medium_load,
            high_load + medium_load + low_load,
            torch.relu(1.0 - high_load),
            torch.relu(1.0 - high_load - medium_load),
        ],
        dim=-1,
    ).detach()
    feature_formula_delta = float((manual_features - causal_features).abs().max().item())
    if feature_formula_delta > 2e-6:
        raise RuntimeError(
            "Causal feature recomputation formula mismatch: "
            f"max_abs_delta={feature_formula_delta}"
        )

    candidate_cache, gates = edge.build_candidate_cache(
        head, cache, causal_features, batch_size=args.policy_batch_size
    )
    adapter = edge.TwoPolicyAdapter(int(cache.policies[0].shape[1]))

    # Contract-level High check before admission: the adapter replaces only two
    # path-feature slices and preserves the exact same High tensor object.
    probe_indices = torch.arange(min(4, len(cache)), device=device)
    probe_batch = runtime.select_cache(candidate_cache, probe_indices)
    probe_policies = list(probe_batch["policies"])
    high_before = probe_policies[0]
    adapted_probe = adapter.adapt_batch(probe_policies, probe_batch)
    policy_high_exact = bool(torch.equal(high_before, adapted_probe[0]))
    policy_high_same_storage = bool(high_before.data_ptr() == adapted_probe[0].data_ptr())
    if not policy_high_exact or not policy_high_same_storage:
        raise RuntimeError("Hattrick-e head changed High policy")

    # Counterfactual audit covers both the sparse backbone and downstream head.
    audit_end = 400 + args.audit_snapshots
    permuted_cache, permuted_policy_audit = strict.build_strict_policy_cache(
        runtime,
        model,
        props,
        400,
        audit_end,
        min(args.policy_batch_size, args.audit_snapshots),
        actual_input_mode="permute",
    )
    zero_prefix = slice_cache(cache, args.audit_snapshots)
    backbone_counterfactual = strict.policy_difference(
        zero_prefix.policies, permuted_cache.policies
    )
    if any(not row["torch_equal"] for row in backbone_counterfactual.values()):
        raise RuntimeError("Actual-TM counterfactual changed sparse backbone policies")
    zero_prefix_features = edge.edge_features(zero_prefix)
    permuted_features = edge.edge_features(permuted_cache)
    feature_counterfactual_delta = float(
        (zero_prefix_features - permuted_features).abs().max().item()
    )
    if feature_counterfactual_delta != 0.0:
        raise RuntimeError("Actual traffic changed Hattrick-e causal features")
    zero_head_cache, zero_gates = edge.build_candidate_cache(
        head, zero_prefix, zero_prefix_features, batch_size=args.audit_snapshots
    )
    permuted_head_cache, permuted_gates = edge.build_candidate_cache(
        head, permuted_cache, permuted_features, batch_size=args.audit_snapshots
    )
    head_policy_counterfactual_delta = float(
        (zero_head_cache.path_features - permuted_head_cache.path_features)
        .abs()
        .max()
        .item()
    )
    gate_counterfactual_delta = float(
        (zero_gates - permuted_gates).abs().max().item()
    )
    if head_policy_counterfactual_delta != 0.0 or gate_counterfactual_delta != 0.0:
        raise RuntimeError(f"Actual traffic changed {HEAD_LABEL} output")

    # During final replay, any model.forward call is an error: only cached
    # strict policies and real actual TMs may reach model.simulate.
    admission_forward_calls = 0

    def forbid_forward(_module, _inputs):
        nonlocal admission_forward_calls
        admission_forward_calls += 1
        raise RuntimeError("Policy forward called during admission replay")

    hook = model.register_forward_pre_hook(forbid_forward)
    try:
        sparse_rows, sparse_summary = runtime.evaluate_cache(
            model,
            props,
            cache,
            adapter=None,
            batch_size=args.admission_batch_size,
        )
        hybrid_rows, hybrid_summary = runtime.evaluate_cache(
            model,
            props,
            candidate_cache,
            adapter=adapter,
            batch_size=args.admission_batch_size,
        )
    finally:
        hook.remove()
    if admission_forward_calls != 0:
        raise RuntimeError("Admission replay invoked model.forward")
    validate_rows(sparse_rows, 400, 500)
    validate_rows(hybrid_rows, 400, 500)
    high_audit = high_exact_audit(sparse_rows, hybrid_rows)

    # Existing Hattrick-e rows are reporting-only. They do not choose or tune
    # either frozen component in this script.
    hattrick_e_rows = read_external_rows(HATTRICK_E_ROWS)
    validate_rows(hattrick_e_rows, 400, 500)
    hattrick_e_summary = runtime.summarize_rows(hattrick_e_rows)
    elapsed = time.perf_counter() - started

    summary_payload = {
        "hybrid": hybrid_summary,
        "sparse_e48": sparse_summary,
        "existing_hattrick_e": hattrick_e_summary,
        "hybrid_minus_sparse_e48": summary_delta(hybrid_summary, sparse_summary),
        "hybrid_minus_existing_hattrick_e": summary_delta(
            hybrid_summary, hattrick_e_summary
        ),
    }
    evaluation_payload = {
        "method": METHOD_NAME,
        "window": [400, 500],
        "load": "2x",
        "strict_esm": True,
        "fixed_components": frozen_audit,
        "head_contract": {
            "label": HEAD_LABEL,
            "edge_count": int(head.edge_count),
            "fixed_feature_count": int(head_payload["feature_count"]),
            "learned_edge_embedding_channels": 6,
            "local_input_width": int(head.local[0].in_features),
            "outputs": ["Medium edge toll", "Low edge toll"],
            "high_output": None,
            "base_medium_low": "sparse-e48 formal serial Medium/Low policies",
            "causal_features": (
                "recomputed after sparse-e48 strict policy inference from current "
                "High/Medium/Low policies, ESM predictions, PTE and capacities"
            ),
            "causal_feature_shape": list(causal_features.shape),
            "causal_feature_formula_max_delta": feature_formula_delta,
            "state_dict_strict": True,
            "missing_keys": list(head_load.missing_keys),
            "unexpected_keys": list(head_load.unexpected_keys),
            "head_selection_payload": head_payload["selection"],
            "head_test_data_read": head_payload.get("test_data_read", "not-recorded"),
            "head_backbone_epoch": head_payload.get("backbone_epoch", "not-recorded"),
            "head_backbone_sha256": head_payload.get(
                "backbone_sha256", "not-recorded"
            ),
        },
        "high_preservation": {
            "adapter_high_torch_equal": policy_high_exact,
            "adapter_high_same_storage": policy_high_same_storage,
            "sequential_admission_rows": high_audit,
        },
        "information_flow": {
            "policy_actual_tm_inputs": "literal zero tensors",
            "policy_predictions": "ESM tm1_pred/tm2_pred/tm3_pred",
            "real_actual_tm_first_use": "sequential model.simulate after policy/head cache frozen",
            "admission_forward_calls": admission_forward_calls,
            "zero_policy_cache_audit": zero_policy_audit,
            "permuted_policy_cache_audit": permuted_policy_audit,
            "backbone_counterfactual": backbone_counterfactual,
            "causal_feature_counterfactual_max_delta": feature_counterfactual_delta,
            "head_policy_counterfactual_max_delta": head_policy_counterfactual_delta,
            "head_gate_counterfactual_max_delta": gate_counterfactual_delta,
        },
        "head_activity": {
            "medium_gate_mean": float(gates[:, 0].mean().item()),
            "low_gate_mean": float(gates[:, 1].mean().item()),
            "medium_policy_max_abs_delta": float(
                (
                    candidate_cache.path_features[:, : cache.policies[1].shape[1]]
                    - cache.policies[1].squeeze(-1)
                )
                .abs()
                .max()
                .item()
            ),
            "low_policy_max_abs_delta": float(
                (
                    candidate_cache.path_features[:, cache.policies[1].shape[1] :]
                    - cache.policies[2].squeeze(-1)
                )
                .abs()
                .max()
                .item()
            ),
        },
        "summary": summary_payload,
        "runtime": {
            "device": str(device),
            "seconds": elapsed,
            "policy_batch_size": args.policy_batch_size,
            "admission_batch_size": args.admission_batch_size,
        },
        "source_hashes": {
            "evaluate_fixed_hybrid.py": sha256(Path(__file__).resolve()),
            "strict_evaluator": sha256(STRICT_EVALUATOR),
            "hattrick_e_rows": sha256(HATTRICK_E_ROWS),
        },
        "sparse_state_load": {
            "strict": True,
            "missing_keys": list(sparse_load.missing_keys),
            "unexpected_keys": list(sparse_load.unexpected_keys),
        },
    }

    hybrid_rows_path = output_dir / "hybrid_rows.csv"
    sparse_rows_path = output_dir / "sparse_e48_rows.csv"
    summary_path = output_dir / "comparison_summary.json"
    evaluation_path = output_dir / "evaluation_audit.json"
    write_csv(hybrid_rows_path, hybrid_rows)
    write_csv(sparse_rows_path, sparse_rows)
    write_json(summary_path, summary_payload)
    evaluation_payload["output_hashes"] = {
        "hybrid_rows.csv": sha256(hybrid_rows_path),
        "sparse_e48_rows.csv": sha256(sparse_rows_path),
        "comparison_summary.json": sha256(summary_path),
    }
    write_json(evaluation_path, evaluation_payload)
    print(json.dumps(evaluation_payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
