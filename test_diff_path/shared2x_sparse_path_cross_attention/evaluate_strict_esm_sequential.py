from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
CLASSES = ("High", "Medium", "Low")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
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


def remove_topology_cache(model) -> None:
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    if hasattr(model, "_spc_transformer_cache"):
        model._spc_transformer_cache = None


def policy_actual_inputs(values, mode: str, num_ods: int, paths_per_od: int):
    """Return a model input tuple whose prediction tensors are untouched."""
    policy_values = list(values)
    replacement_l1 = 0.0
    for index in (2, 4, 6):
        actual = values[index]
        if mode == "zero":
            replacement = torch.zeros_like(actual)
        elif mode == "permute":
            grouped = actual.reshape(
                actual.shape[0], num_ods, paths_per_od, -1
            )
            replacement = torch.flip(grouped, dims=(1,)).reshape_as(actual)
        else:
            raise ValueError(f"Unknown policy actual-input mode: {mode}")
        policy_values[index] = replacement
        replacement_l1 += float((actual - replacement).abs().sum().item())
    return tuple(policy_values), replacement_l1


def build_strict_policy_cache(
    runtime,
    model,
    props,
    start: int,
    end: int,
    batch_size: int,
    actual_input_mode: str,
):
    """Infer policies with zero/permuted actual TM, retaining real TM offline.

    Only the policies produced from this altered model-input tuple enter the
    returned cache. Original actual traffic and oracle values are copied from
    the loader exclusively for the later sequential-admission stage.
    """
    dataset = runtime.DM_Dataset_within_Cluster(props, 0, start, end)
    if int(dataset.max_source_index_read) != end - 1:
        raise RuntimeError("Split-safe reader audit failed")
    path_masks = runtime.shared.base.move_dataset_static(dataset, props.device)
    buckets: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "p0", "p1", "p2",
            "tm0", "tm1", "tm2",
            "ptm0", "ptm1", "ptm2",
            "cap", "of0", "of1", "of2", "om0", "om1", "om2",
        )
    }
    loader = runtime.shared.data_loader(dataset, batch_size, False, seed=0)
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    props.research_return_admitted = False
    remove_topology_cache(model)
    replacement_l1 = 0.0
    model_actual_abs_max = 0.0
    model_actual_nonzero_count = 0
    model.eval()
    with torch.no_grad():
        for inputs in loader:
            original = runtime.shared.unpack_to_device(inputs, props)
            policy_values, batch_l1 = policy_actual_inputs(
                original,
                actual_input_mode,
                int(dataset.num_pairs),
                int(props.num_paths_per_pair),
            )
            replacement_l1 += batch_l1
            for index in (2, 4, 6):
                model_actual_abs_max = max(
                    model_actual_abs_max,
                    float(policy_values[index].abs().max().item()),
                )
                model_actual_nonzero_count += int(
                    torch.count_nonzero(policy_values[index]).item()
                )
            policies = runtime.cached_policy_forward(
                model, props, dataset, policy_values, path_masks
            )
            batch = int(original[2].shape[0])
            capacities = original[1]
            if capacities.shape[0] == 1 and batch > 1:
                capacities = capacities.expand(batch, -1)
            buckets["p0"].append(policies[0].detach())
            buckets["p1"].append(policies[1].detach())
            buckets["p2"].append(policies[2].detach())
            # These original actual tensors never went into the policy call.
            buckets["tm0"].append(original[2].detach())
            buckets["tm1"].append(original[4].detach())
            buckets["tm2"].append(original[6].detach())
            buckets["ptm0"].append(original[3].detach())
            buckets["ptm1"].append(original[5].detach())
            buckets["ptm2"].append(original[7].detach())
            buckets["cap"].append(capacities.detach())
            buckets["of0"].append(original[11].reshape(-1).detach())
            buckets["of1"].append((original[12] - original[11]).reshape(-1).detach())
            buckets["of2"].append((original[13] - original[12]).reshape(-1).detach())
            buckets["om0"].append(original[8].reshape(-1).detach())
            buckets["om1"].append(original[9].reshape(-1).detach())
            buckets["om2"].append(original[10].reshape(-1).detach())
    props.research_return_policy = False
    remove_topology_cache(model)
    joined = {key: torch.cat(value, dim=0) for key, value in buckets.items()}
    cache = runtime.PolicyCache(
        dataset=dataset,
        path_masks=path_masks,
        source_start=start,
        policies=(joined["p0"], joined["p1"], joined["p2"]),
        tms=(joined["tm0"], joined["tm1"], joined["tm2"]),
        predicted_tms=(joined["ptm0"], joined["ptm1"], joined["ptm2"]),
        capacities=joined["cap"],
        oracle_flows=(joined["of0"], joined["of1"], joined["of2"]),
        oracle_mlus=(joined["om0"], joined["om1"], joined["om2"]),
    )
    if replacement_l1 <= 0.0:
        raise RuntimeError("Actual-TM replacement changed no values")
    if actual_input_mode == "zero" and (
        model_actual_abs_max != 0.0 or model_actual_nonzero_count != 0
    ):
        raise RuntimeError("Literal-zero actual-TM policy input audit failed")
    return cache, {
        "mode": actual_input_mode,
        "replacement_l1": replacement_l1,
        "model_actual_input_abs_max": model_actual_abs_max,
        "model_actual_input_nonzero_count": model_actual_nonzero_count,
        "model_was_eval": not model.training,
        "dropout_modules_in_train_mode": [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Dropout) and module.training
        ],
    }


def policy_difference(expected, actual) -> dict:
    result = {}
    for class_name, before, after in zip(CLASSES, expected, actual):
        delta = (before - after).abs()
        before_grouped = before.squeeze(-1).reshape(before.shape[0], -1, 8)
        after_grouped = after.squeeze(-1).reshape_as(before_grouped)
        tv = 0.5 * (before_grouped - after_grouped).abs().sum(dim=-1)
        result[class_name] = {
            "torch_equal": bool(torch.equal(before, after)),
            "different_elements": int(torch.count_nonzero(before != after).item()),
            "max_abs_policy_delta": float(delta.max().item()),
            "mean_abs_policy_delta": float(delta.mean().item()),
            "argmax_changed_od_count": int(
                torch.count_nonzero(
                    before_grouped.argmax(dim=-1) != after_grouped.argmax(dim=-1)
                ).item()
            ),
            "max_od_total_variation": float(tv.max().item()),
        }
    return result


def checkpoint_state(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must be a mapping")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint has no model_state_dict")
    return checkpoint, state


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Cannot write an empty evaluation table")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def validate_rows(rows: list[dict], start: int, end: int) -> None:
    expected = (end - start) * len(CLASSES)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(rows)}")
    keys = {(int(row["snapshot"]), row["class"]) for row in rows}
    expected_keys = {
        (snapshot, class_name)
        for snapshot in range(start, end)
        for class_name in CLASSES
    }
    if keys != expected_keys:
        raise RuntimeError("Snapshot/class row coverage is incomplete or duplicated")
    for row in rows:
        for key, value in row.items():
            if key != "class" and not math.isfinite(float(value)):
                raise RuntimeError(f"Non-finite row value: {key}={value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict-ESM sequential-admission evaluation for one explicit sparse "
            "path cross-attention checkpoint. No checkpoint discovery or ranking."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--start", type=int, default=400)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--policy-batch-size", type=int, default=20)
    parser.add_argument("--admission-batch-size", type=int, default=20)
    parser.add_argument(
        "--counterfactual-audit-snapshots",
        type=int,
        default=1,
        help="Prefix length used to compare zero-actual and permuted-actual inference",
    )
    parser.add_argument(
        "--counterfactual-atol",
        type=float,
        default=None,
        help=(
            "Absolute numerical-noise floor. Default: 1e-5 on CUDA and exact "
            "on CPU. The audit also measures a same-input repeat baseline."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="-",
        help="Artifact directory, or '-' to print JSON without writing files",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")
    if min(args.policy_batch_size, args.admission_batch_size) <= 0:
        raise ValueError("Batch sizes must be positive")
    if not 0 <= args.counterfactual_audit_snapshots <= args.end - args.start:
        raise ValueError("Invalid counterfactual audit length")
    if args.counterfactual_atol is not None and args.counterfactual_atol < 0.0:
        raise ValueError("--counterfactual-atol must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")

    for path in (
        ROOT,
        TEST_DIR,
        TEST_DIR / "shared2x_medium_adapter",
        TEST_DIR / "shared2x_order_regularizer",
        TEST_DIR / "shared2x_full_objectives",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    runtime = load_module(
        "spc_strict_sequential_cache_runtime",
        TEST_DIR / "shared2x_medium_adapter" / "run_experiment.py",
    )
    spc = load_module(
        "spc_strict_sequential_model_runtime",
        THIS_DIR / "run_experiment.py",
    )

    if args.device == "cpu":
        torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    counterfactual_atol = (
        float(args.counterfactual_atol)
        if args.counterfactual_atol is not None
        else (1e-5 if device.type == "cuda" else 0.0)
    )
    props = runtime.build_props(args.level, device)
    checkpoint, state = checkpoint_state(checkpoint_path, device)
    required_keys = (
        "spc_query.weight",
        "spc_medium_kv.weight",
        "medium_pressure_init_adapter.2.weight",
        "medium_pressure_rau_adapter.2.weight",
    )
    missing = [key for key in required_keys if key not in state]
    if missing:
        raise RuntimeError(f"Not an R8,d16 sparse cross-attention checkpoint: {missing}")
    if int(state["spc_query.weight"].shape[0]) != int(spc.ATTENTION_DIM):
        raise RuntimeError("Checkpoint attention width does not match this evaluator")

    model = spc.SparsePathCrossAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    # The evaluated policies are produced with literal zero tensors in every
    # actual-TM input position. Original actual TM is retained only in cache.tms.
    cache, zero_input_audit = build_strict_policy_cache(
        runtime,
        model,
        props,
        args.start,
        args.end,
        args.policy_batch_size,
        actual_input_mode="zero",
    )

    counterfactual = {
        "audited_snapshot_count": args.counterfactual_audit_snapshots,
        "evaluated_policy_literal_zero_input_audit": zero_input_audit,
        "same_input_repeat_audit": None,
        "zero_vs_permuted_audit": None,
        "evaluation_batch_vs_audit_batch": None,
        "pass_all_classes_bitwise": None,
        "pass_all_classes_with_numerical_tolerance": None,
        "classification": None,
        "configured_absolute_tolerance": counterfactual_atol,
    }
    if args.counterfactual_audit_snapshots:
        audit_end = args.start + args.counterfactual_audit_snapshots
        audit_batch_size = min(
            args.policy_batch_size, args.counterfactual_audit_snapshots
        )
        # Both counterfactual arms use the same dataset length, batch shape,
        # eval mode, and cache lifecycle. A second identical zero-input run is
        # the empirical CUDA repeatability floor (sparse/scatter kernels can be
        # non-bitwise even with dropout disabled).
        audit_zero_cache, audit_zero_input = build_strict_policy_cache(
            runtime,
            model,
            props,
            args.start,
            audit_end,
            audit_batch_size,
            actual_input_mode="zero",
        )
        audit_zero_repeat_cache, audit_zero_repeat_input = build_strict_policy_cache(
            runtime,
            model,
            props,
            args.start,
            audit_end,
            audit_batch_size,
            actual_input_mode="zero",
        )
        permuted_cache, permuted_input = build_strict_policy_cache(
            runtime,
            model,
            props,
            args.start,
            audit_end,
            audit_batch_size,
            actual_input_mode="permute",
        )
        zero_prefix = tuple(
            policy[: args.counterfactual_audit_snapshots] for policy in cache.policies
        )
        batch_shape_difference = policy_difference(
            zero_prefix, audit_zero_cache.policies
        )
        repeat_difference = policy_difference(
            audit_zero_cache.policies, audit_zero_repeat_cache.policies
        )
        actual_difference = policy_difference(
            audit_zero_cache.policies, permuted_cache.policies
        )
        per_class_tolerance = {}
        per_class_pass = {}
        for class_name in CLASSES:
            repeat_max = repeat_difference[class_name]["max_abs_policy_delta"]
            tolerance = max(
                counterfactual_atol,
                4.0 * float(repeat_max)
                + (
                    10.0 * torch.finfo(props.dtype).eps
                    if device.type == "cuda"
                    else 0.0
                ),
            )
            per_class_tolerance[class_name] = tolerance
            per_class_pass[class_name] = (
                actual_difference[class_name]["max_abs_policy_delta"] <= tolerance
            )
            actual_difference[class_name]["numerical_tolerance"] = tolerance
            actual_difference[class_name]["within_numerical_tolerance"] = bool(
                per_class_pass[class_name]
            )
        bitwise_passed = all(
            row["torch_equal"] for row in actual_difference.values()
        )
        tolerance_passed = all(per_class_pass.values())
        repeat_is_non_bitwise = any(
            not row["torch_equal"] for row in repeat_difference.values()
        )
        classification = (
            "bitwise invariant"
            if bitwise_passed
            else (
                "CUDA/parallel numerical noise; no material actual-TM dependency"
                if tolerance_passed and repeat_is_non_bitwise
                else (
                    "within configured numerical tolerance"
                    if tolerance_passed
                    else "possible information dependency or numerical instability"
                )
            )
        )
        counterfactual.update(
            {
                "audit_zero_input": audit_zero_input,
                "audit_zero_repeat_input": audit_zero_repeat_input,
                "audit_permuted_input": permuted_input,
                "same_input_repeat_audit": repeat_difference,
                "zero_vs_permuted_audit": actual_difference,
                "evaluation_batch_vs_audit_batch": batch_shape_difference,
                "pass_all_classes_bitwise": bitwise_passed,
                "pass_all_classes_with_numerical_tolerance": tolerance_passed,
                "classification": classification,
            }
        )
        if not tolerance_passed:
            raise RuntimeError(
                "Actual-TM counterfactual exceeds repeatability-aware tolerance"
            )

    # A forward pre-hook deliberately forbids policy inference during replay.
    # evaluate_cache must call only model.simulate with cached policies + real TM.
    admission_forward_calls = 0

    def forbid_forward(_module, _inputs):
        nonlocal admission_forward_calls
        admission_forward_calls += 1
        raise RuntimeError("Policy forward was called during sequential admission")

    hook = model.register_forward_pre_hook(forbid_forward)
    try:
        rows, summary = runtime.evaluate_cache(
            model,
            props,
            cache,
            adapter=None,
            batch_size=args.admission_batch_size,
        )
    finally:
        hook.remove()
    validate_rows(rows, args.start, args.end)
    if admission_forward_calls != 0:
        raise RuntimeError("Admission stage unexpectedly invoked model.forward")

    payload = {
        "evaluation": "R8,d16 sparse cross-attention strict-ESM sequential admission",
        "window": [args.start, args.end],
        "load": "2x",
        "snapshot_count": args.end - args.start,
        "row_count": len(rows),
        "checkpoint_selection_contract": {
            "candidate_checkpoint": "explicit CLI input",
            "checkpoint_discovery_or_ranking_in_this_script": False,
            "test_metrics_used_to_select_checkpoint": False,
        },
        "information_flow_contract": {
            "policy_actual_tm_inputs": "literal zero tensors",
            "policy_prediction_inputs": "ESM tm1_pred/tm2_pred/tm3_pred",
            "real_actual_tm_storage": "offline cache fields only",
            "real_actual_tm_first_use": "model.simulate sequential admission",
            "admission_policy_forward_calls": admission_forward_calls,
            "policy_and_admission_are_separate_stages": True,
        },
        "architecture_load_audit": {
            "model_class": type(model).__name__,
            "attention_dim": int(spc.ATTENTION_DIM),
            "neighbor_ods": int(spc.NEIGHBOR_ODS),
            "feature_count": int(spc.TOTAL_FEATURES),
            "strict_state_dict_load": True,
            "checkpoint_epoch": checkpoint.get("epoch"),
        },
        "counterfactual_policy_audit": counterfactual,
        "metric_contract": {
            "admission": "exact sequential actual-TM Hattrick simulator",
            "class_order": list(CLASSES),
            "oracle": "incremental per-class MF oracle from dataset",
            "norm_fulfill": "class admitted traffic / incremental class MF oracle",
            "summary": "NumPy Mean/P1/P10 over snapshots, independently per class",
        },
        "summary": summary,
        "rows": rows,
        "artifacts": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "output_directory": None
            if args.output_dir == "-"
            else str(Path(args.output_dir).resolve()),
        },
    }

    if args.output_dir != "-":
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        rows_path = output_dir / "sequential_admission_rows.csv"
        summary_path = output_dir / "sequential_admission_summary.json"
        evaluation_path = output_dir / "sequential_admission_evaluation.json"
        write_csv_atomic(rows_path, rows)
        write_json_atomic(summary_path, summary)
        write_json_atomic(evaluation_path, payload)
        payload["artifacts"].update(
            {
                "rows_csv": str(rows_path),
                "rows_csv_sha256": sha256(rows_path),
                "summary_json": str(summary_path),
                "summary_json_sha256": sha256(summary_path),
                "evaluation_json": str(evaluation_path),
                "evaluation_json_sha256_before_self_reference": sha256(evaluation_path),
            }
        )
        # Refresh the manifest with hashes of the two non-self-referential files.
        # The evaluation file cannot contain a stable hash of itself.
        payload["artifacts"].pop("evaluation_json_sha256_before_self_reference")
        write_json_atomic(evaluation_path, payload)

    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
