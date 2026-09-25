from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "shared2x_stage2_tied_preview_smoke_runtime",
    THIS_DIR / "run_experiment.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load Stage2-tied preview runner")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def compare_policies(expected, actual, label: str, require_equal: bool = True) -> dict:
    comparison = {}
    for class_name, before, after in zip(
        ("High", "Medium", "Low"), expected, actual
    ):
        difference = (before - after).abs()
        row = {
            "torch_equal": bool(torch.equal(before, after)),
            "different_elements": int(torch.count_nonzero(before != after).item()),
            "max_abs_delta": float(difference.max().item()),
        }
        comparison[class_name] = row
        if require_equal and not row["torch_equal"]:
            raise RuntimeError(f"{label}: {class_name} policy differs")
    return comparison


def permute_actual_traffic(values, num_ods: int, paths_per_od: int):
    changed = list(values)
    total_l1 = 0.0
    for index in (2, 4, 6):
        actual = values[index]
        grouped = actual.reshape(actual.shape[0], num_ods, paths_per_od, -1)
        permuted = torch.flip(grouped, dims=(1,)).reshape_as(actual)
        changed[index] = permuted
        total_l1 += float((actual - permuted).abs().sum().item())
    if total_l1 == 0.0:
        raise RuntimeError("Actual traffic permutation changed nothing")
    return tuple(changed), total_l1


def timed_policy(model, props, dataset, values, path_masks) -> float:
    started = time.perf_counter()
    with torch.no_grad():
        experiment.full.shared.model_forward(
            model, props, dataset, values, path_masks
        )
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU architecture/projection/latency smoke for tied Stage2 preview"
    )
    parser.add_argument("--snapshot", type=int, default=350)
    parser.add_argument("--timing-runs", type=int, default=5)
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device("cpu")
    props = experiment.full.shared.build_props(4, device)
    props.checkpoint = 0
    dataset = experiment.full.DM_Dataset_within_Cluster(
        props, 0, args.snapshot, args.snapshot + 1
    )
    path_masks = experiment.full.shared.base.move_dataset_static(dataset, device)
    loader = experiment.full.shared.data_loader(
        dataset, batch_size=1, shuffle=False, seed=490
    )
    values = experiment.full.shared.unpack_to_device(next(iter(loader)), props)
    total_paths = int(dataset.pte.shape[0])
    paths_per_od = int(props.num_paths_per_pair)
    num_ods = total_paths // paths_per_od

    torch.manual_seed(20260824)
    native = experiment.NativeHattrick(props).to(device=device, dtype=props.dtype)
    torch.manual_seed(17)
    candidate = experiment.Stage2TiedPreviewHattrick(props).to(
        device=device, dtype=props.dtype
    )
    loaded = candidate.load_state_dict(native.state_dict(), strict=False)
    native_parameters = dict(native.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if not all(
        torch.equal(parameter, candidate_parameters[name])
        for name, parameter in native_parameters.items()
    ):
        raise RuntimeError("Native parameter copy is not exact")

    independent_names = [
        name
        for name in candidate_parameters
        if "medium_scout_init" in name or "medium_scout_rau" in name
    ]
    if independent_names:
        raise RuntimeError(f"Independent scout parameters remain: {independent_names}")
    adapters = (
        candidate.medium_pressure_init_adapter,
        candidate.medium_pressure_rau_adapter,
    )
    if any(
        int(torch.count_nonzero(adapter.body[2].weight).item()) != 0
        or int(torch.count_nonzero(adapter.body[2].bias).item()) != 0
        for adapter in adapters
    ):
        raise RuntimeError("A High residual output layer is not exactly zero")

    expected_pointers = {
        "mlp11": [int(p.data_ptr()) for p in candidate.mlp11.parameters()],
        "mlp12": [int(p.data_ptr()) for p in candidate.mlp12.parameters()],
        "mlp21": [int(p.data_ptr()) for p in candidate.mlp21.parameters()],
        "mlp22": [int(p.data_ptr()) for p in candidate.mlp22.parameters()],
        "mlp_22_violation": [
            int(p.data_ptr()) for p in candidate.mlp_22_violation.parameters()
        ],
    }
    tied_pointers = candidate.preview_parameter_data_ptrs
    if tied_pointers != expected_pointers:
        raise RuntimeError("Preview and formal parameter data_ptrs differ")

    props.mode = "test"
    props.research_return_policy = True
    props.research_return_admitted = False
    props.sim_mf_mlu = 0
    native.eval()
    candidate.eval()

    call_counts = {"mlp21": 0, "mlp22": 0}

    def count_mlp21(_module, _inputs, _output):
        call_counts["mlp21"] += 1

    def count_mlp22(_module, _inputs, _output):
        call_counts["mlp22"] += 1

    hook21 = candidate.mlp21[0].register_forward_hook(count_mlp21)
    hook22 = candidate.mlp22[0].register_forward_hook(count_mlp22)
    with torch.no_grad():
        native_policy, _ = experiment.full.shared.model_forward(
            native, props, dataset, values, path_masks
        )
        zero_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, values, path_masks
        )
    hook21.remove()
    hook22.remove()
    if call_counts["mlp21"] != 2:
        raise RuntimeError(f"Expected tied mlp21 twice, got {call_counts['mlp21']}")
    if call_counts["mlp22"] != int(props.rau2) + 1:
        raise RuntimeError(
            f"Expected tied mlp22 {int(props.rau2) + 1} times, "
            f"got {call_counts['mlp22']}"
        )
    zero_equivalence = compare_policies(
        native_policy, zero_policy, "epoch-0 native equivalence"
    )

    features = candidate._last_preview_features
    split = candidate._last_preview_split_ratios
    if features is None or tuple(features.shape) != (1, total_paths, 4):
        raise RuntimeError("Missing or malformed tied-preview features")
    if split is None:
        raise RuntimeError("Missing tied-preview Medium split")
    split_grouped = split.reshape(1, num_ods, paths_per_od)
    simplex_error = float((split_grouped.sum(dim=-1) - 1.0).abs().max().item())
    if simplex_error > 1e-6:
        raise RuntimeError(f"Preview Medium simplex error {simplex_error}")
    relative = features[..., (1, 3)].reshape(1, num_ods, paths_per_od, 2)
    center_error = float(relative.sum(dim=2).abs().max().item())
    if center_error > 1e-5:
        raise RuntimeError(f"Preview centered-feature error {center_error}")

    # Open the adapters so actual-TM independence is tested on an active path,
    # not merely hidden by epoch-0 zero outputs.
    with torch.no_grad():
        for adapter in adapters:
            adapter.body[2].weight.fill_(0.01)
        active_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, values, path_masks
        )
        changed_values, actual_permutation_l1 = permute_actual_traffic(
            values, num_ods, paths_per_od
        )
        changed_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, changed_values, path_masks
        )
    active_delta = compare_policies(
        zero_policy, active_policy, "active tied-preview delta", require_equal=False
    )
    if max(row["max_abs_delta"] for row in active_delta.values()) == 0.0:
        raise RuntimeError("Opening tied-preview adapters did not affect policy")
    strict_actual = compare_policies(
        active_policy, changed_policy, "strict actual-TM permutation"
    )

    # Exercise the complete inherited forward and all six backward passes used
    # by ordered projection.  Activated adapters ensure both layers receive a
    # real gradient rather than merely occupying zero-filled slices.
    candidate.train()
    props.mode = "train"
    props.research_return_policy = False
    props.research_return_admitted = True
    props.sim_mf_mlu = 0
    candidate.zero_grad(set_to_none=True)
    losses, reported = experiment.full.build_objectives(
        candidate, props, dataset, values, path_masks
    )
    if len(losses) != 6 or any(not torch.isfinite(loss).item() for loss in losses):
        raise RuntimeError("Complete six-objective forward is invalid")
    projection = experiment.full.ordered_project_gradients(candidate, losses)
    parameter_numel = sum(parameter.numel() for parameter in candidate.parameters())
    if int(projection.final_gradient.numel()) != parameter_numel:
        raise RuntimeError("Ordered projection omitted model parameters")
    if len(projection.parameter_shapes) != len(list(candidate.parameters())):
        raise RuntimeError("Ordered projection parameter-shape list is incomplete")
    if not torch.isfinite(projection.final_gradient).all().item():
        raise RuntimeError("Ordered projection returned a non-finite gradient")

    added_parameters = {
        name: parameter.numel()
        for name, parameter in candidate_parameters.items()
        if name not in native_parameters
    }
    offsets = {}
    offset = 0
    for name, parameter in candidate.named_parameters():
        next_offset = offset + parameter.numel()
        if name in added_parameters:
            offsets[name] = [offset, next_offset]
        offset = next_offset
    if sum(end - start for start, end in offsets.values()) != sum(
        added_parameters.values()
    ):
        raise RuntimeError("An added parameter is absent from projection ordering")
    added_raw_gradient_l1 = {
        name: max(
            float(raw[start:end].abs().sum().item())
            for raw in projection.raw_gradients
        )
        for name, (start, end) in offsets.items()
    }
    if any(value == 0.0 for value in added_raw_gradient_l1.values()):
        raise RuntimeError(
            f"An activated added parameter received no six-objective gradient: "
            f"{added_raw_gradient_l1}"
        )

    # Reset exact-zero adapters and compare warmed, cached-transformer CPU
    # policy latency in alternating order.  This is only a coarse <2x gate;
    # CUDA Events remain the formal benchmark.
    with torch.no_grad():
        for adapter in adapters:
            adapter.body[2].weight.zero_()
            adapter.body[2].bias.zero_()
    candidate.eval()
    native.eval()
    props.mode = "test"
    props.research_return_policy = True
    props.research_return_admitted = False
    with torch.no_grad():
        experiment.full.shared.model_forward(native, props, dataset, values, path_masks)
        experiment.full.shared.model_forward(candidate, props, dataset, values, path_masks)
    native_seconds = []
    candidate_seconds = []
    for _ in range(args.timing_runs):
        native_seconds.append(timed_policy(native, props, dataset, values, path_masks))
        candidate_seconds.append(
            timed_policy(candidate, props, dataset, values, path_masks)
        )
    native_median = statistics.median(native_seconds)
    candidate_median = statistics.median(candidate_seconds)
    latency_ratio = candidate_median / max(native_median, 1e-12)
    if latency_ratio >= 2.0:
        raise RuntimeError(f"CPU inference latency ratio {latency_ratio:.3f} >= 2")

    result = {
        "status": "ok",
        "device": "cpu",
        "snapshot": args.snapshot,
        "strict_input": "ESM tm1_pred/tm2_pred; actual TM excluded from preview",
        "actual_permutation_l1": actual_permutation_l1,
        "formal_cascade_inherited_via_super_forward": True,
        "independent_scout_parameter_names": independent_names,
        "tied_parameter_data_ptrs_equal": tied_pointers == expected_pointers,
        "tied_formal_call_counts": call_counts,
        "native_parameters": sum(p.numel() for p in native.parameters()),
        "candidate_parameters": parameter_numel,
        "added_parameters": sum(added_parameters.values()),
        "added_parameter_tensors": added_parameters,
        "missing_state_keys": list(loaded.missing_keys),
        "unexpected_state_keys": list(loaded.unexpected_keys),
        "feature_shape": list(features.shape),
        "feature_min": float(features.min().item()),
        "feature_max": float(features.max().item()),
        "relative_center_max_error": center_error,
        "preview_medium_simplex_max_error": simplex_error,
        "preview_audit": candidate.last_preview_audit,
        "zero_output_native_policy": zero_equivalence,
        "active_preview_policy_delta": active_delta,
        "actual_tm_permutation_policy": strict_actual,
        "projection": {
            "objective_names": list(experiment.full.OBJECTIVE_NAMES),
            "reported_values": [float(value) for value in reported],
            "flat_parameter_count": int(projection.final_gradient.numel()),
            "parameter_tensor_count": len(projection.parameter_shapes),
            "added_parameter_offsets": offsets,
            "added_parameter_max_raw_gradient_l1": added_raw_gradient_l1,
            "final_gradient_l2": float(
                torch.linalg.vector_norm(projection.final_gradient).item()
            ),
        },
        "latency": {
            "timing_runs": args.timing_runs,
            "native_seconds": native_seconds,
            "candidate_seconds": candidate_seconds,
            "native_median_seconds": native_median,
            "candidate_median_seconds": candidate_median,
            "candidate_over_native": latency_ratio,
            "passes_strict_2x": latency_ratio < 2.0,
        },
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
