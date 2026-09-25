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
DEFAULT_R8_CHECKPOINT = (
    THIS_DIR.parent
    / "shared2x_sparse_path_cross_attention"
    / "artifacts"
    / "level2_proxy"
    / "seed_490"
    / "best_model.pt"
)
spec = importlib.util.spec_from_file_location(
    "shared2x_sparse_path_cross_attention_fast_smoke_runtime",
    THIS_DIR / "run_experiment.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load fast sparse path cross-attention runner")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def compare_policies(expected, actual, label: str, require_equal: bool = True):
    result = {}
    for class_name, before, after in zip(("High", "Medium", "Low"), expected, actual):
        difference = (before - after).abs()
        row = {
            "torch_equal": bool(torch.equal(before, after)),
            "different_elements": int(torch.count_nonzero(before != after).item()),
            "max_abs_delta": float(difference.max().item()),
        }
        if require_equal and not row["torch_equal"]:
            raise RuntimeError(f"{label}: {class_name} policy differs")
        result[class_name] = row
    return result


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
        raise RuntimeError("Actual-TM permutation changed nothing")
    return tuple(changed), total_l1


def timed(model, props, dataset, values, path_masks, runs: int):
    elapsed = []
    with torch.no_grad():
        for _ in range(runs):
            started = time.perf_counter()
            experiment.full.shared.model_forward(model, props, dataset, values, path_masks)
            elapsed.append(time.perf_counter() - started)
    return elapsed


def final_layer_modules(model):
    return (
        model.medium_pressure_init_adapter[2],
        model.medium_pressure_rau_adapter[2],
    )


def projection_slice_l1(model, result, parameter_name: str) -> list[float]:
    offset = 0
    selected = None
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if name == parameter_name:
            selected = (offset, offset + count)
            break
        offset += count
    if selected is None:
        raise RuntimeError(f"Missing parameter {parameter_name}")
    start, end = selected
    return [float(value[start:end].abs().sum().item()) for value in result.raw_gradients]


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU smoke for fast sparse cross-attention")
    parser.add_argument("--snapshot", type=int, default=350)
    parser.add_argument("--timing-runs", type=int, default=3)
    parser.add_argument("--r8-checkpoint", type=Path, default=DEFAULT_R8_CHECKPOINT)
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device("cpu")
    props = experiment.full.shared.build_props(4, device)
    props.checkpoint = 0
    dataset = experiment.full.DM_Dataset_within_Cluster(
        props, 0, args.snapshot, args.snapshot + 1
    )
    path_masks = experiment.full.shared.base.move_dataset_static(dataset, device)
    loader = experiment.full.shared.data_loader(dataset, 1, False, seed=490)
    values = experiment.full.shared.unpack_to_device(next(iter(loader)), props)
    pte = dataset.pte.coalesce()
    pte_indices = pte.indices()
    pte_info = (pte, pte_indices[0], pte_indices[1], pte.values())
    total_paths = int(pte.shape[0])
    paths_per_od = int(props.num_paths_per_pair)
    num_ods = total_paths // paths_per_od

    torch.manual_seed(20260824)
    native = experiment.NativeHattrick(props).to(dtype=props.dtype)
    torch.manual_seed(17)
    candidate = experiment.SparsePathCrossAttentionFastHattrick(props).to(dtype=props.dtype)
    loaded = candidate.load_state_dict(native.state_dict(), strict=False)
    native_parameters = dict(native.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if not all(
        torch.equal(value, candidate_parameters[name])
        for name, value in native_parameters.items()
    ):
        raise RuntimeError("Native parameter copy is not exact")
    for layer in final_layer_modules(candidate):
        if torch.count_nonzero(layer.weight).item() or torch.count_nonzero(layer.bias).item():
            raise RuntimeError("Residual output layer is not exactly zero")

    props.mode = "test"
    props.research_return_policy = True
    props.research_return_admitted = False
    props.sim_mf_mlu = 0
    native.eval()
    candidate.eval()
    with torch.no_grad():
        native_policy, _ = experiment.full.shared.model_forward(
            native, props, dataset, values, path_masks
        )
        zero_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, values, path_masks
        )
    zero_equivalence = compare_policies(
        native_policy, zero_policy, "zero-output native equivalence"
    )
    cache_pointer = int(candidate._spc_neighbor_od.data_ptr())
    if candidate.sparse_cache_builds != 1:
        raise RuntimeError("Static sparse cache was not built exactly once")

    candidate.research_spc_capture_attention = True
    candidate.zero_grad(set_to_none=True)
    tm2_probe = values[5].detach().clone().requires_grad_(True)
    features = candidate.compute_medium_pressure_features(
        tm2_probe,
        values[1],
        pte,
        pte_info,
        1,
        paths_per_od,
        props,
        path_masks,
    )
    features.square().mean().backward()
    if tm2_probe.grad is None or not torch.isfinite(tm2_probe.grad).all().item():
        raise RuntimeError("Missing/non-finite tm2_pred gradient")
    attention_gradient_l1 = sum(
        float(parameter.grad.abs().sum().item())
        for name, parameter in candidate.named_parameters()
        if name.startswith("spc_") and parameter.grad is not None
    )
    if attention_gradient_l1 == 0.0:
        raise RuntimeError("Cross-attention gradient is zero")
    if candidate._last_route_attention is None or candidate._last_od_attention is None:
        raise RuntimeError("Requested attention audit was not captured")
    route_simplex_error = float(
        (candidate._last_route_attention.sum(dim=-1) - 1.0).abs().max().item()
    )
    od_simplex_error = float(
        (candidate._last_od_attention.sum(dim=-1) - 1.0).abs().max().item()
    )
    if max(route_simplex_error, od_simplex_error) > 1e-6:
        raise RuntimeError("Attention simplex check failed")
    candidate.research_spc_capture_attention = False
    if candidate.sparse_cache_builds != 1 or int(candidate._spc_neighbor_od.data_ptr()) != cache_pointer:
        raise RuntimeError("Static sparse cache was unexpectedly rebuilt")

    own_od = torch.arange(total_paths) // paths_per_od
    if not torch.equal(candidate._spc_neighbor_od[:, 0].cpu(), own_od):
        raise RuntimeError("Own Medium OD is not the first neighbor")
    sorted_neighbors = torch.sort(candidate._spc_neighbor_od, dim=-1).values
    if bool((sorted_neighbors[:, 1:] == sorted_neighbors[:, :-1]).any().item()):
        raise RuntimeError("Duplicate Medium OD in a sparse neighborhood")

    torch.manual_seed(20260824)
    with torch.no_grad():
        for layer in final_layer_modules(candidate):
            layer.weight.normal_(mean=0.0, std=0.02)
            layer.bias.zero_()
        active_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, values, path_masks
        )
        changed_values, actual_permutation_l1 = permute_actual_traffic(
            values, num_ods, paths_per_od
        )
        changed_actual_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, changed_values, path_masks
        )
    active_delta = compare_policies(
        zero_policy, active_policy, "manually opened residual", require_equal=False
    )
    if max(row["max_abs_delta"] for row in active_delta.values()) == 0.0:
        raise RuntimeError("Nonzero residual output did not affect policy")
    actual_independence = compare_policies(
        active_policy, changed_actual_policy, "actual-TM replacement"
    )

    with torch.no_grad():
        for layer in final_layer_modules(candidate):
            layer.weight.zero_()
            layer.bias.zero_()
        experiment.full.shared.model_forward(native, props, dataset, values, path_masks)
        experiment.full.shared.model_forward(candidate, props, dataset, values, path_masks)
    native_seconds = timed(native, props, dataset, values, path_masks, args.timing_runs)
    candidate_seconds = timed(candidate, props, dataset, values, path_masks, args.timing_runs)
    native_median = statistics.median(native_seconds)
    candidate_median = statistics.median(candidate_seconds)
    latency_ratio = candidate_median / max(native_median, 1e-12)
    if latency_ratio > 2.0:
        raise RuntimeError(f"Cached CPU latency ratio {latency_ratio:.3f} > 2")

    candidate.zero_grad(set_to_none=True)
    candidate.train()
    props.mode = "train"
    props.research_return_policy = False
    props.research_return_admitted = True
    losses, _ = experiment.full.build_objectives(
        candidate, props, dataset, values, path_masks
    )
    projection = experiment.full.ordered_project_gradients(candidate, losses)
    parameter_list = list(candidate.parameters())
    shape_compatible = (
        len(projection.parameter_shapes) == len(parameter_list)
        and all(
            tuple(shape) == tuple(parameter.shape)
            for shape, parameter in zip(projection.parameter_shapes, parameter_list)
        )
        and projection.final_gradient.numel()
        == sum(parameter.numel() for parameter in parameter_list)
    )
    if not shape_compatible or not torch.isfinite(projection.final_gradient).all().item():
        raise RuntimeError("Ordered projection parameter shape/finite audit failed")
    init_output_gradients = projection_slice_l1(
        candidate, projection, "medium_pressure_init_adapter.2.weight"
    )
    rau_output_gradients = projection_slice_l1(
        candidate, projection, "medium_pressure_rau_adapter.2.weight"
    )
    if max(init_output_gradients) == 0.0 or max(rau_output_gradients) == 0.0:
        raise RuntimeError("A zero output layer cannot learn on the first projected step")

    warmstart = None
    if args.r8_checkpoint.is_file():
        warm_model = experiment.SparsePathCrossAttentionFastHattrick(props).to(dtype=props.dtype)
        warmstart = experiment.load_r8d16_warmstart(warm_model, args.r8_checkpoint)
        if warmstart["skipped_target_keys"]:
            raise RuntimeError(
                f"Warm-start did not map target keys: {warmstart['skipped_target_keys']}"
            )
        old_checkpoint = torch.load(
            args.r8_checkpoint, map_location="cpu", weights_only=False
        )["model_state_dict"]
        expected_output = torch.cat(
            (
                old_checkpoint["spc_output.0.weight"][:, :8],
                old_checkpoint["spc_output.0.weight"][:, 16:24],
            ),
            dim=1,
        )[:8, :16]
        if not torch.equal(warm_model.spc_output[0].weight.cpu(), expected_output):
            raise RuntimeError("Warm-start context/query slice is incorrect")
        if torch.count_nonzero(warm_model.medium_pressure_init_adapter[2].weight).item() == 0:
            raise RuntimeError("Trained R8 output adapter was not transferred")

    bytes_per_float = 4
    r8_full_floats = total_paths * 8 * paths_per_od * 16
    fast_full_floats = total_paths * 4 * paths_per_od * 8
    fast_stream_floats = total_paths * paths_per_od * 8
    result = {
        "status": "ok",
        "device": "cpu",
        "snapshot": args.snapshot,
        "forward_is_inherited_native": (
            experiment.SparsePathCrossAttentionFastHattrick.forward
            is experiment.NativeHattrick.forward
        ),
        "strict_input": "ESM tm2_pred and static topology/capacity only",
        "actual_permutation_l1": actual_permutation_l1,
        "feature_shape": list(features.shape),
        "feature_finite": bool(torch.isfinite(features.detach()).all().item()),
        "route_attention_simplex_max_error": route_simplex_error,
        "od_attention_simplex_max_error": od_simplex_error,
        "tm2_pred_gradient_l1": float(tm2_probe.grad.abs().sum().item()),
        "attention_gradient_l1": attention_gradient_l1,
        "sparse_cache_builds": candidate.sparse_cache_builds,
        "static_neighbor_shape": list(candidate._spc_neighbor_od.shape),
        "sparse_audit": candidate.last_spc_audit,
        "native_parameters": sum(parameter.numel() for parameter in native.parameters()),
        "candidate_parameters": sum(parameter.numel() for parameter in candidate.parameters()),
        "added_parameters": sum(
            parameter.numel()
            for name, parameter in candidate_parameters.items()
            if name not in native_parameters
        ),
        "missing_state_keys_after_native_copy": list(loaded.missing_keys),
        "unexpected_state_keys_after_native_copy": list(loaded.unexpected_keys),
        "zero_output_native_policy": zero_equivalence,
        "manual_nonzero_output_policy_delta": active_delta,
        "actual_tm_replacement_policy": actual_independence,
        "ordered_projection": {
            "objectives": list(experiment.full.OBJECTIVE_NAMES),
            "parameter_shapes_compatible": shape_compatible,
            "final_gradient_numel": int(projection.final_gradient.numel()),
            "final_gradient_finite": bool(torch.isfinite(projection.final_gradient).all().item()),
            "init_output_weight_raw_gradient_l1_by_objective": init_output_gradients,
            "rau_output_weight_raw_gradient_l1_by_objective": rau_output_gradients,
        },
        "latency": {
            "timing_runs": args.timing_runs,
            "native_seconds": native_seconds,
            "candidate_seconds": candidate_seconds,
            "native_median_seconds": native_median,
            "candidate_median_seconds": candidate_median,
            "candidate_over_native": latency_ratio,
            "passes_2x": latency_ratio <= 2.0,
        },
        "activation_complexity_batch1_fp32": {
            "r8_full_gather_floats": r8_full_floats,
            "r8_full_gather_mb": r8_full_floats * bytes_per_float / 1_000_000,
            "r4d8_hypothetical_full_gather_floats": fast_full_floats,
            "r4d8_hypothetical_full_gather_mb": fast_full_floats * bytes_per_float / 1_000_000,
            "r4d8_actual_largest_streamed_gather_floats": fast_stream_floats,
            "r4d8_actual_largest_streamed_gather_mb": fast_stream_floats * bytes_per_float / 1_000_000,
            "r8_to_fast_stream_gather_reduction": r8_full_floats / fast_stream_floats,
            "attention_multiply_reduction_r8_to_fast": 4.0,
        },
        "r8_warmstart": warmstart,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
