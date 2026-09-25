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
    "shared2x_semantic_masm_smoke_runtime", THIS_DIR / "run_experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load semantic MASM runner")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def compare_policies(expected, actual, label: str, require_equal: bool = True) -> dict:
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
        raise RuntimeError("Actual traffic permutation changed nothing")
    return tuple(changed), total_l1


def synthetic_semantic_tests() -> dict:
    # Candidate-order equivariance: no candidate index is a model feature.
    torch.manual_seed(9)
    incidence = (torch.rand(2, 8, 7) > 0.62).float()
    incidence[..., 0] = 1.0
    capacity = torch.tensor([2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0])
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, False, True],
            [True, True, True, False, True, True, True, True],
        ]
    )
    overlap, semantics, diversity = (
        experiment.SemanticMASMHattrick.build_semantic_anchors(
            incidence, capacity, mask
        )
    )
    permutation = torch.tensor([5, 2, 7, 0, 3, 6, 1, 4])
    overlap_p, semantics_p, diversity_p = (
        experiment.SemanticMASMHattrick.build_semantic_anchors(
            incidence[:, permutation], capacity, mask[:, permutation]
        )
    )
    expected_overlap = overlap[:, permutation][:, :, permutation]
    overlap_error = float((overlap_p - expected_overlap).abs().max().item())
    semantic_error = float((semantics_p - semantics[:, permutation]).abs().max().item())
    diversity_error = float((diversity_p - diversity[:, permutation]).abs().max().item())
    if max(overlap_error, semantic_error, diversity_error) > 1e-6:
        raise RuntimeError("Fixed semantic construction is not candidate-equivariant")

    # Seven identical shared-edge alternatives do not create seven independent
    # escapes.  Making the eighth route feasible and edge-disjoint must lower
    # the fixed no-clean-alternative conflict seen by that High edge.
    shared = torch.zeros(1, 8, 3)
    shared[:, :7, 0] = 1.0
    shared[:, 7, 1] = 1.0
    base_mask = torch.tensor([[True] * 7 + [False]])
    added_mask = torch.ones(1, 8, dtype=torch.bool)
    _, base_semantics, base_diversity = (
        experiment.SemanticMASMHattrick.build_semantic_anchors(
            shared, torch.ones(3), base_mask
        )
    )
    _, added_semantics, added_diversity = (
        experiment.SemanticMASMHattrick.build_semantic_anchors(
            shared, torch.ones(3), added_mask
        )
    )
    base_conflict = float(
        (
            base_semantics[0, :, 4]
            * base_diversity[0]
            * shared[0, :, 0]
        ).sum().item()
    )
    added_conflict = float(
        (
            added_semantics[0, :, 4]
            * added_diversity[0]
            * shared[0, :, 0]
        ).sum().item()
    )
    if not added_conflict < base_conflict:
        raise RuntimeError("Adding an edge-disjoint alternative did not lower conflict")

    return {
        "candidate_permutation": permutation.tolist(),
        "overlap_max_error": overlap_error,
        "semantic_max_error": semantic_error,
        "diversity_max_error": diversity_error,
        "shared_edge_conflict_before": base_conflict,
        "shared_edge_conflict_after_disjoint_alternative": added_conflict,
        "disjoint_alternative_lowers_conflict": added_conflict < base_conflict,
    }


def timed_policy(model, props, dataset, values, path_masks) -> float:
    started = time.perf_counter()
    with torch.no_grad():
        experiment.full.shared.model_forward(model, props, dataset, values, path_masks)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU smoke for strict semantic MASM")
    parser.add_argument("--snapshot", type=int, default=350)
    parser.add_argument("--timing-runs", type=int, default=5)
    args = parser.parse_args()

    torch.set_num_threads(1)
    if torch.cuda.is_available() and str(torch.tensor(0).device) != "cpu":
        raise RuntimeError("Smoke test must not allocate CUDA tensors")
    synthetic = synthetic_semantic_tests()

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
    total_paths, num_edges = int(dataset.pte.shape[0]), int(dataset.pte.shape[1])
    paths_per_od = int(props.num_paths_per_pair)
    num_ods = total_paths // paths_per_od

    torch.manual_seed(20260824)
    native = experiment.NativeHattrick(props).to(device=device, dtype=props.dtype)
    torch.manual_seed(17)
    candidate = experiment.SemanticMASMHattrick(props).to(
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
    if any("medium_scout" in name for name in candidate_parameters):
        raise RuntimeError("Unanchored learned provisional-scout parameters remain")

    adapters = (
        candidate.medium_pressure_init_adapter,
        candidate.medium_pressure_rau_adapter,
    )
    for adapter in adapters:
        if int(torch.count_nonzero(adapter.body[2].weight).item()) != 0:
            raise RuntimeError("MASM adapter output is not exactly zero")
        if int(torch.count_nonzero(adapter.body[2].bias).item()) != 0:
            raise RuntimeError("MASM adapter output bias is not exactly zero")

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
    native_equivalence = compare_policies(
        native_policy, zero_policy, "epoch-0 native equivalence"
    )
    if candidate.masm_cache_builds != 1:
        raise RuntimeError(f"Expected one MASM cache build, got {candidate.masm_cache_builds}")

    features = candidate._last_masm_features
    if features is None or tuple(features.shape) != (
        1,
        total_paths,
        experiment.TOTAL_FEATURES,
    ):
        raise RuntimeError("Missing or malformed MASM feature tensor")
    message = features[..., experiment.PRESSURE_FEATURES :].reshape(
        1, num_ods, paths_per_od, experiment.MESSAGE_FEATURES
    )
    high_mask = candidate._class_mask(path_masks, 0, total_paths, device).reshape(
        1, num_ods, paths_per_od, 1
    )
    center_error = float((message * high_mask).sum(dim=2).abs().max().item())
    if center_error > 2e-5:
        raise RuntimeError(f"MASM High-OD center error {center_error}")

    encoded = candidate._last_masm_encoded_routes
    expected_fixed = candidate._masm_semantics.unsqueeze(0)
    init_semantic_error = float((encoded - expected_fixed).abs().max().item())
    if init_semantic_error != 0.0:
        raise RuntimeError("Zero residual set encoder is not exactly fixed-semantic")

    # Re-run while all buffers are warm; topology/coverage construction must not
    # occur in each forward.
    cache_pointers_before = {
        "incidence": int(candidate._masm_grouped_incidence.data_ptr()),
        "overlap": int(candidate._masm_overlap.data_ptr()),
        "semantics": int(candidate._masm_semantics.data_ptr()),
        "diversity": int(candidate._masm_diversity.data_ptr()),
    }
    with torch.no_grad():
        experiment.full.shared.model_forward(candidate, props, dataset, values, path_masks)
    cache_pointers_after = {
        "incidence": int(candidate._masm_grouped_incidence.data_ptr()),
        "overlap": int(candidate._masm_overlap.data_ptr()),
        "semantics": int(candidate._masm_semantics.data_ptr()),
        "diversity": int(candidate._masm_diversity.data_ptr()),
    }
    if candidate.masm_cache_builds != 1 or cache_pointers_before != cache_pointers_after:
        raise RuntimeError("MASM static cache was rebuilt on a repeated forward")

    # Open only High adapters.  Then alter every actual TM while predictions
    # remain byte-identical; a strict-ESM policy must remain byte-identical too.
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
        zero_policy, active_policy, "active MASM delta", require_equal=False
    )
    if max(row["max_abs_delta"] for row in active_delta.values()) == 0.0:
        raise RuntimeError("Opening MASM adapters did not affect policy")
    strict_actual = compare_policies(
        active_policy, changed_policy, "strict actual-TM permutation"
    )

    # Complete six-objective ordered projection.  The flat gradient ordering
    # must include every new set-encoder and adapter tensor.
    candidate.train()
    props.mode = "train"
    props.research_return_policy = False
    props.research_return_admitted = True
    candidate.zero_grad(set_to_none=True)
    losses, reported = experiment.full.build_objectives(
        candidate, props, dataset, values, path_masks
    )
    if len(losses) != 6 or any(not torch.isfinite(loss).item() for loss in losses):
        raise RuntimeError("Complete six-objective forward is invalid")
    projection = experiment.full.ordered_project_gradients(candidate, losses)
    parameter_numel = sum(parameter.numel() for parameter in candidate.parameters())
    if int(projection.final_gradient.numel()) != parameter_numel:
        raise RuntimeError("Ordered projection omitted parameters")
    if len(projection.parameter_shapes) != len(list(candidate.parameters())):
        raise RuntimeError("Projection parameter-shape list is incomplete")
    if not torch.isfinite(projection.final_gradient).all().item():
        raise RuntimeError("Projection returned a non-finite gradient")

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
    if sum(end - start for start, end in offsets.values()) != sum(added_parameters.values()):
        raise RuntimeError("An added parameter is absent from projection ordering")
    added_gradient_l1 = sum(
        max(float(raw[start:end].abs().sum().item()) for raw in projection.raw_gradients)
        for start, end in offsets.values()
    )
    if added_gradient_l1 == 0.0:
        raise RuntimeError("All added MASM parameters have zero raw gradient")

    # Coarse cached, single-thread CPU gate.  CUDA Events should be used for a
    # formal latency table; this smoke only rejects an obviously >2x design.
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
        candidate_seconds.append(timed_policy(candidate, props, dataset, values, path_masks))
    native_median = statistics.median(native_seconds)
    candidate_median = statistics.median(candidate_seconds)
    latency_ratio = candidate_median / max(native_median, 1e-12)
    if latency_ratio >= 2.0:
        raise RuntimeError(f"CPU inference latency ratio {latency_ratio:.3f} >= 2")

    result = {
        "status": "ok",
        "device": "cpu",
        "snapshot": args.snapshot,
        "synthetic_semantics": synthetic,
        "strict_input": "ESM tm2_pred; actual traffic excluded from MASM policy",
        "actual_permutation_l1": actual_permutation_l1,
        "formal_cascade_inherited": (
            experiment.SemanticMASMHattrick.forward is experiment.NativeHattrick.forward
        ),
        "native_parameters": sum(p.numel() for p in native.parameters()),
        "candidate_parameters": parameter_numel,
        "added_parameters": sum(added_parameters.values()),
        "added_parameter_tensors": added_parameters,
        "missing_state_keys": list(loaded.missing_keys),
        "unexpected_state_keys": list(loaded.unexpected_keys),
        "feature_shape": list(features.shape),
        "feature_min": float(features.min().item()),
        "feature_max": float(features.max().item()),
        "message_center_max_error": center_error,
        "initial_fixed_semantic_max_error": init_semantic_error,
        "cache_builds": candidate.masm_cache_builds,
        "cache_pointers_stable": cache_pointers_before == cache_pointers_after,
        "masm_audit": candidate.last_masm_audit,
        "zero_output_native_policy": native_equivalence,
        "active_masm_policy_delta": active_delta,
        "actual_tm_permutation_policy": strict_actual,
        "projection": {
            "objective_names": list(experiment.full.OBJECTIVE_NAMES),
            "reported_values": [float(value) for value in reported],
            "flat_parameter_count": int(projection.final_gradient.numel()),
            "parameter_tensor_count": len(projection.parameter_shapes),
            "added_parameter_offsets": offsets,
            "added_parameter_raw_gradient_l1": added_gradient_l1,
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
