from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "shared2x_sparse_od_conflict_smoke_runtime",
    THIS_DIR / "run_experiment.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load sparse OD-conflict experiment runner")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def compare_policies(expected, actual, label: str) -> dict:
    comparison = {}
    for class_name, before, after in zip(
        ("High", "Medium", "Low"), expected, actual
    ):
        difference = (before - after).abs()
        row = {
            "shape": list(before.shape),
            "torch_equal": bool(torch.equal(before, after)),
            "different_elements": int(torch.count_nonzero(before != after).item()),
            "max_abs_delta": float(difference.max().item()),
        }
        comparison[class_name] = row
        if not row["torch_equal"]:
            raise RuntimeError(f"{label}: {class_name} policy is not bitwise equal")
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
        raise RuntimeError("Actual-TM permutation did not change the smoke sample")
    return tuple(changed), total_l1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU cache, strict-ESM, and zero-init smoke for SOCM"
    )
    parser.add_argument("--snapshot", type=int, default=350)
    parser.add_argument(
        "--beta", type=float, choices=experiment.ALLOWED_BETAS, default=4.0
    )
    args = parser.parse_args()

    beta = experiment.configure_beta(args.beta)
    device = torch.device("cpu")
    props = experiment.full.shared.build_props(4, device)
    props.checkpoint = 0
    dataset = experiment.full.DM_Dataset_within_Cluster(
        props, 0, args.snapshot, args.snapshot + 1
    )
    experiment.full.shared.base.move_dataset_static(dataset, device)
    loader = experiment.full.shared.data_loader(
        dataset, batch_size=1, shuffle=False, seed=490
    )
    values = experiment.full.shared.unpack_to_device(next(iter(loader)), props)
    capacities = values[1]
    tm2_pred = values[5].detach().clone().requires_grad_(True)
    pte = dataset.pte.coalesce()
    indices = pte.indices()
    pte_info = [pte, indices[0], indices[1], pte.values()]
    total_paths, num_edges = int(pte.shape[0]), int(pte.shape[1])
    paths_per_od = int(props.num_paths_per_pair)
    num_ods = total_paths // paths_per_od

    # Feature/cache audit with masks, independent of the production-policy
    # models below.  Seven feasible paths per OD exercises both mask branches.
    feature_model = experiment.SparseODConflictHattrick(props).to(
        device=device, dtype=props.dtype
    )
    masks = torch.ones((3, total_paths), dtype=torch.bool, device=device)
    masks[0].reshape(num_ods, paths_per_od)[:, -1] = False
    masks[1].reshape(num_ods, paths_per_od)[:, -1] = False

    def feature_forward():
        return feature_model.compute_medium_pressure_features(
            tm2_pred,
            capacities,
            pte,
            pte_info,
            batch_size=1,
            num_paths_per_pair=paths_per_od,
            props=props,
            path_masks=masks,
        )

    features = feature_forward()
    conflict_pointer = int(feature_model._medium_od_conflict_cache.data_ptr())
    capacity_pointer = int(feature_model._conflict_capacity_cache.data_ptr())
    repeated = feature_forward()
    if feature_model.medium_od_conflict_cache_builds != 1:
        raise RuntimeError("Static OD-conflict cache was unexpectedly rebuilt")
    if int(feature_model._medium_od_conflict_cache.data_ptr()) != conflict_pointer:
        raise RuntimeError("OD-conflict cache storage changed")
    if int(feature_model._conflict_capacity_cache.data_ptr()) != capacity_pointer:
        raise RuntimeError("Capacity-signature cache storage changed")
    if not torch.equal(features, repeated):
        raise RuntimeError("Cached OD-conflict feature replay is not exact")

    expected_shape = (1, total_paths, 2)
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            f"Feature shape mismatch: {tuple(features.shape)} != {expected_shape}"
        )
    conflict = feature_model._medium_od_conflict_cache
    if tuple(conflict.shape) != (total_paths, num_ods):
        raise RuntimeError("OD-conflict cache has the wrong shape")
    if not torch.isfinite(conflict).all().item() or not torch.isfinite(features).all().item():
        raise RuntimeError("Non-finite OD-conflict value or feature")
    if float(conflict.max().item()) <= 0.0:
        raise RuntimeError("OD-conflict cache is identically zero")

    valid_high = masks[0].reshape(num_ods, paths_per_od)
    centered = features[0, :, 1].reshape(num_ods, paths_per_od)
    center_error = float(
        ((centered * valid_high).sum(dim=-1)).abs().max().item()
    )
    if center_error > 1e-4:
        raise RuntimeError(f"High-OD centered feature error: {center_error}")

    features.square().mean().backward()
    gradient = tm2_pred.grad
    if gradient is None or not torch.isfinite(gradient).all().item():
        raise RuntimeError("Missing or non-finite ESM-demand gradient")
    if float(gradient.abs().sum().item()) == 0.0:
        raise RuntimeError("ESM-demand gradient is identically zero")

    # Independent model pair for exact native policy equivalence.  Loading the
    # native state leaves only the zero-output residual adapters as new keys.
    torch.manual_seed(20260824)
    native = experiment.NativeHattrick(props).to(device=device, dtype=props.dtype)
    torch.manual_seed(17)
    candidate = experiment.SparseODConflictHattrick(props).to(
        device=device, dtype=props.dtype
    )
    loaded = candidate.load_state_dict(native.state_dict(), strict=False)
    native_params = dict(native.named_parameters())
    candidate_params = dict(candidate.named_parameters())
    if not all(
        torch.equal(parameter, candidate_params[name])
        for name, parameter in native_params.items()
    ):
        raise RuntimeError("Native parameter copy is not exact")

    props.mode = "test"
    props.research_return_policy = True
    props.research_return_admitted = False
    props.sim_mf_mlu = 0
    native.eval()
    candidate.eval()
    with torch.no_grad():
        native_policy, _ = experiment.full.shared.model_forward(
            native, props, dataset, values, None
        )
        candidate_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, values, None
        )
        candidate_cache_pointer = int(
            candidate._medium_od_conflict_cache.data_ptr()
        )
        permuted_values, actual_permutation_l1 = permute_actual_traffic(
            values, num_ods, paths_per_od
        )
        permuted_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, permuted_values, None
        )

    native_equivalence = compare_policies(
        native_policy, candidate_policy, "zero-init native equivalence"
    )
    strict_esm_equivalence = compare_policies(
        candidate_policy, permuted_policy, "actual-TM permutation"
    )
    if candidate.medium_od_conflict_cache_builds != 1:
        raise RuntimeError("Policy model rebuilt its static conflict cache")
    if int(candidate._medium_od_conflict_cache.data_ptr()) != candidate_cache_pointer:
        raise RuntimeError("Policy model conflict-cache storage changed")

    added_parameters = {
        name: parameter.numel()
        for name, parameter in candidate_params.items()
        if name not in native_params
    }
    cache_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (
            feature_model._medium_od_conflict_cache,
            feature_model._medium_feasible_mask_cache,
            feature_model._conflict_capacity_cache,
        )
    )
    result = {
        "status": "ok",
        "device": "cpu",
        "snapshot": args.snapshot,
        "beta": beta,
        "strict_input": "tm2_pred (ESM) only",
        "actual_permutation_l1": actual_permutation_l1,
        "feature_shape": list(features.shape),
        "feature_min": float(features.detach().min().item()),
        "feature_max": float(features.detach().max().item()),
        "high_od_center_max_sum_error": center_error,
        "tm2_pred_gradient_l1": float(gradient.detach().abs().sum().item()),
        "conflict_cache_shape": list(conflict.shape),
        "conflict_cache_min": float(conflict.min().item()),
        "conflict_cache_max": float(conflict.max().item()),
        "conflict_cache_mean": float(conflict.mean().item()),
        "conflict_cache_builds": feature_model.medium_od_conflict_cache_builds,
        "policy_conflict_cache_builds": candidate.medium_od_conflict_cache_builds,
        "static_cache_bytes": int(cache_bytes),
        "feature_operator_audit": feature_model.last_conflict_operator_audit,
        "native_parameters": sum(p.numel() for p in native.parameters()),
        "candidate_parameters": sum(p.numel() for p in candidate.parameters()),
        "added_parameters": sum(added_parameters.values()),
        "added_parameter_tensors": added_parameters,
        "missing_state_keys": list(loaded.missing_keys),
        "unexpected_state_keys": list(loaded.unexpected_keys),
        "zero_init_native_policy": native_equivalence,
        "actual_permutation_policy": strict_esm_equivalence,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
