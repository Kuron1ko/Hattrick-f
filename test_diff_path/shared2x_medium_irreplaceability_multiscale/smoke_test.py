from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "shared2x_medium_irreplaceability_multiscale_smoke_runtime",
    THIS_DIR / "run_experiment.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load multiscale Medium experiment runner")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU smoke and zero-init policy audit for multiscale scout"
    )
    parser.add_argument("--snapshot", type=int, default=350)
    args = parser.parse_args()

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

    feature_model = experiment.MultiscaleMediumIrreplaceabilityHattrick(props).to(
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
    cache_pointer = int(feature_model._medium_multiscale_power_cache.data_ptr())
    repeated = feature_forward()
    if feature_model.medium_coverage_cache_builds != 1:
        raise RuntimeError("Base coverage cache was unexpectedly rebuilt")
    if feature_model.medium_multiscale_cache_builds != 1:
        raise RuntimeError("Multiscale power cache was unexpectedly rebuilt")
    if int(feature_model._medium_multiscale_power_cache.data_ptr()) != cache_pointer:
        raise RuntimeError("Multiscale cache storage changed")
    if not torch.equal(features, repeated):
        raise RuntimeError("Cached multiscale feature replay is not exact")

    expected_shape = (1, total_paths, 6)
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            f"Feature shape mismatch: {tuple(features.shape)} != {expected_shape}"
        )
    if not torch.isfinite(features).all().item():
        raise RuntimeError("Non-finite multiscale feature")
    valid_high = masks[0].reshape(num_ods, paths_per_od)
    center_errors = []
    for column in (1, 3, 5):
        relative = features[0, :, column].reshape(num_ods, paths_per_od)
        center_errors.append(
            float(((relative * valid_high).sum(dim=-1)).abs().max().item())
        )
    if max(center_errors) > 1e-4:
        raise RuntimeError(f"Centered feature error: {center_errors}")

    powers = feature_model._medium_multiscale_power_cache
    if not torch.equal(powers[0], feature_model._medium_edge_coverage_cache):
        raise RuntimeError("r^1 cache does not equal raw coverage")
    if not torch.allclose(powers[1], powers[0].pow(2), atol=0.0, rtol=0.0):
        raise RuntimeError("r^2 cache mismatch")
    if not torch.allclose(powers[2], powers[0].pow(3), atol=0.0, rtol=0.0):
        raise RuntimeError("r^3 cache mismatch")

    features.square().mean().backward()
    gradient = tm2_pred.grad
    if gradient is None or not torch.isfinite(gradient).all().item():
        raise RuntimeError("Missing or non-finite ESM-demand gradient")
    if float(gradient.abs().sum().item()) == 0.0:
        raise RuntimeError("ESM-demand gradient is identically zero")

    # Independent model pair for the complete zero-output equivalence audit.
    torch.manual_seed(20260824)
    native = experiment.NativeHattrick(props).to(device=device, dtype=props.dtype)
    torch.manual_seed(17)
    candidate = experiment.MultiscaleMediumIrreplaceabilityHattrick(props).to(
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
    policy_comparison = {}
    for name, expected, actual in zip(
        ("High", "Medium", "Low"), native_policy, candidate_policy
    ):
        difference = (expected - actual).abs()
        policy_comparison[name] = {
            "shape": list(expected.shape),
            "torch_equal": bool(torch.equal(expected, actual)),
            "different_elements": int(
                torch.count_nonzero(expected != actual).item()
            ),
            "max_abs_delta": float(difference.max().item()),
        }
        if not torch.equal(expected, actual):
            raise RuntimeError(f"Zero-init {name} policy is not exactly native")

    added_parameters = {
        name: parameter.numel()
        for name, parameter in candidate_params.items()
        if name not in native_params
    }
    cache_bytes = (
        feature_model._medium_edge_coverage_cache.numel()
        * feature_model._medium_edge_coverage_cache.element_size()
        + powers.numel() * powers.element_size()
        + feature_model._medium_feasible_mask_cache.numel()
        * feature_model._medium_feasible_mask_cache.element_size()
    )
    result = {
        "status": "ok",
        "device": "cpu",
        "snapshot": args.snapshot,
        "strict_input": "tm2_pred (ESM) only",
        "feature_shape": list(features.shape),
        "feature_min": float(features.detach().min().item()),
        "feature_max": float(features.detach().max().item()),
        "center_max_sum_errors": center_errors,
        "tm2_pred_gradient_l1": float(gradient.detach().abs().sum().item()),
        "coverage_cache_shape": list(
            feature_model._medium_edge_coverage_cache.shape
        ),
        "power_cache_shape": list(powers.shape),
        "coverage_cache_builds": feature_model.medium_coverage_cache_builds,
        "multiscale_cache_builds": feature_model.medium_multiscale_cache_builds,
        "static_cache_bytes": int(cache_bytes),
        "feature_operator_audit": feature_model.last_multiscale_operator_audit,
        "native_parameters": sum(p.numel() for p in native.parameters()),
        "candidate_parameters": sum(p.numel() for p in candidate.parameters()),
        "added_parameters": sum(added_parameters.values()),
        "added_parameter_tensors": added_parameters,
        "missing_state_keys": list(loaded.missing_keys),
        "unexpected_state_keys": list(loaded.unexpected_keys),
        "policy_comparison": policy_comparison,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
