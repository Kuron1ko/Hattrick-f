from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "shared2x_medium_irreplaceability_smoke_runtime",
    THIS_DIR / "run_experiment.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load Medium-irreplaceability experiment runner")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-snapshot smoke test for the analytic irreplaceability scout"
    )
    parser.add_argument("--snapshot", type=int, default=400)
    parser.add_argument("--gamma", type=float, default=2.0)
    args = parser.parse_args()

    gamma = experiment.configure_gamma(args.gamma)
    device = torch.device("cpu")
    props = experiment.full.shared.build_props(1, device)
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

    model = experiment.MediumIrreplaceabilityHattrick(props).to(
        device=device, dtype=props.dtype
    )
    pte = dataset.pte.coalesce()
    total_paths = int(pte.shape[0])
    paths_per_od = int(props.num_paths_per_pair)
    num_ods = total_paths // paths_per_od

    # Exercise mask handling even though the shared-8sp production dataset is
    # normally unmasked: retain seven feasible paths per OD for High and Medium.
    path_masks = torch.ones((3, total_paths), dtype=torch.bool, device=device)
    path_masks[0].reshape(num_ods, paths_per_od)[:, -1] = False
    path_masks[1].reshape(num_ods, paths_per_od)[:, -1] = False
    indices = pte.indices()
    pte_info = [pte, indices[0], indices[1], pte.values()]

    def compute_features():
        return model.compute_medium_pressure_features(
            tm2_pred,
            capacities,
            pte,
            pte_info,
            batch_size=1,
            num_paths_per_pair=paths_per_od,
            props=props,
            path_masks=path_masks,
        )

    features = compute_features()
    cache_pointer = int(model._medium_edge_coverage_cache.data_ptr())
    repeated = compute_features()
    if model.medium_coverage_cache_builds != 1:
        raise RuntimeError("Static Medium coverage cache was unexpectedly rebuilt")
    if int(model._medium_edge_coverage_cache.data_ptr()) != cache_pointer:
        raise RuntimeError("Static Medium coverage cache storage changed")
    if not torch.equal(features, repeated):
        raise RuntimeError("Cached feature replay is not exact")

    probe_loss = features.square().mean()
    probe_loss.backward()
    gradient = tm2_pred.grad
    expected_shape = (1, total_paths, 2)
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            f"Feature shape mismatch: {tuple(features.shape)} != {expected_shape}"
        )
    if not torch.isfinite(features).all().item():
        raise RuntimeError("Non-finite irreplaceability feature")
    if gradient is None or not torch.isfinite(gradient).all().item():
        raise RuntimeError("Missing or non-finite ESM-demand gradient")
    if float(gradient.abs().sum().item()) == 0.0:
        raise RuntimeError("ESM-demand gradient is identically zero")

    valid_high = path_masks[0].reshape(num_ods, paths_per_od)
    relative = features[0, :, 1].reshape(num_ods, paths_per_od)
    centered_sums = (relative * valid_high).sum(dim=-1)
    max_center_error = float(centered_sums.abs().max().item())
    if max_center_error > 1e-4:
        raise RuntimeError(f"OD-relative feature is not centered: {max_center_error}")

    coverage = model._medium_edge_coverage_cache
    result = {
        "status": "ok",
        "snapshot": args.snapshot,
        "strict_input": "tm2_pred (ESM) only",
        "gamma": gamma,
        "feature_shape": list(features.shape),
        "feature_min": float(features.detach().min().item()),
        "feature_max": float(features.detach().max().item()),
        "feature_mean": float(features.detach().mean().item()),
        "coverage_shape": list(coverage.shape),
        "coverage_min": float(coverage.min().item()),
        "coverage_max": float(coverage.max().item()),
        "coverage_cache_builds": model.medium_coverage_cache_builds,
        "relative_center_max_sum_error": max_center_error,
        "tm2_pred_gradient_l1": float(gradient.detach().abs().sum().item()),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
