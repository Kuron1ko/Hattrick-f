from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "shared2x_virtual_medium_dual_smoke_runtime", THIS_DIR / "run_experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load virtual Medium experiment runner")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-snapshot differentiability smoke test for the virtual Medium scout"
    )
    parser.add_argument("--snapshot", type=int, default=400)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--price-power", type=float, default=2.0)
    args = parser.parse_args()

    settings = experiment.configure_scout(
        args.steps, args.temperature, args.price_power
    )
    device = torch.device("cpu")
    props = experiment.full.shared.build_props(1, device)
    dataset = experiment.full.DM_Dataset_within_Cluster(
        props, 0, args.snapshot, args.snapshot + 1
    )
    path_masks = experiment.full.shared.base.move_dataset_static(dataset, device)
    loader = experiment.full.shared.data_loader(
        dataset, batch_size=1, shuffle=False, seed=490
    )
    values = experiment.full.shared.unpack_to_device(next(iter(loader)), props)
    capacities = values[1]
    tm2_pred = values[5].detach().clone().requires_grad_(True)

    model = experiment.VirtualMediumDualHattrick(props).to(
        device=device, dtype=props.dtype
    )
    pte = dataset.pte.coalesce()
    indices = pte.indices()
    pte_info = [pte, indices[0], indices[1], pte.values()]
    features = model.compute_medium_pressure_features(
        tm2_pred,
        capacities,
        pte,
        pte_info,
        batch_size=1,
        num_paths_per_pair=props.num_paths_per_pair,
        props=props,
        path_masks=path_masks,
    )
    probe_loss = features.square().mean()
    probe_loss.backward()
    gradient = tm2_pred.grad

    expected_shape = (1, int(pte.shape[0]), 2)
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            f"Feature shape mismatch: {tuple(features.shape)} != {expected_shape}"
        )
    if not torch.isfinite(features).all().item():
        raise RuntimeError("Non-finite scout feature")
    if gradient is None or not torch.isfinite(gradient).all().item():
        raise RuntimeError("Missing or non-finite scout gradient")
    if float(gradient.abs().sum().item()) == 0.0:
        raise RuntimeError("Scout gradient is identically zero")

    result = {
        "status": "ok",
        "snapshot": args.snapshot,
        "strict_input": "tm2_pred (ESM) only",
        "settings": {
            "steps": settings.steps,
            "temperature": settings.temperature,
            "price_power": settings.price_power,
        },
        "feature_shape": list(features.shape),
        "feature_min": float(features.detach().min().item()),
        "feature_max": float(features.detach().max().item()),
        "feature_mean": float(features.detach().mean().item()),
        "tm2_pred_gradient_l1": float(gradient.detach().abs().sum().item()),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
