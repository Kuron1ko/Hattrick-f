from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.args_parser import parse_args
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster, custom_collate
from utils.training_utils import loss_mlu, move_to_device


TOPOLOGY = "geant_priomask500_shared_load2x_train"
MODEL = ROOT / f"hattrick_{TOPOLOGY}_8sp.pkl"
OUTPUT = Path(__file__).resolve().parent / "artifacts" / "weighted_mlu_identity.json"


def build_props(device: torch.device):
    props = parse_args(
        [
            "--mode", "train",
            "--topo", TOPOLOGY,
            "--num_paths_per_pair", "8",
            "--num_transformer_layers", "3",
            "--num_gnn_layers", "3",
            "--num_mlp1_hidden_layers", "2",
            "--num_mlp2_hidden_layers", "2",
            "--rau1", "3",
            "--rau2", "3",
            "--rau3", "3",
            "--pred", "1",
            "--dynamic", "0",
            "--pred_type", "esm",
            "--violation", "1",
            "--path_mask", "0",
        ]
    )
    props.device = device
    props.dtype = torch.float32
    props.pg_mlu_lambda = 0.0
    props.edge_cost_objective = False
    props.edge_cost_mode = "weighted_mlu_actual"
    props.future_lookahead = False
    props.future_reservation_strength = 0.0
    props.future_low_reservation_strength = 0.0
    props.research_return_policy = False
    props.research_return_admitted = False
    return props


def flatten_gradients(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
    )


def main() -> None:
    torch.manual_seed(490)
    torch.cuda.manual_seed_all(490)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = build_props(device)
    dataset = DM_Dataset_within_Cluster(props, 0, 0, 1)
    dataset.pte = dataset.pte.to(device=device, dtype=props.dtype).coalesce()
    dataset.padded_edge_ids_per_path = dataset.padded_edge_ids_per_path.to(device)
    move_to_device(dataset.edge_ids_dict_tensor, device)
    move_to_device(dataset.original_pos_edge_ids_dict_tensor, device)
    path_masks = dataset.path_masks.to(device) if dataset.path_masks is not None else None
    inputs = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)))
    values = [
        value.to(device=device, dtype=props.dtype) if torch.is_tensor(value) else value
        for value in inputs
    ]
    node_features = values[0][:1]
    capacities = values[1][:1]

    model = torch.load(MODEL, map_location=device, weights_only=False)
    model.device = device
    model = model.to(device=device, dtype=props.dtype)
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def forward():
        return model(
            props,
            node_features,
            dataset.edge_index,
            capacities,
            dataset.padded_edge_ids_per_path,
            values[2], values[3], values[4], values[5], values[6], values[7],
            dataset.pte,
            dataset.edge_ids_dict_tensor,
            dataset.original_pos_edge_ids_dict_tensor,
            path_masks,
        )

    props.edge_cost_objective = False
    native = forward()
    native_losses = [
        loss_mlu(native[0], values[8])[0],
        loss_mlu(native[1], values[9])[0],
        loss_mlu(native[2], values[10])[0],
    ]
    native_gradient = flatten_gradients(sum(native_losses), parameters)
    native_values = [float(loss.detach().item()) for loss in native_losses]
    native_outputs = [value.detach().clone() for value in native]
    del native, native_losses

    edge_count = int(capacities.shape[1])
    ones = torch.ones(3, edge_count, device=device, dtype=props.dtype)
    model.register_buffer("research_edge_costs", ones, persistent=False)
    props.edge_cost_objective = True
    weighted = forward()
    weighted_losses = [
        loss_mlu(weighted[0], values[8])[0],
        loss_mlu(weighted[1], values[9])[0],
        loss_mlu(weighted[2], values[10])[0],
    ]
    weighted_gradient = flatten_gradients(sum(weighted_losses), parameters)
    weighted_values = [float(loss.detach().item()) for loss in weighted_losses]

    output_differences = [
        float((before - after.detach()).abs().max().item())
        for before, after in zip(native_outputs, weighted)
    ]
    loss_differences = [
        abs(before - after)
        for before, after in zip(native_values, weighted_values)
    ]
    gradient_difference = float((native_gradient - weighted_gradient).abs().max().item())
    result = {
        "model": str(MODEL),
        "snapshot": 0,
        "device": str(device),
        "weights": "all ones",
        "max_abs_output_differences": output_differences,
        "max_abs_loss_difference": max(loss_differences),
        "max_abs_gradient_difference": gradient_difference,
        "outputs_exact": all(value == 0.0 for value in output_differences),
        "losses_exact": all(value == 0.0 for value in loss_differences),
        "gradients_exact": gradient_difference == 0.0,
    }
    result["identity_pass"] = (
        result["outputs_exact"] and result["losses_exact"] and result["gradients_exact"]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["identity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
