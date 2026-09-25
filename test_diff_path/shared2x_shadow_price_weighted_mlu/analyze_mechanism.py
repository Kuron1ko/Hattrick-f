from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EDGE_RUNNER_DIR = HERE.parent / "shared2x_learned_edge_cost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EDGE_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_RUNNER_DIR))

import run_experiment as edge_runner  # noqa: E402
from utils.build_dataset_within_cluster import (  # noqa: E402
    DM_Dataset_within_Cluster,
    custom_collate,
)
from utils.training_utils import move_to_device  # noqa: E402


DEFAULT_ARTIFACT = HERE / "artifacts" / "small_q0p75_delta0p003"


def prepare_dataset(start: int, end: int, device: torch.device):
    props = edge_runner.make_policy_props(start, end, device)
    dataset = DM_Dataset_within_Cluster(props, 0, start, end)
    dataset.pte = dataset.pte.to(device=device, dtype=props.dtype).coalesce()
    dataset.padded_edge_ids_per_path = dataset.padded_edge_ids_per_path.to(device)
    move_to_device(dataset.edge_ids_dict_tensor, device)
    move_to_device(dataset.original_pos_edge_ids_dict_tensor, device)
    path_masks = dataset.path_masks.to(device) if dataset.path_masks is not None else None
    return props, dataset, path_masks


def collect(
    model_path: Path,
    props,
    dataset,
    path_masks,
    score: torch.Tensor,
    strength: float,
) -> dict:
    model = torch.load(model_path, map_location=props.device, weights_only=False)
    model.device = props.device
    model = model.to(device=props.device, dtype=props.dtype)
    model.eval()
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=custom_collate
    )
    rows = {
        "high_price_mass_true": [],
        "high_price_mass_esm": [],
        "high_unweighted_mlu_true": [],
        "high_weighted_mlu_true": [],
        "high_unweighted_mlu_esm": [],
        "high_weighted_mlu_esm": [],
        "high_edge_util_true": [],
        "medium_edge_util_true": [],
    }
    with torch.no_grad():
        for inputs in loader:
            values = tuple(
                value.to(device=props.device, dtype=props.dtype)
                if torch.is_tensor(value)
                else value
                for value in inputs
            )
            policies = edge_runner.cached_policy_forward(
                model, props, dataset, values, path_masks
            )
            capacities = values[1]
            batch = int(values[2].shape[0])
            if capacities.shape[0] == 1 and batch > 1:
                capacities = capacities.expand(batch, -1)
            elif capacities.shape[0] != batch:
                capacities = capacities[:1].expand(batch, -1)

            class_utils_true = []
            class_utils_esm = []
            for class_index, policy in enumerate(policies):
                policy = policy.squeeze(-1)
                true_tm = values[2 + 2 * class_index].squeeze(-1)
                esm_tm = values[3 + 2 * class_index].squeeze(-1)
                true_load = torch.sparse.mm(
                    dataset.pte.t(), (policy * true_tm).t()
                ).t()
                esm_load = torch.sparse.mm(
                    dataset.pte.t(), (policy * esm_tm).t()
                ).t()
                class_utils_true.append(true_load / capacities.clamp_min(1e-9))
                class_utils_esm.append(esm_load / capacities.clamp_min(1e-9))

            high_true = class_utils_true[0]
            high_esm = class_utils_esm[0]
            medium_true = class_utils_true[1]
            rows["high_price_mass_true"].append((high_true * score).sum(dim=1).cpu())
            rows["high_price_mass_esm"].append((high_esm * score).sum(dim=1).cpu())
            rows["high_unweighted_mlu_true"].append(high_true.max(dim=1).values.cpu())
            rows["high_weighted_mlu_true"].append(
                (high_true * (1.0 + strength * score)).max(dim=1).values.cpu()
            )
            rows["high_unweighted_mlu_esm"].append(high_esm.max(dim=1).values.cpu())
            rows["high_weighted_mlu_esm"].append(
                (high_esm * (1.0 + strength * score)).max(dim=1).values.cpu()
            )
            rows["high_edge_util_true"].append(high_true.cpu())
            rows["medium_edge_util_true"].append(medium_true.cpu())
    return {
        key: torch.cat(parts).numpy()
        for key, parts in rows.items()
    }


def scalar_delta(control: np.ndarray, candidate: np.ndarray) -> dict:
    difference = candidate - control
    return {
        "control_mean": float(control.mean()),
        "candidate_mean": float(candidate.mean()),
        "mean_delta": float(difference.mean()),
        "median_delta": float(np.median(difference)),
        "candidate_lower_fraction": float(np.mean(difference < 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--candidate-model", type=Path)
    parser.add_argument("--control-model", type=Path)
    parser.add_argument("--score-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--strength", type=float, default=0.003)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    if args.control_model is not None and args.candidate_model is not None:
        control_path = args.control_model.resolve()
        candidate_path = args.candidate_model.resolve()
    elif "models" in report:
        model_paths = list(report["models"].values())
        control_path = Path(model_paths[0])
        candidate_path = Path(model_paths[1])
    else:
        control_path = Path(report["control_reused"]["model"])
        candidate_path = Path(report["candidate_model"])
    score_artifact = args.score_artifact.resolve()
    payload = torch.load(
        score_artifact / "shadow_price_weights.pt",
        map_location="cpu",
        weights_only=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score = payload["high_shadow_score"].to(device=device, dtype=torch.float32)
    props, dataset, path_masks = prepare_dataset(160, 250, device)
    control = collect(
        control_path, props, dataset, path_masks, score, args.strength
    )
    # Drop the topology cache so the second model creates its own cache.
    candidate = collect(
        candidate_path, props, dataset, path_masks, score, args.strength
    )

    scalar_keys = [key for key, value in control.items() if value.ndim == 1]
    high_edge_delta = (
        candidate["high_edge_util_true"] - control["high_edge_util_true"]
    ).mean(axis=0)
    medium_edge_delta = (
        candidate["medium_edge_util_true"] - control["medium_edge_util_true"]
    ).mean(axis=0)
    score_np = score.cpu().numpy()
    positive = score_np > 0.0
    result = {
        "evaluation_range": [160, 250],
        "scalar_metrics": {
            key: scalar_delta(control[key], candidate[key]) for key in scalar_keys
        },
        "high_util_delta_on_positive_price_edges": float(high_edge_delta[positive].mean()),
        "high_util_delta_on_zero_price_edges": float(high_edge_delta[~positive].mean()),
        "medium_util_delta_on_positive_price_edges": float(
            medium_edge_delta[positive].mean()
        ),
        "medium_util_delta_on_zero_price_edges": float(
            medium_edge_delta[~positive].mean()
        ),
        "score_high_util_delta_correlation": float(
            np.corrcoef(score_np, high_edge_delta)[0, 1]
        ),
        "score_medium_util_delta_correlation": float(
            np.corrcoef(score_np, medium_edge_delta)[0, 1]
        ),
        "edge_rows": [
            {
                "edge_id": int(edge_id),
                "edge": payload["edge_names"][edge_id],
                "score": float(score_np[edge_id]),
                "high_util_delta": float(high_edge_delta[edge_id]),
                "medium_util_delta": float(medium_edge_delta[edge_id]),
            }
            for edge_id in np.argsort(-score_np)
        ],
    }
    output = artifact / "mechanism_analysis.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "edge_rows"}, indent=2))


if __name__ == "__main__":
    main()
