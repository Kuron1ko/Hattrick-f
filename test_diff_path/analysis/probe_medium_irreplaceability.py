from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "test_diff_path"
OUTPUT = (
    ROOT
    / "output"
    / "analysis"
    / "medium_irreplaceability"
    / "probe_snapshots_350_357.json"
)
GAMMAS = (1.0, 2.0, 3.0, 4.0)
TARGET_EDGES = ((0, 4), (4, 0), (0, 19), (19, 0), (1, 13), (13, 1))


def distribution(value: torch.Tensor, absolute: bool = False) -> dict[str, float]:
    flat = value.detach().to(dtype=torch.float64).reshape(-1).cpu()
    if absolute:
        flat = flat.abs()
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "p01": float(torch.quantile(flat, 0.01).item()),
        "p10": float(torch.quantile(flat, 0.10).item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "max": float(flat.max().item()),
    }


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.detach().to(dtype=torch.float64).reshape(-1)
    y = right.detach().to(dtype=torch.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    return float((x * y).sum().div(denom.clamp_min(1e-15)).item())


def average_ranks(value: torch.Tensor) -> torch.Tensor:
    # The edge/path values are almost always unique, but average tied ranks make
    # the Spearman calculation well-defined for exact structural zeros.
    array = value.detach().to(dtype=torch.float64).reshape(-1).cpu().numpy()
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return torch.from_numpy(ranks)


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def main() -> None:
    for path in (ROOT, TEST_DIR, TEST_DIR / "shared2x_order_regularizer"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    shared = importlib.import_module("run_experiment")
    from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster
    from utils.cluster_utils import Cluster_Info

    props = shared.build_props(4, torch.device("cpu"))
    dataset = DM_Dataset_within_Cluster(props, 0, 350, 358)
    num_pairs = int(dataset.num_pairs)
    paths_per_od = int(props.num_paths_per_pair)
    pte = dataset.pte.coalesce().to(dtype=torch.float64)
    dense_pte = pte.to_dense()
    num_edges = int(dense_pte.shape[1])
    incidence = dense_pte.reshape(num_pairs, paths_per_od, num_edges)
    # r[i,e] is the fraction of OD i's candidate paths that contain edge e.
    replacement_fraction = incidence.mean(dim=1)

    demands = torch.stack(
        [
            torch.as_tensor(value)
            .reshape(num_pairs, paths_per_od, 1)[:, 0, 0]
            .to(dtype=torch.float64)
            for value in dataset.list_tms2_pred
        ]
    )
    capacities = torch.stack(
        [torch.as_tensor(value).reshape(-1).to(dtype=torch.float64) for value in dataset.list_capacities]
    )
    if not torch.allclose(capacities, capacities[:1].expand_as(capacities)):
        raise RuntimeError("The probe assumes capacities are stable across snapshots")

    # This is exactly the existing uniform split Medium edge utilization.
    uniform_edge_pressure = torch.einsum("bi,ie->be", demands, replacement_fraction) / capacities
    uniform_path_max = torch.where(
        incidence.unsqueeze(0).bool(),
        uniform_edge_pressure[:, None, None, :],
        torch.full((), -torch.inf, dtype=torch.float64),
    ).amax(dim=-1)
    uniform_path_mean = (
        torch.einsum("pe,be->bp", dense_pte, uniform_edge_pressure)
        / dense_pte.sum(dim=1).clamp_min(1.0).unsqueeze(0)
    ).reshape(len(dataset), num_pairs, paths_per_od)

    cluster_info = Cluster_Info(dataset.list_snapshots[0], props, 0)
    index_to_edge = {index: tuple(map(int, edge)) for edge, index in cluster_info.edges_map.items()}

    records = []
    edge_values_by_gamma: dict[float, torch.Tensor] = {}
    for gamma in GAMMAS:
        weighted_fraction = replacement_fraction.pow(gamma)
        edge_value = torch.einsum("bi,ie->be", demands, weighted_fraction) / capacities
        edge_values_by_gamma[gamma] = edge_value

        # Additive opportunity cost: every edge consumed by a High path has a
        # Medium value. log1p controls the heavy tail; relative is centered only
        # among the eight paths of the same OD.
        raw_path_value = torch.einsum("pe,be->bp", dense_pte, edge_value).reshape(
            len(dataset), num_pairs, paths_per_od
        )
        absolute = torch.log1p(raw_path_value.clamp_min(0.0))
        relative = absolute - absolute.mean(dim=-1, keepdim=True)
        record = {
            "gamma": gamma,
            "edge_value": distribution(edge_value),
            "path_value_raw_additive": distribution(raw_path_value),
            "path_absolute_log1p": distribution(absolute),
            "path_relative_centered_log1p": distribution(relative),
            "path_relative_abs_centered_log1p": distribution(relative, absolute=True),
            "correlation_with_uniform": {
                "edge_pearson_all_snapshots": pearson(edge_value, uniform_edge_pressure),
                "edge_spearman_all_snapshots": spearman(edge_value, uniform_edge_pressure),
                "path_absolute_vs_uniform_max_pearson": pearson(absolute, torch.log1p(uniform_path_max)),
                "path_absolute_vs_uniform_mean_pearson": pearson(absolute, torch.log1p(uniform_path_mean)),
            },
            "finite": bool(
                torch.isfinite(edge_value).all()
                and torch.isfinite(absolute).all()
                and torch.isfinite(relative).all()
            ),
        }
        records.append(record)

    structural = {
        "r_fraction": distribution(replacement_fraction),
        "positive_fraction": float((replacement_fraction > 0).to(torch.float64).mean().item()),
        "od_edge_fraction_ge_0p5": float((replacement_fraction >= 0.5).to(torch.float64).mean().item()),
        "od_edge_fraction_ge_0p75": float((replacement_fraction >= 0.75).to(torch.float64).mean().item()),
        "od_edge_fraction_eq_1": float((replacement_fraction == 1.0).to(torch.float64).mean().item()),
    }

    target_edges = {}
    mean_uniform = uniform_edge_pressure.mean(dim=0)
    for target in TARGET_EDGES:
        if target not in cluster_info.edges_map:
            continue
        edge_index = int(cluster_info.edges_map[target])
        r = replacement_fraction[:, edge_index]
        item = {
            "edge_index": edge_index,
            "capacity": float(capacities[0, edge_index].item()),
            "structural_r": {
                "max": float(r.max().item()),
                "mean_positive": float(r[r > 0].mean().item()) if (r > 0).any() else 0.0,
                "od_count_positive": int((r > 0).sum().item()),
                "od_count_ge_0p5": int((r >= 0.5).sum().item()),
                "od_count_ge_0p75": int((r >= 0.75).sum().item()),
                "od_count_eq_1": int((r == 1.0).sum().item()),
            },
            "uniform_mean": float(mean_uniform[edge_index].item()),
            "uniform_rank_desc": int(
                1 + (mean_uniform > mean_uniform[edge_index]).sum().item()
            ),
            "by_gamma": {},
        }
        for gamma in GAMMAS:
            mean_value = edge_values_by_gamma[gamma].mean(dim=0)
            value = mean_value[edge_index]
            item["by_gamma"][str(int(gamma))] = {
                "mean_edge_value": float(value.item()),
                "rank_desc": int(1 + (mean_value > value).sum().item()),
                "retention_vs_gamma1": float(
                    value.div(edge_values_by_gamma[1.0].mean(dim=0)[edge_index].clamp_min(1e-15)).item()
                ),
            }
        target_edges[f"{target[0]}->{target[1]}"] = item

    top_edges = {}
    for gamma in GAMMAS:
        means = edge_values_by_gamma[gamma].mean(dim=0)
        order = means.argsort(descending=True)
        top_edges[str(int(gamma))] = [
            {
                "edge": f"{index_to_edge[int(index)][0]}->{index_to_edge[int(index)][1]}",
                "edge_index": int(index),
                "mean_edge_value": float(means[index].item()),
                "retention_vs_gamma1": float(
                    means[index]
                    .div(edge_values_by_gamma[1.0].mean(dim=0)[index].clamp_min(1e-15))
                    .item()
                ),
                "max_r": float(replacement_fraction[:, index].max().item()),
                "od_count_ge_0p5": int((replacement_fraction[:, index] >= 0.5).sum().item()),
                "od_count_eq_1": int((replacement_fraction[:, index] == 1.0).sum().item()),
            }
            for index in order[:15].tolist()
        ]

    payload = {
        "method": "Medium candidate-path irreplaceability scout",
        "snapshots": [350, 358],
        "topology": shared.TOPOLOGY,
        "information_contract": "Predicted Medium ESM demand and static candidate paths/capacities only",
        "num_pairs": num_pairs,
        "paths_per_od": paths_per_od,
        "num_edges": num_edges,
        "definition": {
            "r_ie": "fraction of the 8 candidate paths for OD i that contain edge e",
            "edge_value_gamma": "sum_i predicted_medium_i * r_ie^gamma / capacity_e",
            "gamma_1_control": "exactly equal to uniform-split Medium edge utilization",
            "path_raw": "sum of edge_value_gamma along the candidate path",
            "path_absolute": "log1p(path_raw)",
            "path_relative": "path_absolute minus within-OD mean(path_absolute)",
        },
        "structural": structural,
        "records": records,
        "target_edges": target_edges,
        "top_edges_by_mean_value": top_edges,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
