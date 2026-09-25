from __future__ import annotations

import importlib
import json
import math
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
    / "virtual_medium_dual"
    / "probe_snapshots_350_357.json"
)
POWERS = (2.0, 3.0, 4.0, 5.0)
TEMPERATURES = (
    0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0,
    6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0,
)


def quantiles(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1).cpu()
    return {
        "mean": float(flat.mean().item()),
        "p50": float(torch.quantile(flat, 0.50).item()),
        "p90": float(torch.quantile(flat, 0.90).item()),
        "p99": float(torch.quantile(flat, 0.99).item()),
        "max": float(flat.max().item()),
    }


def edge_utilization(
    q: torch.Tensor,
    demand: torch.Tensor,
    pte: torch.Tensor,
    capacity: torch.Tensor,
) -> torch.Tensor:
    flow = (q * demand.unsqueeze(-1)).reshape(-1)
    load = torch.sparse.mm(pte.t(), flow.unsqueeze(-1)).squeeze(-1)
    return load / capacity.clamp_min(1e-9)


def one_snapshot(
    demand: torch.Tensor,
    capacity: torch.Tensor,
    pte: torch.Tensor,
    num_pairs: int,
    power: float,
    temperature: float,
) -> dict[str, torch.Tensor]:
    q0 = torch.full(
        (num_pairs, 8), 1.0 / 8.0, dtype=torch.float64, device=demand.device
    )
    q = q0
    rounds = []
    for _ in range(2):
        util = edge_utilization(q, demand, pte, capacity)
        dual = util.clamp_min(0.0).pow(float(power) - 1.0)
        cost = torch.sparse.mm(pte, dual.unsqueeze(-1)).squeeze(-1).reshape(num_pairs, 8)
        # torch.softmax performs the max-subtraction stabilization internally.
        q_next = torch.softmax(-cost / float(temperature), dim=-1)
        rounds.append((q, util, dual, cost, q_next))
        q = q_next

    q1 = rounds[0][4]
    q2 = rounds[1][4]
    final_util = edge_utilization(q2, demand, pte, capacity)
    final_cost = rounds[1][3]
    absolute = torch.log1p(final_cost.clamp_min(0.0))
    relative = absolute - absolute.mean(dim=-1, keepdim=True)
    tv12 = 0.5 * (q2 - q1).abs().sum(dim=-1)
    entropy = -(q2.clamp_min(1e-15) * q2.clamp_min(1e-15).log()).sum(dim=-1)
    return {
        "uniform_util": rounds[0][1],
        "round1_cost": rounds[0][3],
        "round1_q": q1,
        "round2_util": rounds[1][1],
        "round2_dual": rounds[1][2],
        "round2_cost": final_cost,
        "round2_q": q2,
        "final_util": final_util,
        "absolute_feature": absolute,
        "relative_feature": relative,
        "tv12": tv12,
        "argmax_flip": (q1.argmax(dim=-1) != q2.argmax(dim=-1)).float(),
        "pmax": q2.max(dim=-1).values,
        "entropy": entropy,
        "effective_paths": entropy.exp(),
    }


def main() -> None:
    for path in (ROOT, TEST_DIR, TEST_DIR / "shared2x_order_regularizer"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    shared = importlib.import_module("run_experiment")
    from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster

    device = torch.device("cpu")
    props = shared.build_props(4, device)
    dataset = DM_Dataset_within_Cluster(props, 0, 350, 358)
    pte = dataset.pte.coalesce().to(device=device, dtype=torch.float64)
    num_pairs = int(dataset.num_pairs)
    predicted_medium = [
        torch.as_tensor(value).reshape(num_pairs, 8, 1)[:, 0, 0].to(dtype=torch.float64)
        for value in dataset.list_tms2_pred
    ]
    capacities = [
        torch.as_tensor(value).reshape(-1).to(dtype=torch.float64)
        for value in dataset.list_capacities
    ]

    records = []
    for power in POWERS:
        for temperature in TEMPERATURES:
            buckets: dict[str, list[torch.Tensor]] = {}
            for demand, capacity in zip(predicted_medium, capacities):
                result = one_snapshot(
                    demand, capacity, pte, num_pairs, power, temperature
                )
                for key, value in result.items():
                    buckets.setdefault(key, []).append(value)
            joined = {key: torch.stack(value) for key, value in buckets.items()}
            record = {
                "power": power,
                "temperature": temperature,
                "uniform_edge_util": quantiles(joined["uniform_util"]),
                "round2_edge_util": quantiles(joined["round2_util"]),
                "final_edge_util": quantiles(joined["final_util"]),
                "round2_dual": quantiles(joined["round2_dual"]),
                "round2_path_cost": quantiles(joined["round2_cost"]),
                "absolute_feature_log1p_cost": quantiles(joined["absolute_feature"]),
                "relative_feature_centered_log1p": quantiles(
                    joined["relative_feature"].abs()
                ),
                "q2_pmax": quantiles(joined["pmax"]),
                "q2_pmax_ge_0p9_fraction": float(
                    (joined["pmax"] >= 0.9).float().mean().item()
                ),
                "q2_normalized_entropy_mean": float(
                    joined["entropy"].mean().item() / math.log(8.0)
                ),
                "q2_effective_paths": quantiles(joined["effective_paths"]),
                "round1_to_round2_tv": quantiles(joined["tv12"]),
                "round1_to_round2_argmax_flip_fraction": float(
                    joined["argmax_flip"].mean().item()
                ),
                "finite": bool(
                    all(torch.isfinite(value).all().item() for value in joined.values())
                ),
            }
            records.append(record)

    payload = {
        "method": "two-round virtual Medium dual probe",
        "snapshots": [350, 358],
        "topology": shared.TOPOLOGY,
        "paths_per_od": 8,
        "num_pairs": num_pairs,
        "algorithm": (
            "q0=uniform; repeat twice: u=P^T(q*d)/capacity, "
            "lambda=u^(power-1), cost=P*lambda, q=softmax(-cost/temperature)"
        ),
        "features": {
            "absolute": "log1p(final path cost)",
            "relative": "absolute minus within-OD mean absolute",
        },
        "information_contract": "ESM Medium demand only; no actual TM or oracle input",
        "grid": {"powers": list(POWERS), "temperatures": list(TEMPERATURES)},
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
