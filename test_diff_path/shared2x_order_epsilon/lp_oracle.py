from __future__ import annotations

import json
import sys
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
SHARED_DIR = TEST_DIR / "shared2x_order_regularizer"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(SHARED_DIR))

import run_experiment as shared  # noqa: E402
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster  # noqa: E402


EPSILONS = (0.0025, 0.005, 0.01)
TOL = 1e-7


def _scipy_pte(tensor: torch.Tensor):
    tensor = tensor.coalesce().cpu()
    indices = tensor.indices().numpy()
    values = tensor.values().to(dtype=torch.float64).numpy()
    from scipy.sparse import coo_matrix

    return coo_matrix(
        (values, (indices[0], indices[1])), shape=tensor.shape
    ).tocsr()


def _flows(snapshot, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (snapshot.tm1, snapshot.tm2, snapshot.tm3)
    )


def solve_snapshot(
    snapshot,
    pte,
    approach: str,
    *,
    epsilon: float | None = None,
) -> dict:
    """Solve the exact continuous path-flow lexicographic comparison."""
    if approach not in ("original", "swap", "epsilon"):
        raise ValueError(approach)
    if approach == "epsilon" and epsilon not in EPSILONS:
        raise ValueError(epsilon)
    k = 8
    tms = _flows(snapshot, k)
    num_paths = len(tms[0])
    if num_paths % k:
        raise ValueError("traffic vector is not divisible by K")
    num_pairs = num_paths // k
    capacities = np.asarray(snapshot.capacities, dtype=np.float64).reshape(-1)
    positive_capacity = capacities > 0
    model = gp.Model("shared2x_order_epsilon")
    model.Params.OutputFlag = 0
    model.Params.NumericFocus = 3
    model.Params.FeasibilityTol = 1e-8
    x = [model.addMVar(num_paths, lb=0.0, ub=1.0, name=f"x{c}") for c in range(3)]
    for class_index in range(3):
        matrix = x[class_index].reshape(num_pairs, k)
        model.addConstrs(matrix[pair, :].sum() <= 1.0 for pair in range(num_pairs))
    path_flow = [x[c] * tms[c] for c in range(3)]
    link_flow = [pte.T @ path_flow[c] for c in range(3)]
    total_link = link_flow[0] + link_flow[1] + link_flow[2]
    for edge in range(len(capacities)):
        model.addConstr(total_link[edge] <= capacities[edge])
    totals = [path_flow[c].sum() for c in range(3)]

    def optimize(expr, sense: int) -> float:
        model.setObjective(expr, sense)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi status {model.Status}")
        return float(expr.X) if isinstance(expr, gp.Var) else float(expr.getValue())

    high_star = optimize(totals[0], GRB.MAXIMIZE)
    if approach == "epsilon":
        model.addConstr(totals[0] >= (1.0 - float(epsilon)) * high_star - TOL)
    else:
        model.addConstr(totals[0] >= high_star - TOL)

    high_mlu = model.addVar(lb=0.0, name="high_mlu")
    for edge in np.flatnonzero(positive_capacity):
        model.addConstr(link_flow[0][edge] <= high_mlu * capacities[edge])
    if approach == "original":
        high_mlu_star = optimize(high_mlu, GRB.MINIMIZE)
        model.addConstr(high_mlu <= high_mlu_star + TOL)

    medium_star = optimize(totals[1], GRB.MAXIMIZE)
    model.addConstr(totals[1] >= medium_star - TOL)
    if approach == "swap":
        high_mlu_star = optimize(high_mlu, GRB.MINIMIZE)
        model.addConstr(high_mlu <= high_mlu_star + TOL)
    low_star = optimize(totals[2], GRB.MAXIMIZE)
    model.addConstr(totals[2] >= low_star - TOL)
    total_mlu = model.addVar(lb=0.0, name="total_mlu")
    for edge in np.flatnonzero(positive_capacity):
        model.addConstr(total_link[edge] <= total_mlu * capacities[edge])
    optimize(total_mlu, GRB.MINIMIZE)

    class_path_flows = [np.asarray(path_flow[c].getValue(), dtype=np.float64) for c in range(3)]
    class_totals = [float(value.sum()) for value in class_path_flows]
    high_links = np.asarray(pte.T @ class_path_flows[0].reshape(-1, 1)).reshape(-1)
    all_links = np.asarray(
        pte.T @ sum(class_path_flows).reshape(-1, 1)
    ).reshape(-1)
    exact_high_mlu = float(np.max(high_links[positive_capacity] / capacities[positive_capacity]))
    exact_total_mlu = float(np.max(all_links[positive_capacity] / capacities[positive_capacity]))
    return {
        "approach": approach,
        "epsilon": epsilon,
        "high_star": high_star,
        "high": class_totals[0],
        "medium": class_totals[1],
        "low": class_totals[2],
        "high_mlu": exact_high_mlu,
        "total_mlu": exact_total_mlu,
        "high_path_flow": class_path_flows[0],
        "medium_path_flow": class_path_flows[1],
        "low_path_flow": class_path_flows[2],
    }


def run_level0(output_dir: Path, *, force: bool = False) -> dict:
    complete_path = output_dir / "complete.json"
    if complete_path.exists() and not force:
        return json.loads(complete_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    props = shared.build_props(1, device)
    dataset = DM_Dataset_within_Cluster(props, 0, 0, 32)
    pte = _scipy_pte(dataset.pte)
    rows: list[dict] = []
    path_rows: list[dict] = []
    methods = [("original", None), ("swap", None)] + [
        ("epsilon", epsilon) for epsilon in EPSILONS
    ]
    for local_index, snapshot in enumerate(dataset.list_snapshots):
        solved = {
            (approach, epsilon): solve_snapshot(
                snapshot, pte, approach, epsilon=epsilon
            )
            for approach, epsilon in methods
        }
        baseline = solved[("original", None)]
        oracle_medium = max(
            float(dataset.list_optimal_values_2_mf[local_index])
            - float(dataset.list_optimal_values_1_mf[local_index]),
            1e-12,
        )
        for approach, epsilon in methods:
            result = solved[(approach, epsilon)]
            high_l1 = float(
                np.abs(result["high_path_flow"] - baseline["high_path_flow"]).sum()
            )
            rows.append(
                {
                    "snapshot": local_index,
                    "approach": approach,
                    "epsilon": "" if epsilon is None else epsilon,
                    "high": result["high"],
                    "medium": result["medium"],
                    "low": result["low"],
                    "high_norm": result["high"] / max(result["high_star"], 1e-12),
                    "medium_norm_on_stored_oracle": result["medium"] / oracle_medium,
                    "medium_norm_gain_vs_original": (
                        result["medium"] - baseline["medium"]
                    )
                    / oracle_medium,
                    "high_mlu": result["high_mlu"],
                    "total_mlu": result["total_mlu"],
                    "high_path_flow_l1_vs_original": high_l1,
                }
            )
            for path_id, value in enumerate(result["high_path_flow"]):
                path_rows.append(
                    {
                        "snapshot": local_index,
                        "approach": approach,
                        "epsilon": "" if epsilon is None else epsilon,
                        "class": "High",
                        "path_id": path_id,
                        "flow": float(value),
                    }
                )
    import csv

    def write_csv(path: Path, values: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    write_csv(output_dir / "lp_metrics.csv", rows)
    write_csv(output_dir / "lp_high_path_flows.csv", path_rows)
    summaries = []
    for approach, epsilon in methods:
        selected = [
            row
            for row in rows
            if row["approach"] == approach
            and row["epsilon"] == ("" if epsilon is None else epsilon)
        ]
        gains = np.asarray(
            [row["medium_norm_gain_vs_original"] for row in selected], dtype=np.float64
        )
        summaries.append(
            {
                "approach": approach,
                "epsilon": epsilon,
                "medium_norm_gain_mean": float(gains.mean()),
                "medium_norm_gain_p10": float(np.percentile(gains, 10)),
                "positive_fraction": float((gains > 0).mean()),
                "high_mlu_mean": float(np.mean([row["high_mlu"] for row in selected])),
                "total_mlu_max": float(max(row["total_mlu"] for row in selected)),
            }
        )
    candidates = [row for row in summaries if row["approach"] != "original"]
    go = max(row["medium_norm_gain_mean"] for row in candidates) >= 0.005
    complete = {
        "status": "GO" if go else "NO_GO",
        "neural_scaling_authorized": go,
        "samples": 32,
        "methods": summaries,
        "gate": "at least one swap/epsilon mean Medium NormFulFill ceiling gain >= 0.005",
    }
    complete_path.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    return complete
