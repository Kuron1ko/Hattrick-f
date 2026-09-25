from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import csr_matrix


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[2]
DEFAULT_DETAIL_DIR = THIS_DIR / "artifacts" / "bad_sample_diagnostics" / "snapshot_205"
TOPOLOGY = "geant_priomask500_shared_load2x_train"
K = 8
CLASSES = ("High", "Medium", "Low")

sys.path.insert(0, str(ROOT))
from frameworks.gurobi_utils import GurobiModel  # noqa: E402
from utils.args_parser import parse_args  # noqa: E402
from utils.cluster_utils import Cluster_Info  # noqa: E402
from utils.snapshot_utils import Read_Snapshot  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def path_text(path: list[tuple]) -> str:
    if not path:
        return ""
    nodes = [path[0][0], *[edge[1] for edge in path]]
    return "->".join(str(node) for node in nodes)


def build_props():
    props = parse_args(
        [
            "--topo", TOPOLOGY,
            "--framework", "gurobi",
            "--num_paths_per_pair", str(K),
            "--pred", "0",
            "--priority", "3",
            "--objs", "mf", "mf", "mf",
            "--gur_mode", "flexile",
            "--cluster", "0",
            "--path_mask", "0",
            "--tol", "0.00001",
        ]
    )
    return props


def load_scalar(path: Path, index: int) -> float:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    return float(values[index])


def solve(snapshot_index: int, detail_dir: Path) -> dict:
    props = build_props()
    result_dir = ROOT / "results" / TOPOLOGY / f"{K}sp" / "0"
    filenames = np.loadtxt(
        result_dir / "filenames.txt", dtype="U", delimiter=","
    ).reshape(-1, 3)
    topology_filename, pairs_filename, tm_filename = filenames[snapshot_index]
    snapshot = Read_Snapshot(
        props,
        topology_filename.strip(),
        pairs_filename.strip(),
        tm_filename.strip(),
    )
    cluster = Cluster_Info(snapshot, props, 0)
    paths = cluster.compute_ksp_paths(K, snapshot.pairs)
    pte = csr_matrix(cluster.get_paths_to_edges_matrix(paths).to_dense().numpy())

    opt1 = load_scalar(result_dir / "gt_optimal_values_mf.txt", snapshot_index)
    opt2 = load_scalar(result_dir / "gt_optimal_values_mf_mf.txt", snapshot_index)
    opt3 = load_scalar(result_dir / "gt_optimal_values_mf_mf_mf.txt", snapshot_index)

    solver = GurobiModel(props, snapshot, "flexile")
    solver.model.setParam("OutputFlag", 0)
    solver.model.setParam("NumericFocus", 3)
    solver.model.setParam("FeasibilityTol", 1e-6)
    solver.add_variables(props.objs, snapshot_index)
    solver.add_mlu_variables(props.objs)
    solver.mul_vars_by_tms()
    solver.add_demand_constraints(props.objs)
    solver.add_optimality_constraints(opt1, opt2, props.objs)
    solver.add_capacity_constraints(props.objs, pte)
    solver.define_objective(props.objs)
    solver.model.optimize()
    if solver.model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status {solver.model.Status}")

    pair_keys = list(paths.keys())
    path_rows: list[dict] = []
    class_path_flows: list[np.ndarray] = []
    class_sums: list[float] = []
    for class_index, class_name in enumerate(CLASSES):
        ratios = np.asarray(solver.model_variables[class_index].X, dtype=np.float64)
        tm = np.asarray(solver.tms[class_index], dtype=np.float64).reshape(-1, K)
        flow = ratios * tm
        class_path_flows.append(flow.reshape(-1))
        class_sums.append(float(flow.sum()))
        for pair_index, (source, target) in enumerate(pair_keys):
            for path_index in range(K):
                path_rows.append(
                    {
                        "snapshot": snapshot_index,
                        "class": class_name,
                        "pair_index": pair_index,
                        "source": source,
                        "target": target,
                        "path_index": path_index,
                        "path": path_text(paths[(source, target)][path_index]),
                        "oracle_admission_ratio": float(ratios[pair_index, path_index]),
                        "oracle_path_flow": float(flow[pair_index, path_index]),
                    }
                )
    write_csv(detail_dir / "oracle_recomputed_path_flows.csv", path_rows)

    capacities = np.asarray(solver.capacities, dtype=np.float64).reshape(-1)
    class_link_loads = [np.asarray(pte.T @ flow.reshape(-1, 1)).reshape(-1) for flow in class_path_flows]
    total_link_load = sum(class_link_loads)
    oracle_utilization = total_link_load / capacities
    actual_rows = read_csv(detail_dir / "topology_edges_with_loads.csv")
    edge_names = list(snapshot.graph.edges())
    comparison_rows: list[dict] = []
    for edge_id, (source, target) in enumerate(edge_names):
        actual = actual_rows[edge_id]
        actual_capacity = float(actual["capacity"])
        actual_load = float(actual["admitted_all_load"])
        comparison_rows.append(
            {
                "edge_id": edge_id,
                "source": source,
                "target": target,
                "capacity": actual_capacity,
                "oracle_high_load": float(class_link_loads[0][edge_id]),
                "oracle_medium_load": float(class_link_loads[1][edge_id]),
                "oracle_low_load": float(class_link_loads[2][edge_id]),
                "oracle_total_load": float(total_link_load[edge_id]),
                "oracle_utilization": float(oracle_utilization[edge_id]),
                "actual_admitted_high_load": float(actual["admitted_high_load"]),
                "actual_admitted_medium_load": float(actual["admitted_medium_load"]),
                "actual_admitted_low_load": float(actual["admitted_low_load"]),
                "actual_admitted_total_load": actual_load,
                "actual_admitted_utilization": actual_load / actual_capacity,
                "utilization_difference_actual_minus_oracle": actual_load / actual_capacity - float(oracle_utilization[edge_id]),
            }
        )
    write_csv(detail_dir / "oracle_vs_actual_admitted_edges.csv", comparison_rows)

    summary = {
        "snapshot": snapshot_index,
        "definition": "re-solved ground-truth 3-stage lexicographic max-flow oracle with the repository's 1e-5 preservation tolerance",
        "stored_cumulative_oracle": {"high": opt1, "high_medium": opt2, "all": opt3},
        "stored_incremental_oracle": {
            "high": opt1,
            "medium": opt2 - opt1,
            "low": opt3 - opt2,
        },
        "resolved_final_class_flow": dict(zip((name.lower() for name in CLASSES), class_sums)),
        "resolved_objective": float(solver.model.ObjVal),
        "objective_difference_from_stored": float(solver.model.ObjVal - opt3),
        "max_oracle_utilization": float(oracle_utilization.max()),
        "max_actual_admitted_utilization": max(float(row["actual_admitted_utilization"]) for row in comparison_rows),
        "capacity_feasible": bool(float(oracle_utilization.max()) <= 1.000001),
        "note": "The original run stored scalar objectives but not final path variables; this is a deterministic re-solve of the same LP. Degenerate optimal routings can have different per-link loads while preserving the same objective.",
    }
    (detail_dir / "oracle_recomputed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    solver.model.dispose()
    gp.disposeDefaultEnv()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-solve and export snapshot oracle topology")
    parser.add_argument("--snapshot", type=int, default=205)
    parser.add_argument("--detail-dir", type=Path, default=DEFAULT_DETAIL_DIR)
    args = parser.parse_args()
    summary = solve(args.snapshot, args.detail_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("GRB_LICENSE_FILE", r"D:\kuroresearch\gurobi_license\gurobi.lic")
    main()
