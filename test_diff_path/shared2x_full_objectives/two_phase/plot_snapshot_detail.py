from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import networkx as nx
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DETAIL_DIR = THIS_DIR / "artifacts" / "bad_sample_diagnostics" / "snapshot_205"
CLASSES = ("High", "Medium", "Low")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def load_matrix(path: Path) -> tuple[list[int], np.ndarray]:
    rows = read_csv(path)
    nodes = [int(key) for key in rows[0] if key != "source"]
    matrix = np.asarray(
        [[float(row[str(node)]) for node in nodes] for row in rows],
        dtype=np.float64,
    )
    return nodes, matrix


def plot_topology(detail_dir: Path, detail: dict, edge_rows: list[dict[str, str]]) -> None:
    snapshot_id = int(detail["snapshot"])
    graph = nx.DiGraph()
    positions = {
        int(node["id"]): np.asarray([float(node["x"]), float(node["y"])])
        for node in detail["visualization"]["nodes"]
    }
    edge_list = [(int(row["source"]), int(row["target"])) for row in edge_rows]
    graph.add_nodes_from(positions)
    graph.add_edges_from(edge_list)
    panels = (
        ("esm", "ESM predicted request"),
        ("actual_request", "Actual request"),
        ("admitted", "Actual admitted"),
    )
    maximum = max(
        float(row[f"{basis}_all_utilization"])
        for row in edge_rows
        for basis, _label in panels
    )
    norm = Normalize(0.0, maximum)
    cmap = plt.get_cmap("viridis")
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, (basis, label) in zip(axes, panels):
        values = [float(row[f"{basis}_all_utilization"]) for row in edge_rows]
        max_row = max(edge_rows, key=lambda row: float(row[f"{basis}_all_utilization"]))
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis,
            node_size=330,
            node_color="#f4f4f4",
            edgecolors="#222222",
            linewidths=0.8,
        )
        nx.draw_networkx_labels(graph, positions, ax=axis, font_size=7)
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edgelist=edge_list,
            edge_color=values,
            edge_cmap=cmap,
            edge_vmin=0.0,
            edge_vmax=maximum,
            width=[0.5 + min(value, 4.0) * 0.7 for value in values],
            arrows=True,
            arrowsize=7,
            connectionstyle="arc3,rad=0.07",
            alpha=0.82,
        )
        axis.set_title(
            f"{label}\nmax {max_row['source']}→{max_row['target']} = "
            f"{float(max_row[f'{basis}_all_utilization']):.3f}"
        )
        axis.set_axis_off()
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    figure.colorbar(scalar, ax=axes, shrink=0.75, label="Total link utilization")
    figure.suptitle(f"Snapshot {snapshot_id} · GEANT shared-2x directed topology", fontsize=14)
    figure.savefig(detail_dir / "topology_load_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_traffic_matrices(detail_dir: Path, detail: dict) -> None:
    snapshot_id = int(detail["snapshot"])
    figure, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    for row_index, class_name in enumerate(CLASSES):
        key = class_name.lower()
        nodes, actual = load_matrix(detail_dir / f"traffic_matrix_actual_{key}.csv")
        _nodes, predicted = load_matrix(detail_dir / f"traffic_matrix_esm_{key}.csv")
        error = predicted - actual
        common_max = max(float(actual.max()), float(predicted.max()), 1e-12)
        error_max = max(float(np.abs(error).max()), 1e-12)
        panels = (
            ("Actual", actual, "viridis", Normalize(0.0, common_max)),
            ("ESM prediction", predicted, "viridis", Normalize(0.0, common_max)),
            ("Prediction − actual", error, "coolwarm", TwoSlopeNorm(vmin=-error_max, vcenter=0.0, vmax=error_max)),
        )
        for column_index, (title, matrix, cmap, norm) in enumerate(panels):
            axis = axes[row_index, column_index]
            image = axis.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
            axis.set_title(f"{class_name} · {title}")
            axis.set_xlabel("Destination node")
            axis.set_ylabel("Source node")
            axis.set_xticks(range(len(nodes)), labels=nodes, fontsize=6, rotation=90)
            axis.set_yticks(range(len(nodes)), labels=nodes, fontsize=6)
            figure.colorbar(image, ax=axis, shrink=0.72)
    figure.suptitle(f"Snapshot {snapshot_id} traffic matrices (flow units)", fontsize=14)
    figure.savefig(detail_dir / "traffic_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_oracle_vs_actual(detail_dir: Path, detail: dict) -> None:
    snapshot_id = int(detail["snapshot"])
    edge_rows = read_csv(detail_dir / "oracle_vs_actual_admitted_edges.csv")
    edge_rows = [row for row in edge_rows if float(row["capacity"]) > 0.0]
    positions = {
        int(node["id"]): np.asarray([float(node["x"]), float(node["y"])])
        for node in detail["visualization"]["nodes"]
    }
    graph = nx.DiGraph()
    graph.add_nodes_from(positions)
    edge_list = [(int(row["source"]), int(row["target"])) for row in edge_rows]
    graph.add_edges_from(edge_list)
    panels = (
        ("oracle_utilization", "Re-solved lexicographic oracle"),
        ("actual_admitted_utilization", "Hattrick actual admitted"),
    )
    cmap = plt.get_cmap("viridis")
    norm = Normalize(0.0, 1.0)
    max_capacity = max(float(row["capacity"]) for row in edge_rows)

    def capacity_width(row: dict[str, str]) -> float:
        return 0.7 + 6.3 * np.sqrt(float(row["capacity"]) / max_capacity)

    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for axis, (field, label) in zip(axes, panels):
        values = [min(max(float(row[field]), 0.0), 1.0) for row in edge_rows]
        saturated = sum(float(row[field]) >= 0.999999 for row in edge_rows)
        max_row = max(edge_rows, key=lambda row: float(row[field]))
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=axis,
            node_size=330,
            node_color="#f4f4f4",
            edgecolors="#222222",
            linewidths=0.8,
        )
        nx.draw_networkx_labels(graph, positions, ax=axis, font_size=7)
        nx.draw_networkx_edges(
            graph,
            positions,
            ax=axis,
            edgelist=edge_list,
            edge_color=values,
            edge_cmap=cmap,
            edge_vmin=0.0,
            edge_vmax=1.0,
            width=[capacity_width(row) for row in edge_rows],
            arrows=True,
            arrowsize=7,
            connectionstyle="arc3,rad=0.07",
            alpha=0.85,
        )
        axis.set_title(
            f"{label}\nmax {max_row['source']}→{max_row['target']} = "
            f"{float(max_row[field]):.3f}; saturated links = {saturated}"
        )
        axis.set_axis_off()
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    figure.colorbar(scalar, ax=axes, shrink=0.78, label="Cumulative admitted link load / capacity")
    figure.suptitle(
        f"Snapshot {snapshot_id} · Oracle vs restored-objective Hattrick\n"
        "color = utilization; width = link capacity",
        fontsize=14,
    )
    figure.savefig(
        detail_dir / "oracle_vs_actual_admitted_topology.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render snapshot-detail CSV/JSON plots")
    parser.add_argument("--detail-dir", type=Path, default=DEFAULT_DETAIL_DIR)
    args = parser.parse_args()
    detail_dir = args.detail_dir.resolve()
    detail = json.loads((detail_dir / "snapshot_detail.json").read_text(encoding="utf-8"))
    edge_rows = read_csv(detail_dir / "topology_edges_with_loads.csv")
    plot_topology(detail_dir, detail, edge_rows)
    plot_traffic_matrices(detail_dir, detail)
    plot_oracle_vs_actual(detail_dir, detail)
    print(detail_dir / "topology_load_comparison.png")
    print(detail_dir / "traffic_matrices.png")
    print(detail_dir / "oracle_vs_actual_admitted_topology.png")


if __name__ == "__main__":
    main()
