from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import networkx as nx


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[2]
ARTIFACT_ROOT = THIS_DIR / "artifacts"
DEFAULT_CANDIDATE_DIR = (
    ARTIFACT_ROOT
    / "level2_proxy"
    / "tail_flow_balanced_squared"
    / "multiplier_0p25"
    / "seed_490"
)
DEFAULT_BASELINE_DIR = ARTIFACT_ROOT / "level2_proxy" / "multiplier_0" / "seed_490"
DEFAULT_OUTPUT_DIR = ARTIFACT_ROOT / "bad_sample_diagnostics"

sys.path.insert(0, str(THIS_DIR))
import run_two_phase  # noqa: E402


core = run_two_phase.core
CLASSES = ("High", "Medium", "Low")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def metric_index(rows: list[dict[str, str]]) -> dict[tuple[int, str], dict[str, float]]:
    result: dict[tuple[int, str], dict[str, float]] = {}
    for row in rows:
        converted = {
            key: (float(value) if key not in {"snapshot", "class"} else value)
            for key, value in row.items()
        }
        result[(int(row["snapshot"]), row["class"])] = converted
    return result


def select_bad_samples(candidate: dict, baseline: dict) -> list[dict]:
    snapshots = sorted({snapshot for snapshot, class_name in candidate if class_name == "Medium"})
    scores = []
    for snapshot in snapshots:
        medium = float(candidate[(snapshot, "Medium")]["norm_fulfill"])
        low = float(candidate[(snapshot, "Low")]["norm_fulfill"])
        baseline_medium = float(baseline[(snapshot, "Medium")]["norm_fulfill"])
        scores.append(
            {
                "snapshot": snapshot,
                "medium_norm": medium,
                "low_norm": low,
                "inversion_gap": low - medium,
                "medium_delta_vs_baseline": medium - baseline_medium,
            }
        )

    chosen: list[dict] = []
    used: set[int] = set()
    criteria = (
        ("Medium NormFulFill 最低", lambda row: row["medium_norm"]),
        ("在不重复最低样本后 inversion gap 最大", lambda row: -row["inversion_gap"]),
        ("相对 matched baseline 的 Medium 退化最大", lambda row: row["medium_delta_vs_baseline"]),
    )
    for reason, key in criteria:
        for row in sorted(scores, key=key):
            if row["snapshot"] not in used:
                item = dict(row)
                item["selection_reason"] = reason
                chosen.append(item)
                used.add(row["snapshot"])
                break
    return chosen


def safe_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.size < 2 or float(np.std(actual)) <= 1e-12 or float(np.std(predicted)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def policy_statistics(policy: np.ndarray, positive_mask: np.ndarray) -> tuple[float, float, float]:
    selected = policy[positive_mask] if bool(positive_mask.any()) else policy
    clipped = np.clip(selected, 1e-12, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1) / math.log(policy.shape[1])
    top_share = selected.max(axis=1)
    return float(entropy.mean()), float(top_share.mean()), float(top_share.max())


def tensor_link_load(paths_to_edges: torch.Tensor, path_flow: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(
        paths_to_edges.to(dtype=torch.float32).t(),
        path_flow.to(dtype=torch.float32).reshape(1, -1).t(),
    ).reshape(-1)


def path_text(path: list[tuple]) -> str:
    if not path:
        return ""
    nodes = [path[0][0], *[edge[1] for edge in path]]
    return "->".join(str(node) for node in nodes)


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def matrix_rows(
    node_ids: list,
    pair_keys: list[tuple],
    values: np.ndarray,
) -> list[dict]:
    lookup = {pair: float(value) for pair, value in zip(pair_keys, values)}
    return [
        {"source": source, **{str(target): lookup.get((source, target), 0.0) for target in node_ids}}
        for source in node_ids
    ]


def export_snapshot_detail(
    snapshot_id: int,
    snapshot,
    dataset,
    values,
    policies,
    admitted,
    candidate_metrics: dict,
    baseline_metrics: dict,
    output_dir: Path,
    k: int,
    visualization_path: Path | None,
) -> dict:
    detail_dir = output_dir / f"snapshot_{snapshot_id}"
    detail_dir.mkdir(parents=True, exist_ok=True)
    node_ids = list(snapshot.graph.nodes())
    pair_keys = list(dataset.pij.keys())
    actual_tms = (values[2], values[4], values[6])
    predicted_tms = (values[3], values[5], values[7])
    oracle_totals = (
        values[11],
        values[12] - values[11],
        values[13] - values[12],
    )

    traffic_rows: list[dict] = []
    path_rows: list[dict] = []
    class_flows: dict[str, dict[str, torch.Tensor]] = {}
    matrix_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for class_index, class_name in enumerate(CLASSES):
        actual_od = actual_tms[class_index].detach().reshape(-1, k)[:, 0].float().cpu().numpy()
        predicted_od = predicted_tms[class_index].detach().reshape(-1, k)[:, 0].float().cpu().numpy()
        policy = policies[class_index].detach().reshape(-1, k).float().cpu().numpy()
        admitted_matrix = admitted[class_index].detach().reshape(-1, k).float().cpu().numpy()
        matrix_data[class_name] = (actual_od, predicted_od)
        write_csv(detail_dir / f"traffic_matrix_actual_{class_name.lower()}.csv", matrix_rows(node_ids, pair_keys, actual_od))
        write_csv(detail_dir / f"traffic_matrix_esm_{class_name.lower()}.csv", matrix_rows(node_ids, pair_keys, predicted_od))
        write_csv(detail_dir / f"traffic_matrix_error_{class_name.lower()}.csv", matrix_rows(node_ids, pair_keys, predicted_od - actual_od))

        for pair_index, (source, target) in enumerate(pair_keys):
            clipped = np.clip(policy[pair_index], 1e-12, 1.0)
            entropy = float(-(clipped * np.log(clipped)).sum() / math.log(k))
            top_path = int(np.argmax(policy[pair_index]))
            traffic_rows.append(
                {
                    "snapshot": snapshot_id,
                    "class": class_name,
                    "pair_index": pair_index,
                    "source": source,
                    "target": target,
                    "actual_demand": float(actual_od[pair_index]),
                    "esm_predicted_demand": float(predicted_od[pair_index]),
                    "signed_prediction_error": float(predicted_od[pair_index] - actual_od[pair_index]),
                    "absolute_prediction_error": float(abs(predicted_od[pair_index] - actual_od[pair_index])),
                    "system_admitted": float(admitted_matrix[pair_index].sum()),
                    "raw_fulfill": float(admitted_matrix[pair_index].sum() / max(actual_od[pair_index], 1e-12)),
                    "policy_entropy_normalized": entropy,
                    "top_path_index": top_path,
                    "top_path_share": float(policy[pair_index, top_path]),
                    "top_path": path_text(dataset.pij[(source, target)][top_path]),
                }
            )
            for path_index in range(k):
                share = float(policy[pair_index, path_index])
                path_rows.append(
                    {
                        "snapshot": snapshot_id,
                        "class": class_name,
                        "pair_index": pair_index,
                        "source": source,
                        "target": target,
                        "path_index": path_index,
                        "path": path_text(dataset.pij[(source, target)][path_index]),
                        "policy_share": share,
                        "actual_demand": float(actual_od[pair_index]),
                        "esm_predicted_demand": float(predicted_od[pair_index]),
                        "planned_path_flow_actual_demand": share * float(actual_od[pair_index]),
                        "planned_path_flow_esm_demand": share * float(predicted_od[pair_index]),
                        "admitted_path_flow": float(admitted_matrix[pair_index, path_index]),
                    }
                )

        policy_flat = torch.as_tensor(policy, device=values[2].device).reshape(-1)
        class_flows[class_name] = {
            "esm": policy_flat * predicted_tms[class_index].squeeze(0).squeeze(-1),
            "actual_request": policy_flat * actual_tms[class_index].squeeze(0).squeeze(-1),
            "admitted": admitted[class_index].squeeze(0),
        }

    write_csv(detail_dir / "traffic_matrix_long.csv", traffic_rows)
    write_csv(detail_dir / "system_path_predictions.csv", path_rows)

    capacities = values[1][:1].reshape(-1).float()
    edge_names = list(snapshot.graph.edges())
    link_loads: dict[str, dict[str, torch.Tensor]] = {}
    for basis in ("esm", "actual_request", "admitted"):
        link_loads[basis] = {
            class_name: tensor_link_load(dataset.pte, class_flows[class_name][basis])
            for class_name in CLASSES
        }
        link_loads[basis]["All"] = sum(link_loads[basis][class_name] for class_name in CLASSES)

    bottlenecks = {
        basis: int(torch.argmax(link_loads[basis]["All"] / capacities.clamp_min(1e-12)).item())
        for basis in link_loads
    }
    edge_rows: list[dict] = []
    for edge_id, (source, target) in enumerate(edge_names):
        row = {
            "edge_id": edge_id,
            "source": source,
            "target": target,
            "capacity": float(capacities[edge_id].item()),
        }
        for basis in ("esm", "actual_request", "admitted"):
            for class_name in (*CLASSES, "All"):
                key = class_name.lower()
                load = float(link_loads[basis][class_name][edge_id].item())
                row[f"{basis}_{key}_load"] = load
                row[f"{basis}_{key}_utilization"] = load / max(row["capacity"], 1e-12)
            row[f"{basis}_all_bottleneck"] = int(edge_id == bottlenecks[basis])
        edge_rows.append(row)
    write_csv(detail_dir / "topology_edges_with_loads.csv", edge_rows)

    node_rows = []
    for internal_index, node in enumerate(node_ids):
        outgoing = [row for row in edge_rows if row["source"] == node]
        incoming = [row for row in edge_rows if row["target"] == node]
        node_rows.append(
            {
                "node": node,
                "internal_index": internal_index,
                "in_degree": snapshot.graph.in_degree(node),
                "out_degree": snapshot.graph.out_degree(node),
                "incoming_capacity": sum(float(row["capacity"]) for row in incoming),
                "outgoing_capacity": sum(float(row["capacity"]) for row in outgoing),
            }
        )
    write_csv(detail_dir / "topology_nodes.csv", node_rows)

    class_summary = []
    for class_index, class_name in enumerate(CLASSES):
        saved = candidate_metrics[(snapshot_id, class_name)]
        baseline = baseline_metrics[(snapshot_id, class_name)]
        oracle = float(oracle_totals[class_index].reshape(-1)[0].item())
        admitted_total = float(admitted[class_index].sum().item())
        class_summary.append(
            {
                "class": class_name,
                "actual_demand": float(actual_tms[class_index].sum().item() / k),
                "esm_predicted_demand": float(predicted_tms[class_index].sum().item() / k),
                "system_admitted": admitted_total,
                "oracle_admitted": oracle,
                "shortfall_vs_oracle": oracle - admitted_total,
                "norm_fulfill": admitted_total / max(oracle, 1e-12),
                "baseline_norm_fulfill": float(baseline["norm_fulfill"]),
                "candidate_delta_vs_baseline": float(saved["norm_fulfill"]) - float(baseline["norm_fulfill"]),
            }
        )
    write_csv(detail_dir / "class_summary.csv", class_summary)

    layout = nx.spring_layout(snapshot.graph.to_undirected(), seed=490, iterations=300)
    visual_nodes = [
        {"id": int(node), "x": float(layout[node][0]), "y": float(layout[node][1])}
        for node in node_ids
    ]
    visual_links = [
        {
            "id": int(row["edge_id"]),
            "source": int(row["source"]),
            "target": int(row["target"]),
            "capacity": float(row["capacity"]),
            "esm": float(row["esm_all_utilization"]),
            "actual_request": float(row["actual_request_all_utilization"]),
            "admitted": float(row["admitted_all_utilization"]),
        }
        for row in edge_rows
    ]
    detail = {
        "snapshot": snapshot_id,
        "topology": str(getattr(snapshot, "topo", "")),
        "nodes": len(node_ids),
        "directed_edges": len(edge_names),
        "od_pairs": len(pair_keys),
        "paths_per_pair": k,
        "class_summary": class_summary,
        "bottlenecks": {
            basis: {
                "edge_id": edge_id,
                "source": edge_names[edge_id][0],
                "target": edge_names[edge_id][1],
                "utilization": float((link_loads[basis]["All"][edge_id] / capacities[edge_id]).item()),
            }
            for basis, edge_id in bottlenecks.items()
        },
        "visualization": {"nodes": visual_nodes, "links": visual_links},
    }
    write_json(detail_dir / "snapshot_detail.json", detail)
    write_snapshot_report(detail_dir / "README.md", detail)
    write_static_plots(
        detail_dir,
        snapshot.graph,
        layout,
        edge_rows,
        node_ids,
        pair_keys,
        matrix_data,
    )
    if visualization_path is not None:
        write_topology_visualization(visualization_path, detail)
    return {"directory": str(detail_dir), **detail}


def write_snapshot_report(path: Path, detail: dict) -> None:
    medium = next(row for row in detail["class_summary"] if row["class"] == "Medium")
    low = next(row for row in detail["class_summary"] if row["class"] == "Low")
    lines = [
        f"# Snapshot {detail['snapshot']} 完整输入与系统预测",
        "",
        f"拓扑包含 {detail['nodes']} 个节点、{detail['directed_edges']} 条有向链路、"
        f"{detail['od_pairs']} 个 OD 对，每个 OD 有 {detail['paths_per_pair']} 条 shared 候选路径。",
        "",
        "| Class | Actual demand | ESM prediction | System admitted | Oracle admitted | Shortfall | NormFulFill |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in detail["class_summary"]:
        lines.append(
            f"| {row['class']} | {row['actual_demand']:.6f} | {row['esm_predicted_demand']:.6f} | "
            f"{row['system_admitted']:.6f} | {row['oracle_admitted']:.6f} | "
            f"{row['shortfall_vs_oracle']:+.6f} | {row['norm_fulfill']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"Medium 相对 oracle 少 {medium['shortfall_vs_oracle']:.6f}，即 "
            f"{medium['shortfall_vs_oracle'] / medium['oracle_admitted']:.2%}；"
            f"Low 相对 oracle 的差为 {low['shortfall_vs_oracle']:+.6f}。",
            "",
            "## 数据文件",
            "",
            "- `topology_nodes.csv`：节点、入/出度和容量",
            "- `topology_edges_with_loads.csv`：全部有向链路及 ESM/actual-request/admitted 的分类负载与利用率",
            "- `traffic_matrix_actual_{high,medium,low}.csv`：actual 流量矩阵",
            "- `traffic_matrix_esm_{high,medium,low}.csv`：ESM 预测流量矩阵",
            "- `traffic_matrix_error_{high,medium,low}.csv`：预测减 actual 的误差矩阵",
            "- `traffic_matrix_long.csv`：逐 OD 输入、预测、准入和首选路径",
            "- `system_path_predictions.csv`：逐 OD、逐 class、逐路径的策略份额和准入流量",
            "- `snapshot_detail.json`：机器可读摘要与可视化数据",
            "- `topology_load_comparison.png`：预测请求、actual 请求和最终准入的拓扑负载对比",
            "- `traffic_matrices.png`：三优先级 actual、ESM prediction 和误差矩阵热图",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_topology_visualization(path: Path, detail: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(detail["visualization"], ensure_ascii=False, separators=(",", ":"))
    fragment = f'''<div id="snapshot-205-topology">
  <h2>Snapshot 205 · GEANT shared-2x topology</h2>
  <div class="viz-controls" role="group" aria-label="Load basis">
    <button type="button" class="btn btn-ghost" data-basis="esm" aria-pressed="true">ESM predicted request</button>
    <button type="button" class="btn btn-ghost" data-basis="actual_request" aria-pressed="false">Actual request</button>
    <button type="button" class="btn btn-ghost" data-basis="admitted" aria-pressed="false">Actual admitted</button>
  </div>
  <div class="viz-row text-small" aria-label="Legend">
    <span><svg width="26" height="8" aria-hidden="true"><line x1="1" y1="4" x2="25" y2="4" stroke="var(--muted-foreground)" stroke-width="2"/></svg> utilization &lt; 0.8</span>
    <span><svg width="26" height="8" aria-hidden="true"><line x1="1" y1="4" x2="25" y2="4" stroke="var(--viz-series-2)" stroke-width="3"/></svg> 0.8–1.0</span>
    <span><svg width="26" height="8" aria-hidden="true"><line x1="1" y1="4" x2="25" y2="4" stroke="var(--viz-series-4)" stroke-width="4"/></svg> &gt; 1.0</span>
  </div>
  <div class="topology-plot"></div>
  <div class="text-small topology-summary" aria-live="polite"></div>
</div>
<style>
  #snapshot-205-topology {{ width: 100%; color: var(--foreground); }}
  #snapshot-205-topology .topology-plot {{ width: 100%; min-height: 500px; }}
  #snapshot-205-topology .topology-link {{ fill: none; opacity: .72; }}
  #snapshot-205-topology .topology-node {{ fill: var(--background); stroke: var(--foreground); stroke-width: 1.5; }}
  #snapshot-205-topology .topology-label {{ fill: var(--foreground); font-size: 12px; text-anchor: middle; dominant-baseline: central; pointer-events: none; }}
  #snapshot-205-topology .topology-summary {{ margin-top: 6px; color: var(--muted-foreground); }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById("snapshot-205-topology");
  const data = {data};
  const plot = root.querySelector(".topology-plot");
  const summary = root.querySelector(".topology-summary");
  let basis = "esm";
  function draw() {{
    plot.replaceChildren();
    const width = Math.max(320, plot.getBoundingClientRect().width);
    const height = Math.max(500, Math.min(650, width * 0.78));
    const svg = d3.select(plot).append("svg")
      .attr("viewBox", `0 0 ${{width}} ${{height}}`)
      .attr("width", "100%")
      .attr("height", height)
      .attr("role", "img")
      .attr("aria-label", "GEANT directed topology. Link thickness and color encode selected utilization.");
    svg.append("title").text("Snapshot 205 GEANT topology and link utilization");
    svg.append("desc").text("Twenty-two nodes and seventy-two directed links. Select predicted, requested, or admitted traffic.");
    const xExtent = d3.extent(data.nodes, d => d.x);
    const yExtent = d3.extent(data.nodes, d => d.y);
    const x = d3.scaleLinear().domain(xExtent).range([42, width - 42]);
    const y = d3.scaleLinear().domain(yExtent).range([42, height - 42]);
    const nodeById = new Map(data.nodes.map(d => [d.id, d]));
    const color = value => value > 1.0001 ? "var(--viz-series-4)" : value >= 0.8 ? "var(--viz-series-2)" : "var(--muted-foreground)";
    const curve = d => {{
      const s = nodeById.get(d.source), t = nodeById.get(d.target);
      const sx = x(s.x), sy = y(s.y), tx = x(t.x), ty = y(t.y);
      const dx = tx - sx, dy = ty - sy, length = Math.max(Math.hypot(dx, dy), 1);
      const offset = 7;
      const ox = -dy / length * offset, oy = dx / length * offset;
      return `M${{sx + ox}},${{sy + oy}} L${{tx + ox}},${{ty + oy}}`;
    }};
    const links = svg.append("g").selectAll("path").data(data.links).join("path")
      .attr("class", "topology-link")
      .attr("d", curve)
      .attr("stroke", d => color(d[basis]))
      .attr("stroke-width", d => 1 + Math.min(d[basis], 4) * 1.35)
      .attr("data-tooltip", d => `${{d.source}}→${{d.target}} · utilization ${{d[basis].toFixed(3)}} · capacity ${{d.capacity.toFixed(1)}}`);
    links.append("title").text(d => `${{d.source}}→${{d.target}}: utilization ${{d[basis].toFixed(3)}}`);
    const nodes = svg.append("g").selectAll("g").data(data.nodes).join("g")
      .attr("transform", d => `translate(${{x(d.x)}},${{y(d.y)}})`);
    nodes.append("circle").attr("class", "topology-node").attr("r", 13);
    nodes.append("text").attr("class", "topology-label").text(d => d.id);
    const maxLink = data.links.reduce((a, b) => a[basis] > b[basis] ? a : b);
    summary.textContent = `Maximum: ${{maxLink.source}}→${{maxLink.target}}, utilization ${{maxLink[basis].toFixed(3)}}. Link width is capped visually at utilization 4.`;
  }}
  root.querySelectorAll("button[data-basis]").forEach(button => button.addEventListener("click", () => {{
    basis = button.dataset.basis;
    root.querySelectorAll("button[data-basis]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
    draw();
  }}));
  new ResizeObserver(draw).observe(plot);
  draw();
}})();
</script>'''
    path.write_text(fragment, encoding="utf-8")


def write_static_plots(
    detail_dir: Path,
    graph,
    layout: dict,
    edge_rows: list[dict],
    node_ids: list,
    pair_keys: list[tuple],
    matrix_data: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        # The experiment venv intentionally contains only training dependencies.
        # plot_snapshot_detail.py can render the exported CSV/JSON with a plotting venv.
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    edge_list = [(row["source"], row["target"]) for row in edge_rows]
    panels = (
        ("esm", "ESM predicted request"),
        ("actual_request", "Actual request"),
        ("admitted", "Actual admitted"),
    )
    max_utilization = max(
        float(row[f"{basis}_all_utilization"])
        for row in edge_rows
        for basis, _label in panels
    )
    norm = Normalize(vmin=0.0, vmax=max_utilization)
    cmap = plt.get_cmap("viridis")
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, (basis, label) in zip(axes, panels):
        values = [float(row[f"{basis}_all_utilization"]) for row in edge_rows]
        max_edge = max(edge_rows, key=lambda row: float(row[f"{basis}_all_utilization"]))
        nx.draw_networkx_nodes(
            graph,
            layout,
            ax=axis,
            node_size=330,
            node_color="#f4f4f4",
            edgecolors="#222222",
            linewidths=0.8,
        )
        nx.draw_networkx_labels(graph, layout, ax=axis, font_size=7)
        nx.draw_networkx_edges(
            graph,
            layout,
            ax=axis,
            edgelist=edge_list,
            edge_color=values,
            edge_cmap=cmap,
            edge_vmin=0.0,
            edge_vmax=max_utilization,
            width=[0.5 + min(value, 4.0) * 0.7 for value in values],
            arrows=True,
            arrowsize=7,
            connectionstyle="arc3,rad=0.07",
            alpha=0.82,
        )
        axis.set_title(
            f"{label}\nmax {max_edge['source']}→{max_edge['target']} = "
            f"{float(max_edge[f'{basis}_all_utilization']):.3f}"
        )
        axis.set_axis_off()
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    figure.colorbar(scalar, ax=axes, shrink=0.75, label="Total link utilization")
    figure.suptitle("Snapshot 205 · GEANT shared-2x directed topology", fontsize=14)
    figure.savefig(detail_dir / "topology_load_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    node_index = {node: index for index, node in enumerate(node_ids)}
    matrix_figure, matrix_axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    for row_index, class_name in enumerate(CLASSES):
        actual_od, predicted_od = matrix_data[class_name]
        actual_matrix = np.zeros((len(node_ids), len(node_ids)), dtype=np.float64)
        predicted_matrix = np.zeros_like(actual_matrix)
        for pair_index, (source, target) in enumerate(pair_keys):
            actual_matrix[node_index[source], node_index[target]] = actual_od[pair_index]
            predicted_matrix[node_index[source], node_index[target]] = predicted_od[pair_index]
        error_matrix = predicted_matrix - actual_matrix
        common_max = max(float(actual_matrix.max()), float(predicted_matrix.max()), 1e-12)
        error_max = max(float(np.abs(error_matrix).max()), 1e-12)
        for column_index, (title, matrix, color_map, matrix_norm) in enumerate(
            (
                ("Actual", actual_matrix, "viridis", Normalize(0.0, common_max)),
                ("ESM prediction", predicted_matrix, "viridis", Normalize(0.0, common_max)),
                ("Prediction − actual", error_matrix, "coolwarm", TwoSlopeNorm(vmin=-error_max, vcenter=0.0, vmax=error_max)),
            )
        ):
            axis = matrix_axes[row_index, column_index]
            image = axis.imshow(matrix, cmap=color_map, norm=matrix_norm, interpolation="nearest")
            axis.set_title(f"{class_name} · {title}")
            axis.set_xlabel("Destination node")
            axis.set_ylabel("Source node")
            axis.set_xticks(range(len(node_ids)), labels=node_ids, fontsize=6, rotation=90)
            axis.set_yticks(range(len(node_ids)), labels=node_ids, fontsize=6)
            matrix_figure.colorbar(image, ax=axis, shrink=0.72)
    matrix_figure.suptitle("Snapshot 205 traffic matrices (flow units)", fontsize=14)
    matrix_figure.savefig(detail_dir / "traffic_matrices.png", dpi=180, bbox_inches="tight")
    plt.close(matrix_figure)


def diagnose(args: argparse.Namespace) -> dict:
    candidate_dir = args.candidate_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_metrics = metric_index(read_csv(candidate_dir / "best_evaluation_metrics.csv"))
    baseline_metrics = metric_index(read_csv(baseline_dir / "best_evaluation_metrics.csv"))
    selected = select_bad_samples(candidate_metrics, baseline_metrics)
    selected_ids = {int(row["snapshot"]) for row in selected}
    reasons = {int(row["snapshot"]): row["selection_reason"] for row in selected}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    core.set_seed(args.seed)
    props = core.build_props(2, device)
    props.batch_size = 1
    model = core.load_frozen_ensemble(props, args.seed, device)
    checkpoint_path = candidate_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.assert_teacher_immutable()

    eval_start, eval_end = core.LEVELS[2]["evaluation"]
    dataset = core.DM_Dataset_within_Cluster(props, 0, eval_start, eval_end)
    path_masks = core.base.move_dataset_static(dataset, device)
    loader = core.round1.data_loader(dataset, 1, False, args.seed)
    core.clear_transformer_caches(model)

    prediction_rows: list[dict] = []
    top_od_rows: list[dict] = []
    bottleneck_rows: list[dict] = []
    sample_rows: list[dict] = []
    replay_checks: list[dict] = []
    snapshot_detail: dict | None = None
    k = int(props.num_paths_per_pair)
    pair_keys = list(dataset.pij.keys())

    for local_index, inputs in enumerate(loader):
        snapshot_id = eval_start + local_index
        if snapshot_id not in selected_ids:
            continue
        values = core.base.unpack_to_device(inputs, props)
        snapshot = values[14][0]
        capacities = values[1][:1].reshape(-1).to(dtype=torch.float32)
        actual_tms = (values[2], values[4], values[6])
        predicted_tms = (values[3], values[5], values[7])
        oracle_totals = (
            values[11],
            values[12] - values[11],
            values[13] - values[12],
        )

        props.mode = "test"
        props.sim_mf_mlu = 0
        props.research_return_policy = True
        with torch.no_grad():
            policies, _ = core.model_forward(model, props, dataset, values, path_masks)
        props.research_return_policy = False
        props.sim_mf_mlu = 1
        with torch.no_grad():
            admitted, _ = core.model_forward(model, props, dataset, values, path_masks)
        props.sim_mf_mlu = 0

        if snapshot_id == args.detail_snapshot:
            snapshot_detail = export_snapshot_detail(
                snapshot_id,
                snapshot,
                dataset,
                values,
                policies,
                admitted,
                candidate_metrics,
                baseline_metrics,
                output_dir,
                k,
                args.visualization_path.resolve() if args.visualization_path else None,
            )

        edge_names = list(snapshot.graph.edges())
        cumulative_predicted = torch.zeros_like(actual_tms[0].squeeze(0).squeeze(-1))
        cumulative_requested = torch.zeros_like(cumulative_predicted)
        cumulative_admitted = torch.zeros_like(cumulative_predicted)

        for class_index, class_name in enumerate(CLASSES):
            actual_od = (
                actual_tms[class_index]
                .detach()
                .reshape(-1, k)[:, 0]
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
            predicted_od = (
                predicted_tms[class_index]
                .detach()
                .reshape(-1, k)[:, 0]
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
            policy = (
                policies[class_index]
                .detach()
                .reshape(-1, k)
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
            admitted_matrix = (
                admitted[class_index]
                .detach()
                .reshape(-1, k)
                .to(dtype=torch.float32)
                .cpu()
                .numpy()
            )
            actual_total = float(actual_od.sum())
            predicted_total = float(predicted_od.sum())
            admitted_total = float(admitted_matrix.sum())
            oracle_total = float(oracle_totals[class_index].detach().reshape(-1)[0].item())
            positive = actual_od > 1e-12
            entropy, mean_top_share, max_top_share = policy_statistics(policy, positive)
            absolute_error = np.abs(predicted_od - actual_od)
            wape = float(absolute_error.sum() / max(actual_total, 1e-12))
            prediction_rows.append(
                {
                    "snapshot": snapshot_id,
                    "selection_reason": reasons[snapshot_id],
                    "class": class_name,
                    "actual_demand": actual_total,
                    "esm_predicted_demand": predicted_total,
                    "prediction_bias": (predicted_total - actual_total) / max(actual_total, 1e-12),
                    "prediction_wape": wape,
                    "od_prediction_correlation": finite_or_none(safe_correlation(actual_od, predicted_od)),
                    "system_admitted": admitted_total,
                    "oracle_admitted": oracle_total,
                    "norm_fulfill_replay": admitted_total / max(oracle_total, 1e-12),
                    "raw_fulfill_replay": admitted_total / max(actual_total, 1e-12),
                    "mean_normalized_policy_entropy": entropy,
                    "mean_top1_path_share": mean_top_share,
                    "max_top1_path_share": max_top_share,
                }
            )

            for pair_index in np.argsort(-absolute_error)[:3].tolist():
                source, target = pair_keys[pair_index]
                top_path = int(np.argmax(policy[pair_index]))
                top_od_rows.append(
                    {
                        "snapshot": snapshot_id,
                        "class": class_name,
                        "error_rank": len([r for r in top_od_rows if r["snapshot"] == snapshot_id and r["class"] == class_name]) + 1,
                        "pair_index": pair_index,
                        "source": source,
                        "target": target,
                        "actual_demand": float(actual_od[pair_index]),
                        "esm_predicted_demand": float(predicted_od[pair_index]),
                        "signed_prediction_error": float(predicted_od[pair_index] - actual_od[pair_index]),
                        "system_admitted": float(admitted_matrix[pair_index].sum()),
                        "predicted_top_path_index": top_path,
                        "predicted_top_path_share": float(policy[pair_index, top_path]),
                        "predicted_top_path": path_text(dataset.pij[(source, target)][top_path]),
                    }
                )

            policy_flat = torch.as_tensor(policy, device=device).reshape(-1)
            cumulative_predicted = cumulative_predicted + policy_flat * predicted_tms[class_index].squeeze(0).squeeze(-1)
            cumulative_requested = cumulative_requested + policy_flat * actual_tms[class_index].squeeze(0).squeeze(-1)
            cumulative_admitted = cumulative_admitted + admitted[class_index].squeeze(0)
            stage = ("High", "High+Medium", "All")[class_index]
            for basis, path_flow in (
                ("ESM预测需求下的计划负载", cumulative_predicted),
                ("actual需求下的请求负载（准入前）", cumulative_requested),
                ("actual需求下的准入负载", cumulative_admitted),
            ):
                loads = tensor_link_load(dataset.pte, path_flow)
                ratios = loads / capacities.clamp_min(torch.finfo(torch.float32).tiny)
                edge_id = int(torch.argmax(ratios).item())
                source, target = edge_names[edge_id]
                bottleneck_rows.append(
                    {
                        "snapshot": snapshot_id,
                        "stage": stage,
                        "basis": basis,
                        "edge_id": edge_id,
                        "source": source,
                        "target": target,
                        "load": float(loads[edge_id].item()),
                        "capacity": float(capacities[edge_id].item()),
                        "utilization": float(ratios[edge_id].item()),
                    }
                )

            saved = candidate_metrics[(snapshot_id, class_name)]
            replay_checks.append(
                {
                    "snapshot": snapshot_id,
                    "class": class_name,
                    "saved_norm_fulfill": float(saved["norm_fulfill"]),
                    "replay_norm_fulfill": admitted_total / max(oracle_total, 1e-12),
                    "absolute_error": abs(float(saved["norm_fulfill"]) - admitted_total / max(oracle_total, 1e-12)),
                }
            )

        candidate_medium = candidate_metrics[(snapshot_id, "Medium")]
        candidate_low = candidate_metrics[(snapshot_id, "Low")]
        baseline_medium = baseline_metrics[(snapshot_id, "Medium")]
        sample_rows.append(
            {
                "snapshot": snapshot_id,
                "selection_reason": reasons[snapshot_id],
                "candidate_medium_norm_fulfill": float(candidate_medium["norm_fulfill"]),
                "candidate_low_norm_fulfill": float(candidate_low["norm_fulfill"]),
                "inversion_gap_low_minus_medium": float(candidate_low["norm_fulfill"]) - float(candidate_medium["norm_fulfill"]),
                "baseline_medium_norm_fulfill": float(baseline_medium["norm_fulfill"]),
                "candidate_medium_delta_vs_baseline": float(candidate_medium["norm_fulfill"]) - float(baseline_medium["norm_fulfill"]),
            }
        )

    if {row["snapshot"] for row in sample_rows} != selected_ids:
        raise RuntimeError("Not all selected snapshots were replayed")
    max_replay_error = max(float(row["absolute_error"]) for row in replay_checks)
    if max_replay_error > 2e-5:
        raise RuntimeError(f"Checkpoint replay disagrees with saved metrics: {max_replay_error}")

    sample_rows.sort(key=lambda row: selected_ids_order(selected, int(row["snapshot"])))
    prediction_rows.sort(key=lambda row: (selected_ids_order(selected, int(row["snapshot"])), CLASSES.index(row["class"])))
    top_od_rows.sort(key=lambda row: (selected_ids_order(selected, int(row["snapshot"])), CLASSES.index(row["class"]), int(row["error_rank"])))
    bottleneck_rows.sort(key=lambda row: (selected_ids_order(selected, int(row["snapshot"])), ("High", "High+Medium", "All").index(row["stage"]), row["basis"]))

    write_csv(output_dir / "selected_bad_samples.csv", sample_rows)
    write_csv(output_dir / "system_predictions.csv", prediction_rows)
    write_csv(output_dir / "top_od_prediction_errors.csv", top_od_rows)
    write_csv(output_dir / "bottleneck_links.csv", bottleneck_rows)
    write_csv(output_dir / "checkpoint_replay_checks.csv", replay_checks)

    result = {
        "scope": "Level-2 proxy snapshots 200-249; already viewed model-selection window, not blind test",
        "model": "two-phase frozen-High + tail_flow_balanced_squared, multiplier 0.25",
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "selected_samples": sample_rows,
        "system_predictions": prediction_rows,
        "top_od_prediction_errors": top_od_rows,
        "bottleneck_links": bottleneck_rows,
        "max_checkpoint_replay_error": max_replay_error,
        "snapshot_detail": snapshot_detail,
    }
    write_json(output_dir / "diagnostics.json", result)
    write_report(output_dir / "坏样本诊断.md", result)
    return result


def selected_ids_order(selected: list[dict], snapshot: int) -> int:
    return [int(row["snapshot"]) for row in selected].index(snapshot)


def write_report(path: Path, result: dict) -> None:
    predictions = {
        (int(row["snapshot"]), row["class"]): row for row in result["system_predictions"]
    }
    bottlenecks = {
        (int(row["snapshot"]), row["stage"], row["basis"]): row
        for row in result["bottleneck_links"]
    }
    lines = [
        "# Two-phase Level-2 坏样本与系统预测",
        "",
        f"- 范围：{result['scope']}",
        f"- 模型：{result['model']}，checkpoint epoch {result['checkpoint_epoch']}",
        f"- checkpoint 重放最大绝对误差：{result['max_checkpoint_replay_error']:.3e}",
        "",
        "## 样本总览",
        "",
        "| 快照 | 选取原因 | Medium NFF | Low NFF | Low-Medium gap | 相对 baseline 的 Medium 变化 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in result["selected_samples"]:
        lines.append(
            f"| {row['snapshot']} | {row['selection_reason']} | "
            f"{row['candidate_medium_norm_fulfill']:.4f} | {row['candidate_low_norm_fulfill']:.4f} | "
            f"{row['inversion_gap_low_minus_medium']:+.4f} | {row['candidate_medium_delta_vs_baseline']:+.4f} |"
        )
    lines.extend(["", "## ESM 流量预测与系统准入", ""])
    for sample in result["selected_samples"]:
        snapshot = int(sample["snapshot"])
        lines.extend(
            [
                f"### Snapshot {snapshot}",
                "",
                "| 类别 | actual demand | ESM predicted | bias | WAPE | 系统准入 / oracle | NFF | 平均 top-1 路径份额 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for class_name in CLASSES:
            row = predictions[(snapshot, class_name)]
            lines.append(
                f"| {class_name} | {row['actual_demand']:.3f} | {row['esm_predicted_demand']:.3f} | "
                f"{row['prediction_bias']:+.1%} | {row['prediction_wape']:.1%} | "
                f"{row['system_admitted']:.3f} / {row['oracle_admitted']:.3f} | "
                f"{row['norm_fulfill_replay']:.4f} | {row['mean_top1_path_share']:.1%} |"
            )
        predicted_bn = bottlenecks[(snapshot, "All", "ESM预测需求下的计划负载")]
        actual_bn = bottlenecks[(snapshot, "All", "actual需求下的准入负载")]
        requested_bn = bottlenecks[(snapshot, "All", "actual需求下的请求负载（准入前）")]
        lines.extend(
            [
                "",
                f"系统按 ESM 预测看到的最终瓶颈是 `{predicted_bn['source']}->{predicted_bn['target']}` "
                f"(计划利用率 {predicted_bn['utilization']:.3f})；actual 请求下准入前瓶颈是 "
                f"`{requested_bn['source']}->{requested_bn['target']}` ({requested_bn['utilization']:.3f})；"
                f"顺序准入后的瓶颈是 `{actual_bn['source']}->{actual_bn['target']}` ({actual_bn['utilization']:.6f})。",
                "",
            ]
        )
    lines.extend(
        [
            "## 文件",
            "",
            "- `selected_bad_samples.csv`：选样依据与基线差异",
            "- `system_predictions.csv`：actual/ESM demand、系统准入、oracle、NFF 与策略集中度",
            "- `top_od_prediction_errors.csv`：每类绝对预测误差最大的 3 个 OD 及系统首选路径",
            "- `bottleneck_links.csv`：预测、actual 请求、actual 准入三种口径的逐阶段瓶颈",
            "- `diagnostics.json`：完整机器可读结果",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay representative bad Level-2 samples")
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--detail-snapshot", type=int, default=205)
    parser.add_argument("--visualization-path", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    output = diagnose(parse_cli())
    print(json.dumps({
        "selected": [row["snapshot"] for row in output["selected_samples"]],
        "max_checkpoint_replay_error": output["max_checkpoint_replay_error"],
    }, ensure_ascii=False, indent=2))
