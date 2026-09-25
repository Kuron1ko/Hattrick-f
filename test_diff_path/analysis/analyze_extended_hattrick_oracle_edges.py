from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "analysis" / "hattrick_oracle_edges_extended"
RAW = OUT / "raw"
REGULAR = tuple(range(400, 500, 5))
BAD = (459, 473, 476)
MATERIAL = 0.05


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError("No suitable font found")


def read_rows(snapshot: int) -> list[dict]:
    path = RAW / f"snapshot_{snapshot}" / "oracle_vs_actual_admitted_edges.csv"
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        source = list(csv.DictReader(handle))
    rows = []
    for row in source:
        capacity = float(row["capacity"])
        high_excess = float(row["actual_admitted_high_load"]) - float(row["oracle_high_load"])
        medium_deficit = float(row["oracle_medium_load"]) - float(row["actual_admitted_medium_load"])
        rows.append(
            {
                "snapshot": snapshot,
                "edge_id": int(row["edge_id"]),
                "source": int(row["source"]),
                "target": int(row["target"]),
                "edge": f'{row["source"]}→{row["target"]}',
                "capacity": capacity,
                "oracle_high_load": float(row["oracle_high_load"]),
                "hattrick_high_load": float(row["actual_admitted_high_load"]),
                "oracle_medium_load": float(row["oracle_medium_load"]),
                "hattrick_medium_load": float(row["actual_admitted_medium_load"]),
                "high_excess": high_excess,
                "high_util_excess": high_excess / capacity,
                "medium_deficit": medium_deficit,
                "medium_util_deficit": medium_deficit / capacity,
            }
        )
    return rows


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize(rows: list[dict], snapshots: tuple[int, ...]) -> tuple[list[dict], list[dict], dict]:
    selected = [row for row in rows if int(row["snapshot"]) in snapshots]
    by_edge: dict[int, list[dict]] = defaultdict(list)
    by_snapshot: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        by_edge[int(row["edge_id"])].append(row)
        by_snapshot[int(row["snapshot"])].append(row)

    edge_summary = []
    for edge_id, group in sorted(by_edge.items()):
        high = np.asarray([row["high_util_excess"] for row in group], dtype=np.float64)
        medium = np.asarray([row["medium_util_deficit"] for row in group], dtype=np.float64)
        over = high > 1e-8
        both = over & (medium > 1e-8)
        material = (high > MATERIAL) & (medium > MATERIAL)
        positive_high = high[over]
        medium_when_over = medium[over]
        shared_pressure = np.minimum(np.maximum(high, 0.0), np.maximum(medium, 0.0))
        edge_summary.append(
            {
                "edge_id": edge_id,
                "source": group[0]["source"],
                "target": group[0]["target"],
                "edge": group[0]["edge"],
                "capacity": group[0]["capacity"],
                "n": len(group),
                "high_overuse_count": int(over.sum()),
                "high_overuse_fraction": float(over.mean()),
                "mean_high_excess_all": float(high.mean()),
                "mean_positive_high_excess": float(positive_high.mean()) if len(positive_high) else 0.0,
                "medium_deficit_when_high_overused_count": int(both.sum()),
                "medium_deficit_when_high_overused_fraction_all": float(both.mean()),
                "medium_deficit_given_high_overuse_fraction": float(both.sum() / max(int(over.sum()), 1)),
                "mean_medium_deficit_when_high_overused": float(medium_when_over.mean()) if len(medium_when_over) else 0.0,
                "material_both_count": int(material.sum()),
                "material_both_fraction": float(material.mean()),
                "mean_shared_pressure": float(shared_pressure.mean()),
            }
        )

    edge_summary.sort(
        key=lambda row: (
            row["material_both_fraction"],
            row["mean_shared_pressure"],
            row["medium_deficit_when_high_overused_fraction_all"],
        ),
        reverse=True,
    )

    snapshot_summary = []
    for snapshot, group in sorted(by_snapshot.items()):
        high = np.asarray([row["high_util_excess"] for row in group], dtype=np.float64)
        medium = np.asarray([row["medium_util_deficit"] for row in group], dtype=np.float64)
        over = high > 1e-8
        both = over & (medium > 1e-8)
        material = (high > MATERIAL) & (medium > MATERIAL)
        snapshot_summary.append(
            {
                "snapshot": snapshot,
                "edges": len(group),
                "high_overuse_edges": int(over.sum()),
                "medium_deficit_on_high_overused_edges": int(both.sum()),
                "conditional_fraction": float(both.sum() / max(int(over.sum()), 1)),
                "material_both_edges": int(material.sum()),
                "pearson_all_edges": corr(high, medium),
            }
        )

    high_all = np.asarray([row["high_util_excess"] for row in selected], dtype=np.float64)
    medium_all = np.asarray([row["medium_util_deficit"] for row in selected], dtype=np.float64)
    over_all = high_all > 1e-8
    both_all = over_all & (medium_all > 1e-8)
    material_all = (high_all > MATERIAL) & (medium_all > MATERIAL)
    cond = np.asarray([row["conditional_fraction"] for row in snapshot_summary])
    correlations = np.asarray([row["pearson_all_edges"] for row in snapshot_summary])
    overall = {
        "snapshot_count": len(snapshots),
        "edge_snapshot_count": len(selected),
        "edge_count": len(by_edge),
        "pearson_all_edge_snapshots": corr(high_all, medium_all),
        "high_overuse_edge_snapshot_fraction": float(over_all.mean()),
        "medium_deficit_given_high_overuse_fraction": float(both_all.sum() / max(int(over_all.sum()), 1)),
        "material_both_edge_snapshot_fraction": float(material_all.mean()),
        "per_snapshot_conditional_fraction_mean": float(cond.mean()),
        "per_snapshot_conditional_fraction_min": float(cond.min()),
        "per_snapshot_conditional_fraction_max": float(cond.max()),
        "per_snapshot_pearson_median": float(np.nanmedian(correlations)),
        "per_snapshot_positive_correlation_fraction": float(np.mean(correlations > 0.0)),
        "material_threshold_capacity_fraction": MATERIAL,
    }
    return edge_summary, snapshot_summary, overall


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interpolate_color(value: float, limit: float = 0.45) -> tuple[int, int, int]:
    value = max(-limit, min(limit, value)) / limit
    neutral = np.asarray([244, 244, 244], dtype=np.float64)
    target = np.asarray([202, 52, 55] if value >= 0 else [55, 104, 176], dtype=np.float64)
    rgb = neutral * (1.0 - abs(value)) + target * abs(value)
    return tuple(int(round(channel)) for channel in rgb)


def draw_main_figure(rows: list[dict], top_edges: list[dict], overall: dict) -> None:
    width, height = 3840, 2160
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = font(88, True)
    subtitle = font(42)
    panel_title = font(50, True)
    label = font(36)
    small = font(30)
    tiny = font(25)
    bold_small = font(32, True)
    black = (25, 25, 25)
    gray = (95, 95, 95)
    light = (218, 218, 218)
    red = (202, 52, 55)
    purple = (113, 70, 155)

    draw.text((150, 80), "Hattrick 在高负载下反复占用对 Medium 关键的边", fill=black, font=title)
    draw.text(
        (150, 190),
        "2×负载 · 严格 ESM 推理 · 20 个等间隔时间片（400–495，每隔5个） · 非坏样本筛选",
        fill=gray,
        font=subtitle,
    )

    # Left: heatmap of High edge-utilization excess, with Medium-deficit markers.
    left_x, left_y = 150, 390
    heat_w, heat_h = 2140, 1280
    draw.text((left_x, 305), "同一批边在多个时间片重复出现 High 超额占用", fill=black, font=panel_title)
    edge_ids = [int(row["edge_id"]) for row in top_edges]
    lookup = {(int(row["snapshot"]), int(row["edge_id"])): row for row in rows}
    row_label_w = 310
    cell_w = (heat_w - row_label_w) / len(REGULAR)
    cell_h = heat_h / len(edge_ids)
    for row_index, edge in enumerate(top_edges):
        y0 = left_y + row_index * cell_h
        draw.text((left_x + 5, y0 + cell_h * 0.28), f'{edge["edge"]}  (E{edge["edge_id"]})', fill=black, font=label)
        for col_index, snapshot in enumerate(REGULAR):
            item = lookup[(snapshot, int(edge["edge_id"]))]
            x0 = left_x + row_label_w + col_index * cell_w
            color = interpolate_color(float(item["high_util_excess"]))
            draw.rectangle((x0, y0, x0 + cell_w - 3, y0 + cell_h - 3), fill=color)
            if float(item["medium_util_deficit"]) > MATERIAL:
                radius = min(cell_w, cell_h) * 0.16
                cx, cy = x0 + cell_w / 2, y0 + cell_h / 2
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=black)
    for col_index, snapshot in enumerate(REGULAR):
        x = left_x + row_label_w + (col_index + 0.5) * cell_w
        draw.text((x - 28, left_y + heat_h + 18), str(snapshot), fill=black, font=tiny)
    draw.text((left_x + row_label_w + 650, left_y + heat_h + 72), "时间片", fill=black, font=small)
    draw.line((left_x + row_label_w, left_y + heat_h + 135, left_x + row_label_w + 320, left_y + heat_h + 135), fill=(55, 104, 176), width=24)
    draw.text((left_x + row_label_w + 350, left_y + heat_h + 110), "High 少于 Oracle", fill=black, font=small)
    draw.line((left_x + row_label_w + 760, left_y + heat_h + 135, left_x + row_label_w + 1080, left_y + heat_h + 135), fill=red, width=24)
    draw.text((left_x + row_label_w + 1110, left_y + heat_h + 110), "High 高于 Oracle", fill=black, font=small)
    draw.ellipse((left_x + row_label_w + 1590, left_y + heat_h + 118, left_x + row_label_w + 1626, left_y + heat_h + 154), fill=black)
    draw.text((left_x + row_label_w + 1650, left_y + heat_h + 110), "同时 Medium 缺口 > 5% 容量", fill=black, font=small)

    # Right: paired average bars on High-overuse snapshots.
    right_x, right_y = 2470, 390
    right_w = 1210
    draw.text((right_x, 305), "重复边的平均影响", fill=black, font=panel_title)
    plot_x = right_x + 300
    bar_max_w = right_w - 360
    max_value = max(
        max(float(row["mean_positive_high_excess"]), float(row["mean_medium_deficit_when_high_overused"]))
        for row in top_edges
    )
    max_value = max(max_value, 0.05)
    for row_index, edge in enumerate(top_edges):
        y = right_y + row_index * 158
        draw.text((right_x, y + 38), edge["edge"], fill=black, font=bold_small)
        high_value = float(edge["mean_positive_high_excess"])
        medium_value = max(0.0, float(edge["mean_medium_deficit_when_high_overused"]))
        high_w = bar_max_w * high_value / max_value
        med_w = bar_max_w * medium_value / max_value
        draw.rectangle((plot_x, y + 18, plot_x + high_w, y + 66), fill=red)
        draw.rectangle((plot_x, y + 78, plot_x + med_w, y + 126), fill=purple)
        draw.text((plot_x + high_w + 14, y + 20), f"{high_value:.0%}", fill=black, font=tiny)
        draw.text((plot_x + med_w + 14, y + 80), f"{medium_value:.0%}", fill=black, font=tiny)
        draw.text(
            (right_x + 8, y + 103),
            f'{edge["material_both_count"]}/20',
            fill=gray,
            font=tiny,
        )
    draw.rectangle((right_x, right_y + 1295, right_x + 45, right_y + 1340), fill=red)
    draw.text((right_x + 65, right_y + 1295), "High 平均超额/容量（仅 High 超额时间片）", fill=black, font=small)
    draw.rectangle((right_x, right_y + 1360, right_x + 45, right_y + 1405), fill=purple)
    draw.text((right_x + 65, right_y + 1360), "Medium 平均缺口/容量（同一批时间片）", fill=black, font=small)
    draw.text((right_x, right_y + 1430), "左侧数字：两者均超过5%容量的时间片数", fill=gray, font=small)

    # Bottom evidence line and caveat.
    y = 1900
    draw.line((150, y - 35, 3690, y - 35), fill=light, width=3)
    draw.text(
        (150, y),
        f'全部 1,440 个“边×时间片”样本：High 超额时有 {overall["medium_deficit_given_high_overuse_fraction"]:.0%} 同时出现 Medium 缺口；'
        f'逐边相关系数 r = {overall["pearson_all_edge_snapshots"]:.2f}。',
        fill=black,
        font=label,
    )
    draw.text(
        (150, y + 75),
        "解释边界：Oracle 使用真实流量离线重求且最优路径不唯一；图中规律说明稳定关联，不单独证明因果。",
        fill=gray,
        font=small,
    )
    image.save(OUT / "hattrick_oracle_edge_mechanism_ppt.png", dpi=(200, 200))


def draw_table(top_edges: list[dict], overall: dict) -> None:
    width, height = 3200, 1800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title = font(76, True)
    subtitle = font(36)
    header = font(34, True)
    cell = font(34)
    small = font(29)
    black = (25, 25, 25)
    gray = (100, 100, 100)
    light = (225, 225, 225)
    accent = (202, 52, 55)

    draw.text((120, 65), "重复出现的 High 超额占边及其 Medium 代价", fill=black, font=title)
    draw.text((120, 165), "20 个等间隔时间片；数值均按边容量归一化", fill=gray, font=subtitle)
    columns = [
        ("边", 120, 470),
        ("High超额\n时间片", 470, 900),
        ("High平均超额\n（超额时）", 900, 1450),
        ("High超额时\nMedium也不足", 1450, 2030),
        ("Medium平均缺口\n（同时间片）", 2030, 2600),
        ("两者均>5%\n容量", 2600, 3080),
    ]
    top, row_h = 300, 150
    draw.rectangle((120, top, 3080, top + row_h), fill=(242, 242, 242))
    for text_value, x0, x1 in columns:
        lines = text_value.split("\n")
        for idx, line in enumerate(lines):
            box = draw.textbbox((0, 0), line, font=header)
            tw = box[2] - box[0]
            draw.text(((x0 + x1 - tw) / 2, top + 26 + idx * 48), line, fill=black, font=header)
    for row_index, row in enumerate(top_edges):
        y0 = top + row_h * (row_index + 1)
        if row_index % 2:
            draw.rectangle((120, y0, 3080, y0 + row_h), fill=(249, 249, 249))
        values = [
            f'{row["edge"]}  (E{row["edge_id"]})',
            f'{row["high_overuse_count"]}/20  ({row["high_overuse_fraction"]:.0%})',
            f'{row["mean_positive_high_excess"]:.1%}',
            f'{row["medium_deficit_when_high_overused_count"]}/{row["high_overuse_count"]}  ({row["medium_deficit_given_high_overuse_fraction"]:.0%})',
            f'{row["mean_medium_deficit_when_high_overused"]:.1%}',
            f'{row["material_both_count"]}/20',
        ]
        for (__, x0, x1), value in zip(columns, values):
            box = draw.textbbox((0, 0), value, font=cell)
            tw, th = box[2] - box[0], box[3] - box[1]
            color = accent if x0 == 2600 and int(row["material_both_count"]) >= 8 else black
            draw.text(((x0 + x1 - tw) / 2, y0 + (row_h - th) / 2 - 4), value, fill=color, font=cell)
        draw.line((120, y0 + row_h, 3080, y0 + row_h), fill=light, width=2)
    foot_y = top + row_h * 9 + 60
    draw.text(
        (120, foot_y),
        f'总体：High 超额的边中，{overall["medium_deficit_given_high_overuse_fraction"]:.0%} 同时存在 Medium 缺口；'
        f'两者均超过5%容量的比例为 {overall["material_both_edge_snapshot_fraction"]:.0%}。',
        fill=black,
        font=subtitle,
    )
    draw.text(
        (120, foot_y + 72),
        "注：表格用于展示跨时间片重复性；Oracle 路由存在退化多解，故不把单条边差异解释为唯一因果路径。",
        fill=gray,
        font=small,
    )
    image.save(OUT / "hattrick_oracle_edge_table_ppt.png", dpi=(200, 200))


def write_report(regular_overall: dict, bad_overall: dict, top_edges: list[dict]) -> None:
    lines = [
        "# Hattrick 与 Oracle 的 High 逐边差异：扩展验证",
        "",
        "## 取样",
        "",
        "- 主分析：2×、严格 ESM、独立评估窗 400–499 中每隔 5 个时间片抽取一个，共 20 个。",
        "- 坏样本复核：459、473、476，单独统计，不参与主图选边。",
        "- 每个时间片均重新求解真实流量下的三阶段词典序 max-flow Oracle。",
        "",
        "## 主结果",
        "",
        f'- 20 个时间片、72 条边，共 {regular_overall["edge_snapshot_count"]} 个边-时间片观测。',
        f'- Hattrick High 高于 Oracle 的边中，有 {regular_overall["medium_deficit_given_high_overuse_fraction"]:.1%} 同时出现 Medium 低于 Oracle。',
        f'- High 超额与 Medium 缺口的 pooled Pearson 相关系数为 {regular_overall["pearson_all_edge_snapshots"]:.3f}。',
        f'- {regular_overall["per_snapshot_positive_correlation_fraction"]:.1%} 的时间片内相关系数为正，中位数为 {regular_overall["per_snapshot_pearson_median"]:.3f}。',
        f'- 两者同时超过边容量 5% 的边-时间片占 {regular_overall["material_both_edge_snapshot_fraction"]:.1%}。',
        "",
        "## 最稳定的边",
        "",
        "| Edge | High overuse | Mean High excess | Medium deficit given overuse | Mean Medium deficit | Both >5% |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_edges:
        lines.append(
            f'| {row["edge"]} (E{row["edge_id"]}) | {row["high_overuse_count"]}/20 | '
            f'{row["mean_positive_high_excess"]:.1%} | '
            f'{row["medium_deficit_when_high_overused_count"]}/{row["high_overuse_count"]} | '
            f'{row["mean_medium_deficit_when_high_overused"]:.1%} | {row["material_both_count"]}/20 |'
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            f'- 三个坏样本中的条件共现率为 {bad_overall["medium_deficit_given_high_overuse_fraction"]:.1%}；主分析仍复现该方向，说明不是完全由坏样本筛选产生。',
            "- Oracle 使用真实流量，仅用于离线归因；Hattrick 推理仍严格只使用 ESM。",
            "- 词典序 Oracle 的逐路径最优解不唯一，因此逐边差异是相对一个确定性重求解的稳定关联，而不是唯一必要路径。",
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots = tuple(sorted(set(REGULAR) | set(BAD)))
    rows = [row for snapshot in snapshots for row in read_rows(snapshot)]
    regular_edges, regular_snapshots, regular_overall = summarize(rows, REGULAR)
    bad_edges, bad_snapshots, bad_overall = summarize(rows, BAD)
    top_edges = regular_edges[:8]

    write_csv(OUT / "edge_snapshot_rows.csv", rows)
    write_csv(OUT / "edge_summary_regular20.csv", regular_edges)
    write_csv(OUT / "snapshot_summary_regular20.csv", regular_snapshots)
    write_csv(OUT / "edge_summary_bad3.csv", bad_edges)
    write_csv(OUT / "snapshot_summary_bad3.csv", bad_snapshots)
    payload = {
        "protocol": {
            "load": "2x",
            "inference": "strict ESM",
            "regular_snapshots": list(REGULAR),
            "bad_snapshots_separate": list(BAD),
            "oracle": "ground-truth three-stage lexicographic max-flow re-solve",
            "material_threshold_capacity_fraction": MATERIAL,
        },
        "regular20": regular_overall,
        "bad3": bad_overall,
        "top_edges_regular20": top_edges,
    }
    (OUT / "analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    draw_main_figure(rows, top_edges, regular_overall)
    draw_table(top_edges, regular_overall)
    write_report(regular_overall, bad_overall, top_edges)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
