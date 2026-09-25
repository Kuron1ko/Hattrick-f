from __future__ import annotations

import csv
import math
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "geant" / "4sp" / "100"
TEST_START = 90
TEST_END = 100
CLASSES = ("High", "Medium", "Low")
METHODS = ("Hattrick", "BEST_MC", "SWAN")


def read_floats(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                values.append(float(stripped))
    return np.asarray(values, dtype=np.float64)


def read_demands() -> np.ndarray:
    manifest = ROOT / "manifest" / "geant_manifest.txt"
    demands = []
    with manifest.open("r", encoding="utf-8") as handle:
        rows = [line.strip().split(",") for line in handle if line.strip()]

    for _, _, tm_filename in rows[TEST_START:TEST_END]:
        class_demands = []
        for priority in (1, 2, 3):
            tm_path = ROOT / "traffic_matrices" / f"geant_{priority}" / tm_filename.strip()
            with tm_path.open("rb") as handle:
                tm = pickle.load(handle)
            class_demands.append(float(np.asarray(tm, dtype=np.float64).sum()))
        demands.append(class_demands)
    return np.asarray(demands, dtype=np.float64)


def oracle_fulfill_denominators() -> np.ndarray:
    mf_1 = read_floats(RESULT_DIR / "gt_optimal_values_mf.txt")[TEST_START:TEST_END]
    mf_12 = read_floats(RESULT_DIR / "gt_optimal_values_mf_mf.txt")[TEST_START:TEST_END]
    mf_123 = read_floats(RESULT_DIR / "gt_optimal_values_mf_mf_mf.txt")[TEST_START:TEST_END]
    return np.column_stack((mf_1, mf_12 - mf_1, mf_123 - mf_12))


def oracle_mlu_denominators() -> np.ndarray:
    mlu_1 = read_floats(RESULT_DIR / "gt_optimal_values_mlu.txt")[TEST_START:TEST_END]
    mlu_12 = read_floats(RESULT_DIR / "gt_optimal_values_mlu_mlu.txt")[TEST_START:TEST_END]
    mlu_123 = read_floats(RESULT_DIR / "gt_optimal_values_mlu_mlu_mlu.txt")[TEST_START:TEST_END]
    return np.column_stack((mlu_1, mlu_12, mlu_123))


def split_simulation_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sim = read_floats(path).reshape(-1, 6)[TEST_START:TEST_END]
    mlu = np.column_stack((sim[:, 0], sim[:, 2], sim[:, 4]))
    cumulative_carried = np.column_stack((sim[:, 1], sim[:, 3], sim[:, 5]))
    carried = np.column_stack(
        (
            cumulative_carried[:, 0],
            cumulative_carried[:, 1] - cumulative_carried[:, 0],
            cumulative_carried[:, 2] - cumulative_carried[:, 1],
        )
    )
    return carried, mlu


def build_rows() -> list[dict[str, str | int | float]]:
    demands = read_demands()
    oracle_flow = oracle_fulfill_denominators()
    oracle_mlu = oracle_mlu_denominators()
    rows: list[dict[str, str | int | float]] = []

    hattrick_norm = read_floats(RESULT_DIR / "hattrick_values_esm_sim_mlu_1.txt").reshape(-1, 3)
    hattrick_norm_mlu = read_floats(RESULT_DIR / "hattrick_values_esm_sim_mlu_0.txt").reshape(-1, 3)
    method_data = {
        "Hattrick": (hattrick_norm * oracle_flow, hattrick_norm_mlu * oracle_mlu),
    }
    for method, filename in (
        ("BEST_MC", "flexile_sim_results_esm_mf_mf_mf.txt"),
        ("SWAN", "swan_sim_results_esm_mf_mf_mf.txt"),
    ):
        method_data[method] = split_simulation_file(RESULT_DIR / filename)

    for method in METHODS:
        carried, mlu = method_data[method]
        norm_fulfill = np.divide(carried, oracle_flow, out=np.zeros_like(carried), where=oracle_flow != 0)
        fulfill = np.divide(carried, demands, out=np.zeros_like(carried), where=demands != 0)
        norm_mlu = np.divide(mlu, oracle_mlu, out=np.zeros_like(mlu), where=oracle_mlu != 0)

        for offset in range(TEST_END - TEST_START):
            snapshot_idx = TEST_START + offset
            for class_idx, class_name in enumerate(CLASSES):
                rows.append(
                    {
                        "snapshot_idx": snapshot_idx,
                        "method": method,
                        "class": class_name,
                        "demand": demands[offset, class_idx],
                        "oracle_fulfilled": oracle_flow[offset, class_idx],
                        "carried_traffic": carried[offset, class_idx],
                        "fulfill_ratio": fulfill[offset, class_idx],
                        "norm_fulfill": norm_fulfill[offset, class_idx],
                        "mlu": mlu[offset, class_idx],
                        "norm_mlu": norm_mlu[offset, class_idx],
                    }
                )
    return rows


def write_csv(rows: list[dict[str, str | int | float]]) -> None:
    path = RESULT_DIR / "repro_metrics_100slice.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | float | int]]:
    summary = []
    for method in METHODS:
        for class_name in CLASSES:
            subset = [row for row in rows if row["method"] == method and row["class"] == class_name]
            norm = np.asarray([float(row["norm_fulfill"]) for row in subset])
            fulfill = np.asarray([float(row["fulfill_ratio"]) for row in subset])
            norm_mlu = np.asarray([float(row["norm_mlu"]) for row in subset])
            mlu = np.asarray([float(row["mlu"]) for row in subset])
            summary.append(
                {
                    "method": method,
                    "class": class_name,
                    "n": len(subset),
                    "norm_fulfill_mean": float(norm.mean()),
                    "norm_fulfill_median": float(np.median(norm)),
                    "norm_fulfill_p1": float(np.percentile(norm, 1)),
                    "norm_fulfill_p10": float(np.percentile(norm, 10)),
                    "norm_fulfill_p25": float(np.percentile(norm, 25)),
                    "norm_fulfill_p75": float(np.percentile(norm, 75)),
                    "norm_fulfill_min": float(norm.min()),
                    "norm_fulfill_max": float(norm.max()),
                    "fulfill_ratio_mean": float(fulfill.mean()),
                    "norm_mlu_mean": float(norm_mlu.mean()),
                    "mlu_mean": float(mlu.mean()),
                }
            )

    path = RESULT_DIR / "repro_summary_stats_100slice.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def y_to_px(value: float, y_min: float, y_max: float, plot_top: int, plot_bottom: int) -> int:
    ratio = (value - y_min) / (y_max - y_min)
    return int(plot_bottom - ratio * (plot_bottom - plot_top))


def draw_boxplot(rows: list[dict[str, str | int | float]]) -> None:
    values = {
        (method, class_name): np.asarray(
            [float(row["norm_fulfill"]) for row in rows if row["method"] == method and row["class"] == class_name]
        )
        for class_name in CLASSES
        for method in METHODS
    }
    all_values = np.concatenate(list(values.values()))
    y_min = max(0.0, math.floor((float(all_values.min()) - 0.03) * 20) / 20)
    y_max = max(1.05, math.ceil((float(all_values.max()) + 0.03) * 20) / 20)

    width, height = 1240, 760
    left, right, top, bottom = 95, 60, 92, 620
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28, bold=True)
    label_font = font(18)
    small_font = font(15)
    colors = {"Hattrick": "#2F80ED", "BEST_MC": "#27AE60", "SWAN": "#EB5757"}

    draw.text((left, 28), "GEANT 4sp 100-slice reproduction: NormFulFill on test snapshots 90-99", fill="#202124", font=title_font)
    draw.text((left, 62), "Higher is better. Boxes show p25-p75, center line is median, whiskers show min-max.", fill="#5f6368", font=small_font)

    ticks = np.linspace(y_min, y_max, 6)
    for tick in ticks:
        y = y_to_px(float(tick), y_min, y_max, plot_top, plot_bottom)
        draw.line((plot_left, y, plot_right, y), fill="#E0E0E0", width=1)
        draw.text((20, y - 9), f"{tick:.2f}", fill="#5f6368", font=small_font)

    y_one = y_to_px(1.0, y_min, y_max, plot_top, plot_bottom)
    draw.line((plot_left, y_one, plot_right, y_one), fill="#202124", width=2)
    draw.text((plot_right - 38, y_one - 22), "1.0", fill="#202124", font=small_font)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#202124", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#202124", width=2)

    group_width = (plot_right - plot_left) / len(CLASSES)
    box_width = 56
    for class_idx, class_name in enumerate(CLASSES):
        group_center = plot_left + group_width * (class_idx + 0.5)
        draw.text((group_center - 38, plot_bottom + 28), class_name, fill="#202124", font=label_font)
        offsets = (-74, 0, 74)
        for method, offset in zip(METHODS, offsets):
            vals = values[(method, class_name)]
            q1, median, q3 = np.percentile(vals, [25, 50, 75])
            low, high = vals.min(), vals.max()
            x = int(group_center + offset)
            y_low = y_to_px(float(low), y_min, y_max, plot_top, plot_bottom)
            y_high = y_to_px(float(high), y_min, y_max, plot_top, plot_bottom)
            y_q1 = y_to_px(float(q1), y_min, y_max, plot_top, plot_bottom)
            y_q3 = y_to_px(float(q3), y_min, y_max, plot_top, plot_bottom)
            y_med = y_to_px(float(median), y_min, y_max, plot_top, plot_bottom)
            color = colors[method]

            draw.line((x, y_high, x, y_low), fill=color, width=3)
            draw.line((x - 18, y_high, x + 18, y_high), fill=color, width=3)
            draw.line((x - 18, y_low, x + 18, y_low), fill=color, width=3)
            draw.rectangle((x - box_width // 2, y_q3, x + box_width // 2, y_q1), outline=color, width=3, fill="#FFFFFF")
            draw.line((x - box_width // 2, y_med, x + box_width // 2, y_med), fill=color, width=4)

    legend_x = left
    legend_y = height - 78
    for method in METHODS:
        color = colors[method]
        draw.rectangle((legend_x, legend_y, legend_x + 26, legend_y + 16), fill=color)
        draw.text((legend_x + 34, legend_y - 3), method, fill="#202124", font=label_font)
        legend_x += 170

    image.save(RESULT_DIR / "repro_boxplot_100slice.png")


def draw_cdf_plot(rows: list[dict[str, str | int | float]]) -> None:
    values = {
        (method, class_name): np.sort(
            np.asarray(
                [float(row["norm_fulfill"]) for row in rows if row["method"] == method and row["class"] == class_name]
            )
        )
        for class_name in CLASSES
        for method in METHODS
    }
    x_min = 0.75
    x_max = 1.05

    width, height = 1320, 760
    left, right, top, bottom = 86, 45, 112, 610
    gap = 54
    panel_width = (width - left - right - gap * 2) // len(CLASSES)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28, bold=True)
    label_font = font(18)
    small_font = font(15)
    colors = {"Hattrick": "#2F80ED", "BEST_MC": "#27AE60", "SWAN": "#EB5757"}

    draw.text((left, 30), "GEANT 4sp 100-slice reproduction: CDF of NormFulFill", fill="#202124", font=title_font)
    draw.text((left, 64), "Test snapshots 90-99. Lines are linearly interpolated from empirical CDF points and start at 0.", fill="#5f6368", font=small_font)

    x_ticks = np.arange(x_min, x_max + 0.001, 0.05)
    y_ticks = np.linspace(0.0, 1.0, 6)
    for class_idx, class_name in enumerate(CLASSES):
        panel_left = left + class_idx * (panel_width + gap)
        panel_right = panel_left + panel_width
        panel_top = top
        panel_bottom = bottom

        draw.rectangle((panel_left, panel_top, panel_right, panel_bottom), outline="#202124", width=2)
        draw.text((panel_left + panel_width // 2 - 32, panel_bottom + 30), class_name, fill="#202124", font=label_font)

        for tick in x_ticks:
            x = int(panel_left + ((float(tick) - x_min) / (x_max - x_min)) * panel_width)
            draw.line((x, panel_top, x, panel_bottom), fill="#E8EAED", width=1)
            draw.text((x - 16, panel_bottom + 6), f"{tick:.2f}", fill="#5f6368", font=small_font)
        for tick in y_ticks:
            y = int(panel_bottom - float(tick) * (panel_bottom - panel_top))
            draw.line((panel_left, y, panel_right, y), fill="#E8EAED", width=1)
            if class_idx == 0:
                draw.text((28, y - 9), f"{tick:.1f}", fill="#5f6368", font=small_font)

        for method in METHODS:
            vals = values[(method, class_name)]
            cdf = np.arange(1, len(vals) + 1, dtype=np.float64) / len(vals)
            cdf_x = np.concatenate(([x_min], vals, [x_max]))
            cdf_y = np.concatenate(([0.0], cdf, [1.0]))
            smooth_x = np.linspace(x_min, x_max, 320)
            smooth_y = np.interp(smooth_x, cdf_x, cdf_y)
            points = []
            for value, prob in zip(smooth_x, smooth_y):
                x = int(panel_left + ((float(value) - x_min) / (x_max - x_min)) * panel_width)
                y = int(panel_bottom - float(prob) * (panel_bottom - panel_top))
                points.append((x, y))
            if points:
                draw.line(points, fill=colors[method], width=4)

    draw.text((left + 455, height - 82), "NormFulFill", fill="#202124", font=label_font)
    draw.text((left, top - 28), "CDF", fill="#202124", font=label_font)
    legend_x = left
    legend_y = height - 48
    for method in METHODS:
        color = colors[method]
        draw.line((legend_x, legend_y + 8, legend_x + 32, legend_y + 8), fill=color, width=5)
        draw.text((legend_x + 42, legend_y - 3), method, fill="#202124", font=label_font)
        legend_x += 180

    image.save(RESULT_DIR / "repro_cdf_100slice.png")


def markdown_table(summary: list[dict[str, str | float | int]]) -> str:
    header = "| Method | Class | Mean NormFulFill | P10 | P1 | Mean FulfillRatio | Mean NormMLU |\n"
    sep = "|---|---:|---:|---:|---:|---:|---:|\n"
    lines = [header, sep]
    for row in summary:
        lines.append(
            "| {method} | {class} | {mean:.6f} | {p10:.6f} | {p1:.6f} | {fulfill:.6f} | {nmlu:.6f} |\n".format(
                method=row["method"],
                **{
                    "class": row["class"],
                    "mean": float(row["norm_fulfill_mean"]),
                    "p10": float(row["norm_fulfill_p10"]),
                    "p1": float(row["norm_fulfill_p1"]),
                    "fulfill": float(row["fulfill_ratio_mean"]),
                    "nmlu": float(row["norm_mlu_mean"]),
                },
            )
        )
    return "".join(lines)


def write_report(summary: list[dict[str, str | float | int]]) -> None:
    best = {
        (row["method"], row["class"]): row
        for row in summary
    }
    lines = [
        "# GEANT 100 片第一阶段复现报告\n\n",
        "## 复现目标\n",
        "基于作者开源代码，在 GEANT 数据集上跑通 4sp、前 100 个时间片的 Hattrick 训练/测试流程，"
        "并在测试窗口 90-99 上比较 Hattrick、BEST_MC/Flexile 和 SWAN 的 NormFulFill、FulfillRatio 与 MLU。\n\n",
        "## 环境与范围\n",
        "- Python venv: `D:\\kuroresearch\\.venv-hattrick`\n",
        "- Repo: `D:\\kuroresearch\\Hattrick-main`\n",
        "- 数据: GEANT 三类 traffic matrices 与 ESM 预测矩阵\n",
        "- 训练/验证/测试划分: train 0-80, val 80-90, test 90-100\n",
        "- Gurobi: 使用本机已激活 academic license，在真实 Windows 用户上下文运行\n",
        "- CUDA 状态: 已修复 Hattrick bottleneck edge CUDA 索引逻辑；但本机 NVIDIA driver 546.32 与当前 cu124 wheel 不兼容，训练/测试用 CPU 完成\n\n",
        "## 关键结果\n",
        markdown_table(summary),
        "\n",
        "## 结果文件\n",
        "- `repro_metrics_100slice.csv`: 每个 snapshot、方法、类别的明细指标\n",
        "- `repro_summary_stats_100slice.csv`: 方法/类别汇总统计\n",
        "- `repro_boxplot_100slice.png`: NormFulFill 箱线图\n\n",
        "- `repro_cdf_100slice.png`: NormFulFill CDF 图\n\n",
        "## 解读\n",
        "这是一个 100 片子集复现，不等同于论文完整 GEANT 实验规模。"
        "因此它证明了环境、Gurobi 基线、Hattrick 训练/测试和指标整理链路已跑通，但不能直接声称复现了论文图 4 的完整数值。\n",
        "Hattrick 在该 5 epoch CPU 子集上的平均 NormFulFill 为 "
        f"{float(best[('Hattrick', 'High')]['norm_fulfill_mean']):.4f} / "
        f"{float(best[('Hattrick', 'Medium')]['norm_fulfill_mean']):.4f} / "
        f"{float(best[('Hattrick', 'Low')]['norm_fulfill_mean']):.4f}，"
        "后续若要逼近论文图 4，应扩展到论文测试区间、8sp、更多 epoch，并解决 CUDA wheel/驱动兼容问题。\n",
    ]
    (RESULT_DIR / "repro_summary_100slice.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows)
    summary = summarize(rows)
    draw_boxplot(rows)
    draw_cdf_plot(rows)
    write_report(summary)
    print(f"Wrote {RESULT_DIR / 'repro_metrics_100slice.csv'}")
    print(f"Wrote {RESULT_DIR / 'repro_summary_stats_100slice.csv'}")
    print(f"Wrote {RESULT_DIR / 'repro_boxplot_100slice.png'}")
    print(f"Wrote {RESULT_DIR / 'repro_cdf_100slice.png'}")
    print(f"Wrote {RESULT_DIR / 'repro_summary_100slice.md'}")


if __name__ == "__main__":
    main()
