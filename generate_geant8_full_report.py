from __future__ import annotations

import csv
import math
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "geant" / "8sp" / "0"
TEST_START = 8080
TEST_END = 10773
CLASSES = ("High", "Medium", "Low")
METHODS = ("Hattrick", "BEST_MC", "SWAN")

PAPER_P10 = {
    ("Hattrick", "High"): 1.0000,
    ("Hattrick", "Medium"): 1.0000,
    ("Hattrick", "Low"): 0.9986,
    ("BEST_MC", "High"): 0.9915,
    ("BEST_MC", "Medium"): 0.9652,
    ("BEST_MC", "Low"): 0.9552,
    ("SWAN", "High"): 0.9713,
    ("SWAN", "Medium"): 0.9194,
    ("SWAN", "Low"): 0.9264,
}

PAPER_P1 = {
    ("Hattrick", "High"): 0.9963,
    ("Hattrick", "Medium"): 0.9921,
    ("Hattrick", "Low"): 0.9824,
    ("BEST_MC", "High"): 0.9710,
    ("BEST_MC", "Medium"): 0.9186,
    ("BEST_MC", "Low"): 0.8924,
    ("SWAN", "High"): 0.9445,
    ("SWAN", "Medium"): 0.8320,
    ("SWAN", "Low"): 0.8229,
}


def read_floats(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    values = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                values.append(float(stripped))
    return np.asarray(values, dtype=np.float64)


def read_manifest_rows() -> list[list[str]]:
    manifest = ROOT / "manifest" / "geant_manifest.txt"
    with manifest.open("r", encoding="utf-8") as handle:
        return [line.strip().split(",") for line in handle if line.strip()]


def read_demands() -> np.ndarray:
    demands = []
    for _, _, tm_filename in read_manifest_rows()[TEST_START:TEST_END]:
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

    rows: list[dict[str, str | int | float]] = []
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


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
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
                    "paper_p10": PAPER_P10.get((method, class_name), np.nan),
                    "paper_p1": PAPER_P1.get((method, class_name), np.nan),
                    "delta_p10_vs_paper": float(np.percentile(norm, 10) - PAPER_P10.get((method, class_name), np.nan)),
                    "delta_p1_vs_paper": float(np.percentile(norm, 1) - PAPER_P1.get((method, class_name), np.nan)),
                }
            )
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


def value_to_y(value: float, y_min: float, y_max: float, top: int, bottom: int) -> int:
    return int(bottom - ((value - y_min) / (y_max - y_min)) * (bottom - top))


def value_to_x(value: float, x_min: float, x_max: float, left: int, right: int) -> int:
    value = min(max(value, x_min), x_max)
    return int(left + ((value - x_min) / (x_max - x_min)) * (right - left))


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
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28, bold=True)
    label_font = font(18)
    small_font = font(15)
    colors = {"Hattrick": "#2F80ED", "BEST_MC": "#27AE60", "SWAN": "#EB5757"}

    draw.text((left, 28), "GEANT 8sp full reproduction: NormFulFill boxplot", fill="#202124", font=title_font)

    plot_left, plot_right = left, width - right
    ticks = np.linspace(y_min, y_max, 6)
    for tick in ticks:
        y = value_to_y(float(tick), y_min, y_max, top, bottom)
        draw.line((plot_left, y, plot_right, y), fill="#E0E0E0", width=1)
        draw.text((20, y - 9), f"{tick:.2f}", fill="#5f6368", font=small_font)
    y_one = value_to_y(1.0, y_min, y_max, top, bottom)
    draw.line((plot_left, y_one, plot_right, y_one), fill="#202124", width=2)
    draw.line((plot_left, bottom, plot_right, bottom), fill="#202124", width=2)
    draw.line((plot_left, top, plot_left, bottom), fill="#202124", width=2)

    group_width = (plot_right - plot_left) / len(CLASSES)
    for class_idx, class_name in enumerate(CLASSES):
        group_center = plot_left + group_width * (class_idx + 0.5)
        draw.text((group_center - 38, bottom + 28), class_name, fill="#202124", font=label_font)
        for method, offset in zip(METHODS, (-74, 0, 74)):
            vals = values[(method, class_name)]
            q1, median, q3 = np.percentile(vals, [25, 50, 75])
            low, high = vals.min(), vals.max()
            x = int(group_center + offset)
            y_low = value_to_y(float(low), y_min, y_max, top, bottom)
            y_high = value_to_y(float(high), y_min, y_max, top, bottom)
            y_q1 = value_to_y(float(q1), y_min, y_max, top, bottom)
            y_q3 = value_to_y(float(q3), y_min, y_max, top, bottom)
            y_med = value_to_y(float(median), y_min, y_max, top, bottom)
            color = colors[method]
            draw.line((x, y_high, x, y_low), fill=color, width=3)
            draw.line((x - 18, y_high, x + 18, y_high), fill=color, width=3)
            draw.line((x - 18, y_low, x + 18, y_low), fill=color, width=3)
            draw.rectangle((x - 28, y_q3, x + 28, y_q1), outline=color, width=3, fill="#FFFFFF")
            draw.line((x - 28, y_med, x + 28, y_med), fill=color, width=4)

    legend_x = left
    legend_y = height - 78
    for method in METHODS:
        draw.rectangle((legend_x, legend_y, legend_x + 26, legend_y + 16), fill=colors[method])
        draw.text((legend_x + 34, legend_y - 3), method, fill="#202124", font=label_font)
        legend_x += 170
    image.save(RESULT_DIR / "repro_boxplot_geant8_full.png")


def draw_cdf(rows: list[dict[str, str | int | float]]) -> None:
    values = {
        (method, class_name): np.sort(
            np.asarray(
                [float(row["norm_fulfill"]) for row in rows if row["method"] == method and row["class"] == class_name]
            )
        )
        for class_name in CLASSES
        for method in METHODS
    }
    x_domains = {
        "High": (0.88, 1.00, 0.02, "{:.2f}"),
        "Medium": (0.60, 1.10, 0.10, "{:.1f}"),
        "Low": (0.60, 1.20, 0.20, "{:.1f}"),
    }

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

    draw.text((left, 30), "GEANT 8sp full reproduction: CDF of NormFulFill", fill="#202124", font=title_font)
    draw.text((left, top - 28), "CDF", fill="#202124", font=label_font)

    y_ticks = np.linspace(0.0, 1.0, 6)
    for class_idx, class_name in enumerate(CLASSES):
        x_min, x_max, tick_step, tick_format = x_domains[class_name]
        x_ticks = np.arange(x_min, x_max + tick_step / 2, tick_step)
        panel_left = left + class_idx * (panel_width + gap)
        panel_right = panel_left + panel_width
        draw.rectangle((panel_left, top, panel_right, bottom), outline="#202124", width=2)
        draw.text((panel_left + panel_width // 2 - 32, bottom + 30), class_name, fill="#202124", font=label_font)

        for tick in x_ticks:
            x = value_to_x(float(tick), x_min, x_max, panel_left, panel_right)
            draw.line((x, top, x, bottom), fill="#E8EAED", width=1)
            draw.text((x - 18, bottom + 6), tick_format.format(float(tick)), fill="#5f6368", font=small_font)
        for tick in y_ticks:
            y = int(bottom - float(tick) * (bottom - top))
            draw.line((panel_left, y, panel_right, y), fill="#E8EAED", width=1)
            if class_idx == 0:
                draw.text((28, y - 9), f"{tick:.1f}", fill="#5f6368", font=small_font)

        for method in METHODS:
            vals = values[(method, class_name)]
            cdf = np.arange(1, len(vals) + 1, dtype=np.float64) / len(vals)
            xs = np.concatenate(([x_min], vals, [x_max]))
            ys = np.concatenate(([0.0], cdf, [1.0]))
            points = [
                (
                    value_to_x(float(value), x_min, x_max, panel_left, panel_right),
                    int(bottom - float(prob) * (bottom - top)),
                )
                for value, prob in zip(xs, ys)
            ]
            draw.line(points, fill=colors[method], width=3)

    draw.text((left + 455, height - 82), "NormFulFill", fill="#202124", font=label_font)
    legend_x = left
    legend_y = height - 48
    for method in METHODS:
        draw.line((legend_x, legend_y + 8, legend_x + 32, legend_y + 8), fill=colors[method], width=5)
        draw.text((legend_x + 42, legend_y - 3), method, fill="#202124", font=label_font)
        legend_x += 180
    image.save(RESULT_DIR / "repro_cdf_geant8_full.png")


def markdown_table(summary: list[dict[str, str | float | int]]) -> str:
    lines = [
        "| Method | Class | Mean | Median | P10 | Paper P10 | Delta P10 | P1 | Paper P1 | Delta P1 |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary:
        lines.append(
            "| {method} | {cls} | {mean:.6f} | {median:.6f} | {p10:.6f} | {paper_p10:.6f} | {delta_p10:+.6f} | {p1:.6f} | {paper_p1:.6f} | {delta_p1:+.6f} |\n".format(
                method=row["method"],
                cls=row["class"],
                mean=float(row["norm_fulfill_mean"]),
                median=float(row["norm_fulfill_median"]),
                p10=float(row["norm_fulfill_p10"]),
                paper_p10=float(row["paper_p10"]),
                delta_p10=float(row["delta_p10_vs_paper"]),
                p1=float(row["norm_fulfill_p1"]),
                paper_p1=float(row["paper_p1"]),
                delta_p1=float(row["delta_p1_vs_paper"]),
            )
        )
    return "".join(lines)


def write_report(summary: list[dict[str, str | float | int]]) -> None:
    state_path = RESULT_DIR.parent / "_repro_state" / "state.json"
    state_text = state_path.read_text(encoding="utf-8") if state_path.exists() else "{}"
    lines = [
        "# GEANT Figure 4 Reproduction Report\n\n",
        "## Target\n",
        "This run targets the GEANT + ESM + K=8 experiment corresponding to paper Figure 4. "
        "It compares Hattrick, BEST_MC/Flexile, and SWAN on per-class NormFulFill.\n\n",
        "## Configuration\n",
        "- Dataset split: train 0-6463, validation 6464-8079, test 8080-10772.\n",
        "- Topology: GEANT, static, 8 shortest paths per pair.\n",
        "- Predictor: ESM.\n",
        "- Hattrick target training: 60 one-epoch resumable runs, batch size 64 unless fallback is recorded below.\n",
        "- Paper reference values are the p1 and p10 NormFulFill values stated in the Chinese translation of Section 5.1.\n\n",
        "## Results\n",
        markdown_table(summary),
        "\n## Artifacts\n",
        "- `repro_metrics_geant8_full.csv`: per-snapshot metrics.\n",
        "- `repro_summary_stats_geant8_full.csv`: grouped statistics.\n",
        "- `repro_cdf_geant8_full.png`: empirical CDF from the full test set.\n",
        "- `repro_boxplot_geant8_full.png`: boxplot from the full test set.\n\n",
        "## Runner State\n",
        "```json\n",
        state_text,
        "\n```\n",
    ]
    (RESULT_DIR / "repro_fig4_comparison.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(RESULT_DIR / "repro_metrics_geant8_full.csv", rows)
    summary = summarize(rows)
    write_csv(RESULT_DIR / "repro_summary_stats_geant8_full.csv", summary)
    draw_cdf(rows)
    draw_boxplot(rows)
    write_report(summary)
    print(f"Wrote {RESULT_DIR / 'repro_metrics_geant8_full.csv'}")
    print(f"Wrote {RESULT_DIR / 'repro_summary_stats_geant8_full.csv'}")
    print(f"Wrote {RESULT_DIR / 'repro_cdf_geant8_full.png'}")
    print(f"Wrote {RESULT_DIR / 'repro_boxplot_geant8_full.png'}")
    print(f"Wrote {RESULT_DIR / 'repro_fig4_comparison.md'}")


if __name__ == "__main__":
    main()
