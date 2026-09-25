from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "figures" / "strict_esm_cdf_1x_2x_3x.png"

DATASETS = {
    ("1x", "Hattrick"): ROOT
    / "test_diff_path"
    / "shared1x_edge_toll_head"
    / "onex_level4_hattrick_rows.csv",
    ("1x", "Neo"): ROOT
    / "test_diff_path"
    / "shared1x_edge_toll_head"
    / "onex_level4_edge_toll_rows.csv",
    ("2x", "Hattrick"): ROOT
    / "test_diff_path"
    / "shared2x_edge_toll_head"
    / "strict_esm_2x_hattrick_rows.csv",
    ("2x", "Neo"): ROOT
    / "test_diff_path"
    / "shared2x_edge_toll_head"
    / "strict_esm_2x_candidate_rows.csv",
    ("3x", "Hattrick"): ROOT
    / "test_diff_path"
    / "shared3x_edge_toll_head"
    / "strict_esm_3x_hattrick3_rows.csv",
    ("3x", "Neo"): ROOT
    / "test_diff_path"
    / "shared3x_edge_toll_head"
    / "strict_esm_3x_neo3_rows.csv",
}

CLASSES = ("High", "Medium", "Low")
LOADS = ("1x", "2x", "3x")
METHODS = ("Hattrick", "Neo")
COLORS = {"Hattrick": "#2F7ED8", "Neo": "#F28E2B"}
LINESTYLES = {"Hattrick": "-", "Neo": "--"}


def read_norm_fulfill(path: Path) -> dict[str, np.ndarray]:
    values = {class_name: [] for class_name in CLASSES}
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            values[row["class"]].append(float(row["norm_fulfill"]))
    result = {
        class_name: np.asarray(class_values, dtype=np.float64)
        for class_name, class_values in values.items()
    }
    for class_name, class_values in result.items():
        if class_values.size != 100 or not np.isfinite(class_values).all():
            raise ValueError(
                f"Expected 100 finite {class_name} samples in {path}, "
                f"found {class_values.size}."
            )
    return result


def nice_axis(values: np.ndarray, class_name: str):
    lower = float(np.min(values))
    upper = float(np.max(values))
    observed_span = max(upper - lower, 1e-4)
    if class_name == "High":
        left = min(0.88, lower - 0.08 * observed_span)
        right = max(1.005, upper + 0.08 * observed_span)
        step = 0.02
        tick_format = "%.2f"
    else:
        raw_left = lower - max(0.04, 0.08 * observed_span)
        raw_right = upper + max(0.04, 0.08 * observed_span)
        left = min(0.60, math.floor(raw_left * 10.0) / 10.0)
        right = max(1.10 if class_name == "Medium" else 1.20, math.ceil(raw_right * 10.0) / 10.0)
        step = 0.10 if right - left <= 0.7 else 0.20
        tick_format = "%.1f"
    ticks = np.arange(math.ceil(left / step) * step, right + step * 0.25, step)
    return (left, right), ticks, tick_format


def shared_bandwidth(series: list[np.ndarray], x_limits: tuple[float, float]) -> float:
    pooled = np.concatenate(series)
    axis_span = float(x_limits[1] - x_limits[0])
    observed_span = float(np.ptp(pooled))
    standard_deviation = float(np.std(pooled, ddof=1))
    silverman = 1.06 * standard_deviation * pooled.size ** (-0.2)
    return max(1.5 * silverman, observed_span * 0.04, axis_span * 2e-5, 1e-10)


def smooth_cdf(values: np.ndarray, x_limits: tuple[float, float], bandwidth: float):
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    axis_span = float(x_limits[1] - x_limits[0])
    transition_start = max(
        x_limits[0] + 0.03 * axis_span, ordered[0] - 2.5 * bandwidth
    )
    transition_end = min(
        x_limits[1] - 0.03 * axis_span, ordered[-1] + 2.5 * bandwidth
    )
    if transition_end <= transition_start:
        transition_start = x_limits[0] + 0.03 * axis_span
        transition_end = x_limits[1] - 0.03 * axis_span
    smooth_x = np.linspace(transition_start, transition_end, 1201)
    kernel_cdf = ndtr((smooth_x[:, None] - ordered[None, :]) / bandwidth).mean(axis=1)
    denominator = max(float(kernel_cdf[-1] - kernel_cdf[0]), 1e-12)
    smooth_y = np.clip((kernel_cdf - kernel_cdf[0]) / denominator, 0.0, 1.0)
    smooth_y = np.maximum.accumulate(smooth_y)
    x = np.concatenate(
        ([x_limits[0], transition_start], smooth_x, [transition_end, x_limits[1]])
    )
    y = np.concatenate(([0.0, 0.0], smooth_y, [1.0, 1.0]))
    return x, y


def main() -> None:
    data = {key: read_norm_fulfill(path) for key, path in DATASETS.items()}
    axes_by_class = {}
    for class_name in CLASSES:
        pooled = np.concatenate(
            [data[(load, method)][class_name] for load in LOADS for method in METHODS]
        )
        axes_by_class[class_name] = nice_axis(pooled, class_name)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.2,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.2,
            "axes.edgecolor": "#2B2B2B",
            "axes.linewidth": 0.8,
            "xtick.color": "#5A6572",
            "ytick.color": "#5A6572",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(3, 3, figsize=(13.4, 10.6), sharey=True)
    figure.suptitle(
        "GEANT 8sp — CDF of NormFulFill under strict ESM inference",
        x=0.065,
        y=0.99,
        ha="left",
        fontsize=18,
        fontweight="semibold",
    )
    figure.text(
        0.065,
        0.962,
        "Load-specific Hattrick versus load-specific Hattrick-neo · Level-4 snapshots 400–499",
        ha="left",
        color="#5A6572",
        fontsize=10.2,
    )

    legend_handles = []
    for row_index, load in enumerate(LOADS):
        for column_index, class_name in enumerate(CLASSES):
            axis = axes[row_index, column_index]
            x_limits, ticks, tick_format = axes_by_class[class_name]
            panel_series = [data[(load, method)][class_name] for method in METHODS]
            bandwidth = shared_bandwidth(panel_series, x_limits)
            for method in METHODS:
                x, y = smooth_cdf(data[(load, method)][class_name], x_limits, bandwidth)
                (line,) = axis.plot(
                    x,
                    y,
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    linewidth=2.25,
                    solid_capstyle="round",
                    dash_capstyle="round",
                    label=method,
                    zorder=3 if method == "Neo" else 2,
                )
                if row_index == 0 and column_index == 0:
                    legend_handles.append(line)
            axis.set_title(f"{load} load · {class_name}", pad=7)
            axis.set_xlim(*x_limits)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks(ticks)
            axis.xaxis.set_major_formatter(FormatStrFormatter(tick_format))
            axis.set_yticks(np.arange(0.0, 1.01, 0.2))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            axis.set_xlabel("Normalized FulfillRatio", labelpad=7)
            axis.set_ylabel("CDF", labelpad=7)
            axis.grid(axis="y", color="#DCE2E8", linewidth=0.7, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.margins(x=0, y=0)

    figure.legend(
        legend_handles,
        ("Hattrick", "Hattrick-neo · strict ESM"),
        loc="lower left",
        bbox_to_anchor=(0.06, 0.006),
        ncol=2,
        handlelength=3.2,
        columnspacing=2.2,
    )
    figure.subplots_adjust(
        left=0.065, right=0.985, top=0.91, bottom=0.08, wspace=0.24, hspace=0.43
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
