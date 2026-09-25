from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.special import ndtr


HERE = Path(__file__).resolve().parent
REPORT = HERE / "final_universal_pareto.json"
OUTPUT_STEM = "final_universal_pareto_cdf"
TITLE = "GEANT 8sp — CDF of NormFulFill under strict ESM inference"
SUBTITLE = "One class-symmetric objective and one Pareto guard for all loads · Level-4 snapshots 400–499"
LOADS = ("1", "2", "3")
CLASSES = ("High", "Medium", "Low")
SERIES = (("Hattrick", "baseline"), ("U-Pareto · strict ESM", "candidate"))
COLORS = {"Hattrick": "#2F78D4", "U-Pareto · strict ESM": "#F28E2B"}
LINESTYLES = {"Hattrick": "-", "U-Pareto · strict ESM": "--"}
LEGEND_NCOL = 2
AXES = {
    "High": ((0.88, 1.008), np.arange(0.88, 1.001, 0.02), "%.2f"),
    "Medium": ((0.60, 1.30), np.arange(0.60, 1.21, 0.20), "%.1f"),
    "Low": ((0.60, 1.60), np.arange(0.60, 1.61, 0.20), "%.1f"),
}


def values(payload, load, source, class_name):
    rows = payload["loads"][load]["evaluation"]["rows"][source]
    selected = [
        float(row["norm_fulfill"])
        for row in rows
        if row["class"] == class_name
    ]
    if len(selected) != 100:
        raise ValueError((load, source, class_name, len(selected)))
    return np.asarray(selected, dtype=np.float64)


def shared_bandwidth(series, left, right):
    pooled = np.concatenate(series)
    deviation = float(np.std(pooled, ddof=1))
    quartiles = np.percentile(pooled, (25.0, 75.0))
    robust = float((quartiles[1] - quartiles[0]) / 1.349)
    if robust <= 1e-12:
        robust = deviation
    scale = min(deviation, robust) if deviation > 0 else robust
    silverman = 0.9 * scale * pooled.size ** (-0.2)
    return max(
        1.5 * silverman,
        0.045 * float(np.ptp(pooled)),
        2e-5 * float(right - left),
        1e-10,
    )


def smooth_cdf(samples, left, right, bandwidth, hard_upper=None):
    span = right - left
    start = max(left + 0.025 * span, float(samples.min()) - 3.5 * bandwidth)
    end = min(right - 0.025 * span, float(samples.max()) + 3.5 * bandwidth)
    if hard_upper is not None:
        end = min(end, hard_upper)
    dense_x = np.linspace(start, end, 1401)
    dense_y = ndtr((dense_x[:, None] - samples[None, :]) / bandwidth).mean(axis=1)
    denominator = max(float(dense_y[-1] - dense_y[0]), 1e-12)
    dense_y = np.maximum.accumulate(
        np.clip((dense_y - dense_y[0]) / denominator, 0.0, 1.0)
    )
    return (
        np.concatenate(([left, start], dense_x, [end, right])),
        np.concatenate(([0.0, 0.0], dense_y, [1.0, 1.0])),
    )


def main():
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": "#2A2A2A",
            "axes.linewidth": 0.85,
            "xtick.color": "#596574",
            "ytick.color": "#596574",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(3, 3, figsize=(18.2, 16.9), sharey=True)
    figure.suptitle(
        TITLE,
        x=0.055,
        y=0.988,
        ha="left",
        fontsize=23,
        fontweight="semibold",
    )
    if SUBTITLE:
        figure.text(
            0.055,
            0.965,
            SUBTITLE,
            ha="left",
            color="#596574",
            fontsize=12.5,
        )
    for row_index, load in enumerate(LOADS):
        for column_index, class_name in enumerate(CLASSES):
            axis = axes[row_index, column_index]
            (left, right), ticks, tick_format = AXES[class_name]
            panel = [values(payload, load, source, class_name) for _, source in SERIES]
            bandwidth = shared_bandwidth(panel, left, right)
            for label, source in SERIES:
                x, y = smooth_cdf(
                    values(payload, load, source, class_name),
                    left,
                    right,
                    bandwidth,
                    hard_upper=1.0 if class_name == "High" else None,
                )
                axis.plot(
                    x,
                    y,
                    color=COLORS[label],
                    linestyle=LINESTYLES[label],
                    linewidth=2.65,
                    dash_capstyle="round",
                    solid_capstyle="round",
                    label=label,
                )
            axis.set_xlim(left, right)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks(ticks)
            axis.xaxis.set_major_formatter(FormatStrFormatter(tick_format))
            axis.set_yticks(np.linspace(0.0, 1.0, 6))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            axis.grid(True, color="#DDE3EA", linewidth=0.75)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_title(f"{load}x load · {class_name}", pad=9)
            axis.set_xlabel("Normalized FulfillRatio", labelpad=8)
            axis.set_ylabel("CDF", labelpad=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.012),
        ncol=LEGEND_NCOL,
        fontsize=12.5,
        handlelength=3.2,
        columnspacing=2.0,
    )
    top = 0.945 if SUBTITLE else 0.965
    figure.tight_layout(rect=(0.035, 0.055, 0.995, top), h_pad=2.55, w_pad=2.0)
    figure.savefig(HERE / f"{OUTPUT_STEM}.png", dpi=210, bbox_inches="tight")
    figure.savefig(HERE / f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
