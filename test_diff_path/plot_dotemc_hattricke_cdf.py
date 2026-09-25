from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from plot_strict_esm_cdf import AXES, smooth_cdf


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "output" / "comparisons" / "dotemc_hattricke"
OUTPUT = ROOT / "output" / "figures" / "cdf_hattrick_hattricke_dotemc_1x_2x.png"
LOADS = ("1x", "2x")
CLASSES = ("High", "Medium", "Low")
METHODS = ("Hattrick", "Hatrrick-e", "DOTE-MC")
COLORS = {
    "Hattrick": "#2F7ED8",
    "Hatrrick-e": "#F28E2B",
    "DOTE-MC": "#2AA65A",
}
LINESTYLES = {"Hattrick": "-", "Hatrrick-e": "--", "DOTE-MC": "-."}


def read_values(load: str) -> dict[tuple[str, str], np.ndarray]:
    grouped = {(method, class_name): [] for method in METHODS for class_name in CLASSES}
    path = INPUT_DIR / f"comparison_rows_{load}.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], row["class"])
            if key in grouped:
                grouped[key].append(float(row["norm_fulfill"]))
    return {key: np.asarray(value, dtype=np.float64) for key, value in grouped.items()}


def main() -> None:
    data = {load: read_values(load) for load in LOADS}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10.5,
            "legend.frameon": False,
            "axes.linewidth": 0.75,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.7), sharey=True)
    figure.suptitle(
        "CDF of NormFulFill",
        x=0.065,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="semibold",
    )
    legend_handles = []
    for row_index, load in enumerate(LOADS):
        for column_index, class_name in enumerate(CLASSES):
            axis = axes[row_index, column_index]
            x_limits, ticks, tick_format = AXES[(load, class_name)]
            for method in METHODS:
                x, y = smooth_cdf(data[load][(method, class_name)], x_limits)
                (line,) = axis.plot(
                    x,
                    y,
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    linewidth=2.25,
                    solid_capstyle="round",
                    dash_capstyle="round",
                    label=method,
                    zorder={"Hattrick": 2, "Hatrrick-e": 3, "DOTE-MC": 4}[method],
                )
                if row_index == 0 and column_index == 0:
                    legend_handles.append(line)

            axis.set_title(f"{load} load · {class_name}", pad=8)
            axis.set_xlim(*x_limits)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks(ticks)
            axis.xaxis.set_major_formatter(FormatStrFormatter(tick_format))
            axis.set_yticks(np.arange(0.0, 1.01, 0.2))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            axis.set_xlabel("Normalized FulfillRatio", labelpad=8)
            axis.set_ylabel("CDF", labelpad=8)
            axis.grid(axis="y", color="#DCE2E8", linewidth=0.7, alpha=0.8)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.margins(x=0, y=0)

    figure.legend(
        legend_handles,
        METHODS,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.006),
        ncol=3,
        handlelength=3.2,
        columnspacing=2.2,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.90,
        bottom=0.105,
        wspace=0.24,
        hspace=0.42,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
