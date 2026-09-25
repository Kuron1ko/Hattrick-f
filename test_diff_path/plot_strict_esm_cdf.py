from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "figures" / "strict_esm_cdf_1x_2x.png"

DATASETS = {
    ("1x", "Hattrick"): ROOT
    / "test_diff_path"
    / "shared1x_edge_toll_head"
    / "onex_level4_hattrick_rows.csv",
    ("1x", "Improved"): ROOT
    / "test_diff_path"
    / "shared1x_edge_toll_head"
    / "onex_level4_edge_toll_rows.csv",
    ("2x", "Hattrick"): ROOT
    / "test_diff_path"
    / "shared2x_edge_toll_head"
    / "strict_esm_2x_hattrick_rows.csv",
    ("2x", "Improved"): ROOT
    / "test_diff_path"
    / "shared2x_edge_toll_head"
    / "strict_esm_2x_candidate_rows.csv",
}

CLASSES = ("High", "Medium", "Low")
LOADS = ("1x", "2x")
METHODS = ("Hattrick", "Improved")

# Paper-like limits. The 2x Low upper bound is extended because its measured
# NormFulFill reaches 1.48; using 1.2 would silently clip valid observations.
AXES = {
    ("1x", "High"): ((0.88, 1.005), np.arange(0.88, 1.001, 0.02), "%.2f"),
    ("1x", "Medium"): ((0.60, 1.10), np.arange(0.60, 1.101, 0.10), "%.1f"),
    ("1x", "Low"): ((0.60, 1.20), np.arange(0.60, 1.201, 0.20), "%.1f"),
    ("2x", "High"): ((0.88, 1.005), np.arange(0.88, 1.001, 0.02), "%.2f"),
    ("2x", "Medium"): ((0.60, 1.10), np.arange(0.60, 1.101, 0.10), "%.1f"),
    ("2x", "Low"): ((0.60, 1.52), np.arange(0.60, 1.401, 0.20), "%.1f"),
}

COLORS = {"Hattrick": "#2F7ED8", "Improved": "#F28E2B"}
LINESTYLES = {"Hattrick": "-", "Improved": "--"}


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


def smooth_cdf(values: np.ndarray, x_limits: tuple[float, float]):
    """Return a Gaussian-kernel CDF with explicit flat start/end segments."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    axis_span = float(x_limits[1] - x_limits[0])
    observed_span = float(np.ptp(ordered))
    standard_deviation = float(np.std(ordered, ddof=1))

    # A moderately enlarged Silverman bandwidth removes the sample-by-sample
    # wiggles while preserving the broad distribution shift between methods.
    silverman = 1.06 * standard_deviation * ordered.size ** (-0.2)
    minimum_bandwidth = max(observed_span * 0.04, axis_span * 2e-5, 1e-10)
    bandwidth = max(1.5 * silverman, minimum_bandwidth)

    # Leave 3% of the visible x range flat at both ends. When possible, extend
    # the smooth transition 2.5 bandwidths beyond the observed sample range.
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
    kernel_cdf = ndtr(
        (smooth_x[:, None] - ordered[None, :]) / bandwidth
    ).mean(axis=1)
    denominator = max(float(kernel_cdf[-1] - kernel_cdf[0]), 1e-12)
    smooth_y = np.clip((kernel_cdf - kernel_cdf[0]) / denominator, 0.0, 1.0)
    smooth_y = np.maximum.accumulate(smooth_y)

    # Explicit horizontal CDF=0 and CDF=1 sections reach the axis boundaries.
    x = np.concatenate(
        ([x_limits[0], transition_start], smooth_x, [transition_end, x_limits[1]])
    )
    y = np.concatenate(([0.0, 0.0], smooth_y, [1.0, 1.0]))
    return x, y


def main() -> None:
    data = {key: read_norm_fulfill(path) for key, path in DATASETS.items()}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
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
                x, y = smooth_cdf(data[(load, method)][class_name], x_limits)
                (line,) = axis.plot(
                    x,
                    y,
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    linewidth=2.25,
                    solid_capstyle="round",
                    dash_capstyle="round",
                    label=method,
                    zorder=3 if method == "Improved" else 2,
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
        ("Hattrick", "Hatrrick-e"),
        loc="lower left",
        bbox_to_anchor=(0.06, 0.006),
        ncol=2,
        handlelength=3.2,
        columnspacing=2.2,
    )
    figure.subplots_adjust(
        left=0.065, right=0.985, top=0.90, bottom=0.105, wspace=0.24, hspace=0.42
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
