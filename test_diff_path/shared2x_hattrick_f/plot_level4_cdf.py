from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.special import ndtr


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
OUTPUT_DIR = HERE / "artifacts" / "level4_report"
OUTPUT_PNG = OUTPUT_DIR / "cdf_dote_hattrick_hattrick_f.png"
OUTPUT_PDF = OUTPUT_DIR / "cdf_dote_hattrick_hattrick_f.pdf"
OUTPUT_CURVES = OUTPUT_DIR / "cdf_dote_hattrick_hattrick_f_curves.json"

INPUTS = {
    "DOTE-MC": (
        TEST_DIR
        / "shared2x_order_regularizer"
        / "artifacts"
        / "common_final_existing_methods"
        / "common_metrics.csv"
    ),
    "Hattrick": (
        TEST_DIR
        / "shared2x_full_objectives"
        / "artifacts"
        / "level4_confirmation"
        / "seed_490"
        / "best_evaluation_metrics.csv"
    ),
    "Hattrick-f": (
        HERE
        / "artifacts"
        / "level4_confirmation"
        / "fh_release_low_budget_0p03"
        / "seed_490"
        / "evaluation_metrics.csv"
    ),
}

CLASSES = ("High", "Medium", "Low")
METHODS = ("DOTE-MC", "Hattrick", "Hattrick-f")
COLORS = {
    "DOTE-MC": "#2AA65A",
    "Hattrick": "#2F7ED8",
    "Hattrick-f": "#F28E2B",
}
LINESTYLES = {"DOTE-MC": "-.", "Hattrick": "-", "Hattrick-f": "--"}
AXES = {
    "High": ((0.982, 1.002), np.arange(0.985, 1.001, 0.005), "%.3f"),
    "Medium": ((0.86, 1.04), np.arange(0.86, 1.041, 0.04), "%.2f"),
    "Low": ((0.88, 1.46), np.arange(0.9, 1.41, 0.1), "%.1f"),
}


def read_values(method: str, path: Path) -> dict[str, np.ndarray]:
    values = {class_name: [] for class_name in CLASSES}
    snapshots = {class_name: [] for class_name in CLASSES}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if method == "DOTE-MC" and row.get("method") != "DOTE-MC sensitivity":
                continue
            class_name = row["class"]
            values[class_name].append(float(row["norm_fulfill"]))
            snapshots[class_name].append(int(row["snapshot"]))
    result = {}
    for class_name in CLASSES:
        array = np.asarray(values[class_name], dtype=np.float64)
        if array.size != 100 or not np.isfinite(array).all():
            raise ValueError(
                f"{method}/{class_name}: expected 100 finite values, got {array.size}"
            )
        if sorted(snapshots[class_name]) != list(range(400, 500)):
            raise ValueError(f"{method}/{class_name}: snapshot window is not 400-499")
        result[class_name] = array
    return result


def smooth_cdf(values: np.ndarray, x_limits: tuple[float, float]):
    """Gaussian-kernel CDF with explicit horizontal start/end segments."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    axis_span = float(x_limits[1] - x_limits[0])
    observed_span = float(np.ptp(ordered))
    standard_deviation = float(np.std(ordered, ddof=1))
    silverman = 1.06 * standard_deviation * ordered.size ** (-0.2)
    minimum_bandwidth = max(observed_span * 0.04, axis_span * 2e-5, 1e-10)
    bandwidth = max(1.5 * silverman, minimum_bandwidth)

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
    smooth_y = np.maximum.accumulate(
        np.clip((kernel_cdf - kernel_cdf[0]) / denominator, 0.0, 1.0)
    )
    x = np.concatenate(
        ([x_limits[0], transition_start], smooth_x, [transition_end, x_limits[1]])
    )
    y = np.concatenate(([0.0, 0.0], smooth_y, [1.0, 1.0]))
    return x, y


def main() -> None:
    data = {method: read_values(method, INPUTS[method]) for method in METHODS}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10.5,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.35), sharey=True)
    figure.suptitle(
        "CDF of NormFulFill",
        x=0.066,
        y=0.97,
        ha="left",
        fontsize=18,
        fontweight="semibold",
    )
    legend_handles = []
    exported: dict[str, object] = {
        "protocol": "2x load, strict ESM, independent snapshots 400-499",
        "classes": {},
    }
    for axis, class_name in zip(axes, CLASSES):
        x_limits, ticks, tick_format = AXES[class_name]
        exported["classes"][class_name] = {}
        for method in METHODS:
            x, y = smooth_cdf(data[method][class_name], x_limits)
            line, = axis.plot(
                x,
                y,
                color=COLORS[method],
                linestyle=LINESTYLES[method],
                linewidth=2.35,
                solid_capstyle="round",
                dash_capstyle="round",
                label=method,
                zorder={"Hattrick": 2, "DOTE-MC": 3, "Hattrick-f": 4}[method],
            )
            if class_name == "High":
                legend_handles.append(line)
            exported["classes"][class_name][method] = {
                "x": np.round(x[::4], 7).tolist(),
                "y": np.round(y[::4], 7).tolist(),
                "mean": float(data[method][class_name].mean()),
            }

        axis.set_title(class_name, pad=9)
        axis.set_xlim(*x_limits)
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks(ticks)
        axis.xaxis.set_major_formatter(FormatStrFormatter(tick_format))
        axis.set_yticks(np.arange(0.0, 1.01, 0.2))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.set_xlabel("Normalized FulfillRatio", labelpad=8)
        axis.set_ylabel("CDF", labelpad=8)
        axis.grid(axis="y", color="#DCE2E8", linewidth=0.7, alpha=0.85)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(x=0, y=0)

    figure.legend(
        legend_handles,
        METHODS,
        loc="lower left",
        bbox_to_anchor=(0.06, 0.005),
        ncol=3,
        handlelength=3.2,
        columnspacing=2.2,
    )
    figure.subplots_adjust(
        left=0.066, right=0.985, top=0.84, bottom=0.22, wspace=0.24
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(figure)
    OUTPUT_CURVES.write_text(
        json.dumps(exported, separators=(",", ":")), encoding="utf-8"
    )
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)
    print(OUTPUT_CURVES)


if __name__ == "__main__":
    main()
