from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.special import ndtr


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
SOURCES = {
    "1x": TEST_DIR / "shared1x_priority_fallback_lcr" / "report.json",
    "2x": TEST_DIR
    / "shared2x_teacher_toll_distill"
    / "artifacts"
    / "asymmetric_knn_level4"
    / "final_rows.json",
    "3x": TEST_DIR
    / "shared_multi_asymmetric_knn_toll"
    / "artifacts"
    / "3x_level4"
    / "report.json",
}
LOADS = ("1x", "2x", "3x")
CLASSES = ("High", "Medium", "Low")
SERIES = (("Hattrick", "baseline"), ("Hattrick-neo · strict ESM", "candidate"))
COLORS = {"Hattrick": "#2F78D4", "Hattrick-neo · strict ESM": "#F28E2B"}
LINESTYLES = {"Hattrick": "-", "Hattrick-neo · strict ESM": "--"}
AXES = {
    "High": ((0.88, 1.008), np.arange(0.88, 1.001, 0.02), "%.2f"),
    "Medium": ((0.60, 1.30), np.arange(0.60, 1.21, 0.20), "%.1f"),
    "Low": ((0.60, 1.60), np.arange(0.60, 1.61, 0.20), "%.1f"),
}


def load_rows() -> dict[str, dict[str, list[dict]]]:
    payloads = {
        load: json.loads(path.read_text(encoding="utf-8"))
        for load, path in SOURCES.items()
    }
    rows = {load: payload["rows"] for load, payload in payloads.items()}
    for load in LOADS:
        for source in ("baseline", "candidate"):
            for class_name in CLASSES:
                count = sum(
                    row["class"] == class_name for row in rows[load][source]
                )
                if count != 100:
                    raise ValueError(
                        f"Expected 100 rows for {load}/{source}/{class_name}; got {count}"
                    )
    return rows


def values(rows: dict, load: str, source: str, class_name: str) -> np.ndarray:
    return np.asarray(
        [
            float(row["norm_fulfill"])
            for row in rows[load][source]
            if row["class"] == class_name
        ],
        dtype=np.float64,
    )


def shared_bandwidth(
    series: list[np.ndarray],
    left: float,
    right: float,
) -> float:
    pooled = np.concatenate(series)
    standard_deviation = float(np.std(pooled, ddof=1))
    quartiles = np.percentile(pooled, (25.0, 75.0))
    robust_scale = float((quartiles[1] - quartiles[0]) / 1.349)
    if robust_scale <= 1e-12:
        robust_scale = standard_deviation
    scale = min(standard_deviation, robust_scale) if standard_deviation > 0 else robust_scale
    silverman = 0.9 * scale * pooled.size ** (-0.2)
    observed_span = float(np.ptp(pooled))
    axis_span = float(right - left)
    return max(
        1.5 * silverman,
        0.045 * observed_span,
        2e-5 * axis_span,
        1e-10,
    )


def smooth_monotone_cdf(
    samples: np.ndarray,
    left: float,
    right: float,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-kernel CDF with normalized, exactly flat 0/1 tails."""
    axis_span = float(right - left)
    transition_start = max(
        left + 0.025 * axis_span,
        float(np.min(samples)) - 3.5 * bandwidth,
    )
    transition_end = min(
        right - 0.025 * axis_span,
        float(np.max(samples)) + 3.5 * bandwidth,
    )
    dense_x = np.linspace(transition_start, transition_end, 1401)
    dense_y = ndtr(
        (dense_x[:, None] - samples[None, :]) / bandwidth
    ).mean(axis=1)
    denominator = max(float(dense_y[-1] - dense_y[0]), 1e-12)
    dense_y = np.clip((dense_y - dense_y[0]) / denominator, 0.0, 1.0)
    dense_y = np.maximum.accumulate(dense_y)
    x = np.concatenate(([left, transition_start], dense_x, [transition_end, right]))
    y = np.concatenate(([0.0, 0.0], dense_y, [1.0, 1.0]))
    return x, y


def main() -> None:
    rows = load_rows()
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
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(3, 3, figsize=(18.2, 16.9), sharey=True)
    figure.suptitle(
        "GEANT 8sp — CDF of NormFulFill under strict ESM inference",
        x=0.055,
        y=0.988,
        ha="left",
        fontsize=23,
        fontweight="semibold",
    )
    figure.text(
        0.055,
        0.965,
        "Load-specific Hattrick versus final load-aware Hattrick-neo · Level-4 snapshots 400–499",
        ha="left",
        color="#596574",
        fontsize=12.5,
    )

    for row_index, load in enumerate(LOADS):
        for column_index, class_name in enumerate(CLASSES):
            axis = axes[row_index, column_index]
            (left, right), ticks, tick_format = AXES[class_name]
            panel_values = [
                values(rows, load, source, class_name)
                for _, source in SERIES
            ]
            bandwidth = shared_bandwidth(panel_values, left, right)
            for label, source in SERIES:
                x, y = smooth_monotone_cdf(
                    values(rows, load, source, class_name),
                    left,
                    right,
                    bandwidth,
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
            axis.set_title(f"{load} load · {class_name}", pad=9)
            axis.set_xlabel("Normalized FulfillRatio", labelpad=8)
            axis.set_ylabel("CDF", labelpad=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.012),
        ncol=2,
        fontsize=12.5,
        handlelength=3.2,
        columnspacing=2.0,
    )
    figure.tight_layout(rect=(0.035, 0.055, 0.995, 0.945), h_pad=2.55, w_pad=2.0)
    HERE.mkdir(parents=True, exist_ok=True)
    figure.savefig(HERE / "final_smooth_cdf_1x_2x_3x.png", dpi=210, bbox_inches="tight")
    figure.savefig(HERE / "final_smooth_cdf_1x_2x_3x.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
