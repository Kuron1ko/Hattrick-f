from __future__ import annotations

"""Plot smooth far-horizon strict-ESM raw FulfillRatio CDFs."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr


HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "artifacts" / "far_1x_2x_seed490"
INPUT_CSV = INPUT_DIR / "snapshots.csv"
OUTPUT_PNG = INPUT_DIR / "cdf_far_1x_2x_hattrick_vs_hattrick_f.png"
OUTPUT_JSON = INPUT_DIR / "cdf_evidence.json"

LOADS = ("1x", "2x")
CLASSES = ("High", "Medium", "Low")
METHODS = ("Hattrick", "Hattrick-f")
COLORS = {"Hattrick": "#2F7ED8", "Hattrick-f": "#F28E2B"}
LINESTYLES = {"Hattrick": "-", "Hattrick-f": "--"}
DEFAULT_LEFT = {"High": 0.88, "Medium": 0.60, "Low": 0.60}
BANDWIDTH_BOUNDS = {
    "High": (4.0e-4, 8.0e-3),
    "Medium": (2.0e-3, 2.5e-2),
    "Low": (3.0e-3, 3.0e-2),
}


def read_rows() -> list[dict]:
    with INPUT_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values_for(
    rows: list[dict], load: str, method: str, class_name: str
) -> np.ndarray:
    values = np.asarray(
        [
            float(row["raw_fulfill_ratio"])
            for row in rows
            if row["load"] == load
            and row["method_label"] == method
            and row["class"] == class_name
        ],
        dtype=np.float64,
    )
    if values.size != 500:
        raise RuntimeError(
            f"{load}/{method}/{class_name}: expected 500 values, got {values.size}"
        )
    if float(values.min()) < -1e-8 or float(values.max()) > 1.0 + 1e-8:
        raise RuntimeError("raw FulfillRatio must be in [0,1]")
    return np.clip(values, 0.0, 1.0)


def bandwidth(values: np.ndarray, class_name: str) -> float:
    standard = float(np.std(values, ddof=1))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    robust = float((q75 - q25) / 1.349)
    scales = [value for value in (standard, robust) if value > 1e-12]
    scale = min(scales) if scales else 0.0
    estimate = 0.9 * scale * values.size ** (-0.2)
    lower, upper = BANDWIDTH_BOUNDS[class_name]
    return float(np.clip(estimate, lower, upper))


def axis_left(all_values: list[np.ndarray], class_name: str) -> float:
    minimum = min(float(values.min()) for values in all_values)
    largest_bandwidth = max(bandwidth(values, class_name) for values in all_values)
    data_left = np.floor((minimum - 4.5 * largest_bandwidth) / 0.05) * 0.05
    return float(max(0.0, min(DEFAULT_LEFT[class_name], data_left)))


def smooth_cdf(
    values: np.ndarray, x_left: float, class_name: str
) -> tuple[np.ndarray, np.ndarray, float]:
    width = bandwidth(values, class_name)
    grid = np.linspace(x_left, 1.02, 1600, dtype=np.float64)
    original = ndtr((grid[:, None] - values[None, :]) / width)
    upper_reflection = ndtr(
        (grid[:, None] - (2.0 - values)[None, :]) / width
    )
    cdf = np.mean(original + upper_reflection, axis=1)
    cdf[grid <= float(values.min()) - 4.0 * width] = 0.0
    cdf[grid >= 1.0] = 1.0
    cdf = np.clip(np.maximum.accumulate(cdf), 0.0, 1.0)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    return grid, cdf, width


def main() -> None:
    rows = read_rows()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 13,
            "axes.linewidth": 1.0,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(18.0, 10.8), sharey=True)
    evidence = {
        "metric": "raw FulfillRatio; no far-horizon oracle normalization",
        "inference": "strict ESM prediction only",
        "window": [9000, 9500],
        "seed": 490,
        "series": [],
    }

    for row_index, load in enumerate(LOADS):
        for column_index, class_name in enumerate(CLASSES):
            axis = axes[row_index, column_index]
            series = {
                method: values_for(rows, load, method, class_name)
                for method in METHODS
            }
            x_left = axis_left(list(series.values()), class_name)
            for method in METHODS:
                values = series[method]
                x, y, width = smooth_cdf(values, x_left, class_name)
                axis.plot(
                    x,
                    y,
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    linewidth=2.8,
                    label=method,
                )
                evidence["series"].append(
                    {
                        "load": load,
                        "class": class_name,
                        "method": method,
                        "n": int(values.size),
                        "mean": float(values.mean()),
                        "p1": float(np.percentile(values, 1)),
                        "p10": float(np.percentile(values, 10)),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "bandwidth": width,
                    }
                )
            axis.set_title(f"{load} load · {class_name}", pad=10)
            axis.set_xlim(x_left, 1.02)
            axis.set_ylim(0.0, 1.0)
            axis.set_yticks(np.linspace(0.0, 1.0, 6))
            axis.set_xlabel("FulfillRatio", labelpad=9)
            axis.set_ylabel("CDF", labelpad=8)
            axis.grid(axis="y", color="#D8DEE8", linewidth=0.9, alpha=0.85)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(direction="out", length=4, width=0.9)

    figure.suptitle(
        "Far-horizon CDF of FulfillRatio",
        x=0.065,
        y=0.985,
        ha="left",
        va="top",
        fontsize=25,
        fontweight="bold",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.065, 0.012),
        ncol=2,
        frameon=False,
        handlelength=3.4,
        columnspacing=2.3,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.91,
        bottom=0.105,
        hspace=0.42,
        wspace=0.24,
    )
    figure.savefig(OUTPUT_PNG, dpi=180, facecolor="white")
    plt.close(figure)
    OUTPUT_JSON.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(OUTPUT_PNG.resolve())


if __name__ == "__main__":
    main()
