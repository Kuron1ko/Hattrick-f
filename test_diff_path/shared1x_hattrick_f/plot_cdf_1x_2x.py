from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
OUTPUT_DIR = ROOT / "output" / "comparisons" / "hattrick_f_1x_2x"

CLASSES = ("High", "Medium", "Low")
METHODS = ("Hattrick", "Hattrick-f", "DOTE-MC")
SOURCES = {
    ("1x", "Hattrick"): TEST_DIR
    / "shared1x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "best_evaluation_metrics.csv",
    ("1x", "Hattrick-f"): THIS_DIR
    / "artifacts"
    / "level4_confirmation"
    / "fh_release_low_budget_0p03"
    / "seed_490"
    / "evaluation_metrics.csv",
    ("1x", "DOTE-MC"): ROOT
    / "output"
    / "comparisons"
    / "dotemc_hattricke"
    / "comparison_rows_1x.csv",
    ("2x", "Hattrick"): TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "best_evaluation_metrics.csv",
    ("2x", "Hattrick-f"): TEST_DIR
    / "shared2x_hattrick_f"
    / "artifacts"
    / "level4_confirmation"
    / "fh_release_low_budget_0p03"
    / "seed_490"
    / "evaluation_metrics.csv",
    ("2x", "DOTE-MC"): ROOT
    / "output"
    / "comparisons"
    / "dotemc_hattricke"
    / "comparison_rows_2x.csv",
}

COLORS = {
    "Hattrick": "#2F7ED8",
    "Hattrick-f": "#F28E2B",
    "DOTE-MC": "#2AA65A",
}
LINESTYLES = {"Hattrick": "-", "Hattrick-f": "--", "DOTE-MC": "-."}
X_LIMITS = {
    ("1x", "High"): (0.88, 1.00),
    ("1x", "Medium"): (0.60, 1.10),
    ("1x", "Low"): (0.60, 1.20),
    ("2x", "High"): (0.88, 1.00),
    ("2x", "Medium"): (0.60, 1.10),
    ("2x", "Low"): (0.60, 1.55),
}
BANDWIDTH_BOUNDS = {
    "High": (1.0e-4, 1.4e-3),
    "Medium": (1.5e-3, 1.8e-2),
    "Low": (8.0e-3, 2.5e-2),
}


def read_values(load: str, method: str, class_name: str) -> np.ndarray:
    path = SOURCES[(load, method)]
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    snapshots = []
    for row in rows:
        if row.get("class") != class_name:
            continue
        if method == "DOTE-MC" and row.get("method") != "DOTE-MC":
            continue
        selected.append(float(row["norm_fulfill"]))
        snapshots.append(int(row["snapshot"]))
    if len(selected) != 100 or sorted(snapshots) != list(range(400, 500)):
        raise RuntimeError(
            f"{load} {method} {class_name}: expected snapshots 400-499, "
            f"got n={len(selected)} range={min(snapshots, default=None)}-"
            f"{max(snapshots, default=None)}"
        )
    values = np.asarray(selected, dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"non-finite values in {path}")
    if class_name == "High":
        # High's oracle is a hard upper bound. Values such as 1.00000012 are
        # simulator floating-point noise, not genuine mass above one.
        if float(values.max()) > 1.0 + 1e-5 or float(values.min()) < -1e-8:
            raise RuntimeError(
                f"High NormFulFill violates [0,1] beyond numerical tolerance: "
                f"min={values.min():.9g}, max={values.max():.9g}"
            )
        values = np.clip(values, 0.0, 1.0)
    return values


def robust_bandwidth(values: np.ndarray, class_name: str) -> float:
    standard = float(np.std(values, ddof=1))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    robust = float((q75 - q25) / 1.349)
    scales = [value for value in (standard, robust) if value > 1e-12]
    scale = min(scales) if scales else 0.0
    estimate = 0.9 * scale * values.size ** (-0.2)
    lower, upper = BANDWIDTH_BOUNDS[class_name]
    return float(np.clip(estimate, lower, upper))


def smooth_cdf(
    values: np.ndarray, x_min: float, x_max: float, class_name: str
) -> tuple[np.ndarray, np.ndarray, float]:
    grid = np.linspace(x_min, x_max, 1400, dtype=np.float64)
    bandwidth = robust_bandwidth(values, class_name)
    if class_name == "High":
        grid = np.unique(np.concatenate([grid, np.asarray([1.0])]))
        # Upper-bound reflection makes F(1)=1 exactly while retaining a smooth,
        # monotone curve on x<1. It avoids the false Gaussian-KDE tail above 1.
        original = ndtr((grid[:, None] - values[None, :]) / bandwidth)
        reflected = ndtr(
            (grid[:, None] - (2.0 - values)[None, :]) / bandwidth
        )
        cdf = np.mean(original + reflected, axis=1)
        cdf[grid >= 1.0] = 1.0
    else:
        cdf = np.mean(
            ndtr((grid[:, None] - values[None, :]) / bandwidth), axis=1
        )
        cdf[grid <= float(values.min()) - 4.0 * bandwidth] = 0.0
        cdf[grid >= float(values.max()) + 4.0 * bandwidth] = 1.0
    cdf = np.clip(np.maximum.accumulate(cdf), 0.0, 1.0)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    return grid, cdf, bandwidth


def build_plot() -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.8), sharey=True)
    evidence = {
        "protocol": {
            "inference": "strict ESM prediction only",
            "train": [0, 350],
            "validation_selection_only": [350, 400],
            "independent_evaluation": [400, 500],
            "paths": 8,
        },
        "sources": {f"{load}_{method}": str(path) for (load, method), path in SOURCES.items()},
        "series": [],
    }

    for row_index, load in enumerate(("1x", "2x")):
        for column_index, class_name in enumerate(CLASSES):
            ax = axes[row_index, column_index]
            x_min, x_max = X_LIMITS[(load, class_name)]
            for method in METHODS:
                values = read_values(load, method, class_name)
                x, y, bandwidth = smooth_cdf(values, x_min, x_max, class_name)
                ax.plot(
                    x,
                    y,
                    color=COLORS[method],
                    linestyle=LINESTYLES[method],
                    linewidth=2.8,
                    label=method,
                    zorder={"Hattrick": 2, "Hattrick-f": 3, "DOTE-MC": 4}[method],
                )
                evidence["series"].append(
                    {
                        "load": load,
                        "class": class_name,
                        "method": method,
                        "n": int(values.size),
                        "mean": float(values.mean()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "bandwidth": bandwidth,
                    }
                )
            ax.set_title(f"{load} load · {class_name}", pad=10)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks(np.linspace(0.0, 1.0, 6))
            ax.set_xlabel("Normalized FulfillRatio", labelpad=9)
            ax.set_ylabel("CDF", labelpad=8)
            ax.grid(axis="y", color="#D8DEE8", linewidth=0.9, alpha=0.85)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", length=4, width=0.9)

    fig.suptitle(
        "CDF of NormFulFill",
        x=0.065,
        y=0.985,
        ha="left",
        va="top",
        fontsize=25,
        fontweight="bold",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.065, 0.012),
        ncol=3,
        frameon=False,
        handlelength=3.4,
        columnspacing=2.3,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.91,
        bottom=0.105,
        hspace=0.42,
        wspace=0.24,
    )

    png_path = OUTPUT_DIR / "cdf_hattrick_hattrick_f_dotemc_1x_2x.png"
    pdf_path = OUTPUT_DIR / "cdf_hattrick_hattrick_f_dotemc_1x_2x.pdf"
    evidence_path = OUTPUT_DIR / "cdf_hattrick_hattrick_f_dotemc_1x_2x.json"
    fig.savefig(png_path, dpi=180, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return png_path, pdf_path, evidence_path


if __name__ == "__main__":
    for output in build_plot():
        print(output)
