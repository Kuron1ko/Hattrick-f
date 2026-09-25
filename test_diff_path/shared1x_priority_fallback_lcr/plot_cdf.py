from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPORT = HERE / "report.json"
CLASSES = ("High", "Medium", "Low")
RANGES = {
    "High": (0.88, 1.005),
    "Medium": (0.95, 1.005),
    "Low": (0.95, 1.20),
}


def values(rows: list[dict], class_name: str) -> np.ndarray:
    return np.asarray(
        [row["norm_fulfill"] for row in rows if row["class"] == class_name],
        dtype=np.float64,
    )


def ecdf(samples: np.ndarray, left: float, right: float):
    ordered = np.sort(samples)
    x = np.concatenate(([left, ordered[0]], ordered, [right]))
    y = np.concatenate(
        ((0.0, 0.0), np.arange(1, ordered.size + 1) / ordered.size, (1.0,))
    )
    return x, y


def main() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = payload["rows"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": "#2B2B2B",
            "axes.linewidth": 0.8,
            "xtick.color": "#5A6572",
            "ytick.color": "#5A6572",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.6, 5.25), sharey=True)
    figure.suptitle(
        "GEANT 1× strict-ESM — CDF of NormFulfill",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="semibold",
    )
    figure.text(
        0.07,
        0.925,
        "Original Hattrick versus priority-fallback LCR · Level-4 snapshots 400–499",
        color="#5A6572",
        fontsize=10.5,
    )
    for axis, class_name in zip(axes, CLASSES):
        left, right = RANGES[class_name]
        for label, source, color, linestyle in (
            ("Hattrick", "baseline", "#3478D4", "-"),
            ("Hattrick-PF", "candidate", "#28A45B", "--"),
        ):
            x, y = ecdf(values(rows[source], class_name), left, right)
            axis.step(
                x,
                y,
                where="post",
                color=color,
                linestyle=linestyle,
                linewidth=2.15,
                label=label,
            )
        axis.set_xlim(left, right)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(class_name)
        axis.grid(True, color="#DFE4EA", linewidth=0.75)
    axes[0].set_ylabel("CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.07, 0.005),
        ncol=2,
    )
    figure.supxlabel("NormFulfill", y=0.06)
    figure.tight_layout(rect=(0.03, 0.12, 1.0, 0.89))
    figure.savefig(HERE / "cdf.png", dpi=200, bbox_inches="tight")
    figure.savefig(HERE / "cdf.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
