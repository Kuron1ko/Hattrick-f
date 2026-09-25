from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ARTIFACT_DIR = HERE / "artifacts" / "asymmetric_knn_level4"


def ecdf(values: np.ndarray, left: float, right: float):
    ordered = np.sort(values)
    x = np.concatenate([[left], ordered, [right]])
    y = np.concatenate(
        [[0.0], np.arange(1, len(ordered) + 1) / len(ordered), [1.0]]
    )
    return x, y


def main() -> None:
    payload = json.loads(
        (ARTIFACT_DIR / "final_rows.json").read_text(encoding="utf-8")
    )
    rows = payload["rows"]
    classes = ("High", "Medium", "Low")
    ranges = {
        "High": (0.985, 1.001),
        "Medium": (0.84, 1.01),
        "Low": (0.88, 1.42),
    }
    colors = {"Hattrick": "#3478d4", "Asymmetric kNN toll": "#28a45b"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), sharey=True)
    for axis, class_name in zip(axes, classes):
        left, right = ranges[class_name]
        for label, source in (
            ("Hattrick", "baseline"),
            ("Asymmetric kNN toll", "candidate"),
        ):
            values = np.asarray(
                [
                    row["norm_fulfill"]
                    for row in rows[source]
                    if row["class"] == class_name
                ]
            )
            x, y = ecdf(values, left, right)
            axis.step(
                x,
                y,
                where="post",
                linewidth=2.1,
                color=colors[label],
                label=label,
            )
        axis.set_xlim(left, right)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(class_name, fontsize=12)
        axis.grid(True, color="#dfe3e8", linewidth=0.7)
        axis.tick_params(labelsize=10)
    axes[0].set_ylabel("CDF", fontsize=12)
    fig.suptitle(
        "GEANT 2× strict-ESM: CDF of NormFulfill",
        fontsize=18,
        x=0.08,
        ha="left",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.08, 0.005),
        ncol=2,
        frameon=False,
    )
    fig.supxlabel("NormFulfill", y=0.06, fontsize=12)
    fig.tight_layout(rect=(0.02, 0.11, 1.0, 0.92))
    fig.savefig(ARTIFACT_DIR / "final_cdf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
