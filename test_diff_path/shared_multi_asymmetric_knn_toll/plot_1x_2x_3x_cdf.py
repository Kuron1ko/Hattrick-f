from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
OUTPUT_DIR = HERE / "artifacts" / "combined_1x_2x_3x"
SOURCES = {
    "1×": HERE / "artifacts" / "1x_level4" / "report.json",
    "2×": TEST_DIR
    / "shared2x_teacher_toll_distill"
    / "artifacts"
    / "asymmetric_knn_level4"
    / "final_rows.json",
    "3×": HERE / "artifacts" / "3x_level4" / "report.json",
}
CLASSES = ("High", "Medium", "Low")
LOADS = ("1×", "2×", "3×")
SERIES = (("Hattrick", "baseline"), ("Hattrick-LCR", "candidate"))
COLORS = {"Hattrick": "#3478D4", "Hattrick-LCR": "#28A45B"}
LINESTYLES = {"Hattrick": "-", "Hattrick-LCR": "--"}
AXES = {
    "High": ((0.88, 1.005), np.arange(0.88, 1.001, 0.02), "%.2f"),
    "Medium": ((0.85, 1.20), np.arange(0.85, 1.201, 0.05), "%.2f"),
    "Low": ((0.88, 1.48), np.arange(0.90, 1.401, 0.10), "%.1f"),
}


def load_payloads() -> dict[str, dict]:
    payloads = {
        load: json.loads(path.read_text(encoding="utf-8"))
        for load, path in SOURCES.items()
    }
    for load, payload in payloads.items():
        for source in ("baseline", "candidate"):
            for class_name in CLASSES:
                count = sum(
                    row["class"] == class_name for row in payload["rows"][source]
                )
                if count != 100:
                    raise ValueError(
                        f"Expected 100 {load}/{source}/{class_name} rows, found {count}"
                    )
    return payloads


def values(payload: dict, source: str, class_name: str) -> np.ndarray:
    return np.asarray(
        [
            float(row["norm_fulfill"])
            for row in payload["rows"][source]
            if row["class"] == class_name
        ],
        dtype=np.float64,
    )


def exact_ecdf(samples: np.ndarray, left: float, right: float):
    ordered = np.sort(samples)
    x = np.concatenate(([left, ordered[0]], ordered, [right]))
    y = np.concatenate(
        ((0.0, 0.0), np.arange(1, ordered.size + 1) / ordered.size, (1.0,))
    )
    return x, y


def summarize(samples: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "p1": float(np.percentile(samples, 1)),
        "p10": float(np.percentile(samples, 10)),
    }


def write_combined_report(payloads: dict[str, dict]) -> None:
    report = {
        "method": "priority-asymmetric local case retrieval for edge tolls",
        "display_name": "Hattrick-LCR",
        "strict_esm_inference": True,
        "level4_range": [400, 500],
        "frozen_hyperparameters": {
            "selected_on": "2x Level-2 only",
            "medium_neighbors": 2,
            "low_neighbors": 32,
            "toll_scale": 1.0,
        },
        "results": {},
    }
    for load, payload in payloads.items():
        load_result = {}
        for class_name in CLASSES:
            baseline = summarize(values(payload, "baseline", class_name))
            candidate = summarize(values(payload, "candidate", class_name))
            load_result[class_name] = {
                "Hattrick": baseline,
                "Hattrick-LCR": candidate,
                "delta": {
                    key: candidate[key] - baseline[key]
                    for key in ("mean", "p1", "p10")
                },
            }
        report["results"][load] = load_result
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "combined_level4.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def main() -> None:
    payloads = load_payloads()
    write_combined_report(payloads)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.edgecolor": "#2B2B2B",
            "axes.linewidth": 0.85,
            "xtick.color": "#5A6572",
            "ytick.color": "#5A6572",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(3, 3, figsize=(14.8, 10.8), sharey=True)
    figure.suptitle(
        "GEANT 8sp — CDF of NormFulfill under strict ESM inference",
        x=0.068,
        y=0.985,
        ha="left",
        fontsize=19,
        fontweight="semibold",
    )
    figure.text(
        0.068,
        0.953,
        "Load-specific Hattrick versus Hattrick-LCR · Level-4 snapshots 400–499 · exact empirical CDF",
        ha="left",
        color="#5A6572",
        fontsize=10.5,
    )

    for row_index, load in enumerate(LOADS):
        for column_index, class_name in enumerate(CLASSES):
            axis = axes[row_index, column_index]
            (left, right), ticks, tick_format = AXES[class_name]
            for label, source in SERIES:
                x, y = exact_ecdf(
                    values(payloads[load], source, class_name), left, right
                )
                axis.step(
                    x,
                    y,
                    where="post",
                    color=COLORS[label],
                    linestyle=LINESTYLES[label],
                    linewidth=2.15 if label == "Hattrick" else 2.0,
                    label=label,
                    zorder=2 if label == "Hattrick" else 3,
                )
            axis.set_xlim(left, right)
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks(ticks)
            axis.xaxis.set_major_formatter(FormatStrFormatter(tick_format))
            axis.set_yticks(np.linspace(0.0, 1.0, 6))
            axis.grid(True, color="#DFE4EA", linewidth=0.75, zorder=0)
            axis.set_xlabel(class_name)
            if row_index == 0:
                axis.set_title(class_name, pad=10, fontweight="semibold")
            if column_index == 0:
                axis.set_ylabel(f"{load}\nCDF", labelpad=13, fontweight="semibold")
            else:
                axis.set_ylabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.068, 0.008),
        ncol=2,
        fontsize=11,
    )
    figure.supxlabel("NormFulfill", y=0.048, fontsize=12)
    figure.tight_layout(rect=(0.045, 0.075, 0.995, 0.93), h_pad=2.0, w_pad=1.25)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / "cdf_1x_2x_3x.png", dpi=200, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "cdf_1x_2x_3x.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
