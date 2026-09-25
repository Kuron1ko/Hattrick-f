from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.special import ndtr

import run_experiment as runner

import evaluate_true2x_epoch13_cdf as previous
import run_hattrick_strict2x_research as base
from frameworks.hattrick_system import Hattrick


START, END = 400, 500
OUTPUT_DIR = runner.OUTPUT_ROOT / "true2x_epoch13_vs_epoch23_comparison"
PRIOR_DIR = runner.OUTPUT_ROOT / "true2x_epoch13_comparison"
PRIOR_METRICS = PRIOR_DIR / "true2x_normfulfill_metrics.csv"
PRIOR_PROVENANCE = PRIOR_DIR / "provenance.json"
EPOCH23_CHECKPOINT = (
    runner.OUTPUT_ROOT
    / "level3_validation_only"
    / "baseline"
    / "seed_490"
    / "best_model.pt"
)
TAIL_CHECKPOINT = previous.TAIL_CHECKPOINT
METHODS = (
    "Hattrick (epoch 23)",
    "BEST_MC",
    "Hattrick+Tail (epoch 13)",
)
DOMAINS = {
    "High": (0.88, 1.00, [0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]),
    "Medium": (0.60, 1.10, [0.60, 0.70, 0.80, 0.90, 1.00, 1.10]),
    "Low": (0.60, 1.20, [0.60, 0.80, 1.00, 1.20]),
}


def summarize(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for method in METHODS:
        for class_name in runner.CLASSES:
            selected = [
                row for row in rows if row["method"] == method and row["class"] == class_name
            ]
            values = np.asarray([float(row["norm_fulfill"]) for row in selected])
            fulfill = np.asarray([float(row["fulfill_ratio"]) for row in selected])
            capacity = np.asarray([float(row["admitted_capacity_ratio"]) for row in selected])
            lo, hi, _ = DOMAINS[class_name]
            if values.size != END - START:
                raise RuntimeError(
                    f"Expected {END - START} {method}/{class_name} rows, got {values.size}"
                )
            output.append(
                {
                    "method": method,
                    "class": class_name,
                    "n": int(values.size),
                    "norm_fulfill_mean": float(values.mean()),
                    "norm_fulfill_p1": float(np.percentile(values, 1)),
                    "norm_fulfill_p10": float(np.percentile(values, 10)),
                    "norm_fulfill_min": float(values.min()),
                    "norm_fulfill_max": float(values.max()),
                    "plot_left_censored_fraction": float(np.mean(values < lo)),
                    "plot_right_censored_fraction": float(np.mean(values > hi)),
                    "fulfill_ratio_mean": float(fulfill.mean()),
                    "common_post_admission_mlu_mean": float(capacity.mean()),
                    "common_post_admission_mlu_max": float(capacity.max()),
                    "max_disabled_flow": float(
                        max(float(row["disabled_flow"]) for row in selected)
                    ),
                }
            )
    return output


def inversion_summary(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for method in METHODS:
        indexed = {
            (int(row["snapshot"]), row["class"]): float(row["norm_fulfill"])
            for row in rows
            if row["method"] == method
        }
        gaps = np.asarray(
            [
                indexed[(snapshot, "Low")] - indexed[(snapshot, "Medium")]
                for snapshot in range(START, END)
            ]
        )
        output.append(
            {
                "method": method,
                "inversion_gap_mean": float(gaps.mean()),
                "inversion_positive_gap_mean": float(np.maximum(gaps, 0.0).mean()),
                "inversion_violation_fraction": float(np.mean(gaps > 0.0)),
                "inversion_gap_p90": float(np.percentile(gaps, 90)),
                "inversion_gap_max": float(gaps.max()),
            }
        )
    return output


def smooth_right_censored_cdf(
    values: np.ndarray, domain_lo: float, domain_hi: float, samples: int = 700
) -> tuple[np.ndarray, np.ndarray]:
    """Display-only Gaussian-kernel CDF with values censored to the fixed plot domain."""
    clipped = np.clip(np.asarray(values, dtype=np.float64), domain_lo, domain_hi)
    grid = np.linspace(domain_lo, domain_hi, samples)
    standard_deviation = float(np.std(clipped, ddof=1))
    q25, q75 = np.percentile(clipped, [25, 75])
    robust_scale = float((q75 - q25) / 1.349)
    scales = [value for value in (standard_deviation, robust_scale) if value > 1e-12]
    scale = min(scales) if scales else (domain_hi - domain_lo) / 50.0
    bandwidth = 0.9 * scale * clipped.size ** (-0.2)
    bandwidth = float(
        np.clip(bandwidth, 0.008 * (domain_hi - domain_lo), 0.050 * (domain_hi - domain_lo))
    )
    raw = ndtr((grid[:, None] - clipped[None, :]) / bandwidth).mean(axis=1)
    denominator = max(float(raw[-1] - raw[0]), 1e-12)
    cdf = np.clip((raw - raw[0]) / denominator, 0.0, 1.0)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    return grid, cdf


def plot_smooth_cdf(rows: list[dict], summary: list[dict], output_path: Path) -> None:
    scale = 2
    width, height = 1400 * scale, 800 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    foreground = (36, 39, 44)
    muted = (92, 99, 108)
    grid_color = (222, 226, 232)
    frame = (32, 36, 41)
    colors = {
        METHODS[0]: (52, 120, 218),
        "BEST_MC": (44, 162, 95),
        "Hattrick+Tail (epoch 13)": (232, 91, 91),
    }
    title_font = previous.load_font(28 * scale, bold=True)
    label_font = previous.load_font(18 * scale)
    tick_font = previous.load_font(14 * scale)
    legend_font = previous.load_font(17 * scale)
    small_font = previous.load_font(14 * scale)

    left_margin = 86 * scale
    right_margin = 46 * scale
    gap = 54 * scale
    plot_top = 112 * scale
    plot_bottom = 610 * scale
    panel_width = (width - left_margin - right_margin - 2 * gap) / 3
    panel_lefts = [left_margin + index * (panel_width + gap) for index in range(3)]

    draw.text(
        (left_margin, 28 * scale),
        "GEANT shared 8sp true-2x test (400–499): smoothed CDF of NormFulFill",
        font=title_font,
        fill=foreground,
    )
    draw.text((left_margin, 82 * scale), "CDF", font=small_font, fill=foreground)

    for panel_index, class_name in enumerate(runner.CLASSES):
        x0 = panel_lefts[panel_index]
        x1 = x0 + panel_width
        y0 = plot_top
        y1 = plot_bottom
        domain_lo, domain_hi, x_ticks = DOMAINS[class_name]

        for fraction in np.linspace(0.0, 1.0, 6):
            y = y1 - fraction * (y1 - y0)
            draw.line((x0, y, x1, y), fill=grid_color, width=1 * scale)
            if panel_index == 0:
                previous.text_center(
                    draw,
                    (x0 - 42 * scale, y),
                    f"{fraction:.1f}",
                    tick_font,
                    muted,
                )
        for tick in x_ticks:
            x = x0 + (tick - domain_lo) / (domain_hi - domain_lo) * (x1 - x0)
            draw.line((x, y0, x, y1), fill=grid_color, width=1 * scale)
            previous.text_center(draw, (x, y1 + 17 * scale), f"{tick:.2f}", tick_font, muted)
        draw.rectangle((x0, y0, x1, y1), outline=frame, width=1 * scale)

        for method in METHODS:
            values = np.asarray(
                [
                    float(row["norm_fulfill"])
                    for row in rows
                    if row["method"] == method and row["class"] == class_name
                ]
            )
            xs, ys = smooth_right_censored_cdf(values, domain_lo, domain_hi)
            points = [
                (
                    x0 + (float(value) - domain_lo) / (domain_hi - domain_lo) * (x1 - x0),
                    y1 - float(fraction) * (y1 - y0),
                )
                for value, fraction in zip(xs, ys)
            ]
            draw.line(points, fill=colors[method], width=3 * scale, joint="curve")

        previous.text_center(
            draw, ((x0 + x1) / 2, y1 + 48 * scale), class_name, label_font, foreground
        )

        if class_name == "Low":
            right_censored = {
                row["method"]: float(row["plot_right_censored_fraction"])
                for row in summary
                if row["class"] == "Low"
            }
            left_censored = {
                row["method"]: float(row["plot_left_censored_fraction"])
                for row in summary
                if row["class"] == "Low"
            }
            note = (
                "fixed-axis censoring: "
                f"<0.60 BEST {left_censored['BEST_MC']:.0%}; "
                f">1.20 Hattrick {right_censored[METHODS[0]]:.0%}, "
                f"Tail {right_censored['Hattrick+Tail (epoch 13)']:.0%}"
            )
            previous.text_center(
                draw,
                ((x0 + x1) / 2, y1 + 75 * scale),
                note,
                small_font,
                muted,
            )

    previous.text_center(
        draw,
        (width / 2, plot_bottom + 100 * scale),
        "NormFulFill",
        label_font,
        foreground,
    )

    legend_y = 750 * scale
    hattrick_epoch = "e13" if "13" in METHODS[0] else "e23"
    legend_labels = {
        METHODS[0]: f"Hattrick {hattrick_epoch}",
        "BEST_MC": "BEST_MC",
        "Hattrick+Tail (epoch 13)": "Hattrick+Tail e13",
    }
    item_widths = []
    for method in METHODS:
        box = draw.textbbox((0, 0), legend_labels[method], font=legend_font)
        item_widths.append(42 * scale + box[2] - box[0] + 48 * scale)
    legend_x = (width - sum(item_widths)) / 2
    for method, item_width in zip(METHODS, item_widths):
        draw.line(
            (legend_x, legend_y, legend_x + 32 * scale, legend_y),
            fill=colors[method],
            width=4 * scale,
        )
        draw.text(
            (legend_x + 42 * scale, legend_y - 13 * scale),
            legend_labels[method],
            font=legend_font,
            fill=foreground,
        )
        legend_x += item_width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width // scale, height // scale), Image.Resampling.LANCZOS).save(
        output_path, format="PNG", optimize=True
    )


def main() -> None:
    started = time.perf_counter()
    for path in (PRIOR_METRICS, PRIOR_PROVENANCE, EPOCH23_CHECKPOINT, TAIL_CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(path)
    prior_provenance = json.loads(PRIOR_PROVENANCE.read_text(encoding="utf-8"))
    if prior_provenance.get("window") != [START, END]:
        raise RuntimeError("Prior BEST_MC/Tail cache does not use 400-499")
    if int(prior_provenance.get("tail_checkpoint_epoch", -1)) != 13:
        raise RuntimeError("Prior Tail cache is not epoch 13")
    if prior_provenance.get("tail_checkpoint_sha256") != runner.sha256(TAIL_CHECKPOINT):
        raise RuntimeError("Tail checkpoint hash differs from prior common evaluation")

    cached_rows = runner.read_csv(PRIOR_METRICS)
    reused_rows: list[dict] = []
    for row in cached_rows:
        if row["method"] == "BEST_MC":
            copied = dict(row)
            copied["method"] = "BEST_MC"
            reused_rows.append(copied)
        elif row["method"] == "Hattrick+Tail (epoch 13)":
            copied = dict(row)
            copied["method"] = "Hattrick+Tail (epoch 13)"
            reused_rows.append(copied)
    if len(reused_rows) != 600:
        raise RuntimeError(f"Expected 600 cached BEST_MC/Tail rows, got {len(reused_rows)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runner.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    dataset = runner.DM_Dataset_within_Cluster(props, 0, START, END)
    if int(dataset.max_source_index_read) != END - 1:
        raise RuntimeError("Evaluator read outside the 400-499 held-out test window")
    if base.move_dataset_static(dataset, device) is not None:
        raise RuntimeError("Shared-path dataset unexpectedly contains class masks")

    checkpoint = torch.load(EPOCH23_CHECKPOINT, map_location=device, weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 23:
        raise RuntimeError(f"Expected epoch 23, found {checkpoint.get('epoch')}")
    model = Hattrick(props).to(device=device, dtype=props.dtype).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch23_rows, _ = base.evaluate(model, props, dataset, START)
    for row in epoch23_rows:
        row["method"] = "Hattrick (epoch 23)"

    rows = epoch23_rows + reused_rows
    if len(rows) != len(METHODS) * len(runner.CLASSES) * (END - START):
        raise RuntimeError(f"Unexpected result row count: {len(rows)}")
    summary = summarize(rows)
    inversions = inversion_summary(rows)
    max_capacity = max(float(row["admitted_capacity_ratio"]) for row in rows)
    max_disabled = max(float(row["disabled_flow"]) for row in rows)
    if max_capacity > 1.0001:
        raise RuntimeError(f"Capacity assertion failed: {max_capacity}")
    if max_disabled > 1e-8:
        raise RuntimeError(f"Disabled-flow assertion failed: {max_disabled}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner.write_csv(OUTPUT_DIR / "true2x_epoch13_vs_epoch23_metrics.csv", rows)
    runner.write_json(OUTPUT_DIR / "true2x_epoch13_vs_epoch23_summary.json", summary)
    runner.write_json(OUTPUT_DIR / "true2x_epoch13_vs_epoch23_inversion.json", inversions)
    plot_smooth_cdf(rows, summary, OUTPUT_DIR / "true2x_epoch13_vs_epoch23_smooth_cdf.png")
    runner.write_json(
        OUTPUT_DIR / "provenance.json",
        {
            "scenario": "GEANT shared paths, K=8, true 2x load",
            "window": [START, END],
            "window_role": "held-out test; explicitly authorized by the user",
            "metric_contract": prior_provenance["metric_contract"],
            "plot_contract": {
                "curve": "Gaussian-kernel smoothed CDF; display only",
                "fixed_domains": DOMAINS,
                "censoring": (
                    "values outside fixed reference axes are right/left-censored for display only; "
                    "raw CSV and summary remain uncensored"
                ),
            },
            "dataset_max_source_index_read": int(dataset.max_source_index_read),
            "hattrick_epoch23_checkpoint": str(EPOCH23_CHECKPOINT),
            "hattrick_epoch23": int(checkpoint["epoch"]),
            "hattrick_epoch23_sha256": runner.sha256(EPOCH23_CHECKPOINT),
            "tail_epoch13_checkpoint": str(TAIL_CHECKPOINT),
            "tail_epoch13_sha256": runner.sha256(TAIL_CHECKPOINT),
            "reused_common_metrics": str(PRIOR_METRICS),
            "reused_common_metrics_sha256": runner.sha256(PRIOR_METRICS),
            "max_common_post_admission_mlu": max_capacity,
            "max_disabled_flow": max_disabled,
            "runtime_seconds": float(time.perf_counter() - started),
            "source_sha256": runner.sha256(Path(__file__).resolve()),
        },
    )
    print(json.dumps({"summary": summary, "inversions": inversions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
