from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import run_experiment as runner

import evaluate_strict2x_common_methods as common
import run_hattrick_residual_lp as residual
import run_hattrick_strict2x_research as base
from frameworks.hattrick_system import Hattrick


START, END = 400, 500
OUTPUT_DIR = runner.OUTPUT_ROOT / "true2x_epoch13_comparison"
TAIL_CHECKPOINT = (
    runner.OUTPUT_ROOT
    / "level3_validation_only"
    / "tail_directional_squared"
    / "multiplier_0p25"
    / "seed_490"
    / "best_model.pt"
)
EXISTING_HATTRICK = runner.ROOT / f"hattrick_{runner.TOPOLOGY}_{runner.K}sp.pkl"
METHODS = (
    "Hattrick (existing 60-epoch)",
    "BEST_MC",
    "Hattrick+Tail (epoch 13)",
)


def class_summary(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for method in METHODS:
        for class_name in runner.CLASSES:
            selected = [
                row for row in rows if row["method"] == method and row["class"] == class_name
            ]
            values = np.asarray([float(row["norm_fulfill"]) for row in selected])
            fulfill = np.asarray([float(row["fulfill_ratio"]) for row in selected])
            capacity = np.asarray([float(row["admitted_capacity_ratio"]) for row in selected])
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


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def nice_ticks(values: np.ndarray, count: int = 5) -> tuple[float, float, list[float]]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = max(hi - lo, 1e-3)
    raw = span / max(count - 1, 1)
    exponent = math.floor(math.log10(raw))
    fraction = raw / (10.0**exponent)
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    step = nice_fraction * (10.0**exponent)
    domain_lo = math.floor((lo - 0.04 * span) / step) * step
    domain_hi = math.ceil((hi + 0.04 * span) / step) * step
    ticks: list[float] = []
    current = domain_lo
    while current <= domain_hi + step * 0.25:
        ticks.append(float(current))
        current += step
    return float(domain_lo), float(domain_hi), ticks


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[float, float], value: str, font, fill) -> None:
    box = draw.textbbox((0, 0), value, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), value, font=font, fill=fill)


def plot_cdf(rows: list[dict], output_path: Path) -> None:
    scale = 2
    width, height = 1400 * scale, 800 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    foreground = (36, 39, 44)
    muted = (92, 99, 108)
    grid = (222, 226, 232)
    frame = (32, 36, 41)
    colors = {
        "Hattrick (existing 60-epoch)": (52, 120, 218),
        "BEST_MC": (44, 162, 95),
        "Hattrick+Tail (epoch 13)": (232, 91, 91),
    }
    title_font = load_font(29 * scale, bold=True)
    label_font = load_font(18 * scale)
    tick_font = load_font(14 * scale)
    legend_font = load_font(17 * scale)
    small_font = load_font(15 * scale)

    left_margin = 82 * scale
    right_margin = 46 * scale
    gap = 48 * scale
    plot_top = 112 * scale
    plot_bottom = 620 * scale
    panel_width = (width - left_margin - right_margin - 2 * gap) / 3
    panel_lefts = [left_margin + index * (panel_width + gap) for index in range(3)]

    draw.text(
        (left_margin, 28 * scale),
        "GEANT shared 8sp true-2x test (400–499): CDF of NormFulFill",
        font=title_font,
        fill=foreground,
    )
    draw.text((left_margin, 82 * scale), "CDF", font=small_font, fill=foreground)

    for panel_index, class_name in enumerate(runner.CLASSES):
        x0 = panel_lefts[panel_index]
        x1 = x0 + panel_width
        y0 = plot_top
        y1 = plot_bottom
        all_values = np.asarray(
            [
                float(row["norm_fulfill"])
                for row in rows
                if row["class"] == class_name
            ]
        )
        domain_lo, domain_hi, x_ticks = nice_ticks(all_values, count=6)

        for fraction in np.linspace(0.0, 1.0, 6):
            y = y1 - fraction * (y1 - y0)
            draw.line((x0, y, x1, y), fill=grid, width=1 * scale)
            if panel_index == 0:
                text_center(
                    draw,
                    (x0 - 42 * scale, y),
                    f"{fraction:.1f}",
                    tick_font,
                    muted,
                )
        for tick in x_ticks:
            x = x0 + (tick - domain_lo) / (domain_hi - domain_lo) * (x1 - x0)
            draw.line((x, y0, x, y1), fill=grid, width=1 * scale)
            text_center(draw, (x, y1 + 17 * scale), f"{tick:.2f}", tick_font, muted)
        draw.rectangle((x0, y0, x1, y1), outline=frame, width=1 * scale)

        for method in METHODS:
            values = np.sort(
                np.asarray(
                    [
                        float(row["norm_fulfill"])
                        for row in rows
                        if row["method"] == method and row["class"] == class_name
                    ]
                )
            )
            xs = np.concatenate(([values[0]], values))
            ys = np.concatenate(([0.0], np.arange(1, values.size + 1) / values.size))
            points = []
            for value, fraction in zip(xs, ys):
                x = x0 + (float(value) - domain_lo) / (domain_hi - domain_lo) * (x1 - x0)
                y = y1 - float(fraction) * (y1 - y0)
                points.append((x, y))
            draw.line(points, fill=colors[method], width=3 * scale, joint="curve")

        text_center(draw, ((x0 + x1) / 2, y1 + 48 * scale), class_name, label_font, foreground)

    text_center(
        draw,
        (width / 2, plot_bottom + 84 * scale),
        "NormFulFill",
        label_font,
        foreground,
    )

    legend_y = 744 * scale
    legend_widths = []
    legend_labels = {
        "Hattrick (existing 60-epoch)": "Hattrick",
        "BEST_MC": "BEST_MC",
        "Hattrick+Tail (epoch 13)": "Hattrick+Tail e13",
    }
    for method in METHODS:
        label = legend_labels[method]
        box = draw.textbbox((0, 0), label, font=legend_font)
        legend_widths.append(42 * scale + (box[2] - box[0]) + 45 * scale)
    legend_x = (width - sum(legend_widths)) / 2
    for method, item_width in zip(METHODS, legend_widths):
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
    if not TAIL_CHECKPOINT.exists():
        raise FileNotFoundError(TAIL_CHECKPOINT)
    if not EXISTING_HATTRICK.exists():
        raise FileNotFoundError(EXISTING_HATTRICK)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runner.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    dataset = runner.DM_Dataset_within_Cluster(props, 0, START, END)
    if int(dataset.max_source_index_read) != END - 1:
        raise RuntimeError("Evaluator read outside the 400-499 held-out test window")
    static_masks = base.move_dataset_static(dataset, device)
    if static_masks is not None:
        raise RuntimeError("Shared-path dataset unexpectedly contains class masks")

    tail_checkpoint = torch.load(TAIL_CHECKPOINT, map_location=device, weights_only=False)
    if int(tail_checkpoint.get("epoch", -1)) != 13:
        raise RuntimeError(f"Expected epoch 13, found {tail_checkpoint.get('epoch')}")
    tail_model = Hattrick(props).to(device=device, dtype=props.dtype).eval()
    tail_model.load_state_dict(tail_checkpoint["model_state_dict"])
    tail_rows, _ = base.evaluate(tail_model, props, dataset, START)
    for row in tail_rows:
        row["method"] = "Hattrick+Tail (epoch 13)"

    existing_model = torch.load(EXISTING_HATTRICK, map_location=device, weights_only=False)
    existing_model = existing_model.to(device=device, dtype=props.dtype).eval()
    existing_rows, _ = base.evaluate(existing_model, props, dataset, START)
    for row in existing_rows:
        row["method"] = "Hattrick (existing 60-epoch)"

    num_paths = int(dataset.num_pairs * runner.K)
    masks = [torch.ones(num_paths, dtype=torch.bool, device=device) for _ in range(3)]
    masks_np = [mask.cpu().numpy() for mask in masks]
    pte_info = residual.torch_pte_info(dataset)
    pte_scipy = residual.scipy_pte(dataset)
    simulator = Hattrick(props).to(device=device, dtype=props.dtype).eval()
    loader = base.data_loader(dataset, 1, False, 0)
    best_rows: list[dict] = []
    with torch.no_grad():
        for local_index, inputs in enumerate(loader):
            snapshot = START + local_index
            values = base.unpack_to_device(inputs, props)
            predicted = [
                values[index].squeeze().detach().cpu().numpy().astype(np.float64)
                for index in (3, 5, 7)
            ]
            capacity = values[1][:1].squeeze().detach().cpu().numpy().astype(np.float64)
            policies = common.best_policy(pte_scipy, predicted, masks_np, capacity)
            best_rows.extend(
                common.replay_rows(
                    simulator,
                    props,
                    dataset,
                    values,
                    policies,
                    masks,
                    pte_info,
                    snapshot,
                    "BEST_MC",
                )
            )

    rows = existing_rows + best_rows + tail_rows
    if len(rows) != len(METHODS) * len(runner.CLASSES) * (END - START):
        raise RuntimeError(f"Unexpected result row count: {len(rows)}")
    summary = class_summary(rows)
    inversions = inversion_summary(rows)
    max_capacity = max(float(row["admitted_capacity_ratio"]) for row in rows)
    max_disabled = max(float(row["disabled_flow"]) for row in rows)
    if max_capacity > 1.0001:
        raise RuntimeError(f"Capacity assertion failed: {max_capacity}")
    if max_disabled > 1e-8:
        raise RuntimeError(f"Disabled-flow assertion failed: {max_disabled}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner.write_csv(OUTPUT_DIR / "true2x_normfulfill_metrics.csv", rows)
    runner.write_json(OUTPUT_DIR / "true2x_normfulfill_summary.json", summary)
    runner.write_json(OUTPUT_DIR / "true2x_inversion_summary.json", inversions)
    plot_cdf(rows, OUTPUT_DIR / "true2x_normfulfill_cdf.png")
    runner.write_json(
        OUTPUT_DIR / "provenance.json",
        {
            "scenario": "GEANT shared paths, K=8, true 2x load",
            "window": [START, END],
            "window_role": "held-out test; explicitly authorized by the user",
            "metric_contract": (
                "exact sequential actual-TM admission; incremental MF oracle; "
                "common cumulative admitted link load/capacity"
            ),
            "dataset_max_source_index_read": int(dataset.max_source_index_read),
            "tail_checkpoint": str(TAIL_CHECKPOINT),
            "tail_checkpoint_epoch": int(tail_checkpoint["epoch"]),
            "tail_checkpoint_sha256": runner.sha256(TAIL_CHECKPOINT),
            "existing_hattrick_checkpoint": str(EXISTING_HATTRICK),
            "existing_hattrick_checkpoint_sha256": runner.sha256(EXISTING_HATTRICK),
            "best_mc": (
                "per-snapshot ESM-predicted lexicographic LP with 1e-5 cumulative-priority "
                "preservation, replayed on actual TM"
            ),
            "max_common_post_admission_mlu": max_capacity,
            "max_disabled_flow": max_disabled,
            "runtime_seconds": float(time.perf_counter() - started),
            "source_sha256": runner.sha256(Path(__file__).resolve()),
        },
    )
    print(json.dumps({"summary": summary, "inversions": inversions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
