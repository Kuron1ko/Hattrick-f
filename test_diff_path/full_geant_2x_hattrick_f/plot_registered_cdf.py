from __future__ import annotations

"""Plot registered full-GEANT checkpoints by dataset, method name, and epoch."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from plot_test_cdf import font, monotone_cdf


THIS_DIR = Path(__file__).resolve().parent
REGISTERED_METHODS_PATH = THIS_DIR / "registered_methods.py"
CLASSES = ("High", "Medium", "Low")
TEST_RANGE = (7500, 10200)
COLORS = (
    "#2F7ED8",
    "#F28E2B",
    "#2CA25F",
    "#9467BD",
    "#D62728",
    "#8C6D31",
)
PATTERNS: tuple[tuple[int, ...] | None, ...] = (
    None,
    (13, 8),
    (13, 6, 3, 6),
    (3, 6),
    (19, 7),
    (19, 6, 3, 6, 3, 6),
)
FIXED_AXES = {
    "1x": {
        "High": ((0.88, 1.0), (0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0)),
        "Medium": ((0.6, 1.1), (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)),
        "Low": ((0.6, 1.2), (0.6, 0.8, 1.0, 1.2)),
    },
    "2x": {
        "High": ((0.9, 1.0), (0.9, 0.92, 0.94, 0.96, 0.98, 1.0)),
        "Medium": ((0.9, 1.02), (0.9, 0.92, 0.94, 0.96, 0.98, 1.0)),
        "Low": ((0.9, 1.2), (0.9, 1.0, 1.1, 1.2)),
    },
    "3x": {
        "High": ((0.88, 1.0), (0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0)),
        "Medium": ((0.5, 1.05), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
        "Low": ((0.5, 1.8), (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)),
    },
}


def load_registered_methods():
    module_name = "full_geant_registered_plot_methods"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, REGISTERED_METHODS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {REGISTERED_METHODS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


registered = load_registered_methods()


def load_metrics(path: Path, test_range: tuple[int, int]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[tuple[int, float]]] = {
        class_name: [] for class_name in CLASSES
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            class_name = row["class"]
            if class_name in grouped:
                grouped[class_name].append(
                    (int(row["snapshot"]), float(row["norm_fulfill"]))
                )

    expected = list(range(*test_range))
    result: dict[str, np.ndarray] = {}
    for class_name, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        if [snapshot for snapshot, _value in rows] != expected:
            raise RuntimeError(
                f"{path}: {class_name} is not the complete registered test split"
            )
        values = np.asarray([value for _snapshot, value in rows], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{path}: {class_name} contains non-finite values")
        if class_name == "High":
            if float(values.max()) > 1.00001:
                raise RuntimeError(
                    f"{path}: High NormFulFill exceeds 1 beyond solver tolerance"
                )
            values = np.minimum(values, 1.0)
        result[class_name] = values
    return result


def derive_axis(
    dataset_name: str, class_name: str, arrays: list[np.ndarray]
) -> tuple[tuple[float, float], tuple[float, ...]]:
    limits, ticks = FIXED_AXES[dataset_name][class_name]
    step = float(ticks[1] - ticks[0])
    data_min = min(float(values.min()) for values in arrays)
    data_max = max(float(values.max()) for values in arrays)
    lower = min(limits[0], math.floor((data_min - 0.2 * step) / step) * step)
    upper = max(limits[1], math.ceil((data_max + 0.2 * step) / step) * step)
    if class_name == "High":
        upper = 1.0
    dynamic_ticks = tuple(
        float(value)
        for value in np.arange(lower, upper + 0.5 * step, step)
    )
    return (float(lower), float(upper)), dynamic_ticks


def patterned_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: str,
    width: int,
    pattern: tuple[int, ...] | None,
) -> None:
    if pattern is None:
        draw.line(points, fill=fill, width=width, joint="curve")
        return
    pattern_index = 0
    remaining = float(pattern[0])
    drawing = True
    for start, end in zip(points, points[1:]):
        x0, y0 = start
        x1, y1 = end
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0:
            continue
        used = 0.0
        while used < length:
            take = min(remaining, length - used)
            a = used / length
            b = (used + take) / length
            if drawing:
                draw.line(
                    (
                        x0 + (x1 - x0) * a,
                        y0 + (y1 - y0) * a,
                        x0 + (x1 - x0) * b,
                        y0 + (y1 - y0) * b,
                    ),
                    fill=fill,
                    width=width,
                )
            used += take
            remaining -= take
            if remaining <= 1e-9:
                pattern_index = (pattern_index + 1) % len(pattern)
                remaining = float(pattern[pattern_index])
                drawing = pattern_index % 2 == 0


def tick_text(value: float, step: float) -> str:
    if step >= 0.1:
        return f"{value:.1f}"
    if step >= 0.01:
        return f"{value:.2f}"
    return f"{value:.3f}"


def plot(series: list[dict], output: Path, dataset_name: str) -> None:
    if not 1 <= len(series) <= len(COLORS):
        raise ValueError(f"Plot supports between 1 and {len(COLORS)} series")
    axes = {
        class_name: derive_axis(
            dataset_name,
            class_name,
            [item["values"][class_name] for item in series],
        )
        for class_name in CLASSES
    }

    supersampling = 3
    width, height = 2400, 900
    image = Image.new(
        "RGB", (width * supersampling, height * supersampling), "white"
    )
    draw = ImageDraw.Draw(image)

    def scale(value: float) -> int:
        return int(round(value * supersampling))

    title_font = font(scale(34), bold=True)
    panel_font = font(scale(23))
    label_font = font(scale(19))
    tick_font = font(scale(15))
    legend_font = font(scale(17))
    foreground = "#1F2328"
    muted = "#4E5968"
    grid = "#DCE2E9"

    draw.text(
        (scale(100), scale(38)),
        "CDF of NormFulFill",
        fill=foreground,
        font=title_font,
    )
    left, right, top, bottom = 105, 55, 145, 220
    gap = 90
    panel_width = (width - left - right - 2 * gap) / 3
    panel_height = height - top - bottom

    for panel_index, class_name in enumerate(CLASSES):
        x0 = left + panel_index * (panel_width + gap)
        x1 = x0 + panel_width
        y0 = top
        y1 = top + panel_height
        limits, ticks = axes[class_name]
        step = ticks[1] - ticks[0] if len(ticks) > 1 else limits[1] - limits[0]

        for probability in np.linspace(0.0, 1.0, 6):
            py = y1 - probability * panel_height
            draw.line(
                (scale(x0), scale(py), scale(x1), scale(py)),
                fill=grid,
                width=scale(1),
            )
            draw.text(
                (scale(x0 - 16), scale(py)),
                f"{probability:.1f}",
                fill=muted,
                font=tick_font,
                anchor="rm",
            )

        for tick in ticks:
            if tick < limits[0] - 1e-9 or tick > limits[1] + 1e-9:
                continue
            px = x0 + (tick - limits[0]) / (limits[1] - limits[0]) * panel_width
            draw.line(
                (scale(px), scale(y1), scale(px), scale(y1 + 8)),
                fill=foreground,
                width=scale(1),
            )
            draw.text(
                (scale(px), scale(y1 + 16)),
                tick_text(tick, step),
                fill=muted,
                font=tick_font,
                anchor="ma",
            )

        draw.line(
            (scale(x0), scale(y0), scale(x0), scale(y1)),
            fill=foreground,
            width=scale(1),
        )
        draw.line(
            (scale(x0), scale(y1), scale(x1), scale(y1)),
            fill=foreground,
            width=scale(1),
        )
        draw.text(
            (scale((x0 + x1) / 2), scale(112)),
            f"{dataset_name} load · {class_name}",
            fill=foreground,
            font=panel_font,
            anchor="mm",
        )
        draw.text(
            (scale((x0 + x1) / 2), scale(y1 + 68)),
            "Normalized FulfillRatio",
            fill=foreground,
            font=label_font,
            anchor="mm",
        )

        y_label = Image.new(
            "RGBA", (scale(100), scale(42)), (255, 255, 255, 0)
        )
        y_draw = ImageDraw.Draw(y_label)
        y_draw.text(
            (scale(50), scale(21)),
            "CDF",
            fill=foreground,
            font=label_font,
            anchor="mm",
        )
        y_label = y_label.rotate(
            90, expand=True, resample=Image.Resampling.BICUBIC
        )
        image.paste(
            y_label,
            (
                scale(x0 - 80),
                scale((y0 + y1) / 2 - y_label.height / supersampling / 2),
            ),
            y_label,
        )

        for index, item in enumerate(series):
            curve_x, curve_y = monotone_cdf(item["values"][class_name], limits)
            if class_name == "High":
                # High is capped at one, so the exact right boundary is CDF=1.
                curve_y[-1] = 1.0
            points = [
                (
                    scale(
                        x0
                        + (x - limits[0])
                        / (limits[1] - limits[0])
                        * panel_width
                    ),
                    scale(y1 - y * panel_height),
                )
                for x, y in zip(curve_x, curve_y)
            ]
            pattern = PATTERNS[index]
            patterned_polyline(
                draw,
                points,
                fill=COLORS[index],
                width=scale(3),
                pattern=(
                    None
                    if pattern is None
                    else tuple(scale(length) for length in pattern)
                ),
            )

    legend_columns = min(len(series), 3)
    legend_column_width = (width - 220) / legend_columns
    for index, item in enumerate(series):
        row = index // legend_columns
        column = index % legend_columns
        x0 = 110 + column * legend_column_width
        y0 = height - 70 + row * 32
        pattern = PATTERNS[index]
        patterned_polyline(
            draw,
            [(scale(x0), scale(y0)), (scale(x0 + 70), scale(y0))],
            fill=COLORS[index],
            width=scale(3),
            pattern=(
                None
                if pattern is None
                else tuple(scale(length) for length in pattern)
            ),
        )
        draw.text(
            (scale(x0 + 88), scale(y0)),
            item["label"],
            fill=foreground,
            font=legend_font,
            anchor="lm",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(output, dpi=(240, 240))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each requested model, infer if needed and create one "
            "three-panel CDF against the selected dataset's Hattrick baseline"
        )
    )
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument(
        "--models",
        help=(
            "inline JSON list or path to a JSON file; every item must be "
            '{"name": "METHOD", "epoch": EPOCH}'
        ),
    )
    model_source.add_argument(
        "--method",
        nargs="+",
        help=(
            "one or more registered method names; pair with --epoch. "
            "A single epoch is broadcast to every method"
        ),
    )
    parser.add_argument(
        "--epoch",
        nargs="+",
        type=int,
        help="one epoch per --method, or one epoch shared by all methods",
    )
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument(
        "--dataset",
        default="2x",
        help="registered dataset to infer and plot, for example 1x, 2x, or 3x",
    )
    parser.add_argument(
        "--baseline-epoch",
        type=int,
        default=None,
        help=(
            "override the dataset's registered Hattrick baseline epoch; by "
            "by default the validation-selected epoch is read from the registry"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="re-run inference even if an audited result already exists",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for the generated CDF images and manifest",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("batch size and learning rate must be positive")
    if args.baseline_epoch is not None and args.baseline_epoch <= 0:
        parser.error("--baseline-epoch must be positive")
    try:
        args.dataset, args.dataset_config, _registry = registered.registered_dataset(
            args.dataset
        )
    except KeyError as error:
        parser.error(str(error))
    if args.models is not None:
        if args.epoch is not None:
            parser.error("--epoch is only valid with --method")
        try:
            args.model_list = registered.parse_model_list_argument(args.models)
        except ValueError as error:
            parser.error(str(error))
    else:
        if args.epoch is None:
            parser.error("--method requires --epoch")
        if any(epoch <= 0 for epoch in args.epoch):
            parser.error("every --epoch value must be positive")
        if len(args.epoch) == 1:
            epochs = args.epoch * len(args.method)
        elif len(args.epoch) == len(args.method):
            epochs = args.epoch
        else:
            parser.error(
                "--epoch must contain one value or exactly one value per --method"
            )
        args.model_list = registered.normalize_model_list(
            [
                {"name": name, "epoch": epoch}
                for name, epoch in zip(args.method, epochs)
            ]
        )
    return args


def model_version_name(name: str, epoch: int) -> str:
    """Return a readable, filesystem-safe ``Method-epoch`` identifier."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    if not normalized:
        raise ValueError(f"Method name {name!r} cannot form an output filename")
    return f"{normalized}-{epoch}"


def batch_id(models: list[dict], dataset_name: str) -> str:
    encoded = json.dumps(
        {"dataset": dataset_name, "models": models},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def baseline_epoch(args: argparse.Namespace) -> tuple[str, int]:
    baseline = args.dataset_config.get("baseline")
    if not isinstance(baseline, dict) or "method" not in baseline:
        raise RuntimeError(f"Dataset {args.dataset} has no registered baseline")
    method = str(baseline["method"])
    if args.baseline_epoch is not None:
        return method, args.baseline_epoch
    if "epoch" in baseline:
        return method, int(baseline["epoch"])
    template = baseline.get("selection_file")
    field = str(baseline.get("selection_field", "selected_epoch"))
    if not isinstance(template, str):
        raise RuntimeError(f"Dataset {args.dataset} has no baseline epoch source")
    path = registered.format_path(template, seed=args.seed, epoch=0)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing validation selection for {args.dataset}: {path}"
        )
    value = registered.read_json(path)
    if not isinstance(value, dict) or field not in value:
        raise RuntimeError(f"Baseline selection {path} has no field {field!r}")
    return method, int(value[field])


def main() -> None:
    args = parse_args()
    try:
        baseline_method, selected_baseline_epoch = baseline_epoch(args)
        baseline = registered.infer_registered_method(
            baseline_method,
            selected_baseline_epoch,
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device_name=args.device,
            dataset_name=args.dataset,
            force=args.force_inference,
        )
        requested_results = registered.infer_registered_models(
            args.model_list,
            seed=args.seed,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device_name=args.device,
            dataset_name=args.dataset,
            force=args.force_inference,
        )
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    test_range = tuple(int(value) for value in args.dataset_config["test"])
    baseline_values = load_metrics(Path(baseline["metrics"]), test_range)
    output_dir = (
        args.output_dir
        or registered.RESULT_ROOT / "plots" / args.dataset
        / f"batch_{batch_id(args.model_list, args.dataset)}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[dict] = []
    for index, (spec, result) in enumerate(
        zip(args.model_list, requested_results), start=1
    ):
        figure_name = "_".join(
            (
                model_version_name(baseline["method"], baseline["epoch"]),
                model_version_name(result["method"], result["epoch"]),
            )
        )
        output = output_dir / f"{figure_name}.png"
        series = [
            {
                "label": baseline["method"],
                "values": baseline_values,
            },
            {
                "label": result["method"],
                "values": load_metrics(Path(result["metrics"]), test_range),
            },
        ]
        plot(series, output, args.dataset)
        figures.append(
            {
                "request": spec,
                "candidate": result,
                "baseline": baseline,
                "name": figure_name,
                "output": str(output),
                "series": [item["label"] for item in series],
            }
        )

    manifest = output_dir / "manifest.json"
    registered.write_json(
        manifest,
        {
            "dataset": args.dataset,
            "models": args.model_list,
            "baseline": {
                "name": baseline["method"],
                "epoch": baseline["epoch"],
            },
            "figure_count": len(figures),
            "figures": figures,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "dataset": args.dataset,
                "manifest": str(manifest),
                "figure_count": len(figures),
                "outputs": [item["output"] for item in figures],
                "cache_reused": {
                    f"{baseline['method']}_epoch_{baseline['epoch']}": baseline[
                        "cache_reused"
                    ],
                    "requested": [
                        item["candidate"]["cache_reused"] for item in figures
                    ],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
