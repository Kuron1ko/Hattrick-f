from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator

import _workspace as ws


COLORS = ("#2F7ED8", "#F28E2B", "#2CA25F", "#9467BD")
PATTERNS: tuple[tuple[int, ...] | None, ...] = (
    None,
    (13, 8),
    (13, 6, 3, 6),
    (3, 6),
)
FIXED_AXES = {
    "1x": {
        "High": ((0.88, 1.0), (0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0)),
        "Medium": ((0.6, 1.1), (0.6, 0.7, 0.8, 0.9, 1.0, 1.1)),
        "Low": ((0.6, 1.2), (0.6, 0.8, 1.0, 1.2)),
    },
    "2x": {
        "High": ((0.90, 1.0), (0.90, 0.92, 0.94, 0.96, 0.98, 1.0)),
        "Medium": ((0.84, 1.04), (0.84, 0.88, 0.92, 0.96, 1.0, 1.04)),
        "Low": ((0.90, 1.2), (0.90, 1.0, 1.1, 1.2)),
    },
    "3x": {
        "High": ((0.88, 1.0), (0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0)),
        "Medium": ((0.5, 1.05), (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
        "Low": ((0.5, 1.8), (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)),
    },
}


def load_metrics(path: Path, dataset: str) -> dict[str, np.ndarray]:
    grouped: dict[str, list[tuple[int, float]]] = {
        class_name: [] for class_name in ws.CLASSES
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            class_name = row["class"]
            if class_name in grouped:
                grouped[class_name].append(
                    (int(row["snapshot"]), float(row["norm_fulfill"]))
                )
    expected = list(range(*ws.DATASETS[dataset]["test"]))
    result: dict[str, np.ndarray] = {}
    for class_name, rows in grouped.items():
        rows.sort(key=lambda item: item[0])
        if [snapshot for snapshot, _value in rows] != expected:
            raise RuntimeError(f"{path}: incomplete {class_name} test split")
        values = np.asarray([value for _snapshot, value in rows], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"{path}: non-finite {class_name} values")
        if class_name == "High":
            if float(values.max()) > 1.00001:
                raise RuntimeError(f"{path}: High NormFulFill exceeds solver tolerance")
            values = np.minimum(values, 1.0)
        result[class_name] = values
    return result


def monotone_cdf(
    values: np.ndarray, limits: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Shape-preserving continuous empirical CDF with flat end segments."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    probabilities = (np.arange(values.size, dtype=np.float64) + 0.5) / values.size
    rounded = np.round(values, decimals=10)
    unique_x, inverse = np.unique(rounded, return_inverse=True)
    unique_y = np.zeros(unique_x.size, dtype=np.float64)
    for index in range(unique_x.size):
        unique_y[index] = probabilities[inverse == index].mean()
    span = max(float(values[-1] - values[0]), 1e-6)
    shoulder = max(span * 0.0125, (limits[1] - limits[0]) * 0.001)
    support_x = np.concatenate(
        ([values[0] - shoulder], unique_x, [values[-1] + shoulder])
    )
    support_y = np.concatenate(([0.0], unique_y, [1.0]))
    interpolator = PchipInterpolator(support_x, support_y, extrapolate=False)
    grid = np.linspace(limits[0], limits[1], 1600)
    cdf = np.zeros_like(grid)
    interior = (grid > support_x[0]) & (grid < support_x[-1])
    cdf[interior] = interpolator(grid[interior])
    cdf[grid >= support_x[-1]] = 1.0
    return grid, np.clip(cdf, 0.0, 1.0)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    windows = Path("C:/Windows/Fonts")
    candidates = (
        (windows / "arialbd.ttf", windows / "segoeuib.ttf")
        if bold
        else (windows / "arial.ttf", windows / "segoeui.ttf")
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.truetype(
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size
    )


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


def derive_axis(
    dataset: str, class_name: str, arrays: list[np.ndarray]
) -> tuple[tuple[float, float], tuple[float, ...]]:
    limits, ticks = FIXED_AXES[dataset][class_name]
    # The 2x Medium panel is intentionally a fixed comparison window.  Keep
    # the requested 0.84--1.04 limits even when a tail falls outside it.
    if dataset == "2x" and class_name == "Medium":
        return limits, ticks
    step = float(ticks[1] - ticks[0])
    data_min = min(float(values.min()) for values in arrays)
    data_max = max(float(values.max()) for values in arrays)
    lower = min(limits[0], math.floor((data_min - 0.2 * step) / step) * step)
    upper = max(limits[1], math.ceil((data_max + 0.2 * step) / step) * step)
    if class_name == "High":
        upper = 1.0
    dynamic_ticks = tuple(
        float(value) for value in np.arange(lower, upper + 0.5 * step, step)
    )
    return (float(lower), float(upper)), dynamic_ticks


def tick_text(value: float, step: float) -> str:
    if step >= 0.1:
        return f"{value:.1f}"
    if step >= 0.01:
        return f"{value:.2f}"
    return f"{value:.3f}"


def plot(series: list[dict], output: Path, dataset: str) -> None:
    axes = {
        class_name: derive_axis(
            dataset,
            class_name,
            [item["values"][class_name] for item in series],
        )
        for class_name in ws.CLASSES
    }
    supersampling = 3
    width, height = 2400, 900
    image = Image.new("RGB", (width * supersampling, height * supersampling), "white")
    draw = ImageDraw.Draw(image)

    def scale(value: float) -> int:
        return int(round(value * supersampling))

    title_font = font(scale(34), bold=True)
    panel_font = font(scale(23))
    label_font = font(scale(19))
    tick_font = font(scale(15))
    legend_font = font(scale(17))
    foreground, muted, grid = "#1F2328", "#4E5968", "#DCE2E9"
    draw.text((scale(100), scale(38)), "CDF of NormFulFill", fill=foreground, font=title_font)
    left, right, top, bottom, gap = 105, 55, 145, 220, 90
    panel_width = (width - left - right - 2 * gap) / 3
    panel_height = height - top - bottom
    for panel_index, class_name in enumerate(ws.CLASSES):
        x0 = left + panel_index * (panel_width + gap)
        x1 = x0 + panel_width
        y0, y1 = top, top + panel_height
        limits, ticks = axes[class_name]
        step = ticks[1] - ticks[0]
        for probability in np.linspace(0.0, 1.0, 6):
            py = y1 - probability * panel_height
            draw.line((scale(x0), scale(py), scale(x1), scale(py)), fill=grid, width=scale(1))
            draw.text((scale(x0 - 16), scale(py)), f"{probability:.1f}", fill=muted, font=tick_font, anchor="rm")
        for tick in ticks:
            if tick < limits[0] - 1e-9 or tick > limits[1] + 1e-9:
                continue
            px = x0 + (tick - limits[0]) / (limits[1] - limits[0]) * panel_width
            draw.line((scale(px), scale(y1), scale(px), scale(y1 + 8)), fill=foreground, width=scale(1))
            draw.text((scale(px), scale(y1 + 16)), tick_text(tick, step), fill=muted, font=tick_font, anchor="ma")
        draw.line((scale(x0), scale(y0), scale(x0), scale(y1)), fill=foreground, width=scale(1))
        draw.line((scale(x0), scale(y1), scale(x1), scale(y1)), fill=foreground, width=scale(1))
        draw.text((scale((x0 + x1) / 2), scale(112)), f"{dataset} load · {class_name}", fill=foreground, font=panel_font, anchor="mm")
        draw.text((scale((x0 + x1) / 2), scale(y1 + 68)), "Normalized FulfillRatio", fill=foreground, font=label_font, anchor="mm")
        y_label = Image.new("RGBA", (scale(100), scale(42)), (255, 255, 255, 0))
        ImageDraw.Draw(y_label).text((scale(50), scale(21)), "CDF", fill=foreground, font=label_font, anchor="mm")
        y_label = y_label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
        image.paste(y_label, (scale(x0 - 80), scale((y0 + y1) / 2 - y_label.height / supersampling / 2)), y_label)
        for index, item in enumerate(series):
            curve_x, curve_y = monotone_cdf(item["values"][class_name], limits)
            if class_name == "High":
                curve_y[-1] = 1.0
            points = [
                (
                    scale(x0 + (x - limits[0]) / (limits[1] - limits[0]) * panel_width),
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
                pattern=None if pattern is None else tuple(scale(length) for length in pattern),
            )
    for index, item in enumerate(series):
        # Keep the legend entries together at the lower-left, matching the
        # paper figure instead of spreading two labels across the full canvas.
        x0 = 110 + index * 300
        y0 = height - 70
        pattern = PATTERNS[index]
        patterned_polyline(
            draw,
            [(scale(x0), scale(y0)), (scale(x0 + 70), scale(y0))],
            fill=COLORS[index],
            width=scale(3),
            pattern=None if pattern is None else tuple(scale(length) for length in pattern),
        )
        draw.text((scale(x0 + 88), scale(y0)), item["label"], fill=foreground, font=legend_font, anchor="lm")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width, height), Image.Resampling.LANCZOS).save(output, dpi=(240, 240))
