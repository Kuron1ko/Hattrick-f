from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator


THIS_DIR = Path(__file__).resolve().parent
ARTIFACTS = THIS_DIR / "artifacts"
METHOD_FILES = {
    "Hattrick": ARTIFACTS / "test_hattrick_metrics.csv",
    "Hattrick-f": ARTIFACTS / "test_hattrick_f_metrics.csv",
}
CLASSES = ("High", "Medium", "Low")
X_LIMITS = {
    "High": (0.96, 1.005),
    "Medium": (0.78, 1.06),
    "Low": (0.60, 1.34),
}
X_TICKS = {
    "High": (0.96, 0.97, 0.98, 0.99, 1.00),
    "Medium": (0.80, 0.85, 0.90, 0.95, 1.00, 1.05),
    "Low": (0.60, 0.80, 1.00, 1.20, 1.30),
}
COLORS = {"Hattrick": "#2F7ED8", "Hattrick-f": "#F28E2B"}


def read_values(path: Path) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = {name: [] for name in CLASSES}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["class"]].append(float(row["norm_fulfill"]))
    result = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in grouped.items()
    }
    if any(values.size != 2700 for values in result.values()):
        raise RuntimeError(f"Expected 2700 test values per class in {path}")
    # High cannot exceed its own max-flow oracle. Remove only solver-level
    # positive noise (the observed excess is below 1.5e-7).
    result["High"] = np.minimum(result["High"], 1.0)
    return result


def monotone_cdf(
    values: np.ndarray, limits: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous, shape-preserving empirical CDF with flat end segments."""
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


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    windows = Path("C:/Windows/Fonts")
    candidates = (
        (windows / "arialbd.ttf", windows / "arial.ttf")
        if bold
        else (windows / "arial.ttf", windows / "segoeui.ttf")
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.truetype(
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size
    )


def dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: str,
    width: int,
    dash: float,
    gap: float,
) -> None:
    drawing = True
    remaining = dash
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
                drawing = not drawing
                remaining = dash if drawing else gap


def plot(output: Path) -> None:
    values = {method: read_values(path) for method, path in METHOD_FILES.items()}
    scale = 3
    width, height = 2400, 860
    image = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(image)

    def s(value: float) -> int:
        return int(round(value * scale))

    title_font = font(s(34), bold=True)
    panel_font = font(s(23))
    label_font = font(s(19))
    tick_font = font(s(15))
    legend_font = font(s(18))
    foreground = "#1F2328"
    muted = "#4E5968"
    grid = "#DCE2E9"

    draw.text((s(100), s(40)), "CDF of NormFulFill", fill=foreground, font=title_font)
    left, right, top, bottom = 105, 55, 145, 175
    gap = 90
    panel_width = (width - left - right - 2 * gap) / 3
    panel_height = height - top - bottom

    for panel_index, class_name in enumerate(CLASSES):
        x0 = left + panel_index * (panel_width + gap)
        x1 = x0 + panel_width
        y0 = top
        y1 = top + panel_height
        limits = X_LIMITS[class_name]

        for value in np.linspace(0.0, 1.0, 6):
            py = y1 - value * panel_height
            draw.line((s(x0), s(py), s(x1), s(py)), fill=grid, width=s(1))
            draw.text(
                (s(x0 - 16), s(py)),
                f"{value:.1f}",
                fill=muted,
                font=tick_font,
                anchor="rm",
            )

        for value in X_TICKS[class_name]:
            px = x0 + (value - limits[0]) / (limits[1] - limits[0]) * panel_width
            draw.line((s(px), s(y1), s(px), s(y1 + 8)), fill=foreground, width=s(1))
            draw.text(
                (s(px), s(y1 + 16)),
                f"{value:.2f}" if class_name != "Low" else f"{value:.1f}",
                fill=muted,
                font=tick_font,
                anchor="ma",
            )

        draw.line((s(x0), s(y0), s(x0), s(y1)), fill=foreground, width=s(1))
        draw.line((s(x0), s(y1), s(x1), s(y1)), fill=foreground, width=s(1))
        draw.text(
            (s((x0 + x1) / 2), s(112)),
            f"2x load · {class_name}",
            fill=foreground,
            font=panel_font,
            anchor="mm",
        )
        draw.text(
            (s((x0 + x1) / 2), s(y1 + 68)),
            "Normalized FulfillRatio",
            fill=foreground,
            font=label_font,
            anchor="mm",
        )

        y_label = Image.new("RGBA", (s(100), s(42)), (255, 255, 255, 0))
        y_draw = ImageDraw.Draw(y_label)
        y_draw.text(
            (s(50), s(21)), "CDF", fill=foreground, font=label_font, anchor="mm"
        )
        y_label = y_label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
        image.paste(
            y_label,
            (s(x0 - 80), s((y0 + y1) / 2 - y_label.height / scale / 2)),
            y_label,
        )

        for method in METHOD_FILES:
            curve_x, curve_y = monotone_cdf(values[method][class_name], limits)
            points = [
                (
                    s(
                        x0
                        + (x - limits[0])
                        / (limits[1] - limits[0])
                        * panel_width
                    ),
                    s(y1 - y * panel_height),
                )
                for x, y in zip(curve_x, curve_y)
            ]
            if method == "Hattrick":
                draw.line(points, fill=COLORS[method], width=s(3), joint="curve")
            else:
                dashed_polyline(
                    draw,
                    points,
                    fill=COLORS[method],
                    width=s(3),
                    dash=s(13),
                    gap=s(8),
                )

    legend_y = height - 62
    legend_x = 110
    draw.line(
        (s(legend_x), s(legend_y), s(legend_x + 70), s(legend_y)),
        fill=COLORS["Hattrick"],
        width=s(3),
    )
    draw.text(
        (s(legend_x + 88), s(legend_y)),
        "Hattrick",
        fill=foreground,
        font=legend_font,
        anchor="lm",
    )
    second_x = legend_x + 285
    dashed_polyline(
        draw,
        [(s(second_x), s(legend_y)), (s(second_x + 70), s(legend_y))],
        fill=COLORS["Hattrick-f"],
        width=s(3),
        dash=s(13),
        gap=s(8),
    )
    draw.text(
        (s(second_x + 88), s(legend_y)),
        "Hattrick-f",
        fill=foreground,
        font=legend_font,
        anchor="lm",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image.save(output, dpi=(240, 240))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot full GEANT 2x test CDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS / "test_cdf_hattrick_vs_hattrick_f.png",
    )
    args = parser.parse_args()
    plot(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
