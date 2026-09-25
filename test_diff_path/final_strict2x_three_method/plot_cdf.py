from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator


CLASSES = ("High", "Medium", "Low")
ROLE_ORDER = (
    "native_hattrick",
    "hattrick_e",
    "sparse_cross_attention",
)
ROLE_STYLE = {
    "native_hattrick": {"color": "#2F7ED8", "pattern": None, "zorder": 2},
    "hattrick_e": {"color": "#F28E2B", "pattern": (26.0, 14.0), "zorder": 3},
    "sparse_cross_attention": {
        "color": "#2AA65A",
        "pattern": (26.0, 10.0, 5.0, 10.0),
        "zorder": 4,
    },
}
BASE_AXES = {
    "High": {"limits": (0.88, 1.005), "step": 0.02, "format": "%.2f"},
    "Medium": {"limits": (0.60, 1.10), "step": 0.10, "format": "%.1f"},
    "Low": {"limits": (0.60, 1.52), "step": 0.20, "format": "%.1f"},
}
REQUIRED_COLUMNS = {
    "method",
    "role",
    "snapshot",
    "class",
    "norm_fulfill",
    "demand",
    "oracle_admitted_traffic",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def validate_rows(
    rows: list[dict],
    evaluation_range: tuple[int, int] | None = None,
    expected_roles: tuple[str, ...] = ROLE_ORDER,
) -> dict:
    if not rows:
        raise ValueError("Comparison rows are empty")
    roles = {str(row["role"]) for row in rows}
    if roles != set(expected_roles):
        raise ValueError(f"Expected roles {expected_roles}, found {sorted(roles)}")

    labels_by_role: dict[str, set[str]] = {role: set() for role in expected_roles}
    grouped: dict[tuple[str, str], dict[int, dict]] = {}
    for row_index, row in enumerate(rows):
        role = str(row["role"])
        class_name = str(row["class"])
        if class_name not in CLASSES:
            raise ValueError(f"Row {row_index} has unexpected class {class_name!r}")
        labels_by_role[role].add(str(row["method"]))
        snapshot = int(row["snapshot"])
        for field in ("norm_fulfill", "demand", "oracle_admitted_traffic"):
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError(f"Row {row_index} has non-finite {field}")
        key = (role, class_name)
        by_snapshot = grouped.setdefault(key, {})
        if snapshot in by_snapshot:
            raise ValueError(f"Duplicate row for {role}/{class_name}/{snapshot}")
        by_snapshot[snapshot] = row

    for role, labels in labels_by_role.items():
        if len(labels) != 1:
            raise ValueError(f"Role {role} maps to labels {sorted(labels)}")

    reference_snapshots = None
    for role in expected_roles:
        for class_name in CLASSES:
            snapshots = set(grouped.get((role, class_name), {}))
            if reference_snapshots is None:
                reference_snapshots = snapshots
            elif snapshots != reference_snapshots:
                raise ValueError(
                    f"Snapshot set mismatch for {role}/{class_name}: "
                    f"{len(snapshots)} vs {len(reference_snapshots)}"
                )
    reference_snapshots = reference_snapshots or set()
    if evaluation_range is not None:
        expected = set(range(*evaluation_range))
        if reference_snapshots != expected:
            raise ValueError(
                f"Rows cover {sorted(reference_snapshots)[:3]}..."
                f"{sorted(reference_snapshots)[-3:]}, expected {evaluation_range}"
            )

    # Every method must be normalized by the same snapshot/class demand and
    # oracle. This catches accidental split/load/oracle mixing before plotting.
    denominator_max_delta = 0.0
    native = "native_hattrick"
    for role in expected_roles:
        for class_name in CLASSES:
            for snapshot in reference_snapshots:
                left = grouped[(native, class_name)][snapshot]
                right = grouped[(role, class_name)][snapshot]
                for field in ("demand", "oracle_admitted_traffic"):
                    denominator_max_delta = max(
                        denominator_max_delta,
                        abs(float(left[field]) - float(right[field])),
                    )
    if denominator_max_delta > 1e-5:
        raise ValueError(
            "Methods do not share demand/oracle denominators; max delta "
            f"{denominator_max_delta}"
        )

    return {
        "roles": list(expected_roles),
        "labels": {role: next(iter(labels_by_role[role])) for role in expected_roles},
        "snapshot_start": min(reference_snapshots),
        "snapshot_end_exclusive": max(reference_snapshots) + 1,
        "snapshots_per_class": len(reference_snapshots),
        "rows": len(rows),
        "denominator_max_delta": denominator_max_delta,
    }


def adaptive_axis(class_name: str, series: list[np.ndarray]) -> dict:
    specification = BASE_AXES[class_name]
    base_low, base_high = specification["limits"]
    step = float(specification["step"])
    all_values = np.concatenate(series)
    observed_low = float(np.min(all_values))
    observed_high = float(np.max(all_values))
    observed_span = max(observed_high - observed_low, step)
    margin = max(0.04 * observed_span, 0.20 * step)
    low = base_low
    high = base_high
    if observed_low - margin < low:
        low = math.floor((observed_low - margin) / step) * step
    if observed_high + margin > high:
        high = math.ceil((observed_high + margin) / step) * step
    if high <= low:
        high = low + step
    tick_start = math.ceil((low - 1e-10) / step) * step
    tick_end = math.floor((high + 1e-10) / step) * step
    ticks = np.arange(tick_start, tick_end + 0.5 * step, step)
    return {
        "limits": (float(low), float(high)),
        "ticks": ticks,
        "format": specification["format"],
        "observed": (observed_low, observed_high),
    }


def continuous_empirical_cdf(
    values: np.ndarray,
    x_limits: tuple[float, float],
    points: int = 1201,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Monotone empirical-quantile interpolation with exact flat tails.

    The sorted observations remain the interpolation anchors. PCHIP is applied
    to x as a function of empirical rank, so it cannot invent reversals or
    modify any reported statistic. It only avoids sample-to-sample angular
    wiggles in the display layer.
    """

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.ndim != 1 or ordered.size < 2 or not np.isfinite(ordered).all():
        raise ValueError("CDF requires at least two finite observations")
    axis_span = float(x_limits[1] - x_limits[0])
    observed_span = max(float(np.ptp(ordered)), axis_span * 1e-5)
    transition_padding = max(observed_span * 0.025, axis_span * 0.004)
    transition_start = max(
        x_limits[0] + axis_span * 0.025,
        float(ordered[0]) - transition_padding,
    )
    transition_end = min(
        x_limits[1] - axis_span * 0.025,
        float(ordered[-1]) + transition_padding,
    )
    if transition_end <= transition_start:
        transition_start = x_limits[0] + axis_span * 0.025
        transition_end = x_limits[1] - axis_span * 0.025

    # Forty-one evenly spaced order-statistic anchors retain the empirical
    # distribution's shape while avoiding the visible angular jitter produced
    # by connecting every one of the 100 observations.
    anchor_count = min(41, int(ordered.size))
    anchor_indices = np.unique(
        np.rint(np.linspace(0, ordered.size - 1, anchor_count)).astype(np.int64)
    )
    empirical_anchors = ordered[anchor_indices]
    ranks = (anchor_indices.astype(np.float64) + 0.5) / ordered.size
    control_y = np.concatenate(([0.0], ranks, [1.0]))
    control_x = np.concatenate(
        ([transition_start], empirical_anchors, [transition_end])
    )
    control_x = np.maximum.accumulate(control_x)
    dense_y = np.linspace(0.0, 1.0, int(points))
    dense_x = PchipInterpolator(control_y, control_x)(dense_y)
    dense_x = np.maximum.accumulate(np.clip(dense_x, transition_start, transition_end))

    x = np.concatenate(
        ([x_limits[0], transition_start], dense_x, [transition_end, x_limits[1]])
    )
    y = np.concatenate(([0.0, 0.0], dense_y, [1.0, 1.0]))
    audit = {
        "observations": int(ordered.size),
        "empirical_display_anchors": int(anchor_indices.size),
        "observed_min": float(ordered[0]),
        "observed_max": float(ordered[-1]),
        "transition_start": float(transition_start),
        "transition_end": float(transition_end),
        "left_flat_span": float(transition_start - x_limits[0]),
        "right_flat_span": float(x_limits[1] - transition_end),
        "monotone_x": bool(np.all(np.diff(x) >= -1e-12)),
        "monotone_y": bool(np.all(np.diff(y) >= -1e-12)),
        "exact_flat_zero_tail": bool(y[0] == 0.0 and y[1] == 0.0),
        "exact_flat_one_tail": bool(y[-2] == 1.0 and y[-1] == 1.0),
    }
    return x, y, audit


def load_font(size: int, bold: bool = False):
    windows_fonts = Path("C:/Windows/Fonts")
    candidates = (
        (windows_fonts / "arialbd.ttf", windows_fonts / "arial.ttf")
        if bold
        else (windows_fonts / "arial.ttf", windows_fonts / "segoeui.ttf")
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def draw_patterned_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: str,
    width: int,
    pattern: tuple[float, ...] | None,
) -> None:
    if pattern is None:
        draw.line(points, fill=color, width=width, joint="curve")
        return
    pattern_index = 0
    remaining = float(pattern[0])
    drawing = True
    for start, end in zip(points[:-1], points[1:]):
        x0, y0 = start
        x1, y1 = end
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue
        consumed = 0.0
        while consumed < length - 1e-9:
            advance = min(remaining, length - consumed)
            first_fraction = consumed / length
            second_fraction = (consumed + advance) / length
            segment_start = (x0 + dx * first_fraction, y0 + dy * first_fraction)
            segment_end = (x0 + dx * second_fraction, y0 + dy * second_fraction)
            if drawing:
                draw.line(
                    (segment_start, segment_end),
                    fill=color,
                    width=width,
                )
            consumed += advance
            remaining -= advance
            if remaining <= 1e-9:
                pattern_index = (pattern_index + 1) % len(pattern)
                remaining = float(pattern[pattern_index])
                drawing = pattern_index % 2 == 0


def draw_rotated_text(
    image: Image.Image,
    center: tuple[int, int],
    text: str,
    font,
    fill: str,
) -> None:
    probe = ImageDraw.Draw(image)
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    layer = Image.new("RGBA", (right - left + 16, bottom - top + 16), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((8 - left, 8 - top), text, font=font, fill=fill)
    rotated = layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    position = (center[0] - rotated.width // 2, center[1] - rotated.height // 2)
    image.paste(rotated, position, rotated)


def render_cdf(
    rows_path: Path,
    output_path: Path,
    metadata_path: Path | None = None,
    evaluation_range: tuple[int, int] | None = (400, 500),
) -> dict:
    rows = read_rows(rows_path)
    validation = validate_rows(rows, evaluation_range=evaluation_range)
    labels = validation["labels"]
    grouped: dict[tuple[str, str], np.ndarray] = {}
    for role in ROLE_ORDER:
        for class_name in CLASSES:
            grouped[(role, class_name)] = np.asarray(
                [
                    float(row["norm_fulfill"])
                    for row in rows
                    if row["role"] == role and row["class"] == class_name
                ],
                dtype=np.float64,
            )

    axes_spec = {
        class_name: adaptive_axis(
            class_name, [grouped[(role, class_name)] for role in ROLE_ORDER]
        )
        for class_name in CLASSES
    }
    if output_path.suffix.lower() != ".png":
        raise ValueError("The dependency-free renderer currently requires a .png output")
    width, height = 3168, 1020
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    foreground = "#25282C"
    tick_color = "#5A6572"
    grid_color = "#DCE2E8"
    title_font = load_font(46, bold=True)
    panel_font = load_font(31)
    axis_font = load_font(26)
    tick_font = load_font(23)
    legend_font = load_font(27)
    left_margin, right_margin = 170, 70
    top, bottom = 205, 775
    gap = 125
    panel_width = (width - left_margin - right_margin - 2 * gap) / 3.0
    draw.text((left_margin, 36), "CDF of NormFulFill", font=title_font, fill=foreground)

    curve_audit: dict[str, dict] = {}
    for column_index, class_name in enumerate(CLASSES):
        specification = axes_spec[class_name]
        x_limits = specification["limits"]
        panel_left = left_margin + column_index * (panel_width + gap)
        panel_right = panel_left + panel_width
        draw.text(
            ((panel_left + panel_right) / 2.0, 150),
            class_name,
            font=panel_font,
            fill=foreground,
            anchor="mm",
        )
        for y_tick in np.arange(0.0, 1.01, 0.2):
            y_pixel = bottom - float(y_tick) * (bottom - top)
            draw.line(
                ((panel_left, y_pixel), (panel_right, y_pixel)),
                fill=grid_color,
                width=2,
            )
            draw.line(
                ((panel_left - 10, y_pixel), (panel_left, y_pixel)),
                fill=foreground,
                width=3,
            )
            draw.text(
                (panel_left - 18, y_pixel),
                f"{y_tick:.1f}",
                font=tick_font,
                fill=tick_color,
                anchor="rm",
            )
        for x_tick in specification["ticks"]:
            fraction = (float(x_tick) - x_limits[0]) / (x_limits[1] - x_limits[0])
            x_pixel = panel_left + fraction * panel_width
            draw.line(
                ((x_pixel, bottom), (x_pixel, bottom + 10)),
                fill=foreground,
                width=3,
            )
            draw.text(
                (x_pixel, bottom + 20),
                specification["format"] % float(x_tick),
                font=tick_font,
                fill=tick_color,
                anchor="ma",
            )
        draw.line(
            ((panel_left, top), (panel_left, bottom), (panel_right, bottom)),
            fill=foreground,
            width=3,
        )
        draw.text(
            ((panel_left + panel_right) / 2.0, 865),
            "NormFulFill",
            font=axis_font,
            fill=foreground,
            anchor="mm",
        )
        draw_rotated_text(
            image,
            (int(panel_left - 125), int((top + bottom) / 2)),
            "CDF",
            axis_font,
            foreground,
        )
        for role in ROLE_ORDER:
            x, y, audit = continuous_empirical_cdf(grouped[(role, class_name)], x_limits)
            style = ROLE_STYLE[role]
            points = [
                (
                    panel_left
                    + (float(x_value) - x_limits[0])
                    / (x_limits[1] - x_limits[0])
                    * panel_width,
                    bottom - float(y_value) * (bottom - top),
                )
                for x_value, y_value in zip(x, y)
            ]
            draw_patterned_line(
                draw, points, style["color"], width=6, pattern=style["pattern"]
            )
            curve_audit[f"{role}/{class_name}"] = audit

    legend_y = 965
    legend_x = left_margin
    for role in ROLE_ORDER:
        style = ROLE_STYLE[role]
        draw_patterned_line(
            draw,
            [(legend_x, legend_y), (legend_x + 115, legend_y)],
            style["color"],
            width=6,
            pattern=style["pattern"],
        )
        draw.text(
            (legend_x + 135, legend_y),
            labels[role],
            font=legend_font,
            fill=foreground,
            anchor="lm",
        )
        label_width = draw.textlength(labels[role], font=legend_font)
        legend_x += 135 + label_width + 130

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", dpi=(240, 240), optimize=True)

    metadata = {
        "title": "CDF of NormFulFill",
        "load_factor": 2.0,
        "inference": "strict ESM",
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": sha256(rows_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "curve": (
            "PCHIP monotone interpolation of 41 evenly spaced empirical "
            "order-statistic anchors; "
            "display-only, with exact horizontal CDF=0/1 tails"
        ),
        "renderer": "Pillow dependency-light 240-dpi raster renderer",
        "statistics_modified_by_plotting": False,
        "validation": validation,
        "axes": {
            name: {
                "limits": list(value["limits"]),
                "ticks": [float(item) for item in value["ticks"]],
                "observed": list(value["observed"]),
            }
            for name, value in axes_spec.items()
        },
        "curves": curve_audit,
    }
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper-style continuous empirical CDF for three strict-ESM methods"
    )
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--start", type=int, default=400)
    parser.add_argument("--end", type=int, default=500)
    args = parser.parse_args()
    metadata = render_cdf(
        args.rows.resolve(),
        args.output.resolve(),
        args.metadata.resolve() if args.metadata else None,
        evaluation_range=(args.start, args.end),
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
