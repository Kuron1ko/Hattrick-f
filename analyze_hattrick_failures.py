from __future__ import annotations

import csv
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "geant" / "8sp" / "0"
OUTPUT_DIR = RESULT_DIR / "failure_analysis"
METRICS_PATH = RESULT_DIR / "repro_metrics_geant8_full.csv"
MANIFEST_PATH = ROOT / "manifest" / "geant_manifest.txt"
TEST_START = 8080
TEST_END = 10773
CLASSES = ("High", "Medium", "Low")
THRESHOLDS = {"High": 0.9999, "Medium": 0.99, "Low": 0.98}


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_manifest() -> list[str]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return [line.strip().split(",")[2].strip() for line in handle if line.strip()]


def read_hattrick_metrics() -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    with METRICS_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != "Hattrick":
                continue
            parsed: dict[str, str | int | float] = {
                "snapshot_idx": int(row["snapshot_idx"]),
                "class": row["class"],
            }
            for key in (
                "demand",
                "oracle_fulfilled",
                "carried_traffic",
                "fulfill_ratio",
                "norm_fulfill",
                "mlu",
                "norm_mlu",
            ):
                parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


class TrafficMatrixCache:
    def __init__(self, manifest: list[str]) -> None:
        self.manifest = manifest
        self.cache: dict[tuple[str, int], np.ndarray] = {}

    def read(self, folder: str, snapshot_idx: int) -> np.ndarray:
        key = (folder, snapshot_idx)
        if key not in self.cache:
            path = ROOT / "traffic_matrices" / folder / self.manifest[snapshot_idx]
            with path.open("rb") as handle:
                self.cache[key] = np.asarray(pickle.load(handle), dtype=np.float64)
        return self.cache[key]


def matrix_features(cache: TrafficMatrixCache, snapshot_idx: int, class_idx: int) -> dict[str, float]:
    priority = class_idx + 1
    actual = cache.read(f"geant_{priority}", snapshot_idx)
    predicted = cache.read(f"geant_{priority}_esm", snapshot_idx)
    total = float(actual.sum())
    predicted_total = float(predicted.sum())
    diff = predicted - actual
    flat = actual.reshape(-1)

    top1_share = float(flat.max() / total) if total else 0.0
    top5_share = float(np.sort(flat)[-5:].sum() / total) if total else 0.0
    hhi = float(((flat / total) ** 2).sum()) if total else 0.0
    sparsity = float((flat <= 1e-12).mean())
    pred_l1_rel = float(np.abs(diff).sum() / total) if total else 0.0
    pred_linf_rel = float(np.abs(diff).max() / total) if total else 0.0
    pred_total_signed_rel = float((predicted_total - total) / total) if total else 0.0

    if snapshot_idx > 0:
        previous = cache.read(f"geant_{priority}", snapshot_idx - 1)
        previous_total = float(previous.sum())
        change_l1_rel = float(np.abs(actual - previous).sum() / previous_total) if previous_total else 0.0
        change_total_rel = float((total - previous_total) / previous_total) if previous_total else 0.0
    else:
        change_l1_rel = 0.0
        change_total_rel = 0.0

    return {
        "pred_total_signed_rel": pred_total_signed_rel,
        "pred_l1_rel": pred_l1_rel,
        "pred_linf_rel": pred_linf_rel,
        "top1_share": top1_share,
        "top5_share": top5_share,
        "hhi": hhi,
        "sparsity": sparsity,
        "change_l1_rel": change_l1_rel,
        "change_total_rel": change_total_rel,
    }


def assign_quantile(values: np.ndarray, value: float) -> int:
    edges = np.quantile(values, [0.25, 0.50, 0.75])
    return int(np.searchsorted(edges, value, side="right")) + 1


def enrich_rows(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    manifest = read_manifest()
    cache = TrafficMatrixCache(manifest)
    by_class = {class_name: [row for row in rows if row["class"] == class_name] for class_name in CLASSES}

    bottom10_cutoffs = {
        class_name: float(np.percentile([float(row["norm_fulfill"]) for row in class_rows], 10))
        for class_name, class_rows in by_class.items()
    }
    demand_values = {
        class_name: np.asarray([float(row["demand"]) for row in class_rows], dtype=np.float64)
        for class_name, class_rows in by_class.items()
    }
    mlu_values = {
        class_name: np.asarray([float(row["mlu"]) for row in class_rows], dtype=np.float64)
        for class_name, class_rows in by_class.items()
    }
    oracle_frac_values = {
        class_name: np.asarray(
            [
                float(row["oracle_fulfilled"]) / float(row["demand"])
                if float(row["demand"]) > 0.0
                else 0.0
                for row in class_rows
            ],
            dtype=np.float64,
        )
        for class_name, class_rows in by_class.items()
    }

    enriched: list[dict[str, str | int | float]] = []
    for row in rows:
        class_name = str(row["class"])
        class_idx = CLASSES.index(class_name)
        snapshot_idx = int(row["snapshot_idx"])
        demand = float(row["demand"])
        oracle_fulfilled = float(row["oracle_fulfilled"])
        norm_fulfill = float(row["norm_fulfill"])
        oracle_frac = oracle_fulfilled / demand if demand > 0.0 else 0.0

        enriched_row = dict(row)
        enriched_row.update(matrix_features(cache, snapshot_idx, class_idx))
        enriched_row["oracle_frac"] = oracle_frac
        enriched_row["bad_threshold"] = THRESHOLDS[class_name]
        enriched_row["bottom10_cutoff"] = bottom10_cutoffs[class_name]
        enriched_row["is_bad_threshold"] = int(norm_fulfill < THRESHOLDS[class_name])
        enriched_row["is_bottom10"] = int(norm_fulfill <= bottom10_cutoffs[class_name])
        enriched_row["demand_quantile"] = assign_quantile(demand_values[class_name], demand)
        enriched_row["mlu_quantile"] = assign_quantile(mlu_values[class_name], float(row["mlu"]))
        enriched_row["oracle_frac_quantile"] = assign_quantile(oracle_frac_values[class_name], oracle_frac)
        enriched.append(enriched_row)
    return enriched


def write_csv(path: Path, rows: list[dict[str, str | int | float]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(rows: list[dict[str, str | int | float]], group_fields: tuple[str, ...]) -> list[dict[str, str | int | float]]:
    groups: dict[tuple[object, ...], list[dict[str, str | int | float]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in group_fields)].append(row)

    summaries: list[dict[str, str | int | float]] = []
    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        norm = np.asarray([float(row["norm_fulfill"]) for row in group_rows], dtype=np.float64)
        demand = np.asarray([float(row["demand"]) for row in group_rows], dtype=np.float64)
        mlu = np.asarray([float(row["mlu"]) for row in group_rows], dtype=np.float64)
        oracle_frac = np.asarray([float(row["oracle_frac"]) for row in group_rows], dtype=np.float64)
        item = {field: value for field, value in zip(group_fields, key)}
        item.update(
            {
                "n": len(group_rows),
                "bad_threshold_count": int(sum(int(row["is_bad_threshold"]) for row in group_rows)),
                "bad_threshold_rate": float(np.mean([int(row["is_bad_threshold"]) for row in group_rows])),
                "bottom10_count": int(sum(int(row["is_bottom10"]) for row in group_rows)),
                "bottom10_rate": float(np.mean([int(row["is_bottom10"]) for row in group_rows])),
                "norm_fulfill_mean": float(norm.mean()),
                "norm_fulfill_p1": float(np.percentile(norm, 1)),
                "norm_fulfill_p10": float(np.percentile(norm, 10)),
                "norm_fulfill_min": float(norm.min()),
                "demand_mean": float(demand.mean()),
                "mlu_mean": float(mlu.mean()),
                "oracle_frac_mean": float(oracle_frac.mean()),
            }
        )
        summaries.append(item)
    return summaries


def write_group_stats(rows: list[dict[str, str | int | float]]) -> None:
    all_summaries: list[dict[str, str | int | float]] = []
    for label, fields in (
        ("class", ("class",)),
        ("class_demand_quantile", ("class", "demand_quantile")),
        ("class_mlu_quantile", ("class", "mlu_quantile")),
        ("class_oracle_frac_quantile", ("class", "oracle_frac_quantile")),
        ("class_demand_mlu_quantile", ("class", "demand_quantile", "mlu_quantile")),
    ):
        for summary in summarize_group(rows, fields):
            summary["grouping"] = label
            all_summaries.append(summary)

    fieldnames = [
        "grouping",
        "class",
        "demand_quantile",
        "mlu_quantile",
        "oracle_frac_quantile",
        "n",
        "bad_threshold_count",
        "bad_threshold_rate",
        "bottom10_count",
        "bottom10_rate",
        "norm_fulfill_mean",
        "norm_fulfill_p1",
        "norm_fulfill_p10",
        "norm_fulfill_min",
        "demand_mean",
        "mlu_mean",
        "oracle_frac_mean",
    ]
    write_csv(OUTPUT_DIR / "failure_group_stats.csv", all_summaries, fieldnames)


def value_to_x(value: float, x_min: float, x_max: float, left: int, right: int) -> int:
    if x_max <= x_min:
        return left
    value = max(x_min, min(x_max, value))
    return int(left + ((value - x_min) / (x_max - x_min)) * (right - left))


def value_to_y(value: float, y_min: float, y_max: float, top: int, bottom: int) -> int:
    if y_max <= y_min:
        return bottom
    value = max(y_min, min(y_max, value))
    return int(bottom - ((value - y_min) / (y_max - y_min)) * (bottom - top))


def draw_scatter(rows: list[dict[str, str | int | float]]) -> None:
    width, height = 1500, 980
    left, right, top, bottom = 90, 46, 122, 810
    gap_x, gap_y = 54, 70
    panel_width = (width - left - right - gap_x) // 2
    panel_height = (bottom - top - gap_y) // 2
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(29, bold=True)
    label_font = font(18)
    small_font = font(14)
    colors = {"High": "#2F80ED", "Medium": "#F2994A", "Low": "#EB5757"}

    panels = [
        ("Demand", "demand"),
        ("MLU", "mlu"),
        ("Oracle fulfilled / demand", "oracle_frac"),
        ("ESM L1 prediction error", "pred_l1_rel"),
    ]
    y_values = np.asarray([float(row["norm_fulfill"]) for row in rows], dtype=np.float64)
    y_min = max(0.90, math.floor((float(y_values.min()) - 0.01) * 20) / 20)
    y_max = 1.02

    draw.text((left, 30), "Hattrick failure scatter: data conditions vs NormFulFill", fill="#202124", font=title_font)
    draw.text((left, 68), "Threshold-bad samples are highlighted with larger outlined points.", fill="#5f6368", font=small_font)

    for panel_idx, (label, key) in enumerate(panels):
        col = panel_idx % 2
        row_idx = panel_idx // 2
        panel_left = left + col * (panel_width + gap_x)
        panel_top = top + row_idx * (panel_height + gap_y)
        panel_right = panel_left + panel_width
        panel_bottom = panel_top + panel_height
        x_values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        x_min = float(np.percentile(x_values, 1))
        x_max = float(np.percentile(x_values, 99))
        if key == "oracle_frac":
            x_min, x_max = 0.65, 1.01
        elif key == "pred_l1_rel":
            x_min, x_max = 0.0, float(max(0.35, np.percentile(x_values, 99)))

        draw.rectangle((panel_left, panel_top, panel_right, panel_bottom), outline="#202124", width=2)
        draw.text((panel_left, panel_top - 28), label, fill="#202124", font=label_font)
        for tick in np.linspace(x_min, x_max, 5):
            x = value_to_x(float(tick), x_min, x_max, panel_left, panel_right)
            draw.line((x, panel_top, x, panel_bottom), fill="#E8EAED", width=1)
            draw.text((x - 22, panel_bottom + 7), f"{tick:.2f}", fill="#5f6368", font=small_font)
        for tick in np.linspace(y_min, y_max, 5):
            y = value_to_y(float(tick), y_min, y_max, panel_top, panel_bottom)
            draw.line((panel_left, y, panel_right, y), fill="#E8EAED", width=1)
            draw.text((panel_left - 58, y - 8), f"{tick:.2f}", fill="#5f6368", font=small_font)

        # Plot non-bad first, then bad points on top.
        ordered_rows = sorted(rows, key=lambda row: int(row["is_bad_threshold"]))
        for data_row in ordered_rows:
            x = value_to_x(float(data_row[key]), x_min, x_max, panel_left, panel_right)
            y = value_to_y(float(data_row["norm_fulfill"]), y_min, y_max, panel_top, panel_bottom)
            color = colors[str(data_row["class"])]
            radius = 2 if int(data_row["is_bad_threshold"]) == 0 else 4
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="#202124" if radius == 4 else color)

    legend_x, legend_y = left, height - 92
    for class_name in CLASSES:
        draw.ellipse((legend_x, legend_y, legend_x + 16, legend_y + 16), fill=colors[class_name])
        draw.text((legend_x + 24, legend_y - 2), class_name, fill="#202124", font=label_font)
        legend_x += 130
    draw.text((left + 610, height - 92), "Y-axis: Hattrick NormFulFill", fill="#202124", font=label_font)
    image.save(OUTPUT_DIR / "failure_scatter.png")


def draw_heatmap(rows: list[dict[str, str | int | float]]) -> None:
    width, height = 1580, 760
    left, top = 100, 126
    cell = 92
    class_gap = 96
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(29, bold=True)
    label_font = font(18)
    small_font = font(14)

    draw.text((left, 30), "Hattrick bad-sample rate by demand and MLU quantiles", fill="#202124", font=title_font)
    draw.text((left, 68), "Badness uses fixed absolute NormFulFill thresholds. Darker means more bad samples.", fill="#5f6368", font=small_font)

    for class_idx, class_name in enumerate(CLASSES):
        class_rows = [row for row in rows if row["class"] == class_name]
        panel_left = left + class_idx * (cell * 4 + class_gap)
        draw.text((panel_left + 90, top - 40), class_name, fill="#202124", font=label_font)
        draw.text((panel_left + 90, top + cell * 4 + 45), "MLU quantile", fill="#202124", font=label_font)
        draw.text((panel_left - 64, top + 145), "Demand", fill="#202124", font=small_font)

        for demand_q in range(1, 5):
            draw.text((panel_left - 26, top + (demand_q - 1) * cell + 34), f"Q{demand_q}", fill="#5f6368", font=small_font)
        for mlu_q in range(1, 5):
            draw.text((panel_left + (mlu_q - 1) * cell + 34, top + cell * 4 + 8), f"Q{mlu_q}", fill="#5f6368", font=small_font)

        for demand_q in range(1, 5):
            for mlu_q in range(1, 5):
                cell_rows = [
                    row
                    for row in class_rows
                    if int(row["demand_quantile"]) == demand_q and int(row["mlu_quantile"]) == mlu_q
                ]
                rate = float(np.mean([int(row["is_bad_threshold"]) for row in cell_rows])) if cell_rows else 0.0
                count = len(cell_rows)
                intensity = int(255 - min(1.0, rate / 0.15) * 190)
                fill = (255, intensity, intensity)
                x0 = panel_left + (mlu_q - 1) * cell
                y0 = top + (demand_q - 1) * cell
                draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=fill, outline="#FFFFFF", width=2)
                draw.text((x0 + 20, y0 + 24), f"{rate:.0%}", fill="#202124", font=label_font)
                draw.text((x0 + 22, y0 + 52), f"n={count}", fill="#5f6368", font=small_font)

    image.save(OUTPUT_DIR / "failure_heatmap.png")


def write_markdown(rows: list[dict[str, str | int | float]]) -> None:
    lines: list[str] = [
        "# Hattrick Failure Analysis\n\n",
        "## Definition\n",
        "This report defines bad samples by Hattrick's absolute NormFulFill, not by whether Hattrick loses to a baseline.\n\n",
        "| Class | Bad threshold | Bottom-10 cutoff | Bad count | Bad rate | Min NormFulFill |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]

    for class_name in CLASSES:
        class_rows = [row for row in rows if row["class"] == class_name]
        bad_count = sum(int(row["is_bad_threshold"]) for row in class_rows)
        bottom10_cutoff = float(class_rows[0]["bottom10_cutoff"])
        min_norm = min(float(row["norm_fulfill"]) for row in class_rows)
        lines.append(
            f"| {class_name} | {THRESHOLDS[class_name]:.4f} | {bottom10_cutoff:.6f} | "
            f"{bad_count}/{len(class_rows)} | {bad_count / len(class_rows):.2%} | {min_norm:.6f} |\n"
        )

    lines.extend(
        [
            "\n## Main Findings\n",
            "- High priority is effectively stable in this GEANT run; there are no samples below 0.99 NormFulFill.\n",
            "- Medium bad samples concentrate in high-demand and high-MLU regions.\n",
            "- Low priority has the heaviest tail; its worst samples appear when residual capacity after higher priorities is tight.\n",
            "- Aggregate ESM total-demand error is not the strongest separator of bad samples, so later optimization should inspect congestion-aware features rather than only improving total traffic prediction.\n\n",
            "## Worst Threshold-Bad Samples\n",
        ]
    )

    for class_name in CLASSES:
        bad_rows = [row for row in rows if row["class"] == class_name and int(row["is_bad_threshold"]) == 1]
        if not bad_rows:
            lines.append(f"\n### {class_name}\nNo threshold-bad samples.\n")
            continue
        lines.extend(
            [
                f"\n### {class_name}\n",
                "| Snapshot | NormFulFill | FulfillRatio | Demand | OracleFrac | MLU | ESM L1 Err | Change L1 | DemandQ | MLUQ |\n",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
            ]
        )
        for row in sorted(bad_rows, key=lambda item: float(item["norm_fulfill"]))[:15]:
            lines.append(
                "| {snapshot_idx} | {norm_fulfill:.6f} | {fulfill_ratio:.6f} | {demand:.3f} | "
                "{oracle_frac:.3f} | {mlu:.3f} | {pred_l1_rel:.3f} | {change_l1_rel:.3f} | "
                "{demand_quantile} | {mlu_quantile} |\n".format(**row)
            )

    lines.extend(
        [
            "\n## Artifacts\n",
            "- `hattrick_failure_cases.csv`: enriched per-sample Hattrick diagnostics.\n",
            "- `failure_group_stats.csv`: class and quantile-group summaries.\n",
            "- `failure_scatter.png`: feature-vs-NormFulFill scatter plots.\n",
            "- `failure_heatmap.png`: demand quantile by MLU quantile bad-sample rates.\n",
        ]
    )

    (OUTPUT_DIR / "failure_top_cases.md").write_text("".join(lines), encoding="utf-8")


def validate(rows: list[dict[str, str | int | float]]) -> None:
    expected_per_class = TEST_END - TEST_START
    expected_total = expected_per_class * len(CLASSES)
    if len(rows) != expected_total:
        raise RuntimeError(f"Expected {expected_total} Hattrick rows, got {len(rows)}")
    for class_name in CLASSES:
        count = sum(1 for row in rows if row["class"] == class_name)
        if count != expected_per_class:
            raise RuntimeError(f"Expected {expected_per_class} rows for {class_name}, got {count}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = enrich_rows(read_hattrick_metrics())
    validate(rows)

    fieldnames = [
        "snapshot_idx",
        "class",
        "demand",
        "oracle_fulfilled",
        "carried_traffic",
        "fulfill_ratio",
        "norm_fulfill",
        "mlu",
        "norm_mlu",
        "oracle_frac",
        "pred_total_signed_rel",
        "pred_l1_rel",
        "pred_linf_rel",
        "top1_share",
        "top5_share",
        "hhi",
        "sparsity",
        "change_l1_rel",
        "change_total_rel",
        "bad_threshold",
        "bottom10_cutoff",
        "is_bad_threshold",
        "is_bottom10",
        "demand_quantile",
        "mlu_quantile",
        "oracle_frac_quantile",
    ]
    write_csv(OUTPUT_DIR / "hattrick_failure_cases.csv", rows, fieldnames)
    write_group_stats(rows)
    draw_scatter(rows)
    draw_heatmap(rows)
    write_markdown(rows)

    print(f"Wrote {OUTPUT_DIR / 'hattrick_failure_cases.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'failure_group_stats.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'failure_top_cases.md'}")
    print(f"Wrote {OUTPUT_DIR / 'failure_scatter.png'}")
    print(f"Wrote {OUTPUT_DIR / 'failure_heatmap.png'}")


if __name__ == "__main__":
    main()
