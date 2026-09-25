from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[0]
sys.path.append(str(THIS_DIR))
sys.path.append(str(ROOT))

import run_dotemc_priority_mask_experiment as dm
import run_priority_mask_experiment as pm


K = 8
TRAIN_END = 350
VAL_END = 400
TEST_END = 500
TEST_SNAPSHOTS = TEST_END - VAL_END
EPOCHS = 60
BATCH_SIZE = 8
CLASSES = ("High", "Medium", "Low")
SCENARIOS = {
    "shared": "geant_priomask500_shared",
    "mild": "geant_priomask500_mild",
    "medium": "geant_priomask500_medium",
    "strict": "geant_priomask500_strict",
}
RESULTS = THIS_DIR / "results_dotemc_priority_masks_500slice"
LOGS = THIS_DIR / "logs_dotemc_priority_masks_500slice"
PM_STATE = THIS_DIR / "priority_mask_500_state.json"
DM_STATE = THIS_DIR / "dotemc_priority_mask_500_state.json"
HATTRICK_INTERNAL_METRICS = RESULTS / "hattrick_long500_internal_metrics.csv"
DEFAULT_WEIGHT_LABEL = "w_1_0p01_0p001"
SENS_WEIGHT_LABEL = "w_1_0p1_0p01"


def configure_modules() -> None:
    pm.RESULTS = RESULTS
    pm.LOGS = LOGS / "hattrick_gurobi"
    pm.STATE_FILE = PM_STATE
    pm.TRAIN_END = TRAIN_END
    pm.VAL_END = VAL_END
    pm.TEST_END = TEST_END
    pm.EPOCHS = EPOCHS
    pm.BATCH_SIZE = BATCH_SIZE
    pm.SCENARIOS = SCENARIOS

    dm.RESULTS = RESULTS
    dm.OUTPUTS = RESULTS / "outputs"
    dm.LOGS = LOGS / "dotemc"
    dm.STATE_FILE = DM_STATE
    dm.HATTRICK_RESULTS = HATTRICK_INTERNAL_METRICS
    dm.TRAIN_END = TRAIN_END
    dm.VAL_END = VAL_END
    dm.TEST_END = TEST_END
    dm.EPOCHS = EPOCHS
    dm.BATCH_SIZE = BATCH_SIZE
    dm.SCENARIOS = SCENARIOS
    dm.DEFAULT_WEIGHT_LABEL = DEFAULT_WEIGHT_LABEL


def ensure_dirs() -> None:
    for path in (RESULTS, LOGS, LOGS / "hattrick_gurobi", LOGS / "dotemc"):
        path.mkdir(parents=True, exist_ok=True)


def validate_data() -> None:
    configure_modules()
    for scenario, topo in SCENARIOS.items():
        manifest = ROOT / "manifest" / f"{topo}_manifest.txt"
        if not manifest.exists():
            raise RuntimeError(f"Missing manifest: {manifest}")
        manifest_lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(manifest_lines) != TEST_END:
            raise RuntimeError(f"{manifest} has {len(manifest_lines)} lines, expected {TEST_END}")

        paths_path = ROOT / "topologies" / "paths_dict" / f"{topo}_{K}_paths_dict_cluster_0.pkl"
        if not paths_path.exists():
            raise RuntimeError(f"Missing paths: {paths_path}")
        with paths_path.open("rb") as handle:
            paths = pickle.load(handle)
        counts = [len(value) for value in paths.values()]
        if len(paths) != 462 or min(counts) != K or max(counts) != K:
            raise RuntimeError(f"{paths_path} has invalid path counts")

        if scenario != "shared":
            mask_path = ROOT / "topologies" / "path_masks" / f"{topo}_{K}_path_masks_cluster_0.pkl"
            with mask_path.open("rb") as handle:
                masks = np.asarray(pickle.load(handle), dtype=bool)
            if masks.shape != (3, 462, K):
                raise RuntimeError(f"{mask_path} has shape {masks.shape}")
            if masks.sum(axis=2).min() < 2:
                raise RuntimeError(f"{mask_path} has fewer than two allowed paths")
    print("[ok] 500-slice data validation passed", flush=True)


def prepare() -> None:
    configure_modules()
    ensure_dirs()
    pm.prepare()
    validate_data()


def run_gurobi() -> None:
    configure_modules()
    ensure_dirs()
    pm.run_gurobi()


def set_cli_arg(command: list[str], option: str, value: str | int) -> list[str]:
    updated = list(command)
    try:
        idx = updated.index(option)
    except ValueError as exc:
        raise RuntimeError(f"Missing CLI option {option}") from exc
    updated[idx + 1] = str(value)
    return updated


def train_one_epoch_command(topo: str, path_mask: int, initial_training: int) -> list[str]:
    command = pm.train_command(topo, path_mask)
    command = set_cli_arg(command, "--epochs", 1)
    command = set_cli_arg(command, "--initial_training", initial_training)
    return command


def validate_hattrick_retrained_outputs() -> None:
    expected = (TEST_END - VAL_END) * 3
    for scenario, topo in SCENARIOS.items():
        result_dir = ROOT / "results" / topo / f"{K}sp" / "0"
        for sim in (0, 1):
            path = result_dir / f"hattrick_{scenario}_retrained_values_esm_sim_mlu_{sim}.txt"
            count = pm.line_count(path)
            if count != expected:
                raise RuntimeError(f"{path} has {count} lines, expected {expected}")


def run_hattrick() -> None:
    configure_modules()
    ensure_dirs()
    state = pm.load_state()
    hstate = state.setdefault("hattrick", {})
    for scenario, topo in SCENARIOS.items():
        path_mask = 0 if scenario == "shared" else 1
        for epoch in range(1, EPOCHS + 1):
            key = f"train_{scenario}_retrained_epoch_{epoch:03d}"
            if hstate.get(key):
                print(f"[skip] {key}", flush=True)
                continue
            initial_training = 1 if epoch == 1 else 0
            pm.run_command(key, train_one_epoch_command(topo, path_mask, initial_training))
            hstate[key] = True
            pm.save_state(state)
        for sim in (1, 0):
            key = f"test_{scenario}_retrained_sim{sim}"
            if hstate.get(key):
                print(f"[skip] {key}", flush=True)
                continue
            command = pm.test_command(topo, f"hattrick_{scenario}_retrained", path_mask, "", sim_mf_mlu=sim)
            pm.run_command(key, command)
            hstate[key] = True
            pm.save_state(state)
    validate_hattrick_retrained_outputs()


def run_dotemc() -> None:
    configure_modules()
    ensure_dirs()
    dm.set_seed()
    device = dm.torch.device("cuda" if dm.torch.cuda.is_available() else "cpu")
    dm.validate_data()
    dm.smoke_model(device)
    dm.run_train_stage(device)
    dm.run_test_stage(device)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_hattrick_rows() -> list[dict]:
    configure_modules()
    rows = []
    for scenario, topo in SCENARIOS.items():
        result_dir = ROOT / "results" / topo / f"{K}sp" / "0"
        demands = pm.read_demands(topo)
        oracle = pm.oracle_flow(result_dir)
        norm, norm_mlu = pm.hattrick_values(result_dir, f"{scenario}_retrained")
        method_data = {"Hattrick": (norm * np.maximum(oracle, 1e-9), norm, norm_mlu)}
        for method, filename in (("BEST_MC", "flexile_sim_results_esm_mf_mf_mf.txt"), ("SWAN", "swan_sim_results_esm_mf_mf_mf.txt")):
            carried, mlu = pm.split_simulation(result_dir / filename)
            method_data[method] = (carried, carried / np.maximum(oracle, 1e-9), mlu)
        for method, (carried, norm_fulfill, mlu) in method_data.items():
            for snap_idx in range(TEST_END - VAL_END):
                for class_idx, class_name in enumerate(CLASSES):
                    demand = demands[snap_idx, class_idx]
                    rows.append(
                        {
                            "scenario": scenario,
                            "weights": "na",
                            "topo": topo,
                            "snapshot": VAL_END + snap_idx,
                            "method": method,
                            "class": class_name,
                            "admitted_traffic": float(carried[snap_idx, class_idx]),
                            "demand": float(demand),
                            "fulfill_ratio": float(carried[snap_idx, class_idx] / max(demand, 1e-9)),
                            "mlu": float(mlu[snap_idx, class_idx]),
                            "norm_fulfill": float(norm_fulfill[snap_idx, class_idx]),
                        }
                    )
    write_csv(HATTRICK_INTERNAL_METRICS, rows)
    return rows


def build_dotemc_rows() -> list[dict]:
    configure_modules()
    rows = []
    for scenario in SCENARIOS:
        for weight_label in dm.WEIGHT_CONFIGS:
            metrics_path = dm.model_output_paths(scenario, weight_label)[2]
            for row in read_csv(metrics_path):
                rows.append(
                    {
                        "scenario": row["scenario"],
                        "weights": row["weights"],
                        "topo": row["topo"],
                        "snapshot": int(row["snapshot"]),
                        "method": row["method"],
                        "class": row["class"],
                        "admitted_traffic": float(row["admitted_traffic"]),
                        "demand": float(row["demand"]),
                        "fulfill_ratio": float(row["fulfill_ratio"]),
                        "mlu": float(row["mlu"]),
                        "norm_fulfill": float(row["norm_fulfill"]),
                    }
                )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    groups = sorted({(r["scenario"], r["weights"], r["method"], r["class"]) for r in rows})
    for scenario, weights, method, class_name in groups:
        vals = [r for r in rows if r["scenario"] == scenario and r["weights"] == weights and r["method"] == method and r["class"] == class_name]
        item = {"scenario": scenario, "weights": weights, "method": method, "class": class_name, "n": len(vals)}
        for metric in ("admitted_traffic", "fulfill_ratio", "mlu", "norm_fulfill"):
            arr = np.asarray([float(v[metric]) for v in vals], dtype=np.float64)
            item[f"{metric}_mean"] = float(arr.mean())
            item[f"{metric}_median"] = float(np.median(arr))
            item[f"{metric}_p10"] = float(np.percentile(arr, 10))
            item[f"{metric}_p1"] = float(np.percentile(arr, 1))
            item[f"{metric}_p25"] = float(np.percentile(arr, 25))
            item[f"{metric}_p75"] = float(np.percentile(arr, 75))
        summary.append(item)
    return summary


def paired_gaps(rows: list[dict], scenario: str, class_name: str) -> np.ndarray:
    h = {
        int(r["snapshot"]): float(r["fulfill_ratio"])
        for r in rows
        if r["scenario"] == scenario and r["method"] == "Hattrick" and r["weights"] == "na" and r["class"] == class_name
    }
    d = {
        int(r["snapshot"]): float(r["fulfill_ratio"])
        for r in rows
        if r["scenario"] == scenario and r["method"] == "DOTE_MC" and r["weights"] == DEFAULT_WEIGHT_LABEL and r["class"] == class_name
    }
    keys = sorted(set(h) & set(d))
    if len(keys) != TEST_SNAPSHOTS:
        raise RuntimeError(f"{scenario}/{class_name} has {len(keys)} paired samples, expected {TEST_SNAPSHOTS}")
    return np.asarray([h[key] - d[key] for key in keys], dtype=np.float64)


def bootstrap_gaps(rows: list[dict], iterations: int = 2000) -> list[dict]:
    rng = np.random.default_rng(490)
    out = []
    for scenario in SCENARIOS:
        class_gap_map = {class_name: paired_gaps(rows, scenario, class_name) for class_name in CLASSES}
        for class_name, gaps in class_gap_map.items():
            means = np.empty(iterations, dtype=np.float64)
            for i in range(iterations):
                sample = rng.choice(gaps, size=gaps.shape[0], replace=True)
                means[i] = sample.mean()
            mean_gap = float(gaps.mean())
            ci_low, ci_high = np.percentile(means, [2.5, 97.5])
            out.append(
                {
                    "scenario": scenario,
                    "class": class_name,
                    "n": int(gaps.shape[0]),
                    "mean_gap_hattrick_minus_dotemc": mean_gap,
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "abs_mean_le_0p02": abs(mean_gap) <= 0.02,
                    "ci_contains_zero": ci_low <= 0 <= ci_high,
                    "close_by_rule": abs(mean_gap) <= 0.02 and ci_low <= 0 <= ci_high,
                }
            )

        by_snapshot = []
        for snap_idx in range(TEST_SNAPSHOTS):
            by_snapshot.append(float(np.mean([class_gap_map[class_name][snap_idx] for class_name in CLASSES])))
        gaps = np.asarray(by_snapshot, dtype=np.float64)
        means = np.empty(iterations, dtype=np.float64)
        for i in range(iterations):
            sample = rng.choice(gaps, size=gaps.shape[0], replace=True)
            means[i] = sample.mean()
        mean_gap = float(gaps.mean())
        ci_low, ci_high = np.percentile(means, [2.5, 97.5])
        out.append(
            {
                "scenario": scenario,
                "class": "All",
                "n": int(gaps.shape[0]),
                "mean_gap_hattrick_minus_dotemc": mean_gap,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "abs_mean_le_0p02": abs(mean_gap) <= 0.02,
                "ci_contains_zero": ci_low <= 0 <= ci_high,
                "close_by_rule": abs(mean_gap) <= 0.02 and ci_low <= 0 <= ci_high,
            }
        )
    return out


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def default_plot_rows(rows: list[dict]) -> list[dict]:
    keep = []
    for row in rows:
        if row["method"] == "DOTE_MC" and row["weights"] != DEFAULT_WEIGHT_LABEL:
            continue
        keep.append(row)
    return keep


def draw_cdf(rows: list[dict]) -> None:
    rows = default_plot_rows(rows)
    methods = ("Hattrick", "DOTE_MC", "BEST_MC", "SWAN")
    colors = {"Hattrick": "#2F80ED", "DOTE_MC": "#7B61FF", "BEST_MC": "#27AE60", "SWAN": "#EB5757"}
    width, height = 1420, 1040
    left, top = 84, 108
    panel_w, panel_h = 290, 180
    gap_x, gap_y = 38, 58
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 28), "GEANT 500: DOTE-MC vs Hattrick CDF", fill="#202124", font=font(28, True))
    draw.text((left, 62), "FulfillRatio, test snapshots 400-499, default DOTE-MC weights (1, 0.01, 0.001).", fill="#5f6368", font=font(15))
    x_min, x_max = 0.0, 1.15
    for row_idx, scenario in enumerate(SCENARIOS):
        draw.text((left, top + row_idx * (panel_h + gap_y) - 28), scenario, fill="#202124", font=font(16, True))
        for col_idx, class_name in enumerate(CLASSES):
            px = left + col_idx * (panel_w + gap_x)
            py = top + row_idx * (panel_h + gap_y)
            draw.rectangle((px, py, px + panel_w, py + panel_h), outline="#202124", width=1)
            draw.text((px + panel_w // 2 - 34, py + panel_h + 8), class_name, fill="#202124", font=font(14))
            for tick in np.arange(0, 1.151, 0.25):
                x = int(px + (tick - x_min) / (x_max - x_min) * panel_w)
                draw.line((x, py, x, py + panel_h), fill="#E8EAED")
                draw.text((x - 10, py + panel_h + 24), f"{tick:.2g}", fill="#5f6368", font=font(11))
            for tick in np.linspace(0, 1, 5):
                y = int(py + panel_h - tick * panel_h)
                draw.line((px, y, px + panel_w, y), fill="#E8EAED")
            for method in methods:
                vals = sorted(float(r["fulfill_ratio"]) for r in rows if r["scenario"] == scenario and r["class"] == class_name and r["method"] == method)
                if not vals:
                    continue
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                xs = [x_min] + vals + [x_max]
                ys = [0.0] + list(cdf) + [1.0]
                points = [(int(px + (min(max(v, x_min), x_max) - x_min) / (x_max - x_min) * panel_w), int(py + panel_h - p * panel_h)) for v, p in zip(xs, ys)]
                draw.line(points, fill=colors[method], width=2)
    lx, ly = left, height - 55
    for method in methods:
        draw.line((lx, ly, lx + 36, ly), fill=colors[method], width=5)
        draw.text((lx + 46, ly - 10), method, fill="#202124", font=font(16))
        lx += 190
    image.save(RESULTS / "long500_cdf.png")


def draw_boxplot(rows: list[dict]) -> None:
    rows = default_plot_rows(rows)
    methods = ("Hattrick", "DOTE_MC", "BEST_MC", "SWAN")
    colors = {"Hattrick": "#2F80ED", "DOTE_MC": "#7B61FF", "BEST_MC": "#27AE60", "SWAN": "#EB5757"}
    image = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "GEANT 500: FulfillRatio boxplot", fill="#202124", font=font(28, True))
    draw.text((70, 64), "Default DOTE-MC weights, all classes over test snapshots 400-499.", fill="#5f6368", font=font(15))
    left, top, bottom = 110, 120, 620
    x_gap = 285
    all_vals = [float(r["fulfill_ratio"]) for r in rows]
    y_min, y_max = 0.0, max(1.05, math.ceil((max(all_vals) + 0.05) * 4) / 4)

    def y_of(value: float) -> int:
        return int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))

    for tick in np.arange(0, y_max + 0.001, 0.25):
        y = y_of(float(tick))
        draw.line((left - 40, y, 1210, y), fill="#E8EAED")
        draw.text((32, y - 8), f"{tick:.2f}", fill="#5f6368", font=font(13))
    for idx, scenario in enumerate(SCENARIOS):
        x0 = left + idx * x_gap
        draw.text((x0 - 10, bottom + 22), scenario, fill="#202124", font=font(14))
        for method_idx, method in enumerate(methods):
            vals = np.asarray([float(r["fulfill_ratio"]) for r in rows if r["scenario"] == scenario and r["method"] == method], dtype=np.float64)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            low, high = np.percentile(vals, [5, 95])
            mean = float(vals.mean())
            cx = x0 + method_idx * 38 + 15
            box_left, box_right = cx - 12, cx + 12
            draw.line((cx, y_of(low), cx, y_of(high)), fill=colors[method], width=2)
            draw.line((box_left, y_of(low), box_right, y_of(low)), fill=colors[method], width=2)
            draw.line((box_left, y_of(high), box_right, y_of(high)), fill=colors[method], width=2)
            draw.rectangle((box_left, y_of(q3), box_right, y_of(q1)), outline=colors[method], width=3)
            draw.line((box_left, y_of(med), box_right, y_of(med)), fill=colors[method], width=3)
            draw.ellipse((cx - 3, y_of(mean) - 3, cx + 3, y_of(mean) + 3), fill=colors[method])
    lx, ly = 70, 680
    for method in methods:
        draw.rectangle((lx, ly, lx + 24, ly + 14), fill=colors[method])
        draw.text((lx + 34, ly - 4), method, fill="#202124", font=font(15))
        lx += 160
    image.save(RESULTS / "long500_boxplot.png")


def avg(summary: list[dict], scenario: str, method: str, weights: str = "na") -> float:
    vals = [float(row["fulfill_ratio_mean"]) for row in summary if row["scenario"] == scenario and row["method"] == method and row["weights"] == weights]
    return float(np.mean(vals))


def report_markdown(summary: list[dict], boot: list[dict]) -> str:
    lines = [
        "# GEANT 500 Priority-Mask DOTE-MC vs Hattrick\n\n",
        "This experiment extends the earlier 100-slice priority-mask test to GEANT snapshots 0-499, with train 0-349, validation 350-399, and test 400-499.\n\n",
        "The purpose is to test whether Hattrick's structural advantage over a simple DNN/MLP baseline remains visible when high/medium/low priorities have different allowed path sets.\n\n",
        "## Mean FulfillRatio\n\n",
        "| Scenario | Hattrick | DOTE-MC default | DOTE-MC sensitivity | BEST_MC | SWAN | Hattrick - DOTE default |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    gaps = {}
    for scenario in SCENARIOS:
        hat = avg(summary, scenario, "Hattrick")
        dote = avg(summary, scenario, "DOTE_MC", DEFAULT_WEIGHT_LABEL)
        dote_sens = avg(summary, scenario, "DOTE_MC", SENS_WEIGHT_LABEL)
        best = avg(summary, scenario, "BEST_MC")
        swan = avg(summary, scenario, "SWAN")
        gap = hat - dote
        gaps[scenario] = gap
        lines.append(f"| {scenario} | {hat:.6f} | {dote:.6f} | {dote_sens:.6f} | {best:.6f} | {swan:.6f} | {gap:+.6f} |\n")

    lines.extend(
        [
            "\n## Paired Bootstrap Gap\n\n",
            "Gap is Hattrick minus DOTE-MC default on paired test snapshots. The `All` row averages the three classes per snapshot before bootstrapping.\n\n",
            "| Scenario | Class | Mean gap | 95% CI | Close by rule |\n",
            "|---|---|---:|---:|---|\n",
        ]
    )
    for row in boot:
        if row["class"] != "All":
            continue
        close = "yes" if str(row["close_by_rule"]).lower() == "true" else "no"
        lines.append(
            f"| {row['scenario']} | {row['class']} | {float(row['mean_gap_hattrick_minus_dotemc']):+.6f} | "
            f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}] | {close} |\n"
        )

    all_close = all(
        abs(float(row["mean_gap_hattrick_minus_dotemc"])) <= 0.02
        for row in boot
        if row["class"] == "All"
    )
    shared_advantage = gaps["shared"] > 0.05
    masked_close = all(abs(gaps[scenario]) <= 0.02 for scenario in ("mild", "medium", "strict"))
    masked_hattrick_win = any(gaps[scenario] > 0.03 for scenario in ("mild", "medium", "strict"))

    if shared_advantage and masked_close:
        conclusion = "The result supports the claim that priority-specific path sets weaken Hattrick's structural advantage."
    elif all_close:
        conclusion = "DOTE-MC is close to Hattrick across all four scenarios. In this 500-slice GEANT sub-experiment, Hattrick's structural advantage is not observed."
    elif masked_hattrick_win:
        conclusion = "Hattrick still has a visible advantage in at least one masked scenario, so this result does not support a structure-failure claim."
    else:
        conclusion = "The result is mixed; use per-class tails and MLU before making a strong structure claim."

    lines.extend(
        [
            "\n## Interpretation\n\n",
            conclusion + "\n\n",
            "This should not be written as a proof that Hattrick fails. A safer interpretation is that, under this synthetic priority-specific path-mask setting, a simple MLP baseline can match Hattrick on the tested window. That weakens the evidence that Hattrick's staged GNN/Transformer/RAU structure is the source of the observed performance in this setting.\n\n",
            "The likely reason is that fixing a common 8-path universe and then applying simple masks removes part of the routing-structure complexity that Hattrick was designed to exploit. With only one static topology and predictable GEANT traffic, the MLP can learn a strong direct mapping from predicted demands to path splits.\n\n",
            "This is still a 500-slice sub-experiment, not a full paper-scale result. A stronger claim needs more test windows, multiple random seeds, and preferably real operator path policies rather than synthetic masks.\n",
        ]
    )
    return "".join(lines)


def run_report() -> None:
    configure_modules()
    ensure_dirs()
    pm.validate_gurobi_outputs()
    validate_hattrick_retrained_outputs()
    dm.validate_test_outputs()
    rows = build_hattrick_rows() + build_dotemc_rows()
    summary = summarize(rows)
    boot = bootstrap_gaps(rows)
    write_csv(RESULTS / "long500_metrics.csv", rows)
    write_csv(RESULTS / "long500_summary_stats.csv", summary)
    write_csv(RESULTS / "long500_gap_bootstrap.csv", boot)
    draw_cdf(rows)
    draw_boxplot(rows)
    (RESULTS / "long500_failure_argument.md").write_text(report_markdown(summary, boot), encoding="utf-8")
    print(f"[ok] wrote GEANT 500 report to {RESULTS}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "prepare", "validate", "gurobi", "hattrick", "dotemc", "report"], default="all")
    args = parser.parse_args()
    configure_modules()
    ensure_dirs()
    if args.stage in ("all", "prepare"):
        prepare()
    if args.stage in ("all", "validate"):
        validate_data()
    if args.stage in ("all", "gurobi"):
        run_gurobi()
    if args.stage in ("all", "hattrick"):
        run_hattrick()
    if args.stage in ("all", "dotemc"):
        run_dotemc()
    if args.stage in ("all", "report"):
        run_report()


if __name__ == "__main__":
    main()
