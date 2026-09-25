from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.append(str(THIS_DIR))
sys.path.append(str(ROOT))

import run_dotemc_priority_mask_experiment as dm
import run_priority_mask_experiment as pm


K = 8
TRAIN_END = 350
VAL_END = 400
TEST_END = 500
EPOCHS = 60
BATCH_SIZE = 8
LOAD_FACTOR = 2.0
CLASSES = ("High", "Medium", "Low")
SOURCE_TOPOS = {
    "shared": "geant_priomask500_shared",
    "strict": "geant_priomask500_strict",
}
SCENARIOS = {
    scenario: f"{topo}_load2x_train"
    for scenario, topo in SOURCE_TOPOS.items()
}
DEFAULT_WEIGHT_LABEL = "w_1_0p01_0p001"
SENS_WEIGHT_LABEL = "w_1_0p1_0p01"

RESULTS = THIS_DIR / "results_load2x_retrain_shared_strict"
LOGS = THIS_DIR / "logs_load2x_retrain_shared_strict"
HATTRICK_STATE = THIS_DIR / "load2x_retrain_hattrick_state.json"
DOTE_STATE = THIS_DIR / "load2x_retrain_dotemc_state.json"
BASELINE_RESULTS = THIS_DIR / "results_dotemc_priority_masks_500slice"
FROZEN_RESULTS = THIS_DIR / "results_load2x_stress_500slice"
BASELINE_METRICS = BASELINE_RESULTS / "long500_metrics.csv"
FROZEN_METRICS = FROZEN_RESULTS / "load2x_metrics.csv"
MODEL_AUDIT = RESULTS / "original_hattrick_model_audit.json"


def configure_modules() -> None:
    pm.RESULTS = RESULTS
    pm.LOGS = LOGS / "hattrick_gurobi"
    pm.STATE_FILE = HATTRICK_STATE
    pm.TRAIN_END = TRAIN_END
    pm.VAL_END = VAL_END
    pm.TEST_END = TEST_END
    pm.EPOCHS = EPOCHS
    pm.BATCH_SIZE = BATCH_SIZE
    pm.SCENARIOS = SCENARIOS

    dm.RESULTS = RESULTS
    dm.OUTPUTS = RESULTS / "outputs"
    dm.LOGS = LOGS / "dotemc"
    dm.STATE_FILE = DOTE_STATE
    dm.TRAIN_END = TRAIN_END
    dm.VAL_END = VAL_END
    dm.TEST_END = TEST_END
    dm.EPOCHS = EPOCHS
    dm.BATCH_SIZE = BATCH_SIZE
    dm.SCENARIOS = SCENARIOS
    dm.DEFAULT_WEIGHT_LABEL = DEFAULT_WEIGHT_LABEL


def ensure_dirs() -> None:
    for directory in (RESULTS, LOGS, LOGS / "hattrick_gurobi", LOGS / "dotemc"):
        directory.mkdir(parents=True, exist_ok=True)


def load_hattrick_state() -> dict:
    if HATTRICK_STATE.exists():
        state = json.loads(HATTRICK_STATE.read_text(encoding="utf-8"))
    else:
        state = {}
    state.setdefault("prepare", False)
    state.setdefault("gurobi", {})
    state.setdefault("hattrick", {})
    state.setdefault("report", False)
    return state


def save_hattrick_state(state: dict) -> None:
    HATTRICK_STATE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def manifest_rows(topo: str) -> list[tuple[str, str, str]]:
    path = ROOT / "manifest" / f"{topo}_manifest.txt"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            topology, pairs, tm = [part.strip() for part in line.split(",")]
            rows.append((topology, pairs, tm))
    return rows


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


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def copy_path_artifacts(source_topo: str, target_topo: str, scenario: str) -> None:
    copies = (
        (
            ROOT / "topologies" / "paths_dict" / f"{source_topo}_{K}_paths_dict_cluster_0.pkl",
            ROOT / "topologies" / "paths_dict" / f"{target_topo}_{K}_paths_dict_cluster_0.pkl",
        ),
        (
            ROOT / "topologies" / "paths" / f"{source_topo}_{K}_paths_cluster_0.pkl",
            ROOT / "topologies" / "paths" / f"{target_topo}_{K}_paths_cluster_0.pkl",
        ),
        (
            ROOT / "topologies" / "padded_edge_ids_per_path" / f"{source_topo}_{K}_paths_cluster_0_padded_edge_ids_per_path.pkl",
            ROOT / "topologies" / "padded_edge_ids_per_path" / f"{target_topo}_{K}_paths_cluster_0_padded_edge_ids_per_path.pkl",
        ),
        (
            ROOT / "topologies" / "padded_edge_ids_per_path" / f"{source_topo}_{K}_paths_cluster_0_edge_ids_dict.pkl",
            ROOT / "topologies" / "padded_edge_ids_per_path" / f"{target_topo}_{K}_paths_cluster_0_edge_ids_dict.pkl",
        ),
    )
    for source, target in copies:
        copy_if_exists(source, target)
    if scenario == "strict":
        copy_if_exists(
            ROOT / "topologies" / "path_masks" / f"{source_topo}_{K}_path_masks_cluster_0.pkl",
            ROOT / "topologies" / "path_masks" / f"{target_topo}_{K}_path_masks_cluster_0.pkl",
        )


def scale_pickle(source: Path, target: Path) -> tuple[float, float]:
    with source.open("rb") as handle:
        original = np.asarray(pickle.load(handle))
    scaled = original * LOAD_FACTOR
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(scaled, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return float(original.sum()), float(np.asarray(scaled).sum())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def original_model_metadata() -> dict:
    metadata = {}
    for scenario, topo in SOURCE_TOPOS.items():
        path = ROOT / f"hattrick_{topo}_{K}sp.pkl"
        stat = path.stat()
        metadata[scenario] = {
            "path": str(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(path),
        }
    return metadata


def prepare() -> None:
    configure_modules()
    ensure_dirs()
    state = load_hattrick_state()
    if state["prepare"]:
        print("[skip] complete load2x retraining data already prepared", flush=True)
        validate_data()
        return

    audit_rows = []
    for scenario, source_topo in SOURCE_TOPOS.items():
        target_topo = SCENARIOS[scenario]
        rows = manifest_rows(source_topo)[:TEST_END]
        if len(rows) != TEST_END:
            raise RuntimeError(f"{source_topo} has {len(rows)} rows, expected {TEST_END}")
        copy_path_artifacts(source_topo, target_topo, scenario)

        seen_topologies: set[str] = set()
        seen_pairs: set[str] = set()
        seen_tms: set[str] = set()
        manifest_lines = []
        for snapshot_idx, (topology_filename, pairs_filename, tm_filename) in enumerate(rows):
            manifest_lines.append(f"{topology_filename},{pairs_filename},{tm_filename}\n")
            if topology_filename not in seen_topologies:
                copy_if_exists(
                    ROOT / "topologies" / source_topo / topology_filename,
                    ROOT / "topologies" / target_topo / topology_filename,
                )
                seen_topologies.add(topology_filename)
            if pairs_filename not in seen_pairs:
                copy_if_exists(
                    ROOT / "pairs" / source_topo / pairs_filename,
                    ROOT / "pairs" / target_topo / pairs_filename,
                )
                seen_pairs.add(pairs_filename)
            if tm_filename in seen_tms:
                continue
            for priority in (1, 2, 3):
                for suffix in ("", "_esm"):
                    source = ROOT / "traffic_matrices" / f"{source_topo}_{priority}{suffix}" / tm_filename
                    target = ROOT / "traffic_matrices" / f"{target_topo}_{priority}{suffix}" / tm_filename
                    original_sum, scaled_sum = scale_pickle(source, target)
                    audit_rows.append(
                        {
                            "scenario": scenario,
                            "snapshot": snapshot_idx,
                            "tm_filename": tm_filename,
                            "priority": priority,
                            "matrix": "predicted_esm" if suffix else "ground_truth",
                            "original_sum": original_sum,
                            "scaled_sum": scaled_sum,
                            "ratio": scaled_sum / max(original_sum, 1e-12),
                        }
                    )
            seen_tms.add(tm_filename)
        (ROOT / "manifest" / f"{target_topo}_manifest.txt").write_text("".join(manifest_lines), encoding="utf-8")

    write_csv(RESULTS / "load2x_retrain_data_audit.csv", audit_rows)
    MODEL_AUDIT.write_text(
        json.dumps({"before": original_model_metadata()}, indent=2),
        encoding="utf-8",
    )
    (RESULTS / "load2x_retrain_config.json").write_text(
        json.dumps(
            {
                "load_factor": LOAD_FACTOR,
                "train": [0, TRAIN_END],
                "validation": [TRAIN_END, VAL_END],
                "test": [VAL_END, TEST_END],
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "seed": 490,
                "scenarios": SCENARIOS,
                "scaled_matrices": ["ground_truth", "esm_prediction"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state["prepare"] = True
    save_hattrick_state(state)
    validate_data()


def validate_data() -> None:
    configure_modules()
    for scenario, topo in SCENARIOS.items():
        target_rows = manifest_rows(topo)
        source_rows = manifest_rows(SOURCE_TOPOS[scenario])[:TEST_END]
        if len(target_rows) != TEST_END:
            raise RuntimeError(f"{topo} manifest has {len(target_rows)} rows, expected {TEST_END}")

        source_path = ROOT / "topologies" / "paths_dict" / f"{SOURCE_TOPOS[scenario]}_{K}_paths_dict_cluster_0.pkl"
        target_path = ROOT / "topologies" / "paths_dict" / f"{topo}_{K}_paths_dict_cluster_0.pkl"
        with source_path.open("rb") as handle:
            source_paths = pickle.load(handle)
        with target_path.open("rb") as handle:
            target_paths = pickle.load(handle)
        if source_paths != target_paths or len(target_paths) != 462 or any(len(value) != K for value in target_paths.values()):
            raise RuntimeError(f"Path universe mismatch for {topo}")

        if scenario == "strict":
            source_mask = ROOT / "topologies" / "path_masks" / f"{SOURCE_TOPOS[scenario]}_{K}_path_masks_cluster_0.pkl"
            target_mask = ROOT / "topologies" / "path_masks" / f"{topo}_{K}_path_masks_cluster_0.pkl"
            with source_mask.open("rb") as handle:
                source_masks = np.asarray(pickle.load(handle), dtype=bool)
            with target_mask.open("rb") as handle:
                target_masks = np.asarray(pickle.load(handle), dtype=bool)
            if source_masks.shape != (3, 462, K) or not np.array_equal(source_masks, target_masks):
                raise RuntimeError(f"Strict path mask mismatch for {topo}")

        for source_row, target_row in zip(source_rows, target_rows):
            source_tm = source_row[2]
            target_tm = target_row[2]
            for priority in (1, 2, 3):
                for suffix in ("", "_esm"):
                    with (ROOT / "traffic_matrices" / f"{SOURCE_TOPOS[scenario]}_{priority}{suffix}" / source_tm).open("rb") as handle:
                        original = np.asarray(pickle.load(handle), dtype=np.float64)
                    with (ROOT / "traffic_matrices" / f"{topo}_{priority}{suffix}" / target_tm).open("rb") as handle:
                        scaled = np.asarray(pickle.load(handle), dtype=np.float64)
                    if not np.allclose(scaled, original * LOAD_FACTOR, rtol=1e-7, atol=1e-10):
                        raise RuntimeError(f"Scaling mismatch: {topo}/{priority}{suffix}/{target_tm}")
    print("[ok] full 500-slice load2x retraining data validation passed", flush=True)


def run_gurobi() -> None:
    configure_modules()
    ensure_dirs()
    pm.run_gurobi()


def set_cli_arg(command: list[str], option: str, value: str | int) -> list[str]:
    updated = list(command)
    try:
        index = updated.index(option)
    except ValueError as exc:
        raise RuntimeError(f"Missing CLI option {option}") from exc
    updated[index + 1] = str(value)
    return updated


def train_one_epoch_command(topo: str, path_mask: int, initial_training: int) -> list[str]:
    command = pm.train_command(topo, path_mask)
    command = set_cli_arg(command, "--epochs", 1)
    command = set_cli_arg(command, "--initial_training", initial_training)
    return command


def run_hattrick() -> None:
    configure_modules()
    ensure_dirs()
    state = load_hattrick_state()
    hstate = state["hattrick"]
    for scenario, topo in SCENARIOS.items():
        path_mask = 0 if scenario == "shared" else 1
        for epoch in range(1, EPOCHS + 1):
            key = f"train_{scenario}_epoch_{epoch:03d}"
            if hstate.get(key):
                print(f"[skip] {key}", flush=True)
                continue
            pm.run_command(
                key,
                train_one_epoch_command(topo, path_mask, 1 if epoch == 1 else 0),
            )
            hstate[key] = True
            save_hattrick_state(state)
        for sim in (1, 0):
            key = f"test_{scenario}_sim{sim}"
            if hstate.get(key):
                print(f"[skip] {key}", flush=True)
                continue
            command = pm.test_command(
                topo,
                f"hattrick_{scenario}_retrained_load2x",
                path_mask,
                "",
                sim_mf_mlu=sim,
            )
            pm.run_command(key, command)
            hstate[key] = True
            save_hattrick_state(state)
    validate_hattrick_outputs()


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def validate_hattrick_outputs() -> None:
    expected = (TEST_END - VAL_END) * 3
    for scenario, topo in SCENARIOS.items():
        result_dir = ROOT / "results" / topo / f"{K}sp" / "0"
        for sim in (1, 0):
            output = result_dir / f"hattrick_{scenario}_retrained_load2x_values_esm_sim_mlu_{sim}.txt"
            if line_count(output) != expected:
                raise RuntimeError(f"{output} has {line_count(output)} lines, expected {expected}")
        model_path = ROOT / f"hattrick_{topo}_{K}sp.pkl"
        if not model_path.exists():
            raise RuntimeError(f"Missing retrained model: {model_path}")
    print("[ok] retrained Hattrick outputs passed", flush=True)


def run_dotemc() -> None:
    configure_modules()
    ensure_dirs()
    dm.set_seed()
    device = dm.torch.device("cuda" if dm.torch.cuda.is_available() else "cpu")
    dm.validate_data()
    # The reused DOTE-MC smoke check names its masked scenario "medium".
    # This control only retains shared/strict, so point that check at strict.
    dm.SCENARIOS = {"medium": SCENARIOS["strict"]}
    try:
        dm.smoke_model(device)
    finally:
        dm.SCENARIOS = SCENARIOS
    dm.run_train_stage(device)
    dm.run_test_stage(device)


def build_retrained_rows() -> list[dict]:
    rows = []
    for scenario, topo in SCENARIOS.items():
        result_dir = ROOT / "results" / topo / f"{K}sp" / "0"
        demands = pm.read_demands(topo)
        oracle = pm.oracle_flow(result_dir)
        norm, normalized_mlu = pm.hattrick_values(result_dir, f"{scenario}_retrained_load2x")
        method_data = {"Hattrick": (norm * np.maximum(oracle, 1e-9), norm, normalized_mlu)}
        for method, filename in (
            ("BEST_MC", "flexile_sim_results_esm_mf_mf_mf.txt"),
            ("SWAN", "swan_sim_results_esm_mf_mf_mf.txt"),
        ):
            carried, mlu = pm.split_simulation(result_dir / filename)
            method_data[method] = (carried, carried / np.maximum(oracle, 1e-9), mlu)
        for method, (carried, norm_fulfill, mlu) in method_data.items():
            for local_idx in range(TEST_END - VAL_END):
                for class_idx, class_name in enumerate(CLASSES):
                    demand = demands[local_idx, class_idx]
                    rows.append(
                        {
                            "regime": "retrained_2x",
                            "load_factor": LOAD_FACTOR,
                            "scenario": scenario,
                            "weights": "na",
                            "topo": topo,
                            "snapshot": VAL_END + local_idx,
                            "method": method,
                            "class": class_name,
                            "admitted_traffic": float(carried[local_idx, class_idx]),
                            "demand": float(demand),
                            "fulfill_ratio": float(carried[local_idx, class_idx] / max(demand, 1e-9)),
                            "mlu": float(mlu[local_idx, class_idx]),
                            "norm_fulfill": float(norm_fulfill[local_idx, class_idx]),
                        }
                    )
        for weight_label in dm.WEIGHT_CONFIGS:
            metrics_path = dm.model_output_paths(scenario, weight_label)[2]
            for source in read_csv(metrics_path):
                rows.append(
                    {
                        "regime": "retrained_2x",
                        "load_factor": LOAD_FACTOR,
                        "scenario": scenario,
                        "weights": source["weights"],
                        "topo": source["topo"],
                        "snapshot": int(source["snapshot"]),
                        "method": source["method"],
                        "class": source["class"],
                        "admitted_traffic": float(source["admitted_traffic"]),
                        "demand": float(source["demand"]),
                        "fulfill_ratio": float(source["fulfill_ratio"]),
                        "mlu": float(source["mlu"]),
                        "norm_fulfill": float(source["norm_fulfill"]),
                    }
                )
    return rows


def normalize_existing_rows(path: Path, regime: str, load_factor: float) -> list[dict]:
    rows = []
    for source in read_csv(path):
        if source["scenario"] not in SCENARIOS:
            continue
        rows.append(
            {
                "regime": regime,
                "load_factor": load_factor,
                "scenario": source["scenario"],
                "weights": source["weights"],
                "topo": source["topo"],
                "snapshot": int(source["snapshot"]),
                "method": source["method"],
                "class": source["class"],
                "admitted_traffic": float(source["admitted_traffic"]),
                "demand": float(source["demand"]),
                "fulfill_ratio": float(source["fulfill_ratio"]),
                "mlu": float(source["mlu"]),
                "norm_fulfill": float(source["norm_fulfill"]),
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    groups = sorted({(row["regime"], row["scenario"], row["weights"], row["method"], row["class"]) for row in rows})
    for regime, scenario, weights, method, class_name in groups:
        selected = [
            row
            for row in rows
            if row["regime"] == regime
            and row["scenario"] == scenario
            and row["weights"] == weights
            and row["method"] == method
            and row["class"] == class_name
        ]
        item = {
            "regime": regime,
            "scenario": scenario,
            "weights": weights,
            "method": method,
            "class": class_name,
            "n": len(selected),
        }
        for metric in ("admitted_traffic", "demand", "fulfill_ratio", "mlu", "norm_fulfill"):
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_median"] = float(np.median(values))
            item[f"{metric}_p1"] = float(np.percentile(values, 1))
            item[f"{metric}_p10"] = float(np.percentile(values, 10))
            item[f"{metric}_p25"] = float(np.percentile(values, 25))
            item[f"{metric}_p75"] = float(np.percentile(values, 75))
        summary.append(item)
    return summary


def summary_index(summary: list[dict]) -> dict:
    return {
        (row["regime"], row["scenario"], row["weights"], row["method"], row["class"]): row
        for row in summary
    }


def build_recovery(summary: list[dict]) -> list[dict]:
    indexed = summary_index(summary)
    rows = []
    methods = (
        ("Hattrick", "na"),
        ("DOTE_MC", DEFAULT_WEIGHT_LABEL),
        ("DOTE_MC", SENS_WEIGHT_LABEL),
    )
    for scenario in SCENARIOS:
        for method, weights in methods:
            for class_name in CLASSES:
                baseline = indexed[("baseline_1x", scenario, weights, method, class_name)]
                frozen = indexed[("frozen_2x", scenario, weights, method, class_name)]
                retrained = indexed[("retrained_2x", scenario, weights, method, class_name)]
                item = {
                    "scenario": scenario,
                    "weights": weights,
                    "method": method,
                    "class": class_name,
                }
                for metric in ("fulfill_ratio", "norm_fulfill"):
                    for stat in ("mean", "p1", "p10"):
                        column = f"{metric}_{stat}"
                        baseline_value = float(baseline[column])
                        frozen_value = float(frozen[column])
                        retrained_value = float(retrained[column])
                        drop = baseline_value - frozen_value
                        gain = retrained_value - frozen_value
                        item[f"baseline_1x_{column}"] = baseline_value
                        item[f"frozen_2x_{column}"] = frozen_value
                        item[f"retrained_2x_{column}"] = retrained_value
                        item[f"retrained_minus_frozen_{column}"] = gain
                        item[f"recovery_fraction_{column}"] = gain / drop if abs(drop) > 1e-9 else ""
                rows.append(item)
    return rows


def display_method(method: str, weights: str) -> str:
    if method != "DOTE_MC":
        return method
    if weights == DEFAULT_WEIGHT_LABEL:
        return "DOTE-MC default"
    return "DOTE-MC sensitivity"


def plot_series(rows: list[dict], scenario: str, class_name: str) -> list[tuple[str, np.ndarray]]:
    specs = (
        ("Hattrick frozen", "frozen_2x", "Hattrick", "na"),
        ("Hattrick retrained", "retrained_2x", "Hattrick", "na"),
        ("DOTE frozen", "frozen_2x", "DOTE_MC", DEFAULT_WEIGHT_LABEL),
        ("DOTE retrained", "retrained_2x", "DOTE_MC", DEFAULT_WEIGHT_LABEL),
        ("BEST_MC 2x", "retrained_2x", "BEST_MC", "na"),
        ("SWAN 2x", "retrained_2x", "SWAN", "na"),
    )
    series = []
    for label, regime, method, weights in specs:
        values = np.asarray(
            [
                float(row["norm_fulfill"])
                for row in rows
                if row["scenario"] == scenario
                and row["class"] == class_name
                and row["regime"] == regime
                and row["method"] == method
                and row["weights"] == weights
            ],
            dtype=np.float64,
        )
        series.append((label, values))
    return series


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_cdf(rows: list[dict]) -> None:
    labels = tuple(label for label, _ in plot_series(rows, "shared", "High"))
    colors = ("#2563EB", "#0F766E", "#A855F7", "#E11D48", "#16A34A", "#EA580C")
    width, height = 1500, 900
    left, top = 95, 115
    panel_w, panel_h = 390, 255
    gap_x, gap_y = 70, 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 28), "GEANT 2x load: frozen vs retrained models", fill="#202124", font=font(28, True))
    draw.text((left, 66), "Empirical CDF of NormFulFill, 100 true test snapshots; curves start at y=0.", fill="#5F6368", font=font(15))
    x_min, x_max = 0.0, 1.45
    for row_idx, scenario in enumerate(SCENARIOS):
        for col_idx, class_name in enumerate(CLASSES):
            px = left + col_idx * (panel_w + gap_x)
            py = top + row_idx * (panel_h + gap_y)
            draw.rectangle((px, py, px + panel_w, py + panel_h), outline="#202124", width=1)
            draw.text((px + 8, py - 28), f"{scenario} / {class_name}", fill="#202124", font=font(15, True))
            for tick in np.arange(0, 1.51, 0.25):
                x = int(px + (tick - x_min) / (x_max - x_min) * panel_w)
                draw.line((x, py, x, py + panel_h), fill="#E8EAED")
                if tick <= x_max:
                    draw.text((x - 12, py + panel_h + 8), f"{tick:.2g}", fill="#5F6368", font=font(11))
            for tick in np.linspace(0, 1, 5):
                y = int(py + panel_h - tick * panel_h)
                draw.line((px, y, px + panel_w, y), fill="#E8EAED")
                if col_idx == 0:
                    draw.text((px - 42, y - 7), f"{tick:.2g}", fill="#5F6368", font=font(11))
            for color, (_, values) in zip(colors, plot_series(rows, scenario, class_name)):
                sorted_values = np.sort(values)
                xs = np.concatenate(([0.0], sorted_values, [x_max]))
                ys = np.concatenate(([0.0], np.arange(1, len(sorted_values) + 1) / len(sorted_values), [1.0]))
                points = [
                    (
                        int(px + (min(max(value, x_min), x_max) - x_min) / (x_max - x_min) * panel_w),
                        int(py + panel_h - probability * panel_h),
                    )
                    for value, probability in zip(xs, ys)
                ]
                draw.line(points, fill=color, width=3)
    legend_y = height - 48
    legend_x = left
    for index, (label, color) in enumerate(zip(labels, colors)):
        if index == 3:
            legend_x = 770
            legend_y = height - 48
        draw.line((legend_x, legend_y, legend_x + 34, legend_y), fill=color, width=5)
        draw.text((legend_x + 44, legend_y - 10), label, fill="#202124", font=font(14))
        legend_x += 220
    image.save(RESULTS / "load2x_retrain_cdf.png")


def draw_boxplot(rows: list[dict]) -> None:
    labels = tuple(label for label, _ in plot_series(rows, "shared", "High"))
    colors = ("#2563EB", "#0F766E", "#A855F7", "#E11D48", "#16A34A", "#EA580C")
    width, height = 1500, 900
    left, top = 95, 115
    panel_w, panel_h = 390, 255
    gap_x, gap_y = 70, 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 28), "GEANT 2x load: NormFulFill distributions", fill="#202124", font=font(28, True))
    draw.text((left, 66), "Boxes show P25/P50/P75; whiskers show P5/P95; dots show means.", fill="#5F6368", font=font(15))
    y_min, y_max = 0.0, 1.45
    for row_idx, scenario in enumerate(SCENARIOS):
        for col_idx, class_name in enumerate(CLASSES):
            px = left + col_idx * (panel_w + gap_x)
            py = top + row_idx * (panel_h + gap_y)
            draw.rectangle((px, py, px + panel_w, py + panel_h), outline="#202124", width=1)
            draw.text((px + 8, py - 28), f"{scenario} / {class_name}", fill="#202124", font=font(15, True))

            def y_of(value: float) -> int:
                clipped = min(max(value, y_min), y_max)
                return int(py + panel_h - (clipped - y_min) / (y_max - y_min) * panel_h)

            for tick in np.arange(0, 1.51, 0.25):
                if tick > y_max:
                    continue
                y = y_of(float(tick))
                draw.line((px, y, px + panel_w, y), fill="#E8EAED")
                if col_idx == 0:
                    draw.text((px - 42, y - 7), f"{tick:.2g}", fill="#5F6368", font=font(11))
            for series_idx, (color, (_, values)) in enumerate(zip(colors, plot_series(rows, scenario, class_name))):
                q5, q25, median, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
                mean = float(values.mean())
                cx = px + 35 + series_idx * 59
                draw.line((cx, y_of(q5), cx, y_of(q95)), fill=color, width=2)
                draw.line((cx - 10, y_of(q5), cx + 10, y_of(q5)), fill=color, width=2)
                draw.line((cx - 10, y_of(q95), cx + 10, y_of(q95)), fill=color, width=2)
                draw.rectangle((cx - 16, y_of(q75), cx + 16, y_of(q25)), outline=color, width=3)
                draw.line((cx - 16, y_of(median), cx + 16, y_of(median)), fill=color, width=3)
                draw.ellipse((cx - 4, y_of(mean) - 4, cx + 4, y_of(mean) + 4), fill=color)
                draw.text((cx - 4, py + panel_h + 8), str(series_idx + 1), fill="#5F6368", font=font(11))
    legend_y = height - 48
    legend_x = left
    for index, (label, color) in enumerate(zip(labels, colors)):
        if index == 3:
            legend_x = 770
            legend_y = height - 48
        draw.rectangle((legend_x, legend_y - 7, legend_x + 22, legend_y + 7), fill=color)
        draw.text((legend_x + 32, legend_y - 10), f"{index + 1}: {label}", fill="#202124", font=font(14))
        legend_x += 220
    image.save(RESULTS / "load2x_retrain_boxplot.png")


def classify_hattrick(summary: list[dict], scenario: str) -> tuple[str, dict]:
    indexed = summary_index(summary)
    frozen = indexed[("frozen_2x", scenario, "na", "Hattrick", "Medium")]
    retrained = indexed[("retrained_2x", scenario, "na", "Hattrick", "Medium")]
    best = indexed[("retrained_2x", scenario, "na", "BEST_MC", "Medium")]
    p10 = float(retrained["norm_fulfill_p10"])
    p1 = float(retrained["norm_fulfill_p1"])
    gain = p10 - float(frozen["norm_fulfill_p10"])
    best_gap = float(best["norm_fulfill_p10"]) - p10
    if p10 >= 0.95 and p1 >= 0.90 and best_gap <= 0.03:
        decision = "training-distribution problem"
    elif gain > 0.05 and best_gap > 0.05:
        decision = "mixed training and structure/loss limitation"
    elif gain <= 0.02 and best_gap > 0.05:
        decision = "structure/loss limitation supported"
    else:
        decision = "mixed or inconclusive"
    return decision, {
        "frozen_p10": float(frozen["norm_fulfill_p10"]),
        "retrained_p10": p10,
        "retrained_p1": p1,
        "best_p10": float(best["norm_fulfill_p10"]),
        "gain_p10": gain,
        "best_gap_p10": best_gap,
    }


def report_markdown(summary: list[dict]) -> str:
    indexed = summary_index(summary)
    lines = [
        "# GEANT 2x Load Retraining Control\n\n",
        "This control compares 1x baseline, frozen 1x-trained models on 2x traffic, and models retrained from scratch on complete 2x train/validation data. Architecture, paths, masks, capacity, split, and hyperparameters are unchanged.\n\n",
        "Each result cell is `Mean / P1 / P10` for NormFulFill.\n\n",
        "| Scenario | Method | Class | Frozen 2x | Retrained 2x | BEST_MC 2x |\n",
        "|---|---|---|---:|---:|---:|\n",
    ]
    methods = (
        ("Hattrick", "na"),
        ("DOTE_MC", DEFAULT_WEIGHT_LABEL),
        ("DOTE_MC", SENS_WEIGHT_LABEL),
    )
    for scenario in SCENARIOS:
        for method, weights in methods:
            for class_name in CLASSES:
                frozen = indexed[("frozen_2x", scenario, weights, method, class_name)]
                retrained = indexed[("retrained_2x", scenario, weights, method, class_name)]
                best = indexed[("retrained_2x", scenario, "na", "BEST_MC", class_name)]

                def triple(row: dict) -> str:
                    return f"{float(row['norm_fulfill_mean']):.4f} / {float(row['norm_fulfill_p1']):.4f} / {float(row['norm_fulfill_p10']):.4f}"

                lines.append(
                    f"| {scenario} | {display_method(method, weights)} | {class_name} | "
                    f"{triple(frozen)} | {triple(retrained)} | {triple(best)} |\n"
                )
    lines.extend(
        [
            "\n## Retrained system metrics\n\n",
            "FulfillRatio is `Mean / P1 / P10`; admitted traffic and MLU are means over the 100 test snapshots.\n\n",
            "| Scenario | Method | Class | FulfillRatio | Admitted traffic | MLU |\n",
            "|---|---|---|---:|---:|---:|\n",
        ]
    )
    system_methods = (
        ("Hattrick", "na"),
        ("DOTE_MC", SENS_WEIGHT_LABEL),
        ("BEST_MC", "na"),
        ("SWAN", "na"),
    )
    for scenario in SCENARIOS:
        for method, weights in system_methods:
            for class_name in CLASSES:
                row = indexed[("retrained_2x", scenario, weights, method, class_name)]
                fulfill = (
                    f"{float(row['fulfill_ratio_mean']):.4f} / "
                    f"{float(row['fulfill_ratio_p1']):.4f} / "
                    f"{float(row['fulfill_ratio_p10']):.4f}"
                )
                lines.append(
                    f"| {scenario} | {display_method(method, weights)} | {class_name} | {fulfill} | "
                    f"{float(row['admitted_traffic_mean']):.4f} | {float(row['mlu_mean']):.4f} |\n"
                )
    lines.extend(
        [
            "\n## Decision\n\n",
            "The pre-registered rule focuses on Medium priority; Low values above 1 do not count as recovery because they may indicate priority inversion.\n\n",
            "| Scenario | Frozen P10 | Retrained P10 | Retrained P1 | BEST_MC P10 | Gain | Gap to BEST | Classification |\n",
            "|---|---:|---:|---:|---:|---:|---:|---|\n",
        ]
    )
    decisions = {}
    decision_values = {}
    for scenario in SCENARIOS:
        decision, values = classify_hattrick(summary, scenario)
        decisions[scenario] = decision
        decision_values[scenario] = values
        lines.append(
            f"| {scenario} | {values['frozen_p10']:.4f} | {values['retrained_p10']:.4f} | "
            f"{values['retrained_p1']:.4f} | {values['best_p10']:.4f} | {values['gain_p10']:+.4f} | "
            f"{values['best_gap_p10']:+.4f} | {decision} |\n"
        )
    shared_near_recovery = (
        decision_values["shared"]["gain_p10"] > 0.05
        and decision_values["shared"]["best_gap_p10"] <= 0.03
        and decision_values["shared"]["retrained_p1"] >= 0.90
        and decision_values["shared"]["retrained_p10"] >= 0.94
    )
    if (
        decisions["shared"] == "training-distribution problem" or shared_near_recovery
    ) and decisions["strict"] != "training-distribution problem":
        conclusion = "Shared recovers while strict does not; high load is learnable under the original path assumption, but load plus strong path isolation exposes an additional limitation."
    elif all(value == "training-distribution problem" for value in decisions.values()):
        conclusion = "Both scenarios recover; the frozen-model failure is primarily a training-distribution problem rather than a representational failure."
    elif all(value == "structure/loss limitation supported" for value in decisions.values()):
        conclusion = "Neither scenario recovers; the result supports a structure or loss-function limitation beyond simple load distribution shift."
    else:
        conclusion = "The scenarios give a mixed result; interpret the per-class tails and BEST_MC gaps separately."
    lines.extend(
        [
            "\n## Conclusion\n\n",
            conclusion + "\n\n",
            "This is a single-seed control. A borderline result requires additional seeds before a strong claim.\n",
        ]
    )
    return "".join(lines)


def verify_original_models() -> None:
    audit = json.loads(MODEL_AUDIT.read_text(encoding="utf-8"))
    before = audit["before"]
    after = original_model_metadata()
    audit["after"] = after
    audit["unchanged"] = before == after
    MODEL_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if before != after:
        raise RuntimeError("An original Hattrick model changed during retraining control")


def run_report() -> None:
    configure_modules()
    pm.validate_gurobi_outputs()
    validate_hattrick_outputs()
    dm.validate_test_outputs()
    retrained = build_retrained_rows()
    baseline = normalize_existing_rows(BASELINE_METRICS, "baseline_1x", 1.0)
    frozen = normalize_existing_rows(FROZEN_METRICS, "frozen_2x", LOAD_FACTOR)
    rows = baseline + frozen + retrained
    if len(baseline) != 3000 or len(frozen) != 3000 or len(retrained) != 3000 or len(rows) != 9000:
        raise RuntimeError(
            f"Unexpected row counts: baseline={len(baseline)}, frozen={len(frozen)}, retrained={len(retrained)}"
        )
    if any(not math.isfinite(float(row["norm_fulfill"])) for row in rows):
        raise RuntimeError("NaN or Inf found in NormFulFill")
    summary = summarize(rows)
    recovery = build_recovery(summary)
    write_csv(RESULTS / "load2x_retrain_metrics.csv", rows)
    write_csv(RESULTS / "load2x_retrain_summary_stats.csv", summary)
    write_csv(RESULTS / "load2x_recovery_vs_frozen.csv", recovery)
    draw_cdf(rows)
    draw_boxplot(rows)
    (RESULTS / "load2x_retrain_comparison.md").write_text(report_markdown(summary), encoding="utf-8")
    verify_original_models()
    state = load_hattrick_state()
    state["report"] = True
    save_hattrick_state(state)
    print(f"[ok] wrote load2x retraining control to {RESULTS}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "validate", "gurobi", "hattrick", "dotemc", "report"),
        default="all",
    )
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
