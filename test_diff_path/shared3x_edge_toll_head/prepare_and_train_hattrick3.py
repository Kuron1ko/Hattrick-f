from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
PYTHON = Path(r"D:\kuroresearch\.venv-hattrick\Scripts\python.exe")

SOURCE_TOPOLOGY = "geant_priomask500_shared"
TARGET_TOPOLOGY = "geant_priomask500_shared_load3x_train"
LOAD_FACTOR = 3.0
K = 8
TRAIN_END = 350
VALIDATION_END = 400
FINAL_END = 500
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.002

STATE_PATH = THIS_DIR / "hattrick3_state.json"
AUDIT_PATH = THIS_DIR / "load3x_data_audit.json"
MODEL_PATH = ROOT / f"hattrick_{TARGET_TOPOLOGY}_{K}sp.pkl"
LOG_DIR = THIS_DIR / "logs"


def read_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"prepared": False, "oracles": {}, "epochs": {}, "tested": False}


def write_state(state: dict) -> None:
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def manifest_rows(topology: str) -> list[tuple[str, str, str]]:
    path = ROOT / "manifest" / f"{topology}_manifest.txt"
    rows: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            topo, pairs, tm = [part.strip() for part in line.split(",")]
            rows.append((topo, pairs, tm))
    return rows


def copy_path_artifacts() -> None:
    pairs = (
        (
            ROOT / "topologies" / "paths_dict" / f"{SOURCE_TOPOLOGY}_{K}_paths_dict_cluster_0.pkl",
            ROOT / "topologies" / "paths_dict" / f"{TARGET_TOPOLOGY}_{K}_paths_dict_cluster_0.pkl",
        ),
        (
            ROOT / "topologies" / "paths" / f"{SOURCE_TOPOLOGY}_{K}_paths_cluster_0.pkl",
            ROOT / "topologies" / "paths" / f"{TARGET_TOPOLOGY}_{K}_paths_cluster_0.pkl",
        ),
        (
            ROOT
            / "topologies"
            / "padded_edge_ids_per_path"
            / f"{SOURCE_TOPOLOGY}_{K}_paths_cluster_0_padded_edge_ids_per_path.pkl",
            ROOT
            / "topologies"
            / "padded_edge_ids_per_path"
            / f"{TARGET_TOPOLOGY}_{K}_paths_cluster_0_padded_edge_ids_per_path.pkl",
        ),
        (
            ROOT
            / "topologies"
            / "padded_edge_ids_per_path"
            / f"{SOURCE_TOPOLOGY}_{K}_paths_cluster_0_edge_ids_dict.pkl",
            ROOT
            / "topologies"
            / "padded_edge_ids_per_path"
            / f"{TARGET_TOPOLOGY}_{K}_paths_cluster_0_edge_ids_dict.pkl",
        ),
    )
    for source, target in pairs:
        if not source.exists():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def scaled_copy(source: Path, target: Path) -> tuple[float, float, float]:
    with source.open("rb") as handle:
        original = np.asarray(pickle.load(handle))
    scaled = original * LOAD_FACTOR
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(scaled, handle, protocol=pickle.HIGHEST_PROTOCOL)
    original_sum = float(original.sum())
    scaled_sum = float(scaled.sum())
    return original_sum, scaled_sum, scaled_sum / max(original_sum, 1e-12)


def prepare() -> None:
    state = read_state()
    if state["prepared"]:
        validate_data()
        return

    THIS_DIR.mkdir(parents=True, exist_ok=True)
    copy_path_artifacts()
    source_rows = manifest_rows(SOURCE_TOPOLOGY)[:FINAL_END]
    if len(source_rows) != FINAL_END:
        raise RuntimeError(f"Expected {FINAL_END} source rows, found {len(source_rows)}")

    seen_topologies: set[str] = set()
    seen_pairs: set[str] = set()
    seen_tms: set[str] = set()
    manifest_lines: list[str] = []
    ratios: list[float] = []
    sums = {"actual_original": 0.0, "actual_scaled": 0.0, "esm_original": 0.0, "esm_scaled": 0.0}

    for topology_file, pairs_file, tm_file in source_rows:
        manifest_lines.append(f"{topology_file},{pairs_file},{tm_file}\n")
        if topology_file not in seen_topologies:
            source = ROOT / "topologies" / SOURCE_TOPOLOGY / topology_file
            target = ROOT / "topologies" / TARGET_TOPOLOGY / topology_file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            seen_topologies.add(topology_file)
        if pairs_file not in seen_pairs:
            source = ROOT / "pairs" / SOURCE_TOPOLOGY / pairs_file
            target = ROOT / "pairs" / TARGET_TOPOLOGY / pairs_file
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            seen_pairs.add(pairs_file)
        if tm_file in seen_tms:
            continue
        for priority in (1, 2, 3):
            for suffix in ("", "_esm"):
                source = ROOT / "traffic_matrices" / f"{SOURCE_TOPOLOGY}_{priority}{suffix}" / tm_file
                target = ROOT / "traffic_matrices" / f"{TARGET_TOPOLOGY}_{priority}{suffix}" / tm_file
                original_sum, scaled_sum, ratio = scaled_copy(source, target)
                ratios.append(ratio)
                key = "esm" if suffix else "actual"
                sums[f"{key}_original"] += original_sum
                sums[f"{key}_scaled"] += scaled_sum
        seen_tms.add(tm_file)

    (ROOT / "manifest" / f"{TARGET_TOPOLOGY}_manifest.txt").write_text(
        "".join(manifest_lines), encoding="utf-8"
    )
    payload = {
        "source_topology": SOURCE_TOPOLOGY,
        "target_topology": TARGET_TOPOLOGY,
        "load_factor": LOAD_FACTOR,
        "snapshots": len(source_rows),
        "unique_traffic_matrices": len(seen_tms),
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "sums": sums,
        "split": {
            "hattrick_train": [0, TRAIN_END],
            "hattrick_validation": [TRAIN_END, VALIDATION_END],
            "level4": [VALIDATION_END, FINAL_END],
        },
        "hattrick3_training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "initialization": "from scratch with seed 490",
            "prediction_input": "strict current ESM",
        },
    }
    AUDIT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    state["prepared"] = True
    write_state(state)
    validate_data()


def validate_data() -> None:
    target_rows = manifest_rows(TARGET_TOPOLOGY)
    source_rows = manifest_rows(SOURCE_TOPOLOGY)[:FINAL_END]
    if len(target_rows) != FINAL_END or target_rows != source_rows:
        raise RuntimeError("3x manifest does not exactly match the 500-snapshot source manifest")
    for _, _, tm_file in target_rows:
        for priority in (1, 2, 3):
            for suffix in ("", "_esm"):
                source = ROOT / "traffic_matrices" / f"{SOURCE_TOPOLOGY}_{priority}{suffix}" / tm_file
                target = ROOT / "traffic_matrices" / f"{TARGET_TOPOLOGY}_{priority}{suffix}" / tm_file
                with source.open("rb") as handle:
                    original = np.asarray(pickle.load(handle), dtype=np.float64)
                with target.open("rb") as handle:
                    scaled = np.asarray(pickle.load(handle), dtype=np.float64)
                if not np.allclose(scaled, original * LOAD_FACTOR, rtol=1e-7, atol=1e-10):
                    raise RuntimeError(f"3x scaling mismatch: {target}")


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GUROBI_HOME", r"D:\kuroresearch\gurobi1302\win64")
    env.setdefault("GRB_LICENSE_FILE", r"D:\kuroresearch\gurobi_license\gurobi.lic")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return env


def run_logged(label: str, command: list[str]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{label}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        raise RuntimeError(f"{label} failed with exit code {process.returncode}\n{tail}")


def oracle_command(priority: int, objective: str) -> list[str]:
    return [
        str(PYTHON),
        "frameworks/gurobi_refactored.py",
        "--topo",
        TARGET_TOPOLOGY,
        "--num_paths_per_pair",
        str(K),
        "--opt_start_idx",
        "0",
        "--opt_end_idx",
        str(FINAL_END),
        "--cluster",
        "0",
        "--pred",
        "0",
        "--pred_type",
        "esm",
        "--gur_mode",
        "flexile",
        "--priority",
        str(priority),
        "--objs",
        *([objective] * priority),
        "--path_mask",
        "0",
        "--round",
        "1",
    ]


def build_oracles() -> None:
    state = read_state()
    for objective in ("mf", "mlu"):
        for priority in (1, 2, 3):
            label = f"gt_{'_'.join([objective] * priority)}"
            if state["oracles"].get(label):
                continue
            run_logged(label, oracle_command(priority, objective))
            state["oracles"][label] = True
            write_state(state)

    result_dir = ROOT / "results" / TARGET_TOPOLOGY / f"{K}sp" / "0"
    required = (
        "gt_optimal_values_mf.txt",
        "gt_optimal_values_mf_mf.txt",
        "gt_optimal_values_mf_mf_mf.txt",
        "gt_optimal_values_mlu.txt",
        "gt_optimal_values_mlu_mlu.txt",
        "gt_optimal_values_mlu_mlu_mlu.txt",
    )
    for name in required:
        path = result_dir / name
        count = len(np.loadtxt(path, dtype=np.float64).reshape(-1))
        if count != FINAL_END:
            raise RuntimeError(f"{path} contains {count} values, expected {FINAL_END}")


def train_command(initial_training: int) -> list[str]:
    return [
        str(PYTHON),
        "run_hattrick.py",
        "--topo",
        TARGET_TOPOLOGY,
        "--mode",
        "train",
        "--epochs",
        "1",
        "--batch_size",
        str(BATCH_SIZE),
        "--num_paths_per_pair",
        str(K),
        "--num_transformer_layers",
        "3",
        "--num_gnn_layers",
        "3",
        "--num_mlp1_hidden_layers",
        "2",
        "--num_mlp2_hidden_layers",
        "2",
        "--rau1",
        "3",
        "--rau2",
        "3",
        "--rau3",
        "3",
        "--train_clusters",
        "0",
        "--train_start_indices",
        "0",
        "--train_end_indices",
        str(TRAIN_END),
        "--val_clusters",
        "0",
        "--val_start_indices",
        str(TRAIN_END),
        "--val_end_indices",
        str(VALIDATION_END),
        "--pred",
        "1",
        "--dynamic",
        "0",
        "--lr",
        str(LEARNING_RATE),
        "--pred_type",
        "esm",
        "--initial_training",
        str(initial_training),
        "--violation",
        "1",
        "--path_mask",
        "0",
    ]


def train_hattrick3() -> None:
    state = read_state()
    for epoch in range(1, EPOCHS + 1):
        label = f"epoch_{epoch:03d}"
        if state["epochs"].get(label):
            continue
        run_logged(label, train_command(1 if epoch == 1 else 0))
        state["epochs"][label] = True
        write_state(state)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "prepare", "oracles", "train"), default="all")
    args = parser.parse_args()
    if args.stage in ("all", "prepare"):
        prepare()
    if args.stage in ("all", "oracles"):
        build_oracles()
    if args.stage in ("all", "train"):
        train_hattrick3()
    print(str(MODEL_PATH), flush=True)


if __name__ == "__main__":
    main()
