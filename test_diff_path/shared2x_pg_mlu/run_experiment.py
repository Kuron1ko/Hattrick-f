from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = ROOT.parent / ".venv-hattrick" / "Scripts" / "python.exe"
TOPOLOGY = "geant_priomask500_shared_load2x_train"
K = 8
BASE_MODEL = ROOT / f"hattrick_{TOPOLOGY}_{K}sp.pkl"
RESULT_DIR = ROOT / "results" / TOPOLOGY / f"{K}sp" / "0"
CLASSES = ("High", "Medium", "Low")

PHASES = {
    "small": {
        "train": (0, 128),
        "validation": (128, 160),
        "evaluation": (160, 250),
        "epochs": 12,
        "lr": 0.0002,
        "effective_batch_size": 32,
    },
    "full": {
        "train": (0, 350),
        "validation": (350, 400),
        "evaluation": (400, 500),
        "epochs": 30,
        "lr": 0.0002,
        "effective_batch_size": 32,
    },
}


def common_args() -> list[str]:
    return [
        "--topo", TOPOLOGY,
        "--num_paths_per_pair", str(K),
        "--num_transformer_layers", "3",
        "--num_gnn_layers", "3",
        "--num_mlp1_hidden_layers", "2",
        "--num_mlp2_hidden_layers", "2",
        "--rau1", "3",
        "--rau2", "3",
        "--rau3", "3",
        "--pred", "1",
        "--dynamic", "0",
        "--pred_type", "esm",
        "--violation", "1",
        "--path_mask", "0",
    ]


def run_logged(command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError("\n".join(tail))


def train(phase: str, label: str, multiplier: float) -> tuple[Path, float]:
    cfg = PHASES[phase]
    train_start, train_end = cfg["train"]
    val_start, val_end = cfg["validation"]
    tag = f"pgmlu_{phase}_{label}"
    model_path = ROOT / f"hattrick_{TOPOLOGY}_{K}sp_{tag}.pkl"
    env = os.environ.copy()
    env.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HATTRICK_PG_MLU_LAMBDA": str(multiplier),
            "HATTRICK_RUN_TAG": tag,
            "HATTRICK_INIT_MODEL": str(BASE_MODEL),
            "HATTRICK_FRESH_OPTIMIZER": "1",
            "HATTRICK_EFFECTIVE_BATCH_SIZE": str(cfg["effective_batch_size"]),
        }
    )
    command = [
        str(PYTHON), "run_hattrick.py",
        "--mode", "train",
        "--epochs", str(cfg["epochs"]),
        "--batch_size", "8",
        "--train_clusters", "0",
        "--train_start_indices", str(train_start),
        "--train_end_indices", str(train_end),
        "--val_clusters", "0",
        "--val_start_indices", str(val_start),
        "--val_end_indices", str(val_end),
        "--lr", str(cfg["lr"]),
        "--initial_training", "0",
        *common_args(),
    ]
    started = time.perf_counter()
    print(f"[train] {phase}/{label}, lambda={multiplier:g}", flush=True)
    run_logged(command, env, HERE / "logs" / f"{phase}_{label}_train.log")
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return model_path, time.perf_counter() - started


def evaluate(phase: str, label: str, model_path: Path) -> np.ndarray:
    cfg = PHASES[phase]
    eval_start, eval_end = cfg["evaluation"]
    env = os.environ.copy()
    env.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HATTRICK_PG_MLU_LAMBDA": "0",
        }
    )
    model_name = f"pgmlu_{phase}_{label}"
    command = [
        str(PYTHON), "run_hattrick.py",
        "--mode", "test",
        "--test_cluster", "0",
        "--test_start_idx", str(eval_start),
        "--test_end_idx", str(eval_end),
        "--sim_mf_mlu", "1",
        "--model", model_name,
        "--model_path_override", str(model_path),
        *common_args(),
    ]
    print(f"[test]  {phase}/{label}, strict ESM", flush=True)
    run_logged(command, env, HERE / "logs" / f"{phase}_{label}_test.log")
    values_path = RESULT_DIR / f"{model_name}_values_esm_sim_mlu_1.txt"
    values = np.loadtxt(values_path, dtype=np.float64).reshape(-1, 3)
    expected = eval_end - eval_start
    if values.shape != (expected, 3):
        raise RuntimeError(f"{values_path}: got {values.shape}, expected {(expected, 3)}")
    return values


def summarize(values: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    for index, class_name in enumerate(CLASSES):
        column = values[:, index]
        result[class_name] = {
            "mean": float(column.mean()),
            "p1": float(np.percentile(column, 1)),
            "p10": float(np.percentile(column, 10)),
            "median": float(np.median(column)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, default="small")
    parser.add_argument("--pg-lambda", type=float, default=0.5)
    args = parser.parse_args()
    phase = args.phase
    if args.pg_lambda <= 0:
        raise SystemExit("--pg-lambda must be positive")

    phase_dir = HERE / "artifacts" / phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    runs = (("control", 0.0), (f"pg_l{args.pg_lambda:g}".replace(".", "p"), args.pg_lambda))
    all_values = {}
    records = []
    for label, multiplier in runs:
        model_path, elapsed = train(phase, label, multiplier)
        values = evaluate(phase, label, model_path)
        all_values[label] = values
        np.savetxt(phase_dir / f"{label}_norm_fulfill.csv", values, delimiter=",", header=",".join(CLASSES), comments="")
        summary = summarize(values)
        for class_name in CLASSES:
            records.append(
                {
                    "phase": phase,
                    "method": label,
                    "pg_mlu_lambda": multiplier,
                    "class": class_name,
                    "n": len(values),
                    **summary[class_name],
                    "training_seconds": elapsed,
                    "model_path": str(model_path),
                }
            )

    control_label, candidate_label = runs[0][0], runs[1][0]
    control_summary = summarize(all_values[control_label])
    candidate_summary = summarize(all_values[candidate_label])
    delta = {
        class_name: {
            metric: candidate_summary[class_name][metric] - control_summary[class_name][metric]
            for metric in ("mean", "p1", "p10", "median")
        }
        for class_name in CLASSES
    }
    report = {
        "method": "PG-MLU: actual utilization + lambda * stopgrad(ReLU(actual - ESM))",
        "phase": phase,
        "protocol": PHASES[phase],
        "strict_esm_inference": True,
        "same_initial_model": str(BASE_MODEL),
        "candidate_lambda": args.pg_lambda,
        "summaries": {
            control_label: control_summary,
            candidate_label: candidate_summary,
        },
        "candidate_minus_control": delta,
    }
    (phase_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (phase_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
