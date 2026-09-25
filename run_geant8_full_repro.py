from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

TOPO = "geant"
NUM_PATHS = 8
FINAL_CLUSTER = 0
CHUNK_CLUSTER_BASE = 800000
TOTAL_SNAPSHOTS = 10773
TRAIN_END = 6464
VAL_END = 8080
TEST_END = 10773
DEFAULT_CHUNK_SIZE = 1024
TARGET_EPOCHS = 60

RESULT_BASE = ROOT / "results" / TOPO / f"{NUM_PATHS}sp"
FINAL_DIR = RESULT_BASE / str(FINAL_CLUSTER)
LOG_DIR = RESULT_BASE / "_repro_logs"
STATE_DIR = RESULT_BASE / "_repro_state"
CHUNK_STATE_DIR = STATE_DIR / "chunks"
STATE_FILE = STATE_DIR / "state.json"

GUR_FILES = [
    "filenames.txt",
    "gt_optimal_values_mf.txt",
    "gt_optimal_values_mf_mf.txt",
    "gt_optimal_values_mf_mf_mf.txt",
    "gt_optimal_values_mlu.txt",
    "gt_optimal_values_mlu_mlu.txt",
    "gt_optimal_values_mlu_mlu_mlu.txt",
    "esm_optimal_values_mf.txt",
    "esm_optimal_values_mf_mf.txt",
    "esm_optimal_values_mf_mf_mf.txt",
    "flexile_runtime_1.txt",
    "flexile_runtime_2.txt",
    "flexile_runtime_3.txt",
    "swan_runtime_1.txt",
    "swan_runtime_2.txt",
    "swan_runtime_3.txt",
    "flexile_sim_results_esm_mf_mf_mf.txt",
    "swan_sim_results_esm_mf_mf_mf.txt",
]

OPTIMAL_FILES = [
    "gt_optimal_values_mf.txt",
    "gt_optimal_values_mf_mf.txt",
    "gt_optimal_values_mf_mf_mf.txt",
    "gt_optimal_values_mlu.txt",
    "gt_optimal_values_mlu_mlu.txt",
    "gt_optimal_values_mlu_mlu_mlu.txt",
    "esm_optimal_values_mf.txt",
    "esm_optimal_values_mf_mf.txt",
    "esm_optimal_values_mf_mf_mf.txt",
]

SIM_FILES = [
    "flexile_sim_results_esm_mf_mf_mf.txt",
    "swan_sim_results_esm_mf_mf_mf.txt",
]

RUNTIME_FILES = [
    "flexile_runtime_1.txt",
    "flexile_runtime_2.txt",
    "flexile_runtime_3.txt",
    "swan_runtime_1.txt",
    "swan_runtime_2.txt",
    "swan_runtime_3.txt",
]

EMPTY_PRED_ON_GT_FILES = [
    "esm_optimal_values_mf_on_gt.txt",
    "esm_optimal_values_mf_mf_on_gt.txt",
    "esm_optimal_values_mf_mf_mf_on_gt.txt",
    "swan_esm_1.txt",
    "swan_esm_2.txt",
    "swan_esm_3.txt",
]


def sanitize_output(text: str) -> str:
    text = re.sub(r"Set parameter LicenseID to value \S+", "Set parameter LicenseID to value <redacted>", text)
    text = re.sub(r"\r", "\n", text)
    return text


def line_count(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def ensure_dirs() -> None:
    for path in (RESULT_BASE, FINAL_DIR, LOG_DIR, STATE_DIR, CHUNK_STATE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "training": {
            "completed_epochs": 0,
            "target_epochs": TARGET_EPOCHS,
            "batch_size": 64,
            "checkpoint": 0,
            "fallback_reason": None,
        },
        "hattrick_test": {"sim_mf_mlu_1": False, "sim_mf_mlu_0": False},
        "report": False,
    }


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def env_for_run() -> dict[str, str]:
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env.setdefault("GUROBI_HOME", r"D:\kuroresearch\gurobi1302\win64")
    env.setdefault("GRB_LICENSE_FILE", r"D:\kuroresearch\gurobi_license\gurobi.lic")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_command(label: str, args: list[str], allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    log_path = LOG_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_label}.log"
    print(f"[run] {label}", flush=True)
    started = time.time()
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env_for_run(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - started
    output = sanitize_output((proc.stdout or "") + (proc.stderr or ""))
    log_path.write_text(output, encoding="utf-8")
    print(f"[done] {label} exit={proc.returncode} elapsed={elapsed:.1f}s log={log_path.name}", flush=True)
    if proc.returncode != 0 and not allow_failure:
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(f"Command failed: {label}\n{tail}")
    return proc


def chunks(chunk_size: int) -> list[tuple[int, int, int, int]]:
    result = []
    for chunk_id, start in enumerate(range(0, TOTAL_SNAPSHOTS, chunk_size)):
        end = min(start + chunk_size, TOTAL_SNAPSHOTS)
        cluster = CHUNK_CLUSTER_BASE + chunk_id
        result.append((chunk_id, cluster, start, end))
    return result


def expected_counts_for(length: int) -> dict[str, int]:
    counts = {name: length for name in OPTIMAL_FILES + RUNTIME_FILES + ["filenames.txt"]}
    for name in SIM_FILES:
        counts[name] = length * 6
    return counts


def chunk_dir(cluster: int) -> Path:
    return RESULT_BASE / str(cluster)


def chunk_marker(chunk_id: int) -> Path:
    return CHUNK_STATE_DIR / f"chunk_{chunk_id:03d}.done"


def chunk_complete(chunk_id: int, cluster: int, start: int, end: int) -> bool:
    if not chunk_marker(chunk_id).exists():
        return False
    counts = expected_counts_for(end - start)
    out_dir = chunk_dir(cluster)
    return all(line_count(out_dir / name) == expected for name, expected in counts.items())


def final_gurobi_complete() -> bool:
    counts = expected_counts_for(TOTAL_SNAPSHOTS)
    return all(line_count(FINAL_DIR / name) == expected for name, expected in counts.items())


def gurobi_args(start: int, end: int, cluster: int, pred: int, priority: int, objs: list[str], mode: str) -> list[str]:
    return [
        PYTHON,
        "frameworks/gurobi_refactored.py",
        "--num_paths_per_pair",
        str(NUM_PATHS),
        "--opt_start_idx",
        str(start),
        "--opt_end_idx",
        str(end),
        "--topo",
        TOPO,
        "--framework",
        "gurobi",
        "--pred",
        str(pred),
        "--pred_type",
        "esm",
        "--cluster",
        str(cluster),
        "--priority",
        str(priority),
        "--objs",
        *objs,
        "--gur_mode",
        mode,
        "--tol",
        "0.000001",
    ]


def run_gurobi_chunk(chunk_id: int, cluster: int, start: int, end: int) -> None:
    if chunk_complete(chunk_id, cluster, start, end):
        print(f"[skip] chunk {chunk_id} {start}:{end} already complete", flush=True)
        return

    print(f"[chunk] {chunk_id} cluster={cluster} range={start}:{end}", flush=True)
    length = end - start
    commands = [
        ("gt_mf_1", {"gt_optimal_values_mf.txt": length}, gurobi_args(start, end, cluster, 0, 1, ["mf"], "flexile")),
        ("gt_mf_2", {"gt_optimal_values_mf_mf.txt": length}, gurobi_args(start, end, cluster, 0, 2, ["mf", "mf"], "flexile")),
        ("gt_mf_3", {"gt_optimal_values_mf_mf_mf.txt": length}, gurobi_args(start, end, cluster, 0, 3, ["mf", "mf", "mf"], "flexile")),
        ("gt_mlu_1", {"gt_optimal_values_mlu.txt": length}, gurobi_args(start, end, cluster, 0, 1, ["mlu"], "flexile")),
        ("gt_mlu_2", {"gt_optimal_values_mlu_mlu.txt": length}, gurobi_args(start, end, cluster, 0, 2, ["mlu", "mlu"], "flexile")),
        ("gt_mlu_3", {"gt_optimal_values_mlu_mlu_mlu.txt": length, "filenames.txt": length}, gurobi_args(start, end, cluster, 0, 3, ["mlu", "mlu", "mlu"], "flexile")),
        ("flexile_mf_1", {"esm_optimal_values_mf.txt": length}, gurobi_args(start, end, cluster, 1, 1, ["mf"], "flexile")),
        ("flexile_mf_2", {"esm_optimal_values_mf_mf.txt": length}, gurobi_args(start, end, cluster, 1, 2, ["mf", "mf"], "flexile")),
        ("flexile_mf_3", {"esm_optimal_values_mf_mf_mf.txt": length, "flexile_sim_results_esm_mf_mf_mf.txt": length * 6}, gurobi_args(start, end, cluster, 1, 3, ["mf", "mf", "mf"], "flexile")),
        ("swan_mf_1", {"swan_runtime_1.txt": length}, gurobi_args(start, end, cluster, 1, 1, ["mf"], "swan")),
        ("swan_mf_2", {"swan_runtime_2.txt": length}, gurobi_args(start, end, cluster, 1, 2, ["mf", "mf"], "swan")),
        ("swan_mf_3", {"swan_runtime_3.txt": length, "swan_sim_results_esm_mf_mf_mf.txt": length * 6}, gurobi_args(start, end, cluster, 1, 3, ["mf", "mf", "mf"], "swan")),
    ]
    out_dir = chunk_dir(cluster)
    for suffix, outputs, cmd in commands:
        if all(line_count(out_dir / name) == expected for name, expected in outputs.items()):
            print(f"[skip] chunk {chunk_id} {suffix} outputs complete", flush=True)
            continue
        run_command(f"chunk{chunk_id:03d}_{suffix}", cmd)

    counts = expected_counts_for(end - start)
    out_dir = chunk_dir(cluster)
    missing = [f"{name}:{line_count(out_dir / name)}!={expected}" for name, expected in counts.items() if line_count(out_dir / name) != expected]
    if missing:
        raise RuntimeError(f"Chunk {chunk_id} failed line-count validation: {missing}")
    chunk_marker(chunk_id).write_text(json.dumps({"cluster": cluster, "start": start, "end": end}, indent=2), encoding="utf-8")
    print(f"[chunk] {chunk_id} complete", flush=True)


def run_gurobi_chunks(chunk_size: int, max_chunks: int | None = None) -> None:
    ensure_dirs()
    if final_gurobi_complete():
        print("[skip] final Gurobi results already complete", flush=True)
        return
    selected = chunks(chunk_size)
    if max_chunks is not None:
        selected = selected[:max_chunks]
    for chunk_id, cluster, start, end in selected:
        run_gurobi_chunk(chunk_id, cluster, start, end)


def append_file(src: Path, dest_handle) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    with src.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                dest_handle.write(line if line.endswith("\n") else line + "\n")


def merge_chunks(chunk_size: int) -> None:
    ensure_dirs()
    selected = chunks(chunk_size)
    incomplete = [(chunk_id, start, end) for chunk_id, cluster, start, end in selected if not chunk_complete(chunk_id, cluster, start, end)]
    if incomplete:
        raise RuntimeError(f"Cannot merge; incomplete chunks: {incomplete[:5]}")

    tmp_dir = RESULT_BASE / "_merge_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for name in GUR_FILES:
        with (tmp_dir / name).open("w", encoding="utf-8") as out:
            for _, cluster, _, _ in selected:
                append_file(chunk_dir(cluster) / name, out)

    for name in EMPTY_PRED_ON_GT_FILES:
        (tmp_dir / name).write_text("", encoding="utf-8")

    for name, expected in expected_counts_for(TOTAL_SNAPSHOTS).items():
        actual = line_count(tmp_dir / name)
        if actual != expected:
            raise RuntimeError(f"Merged {name} has {actual} lines, expected {expected}")

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for src in tmp_dir.iterdir():
        shutil.move(str(src), str(FINAL_DIR / src.name))
    shutil.rmtree(tmp_dir)
    print(f"[merge] wrote final Gurobi results to {FINAL_DIR}", flush=True)


def run_environment_checks() -> None:
    run_command("env_nvidia_smi", ["nvidia-smi"])
    run_command(
        "env_torch_cuda",
        [
            PYTHON,
            "-B",
            "-c",
            (
                "import torch, torch_scatter; "
                "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); "
                "print(torch.cuda.get_device_name(0)); "
                "src=torch.tensor([[1.,3.,2.,4.]], device='cuda'); "
                "idx=torch.tensor([0,0,1,1], device='cuda'); "
                "out,arg=torch_scatter.scatter_max(src, idx, dim=1, dim_size=2); "
                "torch.cuda.synchronize(); print('scatter ok', out.cpu().tolist(), arg.cpu().tolist())"
            ),
        ],
    )


def train_command(initial_training: int, batch_size: int, checkpoint: int) -> list[str]:
    return [
        PYTHON,
        "run_hattrick.py",
        "--topo",
        TOPO,
        "--mode",
        "train",
        "--epochs",
        "1",
        "--batch_size",
        str(batch_size),
        "--num_paths_per_pair",
        str(NUM_PATHS),
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
        str(FINAL_CLUSTER),
        "--train_start_indices",
        "0",
        "--train_end_indices",
        str(TRAIN_END),
        "--val_clusters",
        str(FINAL_CLUSTER),
        "--val_start_indices",
        str(TRAIN_END),
        "--val_end_indices",
        str(VAL_END),
        "--pred",
        "1",
        "--dynamic",
        "0",
        "--lr",
        "0.0005",
        "--pred_type",
        "esm",
        "--initial_training",
        str(initial_training),
        "--violation",
        "1",
        "--checkpoint",
        str(checkpoint),
    ]


def run_training(max_epochs: int | None = None) -> None:
    if not final_gurobi_complete():
        raise RuntimeError("Final Gurobi results are not complete; train stage needs oracle files.")
    state = load_state()
    train_state = state["training"]
    target = int(train_state.get("target_epochs", TARGET_EPOCHS))
    if max_epochs is not None:
        target = min(target, int(train_state.get("completed_epochs", 0)) + max_epochs)

    while int(train_state.get("completed_epochs", 0)) < target:
        completed = int(train_state.get("completed_epochs", 0))
        batch_size = int(train_state.get("batch_size", 64))
        checkpoint = int(train_state.get("checkpoint", 0))
        initial_training = 1 if completed == 0 else 0
        label = f"hattrick_train_epoch_{completed + 1:03d}_bs{batch_size}_ckpt{checkpoint}"
        proc = run_command(label, train_command(initial_training, batch_size, checkpoint), allow_failure=True)
        if proc.returncode == 0:
            train_state["completed_epochs"] = completed + 1
            save_state(state)
            continue

        output = sanitize_output((proc.stdout or "") + (proc.stderr or ""))
        if "out of memory" in output.lower() and batch_size == 64:
            print("[fallback] CUDA OOM at batch_size=64; restarting training with batch_size=32 and checkpoint=1", flush=True)
            train_state["completed_epochs"] = 0
            train_state["batch_size"] = 32
            train_state["checkpoint"] = 1
            train_state["fallback_reason"] = "CUDA OOM at batch_size=64"
            save_state(state)
            continue
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(f"Training failed at epoch {completed + 1}\n{tail}")


def test_command(sim_mf_mlu: int) -> list[str]:
    return [
        PYTHON,
        "run_hattrick.py",
        "--topo",
        TOPO,
        "--mode",
        "test",
        "--test_cluster",
        str(FINAL_CLUSTER),
        "--test_start_idx",
        str(VAL_END),
        "--test_end_idx",
        str(TEST_END),
        "--num_paths_per_pair",
        str(NUM_PATHS),
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
        "--pred",
        "1",
        "--dynamic",
        "0",
        "--pred_type",
        "esm",
        "--sim_mf_mlu",
        str(sim_mf_mlu),
        "--violation",
        "1",
        "--checkpoint",
        "0",
    ]


def run_hattrick_tests() -> None:
    state = load_state()
    test_state = state["hattrick_test"]
    for sim in (1, 0):
        key = f"sim_mf_mlu_{sim}"
        output_path = FINAL_DIR / f"hattrick_values_esm_sim_mlu_{sim}.txt"
        if test_state.get(key) and line_count(output_path) == (TEST_END - VAL_END) * 3:
            print(f"[skip] Hattrick test sim_mf_mlu={sim} already complete", flush=True)
            continue
        run_command(f"hattrick_test_sim_{sim}", test_command(sim))
        expected = (TEST_END - VAL_END) * 3
        actual = line_count(output_path)
        if actual != expected:
            raise RuntimeError(f"{output_path.name} has {actual} lines, expected {expected}")
        test_state[key] = True
        save_state(state)


def run_report() -> None:
    run_command("generate_geant8_full_report", [PYTHON, "generate_geant8_full_report.py"])
    state = load_state()
    state["report"] = True
    save_state(state)
    run_command("generate_geant8_full_report_final_state", [PYTHON, "generate_geant8_full_report.py"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "env", "gurobi", "merge", "train", "test", "report"], default="all")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    if args.stage in ("all", "env"):
        run_environment_checks()
    if args.stage in ("all", "gurobi"):
        run_gurobi_chunks(args.chunk_size, args.max_chunks)
    if args.stage in ("all", "merge"):
        merge_chunks(args.chunk_size)
    if args.stage in ("all", "train"):
        run_training(args.max_epochs)
    if args.stage in ("all", "test"):
        run_hattrick_tests()
    if args.stage in ("all", "report"):
        run_report()


if __name__ == "__main__":
    main()
