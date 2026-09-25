from __future__ import annotations

"""Full GEANT 2x experiment for six-loss Hattrick and Hattrick-f.

The fixed half-open snapshot ranges are:
    train      [0, 6000)
    validation [6000, 7500)
    test       [7500, 10200)

Hattrick is trained from scratch with the ordered six-objective projection
    Fh -> Uh -> Fhm -> Uhm -> Fhml -> Uhml.

Hattrick-f starts from the validation-selected Hattrick checkpoint, resets
Adam, keeps every model parameter trainable, and continues with
    Fh -> Fhm -> Fhml.

Only validation snapshots participate in checkpoint/seed selection.  Test
snapshots are first loaded after both selected checkpoints have been fixed.
"""

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import importlib.util
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent

SOURCE_TOPOLOGY = "geant"
TARGET_TOPOLOGY = "geant_full_shared_load2x_train"
NUM_PATHS = 8
LOAD_FACTOR = 2.0
TRAIN_RANGE = (0, 6000)
VALIDATION_RANGE = (6000, 7500)
TEST_RANGE = (7500, 10200)
TOTAL_SNAPSHOTS = TEST_RANGE[1]
CLASSES = ("High", "Medium", "Low")
ORACLE_FILES = (
    "gt_optimal_values_mf.txt",
    "gt_optimal_values_mf_mf.txt",
    "gt_optimal_values_mf_mf_mf.txt",
    "gt_optimal_values_mlu.txt",
    "gt_optimal_values_mlu_mlu.txt",
    "gt_optimal_values_mlu_mlu_mlu.txt",
)
ORACLE_CLUSTER_BASE = 820000

ARTIFACT_ROOT = THIS_DIR / "artifacts"
LOG_ROOT = ARTIFACT_ROOT / "logs"
ORACLE_STATE_ROOT = ARTIFACT_ROOT / "oracle_chunks"
PREPARE_MARKER = ARTIFACT_ROOT / "prepared_2x.json"
SELECTION_PATH = ARTIFACT_ROOT / "selected_models.json"
FINAL_RESULT_DIR = ROOT / "results" / TARGET_TOPOLOGY / f"{NUM_PATHS}sp" / "0"

SHARED_SOURCE = TEST_DIR / "shared2x_order_regularizer" / "run_experiment.py"
FULL_SOURCE = TEST_DIR / "shared2x_full_objectives" / "run_experiment.py"
HATTRICK_F_SOURCE = TEST_DIR / "shared2x_hattrick_f" / "run_experiment.py"

_RUNTIMES: tuple[object, object, object] | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def save_epoch_checkpoint(run_dir: Path, payload: dict) -> Path:
    """Atomically archive one complete training state without pruning it."""
    epoch = int(payload["epoch"])
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def archive_existing_checkpoint(run_dir: Path, source: Path) -> Path:
    """Backfill an existing best/final state into the per-epoch archive."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    epoch = int(checkpoint["epoch"])
    target = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
    if target.exists():
        archived = torch.load(target, map_location="cpu", weights_only=False)
        identity = ("method", "seed", "epoch")
        if any(archived.get(field) != checkpoint.get(field) for field in identity):
            raise RuntimeError(f"Checkpoint archive identity conflict: {target}")
        return target
    atomic_copy(source, target)
    return target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def manifest_rows(topology: str) -> list[tuple[str, str, str]]:
    path = ROOT / "manifest" / f"{topology}_manifest.txt"
    rows: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            fields = tuple(part.strip() for part in line.split(","))
            if len(fields) != 3:
                raise RuntimeError(f"Malformed manifest row in {path}: {line!r}")
            rows.append(fields)  # type: ignore[arg-type]
    return rows


def expected_prepare_config() -> dict:
    source_manifest = ROOT / "manifest" / f"{SOURCE_TOPOLOGY}_manifest.txt"
    return {
        "source_topology": SOURCE_TOPOLOGY,
        "target_topology": TARGET_TOPOLOGY,
        "source_manifest_sha256": sha256(source_manifest),
        "load_factor": LOAD_FACTOR,
        "snapshots": TOTAL_SNAPSHOTS,
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "test": list(TEST_RANGE),
        "scaled_inputs": ["actual", "esm_prediction"],
    }


def target_tm_path(priority: int, predicted: bool, filename: str) -> Path:
    suffix = "_esm" if predicted else ""
    return ROOT / "traffic_matrices" / f"{TARGET_TOPOLOGY}_{priority}{suffix}" / filename


def source_tm_path(priority: int, predicted: bool, filename: str) -> Path:
    suffix = "_esm" if predicted else ""
    return ROOT / "traffic_matrices" / f"{SOURCE_TOPOLOGY}_{priority}{suffix}" / filename


def scale_pickle(source: Path, target: Path) -> None:
    with source.open("rb") as handle:
        original = np.asarray(pickle.load(handle))
    scaled = original * LOAD_FACTOR
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(scaled, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(target)


def cache_file_pairs(source_cluster: int, target_cluster: int) -> list[tuple[Path, Path]]:
    source_prefix = f"{SOURCE_TOPOLOGY}_{NUM_PATHS}_paths_cluster_{source_cluster}"
    target_prefix = f"{TARGET_TOPOLOGY}_{NUM_PATHS}_paths_cluster_{target_cluster}"
    return [
        (
            ROOT / "topologies" / "paths" / f"{source_prefix}.pkl",
            ROOT / "topologies" / "paths" / f"{target_prefix}.pkl",
        ),
        (
            ROOT
            / "topologies"
            / "paths_dict"
            / f"{SOURCE_TOPOLOGY}_{NUM_PATHS}_paths_dict_cluster_{source_cluster}.pkl",
            ROOT
            / "topologies"
            / "paths_dict"
            / f"{TARGET_TOPOLOGY}_{NUM_PATHS}_paths_dict_cluster_{target_cluster}.pkl",
        ),
        (
            ROOT
            / "topologies"
            / "padded_edge_ids_per_path"
            / f"{source_prefix}_padded_edge_ids_per_path.pkl",
            ROOT
            / "topologies"
            / "padded_edge_ids_per_path"
            / f"{target_prefix}_padded_edge_ids_per_path.pkl",
        ),
        (
            ROOT
            / "topologies"
            / "padded_edge_ids_per_path"
            / f"{source_prefix}_edge_ids_dict.pkl",
            ROOT
            / "topologies"
            / "padded_edge_ids_per_path"
            / f"{target_prefix}_edge_ids_dict.pkl",
        ),
    ]


def ensure_cluster_path_cache(target_cluster: int) -> None:
    # All selected GEANT snapshots have the same topology and pair set.  Reusing
    # cluster-0's audited 8-KSP incidence is exact and avoids recomputing paths.
    pairs = cache_file_pairs(0, target_cluster)
    for source, target in pairs:
        if not source.exists():
            # Only the two basic caches are guaranteed in older checkouts.  The
            # padded caches are allowed to be generated by Cluster_Info later.
            if source.parent.name in {"paths", "paths_dict"}:
                raise FileNotFoundError(source)
            continue
        if not target.exists() or sha256(target) != sha256(source):
            atomic_copy(source, target)


def prepared_file_counts_ok() -> bool:
    try:
        rows = manifest_rows(TARGET_TOPOLOGY)
    except FileNotFoundError:
        return False
    if len(rows) != TOTAL_SNAPSHOTS:
        return False
    for priority in (1, 2, 3):
        for predicted in (False, True):
            directory = target_tm_path(priority, predicted, "unused").parent
            if not directory.exists():
                return False
            if sum(1 for path in directory.iterdir() if path.is_file()) < TOTAL_SNAPSHOTS:
                return False
    return True


def validate_prepared_data() -> None:
    target_rows = manifest_rows(TARGET_TOPOLOGY)
    source_rows = manifest_rows(SOURCE_TOPOLOGY)[:TOTAL_SNAPSHOTS]
    if target_rows != source_rows:
        raise RuntimeError("The prepared 2x manifest is not the first 10200 GEANT rows")
    if len(target_rows) != TOTAL_SNAPSHOTS:
        raise RuntimeError(
            f"Prepared manifest has {len(target_rows)} rows; expected {TOTAL_SNAPSHOTS}"
        )

    boundary_indices = (0, 1, 5999, 6000, 7499, 7500, 10199)
    for index in boundary_indices:
        filename = target_rows[index][2]
        for priority in (1, 2, 3):
            for predicted in (False, True):
                with source_tm_path(priority, predicted, filename).open("rb") as handle:
                    source = np.asarray(pickle.load(handle), dtype=np.float64)
                with target_tm_path(priority, predicted, filename).open("rb") as handle:
                    target = np.asarray(pickle.load(handle), dtype=np.float64)
                if not np.allclose(
                    target, source * LOAD_FACTOR, rtol=1e-7, atol=1e-10
                ):
                    kind = "esm" if predicted else "actual"
                    raise RuntimeError(
                        f"2x audit failed at snapshot={index}, class={priority}, kind={kind}"
                    )

    source_topology = ROOT / "topologies" / SOURCE_TOPOLOGY / "t1.json"
    target_topology = ROOT / "topologies" / TARGET_TOPOLOGY / "t1.json"
    source_pairs = ROOT / "pairs" / SOURCE_TOPOLOGY / "t1.pkl"
    target_pairs = ROOT / "pairs" / TARGET_TOPOLOGY / "t1.pkl"
    if sha256(source_topology) != sha256(target_topology):
        raise RuntimeError("Topology changed while preparing the 2x dataset")
    if sha256(source_pairs) != sha256(target_pairs):
        raise RuntimeError("OD pairs changed while preparing the 2x dataset")
    print("[ok] prepared data: exact 2x actual+ESM on all split boundaries", flush=True)


def prepare_data(force: bool = False) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    expected = expected_prepare_config()
    if PREPARE_MARKER.exists() and not force:
        recorded = read_json(PREPARE_MARKER)
        if recorded == expected and prepared_file_counts_ok():
            validate_prepared_data()
            print("[skip] the complete 2x dataset is already prepared", flush=True)
            return

    rows = manifest_rows(SOURCE_TOPOLOGY)
    if len(rows) < TOTAL_SNAPSHOTS:
        raise RuntimeError(
            f"Source GEANT has only {len(rows)} snapshots; {TOTAL_SNAPSHOTS} required"
        )
    rows = rows[:TOTAL_SNAPSHOTS]

    atomic_copy(
        ROOT / "topologies" / SOURCE_TOPOLOGY / "t1.json",
        ROOT / "topologies" / TARGET_TOPOLOGY / "t1.json",
    )
    atomic_copy(
        ROOT / "pairs" / SOURCE_TOPOLOGY / "t1.pkl",
        ROOT / "pairs" / TARGET_TOPOLOGY / "t1.pkl",
    )
    ensure_cluster_path_cache(0)

    manifest_text = "".join(
        f"{topology},{pairs},{tm}\n" for topology, pairs, tm in rows
    )
    target_manifest = ROOT / "manifest" / f"{TARGET_TOPOLOGY}_manifest.txt"
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = target_manifest.with_suffix(f".txt.{os.getpid()}.tmp")
    temporary_manifest.write_text(manifest_text, encoding="utf-8")
    temporary_manifest.replace(target_manifest)

    started = time.perf_counter()
    for index, (_topology, _pairs, filename) in enumerate(rows):
        for priority in (1, 2, 3):
            for predicted in (False, True):
                scale_pickle(
                    source_tm_path(priority, predicted, filename),
                    target_tm_path(priority, predicted, filename),
                )
        if (index + 1) % 250 == 0 or index + 1 == len(rows):
            print(
                f"[prepare] {index + 1}/{len(rows)} snapshots "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )

    validate_prepared_data()
    write_json(PREPARE_MARKER, expected)
    print(f"[done] prepared topology {TARGET_TOPOLOGY}", flush=True)


def oracle_chunks(chunk_size: int) -> list[tuple[int, int, int, int]]:
    chunks: list[tuple[int, int, int, int]] = []
    for chunk_id, start in enumerate(range(0, TOTAL_SNAPSHOTS, chunk_size)):
        end = min(start + chunk_size, TOTAL_SNAPSHOTS)
        chunks.append((chunk_id, ORACLE_CLUSTER_BASE + chunk_id, start, end))
    return chunks


def oracle_chunk_marker(chunk_id: int) -> Path:
    return ORACLE_STATE_ROOT / f"chunk_{chunk_id:04d}.json"


def oracle_chunk_complete(
    chunk_id: int, cluster: int, start: int, end: int
) -> bool:
    marker = oracle_chunk_marker(chunk_id)
    if not marker.exists():
        return False
    try:
        metadata = read_json(marker)
    except (OSError, json.JSONDecodeError):
        return False
    expected_meta = {"cluster": cluster, "start": start, "end": end}
    if metadata != expected_meta:
        return False
    output = ROOT / "results" / TARGET_TOPOLOGY / f"{NUM_PATHS}sp" / str(cluster)
    length = end - start
    return all(line_count(output / name) == length for name in ORACLE_FILES) and (
        line_count(output / "filenames.txt") == length
    )


def gurobi_command(
    start: int, end: int, cluster: int, priority: int, objective: str
) -> list[str]:
    return [
        sys.executable,
        "frameworks/gurobi_refactored.py",
        "--topo",
        TARGET_TOPOLOGY,
        "--framework",
        "gurobi",
        "--num_paths_per_pair",
        str(NUM_PATHS),
        "--opt_start_idx",
        str(start),
        "--opt_end_idx",
        str(end),
        "--cluster",
        str(cluster),
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
        "--tol",
        "0.000001",
    ]


def run_logged_command(label: str, command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = process.stdout or ""
    log_path.write_text(output, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - started
    print(f"[oracle] {label} exit={process.returncode} time={elapsed:.1f}s", flush=True)
    if process.returncode != 0:
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(f"Oracle command failed: {label}\n{tail}")


def run_oracle_chunk(chunk: tuple[int, int, int, int]) -> None:
    chunk_id, cluster, start, end = chunk
    if oracle_chunk_complete(chunk_id, cluster, start, end):
        print(f"[skip] oracle chunk {chunk_id} [{start},{end})", flush=True)
        return
    ensure_cluster_path_cache(cluster)
    output = ROOT / "results" / TARGET_TOPOLOGY / f"{NUM_PATHS}sp" / str(cluster)
    output.mkdir(parents=True, exist_ok=True)

    # MF first because the legacy solver writes placeholder MLU files after an
    # MF pass.  The following true MLU passes intentionally overwrite them.
    for objective in ("mf", "mlu"):
        for priority in (1, 2, 3):
            label = f"chunk{chunk_id:04d}_{objective}{priority}_{start}_{end}"
            run_logged_command(
                label,
                gurobi_command(start, end, cluster, priority, objective),
                LOG_ROOT / "oracle" / f"{label}.log",
            )

    length = end - start
    missing = [
        f"{name}:{line_count(output / name)}"
        for name in (*ORACLE_FILES, "filenames.txt")
        if line_count(output / name) != length
    ]
    if missing:
        raise RuntimeError(f"Oracle chunk {chunk_id} failed validation: {missing}")
    write_json(
        oracle_chunk_marker(chunk_id),
        {"cluster": cluster, "start": start, "end": end},
    )
    print(f"[done] oracle chunk {chunk_id} [{start},{end})", flush=True)


def final_oracle_complete() -> bool:
    return all(
        line_count(FINAL_RESULT_DIR / name) == TOTAL_SNAPSHOTS
        for name in (*ORACLE_FILES, "filenames.txt")
    )


def append_nonempty(source: Path, target_handle) -> None:
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                target_handle.write(line if line.endswith("\n") else line + "\n")


def merge_oracle_chunks(chunks: list[tuple[int, int, int, int]]) -> None:
    incomplete = [
        (chunk_id, start, end)
        for chunk_id, cluster, start, end in chunks
        if not oracle_chunk_complete(chunk_id, cluster, start, end)
    ]
    if incomplete:
        raise RuntimeError(f"Cannot merge incomplete oracle chunks: {incomplete[:8]}")

    result_base = ROOT / "results" / TARGET_TOPOLOGY / f"{NUM_PATHS}sp"
    temporary = result_base / f"_oracle_merge_{os.getpid()}"
    if temporary.exists():
        resolved = temporary.resolve()
        if result_base.resolve() not in resolved.parents:
            raise RuntimeError(f"Unsafe temporary merge path: {resolved}")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)

    for filename in (*ORACLE_FILES, "filenames.txt"):
        with (temporary / filename).open("w", encoding="utf-8") as output:
            for _chunk_id, cluster, _start, _end in chunks:
                source = result_base / str(cluster) / filename
                append_nonempty(source, output)
        if line_count(temporary / filename) != TOTAL_SNAPSHOTS:
            raise RuntimeError(f"Merged oracle file has wrong length: {filename}")

    FINAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in (*ORACLE_FILES, "filenames.txt"):
        (temporary / filename).replace(FINAL_RESULT_DIR / filename)
    temporary.rmdir()
    print(f"[done] merged 2x oracle into {FINAL_RESULT_DIR}", flush=True)


def generate_oracle(chunk_size: int, workers: int) -> None:
    if not PREPARE_MARKER.exists():
        raise RuntimeError("Run --stage prepare before --stage oracle")
    validate_prepared_data()
    if final_oracle_complete():
        print("[skip] complete 2x oracle is already merged", flush=True)
        return
    chunks = oracle_chunks(chunk_size)
    ORACLE_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        for chunk in chunks:
            run_oracle_chunk(chunk)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_oracle_chunk, chunk) for chunk in chunks]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    merge_oracle_chunks(chunks)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtimes() -> tuple[object, object, object]:
    global _RUNTIMES
    if _RUNTIMES is not None:
        return _RUNTIMES
    for directory in (ROOT, TEST_DIR, SHARED_SOURCE.parent, FULL_SOURCE.parent):
        value = str(directory.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)

    # Both audited training modules import the shared runtime under this legacy
    # module name.  Bind it explicitly so path ordering cannot select another
    # run_experiment.py from the research tree.
    shared = load_module("run_experiment", SHARED_SOURCE)
    full = load_module("full_geant_six_loss_runtime", FULL_SOURCE)
    hattrick_f = load_module("full_geant_hattrick_f_runtime", HATTRICK_F_SOURCE)
    _RUNTIMES = (shared, full, hattrick_f)
    return _RUNTIMES


def build_props(
    device: torch.device, *, batch_size: int, epochs: int, learning_rate: float
):
    from utils.args_parser import parse_args

    props = parse_args(
        [
            "--topo",
            TARGET_TOPOLOGY,
            "--framework",
            "hattrick",
            "--mode",
            "train",
            "--epochs",
            str(epochs),
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
            "0",
            "--train_start_indices",
            str(TRAIN_RANGE[0]),
            "--train_end_indices",
            str(TRAIN_RANGE[1]),
            "--val_clusters",
            "0",
            "--val_start_indices",
            str(VALIDATION_RANGE[0]),
            "--val_end_indices",
            str(VALIDATION_RANGE[1]),
            "--pred",
            "1",
            "--dynamic",
            "0",
            "--lr",
            str(learning_rate),
            "--pred_type",
            "esm",
            "--initial_training",
            "1",
            "--violation",
            "1",
            "--path_mask",
            "0",
        ]
    )
    props.device = device
    props.dtype = torch.float32
    props.research_return_admitted = False
    props.research_return_policy = False
    return props


def source_hashes() -> dict[str, str]:
    paths = {
        "run_full_experiment.py": Path(__file__).resolve(),
        "shared_six_loss.py": FULL_SOURCE,
        "hattrick_f.py": HATTRICK_F_SOURCE,
        "ordered_projection.py": FULL_SOURCE.parent / "ordered_projection.py",
        "hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "training_utils.py": ROOT / "utils" / "training_utils.py",
        "dataset.py": ROOT / "utils" / "build_dataset_within_cluster.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def experiment_signature(
    method: str,
    seed: int,
    batch_size: int,
    learning_rate: float,
    low_budget: float,
) -> dict:
    return {
        "method": method,
        "seed": seed,
        "topology": TARGET_TOPOLOGY,
        "load_factor": LOAD_FACTOR,
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "test": list(TEST_RANGE),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "low_budget": low_budget,
        "strict_esm": True,
        "source_sha256": source_hashes(),
    }


# Checkpoints produced before the manual early-stop/finalization commands were
# added contain this runner hash. Those commands changed orchestration only;
# the model, data, optimizer, and objective implementations were unchanged.
LEGACY_RESUME_COMPATIBLE_RUNNER_HASHES = {
    "1d0525e2572a7b4811417498d289b291025b0b6d1e967db488ebab69d2f2ead9",
    # Runner used for epochs 28--40; the later edit only added read-only
    # multi-epoch diagnostic testing.
    "f25973deb1070c481fec1522d7cfe115a3d387acffdd3ec1218552b29f1684b8",
    # Runner that added the Hattrick-only continuation stage. The subsequent
    # edit changes checkpoint retention only, not training computation.
    "3e66e5b6415a577c81e5ea4e2f1e6edc3877301c2e29c9991c5d815f8e774d78",
}


def resume_signature_compatible(saved: dict, requested: dict) -> bool:
    """Require an identical training signature, allowing one audited runner edit."""
    if saved == requested:
        return True

    saved_copy = copy.deepcopy(saved)
    requested_copy = copy.deepcopy(requested)
    saved_sources = saved_copy.get("source_sha256")
    requested_sources = requested_copy.get("source_sha256")
    if not isinstance(saved_sources, dict) or not isinstance(requested_sources, dict):
        return False

    saved_runner = saved_sources.pop("run_full_experiment.py", None)
    requested_sources.pop("run_full_experiment.py", None)
    return (
        saved_runner in LEGACY_RESUME_COMPATIBLE_RUNNER_HASHES
        and saved_copy == requested_copy
    )


def make_datasets(props, *, include_test: bool = False):
    from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster

    train = DM_Dataset_within_Cluster(props, 0, *TRAIN_RANGE)
    validation = DM_Dataset_within_Cluster(props, 0, *VALIDATION_RANGE)
    if int(train.max_source_index_read) != TRAIN_RANGE[1] - 1:
        raise RuntimeError("Train split audit failed")
    if int(validation.max_source_index_read) != VALIDATION_RANGE[1] - 1:
        raise RuntimeError("Validation split audit failed")
    test = None
    if include_test:
        test = DM_Dataset_within_Cluster(props, 0, *TEST_RANGE)
        if int(test.max_source_index_read) != TEST_RANGE[1] - 1:
            raise RuntimeError("Test split audit failed")
    return train, validation, test


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {str(row["class"]): row for row in summary}


def high_min(rows: list[dict]) -> float:
    return min(
        float(row["norm_fulfill"]) for row in rows if row["class"] == "High"
    )


def simulator_safe(summary: list[dict]) -> bool:
    return (
        max(float(row["max_admitted_capacity_ratio"]) for row in summary) <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
    )


def hattrick_rank(rows: list[dict], summary: list[dict]) -> tuple:
    indexed = summary_index(summary)
    minimum = high_min(rows)
    eligible = minimum >= 0.995 - 1e-4 and simulator_safe(summary)
    if eligible:
        return (
            1,
            float(indexed["Medium"]["norm_fulfill_mean"]),
            float(indexed["Medium"]["norm_fulfill_p10"]),
            float(indexed["Medium"]["norm_fulfill_p1"]),
            float(indexed["High"]["norm_fulfill_mean"]),
            float(indexed["Low"]["norm_fulfill_mean"]),
        )
    return (
        0,
        minimum,
        float(indexed["High"]["norm_fulfill_mean"]),
        float(indexed["High"]["norm_fulfill_p1"]),
        float(indexed["Medium"]["norm_fulfill_mean"]),
        float(indexed["Low"]["norm_fulfill_mean"]),
    )


def hattrick_f_rank(
    rows: list[dict],
    summary: list[dict],
    baseline_summary: list[dict],
    low_budget: float,
) -> tuple:
    indexed = summary_index(summary)
    baseline = summary_index(baseline_summary)
    minimum = high_min(rows)
    eligible = (
        minimum >= 0.995 - 1e-4
        and float(indexed["Low"]["norm_fulfill_mean"])
        >= float(baseline["Low"]["norm_fulfill_mean"]) - low_budget
        and simulator_safe(summary)
    )
    if eligible:
        return (
            1,
            float(indexed["Medium"]["norm_fulfill_mean"]),
            float(indexed["Medium"]["norm_fulfill_p10"]),
            float(indexed["Medium"]["norm_fulfill_p1"]),
            float(indexed["High"]["norm_fulfill_mean"]),
            float(indexed["Low"]["norm_fulfill_mean"]),
        )
    return (
        0,
        minimum,
        float(indexed["Low"]["norm_fulfill_mean"])
        - float(baseline["Low"]["norm_fulfill_mean"]),
        float(indexed["High"]["norm_fulfill_mean"]),
        float(indexed["Medium"]["norm_fulfill_mean"]),
        float(indexed["Medium"]["norm_fulfill_p10"]),
    )


def history_row(
    epoch: int,
    rank: tuple,
    rows: list[dict],
    summary: list[dict],
    diagnostics: dict,
    train_metrics: dict,
) -> dict:
    indexed = summary_index(summary)
    return {
        "epoch": epoch,
        "eligible": int(rank[0]),
        "high_norm_mean": indexed["High"]["norm_fulfill_mean"],
        "high_norm_p10": indexed["High"]["norm_fulfill_p10"],
        "high_norm_p1": indexed["High"]["norm_fulfill_p1"],
        "high_norm_min": high_min(rows),
        "medium_norm_mean": indexed["Medium"]["norm_fulfill_mean"],
        "medium_norm_p10": indexed["Medium"]["norm_fulfill_p10"],
        "medium_norm_p1": indexed["Medium"]["norm_fulfill_p1"],
        "low_norm_mean": indexed["Low"]["norm_fulfill_mean"],
        **diagnostics,
        **train_metrics,
    }


def save_validation(
    run_dir: Path,
    epoch: int,
    rows: list[dict],
    summary: list[dict],
    diagnostics: dict,
) -> None:
    write_csv(run_dir / f"validation_epoch_{epoch:03d}_metrics.csv", rows)
    write_json(
        run_dir / f"validation_epoch_{epoch:03d}_summary.json",
        {"classes": summary, "diagnostics": diagnostics},
    )


def train_hattrick(
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    low_budget: float,
) -> Path:
    shared, full, _hattrick_f = load_runtimes()
    from frameworks.hattrick_system import Hattrick
    from utils.AdamOptimizer import ADAMOptimizer

    run_dir = ARTIFACT_ROOT / "hattrick" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = experiment_signature(
        "Hattrick", seed, batch_size, learning_rate, low_budget
    )
    config = {
        **signature,
        "epochs": epochs,
        "ordered_objectives": ["Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml"],
        "selection": (
            "validation only: require every High NormFulFill >=0.995 (1e-4 "
            "numerical tolerance) and simulator safety; then maximize Medium "
            "mean/P10/P1, High mean, Low mean; High-first fallback if infeasible"
        ),
    }
    write_json(run_dir / "config.json", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    props = build_props(
        device, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate
    )
    train_dataset, validation_dataset, _ = make_datasets(props)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)

    final_path = run_dir / "final_model.pt"
    best_path = run_dir / "best_model.pt"
    history_path = run_dir / "train_history.csv"
    history = read_csv(history_path)
    start_epoch = 1
    best_rank: tuple | None = None
    best_epoch: int | None = None

    if final_path.exists():
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        if not resume_signature_compatible(checkpoint.get("signature", {}), signature):
            raise RuntimeError(
                f"Existing Hattrick run differs from requested config: {run_dir}"
            )
        if checkpoint.get("signature") != signature:
            print(
                "[resume] accepted audited orchestration-only runner change",
                flush=True,
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        final_archive = archive_existing_checkpoint(run_dir, final_path)
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            best_rank = tuple(best["rank"])
            best_epoch = int(best["epoch"])
            archive_existing_checkpoint(run_dir, best_path)
        print(
            f"[resume] Hattrick seed={seed} at epoch {start_epoch}; "
            f"retained {final_archive.name}",
            flush=True,
        )

    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        set_seed(seed + epoch * 104729)
        loader = shared.data_loader(
            train_dataset, props.batch_size, True, seed + epoch * 1009
        )
        train_metrics = full.train_epoch(
            model, props, train_dataset, loader, optimizer
        )
        rows, summary, diagnostics = shared.evaluate(
            model, props, validation_dataset, VALIDATION_RANGE[0]
        )
        rank = hattrick_rank(rows, summary)
        save_validation(run_dir, epoch, rows, summary, diagnostics)
        row = history_row(epoch, rank, rows, summary, diagnostics, train_metrics)
        history = [item for item in history if int(item["epoch"]) < epoch]
        history.append(row)
        write_csv(history_path, history)
        payload = {
            "method": "Hattrick",
            "seed": seed,
            "epoch": epoch,
            "rank": rank,
            "signature": signature,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_summary": summary,
            "validation_diagnostics": diagnostics,
        }
        epoch_path = save_epoch_checkpoint(run_dir, payload)
        atomic_copy(epoch_path, final_path)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            atomic_copy(epoch_path, best_path)
        print(
            f"[Hattrick] seed={seed} epoch={epoch}/{epochs} eligible={rank[0]} "
            f"H={row['high_norm_mean']:.6f} Hmin={row['high_norm_min']:.6f} "
            f"M={row['medium_norm_mean']:.6f} L={row['low_norm_mean']:.6f} "
            f"best={best_epoch} saved={epoch_path.name}",
            flush=True,
        )

    if not best_path.exists():
        raise RuntimeError(f"No Hattrick checkpoint was produced in {run_dir}")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    write_json(
        run_dir / "complete.json",
        {
            "status": "COMPLETE",
            "seed": seed,
            "best_epoch": int(best["epoch"]),
            "best_rank": list(best["rank"]),
            "eligible": bool(best["rank"][0]),
            "runtime_seconds_this_invocation": time.perf_counter() - started,
            "best_checkpoint_sha256": sha256(best_path),
            "checkpoint_archive_policy": "every_epoch",
            "saved_epoch_checkpoints": sorted(
                int(path.stem.removeprefix("epoch_"))
                for path in (run_dir / "checkpoints").glob("epoch_*.pt")
            ),
        },
    )
    return best_path


def train_hattrick_f(
    *,
    seed: int,
    phase_a_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    low_budget: float,
) -> Path:
    shared, _full, hattrick_f = load_runtimes()
    from frameworks.hattrick_system import Hattrick
    from utils.AdamOptimizer import ADAMOptimizer

    run_dir = ARTIFACT_ROOT / "hattrick_f" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = experiment_signature(
        "Hattrick-f", seed, batch_size, learning_rate, low_budget
    )
    signature["phase_a_sha256"] = sha256(phase_a_path)
    config = {
        **signature,
        "epochs": epochs,
        "phase_a_checkpoint": str(phase_a_path.resolve()),
        "optimizer_reset": True,
        "all_parameters_trainable": True,
        "ordered_objectives": ["Fh", "Fhm", "Fhml"],
        "persistent_mlu_minimization": False,
        "selection": (
            "validation only: every High NormFulFill >=0.995 (1e-4 tolerance), "
            f"Low mean >= selected Hattrick Low mean - {low_budget}, simulator "
            "safe; then maximize Medium mean/P10/P1, High mean, Low mean"
        ),
    }
    write_json(run_dir / "config.json", config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    props = build_props(
        device, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate
    )
    train_dataset, validation_dataset, _ = make_datasets(props)
    phase_a = torch.load(phase_a_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(phase_a["model_state_dict"])
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    # Required by Hattrick-f: do not retain moments trained for removed U losses.
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)

    baseline_rows, baseline_summary, baseline_diagnostics = shared.evaluate(
        model, props, validation_dataset, VALIDATION_RANGE[0]
    )
    write_csv(run_dir / "phase_a_validation_metrics.csv", baseline_rows)
    write_json(
        run_dir / "phase_a_validation_summary.json",
        {"classes": baseline_summary, "diagnostics": baseline_diagnostics},
    )

    final_path = run_dir / "final_model.pt"
    best_path = run_dir / "best_model.pt"
    history_path = run_dir / "train_history.csv"
    history = read_csv(history_path)
    initial_rank = hattrick_f_rank(
        baseline_rows, baseline_summary, baseline_summary, low_budget
    )
    initial_payload = {
        "method": "Hattrick-f",
        "seed": seed,
        "epoch": 0,
        "rank": initial_rank,
        "signature": signature,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_summary": baseline_summary,
        "validation_diagnostics": baseline_diagnostics,
    }
    if not best_path.exists():
        torch.save(initial_payload, best_path)

    start_epoch = 1
    best_rank = initial_rank
    best_epoch = 0
    if final_path.exists():
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        if not resume_signature_compatible(checkpoint.get("signature", {}), signature):
            raise RuntimeError(
                f"Existing Hattrick-f run differs from requested config: {run_dir}"
            )
        if checkpoint.get("signature") != signature:
            print(
                "[resume] accepted audited orchestration-only runner change",
                flush=True,
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        best_rank = tuple(best["rank"])
        best_epoch = int(best["epoch"])
        print(f"[resume] Hattrick-f seed={seed} at epoch {start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        set_seed(seed + epoch * 104729)
        loader = shared.data_loader(
            train_dataset, props.batch_size, True, seed + epoch * 1009
        )
        train_metrics = hattrick_f.train_epoch(
            model,
            teacher,
            props,
            train_dataset,
            loader,
            optimizer,
            hattrick_f.HIGH_TRAIN_FLOOR,
            None,
            True,
        )
        rows, summary, diagnostics = shared.evaluate(
            model, props, validation_dataset, VALIDATION_RANGE[0]
        )
        rank = hattrick_f_rank(rows, summary, baseline_summary, low_budget)
        save_validation(run_dir, epoch, rows, summary, diagnostics)
        row = history_row(epoch, rank, rows, summary, diagnostics, train_metrics)
        history = [item for item in history if int(item["epoch"]) < epoch]
        history.append(row)
        write_csv(history_path, history)
        payload = {
            "method": "Hattrick-f",
            "seed": seed,
            "epoch": epoch,
            "rank": rank,
            "signature": signature,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_summary": summary,
            "validation_diagnostics": diagnostics,
        }
        torch.save(payload, final_path)
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            torch.save(payload, best_path)
        print(
            f"[Hattrick-f] seed={seed} epoch={epoch}/{epochs} eligible={rank[0]} "
            f"H={row['high_norm_mean']:.6f} Hmin={row['high_norm_min']:.6f} "
            f"M={row['medium_norm_mean']:.6f} L={row['low_norm_mean']:.6f} "
            f"best={best_epoch}",
            flush=True,
        )

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    write_json(
        run_dir / "complete.json",
        {
            "status": "COMPLETE",
            "seed": seed,
            "best_epoch": int(best["epoch"]),
            "best_rank": list(best["rank"]),
            "eligible": bool(best["rank"][0]),
            "phase_a_checkpoint_sha256": sha256(phase_a_path),
            "runtime_seconds_this_invocation": time.perf_counter() - started,
            "best_checkpoint_sha256": sha256(best_path),
        },
    )
    return best_path


def train_models(
    *,
    seeds: list[int],
    hattrick_epochs: int,
    hattrick_f_epochs: int,
    batch_size: int,
    learning_rate: float,
    low_budget: float,
) -> None:
    if not final_oracle_complete():
        raise RuntimeError("Run --stage oracle before --stage train")
    for seed in seeds:
        phase_a = train_hattrick(
            seed=seed,
            epochs=hattrick_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            low_budget=low_budget,
        )
        train_hattrick_f(
            seed=seed,
            phase_a_path=phase_a,
            epochs=hattrick_f_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            low_budget=low_budget,
        )
    select_models(seeds)


def train_hattrick_only(
    *,
    seeds: list[int],
    hattrick_epochs: int,
    batch_size: int,
    learning_rate: float,
    low_budget: float,
) -> None:
    """Resume only the original Hattrick runs without touching Hattrick-f."""
    if not final_oracle_complete():
        raise RuntimeError("Run --stage oracle before --stage hattrick")
    for seed in seeds:
        train_hattrick(
            seed=seed,
            epochs=hattrick_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            low_budget=low_budget,
        )


def pin_hattrick_best_checkpoint(*, seed: int, epoch: int) -> Path:
    """Freeze an already-saved Hattrick checkpoint after user early stopping.

    New runs retain every epoch under ``checkpoints``. Validation CSV/JSON
    files alone still cannot reconstruct a historical model that predates
    this archive policy.
    """
    run_dir = ARTIFACT_ROOT / "hattrick" / f"seed_{seed}"
    best_path = run_dir / "best_model.pt"
    final_path = run_dir / "final_model.pt"
    available: dict[int, Path] = {}
    checkpoint_paths = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
    for path in (*checkpoint_paths, best_path, final_path):
        if path.exists():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            available[int(checkpoint["epoch"])] = path
    if epoch not in available:
        raise RuntimeError(
            f"Cannot pin Hattrick seed={seed} epoch={epoch}; saved model-state "
            f"epochs are {sorted(available)}. Validation summaries do not contain "
            "model weights."
        )

    source = available[epoch]
    if source.resolve() != best_path.resolve():
        atomic_copy(source, best_path)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "Hattrick":
        raise RuntimeError(f"Unexpected method in {best_path}")
    if int(checkpoint.get("seed", -1)) != seed or int(checkpoint["epoch"]) != epoch:
        raise RuntimeError("Pinned Hattrick checkpoint identity audit failed")

    validation_summary_path = run_dir / f"validation_epoch_{epoch:03d}_summary.json"
    validation_metrics_path = run_dir / f"validation_epoch_{epoch:03d}_metrics.csv"
    if not validation_summary_path.exists() or not validation_metrics_path.exists():
        raise FileNotFoundError(
            f"Validation evidence for Hattrick epoch {epoch} is incomplete"
        )
    validation_payload = read_json(validation_summary_path)
    if not isinstance(validation_payload, dict):
        raise RuntimeError("Invalid validation summary payload")

    max_completed_epoch = epoch
    if final_path.exists():
        final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        max_completed_epoch = int(final_checkpoint["epoch"])
    record = {
        "status": "EARLY_STOPPED_BY_USER",
        "method": "Hattrick",
        "seed": seed,
        "best_epoch": epoch,
        "best_rank": [float(value) for value in checkpoint["rank"]],
        "eligible_under_strict_high_gate": bool(checkpoint["rank"][0]),
        "max_completed_epoch_before_stop": max_completed_epoch,
        "remaining_hattrick_epochs_skipped": True,
        "selection_source": "validation only; user accepted the saved best checkpoint",
        "test_range_read": False,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "validation_summary_sha256": sha256(validation_summary_path),
        "validation_metrics_sha256": sha256(validation_metrics_path),
    }
    write_json(run_dir / "manual_early_stop.json", record)
    write_json(run_dir / "complete.json", record)
    print(
        f"[pin] Hattrick seed={seed}: epoch {epoch} is fixed as best "
        f"(training had reached epoch {max_completed_epoch})",
        flush=True,
    )
    return best_path


def pin_hattrick_f_best_checkpoint(*, seed: int, epoch: int) -> Path:
    """Freeze an already-saved Hattrick-f checkpoint after user early stopping."""
    run_dir = ARTIFACT_ROOT / "hattrick_f" / f"seed_{seed}"
    best_path = run_dir / "best_model.pt"
    final_path = run_dir / "final_model.pt"
    available: dict[int, Path] = {}
    for path in (best_path, final_path):
        if path.exists():
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            available[int(checkpoint["epoch"])] = path
    if epoch not in available:
        raise RuntimeError(
            f"Cannot pin Hattrick-f seed={seed} epoch={epoch}; saved model-state "
            f"epochs are {sorted(available)}. Validation summaries do not contain "
            "model weights."
        )

    source = available[epoch]
    if source.resolve() != best_path.resolve():
        atomic_copy(source, best_path)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "Hattrick-f":
        raise RuntimeError(f"Unexpected method in {best_path}")
    if int(checkpoint.get("seed", -1)) != seed or int(checkpoint["epoch"]) != epoch:
        raise RuntimeError("Pinned Hattrick-f checkpoint identity audit failed")

    validation_summary_path = run_dir / f"validation_epoch_{epoch:03d}_summary.json"
    validation_metrics_path = run_dir / f"validation_epoch_{epoch:03d}_metrics.csv"
    if not validation_summary_path.exists() or not validation_metrics_path.exists():
        raise FileNotFoundError(
            f"Validation evidence for Hattrick-f epoch {epoch} is incomplete"
        )

    max_completed_epoch = epoch
    if final_path.exists():
        final_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        max_completed_epoch = int(final_checkpoint["epoch"])
    record = {
        "status": "EARLY_STOPPED_BY_USER",
        "method": "Hattrick-f",
        "seed": seed,
        "best_epoch": epoch,
        "best_rank": [float(value) for value in checkpoint["rank"]],
        "eligible_under_validation_gates": bool(checkpoint["rank"][0]),
        "max_completed_epoch_before_stop": max_completed_epoch,
        "remaining_hattrick_f_epochs_skipped": True,
        "selection_source": "validation only; user accepted the saved best checkpoint",
        "test_range_read": False,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256(best_path),
        "validation_summary_sha256": sha256(validation_summary_path),
        "validation_metrics_sha256": sha256(validation_metrics_path),
    }
    write_json(run_dir / "manual_early_stop.json", record)
    write_json(run_dir / "complete.json", record)
    print(
        f"[pin] Hattrick-f seed={seed}: epoch {epoch} is fixed as best "
        f"(training had reached epoch {max_completed_epoch})",
        flush=True,
    )
    return best_path


def train_hattrick_f_only(
    *,
    seeds: list[int],
    pinned_hattrick_epoch: int,
    hattrick_f_epochs: int,
    batch_size: int,
    learning_rate: float,
    low_budget: float,
) -> None:
    """Skip further Phase-A epochs and start Hattrick-f from a pinned model."""
    if not final_oracle_complete():
        raise RuntimeError("The merged 2x oracle is incomplete")
    for seed in seeds:
        phase_a = pin_hattrick_best_checkpoint(
            seed=seed, epoch=pinned_hattrick_epoch
        )
        train_hattrick_f(
            seed=seed,
            phase_a_path=phase_a,
            epochs=hattrick_f_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            low_budget=low_budget,
        )
    select_models(seeds)


def checkpoint_metadata(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "method": checkpoint["method"],
        "seed": int(checkpoint["seed"]),
        "epoch": int(checkpoint["epoch"]),
        "rank": [float(value) for value in checkpoint["rank"]],
        "validation_summary": checkpoint["validation_summary"],
        "validation_diagnostics": checkpoint["validation_diagnostics"],
    }


def select_models(seeds: Iterable[int]) -> dict:
    selection: dict[str, dict] = {}
    for method, directory in (("Hattrick", "hattrick"), ("Hattrick-f", "hattrick_f")):
        candidates: list[tuple[tuple, Path]] = []
        for seed in seeds:
            path = ARTIFACT_ROOT / directory / f"seed_{seed}" / "best_model.pt"
            complete = path.parent / "complete.json"
            if not path.exists() or not complete.exists():
                raise FileNotFoundError(
                    f"Incomplete {method} training artifact for seed {seed}: {path}"
                )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            candidates.append((tuple(checkpoint["rank"]), path))
        _rank, selected = max(candidates, key=lambda item: item[0])
        metadata = checkpoint_metadata(selected)
        metadata["selection_source"] = "validation only; test range not loaded"
        selection[method] = metadata
        selected_name = "selected_hattrick.pt" if method == "Hattrick" else "selected_hattrick_f.pt"
        atomic_copy(selected, ARTIFACT_ROOT / selected_name)

    payload = {
        "selection_used_test": False,
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "unread_test_at_selection": list(TEST_RANGE),
        "models": selection,
    }
    write_json(SELECTION_PATH, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return payload


def test_diagnostics(rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for class_name in CLASSES:
        selected = [row for row in rows if row["class"] == class_name]
        norms = np.asarray(
            [float(row["norm_fulfill"]) for row in selected], dtype=np.float64
        )
        result[class_name] = {
            "n": int(norms.size),
            "norm_fulfill_min": float(norms.min()),
            "norm_fulfill_max": float(norms.max()),
            "fraction_below_0.995": float(np.mean(norms < 0.995)),
        }
    return result


def paired_test_delta(
    baseline_rows: list[dict], candidate_rows: list[dict]
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for class_name in CLASSES:
        baseline = {
            int(row["snapshot"]): float(row["norm_fulfill"])
            for row in baseline_rows
            if row["class"] == class_name
        }
        candidate = {
            int(row["snapshot"]): float(row["norm_fulfill"])
            for row in candidate_rows
            if row["class"] == class_name
        }
        snapshots = sorted(set(baseline) & set(candidate))
        delta = np.asarray(
            [candidate[index] - baseline[index] for index in snapshots],
            dtype=np.float64,
        )
        result[class_name] = {
            "n": len(snapshots),
            "hattrick_f_minus_hattrick_mean": float(delta.mean()),
            "p1": float(np.percentile(delta, 1)),
            "p10": float(np.percentile(delta, 10)),
            "better_fraction": float(np.mean(delta > 0)),
            "equal_fraction": float(np.mean(np.abs(delta) <= 1e-9)),
        }
    return result


def test_selected_models(
    *, batch_size: int, learning_rate: float, seeds: list[int]
) -> None:
    # Selection is completed before the test dataset is instantiated.  This is
    # the central no-test-leakage ordering guarantee.
    # Re-run the cheap validation-artifact selection so --seeds on this command
    # cannot silently reuse a stale selection made from a different seed set.
    selection = select_models(seeds)
    if not isinstance(selection, dict) or selection.get("selection_used_test") is not False:
        raise RuntimeError("Invalid selected_models.json")

    shared, _full, _hattrick_f = load_runtimes()
    from frameworks.hattrick_system import Hattrick

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = build_props(
        device, batch_size=batch_size, epochs=1, learning_rate=learning_rate
    )
    # Do not instantiate the train or validation datasets during final test.
    from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster

    test_dataset = DM_Dataset_within_Cluster(props, 0, *TEST_RANGE)
    if int(test_dataset.max_source_index_read) != TEST_RANGE[1] - 1:
        raise RuntimeError("Test split audit failed")

    all_rows: dict[str, list[dict]] = {}
    summaries: dict[str, dict] = {}
    for method, filename in (
        ("Hattrick", "selected_hattrick.pt"),
        ("Hattrick-f", "selected_hattrick_f.pt"),
    ):
        path = ARTIFACT_ROOT / filename
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = Hattrick(props).to(device=device, dtype=props.dtype)
        model.load_state_dict(checkpoint["model_state_dict"])
        rows, summary, diagnostics = shared.evaluate(
            model, props, test_dataset, TEST_RANGE[0]
        )
        all_rows[method] = rows
        summaries[method] = {
            "classes": summary,
            "diagnostics": diagnostics,
            "tail_diagnostics": test_diagnostics(rows),
            "checkpoint": checkpoint_metadata(path),
        }
        label = "hattrick" if method == "Hattrick" else "hattrick_f"
        write_csv(ARTIFACT_ROOT / f"test_{label}_metrics.csv", rows)
        write_json(ARTIFACT_ROOT / f"test_{label}_summary.json", summaries[method])

    comparison = {
        "status": "COMPLETE",
        "selection_used_test": False,
        "strict_esm": (
            "routing policy consumes ESM predictions; actual 2x traffic is used only "
            "by sequential admission and metric computation"
        ),
        "test": list(TEST_RANGE),
        "methods": summaries,
        "paired_delta": paired_test_delta(
            all_rows["Hattrick"], all_rows["Hattrick-f"]
        ),
    }
    write_json(ARTIFACT_ROOT / "test_comparison.json", comparison)
    print(json.dumps(comparison, indent=2, ensure_ascii=False), flush=True)


def saved_hattrick_f_epoch_checkpoints(seed: int) -> dict[int, Path]:
    """Return saved Hattrick-f model states without changing model selection."""
    run_dir = ARTIFACT_ROOT / "hattrick_f" / f"seed_{seed}"
    candidates = (
        run_dir / "best_model.pt",
        run_dir / "final_model.pt",
        ARTIFACT_ROOT / "selected_hattrick_f.pt",
    )
    available: dict[int, Path] = {}
    for path in candidates:
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("method") != "Hattrick-f":
            continue
        if int(checkpoint.get("seed", -1)) != seed:
            continue
        available.setdefault(int(checkpoint["epoch"]), path)
    return available


def load_audited_test_cache(
    *,
    checkpoint_path: Path,
    metrics_path: Path,
    summary_path: Path,
    method: str,
    seed: int,
    epoch: int,
) -> tuple[list[dict], dict] | None:
    """Reuse a test result only when its identity and complete range audit pass."""
    if not metrics_path.exists() or not summary_path.exists():
        return None
    try:
        summary = read_json(summary_path)
        if not isinstance(summary, dict):
            return None
        metadata = summary.get("checkpoint")
        if not isinstance(metadata, dict):
            return None
        if metadata.get("method") != method:
            return None
        if int(metadata.get("seed", -1)) != seed:
            return None
        if int(metadata.get("epoch", -1)) != epoch:
            return None
        expected_hash = sha256(checkpoint_path)
        if metadata.get("sha256") != expected_hash:
            return None

        rows = read_csv(metrics_path)
        expected_snapshots = set(range(*TEST_RANGE))
        expected_rows = len(expected_snapshots) * len(CLASSES)
        if len(rows) != expected_rows:
            return None
        for class_name in CLASSES:
            snapshots = {
                int(row["snapshot"])
                for row in rows
                if row.get("class") == class_name
            }
            if snapshots != expected_snapshots:
                return None
        classes = summary.get("classes")
        if not isinstance(classes, list) or len(classes) != len(CLASSES):
            return None
        if {
            str(item.get("class")): int(item.get("n", -1))
            for item in classes
            if isinstance(item, dict)
        } != {class_name: len(expected_snapshots) for class_name in CLASSES}:
            return None
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    return rows, summary


def test_hattrick_f_epochs(
    *,
    seed: int,
    epochs: list[int],
    batch_size: int,
    learning_rate: float,
) -> None:
    """Test multiple saved Hattrick-f epochs without pinning or overwriting them."""
    if len(set(epochs)) != len(epochs):
        raise ValueError("Hattrick-f test epochs must not contain duplicates")
    available = saved_hattrick_f_epoch_checkpoints(seed)
    missing = [epoch for epoch in epochs if epoch not in available]
    if missing:
        raise RuntimeError(
            f"Hattrick-f seed={seed} has no saved model weights for epochs {missing}; "
            f"available epochs are {sorted(available)}"
        )

    hattrick_path = ARTIFACT_ROOT / "selected_hattrick.pt"
    if not hattrick_path.exists():
        raise FileNotFoundError(
            "Missing selected_hattrick.pt; complete validation selection first"
        )
    hattrick_checkpoint = torch.load(
        hattrick_path, map_location="cpu", weights_only=False
    )
    if int(hattrick_checkpoint.get("seed", -1)) != seed:
        raise RuntimeError(
            f"Selected Hattrick seed is {hattrick_checkpoint.get('seed')}, not {seed}"
        )

    baseline_cache_options = (
        (
            ARTIFACT_ROOT / "test_hattrick_epoch_comparison_metrics.csv",
            ARTIFACT_ROOT / "test_hattrick_epoch_comparison_summary.json",
        ),
        (
            ARTIFACT_ROOT / "test_hattrick_metrics.csv",
            ARTIFACT_ROOT / "test_hattrick_summary.json",
        ),
    )
    cached_baseline = None
    baseline_cache_source: str | None = None
    for metrics_path, summary_path in baseline_cache_options:
        cached_baseline = load_audited_test_cache(
            checkpoint_path=hattrick_path,
            metrics_path=metrics_path,
            summary_path=summary_path,
            method="Hattrick",
            seed=seed,
            epoch=int(hattrick_checkpoint["epoch"]),
        )
        if cached_baseline is not None:
            baseline_cache_source = str(metrics_path.resolve())
            break

    cached_candidates: dict[int, tuple[list[dict], dict]] = {}
    candidate_cache_sources: dict[int, str] = {}
    for epoch in epochs:
        cache_options = [
            (
                ARTIFACT_ROOT / f"test_hattrick_f_epoch_{epoch:03d}_metrics.csv",
                ARTIFACT_ROOT / f"test_hattrick_f_epoch_{epoch:03d}_summary.json",
            ),
            (
                ARTIFACT_ROOT / "test_hattrick_f_metrics.csv",
                ARTIFACT_ROOT / "test_hattrick_f_summary.json",
            ),
        ]
        for metrics_path, summary_path in cache_options:
            cached = load_audited_test_cache(
                checkpoint_path=available[epoch],
                metrics_path=metrics_path,
                summary_path=summary_path,
                method="Hattrick-f",
                seed=seed,
                epoch=epoch,
            )
            if cached is not None:
                cached_candidates[epoch] = cached
                candidate_cache_sources[epoch] = str(metrics_path.resolve())
                break

    needs_inference = cached_baseline is None or any(
        epoch not in cached_candidates for epoch in epochs
    )
    shared = None
    Hattrick = None
    device = None
    props = None
    test_dataset = None
    if needs_inference:
        shared, _full, _hattrick_f = load_runtimes()
        from frameworks.hattrick_system import Hattrick as HattrickClass
        from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster

        Hattrick = HattrickClass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        props = build_props(
            device, batch_size=batch_size, epochs=1, learning_rate=learning_rate
        )
        test_dataset = DM_Dataset_within_Cluster(props, 0, *TEST_RANGE)
        if int(test_dataset.max_source_index_read) != TEST_RANGE[1] - 1:
            raise RuntimeError("Test split audit failed")

    def evaluate_checkpoint(path: Path) -> tuple[list[dict], dict]:
        if shared is None or Hattrick is None or device is None or props is None:
            raise RuntimeError("Inference runtime was not initialized")
        if test_dataset is None:
            raise RuntimeError("Test dataset was not initialized")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = Hattrick(props).to(device=device, dtype=props.dtype)
        model.load_state_dict(checkpoint["model_state_dict"])
        rows, summary, diagnostics = shared.evaluate(
            model, props, test_dataset, TEST_RANGE[0]
        )
        result = {
            "classes": summary,
            "diagnostics": diagnostics,
            "tail_diagnostics": test_diagnostics(rows),
            "checkpoint": checkpoint_metadata(path),
        }
        return rows, result

    baseline_reused = cached_baseline is not None
    if cached_baseline is None:
        baseline_rows, baseline_summary = evaluate_checkpoint(hattrick_path)
    else:
        baseline_rows, baseline_summary = cached_baseline
    write_csv(ARTIFACT_ROOT / "test_hattrick_epoch_comparison_metrics.csv", baseline_rows)
    write_json(
        ARTIFACT_ROOT / "test_hattrick_epoch_comparison_summary.json",
        baseline_summary,
    )

    candidates: dict[str, dict] = {}
    for epoch in epochs:
        path = available[epoch]
        reused = epoch in cached_candidates
        if reused:
            rows, summary = cached_candidates[epoch]
        else:
            rows, summary = evaluate_checkpoint(path)
        label = f"epoch_{epoch:03d}"
        write_csv(
            ARTIFACT_ROOT / f"test_hattrick_f_{label}_metrics.csv",
            rows,
        )
        write_json(
            ARTIFACT_ROOT / f"test_hattrick_f_{label}_summary.json",
            summary,
        )
        candidates[label] = {
            "summary": summary,
            "paired_delta_vs_hattrick": paired_test_delta(baseline_rows, rows),
            "test_cache_reused": reused,
            "test_cache_source": candidate_cache_sources.get(epoch),
        }

    comparison = {
        "status": "COMPLETE",
        "selection_used_test": False,
        "purpose": (
            "diagnostic comparison only; do not select an epoch from test results"
        ),
        "strict_esm": (
            "routing policy consumes ESM predictions; actual 2x traffic is used only "
            "by sequential admission and metric computation"
        ),
        "test": list(TEST_RANGE),
        "seed": seed,
        "hattrick": {
            "summary": baseline_summary,
            "test_cache_reused": baseline_reused,
            "test_cache_source": baseline_cache_source,
        },
        "hattrick_f": candidates,
    }
    suffix = "_".join(f"{epoch:03d}" for epoch in epochs)
    output_path = ARTIFACT_ROOT / f"test_hattrick_f_epochs_{suffix}.json"
    write_json(output_path, comparison)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "tested_hattrick_f_epochs": epochs,
                "available_hattrick_f_epochs": sorted(available),
                "comparison": str(output_path.resolve()),
                "best_model_was_modified": False,
                "reused_test_results": {
                    "Hattrick": baseline_reused,
                    **{
                        f"Hattrick-f epoch {epoch}": epoch in cached_candidates
                        for epoch in epochs
                    },
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_checks() -> None:
    source_rows = manifest_rows(SOURCE_TOPOLOGY)
    if len(source_rows) < TOTAL_SNAPSHOTS:
        raise RuntimeError(
            f"GEANT source has {len(source_rows)} rows, expected at least {TOTAL_SNAPSHOTS}"
        )
    missing: list[str] = []
    for index in (0, 5999, 6000, 7499, 7500, 10199):
        topology, pairs, tm = source_rows[index]
        for path in (
            ROOT / "topologies" / SOURCE_TOPOLOGY / topology,
            ROOT / "pairs" / SOURCE_TOPOLOGY / pairs,
            source_tm_path(1, False, tm),
            source_tm_path(2, False, tm),
            source_tm_path(3, False, tm),
            source_tm_path(1, True, tm),
            source_tm_path(2, True, tm),
            source_tm_path(3, True, tm),
        ):
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing boundary inputs:\n" + "\n".join(missing))

    import gurobipy

    load_runtimes()
    report = {
        "status": "READY",
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gurobi": gurobipy.gurobi.version(),
        "source_snapshots": len(source_rows),
        "fixed_splits": {
            "train": list(TRAIN_RANGE),
            "validation": list(VALIDATION_RANGE),
            "test": list(TEST_RANGE),
        },
        "prepared": PREPARE_MARKER.exists() and prepared_file_counts_ok(),
        "oracle_complete": final_oracle_complete(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train/select/test six-loss Hattrick and Hattrick-f on GEANT 2x "
            "with fixed 6000/1500/2700 chronological splits"
        )
    )
    parser.add_argument(
        "--stage",
        choices=(
            "check",
            "prepare",
            "oracle",
            "train",
            "hattrick",
            "hattrick-f",
            "finalize-hattrick-f",
            "select",
            "test",
            "test-hattrick-f-epochs",
            "infer",
            "all",
        ),
        default="check",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[490])
    parser.add_argument("--hattrick-epochs", type=positive_int, default=60)
    parser.add_argument("--hattrick-f-epochs", type=positive_int, default=30)
    parser.add_argument(
        "--hattrick-best-epoch",
        type=positive_int,
        default=None,
        help=(
            "with --stage hattrick-f, pin this already-saved Hattrick epoch "
            "and skip the remaining Hattrick epochs"
        ),
    )
    parser.add_argument(
        "--hattrick-f-best-epoch",
        type=positive_int,
        default=None,
        help=(
            "pin this already-saved Hattrick-f epoch before selection/testing; "
            "required by --stage finalize-hattrick-f"
        ),
    )
    parser.add_argument(
        "--hattrick-f-test-epochs",
        nargs="+",
        type=positive_int,
        default=None,
        help=(
            "saved Hattrick-f epochs to evaluate side by side without changing "
            "best_model.pt"
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "JSON list (or JSON-file path) used by --stage infer; each item "
            "must contain name and epoch"
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="legacy single-model name used by --stage infer",
    )
    parser.add_argument(
        "--epoch",
        type=positive_int,
        default=None,
        help="saved model epoch used by --stage infer",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device for --stage infer",
    )
    parser.add_argument(
        "--dataset",
        default="2x",
        help=(
            "registered dataset used by --stage infer (for example 2x or 3x); "
            "the checkpoint must belong to this dataset"
        ),
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="ignore an audited cached test result and run inference again",
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument(
        "--low-budget",
        type=float,
        default=0.03,
        help="maximum validation Low-mean drop allowed for Hattrick-f",
    )
    parser.add_argument("--oracle-chunk-size", type=positive_int, default=256)
    parser.add_argument(
        "--oracle-workers",
        type=positive_int,
        default=1,
        help="parallel Gurobi processes; use 1 unless the license permits more",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="regenerate only the dedicated target 2x traffic files",
    )
    args = parser.parse_args()
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.low_budget < 0:
        parser.error("--low-budget must be non-negative")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("check", "all"):
        run_checks()
    if args.stage in ("prepare", "all"):
        prepare_data(force=args.force_prepare)
    if args.stage in ("oracle", "all"):
        generate_oracle(args.oracle_chunk_size, args.oracle_workers)
    if args.stage in ("train", "all"):
        train_models(
            seeds=args.seeds,
            hattrick_epochs=args.hattrick_epochs,
            hattrick_f_epochs=args.hattrick_f_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            low_budget=args.low_budget,
        )
    if args.stage == "hattrick":
        train_hattrick_only(
            seeds=args.seeds,
            hattrick_epochs=args.hattrick_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            low_budget=args.low_budget,
        )
    if args.stage == "hattrick-f":
        if args.hattrick_best_epoch is None:
            parser.error("--stage hattrick-f requires --hattrick-best-epoch")
        train_hattrick_f_only(
            seeds=args.seeds,
            pinned_hattrick_epoch=args.hattrick_best_epoch,
            hattrick_f_epochs=args.hattrick_f_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            low_budget=args.low_budget,
        )
    if args.stage == "finalize-hattrick-f":
        if args.hattrick_f_best_epoch is None:
            parser.error(
                "--stage finalize-hattrick-f requires --hattrick-f-best-epoch"
            )
        for seed in args.seeds:
            pin_hattrick_f_best_checkpoint(
                seed=seed, epoch=args.hattrick_f_best_epoch
            )
        select_models(args.seeds)
    if args.stage == "select":
        select_models(args.seeds)
    if args.stage in ("test", "all"):
        if args.hattrick_f_best_epoch is not None:
            for seed in args.seeds:
                pin_hattrick_f_best_checkpoint(
                    seed=seed, epoch=args.hattrick_f_best_epoch
                )
        test_selected_models(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seeds=args.seeds,
        )
    if args.stage == "test-hattrick-f-epochs":
        if args.hattrick_f_test_epochs is None:
            parser.error(
                "--stage test-hattrick-f-epochs requires "
                "--hattrick-f-test-epochs"
            )
        if len(args.seeds) != 1:
            parser.error("multi-epoch diagnostic testing requires exactly one seed")
        test_hattrick_f_epochs(
            seed=args.seeds[0],
            epochs=args.hattrick_f_test_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    if args.stage == "infer":
        if len(args.seeds) != 1:
            parser.error("registered inference requires exactly one seed")
        registered = load_module(
            "full_geant_registered_methods", THIS_DIR / "registered_methods.py"
        )
        try:
            if args.models is not None:
                if args.method is not None or args.epoch is not None:
                    parser.error("--models cannot be combined with --method/--epoch")
                model_list = registered.parse_model_list_argument(args.models)
            else:
                if args.method is None or args.epoch is None:
                    parser.error(
                        "--stage infer requires --models, or legacy "
                        "--method and --epoch"
                    )
                model_list = [{"name": args.method, "epoch": args.epoch}]
            results = registered.infer_registered_models(
                model_list,
                seed=args.seeds[0],
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device_name=args.device,
                dataset_name=args.dataset,
                force=args.force_inference,
                base_runner=(
                    sys.modules[__name__]
                    if args.dataset.casefold() in {"2x", "geant-2x", "geant_2x"}
                    else None
                ),
            )
        except (FileNotFoundError, KeyError, ValueError) as error:
            parser.error(str(error))
        print(
            json.dumps(
                {"models": model_list, "results": results},
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
