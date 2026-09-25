from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import time
from pathlib import Path

import numpy as np

import _workspace as ws


def parse_factor(value: str) -> int:
    match = re.fullmatch(r"([1-9]\d*)x", value.strip().lower())
    if match is None:
        raise argparse.ArgumentTypeError("dataset must use the form <integer>x")
    factor = int(match.group(1))
    if factor == 1:
        raise argparse.ArgumentTypeError("1x is the immutable source dataset")
    return factor


def safe_clear(target: Path) -> None:
    resolved = target.resolve()
    data_root = ws.DATA_ROOT.resolve()
    if resolved.parent != data_root or resolved == data_root:
        raise RuntimeError(f"Refusing to clear unsafe path: {resolved}")
    if target.exists():
        shutil.rmtree(target)


def scale_pickle(source: Path, target: Path, factor: int) -> None:
    with source.open("rb") as handle:
        values = np.asarray(pickle.load(handle))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(values * factor, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an integer-load dataset by scaling every 1x actual and ESM TM"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm replacement without an interactive prompt",
    )
    args = parser.parse_args()
    try:
        factor = parse_factor(args.dataset)
        dataset = f"{factor}x"
        source = ws.DATA_ROOT / "1x"
        target = ws.DATA_ROOT / dataset
        if not (source / "traffic/1").is_dir():
            raise FileNotFoundError(f"Missing source dataset: {source}")
        if target.exists() and any(target.iterdir()):
            if not args.yes:
                answer = input(
                    f"{target} already exists. Clear it and regenerate {dataset}? [y/N]: "
                ).strip().casefold()
                if answer not in {"y", "yes"}:
                    raise SystemExit("generation cancelled; existing data was not changed")
            safe_clear(target)
        target.mkdir(parents=True, exist_ok=True)
        (target / "topology").mkdir(parents=True, exist_ok=True)
        (target / "pairs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "topology/t1.json", target / "topology/t1.json")
        shutil.copy2(source / "pairs/t1.pkl", target / "pairs/t1.pkl")
        shutil.copy2(source / "manifest.txt", target / "manifest.txt")
        source_topology = "geant_full_shared_load1x_train"
        target_topology = f"geant_full_shared_load{factor}x_train"
        for path in (source / "path_cache").glob("*"):
            if path.is_file():
                name = path.name.replace(source_topology, target_topology)
                destination = target / "path_cache" / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        files = []
        for priority in (1, 2, 3):
            for suffix in ("", "_esm"):
                directory = source / "traffic" / f"{priority}{suffix}"
                files.extend(
                    (path, target / "traffic" / f"{priority}{suffix}" / path.name)
                    for path in sorted(directory.glob("*.pkl"))
                )
        started = time.perf_counter()
        for index, (source_file, target_file) in enumerate(files, start=1):
            scale_pickle(source_file, target_file, factor)
            if index % 5000 == 0:
                print(f"[generate] {index}/{len(files)}", flush=True)
        metadata = {
            "schema_version": 1,
            "dataset": dataset,
            "factor": factor,
            "source": "data/1x",
            "scaled": ["actual", "esm_prediction"],
            "files": len(files),
            "elapsed_seconds": time.perf_counter() - started,
            "baseline_required": True,
        }
        ws.write_json(target / "generation.json", metadata)
        manifest_target = ws.ROOT / "manifest" / f"{target_topology}_manifest.txt"
        manifest_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target / "manifest.txt", manifest_target)
    except (argparse.ArgumentTypeError, FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"generate {dataset} finished")
    print(f"data: {target.resolve()}")
    print(f"next: run baseline.py --dataset {dataset} if its baseline is stale or missing")


if __name__ == "__main__":
    main()
