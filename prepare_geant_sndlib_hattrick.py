import argparse
import csv
import os
import pickle
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np


DEMAND_RE = re.compile(
    r"^\s*\S+\s*\(\s*(?P<src>\S+)\s+(?P<dst>\S+)\s*\)\s+\S+\s+(?P<value>[0-9.eE+-]+)"
)
NODE_RE = re.compile(r"^\s*(?P<node>\S+)\s+\(")


def parse_nodes(path: Path) -> list[str]:
    nodes: list[str] = []
    in_nodes = False
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "NODES (":
                in_nodes = True
                continue
            if in_nodes and stripped == ")":
                break
            if in_nodes:
                match = NODE_RE.match(line)
                if match:
                    nodes.append(match.group("node"))
    return nodes


def parse_demands(path: Path, node_to_id: dict[str, int], pair_to_index: dict[tuple[int, int], int]) -> np.ndarray:
    tm = np.zeros((len(pair_to_index), 1), dtype=np.float32)
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            match = DEMAND_RE.match(line)
            if not match:
                continue
            src = node_to_id[match.group("src")]
            dst = node_to_id[match.group("dst")]
            pair_index = pair_to_index.get((src, dst))
            if pair_index is None:
                continue
            # SNDlib marks GEANT demand values as MBITPERSEC. The bundled GEANT
            # topology capacities are in Gbps, so convert Mbps -> Gbps.
            tm[pair_index, 0] = float(match.group("value")) / 1000.0
    return tm


def nonempty_matrix_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("demandMatrix-geant-uhlig-15min-*.txt"))
    nonempty: list[Path] = []
    for path in files:
        with path.open("r", encoding="utf-8-sig") as f:
            if any(DEMAND_RE.match(line) for line in f):
                nonempty.append(path)
    return nonempty


def backup_existing_dirs(base: Path, names: list[str]) -> Path | None:
    existing = [base / name for name in names if (base / name).exists()]
    if not existing:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = base / "_backup" / f"geant_before_sndlib_{stamp}"
    backup_root.mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), str(backup_root / path.name))
    return backup_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SNDlib GEANT matrices to Hattrick pkl format.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("..")
        / "downloads"
        / "sndlib_geant_native"
        / "directed-geant-uhlig-15min-over-4months-ALL-native",
    )
    parser.add_argument("--topo", default="geant")
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    tm_root = repo_root / "traffic_matrices"
    pairs_path = repo_root / "pairs" / args.topo / "t1.pkl"
    manifest_path = repo_root / "manifest" / f"{args.topo}_manifest.txt"
    source_map_path = tm_root / f"{args.topo}_sndlib_source_map.csv"

    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)

    with pairs_path.open("rb") as f:
        pairs = pickle.load(f)
    pairs = np.asarray(pairs, dtype=np.int64)
    pair_to_index = {tuple(pair): i for i, pair in enumerate(pairs.tolist())}

    files = nonempty_matrix_files(args.input_dir)
    manifest_count = sum(1 for _ in manifest_path.open("r", encoding="utf-8"))
    if len(files) != manifest_count:
        raise RuntimeError(
            f"Nonempty SNDlib matrices ({len(files)}) do not match manifest lines ({manifest_count})."
        )

    nodes = parse_nodes(files[0])
    if len(nodes) != len(set(nodes)):
        raise RuntimeError("Duplicate node IDs found in SNDlib node section.")
    node_to_id = {node: i for i, node in enumerate(nodes)}

    output_dirs = [f"{args.topo}_{i}" for i in range(1, 4)]
    pred_dirs = [f"{args.topo}_{i}_esm" for i in range(1, 4)]
    if not args.overwrite:
        backup_root = backup_existing_dirs(tm_root, output_dirs + pred_dirs)
        if backup_root is not None:
            print(f"Backed up existing GEANT traffic dirs to: {backup_root}")

    for dirname in output_dirs:
        (tm_root / dirname).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    totals = np.zeros(3, dtype=np.float64)

    with source_map_path.open("w", newline="", encoding="utf-8") as f_map:
        writer = csv.writer(f_map)
        writer.writerow(["hattrick_file", "sndlib_file"])
        for index, path in enumerate(files, start=1):
            matrix = parse_demands(path, node_to_id, pair_to_index)
            high = rng.uniform(0.45, 0.50, size=matrix.shape).astype(np.float32)
            mid = rng.uniform(0.40, 0.45, size=matrix.shape).astype(np.float32)
            low = (1.0 - high - mid).astype(np.float32)
            split_matrices = [matrix * high, matrix * mid, matrix * low]
            for priority, split_matrix in enumerate(split_matrices, start=1):
                out_path = tm_root / f"{args.topo}_{priority}" / f"t{index}.pkl"
                with out_path.open("wb") as f_out:
                    pickle.dump(split_matrix.astype(np.float32), f_out)
                totals[priority - 1] += float(split_matrix.sum())
            writer.writerow([f"t{index}.pkl", path.name])

            if index % 1000 == 0:
                print(f"Converted {index}/{len(files)} matrices")

    print(f"Converted {len(files)} matrices for topology '{args.topo}'.")
    print(
        "Traffic totals by class (Gbps snapshots sum): "
        f"high={totals[0]:.6f}, medium={totals[1]:.6f}, low={totals[2]:.6f}"
    )
    print(f"Source map: {source_map_path}")


if __name__ == "__main__":
    main()
