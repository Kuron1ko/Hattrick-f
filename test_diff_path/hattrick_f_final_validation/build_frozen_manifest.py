from __future__ import annotations

"""Build the bounded provenance manifest used by final validation.

Topology/path assets and transitive runtime dependencies are hashed in full.
The two 500-snapshot temporal holdouts are intentionally represented by file
counts plus fixed samples; this is an integrity tripwire, not a cryptographic
commitment to every traffic-matrix file in the windows.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "frozen_manifest.json"

EXACT_FILES = (
    ("runtime_dependency", "utils/AdamOptimizer.py"),
    ("runtime_dependency", "utils/robust_proj_utils.py"),
    ("runtime_dependency", "utils/snapshot_utils.py"),
    ("runtime_dependency", "utils/cluster_utils.py"),
    ("runtime_dependency", "utils/build_dataset_within_cluster.py"),
    ("holdout_topology", "topologies/geant/t1.json"),
    ("holdout_pairs", "pairs/geant/t1.pkl"),
    (
        "holdout_8sp_paths_pte_source",
        "topologies/paths/geant_8_paths_cluster_0.pkl",
    ),
    (
        "holdout_8sp_paths_dict",
        "topologies/paths_dict/geant_8_paths_dict_cluster_0.pkl",
    ),
    (
        "holdout_8sp_padded_path_edges",
        "topologies/padded_edge_ids_per_path/"
        "geant_8_paths_cluster_0_padded_edge_ids_per_path.pkl",
    ),
    (
        "holdout_8sp_path_edge_dictionary",
        "topologies/padded_edge_ids_per_path/"
        "geant_8_paths_cluster_0_edge_ids_dict.pkl",
    ),
    (
        "training_topology",
        "topologies/geant_priomask500_shared_load2x_train/t1.json",
    ),
    (
        "training_pairs",
        "pairs/geant_priomask500_shared_load2x_train/t1.pkl",
    ),
    (
        "training_8sp_paths_pte_source",
        "topologies/paths/"
        "geant_priomask500_shared_load2x_train_8_paths_cluster_0.pkl",
    ),
    (
        "training_8sp_paths_dict",
        "topologies/paths_dict/"
        "geant_priomask500_shared_load2x_train_8_paths_dict_cluster_0.pkl",
    ),
    (
        "training_8sp_padded_path_edges",
        "topologies/padded_edge_ids_per_path/"
        "geant_priomask500_shared_load2x_train_8_paths_cluster_0_"
        "padded_edge_ids_per_path.pkl",
    ),
    (
        "training_8sp_path_edge_dictionary",
        "topologies/padded_edge_ids_per_path/"
        "geant_priomask500_shared_load2x_train_8_paths_cluster_0_"
        "edge_ids_dict.pkl",
    ),
    ("holdout_filename_index", "results/geant/8sp/0/filenames.txt"),
    (
        "training_filename_index",
        "results/geant_priomask500_shared_load2x_train/8sp/0/filenames.txt",
    ),
)

WINDOWS = {
    "near": (500, 1000),
    "far": (9000, 9500),
}
CLASSES = {"High": 1, "Medium": 2, "Low": 3}
KINDS = {"actual": "", "esm": "_esm"}
SAMPLE_OFFSETS = (0, 125, 250, 375, 499)
TM_PATTERN = re.compile(r"^t([1-9][0-9]*)\.pkl$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_file_rows() -> list[dict]:
    rows = []
    for role, relative in EXACT_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "role": role,
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def numbered_tm_files(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in directory.iterdir():
        match = TM_PATTERN.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        number = int(match.group(1))
        if number in result:
            raise RuntimeError(f"duplicate traffic-matrix number {number} in {directory}")
        result[number] = path
    return result


def traffic_window_rows() -> list[dict]:
    inventories: dict[Path, dict[int, Path]] = {}
    rows = []
    for window, (start, stop) in WINDOWS.items():
        first_number = start + 1
        last_number = stop
        expected_numbers = set(range(first_number, last_number + 1))
        sample_numbers = [first_number + offset for offset in SAMPLE_OFFSETS]
        if sample_numbers[-1] != last_number:
            raise RuntimeError("fixed traffic sampling must include the final file")
        for class_name, class_index in CLASSES.items():
            for kind, suffix in KINDS.items():
                relative_dir = f"traffic_matrices/geant_{class_index}{suffix}"
                directory = ROOT / relative_dir
                if not directory.is_dir():
                    raise FileNotFoundError(directory)
                if directory not in inventories:
                    inventories[directory] = numbered_tm_files(directory)
                inventory = inventories[directory]
                observed_numbers = expected_numbers & set(inventory)
                missing = sorted(expected_numbers - observed_numbers)
                if missing:
                    raise RuntimeError(
                        f"{relative_dir}/{window} is missing TM numbers {missing[:10]}"
                    )
                samples = []
                for number in sample_numbers:
                    path = inventory[number]
                    samples.append(
                        {
                            "file_number": number,
                            "filename": path.name,
                            "size_bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                    )
                rows.append(
                    {
                        "window": window,
                        "zero_based_snapshot_range": [start, stop],
                        "file_number_range_inclusive": [first_number, last_number],
                        "class": class_name,
                        "kind": kind,
                        "relative_directory": relative_dir,
                        "expected_window_file_count": stop - start,
                        "observed_window_file_count": len(observed_numbers),
                        "total_numbered_tm_files_in_directory": len(inventory),
                        "sample_offsets_from_window_start": list(SAMPLE_OFFSETS),
                        "samples": samples,
                    }
                )
    return rows


def build_manifest() -> dict:
    return {
        "schema_version": 1,
        "root_semantics": (
            "relative paths are resolved against the Hattrick-main repository root"
        ),
        "coverage": {
            "exact_files": (
                "full SHA256 commitment for listed runtime, topology, pairs, and "
                "8sp path/PTE-source assets"
            ),
            "traffic_windows": (
                "directory inventory counts plus first/last and three fixed interior "
                "sample hashes; not a full-content commitment to every TM file"
            ),
        },
        "exact_files": exact_file_rows(),
        "traffic_window_samples": traffic_window_rows(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/check frozen validation manifest")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build_manifest()
    encoded = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    output = args.output.resolve()
    if args.check:
        if not output.is_file():
            raise FileNotFoundError(output)
        current = output.read_text(encoding="utf-8")
        if current != encoded:
            raise RuntimeError(f"frozen manifest differs from current assets: {output}")
        print(output)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
