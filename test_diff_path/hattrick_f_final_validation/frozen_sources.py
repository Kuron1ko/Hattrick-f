from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEST_DIR = HERE.parent

EXPECTED = {
    ROOT / "frameworks/hattrick_system.py": "4f7b15dba81b23f0e23d432e0b65f8ae895b459070dd8249a22c1345b15eb862",
    ROOT / "utils/training_utils.py": "b4b8c08ef353194ae107fc6756bde6f7668f12e5d6e87bf5f26280dbf4057185",
    TEST_DIR / "shared2x_full_objectives/run_experiment.py": "a77c112d25f549e2cbe73b9a41f4ef4cebe10b60061e8683f9faebd86b112a42",
    TEST_DIR / "shared2x_full_objectives/ordered_projection.py": "a8ed960f641e9cb0786ab8edee3b4a3c571255c3e76c939f0d6803ce01543e3a",
    TEST_DIR / "shared2x_hattrick_f/run_experiment.py": "33d9d4d634d6661e935ba5800a90df4b2f5eaeeb0a42fb4465c1528b89025d18",
    TEST_DIR / "shared2x_hattrick_f/run_level4.py": "dc13339b20a28ae9d6520e5466cd4bc5f0c7ec5ec704d5aabcf072884cb6d8d9",
    TEST_DIR / "shared2x_order_regularizer/run_experiment.py": "317d0b796773963352a5b92652e6273074707bad42b3078a734e569366789310",
    ROOT / "utils/AdamOptimizer.py": "1dbb87b114b3845f86718a401e1f4ad2acd09a29d4707706505a35327f3e0325",
    ROOT / "utils/robust_proj_utils.py": "76affbd2e7f27863d6371794eee834fb8549558a3156aec40e1e06227cea0c7e",
    ROOT / "utils/snapshot_utils.py": "657e245fb127c03c29f123f637604fe3f21ced266e29993ed69fb30cff7010ad",
    ROOT / "utils/cluster_utils.py": "f9ccc1af7918cdd0b739bf9cfa6b492a4120bc237ae89c3f485dcd87d1a69df4",
    ROOT / "utils/build_dataset_within_cluster.py": "53c69ca0f24b8d514252dabd58192a3eee1b5584257065e4e1f460a85037be46",
    HERE / "build_frozen_manifest.py": "2bb39dda55651dcdab82e4e619931da83bfc1fe4862fb7bd59c8f0afdf12a8a6",
}

MANIFEST_PATH = HERE / "frozen_manifest.json"
EXPECTED_MANIFEST_SHA256 = (
    "d1c1e9868ca3e7272502f2b90525d3ce80ad0900b3cdcec0362f0d2916de639f"
)
EXPECTED_TRAFFIC_ROWS = {
    (window, class_name, kind)
    for window in ("near", "far")
    for class_name in ("High", "Medium", "Low")
    for kind in ("actual", "esm")
}
EXPECTED_WINDOW_RANGES = {"near": (500, 1000), "far": (9000, 9500)}
EXPECTED_SAMPLE_OFFSETS = (0, 125, 250, 375, 499)
TM_PATTERN = re.compile(r"^t([1-9][0-9]*)\.pkl$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    root = ROOT.resolve()
    if candidate == root or root not in candidate.parents:
        raise RuntimeError(f"manifest path escapes repository root: {relative!r}")
    return candidate


def verify_manifest() -> dict[str, str]:
    manifest_digest = sha256(MANIFEST_PATH)
    if manifest_digest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen manifest hash mismatch: "
            f"expected={EXPECTED_MANIFEST_SHA256}, observed={manifest_digest}"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != 1:
        raise RuntimeError("unsupported frozen manifest schema")

    observed = {str(MANIFEST_PATH.resolve()): manifest_digest}
    exact_paths: set[str] = set()
    for row in manifest.get("exact_files", []):
        relative = str(row["relative_path"])
        if relative in exact_paths:
            raise RuntimeError(f"duplicate exact manifest path: {relative}")
        exact_paths.add(relative)
        path = repository_path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        expected_size = int(row["size_bytes"])
        actual_digest = sha256(path)
        if actual_size != expected_size or actual_digest != str(row["sha256"]):
            raise RuntimeError(
                f"Frozen exact-file audit failed for {path}: "
                f"size {actual_size}/{expected_size}, "
                f"sha256 {actual_digest}/{row['sha256']}"
            )
        observed[str(path)] = actual_digest

    traffic_rows = manifest.get("traffic_window_samples", [])
    row_keys = {
        (str(row["window"]), str(row["class"]), str(row["kind"]))
        for row in traffic_rows
    }
    if row_keys != EXPECTED_TRAFFIC_ROWS or len(traffic_rows) != len(row_keys):
        raise RuntimeError(
            "Frozen traffic-window manifest rows are incomplete or duplicated: "
            f"observed={sorted(row_keys)}"
        )
    directory_inventory: dict[Path, dict[int, Path]] = {}
    for row in traffic_rows:
        window = str(row["window"])
        start, stop = EXPECTED_WINDOW_RANGES[window]
        if tuple(int(value) for value in row["zero_based_snapshot_range"]) != (
            start,
            stop,
        ):
            raise RuntimeError(f"unexpected {window} snapshot range in manifest")
        if tuple(int(value) for value in row["file_number_range_inclusive"]) != (
            start + 1,
            stop,
        ):
            raise RuntimeError(f"unexpected {window} file-number range in manifest")
        if tuple(int(value) for value in row["sample_offsets_from_window_start"]) != (
            EXPECTED_SAMPLE_OFFSETS
        ):
            raise RuntimeError(f"unexpected {window} traffic sampling offsets")

        directory = repository_path(str(row["relative_directory"]))
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        if directory not in directory_inventory:
            inventory: dict[int, Path] = {}
            for path in directory.iterdir():
                match = TM_PATTERN.fullmatch(path.name)
                if match is None or not path.is_file():
                    continue
                number = int(match.group(1))
                if number in inventory:
                    raise RuntimeError(f"duplicate TM number {number} in {directory}")
                inventory[number] = path
            directory_inventory[directory] = inventory
        inventory = directory_inventory[directory]
        window_count = sum(start < number <= stop for number in inventory)
        if window_count != int(row["expected_window_file_count"]):
            raise RuntimeError(
                f"traffic-window count mismatch for {directory}/{window}: "
                f"observed={window_count}, expected={row['expected_window_file_count']}"
            )
        if window_count != int(row["observed_window_file_count"]):
            raise RuntimeError(
                f"traffic-window inventory changed for {directory}/{window}"
            )
        if len(inventory) != int(row["total_numbered_tm_files_in_directory"]):
            raise RuntimeError(
                f"total TM inventory changed for {directory}: "
                f"observed={len(inventory)}, "
                f"expected={row['total_numbered_tm_files_in_directory']}"
            )

        samples = row.get("samples", [])
        expected_numbers = [start + 1 + offset for offset in EXPECTED_SAMPLE_OFFSETS]
        if [int(sample["file_number"]) for sample in samples] != expected_numbers:
            raise RuntimeError(f"traffic samples changed for {directory}/{window}")
        for sample in samples:
            number = int(sample["file_number"])
            path = inventory.get(number)
            if path is None or path.name != str(sample["filename"]):
                raise RuntimeError(
                    f"sampled TM is missing or renamed: {directory}/t{number}.pkl"
                )
            actual_size = path.stat().st_size
            expected_size = int(sample["size_bytes"])
            actual_digest = sha256(path)
            if actual_size != expected_size or actual_digest != str(sample["sha256"]):
                raise RuntimeError(
                    f"sampled TM audit failed for {path}: "
                    f"size {actual_size}/{expected_size}, "
                    f"sha256 {actual_digest}/{sample['sha256']}"
                )
            observed[str(path.resolve())] = actual_digest
    return observed


def verify() -> dict[str, str]:
    observed = {str(path.resolve()): sha256(path) for path in EXPECTED}
    mismatches = {
        str(path.resolve()): {"expected": expected, "observed": observed[str(path.resolve())]}
        for path, expected in EXPECTED.items()
        if observed[str(path.resolve())] != expected
    }
    if mismatches:
        raise RuntimeError(f"Frozen source audit failed: {mismatches}")
    observed.update(verify_manifest())
    return observed


if __name__ == "__main__":
    for path, digest in verify().items():
        print(f"{digest}  {path}")
