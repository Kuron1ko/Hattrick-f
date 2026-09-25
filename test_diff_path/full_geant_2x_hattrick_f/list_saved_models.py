from __future__ import annotations

"""List saved registered checkpoints and their validation metrics."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch

import registered_methods as registered


COLUMNS = (
    "epoch",
    "eligible",
    "high_norm_mean",
    "high_norm_p10",
    "high_norm_p1",
    "high_norm_min",
    "medium_norm_mean",
    "medium_norm_p10",
    "medium_norm_p1",
    "low_norm_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List the saved checkpoints for one registered method and show "
            "the validation metrics stored inside each checkpoint"
        )
    )
    parser.add_argument(
        "--dataset",
        default="2x",
        help="registered traffic condition, for example 1x, 2x, or 3x",
    )
    parser.add_argument(
        "--method",
        default=None,
        help=(
            "registered method name, for example Hattrick or Hattrick-f3; "
            "omit it to print one validation-metrics table per saved method"
        ),
    )
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="also show model size and the saved training configuration",
    )
    args = parser.parse_args()
    if args.show_config and args.method is None:
        parser.error("--show-config requires --method")
    return args


def number(row: dict, field: str, path: Path) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Missing metric {field!r} in {path}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite metric {field!r} in {path}")
    return value


def history_path(config: dict, seed: int, epoch: int) -> Path:
    inspected: set[Path] = set()
    for checkpoint in registered.checkpoint_candidates(
        config, seed=seed, epoch=epoch
    ):
        for directory in (checkpoint.parent, checkpoint.parent.parent):
            candidate = directory / "train_history.csv"
            if candidate in inspected:
                continue
            inspected.add(candidate)
            if candidate.is_file():
                return candidate
    raise RuntimeError(
        "Cannot find train_history.csv beside the registered checkpoints"
    )


def load_history(path: Path) -> dict[int, dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[int, dict] = {}
    for row in rows:
        try:
            epoch = int(row["epoch"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"Malformed training history: {path}") from error
        result[epoch] = row
    return result


def checkpoint_row(history: dict[int, dict], path: Path, epoch: int) -> dict:
    if epoch not in history:
        raise RuntimeError(f"Epoch {epoch} is missing from {path}")
    source = history[epoch]
    eligible = number(source, "eligible", path)
    return {
        "epoch": epoch,
        "eligible": int(round(eligible)),
        **{
            field: number(source, field, path)
            for field in COLUMNS
            if field not in ("epoch", "eligible")
        },
    }


def cell(value: object) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def print_table(rows: list[dict]) -> None:
    values = [[cell(row[column]) for column in COLUMNS] for row in rows]
    widths = [
        max(len(column), *(len(row[index]) for row in values))
        for index, column in enumerate(COLUMNS)
    ]
    print(" ".join(column.ljust(widths[index]) for index, column in enumerate(COLUMNS)))
    print(" ".join("-" * width for width in widths))
    for row in values:
        print(" ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def checkpoint_details(
    method: str, dataset: str, seed: int, epoch: int
) -> tuple[Path, dict, int, int]:
    _canonical, path, _identity = registered.resolve_checkpoint(
        method, epoch, seed=seed, dataset_name=dataset
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Checkpoint has no model_state_dict: {path}")
    parameter_count = sum(int(tensor.numel()) for tensor in state.values())
    tensor_bytes = sum(
        int(tensor.numel() * tensor.element_size()) for tensor in state.values()
    )
    return path, checkpoint, parameter_count, tensor_bytes


def list_methods(dataset: str, registry_value: dict, seed: int) -> None:
    tables: list[tuple[str, list[dict]]] = []
    for method, method_config in registry_value["methods"].items():
        if dataset not in method_config.get("datasets", {}):
            continue
        epochs = registered.available_epochs(
            method, seed=seed, dataset_name=dataset
        )
        if not epochs:
            continue
        training_history_path = history_path(method_config["datasets"][dataset], seed, epochs[0])
        history = load_history(training_history_path)
        rows = [
            checkpoint_row(history, training_history_path, epoch)
            for epoch in epochs
        ]
        tables.append((method, rows))
    if not tables:
        raise RuntimeError(f"No saved methods for dataset={dataset}, seed={seed}")
    for index, (method, rows) in enumerate(tables):
        if index:
            print()
        print(f"method: {method}")
        print_table(rows)


def print_config(method: str, dataset: str, seed: int, epoch: int) -> None:
    path, checkpoint, parameter_count, tensor_bytes = checkpoint_details(
        method, dataset, seed, epoch
    )
    config = checkpoint.get("config", checkpoint.get("signature"))
    print()
    print(f"method: {method}")
    print(f"dataset: {dataset}")
    print(f"configuration_epoch: {epoch}")
    print(f"parameters: {parameter_count}")
    print(f"model_tensor_bytes: {tensor_bytes}")
    print(f"checkpoint_bytes: {path.stat().st_size}")
    print(f"checkpoint: {path}")
    print("config:")
    print(json.dumps(config, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    try:
        dataset, _dataset_config, registry_value = registered.registered_dataset(
            args.dataset
        )
        if args.method is None:
            list_methods(dataset, registry_value, args.seed)
            return
        canonical, method_config, _context = registered.registered_method(
            args.method, dataset
        )
        epochs = registered.available_epochs(
            canonical, seed=args.seed, dataset_name=dataset
        )
        if not epochs:
            raise RuntimeError(
                f"No saved {dataset} {canonical} checkpoints for seed={args.seed}"
            )
        training_history_path = history_path(method_config, args.seed, epochs[0])
        history = load_history(training_history_path)
        rows = [
            checkpoint_row(history, training_history_path, epoch)
            for epoch in epochs
        ]
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print_table(rows)
    if args.show_config:
        print_config(canonical, dataset, args.seed, epochs[-1])


if __name__ == "__main__":
    main()
