from __future__ import annotations

import argparse
import json
import time

import torch

import _workspace as ws
from method import hattrick_system as system


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict-ESM inference with an audited local cache"
    )
    parser.add_argument("--dataset", required=True, choices=tuple(ws.DATASETS))
    parser.add_argument("--method", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        args.method = ws.canonical_method(args.method)
    except ValueError as error:
        parser.error(str(error))
    if args.epoch <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epoch, batch size, and learning rate must be positive")
    return args


def infer_one(args: argparse.Namespace) -> dict:
    dataset = ws.canonical_dataset(args.dataset)
    method = ws.canonical_method(args.method)
    checkpoint_path = ws.resolve_checkpoint(dataset, method, args.epoch, args.seed)
    checkpoint_hash = ws.sha256(checkpoint_path)
    output = ws.cache_dir(dataset, method, args.seed, args.epoch)
    metrics_path = output / "metrics.csv"
    summary_path = output / "summary.json"
    metadata_path = output / "cache.json"
    print("infer start", flush=True)
    if not args.force and metrics_path.is_file() and metadata_path.is_file():
        metadata = ws.read_json(metadata_path)
        if (
            isinstance(metadata, dict)
            and metadata.get("checkpoint_sha256") == checkpoint_hash
            and metadata.get("test") == list(ws.DATASETS[dataset]["test"])
        ):
            ws.validate_test_metrics(metrics_path, dataset)
            result = {
                "cache_reused": True,
                "cache": str(output.resolve()),
                "metrics": str(metrics_path.resolve()),
                "summary": str(summary_path.resolve()) if summary_path.is_file() else None,
            }
            print(f"cache found at {output.resolve()}", flush=True)
            return result

    ws.ensure_workspace_inputs(dataset)
    device = ws.resolve_device(args.device)
    props = system.build_props(
        dataset,
        device=device,
        batch_size=args.batch_size,
        epochs=1,
        learning_rate=args.learning_rate,
    )
    from frameworks.hattrick_system import Hattrick
    from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster

    test_start, test_end = ws.DATASETS[dataset]["test"]
    test_dataset = DM_Dataset_within_Cluster(props, 0, test_start, test_end)
    if int(test_dataset.max_source_index_read) != test_end - 1:
        raise RuntimeError("Test split audit failed")
    checkpoint = ws.load_checkpoint(checkpoint_path, device=device)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    started = time.perf_counter()
    rows, summary, diagnostics = system.evaluate(
        model, props, test_dataset, test_start
    )
    ws.finite_summary(summary)
    output.mkdir(parents=True, exist_ok=True)
    ws.write_csv(metrics_path, rows)
    ws.write_json(summary_path, {"classes": summary, "diagnostics": diagnostics})
    metadata = {
        "schema_version": 1,
        "dataset": dataset,
        "method": method,
        "seed": args.seed,
        "epoch": args.epoch,
        "strict_esm": True,
        "test": [test_start, test_end],
        "checkpoint": str(checkpoint_path.relative_to(ws.ROOT)),
        "checkpoint_sha256": checkpoint_hash,
        "metrics": "metrics.csv",
        "summary": "summary.json",
        "runtime_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    ws.write_json(metadata_path, metadata)
    ws.validate_test_metrics(metrics_path, dataset)
    print(f"infer finished at {output.resolve()}", flush=True)
    return {
        "cache_reused": False,
        "cache": str(output.resolve()),
        "metrics": str(metrics_path.resolve()),
        "summary": str(summary_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    try:
        result = infer_one(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
