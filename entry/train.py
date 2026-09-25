from __future__ import annotations

import argparse
import importlib

import _workspace as ws


def parse_args() -> tuple[argparse.Namespace, object]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--dataset", required=True)
    bootstrap.add_argument("--method", required=True)
    known, _unknown = bootstrap.parse_known_args()
    try:
        dataset = ws.canonical_dataset(known.dataset)
        method = ws.canonical_method(known.method)
    except ValueError as error:
        bootstrap.error(str(error))
    module_name = (
        "method.hattrick_system" if method == "Hattrick"
        else "method.hattrick_f3_system"
    )
    method_module = importlib.import_module(module_name)
    parser = argparse.ArgumentParser(
        description=f"Train or automatically resume {method} on {dataset}"
    )
    parser.add_argument("--dataset", required=True, choices=tuple(ws.DATASETS))
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=490)
    method_module.register_arguments(parser)
    args = parser.parse_args()
    args.dataset = dataset
    args.method = method
    if args.epochs <= 0 or args.top_k <= 0 or args.top_k > args.epochs:
        parser.error("--epochs and --top-k must be positive, and top-k <= epochs")
    if args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("--batch-size and --learning-rate must be positive")
    return args, method_module


def main() -> None:
    args, method_module = parse_args()
    try:
        method_module.run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
