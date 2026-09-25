from __future__ import annotations

"""Dedicated entry point for the complete GEANT 3x data and Oracle workspace."""

import argparse
import importlib.util
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
RUNTIME_PATH = TEST_DIR / "full_geant_3x_hattrick" / "run_full_experiment.py"


def load_runtime():
    spec = importlib.util.spec_from_file_location(
        "full_geant_3x_workspace_runtime", RUNTIME_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import 3x workspace runtime: {RUNTIME_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runtime = load_runtime()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and audit the full 10200-snapshot GEANT 3x workspace"
    )
    parser.add_argument(
        "--stage", choices=("check", "prepare", "oracle", "all"), default="check"
    )
    parser.add_argument("--oracle-chunk-size", type=positive_int, default=256)
    parser.add_argument("--oracle-workers", type=positive_int, default=1)
    parser.add_argument("--force-prepare", action="store_true")
    args = parser.parse_args()
    # Fields consumed only by the shared read-only status reporter.
    args.epochs = runtime.DEFAULT_EPOCHS
    args.save_every = runtime.DEFAULT_SAVE_EVERY
    args.top_k = runtime.DEFAULT_TOP_K
    runtime.check(args)
    if args.stage in ("prepare", "all"):
        runtime.base.prepare_data(force=args.force_prepare)
    if args.stage in ("oracle", "all"):
        runtime.base.generate_oracle(args.oracle_chunk_size, args.oracle_workers)


if __name__ == "__main__":
    main()
