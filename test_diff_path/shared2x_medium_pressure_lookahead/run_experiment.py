from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
FULL_DIR = TEST_DIR / "shared2x_full_objectives"
for item in (str(ROOT), str(TEST_DIR), str(FULL_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


full = load_module(
    "shared2x_medium_pressure_full_runtime", FULL_DIR / "run_experiment.py"
)
NativeHattrick = full.Hattrick


class MediumPressureHattrick(NativeHattrick):
    """Native serial Hattrick plus persistent global Medium pressure preview."""

    def __init__(self, props):
        props.future_lookahead = False
        super().__init__(props)
        self.enable_medium_pressure_lookahead()


full.Hattrick = MediumPressureHattrick
full.OUTPUT_ROOT = THIS_DIR / "artifacts"


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "ordered_projection.py": FULL_DIR / "ordered_projection.py",
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "utils/training_utils.py": ROOT / "utils" / "training_utils.py",
        "shared2x_order_regularizer/run_experiment.py": (
            TEST_DIR / "shared2x_order_regularizer" / "run_experiment.py"
        ),
    }
    return {name: full.sha256(path) for name, path in paths.items()}


full.source_hashes = source_hashes


def method_description() -> dict:
    return {
        "method": "Hattrick-LA: persistent global Medium-pressure lookahead",
        "architecture": (
            "Native High -> High+Medium -> High+Medium+Low cascade. "
            "A zero-initialized residual adapter gives the initial High head and "
            "every High RAU iteration two path features: max and mean global "
            "predicted-Medium edge utilization under a uniform feasible preview."
        ),
        "projection": "unchanged six-objective ordered gradient projection",
        "objectives": list(full.OBJECTIVE_NAMES),
        "inference_information": (
            "ESM-predicted High/Medium/Low traffic, topology, capacities, and "
            "candidate paths only; actual traffic is used only by the training "
            "admission loss and final evaluation replay"
        ),
        "zero_initialization": (
            "Both lookahead residual output layers start at zero, so epoch-0 "
            "policy is exactly native Hattrick for the same initialization."
        ),
        "source_sha256": source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared-2x Hattrick global Medium-pressure lookahead experiment"
    )
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = full.run_one(args.level, args.seed, force=args.force)
    description = method_description()
    description.update(
        {
            "level": args.level,
            "seed": args.seed,
            "run_directory": str(run_dir),
        }
    )
    full.write_json(run_dir / "lookahead_method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
