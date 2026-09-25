from __future__ import annotations

"""Run persistent Stage-2 training without opening the 400-499 test window."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BASE_RUNNER = THIS_DIR / "run_experiment.py"
PRECOMMIT = THIS_DIR / "level4_precommit_manifest.json"
EXPECTED_BASE_SHA256 = (
    "c4061d15495d4d5ecfdf4e3ca7dd77fa0a64a732bfb0c1c1d0ceefb61a1df14f"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base():
    if sha256(BASE_RUNNER) != EXPECTED_BASE_SHA256:
        raise RuntimeError("Persistent base runner changed")
    spec = importlib.util.spec_from_file_location(
        "persistent_stage2_level4_validation_only_base", BASE_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Level-4 persistent Stage-2 training with validation-only evaluation"
    )
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    base = load_base()

    original_spec = dict(base.parent.full.shared.LEVELS[4])
    if tuple(original_spec["train"]) != (0, 350):
        raise RuntimeError(f"Unexpected Level-4 train split: {original_spec['train']}")
    if tuple(original_spec["validation"]) != (350, 400):
        raise RuntimeError(
            f"Unexpected Level-4 validation split: {original_spec['validation']}"
        )
    # The generic runner normally evaluates 400-499 after epoch 60.  Replace
    # only that reporting split with validation, so neither model construction,
    # training, checkpoint ranking nor automatic final reporting opens test.
    validation_only_spec = dict(original_spec)
    validation_only_spec["evaluation"] = (350, 400)
    base.parent.full.shared.LEVELS[4] = validation_only_spec

    def source_hashes() -> dict[str, str]:
        if not PRECOMMIT.exists():
            raise RuntimeError("Level-4 precommit manifest is missing")
        values = base.source_hashes()
        values["run_level4_validation_only.py"] = sha256(Path(__file__).resolve())
        values["level4_precommit_manifest.json"] = sha256(PRECOMMIT)
        protocol = json.dumps(
            {
                "train": [0, 350],
                "validation": [350, 400],
                "automatic_evaluation": [350, 400],
                "forbidden_test": [400, 500],
                "test_data_read": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        values["level4_validation_only/protocol"] = hashlib.sha256(
            protocol
        ).hexdigest()
        return values

    base.parent.full.source_hashes = source_hashes
    run_dir = base.parent.full.run_one(4, args.seed, force=args.force)
    description = base.method_description()
    description.update(
        {
            "level": 4,
            "seed": args.seed,
            "run_directory": str(run_dir),
            "training_split": [0, 350],
            "validation_and_automatic_reporting_split": [350, 400],
            "forbidden_test_split": [400, 500],
            "test_data_read": False,
            "selection": "external validation-only immutable watcher",
            "source_sha256": source_hashes(),
        }
    )
    base.parent.full.write_json(run_dir / "method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
