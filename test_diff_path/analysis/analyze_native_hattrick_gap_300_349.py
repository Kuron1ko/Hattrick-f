from __future__ import annotations

"""Strict-ESM 300--349 Low gap to the frozen original Hattrick checkpoint."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "test_diff_path"
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
STRICT_EVALUATOR = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "evaluate_strict_esm_sequential.py"
)
NATIVE_CHECKPOINT = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "final_model.pt"
)
OUTPUT_DIR = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention_persistent"
    / "artifacts"
    / "analysis_epoch41_esm_scale_low_guard_300_349"
)
SCALE_REPORT = OUTPUT_DIR / "summary.json"
PRECOMMIT = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention_persistent"
    / "level4_precommit_manifest.json"
)
EXPECTED_NATIVE_SHA256 = (
    "394e0fe592491f70836b92a9a17543ea53d3fe1fbfc5b3f156bef0e4c3a78622"
)
START, STOP = 300, 350


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_hashes() -> dict[str, str]:
    manifest = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    if manifest.get("test_data_read") is not False or len(manifest["files"]) != 27:
        raise RuntimeError("Unexpected precommit manifest")
    hashes = {}
    for item in manifest["files"]:
        path = Path(item["path"])
        actual = sha256(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Frozen file changed: {path}")
        hashes[str(path.resolve())] = actual
    return hashes


def main() -> None:
    if not SCALE_REPORT.exists() or (START, STOP) != (300, 350):
        raise RuntimeError("Missing scale report or split guard changed")
    before = frozen_hashes()
    native_sha = sha256(NATIVE_CHECKPOINT)
    if native_sha != EXPECTED_NATIVE_SHA256:
        raise RuntimeError("Original Hattrick checkpoint changed")

    edge = load_module("native_gap_edge", EDGE_RUNNER)
    strict = load_module("native_gap_strict", STRICT_EVALUATOR)
    runtime = edge.runtime
    runtime.set_seed(20260824)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    payload = torch.load(NATIVE_CHECKPOINT, map_location=device, weights_only=False)
    if int(payload.get("epoch", -1)) != 60:
        raise RuntimeError("Original Hattrick checkpoint is not epoch 60")
    model = runtime.Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    calls = []
    original_constructor = runtime.DM_Dataset_within_Cluster

    def guarded_constructor(current_props, mode, start, end, *args, **kwargs):
        start, end = int(start), int(end)
        if start < START or end > STOP or start >= end:
            raise RuntimeError(f"Forbidden dataset construction: [{start},{end})")
        calls.append([start, end])
        return original_constructor(current_props, mode, start, end, *args, **kwargs)

    runtime.DM_Dataset_within_Cluster = guarded_constructor
    cache, audit = strict.build_strict_policy_cache(
        runtime, model, props, START, STOP, 25, actual_input_mode="zero"
    )
    if calls != [[START, STOP]] or int(cache.dataset.max_source_index_read) != 349:
        raise RuntimeError("Split audit failed")
    rows, summary = runtime.evaluate_cache(model, props, cache, None, batch_size=25)
    indexed = runtime.summary_index(summary)
    original = {
        class_name: {
            suffix: float(indexed[class_name][f"norm_fulfill_{suffix}"])
            for suffix in ("mean", "p1", "p10")
        }
        for class_name in ("High", "Medium", "Low")
    }

    scale = json.loads(SCALE_REPORT.read_text(encoding="utf-8"))
    methods = {
        "core": scale["results"]["core"]["absolute_low"],
        "full_guard": scale["results"]["full_guard"]["absolute_low"],
        "esm_selected": scale["results"]["esm_selected"]["absolute_low"],
    }
    gaps = {
        name: {
            suffix: float(metrics[suffix]) - original["Low"][suffix]
            for suffix in ("mean", "p1", "p10")
        }
        for name, metrics in methods.items()
    }
    after = frozen_hashes()
    if after != before:
        raise RuntimeError("Frozen file changed during native comparison")
    result = {
        "scope": {
            "window": [START, STOP],
            "maximum_source_index_read": 349,
            "dataset_constructor_calls": calls,
            "forbidden_window_constructed_or_read": False,
        },
        "original_hattrick": {
            "checkpoint": str(NATIVE_CHECKPOINT.resolve()),
            "sha256": native_sha,
            "epoch": 60,
            "strict_esm_audit": audit,
            "norm_fulfill": original,
        },
        "low_gap_method_minus_original_hattrick": gaps,
        "all_27_frozen_hashes_unchanged": True,
    }
    target = OUTPUT_DIR / "original_hattrick_gap_300_349.json"
    if target.exists():
        raise RuntimeError(f"Refusing to overwrite {target}")
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
