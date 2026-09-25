from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
SPARSE_DIR = TEST_DIR / "shared2x_sparse_path_cross_attention"
EDGE_DIR = TEST_DIR / "shared2x_edge_toll_head"
SPARSE_RUNNER = SPARSE_DIR / "run_experiment.py"
EDGE_RUNNER = EDGE_DIR / "run_experiment.py"
STRICT_EVALUATOR = SPARSE_DIR / "evaluate_strict_esm_sequential.py"
SELECTION_MANIFEST = (
    SPARSE_DIR
    / "checkpoint_selection_validation_only"
    / "manifest_frozen_epoch_060.json"
)
SPARSE_CHECKPOINT = (
    SPARSE_DIR
    / "checkpoint_selection_validation_only"
    / "archive"
    / "epoch_048.pt"
)
OUTPUT_DIR = THIS_DIR / "artifacts"

EXPECTED_HASHES = {
    SPARSE_CHECKPOINT: "55dfbe1506148099c838f3e5e08ca871f280877e56748ae3ea5af574c0a3acad",
    SELECTION_MANIFEST: "a481a9ff750a00a5ab8abc5a76a66e8db05c82c0cecca7d1f807437282e09d6d",
    SPARSE_RUNNER: "3b40dff9801bc71e6606888d8a690adbe2e695b52fd1a4710f6bb97b71b19f56",
    EDGE_RUNNER: "839c6a8b2a091553fe817abedc97016b35e498980a4eb2d8f936cb5cbae4c866",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> dict[str, str]:
    hashes = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {path}: {actual} != {expected}")
        hashes[str(path.resolve())] = actual
    manifest = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    selected = manifest["selection"]["selected"]
    if manifest.get("test_data_read") is not False or int(selected["epoch"]) != 48:
        raise RuntimeError("The sparse backbone was not frozen validation-only at epoch 48")
    return hashes


def public(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in ("rows", "state")}


def main() -> None:
    frozen_hashes = verify_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    strict = load_module("sparse_e48_head_train_strict", STRICT_EVALUATOR)
    edge = load_module("sparse_e48_head_train_edge", EDGE_RUNNER)
    sparse = load_module("sparse_e48_head_train_model", SPARSE_RUNNER)
    runtime = edge.runtime

    runtime.set_seed(20260824)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    payload = torch.load(SPARSE_CHECKPOINT, map_location=device, weights_only=False)
    if int(payload.get("epoch", -1)) != 48:
        raise RuntimeError("Sparse checkpoint payload is not epoch 48")
    model = sparse.SparsePathCrossAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Literal-zero actual TMs are passed to every policy forward. The returned
    # caches retain actual TMs only for differentiable admission loss/evaluation.
    train_cache, train_information_audit = strict.build_strict_policy_cache(
        runtime, model, props, 0, 318, 32, actual_input_mode="zero"
    )
    safety_cache, safety_information_audit = strict.build_strict_policy_cache(
        runtime, model, props, 318, 350, 32, actual_input_mode="zero"
    )
    validation_cache, validation_information_audit = strict.build_strict_policy_cache(
        runtime, model, props, 350, 400, 25, actual_input_mode="zero"
    )

    train_features = edge.edge_features(train_cache)
    safety_features = edge.edge_features(safety_cache)
    validation_features = edge.edge_features(validation_cache)
    train_baseline_norm = edge.baseline_normalized(model, props, train_cache)
    safety_rows, safety_summary = runtime.evaluate_cache(
        model, props, safety_cache, None, batch_size=32
    )
    validation_rows, validation_summary = runtime.evaluate_cache(
        model, props, validation_cache, None, batch_size=25
    )

    started = time.perf_counter()
    all_candidates = []
    histories = []
    settings = ((10.0, 0.25), (25.0, 0.25), (50.0, 0.5))
    for index, (safety_weight, low_reward) in enumerate(settings):
        candidates, history = edge.train_one(
            model,
            props,
            train_cache,
            train_features,
            train_baseline_norm,
            safety_cache,
            safety_features,
            safety_rows,
            safety_weight,
            low_reward,
            20260901 + index,
        )
        all_candidates.extend(candidates)
        histories.append(
            {
                "safety_weight": safety_weight,
                "low_reward": low_reward,
                "history": history,
            }
        )

    # Selection is exactly the existing Hattrick-e safety-window rule. The
    # independent validation window reports generalization but cannot re-rank.
    safety_winner = max(all_candidates, key=edge.rank)
    head = edge.EdgeTollHead(
        int(train_cache.capacities.shape[1]), int(train_features.shape[-1])
    ).to(device)
    head.load_state_dict(safety_winner["state"], strict=True)
    head.eval()
    validation_result = edge.evaluate_head(
        model,
        props,
        head,
        validation_cache,
        validation_features,
        validation_rows,
    )
    elapsed = time.perf_counter() - started

    checkpoint_path = OUTPUT_DIR / "best_edge_toll_head.pt"
    torch.save(
        {
            "state_dict": safety_winner["state"],
            "feature_count": int(train_features.shape[-1]),
            "edge_count": int(train_cache.capacities.shape[1]),
            "selection": public(safety_winner),
            "backbone_epoch": 48,
            "backbone_sha256": EXPECTED_HASHES[SPARSE_CHECKPOINT],
            "test_data_read": False,
        },
        checkpoint_path,
    )
    report = {
        "method": "sparse-e48-matched ESM edge-toll residual head",
        "status": "training/safety-selection/independent-validation only",
        "test_data_read": False,
        "protocol": {
            "train": [0, 318],
            "safety_selection": [318, 350],
            "independent_validation": [350, 400],
            "forbidden_during_training": [400, 500],
            "settings": [list(value) for value in settings],
            "selection_rule": "unchanged shared2x_edge_toll_head.rank on safety window",
            "actual_tm_online": False,
        },
        "information_audits": {
            "train": train_information_audit,
            "safety": safety_information_audit,
            "validation": validation_information_audit,
        },
        "frozen_input_hashes": frozen_hashes,
        "safety_baseline": runtime.summary_index(safety_summary),
        "safety_candidates": [public(row) for row in all_candidates],
        "safety_winner": public(safety_winner),
        "validation_baseline": runtime.summary_index(validation_summary),
        "validation": public(validation_result),
        "histories": histories,
        "runtime": {"device": str(device), "seconds": elapsed},
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
        },
    }
    runtime.write_json(OUTPUT_DIR / "training_validation.json", report)
    runtime.write_csv(OUTPUT_DIR / "validation_baseline_rows.csv", validation_rows)
    runtime.write_csv(
        OUTPUT_DIR / "validation_candidate_rows.csv", validation_result["rows"]
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
