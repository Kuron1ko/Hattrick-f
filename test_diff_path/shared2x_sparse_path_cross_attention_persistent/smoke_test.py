from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
PERSISTENT_RUNNER = THIS_DIR / "run_experiment.py"
STRICT_EVALUATOR = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "evaluate_strict_esm_sequential.py"
)
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
CHECKPOINT = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "checkpoint_selection_validation_only"
    / "archive"
    / "epoch_048.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "55dfbe1506148099c838f3e5e08ca871f280877e56748ae3ea5af574c0a3acad"
)


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


def main() -> None:
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Sparse epoch-48 checkpoint hash mismatch")
    persistent = load_module("persistent_stage2_smoke_model", PERSISTENT_RUNNER)
    strict = load_module("persistent_stage2_smoke_strict", STRICT_EVALUATOR)
    edge = load_module("persistent_stage2_smoke_runtime", EDGE_RUNNER)
    runtime = edge.runtime

    torch.set_num_threads(1)
    device = torch.device("cpu")
    props = runtime.build_props(4, device)
    payload = torch.load(CHECKPOINT, map_location=device, weights_only=False)

    parent_model = persistent.parent.SparsePathCrossAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    parent_load = parent_model.load_state_dict(payload["model_state_dict"], strict=True)
    parent_model.eval()

    candidate_model = persistent.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    candidate_load = candidate_model.load_state_dict(
        payload["model_state_dict"], strict=False
    )
    expected_missing = {
        f"{name}.{suffix}"
        for name in ("stage2_high_init_adapter", "stage2_high_rau_adapter")
        for suffix in (
            "0.weight",
            "0.bias",
            "2.weight",
            "2.bias",
        )
    }
    if set(candidate_load.missing_keys) != expected_missing:
        raise RuntimeError(
            f"Unexpected candidate missing keys: {candidate_load.missing_keys}"
        )
    if candidate_load.unexpected_keys:
        raise RuntimeError(
            f"Unexpected candidate state keys: {candidate_load.unexpected_keys}"
        )
    candidate_model.eval()

    parent_cache, parent_audit = strict.build_strict_policy_cache(
        runtime, parent_model, props, 32, 33, 1, actual_input_mode="zero"
    )
    candidate_cache, candidate_audit = strict.build_strict_policy_cache(
        runtime, candidate_model, props, 32, 33, 1, actual_input_mode="zero"
    )
    policy_audit = strict.policy_difference(
        parent_cache.policies, candidate_cache.policies
    )
    if any(not value["torch_equal"] for value in policy_audit.values()):
        raise RuntimeError(f"Zero-output candidate changed parent policy: {policy_audit}")
    if candidate_model.last_stage2_context_audit.get("site") != "mlp22":
        raise RuntimeError("Formal Stage-2 context injection did not execute")

    result = {
        "status": "passed",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "parent_state_strict": {
            "missing": list(parent_load.missing_keys),
            "unexpected": list(parent_load.unexpected_keys),
        },
        "candidate_missing_keys_exact": sorted(expected_missing),
        "zero_output_policy_equivalence": policy_audit,
        "parent_information_audit": parent_audit,
        "candidate_information_audit": candidate_audit,
        "stage2_context_audit": candidate_model.last_stage2_context_audit,
        "source_sha256": {
            "run_experiment.py": sha256(PERSISTENT_RUNNER),
            "smoke_test.py": sha256(Path(__file__).resolve()),
            "strict_evaluator": sha256(STRICT_EVALUATOR),
        },
    }
    output = THIS_DIR / "smoke_test.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
