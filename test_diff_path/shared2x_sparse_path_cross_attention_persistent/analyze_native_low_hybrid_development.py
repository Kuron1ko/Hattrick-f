from __future__ import annotations

"""Development-only strict-ESM hybrid check on snapshots below 350.

The candidate keeps the frozen persistent core's High and Medium policies and
substitutes only the native Hattrick Low policy.  Actual traffic is retained
solely for offline sequential-admission scoring.
"""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
CORE_RUNNER = THIS_DIR / "run_experiment.py"
CORE_CHECKPOINT = THIS_DIR / "level4_selection_validation_only" / "selected_checkpoint.pt"
NATIVE_CHECKPOINT = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "final_model.pt"
)
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
GUARD_LIBRARY = THIS_DIR / "train_low_guard.py"
STRICT_EVALUATOR = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "evaluate_strict_esm_sequential.py"
)
OUTPUT = THIS_DIR / "artifacts" / "development_native_low_hybrid_300_349.json"

for item in (str(ROOT), str(TEST_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

from frameworks.hattrick_system import Hattrick


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    torch.manual_seed(20260824)
    core_module = load("hybrid_core_model", CORE_RUNNER)
    edge = load("hybrid_edge", EDGE_RUNNER)
    guard = load("hybrid_guard", GUARD_LIBRARY)
    strict = load("hybrid_strict", STRICT_EVALUATOR)
    runtime = edge.runtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)

    core_payload = torch.load(CORE_CHECKPOINT, map_location=device, weights_only=False)
    core = core_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    core.load_state_dict(core_payload["model_state_dict"], strict=True)
    core.eval()

    native_payload = torch.load(NATIVE_CHECKPOINT, map_location=device, weights_only=False)
    native = Hattrick(props).to(device=device, dtype=props.dtype)
    native.load_state_dict(native_payload["model_state_dict"], strict=True)
    native.eval()

    core_cache, core_audit = strict.build_strict_policy_cache(
        runtime, core, props, 300, 350, 25, actual_input_mode="zero"
    )
    native_cache, native_audit = strict.build_strict_policy_cache(
        runtime, native, props, 300, 350, 25, actual_input_mode="zero"
    )
    for left, right in zip(core_cache.predicted_tms, native_cache.predicted_tms):
        if not torch.equal(left, right):
            raise RuntimeError("Core and native caches do not share identical ESM inputs")
    core_rows, core_summary = runtime.evaluate_cache(
        core, props, core_cache, None, batch_size=25
    )
    native_rows, native_summary = runtime.evaluate_cache(
        core, props, native_cache, None, batch_size=25
    )
    hybrid_cache = replace(
        core_cache, path_features=native_cache.policies[2].squeeze(-1)
    )
    hybrid_rows, hybrid_summary = runtime.evaluate_cache(
        core, props, hybrid_cache, guard.LowOnlyAdapter(), batch_size=25
    )

    report = {
        "status": "development-only; snapshots 350-499 not read",
        "window": [300, 350],
        "method": "persistent High/Medium plus native-Hattrick strict-ESM Low",
        "actual_tm_policy_input": False,
        "core_information_audit": core_audit,
        "native_information_audit": native_audit,
        "core": runtime.summary_index(core_summary),
        "native": runtime.summary_index(native_summary),
        "hybrid": runtime.summary_index(hybrid_summary),
        "hybrid_vs_core": edge.diagnostics(core_rows, hybrid_rows),
        "hybrid_vs_native": edge.diagnostics(native_rows, hybrid_rows),
        "high_policy_torch_equal": bool(
            torch.equal(core_cache.policies[0], hybrid_cache.policies[0])
        ),
        "medium_policy_torch_equal": bool(
            torch.equal(core_cache.policies[1], hybrid_cache.policies[1])
        ),
        "test_data_read": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
