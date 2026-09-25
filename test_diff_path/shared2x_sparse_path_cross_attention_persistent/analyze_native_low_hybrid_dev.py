from __future__ import annotations

"""Development-only native-Low hybrids on snapshots 300-349.

This analyzer never constructs a dataset containing snapshot 350 or later.
Every policy forward receives literal-zero actual-TM tensors.  Candidate
hybrids always reuse the frozen persistent core's High and Medium policy
tensors verbatim and replace only Low.
"""

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
CORE_CHECKPOINT = THIS_DIR / "level4_selection_validation_only" / "selected_checkpoint.pt"
NATIVE_CHECKPOINT = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "final_model.pt"
)
MODEL_RUNNER = THIS_DIR / "run_experiment.py"
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
STRICT_RUNNER = (
    TEST_DIR / "shared2x_sparse_path_cross_attention" / "evaluate_strict_esm_sequential.py"
)
OUTPUT = THIS_DIR / "artifacts" / "development_native_low_hybrid_300_349.json"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LowOnlyAdapter:
    def adapt_batch(self, policies, batch):
        policies[2] = batch["path_features"].unsqueeze(-1)
        return policies


def evaluate(runtime, edge, model, props, core_cache, low, baseline_rows):
    candidate_cache = replace(core_cache, path_features=low.squeeze(-1))
    rows, summary = runtime.evaluate_cache(
        model, props, candidate_cache, LowOnlyAdapter(), batch_size=25
    )
    return {
        "summary": runtime.summary_index(summary),
        "vs_core": edge.diagnostics(baseline_rows, rows),
        "rows": rows,
    }


def transplant_native_stage3(core_model, native_state):
    copied = []
    own = core_model.state_dict()
    prefixes = ("mlp31.", "mlp32.", "mlp_32_violation.")
    with torch.no_grad():
        for name, value in native_state.items():
            if name.startswith(prefixes):
                if name not in own or own[name].shape != value.shape:
                    raise RuntimeError(f"Incompatible Stage-3 parameter: {name}")
                own[name].copy_(value.to(device=own[name].device, dtype=own[name].dtype))
                copied.append(name)
    if not any(name.startswith("mlp31.") for name in copied):
        raise RuntimeError("No mlp31 parameters copied")
    if not any(name.startswith("mlp32.") for name in copied):
        raise RuntimeError("No mlp32 parameters copied")
    return copied


def policy_exact(core_cache, candidate_cache):
    return {
        "High": bool(torch.equal(core_cache.policies[0], candidate_cache.policies[0])),
        "Medium": bool(torch.equal(core_cache.policies[1], candidate_cache.policies[1])),
    }


def strip_rows(value):
    return {key: item for key, item in value.items() if key != "rows"}


def main():
    torch.manual_seed(20260824)
    model_module = load("native_low_hybrid_model", MODEL_RUNNER)
    edge = load("native_low_hybrid_edge", EDGE_RUNNER)
    strict = load("native_low_hybrid_strict", STRICT_RUNNER)
    runtime = edge.runtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)

    core_payload = torch.load(CORE_CHECKPOINT, map_location=device, weights_only=False)
    core_model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    core_model.load_state_dict(core_payload["model_state_dict"], strict=True)
    core_model.eval()

    native_payload = torch.load(NATIVE_CHECKPOINT, map_location=device, weights_only=False)
    native_model = runtime.Hattrick(props).to(device=device, dtype=props.dtype)
    native_model.load_state_dict(native_payload["model_state_dict"], strict=True)
    native_model.eval()

    started = time.perf_counter()
    core_cache, core_audit = strict.build_strict_policy_cache(
        runtime, core_model, props, 300, 350, 25, actual_input_mode="zero"
    )
    core_seconds = time.perf_counter() - started
    core_rows, core_summary = runtime.evaluate_cache(
        core_model, props, core_cache, None, batch_size=25
    )

    started = time.perf_counter()
    native_cache, native_audit = strict.build_strict_policy_cache(
        runtime, native_model, props, 300, 350, 25, actual_input_mode="zero"
    )
    native_seconds = time.perf_counter() - started
    native_rows, native_summary = runtime.evaluate_cache(
        native_model, props, native_cache, None, batch_size=25
    )

    # Full independent native forward, but only its Low tensor is consumed.
    independent = evaluate(
        runtime,
        edge,
        core_model,
        props,
        core_cache,
        native_cache.policies[2],
        core_rows,
    )

    # One-forward alternative: keep the persistent core, replace only its
    # native Stage-3 modules, and again consume only the resulting Low tensor.
    transplanted_model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    transplanted_model.load_state_dict(core_payload["model_state_dict"], strict=True)
    copied = transplant_native_stage3(
        transplanted_model, native_payload["model_state_dict"]
    )
    transplanted_model.eval()
    started = time.perf_counter()
    transplanted_cache, transplanted_audit = strict.build_strict_policy_cache(
        runtime, transplanted_model, props, 300, 350, 25, actual_input_mode="zero"
    )
    transplanted_seconds = time.perf_counter() - started
    transplanted = evaluate(
        runtime,
        edge,
        core_model,
        props,
        core_cache,
        transplanted_cache.policies[2],
        core_rows,
    )

    # Explicitly construct hybrid caches and prove that upstream tensors are
    # byte-identical to the core policy, not merely close after replay.
    independent_cache = replace(
        core_cache,
        policies=(core_cache.policies[0], core_cache.policies[1], native_cache.policies[2]),
    )
    transplanted_hybrid_cache = replace(
        core_cache,
        policies=(
            core_cache.policies[0],
            core_cache.policies[1],
            transplanted_cache.policies[2],
        ),
    )

    result = {
        "status": "development-only; no snapshot >=350 constructed",
        "window": [300, 350],
        "strict_esm": True,
        "core": runtime.summary_index(core_summary),
        "native": runtime.summary_index(native_summary),
        "core_vs_native": edge.diagnostics(native_rows, core_rows),
        "independent_native_low_hybrid": strip_rows(independent),
        "independent_native_low_hybrid_vs_native": edge.diagnostics(
            native_rows, independent["rows"]
        ),
        "native_stage3_transplant_low_hybrid": strip_rows(transplanted),
        "native_stage3_transplant_low_hybrid_vs_native": edge.diagnostics(
            native_rows, transplanted["rows"]
        ),
        "upstream_policy_torch_equal": {
            "independent_native_low": policy_exact(core_cache, independent_cache),
            "native_stage3_transplant": policy_exact(core_cache, transplanted_hybrid_cache),
        },
        "information_audits": {
            "core": core_audit,
            "native": native_audit,
            "native_stage3_transplant": transplanted_audit,
        },
        "stage3_parameter_tensors_copied": len(copied),
        "policy_cache_seconds": {
            "core": core_seconds,
            "native": native_seconds,
            "stage3_transplant": transplanted_seconds,
            "independent_hybrid_estimated_total": core_seconds + native_seconds,
            "independent_hybrid_ratio_to_core": (core_seconds + native_seconds)
            / max(core_seconds, 1e-9),
            "stage3_transplant_ratio_to_core": transplanted_seconds
            / max(core_seconds, 1e-9),
        },
        "test_data_read": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
