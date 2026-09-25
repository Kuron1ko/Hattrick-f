from __future__ import annotations

"""Strict-ESM residual mirror-routing development check on snapshots 0-349."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
MODEL_RUNNER = THIS_DIR / "run_experiment.py"
CORE = THIS_DIR / "level4_selection_validation_only" / "selected_checkpoint.pt"
NATIVE = TEST_DIR / "shared2x_full_objectives" / "artifacts" / "level4_confirmation" / "seed_490" / "final_model.pt"
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
GUARD_LIBRARY = THIS_DIR / "train_low_guard.py"
STRICT = TEST_DIR / "shared2x_sparse_path_cross_attention" / "evaluate_strict_esm_sequential.py"
OUTPUT = THIS_DIR / "artifacts" / "development_low_residual_mirror_000_349.json"
K = 8

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


def exact_predicted_residual(model, props, cache, edge):
    zero_low = torch.zeros_like(cache.policies[2])
    with torch.no_grad():
        admitted = model.simulate(
            [cache.policies[0], cache.policies[1], zero_low],
            list(cache.predicted_tms),
            cache.capacities,
            edge.pte_info(cache),
            len(cache),
            props,
            rate_cap=props.rate_cap,
        )[:2]
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    loads = torch.zeros_like(cache.capacities, dtype=torch.float32)
    for ratio, demand in zip(admitted, cache.predicted_tms[:2]):
        flow = ratio.to(dtype=torch.float32) * demand.squeeze(-1).to(dtype=torch.float32)
        loads = loads + torch.sparse.mm(pte.t(), flow.t()).t()
    return (cache.capacities.to(dtype=torch.float32) - loads).clamp_min(1e-5)


def mirror_route(base, demand, residual, pte, steps: int, eta: float):
    batch = len(base)
    grouped_base = base.reshape(batch, -1, K)
    valid = grouped_base > 0
    logits = torch.log(grouped_base.clamp_min(1e-12))
    policy = grouped_base.clone()
    for _ in range(steps):
        flat = policy.reshape_as(base)
        flow = flat.to(dtype=torch.float32) * demand.to(dtype=torch.float32)
        edge_load = torch.sparse.mm(pte.t(), flow.t()).t()
        edge_price = edge_load / residual
        path_cost = torch.sparse.mm(pte, edge_price.t()).t().reshape_as(policy)
        masked = torch.where(valid, path_cost, torch.full_like(path_cost, float("nan")))
        center = torch.nanmean(masked, dim=-1, keepdim=True)
        centered = torch.where(valid, path_cost - center, torch.zeros_like(path_cost))
        scale = torch.sqrt(
            (centered.square().sum(dim=-1, keepdim=True) / valid.sum(dim=-1, keepdim=True).clamp_min(1))
            + 1e-8
        )
        logits = logits - float(eta) * centered / scale
        masked_logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
        policy = torch.softmax(masked_logits, dim=-1) * grouped_base.sum(dim=-1, keepdim=True)
        policy = torch.where(valid, policy, torch.zeros_like(policy))
    return policy.reshape_as(base)


def public(result):
    return {key: value for key, value in result.items() if key != "rows"}


def rows_in_window(rows, start, end):
    return [row for row in rows if start <= int(row["snapshot"]) < end]


def main() -> None:
    torch.manual_seed(20260824)
    model_module = load("mirror_model", MODEL_RUNNER)
    edge = load("mirror_edge", EDGE_RUNNER)
    guard = load("mirror_guard", GUARD_LIBRARY)
    strict = load("mirror_strict", STRICT)
    runtime = edge.runtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    payload = torch.load(CORE, map_location=device, weights_only=False)
    model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    native_payload = torch.load(NATIVE, map_location=device, weights_only=False)
    native = Hattrick(props).to(device=device, dtype=props.dtype)
    native.load_state_dict(native_payload["model_state_dict"], strict=True)
    native.eval()
    cache, audit = strict.build_strict_policy_cache(
        runtime, model, props, 0, 350, 25, actual_input_mode="zero"
    )
    native_cache, native_audit = strict.build_strict_policy_cache(
        runtime, native, props, 0, 350, 25, actual_input_mode="zero"
    )
    baseline_rows, baseline_summary = runtime.evaluate_cache(
        model, props, cache, None, batch_size=25
    )
    native_rows, native_summary = runtime.evaluate_cache(
        model, props, native_cache, None, batch_size=25
    )
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    residual = exact_predicted_residual(model, props, cache, edge)
    base = cache.policies[2].squeeze(-1)
    demand = cache.predicted_tms[2].squeeze(-1)
    candidates = {}
    for steps in (1, 2, 4, 8, 16):
        for eta in (0.125, 0.25, 0.5, 1.0):
            with torch.no_grad():
                low = mirror_route(base, demand, residual, pte, steps, eta)
            candidate = replace(cache, path_features=low)
            rows, summary = runtime.evaluate_cache(
                model, props, candidate, guard.LowOnlyAdapter(), batch_size=25
            )
            key = f"steps={steps},eta={eta}"
            candidates[key] = {
                "summary": runtime.summary_index(summary),
                "vs_core": edge.diagnostics(baseline_rows, rows),
                "vs_native": edge.diagnostics(native_rows, rows),
                "blocks_vs_core": {
                    f"{start}-{start + 49}": edge.diagnostics(
                        rows_in_window(baseline_rows, start, start + 50),
                        rows_in_window(rows, start, start + 50),
                    )
                    for start in range(0, 350, 50)
                },
                "blocks_vs_native": {
                    f"{start}-{start + 49}": edge.diagnostics(
                        rows_in_window(native_rows, start, start + 50),
                        rows_in_window(rows, start, start + 50),
                    )
                    for start in range(0, 350, 50)
                },
                "policy_tv_mean": float(
                    (0.5 * (low - base).abs().reshape(len(cache), -1, K).sum(dim=-1)).mean().item()
                ),
            }
    ranked = sorted(
        candidates,
        key=lambda key: (
            candidates[key]["vs_native"]["Low"]["mean_gain"],
            candidates[key]["vs_native"]["Low"]["p1_gain"],
            candidates[key]["vs_native"]["Low"]["p10_gain"],
        ),
        reverse=True,
    )
    report = {
        "status": "development-only; snapshots 350-499 not read",
        "window": [0, 350],
        "method": "strict-ESM residual-capacity entropic mirror routing for Low only",
        "information_audit": audit,
        "native_information_audit": native_audit,
        "core": runtime.summary_index(baseline_summary),
        "native": runtime.summary_index(native_summary),
        "ranked_keys": ranked,
        "candidates": candidates,
        "test_data_read": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ranked": [
        {
            "key": key,
            "vs_core": candidates[key]["vs_core"]["Low"],
            "vs_native": candidates[key]["vs_native"]["Low"],
            "worst_block_mean_vs_native": min(
                block["Low"]["mean_gain"]
                for block in candidates[key]["blocks_vs_native"].values()
            ),
        }
        for key in ranked
    ]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
