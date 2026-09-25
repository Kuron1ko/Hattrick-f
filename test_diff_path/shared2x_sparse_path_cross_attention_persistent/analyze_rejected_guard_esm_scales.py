from __future__ import annotations

"""Development-only ESM scale analysis on snapshots 300-349.

Never constructs 350-499.  Actual TM is used only to score policies after each
policy and its per-snapshot scale have been chosen from ESM predictions.
"""

import importlib.util
import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
CORE = THIS_DIR / "level4_selection_validation_only" / "selected_checkpoint.pt"
HEAD = THIS_DIR / "artifacts" / "level4_low_guard_frozen" / "frozen_low_guard_prevalidation.pt"
MODEL_RUNNER = THIS_DIR / "run_experiment.py"
GUARD_LIBRARY = THIS_DIR / "train_low_guard.py"
EDGE_RUNNER = THIS_DIR.parent / "shared2x_edge_toll_head" / "run_experiment.py"
STRICT = (
    THIS_DIR.parent
    / "shared2x_sparse_path_cross_attention"
    / "evaluate_strict_esm_sequential.py"
)
OUTPUT = THIS_DIR / "artifacts" / "development_rejected_guard_esm_scales.json"
SCALES = (0.0, 0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def predicted_low_score(model, props, cache, low: torch.Tensor, edge, batch_size=25):
    values = []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            stop = min(start + batch_size, len(cache))
            indices = torch.arange(start, stop, device=props.device)
            batch = edge.runtime.select_cache(cache, indices)
            predicted_tms = tuple(
                value.index_select(0, indices) for value in cache.predicted_tms
            )
            policies = [
                batch["policies"][0],
                batch["policies"][1],
                low.index_select(0, indices).unsqueeze(-1),
            ]
            ratios = model.simulate(
                policies,
                list(predicted_tms),
                batch["capacities"],
                edge.pte_info(cache),
                len(indices),
                props,
                rate_cap=props.rate_cap,
            )[:3]
            predicted = predicted_tms[2].squeeze(-1)
            admitted = ratios[2].reshape(len(indices), -1) * predicted
            values.append(admitted.sum(dim=1) / predicted.sum(dim=1).clamp_min(1e-9))
    return torch.cat(values)


def evaluate_low(edge, guard, model, props, cache, low, baseline_rows):
    candidate = replace(cache, path_features=low)
    rows, summary = edge.runtime.evaluate_cache(
        model, props, candidate, guard.LowOnlyAdapter(), batch_size=25
    )
    return {
        "summary": edge.runtime.summary_index(summary),
        "diagnostics": edge.diagnostics(baseline_rows, rows),
        "rows": rows,
    }


def low_values(rows):
    return np.asarray(
        [float(row["norm_fulfill"]) for row in rows if row["class"] == "Low"],
        dtype=np.float64,
    )


def public(value):
    return {key: item for key, item in value.items() if key != "rows"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, default=CORE)
    parser.add_argument("--head", type=Path, default=HEAD)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--start", type=int, default=300)
    parser.add_argument("--end", type=int, default=350)
    args = parser.parse_args()
    if not (0 <= args.start < args.end <= 350):
        raise RuntimeError("Development analyzer is code-limited to snapshots 0-349")
    torch.manual_seed(20260824)
    model_module = load("dev_scale_model", MODEL_RUNNER)
    guard = load("dev_scale_guard", GUARD_LIBRARY)
    edge = load("dev_scale_edge", EDGE_RUNNER)
    strict = load("dev_scale_strict", STRICT)
    runtime = edge.runtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    core_payload = torch.load(args.core, map_location=device, weights_only=False)
    model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    model.load_state_dict(core_payload["model_state_dict"], strict=True)
    model.eval()
    head_payload = torch.load(args.head, map_location=device, weights_only=False)
    head = guard.LowGuard(
        int(head_payload["edge_count"]),
        int(head_payload["feature_count"]),
        float(head_payload["max_toll"]),
    ).to(device)
    head.load_state_dict(head_payload["state_dict"], strict=True)
    head.eval()

    cache, information_audit = strict.build_strict_policy_cache(
        runtime, model, props, args.start, args.end, 25, actual_input_mode="zero"
    )
    features = edge.edge_features(cache)
    baseline_rows, baseline_summary = runtime.evaluate_cache(
        model, props, cache, None, batch_size=25
    )
    with torch.no_grad():
        tolls = head(features)
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    base_low = cache.policies[2].squeeze(-1)

    lows = []
    predicted_scores = []
    fixed = {}
    for scale in SCALES:
        low = guard.route_low(base_low, float(scale) * tolls, pte)
        score = predicted_low_score(model, props, cache, low, edge)
        evaluation = evaluate_low(edge, guard, model, props, cache, low, baseline_rows)
        lows.append(low)
        predicted_scores.append(score)
        fixed[str(scale)] = evaluation

    score_matrix = torch.stack(predicted_scores, dim=1)
    low_stack = torch.stack(lows, dim=1)
    best_index = score_matrix.argmax(dim=1)
    batch_index = torch.arange(len(cache), device=device)
    selected_low = low_stack[batch_index, best_index]
    selected = evaluate_low(
        edge, guard, model, props, cache, selected_low, baseline_rows
    )

    base_actual = low_values(baseline_rows)
    full_actual = low_values(fixed["1.0"]["rows"])
    selected_actual = low_values(selected["rows"])
    predicted_gain = (
        score_matrix[:, -1] - score_matrix[:, 0]
    ).detach().cpu().numpy()
    actual_gain = full_actual - base_actual
    correlation = float(np.corrcoef(predicted_gain, actual_gain)[0, 1])
    scale_counts = {
        str(scale): int((best_index == index).sum().item())
        for index, scale in enumerate(SCALES)
    }
    selected_scales = np.asarray([SCALES[index] for index in best_index.cpu().tolist()])
    report = {
        "status": "development-only; 350-499 not read",
        "window": [args.start, args.end],
        "core_checkpoint": str(args.core.resolve()),
        "guard_checkpoint": str(args.head.resolve()),
        "scales": list(SCALES),
        "selection": "per-snapshot argmax predicted Low admitted fraction",
        "information_audit": information_audit,
        "baseline": runtime.summary_index(baseline_summary),
        "fixed_scales": {key: public(value) for key, value in fixed.items()},
        "esm_selected": public(selected),
        "selected_scale_counts": scale_counts,
        "selected_scale_mean": float(selected_scales.mean()),
        "full_scale_predicted_vs_actual_gain_correlation": correlation,
        "full_scale_predicted_gain_min_mean_max": [
            float(predicted_gain.min()),
            float(predicted_gain.mean()),
            float(predicted_gain.max()),
        ],
        "full_scale_actual_gain_min_mean_max": [
            float(actual_gain.min()),
            float(actual_gain.mean()),
            float(actual_gain.max()),
        ],
        "selected_actual_gain_min_mean_max": [
            float((selected_actual - base_actual).min()),
            float((selected_actual - base_actual).mean()),
            float((selected_actual - base_actual).max()),
        ],
        "test_data_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
