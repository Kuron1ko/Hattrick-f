from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
METHOD_PATH = THIS_DIR / "run_causal_esm_low.py"
for item in (str(THIS_DIR), str(THIS_DIR.parent), str(THIS_DIR.parent.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)
spec = importlib.util.spec_from_file_location("joint_calibration_runtime", METHOD_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {METHOD_PATH}")
method = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = method
spec.loader.exec_module(method)
edge = method.edge
runtime = method.runtime


def public(result):
    return {k: v for k, v in result.items() if k != "rows"}


def rank(result):
    diag = result["diagnostics"]
    worst = max(
        diag["Medium"]["ecdf_max_upward_violation"],
        diag["Low"]["ecdf_max_upward_violation"],
    )
    total = (
        diag["Medium"]["ecdf_max_upward_violation"]
        + diag["Low"]["ecdf_max_upward_violation"]
    )
    return (
        -worst,
        -total,
        diag["Medium"]["mean_gain"],
        diag["Low"]["mean_gain"],
        diag["Medium"]["p10_gain"],
    )


def main():
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    saved = torch.load(THIS_DIR / "best_edge_toll_head.pt", map_location=device)
    head = edge.EdgeTollHead(saved["edge_count"], saved["feature_count"]).to(device)
    head.load_state_dict(saved["state_dict"])
    head.eval()

    history = runtime.build_policy_cache(model, props, 0, edge.SAFETY[0], batch_size=64)
    development = runtime.build_policy_cache(
        model, props, edge.SAFETY[0], edge.VALIDATION[1], batch_size=64
    )
    dev_rows, _ = runtime.evaluate_cache(model, props, development, None, batch_size=32)
    ordinary_dev = edge.edge_features(development)

    causal_configs = [
        (strength, decay, window)
        for strength in (0.5, 0.75, 1.0)
        for decay in (0.8, 0.95)
        for window in (8, 32)
    ]
    cached_corrected = {}
    for strength, decay, window in causal_configs:
        corrected_tms = method.causal_predictions(
            history, development, strength, decay, window
        )
        cached_corrected[(strength, decay, window)] = edge.edge_features(
            replace(development, predicted_tms=corrected_tms)
        )

    candidates = []
    for strength, decay, window in causal_configs:
        corrected_features = cached_corrected[(strength, decay, window)]
        for medium_scale in (0.55, 0.70, 0.85, 1.0, 1.15):
            for low_scale in (1.0, 1.1, 1.2, 1.3, 1.4):
                result = method.evaluate(
                    model,
                    props,
                    head,
                    development,
                    ordinary_dev,
                    corrected_features,
                    low_scale,
                    dev_rows,
                    medium_scale=medium_scale,
                )
                result.update(
                    {
                        "strength": strength,
                        "decay": decay,
                        "window": window,
                        "medium_scale": medium_scale,
                        "low_scale": low_scale,
                    }
                )
                candidates.append(result)
    winner = max(candidates, key=rank)

    final_history = runtime.build_policy_cache(model, props, 0, edge.FINAL[0], batch_size=64)
    final = runtime.build_policy_cache(model, props, *edge.FINAL, batch_size=64)
    final_rows, final_summary = runtime.evaluate_cache(
        model, props, final, None, batch_size=32
    )
    corrected_final_tms = method.causal_predictions(
        final_history,
        final,
        winner["strength"],
        winner["decay"],
        winner["window"],
    )
    final_result = method.evaluate(
        model,
        props,
        head,
        final,
        edge.edge_features(final),
        edge.edge_features(replace(final, predicted_tms=corrected_final_tms)),
        winner["low_scale"],
        final_rows,
        medium_scale=winner["medium_scale"],
    )
    payload = {
        "method": "Causal ESM-calibrated asymmetric edge-toll head",
        "architecture": {
            "high": "unchanged Hattrick",
            "medium": "ESM edge toll with calibrated response",
            "low": "causally error-corrected ESM edge toll with calibrated response",
        },
        "development_range": [edge.SAFETY[0], edge.VALIDATION[1]],
        "level4_range": list(edge.FINAL),
        "current_actual_tm_used": False,
        "checkpoint": str(checkpoint),
        "development_winner": public(winner),
        "level4_baseline": runtime.summary_index(final_summary),
        "level4_candidate": public(final_result),
    }
    runtime.write_json(THIS_DIR / "joint_calibrated_level4.json", payload)
    runtime.write_csv(THIS_DIR / "joint_level4_baseline_rows.csv", final_rows)
    runtime.write_csv(THIS_DIR / "joint_level4_candidate_rows.csv", final_result["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
