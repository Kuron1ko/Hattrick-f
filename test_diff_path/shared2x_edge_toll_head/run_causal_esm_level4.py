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
spec = importlib.util.spec_from_file_location("causal_level4_runtime", METHOD_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {METHOD_PATH}")
method = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = method
spec.loader.exec_module(method)
edge = method.edge
runtime = method.runtime


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

    validation_payload = json.loads(
        (THIS_DIR / "causal_esm_low_validation.json").read_text(encoding="utf-8")
    )
    winner = max(
        validation_payload["validation_diagnostic_sweep"],
        key=lambda x: (
            -x["diagnostics"]["Low"]["ecdf_max_upward_violation"],
            x["diagnostics"]["Low"]["mean_gain"],
            x["diagnostics"]["Low"]["p10_gain"],
        ),
    )
    config = {
        "strength": float(winner["strength"]),
        "decay": float(winner["decay"]),
        "window": int(winner["window"]),
        "low_scale": float(winner["low_scale"]),
    }

    history = runtime.build_policy_cache(model, props, 0, edge.FINAL[0], batch_size=64)
    final = runtime.build_policy_cache(model, props, *edge.FINAL, batch_size=64)
    baseline_rows, baseline_summary = runtime.evaluate_cache(
        model, props, final, None, batch_size=32
    )
    ordinary_features = edge.edge_features(final)
    corrected_tms = method.causal_predictions(
        history,
        final,
        config["strength"],
        config["decay"],
        config["window"],
    )
    corrected_cache = replace(final, predicted_tms=corrected_tms)
    result = method.evaluate(
        model,
        props,
        head,
        final,
        ordinary_features,
        edge.edge_features(corrected_cache),
        config["low_scale"],
        baseline_rows,
    )
    payload = {
        "method": "Causal ESM-calibrated asymmetric edge-toll head",
        "level": 4,
        "range": list(edge.FINAL),
        "selection": "configuration frozen from 350-399 validation",
        "config": config,
        "current_actual_tm_used": False,
        "checkpoint": str(checkpoint),
        "baseline": runtime.summary_index(baseline_summary),
        "candidate": method.public(result),
    }
    runtime.write_json(THIS_DIR / "causal_esm_level4.json", payload)
    runtime.write_csv(THIS_DIR / "level4_baseline_rows.csv", baseline_rows)
    runtime.write_csv(THIS_DIR / "level4_candidate_rows.csv", result["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
