from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
RISK_PATH = THIS_DIR / "run_risk_controller.py"
for item in (str(THIS_DIR), str(THIS_DIR.parent), str(THIS_DIR.parent.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)
spec = importlib.util.spec_from_file_location("risk_runtime", RISK_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {RISK_PATH}")
risk = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = risk
spec.loader.exec_module(risk)
edge = risk.edge
runtime = risk.runtime


def scaled_cache(cache, tolls, low_scale):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    routed = []
    for output_index, class_index in enumerate((1, 2)):
        base = cache.policies[class_index].squeeze(-1)
        scale = 1.0 if output_index == 0 else float(low_scale)
        routed.append(risk.route_from_tolls(base, tolls[:, :, output_index] * scale, pte))
    return replace(cache, path_features=torch.cat(routed, dim=1))


def evaluate(model, props, cache, tolls, scale, baseline_rows):
    adapted = scaled_cache(cache, tolls, scale)
    adapter = edge.TwoPolicyAdapter(int(cache.policies[0].shape[1]))
    rows, summary = runtime.evaluate_cache(model, props, adapted, adapter, batch_size=32)
    baseline_by_key = {(int(r["snapshot"]), r["class"]): r for r in baseline_rows}
    for row in rows:
        if row["class"] == "High":
            row.update(baseline_by_key[(int(row["snapshot"]), "High")])
    return {
        "scale": float(scale),
        "summary": runtime.summary_index(summary),
        "diagnostics": edge.diagnostics(baseline_rows, rows),
        "rows": rows,
    }


def public(result):
    return {k: v for k, v in result.items() if k != "rows"}


def low_rank(result):
    low = result["diagnostics"]["Low"]
    return (
        -low["ecdf_max_upward_violation"],
        low["mean_gain"],
        low["p10_gain"],
        low["p1_gain"],
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

    data = {}
    for name, bounds in (("safety", edge.SAFETY), ("validation", edge.VALIDATION)):
        cache = runtime.build_policy_cache(model, props, *bounds, batch_size=32)
        baseline_rows, _ = runtime.evaluate_cache(model, props, cache, None, batch_size=32)
        features = edge.edge_features(cache)
        tolls, _ = risk.head_outputs(head, cache, features)
        data[name] = (cache, tolls, baseline_rows)

    scales = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
    safety_cache, safety_tolls, safety_rows = data["safety"]
    safety = [
        evaluate(model, props, safety_cache, safety_tolls, scale, safety_rows)
        for scale in scales
    ]
    winner = max(safety, key=low_rank)
    validation_cache, validation_tolls, validation_rows = data["validation"]
    validation = evaluate(
        model, props, validation_cache, validation_tolls, winner["scale"], validation_rows
    )
    diagnostic_validation = [
        evaluate(model, props, validation_cache, validation_tolls, scale, validation_rows)
        for scale in scales
    ]
    payload = {
        "method": "Low-only response scaling",
        "checkpoint": str(checkpoint),
        "selection": "scale selected only on safety set",
        "safety_sweep": [public(x) for x in safety],
        "safety_winner": public(winner),
        "validation": public(validation),
        "validation_diagnostic_sweep": [public(x) for x in diagnostic_validation],
    }
    runtime.write_json(THIS_DIR / "low_response_validation.json", payload)
    runtime.write_csv(THIS_DIR / "low_response_validation_rows.csv", validation["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
