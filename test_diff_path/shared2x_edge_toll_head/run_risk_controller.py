from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
HEAD_PATH = THIS_DIR / "run_experiment.py"
for item in (str(THIS_DIR), str(THIS_DIR.parent), str(THIS_DIR.parent.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)
spec = importlib.util.spec_from_file_location("edge_toll_head_runtime", HEAD_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {HEAD_PATH}")
edge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = edge
spec.loader.exec_module(edge)
runtime = edge.runtime


class RiskNet(nn.Module):
    def __init__(self, input_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_count, 96),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 2),
        )

    def forward(self, features):
        return self.net(features)


def route_from_tolls(base, toll, pte):
    batch = int(base.shape[0])
    path_cost = torch.sparse.mm(pte, toll.transpose(0, 1)).transpose(0, 1)
    grouped_base = base.reshape(batch, -1, edge.K)
    grouped_cost = path_cost.reshape(batch, -1, edge.K)
    valid = grouped_base > 0
    logits = torch.log(grouped_base.clamp_min(1e-12)) - grouped_cost
    logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
    routed = torch.softmax(logits, dim=-1) * grouped_base.sum(dim=-1, keepdim=True)
    return torch.where(valid, routed, torch.zeros_like(routed)).reshape_as(base)


def head_outputs(head, cache, features, batch_size=32):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    toll_chunks = []
    gate_chunks = []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            stop = min(start + batch_size, len(cache))
            indices = torch.arange(start, stop, device=features.device)
            base = [
                cache.policies[c].index_select(0, indices).squeeze(-1)
                for c in (1, 2)
            ]
            _, tolls, gates = head(features.index_select(0, indices), base, pte)
            toll_chunks.append(tolls)
            gate_chunks.append(gates)
    return torch.cat(toll_chunks, dim=0), torch.cat(gate_chunks, dim=0)


def controller_features(edge_features, tolls, gates):
    # Preserve edge identity, then add robust global summaries. All are ESM-visible.
    flat = edge_features.flatten(1)
    stats = [
        edge_features.mean(dim=1),
        edge_features.std(dim=1),
        edge_features.amin(dim=1),
        edge_features.amax(dim=1),
    ]
    quantiles = torch.quantile(
        edge_features, torch.tensor([0.1, 0.5, 0.9], device=edge_features.device), dim=1
    ).permute(1, 0, 2).flatten(1)
    return torch.cat([flat, *stats, quantiles, tolls.flatten(1), gates], dim=1)


def candidate_cache(cache, tolls, accept):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    amplitude = accept.to(dtype=tolls.dtype).reshape(-1, 1)
    routed = []
    for output_index, class_index in enumerate((1, 2)):
        base = cache.policies[class_index].squeeze(-1)
        routed.append(
            route_from_tolls(base, tolls[:, :, output_index] * amplitude, pte)
        )
    path_features = torch.cat(routed, dim=1)
    return replace(cache, path_features=path_features)


def evaluate_mask(model, props, cache, tolls, accept, baseline_rows):
    selected_cache = candidate_cache(cache, tolls, accept)
    adapter = edge.TwoPolicyAdapter(int(cache.policies[0].shape[1]))
    rows, summary = runtime.evaluate_cache(
        model, props, selected_cache, adapter, batch_size=32
    )
    baseline_by_key = {
        (int(row["snapshot"]), row["class"]): row for row in baseline_rows
    }
    for row in rows:
        if row["class"] == "High":
            row.update(baseline_by_key[(int(row["snapshot"]), "High")])
    return {
        "summary": runtime.summary_index(summary),
        "diagnostics": edge.diagnostics(baseline_rows, rows),
        "coverage": float(accept.float().mean().item()),
        "accepted": [bool(value) for value in accept.detach().cpu().tolist()],
        "rows": rows,
    }


def gain_labels(model, props, cache, tolls, baseline_rows):
    accepted = torch.ones(len(cache), dtype=torch.bool, device=props.device)
    result = evaluate_mask(model, props, cache, tolls, accepted, baseline_rows)
    labels = []
    for class_name in ("Medium", "Low"):
        base = edge.class_values(baseline_rows, class_name)
        cand = edge.class_values(result["rows"], class_name)
        labels.append(torch.tensor(cand - base, device=props.device, dtype=torch.float32))
    return torch.stack(labels, dim=1), result


def fit_ensemble(features, labels, seeds):
    mean = features.mean(dim=0, keepdim=True)
    scale = features.std(dim=0, keepdim=True).clamp_min(1e-4)
    x = ((features - mean) / scale).clamp(-8.0, 8.0)
    label_scale = labels.std(dim=0, keepdim=True).clamp_min(0.002)
    y = labels / label_scale
    states = []
    n = len(features)
    for seed in seeds:
        torch.manual_seed(seed)
        net = RiskNet(int(features.shape[1])).to(features.device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.003)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bootstrap = torch.randint(0, n, (n,), generator=generator).to(features.device)
        for epoch in range(180):
            permutation = bootstrap[torch.randperm(n, generator=generator).to(features.device)]
            for start in range(0, n, 48):
                indices = permutation[start : start + 48]
                prediction = net(x.index_select(0, indices))
                target = y.index_select(0, indices)
                regression = nn.functional.smooth_l1_loss(prediction, target)
                sign_target = (target >= 0).to(dtype=prediction.dtype)
                classification = nn.functional.binary_cross_entropy_with_logits(
                    prediction * 2.0, sign_target
                )
                loss = regression + 0.2 * classification
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        states.append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    return states, mean, scale, label_scale


def predict_ensemble(features, states, mean, scale, label_scale):
    x = ((features - mean) / scale).clamp(-8.0, 8.0)
    predictions = []
    with torch.no_grad():
        for state in states:
            net = RiskNet(int(features.shape[1])).to(features.device)
            net.load_state_dict(state)
            net.eval()
            predictions.append(net(x) * label_scale)
    return torch.stack(predictions, dim=0)


def public(result):
    return {k: v for k, v in result.items() if k != "rows"}


def selection_rank(result):
    diag = result["diagnostics"]
    worst = max(
        diag["Medium"]["ecdf_max_upward_violation"],
        diag["Low"]["ecdf_max_upward_violation"],
    )
    utility = diag["Medium"]["mean_gain"] + 0.25 * diag["Low"]["mean_gain"]
    return (
        int(diag["all_cdfs_noninferior"]),
        -worst,
        utility,
        diag["Medium"]["p10_gain"],
        result["coverage"],
    )


def main():
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    caches = {
        "train": runtime.build_policy_cache(model, props, *edge.TRAIN, batch_size=64),
        "safety": runtime.build_policy_cache(model, props, *edge.SAFETY, batch_size=32),
        "validation": runtime.build_policy_cache(model, props, *edge.VALIDATION, batch_size=32),
    }
    rows = {}
    features = {}
    for name, cache in caches.items():
        rows[name], _ = runtime.evaluate_cache(model, props, cache, None, batch_size=32)
        features[name] = edge.edge_features(cache)

    saved = torch.load(THIS_DIR / "best_edge_toll_head.pt", map_location=device)
    head = edge.EdgeTollHead(saved["edge_count"], saved["feature_count"]).to(device)
    head.load_state_dict(saved["state_dict"])
    head.eval()
    tolls = {}
    control_x = {}
    for name, cache in caches.items():
        tolls[name], gates = head_outputs(head, cache, features[name])
        control_x[name] = controller_features(features[name], tolls[name], gates)

    train_labels, train_full = gain_labels(
        model, props, caches["train"], tolls["train"], rows["train"]
    )
    safety_labels, safety_full = gain_labels(
        model, props, caches["safety"], tolls["safety"], rows["safety"]
    )
    started = time.perf_counter()
    states, mean, scale, label_scale = fit_ensemble(
        control_x["train"], train_labels, range(20261001, 20261010)
    )
    safety_predictions = predict_ensemble(
        control_x["safety"], states, mean, scale, label_scale
    )
    candidates = []
    # Ensemble lower-confidence score; beta and threshold are calibrated only on safety.
    for beta in (0.0, 0.5, 1.0, 1.5, 2.0):
        lower = safety_predictions.mean(dim=0) - beta * safety_predictions.std(dim=0)
        score = lower.amin(dim=1)
        quantiles = torch.linspace(0.0, 1.0, 33, device=device)
        thresholds = torch.unique(torch.quantile(score, quantiles))
        thresholds = torch.cat([thresholds, score.max().reshape(1) + 1e-8])
        for threshold in thresholds:
            accept = score >= threshold
            result = evaluate_mask(
                model, props, caches["safety"], tolls["safety"], accept, rows["safety"]
            )
            result["beta"] = float(beta)
            result["threshold"] = float(threshold.item())
            candidates.append(result)
    winner = max(candidates, key=selection_rank)

    validation_predictions = predict_ensemble(
        control_x["validation"], states, mean, scale, label_scale
    )
    validation_lower = validation_predictions.mean(dim=0) - winner["beta"] * validation_predictions.std(dim=0)
    validation_score = validation_lower.amin(dim=1)
    validation_accept = validation_score >= winner["threshold"]
    validation = evaluate_mask(
        model,
        props,
        caches["validation"],
        tolls["validation"],
        validation_accept,
        rows["validation"],
    )
    validation["beta"] = winner["beta"]
    validation["threshold"] = winner["threshold"]

    payload = {
        "method": "ESM edge-toll head with ensemble risk rejection",
        "protocol": {
            "risk_train": list(edge.TRAIN),
            "threshold_selection": list(edge.SAFETY),
            "independent_validation": list(edge.VALIDATION),
            "actual_tm_online": False,
        },
        "checkpoint": str(checkpoint),
        "seconds": time.perf_counter() - started,
        "train_full": public(train_full),
        "safety_full": public(safety_full),
        "safety_winner": public(winner),
        "validation": public(validation),
    }
    runtime.write_json(THIS_DIR / "risk_controller_validation.json", payload)
    runtime.write_csv(THIS_DIR / "risk_validation_baseline_rows.csv", rows["validation"])
    runtime.write_csv(THIS_DIR / "risk_validation_candidate_rows.csv", validation["rows"])
    torch.save(
        {
            "states": states,
            "mean": mean.detach().cpu(),
            "scale": scale.detach().cpu(),
            "label_scale": label_scale.detach().cpu(),
            "beta": winner["beta"],
            "threshold": winner["threshold"],
        },
        THIS_DIR / "best_risk_controller.pt",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
