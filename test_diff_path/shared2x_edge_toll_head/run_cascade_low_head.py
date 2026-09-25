from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
RISK_PATH = THIS_DIR / "run_risk_controller.py"
for item in (str(THIS_DIR), str(THIS_DIR.parent), str(THIS_DIR.parent.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)
spec = importlib.util.spec_from_file_location("cascade_runtime", RISK_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {RISK_PATH}")
risk = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = risk
spec.loader.exec_module(risk)
edge = risk.edge
runtime = risk.runtime


class LowResidualHead(nn.Module):
    def __init__(self, edge_count: int, feature_count: int, max_toll: float):
        super().__init__()
        self.max_toll = float(max_toll)
        self.embedding = nn.Parameter(torch.zeros(edge_count, 6))
        self.net = nn.Sequential(
            nn.Linear(feature_count + 6, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        embedding = self.embedding.unsqueeze(0).expand(len(features), -1, -1)
        return self.max_toll * torch.tanh(
            self.net(torch.cat([features, embedding], dim=-1)).squeeze(-1)
        )


def prepare_primary(head, cache, base_features):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    tolls, _ = risk.head_outputs(head, cache, base_features)
    medium = risk.route_from_tolls(
        cache.policies[1].squeeze(-1), tolls[:, :, 0], pte
    )
    capacities = cache.capacities.to(dtype=torch.float32).clamp_min(1e-9)
    high_flow = cache.policies[0].squeeze(-1) * cache.predicted_tms[0].squeeze(-1)
    medium_flow = medium * cache.predicted_tms[1].squeeze(-1)
    high_load = torch.sparse.mm(pte.t(), high_flow.t()).t() / capacities
    medium_load = torch.sparse.mm(pte.t(), medium_flow.t()).t() / capacities
    post_medium = high_load + medium_load
    cascade_features = torch.cat(
        [
            base_features,
            medium_load.unsqueeze(-1),
            post_medium.unsqueeze(-1),
            torch.relu(1.0 - post_medium).unsqueeze(-1),
            tolls[:, :, 0:1],
            tolls[:, :, 1:2],
        ],
        dim=-1,
    ).detach()
    return medium.detach(), tolls[:, :, 1].detach(), cascade_features


def build_cache(cache, medium, primary_low_toll, residual_toll):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    low = risk.route_from_tolls(
        cache.policies[2].squeeze(-1), primary_low_toll + residual_toll, pte
    )
    return replace(cache, path_features=torch.cat([medium, low], dim=1))


def evaluate(model, props, head, cache, prepared, baseline_rows):
    medium, primary_low_toll, features = prepared
    with torch.no_grad():
        residual = head(features)
    candidate_cache = build_cache(cache, medium, primary_low_toll, residual)
    adapter = edge.TwoPolicyAdapter(int(cache.policies[0].shape[1]))
    rows, summary = runtime.evaluate_cache(
        model, props, candidate_cache, adapter, batch_size=32
    )
    baseline_by_key = {(int(r["snapshot"]), r["class"]): r for r in baseline_rows}
    for row in rows:
        if row["class"] == "High":
            row.update(baseline_by_key[(int(row["snapshot"]), "High")])
    return {
        "summary": runtime.summary_index(summary),
        "diagnostics": edge.diagnostics(baseline_rows, rows),
        "residual_abs_mean": float(residual.abs().mean().item()),
        "rows": rows,
        "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
    }


def public(result):
    return {k: v for k, v in result.items() if k not in ("rows", "state")}


def low_rank(result):
    low = result["diagnostics"]["Low"]
    return (
        -low["ecdf_max_upward_violation"],
        low["mean_gain"],
        low["p10_gain"],
        low["p1_gain"],
    )


def train_one(
    model,
    props,
    train_cache,
    train_prepared,
    train_baseline,
    safety_cache,
    safety_prepared,
    safety_rows,
    max_toll,
    safety_weight,
    seed,
):
    torch.manual_seed(seed)
    medium, primary_low_toll, features = train_prepared
    head = LowResidualHead(
        int(train_cache.capacities.shape[1]), int(features.shape[-1]), max_toll
    ).to(props.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=2e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pte = train_cache.dataset.pte.coalesce().to(dtype=torch.float32)
    candidates = []
    history = []
    for epoch in range(1, 81):
        permutation = torch.randperm(len(train_cache), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(train_cache), 32):
            cpu_indices = permutation[start : start + 32]
            indices = cpu_indices.to(props.device)
            residual = head(features.index_select(0, indices))
            low = risk.route_from_tolls(
                train_cache.policies[2].index_select(0, indices).squeeze(-1),
                primary_low_toll.index_select(0, indices) + residual,
                pte,
            )
            policies = [
                train_cache.policies[0].index_select(0, indices),
                medium.index_select(0, indices).unsqueeze(-1),
                low.unsqueeze(-1),
            ]
            candidate = edge.normalized_fulfillment(
                model, props, train_cache, indices, policies
            )[:, 2]
            baseline = train_baseline.index_select(0, indices)[:, 2]
            gain = candidate - baseline
            violation = torch.relu(0.0005 - gain)
            hard_count = max(1, int(math.ceil(len(indices) * 0.30)))
            hard_safety = torch.topk(violation, hard_count).values.mean()
            loss = (
                -gain.mean()
                + float(safety_weight) * hard_safety
                + 3e-4 * residual.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
        if epoch == 1 or epoch % 10 == 0:
            result = evaluate(
                model, props, head, safety_cache, safety_prepared, safety_rows
            )
            result["epoch"] = epoch
            result["max_toll"] = float(max_toll)
            result["safety_weight"] = float(safety_weight)
            candidates.append(result)
            history.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss / len(train_cache),
                    "low": result["diagnostics"]["Low"],
                }
            )
    return candidates, history


def main():
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    primary_saved = torch.load(THIS_DIR / "best_edge_toll_head.pt", map_location=device)
    primary = edge.EdgeTollHead(
        primary_saved["edge_count"], primary_saved["feature_count"]
    ).to(device)
    primary.load_state_dict(primary_saved["state_dict"])
    primary.eval()

    caches = {}
    rows = {}
    prepared = {}
    for name, bounds, batch_size in (
        ("train", edge.TRAIN, 64),
        ("safety", edge.SAFETY, 32),
        ("validation", edge.VALIDATION, 32),
    ):
        cache = runtime.build_policy_cache(model, props, *bounds, batch_size=batch_size)
        caches[name] = cache
        rows[name], _ = runtime.evaluate_cache(model, props, cache, None, batch_size=32)
        prepared[name] = prepare_primary(primary, cache, edge.edge_features(cache))
    train_baseline = edge.baseline_normalized(model, props, caches["train"])

    started = time.perf_counter()
    candidates = []
    histories = []
    for index, (max_toll, safety_weight) in enumerate(
        ((0.75, 5.0), (1.25, 15.0), (1.75, 40.0))
    ):
        current, history = train_one(
            model,
            props,
            caches["train"],
            prepared["train"],
            train_baseline,
            caches["safety"],
            prepared["safety"],
            rows["safety"],
            max_toll,
            safety_weight,
            20261101 + index,
        )
        candidates.extend(current)
        histories.append(
            {"max_toll": max_toll, "safety_weight": safety_weight, "history": history}
        )
    winner = max(candidates, key=low_rank)
    low_head = LowResidualHead(
        int(caches["train"].capacities.shape[1]),
        int(prepared["train"][2].shape[-1]),
        winner["max_toll"],
    ).to(device)
    low_head.load_state_dict(winner["state"])
    low_head.eval()
    validation = evaluate(
        model,
        props,
        low_head,
        caches["validation"],
        prepared["validation"],
        rows["validation"],
    )
    payload = {
        "method": "Cascaded ESM Low residual edge head",
        "architecture": "Medium edge toll frozen; Low head sees post-Medium ESM residual capacity",
        "checkpoint": str(checkpoint),
        "seconds": time.perf_counter() - started,
        "safety_candidates": [public(x) for x in candidates],
        "safety_winner": public(winner),
        "validation": public(validation),
        "histories": histories,
    }
    runtime.write_json(THIS_DIR / "cascade_low_validation.json", payload)
    runtime.write_csv(THIS_DIR / "cascade_low_validation_rows.csv", validation["rows"])
    torch.save(
        {
            "state_dict": winner["state"],
            "max_toll": winner["max_toll"],
            "feature_count": int(prepared["train"][2].shape[-1]),
            "edge_count": int(caches["train"].capacities.shape[1]),
            "selection": public(winner),
        },
        THIS_DIR / "best_cascade_low_head.pt",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
