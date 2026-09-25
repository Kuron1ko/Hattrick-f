from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
CASCADE_PATH = THIS_DIR / "run_cascade_low_head.py"
for item in (str(THIS_DIR), str(THIS_DIR.parent), str(THIS_DIR.parent.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)
spec = importlib.util.spec_from_file_location("global_cdf_runtime", CASCADE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {CASCADE_PATH}")
cascade = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cascade
spec.loader.exec_module(cascade)
edge = cascade.edge
risk = cascade.risk
runtime = cascade.runtime


class GlobalLowCDFHead(nn.Module):
    def __init__(self, edge_count: int, max_toll: float):
        super().__init__()
        self.max_toll = float(max_toll)
        self.raw = nn.Parameter(torch.zeros(edge_count))

    def forward(self, features):
        residual = self.max_toll * torch.tanh(self.raw)
        return residual.unsqueeze(0).expand(len(features), -1)


def public(result):
    return {k: v for k, v in result.items() if k not in ("rows", "state")}


def low_rank(result):
    low = result["diagnostics"]["Low"]
    return (
        -low["ecdf_max_upward_violation"],
        low["p1_gain"],
        low["p10_gain"],
        low["mean_gain"],
        -result["residual_abs_mean"],
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
    cdf_weight,
    seed,
):
    torch.manual_seed(seed)
    medium, primary_low_toll, features = train_prepared
    head = GlobalLowCDFHead(int(train_cache.capacities.shape[1]), max_toll).to(props.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.005, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pte = train_cache.dataset.pte.coalesce().to(dtype=torch.float32)
    candidates = []
    history = []
    for epoch in range(1, 101):
        permutation = torch.randperm(len(train_cache), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(train_cache), 48):
            cpu_indices = permutation[start : start + 48]
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
            sorted_candidate = torch.sort(candidate).values
            sorted_baseline = torch.sort(baseline).values
            shortfall = torch.relu(sorted_baseline + 0.0003 - sorted_candidate)
            tail_count = max(1, int(math.ceil(len(indices) * 0.4)))
            distribution_safety = (
                shortfall.mean()
                + torch.topk(shortfall, tail_count).values.mean()
                + shortfall.max()
            )
            gain = candidate.mean() - baseline.mean()
            residual_vector = head.raw.tanh() * float(max_toll)
            loss = (
                -gain
                + float(cdf_weight) * distribution_safety
                + 5e-4 * residual_vector.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
        if epoch == 1 or epoch % 10 == 0:
            result = cascade.evaluate(
                model, props, head, safety_cache, safety_prepared, safety_rows
            )
            result["epoch"] = epoch
            result["max_toll"] = float(max_toll)
            result["cdf_weight"] = float(cdf_weight)
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
        prepared[name] = cascade.prepare_primary(primary, cache, edge.edge_features(cache))
    train_baseline = edge.baseline_normalized(model, props, caches["train"])

    started = time.perf_counter()
    candidates = []
    histories = []
    for index, (max_toll, cdf_weight) in enumerate(
        ((0.20, 5.0), (0.40, 15.0), (0.75, 40.0))
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
            cdf_weight,
            20261201 + index,
        )
        candidates.extend(current)
        histories.append(
            {"max_toll": max_toll, "cdf_weight": cdf_weight, "history": history}
        )
    winner = max(candidates, key=low_rank)
    head = GlobalLowCDFHead(
        int(caches["train"].capacities.shape[1]), winner["max_toll"]
    ).to(device)
    head.load_state_dict(winner["state"])
    head.eval()
    validation = cascade.evaluate(
        model,
        props,
        head,
        caches["validation"],
        prepared["validation"],
        rows["validation"],
    )
    payload = {
        "method": "Global 72-edge Low correction with order-statistic CDF loss",
        "architecture": "72 shared topology parameters; Medium policy frozen",
        "checkpoint": str(checkpoint),
        "seconds": time.perf_counter() - started,
        "safety_candidates": [public(x) for x in candidates],
        "safety_winner": public(winner),
        "validation": public(validation),
        "histories": histories,
    }
    runtime.write_json(THIS_DIR / "global_low_cdf_validation.json", payload)
    runtime.write_csv(THIS_DIR / "global_low_cdf_validation_rows.csv", validation["rows"])
    torch.save(
        {
            "state_dict": winner["state"],
            "max_toll": winner["max_toll"],
            "selection": public(winner),
        },
        THIS_DIR / "best_global_low_cdf_head.pt",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
