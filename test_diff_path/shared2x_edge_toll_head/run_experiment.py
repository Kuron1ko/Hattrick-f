from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
RUNTIME_PATH = TEST_DIR / "shared2x_medium_adapter" / "run_experiment.py"
for item in (str(ROOT), str(TEST_DIR), str(RUNTIME_PATH.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)
spec = importlib.util.spec_from_file_location("edge_toll_runtime", RUNTIME_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {RUNTIME_PATH}")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


TRAIN = (0, 318)
SAFETY = (318, 350)
VALIDATION = (350, 400)
FINAL = (400, 500)
K = 8


class TwoPolicyAdapter:
    def __init__(self, path_count: int):
        self.path_count = int(path_count)

    def adapt_batch(self, policies, batch):
        features = batch["path_features"]
        policies[1] = features[:, : self.path_count].unsqueeze(-1)
        policies[2] = features[:, self.path_count :].unsqueeze(-1)
        return policies


class EdgeTollHead(nn.Module):
    def __init__(self, edge_count: int, feature_count: int, max_toll: float = 1.25):
        super().__init__()
        self.edge_count = int(edge_count)
        self.max_toll = float(max_toll)
        self.edge_embedding = nn.Parameter(torch.zeros(edge_count, 6))
        self.local = nn.Sequential(
            nn.Linear(feature_count + 6, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 2),
        )
        self.gate = nn.Sequential(
            nn.Linear(feature_count * 2, 16),
            nn.SiLU(),
            nn.Linear(16, 2),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -1.5)

    def forward(self, edge_features, base_policies, pte):
        batch = edge_features.shape[0]
        embedding = self.edge_embedding.unsqueeze(0).expand(batch, -1, -1)
        local_input = torch.cat([edge_features, embedding], dim=-1)
        tolls = self.max_toll * torch.tanh(self.local(local_input))
        pooled = torch.cat(
            [edge_features.mean(dim=1), edge_features.amax(dim=1)], dim=-1
        )
        gates = torch.sigmoid(self.gate(pooled))
        tolls = tolls * gates.unsqueeze(1)
        policies = []
        for class_index, base in enumerate(base_policies):
            path_cost = torch.sparse.mm(
                pte, tolls[:, :, class_index].transpose(0, 1)
            ).transpose(0, 1)
            grouped_base = base.reshape(batch, -1, K)
            grouped_cost = path_cost.reshape(batch, -1, K)
            valid = grouped_base > 0
            logits = torch.log(grouped_base.clamp_min(1e-12)) - grouped_cost
            logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
            routed = torch.softmax(logits, dim=-1) * grouped_base.sum(
                dim=-1, keepdim=True
            )
            routed = torch.where(valid, routed, torch.zeros_like(routed))
            policies.append(routed.reshape_as(base))
        return policies, tolls, gates


def edge_features(cache):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    capacities = cache.capacities.to(dtype=torch.float32).clamp_min(1e-9)
    loads = []
    for policy, demand in zip(cache.policies, cache.predicted_tms):
        flow = policy.squeeze(-1).to(dtype=torch.float32) * demand.squeeze(-1).to(
            dtype=torch.float32
        )
        loads.append(torch.sparse.mm(pte.t(), flow.t()).t() / capacities)
    high = loads[0]
    medium = loads[1]
    low = loads[2]
    return torch.stack(
        [
            high,
            medium,
            low,
            high + medium,
            high + medium + low,
            torch.relu(1.0 - high),
            torch.relu(1.0 - high - medium),
        ],
        dim=-1,
    ).detach()


def pte_info(cache):
    pte = cache.dataset.pte.coalesce()
    indices = pte.indices()
    return pte, indices[0], indices[1], pte.values()


def normalized_fulfillment(model, props, cache, indices, policies):
    batch = runtime.select_cache(cache, indices)
    ratios = model.simulate(
        policies,
        list(batch["tms"]),
        batch["capacities"],
        pte_info(cache),
        int(indices.numel()),
        props,
        rate_cap=props.rate_cap,
    )[:3]
    admitted = [
        ratio.reshape(indices.numel(), -1) * tm.squeeze(-1)
        for ratio, tm in zip(ratios, batch["tms"])
    ]
    return torch.stack(
        [
            flow.sum(dim=1) / oracle.clamp_min(1e-9)
            for flow, oracle in zip(admitted, batch["oracle_flows"])
        ],
        dim=1,
    )


def baseline_normalized(model, props, cache, batch_size=32):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            indices = torch.arange(
                start, min(start + batch_size, len(cache)), device=props.device
            )
            policies = [value.index_select(0, indices) for value in cache.policies]
            chunks.append(normalized_fulfillment(model, props, cache, indices, policies))
    return torch.cat(chunks, dim=0)


def build_candidate_cache(head, cache, features, batch_size=32):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    medium = []
    low = []
    gates = []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            stop = min(start + batch_size, len(cache))
            indices = torch.arange(start, stop, device=features.device)
            base = [
                cache.policies[class_index].index_select(0, indices).squeeze(-1)
                for class_index in (1, 2)
            ]
            policies, _, batch_gates = head(
                features.index_select(0, indices), base, pte
            )
            medium.append(policies[0])
            low.append(policies[1])
            gates.append(batch_gates)
    path_features = torch.cat(
        [torch.cat(medium, dim=0), torch.cat(low, dim=0)], dim=1
    )
    return replace(cache, path_features=path_features), torch.cat(gates, dim=0)


def class_values(rows, class_name):
    return np.asarray(
        [float(row["norm_fulfill"]) for row in rows if row["class"] == class_name],
        dtype=np.float64,
    )


def ecdf_violation(baseline, candidate):
    baseline = np.sort(np.asarray(baseline))
    candidate = np.sort(np.asarray(candidate))
    grid = np.unique(np.concatenate([baseline, candidate]))
    base_cdf = np.searchsorted(baseline, grid, side="right") / len(baseline)
    cand_cdf = np.searchsorted(candidate, grid, side="right") / len(candidate)
    return float(np.max(cand_cdf - base_cdf))


def diagnostics(baseline_rows, candidate_rows):
    result = {}
    for class_name in ("High", "Medium", "Low"):
        baseline = class_values(baseline_rows, class_name)
        candidate = class_values(candidate_rows, class_name)
        delta = candidate - baseline
        result[class_name] = {
            "mean_gain": float(delta.mean()),
            "p1_gain": float(np.quantile(candidate, 0.01) - np.quantile(baseline, 0.01)),
            "p10_gain": float(np.quantile(candidate, 0.10) - np.quantile(baseline, 0.10)),
            "paired_negative_count": int(np.sum(delta < -1e-7)),
            "ecdf_max_upward_violation": ecdf_violation(baseline, candidate),
        }
    result["all_cdfs_noninferior"] = all(
        result[name]["ecdf_max_upward_violation"] <= 1e-12
        for name in ("High", "Medium", "Low")
    )
    return result


def evaluate_head(model, props, head, cache, features, baseline_rows):
    candidate_cache, gates = build_candidate_cache(head, cache, features)
    adapter = TwoPolicyAdapter(int(cache.policies[0].shape[1]))
    rows, summary = runtime.evaluate_cache(
        model, props, candidate_cache, adapter, batch_size=32
    )
    baseline_by_key = {
        (int(row["snapshot"]), row["class"]): row for row in baseline_rows
    }
    for row in rows:
        if row["class"] == "High":
            row.update(baseline_by_key[(int(row["snapshot"]), "High")])
    return {
        "summary": runtime.summary_index(summary),
        "diagnostics": diagnostics(baseline_rows, rows),
        "gate_mean_medium": float(gates[:, 0].mean().item()),
        "gate_mean_low": float(gates[:, 1].mean().item()),
        "rows": rows,
        "state": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
    }


def public(result):
    return {key: value for key, value in result.items() if key not in ("rows", "state")}


def rank(result):
    diag = result["diagnostics"]
    worst = max(
        diag["Medium"]["ecdf_max_upward_violation"],
        diag["Low"]["ecdf_max_upward_violation"],
    )
    return (
        int(diag["all_cdfs_noninferior"]),
        -worst,
        diag["Medium"]["mean_gain"],
        diag["Medium"]["p10_gain"],
        diag["Low"]["mean_gain"],
    )


def train_one(
    model,
    props,
    train_cache,
    train_features,
    train_baseline_norm,
    safety_cache,
    safety_features,
    safety_rows,
    safety_weight,
    low_reward,
    seed,
):
    torch.manual_seed(seed)
    edge_count = int(train_cache.capacities.shape[1])
    head = EdgeTollHead(edge_count, int(train_features.shape[-1])).to(props.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=1e-4)
    pte = train_cache.dataset.pte.coalesce().to(dtype=torch.float32)
    history = []
    candidates = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = 32
    for epoch in range(1, 81):
        permutation = torch.randperm(len(train_cache), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(train_cache), batch_size):
            cpu_indices = permutation[start : start + batch_size]
            indices = cpu_indices.to(props.device)
            base = [
                train_cache.policies[class_index]
                .index_select(0, indices)
                .squeeze(-1)
                for class_index in (1, 2)
            ]
            routed, tolls, gates = head(
                train_features.index_select(0, indices), base, pte
            )
            policies = [
                train_cache.policies[0].index_select(0, indices),
                routed[0].unsqueeze(-1),
                routed[1].unsqueeze(-1),
            ]
            candidate = normalized_fulfillment(
                model, props, train_cache, indices, policies
            )
            baseline = train_baseline_norm.index_select(0, indices)
            medium_gain = candidate[:, 1] - baseline[:, 1]
            low_gain = candidate[:, 2] - baseline[:, 2]
            medium_violation = torch.relu(0.0002 - medium_gain)
            low_violation = torch.relu(0.0002 - low_gain)
            hard_count = max(1, int(math.ceil(float(indices.numel()) * 0.25)))
            safety = (
                torch.topk(medium_violation, hard_count).values.mean()
                + 1.5 * torch.topk(low_violation, hard_count).values.mean()
            )
            utility = medium_gain + float(low_reward) * low_gain
            regularizer = 2e-4 * tolls.square().mean() + 2e-4 * gates.mean()
            loss = -utility.mean() + float(safety_weight) * safety + regularizer
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * int(indices.numel())
        if epoch % 10 == 0 or epoch == 1:
            result = evaluate_head(
                model, props, head, safety_cache, safety_features, safety_rows
            )
            result["epoch"] = epoch
            result["safety_weight"] = float(safety_weight)
            result["low_reward"] = float(low_reward)
            candidates.append(result)
            history.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss / len(train_cache),
                    "diagnostics": result["diagnostics"],
                }
            )
    return candidates, history


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_cache = runtime.build_policy_cache(model, props, *TRAIN, batch_size=64)
    safety_cache = runtime.build_policy_cache(model, props, *SAFETY, batch_size=32)
    validation_cache = runtime.build_policy_cache(model, props, *VALIDATION, batch_size=32)
    train_features = edge_features(train_cache)
    safety_features = edge_features(safety_cache)
    validation_features = edge_features(validation_cache)
    train_baseline_norm = baseline_normalized(model, props, train_cache)
    safety_rows, safety_summary = runtime.evaluate_cache(
        model, props, safety_cache, None, batch_size=32
    )
    validation_rows, validation_summary = runtime.evaluate_cache(
        model, props, validation_cache, None, batch_size=32
    )

    started = time.perf_counter()
    all_candidates = []
    histories = []
    for index, (safety_weight, low_reward) in enumerate(
        ((10.0, 0.25), (25.0, 0.25), (50.0, 0.5))
    ):
        candidates, history = train_one(
            model,
            props,
            train_cache,
            train_features,
            train_baseline_norm,
            safety_cache,
            safety_features,
            safety_rows,
            safety_weight,
            low_reward,
            20260901 + index,
        )
        all_candidates.extend(candidates)
        histories.append(
            {
                "safety_weight": safety_weight,
                "low_reward": low_reward,
                "history": history,
            }
        )
    safety_winner = max(all_candidates, key=rank)
    head = EdgeTollHead(
        int(train_cache.capacities.shape[1]), int(train_features.shape[-1])
    ).to(device)
    head.load_state_dict(safety_winner["state"])
    validation_result = evaluate_head(
        model,
        props,
        head,
        validation_cache,
        validation_features,
        validation_rows,
    )
    elapsed = time.perf_counter() - started
    payload = {
        "method": "ESM-conditioned edge-toll residual head",
        "architecture": {
            "edge_features": int(train_features.shape[-1]),
            "edge_count": int(train_cache.capacities.shape[1]),
            "outputs": "Medium and Low edge tolls plus two abstention gates",
            "high_policy": "unchanged Hattrick",
        },
        "protocol": {
            "train": list(TRAIN),
            "safety_selection": list(SAFETY),
            "independent_validation": list(VALIDATION),
            "final": list(FINAL),
            "actual_tm_online": False,
        },
        "checkpoint": str(checkpoint),
        "seconds": elapsed,
        "safety_baseline": runtime.summary_index(safety_summary),
        "safety_candidates": [public(row) for row in all_candidates],
        "safety_winner": public(safety_winner),
        "validation_baseline": runtime.summary_index(validation_summary),
        "validation": public(validation_result),
        "histories": histories,
    }
    runtime.write_json(THIS_DIR / "level3_training_validation.json", payload)
    runtime.write_csv(THIS_DIR / "validation_baseline_rows.csv", validation_rows)
    runtime.write_csv(
        THIS_DIR / "validation_candidate_rows.csv", validation_result["rows"]
    )
    torch.save(
        {
            "state_dict": safety_winner["state"],
            "feature_count": int(train_features.shape[-1]),
            "edge_count": int(train_cache.capacities.shape[1]),
            "selection": public(safety_winner),
        },
        THIS_DIR / "best_edge_toll_head.pt",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
