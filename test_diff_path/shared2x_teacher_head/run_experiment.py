from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
RUNTIME_DIR = TEST_DIR / "shared2x_medium_adapter"
SANDWICH_DIR = TEST_DIR / "shared2x_sandwich_flow"
for item in (str(ROOT), str(TEST_DIR), str(RUNTIME_DIR), str(SANDWICH_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

spec = importlib.util.spec_from_file_location(
    "shared2x_teacher_head_runtime", RUNTIME_DIR / "run_experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load frozen-policy runtime")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)

from planner import SandwichMediumPlanner, od_incidence


K = 8
OUTPUT_ROOT = THIS_DIR / "artifacts"
LEVELS = {
    1: {"label": "level1_eight_samples", "train": (0, 160), "validation": (160, 200), "evaluation": (200, 208), "epochs": 20, "backbone_level": 2},
    2: {"label": "level2_proxy", "train": (0, 160), "validation": (160, 200), "evaluation": (200, 250), "epochs": 20, "backbone_level": 2},
    3: {"label": "level3_validation", "train": (0, 300), "validation": (300, 350), "evaluation": (350, 400), "epochs": 30, "backbone_level": 4},
    4: {"label": "level4_confirmation", "train": (0, 300), "validation": (300, 350), "evaluation": (400, 500), "epochs": 30, "backbone_level": 4},
}


class TeacherMediumHead(nn.Module):
    """Small ESM-conditioned Medium-only admission and routing head."""

    def __init__(self, num_pairs: int, feature_dim: int = 9, hidden_dim: int = 24):
        super().__init__()
        self.num_pairs = int(num_pairs)
        self.pair_embedding = nn.Embedding(num_pairs, 8)
        self.rank_embedding = nn.Embedding(K, 4)
        self.network = nn.Sequential(
            nn.Linear(feature_dim + 12, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        base_policy: torch.Tensor,
        features: torch.Tensor,
        enabled: torch.Tensor | None,
    ) -> torch.Tensor:
        flat = base_policy.squeeze(-1)
        batch, path_count = flat.shape
        pair_ids = torch.arange(self.num_pairs, device=flat.device).repeat_interleave(K)
        rank_ids = torch.arange(K, device=flat.device).repeat(self.num_pairs)
        learned = torch.cat(
            [
                self.pair_embedding(pair_ids).unsqueeze(0).expand(batch, -1, -1),
                self.rank_embedding(rank_ids).unsqueeze(0).expand(batch, -1, -1),
            ],
            dim=-1,
        )
        output = self.network(torch.cat([features, learned], dim=-1)).reshape(
            batch, self.num_pairs, K
        )
        base = flat.reshape(batch, self.num_pairs, K).clamp_min(1e-12)
        route_logits = base.log() + output
        if enabled is not None:
            mask = enabled.reshape(1, self.num_pairs, K)
            route_logits = route_logits.masked_fill(~mask, torch.finfo(route_logits.dtype).min)
        route = torch.softmax(route_logits, dim=-1)
        base_mass = flat.reshape(batch, self.num_pairs, K).sum(dim=-1, keepdim=True)
        policy = (base_mass * route).reshape(batch, path_count)
        return policy.unsqueeze(-1)


def build_planner(cache: runtime.PolicyCache) -> SandwichMediumPlanner:
    pte = cache.dataset.pte.coalesce().cpu()
    indices = pte.indices().numpy()
    values = pte.values().numpy().astype(np.float64, copy=False)
    path_count, edge_count = pte.shape
    links = sparse.coo_matrix(
        (values, (indices[1], indices[0])), shape=(edge_count, path_count)
    ).tocsr()
    return SandwichMediumPlanner(links, od_incidence(path_count // K, K))


def baseline_admitted(model, props, cache: runtime.PolicyCache) -> tuple[torch.Tensor, ...]:
    buckets = [[], [], []]
    for offset in range(0, len(cache), 16):
        indices = torch.arange(offset, min(offset + 16, len(cache)), device=props.device)
        batch = runtime.select_cache(cache, indices)
        admitted, _ = runtime.simulate_admission(model, props, cache.dataset, batch, None)
        for class_index in range(3):
            buckets[class_index].append(admitted[class_index].detach())
    return tuple(torch.cat(values, dim=0) for values in buckets)


def build_teacher(model, props, cache: runtime.PolicyCache) -> torch.Tensor:
    planner = build_planner(cache)
    admitted = baseline_admitted(model, props, cache)
    medium_mask = None
    if cache.path_masks is not None:
        medium_mask = cache.path_masks[1].reshape(-1).detach().cpu().numpy().astype(bool)
    policies = []
    for sample in range(len(cache)):
        demand = (
            cache.tms[1][sample].reshape(planner.num_pairs, K)[:, 0]
            .detach().cpu().numpy().astype(np.float64)
        )
        plan = planner.solve(
            cache.capacities[sample].detach().cpu().numpy(),
            admitted[0][sample].detach().cpu().numpy(),
            admitted[2][sample].detach().cpu().numpy(),
            demand,
            medium_mask,
        )
        denominator = np.repeat(demand, K)
        policy = np.divide(
            plan.path_flow,
            denominator,
            out=np.zeros_like(plan.path_flow),
            where=denominator > 1e-12,
        )
        policies.append(policy)
    return torch.as_tensor(
        np.stack(policies), device=props.device, dtype=props.dtype
    ).unsqueeze(-1)


def add_features(cache: runtime.PolicyCache) -> runtime.PolicyCache:
    return runtime.add_dynamic_path_features(cache)


def adapt_cache(
    head: TeacherMediumHead,
    cache: runtime.PolicyCache,
) -> runtime.PolicyCache:
    enabled = None if cache.path_masks is None else cache.path_masks[1].reshape(-1)
    outputs = []
    head.eval()
    with torch.no_grad():
        for offset in range(0, len(cache), 32):
            stop = min(offset + 32, len(cache))
            outputs.append(
                head(
                    cache.policies[1][offset:stop],
                    cache.path_features[offset:stop],
                    enabled,
                )
            )
    return replace(
        cache,
        policies=(cache.policies[0], torch.cat(outputs, dim=0), cache.policies[2]),
    )


def train_epoch(
    head: TeacherMediumHead,
    optimizer: torch.optim.Optimizer,
    cache: runtime.PolicyCache,
    teacher: torch.Tensor,
    batch_size: int = 32,
) -> float:
    head.train()
    enabled = None if cache.path_masks is None else cache.path_masks[1].reshape(-1)
    order = torch.randperm(len(cache), device=teacher.device)
    total = 0.0
    batches = 0
    for offset in range(0, len(cache), batch_size):
        ids = order[offset : offset + batch_size]
        prediction = head(
            cache.policies[1].index_select(0, ids),
            cache.path_features.index_select(0, ids),
            enabled,
        ).squeeze(-1).reshape(-1, head.num_pairs, K)
        target = teacher.index_select(0, ids).squeeze(-1).reshape(-1, head.num_pairs, K)
        base = cache.policies[1].index_select(0, ids).squeeze(-1).reshape(-1, head.num_pairs, K)
        changed = ((target - base).abs().sum(dim=-1) > 1e-4).to(prediction.dtype)
        target_mass = target.sum(dim=-1)
        prediction_mass = prediction.sum(dim=-1)
        target_route = target / target_mass.unsqueeze(-1).clamp_min(1e-9)
        prediction_route = prediction / prediction_mass.unsqueeze(-1).clamp_min(1e-9)
        route_kl = (
            target_route.clamp_min(1e-9)
            * (target_route.clamp_min(1e-9).log() - prediction_route.clamp_min(1e-9).log())
        ).sum(dim=-1)
        weight = changed + 0.05
        loss = (route_kl * weight).sum() / weight.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach().item())
        batches += 1
    return total / max(batches, 1)


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def gaps(candidate: list[dict], baseline: list[dict]) -> dict[str, float]:
    c = summary_index(candidate)
    b = summary_index(baseline)
    return {
        f"{name.lower()}_{metric}_gap": float(c[name][f"norm_fulfill_{metric}"])
        - float(b[name][f"norm_fulfill_{metric}"])
        for name in runtime.CLASSES
        for metric in ("mean", "p1", "p10")
    }


def rank(summary: list[dict], baseline: list[dict]) -> tuple:
    delta = gaps(summary, baseline)
    high = summary_index(summary)["High"]["norm_fulfill_mean"]
    safe = high >= 0.995 and delta["low_mean_gap"] >= -0.003 and delta["low_p10_gap"] >= -0.01
    return (
        int(safe),
        min(delta["medium_mean_gap"], delta["medium_p1_gap"], delta["medium_p10_gap"]),
        delta["medium_mean_gap"],
    )


def safe_remove(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise RuntimeError(f"Unsafe removal target: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def run_one(level: int, seed: int, force: bool) -> Path:
    setting = LEVELS[level]
    run_dir = OUTPUT_ROOT / setting["label"] / f"seed_{seed}"
    if force:
        safe_remove(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(int(setting["backbone_level"]), device)
    model, checkpoint = runtime.load_backbone(int(setting["backbone_level"]), 490, props, device)
    train = add_features(runtime.build_policy_cache(model, props, *setting["train"], batch_size=16))
    validation = add_features(runtime.build_policy_cache(model, props, *setting["validation"], batch_size=16))
    evaluation = add_features(runtime.build_policy_cache(model, props, *setting["evaluation"], batch_size=16))
    teacher = build_teacher(model, props, train)
    head = TeacherMediumHead(train.policies[1].shape[1] // K).to(device=device, dtype=props.dtype)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    validation_rows_base, validation_summary_base = runtime.evaluate_cache(model, props, validation, None)
    best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    _, initial_validation_summary = runtime.evaluate_cache(
        model, props, adapt_cache(head, validation), None
    )
    best_rank = rank(initial_validation_summary, validation_summary_base)
    history = [{"epoch": 0, "loss": None, "rank": list(best_rank), "gaps": gaps(initial_validation_summary, validation_summary_base)}]
    for epoch in range(1, int(setting["epochs"]) + 1):
        loss = train_epoch(head, optimizer, train, teacher)
        candidate = adapt_cache(head, validation)
        _, summary = runtime.evaluate_cache(model, props, candidate, None)
        current_rank = rank(summary, validation_summary_base)
        history.append({"epoch": epoch, "loss": loss, "rank": list(current_rank), "gaps": gaps(summary, validation_summary_base)})
        if current_rank > best_rank:
            best_rank = current_rank
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    baseline_rows, baseline_summary = runtime.evaluate_cache(model, props, evaluation, None)
    candidate_cache = adapt_cache(head, evaluation)
    candidate_rows, candidate_summary = runtime.evaluate_cache(model, props, candidate_cache, None)
    delta = gaps(candidate_summary, baseline_summary)
    feasible = (
        summary_index(candidate_summary)["High"]["norm_fulfill_mean"] >= 0.995
        and abs(delta["high_mean_gap"]) <= 1e-6
        and delta["medium_mean_gap"] > 0
        and delta["medium_p1_gap"] > 0
        and delta["medium_p10_gap"] > 0
        and delta["low_mean_gap"] >= -0.003
        and delta["low_p10_gap"] >= -0.01
    )
    torch.save({"head_state_dict": best_state, "checkpoint": str(checkpoint)}, run_dir / "best_head.pt")
    runtime.write_csv(run_dir / "baseline_metrics.csv", baseline_rows)
    runtime.write_csv(run_dir / "candidate_metrics.csv", candidate_rows)
    runtime.write_json(run_dir / "training_history.json", history)
    result = {
        "status": "complete",
        "feasible": feasible,
        "gaps": delta,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "best_validation_rank": list(best_rank),
        "parameters": sum(parameter.numel() for parameter in head.parameters()),
        "inference_contract": "teacher uses actual TM only in training; deployed head uses ESM-derived path features and frozen Hattrick policies only",
    }
    runtime.write_json(run_dir / "summary.json", result)
    runtime.write_json(run_dir / "complete.json", {key: result[key] for key in ("status", "feasible", "gaps")})
    print(json.dumps({key: result[key] for key in ("status", "feasible", "gaps", "parameters")}), flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=tuple(LEVELS), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_one(args.level, args.seed, args.force)


if __name__ == "__main__":
    main()
