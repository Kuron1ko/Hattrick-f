from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
SHARED_DIR = TEST_DIR / "shared2x_order_regularizer"
for item in (str(ROOT), str(TEST_DIR), str(SHARED_DIR), str(THIS_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

shared_spec = importlib.util.spec_from_file_location(
    "shared2x_medium_adapter_runtime", SHARED_DIR / "run_experiment.py"
)
if shared_spec is None or shared_spec.loader is None:
    raise RuntimeError("Unable to load the shared 2x experiment runtime")
shared = importlib.util.module_from_spec(shared_spec)
sys.modules[shared_spec.name] = shared
shared_spec.loader.exec_module(shared)

from adapter import (
    CausalMediumLowAdapter,
    DynamicCausalAdapter,
    EndpointSharedMediumAdapter,
    PriorityIsolatedMediumAdapter,
    SlackAwareCausalAdapter,
)
from frameworks.hattrick_system import Hattrick
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster


K = 8
CLASSES = ("High", "Medium", "Low")
OUTPUT_ROOT = THIS_DIR / "artifacts"
SPECS = {
    1: {
        "label": "level1_correctness",
        "gradient": (0, 24),
        "safety": (24, 32),
        "validation": (32, 40),
        "evaluation": (32, 40),
        "epochs": 2,
    },
    2: {
        "label": "level2_proxy",
        "gradient": (0, 128),
        "safety": (128, 160),
        "validation": (160, 200),
        "evaluation": (200, 250),
        "epochs": 12,
    },
    3: {
        "label": "level3_validation_only",
        "gradient": (0, 318),
        "safety": (318, 350),
        "validation": (350, 400),
        "evaluation": (350, 400),
        "epochs": 30,
    },
    4: {
        "label": "level4_confirmation",
        "gradient": (0, 318),
        "safety": (318, 350),
        "validation": (350, 400),
        "evaluation": (400, 500),
        "epochs": 60,
    },
}


@dataclass
class PolicyCache:
    dataset: DM_Dataset_within_Cluster
    path_masks: torch.Tensor | None
    source_start: int
    policies: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    tms: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    predicted_tms: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    capacities: torch.Tensor
    oracle_flows: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    oracle_mlus: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    path_features: torch.Tensor | None = None

    def __len__(self) -> int:
        return int(self.tms[0].shape[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def backbone_checkpoint(level: int, backbone_seed: int) -> Path:
    phase_a = TEST_DIR / "shared2x_order_epsilon" / "artifacts" / "phase_a"
    if level == 1:
        return phase_a / "level1_correctness" / f"seed_{backbone_seed}" / "best_model.pt"
    if level == 2:
        return phase_a / "level2_proxy" / f"seed_{backbone_seed}" / "best_model.pt"
    if level == 3:
        return phase_a / "level3_validation_only" / f"seed_{backbone_seed}" / "best_model.pt"
    return (
        TEST_DIR
        / "shared2x_full_objectives"
        / "artifacts"
        / "level4_confirmation"
        / f"seed_{backbone_seed}"
        / "best_model.pt"
    )


def build_props(level: int, device: torch.device):
    props = shared.build_props(level, device)
    props.research_return_policy = False
    props.research_return_admitted = False
    props.sim_mf_mlu = 0
    props.mode = "test"
    return props


def load_backbone(level: int, backbone_seed: int, props, device: torch.device) -> tuple[Hattrick, Path]:
    checkpoint_path = backbone_checkpoint(level, backbone_seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing frozen backbone: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint_path


def cached_policy_forward(model: Hattrick, props, dataset, values, path_masks):
    """Run the frozen test-time policy head with a batch-safe topology cache.

    Hattrick's native non-dynamic test cache stores the first batch dimension.
    Re-expand the cached topology representation and static capacities for each
    subsequent batch; the underlying numerical model is unchanged.
    """
    (
        node_features,
        capacities,
        tm1,
        tm1_pred,
        tm2,
        tm2_pred,
        tm3,
        tm3_pred,
        *_rest,
    ) = values
    batch = int(tm1.shape[0])
    if not props.dynamic:
        node_features = node_features[:1]
        if hasattr(model, "transformer_output"):
            topology_cache = model.transformer_output[:1]
            model.transformer_output = topology_cache.expand(batch, -1, -1, -1)
            node_features = node_features[:1].expand(batch, -1, -1)
            capacities = capacities[:1].expand(batch, -1)
        else:
            capacities = capacities[:1]
    output = model(
        props,
        node_features,
        dataset.edge_index,
        capacities,
        dataset.padded_edge_ids_per_path,
        tm1,
        tm1_pred,
        tm2,
        tm2_pred,
        tm3,
        tm3_pred,
        dataset.pte,
        dataset.edge_ids_dict_tensor,
        dataset.original_pos_edge_ids_dict_tensor,
        path_masks,
    )
    if not props.dynamic and hasattr(model, "transformer_output"):
        model.transformer_output = model.transformer_output[:1]
    return output


def build_policy_cache(
    model: Hattrick,
    props,
    start: int,
    end: int,
    batch_size: int,
) -> PolicyCache:
    dataset = DM_Dataset_within_Cluster(props, 0, start, end)
    if int(dataset.max_source_index_read) != end - 1:
        raise RuntimeError("Split-safe reader audit failed")
    path_masks = shared.base.move_dataset_static(dataset, props.device)
    buckets: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "p0", "p1", "p2", "tm0", "tm1", "tm2", "ptm0", "ptm1", "ptm2", "cap",
            "of0", "of1", "of2", "om0", "om1", "om2",
        )
    }
    loader = shared.data_loader(dataset, batch_size, False, 0)
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    with torch.no_grad():
        for inputs in loader:
            values = shared.unpack_to_device(inputs, props)
            policies = cached_policy_forward(model, props, dataset, values, path_masks)
            batch = int(values[2].shape[0])
            capacities = values[1]
            if capacities.shape[0] == 1 and batch > 1:
                capacities = capacities.expand(batch, -1)
            buckets["p0"].append(policies[0].detach())
            buckets["p1"].append(policies[1].detach())
            buckets["p2"].append(policies[2].detach())
            buckets["tm0"].append(values[2].detach())
            buckets["tm1"].append(values[4].detach())
            buckets["tm2"].append(values[6].detach())
            buckets["ptm0"].append(values[3].detach())
            buckets["ptm1"].append(values[5].detach())
            buckets["ptm2"].append(values[7].detach())
            buckets["cap"].append(capacities.detach())
            buckets["of0"].append(values[11].reshape(-1).detach())
            buckets["of1"].append((values[12] - values[11]).reshape(-1).detach())
            buckets["of2"].append((values[13] - values[12]).reshape(-1).detach())
            buckets["om0"].append(values[8].reshape(-1).detach())
            buckets["om1"].append(values[9].reshape(-1).detach())
            buckets["om2"].append(values[10].reshape(-1).detach())
    props.research_return_policy = False
    joined = {key: torch.cat(value, dim=0) for key, value in buckets.items()}
    return PolicyCache(
        dataset=dataset,
        path_masks=path_masks,
        source_start=start,
        policies=(joined["p0"], joined["p1"], joined["p2"]),
        tms=(joined["tm0"], joined["tm1"], joined["tm2"]),
        predicted_tms=(joined["ptm0"], joined["ptm1"], joined["ptm2"]),
        capacities=joined["cap"],
        oracle_flows=(joined["of0"], joined["of1"], joined["of2"]),
        oracle_mlus=(joined["om0"], joined["om1"], joined["om2"]),
    )


def select_cache(cache: PolicyCache, indices: torch.Tensor) -> dict:
    selected = {
        "policies": tuple(value.index_select(0, indices) for value in cache.policies),
        "tms": tuple(value.index_select(0, indices) for value in cache.tms),
        "capacities": cache.capacities.index_select(0, indices),
        "oracle_flows": tuple(value.index_select(0, indices) for value in cache.oracle_flows),
        "oracle_mlus": tuple(value.index_select(0, indices) for value in cache.oracle_mlus),
    }
    if cache.path_features is not None:
        selected["path_features"] = cache.path_features.index_select(0, indices)
    return selected


def add_dynamic_path_features(cache: PolicyCache) -> PolicyCache:
    import torch_scatter

    pte = cache.dataset.pte.coalesce()
    pte_indices = pte.indices()
    row_indices, col_indices = pte_indices[0], pte_indices[1]
    pte_values = pte.values().to(dtype=torch.float32)
    path_count = int(pte.shape[0])
    capacities = cache.capacities.to(dtype=torch.float32).clamp_min(1e-9)
    predicted = [value.squeeze(-1).to(dtype=torch.float32) for value in cache.predicted_tms]
    policies = [value.squeeze(-1).to(dtype=torch.float32) for value in cache.policies]

    high_flow = policies[0] * predicted[0]
    medium_flow = policies[1] * predicted[1]
    high_load = torch.sparse.mm(pte.t(), high_flow.t()).t()
    high_medium_load = high_load + torch.sparse.mm(pte.t(), medium_flow.t()).t()

    def path_bottleneck(link_utilization: torch.Tensor) -> torch.Tensor:
        gathered = link_utilization[:, col_indices] * pte_values
        return torch_scatter.scatter_max(
            gathered, row_indices, dim=1, dim_size=path_count
        )[0]

    high_bottleneck = path_bottleneck(high_load / capacities)
    high_medium_bottleneck = path_bottleneck(high_medium_load / capacities)
    hop_count = torch.sparse.sum(pte, dim=1).to_dense().to(dtype=torch.float32)
    hop_count = hop_count / hop_count.max().clamp_min(1.0)
    hop_count = hop_count.reshape(1, -1).expand(len(cache), -1)

    normalized_demands = []
    for demand in predicted:
        od_demand = demand.reshape(len(cache), -1, K)[:, :, 0]
        scale = od_demand.mean(dim=1, keepdim=True).clamp_min(1e-9)
        normalized_demands.append(torch.log1p(demand / scale))
    features = torch.stack(
        [
            policies[0], policies[1], policies[2],
            normalized_demands[0], normalized_demands[1], normalized_demands[2],
            torch.log1p(high_bottleneck.clamp_min(0.0)),
            torch.log1p(high_medium_bottleneck.clamp_min(0.0)),
            hop_count,
        ],
        dim=-1,
    ).detach()
    return replace(cache, path_features=features)


def simulate_admission(
    model: Hattrick,
    props,
    dataset: DM_Dataset_within_Cluster,
    batch: dict,
    adapter: PriorityIsolatedMediumAdapter | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    policies = list(batch["policies"])
    if adapter is not None:
        if hasattr(adapter, "adapt_batch"):
            policies = adapter.adapt_batch(policies, batch)
        elif hasattr(adapter, "adapt_policies"):
            policies = adapter.adapt_policies(policies)
        else:
            policies[1] = adapter(policies[1])
    pte = dataset.pte.coalesce()
    pte_indices = pte.indices()
    pte_info = (pte, pte_indices[0], pte_indices[1], pte.values())
    admitted_ratios = model.simulate(
        policies,
        list(batch["tms"]),
        batch["capacities"],
        pte_info,
        int(batch["tms"][0].shape[0]),
        props,
        rate_cap=props.rate_cap,
    )[:3]
    admitted = tuple(
        ratio.reshape(ratio.shape[0], -1) * tm.squeeze(-1)
        for ratio, tm in zip(admitted_ratios, batch["tms"])
    )
    normalized = torch.stack(
        [
            flow.sum(dim=1) / oracle.clamp_min(1e-9)
            for flow, oracle in zip(admitted, batch["oracle_flows"])
        ],
        dim=1,
    )
    return admitted, normalized


def lower_tail_score(
    values: torch.Tensor,
    fraction: float = 0.25,
    tail_weight: float = 0.5,
) -> torch.Tensor:
    count = max(1, int(math.ceil(float(values.numel()) * fraction)))
    tail = torch.topk(values, count, largest=False).values.mean()
    return (1.0 - float(tail_weight)) * values.mean() + float(tail_weight) * tail


def p10_tensor(values: torch.Tensor) -> torch.Tensor:
    return torch.quantile(values, 0.1)


def train_batch(
    model: Hattrick,
    props,
    cache: PolicyCache,
    batch: dict,
    adapter: torch.nn.Module,
    step_size: float,
    low_mean_floor: float,
    low_p10_floor: float,
    low_alignment: float,
    medium_tail_weight: float,
    guard_mode: str,
    high_step_scale: float,
) -> dict[str, float]:
    causal_shield = hasattr(adapter, "low_parameters")
    full_parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    full_before = [parameter.detach().clone() for parameter in full_parameters]
    with torch.no_grad():
        _, baseline_norm = simulate_admission(model, props, cache.dataset, batch, None)
    _, current_norm = simulate_admission(model, props, cache.dataset, batch, adapter)
    medium_score = lower_tail_score(current_norm[:, 1], tail_weight=medium_tail_weight)
    low_score = lower_tail_score(current_norm[:, 2])
    if hasattr(adapter, "medium_parameters"):
        parameters = list(adapter.medium_parameters())
    else:
        parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    medium_gradients = torch.autograd.grad(-medium_score, parameters, retain_graph=True)
    low_gradients = torch.autograd.grad(-low_score, parameters)

    dot = sum(torch.sum(medium * low) for medium, low in zip(medium_gradients, low_gradients))
    medium_norm_sq = sum(torch.sum(value * value) for value in medium_gradients)
    low_norm_sq = sum(torch.sum(value * value) for value in low_gradients)
    conflict = bool(dot.item() < 0.0 and low_norm_sq.item() > 1e-20)
    if causal_shield:
        # Low is repaired by a causally downstream head.  Constraining the Medium
        # gradient here would double-count Low and needlessly reduce Medium.
        directions = list(medium_gradients)
    else:
        target_alignment = float(low_alignment) * torch.sqrt(medium_norm_sq * low_norm_sq)
        correction = torch.clamp(target_alignment - dot, min=0.0) / low_norm_sq.clamp_min(1e-20)
        directions = [
            medium + correction * low
            for medium, low in zip(medium_gradients, low_gradients)
        ]
    if not all(torch.isfinite(direction).all().item() for direction in directions):
        raise RuntimeError("Non-finite adapter gradient")
    total_elements = sum(direction.numel() for direction in directions)
    rms = torch.sqrt(sum(direction.square().sum() for direction in directions) / total_elements)
    if float(rms.item()) <= 1e-20:
        return {"accepted": 0.0, "backtracks": 0.0, "conflict": float(conflict), "step": 0.0}
    directions = [direction / rms for direction in directions]
    if hasattr(adapter, "high_parameters"):
        high_parameter_ids = {id(parameter) for parameter in adapter.high_parameters()}
        directions = [
            direction * float(high_step_scale)
            if id(parameter) in high_parameter_ids
            else direction
            for parameter, direction in zip(parameters, directions)
        ]

    before = [parameter.detach().clone() for parameter in parameters]
    current_medium_score = float(medium_score.detach().item())
    accepted_step = 0.0
    backtracks = 0
    for power in range(9):
        candidate_step = float(step_size) * (0.5 ** power)
        with torch.no_grad():
            for parameter, original, direction in zip(parameters, before, directions):
                parameter.copy_(original - candidate_step * direction)
            adapter.project_parameter_box()
            _, candidate_norm = simulate_admission(model, props, cache.dataset, batch, adapter)
            candidate_medium_score = float(
                lower_tail_score(candidate_norm[:, 1], tail_weight=medium_tail_weight).item()
            )
            low_mean_gap = float((candidate_norm[:, 2].mean() - baseline_norm[:, 2].mean()).item())
            low_p10_gap = float((p10_tensor(candidate_norm[:, 2]) - p10_tensor(baseline_norm[:, 2])).item())
        medium_ok = candidate_medium_score >= current_medium_score - 1e-7
        low_ok = (
            low_mean_gap >= -float(low_mean_floor)
            and low_p10_gap >= -float(low_p10_floor)
        )
        if medium_ok and (causal_shield or low_ok):
            accepted_step = candidate_step
            backtracks = power
            break
    if accepted_step == 0.0:
        with torch.no_grad():
            for parameter, original in zip(parameters, before):
                parameter.copy_(original)
        backtracks = 9

    shield_accepted = 0.0
    shield_step = 0.0
    if causal_shield:
        _, shield_current_norm = simulate_admission(model, props, cache.dataset, batch, adapter)
        shield_current_score = lower_tail_score(shield_current_norm[:, 2])
        shield_parameters = list(adapter.low_parameters())
        shield_gradients = torch.autograd.grad(-shield_current_score, shield_parameters)
        shield_rms = torch.sqrt(
            sum(value.square().sum() for value in shield_gradients)
            / sum(value.numel() for value in shield_gradients)
        )
        if float(shield_rms.item()) > 1e-20:
            shield_directions = [value / shield_rms for value in shield_gradients]
            shield_before = [parameter.detach().clone() for parameter in shield_parameters]
            current_low_score = float(shield_current_score.detach().item())
            current_medium_mean = float(shield_current_norm[:, 1].mean().detach().item())
            for power in range(9):
                candidate_step = float(step_size) * (0.5 ** power)
                with torch.no_grad():
                    for parameter, original, direction in zip(
                        shield_parameters, shield_before, shield_directions
                    ):
                        parameter.copy_(original - candidate_step * direction)
                    adapter.project_parameter_box()
                    _, shield_candidate_norm = simulate_admission(
                        model, props, cache.dataset, batch, adapter
                    )
                    candidate_low_score = float(lower_tail_score(shield_candidate_norm[:, 2]).item())
                    candidate_medium_mean = float(shield_candidate_norm[:, 1].mean().item())
                    shield_low_mean_gap = float(
                        (shield_candidate_norm[:, 2].mean() - baseline_norm[:, 2].mean()).item()
                    )
                    shield_low_p10_gap = float(
                        (p10_tensor(shield_candidate_norm[:, 2]) - p10_tensor(baseline_norm[:, 2])).item()
                    )
                shield_low_ok = (
                    shield_low_mean_gap >= -float(low_mean_floor)
                    and shield_low_p10_gap >= -float(low_p10_floor)
                )
                if (
                    candidate_low_score >= current_low_score - 1e-7
                    and abs(candidate_medium_mean - current_medium_mean) <= 1e-6
                    and (guard_mode == "global" or shield_low_ok)
                ):
                    shield_accepted = 1.0
                    shield_step = candidate_step
                    break
            if shield_accepted == 0.0:
                with torch.no_grad():
                    for parameter, original in zip(shield_parameters, shield_before):
                        parameter.copy_(original)
        with torch.no_grad():
            _, final_norm = simulate_admission(model, props, cache.dataset, batch, adapter)
            final_medium_score = float(
                lower_tail_score(final_norm[:, 1], tail_weight=medium_tail_weight).item()
            )
            final_low_mean_gap = float(
                (final_norm[:, 2].mean() - baseline_norm[:, 2].mean()).item()
            )
            final_low_p10_gap = float(
                (p10_tensor(final_norm[:, 2]) - p10_tensor(baseline_norm[:, 2])).item()
            )
        final_low_ok = (
            final_low_mean_gap >= -float(low_mean_floor)
            and final_low_p10_gap >= -float(low_p10_floor)
        )
        if final_medium_score < current_medium_score - 1e-7 or (
            guard_mode != "global" and not final_low_ok
        ):
            with torch.no_grad():
                for parameter, original in zip(full_parameters, full_before):
                    parameter.copy_(original)
            accepted_step = 0.0
            shield_accepted = 0.0
            shield_step = 0.0
    return {
        "accepted": float(accepted_step > 0.0),
        "backtracks": float(backtracks),
        "conflict": float(conflict),
        "step": accepted_step,
        "shield_accepted": shield_accepted,
        "shield_step": shield_step,
    }


def summarize_rows(rows: list[dict]) -> list[dict]:
    return shared.base.summarize_rows(rows)


def evaluate_cache(
    model: Hattrick,
    props,
    cache: PolicyCache,
    adapter: torch.nn.Module | None,
    batch_size: int = 8,
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    path_masks = cache.path_masks
    flat_masks = [path_masks[index].reshape(-1) for index in range(3)] if path_masks is not None else None
    with torch.no_grad():
        for offset in range(0, len(cache), batch_size):
            stop = min(offset + batch_size, len(cache))
            indices = torch.arange(offset, stop, device=props.device)
            batch = select_cache(cache, indices)
            admitted, normalized = simulate_admission(model, props, cache.dataset, batch, adapter)
            cumulative = torch.zeros_like(admitted[0])
            for class_index, class_name in enumerate(CLASSES):
                cumulative = cumulative + admitted[class_index]
                loads = torch.sparse.mm(
                    cache.dataset.pte.to(dtype=torch.float32).t(), cumulative.to(dtype=torch.float32).t()
                ).t()
                capacity_ratio = loads / batch["capacities"].clamp_min(1e-9)
                raw_mlu = capacity_ratio.max(dim=1).values
                disabled = torch.zeros(stop - offset, device=props.device)
                if flat_masks is not None:
                    disabled_paths = ~flat_masks[class_index]
                    if disabled_paths.any():
                        disabled = admitted[class_index][:, disabled_paths].abs().max(dim=1).values
                admitted_total = admitted[class_index].sum(dim=1)
                demand = batch["tms"][class_index].sum(dim=1).squeeze(-1) / K
                for local in range(stop - offset):
                    oracle = max(float(batch["oracle_flows"][class_index][local].item()), 1e-9)
                    rows.append(
                        {
                            "snapshot": cache.source_start + offset + local,
                            "class": class_name,
                            "admitted_traffic": float(admitted_total[local].item()),
                            "demand": float(demand[local].item()),
                            "fulfill_ratio": float((admitted_total[local] / demand[local].clamp_min(1e-9)).item()),
                            "oracle_admitted_traffic": oracle,
                            "norm_fulfill": float(normalized[local, class_index].item()),
                            "raw_mlu": float(raw_mlu[local].item()),
                            "oracle_mlu": float(batch["oracle_mlus"][class_index][local].item()),
                            "normalized_mlu": float(
                                (raw_mlu[local] / batch["oracle_mlus"][class_index][local].clamp_min(1e-9)).item()
                            ),
                            "disabled_flow": float(disabled[local].item()),
                            "admitted_capacity_ratio": float(capacity_ratio[local].max().item()),
                        }
                    )
    if any(
        not math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key != "class"
    ):
        raise RuntimeError("Evaluation produced NaN or Inf")
    return rows, summarize_rows(rows)


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def metric_gaps(candidate: list[dict], baseline: list[dict]) -> dict[str, float]:
    c = summary_index(candidate)
    b = summary_index(baseline)
    result: dict[str, float] = {}
    for class_name in CLASSES:
        for suffix in ("mean", "p1", "p10"):
            key = f"norm_fulfill_{suffix}"
            result[f"{class_name.lower()}_{suffix}_gap"] = float(c[class_name][key]) - float(b[class_name][key])
    return result


def feasible(
    candidate: list[dict],
    baseline: list[dict],
    low_mean_floor: float = 0.006,
    low_p10_floor: float = 0.01,
    high_exact: bool = True,
) -> bool:
    indexed = summary_index(candidate)
    baseline_indexed = summary_index(baseline)
    gaps = metric_gaps(candidate, baseline)
    high_floor = min(0.995, float(baseline_indexed["High"]["norm_fulfill_mean"]) - 1e-7)
    return (
        float(indexed["High"]["norm_fulfill_mean"]) >= high_floor
        and (not high_exact or abs(gaps["high_mean_gap"]) <= 1e-6)
        and gaps["low_mean_gap"] >= -low_mean_floor
        and gaps["low_p10_gap"] >= -low_p10_floor
        and max(float(row["max_admitted_capacity_ratio"]) for row in candidate) <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in candidate) <= 1e-8
    )


def rank_checkpoint(
    candidate: list[dict], baseline: list[dict], high_exact: bool = True
) -> tuple[float, ...]:
    gaps = metric_gaps(candidate, baseline)
    if feasible(candidate, baseline, high_exact=high_exact):
        worst_medium_gain = min(
            gaps["medium_mean_gap"], gaps["medium_p1_gap"], gaps["medium_p10_gap"]
        )
        return (1.0, worst_medium_gain, gaps["medium_p10_gap"], gaps["medium_mean_gap"])
    indexed = summary_index(candidate)
    return (
        0.0,
        float(indexed["High"]["norm_fulfill_mean"]),
        gaps["low_p10_gap"],
        gaps["medium_p10_gap"],
    )


def safe_remove_run_dir(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise RuntimeError(f"Refusing unsafe removal: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def run_one(
    level: int,
    seed: int,
    backbone_seed: int,
    epochs: int | None,
    step_size: float,
    adapter_kind: str,
    low_alignment: float,
    medium_tail_weight: float,
    guard_mode: str,
    high_step_scale: float,
    force: bool,
) -> Path:
    if level not in SPECS:
        raise ValueError(f"level must be one of {tuple(SPECS)}")
    spec = SPECS[level]
    total_epochs = int(spec["epochs"] if epochs is None else epochs)
    run_dir = (
        OUTPUT_ROOT
        / spec["label"]
        / adapter_kind
        / f"low_align_{format(float(low_alignment), '.3g').replace('.', 'p')}"
        / f"medium_tail_{format(float(medium_tail_weight), '.3g').replace('.', 'p')}"
        / f"guard_{guard_mode}"
        / f"high_step_{format(float(high_step_scale), '.3g').replace('.', 'p')}"
        / f"backbone_{backbone_seed}"
        / f"order_seed_{seed}"
    )
    if (run_dir / "complete.json").exists() and not force:
        print(f"[skip] {run_dir}", flush=True)
        return run_dir
    if force:
        safe_remove_run_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    props = build_props(level, device)
    model, checkpoint_path = load_backbone(level, backbone_seed, props, device)
    config = {
        "method": "Priority-Isolated Medium Adapter (PIMA)",
        "level": level,
        "label": spec["label"],
        "seed": seed,
        "backbone_seed": backbone_seed,
        "topology": shared.TOPOLOGY,
        "load_factor": 2.0,
        "paths_per_pair": K,
        "gradient": list(spec["gradient"]),
        "safety": list(spec["safety"]),
        "validation": list(spec["validation"]),
        "evaluation": list(spec["evaluation"]),
        "epochs": total_epochs,
        "batch_size": 8,
        "step_size": float(step_size),
        "adapter_kind": adapter_kind,
        "low_alignment": float(low_alignment),
        "medium_tail_weight": float(medium_tail_weight),
        "guard_mode": guard_mode,
        "high_step_scale": float(high_step_scale),
        "batch_guard": {"low_mean_gap_floor": -0.001, "low_p10_gap_floor": -0.005},
        "checkpoint_guard": {"high_mean_floor": 0.995, "low_mean_gap_floor": -0.006, "low_p10_gap_floor": -0.01},
        "early_stopping": "stop after 10 consecutive full epoch rollbacks",
        "backbone_checkpoint": str(checkpoint_path),
        "backbone_sha256": sha256(checkpoint_path),
        "architecture": {
            "trainable_scope": "frozen Hattrick; isolated Medium residual followed by an optional causal Low shield; High always bypasses",
            "residual": "per-OD base-measure-centred tanh multiplier",
            "objective": "equal mixture of Medium mean and lower-quartile normalized fulfillment",
            "gradient_guard": "constrain Medium step to a positive-alignment cone around the Low descent half-space",
            "step_guard": "exact actual-TM sequential-admission replay with backtracking",
        },
        "source_sha256": {
            "adapter.py": sha256(THIS_DIR / "adapter.py"),
            "run_experiment.py": sha256(Path(__file__).resolve()),
            "frameworks/hattrick_system.py": sha256(ROOT / "frameworks" / "hattrick_system.py"),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
    }
    write_json(run_dir / "config.json", config)

    print("[cache] frozen-backbone policies", flush=True)
    gradient_cache = build_policy_cache(model, props, *spec["gradient"], batch_size=8)
    safety_cache = build_policy_cache(model, props, *spec["safety"], batch_size=8)
    validation_cache = build_policy_cache(model, props, *spec["validation"], batch_size=8)
    evaluation_cache = build_policy_cache(model, props, *spec["evaluation"], batch_size=8)
    path_count = int(gradient_cache.policies[1].shape[1])
    if path_count % K:
        raise RuntimeError(f"Path count {path_count} is not divisible by K={K}")
    if adapter_kind == "dynamic_causal":
        gradient_cache = add_dynamic_path_features(gradient_cache)
        safety_cache = add_dynamic_path_features(safety_cache)
        validation_cache = add_dynamic_path_features(validation_cache)
        evaluation_cache = add_dynamic_path_features(evaluation_cache)
        adapter = DynamicCausalAdapter(feature_dim=9, hidden_dim=16, paths_per_pair=K)
    elif adapter_kind == "pair":
        adapter = PriorityIsolatedMediumAdapter(path_count // K, K)
    elif adapter_kind in ("endpoint", "causal_endpoint", "slack_causal"):
        pairs = list(gradient_cache.dataset.pij.keys())
        if len(pairs) != path_count // K:
            raise RuntimeError("OD pair order does not match the emitted policy")
        sources = torch.tensor([int(pair[0]) for pair in pairs], dtype=torch.long)
        destinations = torch.tensor([int(pair[1]) for pair in pairs], dtype=torch.long)
        if adapter_kind == "endpoint":
            adapter = EndpointSharedMediumAdapter(sources, destinations, K)
        elif adapter_kind == "causal_endpoint":
            adapter = CausalMediumLowAdapter(sources, destinations, K)
        else:
            adapter = SlackAwareCausalAdapter(sources, destinations, K)
    else:
        raise ValueError(f"Unknown adapter kind: {adapter_kind}")
    adapter = adapter.to(device=device, dtype=props.dtype)

    sample_policy = gradient_cache.policies[1][:8]
    zero_policy = adapter(sample_policy)
    zero_policy_delta = float((zero_policy - sample_policy).abs().max().item())
    zero_low_policy_delta = 0.0
    if hasattr(adapter, "adapt_batch"):
        sample_indices = torch.arange(0, 8, device=device)
        sample_batch = select_cache(gradient_cache, sample_indices)
        sample_policies = list(sample_batch["policies"])
        zero_policies = adapter.adapt_batch(sample_policies, sample_batch)
        zero_policy_delta = max(
            float((actual - expected).abs().max().item())
            for actual, expected in zip(zero_policies, sample_policies)
        )
        zero_low_policy_delta = float(
            (zero_policies[2] - sample_policies[2]).abs().max().item()
        )
    elif hasattr(adapter, "adapt_policies"):
        sample_policies = [value[:8] for value in gradient_cache.policies]
        zero_policies = adapter.adapt_policies(sample_policies)
        zero_policy_delta = max(
            float((actual - expected).abs().max().item())
            for actual, expected in zip(zero_policies, sample_policies)
        )
        zero_low_policy_delta = float(
            (zero_policies[2] - sample_policies[2]).abs().max().item()
        )
    mass_delta = float(
        (
            zero_policy.reshape(8, -1, K).sum(-1)
            - sample_policy.reshape(8, -1, K).sum(-1)
        ).abs().max().item()
    )
    if zero_policy_delta != 0.0 or mass_delta != 0.0:
        raise RuntimeError(f"Zero adapter failed exact identity: policy={zero_policy_delta}, mass={mass_delta}")

    baseline_safety_rows, baseline_safety = evaluate_cache(model, props, safety_cache, None)
    baseline_validation_rows, baseline_validation = evaluate_cache(model, props, validation_cache, None)
    baseline_evaluation_rows, baseline_evaluation = evaluate_cache(model, props, evaluation_cache, None)
    write_csv(run_dir / "baseline_safety_metrics.csv", baseline_safety_rows)
    write_csv(run_dir / "baseline_validation_metrics.csv", baseline_validation_rows)
    write_csv(run_dir / "baseline_evaluation_metrics.csv", baseline_evaluation_rows)
    write_json(
        run_dir / "baseline_summary.json",
        {"safety": baseline_safety, "validation": baseline_validation, "evaluation": baseline_evaluation},
    )

    zero_rows, zero_summary = evaluate_cache(model, props, evaluation_cache, adapter)
    zero_row_delta = max(
        abs(float(a["norm_fulfill"]) - float(b["norm_fulfill"]))
        for a, b in zip(zero_rows, baseline_evaluation_rows)
    )
    audit = {
        "zero_policy_max_abs_delta": zero_policy_delta,
        "zero_low_policy_max_abs_delta": zero_low_policy_delta,
        "zero_od_mass_max_abs_delta": mass_delta,
        "zero_evaluation_norm_fulfill_max_abs_delta": zero_row_delta,
        "high_low_bypass_is_structural": True,
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
    }
    write_json(run_dir / "identity_audit.json", audit)
    if zero_row_delta > 1e-6:
        raise RuntimeError(f"Zero adapter evaluation mismatch: {zero_row_delta}")

    best_path = run_dir / "best_adapter.pt"
    high_exact = adapter_kind != "slack_causal"
    initial_rank = rank_checkpoint(
        baseline_validation, baseline_validation, high_exact=high_exact
    )
    torch.save(
        {"epoch": 0, "rank": initial_rank, "adapter_state_dict": adapter.state_dict(), "config": config},
        best_path,
    )
    best_rank = initial_rank
    best_epoch = 0
    history: list[dict] = []
    consecutive_rollbacks = 0
    high_budget_open = adapter_kind == "slack_causal"
    started = time.perf_counter()
    for epoch in range(1, total_epochs + 1):
        epoch_state = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch * 1009)
        permutation = torch.randperm(len(gradient_cache), generator=generator)
        batch_stats: list[dict[str, float]] = []
        for offset in range(0, len(gradient_cache), 8):
            selected = permutation[offset : offset + 8].to(device=device)
            batch = select_cache(gradient_cache, selected)
            batch_stats.append(
                train_batch(
                    model,
                    props,
                    gradient_cache,
                    batch,
                    adapter,
                    step_size,
                    low_mean_floor=0.001,
                    low_p10_floor=0.005,
                    low_alignment=low_alignment,
                    medium_tail_weight=medium_tail_weight,
                    guard_mode=guard_mode,
                    high_step_scale=high_step_scale if high_budget_open else 0.0,
                )
            )

        candidate_state = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
        high_projection_scale = 1.0
        if adapter_kind == "slack_causal" and high_budget_open:
            # Preserve the full Medium/Low update and project only the isolated
            # High head onto the High>=0.995 boundary on held-out caches.
            trial_safety_rows, trial_safety_summary = evaluate_cache(
                model, props, safety_cache, adapter
            )
            trial_validation_rows, trial_validation_summary = evaluate_cache(
                model, props, validation_cache, adapter
            )
            baseline_safety_high = summary_index(baseline_safety)["High"]["norm_fulfill_mean"]
            baseline_validation_high = summary_index(baseline_validation)["High"]["norm_fulfill_mean"]
            safety_high_target = min(0.99505, float(baseline_safety_high) - 1e-7)
            validation_high_target = min(0.99505, float(baseline_validation_high) - 1e-7)

            def high_is_safe(
                safety_summary: list[dict], validation_summary: list[dict]
            ) -> bool:
                return (
                    summary_index(safety_summary)["High"]["norm_fulfill_mean"]
                    >= safety_high_target
                    and summary_index(validation_summary)["High"]["norm_fulfill_mean"]
                    >= validation_high_target
                )

            if not high_is_safe(trial_safety_summary, trial_validation_summary):
                safe_scale, unsafe_scale = 0.0, 1.0
                for _ in range(10):
                    scale = 0.5 * (safe_scale + unsafe_scale)
                    projected = {
                        key: (
                            epoch_state[key] + scale * (candidate_state[key] - epoch_state[key])
                            if key.startswith("high_head.")
                            else candidate_state[key]
                        )
                        for key in candidate_state
                    }
                    adapter.load_state_dict(projected)
                    trial_safety_rows, trial_safety_summary = evaluate_cache(
                        model, props, safety_cache, adapter
                    )
                    trial_validation_rows, trial_validation_summary = evaluate_cache(
                        model, props, validation_cache, adapter
                    )
                    if high_is_safe(trial_safety_summary, trial_validation_summary):
                        safe_scale = scale
                    else:
                        unsafe_scale = scale
                high_projection_scale = safe_scale
                candidate_state = {
                    key: (
                        epoch_state[key]
                        + high_projection_scale * (candidate_state[key] - epoch_state[key])
                        if key.startswith("high_head.")
                        else candidate_state[key]
                    )
                    for key in candidate_state
                }
                adapter.load_state_dict(candidate_state)

        safety_rows: list[dict] = []
        safety_summary: list[dict] = []
        safety_scale = 0.0
        for power in range(9):
            scale = 0.5 ** power
            if power > 0:
                interpolated = {
                    key: epoch_state[key] + scale * (candidate_state[key] - epoch_state[key])
                    for key in epoch_state
                }
                adapter.load_state_dict(interpolated)
            safety_rows, safety_summary = evaluate_cache(model, props, safety_cache, adapter)
            safety_mean_floor = 0.006 if guard_mode == "global" else 0.002
            safety_p10_floor = 0.01 if guard_mode == "global" else 0.0075
            if feasible(
                safety_summary,
                baseline_safety,
                low_mean_floor=safety_mean_floor,
                low_p10_floor=safety_p10_floor,
                high_exact=high_exact,
            ):
                safety_scale = scale
                break
        rolled_back = safety_scale == 0.0
        if rolled_back:
            adapter.load_state_dict(epoch_state)
            safety_rows, safety_summary = evaluate_cache(model, props, safety_cache, adapter)
        validation_rows, validation_summary = evaluate_cache(model, props, validation_cache, adapter)
        if adapter_kind == "slack_causal":
            safety_stop = min(
                0.99515,
                float(summary_index(baseline_safety)["High"]["norm_fulfill_mean"]) - 1e-7,
            )
            validation_stop = min(
                0.99515,
                float(summary_index(baseline_validation)["High"]["norm_fulfill_mean"]) - 1e-7,
            )
            high_budget_open = (
                float(summary_index(safety_summary)["High"]["norm_fulfill_mean"])
                > safety_stop
                and float(summary_index(validation_summary)["High"]["norm_fulfill_mean"])
                > validation_stop
            )
        rank = rank_checkpoint(
            validation_summary, baseline_validation, high_exact=high_exact
        )
        gaps = metric_gaps(validation_summary, baseline_validation)
        safety_mean_floor = 0.006 if guard_mode == "global" else 0.002
        safety_p10_floor = 0.01 if guard_mode == "global" else 0.0075
        if rank > best_rank and feasible(
            safety_summary,
            baseline_safety,
            safety_mean_floor,
            safety_p10_floor,
            high_exact=high_exact,
        ):
            best_rank = rank
            best_epoch = epoch
            torch.save(
                {"epoch": epoch, "rank": rank, "adapter_state_dict": adapter.state_dict(), "config": config},
                best_path,
            )
        write_csv(run_dir / f"safety_epoch_{epoch:03d}_metrics.csv", safety_rows)
        write_csv(run_dir / f"validation_epoch_{epoch:03d}_metrics.csv", validation_rows)
        write_json(
            run_dir / f"validation_epoch_{epoch:03d}_summary.json",
            {"safety": safety_summary, "validation": validation_summary, "gaps": gaps},
        )
        history_row = {
            "epoch": epoch,
            "rolled_back": int(rolled_back),
            "safety_scale": safety_scale,
            "high_projection_scale": high_projection_scale,
            "high_budget_open": int(high_budget_open),
            "accepted_batches": sum(item["accepted"] for item in batch_stats),
            "mean_backtracks": float(np.mean([item["backtracks"] for item in batch_stats])),
            "conflict_batches": sum(item["conflict"] for item in batch_stats),
            "mean_accepted_step": float(np.mean([item["step"] for item in batch_stats])),
            "shield_accepted_batches": sum(item["shield_accepted"] for item in batch_stats),
            "mean_shield_step": float(np.mean([item["shield_step"] for item in batch_stats])),
            **gaps,
            "validation_high_mean": summary_index(validation_summary)["High"]["norm_fulfill_mean"],
            "validation_medium_mean": summary_index(validation_summary)["Medium"]["norm_fulfill_mean"],
            "validation_medium_p10": summary_index(validation_summary)["Medium"]["norm_fulfill_p10"],
            "validation_low_mean": summary_index(validation_summary)["Low"]["norm_fulfill_mean"],
            "validation_low_p10": summary_index(validation_summary)["Low"]["norm_fulfill_p10"],
        }
        history.append(history_row)
        write_csv(run_dir / "train_history.csv", history)
        print(
            f"[epoch {epoch:03d}] rollback={int(rolled_back)} "
            f"H={summary_index(validation_summary)['High']['norm_fulfill_mean']:.6f} "
            f"Mmean={gaps['medium_mean_gap']:+.6f} Mp10={gaps['medium_p10_gap']:+.6f} "
            f"Lmean={gaps['low_mean_gap']:+.6f} Lp10={gaps['low_p10_gap']:+.6f}",
            flush=True,
        )
        consecutive_rollbacks = consecutive_rollbacks + 1 if rolled_back else 0
        if consecutive_rollbacks >= 10:
            print(f"[early-stop] {consecutive_rollbacks} consecutive safety rollbacks", flush=True)
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    adapter.load_state_dict(best["adapter_state_dict"])
    final_rows, final_summary = evaluate_cache(model, props, evaluation_cache, adapter)
    final_gaps = metric_gaps(final_summary, baseline_evaluation)
    write_csv(run_dir / "evaluation_metrics.csv", final_rows)
    write_json(
        run_dir / "evaluation_summary.json",
        {
            "best_epoch": int(best["epoch"]),
            "baseline": baseline_evaluation,
            "candidate": final_summary,
            "gaps": final_gaps,
            "feasible": feasible(
                final_summary, baseline_evaluation, high_exact=high_exact
            ),
        },
    )
    complete = {
        "status": "complete",
        "best_epoch": int(best["epoch"]),
        "best_rank": list(best["rank"]),
        "identity_audit": audit,
        "evaluation_gaps": final_gaps,
        "evaluation_feasible": feasible(
            final_summary, baseline_evaluation, high_exact=high_exact
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(run_dir / "complete.json", complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)
    return run_dir


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Priority-Isolated Medium Adapter experiment")
    parser.add_argument("--level", type=int, choices=tuple(SPECS), required=True)
    parser.add_argument("--seed", type=int, default=490, help="Adapter data-order seed")
    parser.add_argument("--backbone-seed", type=int, default=490)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument(
        "--adapter-kind",
        choices=("slack_causal", "dynamic_causal", "causal_endpoint", "endpoint", "pair"),
        default="causal_endpoint",
    )
    parser.add_argument("--low-alignment", type=float, default=0.25)
    parser.add_argument("--medium-tail-weight", type=float, default=0.1)
    parser.add_argument("--guard-mode", choices=("batch", "global"), default="batch")
    parser.add_argument(
        "--high-step-scale",
        type=float,
        default=1.0,
        help="Relative update rate for the isolated High head in slack_causal",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_cli()
    run_one(
        level=arguments.level,
        seed=arguments.seed,
        backbone_seed=arguments.backbone_seed,
        epochs=arguments.epochs,
        step_size=arguments.step_size,
        adapter_kind=arguments.adapter_kind,
        low_alignment=arguments.low_alignment,
        medium_tail_weight=arguments.medium_tail_weight,
        guard_mode=arguments.guard_mode,
        high_step_scale=arguments.high_step_scale,
        force=arguments.force,
    )
