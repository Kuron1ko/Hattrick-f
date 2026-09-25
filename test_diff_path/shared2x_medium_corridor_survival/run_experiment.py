from __future__ import annotations

"""Training-only Medium corridor-survival objective for persistent Hattrick.

This branch deliberately does *not* add another inference feature or head.  It
keeps the persistent sparse-attention Hattrick policy unchanged and augments
the existing cumulative High+Medium objective with a differentiable estimate
of how much Medium demand can still use its best candidate corridor after High
has been sequentially admitted.

The objective is supervised with actual training traffic, while policy logits
remain functions of ESM predictions only.  Therefore it learns how to correct
ESM error but has exactly zero deployment-time cost.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
PERSISTENT_DIR = TEST_DIR / "shared2x_sparse_path_cross_attention_persistent"
PERSISTENT_RUNNER = PERSISTENT_DIR / "run_experiment.py"
for item in (str(ROOT), str(TEST_DIR), str(PERSISTENT_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


persistent = load_module(
    "shared2x_medium_corridor_survival_persistent", PERSISTENT_RUNNER
)
full = persistent.parent.full


# Tuned only on Level-1/Level-2 development splits.  The values are overwritten
# from CLI before training begins and included in source/settings fingerprints.
CORRIDOR_WEIGHT = 0.5
EDGE_TEMPERATURE = 0.05
PATH_TEMPERATURE = 0.05
MAX_RESIDUAL_RATIO = 4.0


def _segment_softmin(
    values: torch.Tensor,
    segment: torch.Tensor,
    segments: int,
    temperature: float,
) -> torch.Tensor:
    """Length-normalized soft-min over sparse path-edge memberships.

    ``values`` is [B,nnz] and ``segment`` maps each nonzero to one path.  The
    log-*mean*-exp form makes equal-edge paths independent of path length.
    """

    if values.ndim != 2 or segment.ndim != 1:
        raise ValueError("Unexpected segment soft-min shapes")
    if int(values.shape[1]) != int(segment.numel()):
        raise ValueError("Sparse values/segment width mismatch")
    tau = float(temperature)
    if not tau > 0.0:
        raise ValueError("Edge temperature must be positive")

    batch = int(values.shape[0])
    index = segment.reshape(1, -1).expand(batch, -1)
    scaled = -values / tau
    maxima = torch.full(
        (batch, segments),
        -torch.inf,
        dtype=scaled.dtype,
        device=scaled.device,
    )
    maxima.scatter_reduce_(1, index, scaled, reduce="amax", include_self=True)
    centered = torch.exp(scaled - maxima.gather(1, index))
    sums = torch.zeros_like(maxima)
    sums.scatter_add_(1, index, centered)
    counts = torch.zeros_like(maxima)
    counts.scatter_add_(1, index, torch.ones_like(centered))
    if bool((counts <= 0).any().item()):
        raise RuntimeError("A candidate path contains no edge")
    log_mean_exp = maxima + torch.log(sums / counts)
    return -tau * log_mean_exp


def _smooth_cap_one(value: torch.Tensor, sharpness: float = 8.0) -> torch.Tensor:
    """Smooth approximation of min(value, 1), anchored exactly at zero."""

    k = float(sharpness)
    baseline = F.softplus(value.new_tensor(-k)) / k
    return value - F.softplus(k * (value - 1.0)) / k + baseline


def medium_corridor_survival(
    admitted_high: torch.Tensor,
    capacities: torch.Tensor,
    medium_demand: torch.Tensor,
    paths_to_edges: torch.Tensor,
    medium_path_mask: torch.Tensor | None,
    paths_per_od: int,
    *,
    edge_temperature: float = EDGE_TEMPERATURE,
    path_temperature: float = PATH_TEMPERATURE,
    max_residual_ratio: float = MAX_RESIDUAL_RATIO,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return demand-weighted survival of Medium's best candidate corridor.

    High is not tied to any teacher path.  Its admitted path flow determines
    residual edge capacity.  For every Medium OD/path we take a soft minimum of
    residual-capacity / OD-demand along that path, smoothly cap it at one, then
    take a normalized soft maximum over the K candidate paths.
    """

    pte = paths_to_edges.coalesce()
    row, col = pte.indices()
    total_paths, _num_edges = int(pte.shape[0]), int(pte.shape[1])
    k = int(paths_per_od)
    if total_paths % k:
        raise RuntimeError("Path count is not divisible by paths_per_od")
    num_ods = total_paths // k
    batch = int(admitted_high.shape[0])
    if tuple(admitted_high.shape) != (batch, total_paths):
        raise RuntimeError("Unexpected admitted-High shape")

    capacity = capacities.to(dtype=torch.float32)
    if int(capacity.shape[0]) == 1 and batch > 1:
        capacity = capacity.expand(batch, -1)
    high_load = torch.sparse.mm(
        pte.to(dtype=torch.float32).t(), admitted_high.to(dtype=torch.float32).t()
    ).t()
    # Sequential admission should make this nonnegative; clamp only absorbs
    # floating-point overshoot at exactly saturated edges.
    residual = (capacity - high_load).clamp_min(0.0)

    demand = medium_demand.to(dtype=torch.float32).reshape(batch, num_ods, k)
    demand_by_od = demand.mean(dim=-1)
    demand_by_path = demand_by_od.repeat_interleave(k, dim=1)
    selected_ratio = residual[:, col] / demand_by_path[:, row].clamp_min(1e-6)
    selected_ratio = selected_ratio.clamp(max=float(max_residual_ratio))
    path_bottleneck = _segment_softmin(
        selected_ratio, row, total_paths, edge_temperature
    )
    path_score = _smooth_cap_one(path_bottleneck).reshape(batch, num_ods, k)

    if medium_path_mask is None:
        valid = torch.ones(
            (num_ods, k), dtype=torch.bool, device=path_score.device
        )
    else:
        valid = medium_path_mask.reshape(num_ods, k).to(
            device=path_score.device, dtype=torch.bool
        )
    valid_count = valid.sum(dim=-1)
    if bool((valid_count <= 0).any().item()):
        raise RuntimeError("A Medium OD has no enabled path")
    tau_path = float(path_temperature)
    if not tau_path > 0.0:
        raise ValueError("Path temperature must be positive")
    logits = (path_score / tau_path).masked_fill(
        ~valid.reshape(1, num_ods, k), -torch.inf
    )
    # log-mean-exp is a smooth max without rewarding an OD merely for having
    # more enabled paths.
    od_survival = tau_path * (
        torch.logsumexp(logits, dim=-1)
        - torch.log(valid_count.to(dtype=logits.dtype)).reshape(1, num_ods)
    )
    active = demand_by_od > 1e-6
    weights = demand_by_od * active
    snapshot_survival = (od_survival * weights).sum(dim=-1) / weights.sum(
        dim=-1
    ).clamp_min(1e-6)
    score = snapshot_survival.mean()
    if not torch.isfinite(score).item():
        raise RuntimeError("Non-finite Medium corridor-survival score")
    return score, {
        "high_load": high_load,
        "residual": residual,
        "path_bottleneck": path_bottleneck,
        "od_survival": od_survival,
        "snapshot_survival": snapshot_survival,
    }


def build_objectives(model, props, dataset, values, path_masks):
    (
        _node_features,
        capacities,
        _tm1,
        _tm1_pred,
        tm2,
        _tm2_pred,
        _tm3,
        _tm3_pred,
        opt1,
        opt2,
        opt3,
        opt1_mf,
        opt2_mf,
        opt3_mf,
        _snapshots,
    ) = values
    output, effective_capacities = full.shared.model_forward(
        model, props, dataset, values, path_masks
    )
    (
        edges_high,
        edges_high_medium,
        edges_all,
        _edges_high_final,
        _edges_high_medium_final,
        all_traffic,
        admitted_high,
        admitted_medium,
        _admitted_low,
    ) = output

    loss_fh, value_fh = full.loss_mf(admitted_high, opt1_mf.detach())
    loss_uh, value_uh = full.loss_mlu(edges_high, opt1.detach())
    loss_fhm, value_fhm = full.loss_mf(
        admitted_high + admitted_medium, opt2_mf.detach()
    )
    loss_uhm, value_uhm = full.loss_mlu(edges_high_medium, opt2.detach())
    loss_fhml, value_fhml = full.loss_mf(all_traffic, opt3_mf.detach())
    loss_uhml, value_uhml = full.loss_mlu(edges_all, opt3.detach())

    medium_mask = None if path_masks is None else path_masks[1]
    corridor_score, diagnostics = medium_corridor_survival(
        admitted_high,
        effective_capacities,
        tm2,
        dataset.pte,
        medium_mask,
        int(props.num_paths_per_pair),
        edge_temperature=EDGE_TEMPERATURE,
        path_temperature=PATH_TEMPERATURE,
        max_residual_ratio=MAX_RESIDUAL_RATIO,
    )
    # Match loss_mf's detached value normalization so lambda is interpretable
    # across batches.  Fh and Uh remain earlier projected priorities.  Blending
    # here keeps the exact six-objective projection and turns Fhm's gradient
    # into "admit Medium while preserving at least one viable corridor".
    loss_corridor = -corridor_score / corridor_score.detach().clamp_min(1e-6)
    loss_fhm_augmented = loss_fhm + float(CORRIDOR_WEIGHT) * loss_corridor
    model._last_medium_corridor_diagnostics = {
        "score": float(corridor_score.detach().item()),
        "snapshot_min": float(diagnostics["snapshot_survival"].detach().min().item()),
        "snapshot_mean": float(
            diagnostics["snapshot_survival"].detach().mean().item()
        ),
        "max_high_load_ratio": float(
            (
                diagnostics["high_load"]
                / effective_capacities.to(dtype=torch.float32)
            )
            .detach()
            .max()
            .item()
        ),
    }
    return (
        (
            loss_fh,
            loss_uh,
            loss_fhm_augmented,
            loss_uhm,
            loss_fhml,
            loss_uhml,
        ),
        (value_fh, value_uh, value_fhm, value_uhm, value_fhml, value_uhml),
    )


full.build_objectives = build_objectives


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "persistent_runner.py": PERSISTENT_RUNNER,
        "ordered_projection.py": (
            TEST_DIR / "shared2x_full_objectives" / "ordered_projection.py"
        ),
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
    }
    hashes = {name: full.sha256(path) for name, path in paths.items()}
    settings = json.dumps(
        {
            "corridor_weight": CORRIDOR_WEIGHT,
            "edge_temperature": EDGE_TEMPERATURE,
            "path_temperature": PATH_TEMPERATURE,
            "max_residual_ratio": MAX_RESIDUAL_RATIO,
            "objective_slot": "blend into Fhm after Fh and Uh",
            "policy_architecture_change": False,
            "inference_cost": 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["medium_corridor_survival/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


full.source_hashes = source_hashes


def method_description() -> dict:
    return {
        "method": "Hattrick Medium Corridor Survival (MCS)",
        "primary_target": "Medium",
        "architecture": "unchanged persistent serial Hattrick policy",
        "projection": (
            "unchanged six-objective ordered projection; MCS is blended into Fhm, "
            "whose gradient is projected against prior Fh and Uh"
        ),
        "training_signal": (
            "actual training High admission and Medium demand supervise a policy "
            "whose inputs remain strict ESM predictions"
        ),
        "formula": (
            "residual=C-PTE^T admitted_H; path=softmin_edges(residual/d_M); "
            "OD=softmax_paths(smooth_min(path,1)); maximize demand-weighted OD score"
        ),
        "corridor_weight": CORRIDOR_WEIGHT,
        "edge_temperature": EDGE_TEMPERATURE,
        "path_temperature": PATH_TEMPERATURE,
        "max_residual_ratio": MAX_RESIDUAL_RATIO,
        "new_parameters": 0,
        "deployment_extra_ops": 0,
        "strict_inference": (
            "unchanged: ESM predictions/topology/capacities/masks only; MCS is absent "
            "from eval/inference"
        ),
        "source_sha256": source_hashes(),
    }


def main() -> None:
    global CORRIDOR_WEIGHT, EDGE_TEMPERATURE, PATH_TEMPERATURE
    parser = argparse.ArgumentParser(
        description="Training-only Medium corridor survival for persistent Hattrick"
    )
    parser.add_argument("--level", type=int, choices=(1, 2), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--edge-temperature", type=float, default=0.05)
    parser.add_argument("--path-temperature", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.weight < 0.0:
        raise ValueError("Weight must be nonnegative")
    CORRIDOR_WEIGHT = float(args.weight)
    EDGE_TEMPERATURE = float(args.edge_temperature)
    PATH_TEMPERATURE = float(args.path_temperature)

    tag = (
        f"w{CORRIDOR_WEIGHT:g}_te{EDGE_TEMPERATURE:g}_tp{PATH_TEMPERATURE:g}"
        .replace(".", "p")
    )
    full.OUTPUT_ROOT = THIS_DIR / "artifacts" / tag
    run_dir = full.run_one(args.level, args.seed, force=args.force)
    description = method_description()
    description.update(
        {"level": args.level, "seed": args.seed, "run_directory": str(run_dir)}
    )
    full.write_json(run_dir / "medium_corridor_survival_method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
