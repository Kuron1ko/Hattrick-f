from __future__ import annotations

"""Dense Medium-survivability projection for the persistent Hattrick model.

This is an isolated development runner.  It does not edit the frozen persistent
runner/checkpoints.  The new objective is evaluated from the *emitted High
policy* and ESM predictions only:

  residual edge capacity after predicted High
      -> smooth bottleneck for every feasible Medium candidate path
      -> smooth best-path envelope for every Medium OD.

The objective is placed immediately after Fh in ordered projection.  Hence it
may move High inside the local equal-Fh face before Uh is optimized; it does not
name or freeze a High path and it is not protected from the primary Fh loss.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
PERSISTENT_RUNNER = (
    TEST_DIR / "shared2x_sparse_path_cross_attention_persistent" / "run_experiment.py"
)
for item in (str(ROOT), str(TEST_DIR), str(PERSISTENT_RUNNER.parent)):
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


persistent = load_module("medium_survival_persistent_runtime", PERSISTENT_RUNNER)
base = persistent.parent.full
native_build_objectives = base.build_objectives

EDGE_TAU = 0.10
PATH_TAU = 0.20
AUX_WEIGHT = 1.0
OBJECTIVE_NAMES = ("Fh", "Sm", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")


def _incidence(model, pte: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (int(pte.shape[0]), int(pte.shape[1]))
    cached = getattr(model, "_medium_survival_incidence", None)
    if cached is None or tuple(cached.shape) != shape or cached.device != pte.device:
        cached = pte.to_dense().to(dtype=torch.bool)
        # A plain non-parameter cache keeps checkpoints architecture-compatible.
        model._medium_survival_incidence = cached
        model._medium_survival_path_lengths = cached.sum(dim=-1).clamp_min(1)
    return cached, model._medium_survival_path_lengths


def emitted_policy(model, props, dataset, values, path_masks):
    previous = bool(getattr(props, "research_return_policy", False))
    props.research_return_policy = True
    try:
        policies, _ = base.shared.model_forward(
            model, props, dataset, values, path_masks
        )
    finally:
        props.research_return_policy = previous
    return policies


def medium_survivability_loss(model, props, dataset, values, path_masks):
    (
        _node_features,
        capacities,
        _tm1,
        tm1_pred,
        _tm2,
        tm2_pred,
        *_rest,
    ) = values
    high_policy, _medium_policy, _low_policy = emitted_policy(
        model, props, dataset, values, path_masks
    )
    high_policy = high_policy.squeeze(-1)
    batch_size, total_paths = high_policy.shape
    paths_per_od = int(props.num_paths_per_pair)
    if total_paths % paths_per_od:
        raise RuntimeError("Path count is not divisible by paths per OD")
    num_ods = total_paths // paths_per_od

    pte = dataset.pte.coalesce()
    high_path_flow = high_policy * tm1_pred.squeeze(-1)
    high_edge_load = torch.sparse.mm(
        pte.to(dtype=torch.float32).t(), high_path_flow.to(dtype=torch.float32).t()
    ).t()
    edge_capacity = capacities
    if edge_capacity.shape[0] == 1 and batch_size > 1:
        edge_capacity = edge_capacity.expand(batch_size, -1)
    if tuple(edge_capacity.shape) != tuple(high_edge_load.shape):
        raise RuntimeError(
            f"Capacity/load mismatch: {tuple(edge_capacity.shape)} / "
            f"{tuple(high_edge_load.shape)}"
        )
    residual = edge_capacity.to(dtype=torch.float32) - high_edge_load

    demand = tm2_pred.squeeze(-1).reshape(batch_size, num_ods, paths_per_od).mean(
        dim=-1
    )
    demand = demand.to(dtype=torch.float32).clamp_min(1e-6)
    demand_per_path = demand.repeat_interleave(paths_per_od, dim=1)

    incidence, path_lengths = _incidence(model, pte)
    residual_ratio = residual.unsqueeze(1) / demand_per_path.unsqueeze(-1)
    edge_logits = (-residual_ratio / EDGE_TAU).masked_fill(
        ~incidence.reshape(1, total_paths, -1), -torch.inf
    )
    # Normalized smooth minimum avoids rewarding shorter candidate paths merely
    # because they contain fewer terms in logsumexp.
    path_bottleneck = -EDGE_TAU * (
        torch.logsumexp(edge_logits, dim=-1)
        - path_lengths.to(dtype=edge_logits.dtype).log().reshape(1, total_paths)
    )

    medium_mask = model.path_mask_for_class(path_masks, 1)
    if medium_mask is None:
        medium_mask = torch.ones(total_paths, dtype=torch.bool, device=pte.device)
    medium_mask = medium_mask.reshape(num_ods, paths_per_od).to(dtype=torch.bool)
    feasible_count = medium_mask.sum(dim=-1)
    if bool((feasible_count == 0).any().item()):
        raise RuntimeError("A Medium OD has no feasible candidate path")
    path_bottleneck = path_bottleneck.reshape(batch_size, num_ods, paths_per_od)
    path_logits = (path_bottleneck / PATH_TAU).masked_fill(
        ~medium_mask.reshape(1, num_ods, paths_per_od), -torch.inf
    )
    # log-mean-exp is a smooth best-path envelope with an option-count bonus:
    # two comparable surviving paths score above one isolated survivor.
    od_envelope = PATH_TAU * (
        torch.logsumexp(path_logits, dim=-1)
        - feasible_count.to(dtype=path_logits.dtype).log().reshape(1, num_ods)
    )
    demand_weight = demand.detach() / demand.detach().mean(dim=1, keepdim=True)
    score_by_sample = (demand_weight * od_envelope).mean(dim=1)
    raw_score = score_by_sample.mean()
    # Match Hattrick's existing unit-value/self-normalized objective convention.
    loss = -AUX_WEIGHT * raw_score / raw_score.detach().abs().clamp_min(1e-6)
    diagnostics = {
        "reported_Sm": float(raw_score.detach().item()),
        "reported_Sm_min": float(score_by_sample.detach().min().item()),
        "reported_Sm_max": float(score_by_sample.detach().max().item()),
        "reported_high_predicted_mlu": float(
            (high_edge_load / edge_capacity.clamp_min(1e-6)).amax(dim=1).mean().detach().item()
        ),
    }
    return loss, diagnostics


def build_objectives(model, props, dataset, values, path_masks):
    native_losses, native_reported = native_build_objectives(
        model, props, dataset, values, path_masks
    )
    survival_loss, diagnostics = medium_survivability_loss(
        model, props, dataset, values, path_masks
    )
    losses = (
        native_losses[0],
        survival_loss,
        native_losses[1],
        *native_losses[2:],
    )
    # The base trainer expects one reported scalar for every objective.  Detailed
    # score extrema are retained on the model for smoke/audit scripts.
    model._last_medium_survival_diagnostics = diagnostics
    reported = (
        native_reported[0],
        diagnostics["reported_Sm"],
        native_reported[1],
        *native_reported[2:],
    )
    return losses, reported


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "persistent_runner.py": PERSISTENT_RUNNER,
        "ordered_projection.py": TEST_DIR / "shared2x_full_objectives" / "ordered_projection.py",
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
    }
    hashes = {name: base.sha256(path) for name, path in paths.items()}
    settings = json.dumps(
        {
            "edge_tau": EDGE_TAU,
            "path_tau": PATH_TAU,
            "aux_weight": AUX_WEIGHT,
            "objective_order": OBJECTIVE_NAMES,
            "teacher": None,
            "policy_traffic": "ESM prediction",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["medium_survival/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


def configure(aux_weight: float, edge_tau: float, path_tau: float) -> None:
    global AUX_WEIGHT, EDGE_TAU, PATH_TAU
    AUX_WEIGHT = float(aux_weight)
    EDGE_TAU = float(edge_tau)
    PATH_TAU = float(path_tau)
    tag = (
        f"w{AUX_WEIGHT:g}_te{EDGE_TAU:g}_tp{PATH_TAU:g}"
        .replace(".", "p")
    )
    base.OBJECTIVE_NAMES = OBJECTIVE_NAMES
    base.build_objectives = build_objectives
    base.source_hashes = source_hashes
    base.OUTPUT_ROOT = THIS_DIR / "artifacts" / tag


def main() -> None:
    parser = argparse.ArgumentParser(description="Medium-survivability projection")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--aux-weight", type=float, default=1.0)
    parser.add_argument("--edge-tau", type=float, default=0.10)
    parser.add_argument("--path-tau", type=float, default=0.20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configure(args.aux_weight, args.edge_tau, args.path_tau)
    run_dir = base.run_one(args.level, args.seed, force=args.force)
    method = {
        "method": "Medium candidate-survivability projection",
        "objective_order": list(OBJECTIVE_NAMES),
        "mechanism": (
            "Fh -> ESM residual Medium path envelope -> Uh -> Fhm -> Uhm -> "
            "Fhml -> Uhml"
        ),
        "does_not_fix_high_path": True,
        "strict_inference_unchanged": True,
        "teacher": None,
        "edge_tau": EDGE_TAU,
        "path_tau": PATH_TAU,
        "aux_weight": AUX_WEIGHT,
        "source_sha256": source_hashes(),
    }
    base.write_json(run_dir / "method.json", method)
    print(json.dumps(method, indent=2), flush=True)


if __name__ == "__main__":
    main()
