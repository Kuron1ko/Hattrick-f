from __future__ import annotations

"""Distill only DOTE's Medium-critical residual-capacity signature.

The student never matches a DOTE path distribution.  A frozen DOTE model is
used offline during training to produce its High edge load from the same ESM
predictions.  Student and teacher are compared only on the 72-dimensional
post-High residual-capacity vector, weighted by the predicted Medium candidate
footprint.  Inference remains the unchanged serial persistent Hattrick model.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
PERSISTENT_RUNNER = (
    TEST_DIR / "shared2x_sparse_path_cross_attention_persistent" / "run_experiment.py"
)
DOTE_RUNNER = TEST_DIR / "run_dotemc_priority_mask_experiment.py"
DOTE_CHECKPOINT = (
    TEST_DIR
    / "results_load2x_retrain_shared_strict"
    / "outputs"
    / "shared"
    / "w_1_0p1_0p01"
    / "best_model.pt"
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


persistent = load_module("dote_signature_persistent_runtime", PERSISTENT_RUNNER)
dote_runtime = load_module("dote_signature_teacher_runtime", DOTE_RUNNER)
base = persistent.parent.full
native_build_objectives = base.build_objectives

AUX_WEIGHT = 1.0
OBJECTIVE_NAMES = ("Fh", "Sd", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
_teacher_cache: dict[str, object] = {}


def _teacher(device: torch.device):
    key = str(device)
    if key not in _teacher_cache:
        model, mean, std = dote_runtime.load_checkpoint(DOTE_CHECKPOINT, device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _teacher_cache[key] = (model, mean, std)
    return _teacher_cache[key]


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


def edge_load(policy: torch.Tensor, demand: torch.Tensor, pte: torch.Tensor):
    policy = policy.squeeze(-1)
    path_flow = policy * demand.squeeze(-1)
    return torch.sparse.mm(
        pte.to(dtype=torch.float32).t(), path_flow.to(dtype=torch.float32).t()
    ).t()


def signature_loss(model, props, dataset, values, path_masks):
    (
        _node_features,
        capacities,
        _tm1,
        tm1_pred,
        _tm2,
        tm2_pred,
        _tm3,
        tm3_pred,
        *_rest,
    ) = values
    student_high, _student_medium, _student_low = emitted_policy(
        model, props, dataset, values, path_masks
    )
    batch_size, total_paths, _ = student_high.shape
    paths_per_od = int(props.num_paths_per_pair)
    num_ods = total_paths // paths_per_od
    pte = dataset.pte.coalesce()
    capacities = capacities.to(dtype=torch.float32)
    if capacities.shape[0] == 1 and batch_size > 1:
        capacities = capacities.expand(batch_size, -1)

    teacher, mean, std = _teacher(tm1_pred.device)
    with torch.no_grad():
        features = dote_runtime.make_inputs(
            tm1_pred.float(),
            tm2_pred.float(),
            tm3_pred.float(),
            num_ods,
            mean,
            std,
        )
        teacher_high = teacher(features, None)[:, 0].reshape(
            batch_size, total_paths, 1
        )

    student_high_load = edge_load(student_high, tm1_pred, pte)
    teacher_high_load = edge_load(teacher_high, tm1_pred, pte)
    student_residual_fraction = 1.0 - student_high_load / capacities.clamp_min(1e-6)
    teacher_residual_fraction = 1.0 - teacher_high_load / capacities.clamp_min(1e-6)

    # The Medium candidate footprint is deliberately policy-free: every feasible
    # candidate receives equal mass within its OD.  It identifies which residual
    # edges matter to Medium without importing DOTE's Medium route choice.
    medium_mask = model.path_mask_for_class(path_masks, 1)
    if medium_mask is None:
        medium_mask = torch.ones(total_paths, dtype=torch.bool, device=pte.device)
    medium_mask = medium_mask.reshape(num_ods, paths_per_od).to(dtype=torch.bool)
    counts = medium_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    uniform = (
        medium_mask.to(dtype=torch.float32) / counts.to(dtype=torch.float32)
    ).reshape(1, total_paths)
    medium_path_flow = uniform * tm2_pred.squeeze(-1).to(dtype=torch.float32)
    medium_edge_footprint = torch.sparse.mm(
        pte.to(dtype=torch.float32).t(), medium_path_flow.t()
    ).t()
    criticality = medium_edge_footprint / capacities.clamp_min(1e-6)
    criticality = criticality / criticality.mean(dim=1, keepdim=True).clamp_min(1e-6)

    per_edge = F.smooth_l1_loss(
        student_residual_fraction,
        teacher_residual_fraction,
        beta=0.02,
        reduction="none",
    )
    raw = (criticality.detach() * per_edge).mean()
    loss = AUX_WEIGHT * raw / raw.detach().clamp_min(1e-6)
    cosine = F.cosine_similarity(
        student_residual_fraction,
        teacher_residual_fraction,
        dim=1,
    ).mean()
    model._last_dote_signature_diagnostics = {
        "reported_Sd": float(raw.detach().item()),
        "reported_signature_cosine": float(cosine.detach().item()),
        "reported_student_predicted_high_mlu": float(
            (student_high_load / capacities).amax(dim=1).mean().detach().item()
        ),
        "reported_teacher_predicted_high_mlu": float(
            (teacher_high_load / capacities).amax(dim=1).mean().detach().item()
        ),
    }
    return loss, model._last_dote_signature_diagnostics


def build_objectives(model, props, dataset, values, path_masks):
    native_losses, native_reported = native_build_objectives(
        model, props, dataset, values, path_masks
    )
    distill_loss, diagnostics = signature_loss(model, props, dataset, values, path_masks)
    losses = (
        native_losses[0],
        distill_loss,
        native_losses[1],
        *native_losses[2:],
    )
    reported = (
        native_reported[0],
        diagnostics["reported_Sd"],
        native_reported[1],
        *native_reported[2:],
    )
    return losses, reported


def source_hashes() -> dict[str, str]:
    paths = {
        "run_dote_signature_experiment.py": Path(__file__).resolve(),
        "persistent_runner.py": PERSISTENT_RUNNER,
        "dote_runner.py": DOTE_RUNNER,
        "dote_checkpoint": DOTE_CHECKPOINT,
        "ordered_projection.py": TEST_DIR / "shared2x_full_objectives" / "ordered_projection.py",
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
    }
    hashes = {name: base.sha256(path) for name, path in paths.items()}
    settings = json.dumps(
        {
            "aux_weight": AUX_WEIGHT,
            "objective_order": OBJECTIVE_NAMES,
            "target": "Medium-critical post-High residual-capacity vector",
            "path_policy_distillation": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["dote_signature/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


def configure(aux_weight: float) -> None:
    global AUX_WEIGHT
    AUX_WEIGHT = float(aux_weight)
    tag = f"w{AUX_WEIGHT:g}".replace(".", "p")
    base.OBJECTIVE_NAMES = OBJECTIVE_NAMES
    base.build_objectives = build_objectives
    base.source_hashes = source_hashes
    base.OUTPUT_ROOT = THIS_DIR / "artifacts_dote_signature" / tag


def main() -> None:
    parser = argparse.ArgumentParser(description="DOTE residual-signature projection")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--aux-weight", type=float, default=1.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configure(args.aux_weight)
    run_dir = base.run_one(args.level, args.seed, force=args.force)
    method = {
        "method": "DOTE Medium-critical residual signature projection",
        "objective_order": list(OBJECTIVE_NAMES),
        "path_policy_distillation": False,
        "teacher_used_at_inference": False,
        "strict_esm_teacher_and_student": True,
        "aux_weight": AUX_WEIGHT,
        "source_sha256": source_hashes(),
    }
    base.write_json(run_dir / "method.json", method)
    print(json.dumps(method, indent=2), flush=True)


if __name__ == "__main__":
    main()
