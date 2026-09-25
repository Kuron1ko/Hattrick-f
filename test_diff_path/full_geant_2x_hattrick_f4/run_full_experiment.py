from __future__ import annotations

"""Train Hattrick-f4 on the complete GEANT 2x split.

Phase A is the validation-selected original six-loss Hattrick checkpoint.
Phase B removes MLU and uses Medium's ESM demand to build a training-only,
active-set crossover target for High's emitted path policy.  The objective
order is High-tail -> Fh -> crossover -> Fhm -> Fhml.  Checkpoints contain an
ordinary Hattrick state dict; no crossover code runs during inference.

Ranges are fixed: train [0,6000), validation [6000,7500), test [7500,10200).
This trainer never instantiates the test split.  Validation selects top-K,
resume state is saved each epoch, and archives are saved every five epochs by
default.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

# The dedicated 2x traffic snapshots were serialized by NumPy 2.x.  Keep them
# readable when the runner is executed in the repository's NumPy 1.x runtime.
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
INFRA_PATH = TEST_DIR / "full_geant_2x_hattrick_f3" / "run_full_experiment.py"
DEFAULT_PHASE_A = (
    TEST_DIR / "full_geant_2x_hattrick_f" / "artifacts" / "hattrick"
    / "seed_490" / "best_model.pt"
)
ARTIFACT_ROOT = THIS_DIR / "artifacts"
TRAIN_RANGE = (0, 6000)
VALIDATION_RANGE = (6000, 7500)
TEST_RANGE = (7500, 10200)
OBJECTIVE_NAMES = ("Htail", "Fh", "Cx", "Fhm", "Fhml")
DEFAULT_EPOCHS = 40
DEFAULT_TOP_K = 5
DEFAULT_SAVE_EVERY = 5
DEFAULT_HIGH_FLOOR = 0.995
DEFAULT_CROSSOVER_STEP = 0.35
DEFAULT_EDGE_TAU = 0.10
DEFAULT_PATH_TAU = 0.20
DEFAULT_ACTIVATION_MARGIN = 0.05


def load_infrastructure():
    spec = importlib.util.spec_from_file_location(
        "full_geant_2x_hattrick_f4_infrastructure", INFRA_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import full-data infrastructure: {INFRA_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


infra = load_infrastructure()
base = infra.base


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def forward_components(model, shared, props, dataset, values, path_masks):
    previous = bool(getattr(props, "research_return_policy", False))
    props.research_return_policy = False
    try:
        output, _ = shared.model_forward(model, props, dataset, values, path_masks)
    finally:
        props.research_return_policy = previous
    (
        _edges_high, _edges_high_medium, _edges_all, _edges_high_final,
        _edges_high_medium_final, all_traffic, admitted_high, admitted_medium,
        _admitted_low,
    ) = output
    return {
        "all_traffic": all_traffic,
        "admitted_high": admitted_high,
        "admitted_medium": admitted_medium,
        "opt1_mf": values[11],
        "opt2_mf": values[12],
        "opt3_mf": values[13],
    }


def emitted_policy(model, shared, props, dataset, values, path_masks):
    previous = bool(getattr(props, "research_return_policy", False))
    props.research_return_policy = True
    try:
        policies, _ = shared.model_forward(model, props, dataset, values, path_masks)
    finally:
        props.research_return_policy = previous
    return policies


def class_mask(model, path_masks, class_index: int, total_paths: int, device):
    mask = model.path_mask_for_class(path_masks, class_index)
    if mask is None:
        return torch.ones(total_paths, dtype=torch.bool, device=device)
    mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
    if mask.numel() != total_paths:
        raise RuntimeError(
            f"Class-{class_index} mask has {mask.numel()} entries; "
            f"expected {total_paths}"
        )
    return mask


def high_tail_loss(admitted_high, oracle_high, floor: float):
    batch_size = admitted_high.shape[0]
    admitted = admitted_high.reshape(batch_size, -1).sum(dim=1)
    oracle = oracle_high.detach().reshape(batch_size, -1).sum(dim=1)
    ratio = torch.where(
        oracle > 1e-12,
        admitted / oracle.clamp_min(1e-12),
        torch.ones_like(admitted),
    )
    violation = torch.relu(float(floor) - ratio)
    return violation.mean(), {
        "reported_high_ratio_mean": float(ratio.detach().mean().item()),
        "reported_high_ratio_min": float(ratio.detach().min().item()),
        "high_tail_active_fraction": float(
            (violation.detach() > 0).float().mean().item()
        ),
        "high_tail_violation_mean": float(violation.detach().mean().item()),
        "high_tail_violation_max": float(violation.detach().max().item()),
    }


def _incidence(model, pte: torch.Tensor):
    shape = (int(pte.shape[0]), int(pte.shape[1]))
    cached = getattr(model, "_f4_path_edge_incidence", None)
    if cached is None or tuple(cached.shape) != shape or cached.device != pte.device:
        cached = pte.to_dense().to(dtype=torch.bool)
        model._f4_path_edge_incidence = cached
        model._f4_path_lengths = cached.sum(dim=-1).clamp_min(1)
    return cached, model._f4_path_lengths


def active_set_crossover_loss(
    *, model, high_policy, capacities, tm1_pred, tm2_pred, pte, path_masks,
    paths_per_od: int, crossover_step: float, edge_tau: float, path_tau: float,
    activation_margin: float,
):
    """Create a stopped Medium-selected crossover target for High paths."""
    high = high_policy.squeeze(-1)
    batch_size, total_paths = high.shape
    if total_paths % paths_per_od:
        raise RuntimeError("Path count is not divisible by paths per OD")
    num_ods = total_paths // paths_per_od
    pte = pte.coalesce()
    incidence, path_lengths = _incidence(model, pte)
    high_mask = class_mask(model, path_masks, 0, total_paths, high.device)
    medium_mask = class_mask(model, path_masks, 1, total_paths, high.device)
    high_mask_od = high_mask.reshape(num_ods, paths_per_od)
    medium_mask_od = medium_mask.reshape(num_ods, paths_per_od)
    if bool((high_mask_od.sum(dim=1) == 0).any().item()):
        raise RuntimeError("A High OD has no feasible candidate path")
    if bool((medium_mask_od.sum(dim=1) == 0).any().item()):
        raise RuntimeError("A Medium OD has no feasible candidate path")

    with torch.no_grad():
        high_detached = high.detach().float()
        high_path_flow = high_detached * tm1_pred.detach().squeeze(-1).float()
        high_edge_load = torch.sparse.mm(pte.float().t(), high_path_flow.t()).t()
        edge_capacity = capacities.detach().float()
        if edge_capacity.shape[0] == 1 and batch_size > 1:
            edge_capacity = edge_capacity.expand(batch_size, -1)
        if tuple(edge_capacity.shape) != tuple(high_edge_load.shape):
            raise RuntimeError(
                f"Capacity/load mismatch: {tuple(edge_capacity.shape)} / "
                f"{tuple(high_edge_load.shape)}"
            )
        residual_fraction = (
            edge_capacity - high_edge_load
        ) / edge_capacity.clamp_min(1e-6)

        # Soft active edge of each Medium path, followed by its soft best paths.
        edge_logits = (-residual_fraction.unsqueeze(1) / edge_tau).masked_fill(
            ~incidence.reshape(1, total_paths, -1), -torch.inf
        )
        edge_active = torch.softmax(edge_logits, dim=-1)
        path_bottleneck = -edge_tau * (
            torch.logsumexp(edge_logits, dim=-1)
            - path_lengths.float().log().reshape(1, total_paths)
        )
        bottleneck_od = path_bottleneck.reshape(batch_size, num_ods, paths_per_od)
        medium_logits = (bottleneck_od / path_tau).masked_fill(
            ~medium_mask_od.reshape(1, num_ods, paths_per_od), -torch.inf
        )
        medium_path_active = torch.softmax(medium_logits, dim=-1)
        medium_demand = (
            tm2_pred.detach().squeeze(-1)
            .reshape(batch_size, num_ods, paths_per_od).mean(dim=-1).float()
        )
        demand_weight = medium_demand / medium_demand.mean(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        edge_active_od = edge_active.reshape(
            batch_size, num_ods, paths_per_od, -1
        )
        edge_pressure = (
            demand_weight.unsqueeze(-1).unsqueeze(-1)
            * medium_path_active.unsqueeze(-1) * edge_active_od
        ).sum(dim=(1, 2))

        # Paths crossing Medium's active edges get a larger crossover price.
        high_path_cost = torch.matmul(edge_pressure, incidence.float().t())
        high_cost_od = high_path_cost.reshape(batch_size, num_ods, paths_per_od)
        mask3 = high_mask_od.reshape(1, num_ods, paths_per_od)
        masked_min = high_cost_od.masked_fill(~mask3, torch.inf).amin(
            dim=-1, keepdim=True
        )
        masked_max = high_cost_od.masked_fill(~mask3, -torch.inf).amax(
            dim=-1, keepdim=True
        )
        spread = (masked_max - masked_min).clamp_min(0.0)
        mean_abs = (
            high_cost_od.abs().masked_fill(~mask3, 0.0).sum(dim=-1, keepdim=True)
            / high_mask_od.sum(dim=-1).reshape(1, num_ods, 1).clamp_min(1)
        )
        relative_spread = spread / mean_abs.clamp_min(1e-6)
        active_od = relative_spread.squeeze(-1) >= activation_margin
        normalized_cost = (high_cost_od - masked_min) / spread.clamp_min(1e-6)
        current_od = high_detached.reshape(batch_size, num_ods, paths_per_od)
        target_logits = (
            torch.log(current_od.clamp_min(1e-9))
            - crossover_step * normalized_cost
        ).masked_fill(~mask3, -torch.inf)
        target = torch.softmax(target_logits, dim=-1)
        target = torch.where(active_od.unsqueeze(-1), target, current_od)

    current = high.reshape(batch_size, num_ods, paths_per_od).clamp_min(1e-9)
    per_od = -(target * torch.log(current)).sum(dim=-1)
    loss = per_od.masked_select(active_od).mean() if bool(active_od.any()) else high.sum() * 0.0
    current_entropy = -(current.detach() * torch.log(current.detach())).sum(dim=-1)
    target_entropy = -(target * torch.log(target.clamp_min(1e-9))).sum(dim=-1)
    l1 = (target - current.detach()).abs().sum(dim=-1)
    active_float = active_od.float()
    active_count = active_float.sum().clamp_min(1.0)
    diagnostics = {
        "crossover_active_od_fraction": float(active_float.mean().item()),
        "crossover_target_l1_active": float(
            (l1 * active_float).sum().item() / active_count.item()
        ),
        "crossover_current_entropy_active": float(
            (current_entropy * active_float).sum().item() / active_count.item()
        ),
        "crossover_target_entropy_active": float(
            (target_entropy * active_float).sum().item() / active_count.item()
        ),
        "crossover_relative_spread_mean": float(relative_spread.mean().item()),
        "predicted_high_mlu_mean": float(
            (high_edge_load / edge_capacity.clamp_min(1e-6)).amax(dim=1).mean().item()
        ),
    }
    return loss, diagnostics


def train_epoch(*, shared, full, model, props, dataset, loader, optimizer, args):
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    props.research_return_policy = False
    path_masks = shared.base.move_dataset_static(dataset, props.device)
    totals: dict[str, float] = {}
    first_probe: dict[str, float] = {}
    count = 0
    for batch_index, inputs in enumerate(loader):
        values = shared.unpack_to_device(inputs, props)
        components = forward_components(model, shared, props, dataset, values, path_masks)
        high_policy, _medium_policy, _low_policy = emitted_policy(
            model, shared, props, dataset, values, path_masks
        )
        loss_tail, tail_diag = high_tail_loss(
            components["admitted_high"], components["opt1_mf"], args.high_floor
        )
        loss_fh, value_fh = full.loss_mf(
            components["admitted_high"], components["opt1_mf"].detach()
        )
        loss_cx, cx_diag = active_set_crossover_loss(
            model=model, high_policy=high_policy, capacities=values[1],
            tm1_pred=values[3], tm2_pred=values[5], pte=dataset.pte,
            path_masks=path_masks, paths_per_od=int(props.num_paths_per_pair),
            crossover_step=args.crossover_step, edge_tau=args.edge_tau,
            path_tau=args.path_tau, activation_margin=args.activation_margin,
        )
        loss_fhm, value_fhm = full.loss_mf(
            components["admitted_high"] + components["admitted_medium"],
            components["opt2_mf"].detach(),
        )
        loss_fhml, value_fhml = full.loss_mf(
            components["all_traffic"], components["opt3_mf"].detach()
        )
        losses = (loss_tail, loss_fh, loss_cx, loss_fhm, loss_fhml)
        if any(not torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError("Non-finite Hattrick-f4 objective")
        projected = full.ordered_project_gradients(model, losses)
        if batch_index == 0:
            first_probe = full.projection_diagnostics(projected, OBJECTIVE_NAMES)
        full.assign_gradients_and_step(
            model, projected.final_gradient, optimizer, projected.parameter_shapes
        )
        diagnostics = {
            "reported_Fh": float(value_fh),
            "reported_Fhm": float(value_fhm),
            "reported_Fhml": float(value_fhml),
            **tail_diag, **cx_diag,
        }
        for name, value in diagnostics.items():
            totals[name] = totals.get(name, 0.0) + float(value)
        count += 1
    props.research_return_admitted = False
    props.research_return_policy = False
    result = {name: value / max(count, 1) for name, value in totals.items()}
    result.update(first_probe)
    result["train_batches"] = count
    return result


def source_hashes() -> dict[str, str]:
    paths = {
        "hattrick_f4_runner.py": Path(__file__).resolve(),
        "full_data_infrastructure.py": INFRA_PATH,
        "full_geant_base_runner.py": infra.BASE_RUNNER_PATH,
        "six_objective_runtime.py": base.FULL_SOURCE,
        "ordered_projection.py": base.FULL_SOURCE.parent / "ordered_projection.py",
        "hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "training_utils.py": ROOT / "utils" / "training_utils.py",
        "dataset.py": ROOT / "utils" / "build_dataset_within_cluster.py",
    }
    return {name: base.sha256(path) for name, path in paths.items()}


def experiment_config(args, phase_a: Path) -> dict:
    return {
        "method": "Hattrick-f4", "seed": args.seed,
        "topology": base.TARGET_TOPOLOGY, "load_factor": 2.0,
        "strict_esm": True, "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE), "test": list(TEST_RANGE),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "low_budget": args.low_budget,
        "top_k": args.top_k, "save_every": args.save_every,
        "phase_a_checkpoint": str(phase_a.resolve()),
        "phase_a_sha256": base.sha256(phase_a),
        "optimizer_reset": True, "all_parameters_trainable": True,
        "f4": {
            "kind": "training-only central-path active-set crossover",
            "objective_order": list(OBJECTIVE_NAMES),
            "inference_architecture_changed": False,
            "policy_traffic": "ESM predictions only",
            "corrector_traffic": "training actual traffic and Oracle labels",
            "high_floor": args.high_floor,
            "crossover_step": args.crossover_step,
            "edge_temperature": args.edge_tau,
            "path_temperature": args.path_tau,
            "activation_relative_spread": args.activation_margin,
            "target_gradient": "stopped",
        },
        "selection": (
            "validation only: strict High per-snapshot floor, Low mean budget, "
            "simulator safety; then Medium mean/P10/P1, High mean, Low mean"
        ),
        "source_sha256": source_hashes(),
    }


def run_directory(seed: int) -> Path:
    return ARTIFACT_ROOT / f"seed_{seed}"


def check_inputs(args) -> None:
    phase_a = args.phase_a_checkpoint.resolve()
    if not phase_a.exists():
        raise FileNotFoundError(f"Missing Phase-A Hattrick checkpoint: {phase_a}")
    if not base.final_oracle_complete():
        raise RuntimeError("The full 2x Oracle/data is incomplete")
    checkpoint = torch.load(phase_a, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "Hattrick":
        raise RuntimeError(f"Phase-A checkpoint is not Hattrick: {phase_a}")
    report = {
        "status": "READY", "method": "Hattrick-f4",
        "phase_a_epoch": int(checkpoint["epoch"]),
        "phase_a_sha256": base.sha256(phase_a),
        "train": list(TRAIN_RANGE), "validation": list(VALIDATION_RANGE),
        "test_reserved_for_unified_inference": list(TEST_RANGE),
        "strict_esm": True, "epochs": args.epochs,
        "save_every": args.save_every, "top_k": args.top_k,
        "training_only_crossover": {
            "high_floor": args.high_floor, "step": args.crossover_step,
            "edge_tau": args.edge_tau, "path_tau": args.path_tau,
            "activation_margin": args.activation_margin,
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def train(args) -> None:
    check_inputs(args)
    phase_a_path = args.phase_a_checkpoint.resolve()
    run_dir = run_directory(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = experiment_config(args, phase_a_path)
    config_sha256 = canonical_json_sha256(config)
    config_path = run_dir / "config.json"
    resume_path = run_dir / "resume_state.pt"
    if config_path.exists():
        if base.read_json(config_path) != config:
            raise RuntimeError(f"Existing Hattrick-f4 configuration differs: {config_path}")
    else:
        base.write_json(config_path, config)
    if resume_path.exists() and not args.resume:
        raise RuntimeError(
            f"Existing resume state found: {resume_path}. Use --resume to continue."
        )
    if args.resume and not resume_path.exists():
        raise FileNotFoundError(f"--resume requested but no state exists: {resume_path}")

    shared, full, _hattrick_f = base.load_runtimes()
    from frameworks.hattrick_system import Hattrick
    from utils.AdamOptimizer import ADAMOptimizer

    device = infra.resolve_device(args.device)
    base.set_seed(args.seed)
    props = base.build_props(
        device, batch_size=args.batch_size, epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    train_dataset, validation_dataset, test_dataset = base.make_datasets(props)
    if test_dataset is not None:
        raise RuntimeError("Training unexpectedly instantiated the test split")
    phase_a = torch.load(phase_a_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(phase_a["model_state_dict"])
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    baseline_rows, baseline_summary, baseline_diag = shared.evaluate(
        model, props, validation_dataset, VALIDATION_RANGE[0]
    )
    base.write_csv(run_dir / "phase_a_validation_metrics.csv", baseline_rows)
    base.write_json(
        run_dir / "phase_a_validation_summary.json",
        {"classes": baseline_summary, "diagnostics": baseline_diag},
    )
    history_path = run_dir / "train_history.csv"
    history = base.read_csv(history_path)
    top_entries = infra.load_top_k(run_dir)
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("method") != "Hattrick-f4":
            raise RuntimeError("Resume-state method is not Hattrick-f4")
        if int(checkpoint.get("seed", -1)) != args.seed:
            raise RuntimeError("Resume-state seed mismatch")
        if checkpoint.get("config") != config:
            raise RuntimeError("Resume-state configuration mismatch")
        if checkpoint.get("config_sha256") != config_sha256:
            raise RuntimeError("Resume-state configuration hash mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[resume] Hattrick-f4 at epoch {start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        base.set_seed(args.seed + epoch * 104729)
        loader = shared.data_loader(
            train_dataset, props.batch_size, True, args.seed + epoch * 1009
        )
        train_metrics = train_epoch(
            shared=shared, full=full, model=model, props=props,
            dataset=train_dataset, loader=loader, optimizer=optimizer, args=args,
        )
        rows, summary, diagnostics = shared.evaluate(
            model, props, validation_dataset, VALIDATION_RANGE[0]
        )
        rank = base.hattrick_f_rank(rows, summary, baseline_summary, args.low_budget)
        base.save_validation(run_dir, epoch, rows, summary, diagnostics)
        row = base.history_row(epoch, rank, rows, summary, diagnostics, train_metrics)
        history = [item for item in history if int(item["epoch"]) < epoch]
        history.append(row)
        base.write_csv(history_path, history)
        payload = {
            "method": "Hattrick-f4", "seed": args.seed, "epoch": epoch,
            "rank": tuple(float(value) for value in rank), "config": config,
            "config_sha256": config_sha256,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_summary": summary,
            "validation_diagnostics": diagnostics,
        }
        torch.save(payload, resume_path)
        archive_path = None
        if epoch % args.save_every == 0 or epoch == args.epochs:
            archive_path = infra.save_epoch_checkpoint(run_dir, payload)
        top_entries = infra.update_top_k(
            run_dir=run_dir, entries=top_entries, payload=payload, top_k=args.top_k
        )
        retained = [int(entry["epoch"]) for entry in top_entries]
        print(
            f"[Hattrick-f4] epoch={epoch}/{args.epochs} eligible={rank[0]} "
            f"H={row['high_norm_mean']:.6f} Hmin={row['high_norm_min']:.6f} "
            f"M={row['medium_norm_mean']:.6f} L={row['low_norm_mean']:.6f} "
            f"archive={archive_path.name if archive_path else '-'} "
            f"best{args.top_k}={retained}", flush=True,
        )
    if len(top_entries) != args.top_k:
        raise RuntimeError(f"Expected {args.top_k} retained checkpoints, found {len(top_entries)}")
    complete = {
        "status": "COMPLETE", "method": "Hattrick-f4",
        "selection_used_test": False, "strict_esm": True, "seed": args.seed,
        "epochs": args.epochs, "save_every": args.save_every,
        "top_k": args.top_k, "best_checkpoints": top_entries,
        "top5": top_entries, "selected_epoch": int(top_entries[0]["epoch"]),
        "saved_epoch_checkpoints": infra.saved_epoch_numbers(run_dir),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "resume_state": str(resume_path.resolve()),
        "resume_state_sha256": base.sha256(resume_path),
        "config_sha256": config_sha256,
    }
    base.write_json(run_dir / "complete.json", complete)
    print(json.dumps(complete, indent=2, ensure_ascii=False), flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train full-GEANT 2x Hattrick-f4 active-set crossover"
    )
    parser.add_argument("--stage", choices=("check", "train", "all"), default="check")
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--epochs", type=positive_int, default=DEFAULT_EPOCHS)
    parser.add_argument("--top-k", type=positive_int, default=DEFAULT_TOP_K)
    parser.add_argument("--save-every", type=positive_int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=positive_float, default=0.0005)
    parser.add_argument("--low-budget", type=non_negative_float, default=0.03)
    parser.add_argument("--high-floor", type=positive_float, default=DEFAULT_HIGH_FLOOR)
    parser.add_argument("--crossover-step", type=positive_float, default=DEFAULT_CROSSOVER_STEP)
    parser.add_argument("--edge-tau", type=positive_float, default=DEFAULT_EDGE_TAU)
    parser.add_argument("--path-tau", type=positive_float, default=DEFAULT_PATH_TAU)
    parser.add_argument(
        "--activation-margin", type=non_negative_float,
        default=DEFAULT_ACTIVATION_MARGIN,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--phase-a-checkpoint", type=Path, default=DEFAULT_PHASE_A)
    args = parser.parse_args()
    if args.top_k > args.epochs:
        parser.error("--top-k cannot exceed --epochs")
    if not 0.0 < args.high_floor <= 1.0:
        parser.error("--high-floor must be in (0, 1]")
    if not 0.0 < args.crossover_step <= 1.0:
        parser.error("--crossover-step must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    if args.stage in ("check", "all"):
        check_inputs(args)
    if args.stage in ("train", "all"):
        train(args)


if __name__ == "__main__":
    main()
