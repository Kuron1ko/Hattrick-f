from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

import _workspace as ws
from utils.AdamOptimizer import ADAMOptimizer
from utils.args_parser import parse_args
from utils.build_dataset_within_cluster import (
    DM_Dataset_within_Cluster,
    custom_collate,
)
from utils.robust_proj_utils import assign_gradients_and_step, flatten_grads
from utils.training_utils import loss_mf, loss_mlu


METHOD_NAME = "Hattrick"
OBJECTIVE_NAMES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
CLASSES = ("High", "Medium", "Low")
NUM_PATHS = 8


def register_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def build_props(
    dataset: str,
    *,
    device: torch.device,
    batch_size: int,
    epochs: int,
    learning_rate: float,
):
    dataset = ws.canonical_dataset(dataset)
    spec = ws.DATASETS[dataset]
    train_start, train_end = spec["train"]
    validation_start, validation_end = spec["validation"]
    props = parse_args(
        [
            "--topo", str(spec["topology"]),
            "--framework", "hattrick",
            "--mode", "train",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--num_paths_per_pair", str(NUM_PATHS),
            "--num_transformer_layers", "3",
            "--num_gnn_layers", "3",
            "--num_mlp1_hidden_layers", "2",
            "--num_mlp2_hidden_layers", "2",
            "--rau1", "3",
            "--rau2", "3",
            "--rau3", "3",
            "--train_clusters", "0",
            "--train_start_indices", str(train_start),
            "--train_end_indices", str(train_end),
            "--val_clusters", "0",
            "--val_start_indices", str(validation_start),
            "--val_end_indices", str(validation_end),
            "--pred", "1",
            "--dynamic", "0",
            "--lr", str(learning_rate),
            "--pred_type", "esm",
            "--initial_training", "1",
            "--violation", "1",
            "--path_mask", "0",
        ]
    )
    props.device = device
    props.dtype = torch.float32
    props.workspace_root = str(ws.ROOT)
    props.dataset_name = dataset
    props.data_dir = str(ws.DATA_ROOT / dataset)
    props.baseline_dir = str(ws.BASE_ROOT / dataset)
    props.path_cache_dir = str(ws.DATA_ROOT / dataset / "path_cache")
    props.research_return_admitted = False
    props.research_return_policy = False
    return props


def make_datasets(props, dataset: str, *, include_test: bool = False):
    dataset = ws.canonical_dataset(dataset)
    spec = ws.DATASETS[dataset]
    train = DM_Dataset_within_Cluster(props, 0, *spec["train"])
    validation = DM_Dataset_within_Cluster(props, 0, *spec["validation"])
    if int(train.max_source_index_read) != spec["train"][1] - 1:
        raise RuntimeError("Train split audit failed")
    if int(validation.max_source_index_read) != spec["validation"][1] - 1:
        raise RuntimeError("Validation split audit failed")
    test = None
    if include_test:
        test = DM_Dataset_within_Cluster(props, 0, *spec["test"])
        if int(test.max_source_index_read) != spec["test"][1] - 1:
            raise RuntimeError("Test split audit failed")
    return train, validation, test


def data_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
        generator=generator,
    )


def move_dataset_static(dataset, device: torch.device) -> torch.Tensor | None:
    dataset.pte = dataset.pte.to(device=device, dtype=torch.float32).coalesce()
    dataset.padded_edge_ids_per_path = dataset.padded_edge_ids_per_path.to(device)
    for mapping in (
        dataset.edge_ids_dict_tensor,
        dataset.original_pos_edge_ids_dict_tensor,
    ):
        for key in mapping:
            mapping[key] = mapping[key].to(device)
    return dataset.path_masks.to(device) if dataset.path_masks is not None else None


def unpack_to_device(inputs, props):
    *tensors, snapshots = inputs
    tensors = [
        value.to(device=props.device, dtype=props.dtype) for value in tensors
    ]
    return (*tensors, snapshots)


def model_forward(model, props, dataset, values, path_masks):
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
    if not props.dynamic:
        node_features = node_features[:1]
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
    return output, capacities


def build_objectives(model, props, dataset, values, path_masks):
    (
        _node_features,
        _capacities,
        _tm1,
        _tm1_pred,
        _tm2,
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
    output, _ = model_forward(model, props, dataset, values, path_masks)
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
    # Complete paper order: High flow, High MLU, cumulative High+Medium flow,
    # High+Medium MLU, total flow, total MLU.
    loss_fh, value_fh = loss_mf(admitted_high, opt1_mf.detach())
    loss_uh, value_uh = loss_mlu(edges_high, opt1.detach())
    loss_fhm, value_fhm = loss_mf(
        admitted_high + admitted_medium, opt2_mf.detach()
    )
    loss_uhm, value_uhm = loss_mlu(edges_high_medium, opt2.detach())
    loss_fhml, value_fhml = loss_mf(all_traffic, opt3_mf.detach())
    loss_uhml, value_uhml = loss_mlu(edges_all, opt3.detach())
    return (
        (loss_fh, loss_uh, loss_fhm, loss_uhm, loss_fhml, loss_uhml),
        (value_fh, value_uh, value_fhm, value_uhm, value_fhml, value_uhml),
    )


@dataclass
class ProjectionResult:
    final_gradient: torch.Tensor
    parameter_shapes: list[torch.Size]
    raw_gradients: list[torch.Tensor]
    projected_gradients: list[torch.Tensor]


def _project_away_from_basis(
    vector: torch.Tensor, orthonormal_basis: list[torch.Tensor]
) -> torch.Tensor:
    projected = vector.to(dtype=torch.float64)
    for _ in range(2):
        for basis_vector in orthonormal_basis:
            projected = projected - torch.dot(projected, basis_vector) * basis_vector
    return projected


def ordered_project_gradients(
    model: torch.nn.Module,
    losses: Sequence[torch.Tensor],
    *,
    zero_tolerance: float = 1e-12,
) -> ProjectionResult:
    if not losses:
        raise ValueError("At least one ordered objective is required")
    raw_gradients: list[torch.Tensor] = []
    shapes: list[torch.Size] | None = None
    for index, loss in enumerate(losses):
        if loss.ndim != 0:
            raise ValueError(f"Objective {index} must be scalar")
        gradient, current_shapes = flatten_grads(
            model,
            loss,
            retain_graph=index < len(losses) - 1,
            zero_grad=index < len(losses) - 1,
        )
        if shapes is None:
            shapes = current_shapes
        raw_gradients.append(gradient.detach().to(dtype=torch.float32))
    assert shapes is not None
    basis: list[torch.Tensor] = []
    projected_gradients: list[torch.Tensor] = []
    for raw in raw_gradients:
        work = raw.to(dtype=torch.float64)
        raw_norm = torch.linalg.vector_norm(work)
        if not torch.isfinite(raw_norm).item():
            raise RuntimeError("Non-finite raw objective gradient")
        if float(raw_norm.item()) <= zero_tolerance:
            projected_gradients.append(torch.zeros_like(raw))
            continue
        projected = _project_away_from_basis(work, basis)
        projected_norm = torch.linalg.vector_norm(projected)
        if not torch.isfinite(projected_norm).item():
            raise RuntimeError("Non-finite projected objective gradient")
        if float(projected_norm.item()) <= zero_tolerance:
            projected_gradients.append(torch.zeros_like(raw))
            continue
        basis.append(projected / projected_norm)
        projected_gradients.append(projected.to(dtype=raw.dtype))
    final_gradient = torch.stack(projected_gradients, dim=0).sum(dim=0)
    if not torch.isfinite(final_gradient).all().item():
        raise RuntimeError("Non-finite final ordered gradient")
    return ProjectionResult(
        final_gradient=final_gradient,
        parameter_shapes=shapes,
        raw_gradients=raw_gradients,
        projected_gradients=projected_gradients,
    )


def projection_diagnostics(
    result: ProjectionResult, names: Sequence[str]
) -> dict[str, float]:
    if len(names) != len(result.raw_gradients):
        raise ValueError("Objective names and gradients must have the same length")
    diagnostics: dict[str, float] = {}
    for name, raw, projected in zip(
        names, result.raw_gradients, result.projected_gradients
    ):
        diagnostics[f"raw_gradient_norm_{name}"] = float(
            torch.linalg.vector_norm(raw).item()
        )
        diagnostics[f"projected_gradient_norm_{name}"] = float(
            torch.linalg.vector_norm(projected).item()
        )
    for lower_index in range(1, len(result.projected_gradients)):
        lower = result.projected_gradients[lower_index].to(dtype=torch.float64)
        for higher_index in range(lower_index):
            higher = result.projected_gradients[higher_index].to(dtype=torch.float64)
            denominator = torch.linalg.vector_norm(lower) * torch.linalg.vector_norm(
                higher
            )
            cosine = 0.0
            if float(denominator.item()) > 1e-20:
                cosine = float((torch.dot(lower, higher) / denominator).item())
            diagnostics[
                f"projected_cosine_{names[lower_index]}_vs_{names[higher_index]}"
            ] = cosine
    return diagnostics


def train_epoch(model, props, dataset, loader, optimizer) -> dict[str, float]:
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    path_masks = move_dataset_static(dataset, props.device)
    totals = {f"reported_{name}": 0.0 for name in OBJECTIVE_NAMES}
    first_probe: dict[str, float] = {}
    count = 0
    for batch_index, inputs in enumerate(loader):
        values = unpack_to_device(inputs, props)
        losses, reported = build_objectives(
            model, props, dataset, values, path_masks
        )
        if any(not torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError("Non-finite six-objective loss")
        result = ordered_project_gradients(model, losses)
        if batch_index == 0:
            first_probe = projection_diagnostics(result, OBJECTIVE_NAMES)
        assign_gradients_and_step(
            model,
            result.final_gradient,
            optimizer,
            result.parameter_shapes,
        )
        for name, value in zip(OBJECTIVE_NAMES, reported):
            totals[f"reported_{name}"] += float(value)
        count += 1
    props.research_return_admitted = False
    means = {name: value / max(count, 1) for name, value in totals.items()}
    means.update(first_probe)
    means["train_batches"] = count
    return means


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_rows(rows: list[dict]) -> list[dict]:
    summary = []
    for class_name in CLASSES:
        selected = [row for row in rows if row["class"] == class_name]
        item = {"class": class_name, "n": len(selected)}
        for metric in (
            "admitted_traffic",
            "demand",
            "fulfill_ratio",
            "norm_fulfill",
            "normalized_mlu",
            "raw_mlu",
            "admitted_capacity_ratio",
        ):
            values = [float(row[metric]) for row in selected]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_p1"] = _percentile(values, 1)
            item[f"{metric}_p10"] = _percentile(values, 10)
        item["max_disabled_flow"] = max(
            float(row["disabled_flow"]) for row in selected
        )
        item["max_admitted_capacity_ratio"] = max(
            float(row["admitted_capacity_ratio"]) for row in selected
        )
        summary.append(item)
    return summary


def evaluation_diagnostics(rows: list[dict]) -> dict[str, float]:
    by_key = {(int(row["snapshot"]), row["class"]): row for row in rows}
    snapshots = sorted({int(row["snapshot"]) for row in rows})
    gaps = np.asarray(
        [
            float(by_key[(snapshot, "Low")]["norm_fulfill"])
            - float(by_key[(snapshot, "Medium")]["norm_fulfill"])
            for snapshot in snapshots
        ],
        dtype=np.float64,
    )
    return {
        "inversion_raw_gap_mean": float(gaps.mean()),
        "inversion_positive_gap_mean": float(np.maximum(gaps, 0.0).mean()),
        "inversion_violation_fraction": float((gaps > 0).mean()),
        "inversion_gap_p90": float(np.percentile(gaps, 90)),
        "inversion_gap_max": float(gaps.max()),
    }


def evaluate(model, props, dataset, start_index: int):
    """Strict prediction inference: route on ESM, realize metrics on actual TM."""
    model.eval()
    props.mode = "test"
    props.research_return_admitted = False
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    path_masks = move_dataset_static(dataset, props.device)
    flat_masks = (
        [path_masks[index].reshape(-1) for index in range(3)]
        if path_masks is not None
        else None
    )
    rows: list[dict] = []
    loader = data_loader(dataset, 1, False, 0)
    with torch.no_grad():
        for local_index, inputs in enumerate(loader):
            values = unpack_to_device(inputs, props)
            (
                _node_features,
                capacities_full,
                tm1,
                _tm1_pred,
                tm2,
                _tm2_pred,
                tm3,
                _tm3_pred,
                opt1,
                opt2,
                opt3,
                opt1_mf,
                opt2_mf,
                opt3_mf,
                _snapshots,
            ) = values
            props.sim_mf_mlu = 1
            admitted, capacities = model_forward(
                model, props, dataset, values, path_masks
            )
            admitted = tuple(item.reshape(1, -1) for item in admitted)
            props.sim_mf_mlu = 0
            edge_utils, _ = model_forward(model, props, dataset, values, path_masks)
            oracle_flow = (opt1_mf, opt2_mf - opt1_mf, opt3_mf - opt2_mf)
            oracle_mlu = (opt1, opt2, opt3)
            tms = (tm1, tm2, tm3)
            cumulative_admitted = torch.zeros_like(admitted[0])
            for class_index, class_name in enumerate(CLASSES):
                class_admitted = admitted[class_index]
                cumulative_admitted = cumulative_admitted + class_admitted
                admitted_total = float(class_admitted.sum().item())
                demand = float((tms[class_index].sum() / NUM_PATHS).item())
                oracle = max(float(oracle_flow[class_index].item()), 1e-9)
                raw_mlu = float(edge_utils[class_index].max().item())
                normalized_mlu = raw_mlu / max(
                    float(oracle_mlu[class_index].item()), 1e-9
                )
                disabled_flow = 0.0
                if flat_masks is not None:
                    disabled = ~flat_masks[class_index]
                    if disabled.any():
                        disabled_flow = float(
                            class_admitted[:, disabled].abs().max().item()
                        )
                admitted_on_links = torch.sparse.mm(
                    dataset.pte.to(dtype=torch.float32).t(),
                    cumulative_admitted.to(dtype=torch.float32).t(),
                ).t()
                admitted_capacity_ratio = float(
                    (
                        admitted_on_links
                        / capacities_full[:1].to(dtype=torch.float32)
                    )
                    .max()
                    .item()
                )
                rows.append(
                    {
                        "snapshot": start_index + local_index,
                        "class": class_name,
                        "admitted_traffic": admitted_total,
                        "demand": demand,
                        "fulfill_ratio": admitted_total / max(demand, 1e-9),
                        "oracle_admitted_traffic": oracle,
                        "norm_fulfill": admitted_total / oracle,
                        "raw_mlu": raw_mlu,
                        "oracle_mlu": float(oracle_mlu[class_index].item()),
                        "normalized_mlu": normalized_mlu,
                        "disabled_flow": disabled_flow,
                        "admitted_capacity_ratio": admitted_capacity_ratio,
                    }
                )
    if any(
        not math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key != "class"
    ):
        raise RuntimeError("Evaluation produced NaN or Inf")
    summary = summarize_rows(rows)
    return rows, summary, evaluation_diagnostics(rows)


def _resume_candidate(run_dir: Path) -> Path | None:
    for name in ("resume_state.pt", "final_model.pt"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            value = ws.load_checkpoint(path)
        except RuntimeError:
            continue
        if "optimizer_state_dict" in value:
            return path
    return None


def _validate_resume_hyperparameters(checkpoint: dict, args: Namespace) -> None:
    config = checkpoint.get("entry_config") or checkpoint.get("config") or {}
    if not isinstance(config, dict):
        return
    requested = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "top_k": args.top_k,
    }
    if args.method == "Hattrick-f3":
        requested.update(
            {
                "anneal_epochs": args.anneal_epochs,
                "low_budget": args.low_budget,
            }
        )
    annealing = config.get("annealing")
    if isinstance(annealing, dict) and "anneal_epochs" in annealing:
        config = {**config, "anneal_epochs": annealing["anneal_epochs"]}
    mismatches = []
    for key, value in requested.items():
        if key not in config:
            continue
        saved = config[key]
        same = (
            abs(float(saved) - value) <= 1e-12
            if isinstance(value, float)
            else saved == value
        )
        if not same:
            mismatches.append(f"{key}: saved={saved}, requested={value}")
    if mismatches:
        raise RuntimeError(
            "Automatic resume requires unchanged hyperparameters except --epochs:\n"
            + "\n".join(mismatches)
        )


def inspect_training(args: Namespace) -> dict:
    dataset = ws.canonical_dataset(args.dataset)
    method = ws.canonical_method(args.method)
    run_directory = ws.run_dir(dataset, method, args.seed)
    resume_path = _resume_candidate(run_directory)
    current_epoch = None
    if resume_path is not None:
        checkpoint = ws.load_checkpoint(resume_path)
        _validate_resume_hyperparameters(checkpoint, args)
        current_epoch = int(checkpoint["epoch"])
    checkpoints = ws.saved_checkpoints(dataset, method, args.seed)
    report = {
        "dataset": dataset,
        "method": method,
        "seed": args.seed,
        "target_epochs": args.epochs,
        "mode": "resume" if resume_path else "initial",
        "resume_state": (
            str(resume_path.relative_to(ws.ROOT)) if resume_path else None
        ),
        "current_epoch": current_epoch,
        "saved_epochs": list(checkpoints),
        "device": str(ws.resolve_device(args.device)),
        "data": str((ws.DATA_ROOT / dataset).relative_to(ws.ROOT)),
        "baseline": str((ws.BASE_ROOT / dataset).relative_to(ws.ROOT)),
        "model_output": str(run_directory.relative_to(ws.ROOT)),
        "automatic_resume": True,
    }
    if resume_path is None and checkpoints:
        report["blocked"] = (
            "Saved inference checkpoints exist, but none contains a resumable "
            "optimizer state. Use a new --seed for an initial run."
        )
    return report


def _entry_config(args: Namespace, *, phase_a: Path | None) -> dict:
    value = {
        "schema_version": 1,
        "method": args.method,
        "dataset": args.dataset,
        "seed": args.seed,
        "strict_esm": True,
        "train": list(ws.DATASETS[args.dataset]["train"]),
        "validation": list(ws.DATASETS[args.dataset]["validation"]),
        "test": list(ws.DATASETS[args.dataset]["test"]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "top_k": args.top_k,
        "selection_rules": str(ws.RULES_PATH.relative_to(ws.ROOT)),
        "runtime": "method/hattrick_system.py",
    }
    if args.method == "Hattrick-f3":
        value.update(
            {
                "anneal_epochs": args.anneal_epochs,
                "low_budget": args.low_budget,
                "phase_a_checkpoint": (
                    str(phase_a.relative_to(ws.ROOT)) if phase_a else None
                ),
                "optimizer_reset_on_initial_training": True,
                "algorithm": "method/hattrick_f3_system.py",
            }
        )
    return value


def _save_epoch(
    *,
    args: Namespace,
    epoch: int,
    model,
    optimizer,
    rank: tuple[float, ...],
    summary: list[dict],
    diagnostics: dict,
    train_metrics: dict,
    anneal_alpha: float | None,
    phase_a: Path | None,
) -> Path:
    run_directory = ws.run_dir(args.dataset, args.method, args.seed)
    payload = {
        "method": args.method,
        "dataset": args.dataset,
        "seed": args.seed,
        "epoch": epoch,
        "rank": tuple(float(value) for value in rank),
        "entry_config": _entry_config(args, phase_a=phase_a),
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_summary": summary,
        "validation_diagnostics": diagnostics,
        "train_metrics": train_metrics,
    }
    if anneal_alpha is not None:
        payload["anneal_alpha"] = anneal_alpha
    checkpoint = run_directory / "checkpoints" / f"epoch_{epoch:03d}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    torch.save(payload, run_directory / "resume_state.pt")
    return checkpoint


def _history_row(
    *,
    epoch: int,
    rank: tuple[float, ...],
    summary: list[dict],
    diagnostics: dict,
    train_metrics: dict,
    anneal_alpha: float | None,
) -> dict:
    """Build the stable, legacy-compatible training-history schema."""
    metrics = ws.metric_fields(summary)
    row = {
        "epoch": int(epoch),
        "eligible": int(bool(rank[0])),
        "high_norm_mean": metrics["h_mean"],
        "high_norm_p10": metrics["h_p10"],
        "high_norm_p1": metrics["h_p1"],
        "medium_norm_mean": metrics["m_mean"],
        "medium_norm_p10": metrics["m_p10"],
        "medium_norm_p1": metrics["m_p1"],
        "low_norm_mean": metrics["l_mean"],
        "low_norm_p10": metrics["l_p10"],
        "low_norm_p1": metrics["l_p1"],
        **diagnostics,
        **train_metrics,
    }
    if anneal_alpha is not None:
        row["anneal_alpha"] = float(anneal_alpha)
    return row


def _recover_history(
    *,
    args: Namespace,
    history_path: Path,
) -> list[dict]:
    """Recover history rows when a checkpoint was saved before logging failed."""
    history = ws.read_csv(history_path)
    checkpoints = ws.saved_checkpoints(args.dataset, args.method, args.seed)
    if not checkpoints:
        return history
    high_mean_max = ws.maximum_high_mean(
        args.dataset, args.method, args.seed, checkpoints
    )
    by_epoch = {int(row["epoch"]): row for row in history}
    changed = False

    # Eligibility depends on the current maximum High mean, so refresh old
    # history rows whenever training discovers a new maximum.
    window = float(ws.load_rules()["high"]["mean_window_from_max"])
    threshold = high_mean_max - window
    for row in by_epoch.values():
        high_text = row.get("high_norm_mean", row.get("h_mean"))
        if high_text not in (None, ""):
            eligible = str(int(float(high_text) > threshold))
            if str(row.get("eligible", "")) != eligible:
                row["eligible"] = eligible
                changed = True

    for epoch, path in checkpoints.items():
        if epoch in by_epoch:
            continue
        checkpoint = ws.load_checkpoint(path)
        try:
            summary = ws.checkpoint_summary(
                args.dataset, args.method, args.seed, epoch, checkpoint
            )
            rank = ws.selection_rank(
                args.dataset,
                args.method,
                args.seed,
                epoch,
                checkpoint,
                high_mean_max=high_mean_max,
            )
        except (KeyError, RuntimeError, ValueError):
            continue
        by_epoch[epoch] = _history_row(
            epoch=epoch,
            rank=rank,
            summary=summary,
            diagnostics=checkpoint.get("validation_diagnostics") or {},
            train_metrics=checkpoint.get("train_metrics") or {},
            anneal_alpha=checkpoint.get("anneal_alpha"),
        )
        changed = True
        print(f"[history] recovered epoch {epoch} from checkpoint", flush=True)

    recovered = [by_epoch[epoch] for epoch in sorted(by_epoch)]
    if changed:
        ws.write_csv(history_path, recovered)
    return recovered


def run_training(args: Namespace) -> None:
    args.dataset = ws.canonical_dataset(args.dataset)
    args.method = ws.canonical_method(args.method)
    report = inspect_training(args)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    if report.get("blocked"):
        raise RuntimeError(str(report["blocked"]))
    ws.ensure_workspace_inputs(args.dataset)

    from frameworks.hattrick_system import Hattrick

    device = ws.resolve_device(args.device)
    set_seed(args.seed)
    props = build_props(
        args.dataset,
        device=device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    train_dataset, validation_dataset, test_dataset = make_datasets(
        props, args.dataset, include_test=False
    )
    if test_dataset is not None:
        raise RuntimeError("Training unexpectedly instantiated the test split")
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    run_directory = ws.run_dir(args.dataset, args.method, args.seed)
    run_directory.mkdir(parents=True, exist_ok=True)
    resume_path = _resume_candidate(run_directory)
    phase_a: Path | None = None
    start_epoch = 1

    if resume_path is not None:
        checkpoint = ws.load_checkpoint(resume_path, device=device)
        _validate_resume_hyperparameters(checkpoint, args)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if args.method == "Hattrick-f3":
            phase_text = (checkpoint.get("entry_config") or {}).get(
                "phase_a_checkpoint"
            )
            if phase_text:
                phase_a = ws.ROOT / phase_text
        print(
            f"[resume] {args.dataset} {args.method} from epoch {start_epoch}",
            flush=True,
        )
    elif args.method == "Hattrick-f3":
        if args.phase_a_epoch is None:
            phase_epoch, phase_a, phase_checkpoint = ws.select_checkpoint(
                args.dataset, "Hattrick", args.seed
            )
        else:
            phase_epoch = args.phase_a_epoch
            phase_a = ws.resolve_checkpoint(
                args.dataset, "Hattrick", phase_epoch, args.seed
            )
            phase_checkpoint = ws.load_checkpoint(phase_a)
        model.load_state_dict(phase_checkpoint["model_state_dict"])
        optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
        print(
            f"[phase-a] Hattrick epoch {phase_epoch}: "
            f"{phase_a.relative_to(ws.ROOT)}",
            flush=True,
        )

    if start_epoch > args.epochs:
        print(
            f"[done] current epoch {start_epoch - 1} already reaches target "
            f"{args.epochs}",
            flush=True,
        )
        return
    if args.method == "Hattrick-f3" and phase_a is None:
        _phase_epoch, phase_a, _phase_checkpoint = ws.select_checkpoint(
            args.dataset, "Hattrick", args.seed
        )

    history_path = run_directory / "train_history.csv"
    history = _recover_history(args=args, history_path=history_path)
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        set_seed(args.seed + epoch * 104729)
        loader = data_loader(
            train_dataset, props.batch_size, True, args.seed + epoch * 1009
        )
        alpha: float | None = None
        if args.method == "Hattrick":
            train_metrics = train_epoch(
                model, props, train_dataset, loader, optimizer
            )
        else:
            from method import hattrick_f3_system as f3

            alpha = float(f3.anneal_alpha(epoch, args.anneal_epochs))
            train_metrics = f3.train_epoch(
                model=model,
                props=props,
                dataset=train_dataset,
                loader=loader,
                optimizer=optimizer,
                alpha=alpha,
            )
        rows, summary, diagnostics = evaluate(
            model,
            props,
            validation_dataset,
            ws.DATASETS[args.dataset]["validation"][0],
        )
        ws.finite_summary(summary)
        ws.write_csv(
            run_directory / f"validation_epoch_{epoch:03d}_metrics.csv", rows
        )
        ws.write_json(
            run_directory / f"validation_epoch_{epoch:03d}_summary.json",
            {"classes": summary, "diagnostics": diagnostics},
        )
        provisional = {"validation_summary": summary, "rank": ()}
        metrics = ws.metric_fields(summary)
        try:
            previous_high_mean_max = ws.maximum_high_mean(
                args.dataset, args.method, args.seed
            )
        except RuntimeError:
            previous_high_mean_max = metrics["h_mean"]
        high_mean_max = max(previous_high_mean_max, metrics["h_mean"])
        rank = ws.selection_rank(
            args.dataset,
            args.method,
            args.seed,
            epoch,
            provisional,
            high_mean_max=high_mean_max,
        )
        checkpoint_path = _save_epoch(
            args=args,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            rank=rank,
            summary=summary,
            diagnostics=diagnostics,
            train_metrics=train_metrics,
            anneal_alpha=alpha,
            phase_a=phase_a,
        )
        entries = ws.save_top_k(
            args.dataset,
            args.method,
            args.seed,
            args.top_k,
        )
        row = _history_row(
            epoch=epoch,
            rank=rank,
            summary=summary,
            diagnostics=diagnostics,
            train_metrics=train_metrics,
            anneal_alpha=alpha,
        )
        threshold = high_mean_max - float(
            ws.load_rules()["high"]["mean_window_from_max"]
        )
        for item in history:
            high_text = item.get("high_norm_mean", item.get("h_mean"))
            if high_text not in (None, ""):
                item["eligible"] = int(float(high_text) > threshold)
        history = [item for item in history if int(item["epoch"]) < epoch]
        history.append(row)
        ws.write_csv(history_path, history)
        print(
            f"[{args.dataset} {args.method}] epoch={epoch}/{args.epochs} "
            f"eligible={int(bool(rank[0]))} H={metrics['h_mean']:.6f} "
            f"M={metrics['m_mean']:.6f} L={metrics['l_mean']:.6f} "
            f"alpha={'-' if alpha is None else f'{alpha:.4f}'} "
            f"top{args.top_k}={[entry['epoch'] for entry in entries]}",
            flush=True,
        )

    entries = ws.save_top_k(
        args.dataset,
        args.method,
        args.seed,
        args.top_k,
    )
    ws.write_json(
        run_directory / "complete.json",
        {
            "status": "COMPLETE",
            "method": args.method,
            "dataset": args.dataset,
            "seed": args.seed,
            "epochs": args.epochs,
            "selection_used_test": False,
            "selected_epoch": entries[0]["epoch"],
            "top_k": entries,
            "runtime_seconds_this_invocation": time.perf_counter() - started,
        },
    )
    print(
        f"[complete] selected epoch {entries[0]['epoch']} at "
        f"{run_directory.relative_to(ws.ROOT)}",
        flush=True,
    )


def run(args: Namespace) -> None:
    args.method = METHOD_NAME
    run_training(args)
