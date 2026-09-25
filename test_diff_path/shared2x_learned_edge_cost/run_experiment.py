from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = ROOT.parent / ".venv-hattrick" / "Scripts" / "python.exe"
TOPOLOGY = "geant_priomask500_shared_load2x_train"
K = 8
BASE_MODEL = ROOT / f"hattrick_{TOPOLOGY}_{K}sp.pkl"
RESULT_DIR = ROOT / "results" / TOPOLOGY / f"{K}sp" / "0"
CLASSES = ("High", "Medium", "Low")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.args_parser import parse_args
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster, custom_collate
from utils.training_utils import move_to_device


PHASES = {
    "small": {
        "train": (0, 128),
        "validation": (128, 160),
        "evaluation": (160, 250),
        "epochs": 12,
        "lr": 0.0002,
        "effective_batch_size": 32,
    },
    "full": {
        "train": (0, 350),
        "validation": (350, 400),
        "evaluation": (400, 500),
        "epochs": 30,
        "lr": 0.0002,
        "effective_batch_size": 32,
    },
}


def common_args() -> list[str]:
    return [
        "--topo", TOPOLOGY,
        "--num_paths_per_pair", str(K),
        "--num_transformer_layers", "3",
        "--num_gnn_layers", "3",
        "--num_mlp1_hidden_layers", "2",
        "--num_mlp2_hidden_layers", "2",
        "--rau1", "3",
        "--rau2", "3",
        "--rau3", "3",
        "--pred", "1",
        "--dynamic", "0",
        "--pred_type", "esm",
        "--violation", "1",
        "--path_mask", "0",
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def run_logged(command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        raise RuntimeError("\n".join(tail))


def make_policy_props(start: int, end: int, device: torch.device):
    props = parse_args(
        [
            "--mode", "test",
            "--test_cluster", "0",
            "--test_start_idx", str(start),
            "--test_end_idx", str(end),
            "--sim_mf_mlu", "0",
            *common_args(),
        ]
    )
    props.device = device
    props.dtype = torch.float32
    props.pg_mlu_lambda = 0.0
    props.edge_cost_objective = False
    props.research_return_policy = True
    props.research_return_admitted = False
    return props


def cached_policy_forward(model, props, dataset, values, path_masks):
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
    node_features = node_features[:1]
    if hasattr(model, "transformer_output"):
        topology_cache = model.transformer_output[:1]
        model.transformer_output = topology_cache.expand(batch, -1, -1, -1)
        node_features = node_features.expand(batch, -1, -1)
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
    if hasattr(model, "transformer_output"):
        model.transformer_output = model.transformer_output[:1]
    return output


def fit_edge_costs(
    start: int,
    end: int,
    quantile: float,
    output_path: Path,
    target: str = "gap",
    ratio_epsilon: float = 0.01,
    weight_blend: float = 1.0,
) -> dict:
    """Fit 3 x |E| static costs by edge-level quantile regression."""
    if target not in {"gap", "ratio"}:
        raise ValueError("target must be 'gap' or 'ratio'")
    if ratio_epsilon <= 0:
        raise ValueError("ratio_epsilon must be positive")
    if weight_blend < 0:
        raise ValueError("weight_blend must be non-negative")
    set_seed(490)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = make_policy_props(start, end, device)
    dataset = DM_Dataset_within_Cluster(props, 0, start, end)
    if int(dataset.max_source_index_read) != end - 1:
        raise RuntimeError("Split-safe reader audit failed while fitting edge costs")
    dataset.pte = dataset.pte.to(device=device, dtype=props.dtype).coalesce()
    dataset.padded_edge_ids_per_path = dataset.padded_edge_ids_per_path.to(device)
    move_to_device(dataset.edge_ids_dict_tensor, device)
    move_to_device(dataset.original_pos_edge_ids_dict_tensor, device)
    path_masks = dataset.path_masks.to(device) if dataset.path_masks is not None else None

    model = torch.load(BASE_MODEL, map_location=device, weights_only=False)
    model.device = device
    model = model.to(device=device, dtype=props.dtype)
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    risk_batches: list[torch.Tensor] = []
    gap_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for inputs in loader:
            values = tuple(
                value.to(device=device, dtype=props.dtype) if torch.is_tensor(value) else value
                for value in inputs
            )
            policies = cached_policy_forward(model, props, dataset, values, path_masks)
            true_tms = (values[2], values[4], values[6])
            predicted_tms = (values[3], values[5], values[7])
            capacities = values[1]
            batch = int(true_tms[0].shape[0])
            if capacities.shape[0] == 1 and batch > 1:
                capacities = capacities.expand(batch, -1)
            elif capacities.shape[0] != batch:
                capacities = capacities[:1].expand(batch, -1)

            true_cumulative = torch.zeros_like(capacities)
            predicted_cumulative = torch.zeros_like(capacities)
            stage_risks = []
            stage_gaps = []
            for policy, true_tm, predicted_tm in zip(policies, true_tms, predicted_tms):
                policy = policy.squeeze(-1)
                true_path_flow = policy * true_tm.squeeze(-1)
                predicted_path_flow = policy * predicted_tm.squeeze(-1)
                true_cumulative = true_cumulative + torch.sparse.mm(
                    dataset.pte.t(), true_path_flow.t()
                ).t()
                predicted_cumulative = predicted_cumulative + torch.sparse.mm(
                    dataset.pte.t(), predicted_path_flow.t()
                ).t()
                true_utilization = true_cumulative / capacities.clamp_min(1e-9)
                predicted_utilization = predicted_cumulative / capacities.clamp_min(1e-9)
                positive_gap = torch.relu(true_utilization - predicted_utilization)
                stage_gaps.append(positive_gap)
                if target == "ratio":
                    stage_risks.append(
                        (true_utilization + ratio_epsilon)
                        / (predicted_utilization + ratio_epsilon)
                    )
                else:
                    stage_risks.append(positive_gap)
            risk_batches.append(torch.stack(stage_risks, dim=1).cpu())
            gap_batches.append(torch.stack(stage_gaps, dim=1).cpu())

    risks = torch.cat(risk_batches, dim=0)
    gaps = torch.cat(gap_batches, dim=0)
    # For a constant predictor, the empirical quantile is the exact minimizer
    # of pinball loss.  This is quantile regression with only 3*|E| parameters.
    weights = torch.quantile(risks, quantile, dim=0).to(dtype=torch.float32)
    if target == "ratio":
        # Never make an edge look safer than vanilla MLU.  Values above one
        # encode the learned multiplicative ESM underprediction factor.
        weights = weights.clamp_min(1.0)
        weights = 1.0 + weight_blend * (weights - 1.0)
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise RuntimeError("Invalid learned edge costs")
    if torch.any(weights.sum(dim=1) <= 0):
        raise RuntimeError("A stage learned all-zero edge costs")

    means = gaps.mean(dim=0)
    frequency = (gaps > 0).to(torch.float32).mean(dim=0)
    payload = {
        "method": (
            "static edge real-to-ESM utilization-ratio quantile regression"
            if target == "ratio"
            else "static one-sided edge-utilization quantile regression"
        ),
        "strict_esm_policy": True,
        "train_range": [start, end],
        "quantile": quantile,
        "target": target,
        "ratio_epsilon": ratio_epsilon if target == "ratio" else None,
        "weight_blend": weight_blend if target == "ratio" else None,
        "weights": weights,
        "mean_positive_error": means,
        "underprediction_frequency": frequency,
        "edge_index": dataset.edge_index.detach().cpu(),
        "n_samples": int(gaps.shape[0]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    stats = {
        "n_samples": int(gaps.shape[0]),
        "n_edges": int(gaps.shape[2]),
        "quantile": quantile,
        "target": target,
        "ratio_epsilon": ratio_epsilon if target == "ratio" else None,
        "weight_blend": weight_blend if target == "ratio" else None,
        "stage_weight_min": weights.min(dim=1).values.tolist(),
        "stage_weight_mean": weights.mean(dim=1).tolist(),
        "stage_weight_max": weights.max(dim=1).values.tolist(),
        "stage_zero_edges": (weights == 0).sum(dim=1).tolist(),
        "stage_unit_edges": (weights == 1).sum(dim=1).tolist(),
        "stage_underprediction_frequency_mean": frequency.mean(dim=1).tolist(),
        "path": str(output_path),
    }
    (output_path.parent / "edge_cost_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    del model, dataset, risks, gaps, risk_batches, gap_batches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def train_model(
    phase: str,
    label: str,
    edge_cost_path: Path | None,
    edge_cost_mode: str,
    edge_cost_power: float,
    edge_cost_strength: float,
    future_lookahead: bool,
    future_reservation_strength: float,
    future_low_reservation_strength: float,
    hm_overflow_lambda: float = 0.0,
) -> tuple[Path, float]:
    cfg = PHASES[phase]
    train_start, train_end = cfg["train"]
    val_start, val_end = cfg["validation"]
    tag = f"edgecost_{phase}_{label}"
    model_path = ROOT / f"hattrick_{TOPOLOGY}_{K}sp_{tag}.pkl"
    env = os.environ.copy()
    env.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HATTRICK_PG_MLU_LAMBDA": "0",
            "HATTRICK_HM_OVERFLOW_LAMBDA": str(hm_overflow_lambda),
            "HATTRICK_EDGE_COST_OBJECTIVE": "1" if edge_cost_path else "0",
            "HATTRICK_EDGE_COST_PATH": str(edge_cost_path) if edge_cost_path else "",
            "HATTRICK_EDGE_COST_MODE": edge_cost_mode,
            "HATTRICK_EDGE_COST_POWER": str(edge_cost_power),
            "HATTRICK_EDGE_COST_STRENGTH": str(edge_cost_strength),
            "HATTRICK_FUTURE_LOOKAHEAD": "1" if future_lookahead else "0",
            "HATTRICK_FUTURE_RESERVATION_STRENGTH": str(future_reservation_strength),
            "HATTRICK_FUTURE_LOW_RESERVATION_STRENGTH": str(
                future_low_reservation_strength
            ),
            "HATTRICK_RUN_TAG": tag,
            "HATTRICK_INIT_MODEL": str(BASE_MODEL),
            "HATTRICK_FRESH_OPTIMIZER": "1",
            "HATTRICK_EFFECTIVE_BATCH_SIZE": str(cfg["effective_batch_size"]),
        }
    )
    command = [
        str(PYTHON), "run_hattrick.py",
        "--mode", "train",
        "--epochs", str(cfg["epochs"]),
        "--batch_size", "8",
        "--train_clusters", "0",
        "--train_start_indices", str(train_start),
        "--train_end_indices", str(train_end),
        "--val_clusters", "0",
        "--val_start_indices", str(val_start),
        "--val_end_indices", str(val_end),
        "--lr", str(cfg["lr"]),
        "--initial_training", "0",
        *common_args(),
    ]
    started = time.perf_counter()
    print(f"[train] {phase}/{label}", flush=True)
    run_logged(command, env, HERE / "logs" / f"{phase}_{label}_train.log")
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return model_path, time.perf_counter() - started


def evaluate(
    phase: str,
    label: str,
    model_path: Path,
    future_lookahead: bool = False,
) -> np.ndarray:
    cfg = PHASES[phase]
    eval_start, eval_end = cfg["evaluation"]
    model_name = f"edgecost_{phase}_{label}"
    env = os.environ.copy()
    env.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HATTRICK_PG_MLU_LAMBDA": "0",
            "HATTRICK_EDGE_COST_OBJECTIVE": "0",
            "HATTRICK_EDGE_COST_PATH": "",
            "HATTRICK_EDGE_COST_MODE": "linear",
            "HATTRICK_FUTURE_LOOKAHEAD": "1" if future_lookahead else "0",
            "HATTRICK_FUTURE_RESERVATION_STRENGTH": "0",
            "HATTRICK_FUTURE_LOW_RESERVATION_STRENGTH": "0",
        }
    )
    command = [
        str(PYTHON), "run_hattrick.py",
        "--mode", "test",
        "--test_cluster", "0",
        "--test_start_idx", str(eval_start),
        "--test_end_idx", str(eval_end),
        "--sim_mf_mlu", "1",
        "--model", model_name,
        "--model_path_override", str(model_path),
        *common_args(),
    ]
    print(f"[test]  {phase}/{label}, strict ESM", flush=True)
    run_logged(command, env, HERE / "logs" / f"{phase}_{label}_test.log")
    values_path = RESULT_DIR / f"{model_name}_values_esm_sim_mlu_1.txt"
    values = np.loadtxt(values_path, dtype=np.float64).reshape(-1, 3)
    expected = eval_end - eval_start
    if values.shape != (expected, 3):
        raise RuntimeError(f"{values_path}: got {values.shape}, expected {(expected, 3)}")
    return values


def summarize(values: np.ndarray) -> dict[str, dict[str, float]]:
    result = {}
    for index, class_name in enumerate(CLASSES):
        column = values[:, index]
        result[class_name] = {
            "mean": float(column.mean()),
            "p1": float(np.percentile(column, 1)),
            "p10": float(np.percentile(column, 10)),
            "median": float(np.median(column)),
        }
    return result


def write_summary_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=PHASES, default="small")
    parser.add_argument("--quantile", type=float, default=0.9)
    parser.add_argument(
        "--objective",
        choices=("linear", "congestion", "weighted_mlu", "weighted_mlu_actual"),
        default="weighted_mlu",
    )
    parser.add_argument("--weight-target", choices=("auto", "gap", "ratio"), default="auto")
    parser.add_argument("--ratio-epsilon", type=float, default=0.01)
    parser.add_argument("--weight-blend", type=float, default=1.0)
    parser.add_argument("--power", type=float, default=4.0)
    parser.add_argument("--strength", type=float, default=0.25)
    parser.add_argument("--future-lookahead", action="store_true")
    parser.add_argument("--future-reservation-strength", type=float, default=0.0)
    parser.add_argument("--future-low-reservation-strength", type=float, default=None)
    args = parser.parse_args()
    if not 0.5 <= args.quantile < 1.0:
        raise SystemExit("--quantile must be in [0.5, 1.0)")
    if args.power <= 1.0:
        raise SystemExit("--power must be greater than 1")
    if args.strength < 0.0:
        raise SystemExit("--strength must be non-negative")
    if args.ratio_epsilon <= 0.0:
        raise SystemExit("--ratio-epsilon must be positive")
    if args.weight_blend < 0.0:
        raise SystemExit("--weight-blend must be non-negative")
    if args.future_reservation_strength < 0.0:
        raise SystemExit("--future-reservation-strength must be non-negative")
    if (
        args.future_low_reservation_strength is not None
        and args.future_low_reservation_strength < 0.0
    ):
        raise SystemExit("--future-low-reservation-strength must be non-negative")

    phase = args.phase
    cfg = PHASES[phase]
    low_reservation_strength = (
        args.future_reservation_strength
        if args.future_low_reservation_strength is None
        else args.future_low_reservation_strength
    )
    weight_target = (
        "ratio" if args.objective in {"weighted_mlu", "weighted_mlu_actual"} else "gap"
        if args.weight_target == "auto"
        else args.weight_target
    )
    if args.weight_target != "auto":
        weight_target = args.weight_target
    experiment_suffix = (
        (
            (
                f"{args.objective}_{weight_target}_q{args.quantile:g}_"
                f"eps{args.ratio_epsilon:g}_blend{args.weight_blend:g}"
            )
            if args.objective in {"weighted_mlu", "weighted_mlu_actual"}
            else f"{args.objective}_p{args.power:g}_a{args.strength:g}_q{args.quantile:g}"
        )
        .replace(".", "p")
    )
    if args.future_lookahead:
        experiment_suffix = f"{experiment_suffix}_lookahead"
    if args.future_reservation_strength > 0.0:
        experiment_suffix = (
            f"{experiment_suffix}_reserve{args.future_reservation_strength:g}"
            .replace(".", "p")
        )
    if low_reservation_strength != args.future_reservation_strength:
        experiment_suffix = (
            f"{experiment_suffix}_lowreserve{low_reservation_strength:g}"
            .replace(".", "p")
        )
    phase_dir = HERE / "artifacts" / f"{phase}_{experiment_suffix}"
    phase_dir.mkdir(parents=True, exist_ok=True)
    label = f"learned_{experiment_suffix}"
    edge_cost_path = phase_dir / f"edge_costs_q{args.quantile:g}.pt"
    weight_stats = fit_edge_costs(
        *cfg["train"],
        args.quantile,
        edge_cost_path,
        target=weight_target,
        ratio_epsilon=args.ratio_epsilon,
        weight_blend=args.weight_blend,
    )

    control_label = f"control_{experiment_suffix}"
    runs = ((control_label, None), (label, edge_cost_path))
    all_values: dict[str, np.ndarray] = {}
    records: list[dict] = []
    models: dict[str, str] = {}
    for run_label, costs in runs:
        run_lookahead = bool(args.future_lookahead and costs is not None)
        model_path, elapsed = train_model(
            phase,
            run_label,
            costs,
            args.objective,
            args.power,
            args.strength,
            run_lookahead,
            args.future_reservation_strength if costs is not None else 0.0,
            low_reservation_strength if costs is not None else 0.0,
        )
        values = evaluate(phase, run_label, model_path, run_lookahead)
        all_values[run_label] = values
        models[run_label] = str(model_path)
        np.savetxt(
            phase_dir / f"{run_label}_norm_fulfill.csv",
            values,
            delimiter=",",
            header=",".join(CLASSES),
            comments="",
        )
        summary = summarize(values)
        for class_name in CLASSES:
            records.append(
                {
                    "phase": phase,
                    "method": run_label,
                    "class": class_name,
                    "n": len(values),
                    **summary[class_name],
                    "training_seconds": elapsed,
                    "model_path": str(model_path),
                }
            )

    control = summarize(all_values[control_label])
    candidate = summarize(all_values[label])
    delta = {
        class_name: {
            metric: candidate[class_name][metric] - control[class_name][metric]
            for metric in ("mean", "p1", "p10", "median")
        }
        for class_name in CLASSES
    }
    gates = {
        "high_mean_at_least_0.995": candidate["High"]["mean"] >= 0.995,
        "low_mean_drop_at_most_0.003": delta["Low"]["mean"] >= -0.003,
        "low_p10_drop_at_most_0.005": delta["Low"]["p10"] >= -0.005,
        "medium_mean_gain_at_least_0.005": delta["Medium"]["mean"] >= 0.005,
    }
    report = {
        "method": (
            "three learned edge ratios inside native real-utilization cross-stage weighted MLU"
            if args.objective == "weighted_mlu_actual"
            else
            "three learned real/ESM edge ratios inside max-over-edges weighted MLU"
            if args.objective == "weighted_mlu"
            else "three learned risk weights with a utilization-dependent edge price replacing the three MLU losses"
        ),
        "formula": (
            "WMLU_s = native_cross_stage_max_e(weight[s,e] * real_train_util[s,e])"
            if args.objective == "weighted_mlu_actual"
            else "WMLU_s = max_e(weight[s,e] * ESM_util[s,e])"
            if args.objective == "weighted_mlu"
            else (
                "C_s = sum_e ESM_load[s,e] * (1 + strength*v[s,e]/mean(v[s])) "
                "* ESM_util[s,e]^(power-1)"
                if args.objective == "congestion"
                else "C_s = sum_e v[s,e] * ESM_edge_load[s,e]"
            )
        ),
        "weight_supervision": (
            "max(1, q-quantile((real_util+epsilon)/(ESM_util+epsilon)))"
            if weight_target == "ratio"
            else "q-quantile of ReLU(real cumulative utilization - ESM cumulative utilization)"
        ),
        "strict_esm_inference": True,
        "future_lookahead": args.future_lookahead,
        "future_reservation_strength": args.future_reservation_strength,
        "future_low_reservation_strength": low_reservation_strength,
        "future_reservation_formula": (
            "High *= 1 + beta_medium*ESM_Medium_util + beta_low*ESM_Low_util; "
            "HighMedium *= 1 + beta_low*ESM_Low_util"
            if args.future_reservation_strength > 0.0
            else None
        ),
        "future_lookahead_inputs": (
            ["High ESM", "Medium ESM", "Low ESM"]
            if args.future_lookahead
            else ["High ESM"]
        ),
        "lookahead_initialization": (
            "new Medium/Low input columns initialized to zero"
            if args.future_lookahead
            else None
        ),
        "phase": phase,
        "protocol": cfg,
        "same_initial_model": str(BASE_MODEL),
        "quantile": args.quantile,
        "weight_target": weight_target,
        "ratio_epsilon": args.ratio_epsilon if weight_target == "ratio" else None,
        "weight_blend": args.weight_blend if weight_target == "ratio" else None,
        "objective": args.objective,
        "power": args.power,
        "strength": args.strength,
        "edge_cost_stats": weight_stats,
        "models": models,
        "summaries": {control_label: control, label: candidate},
        "candidate_minus_control": delta,
        "expansion_gates": gates,
        "expand": all(gates.values()),
    }
    (phase_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary_csv(phase_dir / "summary.csv", records)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
