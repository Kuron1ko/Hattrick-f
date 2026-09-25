from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
OUTPUT_ROOT = THIS_DIR / "results_hattrick_improvement_strict2x"
TOPOLOGY = "geant_priomask500_strict_load2x_train"
K = 8
CLASSES = ("High", "Medium", "Low")

sys.path.insert(0, str(ROOT))

from frameworks.hattrick_system import Hattrick
from utils.AdamOptimizer import ADAMOptimizer
from utils.args_parser import parse_args
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster, custom_collate
from utils.robust_proj_utils import assign_gradients_and_step, project_gradients_one_optimizer_robust
from utils.training_utils import loss_mf, loss_mlu, train as unchanged_train


LEVELS = {
    1: {
        "train": (0, 32),
        "validation": (32, 40),
        "evaluation": (32, 40),
        "epochs": 2,
        "label": "level1_correctness",
    },
    2: {
        "train": (0, 160),
        "validation": (160, 200),
        "evaluation": (200, 250),
        "epochs": 12,
        "label": "level2_proxy",
    },
    3: {
        "train": (0, 350),
        "validation": (350, 400),
        "evaluation": (350, 400),
        "epochs": 30,
        "label": "level3_validation_only",
    },
    4: {
        "train": (0, 350),
        "validation": (350, 400),
        "evaluation": (400, 500),
        "epochs": 60,
        "label": "level4_test",
    },
}

APPROACHES = ("unchanged", "final_medium_pcgrad")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def build_props(level: int, device: torch.device):
    spec = LEVELS[level]
    train_start, train_end = spec["train"]
    val_start, val_end = spec["validation"]
    props = parse_args(
        [
            "--topo", TOPOLOGY,
            "--mode", "train",
            "--epochs", str(spec["epochs"]),
            "--batch_size", "8",
            "--num_paths_per_pair", str(K),
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
            "--val_start_indices", str(val_start),
            "--val_end_indices", str(val_end),
            "--pred", "1",
            "--dynamic", "0",
            "--lr", "0.0005",
            "--pred_type", "esm",
            "--initial_training", "1",
            "--violation", "1",
            "--path_mask", "1",
        ]
    )
    props.device = device
    props.dtype = torch.float32
    props.research_return_admitted = False
    return props


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def move_dataset_static(dataset: DM_Dataset_within_Cluster, device: torch.device) -> torch.Tensor | None:
    dataset.pte = dataset.pte.to(device=device, dtype=torch.float32).coalesce()
    dataset.padded_edge_ids_per_path = dataset.padded_edge_ids_per_path.to(device)
    for mapping in (dataset.edge_ids_dict_tensor, dataset.original_pos_edge_ids_dict_tensor):
        for key in mapping:
            mapping[key] = mapping[key].to(device)
    return dataset.path_masks.to(device) if dataset.path_masks is not None else None


def unpack_to_device(inputs, props):
    (
        node_features,
        capacities,
        tm1,
        tm1_pred,
        tm2,
        tm2_pred,
        tm3,
        tm3_pred,
        opt1,
        opt2,
        opt3,
        opt1_mf,
        opt2_mf,
        opt3_mf,
        snapshots,
    ) = inputs
    tensors = [
        node_features,
        capacities,
        tm1,
        tm1_pred,
        tm2,
        tm2_pred,
        tm3,
        tm3_pred,
        opt1,
        opt2,
        opt3,
        opt1_mf,
        opt2_mf,
        opt3_mf,
    ]
    tensors = [value.to(device=props.device, dtype=props.dtype) for value in tensors]
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


def train_final_medium_epoch(model, props, dataset, loader, optimizer) -> dict:
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    path_masks = move_dataset_static(dataset, props.device)
    totals = {"high_mlu": 0.0, "medium_admitted": 0.0, "high_medium_mlu": 0.0, "total_admitted": 0.0}
    count = 0
    for inputs in loader:
        values = unpack_to_device(inputs, props)
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
            _opt3,
            opt1_mf,
            opt2_mf,
            _opt3_mf,
            _snapshots,
        ) = values
        output, _ = model_forward(model, props, dataset, values, path_masks)
        (
            edges_util_high,
            _edges_util_high_medium_max_stage,
            _edges_util_all,
            _edges_util_high_final,
            edges_util_high_medium_final,
            all_traffic,
            _admitted_high,
            admitted_medium,
            _admitted_low,
        ) = output
        loss1, value1 = loss_mlu(edges_util_high, opt1)
        loss2, value2 = loss_mf(admitted_medium, opt2_mf - opt1_mf)
        loss3, value3 = loss_mlu(edges_util_high_medium_final, opt2)
        loss4, value4 = loss_mf(all_traffic, _opt3_mf)
        losses = (loss1, loss2, loss3, loss4)
        if any(not torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError("Non-finite candidate loss")
        final_grads, shapes, *_ = project_gradients_one_optimizer_robust(
            model, loss1, loss2, loss3, loss4, optimizer
        )
        if not torch.isfinite(final_grads).all().item():
            raise RuntimeError("Non-finite candidate gradients")
        assign_gradients_and_step(model, final_grads, optimizer, shapes)
        totals["high_mlu"] += float(value1)
        totals["medium_admitted"] += float(value2)
        totals["high_medium_mlu"] += float(value3)
        totals["total_admitted"] += float(value4)
        count += 1
    props.research_return_admitted = False
    return {key: value / max(count, 1) for key, value in totals.items()}


def data_loader(dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate,
        generator=generator if shuffle else None,
    )


def percentile(values: list[float], q: float) -> float:
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
            item[f"{metric}_p1"] = percentile(values, 1)
            item[f"{metric}_p10"] = percentile(values, 10)
        item["max_disabled_flow"] = max(float(row["disabled_flow"]) for row in selected)
        item["max_admitted_capacity_ratio"] = max(float(row["admitted_capacity_ratio"]) for row in selected)
        summary.append(item)
    return summary


def evaluate(model, props, dataset, start_index: int) -> tuple[list[dict], list[dict]]:
    model.eval()
    props.mode = "test"
    props.research_return_admitted = False
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    path_masks = move_dataset_static(dataset, props.device)
    flat_masks = [path_masks[index].reshape(-1) for index in range(3)] if path_masks is not None else None
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
            admitted, capacities = model_forward(model, props, dataset, values, path_masks)
            admitted = tuple(item.reshape(1, -1) for item in admitted)
            props.sim_mf_mlu = 0
            edge_utils, _ = model_forward(model, props, dataset, values, path_masks)
            oracle_flow = (
                opt1_mf,
                opt2_mf - opt1_mf,
                opt3_mf - opt2_mf,
            )
            oracle_mlu = (opt1, opt2, opt3)
            tms = (tm1, tm2, tm3)
            cumulative_admitted = torch.zeros_like(admitted[0])
            for class_index, class_name in enumerate(CLASSES):
                class_admitted = admitted[class_index]
                cumulative_admitted = cumulative_admitted + class_admitted
                admitted_total = float(class_admitted.sum().item())
                demand = float((tms[class_index].sum() / K).item())
                oracle = max(float(oracle_flow[class_index].item()), 1e-9)
                raw_mlu = float(edge_utils[class_index].max().item())
                normalized_mlu = raw_mlu / max(float(oracle_mlu[class_index].item()), 1e-9)
                disabled_flow = 0.0
                if flat_masks is not None:
                    disabled = ~flat_masks[class_index]
                    if disabled.any():
                        disabled_flow = float(class_admitted[:, disabled].abs().max().item())
                admitted_on_links = torch.sparse.mm(
                    dataset.pte.to(dtype=torch.float32).t(),
                    cumulative_admitted.to(dtype=torch.float32).t(),
                ).t()
                admitted_capacity_ratio = float(
                    (admitted_on_links / capacities_full[:1].to(dtype=torch.float32)).max().item()
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
    if any(not math.isfinite(float(value)) for row in rows for key, value in row.items() if key not in ("class",)):
        raise RuntimeError("Evaluation produced NaN or Inf")
    summary = summarize_rows(rows)
    return rows, summary


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def checkpoint_rank(summary: list[dict]) -> tuple:
    indexed = summary_index(summary)
    high = indexed["High"]
    medium = indexed["Medium"]
    feasible = (
        float(high["norm_fulfill_mean"]) >= 0.98
        and float(indexed["Low"]["normalized_mlu_mean"]) <= 1.05
        and max(float(row["max_admitted_capacity_ratio"]) for row in summary) <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
    )
    if feasible:
        return (
            1,
            float(medium["norm_fulfill_p10"]),
            float(medium["norm_fulfill_p1"]),
            -abs(float(medium["normalized_mlu_mean"]) - 1.0),
        )
    return (
        0,
        float(high["norm_fulfill_mean"]),
        float(high["norm_fulfill_p10"]),
        float(medium["norm_fulfill_p10"]),
    )


def run_one(level: int, approach: str, seed: int, force: bool = False) -> Path:
    spec = LEVELS[level]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = OUTPUT_ROOT / spec["label"] / approach / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not force:
        print(f"[skip] complete {run_dir}", flush=True)
        return run_dir

    set_seed(seed)
    props = build_props(level, device)
    train_start, train_end = spec["train"]
    val_start, val_end = spec["validation"]
    eval_start, eval_end = spec["evaluation"]
    config = {
        "level": level,
        "label": spec["label"],
        "approach": approach,
        "seed": seed,
        "topology": TOPOLOGY,
        "train": [train_start, train_end],
        "validation": [val_start, val_end],
        "evaluation": [eval_start, eval_end],
        "evaluation_is_final_test": level == 4,
        "epochs": spec["epochs"],
        "batch_size": props.batch_size,
        "learning_rate": props.lr,
        "checkpoint_rule": "among exact mask/capacity checkpoints with High mean >= 0.98 and final cumulative normalized pre-admission MLU <= 1.05, maximize validation Medium NormFulFill P10 then P1; if none meet the guards, maximize High mean then High P10 before Medium P10 and label the checkpoint as a fallback",
        "candidate_objectives": (
            ["High max-stage normalized MLU", "final admitted Medium", "final cumulative High+Medium MLU", "total admitted traffic"]
            if approach == "final_medium_pcgrad"
            else ["unchanged Hattrick objectives"]
        ),
        "source_sha256": {
            "frameworks/hattrick_system.py": sha256(ROOT / "frameworks" / "hattrick_system.py"),
            "utils/build_dataset_within_cluster.py": sha256(ROOT / "utils" / "build_dataset_within_cluster.py"),
            "utils/training_utils.py": sha256(ROOT / "utils" / "training_utils.py"),
            "utils/robust_proj_utils.py": sha256(ROOT / "utils" / "robust_proj_utils.py"),
            "test_diff_path/run_hattrick_strict2x_research.py": sha256(Path(__file__).resolve()),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "data_access": {
            "train": [train_start, train_end],
            "validation": [val_start, val_end],
            "evaluation": [eval_start, eval_end],
            "max_source_index_read": max(train_end, val_end, eval_end) - 1,
            "reader_contract": "np.loadtxt skiprows/max_rows; no outcome or manifest rows at or beyond each split end are read",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    train_dataset = DM_Dataset_within_Cluster(props, 0, train_start, train_end)
    val_dataset = DM_Dataset_within_Cluster(props, 0, val_start, val_end)
    eval_dataset = DM_Dataset_within_Cluster(props, 0, eval_start, eval_end)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    best_rank = None
    best_epoch = None
    history: list[dict] = []

    for epoch in range(1, int(spec["epochs"]) + 1):
        props.mode = "train"
        props.sim_mf_mlu = 0
        props.research_return_admitted = False
        loader = data_loader(train_dataset, props.batch_size, True, seed + epoch * 1009)
        if approach == "unchanged":
            unchanged_train(epoch - 1, int(spec["epochs"]), model, props, [train_dataset], [loader], [optimizer])
            train_metrics = {}
        elif approach == "final_medium_pcgrad":
            train_metrics = train_final_medium_epoch(model, props, train_dataset, loader, optimizer)
        else:
            raise ValueError(f"Unknown approach: {approach}")

        val_rows, val_summary = evaluate(model, props, val_dataset, val_start)
        write_csv(run_dir / f"validation_epoch_{epoch:03d}_metrics.csv", val_rows)
        (run_dir / f"validation_epoch_{epoch:03d}_summary.json").write_text(
            json.dumps(val_summary, indent=2), encoding="utf-8"
        )
        rank = checkpoint_rank(val_summary)
        indexed = summary_index(val_summary)
        history_row = {
            "epoch": epoch,
            "high_norm_mean": indexed["High"]["norm_fulfill_mean"],
            "medium_norm_mean": indexed["Medium"]["norm_fulfill_mean"],
            "medium_norm_p1": indexed["Medium"]["norm_fulfill_p1"],
            "medium_norm_p10": indexed["Medium"]["norm_fulfill_p10"],
            "medium_normalized_mlu_mean": indexed["Medium"]["normalized_mlu_mean"],
            "max_disabled_flow": max(row["max_disabled_flow"] for row in val_summary),
            "max_admitted_capacity_ratio": max(row["max_admitted_capacity_ratio"] for row in val_summary),
            "checkpoint_feasible": rank[0],
            **train_metrics,
        }
        history.append(history_row)
        write_csv(run_dir / "train_history.csv", history)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            },
            run_dir / "final_model.pt",
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "rank": rank,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                run_dir / "best_model.pt",
            )
        print(
            f"[{spec['label']}] {approach} seed={seed} epoch={epoch}/{spec['epochs']} "
            f"high_mean={history_row['high_norm_mean']:.4f} "
            f"medium_p1={history_row['medium_norm_p1']:.4f} "
            f"medium_p10={history_row['medium_norm_p10']:.4f} best_epoch={best_epoch}",
            flush=True,
        )

    evaluations = []
    final_reference_rows = None
    for checkpoint_name in ("final", "best"):
        checkpoint = torch.load(run_dir / f"{checkpoint_name}_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        rows, summary = evaluate(model, props, eval_dataset, eval_start)
        for row in rows:
            row["checkpoint"] = checkpoint_name
            row["approach"] = approach
            row["seed"] = seed
            row["level"] = level
        write_csv(run_dir / f"{checkpoint_name}_evaluation_metrics.csv", rows)
        (run_dir / f"{checkpoint_name}_evaluation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        evaluations.append({"checkpoint": checkpoint_name, "epoch": checkpoint["epoch"], "summary": summary})
        if checkpoint_name == "final":
            final_reference_rows = rows

    recovery_checkpoint = torch.load(run_dir / "final_model.pt", map_location=device, weights_only=False)
    recovery_model = Hattrick(props).to(device=device, dtype=props.dtype)
    recovery_optimizer = ADAMOptimizer(recovery_model.parameters(), lr=props.lr)
    recovery_model.load_state_dict(recovery_checkpoint["model_state_dict"])
    recovery_optimizer.load_state_dict(recovery_checkpoint["optimizer_state_dict"])
    recovery_rows, _ = evaluate(recovery_model, props, eval_dataset, eval_start)
    numeric_fields = (
        "admitted_traffic",
        "demand",
        "fulfill_ratio",
        "oracle_admitted_traffic",
        "norm_fulfill",
        "raw_mlu",
        "oracle_mlu",
        "normalized_mlu",
        "disabled_flow",
        "admitted_capacity_ratio",
    )
    max_recovery_delta = max(
        abs(float(expected[field]) - float(actual[field]))
        for expected, actual in zip(final_reference_rows, recovery_rows)
        for field in numeric_fields
    )
    recovery = {
        "checkpoint_epoch": recovery_checkpoint["epoch"],
        "model_and_optimizer_loaded": True,
        "optimizer_parameter_states": len(recovery_optimizer.state),
        "evaluation_rows": len(recovery_rows),
        "max_numeric_delta_from_saved_final_evaluation": max_recovery_delta,
        # Float32 inference can differ by one ULP after a save/load boundary on
        # the same device (observed maximum: 4.768e-7).  Treat that as exact
        # numerical recovery while still rejecting any material mismatch.
        "passes": len(recovery_rows) == len(final_reference_rows) and max_recovery_delta <= 1e-6,
    }
    (run_dir / "checkpoint_recovery.json").write_text(json.dumps(recovery, indent=2), encoding="utf-8")
    if not recovery["passes"]:
        raise RuntimeError(f"Checkpoint recovery mismatch: {recovery}")

    complete = {
        "best_epoch": best_epoch,
        "best_rank": best_rank,
        "selection_status": "guard_feasible" if best_rank[0] else "high_first_fallback_no_guard_feasible",
        "checkpoint_recovery": recovery,
        "evaluations": evaluations,
    }
    complete_path.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    return run_dir


def build_level_summary(level: int) -> Path:
    spec = LEVELS[level]
    level_dir = OUTPUT_ROOT / spec["label"]
    rows = []
    for approach in APPROACHES:
        approach_dir = level_dir / approach
        if not approach_dir.exists():
            continue
        for seed_dir in sorted(approach_dir.glob("seed_*")):
            seed = int(seed_dir.name.split("_")[-1])
            for checkpoint in ("final", "best"):
                path = seed_dir / f"{checkpoint}_evaluation_summary.json"
                if not path.exists():
                    continue
                summary = json.loads(path.read_text(encoding="utf-8"))
                for item in summary:
                    rows.append(
                        {
                            "level": level,
                            "approach": approach,
                            "seed": seed,
                            "checkpoint": checkpoint,
                            **item,
                        }
                    )
    output = level_dir / "level_summary.csv"
    write_csv(output, rows)
    return output


def analyze_level2_gate(seeds: list[int]) -> Path:
    level_dir = OUTPUT_ROOT / LEVELS[2]["label"]
    results = []
    for seed in seeds:
        baseline_path = level_dir / "unchanged" / f"seed_{seed}" / "best_evaluation_metrics.csv"
        candidate_path = level_dir / "final_medium_pcgrad" / f"seed_{seed}" / "best_evaluation_metrics.csv"
        if not baseline_path.exists() or not candidate_path.exists():
            continue
        baseline = read_csv(baseline_path)
        candidate = read_csv(candidate_path)
        baseline_by_key = {(int(row["snapshot"]), row["class"]): row for row in baseline}
        candidate_by_key = {(int(row["snapshot"]), row["class"]): row for row in candidate}
        if baseline_by_key.keys() != candidate_by_key.keys():
            raise RuntimeError(f"Level-2 paired row mismatch for seed {seed}")

        def values(rows, class_name, metric):
            return np.asarray([float(row[metric]) for row in rows if row["class"] == class_name], dtype=np.float64)

        base_medium = values(baseline, "Medium", "norm_fulfill")
        cand_medium = values(candidate, "Medium", "norm_fulfill")
        base_high = values(baseline, "High", "norm_fulfill")
        cand_high = values(candidate, "High", "norm_fulfill")
        base_capacity = values(baseline, "Low", "admitted_capacity_ratio")
        cand_capacity = values(candidate, "Low", "admitted_capacity_ratio")
        base_mlu = values(baseline, "Low", "normalized_mlu")
        cand_mlu = values(candidate, "Low", "normalized_mlu")
        paired_medium = cand_medium - base_medium
        max_disabled = max(float(row["disabled_flow"]) for row in candidate)
        item = {
            "seed": seed,
            "baseline_medium_p10": percentile(base_medium.tolist(), 10),
            "candidate_medium_p10": percentile(cand_medium.tolist(), 10),
            "medium_p10_gap": percentile(cand_medium.tolist(), 10) - percentile(base_medium.tolist(), 10),
            "baseline_medium_p1": percentile(base_medium.tolist(), 1),
            "candidate_medium_p1": percentile(cand_medium.tolist(), 1),
            "medium_p1_gap": percentile(cand_medium.tolist(), 1) - percentile(base_medium.tolist(), 1),
            "baseline_high_mean": float(base_high.mean()),
            "candidate_high_mean": float(cand_high.mean()),
            "high_mean_gap": float(cand_high.mean() - base_high.mean()),
            "baseline_admitted_capacity_ratio_mean": float(base_capacity.mean()),
            "candidate_admitted_capacity_ratio_mean": float(cand_capacity.mean()),
            "admitted_capacity_ratio_mean_gap": float(cand_capacity.mean() - base_capacity.mean()),
            "baseline_normalized_mlu_mean": float(base_mlu.mean()),
            "candidate_normalized_mlu_mean": float(cand_mlu.mean()),
            "normalized_mlu_mean_gap": float(cand_mlu.mean() - base_mlu.mean()),
            "candidate_max_admitted_capacity_ratio": float(cand_capacity.max()),
            "candidate_max_disabled_flow": max_disabled,
            "paired_medium_mean_gap": float(paired_medium.mean()),
            "paired_medium_median_gap": float(np.median(paired_medium)),
            "paired_medium_positive_slices": int((paired_medium > 0).sum()),
            "paired_medium_total_slices": int(paired_medium.size),
            "paired_medium_min_gap": float(paired_medium.min()),
            "paired_medium_max_gap": float(paired_medium.max()),
        }
        item["passes"] = bool(
            item["medium_p10_gap"] >= 0.02
            and item["high_mean_gap"] >= -0.01
            and item["normalized_mlu_mean_gap"] <= 0.01
            and item["admitted_capacity_ratio_mean_gap"] <= 0.01
            and item["candidate_max_admitted_capacity_ratio"] <= 1.0001
            and item["candidate_max_disabled_flow"] <= 1e-8
            and item["paired_medium_positive_slices"] >= math.ceil(0.5 * item["paired_medium_total_slices"])
        )
        results.append(item)

    decision = {
        "level": 2,
        "evaluation_window": list(LEVELS[2]["evaluation"]),
        "final_test_opened": False,
        "required_seeds": seeds,
        "results": results,
        "promote": len(results) == len(seeds) and len(results) >= 2 and all(row["passes"] for row in results),
        "rule": {
            "medium_p10_gap_min": 0.02,
            "high_mean_gap_min": -0.01,
            "normalized_mlu_mean_gap_max": 0.01,
            "admitted_capacity_ratio_mean_gap_max": 0.01,
            "candidate_max_admitted_capacity_ratio": 1.0001,
            "candidate_max_disabled_flow": 1e-8,
            "paired_positive_fraction_min": 0.5,
        },
    }
    output = level_dir / "level2_gate.json"
    output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=sorted(LEVELS), required=True)
    parser.add_argument("--approaches", nargs="+", choices=APPROACHES, default=list(APPROACHES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[490])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.level == 4 and len(args.seeds) < 3:
        raise SystemExit("Level 4 requires at least three seeds")
    if args.level in (2, 3) and len(args.seeds) < 2:
        raise SystemExit(f"Level {args.level} requires at least two seeds")
    for approach in args.approaches:
        for seed in args.seeds:
            run_one(args.level, approach, seed, force=args.force)
    summary = build_level_summary(args.level)
    print(f"[ok] wrote {summary}", flush=True)
    if args.level == 2 and set(args.approaches) == set(APPROACHES):
        gate = analyze_level2_gate(args.seeds)
        print(f"[ok] wrote {gate}", flush=True)


if __name__ == "__main__":
    main()
