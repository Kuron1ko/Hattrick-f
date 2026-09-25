from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.append(str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torch_scatter import scatter_max

from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster, custom_collate


RESULTS = ROOT / "test_diff_path" / "results_dotemc_priority_masks_60epoch"
OUTPUTS = RESULTS / "outputs"
LOGS = ROOT / "test_diff_path" / "logs_dotemc_priority_masks_60epoch"
STATE_FILE = ROOT / "test_diff_path" / "dotemc_priority_mask_state.json"
HATTRICK_RESULTS = ROOT / "test_diff_path" / "results_priority_masks_60epoch" / "priority_mask_metrics_100slice.csv"

K = 8
TRAIN_END = 80
VAL_END = 90
TEST_END = 100
EPOCHS = 60
BATCH_SIZE = 8
LR = 0.0005
HIDDEN_DIM = 1024
NUM_HIDDEN_LAYERS = 2
ZERO_CAP_MASK = 0.01
CLASSES = ("High", "Medium", "Low")
SCENARIOS = {
    "shared": "geant_priomask_shared",
    "mild": "geant_priomask_mild",
    "medium": "geant_priomask_medium",
    "strict": "geant_priomask_strict",
}
WEIGHT_CONFIGS = {
    "w_1_0p01_0p001": (1.0, 0.01, 0.001),
    "w_1_0p1_0p01": (1.0, 0.1, 0.01),
}
DEFAULT_WEIGHT_LABEL = "w_1_0p01_0p001"


def ensure_dirs() -> None:
    for path in (RESULTS, OUTPUTS, LOGS, RESULTS / "models"):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 490) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"validate": False, "smoke": False, "train": {}, "test": {}, "report": False}


def save_state(state: dict) -> None:
    ensure_dirs()
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def make_props(topo: str, path_mask: int, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        topo=topo,
        weight=None,
        metric="mlu",
        sim_mf_mlu=0,
        mode="train",
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        num_paths_per_pair=K,
        framework="hattrick",
        num_transformer_layers=3,
        num_heads=0,
        num_gnn_layers=3,
        num_mlp1_hidden_layers=2,
        num_mlp2_hidden_layers=2,
        dropout=0,
        rau1=3,
        rau2=3,
        rau3=3,
        failure_id=None,
        dynamic=0,
        dtype=torch.float32,
        initial_training=1,
        meta_learning=0,
        violation=1,
        round=1,
        rate_cap=0,
        index=1,
        train_start_indices=[0],
        train_end_indices=[TRAIN_END],
        val_start_indices=[TRAIN_END],
        val_end_indices=[VAL_END],
        test_start_idx=VAL_END,
        test_end_idx=TEST_END,
        train_clusters=[0],
        val_clusters=[0],
        test_cluster=0,
        opt_start_idx=0,
        opt_end_idx=TEST_END,
        priority=1,
        cluster=0,
        additive_loss=1,
        checkpoint=0,
        tol=0.00001,
        zero_cap_mask=ZERO_CAP_MASK,
        path_mask=path_mask,
        model_path_override="",
        pred=1,
        pred_type="esm",
        combine_tms=0,
        objs=None,
        gur_mode="flexile",
        model="dotemc",
        device=device,
    )


class DotemcMLP(nn.Module):
    def __init__(self, input_dim: int, num_pairs: int, k: int, hidden_dim: int = HIDDEN_DIM, hidden_layers: int = NUM_HIDDEN_LAYERS):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 3 * num_pairs * k))
        self.net = nn.Sequential(*layers)
        self.num_pairs = num_pairs
        self.k = k

    def forward(self, inputs: torch.Tensor, path_masks: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.net(inputs).reshape(inputs.shape[0], 3, self.num_pairs, self.k)
        if path_masks is not None:
            mask = path_masks.reshape(3, self.num_pairs, self.k).unsqueeze(0).to(device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.exp(torch.nn.functional.log_softmax(logits, dim=-1))


def compact_tm(tm: torch.Tensor, num_pairs: int) -> torch.Tensor:
    return tm.reshape(tm.shape[0], num_pairs, K, 1)[:, :, 0, 0]


def make_inputs(tms1_pred: torch.Tensor, tms2_pred: torch.Tensor, tms3_pred: torch.Tensor, num_pairs: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    raw = torch.cat(
        (
            compact_tm(tms1_pred, num_pairs),
            compact_tm(tms2_pred, num_pairs),
            compact_tm(tms3_pred, num_pairs),
        ),
        dim=1,
    )
    return (raw - mean) / std


def class_oracles(opt1_mf: torch.Tensor, opt2_mf: torch.Tensor, opt3_mf: torch.Tensor) -> torch.Tensor:
    return torch.stack((opt1_mf, opt2_mf - opt1_mf, opt3_mf - opt2_mf), dim=1).clamp_min(1e-9)


def path_edge_info(paths_to_edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pte = paths_to_edges.coalesce()
    row_indices, col_indices = pte.indices()
    values = pte.values()
    return pte, row_indices, col_indices, values


def stage_admit(
    split_ratios: torch.Tensor,
    tm: torch.Tensor,
    residual_capacities: torch.Tensor,
    paths_to_edges: torch.Tensor,
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tunnel_load = split_ratios * tm.squeeze(-1)
    link_load = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), tunnel_load.to(dtype=torch.float32).t()).t()
    edge_utils = link_load / residual_capacities
    max_util, _ = scatter_max(edge_utils[:, col_indices] * values, row_indices, dim=1, dim_size=paths_to_edges.shape[0])
    scale = torch.where(max_util < 1.0, torch.ones_like(max_util), max_util)
    adjusted_split = split_ratios / scale.clamp_min(1e-9)
    adjusted_tunnel_load = adjusted_split * tm.squeeze(-1)
    adjusted_link_load = torch.sparse.mm(paths_to_edges.to(dtype=torch.float32).t(), adjusted_tunnel_load.to(dtype=torch.float32).t()).t()
    return adjusted_split, adjusted_link_load


def simulate_admission(
    split_ratios: torch.Tensor,
    tms: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    capacities: torch.Tensor,
    pte_info: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    paths_to_edges, row_indices, col_indices, values = pte_info
    batch_size = split_ratios.shape[0]
    split_flat = split_ratios.reshape(batch_size, 3, -1)
    cumulative_load = torch.zeros_like(capacities, dtype=torch.float32)
    residual = capacities.to(dtype=torch.float32)
    admitted = []
    mlus = []
    for class_idx, tm in enumerate(tms):
        adjusted_split, class_link_load = stage_admit(
            split_flat[:, class_idx],
            tm,
            residual,
            paths_to_edges,
            row_indices,
            col_indices,
            values,
        )
        class_tunnel_load = adjusted_split * tm.squeeze(-1)
        admitted.append(class_tunnel_load.sum(dim=1))
        cumulative_load = cumulative_load + class_link_load
        mlus.append((cumulative_load / capacities.to(dtype=torch.float32).clamp_min(1e-9)).max(dim=1).values)
        residual = capacities.to(dtype=torch.float32) - cumulative_load
        residual = torch.where(residual <= 0, torch.full_like(residual, ZERO_CAP_MASK), residual)
    return torch.stack(admitted, dim=1), torch.stack(mlus, dim=1)


def load_dataset(topo: str, path_mask: int, start: int, end: int, device: torch.device) -> DM_Dataset_within_Cluster:
    props = make_props(topo, path_mask, device)
    return DM_Dataset_within_Cluster(props, 0, start, end)


def prepare_runtime_dataset(dataset: DM_Dataset_within_Cluster, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataset.pte = dataset.pte.to(device=device, dtype=torch.float32).coalesce()
    info = path_edge_info(dataset.pte)
    path_masks = dataset.path_masks.to(device=device) if dataset.path_masks is not None else None
    return (*info, path_masks)


def compute_input_stats(dataset: DM_Dataset_within_Cluster, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    chunks = []
    num_pairs = dataset.num_pairs
    zero = torch.zeros(1, 3 * num_pairs, device=device)
    one = torch.ones(1, 3 * num_pairs, device=device)
    for inputs in loader:
        _, _, _, tm1_pred, _, tm2_pred, _, tm3_pred, *_ = inputs
        raw = torch.cat(
            (
                compact_tm(tm1_pred.to(device=device, dtype=torch.float32), num_pairs),
                compact_tm(tm2_pred.to(device=device, dtype=torch.float32), num_pairs),
                compact_tm(tm3_pred.to(device=device, dtype=torch.float32), num_pairs),
            ),
            dim=1,
        )
        chunks.append(raw)
    if not chunks:
        return zero, one
    data = torch.cat(chunks, dim=0)
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return mean, std


def train_one(scenario: str, topo: str, weight_label: str, weights: tuple[float, float, float], device: torch.device) -> None:
    path_mask = 0 if scenario == "shared" else 1
    train_ds = load_dataset(topo, path_mask, 0, TRAIN_END, device)
    val_ds = load_dataset(topo, path_mask, TRAIN_END, VAL_END, device)
    _, _, _, _, train_masks = prepare_runtime_dataset(train_ds, device)
    val_info = prepare_runtime_dataset(val_ds, device)
    val_masks = val_info[-1]
    pte_info_train = path_edge_info(train_ds.pte)
    pte_info_val = val_info[:4]

    mean, std = compute_input_stats(train_ds, device)
    model = DotemcMLP(3 * train_ds.num_pairs, train_ds.num_pairs, K).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    weight_tensor = torch.tensor(weights, device=device, dtype=torch.float32)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)
    best_val = math.inf
    best_epoch = -1
    out_dir = OUTPUTS / scenario / weight_label
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"train_{scenario}_{weight_label}.csv"
    with log_path.open("w", newline="", encoding="utf-8") as log_handle:
        writer = csv.DictWriter(log_handle, fieldnames=["epoch", "train_loss", "val_loss", "val_norm_high", "val_norm_medium", "val_norm_low"])
        writer.writeheader()
        for epoch in range(EPOCHS):
            model.train()
            train_losses = []
            for inputs in train_loader:
                _, capacities, tm1, tm1_pred, tm2, tm2_pred, tm3, tm3_pred, opt1, _, _, opt1_mf, opt2_mf, opt3_mf, _ = inputs
                capacities = capacities.to(device=device, dtype=torch.float32)
                tm1 = tm1.to(device=device, dtype=torch.float32)
                tm2 = tm2.to(device=device, dtype=torch.float32)
                tm3 = tm3.to(device=device, dtype=torch.float32)
                tm1_pred = tm1_pred.to(device=device, dtype=torch.float32)
                tm2_pred = tm2_pred.to(device=device, dtype=torch.float32)
                tm3_pred = tm3_pred.to(device=device, dtype=torch.float32)
                opt1_mf = opt1_mf.to(device=device, dtype=torch.float32)
                opt2_mf = opt2_mf.to(device=device, dtype=torch.float32)
                opt3_mf = opt3_mf.to(device=device, dtype=torch.float32)
                features = make_inputs(tm1_pred, tm2_pred, tm3_pred, train_ds.num_pairs, mean, std)
                split_ratios = model(features, train_masks)
                admitted, _ = simulate_admission(split_ratios, (tm1, tm2, tm3), capacities, pte_info_train)
                norm = admitted / class_oracles(opt1_mf, opt2_mf, opt3_mf)
                loss = -(norm * weight_tensor).sum(dim=1).mean()
                if torch.isnan(loss):
                    raise RuntimeError(f"NaN loss in {scenario} {weight_label} epoch {epoch + 1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            val_loss, val_norm = evaluate_loss(model, val_loader, val_ds, val_masks, pte_info_val, mean, std, weight_tensor, device)
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(train_losses)),
                    "val_loss": val_loss,
                    "val_norm_high": val_norm[0],
                    "val_norm_medium": val_norm[1],
                    "val_norm_low": val_norm[2],
                }
            )
            log_handle.flush()
            print(
                f"[DOTE-MC] {scenario} {weight_label} epoch {epoch + 1:02d}/{EPOCHS} "
                f"train={float(np.mean(train_losses)):.6f} val={val_loss:.6f} "
                f"norm={val_norm[0]:.4f}/{val_norm[1]:.4f}/{val_norm[2]:.4f}",
                flush=True,
            )
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                save_checkpoint(out_dir / "best_model.pt", model, mean, std, train_ds.num_pairs, weights, best_epoch, best_val)
        save_checkpoint(out_dir / "final_model.pt", model, mean, std, train_ds.num_pairs, weights, EPOCHS, best_val)


def evaluate_loss(
    model: DotemcMLP,
    loader: DataLoader,
    dataset: DM_Dataset_within_Cluster,
    path_masks: torch.Tensor | None,
    pte_info: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
) -> tuple[float, list[float]]:
    model.eval()
    losses = []
    norms = []
    with torch.no_grad():
        for inputs in loader:
            _, capacities, tm1, tm1_pred, tm2, tm2_pred, tm3, tm3_pred, _, _, _, opt1_mf, opt2_mf, opt3_mf, _ = inputs
            capacities = capacities.to(device=device, dtype=torch.float32)
            tm1 = tm1.to(device=device, dtype=torch.float32)
            tm2 = tm2.to(device=device, dtype=torch.float32)
            tm3 = tm3.to(device=device, dtype=torch.float32)
            tm1_pred = tm1_pred.to(device=device, dtype=torch.float32)
            tm2_pred = tm2_pred.to(device=device, dtype=torch.float32)
            tm3_pred = tm3_pred.to(device=device, dtype=torch.float32)
            opt1_mf = opt1_mf.to(device=device, dtype=torch.float32)
            opt2_mf = opt2_mf.to(device=device, dtype=torch.float32)
            opt3_mf = opt3_mf.to(device=device, dtype=torch.float32)
            features = make_inputs(tm1_pred, tm2_pred, tm3_pred, dataset.num_pairs, mean, std)
            split_ratios = model(features, path_masks)
            admitted, _ = simulate_admission(split_ratios, (tm1, tm2, tm3), capacities, pte_info)
            norm = admitted / class_oracles(opt1_mf, opt2_mf, opt3_mf)
            losses.append(float((-(norm * weights).sum(dim=1).mean()).cpu()))
            norms.append(norm.detach().cpu().numpy())
    all_norm = np.concatenate(norms, axis=0)
    return float(np.mean(losses)), [float(v) for v in all_norm.mean(axis=0)]


def save_checkpoint(path: Path, model: DotemcMLP, mean: torch.Tensor, std: torch.Tensor, num_pairs: int, weights: tuple[float, float, float], epoch: int, best_val: float) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
            "num_pairs": num_pairs,
            "k": K,
            "hidden_dim": HIDDEN_DIM,
            "hidden_layers": NUM_HIDDEN_LAYERS,
            "weights": weights,
            "epoch": epoch,
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[DotemcMLP, torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = DotemcMLP(
        3 * int(checkpoint["num_pairs"]),
        int(checkpoint["num_pairs"]),
        int(checkpoint["k"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["hidden_layers"]),
    ).to(device=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = checkpoint["mean"].to(device=device, dtype=torch.float32)
    std = checkpoint["std"].to(device=device, dtype=torch.float32)
    return model, mean, std


def model_output_paths(scenario: str, weight_label: str) -> tuple[Path, Path, Path]:
    out_dir = OUTPUTS / scenario / weight_label
    return out_dir / "dotemc_values_esm_sim_mlu_1.txt", out_dir / "dotemc_values_esm_sim_mlu_0.txt", out_dir / "dotemc_metrics.csv"


def test_one(scenario: str, topo: str, weight_label: str, device: torch.device) -> None:
    path_mask = 0 if scenario == "shared" else 1
    ds = load_dataset(topo, path_mask, VAL_END, TEST_END, device)
    pte_info = prepare_runtime_dataset(ds, device)[:4]
    path_masks = ds.path_masks.to(device=device) if ds.path_masks is not None else None
    model, mean, std = load_checkpoint(OUTPUTS / scenario / weight_label / "best_model.pt", device)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=custom_collate)
    values_path, mlu_path, metrics_path = model_output_paths(scenario, weight_label)
    rows = []
    with values_path.open("w", encoding="utf-8") as value_handle, mlu_path.open("w", encoding="utf-8") as mlu_handle:
        with torch.no_grad():
            for local_idx, inputs in enumerate(loader):
                _, capacities, tm1, tm1_pred, tm2, tm2_pred, tm3, tm3_pred, _, _, _, opt1_mf, opt2_mf, opt3_mf, _ = inputs
                capacities = capacities.to(device=device, dtype=torch.float32)
                tm1 = tm1.to(device=device, dtype=torch.float32)
                tm2 = tm2.to(device=device, dtype=torch.float32)
                tm3 = tm3.to(device=device, dtype=torch.float32)
                tm1_pred = tm1_pred.to(device=device, dtype=torch.float32)
                tm2_pred = tm2_pred.to(device=device, dtype=torch.float32)
                tm3_pred = tm3_pred.to(device=device, dtype=torch.float32)
                opt1_mf = opt1_mf.to(device=device, dtype=torch.float32)
                opt2_mf = opt2_mf.to(device=device, dtype=torch.float32)
                opt3_mf = opt3_mf.to(device=device, dtype=torch.float32)
                features = make_inputs(tm1_pred, tm2_pred, tm3_pred, ds.num_pairs, mean, std)
                split_ratios = model(features, path_masks)
                if path_masks is not None:
                    mask4d = path_masks.reshape(3, ds.num_pairs, K).unsqueeze(0)
                    disabled_sum = float(split_ratios.masked_select(~mask4d).sum().detach().cpu())
                    if disabled_sum != 0.0:
                        raise RuntimeError(f"Disabled paths received {disabled_sum} split ratio in {scenario}")
                allowed_sums = split_ratios.sum(dim=-1)
                if not torch.allclose(allowed_sums, torch.ones_like(allowed_sums), atol=1e-5):
                    raise RuntimeError(f"Split ratios do not sum to one in {scenario} {weight_label}")
                admitted, mlus = simulate_admission(split_ratios, (tm1, tm2, tm3), capacities, pte_info)
                oracle = class_oracles(opt1_mf, opt2_mf, opt3_mf)
                demand = torch.stack(
                    (
                        compact_tm(tm1, ds.num_pairs).sum(dim=1),
                        compact_tm(tm2, ds.num_pairs).sum(dim=1),
                        compact_tm(tm3, ds.num_pairs).sum(dim=1),
                    ),
                    dim=1,
                )
                norm = admitted / oracle
                fulfill = admitted / demand.clamp_min(1e-9)
                for class_idx, class_name in enumerate(CLASSES):
                    value_handle.write(f"{float(norm[0, class_idx].detach().cpu())}\n")
                    mlu_handle.write(f"{float(mlus[0, class_idx].detach().cpu())}\n")
                    rows.append(
                        {
                            "scenario": scenario,
                            "weights": weight_label,
                            "topo": topo,
                            "snapshot": VAL_END + local_idx,
                            "method": "DOTE_MC",
                            "class": class_name,
                            "admitted_traffic": float(admitted[0, class_idx].detach().cpu()),
                            "demand": float(demand[0, class_idx].detach().cpu()),
                            "fulfill_ratio": float(fulfill[0, class_idx].detach().cpu()),
                            "mlu": float(mlus[0, class_idx].detach().cpu()),
                            "norm_fulfill": float(norm[0, class_idx].detach().cpu()),
                        }
                    )
    write_csv(metrics_path, rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_data() -> None:
    for scenario, topo in SCENARIOS.items():
        paths_path = ROOT / "topologies" / "paths_dict" / f"{topo}_{K}_paths_dict_cluster_0.pkl"
        with paths_path.open("rb") as handle:
            paths = pickle.load(handle)
        lengths = [len(value) for value in paths.values()]
        if len(paths) != 462 or min(lengths) != K or max(lengths) != K:
            raise RuntimeError(f"{paths_path} has invalid path counts")
        mask_path = ROOT / "topologies" / "path_masks" / f"{topo}_{K}_path_masks_cluster_0.pkl"
        if scenario == "shared":
            continue
        with mask_path.open("rb") as handle:
            masks = np.asarray(pickle.load(handle), dtype=bool)
        if masks.shape != (3, 462, K):
            raise RuntimeError(f"{mask_path} has shape {masks.shape}")
        if masks.sum(axis=2).min() < 2:
            raise RuntimeError(f"{mask_path} has fewer than two allowed paths")


def smoke_model(device: torch.device) -> None:
    topo = SCENARIOS["medium"]
    ds = load_dataset(topo, 1, 0, 1, device)
    pte_info = prepare_runtime_dataset(ds, device)[:4]
    path_masks = ds.path_masks.to(device=device)
    mean, std = compute_input_stats(ds, device)
    model = DotemcMLP(3 * ds.num_pairs, ds.num_pairs, K).to(device=device)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=custom_collate)
    inputs = next(iter(loader))
    _, capacities, tm1, tm1_pred, tm2, tm2_pred, tm3, tm3_pred, _, _, _, opt1_mf, opt2_mf, opt3_mf, _ = inputs
    capacities = capacities.to(device=device, dtype=torch.float32)
    tm1 = tm1.to(device=device, dtype=torch.float32)
    tm2 = tm2.to(device=device, dtype=torch.float32)
    tm3 = tm3.to(device=device, dtype=torch.float32)
    features = make_inputs(tm1_pred.to(device=device, dtype=torch.float32), tm2_pred.to(device=device, dtype=torch.float32), tm3_pred.to(device=device, dtype=torch.float32), ds.num_pairs, mean, std)
    split = model(features, path_masks)
    if split.shape != (1, 3, ds.num_pairs, K):
        raise RuntimeError(f"Unexpected DOTE-MC output shape {tuple(split.shape)}")
    mask4d = path_masks.reshape(3, ds.num_pairs, K).unsqueeze(0)
    disabled_sum = float(split.masked_select(~mask4d).sum().detach().cpu())
    if disabled_sum != 0.0:
        raise RuntimeError(f"Disabled split ratio is non-zero: {disabled_sum}")
    if not torch.allclose(split.sum(dim=-1), torch.ones_like(split.sum(dim=-1)), atol=1e-5):
        raise RuntimeError("Split ratios do not sum to one")
    admitted, mlus = simulate_admission(split, (tm1, tm2, tm3), capacities, pte_info)
    if admitted.shape != (1, 3) or mlus.shape != (1, 3):
        raise RuntimeError("Simulation shape check failed")


def run_train_stage(device: torch.device) -> None:
    state = load_state()
    for scenario, topo in SCENARIOS.items():
        for weight_label, weights in WEIGHT_CONFIGS.items():
            key = f"{scenario}_{weight_label}"
            if state["train"].get(key):
                print(f"[skip] train {key}", flush=True)
                continue
            start = time.time()
            train_one(scenario, topo, weight_label, weights, device)
            state["train"][key] = {"done": True, "elapsed_sec": time.time() - start}
            save_state(state)


def run_test_stage(device: torch.device) -> None:
    state = load_state()
    for scenario, topo in SCENARIOS.items():
        for weight_label in WEIGHT_CONFIGS:
            key = f"{scenario}_{weight_label}"
            if state["test"].get(key):
                print(f"[skip] test {key}", flush=True)
                continue
            test_one(scenario, topo, weight_label, device)
            state["test"][key] = True
            save_state(state)
    validate_test_outputs()


def line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def validate_test_outputs() -> None:
    for scenario in SCENARIOS:
        for weight_label in WEIGHT_CONFIGS:
            values_path, mlu_path, metrics_path = model_output_paths(scenario, weight_label)
            if line_count(values_path) != (TEST_END - VAL_END) * 3:
                raise RuntimeError(f"{values_path} has invalid line count")
            if line_count(mlu_path) != (TEST_END - VAL_END) * 3:
                raise RuntimeError(f"{mlu_path} has invalid line count")
            if len(read_csv(metrics_path)) != (TEST_END - VAL_END) * 3:
                raise RuntimeError(f"{metrics_path} has invalid row count")


def build_combined_rows() -> list[dict]:
    rows = []
    for row in read_csv(HATTRICK_RESULTS):
        if row["run_type"] != "retrained":
            continue
        rows.append(
            {
                "scenario": row["scenario"],
                "weights": "na",
                "topo": row["topo"],
                "snapshot": int(row["snapshot"]),
                "method": row["method"],
                "class": row["class"],
                "admitted_traffic": float(row["admitted_traffic"]),
                "demand": float(row["demand"]),
                "fulfill_ratio": float(row["fulfill_ratio"]),
                "mlu": float(row["mlu"]),
                "norm_fulfill": float(row["norm_fulfill"]),
            }
        )
    for scenario in SCENARIOS:
        for weight_label in WEIGHT_CONFIGS:
            rows.extend(read_csv(model_output_paths(scenario, weight_label)[2]))
    for row in rows:
        for key in ("snapshot", "admitted_traffic", "demand", "fulfill_ratio", "mlu", "norm_fulfill"):
            if key == "snapshot":
                row[key] = int(row[key])
            else:
                row[key] = float(row[key])
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    groups = sorted({(r["scenario"], r["weights"], r["method"], r["class"]) for r in rows})
    for scenario, weights, method, class_name in groups:
        vals = [r for r in rows if r["scenario"] == scenario and r["weights"] == weights and r["method"] == method and r["class"] == class_name]
        item = {"scenario": scenario, "weights": weights, "method": method, "class": class_name, "n": len(vals)}
        for metric in ("admitted_traffic", "fulfill_ratio", "mlu", "norm_fulfill"):
            arr = np.asarray([float(v[metric]) for v in vals], dtype=np.float64)
            item[f"{metric}_mean"] = float(arr.mean())
            item[f"{metric}_median"] = float(np.median(arr))
            item[f"{metric}_p10"] = float(np.percentile(arr, 10))
            item[f"{metric}_p1"] = float(np.percentile(arr, 1))
        summary.append(item)
    return summary


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def default_rows(rows: list[dict]) -> list[dict]:
    keep = []
    for row in rows:
        if row["method"] == "DOTE_MC" and row["weights"] != DEFAULT_WEIGHT_LABEL:
            continue
        keep.append(row)
    return keep


def draw_cdf(rows: list[dict]) -> None:
    rows = default_rows(rows)
    methods = ("Hattrick", "DOTE_MC", "BEST_MC", "SWAN")
    colors = {"Hattrick": "#2F80ED", "DOTE_MC": "#7B61FF", "BEST_MC": "#27AE60", "SWAN": "#EB5757"}
    width, height = 1420, 1040
    left, top = 84, 108
    panel_w, panel_h = 290, 180
    gap_x, gap_y = 38, 58
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 28), "DOTE-MC vs Hattrick: CDF of FulfillRatio", fill="#202124", font=font(28, True))
    draw.text((left, 62), f"Default DOTE-MC weights {WEIGHT_CONFIGS[DEFAULT_WEIGHT_LABEL]}; empirical CDF over snapshots 90-99.", fill="#5f6368", font=font(15))
    x_min, x_max = 0.0, 1.15
    for row_idx, scenario in enumerate(SCENARIOS):
        draw.text((left, top + row_idx * (panel_h + gap_y) - 28), scenario, fill="#202124", font=font(16, True))
        for col_idx, class_name in enumerate(CLASSES):
            px = left + col_idx * (panel_w + gap_x)
            py = top + row_idx * (panel_h + gap_y)
            draw.rectangle((px, py, px + panel_w, py + panel_h), outline="#202124", width=1)
            draw.text((px + panel_w // 2 - 34, py + panel_h + 8), class_name, fill="#202124", font=font(14))
            for tick in np.arange(0, 1.151, 0.25):
                x = int(px + (tick - x_min) / (x_max - x_min) * panel_w)
                draw.line((x, py, x, py + panel_h), fill="#E8EAED")
                draw.text((x - 10, py + panel_h + 24), f"{tick:.2g}", fill="#5f6368", font=font(11))
            for tick in np.linspace(0, 1, 5):
                y = int(py + panel_h - tick * panel_h)
                draw.line((px, y, px + panel_w, y), fill="#E8EAED")
            for method in methods:
                vals = sorted(float(r["fulfill_ratio"]) for r in rows if r["scenario"] == scenario and r["class"] == class_name and r["method"] == method)
                if not vals:
                    continue
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                xs = [x_min] + vals + [x_max]
                ys = [0.0] + list(cdf) + [1.0]
                points = [(int(px + (min(max(v, x_min), x_max) - x_min) / (x_max - x_min) * panel_w), int(py + panel_h - p * panel_h)) for v, p in zip(xs, ys)]
                draw.line(points, fill=colors[method], width=2)
    lx, ly = left, height - 55
    for method in methods:
        draw.line((lx, ly, lx + 36, ly), fill=colors[method], width=5)
        draw.text((lx + 46, ly - 10), method, fill="#202124", font=font(16))
        lx += 190
    image.save(RESULTS / "dotemc_vs_hattrick_cdf.png")


def draw_boxplot(rows: list[dict]) -> None:
    rows = default_rows(rows)
    methods = ("Hattrick", "DOTE_MC", "BEST_MC", "SWAN")
    colors = {"Hattrick": "#2F80ED", "DOTE_MC": "#7B61FF", "BEST_MC": "#27AE60", "SWAN": "#EB5757"}
    image = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "DOTE-MC vs Hattrick: FulfillRatio boxplot", fill="#202124", font=font(28, True))
    draw.text((70, 64), "Default DOTE-MC weights; all classes over snapshots 90-99.", fill="#5f6368", font=font(15))
    left, top, bottom = 110, 120, 620
    x_gap = 285
    all_vals = [float(r["fulfill_ratio"]) for r in rows]
    y_min, y_max = 0.0, max(1.05, math.ceil((max(all_vals) + 0.05) * 4) / 4)

    def y_of(value: float) -> int:
        return int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))

    for tick in np.arange(0, y_max + 0.001, 0.25):
        y = y_of(float(tick))
        draw.line((left - 40, y, 1210, y), fill="#E8EAED")
        draw.text((32, y - 8), f"{tick:.2f}", fill="#5f6368", font=font(13))
    for idx, scenario in enumerate(SCENARIOS):
        x0 = left + idx * x_gap
        draw.text((x0 - 10, bottom + 22), scenario, fill="#202124", font=font(14))
        for method_idx, method in enumerate(methods):
            vals = np.asarray([float(r["fulfill_ratio"]) for r in rows if r["scenario"] == scenario and r["method"] == method], dtype=np.float64)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            low, high = np.percentile(vals, [5, 95])
            mean = float(vals.mean())
            cx = x0 + method_idx * 38 + 15
            box_left, box_right = cx - 12, cx + 12
            draw.line((cx, y_of(low), cx, y_of(high)), fill=colors[method], width=2)
            draw.line((box_left, y_of(low), box_right, y_of(low)), fill=colors[method], width=2)
            draw.line((box_left, y_of(high), box_right, y_of(high)), fill=colors[method], width=2)
            draw.rectangle((box_left, y_of(q3), box_right, y_of(q1)), outline=colors[method], width=3)
            draw.line((box_left, y_of(med), box_right, y_of(med)), fill=colors[method], width=3)
            draw.ellipse((cx - 3, y_of(mean) - 3, cx + 3, y_of(mean) + 3), fill=colors[method])
    lx, ly = 70, 680
    for method in methods:
        draw.rectangle((lx, ly, lx + 24, ly + 14), fill=colors[method])
        draw.text((lx + 34, ly - 4), method, fill="#202124", font=font(15))
        lx += 160
    image.save(RESULTS / "dotemc_vs_hattrick_boxplot.png")


def avg_summary(summary: list[dict], scenario: str, method: str, weights: str = "na") -> float:
    vals = [float(row["fulfill_ratio_mean"]) for row in summary if row["scenario"] == scenario and row["method"] == method and row["weights"] == weights]
    return float(np.mean(vals))


def structure_report(summary: list[dict]) -> str:
    lines = [
        "# DOTE-MC Priority-Mask Structure Analysis\n\n",
        "Goal: test whether Hattrick's structural advantage over a simple DNN baseline disappears when priorities have different path sets.\n\n",
        "## Mean FulfillRatio, averaged over classes\n\n",
        "| Scenario | Hattrick | DOTE-MC default | DOTE-MC sensitivity | BEST_MC | SWAN | Hattrick - DOTE default |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    gaps = {}
    for scenario in SCENARIOS:
        hat = avg_summary(summary, scenario, "Hattrick")
        dote_default = avg_summary(summary, scenario, "DOTE_MC", DEFAULT_WEIGHT_LABEL)
        dote_sens = avg_summary(summary, scenario, "DOTE_MC", "w_1_0p1_0p01")
        best = avg_summary(summary, scenario, "BEST_MC")
        swan = avg_summary(summary, scenario, "SWAN")
        gap = hat - dote_default
        gaps[scenario] = gap
        lines.append(f"| {scenario} | {hat:.6f} | {dote_default:.6f} | {dote_sens:.6f} | {best:.6f} | {swan:.6f} | {gap:+.6f} |\n")
    close_masked = all(abs(gaps[s]) <= 0.03 for s in ("mild", "medium", "strict"))
    shared_advantage = gaps["shared"] > 0.05
    if shared_advantage and close_masked:
        conclusion = "Hattrick has a clear shared-path advantage, but that advantage largely disappears under priority-specific path masks."
    elif all(abs(gap) <= 0.03 for gap in gaps.values()):
        conclusion = "On this 100-slice test, DOTE-MC is close to Hattrick in all scenarios; this does not prove Hattrick fails, but it weakens the evidence for a structural advantage at this scale."
    elif any(gaps[s] > 0.03 for s in ("mild", "medium", "strict")):
        conclusion = "Hattrick still has a meaningful advantage in at least one masked scenario, so the result does not support a structure-failure claim."
    else:
        conclusion = "The result is mixed; inspect per-class tails before making a structure claim."
    lines.extend(
        [
            "\n## Conclusion\n\n",
            conclusion + "\n\n",
            "This is a 100-slice experiment with 10 test snapshots. Treat it as preliminary evidence; a stronger claim needs a larger test window and multiple seeds.\n",
        ]
    )
    return "".join(lines)


def run_report_stage() -> None:
    rows = build_combined_rows()
    summary = summarize(rows)
    write_csv(RESULTS / "dotemc_priority_mask_metrics_100slice.csv", rows)
    write_csv(RESULTS / "dotemc_priority_mask_summary_stats.csv", summary)
    draw_cdf(rows)
    draw_boxplot(rows)
    (RESULTS / "dotemc_structure_failure_analysis.md").write_text(structure_report(summary), encoding="utf-8")
    state = load_state()
    state["report"] = True
    save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "validate", "smoke", "train", "test", "report"], default="all")
    args = parser.parse_args()
    ensure_dirs()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage in ("all", "validate"):
        validate_data()
        state = load_state()
        state["validate"] = True
        save_state(state)
        print("[ok] data and mask validation passed", flush=True)
    if args.stage in ("all", "smoke"):
        smoke_model(device)
        state = load_state()
        state["smoke"] = True
        save_state(state)
        print("[ok] DOTE-MC smoke test passed", flush=True)
    if args.stage in ("all", "train"):
        run_train_stage(device)
    if args.stage in ("all", "test"):
        run_test_stage(device)
    if args.stage in ("all", "report"):
        run_report_stage()
        print(f"[ok] wrote report to {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
