from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
PROBE_PATH = THIS_DIR / "probe_onex_transfer.py"
spec = importlib.util.spec_from_file_location("shared1x_train_runtime", PROBE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load 1x ESM-SAR runtime")
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)
method = probe.method
runtime = probe.runtime

from frameworks.hattrick_system import Hattrick
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster


LEVELS = {
    1: {"label": "level1_correctness", "train": (0, 32), "validation": (32, 40), "epochs": 2},
    2: {"label": "level2_proxy", "train": (0, 160), "validation": (160, 200), "epochs": 12},
    4: {"label": "level4_full", "train": (0, 350), "validation": (350, 400), "epochs": 60},
}
SAR_STEPS_TRAIN = 4
SAR_STEPS_INFERENCE = 24
SAR_LR = 0.06
LOW_WEIGHT = 0.25
ANCHOR_WEIGHT = 0.01
HIGH_WEIGHT = 8.0
MODEL_LR = 5e-4
BATCH_SIZE = 8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def clear_topology_cache(model: Hattrick) -> None:
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")


def expanded_capacities(capacities: torch.Tensor, batch: int) -> torch.Tensor:
    return capacities.expand(batch, -1) if capacities.shape[0] == 1 and batch > 1 else capacities


def model_policies(model, props, dataset, values, path_masks):
    (
        node_features, capacities,
        tm1, tm1_pred, tm2, tm2_pred, tm3, tm3_pred,
        *_rest,
    ) = values
    batch = int(tm1.shape[0])
    if not props.dynamic:
        node_features = node_features[:1]
        capacities_for_model = capacities[:1]
    else:
        capacities_for_model = capacities
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    policies = model(
        props,
        node_features,
        dataset.edge_index,
        capacities_for_model,
        dataset.padded_edge_ids_per_path,
        tm1, tm1_pred, tm2, tm2_pred, tm3, tm3_pred,
        dataset.pte,
        dataset.edge_ids_dict_tensor,
        dataset.original_pos_edge_ids_dict_tensor,
        path_masks,
    )
    return list(policies), (tm1_pred, tm2_pred, tm3_pred), expanded_capacities(capacities, batch)


def predicted_fulfillment(model, props, dataset, policies, predicted_tms, capacities):
    batch = int(predicted_tms[0].shape[0])
    pte = dataset.pte.coalesce()
    indices = pte.indices()
    pte_info = (pte, indices[0], indices[1], pte.values())
    admitted_ratios = model.simulate(
        policies, list(predicted_tms), capacities, pte_info, batch, props,
        rate_cap=props.rate_cap,
    )[:3]
    values = []
    for ratio, tm in zip(admitted_ratios, predicted_tms):
        admitted = (ratio.reshape(batch, -1) * tm.squeeze(-1)).sum(dim=1)
        demand = tm.sum(dim=1).squeeze(-1) / 8
        values.append(admitted / demand.clamp_min(1e-9))
    return torch.stack(values, dim=1)


def searched_logit_deltas(model, props, dataset, base_policies, predicted_tms, capacities):
    base_medium = base_policies[1].squeeze(-1).detach()
    base_low = base_policies[2].squeeze(-1).detach()
    medium_logits = torch.nn.Parameter(torch.log(base_medium.clamp_min(1e-12)))
    low_logits = torch.nn.Parameter(torch.log(base_low.clamp_min(1e-12)))
    optimizer = torch.optim.Adam([medium_logits, low_logits], lr=SAR_LR)
    for _ in range(SAR_STEPS_TRAIN):
        optimizer.zero_grad(set_to_none=True)
        medium = method.routed_policy(medium_logits, base_medium)
        low = method.routed_policy(low_logits, base_low)
        policies = [base_policies[0].detach(), medium.unsqueeze(-1), low.unsqueeze(-1)]
        fulfill = predicted_fulfillment(
            model, props, dataset, policies, predicted_tms, capacities
        )
        grouped_medium = medium.reshape(len(medium), -1, 8)
        grouped_low = low.reshape(len(low), -1, 8)
        grouped_base_medium = base_medium.reshape(len(medium), -1, 8)
        grouped_base_low = base_low.reshape(len(low), -1, 8)
        anchor = (
            grouped_medium
            * torch.log((grouped_medium + 1e-12) / (grouped_base_medium + 1e-12))
        ).sum(dim=-1).mean()
        anchor = anchor + (
            grouped_low
            * torch.log((grouped_low + 1e-12) / (grouped_base_low + 1e-12))
        ).sum(dim=-1).mean()
        objective = (
            fulfill[:, 1].mean()
            + LOW_WEIGHT * fulfill[:, 2].mean()
            - ANCHOR_WEIGHT * anchor
        )
        (-objective).backward()
        optimizer.step()
    return (
        medium_logits.detach() - torch.log(base_medium.clamp_min(1e-12)),
        low_logits.detach() - torch.log(base_low.clamp_min(1e-12)),
    )


def training_loss(model, props, dataset, base_policies, predicted_tms, capacities):
    medium_delta, low_delta = searched_logit_deltas(
        model, props, dataset, base_policies, predicted_tms, capacities
    )
    base_medium = base_policies[1].squeeze(-1)
    base_low = base_policies[2].squeeze(-1)
    medium = method.routed_policy(
        torch.log(base_medium.clamp_min(1e-12)) + medium_delta, base_medium
    )
    low = method.routed_policy(
        torch.log(base_low.clamp_min(1e-12)) + low_delta, base_low
    )
    policies = [base_policies[0], medium.unsqueeze(-1), low.unsqueeze(-1)]
    fulfill = predicted_fulfillment(
        model, props, dataset, policies, predicted_tms, capacities
    )
    grouped_medium = medium.reshape(len(medium), -1, 8)
    grouped_low = low.reshape(len(low), -1, 8)
    grouped_base_medium = base_medium.reshape(len(medium), -1, 8)
    grouped_base_low = base_low.reshape(len(low), -1, 8)
    anchor = (
        grouped_medium
        * torch.log((grouped_medium + 1e-12) / (grouped_base_medium + 1e-12))
    ).sum(dim=-1).mean()
    anchor = anchor + (
        grouped_low
        * torch.log((grouped_low + 1e-12) / (grouped_base_low + 1e-12))
    ).sum(dim=-1).mean()
    objective = (
        HIGH_WEIGHT * fulfill[:, 0].mean()
        + fulfill[:, 1].mean()
        + LOW_WEIGHT * fulfill[:, 2].mean()
        - ANCHOR_WEIGHT * anchor
    )
    return -objective, fulfill.detach(), anchor.detach()


def train_epoch(model, props, dataset, loader, path_masks, optimizer) -> dict:
    model.train()
    clear_topology_cache(model)
    totals = torch.zeros(3, device=props.device)
    total_loss = 0.0
    total_anchor = 0.0
    batches = 0
    for inputs in loader:
        values = runtime.shared.unpack_to_device(inputs, props)
        optimizer.zero_grad(set_to_none=True)
        policies, predicted_tms, capacities = model_policies(
            model, props, dataset, values, path_masks
        )
        loss, fulfill, anchor = training_loss(
            model, props, dataset, policies, predicted_tms, capacities
        )
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite ESM-SAR training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 25.0)
        optimizer.step()
        total_loss += float(loss.item())
        total_anchor += float(anchor.item())
        totals += fulfill.mean(dim=0)
        batches += 1
    values = totals / max(batches, 1)
    return {
        "loss": total_loss / max(batches, 1),
        "predicted_high": float(values[0].item()),
        "predicted_medium": float(values[1].item()),
        "predicted_low": float(values[2].item()),
        "route_kl": total_anchor / max(batches, 1),
        "batches": batches,
    }


def corrected_cache(model, props, cache, strict: bool):
    config = (SAR_STEPS_INFERENCE, SAR_LR, LOW_WEIGHT, ANCHOR_WEIGHT)
    if not strict:
        return method.correct_cache(model, props, cache, *config)
    values = []
    for index in range(len(cache)):
        one = method.correct_cache(
            model, props, probe.slice_cache(cache, index, index + 1), *config
        )
        values.append(one.path_features)
    return cache.__class__(
        **{**cache.__dict__, "path_features": torch.cat(values, dim=0)}
    )


def evaluate_model(model, props, start: int, stop: int, strict: bool) -> dict:
    clear_topology_cache(model)
    model.eval()
    cache = runtime.build_policy_cache(model, props, start, stop, batch_size=32)
    base_rows, base_summary = runtime.evaluate_cache(model, props, cache, None, batch_size=32)
    refined = corrected_cache(model, props, cache, strict)
    adapter = method.CorrectedPolicyAdapter(int(cache.policies[0].shape[1]))
    rows, summary = runtime.evaluate_cache(model, props, refined, adapter, batch_size=32)
    return {
        "base_rows": base_rows,
        "base_summary": base_summary,
        "rows": rows,
        "summary": summary,
    }


def paired_class_values(rows: list[dict], class_name: str) -> np.ndarray:
    ordered = sorted(
        (
            (int(row["snapshot"]), float(row["norm_fulfill"]))
            for row in rows
            if row["class"] == class_name
        ),
        key=lambda item: item[0],
    )
    return np.asarray([value for _snapshot, value in ordered], dtype=np.float64)


def metric_score(
    summary: list[dict],
    original: list[dict],
    rows: list[dict],
    original_rows: list[dict],
) -> tuple:
    values = runtime.summary_index(summary)
    high_ok = float(values["High"]["norm_fulfill_mean"]) >= 0.995
    candidate_low = paired_class_values(rows, "Low")
    baseline_low = paired_class_values(original_rows, "Low")
    if candidate_low.shape != baseline_low.shape:
        raise RuntimeError("Candidate and original validation rows are not paired")
    low_delta = candidate_low - baseline_low
    low_mean_delta = float(low_delta.mean())
    low_se = float(low_delta.std(ddof=1) / math.sqrt(len(low_delta))) if len(low_delta) > 1 else 0.0
    # "No significant decline": the paired one-sided 95% upper confidence bound
    # must reach the small practical tolerance (-0.003). This avoids selecting on
    # one noisy validation mean while keeping Low as a real constraint.
    low_upper_95 = low_mean_delta + 1.645 * low_se
    low_ok = low_upper_95 >= -0.003
    medium = values["Medium"]
    return (
        int(high_ok and low_ok),
        float(medium["norm_fulfill_mean"])
        + 0.35 * float(medium["norm_fulfill_p1"])
        + 0.35 * float(medium["norm_fulfill_p10"]),
        low_upper_95,
        low_mean_delta,
        float(values["High"]["norm_fulfill_mean"]),
    )


def safe_remove(path: Path) -> None:
    root = (THIS_DIR / "artifacts").resolve()
    resolved = path.resolve()
    if root not in resolved.parents or resolved == root:
        raise RuntimeError(f"Unsafe run directory: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def run(level: int, seed: int, force: bool) -> None:
    setting = LEVELS[level]
    run_dir = THIS_DIR / "artifacts" / setting["label"] / f"seed_{seed}"
    if force:
        safe_remove(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime.shared.TOPOLOGY = probe.TOPOLOGY
    props = runtime.build_props(4, device)
    props.checkpoint = 0
    props.research_return_policy = True
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_LR)

    train_start, train_stop = setting["train"]
    val_start, val_stop = setting["validation"]
    train_dataset = DM_Dataset_within_Cluster(props, 0, train_start, train_stop)
    loader = runtime.shared.data_loader(train_dataset, BATCH_SIZE, True, seed)
    path_masks = runtime.shared.base.move_dataset_static(train_dataset, device)

    original_model, _ = probe.load_onex(device)
    original_evaluation = evaluate_model(
        original_model, props, val_start, val_stop, strict=False
    )
    original_validation = original_evaluation["base_summary"]
    original_validation_rows = original_evaluation["base_rows"]
    del original_model
    clear_topology_cache(model)

    history = []
    best_score = None
    best_epoch = 0
    best_path = run_dir / "best_model.pt"
    validation_every = 1 if level in (1, 2) else 5
    started = time.perf_counter()
    for epoch in range(1, int(setting["epochs"]) + 1):
        train_metrics = train_epoch(
            model, props, train_dataset, loader, path_masks, optimizer
        )
        record = {"epoch": epoch, **train_metrics}
        if epoch % validation_every == 0 or epoch == int(setting["epochs"]):
            validation = evaluate_model(model, props, val_start, val_stop, strict=False)
            score = metric_score(
                validation["summary"],
                original_validation,
                validation["rows"],
                original_validation_rows,
            )
            record["validation"] = method.compact(validation["summary"])
            record["score"] = list(score)
            if best_score is None or score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "score": score,
                        "seed": seed,
                    },
                    best_path,
                )
        history.append(record)
        (run_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    clear_topology_cache(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_validation = evaluate_model(model, props, val_start, val_stop, strict=True)
    payload = {
        "method": "from-scratch 1x Hattrick trained through first-order unrolled ESM-SAR",
        "level": level,
        "seed": seed,
        "topology": probe.TOPOLOGY,
        "train_range": list(setting["train"]),
        "validation_range": list(setting["validation"]),
        "epochs": setting["epochs"],
        "best_epoch": best_epoch,
        "elapsed_seconds": time.perf_counter() - started,
        "config": {
            "model_lr": MODEL_LR,
            "batch_size": BATCH_SIZE,
            "sar_train_steps": SAR_STEPS_TRAIN,
            "sar_inference_steps": SAR_STEPS_INFERENCE,
            "sar_lr": SAR_LR,
            "high_weight": HIGH_WEIGHT,
            "medium_weight": 1.0,
            "low_weight": LOW_WEIGHT,
            "anchor_weight": ANCHOR_WEIGHT,
            "selection_rule": (
                "High mean >= 0.995; Low paired one-sided 95% upper bound >= -0.003; "
                "then maximize Medium mean + 0.35*P1 + 0.35*P10"
            ),
        },
        "original_hattrick_validation": method.compact(original_validation),
        "candidate_base_validation": method.compact(final_validation["base_summary"]),
        "candidate_esm_sar_validation": method.compact(final_validation["summary"]),
        "candidate_vs_original_delta": method.gaps(
            final_validation["summary"], original_validation
        ),
        "checkpoint": str(best_path),
        "checkpoint_sha256": runtime.sha256(best_path),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    runtime.write_csv(run_dir / "validation_rows.csv", final_validation["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=tuple(LEVELS), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli()
    run(args.level, args.seed, args.force)
