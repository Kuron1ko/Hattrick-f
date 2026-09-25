from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
OUTPUT_DIR = ROOT / "output" / "comparisons" / "dotemc_hattricke"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


for path in (ROOT, THIS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

dote = load_module(
    "dotemc_hattricke_dote_runtime",
    THIS_DIR / "run_dotemc_priority_mask_experiment.py",
)
edge = load_module(
    "dotemc_hattricke_edge_runtime",
    THIS_DIR / "shared2x_edge_toll_head" / "run_experiment.py",
)
onex = load_module(
    "dotemc_hattricke_onex_runtime",
    THIS_DIR / "shared1x_esm_sar" / "probe_onex_transfer.py",
)


CLASSES = ("High", "Medium", "Low")
WINDOW = (400, 500)
DOTE_CHECKPOINTS = {
    "1x": THIS_DIR
    / "results_dotemc_priority_masks_500slice"
    / "outputs"
    / "shared"
    / "w_1_0p1_0p01"
    / "best_model.pt",
    "2x": THIS_DIR
    / "results_load2x_retrain_shared_strict"
    / "outputs"
    / "shared"
    / "w_1_0p1_0p01"
    / "best_model.pt",
}
REFERENCE_ROWS = {
    ("1x", "Hattrick"): THIS_DIR
    / "shared1x_edge_toll_head"
    / "onex_level4_hattrick_rows.csv",
    ("1x", "Hatrrick-e"): THIS_DIR
    / "shared1x_edge_toll_head"
    / "onex_level4_edge_toll_rows.csv",
    ("2x", "Hattrick"): THIS_DIR
    / "shared2x_edge_toll_head"
    / "strict_esm_2x_hattrick_rows.csv",
    ("2x", "Hatrrick-e"): THIS_DIR
    / "shared2x_edge_toll_head"
    / "strict_esm_2x_candidate_rows.csv",
}


class AllPolicyAdapter:
    def __init__(self, path_count: int):
        self.path_count = int(path_count)

    def adapt_batch(self, policies, batch):
        values = batch["path_features"]
        for class_index in range(3):
            start = class_index * self.path_count
            stop = start + self.path_count
            policies[class_index] = values[:, start:stop].unsqueeze(-1)
        return policies


def dote_path_features(model, mean, std, cache, batch_size: int = 64):
    if cache.path_masks is not None:
        raise RuntimeError("The current comparison expects the shared 8-path dataset")
    outputs = []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            stop = min(start + batch_size, len(cache))
            predicted = [value[start:stop] for value in cache.predicted_tms]
            inputs = dote.make_inputs(
                predicted[0].to(dtype=torch.float32),
                predicted[1].to(dtype=torch.float32),
                predicted[2].to(dtype=torch.float32),
                cache.dataset.num_pairs,
                mean,
                std,
            )
            outputs.append(model(inputs, None).reshape(stop - start, -1))
    return torch.cat(outputs, dim=0)


def evaluate_dote(load: str, device: torch.device):
    if load == "1x":
        base_model, props = onex.load_onex(device)
        runtime = onex.runtime
        backbone = onex.MODEL_PATH
    else:
        runtime = edge.runtime
        props = runtime.build_props(4, device)
        base_model, backbone = runtime.load_backbone(4, 490, props, device)

    cache = runtime.build_policy_cache(
        base_model, props, WINDOW[0], WINDOW[1], batch_size=64
    )
    checkpoint = DOTE_CHECKPOINTS[load]
    model, mean, std = dote.load_checkpoint(checkpoint, device)
    policies = dote_path_features(model, mean, std, cache)

    # Strict-information audit: changing all actual traffic matrices must not
    # affect a policy that is generated solely from the stored ESM predictions.
    counterfactual_cache = replace(
        cache, tms=tuple(torch.zeros_like(value) for value in cache.tms)
    )
    counterfactual = dote_path_features(model, mean, std, counterfactual_cache)
    actual_tm_policy_max_abs_diff = float(
        (policies - counterfactual).abs().max().item()
    )

    policy_cache = replace(cache, path_features=policies)
    adapter = AllPolicyAdapter(int(cache.policies[0].shape[1]))
    rows, _ = runtime.evaluate_cache(
        base_model, props, policy_cache, adapter, batch_size=32
    )
    for row in rows:
        row["method"] = "DOTE-MC"
    return rows, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": runtime.sha256(checkpoint),
        "backbone_used_as_common_simulator": str(backbone),
        "actual_tm_policy_max_abs_diff": actual_tm_policy_max_abs_diff,
    }


def read_rows(path: Path, method: str) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["method"] = method
    return rows


def summarize(load: str, rows: list[dict]) -> list[dict]:
    output = []
    for method in ("Hattrick", "Hatrrick-e", "DOTE-MC"):
        for class_name in CLASSES:
            selected = [
                row
                for row in rows
                if row["method"] == method and row["class"] == class_name
            ]
            norm = np.asarray(
                [float(row["norm_fulfill"]) for row in selected], dtype=np.float64
            )
            fulfill = np.asarray(
                [float(row["fulfill_ratio"]) for row in selected], dtype=np.float64
            )
            output.append(
                {
                    "load": load,
                    "method": method,
                    "class": class_name,
                    "n": int(norm.size),
                    "norm_fulfill_mean": float(norm.mean()),
                    "norm_fulfill_p1": float(np.percentile(norm, 1)),
                    "norm_fulfill_p10": float(np.percentile(norm, 10)),
                    "fulfill_ratio_mean": float(fulfill.mean()),
                }
            )
    return output


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "load",
        "method",
        "snapshot",
        "class",
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.manual_seed(20260824)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    provenance = {}
    for load in ("1x", "2x"):
        rows = []
        for method in ("Hattrick", "Hatrrick-e"):
            rows.extend(read_rows(REFERENCE_ROWS[(load, method)], method))
        dote_rows, dote_provenance = evaluate_dote(load, device)
        rows.extend(dote_rows)
        for row in rows:
            row["load"] = load
        if len(rows) != 900:
            raise RuntimeError(f"Expected 900 rows for {load}, got {len(rows)}")
        write_rows(OUTPUT_DIR / f"comparison_rows_{load}.csv", rows)
        all_summaries.extend(summarize(load, rows))
        provenance[load] = dote_provenance

    indexed = {
        (row["load"], row["method"], row["class"]): row
        for row in all_summaries
    }
    deltas = []
    for load in ("1x", "2x"):
        for method in ("Hatrrick-e", "DOTE-MC"):
            for class_name in CLASSES:
                current = indexed[(load, method, class_name)]
                baseline = indexed[(load, "Hattrick", class_name)]
                deltas.append(
                    {
                        "load": load,
                        "method": method,
                        "class": class_name,
                        "mean_delta_vs_hattrick": current["norm_fulfill_mean"]
                        - baseline["norm_fulfill_mean"],
                        "p1_delta_vs_hattrick": current["norm_fulfill_p1"]
                        - baseline["norm_fulfill_p1"],
                        "p10_delta_vs_hattrick": current["norm_fulfill_p10"]
                        - baseline["norm_fulfill_p10"],
                    }
                )
    report = {
        "dataset": "GEANT shared 8sp, Level-4 snapshots 400-499",
        "protocol": {
            "1x": "DOTE-MC trained on 1x data and evaluated on 1x",
            "2x": "DOTE-MC retrained on 2x data and evaluated on 2x",
            "weights": [1.0, 0.1, 0.01],
            "inference": "strict ESM predictions only; actual traffic is evaluation only",
            "common_evaluator": "Hattrick exact sequential admission simulator",
        },
        "summary": all_summaries,
        "deltas_vs_hattrick": deltas,
        "provenance": provenance,
    }
    (OUTPUT_DIR / "comparison_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_table(OUTPUT_DIR / "comparison_summary.csv", all_summaries)
    write_table(OUTPUT_DIR / "deltas_vs_hattrick.csv", deltas)
    print(OUTPUT_DIR / "comparison_summary.json", flush=True)


if __name__ == "__main__":
    main()
