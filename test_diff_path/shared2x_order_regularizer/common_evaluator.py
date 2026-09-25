from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import run_experiment as runner

import run_dotemc_priority_mask_experiment as dote
import run_hattrick_strict2x_research as base
from frameworks.hattrick_system import Hattrick


HATTRICK_CHECKPOINT = runner.ROOT / f"hattrick_{runner.TOPOLOGY}_{runner.K}sp.pkl"
DOTE_CHECKPOINT = (
    runner.TEST_DIR
    / "results_load2x_retrain_shared_strict"
    / "outputs"
    / "shared"
    / "w_1_0p1_0p01"
    / "best_model.pt"
)
OUTPUT_DIR = runner.OUTPUT_ROOT / "common_final_existing_methods"
START, END = 400, 500


def replay_dote(
    model,
    mean: torch.Tensor,
    std: torch.Tensor,
    simulator: Hattrick,
    props,
    dataset,
) -> list[dict]:
    rows: list[dict] = []
    loader = base.data_loader(dataset, 1, False, 0)
    pte = dataset.pte.coalesce()
    pte_indices = pte.indices()
    pte_info = (pte, pte_indices[0], pte_indices[1], pte.values())
    with torch.no_grad():
        for local_index, inputs in enumerate(loader):
            values = base.unpack_to_device(inputs, props)
            (
                _node_features,
                capacities,
                tm1,
                tm1_pred,
                tm2,
                tm2_pred,
                tm3,
                tm3_pred,
                _opt1,
                _opt2,
                _opt3,
                opt1_mf,
                opt2_mf,
                opt3_mf,
                _snapshots,
            ) = values
            features = dote.make_inputs(
                tm1_pred.to(dtype=torch.float32),
                tm2_pred.to(dtype=torch.float32),
                tm3_pred.to(dtype=torch.float32),
                dataset.num_pairs,
                mean,
                std,
            )
            policy = model(features, None)
            policy_tensors = [
                policy[:, class_index].reshape(1, -1, 1)
                for class_index in range(3)
            ]
            admitted_fraction = simulator.simulate(
                policy_tensors,
                [tm1, tm2, tm3],
                capacities[:1],
                pte_info,
                1,
                props,
                rate_cap=props.rate_cap,
            )[:3]
            tms = (tm1, tm2, tm3)
            oracle = (opt1_mf, opt2_mf - opt1_mf, opt3_mf - opt2_mf)
            cumulative = torch.zeros_like(tm1.squeeze(-1))
            for class_index, class_name in enumerate(runner.CLASSES):
                admitted = (
                    admitted_fraction[class_index].reshape(1, -1)
                    * tms[class_index].squeeze(-1)
                )
                cumulative = cumulative + admitted
                link_load = torch.sparse.mm(
                    dataset.pte.to(dtype=torch.float32).t(),
                    cumulative.to(dtype=torch.float32).t(),
                ).t()
                total = float(admitted.sum().item())
                demand = float((tms[class_index].sum() / runner.K).item())
                oracle_flow = max(float(oracle[class_index].item()), 1e-9)
                rows.append(
                    {
                        "method": "DOTE-MC sensitivity",
                        "snapshot": START + local_index,
                        "class": class_name,
                        "admitted_traffic": total,
                        "demand": demand,
                        "fulfill_ratio": total / max(demand, 1e-9),
                        "oracle_admitted_traffic": oracle_flow,
                        "norm_fulfill": total / oracle_flow,
                        "admitted_capacity_ratio": float(
                            (link_load / capacities[:1].to(dtype=torch.float32)).max().item()
                        ),
                        "disabled_flow": 0.0,
                    }
                )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for method in ("Hattrick", "DOTE-MC sensitivity"):
        selected_method = [row for row in rows if row["method"] == method]
        for class_name in runner.CLASSES:
            selected = [row for row in selected_method if row["class"] == class_name]
            norm = np.asarray([float(row["norm_fulfill"]) for row in selected])
            fulfill = np.asarray([float(row["fulfill_ratio"]) for row in selected])
            capacity = np.asarray(
                [float(row["admitted_capacity_ratio"]) for row in selected]
            )
            output.append(
                {
                    "method": method,
                    "class": class_name,
                    "n": len(selected),
                    "norm_fulfill_mean": float(norm.mean()),
                    "norm_fulfill_p1": float(np.percentile(norm, 1)),
                    "norm_fulfill_p10": float(np.percentile(norm, 10)),
                    "fulfill_ratio_mean": float(fulfill.mean()),
                    "common_post_admission_mlu_mean": float(capacity.mean()),
                    "common_post_admission_mlu_max": float(capacity.max()),
                    "max_disabled_flow": float(
                        max(float(row["disabled_flow"]) for row in selected)
                    ),
                }
            )
    return output


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runner.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    dataset = runner.DM_Dataset_within_Cluster(props, 0, START, END)
    if int(dataset.max_source_index_read) != END - 1:
        raise RuntimeError("Final existing-method evaluator read outside 400-499")
    masks = base.move_dataset_static(dataset, device)
    if masks is not None:
        raise RuntimeError("Shared-path evaluator unexpectedly received path masks")

    hattrick = torch.load(HATTRICK_CHECKPOINT, map_location=device, weights_only=False)
    hattrick = hattrick.to(device=device, dtype=props.dtype).eval()
    hattrick_rows, _ = base.evaluate(hattrick, props, dataset, START)
    for row in hattrick_rows:
        row["method"] = "Hattrick"

    dote_model, dote_mean, dote_std = dote.load_checkpoint(DOTE_CHECKPOINT, device)
    simulator = Hattrick(props).to(device=device, dtype=props.dtype).eval()
    dote_rows = replay_dote(
        dote_model,
        dote_mean,
        dote_std,
        simulator,
        props,
        dataset,
    )
    rows = hattrick_rows + dote_rows
    summary = summarize(rows)
    if max(float(row["common_post_admission_mlu_max"]) for row in summary) > 1.0001:
        raise RuntimeError("Common evaluator capacity assertion failed")
    if max(float(row["max_disabled_flow"]) for row in summary) > 1e-8:
        raise RuntimeError("Common evaluator disabled-flow assertion failed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner.write_csv(OUTPUT_DIR / "common_metrics.csv", rows)
    runner.write_json(OUTPUT_DIR / "common_summary.json", summary)
    runner.write_json(
        OUTPUT_DIR / "provenance.json",
        {
            "window": [START, END],
            "window_status": "previously viewed confirmation window; no regularized checkpoint evaluated",
            "metric_contract": "exact sequential actual-TM admission; cumulative admitted link load/capacity",
            "dataset_max_source_index_read": int(dataset.max_source_index_read),
            "hattrick_checkpoint": str(HATTRICK_CHECKPOINT),
            "hattrick_checkpoint_sha256": runner.sha256(HATTRICK_CHECKPOINT),
            "dote_checkpoint": str(DOTE_CHECKPOINT),
            "dote_checkpoint_sha256": runner.sha256(DOTE_CHECKPOINT),
            "source_sha256": runner.sha256(Path(__file__).resolve()),
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
