from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import block_diag, coo_matrix, csr_matrix, hstack, vstack

import run_dotemc_priority_mask_experiment as dote
import run_hattrick_residual_lp as residual
import run_hattrick_strict2x_research as research
from frameworks.hattrick_system import Hattrick


SEEDS = (490, 491, 492)
MANIFEST = residual.OUTPUT_ROOT / "final_frozen_manifest.json"
DOTE_CHECKPOINT = (
    residual.ROOT
    / "test_diff_path"
    / "results_load2x_stress_500slice"
    / "outputs"
    / "strict"
    / "w_1_0p1_0p01"
    / "best_model.pt"
)
SWAN_DIR = (
    residual.ROOT.parent
    / "scratch"
    / "split_ratios"
    / research.TOPOLOGY
    / "swan"
    / "esm"
)
PRESERVE = 1.0 - 1e-5


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    own_hash = residual.sha256(Path(__file__).resolve())
    if not manifest.get("authorized", False):
        raise RuntimeError("Final test remains sealed")
    if manifest["source_sha256"]["test_diff_path/evaluate_strict2x_common_methods.py"] != own_hash:
        raise RuntimeError("Common evaluator differs from the frozen manifest")
    if manifest["source_sha256"]["test_diff_path/run_dotemc_priority_mask_experiment.py"] != residual.sha256(
        Path(dote.__file__).resolve()
    ):
        raise RuntimeError("DOTE evaluator source differs from the frozen manifest")
    if manifest["comparison_artifact_sha256"]["dote_checkpoint"] != residual.sha256(DOTE_CHECKPOINT):
        raise RuntimeError("DOTE comparison checkpoint differs from the frozen manifest")
    return manifest


def class_matrix(pte: csr_matrix, active: np.ndarray, demand: np.ndarray) -> csr_matrix:
    return pte[active, :].T.multiply(demand[active])


def pair_matrix(active: np.ndarray, num_pairs: int) -> csr_matrix:
    return coo_matrix(
        (np.ones(active.size), (active // research.K, np.arange(active.size))),
        shape=(num_pairs, active.size),
    ).tocsr()


def solve_stage(
    pte: csr_matrix,
    demands: list[np.ndarray],
    masks: list[np.ndarray],
    capacity: np.ndarray,
    preserve_high: float | None,
    preserve_high_medium: float | None,
) -> tuple[list[np.ndarray], float]:
    num_classes = len(demands)
    num_pairs = demands[0].size // research.K
    active = [np.flatnonzero(masks[index]) for index in range(num_classes)]
    lengths = [indices.size for indices in active]
    coefficients = [demands[index][active[index]] for index in range(num_classes)]
    link = hstack(
        [class_matrix(pte, active[index], demands[index]) for index in range(num_classes)],
        format="csr",
    )
    pairs = block_diag([pair_matrix(indices, num_pairs) for indices in active], format="csr")
    constraints = [link, pairs]
    rhs = [capacity, np.ones(num_pairs * num_classes)]
    if preserve_high is not None:
        row = np.zeros(sum(lengths), dtype=np.float64)
        row[: lengths[0]] = -coefficients[0]
        constraints.append(csr_matrix(row.reshape(1, -1)))
        rhs.append(np.asarray([-preserve_high], dtype=np.float64))
    if preserve_high_medium is not None:
        row = np.zeros(sum(lengths), dtype=np.float64)
        row[: lengths[0]] = -coefficients[0]
        row[lengths[0] : lengths[0] + lengths[1]] = -coefficients[1]
        constraints.append(csr_matrix(row.reshape(1, -1)))
        rhs.append(np.asarray([-preserve_high_medium], dtype=np.float64))
    objective = -np.concatenate(coefficients)
    result = linprog(
        objective,
        A_ub=vstack(constraints, format="csr"),
        b_ub=np.concatenate(rhs),
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"BEST LP failed: {result.status} {result.message}")
    policies = []
    offset = 0
    for index, length in enumerate(lengths):
        policy = np.zeros(demands[index].size, dtype=np.float32)
        policy[active[index]] = np.clip(result.x[offset : offset + length], 0.0, 1.0)
        policies.append(policy)
        offset += length
    return policies, float(-result.fun)


def best_policy(
    pte: csr_matrix, demands: list[np.ndarray], masks: list[np.ndarray], capacity: np.ndarray
) -> list[np.ndarray]:
    _, high = solve_stage(pte, demands[:1], masks[:1], capacity, None, None)
    _, high_medium = solve_stage(
        pte, demands[:2], masks[:2], capacity, PRESERVE * high, None
    )
    policy, _ = solve_stage(
        pte,
        demands,
        masks,
        capacity,
        PRESERVE * high,
        PRESERVE * high_medium,
    )
    return policy


def replay_rows(
    simulator: Hattrick,
    props,
    dataset,
    values,
    policies: list[np.ndarray],
    masks: list[torch.Tensor],
    pte_info,
    snapshot: int,
    method: str,
) -> list[dict]:
    (
        _node_features,
        capacities_full,
        tm1,
        _tm1_pred,
        tm2,
        _tm2_pred,
        tm3,
        _tm3_pred,
        _opt1,
        _opt2,
        _opt3,
        opt1_mf,
        opt2_mf,
        opt3_mf,
        _snapshots,
    ) = values
    tensors = [
        torch.from_numpy(np.asarray(policy, dtype=np.float32)).to(props.device).reshape(1, -1, 1)
        for policy in policies
    ]
    admitted_fraction = simulator.simulate(
        tensors, [tm1, tm2, tm3], capacities_full[:1], pte_info, 1, props, rate_cap=props.rate_cap
    )[:3]
    tms = (tm1, tm2, tm3)
    oracles = (opt1_mf, opt2_mf - opt1_mf, opt3_mf - opt2_mf)
    cumulative = torch.zeros_like(tm1.squeeze(-1))
    rows = []
    for class_index, class_name in enumerate(research.CLASSES):
        admitted = admitted_fraction[class_index].reshape(1, -1) * tms[class_index].squeeze(-1)
        cumulative = cumulative + admitted
        link_load = torch.sparse.mm(
            dataset.pte.to(dtype=torch.float32).t(), cumulative.to(dtype=torch.float32).t()
        ).t()
        oracle = max(float(oracles[class_index].item()), 1e-9)
        total = float(admitted.sum().item())
        demand = float((tms[class_index].sum() / research.K).item())
        disabled = ~masks[class_index].reshape(-1)
        rows.append(
            {
                "method": method,
                "snapshot": snapshot,
                "class": class_name,
                "admitted_traffic": total,
                "demand": demand,
                "fulfill_ratio": total / max(demand, 1e-9),
                "oracle_admitted_traffic": oracle,
                "norm_fulfill": total / oracle,
                "admitted_capacity_ratio": float(
                    (link_load / capacities_full[:1].to(dtype=torch.float32)).max().item()
                ),
                "disabled_flow": float(admitted[:, disabled].abs().max().item()) if disabled.any() else 0.0,
            }
        )
    return rows


def sha256_tree(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(residual.sha256(path)))
    return digest.hexdigest()


def summarize(method_rows: list[dict]) -> list[dict]:
    output = []
    for method in sorted({row["method"] for row in method_rows}):
        selected = [row for row in method_rows if row["method"] == method]
        for class_name in research.CLASSES:
            rows = [row for row in selected if row["class"] == class_name]
            norm = np.asarray([float(row["norm_fulfill"]) for row in rows])
            admitted = np.asarray([float(row["admitted_traffic"]) for row in rows])
            output.append(
                {
                    "method": method,
                    "class": class_name,
                    "n": len(rows),
                    "norm_fulfill_mean": float(norm.mean()),
                    "norm_fulfill_p1": float(np.percentile(norm, 1)),
                    "norm_fulfill_p10": float(np.percentile(norm, 10)),
                    "admitted_traffic_mean": float(admitted.mean()),
                    "common_post_admission_mlu_mean": float(
                        np.mean([float(row["admitted_capacity_ratio"]) for row in rows])
                    ),
                    "max_admitted_capacity_ratio": float(
                        max(float(row["admitted_capacity_ratio"]) for row in rows)
                    ),
                    "max_disabled_flow": float(max(float(row["disabled_flow"]) for row in rows)),
                }
            )
    return output


def main() -> None:
    manifest = load_manifest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = research.build_props(4, device)
    dataset = research.DM_Dataset_within_Cluster(props, 0, 400, 500)
    static_masks = research.move_dataset_static(dataset, device)
    if static_masks is None:
        raise RuntimeError("Common evaluator requires strict path masks")
    masks = [static_masks[index].reshape(-1).to(dtype=torch.bool) for index in range(3)]
    masks_np = [mask.cpu().numpy() for mask in masks]
    pte_info = residual.torch_pte_info(dataset)
    pte_scipy = residual.scipy_pte(dataset)
    simulator = Hattrick(props).to(device=device, dtype=props.dtype).eval()
    dote_model, dote_mean, dote_std = dote.load_checkpoint(DOTE_CHECKPOINT, device)
    loader = research.data_loader(dataset, 1, False, 0)

    generated = []
    swan_paths = []
    with torch.no_grad():
        for local_index, inputs in enumerate(loader):
            snapshot = 400 + local_index
            values = research.unpack_to_device(inputs, props)
            tm1_pred, tm2_pred, tm3_pred = values[3], values[5], values[7]
            features = dote.make_inputs(
                tm1_pred.to(dtype=torch.float32),
                tm2_pred.to(dtype=torch.float32),
                tm3_pred.to(dtype=torch.float32),
                dataset.num_pairs,
                dote_mean,
                dote_std,
            )
            dote_policy = dote_model(features, static_masks).reshape(3, -1).cpu().numpy()
            generated.extend(
                replay_rows(
                    simulator,
                    props,
                    dataset,
                    values,
                    [dote_policy[index] for index in range(3)],
                    masks,
                    pte_info,
                    snapshot,
                    "DOTE_MC",
                )
            )

            predicted = [
                values[index].squeeze().detach().cpu().numpy().astype(np.float64)
                for index in (3, 5, 7)
            ]
            capacity = values[1][:1].squeeze().detach().cpu().numpy().astype(np.float64)
            best = best_policy(pte_scipy, predicted, masks_np, capacity)
            generated.extend(
                replay_rows(
                    simulator, props, dataset, values, best, masks, pte_info, snapshot, "BEST_MC"
                )
            )

            swan_path = SWAN_DIR / f"{snapshot}.pkl"
            swan_paths.append(swan_path)
            with swan_path.open("rb") as handle:
                swan = [np.asarray(item, dtype=np.float32).reshape(-1) for item in pickle.load(handle)]
            generated.extend(
                replay_rows(
                    simulator, props, dataset, values, swan, masks, pte_info, snapshot, "SWAN"
                )
            )

    # The Hattrick and hybrid runners already use this same exact simulator and
    # oracle contract.  Pool their three frozen seed CSVs into the neutral table.
    level_dir = residual.OUTPUT_ROOT / research.LEVELS[4]["label"]
    all_rows = list(generated)
    for seed in SEEDS:
        for directory, method in (("unchanged", "Hattrick"), (residual.APPROACH, residual.APPROACH)):
            rows = research.read_csv(level_dir / directory / f"seed_{seed}" / "best_evaluation_metrics.csv")
            for row in rows:
                all_rows.append(
                    {
                        "method": method,
                        "snapshot": int(row["snapshot"]),
                        "class": row["class"],
                        "admitted_traffic": float(row["admitted_traffic"]),
                        "demand": float(row["demand"]),
                        "fulfill_ratio": float(row["fulfill_ratio"]),
                        "oracle_admitted_traffic": float(row["oracle_admitted_traffic"]),
                        "norm_fulfill": float(row["norm_fulfill"]),
                        "admitted_capacity_ratio": float(row["admitted_capacity_ratio"]),
                        "disabled_flow": float(row["disabled_flow"]),
                    }
                )
    output_dir = level_dir / "common_evaluator"
    output_dir.mkdir(parents=True, exist_ok=True)
    research.write_csv(output_dir / "common_method_metrics.csv", all_rows)
    summary = summarize(all_rows)
    (output_dir / "common_method_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    provenance = {
        "metric_contract": "exact sequential actual-TM admission, incremental MF oracle, common cumulative admitted load/capacity",
        "hattrick_seed_count": 3,
        "hybrid_seed_count": 3,
        "dote_checkpoint": str(DOTE_CHECKPOINT),
        "dote_checkpoint_sha256": residual.sha256(DOTE_CHECKPOINT),
        "swan_cache_sha256": sha256_tree(swan_paths),
        "best_policy": "ESM-predicted lexicographic LP with 1e-5 cumulative-priority preservation",
        "dataset_max_source_index_read": dataset.max_source_index_read,
        "source_sha256": {
            "test_diff_path/evaluate_strict2x_common_methods.py": residual.sha256(Path(__file__).resolve()),
            "final_frozen_manifest.json": residual.sha256(MANIFEST),
        },
        "manifest_frozen_at_utc": manifest["frozen_at_utc"],
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
