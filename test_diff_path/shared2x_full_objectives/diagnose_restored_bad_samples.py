from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
TWO_PHASE_DIR = THIS_DIR / "two_phase"
DEFAULT_RUN_DIR = THIS_DIR / "artifacts" / "level4_confirmation" / "seed_490"
DEFAULT_OUTPUT_DIR = THIS_DIR / "artifacts" / "restored_bad_sample_diagnostics"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(TWO_PHASE_DIR))

import diagnose_bad_samples as detail  # noqa: E402
from frameworks.hattrick_system import Hattrick  # noqa: E402
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster  # noqa: E402


CLASSES = ("High", "Medium", "Low")


def metric_index(rows: list[dict[str, str]]) -> dict[tuple[int, str], dict]:
    result: dict[tuple[int, str], dict] = {}
    text_fields = {"snapshot", "class", "checkpoint"}
    for row in rows:
        converted = {
            key: (value if key in text_fields else float(value))
            for key, value in row.items()
        }
        result[(int(row["snapshot"]), row["class"])] = converted
    return result


def select_bad_samples(metrics: dict[tuple[int, str], dict]) -> list[dict]:
    snapshots = sorted(snapshot for snapshot, name in metrics if name == "Medium")
    rows: list[dict] = []
    for snapshot in snapshots:
        high = metrics[(snapshot, "High")]
        medium = metrics[(snapshot, "Medium")]
        low = metrics[(snapshot, "Low")]
        rows.append(
            {
                "snapshot": snapshot,
                "high_norm_fulfill": float(high["norm_fulfill"]),
                "medium_norm_fulfill": float(medium["norm_fulfill"]),
                "low_norm_fulfill": float(low["norm_fulfill"]),
                "high_shortfall": float(high["oracle_admitted_traffic"])
                - float(high["admitted_traffic"]),
                "medium_shortfall": float(medium["oracle_admitted_traffic"])
                - float(medium["admitted_traffic"]),
                "low_excess": float(low["admitted_traffic"])
                - float(low["oracle_admitted_traffic"]),
                "inversion_gap_low_minus_medium": float(low["norm_fulfill"])
                - float(medium["norm_fulfill"]),
            }
        )

    criteria = (
        ("Medium NormFulFill 最低", lambda row: row["medium_norm_fulfill"]),
        ("Medium 绝对流量缺口最大", lambda row: -row["medium_shortfall"]),
        ("Low-Medium inversion gap 最大", lambda row: -row["inversion_gap_low_minus_medium"]),
    )
    selected: list[dict] = []
    used: set[int] = set()
    for reason, key in criteria:
        for row in sorted(rows, key=key):
            if int(row["snapshot"]) in used:
                continue
            chosen = dict(row)
            chosen["selection_reason"] = reason
            selected.append(chosen)
            used.add(int(row["snapshot"]))
            break
    return selected


def replay(args: argparse.Namespace) -> dict:
    runner = detail.core.round1
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "best_evaluation_metrics.csv"
    metrics = metric_index(detail.read_csv(metrics_path))
    selected = select_bad_samples(metrics)
    selected_ids = {int(row["snapshot"]) for row in selected}

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    runner.set_seed(args.seed)
    props = runner.build_props(4, device)
    props.batch_size = 1
    checkpoint_path = run_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")

    eval_start, eval_end = runner.LEVELS[4]["evaluation"]
    dataset = DM_Dataset_within_Cluster(props, 0, eval_start, eval_end)
    if int(dataset.max_source_index_read) != eval_end - 1:
        raise RuntimeError("Split-safe data audit failed")
    path_masks = runner.base.move_dataset_static(dataset, device)
    loader = runner.data_loader(dataset, 1, False, args.seed)
    replay_rows: list[dict] = []
    exported: list[int] = []
    k = int(props.num_paths_per_pair)

    for local_index, inputs in enumerate(loader):
        snapshot_id = eval_start + local_index
        if snapshot_id not in selected_ids:
            continue
        values = runner.unpack_to_device(inputs, props)
        snapshot = values[14][0]

        props.mode = "test"
        props.sim_mf_mlu = 0
        props.research_return_policy = True
        with torch.no_grad():
            policies, _ = runner.model_forward(
                model, props, dataset, values, path_masks
            )
        props.research_return_policy = False
        props.sim_mf_mlu = 1
        with torch.no_grad():
            admitted, _ = runner.model_forward(
                model, props, dataset, values, path_masks
            )
        props.sim_mf_mlu = 0

        detail.export_snapshot_detail(
            snapshot_id,
            snapshot,
            dataset,
            values,
            policies,
            admitted,
            metrics,
            metrics,
            output_dir,
            k,
            None,
        )
        exported.append(snapshot_id)
        oracle_totals = (
            values[11],
            values[12] - values[11],
            values[13] - values[12],
        )
        for class_index, class_name in enumerate(CLASSES):
            replay_norm = float(admitted[class_index].sum().item()) / max(
                float(oracle_totals[class_index].reshape(-1)[0].item()), 1e-12
            )
            saved_norm = float(metrics[(snapshot_id, class_name)]["norm_fulfill"])
            replay_rows.append(
                {
                    "snapshot": snapshot_id,
                    "class": class_name,
                    "saved_norm_fulfill": saved_norm,
                    "replay_norm_fulfill": replay_norm,
                    "absolute_error": abs(saved_norm - replay_norm),
                }
            )

    if set(exported) != selected_ids:
        raise RuntimeError(f"Missing selected snapshots: {selected_ids - set(exported)}")
    max_replay_error = max(float(row["absolute_error"]) for row in replay_rows)
    if max_replay_error > 2e-5:
        raise RuntimeError(f"Checkpoint replay mismatch: {max_replay_error}")

    detail.write_csv(output_dir / "selected_bad_samples.csv", selected)
    detail.write_csv(output_dir / "checkpoint_replay_checks.csv", replay_rows)
    result = {
        "scope": f"Level-4 confirmation snapshots {eval_start}-{eval_end - 1}; already viewed confirmation window",
        "model": "original Hattrick with restored Fh and cumulative Fhm objectives; no two-phase modification",
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selected_samples": selected,
        "max_checkpoint_replay_error": max_replay_error,
        "output_dir": str(output_dir),
    }
    detail.write_json(output_dir / "diagnostics.json", result)
    return result


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay bad samples from restored six-objective original Hattrick"
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = replay(parse_cli())
    print(
        json.dumps(
            {
                "selected": [row["snapshot"] for row in result["selected_samples"]],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "max_checkpoint_replay_error": result["max_checkpoint_replay_error"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
