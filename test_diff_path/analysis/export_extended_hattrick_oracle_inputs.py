from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "test_diff_path"
FULL_DIR = TEST_DIR / "shared2x_full_objectives"
TWO_PHASE_DIR = FULL_DIR / "two_phase"
DEFAULT_RUN_DIR = FULL_DIR / "artifacts" / "level4_confirmation" / "seed_490"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "analysis" / "hattrick_oracle_edges_extended" / "raw"
DEFAULT_SNAPSHOTS = tuple(sorted(set(range(400, 500, 5)) | {459, 473, 476}))
CLASSES = ("High", "Medium", "Low")

for path in (ROOT, FULL_DIR, TWO_PHASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnose_bad_samples as detail  # noqa: E402
import diagnose_restored_bad_samples as restored  # noqa: E402
from frameworks.hattrick_system import Hattrick  # noqa: E402
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster  # noqa: E402


def replay(args: argparse.Namespace) -> dict:
    runner = detail.core.round1
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = tuple(sorted(set(int(value) for value in args.snapshots)))

    metrics = restored.metric_index(detail.read_csv(run_dir / "best_evaluation_metrics.csv"))
    available = {snapshot for snapshot, class_name in metrics if class_name == "Medium"}
    missing = sorted(set(snapshots) - available)
    if missing:
        raise ValueError(f"Snapshots outside the saved evaluation window: {missing}")

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

    exported: list[int] = []
    replay_rows: list[dict] = []
    k = int(props.num_paths_per_pair)
    selected = set(snapshots)
    for local_index, inputs in enumerate(loader):
        snapshot_id = eval_start + local_index
        if snapshot_id not in selected:
            continue
        values = runner.unpack_to_device(inputs, props)
        snapshot = values[14][0]

        props.mode = "test"
        props.sim_mf_mlu = 0
        props.research_return_policy = True
        with torch.no_grad():
            policies, _ = runner.model_forward(model, props, dataset, values, path_masks)
        props.research_return_policy = False
        props.sim_mf_mlu = 1
        with torch.no_grad():
            admitted, _ = runner.model_forward(model, props, dataset, values, path_masks)
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
        oracle_totals = (values[11], values[12] - values[11], values[13] - values[12])
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

    if set(exported) != selected:
        raise RuntimeError(f"Missing snapshots: {sorted(selected - set(exported))}")
    max_replay_error = max(float(row["absolute_error"]) for row in replay_rows)
    if max_replay_error > 2e-5:
        raise RuntimeError(f"Checkpoint replay mismatch: {max_replay_error}")

    detail.write_csv(output_dir / "checkpoint_replay_checks.csv", replay_rows)
    payload = {
        "model": "original six-objective Hattrick",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "inference": "strict ESM",
        "evaluation_window": [eval_start, eval_end],
        "snapshots": list(snapshots),
        "regular_grid_snapshots": list(range(400, 500, 5)),
        "previously_selected_bad_snapshots": [459, 473, 476],
        "device": str(device),
        "max_checkpoint_replay_error": max_replay_error,
    }
    detail.write_json(output_dir / "replay_manifest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Hattrick edge loads for an extended snapshot sample")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--snapshots", type=int, nargs="+", default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(replay(parse_args()), indent=2, ensure_ascii=False))
