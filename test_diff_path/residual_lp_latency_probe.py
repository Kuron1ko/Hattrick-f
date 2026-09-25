from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import run_hattrick_residual_lp as residual
import run_hattrick_strict2x_research as research
from frameworks.hattrick_system import Hattrick


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure full Hattrick+residual-LP evaluation latency.")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.level == 4:
        raise RuntimeError("Level 4 remains sealed until the final-test manifest is frozen.")

    spec = research.LEVELS[args.level]
    start, end = spec["evaluation"]
    research.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = research.build_props(args.level, device)
    props.research_return_policy = False
    dataset = research.DM_Dataset_within_Cluster(props, 0, start, end)
    checkpoint_path = residual.selected_baseline_dir(args.level, args.seed) / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(checkpoint["model_state_dict"])

    # The first full pass warms the CUDA kernels and HiGHS machinery.  Timed
    # passes include neural inference, tensor/CPU transfers, both LPs, exact
    # actual-traffic replay, and metric construction for every snapshot.
    residual.evaluate_hybrid(model, props, dataset, start)
    durations = []
    for _ in range(args.repeats):
        synchronize(device)
        before = time.perf_counter()
        residual.evaluate_hybrid(model, props, dataset, start)
        synchronize(device)
        durations.append(time.perf_counter() - before)

    count = end - start
    milliseconds = [duration * 1000.0 / count for duration in durations]
    output = {
        "level": args.level,
        "seed": args.seed,
        "evaluation": [start, end],
        "snapshots_per_repeat": count,
        "repeats": args.repeats,
        "device": str(device),
        "scope": "neural inference + predicted High replay + Medium/Low LPs + exact actual-TM replay + metrics",
        "warmup_full_passes": 1,
        "seconds_per_repeat": durations,
        "milliseconds_per_snapshot": milliseconds,
        "median_milliseconds_per_snapshot": statistics.median(milliseconds),
        "dataset_access": {
            "loaded_index_range": list(dataset.loaded_index_range),
            "max_source_index_read": dataset.max_source_index_read,
        },
        "source_sha256": {
            "test_diff_path/residual_lp_latency_probe.py": residual.sha256(Path(__file__).resolve()),
            "test_diff_path/run_hattrick_residual_lp.py": residual.sha256(Path(residual.__file__).resolve()),
            "frameworks/hattrick_system.py": residual.sha256(residual.ROOT / "frameworks" / "hattrick_system.py"),
            "utils/build_dataset_within_cluster.py": residual.sha256(
                residual.ROOT / "utils" / "build_dataset_within_cluster.py"
            ),
            "matched_baseline_checkpoint": residual.sha256(checkpoint_path),
        },
    }
    run_dir = (
        residual.OUTPUT_ROOT
        / spec["label"]
        / residual.APPROACH
        / f"seed_{args.seed}"
    )
    path = run_dir / "end_to_end_latency.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
