from __future__ import annotations

import argparse
import json
from pathlib import Path

import _workspace as ws


ORACLE_FILES = (
    "gt_optimal_values_mf.txt",
    "gt_optimal_values_mf_mf.txt",
    "gt_optimal_values_mf_mf_mf.txt",
    "gt_optimal_values_mlu.txt",
    "gt_optimal_values_mlu_mlu.txt",
    "gt_optimal_values_mlu_mlu_mlu.txt",
)


def nonempty_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def audit(dataset: str) -> dict:
    root = ws.BASE_ROOT / dataset
    expected = ws.DATASETS[dataset]["test"][1]
    files = {name: nonempty_lines(root / name) for name in (*ORACLE_FILES, "filenames.txt")}
    return {
        "dataset": dataset,
        "path": str(root.resolve()),
        "expected_snapshots": expected,
        "files": files,
        "complete": all(count == expected for count in files.values()),
        "gurobi_source": str((ws.ROOT / "frameworks/gurobi_refactored.py").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the local Gurobi baseline copied into baseresult/"
    )
    parser.add_argument("--dataset", required=True, choices=tuple(ws.DATASETS))
    parser.add_argument(
        "--show-recompute-command",
        action="store_true",
        help="show the vendored solver entry used to rebuild individual chunks",
    )
    args = parser.parse_args()
    result = audit(args.dataset)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["complete"]:
        raise SystemExit(
            f"baseline {args.dataset} is incomplete. The vendored Gurobi implementation is "
            "frameworks/gurobi_refactored.py; regenerate the missing oracle before training."
        )
    if args.show_recompute_command:
        topology = ws.DATASETS[args.dataset]["topology"]
        print("Example single chunk/pass (repeat priorities 1-3 for mf and mlu):")
        print(
            "python frameworks/gurobi_refactored.py "
            f"--topo {topology} --framework gurobi --num_paths_per_pair 8 "
            "--opt_start_idx 0 --opt_end_idx 256 --cluster 0 --pred 0 "
            "--gur_mode flexile --priority 1 --objs mf --path_mask 0 --tol 0.000001"
        )
    print(f"baseline{args.dataset} finished")


if __name__ == "__main__":
    main()
