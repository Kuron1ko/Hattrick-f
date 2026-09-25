from __future__ import annotations

import argparse

import _workspace as ws
from list import COLUMNS, format_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the validation-best checkpoint using selection_rules.json"
    )
    parser.add_argument("--dataset", required=True, choices=tuple(ws.DATASETS))
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=490)
    args = parser.parse_args()
    try:
        method = ws.canonical_method(args.method)
        epoch, path, checkpoint = ws.select_checkpoint(args.dataset, method, args.seed)
        summary = ws.checkpoint_summary(
            args.dataset, method, args.seed, epoch, checkpoint
        )
        metrics = ws.metric_fields(summary)
        high_mean_max = ws.maximum_high_mean(args.dataset, method, args.seed)
        rank = ws.selection_rank(
            args.dataset,
            method,
            args.seed,
            epoch,
            checkpoint,
            high_mean_max=high_mean_max,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("dataset method       " + " ".join(name.ljust(width) for name, width in COLUMNS).rstrip())
    print("-" * 145)
    prefix = f"{args.dataset:<7} {method:<12} "
    print(prefix + format_row({"epoch": epoch, "eligible": int(bool(rank[0])), **metrics}))
    print(f"checkpoint: {path.relative_to(ws.ROOT)}")
    window = float(ws.load_rules()["high"]["mean_window_from_max"])
    print(f"h_mean_max: {high_mean_max:.8f}; candidate: h_mean > {high_mean_max - window:.8f}")
    print(f"rules: {ws.RULES_PATH.relative_to(ws.ROOT)}")


if __name__ == "__main__":
    main()
