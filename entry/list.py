from __future__ import annotations

import argparse

import _workspace as ws


COLUMNS = (
    ("epoch", 7),
    ("eligible", 10),
    ("h_mean", 10),
    ("h_p1", 10),
    ("h_p10", 10),
    ("m_mean", 10),
    ("m_p1", 10),
    ("m_p10", 10),
    ("l_mean", 10),
    ("l_p1", 10),
    ("l_p10", 10),
)


def format_row(row: dict) -> str:
    values = []
    for name, width in COLUMNS:
        value = row[name]
        text = str(value) if name in {"epoch", "eligible"} else f"{float(value):.5f}"
        values.append(text.ljust(width))
    return " ".join(values).rstrip()


def rows_for(dataset: str, method: str, seed: int) -> list[dict]:
    checkpoints = ws.saved_checkpoints(dataset, method, seed)
    high_mean_max = ws.maximum_high_mean(dataset, method, seed, checkpoints)
    result = []
    for epoch, path in checkpoints.items():
        checkpoint = ws.load_checkpoint(path)
        try:
            summary = ws.checkpoint_summary(dataset, method, seed, epoch, checkpoint)
            metrics = ws.metric_fields(summary)
            rank = ws.selection_rank(
                dataset,
                method,
                seed,
                epoch,
                checkpoint,
                high_mean_max=high_mean_max,
            )
        except (KeyError, RuntimeError, ValueError):
            continue
        result.append({"epoch": epoch, "eligible": int(bool(rank[0])), **metrics})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List saved epochs and validation metrics from model/"
    )
    parser.add_argument("--dataset", required=True, choices=tuple(ws.DATASETS))
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, default=490)
    args = parser.parse_args()
    try:
        method = ws.canonical_method(args.method)
        rows = rows_for(args.dataset, method, args.seed)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not rows:
        raise SystemExit(
            f"No saved checkpoints with validation metrics for {args.dataset} {method}"
        )
    print(" ".join(name.ljust(width) for name, width in COLUMNS).rstrip())
    print("-" * (sum(width for _name, width in COLUMNS) + len(COLUMNS) - 1))
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    main()
