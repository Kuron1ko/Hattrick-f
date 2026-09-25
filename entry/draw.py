from __future__ import annotations

import argparse
import re
from pathlib import Path

import _workspace as ws
from _plotting import load_metrics, plot
from infer import infer_one


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Infer when needed and draw two models on one dataset"
    )
    parser.add_argument("--dataset", required=True, choices=tuple(ws.DATASETS))
    parser.add_argument("--method1", required=True)
    parser.add_argument("--epoch1", type=int, required=True)
    parser.add_argument("--method2", required=True)
    parser.add_argument("--epoch2", type=int, required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--force-inference", action="store_true")
    args = parser.parse_args()
    try:
        methods = (ws.canonical_method(args.method1), ws.canonical_method(args.method2))
        epochs = (args.epoch1, args.epoch2)
        results = []
        for method, epoch in zip(methods, epochs):
            infer_args = argparse.Namespace(
                dataset=args.dataset,
                method=method,
                epoch=epoch,
                seed=args.seed,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                force=args.force_inference,
            )
            results.append(infer_one(infer_args))
        series = [
            {
                "label": method,
                "values": load_metrics(Path(result["metrics"]), args.dataset),
            }
            for method, result in zip(methods, results)
        ]
        filename = (
            f"{safe_name(methods[0])}-{epochs[0]}-"
            f"{safe_name(methods[1])}-{epochs[1]}.png"
        )
        output = ws.PICTURE_ROOT / args.dataset / filename
        plot(series, output, args.dataset)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"picture finished at {output.resolve()}")


if __name__ == "__main__":
    main()
