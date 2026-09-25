from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
spec = importlib.util.spec_from_file_location(
    "shared2x_sparse_path_cross_attention_optimized_equivalence_runtime",
    THIS_DIR / "run_experiment.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load optimized sparse-attention runner")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


DEFAULT_CHECKPOINT = (
    ROOT
    / "test_diff_path/shared2x_sparse_path_cross_attention/artifacts"
    / "level2_proxy/seed_490/final_model.pt"
)


def state_dict_from_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model_state_dict"]


def maximum_delta(left, right) -> float:
    return max(float((a - b).abs().max().item()) for a, b in zip(left, right))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU checkpoint/output equivalence for optimized R8/d16 attention"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 16, 20))
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device("cpu")
    checkpoint = state_dict_from_checkpoint(args.checkpoint.resolve())
    rows = []
    for batch_size in args.batch_sizes:
        props = experiment.full.shared.build_props(2, device)
        props.mode = "test"
        props.checkpoint = 0
        props.research_return_policy = True
        props.research_return_admitted = False
        dataset = experiment.full.DM_Dataset_within_Cluster(
            props, 0, 400, 400 + int(batch_size)
        )
        masks = experiment.full.shared.base.move_dataset_static(dataset, device)
        values = experiment.full.shared.unpack_to_device(
            next(
                iter(
                    experiment.full.shared.data_loader(
                        dataset, int(batch_size), False, 490
                    )
                )
            ),
            props,
        )
        reference = experiment.OriginalSparsePathCrossAttentionHattrick(props).eval()
        optimized = experiment.OptimizedSparsePathCrossAttentionHattrick(props).eval()
        reference.load_state_dict(checkpoint, strict=True)
        optimized.load_state_dict(checkpoint, strict=True)
        if list(reference.state_dict()) != list(optimized.state_dict()):
            raise RuntimeError("Persistent state_dict keys differ")

        with torch.no_grad():
            started = time.perf_counter()
            reference_policy, _ = experiment.full.shared.model_forward(
                reference, props, dataset, values, masks
            )
            reference_seconds = time.perf_counter() - started
            started = time.perf_counter()
            optimized_policy, _ = experiment.full.shared.model_forward(
                optimized, props, dataset, values, masks
            )
            optimized_first_seconds = time.perf_counter() - started
            started = time.perf_counter()
            optimized_repeat_policy, _ = experiment.full.shared.model_forward(
                optimized, props, dataset, values, masks
            )
            optimized_repeat_seconds = time.perf_counter() - started

        row = {
            "batch_size": int(batch_size),
            "state_dict_strict": True,
            "reference_first_seconds": reference_seconds,
            "optimized_first_seconds": optimized_first_seconds,
            "optimized_repeat_seconds": optimized_repeat_seconds,
            "feature_max_delta": float(
                (
                    reference._last_spc_features
                    - optimized._last_spc_features
                ).abs().max().item()
            ),
            "route_attention_max_delta": float(
                (
                    reference._last_route_attention
                    - optimized._last_route_attention
                ).abs().max().item()
            ),
            "od_attention_max_delta": float(
                (
                    reference._last_od_attention
                    - optimized._last_od_attention
                ).abs().max().item()
            ),
            "policy_max_delta": maximum_delta(
                reference_policy, optimized_policy
            ),
            "optimized_repeat_policy_max_delta": maximum_delta(
                optimized_policy, optimized_repeat_policy
            ),
            "optimized_cache_builds": optimized.sparse_cache_builds,
        }
        if row["feature_max_delta"] > 5e-6:
            raise RuntimeError(f"Feature equivalence failed: {row}")
        if row["policy_max_delta"] > 1e-6:
            raise RuntimeError(f"Policy equivalence failed: {row}")
        if row["optimized_repeat_policy_max_delta"] != 0.0:
            raise RuntimeError(f"Steady-state replay is not exact: {row}")
        if row["optimized_cache_builds"] != 1:
            raise RuntimeError(f"Static cache rebuilt: {row}")
        rows.append(row)
        del reference, optimized, dataset, values
        gc.collect()

    print(
        json.dumps(
            {
                "status": "ok",
                "device": "cpu",
                "checkpoint": str(args.checkpoint.resolve()),
                "mathematics": "unchanged R=8, K=8, d=16",
                "rows": rows,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
