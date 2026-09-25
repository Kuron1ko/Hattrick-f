from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
ORIGINAL_DIR = TEST_DIR / "shared2x_sparse_path_cross_attention"
ORIGINAL_RUNNER = ORIGINAL_DIR / "run_experiment.py"
for item in (str(ROOT), str(TEST_DIR), str(ORIGINAL_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


original = load_module(
    "shared2x_sparse_path_cross_attention_optimized_original",
    ORIGINAL_RUNNER,
)
full = original.full
NativeHattrick = original.NativeHattrick
OriginalSparsePathCrossAttentionHattrick = (
    original.SparsePathCrossAttentionHattrick
)

ATTENTION_DIM = original.ATTENTION_DIM
NEIGHBOR_ODS = original.NEIGHBOR_ODS
ANCHOR_FEATURES = original.ANCHOR_FEATURES
ATTENTION_FEATURES = original.ATTENTION_FEATURES
TOTAL_FEATURES = original.TOTAL_FEATURES

# One GEANT OD worth of High paths.  At B=20 the largest selected K/V tensor is
# about 38 MiB instead of the original 303 MiB full-path gather.
HIGH_PATH_CHUNK = 462


class OptimizedSparsePathCrossAttentionHattrick(
    OriginalSparsePathCrossAttentionHattrick
):
    """R=8, d=16 cross-attention with equivalent chunked contractions.

    Parameters and persistent state are exactly those of the original class, so
    its checkpoints load with ``strict=True``.  Only non-persistent topology
    buffers and the contraction schedule differ.
    """

    def __init__(self, props):
        super().__init__(props)
        self.register_buffer(
            "_spco_capacity_features", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_spco_route_mask", torch.empty(0, dtype=torch.bool), persistent=False
        )
        self._spco_static_key = None

    def forward(self, *args, **kwargs):
        """Repair native static-test capacity broadcasting without changing values.

        The repository core expands a width-one capacity batch only on the first
        test forward.  On later cached-transformer forwards, batch>1 reaches a
        ``torch.gather`` with capacity [1,E] and indices [B,P].  Supplying the
        same zero-stride expanded view on every call is mathematically identical
        to the first-forward behavior and permits steady-state benchmarking.
        """

        if len(args) < 6:
            return super().forward(*args, **kwargs)
        forwarded = list(args)
        props = forwarded[0]
        capacities = forwarded[3]
        traffic_batch = int(forwarded[5].shape[0])
        if (
            not bool(getattr(props, "dynamic", False))
            and int(capacities.shape[0]) == 1
            and traffic_batch > 1
            and hasattr(self, "transformer_output")
        ):
            # GNN is still invoked before the native forward consults its
            # transformer cache, so node features and capacity must carry the
            # same harmless repeated batch into that call.
            forwarded[1] = forwarded[1].expand(traffic_batch, -1, -1)
            forwarded[3] = capacities.expand(traffic_batch, -1)
        return super().forward(*forwarded, **kwargs)

    @staticmethod
    def _mask_identity(path_masks):
        if path_masks is None:
            return None
        medium_mask = path_masks[1]
        return (
            int(medium_mask.data_ptr()),
            int(getattr(medium_mask, "_version", 0)),
            tuple(medium_mask.shape),
            str(medium_mask.device),
        )

    def _ensure_sparse_cache(
        self,
        paths_to_edges,
        capacities,
        path_masks,
        paths_per_od: int,
    ) -> None:
        """Use the original builder once, then trust this runner's static topology.

        The shared experiment fixes one topology, capacity vector, PTE and mask
        for the lifetime of a model.  Avoiding a CUDA->CPU capacity copy plus
        several ``.item()`` synchronizations on every forward is therefore
        mathematically neutral in this runner.
        """

        pte = paths_to_edges.coalesce()
        key = (
            int(pte.shape[0]),
            int(pte.shape[1]),
            int(pte._nnz()),
            str(pte.device),
            int(paths_per_od),
            self._mask_identity(path_masks),
        )
        cache_valid = (
            self._spco_static_key == key
            and self._spc_cache_shape is not None
            and tuple(self._spc_neighbor_path.shape)
            == (int(pte.shape[0]), NEIGHBOR_ODS, int(paths_per_od))
            and self._spc_neighbor_path.device == pte.device
            and tuple(self._spco_route_mask.shape)
            == (int(pte.shape[0]), NEIGHBOR_ODS, int(paths_per_od))
        )
        if cache_valid:
            return

        super()._ensure_sparse_cache(
            pte, capacities, path_masks, int(paths_per_od)
        )
        self._spco_route_mask = self._spc_medium_mask_signature[
            self._spc_neighbor_path
        ].reshape(int(pte.shape[0]), NEIGHBOR_ODS, int(paths_per_od))
        self._spco_capacity_features = torch.empty(
            0, device=pte.device, dtype=torch.float32
        )
        self._spco_static_key = key

    def _capacity_features(self, capacities, pte, pte_info):
        """Cache static summaries and always expand them to the traffic batch.

        Expanding fixes the original steady-state test bug: after the topology
        transformer is cached, native ``model_forward`` supplies capacity with
        batch width one even when cached path embeddings have B>1.
        """

        batch_size = int(capacities.shape[0])
        if int(self._spco_capacity_features.numel()) == 0:
            computed = super()._capacity_features(capacities, pte, pte_info)
            self._spco_capacity_features = computed[:1].detach()
        cached = self._spco_capacity_features
        if int(cached.shape[0]) != 1:
            raise RuntimeError("Static capacity cache must have batch width one")
        return cached.expand(batch_size, -1, -1)

    def compute_medium_pressure_features(
        self,
        tm2_pred,
        capacities,
        paths_to_edges,
        pte_info,
        batch_size,
        num_paths_per_pair,
        props,
        path_masks,
    ):
        pte = paths_to_edges.coalesce()
        total_paths = int(pte.shape[0])
        paths_per_od = int(num_paths_per_pair)
        num_ods = total_paths // paths_per_od
        self._ensure_sparse_cache(pte, capacities, path_masks, paths_per_od)
        path_embeddings = self._current_path_embeddings(batch_size)
        capacity_features = self._capacity_features(capacities, pte, pte_info)
        # Cached static test-mode capacity has width one, while path embeddings
        # retain the requested benchmark batch.  Expand without materializing.
        if int(capacity_features.shape[0]) == 1 and int(batch_size) > 1:
            capacity_features = capacity_features.expand(batch_size, -1, -1)
        if int(capacity_features.shape[0]) != int(batch_size):
            raise RuntimeError("Capacity-feature batch does not match traffic")
        model_dtype = path_embeddings.dtype

        query_inputs = torch.cat(
            (path_embeddings, capacity_features.to(dtype=model_dtype)), dim=-1
        )
        query = self.spc_query(query_inputs)
        safe_demand = tm2_pred.to(dtype=torch.float32).clamp_min(0.0)
        medium_inputs = torch.cat(
            (
                path_embeddings,
                torch.log1p(safe_demand).to(dtype=model_dtype),
                capacity_features.to(dtype=model_dtype),
            ),
            dim=-1,
        )
        medium_kv = self.spc_medium_kv(medium_inputs)

        # Route attention is independent across High paths.  Chunk only that
        # axis, select one small K/V block, and express both dot-product/sum
        # contractions as batched matmul.  This avoids the original two
        # [B,P,R,K,D] broadcast products and its full 303 MiB B=20 K/V gather.
        route_message_chunks = []
        route_attention_chunks = []
        scale = math.sqrt(float(ATTENTION_DIM))
        for start in range(0, total_paths, HIGH_PATH_CHUNK):
            end = min(total_paths, start + HIGH_PATH_CHUNK)
            chunk_paths = end - start
            neighbor_paths = self._spc_neighbor_path[start:end]
            gathered_kv = medium_kv[:, neighbor_paths.reshape(-1), :].reshape(
                batch_size,
                chunk_paths,
                NEIGHBOR_ODS,
                paths_per_od,
                ATTENTION_DIM,
            )
            query_chunk = query[:, start:end, :]
            route_scores = torch.matmul(
                gathered_kv,
                query_chunk.unsqueeze(2).unsqueeze(-1),
            ).squeeze(-1) / scale
            route_scores = route_scores + self.route_overlap_scale * (
                self._spc_neighbor_overlap[start:end]
                .reshape(1, chunk_paths, NEIGHBOR_ODS, paths_per_od)
                .to(dtype=route_scores.dtype)
            )
            route_scores = route_scores.masked_fill(
                ~self._spco_route_mask[start:end].reshape(
                    1, chunk_paths, NEIGHBOR_ODS, paths_per_od
                ),
                torch.finfo(route_scores.dtype).min,
            )
            route_attention_chunk = torch.softmax(route_scores, dim=-1)
            route_message_chunk = torch.matmul(
                route_attention_chunk.unsqueeze(-2), gathered_kv
            ).squeeze(-2)
            route_attention_chunks.append(route_attention_chunk)
            route_message_chunks.append(route_message_chunk)

        route_attention = torch.cat(route_attention_chunks, dim=1)
        route_message = torch.cat(route_message_chunks, dim=1)

        od_query = self.spc_od_query(query).unsqueeze(2)
        od_key = self.spc_od_key(route_message)
        od_value = self.spc_od_value(route_message)
        od_scores = torch.matmul(
            od_query, od_key.transpose(-2, -1)
        ).squeeze(2) / scale
        od_scores = od_scores + self.od_overlap_scale * (
            self._spc_neighbor_od_overlap.reshape(
                1, total_paths, NEIGHBOR_ODS
            ).to(dtype=od_scores.dtype)
        )
        demand_by_od = safe_demand.reshape(
            batch_size, num_ods, paths_per_od, 1
        ).mean(dim=2).squeeze(-1)
        neighbor_demand = demand_by_od[:, self._spc_neighbor_od]
        od_scores = od_scores + self.od_demand_scale * torch.log1p(
            neighbor_demand
        ).to(dtype=od_scores.dtype)
        od_attention = torch.softmax(od_scores, dim=-1)
        context = torch.matmul(
            od_attention.unsqueeze(-2), od_value
        ).squeeze(-2)
        attention_features = self.spc_output(torch.cat((query, context), dim=-1))

        anchor = NativeHattrick.compute_medium_pressure_features(
            self,
            tm2_pred,
            capacities,
            pte,
            pte_info,
            batch_size,
            paths_per_od,
            props,
            path_masks,
        )
        features = torch.cat(
            (anchor, attention_features.to(dtype=anchor.dtype)), dim=-1
        )
        if tuple(features.shape) != (batch_size, total_paths, TOTAL_FEATURES):
            raise RuntimeError(
                f"Unexpected optimized sparse-attention shape {tuple(features.shape)}"
            )
        if not torch.isfinite(features).all().item():
            raise RuntimeError("Optimized sparse cross-attention is non-finite")

        self._last_route_attention = route_attention.detach()
        self._last_od_attention = od_attention.detach()
        self._last_spc_features = features.detach()
        self._last_spc_audit = {
            "attention_dim": ATTENTION_DIM,
            "neighbor_ods": NEIGHBOR_ODS,
            "paths_per_medium_od": paths_per_od,
            "high_path_chunk": HIGH_PATH_CHUNK,
            "pairs_per_high_path": NEIGHBOR_ODS * paths_per_od,
            "high_medium_path_pairs_per_snapshot": (
                total_paths * NEIGHBOR_ODS * paths_per_od
            ),
            "feature_count": TOTAL_FEATURES,
            "contractions": "chunked selected K/V plus batched matmul",
            "formal_forward": (
                "original cascade via a capacity-broadcast-only wrapper"
            ),
            "strict_policy_inputs": [
                "tm2_pred",
                "path_embeddings",
                "paths_to_edges",
                "capacities",
                "path_masks",
            ],
        }
        return features


full.Hattrick = OptimizedSparsePathCrossAttentionHattrick
full.OUTPUT_ROOT = THIS_DIR / "artifacts"


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "original_sparse_path_cross_attention/run_experiment.py": ORIGINAL_RUNNER,
        "ordered_projection.py": original.FULL_DIR / "ordered_projection.py",
        "shared2x_full_objectives/run_experiment.py": (
            original.FULL_DIR / "run_experiment.py"
        ),
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
    }
    hashes = {name: full.sha256(path) for name, path in paths.items()}
    settings = json.dumps(
        {
            "attention_dim": ATTENTION_DIM,
            "neighbor_ods": NEIGHBOR_ODS,
            "high_path_chunk": HIGH_PATH_CHUNK,
            "math": "same route/OD softmax; broadcast products replaced by matmul",
            "checkpoint_compatible": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["sparse_path_cross_attention_optimized/settings"] = hashlib.sha256(
        settings
    ).hexdigest()
    return hashes


full.source_hashes = source_hashes


def benchmark_model_class():
    return OptimizedSparsePathCrossAttentionHattrick


def method_description() -> dict[str, object]:
    return {
        "method": "Optimized Hattrick sparse path cross-attention",
        "mathematics": "unchanged R=8, K=8 and d=16 two-level attention",
        "optimization": (
            "Chunk High paths by 462, materialize only a small selected K/V "
            "block, and replace four broadcast multiply/reduce pairs by batched "
            "matrix multiplications. Cache static capacity summaries/masks."
        ),
        "checkpoint_compatibility": (
            "No new persistent buffers or parameters; original R8/d16 state_dict "
            "loads with strict=True."
        ),
        "formal_cascade": "unchanged High -> High+Medium -> High+Medium+Low",
        "projection": "unchanged six-objective ordered gradient projection",
        "strict_inference": "ESM and static topology/capacity inputs only",
        "source_sha256": source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimized shared-2x sparse High/Medium cross-attention"
    )
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = full.run_one(args.level, args.seed, force=args.force)
    description = method_description()
    description.update(
        {"level": args.level, "seed": args.seed, "run_directory": str(run_dir)}
    )
    full.write_json(
        run_dir / "sparse_path_cross_attention_optimized_method.json",
        description,
    )
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
