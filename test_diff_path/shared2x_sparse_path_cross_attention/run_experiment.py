from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
FULL_DIR = TEST_DIR / "shared2x_full_objectives"
for item in (str(ROOT), str(TEST_DIR), str(FULL_DIR)):
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


full = load_module(
    "shared2x_sparse_path_cross_attention_full_runtime",
    FULL_DIR / "run_experiment.py",
)
NativeHattrick = full.Hattrick

ATTENTION_DIM = 16
NEIGHBOR_ODS = 8
ANCHOR_FEATURES = 2
ATTENTION_FEATURES = ATTENTION_DIM
TOTAL_FEATURES = ANCHOR_FEATURES + ATTENTION_FEATURES


class ZeroOutputAdapter(nn.Sequential):
    """Hattrick-style residual whose final affine map starts at zero."""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.kaiming_normal_(self[0].weight, a=0.02, nonlinearity="leaky_relu")
        nn.init.zeros_(self[0].bias)
        nn.init.zeros_(self[2].weight)
        nn.init.zeros_(self[2].bias)


class SparsePathCrossAttentionHattrick(NativeHattrick):
    """Serial Hattrick with sparse High-path to Medium-path attention.

    Every High candidate path keeps its own OD and attends to eight Medium ODs:
    its own OD plus seven static inverse-capacity-overlap neighbors. Attention
    first resolves the eight concrete paths inside each Medium OD, then resolves
    the eight OD messages. The native uniform Medium-pressure feature remains
    as a separate two-scalar anchor. Formal Medium/Low stages are inherited.
    """

    def __init__(self, props):
        props.future_lookahead = False
        super().__init__(props)
        path_dim = int(self.input_dim)
        # Query: path embedding + max/mean inverse path capacity.
        self.spc_query = nn.Linear(path_dim + 2, ATTENTION_DIM)
        # Medium K/V token: path embedding + ESM demand + capacity summaries.
        # A shared K/V projection halves gathered activation memory.
        self.spc_medium_kv = nn.Linear(path_dim + 3, ATTENTION_DIM)
        self.spc_od_query = nn.Linear(ATTENTION_DIM, ATTENTION_DIM, bias=False)
        self.spc_od_key = nn.Linear(ATTENTION_DIM, ATTENTION_DIM, bias=False)
        self.spc_od_value = nn.Linear(ATTENTION_DIM, ATTENTION_DIM, bias=False)
        self.spc_output = nn.Sequential(
            nn.Linear(ATTENTION_DIM * 2, ATTENTION_DIM),
            nn.LeakyReLU(negative_slope=0.02),
            nn.LayerNorm(ATTENTION_DIM),
        )
        # Length-one tensors are scalar-equivalent under broadcasting and are
        # compatible with Hattrick's ordered-gradient slicing utility, which
        # cannot reconstruct a zero-dimensional parameter shape.
        self.route_overlap_scale = nn.Parameter(torch.tensor([1.0]))
        self.od_overlap_scale = nn.Parameter(torch.tensor([1.0]))
        self.od_demand_scale = nn.Parameter(torch.tensor([0.25]))

        for module in (
            self.spc_query,
            self.spc_medium_kv,
            self.spc_od_query,
            self.spc_od_key,
            self.spc_od_value,
            self.spc_output[0],
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        self.medium_pressure_init_adapter = ZeroOutputAdapter(
            self.mlp_11_dim + TOTAL_FEATURES
        )
        self.medium_pressure_rau_adapter = ZeroOutputAdapter(
            self.mlp_12_dim + TOTAL_FEATURES
        )
        self.medium_pressure_feature_count = TOTAL_FEATURES
        self.research_medium_pressure_lookahead_enabled = True

        # Lazy non-persistent topology cache.
        self.register_buffer("_spc_incidence", torch.empty(0), persistent=False)
        self.register_buffer("_spc_path_lengths", torch.empty(0), persistent=False)
        self.register_buffer(
            "_spc_neighbor_od", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_spc_neighbor_path", torch.empty(0, dtype=torch.long), persistent=False
        )
        self.register_buffer("_spc_neighbor_overlap", torch.empty(0), persistent=False)
        self.register_buffer("_spc_neighbor_od_overlap", torch.empty(0), persistent=False)
        self.register_buffer("_spc_capacity_signature", torch.empty(0), persistent=False)
        self.register_buffer(
            "_spc_medium_mask_signature",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self._spc_cache_shape: tuple[int, int, int] | None = None
        self._spc_cache_builds = 0
        self._spc_transformer_cache = None
        self._last_route_attention = None
        self._last_od_attention = None
        self._last_spc_features = None
        self._last_spc_audit: dict[str, object] = {}

    @property
    def sparse_cache_builds(self) -> int:
        return int(self._spc_cache_builds)

    @property
    def last_spc_audit(self) -> dict[str, object]:
        return dict(self._last_spc_audit)

    def compute_transformer_output(self, *args, **kwargs):
        output = super().compute_transformer_output(*args, **kwargs)
        self._spc_transformer_cache = output
        return output

    def _current_path_embeddings(self, batch_size: int) -> torch.Tensor:
        output = self._spc_transformer_cache
        if output is None and hasattr(self, "transformer_output"):
            output = self.transformer_output
        if output is None:
            raise RuntimeError("Sparse cross-attention has no path embeddings")
        if int(output.shape[0]) == 1 and int(batch_size) > 1:
            output = output.expand(batch_size, -1, -1, -1)
        if int(output.shape[0]) != int(batch_size):
            raise RuntimeError("Path-embedding batch mismatch")
        path_embeddings = output[:, :, 0, :]
        self._spc_transformer_cache = None
        return path_embeddings

    def _class_mask(self, path_masks, class_index: int, total_paths: int, device):
        mask = self.path_mask_for_class(path_masks, class_index)
        if mask is None:
            return torch.ones(total_paths, dtype=torch.bool, device=device)
        mask = mask.reshape(-1).to(device=device, dtype=torch.bool)
        if int(mask.numel()) != total_paths:
            raise RuntimeError("Path mask and PTE have different widths")
        return mask

    @staticmethod
    def _capacity_reference(capacities, num_edges: int) -> torch.Tensor:
        values = capacities.detach().to(device="cpu", dtype=torch.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if tuple(values.shape[1:]) != (num_edges,):
            raise RuntimeError("Unexpected capacity shape")
        reference = values[0]
        if int(values.shape[0]) > 1 and not torch.equal(
            values, reference.reshape(1, -1).expand_as(values)
        ):
            raise RuntimeError("Static neighbor cache requires one topology per batch")
        if not torch.isfinite(reference).all().item() or bool((reference <= 0).any().item()):
            raise RuntimeError("Sparse attention requires finite positive capacities")
        return reference

    def _ensure_sparse_cache(
        self,
        paths_to_edges,
        capacities,
        path_masks,
        paths_per_od: int,
    ) -> None:
        pte = paths_to_edges.coalesce()
        total_paths, num_edges = int(pte.shape[0]), int(pte.shape[1])
        if total_paths % int(paths_per_od) != 0:
            raise RuntimeError("Path count is not divisible by paths per OD")
        num_ods = total_paths // int(paths_per_od)
        if num_ods < NEIGHBOR_ODS:
            raise RuntimeError("Not enough Medium ODs for requested neighborhood")
        capacity_reference = self._capacity_reference(capacities, num_edges)
        medium_mask = self._class_mask(
            path_masks, 1, total_paths, pte.device
        ).detach().to(device="cpu")
        shape = (total_paths, num_ods, int(paths_per_od))
        cache_valid = (
            self._spc_cache_shape == shape
            and tuple(self._spc_neighbor_od.shape) == (total_paths, NEIGHBOR_ODS)
            and self._spc_neighbor_od.device == pte.device
            and torch.equal(self._spc_capacity_signature.cpu(), capacity_reference)
            and torch.equal(self._spc_medium_mask_signature.cpu(), medium_mask)
        )
        if cache_valid:
            return

        indices = pte.indices().detach().to(device="cpu")
        incidence = torch.zeros((total_paths, num_edges), dtype=torch.float32)
        incidence[indices[0], indices[1]] = 1.0
        path_lengths = incidence.sum(dim=-1).clamp_min(1.0)
        inverse_capacity = capacity_reference.reciprocal()
        weighted = incidence * inverse_capacity.sqrt().reshape(1, num_edges)
        # [P,P], used only while building the static top-OD neighborhood.
        overlap = weighted @ weighted.t()
        medium_footprint = (incidence * inverse_capacity.reshape(1, num_edges)).sum(
            dim=-1
        ).clamp_min(1e-12)
        overlap.div_(medium_footprint.reshape(1, total_paths)).clamp_(0.0, 1.0)
        overlap_by_od = overlap.reshape(total_paths, num_ods, paths_per_od)
        medium_mask_by_od = medium_mask.reshape(num_ods, paths_per_od)
        masked_overlap = overlap_by_od.masked_fill(
            ~medium_mask_by_od.reshape(1, num_ods, paths_per_od),
            -torch.inf,
        )
        od_overlap = masked_overlap.amax(dim=-1)

        own_od = torch.arange(total_paths, dtype=torch.long) // int(paths_per_od)
        rows = torch.arange(total_paths, dtype=torch.long)
        other_scores = od_overlap.clone()
        other_scores[rows, own_od] = -torch.inf
        other_od = torch.topk(
            other_scores, k=NEIGHBOR_ODS - 1, dim=-1, sorted=True
        ).indices
        neighbor_od = torch.cat((own_od.unsqueeze(-1), other_od), dim=-1)
        path_offsets = torch.arange(paths_per_od, dtype=torch.long)
        neighbor_path = (
            neighbor_od.unsqueeze(-1) * int(paths_per_od)
            + path_offsets.reshape(1, 1, paths_per_od)
        )
        gathered_overlap = overlap.gather(1, neighbor_path.reshape(total_paths, -1)).reshape(
            total_paths, NEIGHBOR_ODS, paths_per_od
        )
        gathered_od_overlap = od_overlap.gather(1, neighbor_od)

        self._spc_incidence = incidence.to(device=pte.device)
        self._spc_path_lengths = path_lengths.to(device=pte.device)
        self._spc_neighbor_od = neighbor_od.to(device=pte.device)
        self._spc_neighbor_path = neighbor_path.to(device=pte.device)
        self._spc_neighbor_overlap = gathered_overlap.to(device=pte.device)
        self._spc_neighbor_od_overlap = gathered_od_overlap.to(device=pte.device)
        self._spc_capacity_signature = capacity_reference.to(device=pte.device)
        self._spc_medium_mask_signature = medium_mask.to(device=pte.device)
        self._spc_cache_shape = shape
        self._spc_cache_builds += 1

    @staticmethod
    def _path_max(
        edge_values,
        row_indices,
        col_indices,
        pte_values,
        total_paths: int,
    ):
        selected = edge_values[:, col_indices] * pte_values.reshape(1, -1)
        output = torch.zeros(
            (edge_values.shape[0], total_paths),
            dtype=edge_values.dtype,
            device=edge_values.device,
        )
        output.scatter_reduce_(
            1,
            row_indices.reshape(1, -1).expand(edge_values.shape[0], -1),
            selected,
            reduce="amax",
            include_self=True,
        )
        return output

    def _capacity_features(self, capacities, pte, pte_info):
        _, row_indices, col_indices, pte_values = pte_info
        pte_float = pte.to(dtype=torch.float32)
        inverse_capacity = capacities.to(dtype=torch.float32).clamp_min(1e-6).reciprocal()
        path_sum = torch.sparse.mm(pte_float, inverse_capacity.t()).t()
        path_mean = path_sum / self._spc_path_lengths.reshape(1, -1)
        path_max = self._path_max(
            inverse_capacity,
            row_indices,
            col_indices,
            pte_values.to(dtype=torch.float32),
            int(pte.shape[0]),
        )
        return torch.stack((torch.log1p(path_max), torch.log1p(path_mean)), dim=-1)

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
        total_paths, num_edges = int(pte.shape[0]), int(pte.shape[1])
        paths_per_od = int(num_paths_per_pair)
        num_ods = total_paths // paths_per_od
        self._ensure_sparse_cache(pte, capacities, path_masks, paths_per_od)
        path_embeddings = self._current_path_embeddings(batch_size)
        capacity_features = self._capacity_features(capacities, pte, pte_info)
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

        # Static gather: [B,P,R,K,D], R=8 ODs and K=8 concrete paths/OD.
        neighbor_flat = self._spc_neighbor_path.reshape(-1)
        gathered_kv = medium_kv[:, neighbor_flat, :].reshape(
            batch_size,
            total_paths,
            NEIGHBOR_ODS,
            paths_per_od,
            ATTENTION_DIM,
        )
        route_scores = (
            query.unsqueeze(2).unsqueeze(3) * gathered_kv
        ).sum(dim=-1) / math.sqrt(float(ATTENTION_DIM))
        route_scores = route_scores + self.route_overlap_scale * self._spc_neighbor_overlap.reshape(
            1, total_paths, NEIGHBOR_ODS, paths_per_od
        ).to(dtype=route_scores.dtype)
        medium_mask = self._spc_medium_mask_signature[
            self._spc_neighbor_path
        ].reshape(1, total_paths, NEIGHBOR_ODS, paths_per_od)
        route_scores = route_scores.masked_fill(
            ~medium_mask, torch.finfo(route_scores.dtype).min
        )
        route_attention = torch.softmax(route_scores, dim=-1)
        route_message = (
            route_attention.unsqueeze(-1) * gathered_kv
        ).sum(dim=3)

        od_query = self.spc_od_query(query).unsqueeze(2)
        od_key = self.spc_od_key(route_message)
        od_value = self.spc_od_value(route_message)
        od_scores = (od_query * od_key).sum(dim=-1) / math.sqrt(float(ATTENTION_DIM))
        od_scores = od_scores + self.od_overlap_scale * self._spc_neighbor_od_overlap.reshape(
            1, total_paths, NEIGHBOR_ODS
        ).to(dtype=od_scores.dtype)
        demand_by_od = safe_demand.reshape(
            batch_size, num_ods, paths_per_od, 1
        ).mean(dim=2).squeeze(-1)
        neighbor_demand = demand_by_od[:, self._spc_neighbor_od]
        od_scores = od_scores + self.od_demand_scale * torch.log1p(
            neighbor_demand
        ).to(dtype=od_scores.dtype)
        od_attention = torch.softmax(od_scores, dim=-1)
        context = (od_attention.unsqueeze(-1) * od_value).sum(dim=2)
        attention_features = self.spc_output(torch.cat((query, context), dim=-1))

        # Independent cheap global anchor guards against a poor early attention
        # neighborhood without erasing Medium OD identity from the main branch.
        anchor = super().compute_medium_pressure_features(
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
            raise RuntimeError(f"Unexpected sparse-attention shape {tuple(features.shape)}")
        if not torch.isfinite(features).all().item():
            raise RuntimeError("Sparse cross-attention produced non-finite features")

        self._last_route_attention = route_attention.detach()
        self._last_od_attention = od_attention.detach()
        self._last_spc_features = features.detach()
        self._last_spc_audit = {
            "attention_dim": ATTENTION_DIM,
            "neighbor_ods": NEIGHBOR_ODS,
            "paths_per_medium_od": paths_per_od,
            "pairs_per_high_path": NEIGHBOR_ODS * paths_per_od,
            "high_medium_path_pairs_per_snapshot": (
                total_paths * NEIGHBOR_ODS * paths_per_od
            ),
            "feature_count": TOTAL_FEATURES,
            "formal_forward_inherited": type(self).forward is NativeHattrick.forward,
            "strict_policy_inputs": [
                "tm2_pred",
                "path_embeddings",
                "paths_to_edges",
                "capacities",
                "path_masks",
            ],
        }
        return features


full.Hattrick = SparsePathCrossAttentionHattrick
full.OUTPUT_ROOT = THIS_DIR / "artifacts"


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "ordered_projection.py": FULL_DIR / "ordered_projection.py",
        "shared2x_full_objectives/run_experiment.py": FULL_DIR / "run_experiment.py",
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "utils/training_utils.py": ROOT / "utils" / "training_utils.py",
    }
    hashes = {name: full.sha256(path) for name, path in paths.items()}
    settings = json.dumps(
        {
            "attention_dim": ATTENTION_DIM,
            "neighbor_ods": NEIGHBOR_ODS,
            "neighbor_rule": "own OD plus top-7 max inverse-capacity path overlap",
            "attention": "one route-softmax then one OD-softmax, one head",
            "anchor_features": ANCHOR_FEATURES,
            "zero_output_adapters": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["sparse_path_cross_attention/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


full.source_hashes = source_hashes


def method_description() -> dict:
    return {
        "method": "Hattrick sparse High-path/Medium-path cross-attention",
        "architecture": (
            "Each High path attends first to the eight concrete paths of eight "
            "Medium ODs (own OD plus seven structural neighbors), then across "
            "those OD messages. The 16-d context and two uniform-pressure anchors "
            "feed High init and every High RAU through zero-output adapters."
        ),
        "tensor_contract": {
            "high_query": "[B,3696,16]",
            "medium_tokens": "[B,462,8,16]",
            "neighbor_od": "[3696,8] static",
            "gathered_pairs": "[B,3696,8,8,16]",
            "output_features": "[B,3696,18]",
        },
        "formal_cascade": "Inherited native High -> High+Medium -> High+Medium+Low",
        "formal_medium_stage": "Unchanged and recomputed after High",
        "projection": "unchanged six-objective ordered gradient projection",
        "objectives": list(full.OBJECTIVE_NAMES),
        "inference_information": (
            "ESM tm2_pred, topology/capacity path embeddings, PTE, capacities, "
            "and feasibility masks only; actual traffic is never a policy input"
        ),
        "zero_initialization": (
            "Both High residual output affine maps have exactly zero weight/bias, "
            "preserving the matched native epoch-0 policy bitwise."
        ),
        "source_sha256": source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared-2x sparse High/Medium path cross-attention"
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
    full.write_json(run_dir / "sparse_path_cross_attention_method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
