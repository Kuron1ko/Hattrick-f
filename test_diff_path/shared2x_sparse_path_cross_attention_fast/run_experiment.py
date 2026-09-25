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
    "shared2x_sparse_path_cross_attention_fast_full_runtime",
    FULL_DIR / "run_experiment.py",
)
NativeHattrick = full.Hattrick

ATTENTION_DIM = 8
NEIGHBOR_ODS = 4
ANCHOR_FEATURES = 2
ATTENTION_FEATURES = ATTENTION_DIM
TOTAL_FEATURES = ANCHOR_FEATURES + ATTENTION_FEATURES
ACTIVE_R8_WARMSTART: Path | None = None


class ZeroOutputAdapter(nn.Sequential):
    """Hattrick-style residual whose output affine map starts at zero."""

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


def _checkpoint_state(checkpoint_or_path) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(checkpoint_or_path, (str, Path)):
        path = Path(checkpoint_or_path).resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        source = str(path)
    else:
        checkpoint = checkpoint_or_path
        source = "in-memory"
    if not isinstance(checkpoint, dict):
        raise RuntimeError("R8d16 warm-start checkpoint is not a mapping")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise RuntimeError("R8d16 warm-start has no model state mapping")
    return state, source


def load_r8d16_warmstart(model: nn.Module, checkpoint_or_path) -> dict[str, object]:
    """Slice an R=8,d=16 checkpoint into R=4,d=8.

    This is a lossy initialization, not an equivalence transform.  Native
    same-shaped parameters are copied exactly.  Attention channels 0:8 are
    retained; the compressed output projection keeps query channels 0:8 and
    context channels 16:24.  The High adapters retain their shared prefix and
    the first eight learned attention-feature columns.
    """

    old, source = _checkpoint_state(checkpoint_or_path)
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    exact: list[str] = []
    sliced: list[str] = []
    skipped: list[str] = []

    special_prefix_slice = {
        "spc_query.weight",
        "spc_query.bias",
        "spc_medium_kv.weight",
        "spc_medium_kv.bias",
        "spc_od_query.weight",
        "spc_od_key.weight",
        "spc_od_value.weight",
        "spc_output.0.bias",
        "spc_output.2.weight",
        "spc_output.2.bias",
        "medium_pressure_init_adapter.0.weight",
        "medium_pressure_rau_adapter.0.weight",
    }
    for name, wanted in target.items():
        value = old.get(name)
        if not isinstance(value, torch.Tensor):
            skipped.append(name)
            continue
        if name == "spc_output.0.weight":
            columns = torch.cat((value[:, :8], value[:, 16:24]), dim=1)
            candidate = columns[:8, :16]
        elif name in special_prefix_slice:
            slices = tuple(slice(0, size) for size in wanted.shape)
            candidate = value[slices]
        elif tuple(value.shape) == tuple(wanted.shape):
            candidate = value
        else:
            skipped.append(name)
            continue
        if tuple(candidate.shape) != tuple(wanted.shape):
            skipped.append(name)
            continue
        mapped[name] = candidate.to(dtype=wanted.dtype)
        if tuple(value.shape) == tuple(wanted.shape):
            exact.append(name)
        else:
            sliced.append(name)

    loaded = model.load_state_dict(mapped, strict=False)
    handled_old = set(mapped)
    return {
        "source": source,
        "mode": "lossy R8d16 -> R4d8 channel/feature slicing",
        "exact_keys": exact,
        "sliced_keys": sliced,
        "skipped_target_keys": skipped,
        "unused_source_keys": sorted(set(old) - handled_old),
        "load_missing_keys": list(loaded.missing_keys),
        "load_unexpected_keys": list(loaded.unexpected_keys),
    }


class SparsePathCrossAttentionFastHattrick(NativeHattrick):
    """R4,d8 serial Hattrick with streamed High/Medium path attention."""

    def __init__(self, props):
        props.future_lookahead = False
        super().__init__(props)
        path_dim = int(self.input_dim)
        self.spc_query = nn.Linear(path_dim + 2, ATTENTION_DIM)
        self.spc_medium_kv = nn.Linear(path_dim + 3, ATTENTION_DIM)
        self.spc_od_query = nn.Linear(ATTENTION_DIM, ATTENTION_DIM, bias=False)
        self.spc_od_key = nn.Linear(ATTENTION_DIM, ATTENTION_DIM, bias=False)
        self.spc_od_value = nn.Linear(ATTENTION_DIM, ATTENTION_DIM, bias=False)
        self.spc_output = nn.Sequential(
            nn.Linear(ATTENTION_DIM * 2, ATTENTION_DIM),
            nn.LeakyReLU(negative_slope=0.02),
            nn.LayerNorm(ATTENTION_DIM),
        )
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
        self.research_spc_capture_attention = False
        self.r8_warmstart_manifest = None
        if ACTIVE_R8_WARMSTART is not None:
            self.r8_warmstart_manifest = load_r8d16_warmstart(
                self, ACTIVE_R8_WARMSTART
            )

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
        self, paths_to_edges, capacities, path_masks, paths_per_od: int
    ) -> None:
        pte = paths_to_edges.coalesce()
        total_paths, num_edges = int(pte.shape[0]), int(pte.shape[1])
        if total_paths % int(paths_per_od) != 0:
            raise RuntimeError("Path count is not divisible by paths per OD")
        num_ods = total_paths // int(paths_per_od)
        if num_ods < NEIGHBOR_ODS:
            raise RuntimeError("Not enough Medium ODs for requested neighborhood")
        capacity_reference = self._capacity_reference(capacities, num_edges)
        medium_mask = self._class_mask(path_masks, 1, total_paths, pte.device).detach().cpu()
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

        indices = pte.indices().detach().cpu()
        incidence = torch.zeros((total_paths, num_edges), dtype=torch.float32)
        incidence[indices[0], indices[1]] = 1.0
        path_lengths = incidence.sum(dim=-1).clamp_min(1.0)
        inverse_capacity = capacity_reference.reciprocal()
        weighted = incidence * inverse_capacity.sqrt().reshape(1, num_edges)
        # P x P exists only during this one-time CPU cache construction.
        overlap = weighted @ weighted.t()
        medium_footprint = (incidence * inverse_capacity.reshape(1, num_edges)).sum(
            dim=-1
        ).clamp_min(1e-12)
        overlap.div_(medium_footprint.reshape(1, total_paths)).clamp_(0.0, 1.0)
        overlap_by_od = overlap.reshape(total_paths, num_ods, paths_per_od)
        medium_mask_by_od = medium_mask.reshape(num_ods, paths_per_od)
        masked_overlap = overlap_by_od.masked_fill(
            ~medium_mask_by_od.reshape(1, num_ods, paths_per_od), -torch.inf
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
        offsets = torch.arange(paths_per_od, dtype=torch.long)
        neighbor_path = (
            neighbor_od.unsqueeze(-1) * int(paths_per_od)
            + offsets.reshape(1, 1, paths_per_od)
        )
        gathered_overlap = overlap.gather(
            1, neighbor_path.reshape(total_paths, -1)
        ).reshape(total_paths, NEIGHBOR_ODS, paths_per_od)
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
    def _path_max(edge_values, row_indices, col_indices, pte_values, total_paths: int):
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
        inverse_capacity = capacities.to(dtype=torch.float32).clamp_min(1e-6).reciprocal()
        path_sum = torch.sparse.mm(pte.to(dtype=torch.float32), inverse_capacity.t()).t()
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
        total_paths = int(pte.shape[0])
        paths_per_od = int(num_paths_per_pair)
        num_ods = total_paths // paths_per_od
        self._ensure_sparse_cache(pte, capacities, path_masks, paths_per_od)
        path_embeddings = self._current_path_embeddings(batch_size)
        capacity_features = self._capacity_features(capacities, pte, pte_info)
        model_dtype = path_embeddings.dtype

        query = self.spc_query(
            torch.cat((path_embeddings, capacity_features.to(dtype=model_dtype)), dim=-1)
        )
        safe_demand = tm2_pred.to(dtype=torch.float32).clamp_min(0.0)
        medium_kv = self.spc_medium_kv(
            torch.cat(
                (
                    path_embeddings,
                    torch.log1p(safe_demand).to(dtype=model_dtype),
                    capacity_features.to(dtype=model_dtype),
                ),
                dim=-1,
            )
        )

        # Stream one neighbor OD at a time.  The largest gather is B x P x K x d,
        # never B x P x R x K x d.  Only the four reduced OD messages survive.
        route_messages = []
        captured_route_attention = []
        route_simplex_error = 0.0
        scale = math.sqrt(float(ATTENTION_DIM))
        for slot in range(NEIGHBOR_ODS):
            indices = self._spc_neighbor_path[:, slot, :].reshape(-1)
            gathered = medium_kv.index_select(1, indices).reshape(
                batch_size, total_paths, paths_per_od, ATTENTION_DIM
            )
            route_scores = torch.einsum("bpd,bpkd->bpk", query, gathered) / scale
            route_scores = route_scores + self.route_overlap_scale * self._spc_neighbor_overlap[
                :, slot, :
            ].reshape(1, total_paths, paths_per_od).to(dtype=route_scores.dtype)
            medium_mask = self._spc_medium_mask_signature[
                self._spc_neighbor_path[:, slot, :]
            ].reshape(1, total_paths, paths_per_od)
            route_scores = route_scores.masked_fill(
                ~medium_mask, torch.finfo(route_scores.dtype).min
            )
            route_attention = torch.softmax(route_scores, dim=-1)
            route_messages.append(
                torch.einsum("bpk,bpkd->bpd", route_attention, gathered)
            )
            if self.research_spc_capture_attention:
                with torch.no_grad():
                    route_simplex_error = max(
                        route_simplex_error,
                        float((route_attention.sum(dim=-1) - 1.0).abs().max().item()),
                    )
                captured_route_attention.append(route_attention.detach())

        route_message = torch.stack(route_messages, dim=2)
        od_query = self.spc_od_query(query).unsqueeze(2)
        od_key = self.spc_od_key(route_message)
        od_value = self.spc_od_value(route_message)
        od_scores = (od_query * od_key).sum(dim=-1) / scale
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
        features = torch.cat((anchor, attention_features.to(dtype=anchor.dtype)), dim=-1)
        if tuple(features.shape) != (batch_size, total_paths, TOTAL_FEATURES):
            raise RuntimeError(f"Unexpected sparse-attention shape {tuple(features.shape)}")
        if self.research_spc_capture_attention and not torch.isfinite(features).all().item():
            raise RuntimeError("Sparse cross-attention produced non-finite features")

        self._last_route_attention = (
            torch.stack(captured_route_attention, dim=2)
            if captured_route_attention
            else None
        )
        self._last_od_attention = od_attention.detach() if self.research_spc_capture_attention else None
        self._last_spc_features = (
            features.detach() if self.research_spc_capture_attention else None
        )
        self._last_spc_audit = {
            "attention_dim": ATTENTION_DIM,
            "neighbor_ods": NEIGHBOR_ODS,
            "paths_per_medium_od": paths_per_od,
            "pairs_per_high_path": NEIGHBOR_ODS * paths_per_od,
            "high_medium_path_pairs_per_snapshot": total_paths * NEIGHBOR_ODS * paths_per_od,
            "feature_count": TOTAL_FEATURES,
            "route_attention_simplex_max_error": route_simplex_error,
            "od_attention_simplex_max_error": (
                float((od_attention.detach().sum(dim=-1) - 1.0).abs().max().item())
                if self.research_spc_capture_attention
                else None
            ),
            "largest_gather_shape": [batch_size, total_paths, paths_per_od, ATTENTION_DIM],
            "forbidden_full_5d_gather_materialized": False,
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


full.Hattrick = SparsePathCrossAttentionFastHattrick
full.OUTPUT_ROOT = THIS_DIR / "artifacts" / "from_scratch"


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "ordered_projection.py": FULL_DIR / "ordered_projection.py",
        "shared2x_full_objectives/run_experiment.py": FULL_DIR / "run_experiment.py",
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "utils/training_utils.py": ROOT / "utils" / "training_utils.py",
    }
    hashes = {name: full.sha256(path) for name, path in paths.items()}
    if ACTIVE_R8_WARMSTART is not None:
        hashes["warmstart_r8_checkpoint"] = full.sha256(ACTIVE_R8_WARMSTART)
    settings = json.dumps(
        {
            "attention_dim": ATTENTION_DIM,
            "neighbor_ods": NEIGHBOR_ODS,
            "neighbor_rule": "own OD plus top-3 max inverse-capacity path overlap",
            "attention": "streamed route-softmax then OD-softmax, one head",
            "anchor_features": ANCHOR_FEATURES,
            "zero_output_adapters": True,
            "strict_esm": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["sparse_path_cross_attention_fast/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


full.source_hashes = source_hashes


def method_description() -> dict:
    return {
        "method": "Hattrick fast sparse High-path/Medium-path cross-attention",
        "architecture": (
            "Each High path attends to the eight paths of four Medium ODs "
            "(own OD plus three structural neighbors), one OD at a time, then "
            "attends across the four reduced messages. The 8-d context and two "
            "native pressure anchors feed zero-output High adapters."
        ),
        "tensor_contract": {
            "high_query": "[B,3696,8]",
            "medium_tokens": "[B,3696,8]",
            "neighbor_od": "[3696,4] static",
            "largest_streamed_gather": "[B,3696,8,8]",
            "forbidden_full_gather": "[B,3696,4,8,8] is never materialized",
            "reduced_od_messages": "[B,3696,4,8]",
            "output_features": "[B,3696,10]",
        },
        "formal_cascade": "Inherited native High -> High+Medium -> High+Medium+Low",
        "formal_medium_stage": "Unchanged and recomputed after High",
        "projection": "unchanged six-objective ordered gradient projection",
        "objectives": list(full.OBJECTIVE_NAMES),
        "inference_information": (
            "ESM tm2_pred, topology/capacity path embeddings, PTE, capacities, "
            "and feasibility masks only; actual traffic is never a policy input"
        ),
        "warmstart": (
            "Optional lossy slicing of an R8,d16 checkpoint: first eight latent "
            "channels, four-neighbor runtime cache, and matching High-adapter "
            "feature columns. It is an initialization, not exact equivalence."
        ),
        "source_sha256": source_hashes(),
    }


def main() -> None:
    global ACTIVE_R8_WARMSTART
    parser = argparse.ArgumentParser(
        description="Shared-2x fast sparse High/Medium path cross-attention"
    )
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--warmstart-r8",
        type=Path,
        default=None,
        help="Optional R=8,d=16 model checkpoint to slice into this model",
    )
    args = parser.parse_args()

    if args.warmstart_r8 is not None:
        ACTIVE_R8_WARMSTART = args.warmstart_r8.resolve()
        if not ACTIVE_R8_WARMSTART.is_file():
            raise FileNotFoundError(ACTIVE_R8_WARMSTART)
        tag = full.sha256(ACTIVE_R8_WARMSTART)[:10]
        full.OUTPUT_ROOT = THIS_DIR / "artifacts" / f"warmstart_r8_{tag}"
    else:
        ACTIVE_R8_WARMSTART = None
        full.OUTPUT_ROOT = THIS_DIR / "artifacts" / "from_scratch"

    run_dir = full.run_one(args.level, args.seed, force=args.force)
    description = method_description()
    description.update(
        {
            "level": args.level,
            "seed": args.seed,
            "run_directory": str(run_dir),
            "r8_warmstart": str(ACTIVE_R8_WARMSTART) if ACTIVE_R8_WARMSTART else None,
        }
    )
    full.write_json(run_dir / "sparse_path_cross_attention_fast_method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
