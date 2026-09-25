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
LEARNED_DIR = TEST_DIR / "shared2x_learned_medium_scout"
for item in (str(ROOT), str(TEST_DIR), str(FULL_DIR), str(LEARNED_DIR)):
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


learned = load_module(
    "shared2x_semantic_masm_learned_runtime",
    LEARNED_DIR / "run_experiment.py",
)
full = learned.full
NativeHattrick = learned.NativeHattrick


TOKEN_DIM = 8
ATTENTION_HEADS = 2
ATTENTION_HEAD_DIM = TOKEN_DIM // ATTENTION_HEADS
TAIL_POWER = 4.0
PRESSURE_FEATURES = 2
MESSAGE_FEATURES = 4 * TOKEN_DIM
TOTAL_FEATURES = PRESSURE_FEATURES + MESSAGE_FEATURES
ADAPTER_HIDDEN_DIM = 32
SET_HIDDEN_DIM = 16


class SemanticMASMHattrick(learned.LearnedMediumScoutHattrick):
    """Serial Hattrick with an anchored Medium alternative-set message.

    The formal native High -> Medium -> Low cascade is unchanged.  Before
    High, the *predicted* Medium demand is combined with eight fixed route
    semantics and a small permutation-equivariant residual set encoder.  The
    resulting vector messages travel path -> edge -> High path through the
    native PTE.  No provisional Medium split from formal Stage2 is used.
    """

    def __init__(self, props):
        props.future_lookahead = False
        super().__init__(props)

        # Remove the unconstrained provisional routing network inherited only
        # to reuse its transformer hook and sparse-path helper functions.
        del self.medium_scout_init
        del self.medium_scout_rau

        embedding_dim = int(self.input_dim)
        self.masm_path_residual = nn.Linear(embedding_dim, TOKEN_DIM)
        self.masm_query = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.masm_key = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.masm_value = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.masm_attention_output = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.masm_ffn = nn.Sequential(
            nn.Linear(TOKEN_DIM, SET_HIDDEN_DIM),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(SET_HIDDEN_DIM, TOKEN_DIM),
        )

        # The learned encoder is a bounded residual around fixed semantics.
        # At initialization its output is exactly the fixed semantic token.
        nn.init.zeros_(self.masm_path_residual.weight)
        nn.init.zeros_(self.masm_path_residual.bias)
        for layer in (self.masm_query, self.masm_key, self.masm_value):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.masm_attention_output.weight)
        nn.init.zeros_(self.masm_attention_output.bias)
        nn.init.kaiming_normal_(
            self.masm_ffn[0].weight, a=0.02, nonlinearity="leaky_relu"
        )
        nn.init.zeros_(self.masm_ffn[0].bias)
        nn.init.zeros_(self.masm_ffn[2].weight)
        nn.init.zeros_(self.masm_ffn[2].bias)

        self.medium_pressure_init_adapter = learned.ZeroOutputResidual(
            self.mlp_11_dim + TOTAL_FEATURES, ADAPTER_HIDDEN_DIM
        )
        self.medium_pressure_rau_adapter = learned.ZeroOutputResidual(
            self.mlp_12_dim + TOTAL_FEATURES, ADAPTER_HIDDEN_DIM
        )
        self.medium_pressure_feature_count = TOTAL_FEATURES
        self.research_medium_pressure_lookahead_enabled = True

        # All topology-derived tensors are built once.  They are deliberately
        # non-persistent so a native checkpoint remains directly loadable.
        self.register_buffer("_masm_grouped_incidence", torch.empty(0), persistent=False)
        self.register_buffer("_masm_overlap", torch.empty(0), persistent=False)
        self.register_buffer("_masm_semantics", torch.empty(0), persistent=False)
        self.register_buffer("_masm_diversity", torch.empty(0), persistent=False)
        self.register_buffer("_masm_medium_mask", torch.empty(0), persistent=False)
        self.register_buffer("_masm_capacity_reference", torch.empty(0), persistent=False)
        self._masm_cache_signature: tuple[int, int, int, int] | None = None
        self._masm_cache_builds = 0
        self._last_masm_features = None
        self._last_masm_encoded_routes = None
        self._last_masm_audit: dict[str, object] = {}

    @property
    def masm_cache_builds(self) -> int:
        return int(self._masm_cache_builds)

    @property
    def last_masm_audit(self) -> dict[str, object]:
        return dict(self._last_masm_audit)

    @staticmethod
    def build_semantic_anchors(
        incidence: torch.Tensor,
        capacities: torch.Tensor,
        feasible_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return weighted-Jaccard overlap, d=8 semantics and diversity split.

        Args:
            incidence: binary route incidence [OD, K, E].
            capacities: static edge capacities [E].
            feasible_mask: Medium route feasibility [OD, K].

        Candidate position is never encoded, hence permuting K permutes the
        first two outputs and the diversity split in exactly the same way.
        """

        if incidence.dim() != 3:
            raise RuntimeError("incidence must have shape [OD,K,E]")
        num_ods, paths_per_od, num_edges = incidence.shape
        if tuple(feasible_mask.shape) != (num_ods, paths_per_od):
            raise RuntimeError("feasible mask does not match incidence")
        capacity = capacities.reshape(-1).to(
            device=incidence.device, dtype=torch.float32
        )
        if int(capacity.numel()) != int(num_edges):
            raise RuntimeError("capacity width does not match incidence")

        routes = incidence.to(dtype=torch.float32).clamp_min(0.0)
        mask = feasible_mask.to(device=routes.device, dtype=torch.bool)
        mask_f = mask.to(dtype=routes.dtype)
        inverse_capacity = capacity.clamp_min(1e-8).reciprocal()
        weighted_routes = routes * inverse_capacity.reshape(1, 1, -1)
        footprint = weighted_routes.sum(dim=-1)
        shared = torch.einsum("oke,ole->okl", routes, weighted_routes)
        union = footprint.unsqueeze(2) + footprint.unsqueeze(1) - shared
        overlap = torch.where(union > 0, shared / union.clamp_min(1e-12), 0.0)

        pair_valid = mask.unsqueeze(2) & mask.unsqueeze(1)
        overlap = overlap * pair_valid.to(dtype=overlap.dtype)
        eye = torch.eye(paths_per_od, dtype=torch.bool, device=routes.device)
        other_valid = pair_valid & ~eye.reshape(1, paths_per_od, paths_per_od)
        other_count = other_valid.sum(dim=-1)
        overlap_sum = (overlap * other_valid).sum(dim=-1)
        common_overlap = torch.where(
            other_count > 0,
            overlap_sum / other_count.clamp_min(1).to(dtype=overlap.dtype),
            torch.ones_like(overlap_sum),
        )
        masked_overlap = overlap.masked_fill(~other_valid, float("inf"))
        minimum_overlap = masked_overlap.amin(dim=-1)
        minimum_overlap = torch.where(
            other_count > 0, minimum_overlap, torch.ones_like(minimum_overlap)
        )
        common_overlap = common_overlap * mask_f
        minimum_overlap = minimum_overlap * mask_f
        clean_alternative = (1.0 - minimum_overlap).clamp(0.0, 1.0) * mask_f

        route_length = routes.sum(dim=-1).clamp_min(1.0)
        inverse_bottleneck = weighted_routes.amax(dim=-1)
        inverse_mean = weighted_routes.sum(dim=-1) / route_length

        def normalize(values: torch.Tensor) -> torch.Tensor:
            logged = torch.log1p(values.clamp_min(0.0))
            scale = logged.amax().clamp_min(1e-8)
            return logged / scale

        inverse_bottleneck = normalize(inverse_bottleneck) * mask_f
        inverse_mean = normalize(inverse_mean) * mask_f
        length_normalized = route_length / route_length.amax().clamp_min(1.0)
        length_normalized = length_normalized * mask_f

        # Similar alternatives share mass; genuinely different alternatives
        # retain more mass.  The normalized weights sum to one per Medium OD.
        raw_diversity = mask_f / (1.0 + overlap_sum)
        diversity = raw_diversity / raw_diversity.sum(dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        diversity_semantic = diversity * mask_f.sum(dim=-1, keepdim=True)

        semantics = torch.stack(
            (
                mask_f,
                inverse_bottleneck,
                inverse_mean,
                common_overlap,
                minimum_overlap,
                clean_alternative,
                diversity_semantic,
                length_normalized,
            ),
            dim=-1,
        )
        return overlap, semantics, diversity

    def _ensure_masm_cache(
        self,
        paths_to_edges: torch.Tensor,
        capacities: torch.Tensor,
        medium_mask: torch.Tensor,
        paths_per_od: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pte = paths_to_edges.coalesce().to(dtype=torch.float32)
        total_paths, num_edges = int(pte.shape[0]), int(pte.shape[1])
        if total_paths % paths_per_od != 0:
            raise RuntimeError("Path count is not divisible by paths per OD")
        num_ods = total_paths // paths_per_od
        signature = (total_paths, num_edges, int(pte._nnz()), paths_per_od)

        reference_capacity = capacities.to(dtype=torch.float32)
        if reference_capacity.dim() == 2:
            if not torch.equal(
                reference_capacity,
                reference_capacity[:1].expand_as(reference_capacity),
            ):
                raise RuntimeError("MASM requires topology-static capacities in a batch")
            reference_capacity = reference_capacity[0]
        reference_capacity = reference_capacity.reshape(-1)

        valid = (
            self._masm_cache_signature == signature
            and tuple(self._masm_grouped_incidence.shape)
            == (num_ods, paths_per_od, num_edges)
            and self._masm_grouped_incidence.device == pte.device
            and torch.equal(self._masm_capacity_reference, reference_capacity)
            and torch.equal(self._masm_medium_mask, medium_mask.reshape(num_ods, paths_per_od))
        )
        if not valid:
            incidence = pte.to_dense().reshape(num_ods, paths_per_od, num_edges)
            overlap, semantics, diversity = self.build_semantic_anchors(
                incidence,
                reference_capacity,
                medium_mask.reshape(num_ods, paths_per_od),
            )
            self._masm_grouped_incidence = incidence
            self._masm_overlap = overlap
            self._masm_semantics = semantics
            self._masm_diversity = diversity
            self._masm_medium_mask = medium_mask.reshape(num_ods, paths_per_od)
            self._masm_capacity_reference = reference_capacity.detach().clone()
            self._masm_cache_signature = signature
            self._masm_cache_builds += 1
        return (
            self._masm_grouped_incidence,
            self._masm_overlap,
            self._masm_semantics,
            self._masm_diversity,
        )

    def _encode_alternative_set(
        self,
        path_embeddings: torch.Tensor,
        semantics: torch.Tensor,
        overlap: torch.Tensor,
        medium_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_ods, paths_per_od, _ = path_embeddings.shape
        fixed = semantics.unsqueeze(0).expand(batch_size, -1, -1, -1)
        path_residual = self.masm_path_residual(path_embeddings)
        base = fixed + path_residual

        def heads(values: torch.Tensor) -> torch.Tensor:
            return values.reshape(
                batch_size, num_ods, paths_per_od, ATTENTION_HEADS, ATTENTION_HEAD_DIM
            ).permute(0, 1, 3, 2, 4)

        query = heads(self.masm_query(base))
        key = heads(self.masm_key(base))
        value = heads(self.masm_value(base))
        scores = torch.einsum("bohkd,bohld->bohkl", query, key)
        scores = scores / math.sqrt(float(ATTENTION_HEAD_DIM))
        scores = scores - 2.0 * overlap.reshape(
            1, num_ods, 1, paths_per_od, paths_per_od
        )
        key_mask = medium_mask.reshape(1, num_ods, 1, 1, paths_per_od)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bohkl,bohld->bohkd", weights, value)
        context = context.permute(0, 1, 3, 2, 4).reshape(
            batch_size, num_ods, paths_per_od, TOKEN_DIM
        )
        attention_residual = self.masm_attention_output(context)
        ffn_residual = self.masm_ffn(base + attention_residual)
        learned_residual = path_residual + attention_residual + ffn_residual
        encoded = fixed + 0.25 * torch.tanh(learned_residual)
        return encoded * medium_mask.reshape(1, num_ods, paths_per_od, 1)

    @staticmethod
    def _sparse_vector_path_pools(
        edge_tokens: torch.Tensor,
        pte: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        pte_values: torch.Tensor,
        path_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return edge-token maximum and mean on every path."""

        batch_size, num_edges, width = edge_tokens.shape
        total_paths = int(pte.shape[0])
        flattened = edge_tokens.permute(1, 0, 2).reshape(num_edges, -1)
        summed = torch.sparse.mm(pte, flattened)
        means = summed.reshape(total_paths, batch_size, width).permute(1, 0, 2)
        means = means / path_lengths.reshape(1, total_paths, 1)

        selected = edge_tokens[:, col_indices, :] * pte_values.reshape(1, -1, 1)
        maxima = torch.full(
            (batch_size, total_paths, width),
            -torch.inf,
            device=edge_tokens.device,
            dtype=edge_tokens.dtype,
        )
        maxima.scatter_reduce_(
            1,
            row_indices.reshape(1, -1, 1).expand(batch_size, -1, width),
            selected,
            reduce="amax",
            include_self=True,
        )
        maxima = torch.where(torch.isfinite(maxima), maxima, 0.0)
        return maxima, means

    @staticmethod
    def _center_high_vectors(
        values: torch.Tensor,
        high_mask: torch.Tensor,
        paths_per_od: int,
    ) -> torch.Tensor:
        batch_size, total_paths, width = values.shape
        grouped = values.reshape(batch_size, -1, paths_per_od, width)
        grouped_mask = high_mask.reshape(1, -1, paths_per_od, 1).to(values.dtype)
        counts = grouped_mask.sum(dim=2, keepdim=True)
        if bool((counts == 0).any().item()):
            raise RuntimeError("At least one High OD has no feasible path")
        mean = (grouped * grouped_mask).sum(dim=2, keepdim=True) / counts
        return ((grouped - mean) * grouped_mask).reshape(batch_size, total_paths, width)

    @staticmethod
    def _signed_log1p(values: torch.Tensor) -> torch.Tensor:
        return torch.sign(values) * torch.log1p(values.abs())

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
        pte, row_indices, col_indices, pte_values = pte_info
        pte = pte.coalesce().to(dtype=torch.float32)
        row_indices = row_indices.to(device=pte.device)
        col_indices = col_indices.to(device=pte.device)
        pte_values = pte_values.to(device=pte.device, dtype=torch.float32)
        total_paths, num_edges = int(pte.shape[0]), int(pte.shape[1])
        paths_per_od = int(num_paths_per_pair)
        num_ods = total_paths // paths_per_od

        # Keep the original uniform Medium-pressure scalars as an explicit,
        # interpretable absolute-load anchor.
        pressure_anchor = NativeHattrick.compute_medium_pressure_features(
            self,
            tm2_pred,
            capacities,
            paths_to_edges,
            pte_info,
            batch_size,
            paths_per_od,
            props,
            path_masks,
        ).to(dtype=torch.float32)

        high_mask = self._class_mask(path_masks, 0, total_paths, pte.device)
        medium_mask_flat = self._class_mask(path_masks, 1, total_paths, pte.device)
        medium_mask = medium_mask_flat.reshape(num_ods, paths_per_od)
        incidence, overlap, semantics, diversity = self._ensure_masm_cache(
            pte, capacities, medium_mask_flat, paths_per_od
        )
        path_embeddings = self._current_path_embeddings(batch_size).reshape(
            batch_size, num_ods, paths_per_od, -1
        )
        encoded = self._encode_alternative_set(
            path_embeddings, semantics, overlap, medium_mask
        )

        safe_tm2 = tm2_pred.to(dtype=torch.float32).clamp_min(0.0)
        demand_by_od = safe_tm2.reshape(
            batch_size, num_ods, paths_per_od, -1
        ).mean(dim=(2, 3))
        route_scale = demand_by_od.unsqueeze(-1) * diversity.unsqueeze(0)
        route_messages = encoded * route_scale.unsqueeze(-1)

        capacity = capacities.to(dtype=torch.float32).clamp_min(1e-8)
        if int(capacity.shape[0]) == 1 and int(batch_size) > 1:
            capacity = capacity.expand(batch_size, -1)
        od_edge_tokens = torch.einsum(
            "bokd,oke->boed", route_messages, incidence
        ) / capacity.reshape(batch_size, 1, num_edges, 1)
        edge_sum = od_edge_tokens.sum(dim=1)

        # Tail OD selection remains tied to fixed semantics.  It emphasizes an
        # OD only when its predicted demand reaches an edge through routes with
        # no clean alternative; a learned latent cannot redefine this notion.
        fixed_messages = semantics.unsqueeze(0) * route_scale.unsqueeze(-1)
        fixed_od_edge = torch.einsum(
            "bokd,oke->boed", fixed_messages, incidence
        ) / capacity.reshape(batch_size, 1, num_edges, 1)
        fixed_risk = fixed_od_edge[..., 0] * (1.0 + fixed_od_edge[..., 4])
        tail_mass = fixed_risk.clamp_min(0.0).pow(TAIL_POWER)
        tail_weights = tail_mass / tail_mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
        edge_tail = (tail_weights.unsqueeze(-1) * od_edge_tokens).sum(dim=1)

        path_lengths = incidence.reshape(total_paths, num_edges).sum(dim=-1).clamp_min(1.0)
        sum_max, sum_mean = self._sparse_vector_path_pools(
            edge_sum, pte, row_indices, col_indices, pte_values, path_lengths
        )
        tail_max, tail_mean = self._sparse_vector_path_pools(
            edge_tail, pte, row_indices, col_indices, pte_values, path_lengths
        )
        message = torch.cat((sum_max, sum_mean, tail_max, tail_mean), dim=-1)
        message = self._signed_log1p(message)
        message = self._center_high_vectors(message, high_mask, paths_per_od)
        pressure_anchor = pressure_anchor * high_mask.reshape(1, -1, 1)
        features = torch.cat((pressure_anchor, message), dim=-1).to(tm2_pred.dtype)

        if tuple(features.shape) != (batch_size, total_paths, TOTAL_FEATURES):
            raise RuntimeError(f"Unexpected MASM feature shape {tuple(features.shape)}")
        if not torch.isfinite(features).all().item():
            raise RuntimeError("Semantic MASM produced non-finite features")

        self._last_masm_features = features.detach()
        self._last_masm_encoded_routes = encoded.detach()
        self._last_masm_audit = {
            "token_dim": TOKEN_DIM,
            "attention_heads": ATTENTION_HEADS,
            "tail_power": TAIL_POWER,
            "feature_count": TOTAL_FEATURES,
            "cache_builds": self.masm_cache_builds,
            "total_paths": total_paths,
            "num_ods": num_ods,
            "paths_per_od": paths_per_od,
            "num_edges": num_edges,
            "formal_medium_stage_inherited": type(self).forward is NativeHattrick.forward,
            "strict_policy_inputs": [
                "tm2_pred",
                "path_embeddings",
                "paths_to_edges",
                "capacities",
                "path_masks",
            ],
            "fixed_semantic_channels": [
                "feasible_coverage",
                "inverse_bottleneck_capacity",
                "inverse_mean_capacity",
                "mean_common_edge_overlap",
                "minimum_overlap_no_clean_alternative",
                "clean_alternative",
                "diversity_mass",
                "normalized_path_length",
            ],
        }
        return features


full.Hattrick = SemanticMASMHattrick
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
            "token_dim": TOKEN_DIM,
            "attention_heads": ATTENTION_HEADS,
            "tail_power": TAIL_POWER,
            "pressure_features": PRESSURE_FEATURES,
            "message_features": MESSAGE_FEATURES,
            "zero_output_residuals": 2,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["semantic_masm/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


full.source_hashes = source_hashes


def method_description() -> dict:
    return {
        "method": "Hattrick semantic Medium Alternative-Set Message (MASM)",
        "architecture": (
            "Before native High, each Medium OD's eight alternatives receive a fixed "
            "d=8 semantic token plus a bounded permutation-equivariant residual. "
            "PTE path-to-edge sum and fixed-semantic tail aggregation are returned "
            "to High using per-path edge maximum and mean."
        ),
        "fixed_semantics": [
            "feasible coverage",
            "inverse bottleneck and mean capacity",
            "common-edge overlap",
            "clean/no-clean alternative",
            "diversity mass",
            "path length",
        ],
        "pressure_anchor": "native uniform-Medium maximum and mean edge pressure",
        "formal_cascade": "Inherited native High -> High+Medium -> High+Medium+Low",
        "formal_medium_stage": "Unchanged and recomputed only after High",
        "projection": "unchanged complete six-objective ordered gradient projection",
        "objectives": list(full.OBJECTIVE_NAMES),
        "inference_information": (
            "Strict ESM tm2_pred, topology/path embeddings, PTE, capacities and "
            "feasibility masks only; actual traffic is excluded from the policy message"
        ),
        "zero_initialization": (
            "Both High residual output layers are exactly zero; a native state load "
            "therefore gives bitwise native epoch-0 policy."
        ),
        "source_sha256": source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared-2x semantic MASM experiment")
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = full.run_one(args.level, args.seed, force=args.force)
    description = method_description()
    description.update(
        {"level": args.level, "seed": args.seed, "run_directory": str(run_dir)}
    )
    full.write_json(run_dir / "semantic_masm_method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
