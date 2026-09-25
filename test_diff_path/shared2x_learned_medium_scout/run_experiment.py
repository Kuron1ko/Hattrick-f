from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
    "shared2x_learned_medium_scout_full_runtime",
    FULL_DIR / "run_experiment.py",
)
NativeHattrick = full.Hattrick

SCOUT_HIDDEN_DIM = 32
SCOUT_FEATURE_COUNT = 4


class ZeroOutputResidual(nn.Module):
    """Small residual with an exactly zero, directly trainable output layer."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.kaiming_normal_(
            self.body[0].weight,
            a=0.02,
            nonlinearity="leaky_relu",
        )
        nn.init.zeros_(self.body[0].bias)
        # This preserves exact epoch-0 equivalence while exposing all hidden
        # channels to the first optimizer step.  A single scalar gate opened
        # too slowly under ordered projection in the Level-1 screen.
        nn.init.zeros_(self.body[2].weight)
        nn.init.zeros_(self.body[2].bias)

    def forward(self, inputs):
        return self.body(inputs)


class LearnedMediumScoutHattrick(NativeHattrick):
    """Native serial Hattrick with a learned provisional-Medium scout.

    Before High, a shared path MLP proposes a Medium routing from the current
    path embedding, ESM Medium demand, and inverse-capacity summaries.  One
    edge aggregation and one tiny RAU update refine that proposal.  The final
    provisional Medium edge load and a same-OD overlap signal become four
    High-path features.  The inherited formal Medium stage still runs after
    High and is not modified.
    """

    def __init__(self, props):
        props.future_lookahead = False
        super().__init__(props)
        hidden = SCOUT_HIDDEN_DIM

        # path embedding + log Medium demand + max/mean inverse capacity
        scout_base_dim = int(self.input_dim) + 3
        self.medium_scout_init = nn.Sequential(
            nn.Linear(scout_base_dim, hidden),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(hidden, 1),
        )
        # Base features plus initial logit/probability and three edge-feedback
        # values: absolute max/mean and within-OD relative mean pressure.
        self.medium_scout_rau = nn.Sequential(
            nn.Linear(scout_base_dim + 5, hidden),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(hidden, 1),
        )
        self._initialize_scout_mlp(self.medium_scout_init)
        self._initialize_scout_mlp(self.medium_scout_rau)

        # The core High hook is deliberately reused, but with four richer
        # features and exact zero-output residuals rather than a core-file change.
        self.medium_pressure_init_adapter = ZeroOutputResidual(
            self.mlp_11_dim + SCOUT_FEATURE_COUNT, hidden
        )
        self.medium_pressure_rau_adapter = ZeroOutputResidual(
            self.mlp_12_dim + SCOUT_FEATURE_COUNT, hidden
        )
        self.medium_pressure_feature_count = SCOUT_FEATURE_COUNT
        self.research_medium_pressure_lookahead_enabled = True

        # Static path incidence is materialized lazily from the runtime PTE.
        # It is non-persistent to retain native checkpoint portability.
        self.register_buffer(
            "_learned_scout_incidence", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_learned_scout_path_lengths", torch.empty(0), persistent=False
        )
        self._learned_scout_pte_shape: tuple[int, int] | None = None
        self._learned_scout_transformer_cache = None
        self._last_scout_split_ratios = None
        self._last_scout_features = None
        self._last_scout_audit: dict[str, object] = {}

    @staticmethod
    def _initialize_scout_mlp(module: nn.Sequential) -> None:
        nn.init.kaiming_normal_(
            module[0].weight,
            a=0.02,
            nonlinearity="leaky_relu",
        )
        nn.init.zeros_(module[0].bias)
        # Small initial logits keep the provisional policy close to diffuse
        # while preserving trainable, non-identical path scores.
        nn.init.normal_(module[2].weight, mean=0.0, std=0.02)
        nn.init.zeros_(module[2].bias)

    @property
    def last_scout_audit(self) -> dict[str, object]:
        return dict(self._last_scout_audit)

    def compute_transformer_output(self, *args, **kwargs):
        output = super().compute_transformer_output(*args, **kwargs)
        self._learned_scout_transformer_cache = output
        return output

    def _current_path_embeddings(self, batch_size: int) -> torch.Tensor:
        # Train/dynamic mode always produces a fresh temporary cache. Prefer it
        # over a static test cache that may remain from validation in the prior
        # epoch. Static test replay reaches ``self.transformer_output`` only
        # when no fresh transformer call occurred.
        output = self._learned_scout_transformer_cache
        if output is None and hasattr(self, "transformer_output"):
            output = self.transformer_output
        if output is None:
            raise RuntimeError(
                "Learned Medium scout has no current transformer path embedding"
            )
        if int(output.shape[0]) == 1 and int(batch_size) > 1:
            output = output.expand(batch_size, -1, -1, -1)
        if int(output.shape[0]) != int(batch_size):
            raise RuntimeError(
                f"Path-embedding batch {output.shape[0]} != {batch_size}"
            )
        path_embeddings = output[:, :, 0, :]
        # Do not retain a completed training graph beyond this forward pass.
        self._learned_scout_transformer_cache = None
        return path_embeddings

    def _ensure_incidence_cache(self, paths_to_edges) -> tuple[torch.Tensor, torch.Tensor]:
        pte = paths_to_edges.coalesce()
        shape = (int(pte.shape[0]), int(pte.shape[1]))
        valid = (
            self._learned_scout_pte_shape == shape
            and tuple(self._learned_scout_incidence.shape) == shape
            and self._learned_scout_incidence.device == pte.device
        )
        if not valid:
            incidence = pte.to(dtype=torch.float32).to_dense()
            lengths = incidence.sum(dim=-1).clamp_min(1.0)
            self._learned_scout_incidence = incidence
            self._learned_scout_path_lengths = lengths
            self._learned_scout_pte_shape = shape
        return self._learned_scout_incidence, self._learned_scout_path_lengths

    @staticmethod
    def _masked_center(
        values: torch.Tensor,
        mask: torch.Tensor,
        num_paths_per_pair: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, total_paths = values.shape
        paths_per_od = int(num_paths_per_pair)
        grouped = values.reshape(batch_size, -1, paths_per_od)
        grouped_mask = mask.reshape(1, -1, paths_per_od).to(dtype=values.dtype)
        counts = grouped_mask.sum(dim=-1, keepdim=True)
        if bool((counts == 0).any().item()):
            raise RuntimeError("At least one High OD has no feasible path")
        mean = (grouped * grouped_mask).sum(dim=-1, keepdim=True) / counts
        absolute = grouped * grouped_mask
        relative = (grouped - mean) * grouped_mask
        return absolute.reshape(batch_size, total_paths), relative.reshape(
            batch_size, total_paths
        )

    def _class_mask(self, path_masks, class_index: int, total_paths: int, device):
        mask = self.path_mask_for_class(path_masks, class_index)
        if mask is None:
            return torch.ones(total_paths, dtype=torch.bool, device=device)
        mask = mask.reshape(-1).to(device=device, dtype=torch.bool)
        if int(mask.numel()) != int(total_paths):
            raise RuntimeError("Path mask and PTE have different path counts")
        return mask

    @staticmethod
    def _path_max(
        edge_values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        pte_values: torch.Tensor,
        total_paths: int,
    ) -> torch.Tensor:
        selected = edge_values[:, col_indices] * pte_values.reshape(1, -1)
        output = torch.zeros(
            (edge_values.shape[0], total_paths),
            device=edge_values.device,
            dtype=edge_values.dtype,
        )
        output.scatter_reduce_(
            1,
            row_indices.reshape(1, -1).expand(edge_values.shape[0], -1),
            selected,
            reduce="amax",
            include_self=True,
        )
        return output

    def _path_edge_summaries(
        self,
        edge_values: torch.Tensor,
        pte_float,
        row_indices,
        col_indices,
        pte_values,
        path_lengths,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        path_sum = torch.sparse.mm(pte_float, edge_values.t()).t()
        path_mean = path_sum / path_lengths.reshape(1, -1)
        path_max = self._path_max(
            edge_values,
            row_indices,
            col_indices,
            pte_values,
            int(pte_float.shape[0]),
        )
        return path_max, path_mean

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
        """Run one learned provisional-Medium RAU and return High features."""

        pte, row_indices, col_indices, pte_values = pte_info
        pte = pte.coalesce()
        pte_float = pte.to(dtype=torch.float32)
        row_indices = row_indices.to(device=pte.device)
        col_indices = col_indices.to(device=pte.device)
        pte_values = pte_values.to(device=pte.device, dtype=torch.float32)
        total_paths, num_edges = int(pte.shape[0]), int(pte.shape[1])
        paths_per_od = int(num_paths_per_pair)
        if total_paths % paths_per_od != 0:
            raise RuntimeError("Path count is not divisible by paths per OD")
        num_ods = total_paths // paths_per_od
        if int(tm2_pred.shape[0]) != int(batch_size):
            raise RuntimeError("Medium prediction batch does not match batch_size")

        path_embeddings = self._current_path_embeddings(batch_size)
        incidence, path_lengths = self._ensure_incidence_cache(pte)
        safe_capacity = capacities.to(dtype=torch.float32).clamp_min(1e-6)
        inverse_capacity = safe_capacity.reciprocal()
        capacity_max, capacity_mean = self._path_edge_summaries(
            inverse_capacity,
            pte_float,
            row_indices,
            col_indices,
            pte_values,
            path_lengths,
        )
        capacity_features = torch.stack(
            (torch.log1p(capacity_max), torch.log1p(capacity_mean)), dim=-1
        )

        safe_tm2 = tm2_pred.to(dtype=torch.float32).clamp_min(0.0)
        demand_feature = torch.log1p(safe_tm2)
        scout_dtype = path_embeddings.dtype
        base_inputs = torch.cat(
            (
                path_embeddings,
                demand_feature.to(dtype=scout_dtype),
                capacity_features.to(dtype=scout_dtype),
            ),
            dim=-1,
        )
        initial_logits = self.medium_scout_init(base_inputs)
        medium_mask = self._class_mask(
            path_masks, 1, total_paths, pte.device
        )
        initial_edge_util, _, initial_split = self.compute_edge_utils(
            initial_logits,
            pte,
            safe_tm2.to(dtype=tm2_pred.dtype),
            capacities,
            props,
            batch_size,
            paths_per_od,
            add_epsilon=False,
            path_mask=medium_mask,
        )
        initial_max, initial_mean = self._path_edge_summaries(
            initial_edge_util.to(dtype=torch.float32).clamp_min(0.0),
            pte_float,
            row_indices,
            col_indices,
            pte_values,
            path_lengths,
        )
        high_mask = self._class_mask(path_masks, 0, total_paths, pte.device)
        initial_max_log = torch.log1p(initial_max)
        initial_mean_log = torch.log1p(initial_mean)
        _, initial_relative = self._masked_center(
            initial_mean_log, high_mask, paths_per_od
        )
        rau_inputs = torch.cat(
            (
                base_inputs,
                initial_logits,
                initial_split,
                initial_max_log.unsqueeze(-1).to(dtype=scout_dtype),
                initial_mean_log.unsqueeze(-1).to(dtype=scout_dtype),
                initial_relative.unsqueeze(-1).to(dtype=scout_dtype),
            ),
            dim=-1,
        )
        final_logits = initial_logits + self.medium_scout_rau(rau_inputs)
        final_edge_util, _, final_split = self.compute_edge_utils(
            final_logits,
            pte,
            safe_tm2.to(dtype=tm2_pred.dtype),
            capacities,
            props,
            batch_size,
            paths_per_od,
            add_epsilon=False,
            path_mask=medium_mask,
        )

        final_edge_util_f32 = final_edge_util.to(dtype=torch.float32).clamp_min(0.0)
        final_max, final_mean = self._path_edge_summaries(
            final_edge_util_f32,
            pte_float,
            row_indices,
            col_indices,
            pte_values,
            path_lengths,
        )
        global_absolute, _ = self._masked_center(
            torch.log1p(final_max), high_mask, paths_per_od
        )
        _, global_relative = self._masked_center(
            torch.log1p(final_mean), high_mask, paths_per_od
        )

        # Preserve Medium OD identity: for High candidate (o,k), measure its
        # capacity-normalized overlap with that same OD's learned provisional
        # Medium mixture, rather than only with globally aggregated load.
        grouped_incidence = incidence.reshape(
            num_ods, paths_per_od, num_edges
        )
        grouped_split = final_split.reshape(batch_size, num_ods, paths_per_od).to(
            dtype=torch.float32
        )
        demand_by_od = safe_tm2.reshape(
            batch_size, num_ods, paths_per_od, 1
        ).mean(dim=2).squeeze(-1)
        od_edge_probability = torch.einsum(
            "bok,oke->boe", grouped_split, grouped_incidence
        )
        od_edge_util = (
            od_edge_probability
            * demand_by_od.unsqueeze(-1)
            / safe_capacity.unsqueeze(1)
        )
        same_od_raw = torch.einsum(
            "oke,boe->bok", grouped_incidence, od_edge_util
        ).reshape(batch_size, total_paths)
        same_od_absolute, same_od_relative = self._masked_center(
            torch.log1p(same_od_raw.clamp_min(0.0)), high_mask, paths_per_od
        )

        features = torch.stack(
            (
                global_absolute,
                global_relative,
                same_od_absolute,
                same_od_relative,
            ),
            dim=-1,
        ).to(dtype=tm2_pred.dtype)
        if not torch.isfinite(features).all().item():
            raise RuntimeError("Learned Medium scout produced non-finite features")

        self._last_scout_split_ratios = final_split.detach()
        self._last_scout_features = features.detach()
        self._last_scout_audit = {
            "feature_count": SCOUT_FEATURE_COUNT,
            "provisional_medium_rau_steps": 1,
            "edge_aggregations": 2,
            "total_paths": total_paths,
            "num_ods": num_ods,
            "num_edges": num_edges,
            "formal_medium_stage_inherited": type(self).forward is NativeHattrick.forward,
            "strict_policy_inputs": [
                "tm2_pred",
                "path_embeddings",
                "paths_to_edges",
                "capacities",
                "path_masks",
            ],
        }
        return features


full.Hattrick = LearnedMediumScoutHattrick
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
            "hidden_dim": SCOUT_HIDDEN_DIM,
            "features": SCOUT_FEATURE_COUNT,
            "provisional_rau_steps": 1,
            "zero_output_residuals": 2,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["learned_medium_scout/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


full.source_hashes = source_hashes


def method_description() -> dict:
    return {
        "method": "Hattrick learned provisional-Medium scout",
        "architecture": (
            "Before native High, one shared 32-wide path MLP proposes Medium "
            "routing and one 32-wide edge-feedback RAU refines it. Four "
            "provisional edge-load/same-OD-conflict features feed High init and "
            "every High RAU through exact zero-output residuals."
        ),
        "feature_order": [
            "global_edge_load_absolute",
            "global_edge_load_within_high_od_relative",
            "same_medium_od_conflict_absolute",
            "same_medium_od_conflict_within_high_od_relative",
        ],
        "formal_cascade": "Inherited native High -> High+Medium -> High+Medium+Low",
        "formal_medium_stage": "Unchanged and recomputed after High",
        "projection": "unchanged six-objective ordered gradient projection",
        "objectives": list(full.OBJECTIVE_NAMES),
        "inference_information": (
            "ESM tm2_pred, path embeddings derived from topology/capacity, PTE, "
            "capacities, and feasibility masks only; actual traffic is not read "
            "by the policy scout"
        ),
        "zero_initialization": (
            "Both High residual output layers are initialized to exact zero, so "
            "the epoch-0 policy is bitwise native after matching native weights."
        ),
        "source_sha256": source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared-2x Hattrick learned provisional-Medium scout"
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
    full.write_json(run_dir / "learned_medium_scout_method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
