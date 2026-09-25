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
    "shared2x_medium_set_message_full_runtime",
    FULL_DIR / "run_experiment.py",
)
NativeHattrick = full.Hattrick


ANCHOR_FEATURES = 2
MESSAGE_TOKEN_DIM = 8
MESSAGE_FEATURES = 2 * MESSAGE_TOKEN_DIM
TOTAL_FEATURES = ANCHOR_FEATURES + MESSAGE_FEATURES
HIDDEN_DIM = 32


def _zero_output_branch(input_dim: int, hidden_dim: int) -> nn.Sequential:
    branch = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LeakyReLU(negative_slope=0.02),
        nn.Linear(hidden_dim, 1),
    )
    nn.init.kaiming_normal_(
        branch[0].weight, a=0.02, nonlinearity="leaky_relu"
    )
    nn.init.zeros_(branch[0].bias)
    nn.init.zeros_(branch[2].weight)
    nn.init.zeros_(branch[2].bias)
    return branch


class DualBranchAdapter(nn.Module):
    """Keep the scalar pressure anchor separate from the vector message."""

    def __init__(self, base_width: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.base_width = int(base_width)
        self.scalar_branch = _zero_output_branch(
            self.base_width + ANCHOR_FEATURES, hidden_dim
        )
        self.vector_branch = _zero_output_branch(
            self.base_width + MESSAGE_FEATURES, hidden_dim
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        expected = self.base_width + TOTAL_FEATURES
        if int(inputs.shape[-1]) != expected:
            raise RuntimeError(
                f"Dual-branch input width {inputs.shape[-1]} != {expected}"
            )
        base = inputs[..., : self.base_width]
        anchor = inputs[
            ..., self.base_width : self.base_width + ANCHOR_FEATURES
        ]
        vector = inputs[..., self.base_width + ANCHOR_FEATURES :]
        return self.scalar_branch(torch.cat((base, anchor), dim=-1)) + self.vector_branch(
            torch.cat((base, vector), dim=-1)
        )


class MediumSetMessageHattrick(NativeHattrick):
    """Native serial Hattrick with a vectorized Medium-set message to High.

    The formal High -> Medium -> Low cascade remains untouched.  Before High,
    Medium candidate paths are encoded jointly within each OD by a tiny
    DeepSets block.  Uniform and learned-gate messages are scattered to edges,
    capacity-normalized, and gathered back to every High candidate path.  The
    original two uniform-pressure scalars remain an independent anchor branch.
    """

    def __init__(self, props):
        props.future_lookahead = False
        super().__init__(props)
        base_width = int(self.input_dim) + 3
        self.medium_set_path_token = nn.Sequential(
            nn.Linear(base_width, HIDDEN_DIM),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(HIDDEN_DIM, MESSAGE_TOKEN_DIM),
            nn.Tanh(),
        )
        self.medium_set_context = nn.Sequential(
            nn.Linear(3 * MESSAGE_TOKEN_DIM, HIDDEN_DIM),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(HIDDEN_DIM, MESSAGE_TOKEN_DIM),
            nn.Tanh(),
        )
        self.medium_set_gate = nn.Linear(MESSAGE_TOKEN_DIM, 1)
        self._initialize_message_mlp(self.medium_set_path_token)
        self._initialize_message_mlp(self.medium_set_context)
        # A zero gate is exactly uniform, while the explicit uniform channel is
        # retained even after this learned gate moves away from uniform.
        nn.init.zeros_(self.medium_set_gate.weight)
        nn.init.zeros_(self.medium_set_gate.bias)

        self.medium_pressure_init_adapter = DualBranchAdapter(self.mlp_11_dim)
        self.medium_pressure_rau_adapter = DualBranchAdapter(self.mlp_12_dim)
        self.medium_pressure_feature_count = TOTAL_FEATURES
        self.research_medium_pressure_lookahead_enabled = True

        self.register_buffer(
            "_medium_set_path_lengths", torch.empty(0), persistent=False
        )
        self._medium_set_pte_signature: tuple[int, int, int] | None = None
        self._medium_set_cache_builds = 0
        self._medium_set_transformer_cache = None
        self._last_medium_set_gate_split = None
        self._last_medium_set_uniform_split = None
        self._last_medium_set_anchor_features = None
        self._last_medium_set_vector_features = None
        self._last_medium_set_audit: dict[str, object] = {}

    @staticmethod
    def _initialize_message_mlp(module: nn.Sequential) -> None:
        for layer in module:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, a=0.02, nonlinearity="leaky_relu"
                )
                nn.init.zeros_(layer.bias)

    @property
    def medium_set_cache_builds(self) -> int:
        return int(self._medium_set_cache_builds)

    @property
    def last_medium_set_audit(self) -> dict[str, object]:
        return dict(self._last_medium_set_audit)

    def compute_transformer_output(self, *args, **kwargs):
        output = super().compute_transformer_output(*args, **kwargs)
        self._medium_set_transformer_cache = output
        return output

    def _current_path_embeddings(self, batch_size: int) -> torch.Tensor:
        output = self._medium_set_transformer_cache
        if output is None and hasattr(self, "transformer_output"):
            output = self.transformer_output
        if output is None:
            raise RuntimeError("Medium set-message has no current path embeddings")
        if int(output.shape[0]) == 1 and int(batch_size) > 1:
            output = output.expand(batch_size, -1, -1, -1)
        if int(output.shape[0]) != int(batch_size):
            raise RuntimeError(
                f"Path-embedding batch {output.shape[0]} != {batch_size}"
            )
        path_embeddings = output[:, :, 0, :]
        self._medium_set_transformer_cache = None
        return path_embeddings

    def _ensure_path_length_cache(self, paths_to_edges) -> torch.Tensor:
        pte = paths_to_edges.coalesce()
        signature = (int(pte.shape[0]), int(pte.shape[1]), int(pte._nnz()))
        valid = (
            self._medium_set_pte_signature == signature
            and int(self._medium_set_path_lengths.numel()) == int(pte.shape[0])
            and self._medium_set_path_lengths.device == pte.device
        )
        if not valid:
            lengths = torch.sparse.sum(
                pte.to(dtype=torch.float32), dim=1
            ).to_dense().clamp_min(1.0)
            self._medium_set_path_lengths = lengths
            self._medium_set_pte_signature = signature
            self._medium_set_cache_builds += 1
        return self._medium_set_path_lengths

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

    @staticmethod
    def _sparse_batch_mm(matrix: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Apply sparse [O,I] to dense [B,I,D], returning [B,O,D]."""

        if values.dim() != 3 or int(values.shape[1]) != int(matrix.shape[1]):
            raise RuntimeError("Sparse batch-mm dimensions do not agree")
        batch_size, _, width = values.shape
        flattened = values.permute(1, 0, 2).reshape(matrix.shape[1], -1)
        output = torch.sparse.mm(matrix, flattened)
        return output.reshape(matrix.shape[0], batch_size, width).permute(1, 0, 2)

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
        if total_paths % paths_per_od != 0:
            raise RuntimeError("Path count is not divisible by paths per OD")
        num_ods = total_paths // paths_per_od
        if int(tm2_pred.shape[0]) != int(batch_size):
            raise RuntimeError("Medium prediction batch does not match batch_size")

        # Preserve the original, interpretable two-scalar pressure channel.
        anchor = NativeHattrick.compute_medium_pressure_features(
            self,
            tm2_pred,
            capacities,
            paths_to_edges,
            pte_info,
            batch_size,
            paths_per_od,
            props,
            path_masks,
        )

        path_embeddings = self._current_path_embeddings(batch_size)
        path_lengths = self._ensure_path_length_cache(pte)
        safe_capacity = capacities.to(dtype=torch.float32).clamp_min(1e-8)
        inverse_capacity = safe_capacity.reciprocal()
        capacity_sum = torch.sparse.mm(pte, inverse_capacity.t()).t()
        capacity_mean = capacity_sum / path_lengths.reshape(1, -1)
        capacity_max = self._path_max(
            inverse_capacity,
            row_indices,
            col_indices,
            pte_values,
            total_paths,
        )

        safe_tm2 = tm2_pred.to(dtype=torch.float32).clamp_min(0.0)
        base_inputs = torch.cat(
            (
                path_embeddings,
                torch.log1p(safe_tm2).to(dtype=path_embeddings.dtype),
                torch.stack(
                    (torch.log1p(capacity_max), torch.log1p(capacity_mean)),
                    dim=-1,
                ).to(dtype=path_embeddings.dtype),
            ),
            dim=-1,
        )
        path_tokens = self.medium_set_path_token(base_inputs).reshape(
            batch_size, num_ods, paths_per_od, MESSAGE_TOKEN_DIM
        )

        medium_mask = self._class_mask(path_masks, 1, total_paths, pte.device)
        grouped_mask = medium_mask.reshape(1, num_ods, paths_per_od)
        mask_float = grouped_mask.to(dtype=path_tokens.dtype).unsqueeze(-1)
        counts = mask_float.sum(dim=2, keepdim=True)
        if bool((counts == 0).any().item()):
            raise RuntimeError("At least one Medium OD has no feasible path")
        path_tokens = path_tokens * mask_float
        od_mean = path_tokens.sum(dim=2, keepdim=True) / counts
        contextual_inputs = torch.cat(
            (
                path_tokens,
                od_mean.expand_as(path_tokens),
                path_tokens - od_mean,
            ),
            dim=-1,
        )
        contextual_tokens = self.medium_set_context(contextual_inputs) * mask_float

        gate_logits = self.medium_set_gate(contextual_tokens).squeeze(-1)
        gate_logits = gate_logits.masked_fill(
            ~grouped_mask, torch.finfo(gate_logits.dtype).min
        )
        learned_split = torch.softmax(gate_logits, dim=2) * grouped_mask
        uniform_split = grouped_mask.to(dtype=path_tokens.dtype) / counts.squeeze(-1)

        demand_by_od = safe_tm2.reshape(
            batch_size, num_ods, paths_per_od, 1
        ).mean(dim=2).squeeze(-1)
        demand_scale = demand_by_od.unsqueeze(-1).unsqueeze(-1)
        uniform_messages = (
            demand_scale * uniform_split.unsqueeze(-1) * path_tokens
        )
        learned_messages = (
            demand_scale * learned_split.unsqueeze(-1) * contextual_tokens
        )
        path_messages = torch.cat(
            (uniform_messages, learned_messages), dim=-1
        ).reshape(batch_size, total_paths, MESSAGE_FEATURES)

        edge_messages = self._sparse_batch_mm(pte.transpose(0, 1), path_messages)
        edge_messages = edge_messages / safe_capacity.unsqueeze(-1)
        path_message_features = self._sparse_batch_mm(pte, edge_messages)
        path_message_features = path_message_features / path_lengths.reshape(1, -1, 1)
        path_message_features = self._signed_log1p(path_message_features)

        high_mask = self._class_mask(path_masks, 0, total_paths, pte.device)
        path_message_features = path_message_features * high_mask.reshape(1, -1, 1)
        vector = path_message_features.to(dtype=tm2_pred.dtype)
        features = torch.cat((anchor, vector), dim=-1)
        if tuple(features.shape) != (batch_size, total_paths, TOTAL_FEATURES):
            raise RuntimeError(f"Unexpected set-message feature shape {features.shape}")
        if not torch.isfinite(features).all().item():
            raise RuntimeError("Medium set-message produced non-finite features")

        self._last_medium_set_gate_split = learned_split.detach().reshape(
            batch_size, total_paths
        )
        self._last_medium_set_uniform_split = uniform_split.detach().reshape(
            1, total_paths
        )
        self._last_medium_set_anchor_features = anchor.detach()
        self._last_medium_set_vector_features = vector.detach()
        self._last_medium_set_audit = {
            "feature_count": TOTAL_FEATURES,
            "anchor_features": ANCHOR_FEATURES,
            "message_features": MESSAGE_FEATURES,
            "token_dim": MESSAGE_TOKEN_DIM,
            "total_paths": total_paths,
            "num_ods": num_ods,
            "num_edges": num_edges,
            "cache_builds": self.medium_set_cache_builds,
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


full.Hattrick = MediumSetMessageHattrick
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
            "anchor_features": ANCHOR_FEATURES,
            "message_token_dim": MESSAGE_TOKEN_DIM,
            "message_features": MESSAGE_FEATURES,
            "context": "token, OD mean, token-minus-mean",
            "channels": ["uniform", "learned zero-logit gate"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["medium_set_message/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


full.source_hashes = source_hashes


def method_description() -> dict[str, object]:
    return {
        "method": "Hattrick Medium candidate-set message",
        "architecture": (
            "Unchanged native High -> Medium -> Low cascade. Before High, a tiny "
            "DeepSets encoder maps the complete eight-path Medium candidate set "
            "through capacity-normalized edge vector messages to every High path."
        ),
        "anchor_branch": (
            "Original two uniform-Medium pressure scalars (path max and mean "
            "log1p utilization), with its own zero-output High adapter branch."
        ),
        "vector_branch": {
            "token_dim": MESSAGE_TOKEN_DIM,
            "output_dim": MESSAGE_FEATURES,
            "context": "path token + feasible OD mean + token-minus-mean",
            "channels": [
                "explicit uniform Medium message",
                "zero-logit learned-gate Medium message",
            ],
            "adapter": "independent zero-output High init/RAU residual branch",
        },
        "projection": "unchanged six-objective ordered gradient projection",
        "objectives": list(full.OBJECTIVE_NAMES),
        "inference_information": (
            "ESM Medium prediction, static topology/candidate paths/masks and "
            "capacities only; actual traffic is excluded from policy inference"
        ),
        "output_root": str(full.OUTPUT_ROOT),
        "source_sha256": source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared-2x six-objective Hattrick Medium set-message experiment"
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
    full.write_json(run_dir / "medium_set_message_method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
