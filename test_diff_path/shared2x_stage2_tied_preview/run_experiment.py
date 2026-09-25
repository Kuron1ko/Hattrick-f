from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch


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
    "shared2x_stage2_tied_preview_learned_runtime",
    LEARNED_DIR / "run_experiment.py",
)
full = learned.full
NativeHattrick = learned.NativeHattrick
FEATURE_COUNT = 4


class Stage2TiedPreviewHattrick(learned.LearnedMediumScoutHattrick):
    """Preview Medium with the formal Stage-2 weights and input construction.

    A native no-lookahead Stage 1 is replayed before the actual High decision.
    Its output is passed through the *same* ``mlp21`` and one exact ``mlp22``
    RAU used by formal Stage 2.  No independent provisional-routing head is
    retained.  Four load summaries from that tied preview feed the inherited
    zero-output High residual adapters; the formal serial cascade remains the
    superclass forward.
    """

    def __init__(self, props):
        super().__init__(props)
        # The parent supplies useful topology helpers and four-feature
        # zero-output adapters.  Remove both independent scout heads so every
        # provisional Medium logit comes from formal mlp21/mlp22 parameters.
        del self.medium_scout_init
        del self.medium_scout_rau
        self._preview_tm1_pred = None
        self._preview_padded_edge_ids = None
        self._last_preview_split_ratios = None
        self._last_preview_features = None
        self._last_preview_audit: dict[str, object] = {}

    @property
    def last_preview_audit(self) -> dict[str, object]:
        return dict(self._last_preview_audit)

    @property
    def preview_parameter_data_ptrs(self) -> dict[str, list[int]]:
        # These are the objects directly invoked by the preview.  There is no
        # copied/twin state dict whose synchronization could drift.
        return {
            "mlp11": [int(parameter.data_ptr()) for parameter in self.mlp11.parameters()],
            "mlp12": [int(parameter.data_ptr()) for parameter in self.mlp12.parameters()],
            "mlp21": [int(parameter.data_ptr()) for parameter in self.mlp21.parameters()],
            "mlp22": [int(parameter.data_ptr()) for parameter in self.mlp22.parameters()],
            "mlp_22_violation": [
                int(parameter.data_ptr())
                for parameter in self.mlp_22_violation.parameters()
            ],
        }

    def forward(
        self,
        props,
        node_features,
        edge_index,
        capacities,
        padded_edge_ids_per_path,
        tm1,
        tm1_pred,
        tm2,
        tm2_pred,
        tm3,
        tm3_pred,
        paths_to_edges,
        edge_ids_dict_tensor,
        original_pos_edge_ids_dict_tensor,
        path_masks=None,
    ):
        # The framework hook receives tm2_pred but not tm1_pred or padded path
        # edge IDs.  Stash only prediction/topology inputs for the duration of
        # this acyclic forward; actual TMs are deliberately never retained.
        if self._preview_tm1_pred is not None:
            raise RuntimeError("Nested Stage2-tied preview forward is unsupported")
        self._preview_tm1_pred = tm1_pred
        self._preview_padded_edge_ids = padded_edge_ids_per_path
        try:
            return super().forward(
                props,
                node_features,
                edge_index,
                capacities,
                padded_edge_ids_per_path,
                tm1,
                tm1_pred,
                tm2,
                tm2_pred,
                tm3,
                tm3_pred,
                paths_to_edges,
                edge_ids_dict_tensor,
                original_pos_edge_ids_dict_tensor,
                path_masks,
            )
        finally:
            self._preview_tm1_pred = None
            self._preview_padded_edge_ids = None

    def _current_transformer_context(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self._learned_scout_transformer_cache
        if output is None and hasattr(self, "transformer_output"):
            output = self.transformer_output
        if output is None:
            raise RuntimeError("Stage2-tied preview has no transformer output")
        if int(output.shape[0]) == 1 and int(batch_size) > 1:
            output = output.expand(batch_size, -1, -1, -1)
        if int(output.shape[0]) != int(batch_size):
            raise RuntimeError(
                f"Transformer batch {output.shape[0]} != {batch_size}"
            )
        self._learned_scout_transformer_cache = None
        return output[:, :, 0, :], output[:, :, 1:, :]

    def _replay_native_stage1(
        self,
        props,
        path_embeddings,
        path_edge_embeddings,
        tm1_pred,
        capacities,
        padded_edge_ids_per_path,
        paths_to_edges,
        pte_info,
        batch_size: int,
        paths_per_od: int,
        path_masks,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay the formal native Stage-1 construction without lookahead."""

        total_paths = int(paths_to_edges.shape[0])
        high_mask = self.path_mask_for_class(path_masks, 0)
        high_inputs = torch.cat((path_embeddings, tm1_pred), dim=-1)
        high_logits = self.forward_pass_mlp(
            high_inputs, self.mlp11, self.num_mlp1_hidden_layers
        )
        refined_high_logits = high_logits
        for _ in range(int(props.rau1)):
            current_logits = refined_high_logits
            edge_high, _, split_high = self.compute_edge_utils(
                current_logits,
                paths_to_edges,
                tm1_pred,
                capacities,
                props,
                batch_size,
                paths_per_od,
                add_epsilon=True,
                path_mask=high_mask,
            )
            mlu_high = self.compute_mlu(
                edge_high,
                batch_size,
                total_paths,
                subtract_epsilon=True,
            )
            (
                bottleneck_embedding,
                bottleneck_utilization,
                _,
            ) = self.compute_bottleneck_link_mlu_per_path(
                edge_high,
                padded_edge_ids_per_path,
                path_edge_embeddings,
                batch_size,
                total_paths,
                pte_info,
            )
            rau_inputs = torch.cat(
                (
                    bottleneck_embedding,
                    bottleneck_utilization,
                    mlu_high,
                    tm1_pred,
                    split_high,
                ),
                dim=-1,
            ).squeeze(0)
            delta = self.forward_pass_mlp(
                rau_inputs, self.mlp12, self.num_mlp2_hidden_layers
            )
            refined_high_logits = current_logits + delta

        edge_high, _, _ = self.compute_edge_utils(
            refined_high_logits,
            paths_to_edges,
            tm1_pred,
            capacities,
            props,
            batch_size,
            paths_per_od,
            add_epsilon=False,
            path_mask=high_mask,
        )
        mlu_high = self.compute_mlu(
            edge_high,
            batch_size,
            total_paths,
            subtract_epsilon=False,
        )
        return refined_high_logits, mlu_high

    def _run_tied_stage2_one_step(
        self,
        props,
        path_embeddings,
        path_edge_embeddings,
        provisional_high_logits,
        mlu11,
        tm1_pred,
        tm2_pred,
        capacities,
        padded_edge_ids_per_path,
        paths_to_edges,
        pte_info,
        batch_size: int,
        paths_per_od: int,
        path_masks,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute formal Stage-2 initialization and exactly one formal RAU."""

        total_paths = int(paths_to_edges.shape[0])
        high_mask = self.path_mask_for_class(path_masks, 0)
        medium_mask = self.path_mask_for_class(path_masks, 1)

        # Exact formal mlp21 input and High-column residual (framework lines
        # 577-582).  Concatenation avoids an in-place view mutation while
        # retaining the identical values.
        stage2_inputs = torch.cat(
            (path_embeddings, tm1_pred, tm2_pred, mlu11), dim=-1
        )
        stage2_logits = self.forward_pass_mlp(
            stage2_inputs, self.mlp21, self.num_mlp1_hidden_layers
        )
        stage2_logits = torch.cat(
            (
                stage2_logits[:, :, :1] + provisional_high_logits,
                stage2_logits[:, :, 1:],
            ),
            dim=-1,
        )
        high_logits = stage2_logits[:, :, :1]
        medium_logits = stage2_logits[:, :, 1:]

        if props.violation:
            edge_high, high_tunnels, split_high = self.compute_edge_utils(
                high_logits,
                paths_to_edges,
                tm1_pred,
                capacities,
                props,
                batch_size,
                paths_per_od,
                add_epsilon=True,
                violation=True,
                path_mask=high_mask,
            )
        else:
            edge_high, _, split_high = self.compute_edge_utils(
                high_logits,
                paths_to_edges,
                tm1_pred,
                capacities,
                props,
                batch_size,
                paths_per_od,
                add_epsilon=True,
                path_mask=high_mask,
            )
            high_tunnels = None
        edge_medium, _, split_medium = self.compute_edge_utils(
            medium_logits,
            paths_to_edges,
            tm2_pred,
            capacities,
            props,
            batch_size,
            paths_per_od,
            add_epsilon=True,
            path_mask=medium_mask,
        )
        combined_edges = edge_high + edge_medium
        mlu12 = self.compute_mlu(
            edge_high, batch_size, total_paths, subtract_epsilon=True
        )
        mlu22 = self.compute_mlu(
            combined_edges, batch_size, total_paths, subtract_epsilon=True
        )
        (
            high_bottleneck_embedding,
            high_bottleneck_utilization,
            high_bottleneck_capacity,
        ) = self.compute_bottleneck_link_mlu_per_path(
            edge_high,
            padded_edge_ids_per_path,
            path_edge_embeddings,
            batch_size,
            total_paths,
            pte_info,
            capacities,
        )
        (
            combined_bottleneck_embedding,
            combined_bottleneck_utilization,
            _,
        ) = self.compute_bottleneck_link_mlu_per_path(
            combined_edges,
            padded_edge_ids_per_path,
            path_edge_embeddings,
            batch_size,
            total_paths,
            pte_info,
        )

        # Exact formal mlp22 input order (framework lines 625-636).
        stage2_rau_inputs = torch.cat(
            (
                mlu11,
                tm1_pred,
                high_bottleneck_embedding,
                high_bottleneck_utilization,
                mlu12,
                split_high,
                tm2_pred,
                combined_bottleneck_embedding,
                combined_bottleneck_utilization,
                mlu22,
                split_medium,
            ),
            dim=-1,
        ).squeeze(0)

        if props.violation:
            violation_metric = (
                high_tunnels.unsqueeze(-1)
                / high_bottleneck_capacity.unsqueeze(-1)
            ) / mlu11.detach()
            violation_data = self.mlp_22_violation(violation_metric)
            delta = self.forward_pass_mlp(
                stage2_rau_inputs,
                self.mlp22,
                self.num_mlp2_hidden_layers,
                violation_data=[violation_data],
            )
        else:
            delta = self.forward_pass_mlp(
                stage2_rau_inputs, self.mlp22, self.num_mlp2_hidden_layers
            )
        refined_stage2_logits = stage2_logits + delta

        final_high_edges, _, _ = self.compute_edge_utils(
            refined_stage2_logits[:, :, :1],
            paths_to_edges,
            tm1_pred,
            capacities,
            props,
            batch_size,
            paths_per_od,
            add_epsilon=False,
            path_mask=high_mask,
        )
        final_medium_edges, _, final_medium_split = self.compute_edge_utils(
            refined_stage2_logits[:, :, 1:],
            paths_to_edges,
            tm2_pred,
            capacities,
            props,
            batch_size,
            paths_per_od,
            add_epsilon=False,
            path_mask=medium_mask,
        )
        return final_high_edges + final_medium_edges, final_medium_split

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
        """Return max/mean absolute and centered tied-preview features."""

        if self._preview_tm1_pred is None or self._preview_padded_edge_ids is None:
            raise RuntimeError(
                "Stage2-tied features must be computed inside the complete forward"
            )
        tm1_pred = self._preview_tm1_pred
        padded_edge_ids = self._preview_padded_edge_ids
        pte, row_indices, col_indices, pte_values = pte_info
        pte = pte.coalesce()
        pte_float = pte.to(dtype=torch.float32)
        total_paths = int(pte.shape[0])
        paths_per_od = int(num_paths_per_pair)
        if total_paths % paths_per_od != 0:
            raise RuntimeError("Path count is not divisible by paths per OD")
        if int(tm1_pred.shape[0]) != int(batch_size) or int(tm2_pred.shape[0]) != int(batch_size):
            raise RuntimeError("Prediction batch does not match batch_size")

        path_embeddings, path_edge_embeddings = self._current_transformer_context(
            batch_size
        )
        _, path_lengths = self._ensure_incidence_cache(pte)
        provisional_high_logits, mlu11 = self._replay_native_stage1(
            props,
            path_embeddings,
            path_edge_embeddings,
            tm1_pred,
            capacities,
            padded_edge_ids,
            pte,
            pte_info,
            batch_size,
            paths_per_od,
            path_masks,
        )
        combined_edges, medium_split = self._run_tied_stage2_one_step(
            props,
            path_embeddings,
            path_edge_embeddings,
            provisional_high_logits,
            mlu11,
            tm1_pred,
            tm2_pred,
            capacities,
            padded_edge_ids,
            pte,
            pte_info,
            batch_size,
            paths_per_od,
            path_masks,
        )

        row_indices = row_indices.to(device=pte.device)
        col_indices = col_indices.to(device=pte.device)
        pte_values = pte_values.to(device=pte.device, dtype=torch.float32)
        path_max, path_mean = self._path_edge_summaries(
            combined_edges.to(dtype=torch.float32).clamp_min(0.0),
            pte_float,
            row_indices,
            col_indices,
            pte_values,
            path_lengths,
        )
        high_mask = self._class_mask(path_masks, 0, total_paths, pte.device)
        max_absolute, max_centered = self._masked_center(
            torch.log1p(path_max), high_mask, paths_per_od
        )
        mean_absolute, mean_centered = self._masked_center(
            torch.log1p(path_mean), high_mask, paths_per_od
        )
        features = torch.stack(
            (max_absolute, max_centered, mean_absolute, mean_centered), dim=-1
        ).to(dtype=tm2_pred.dtype)
        if not torch.isfinite(features).all().item():
            raise RuntimeError("Stage2-tied preview produced non-finite features")

        self._last_preview_split_ratios = medium_split.detach()
        self._last_preview_features = features.detach()
        self._last_preview_audit = {
            "feature_count": FEATURE_COUNT,
            "provisional_native_high_rau_steps": int(props.rau1),
            "tied_formal_stage2_rau_steps": 1,
            "formal_mlp21_object_id": id(self.mlp21),
            "formal_mlp22_object_id": id(self.mlp22),
            "independent_scout_parameter_count": 0,
            "strict_policy_inputs": [
                "tm1_pred",
                "tm2_pred",
                "transformer_topology_embeddings",
                "paths_to_edges",
                "capacities",
                "path_masks",
            ],
        }
        return features


full.Hattrick = Stage2TiedPreviewHattrick
full.OUTPUT_ROOT = THIS_DIR / "artifacts"


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "shared2x_learned_medium_scout/run_experiment.py": (
            LEARNED_DIR / "run_experiment.py"
        ),
        "ordered_projection.py": FULL_DIR / "ordered_projection.py",
        "shared2x_full_objectives/run_experiment.py": FULL_DIR / "run_experiment.py",
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "utils/training_utils.py": ROOT / "utils" / "training_utils.py",
    }
    hashes = {name: full.sha256(path) for name, path in paths.items()}
    settings = json.dumps(
        {
            "preview_high": "full native Stage1 without lookahead residual",
            "preview_medium": "formal mlp21 plus one exact formal mlp22 RAU",
            "independent_scout_heads": 0,
            "features": [
                "combined_max_absolute",
                "combined_max_centered",
                "combined_mean_absolute",
                "combined_mean_centered",
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["stage2_tied_preview/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


full.source_hashes = source_hashes


def method_description() -> dict:
    return {
        "method": "Hattrick formal-Stage2-tied one-step Medium preview",
        "architecture": (
            "A native no-lookahead Stage1 replay feeds the existing formal mlp21 "
            "and one exact formal mlp22 bottleneck RAU before actual High. Four "
            "preview features enter zero-output High adapters; the inherited "
            "formal serial cascade then runs unchanged."
        ),
        "parameter_sharing": (
            "Preview invokes the same mlp11/mlp12/mlp21/mlp22 and violation "
            "parameter objects as the formal cascade; no independent route head."
        ),
        "feature_order": [
            "combined_max_absolute",
            "combined_max_within_high_od_centered",
            "combined_mean_absolute",
            "combined_mean_within_high_od_centered",
        ],
        "projection": "unchanged six-objective ordered gradient projection",
        "objectives": list(full.OBJECTIVE_NAMES),
        "inference_information": (
            "ESM tm1_pred/tm2_pred, topology transformer embeddings, PTE, "
            "capacities, padded path-edge IDs, and masks only; actual traffic "
            "is not retained or read by the preview"
        ),
        "output_root": str(full.OUTPUT_ROOT),
        "source_sha256": source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared-2x formal Stage2-tied one-step Medium preview"
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
    full.write_json(run_dir / "stage2_tied_preview_method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
