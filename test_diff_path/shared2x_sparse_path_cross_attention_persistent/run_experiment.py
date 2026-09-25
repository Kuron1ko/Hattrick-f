from __future__ import annotations

"""Persist Medium OD/path context into the *formal* Stage-2 High updates.

The parent sparse-attention model exposes detailed Medium conflict information
to Stage 1.  Native Hattrick subsequently updates both High and Medium inside
Stage 2, but those final High updates only receive the native aggregated RAU
features.  This variant feeds the already-computed sparse context into the High
component of mlp21/mlp22 as zero-output residuals.  It does not preview Stage 2,
change the serial order, or add a training objective.
"""

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
PARENT_DIR = TEST_DIR / "shared2x_sparse_path_cross_attention"
PARENT_RUNNER = PARENT_DIR / "run_experiment.py"
for item in (str(ROOT), str(TEST_DIR), str(PARENT_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


parent = load_module(
    "shared2x_sparse_path_cross_attention_persistent_parent", PARENT_RUNNER
)


class PersistentStage2SparseAttentionHattrick(
    parent.SparsePathCrossAttentionHattrick
):
    """Carry the same Medium path message through final Stage-2 High updates."""

    def __init__(self, props):
        super().__init__(props)
        self.stage2_high_init_adapter = parent.ZeroOutputAdapter(
            self.mlp_21_dim + parent.TOTAL_FEATURES
        )
        self.stage2_high_rau_adapter = parent.ZeroOutputAdapter(
            self.mlp_22_dim + parent.TOTAL_FEATURES
        )
        self._spc_features_live = None
        self._last_stage2_context_audit: dict[str, object] = {}

    @property
    def last_stage2_context_audit(self) -> dict[str, object]:
        return dict(self._last_stage2_context_audit)

    def compute_medium_pressure_features(self, *args, **kwargs):
        features = super().compute_medium_pressure_features(*args, **kwargs)
        # Keep the live graph for the actual Stage-2 call.  The parent already
        # stores a detached diagnostic copy separately.
        self._spc_features_live = features
        return features

    @staticmethod
    def _align_context(inputs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2 and context.ndim == 3 and context.shape[0] == 1:
            context = context.squeeze(0)
        if inputs.shape[:-1] != context.shape[:-1]:
            raise RuntimeError(
                "Stage-2 sparse-context prefix mismatch: "
                f"inputs={tuple(inputs.shape)}, context={tuple(context.shape)}"
            )
        return context.to(device=inputs.device, dtype=inputs.dtype)

    def forward_pass_mlp(self, inputs, mlp, num_hidden_layers, violation_data=None):
        output = super().forward_pass_mlp(
            inputs, mlp, num_hidden_layers, violation_data=violation_data
        )
        adapter = None
        site = None
        if mlp is self.mlp21:
            adapter = self.stage2_high_init_adapter
            site = "mlp21"
        elif mlp is self.mlp22:
            adapter = self.stage2_high_rau_adapter
            site = "mlp22"
        if adapter is None:
            return output

        context = self._spc_features_live
        if context is None:
            raise RuntimeError(f"{site} ran before sparse Medium context was computed")
        context = self._align_context(inputs, context)
        high_residual = adapter(torch.cat((inputs, context), dim=-1))
        if output.shape[-1] != 2 or high_residual.shape != output[..., :1].shape:
            raise RuntimeError(
                f"Unexpected {site} output/residual shapes: "
                f"{tuple(output.shape)} / {tuple(high_residual.shape)}"
            )
        output = torch.cat(
            (output[..., :1] + high_residual, output[..., 1:]), dim=-1
        )
        self._last_stage2_context_audit = {
            "site": site,
            "input_shape": list(inputs.shape),
            "context_shape": list(context.shape),
            "residual_shape": list(high_residual.shape),
            "only_high_channel_modified": True,
            "formal_stage2_call": True,
        }
        return output


parent.full.Hattrick = PersistentStage2SparseAttentionHattrick
parent.full.OUTPUT_ROOT = THIS_DIR / "artifacts"


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "parent_sparse_runner.py": PARENT_RUNNER,
        "ordered_projection.py": (
            TEST_DIR / "shared2x_full_objectives" / "ordered_projection.py"
        ),
        "shared2x_full_objectives/run_experiment.py": (
            TEST_DIR / "shared2x_full_objectives" / "run_experiment.py"
        ),
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
    }
    hashes = {name: parent.full.sha256(path) for name, path in paths.items()}
    settings = json.dumps(
        {
            "parent_attention_dim": parent.ATTENTION_DIM,
            "parent_neighbor_ods": parent.NEIGHBOR_ODS,
            "stage2_injection": ["mlp21.high", "mlp22.high"],
            "zero_output_adapters": True,
            "new_objectives": 0,
            "stage2_preview": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["persistent_stage2/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


parent.full.source_hashes = source_hashes


def method_description() -> dict:
    return {
        "method": "persistent sparse Medium context for formal Stage-2 High",
        "architecture": (
            "The parent 18-dimensional sparse Medium OD/path message still feeds "
            "Stage-1 High, and now also feeds zero-output residuals on the High "
            "channel of formal Stage-2 mlp21 and every mlp22 RAU update."
        ),
        "formal_cascade": "unchanged High -> High+Medium -> High+Medium+Low",
        "formal_projection": "unchanged six-objective ordered gradient projection",
        "stage2_preview": False,
        "extra_objectives": [],
        "medium_low_outputs": "native formal Stage-2/Stage-3 outputs",
        "strict_inference_inputs": (
            "ESM predictions, topology/path embeddings, capacities, PTE and masks only"
        ),
        "zero_initialization": (
            "Both new Stage-2 High residual output layers start at exact zero; "
            "the parent policy is therefore preserved before training."
        ),
        "source_sha256": source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent sparse Medium context in formal Stage-2 High"
    )
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = parent.full.run_one(args.level, args.seed, force=args.force)
    description = method_description()
    description.update(
        {"level": args.level, "seed": args.seed, "run_directory": str(run_dir)}
    )
    parent.full.write_json(run_dir / "method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
