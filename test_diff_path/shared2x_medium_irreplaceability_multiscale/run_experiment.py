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
SINGLE_DIR = TEST_DIR / "shared2x_medium_irreplaceability"
for item in (str(ROOT), str(TEST_DIR), str(FULL_DIR), str(SINGLE_DIR)):
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


single = load_module(
    "shared2x_medium_irreplaceability_multiscale_single_runtime",
    SINGLE_DIR / "run_experiment.py",
)
full = single.full
NativeHattrick = single.NativeHattrick


class MultiscaleMediumIrreplaceabilityHattrick(
    single.MediumIrreplaceabilityHattrick
):
    """Serial Hattrick with cached r^1/r^2/r^3 Medium path coverage."""

    FEATURE_POWERS = (1, 2, 3)

    def __init__(self, props):
        # The parent constructs the native two-feature zero-init adapters and
        # static coverage buffers. Replace only those adapters with the exact
        # same design at six input features; the native cascade is untouched.
        super().__init__(props)
        self._replace_medium_pressure_adapters(feature_count=6)
        self.register_buffer(
            "_medium_multiscale_power_cache", torch.empty(0), persistent=False
        )
        self._medium_multiscale_source_build = -1
        self._medium_multiscale_cache_builds = 0
        self._last_multiscale_operator_audit: dict[str, int] = {}

    def _replace_medium_pressure_adapters(self, feature_count: int) -> None:
        reference = self.mlp11[0]
        device = reference.weight.device
        dtype = reference.weight.dtype
        hidden_dim = max(16, int(self.input_dim) * 2)

        def residual_adapter(base_width: int) -> nn.Sequential:
            adapter = nn.Sequential(
                nn.Linear(base_width + feature_count, hidden_dim),
                nn.LeakyReLU(negative_slope=0.02),
                nn.Linear(hidden_dim, 1),
            ).to(device=device, dtype=dtype)
            nn.init.kaiming_normal_(
                adapter[0].weight, nonlinearity="leaky_relu"
            )
            nn.init.zeros_(adapter[0].bias)
            nn.init.zeros_(adapter[2].weight)
            nn.init.zeros_(adapter[2].bias)
            return adapter

        self.medium_pressure_init_adapter = residual_adapter(self.mlp_11_dim)
        self.medium_pressure_rau_adapter = residual_adapter(self.mlp_12_dim)
        self.medium_pressure_feature_count = feature_count

    @property
    def medium_multiscale_cache_builds(self) -> int:
        return int(self._medium_multiscale_cache_builds)

    @property
    def last_multiscale_operator_audit(self) -> dict[str, int]:
        return dict(self._last_multiscale_operator_audit)

    def _ensure_multiscale_power_cache(
        self,
        paths_to_edges,
        path_masks,
        num_paths_per_pair: int,
    ) -> torch.Tensor:
        coverage = super()._ensure_medium_coverage_cache(
            paths_to_edges, path_masks, num_paths_per_pair
        )
        source_build = self.medium_coverage_cache_builds
        cache_valid = (
            self._medium_multiscale_source_build == source_build
            and tuple(self._medium_multiscale_power_cache.shape)
            == (len(self.FEATURE_POWERS), *tuple(coverage.shape))
            and self._medium_multiscale_power_cache.device == coverage.device
        )
        if cache_valid:
            return self._medium_multiscale_power_cache

        self._medium_multiscale_power_cache = torch.stack(
            tuple(coverage.pow(power) for power in self.FEATURE_POWERS), dim=0
        )
        # The reused single-scale helper creates an r^gamma cache. It is not
        # needed here, so release it after the r^1/r^2/r^3 cache is ready.
        self._medium_edge_weight_cache = torch.empty(
            0, device=coverage.device, dtype=coverage.dtype
        )
        self._medium_multiscale_source_build = source_build
        self._medium_multiscale_cache_builds += 1
        return self._medium_multiscale_power_cache

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
        """Return [absolute, centered] path value for gamma 1, 2 and 3."""

        del pte_info, props
        pte = paths_to_edges.coalesce()
        total_paths, num_edges = (int(pte.shape[0]), int(pte.shape[1]))
        if total_paths % int(num_paths_per_pair) != 0:
            raise RuntimeError("Path count is not divisible by paths per OD")
        num_ods = total_paths // int(num_paths_per_pair)
        if int(batch_size) != int(tm2_pred.shape[0]):
            raise RuntimeError("Batch-size argument does not match Medium prediction")

        powers = self._ensure_multiscale_power_cache(
            pte, path_masks, int(num_paths_per_pair)
        )
        demand = (
            tm2_pred.squeeze(-1)
            .reshape(batch_size, num_ods, num_paths_per_pair)
            .to(dtype=torch.float32)
            .mean(dim=-1)
        )
        cap = capacities.to(device=demand.device, dtype=torch.float32)
        if int(cap.shape[-1]) != num_edges:
            raise RuntimeError(
                f"Capacity width {cap.shape[-1]} does not match {num_edges} edges"
            )
        if int(cap.shape[0]) not in (1, int(batch_size)):
            raise RuntimeError("Capacities must have batch width one or match demand")

        high_mask = self._high_feasible_mask(path_masks, total_paths, pte.device)
        high_mask_by_od = high_mask.reshape(1, num_ods, num_paths_per_pair)
        mask_float = high_mask_by_od.to(dtype=torch.float32)
        feasible_counts = mask_float.sum(dim=-1, keepdim=True)
        if bool((feasible_counts == 0).any().item()):
            raise RuntimeError("At least one High OD has no feasible candidate path")

        pte_float = pte.to(dtype=torch.float32)
        features = []
        for coverage_power in powers.unbind(dim=0):
            edge_value = (demand @ coverage_power) / cap.clamp_min(1e-8)
            path_value = torch.sparse.mm(pte_float, edge_value.t()).t()
            absolute = torch.log1p(path_value.clamp_min(0.0)).reshape(
                batch_size, num_ods, num_paths_per_pair
            )
            od_mean = (absolute * mask_float).sum(
                dim=-1, keepdim=True
            ) / feasible_counts
            centered = (absolute - od_mean) * mask_float
            absolute = absolute * mask_float
            features.extend(
                (
                    absolute.reshape(batch_size, total_paths),
                    centered.reshape(batch_size, total_paths),
                )
            )

        self._last_multiscale_operator_audit = {
            "dense_demand_by_coverage_matmul": len(self.FEATURE_POWERS),
            "sparse_path_by_edge_matmul": len(self.FEATURE_POWERS),
            "feature_columns": len(features),
        }
        return torch.stack(features, dim=-1).to(dtype=tm2_pred.dtype)


full.Hattrick = MultiscaleMediumIrreplaceabilityHattrick
full.OUTPUT_ROOT = THIS_DIR / "artifacts"


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "shared2x_medium_irreplaceability/run_experiment.py": (
            SINGLE_DIR / "run_experiment.py"
        ),
        "ordered_projection.py": FULL_DIR / "ordered_projection.py",
        "shared2x_full_objectives/run_experiment.py": FULL_DIR / "run_experiment.py",
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "utils/training_utils.py": ROOT / "utils" / "training_utils.py",
    }
    hashes = {name: full.sha256(path) for name, path in paths.items()}
    settings = json.dumps(
        {"coverage_powers": list(MultiscaleMediumIrreplaceabilityHattrick.FEATURE_POWERS),
         "features_per_power": ["absolute_log1p", "centered_only"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["medium_irreplaceability_multiscale/settings"] = hashlib.sha256(
        settings
    ).hexdigest()
    return hashes


full.source_hashes = source_hashes


def method_description() -> dict:
    return {
        "method": "Hattrick multiscale analytic Medium-irreplaceability scout",
        "architecture": (
            "Native High -> High+Medium -> High+Medium+Low serial cascade with "
            "six-feature zero-initialized residual adapters at High init and RAU."
        ),
        "coverage_powers": list(
            MultiscaleMediumIrreplaceabilityHattrick.FEATURE_POWERS
        ),
        "feature_order": [
            "gamma1_absolute", "gamma1_centered",
            "gamma2_absolute", "gamma2_centered",
            "gamma3_absolute", "gamma3_centered",
        ],
        "projection": "unchanged six-objective ordered gradient projection",
        "objectives": list(full.OBJECTIVE_NAMES),
        "inference_information": (
            "ESM-predicted Medium traffic, static candidate paths and masks, topology, "
            "and capacities only; current actual traffic is never a policy input"
        ),
        "output_root": str(full.OUTPUT_ROOT),
        "source_sha256": source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared-2x six-objective Hattrick multiscale Medium scout"
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
        run_dir / "medium_irreplaceability_multiscale_method.json", description
    )
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
