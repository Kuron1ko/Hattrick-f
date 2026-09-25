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
    "shared2x_medium_irreplaceability_full_runtime",
    FULL_DIR / "run_experiment.py",
)
NativeHattrick = full.Hattrick


ACTIVE_GAMMA = 2.0


def validate_gamma(gamma: float) -> float:
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive")
    if gamma > 16.0:
        raise ValueError("gamma above 16 is disabled to avoid numerical underflow")
    return gamma


def number_label(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def configure_gamma(gamma: float = 2.0) -> float:
    """Select a gamma-specific, isolated experiment output tree."""

    global ACTIVE_GAMMA
    ACTIVE_GAMMA = validate_gamma(gamma)
    full.OUTPUT_ROOT = THIS_DIR / "artifacts" / f"gamma_{number_label(ACTIVE_GAMMA)}"
    return ACTIVE_GAMMA


class MediumIrreplaceabilityHattrick(NativeHattrick):
    """Native serial Hattrick plus an analytic Medium-irreplaceability preview."""

    def __init__(self, props):
        props.future_lookahead = False
        super().__init__(props)
        self.medium_irreplaceability_gamma = validate_gamma(ACTIVE_GAMMA)
        # PTE is supplied to forward rather than construction, so coverage is
        # built lazily once.  Non-persistent buffers follow model.to(device)
        # without changing checkpoint compatibility.
        self.register_buffer(
            "_medium_edge_coverage_cache", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_medium_edge_weight_cache", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_medium_feasible_mask_cache",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self._medium_coverage_shape: tuple[int, int, int] | None = None
        self._medium_coverage_cache_builds = 0
        self.enable_medium_pressure_lookahead()

    @property
    def medium_coverage_cache_builds(self) -> int:
        return int(self._medium_coverage_cache_builds)

    def _medium_feasible_mask(self, path_masks, total_paths: int, device):
        mask = self.path_mask_for_class(path_masks, 1)
        if mask is None:
            return torch.ones(total_paths, dtype=torch.bool, device=device)
        mask = mask.reshape(-1).to(device=device, dtype=torch.bool)
        if int(mask.numel()) != total_paths:
            raise RuntimeError(
                f"Medium path-mask width {mask.numel()} does not match {total_paths} paths"
            )
        return mask

    def _high_feasible_mask(self, path_masks, total_paths: int, device):
        mask = self.path_mask_for_class(path_masks, 0)
        if mask is None:
            return torch.ones(total_paths, dtype=torch.bool, device=device)
        mask = mask.reshape(-1).to(device=device, dtype=torch.bool)
        if int(mask.numel()) != total_paths:
            raise RuntimeError(
                f"High path-mask width {mask.numel()} does not match {total_paths} paths"
            )
        return mask

    def _ensure_medium_coverage_cache(
        self,
        paths_to_edges,
        path_masks,
        num_paths_per_pair: int,
    ) -> torch.Tensor:
        """Cache r[o,e], the feasible Medium path coverage of every edge."""

        pte = paths_to_edges.coalesce()
        total_paths, num_edges = (int(pte.shape[0]), int(pte.shape[1]))
        if total_paths % int(num_paths_per_pair) != 0:
            raise RuntimeError("Path count is not divisible by paths per OD")
        num_ods = total_paths // int(num_paths_per_pair)
        medium_mask = self._medium_feasible_mask(
            path_masks, total_paths, pte.device
        )
        expected_shape = (num_ods, num_edges, int(num_paths_per_pair))
        cache_valid = (
            self._medium_coverage_shape == expected_shape
            and tuple(self._medium_edge_coverage_cache.shape) == (num_ods, num_edges)
            and self._medium_edge_coverage_cache.device == pte.device
            and self._medium_feasible_mask_cache.device == pte.device
            and torch.equal(self._medium_feasible_mask_cache, medium_mask)
        )
        if cache_valid:
            return self._medium_edge_coverage_cache

        # This static one-time construction traverses only sparse PTE entries.
        # Building on CPU avoids nondeterministic CUDA indexed accumulation when
        # the inherited runner enables deterministic algorithms.
        mask_cpu = medium_mask.detach().to(device="cpu")
        feasible_counts = mask_cpu.reshape(num_ods, num_paths_per_pair).sum(dim=1)
        if bool((feasible_counts == 0).any().item()):
            bad_od = int((feasible_counts == 0).nonzero(as_tuple=False)[0].item())
            raise RuntimeError(f"Medium OD {bad_od} has no feasible candidate path")

        indices = pte.indices().detach().to(device="cpu")
        values = pte.values().detach().to(device="cpu", dtype=torch.float32)
        rows, cols = indices[0], indices[1]
        keep = mask_cpu[rows]
        rows = rows[keep]
        cols = cols[keep]
        values = values[keep]
        od_indices = torch.div(rows, num_paths_per_pair, rounding_mode="floor")
        coverage_cpu = torch.zeros((num_ods, num_edges), dtype=torch.float32)
        coverage_cpu.index_put_((od_indices, cols), values, accumulate=True)
        coverage_cpu = coverage_cpu / feasible_counts.to(torch.float32).unsqueeze(1)
        coverage = coverage_cpu.to(device=pte.device)

        self._medium_edge_coverage_cache = coverage
        self._medium_edge_weight_cache = coverage.pow(
            self.medium_irreplaceability_gamma
        )
        self._medium_feasible_mask_cache = medium_mask.detach().clone()
        self._medium_coverage_shape = expected_shape
        self._medium_coverage_cache_builds += 1
        return self._medium_edge_coverage_cache

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
        """Map analytic Medium edge value to absolute and relative path features.

        ``r[o,e]`` is the fraction of feasible paths of Medium OD ``o`` that
        contain edge ``e``.  With ESM demand ``d``, edge value is
        ``(d @ r**gamma) / capacity``.  The first returned feature is the log1p
        sum of these values along a High path.  The second is its centered
        relative deviation within the same High OD.
        """

        del pte_info  # The analytic scout only needs the coalesced sparse PTE.
        pte = paths_to_edges.coalesce()
        total_paths, num_edges = (int(pte.shape[0]), int(pte.shape[1]))
        if total_paths % int(num_paths_per_pair) != 0:
            raise RuntimeError("Path count is not divisible by paths per OD")
        num_ods = total_paths // int(num_paths_per_pair)
        if int(batch_size) != int(tm2_pred.shape[0]):
            raise RuntimeError("Batch-size argument does not match Medium prediction")

        self._ensure_medium_coverage_cache(
            pte, path_masks, int(num_paths_per_pair)
        )
        demand = (
            tm2_pred.squeeze(-1)
            .reshape(batch_size, num_ods, num_paths_per_pair)
            .to(dtype=torch.float32)
            .mean(dim=-1)
        )
        edge_value = demand @ self._medium_edge_weight_cache

        cap = capacities.to(device=edge_value.device, dtype=torch.float32)
        if int(cap.shape[-1]) != num_edges:
            raise RuntimeError(
                f"Capacity width {cap.shape[-1]} does not match {num_edges} edges"
            )
        if int(cap.shape[0]) not in (1, int(batch_size)):
            raise RuntimeError("Capacities must have batch width one or match demand")
        edge_value = edge_value / cap.clamp_min(1e-8)

        pte_float = pte.to(dtype=torch.float32)
        path_value = torch.sparse.mm(pte_float, edge_value.t()).t()
        absolute = torch.log1p(path_value.clamp_min(0.0))

        high_mask = self._high_feasible_mask(path_masks, total_paths, pte.device)
        high_mask_by_od = high_mask.reshape(1, num_ods, num_paths_per_pair)
        absolute_by_od = absolute.reshape(batch_size, num_ods, num_paths_per_pair)
        mask_float = high_mask_by_od.to(dtype=absolute_by_od.dtype)
        feasible_counts = mask_float.sum(dim=-1, keepdim=True)
        if bool((feasible_counts == 0).any().item()):
            raise RuntimeError("At least one High OD has no feasible candidate path")
        od_mean = (absolute_by_od * mask_float).sum(
            dim=-1, keepdim=True
        ) / feasible_counts
        # The absolute feature already carries the global value scale.  The
        # second feature only needs to rank alternatives within one High OD;
        # keeping it centered (rather than dividing by a possibly small OD
        # mean) avoids an unnecessary heavy tail during early training.
        relative = absolute_by_od - od_mean
        absolute_by_od = absolute_by_od * mask_float
        relative = relative * mask_float

        features = torch.stack(
            (
                absolute_by_od.reshape(batch_size, total_paths),
                relative.reshape(batch_size, total_paths),
            ),
            dim=-1,
        )
        return features.to(dtype=tm2_pred.dtype)


full.Hattrick = MediumIrreplaceabilityHattrick


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
        {"gamma": ACTIVE_GAMMA}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    hashes["medium_irreplaceability/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


full.source_hashes = source_hashes
configure_gamma()


def method_description() -> dict:
    return {
        "method": "Hattrick analytic Medium-irreplaceability scout",
        "architecture": (
            "Native High -> High+Medium -> High+Medium+Low cascade with the native "
            "zero-initialized Medium-pressure adapters. Static feasible-path edge "
            "coverage supplies analytic absolute and within-OD relative path value."
        ),
        "gamma": ACTIVE_GAMMA,
        "coverage": "r[o,e] = feasible Medium paths containing e / feasible paths of OD o",
        "edge_value": "(ESM Medium OD demand @ r**gamma) / edge capacity",
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
        description="Shared-2x six-objective Hattrick Medium-irreplaceability experiment"
    )
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configure_gamma(args.gamma)
    run_dir = full.run_one(args.level, args.seed, force=args.force)
    description = method_description()
    description.update(
        {"level": args.level, "seed": args.seed, "run_directory": str(run_dir)}
    )
    full.write_json(run_dir / "medium_irreplaceability_method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
