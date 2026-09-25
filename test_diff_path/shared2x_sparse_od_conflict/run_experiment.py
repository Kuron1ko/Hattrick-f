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
    "shared2x_sparse_od_conflict_full_runtime",
    FULL_DIR / "run_experiment.py",
)
NativeHattrick = full.Hattrick


ALLOWED_BETAS = (2.0, 4.0, 8.0)
ACTIVE_BETA = 4.0


def validate_beta(beta: float) -> float:
    beta = float(beta)
    if not math.isfinite(beta) or beta not in ALLOWED_BETAS:
        choices = ", ".join(format(value, "g") for value in ALLOWED_BETAS)
        raise ValueError(f"beta must be one of {{{choices}}}")
    return beta


def number_label(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def configure_beta(beta: float = 4.0) -> float:
    """Select a beta-specific output tree before constructing the model."""

    global ACTIVE_BETA
    ACTIVE_BETA = validate_beta(beta)
    full.OUTPUT_ROOT = THIS_DIR / "artifacts" / f"beta_{number_label(ACTIVE_BETA)}"
    return ACTIVE_BETA


class SparseODConflictHattrick(NativeHattrick):
    """Serial Hattrick with an OD-specific Medium best-response preview.

    The static conflict matrix keeps the Medium OD identity until after a
    smooth minimum over that OD's feasible routes.  Only ESM-predicted Medium
    demand is used online.  The native Medium and Low stages are unchanged.
    """

    def __init__(self, props):
        props.future_lookahead = False
        super().__init__(props)
        self.medium_od_conflict_beta = validate_beta(ACTIVE_BETA)
        # PTE, masks, and capacities arrive at forward time.  These
        # non-persistent buffers therefore materialize lazily and do not alter
        # native checkpoint compatibility.
        self.register_buffer(
            "_medium_od_conflict_cache", torch.empty(0), persistent=False
        )
        self.register_buffer(
            "_medium_feasible_mask_cache",
            torch.empty(0, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "_conflict_capacity_cache", torch.empty(0), persistent=False
        )
        self._medium_od_conflict_shape: tuple[int, int, int, int] | None = None
        self._medium_od_conflict_cache_builds = 0
        self._last_conflict_operator_audit: dict[str, int] = {}
        self.enable_medium_pressure_lookahead()

    @property
    def medium_od_conflict_cache_builds(self) -> int:
        return int(self._medium_od_conflict_cache_builds)

    @property
    def last_conflict_operator_audit(self) -> dict[str, int]:
        return dict(self._last_conflict_operator_audit)

    def _class_feasible_mask(
        self,
        path_masks,
        class_index: int,
        total_paths: int,
        device,
        class_name: str,
    ) -> torch.Tensor:
        mask = self.path_mask_for_class(path_masks, class_index)
        if mask is None:
            return torch.ones(total_paths, dtype=torch.bool, device=device)
        mask = mask.reshape(-1).to(device=device, dtype=torch.bool)
        if int(mask.numel()) != total_paths:
            raise RuntimeError(
                f"{class_name} path-mask width {mask.numel()} does not match "
                f"{total_paths} paths"
            )
        return mask

    def _medium_feasible_mask(self, path_masks, total_paths: int, device):
        return self._class_feasible_mask(
            path_masks, 1, total_paths, device, "Medium"
        )

    def _high_feasible_mask(self, path_masks, total_paths: int, device):
        return self._class_feasible_mask(
            path_masks, 0, total_paths, device, "High"
        )

    @staticmethod
    def _capacity_reference(capacities, num_edges: int, device) -> torch.Tensor:
        cap = capacities.detach().to(device=device, dtype=torch.float32)
        if cap.ndim == 1:
            cap = cap.reshape(1, -1)
        if cap.ndim != 2 or int(cap.shape[1]) != num_edges:
            raise RuntimeError(
                f"Capacities must have shape [B,{num_edges}], got {tuple(cap.shape)}"
            )
        reference = cap[0]
        if int(cap.shape[0]) > 1 and not torch.equal(
            cap, reference.reshape(1, -1).expand_as(cap)
        ):
            raise RuntimeError(
                "Sparse OD-conflict cache requires static capacities within a batch"
            )
        if not torch.isfinite(reference).all().item():
            raise RuntimeError("Capacities contain a non-finite value")
        if bool((reference <= 0).any().item()):
            raise RuntimeError("Capacities must be strictly positive")
        return reference

    def _ensure_medium_od_conflict_cache(
        self,
        paths_to_edges,
        capacities,
        path_masks,
        num_paths_per_pair: int,
    ) -> torch.Tensor:
        """Cache dense C[p,o] with a smooth Medium best-route response.

        For High candidate path ``p`` and Medium candidate path ``q``, the
        overlap is the fraction of ``q``'s inverse-capacity footprint shared
        with ``p``.  ``C[p,o]`` is the normalized smooth minimum of this
        overlap over feasible paths of Medium OD ``o``.  The mean inside the
        log makes one surviving route more costly than many surviving routes.
        """

        pte = paths_to_edges.coalesce()
        total_paths, num_edges = (int(pte.shape[0]), int(pte.shape[1]))
        paths_per_od = int(num_paths_per_pair)
        if paths_per_od <= 0 or total_paths % paths_per_od != 0:
            raise RuntimeError("Path count is not divisible by paths per OD")
        num_ods = total_paths // paths_per_od
        medium_mask = self._medium_feasible_mask(
            path_masks, total_paths, pte.device
        )
        feasible_counts = medium_mask.reshape(num_ods, paths_per_od).sum(dim=1)
        if bool((feasible_counts == 0).any().item()):
            bad_od = int((feasible_counts == 0).nonzero(as_tuple=False)[0].item())
            raise RuntimeError(f"Medium OD {bad_od} has no feasible candidate path")

        capacity_reference = self._capacity_reference(
            capacities, num_edges, pte.device
        )
        expected_shape = (total_paths, num_ods, paths_per_od, num_edges)
        cache_valid = (
            self._medium_od_conflict_shape == expected_shape
            and tuple(self._medium_od_conflict_cache.shape)
            == (total_paths, num_ods)
            and self._medium_od_conflict_cache.device == pte.device
            and self._medium_feasible_mask_cache.device == pte.device
            and self._conflict_capacity_cache.device == pte.device
            and torch.equal(self._medium_feasible_mask_cache, medium_mask)
            and torch.equal(self._conflict_capacity_cache, capacity_reference)
        )
        if cache_valid:
            return self._medium_od_conflict_cache

        # Build once on CPU.  GEANT has 3696 paths and 72 edges, so the only
        # large temporary is the 3696x3696 pairwise overlap (about 52 MiB).
        # In-place normalization/exponentiation avoids additional copies and
        # keeps deterministic-algorithm mode independent of CUDA atomics.
        indices = pte.indices().detach().to(device="cpu")
        incidence = torch.zeros((total_paths, num_edges), dtype=torch.float32)
        incidence[indices[0], indices[1]] = 1.0
        inv_capacity = capacity_reference.detach().to(device="cpu").reciprocal()
        weighted_medium_paths = incidence * inv_capacity.reshape(1, num_edges)
        medium_path_footprint = weighted_medium_paths.sum(dim=1).clamp_min(1e-12)

        overlap = incidence @ weighted_medium_paths.t()
        overlap.div_(medium_path_footprint.reshape(1, total_paths))
        overlap.clamp_(min=0.0, max=1.0)
        overlap.mul_(-self.medium_od_conflict_beta).exp_()
        overlap = overlap.reshape(total_paths, num_ods, paths_per_od)

        medium_mask_cpu = medium_mask.detach().to(device="cpu").reshape(
            num_ods, paths_per_od
        )
        overlap.mul_(medium_mask_cpu.reshape(1, num_ods, paths_per_od))
        mean_survival = overlap.sum(dim=-1) / feasible_counts.detach().to(
            device="cpu", dtype=torch.float32
        ).reshape(1, num_ods)
        conflict_cpu = -torch.log(mean_survival.clamp_min(1e-30))
        conflict_cpu.div_(self.medium_od_conflict_beta).clamp_(min=0.0, max=1.0)

        self._medium_od_conflict_cache = conflict_cpu.to(device=pte.device)
        self._medium_feasible_mask_cache = medium_mask.detach().clone()
        self._conflict_capacity_cache = capacity_reference.detach().clone()
        self._medium_od_conflict_shape = expected_shape
        self._medium_od_conflict_cache_builds += 1
        return self._medium_od_conflict_cache

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
        """Return absolute and within-High-OD centered conflict features."""

        del pte_info, props
        pte = paths_to_edges.coalesce()
        total_paths = int(pte.shape[0])
        paths_per_od = int(num_paths_per_pair)
        if paths_per_od <= 0 or total_paths % paths_per_od != 0:
            raise RuntimeError("Path count is not divisible by paths per OD")
        num_ods = total_paths // paths_per_od
        if int(batch_size) != int(tm2_pred.shape[0]):
            raise RuntimeError("Batch-size argument does not match Medium prediction")

        conflict = self._ensure_medium_od_conflict_cache(
            pte, capacities, path_masks, paths_per_od
        )
        demand = (
            tm2_pred.squeeze(-1)
            .reshape(batch_size, num_ods, paths_per_od)
            .to(dtype=torch.float32)
            .mean(dim=-1)
        )
        path_harm = demand @ conflict.t()
        absolute = torch.log1p(path_harm.clamp_min(0.0))

        high_mask = self._high_feasible_mask(path_masks, total_paths, pte.device)
        high_mask_by_od = high_mask.reshape(1, num_ods, paths_per_od)
        mask_float = high_mask_by_od.to(dtype=absolute.dtype)
        feasible_counts = mask_float.sum(dim=-1, keepdim=True)
        if bool((feasible_counts == 0).any().item()):
            raise RuntimeError("At least one High OD has no feasible candidate path")
        absolute_by_od = absolute.reshape(batch_size, num_ods, paths_per_od)
        od_mean = (absolute_by_od * mask_float).sum(
            dim=-1, keepdim=True
        ) / feasible_counts
        centered = (absolute_by_od - od_mean) * mask_float
        absolute_by_od = absolute_by_od * mask_float

        self._last_conflict_operator_audit = {
            "dense_medium_demand_by_od_conflict_matmul": 1,
            "cached_high_paths": total_paths,
            "cached_medium_ods": num_ods,
            "feature_columns": 2,
        }
        return torch.stack(
            (
                absolute_by_od.reshape(batch_size, total_paths),
                centered.reshape(batch_size, total_paths),
            ),
            dim=-1,
        ).to(dtype=tm2_pred.dtype)


full.Hattrick = SparseODConflictHattrick


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
            "beta": ACTIVE_BETA,
            "conflict": "smooth-min normalized shared inverse-capacity overlap",
            "features": ["absolute_log1p", "high_od_centered"],
            "sparsification": None,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hashes["sparse_od_conflict/settings"] = hashlib.sha256(settings).hexdigest()
    return hashes


full.source_hashes = source_hashes
configure_beta()


def method_description() -> dict:
    return {
        "method": "Hattrick sparse-OD-conflict scout (dense GEANT cache MVP)",
        "architecture": (
            "Native High -> High+Medium -> High+Medium+Low serial cascade with "
            "the native two-feature zero-initialized High init/RAU adapters."
        ),
        "beta": ACTIVE_BETA,
        "conflict_matrix": "C[p,o] with shape [3696,462] on GEANT shared-8sp",
        "path_overlap": (
            "shared inverse-capacity footprint divided by the Medium path's "
            "inverse-capacity footprint"
        ),
        "medium_best_response": (
            "normalized smooth-min over every feasible path of each Medium OD"
        ),
        "feature_order": ["absolute_log1p", "high_od_centered"],
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
        description="Shared-2x six-objective Hattrick sparse OD-conflict experiment"
    )
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--beta", type=float, choices=ALLOWED_BETAS, default=4.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configure_beta(args.beta)
    run_dir = full.run_one(args.level, args.seed, force=args.force)
    description = method_description()
    description.update(
        {"level": args.level, "seed": args.seed, "run_directory": str(run_dir)}
    )
    full.write_json(run_dir / "sparse_od_conflict_method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
