from __future__ import annotations

import argparse
import dataclasses
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
    "shared2x_virtual_medium_dual_full_runtime", FULL_DIR / "run_experiment.py"
)
NativeHattrick = full.Hattrick


@dataclasses.dataclass(frozen=True)
class ScoutSettings:
    steps: int = 2
    temperature: float = 2.0
    price_power: float = 2.0

    def validate(self) -> "ScoutSettings":
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.price_power) or self.price_power <= 0.0:
            raise ValueError("price_power must be finite and positive")
        if self.price_power > 8.0:
            raise ValueError("price_power above 8 is disabled to avoid unstable scout prices")
        return self


ACTIVE_SETTINGS = ScoutSettings()


def number_label(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def settings_label(settings: ScoutSettings) -> str:
    return (
        f"steps_{settings.steps}"
        f"__temperature_{number_label(settings.temperature)}"
        f"__price_power_{number_label(settings.price_power)}"
    )


def configure_scout(
    steps: int = 2,
    temperature: float = 2.0,
    price_power: float = 2.0,
) -> ScoutSettings:
    """Select one isolated virtual-dual experiment configuration."""

    global ACTIVE_SETTINGS
    ACTIVE_SETTINGS = ScoutSettings(
        steps=int(steps),
        temperature=float(temperature),
        price_power=float(price_power),
    ).validate()
    full.OUTPUT_ROOT = THIS_DIR / "artifacts" / settings_label(ACTIVE_SETTINGS)
    return ACTIVE_SETTINGS


class VirtualMediumDualHattrick(NativeHattrick):
    """Serial Hattrick with a differentiable virtual-Medium dual scout.

    The scout never admits traffic.  Starting from a uniform feasible Medium
    split, it performs a small number of soft best-response rounds using only
    ESM-predicted Medium demand.  Edge marginal prices are mapped back to every
    path and consumed by the existing zero-initialized High residual adapters.
    """

    def __init__(self, props):
        props.future_lookahead = False
        super().__init__(props)
        settings = ACTIVE_SETTINGS.validate()
        self.virtual_medium_steps = settings.steps
        self.virtual_medium_temperature = settings.temperature
        self.virtual_medium_price_power = settings.price_power
        self.enable_medium_pressure_lookahead()

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
        """Return absolute and within-OD relative Medium path prices.

        For each scout round, the current Medium path logits are converted to
        predicted edge utilization by the native sparse path-to-edge operator.
        The convex congestion marginal ``utilization ** price_power`` is then
        summed along each candidate path.  Except on the last round, its
        negative value becomes the next soft path response, scaled by
        ``temperature``.  All operations remain in the autograd graph.
        """

        steps = int(self.virtual_medium_steps)
        temperature = float(self.virtual_medium_temperature)
        price_power = float(self.virtual_medium_price_power)
        if steps < 1 or temperature <= 0.0 or price_power <= 0.0:
            raise RuntimeError("Invalid virtual Medium scout configuration")

        pte_float = paths_to_edges.coalesce().to(dtype=torch.float32)
        medium_mask = self.path_mask_for_class(path_masks, 1)
        preview_logits = torch.zeros_like(tm2_pred)
        edge_price = None

        for step in range(steps):
            medium_edge_util, _, _ = self.compute_edge_utils(
                preview_logits,
                paths_to_edges,
                tm2_pred,
                capacities,
                props,
                batch_size,
                num_paths_per_pair,
                add_epsilon=False,
                path_mask=medium_mask,
            )
            # A modest cap prevents accidental overflow for exploratory powers
            # while leaving the normal GEANT utilization range untouched.
            safe_util = medium_edge_util.to(dtype=torch.float32).clamp(0.0, 64.0)
            edge_price = safe_util.pow(price_power)
            if step + 1 < steps:
                path_price = torch.sparse.mm(pte_float, edge_price.t()).t()
                preview_logits = (
                    -path_price / temperature
                ).unsqueeze(-1).to(dtype=tm2_pred.dtype)

        if edge_price is None:
            raise RuntimeError("Virtual Medium scout produced no edge price")

        path_price_sum = torch.sparse.mm(pte_float, edge_price.t()).t()
        absolute_price = torch.log1p(path_price_sum.clamp_min(0.0))
        grouped_price = absolute_price.reshape(
            batch_size, -1, num_paths_per_pair
        )
        relative_price = grouped_price - grouped_price.mean(dim=-1, keepdim=True)
        features = torch.stack(
            (absolute_price, relative_price.reshape(batch_size, -1)), dim=-1
        )
        return features.to(dtype=tm2_pred.dtype)


full.Hattrick = VirtualMediumDualHattrick


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "ordered_projection.py": FULL_DIR / "ordered_projection.py",
        "shared2x_full_objectives/run_experiment.py": FULL_DIR / "run_experiment.py",
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "utils/training_utils.py": ROOT / "utils" / "training_utils.py",
    }
    hashes = {name: full.sha256(path) for name, path in paths.items()}
    encoded = json.dumps(
        dataclasses.asdict(ACTIVE_SETTINGS), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    hashes["virtual_medium_dual/settings"] = hashlib.sha256(encoded).hexdigest()
    return hashes


full.source_hashes = source_hashes
configure_scout()


def method_description() -> dict:
    settings = dataclasses.asdict(ACTIVE_SETTINGS)
    return {
        "method": "Hattrick virtual-Medium marginal-dual scout",
        "architecture": (
            "Native High -> High+Medium -> High+Medium+Low cascade. Before High, "
            "a non-admitting differentiable Medium scout performs soft path-price "
            "responses and supplies absolute/within-OD-relative path-price features to the native "
            "zero-initialized Medium-pressure adapters."
        ),
        "scout": settings,
        "projection": "unchanged six-objective ordered gradient projection",
        "objectives": list(full.OBJECTIVE_NAMES),
        "inference_information": (
            "The scout consumes ESM-predicted Medium traffic, topology, capacities, "
            "candidate paths, and feasibility masks only. Current actual traffic is "
            "not an inference input."
        ),
        "output_root": str(full.OUTPUT_ROOT),
        "source_sha256": source_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shared-2x six-objective Hattrick virtual-Medium dual experiment"
    )
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--price-power", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    configure_scout(args.steps, args.temperature, args.price_power)
    run_dir = full.run_one(args.level, args.seed, force=args.force)
    description = method_description()
    description.update(
        {"level": args.level, "seed": args.seed, "run_directory": str(run_dir)}
    )
    full.write_json(run_dir / "virtual_medium_dual_method.json", description)
    print(json.dumps(description, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
