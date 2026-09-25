from __future__ import annotations

"""Protocol-fixed strict-ESM robustness tests for frozen Hattrick checkpoints.

This script extends ``evaluate_holdout.py`` with class-specific prediction
bias, demand-preserving OD-shape noise, and physical-link capacity derating.
It reports raw admission metrics only; no 1x oracle is loaded or reused.
"""

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import evaluate_holdout as base


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "artifacts" / "robustness"
FAMILIES = ("class_bias", "shape_noise", "capacity_derating")
PREDICTION_FAMILIES = ("class_bias", "shape_noise")
DEFAULT_SEEDS = (490, 491, 492)
REQUIRED_METHODS = ("phase_a", "hattrick_f", "six_loss_control")
# Fixed before reading any capacity-fault result.  Each formal 500-snapshot
# window contributes the same 20 uniformly-spaced audit points.
CAPACITY_AUDIT_OFFSETS = tuple(range(0, 500, 25))
CLASS_BIAS_SCENARIOS = (
    ("high_bias_0p8", (0.8, 1.0, 1.0)),
    ("high_bias_1p2", (1.2, 1.0, 1.0)),
    ("medium_bias_0p8", (1.0, 0.8, 1.0)),
    ("medium_bias_1p2", (1.0, 1.2, 1.0)),
)
NOISE_SIGMAS = (0.1, 0.2)
NOISE_SEEDS = (20260825, 20260826, 20260827)
DERATING_FACTORS = (0.5, 0.1)
NEAR_OUTAGE_FACTOR = 0.01


@dataclass(frozen=True)
class PhysicalLink:
    index: int
    left: str
    right: str
    directed_edge_indices: tuple[int, int]

    @property
    def label(self) -> str:
        return f"{self.left}<->{self.right}"


def physical_links(directed_edges: list[tuple]) -> list[PhysicalLink]:
    """Pair reverse directed edges without relying on node sortability."""

    grouped: dict[frozenset, list[tuple[int, object, object]]] = {}
    order: list[frozenset] = []
    for edge_index, (left, right) in enumerate(directed_edges):
        key = frozenset((left, right))
        if len(key) != 2:
            raise RuntimeError(f"self-loop cannot define a physical link: {(left, right)}")
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((edge_index, left, right))

    links = []
    for link_index, key in enumerate(order):
        members = grouped[key]
        if len(members) != 2:
            raise RuntimeError(
                f"physical link {tuple(key)} has {len(members)} directed edges"
            )
        first, second = members
        if not (first[1] == second[2] and first[2] == second[1]):
            raise RuntimeError(f"physical link lacks exact reverse edges: {members}")
        links.append(
            PhysicalLink(
                index=link_index,
                left=str(first[1]),
                right=str(first[2]),
                directed_edge_indices=(int(first[0]), int(second[0])),
            )
        )
    if len(links) != 36:
        raise RuntimeError(f"GEANT must contain 36 physical links, found {len(links)}")
    return links


def class_scaled_prediction(
    prediction: tuple[torch.Tensor, ...], factors: tuple[float, float, float]
) -> tuple[torch.Tensor, ...]:
    return tuple(
        value * float(factor) for value, factor in zip(prediction, factors)
    )


def _noise_rng(seed: int, snapshot: int, class_index: int) -> np.random.Generator:
    # SeedSequence makes each class/snapshot draw invariant to batch size/order.
    sequence = np.random.SeedSequence([int(seed), int(snapshot), int(class_index)])
    return np.random.default_rng(sequence)


def demand_preserving_shape_noise(
    prediction: tuple[torch.Tensor, ...],
    snapshots: list[int],
    sigma: float,
    seed: int,
    num_pairs: int,
) -> tuple[tuple[torch.Tensor, ...], float, float]:
    """Apply OD lognormal noise while preserving each class/snapshot total."""

    output = []
    maximum_relative_error = 0.0
    total_variations: list[float] = []
    for class_index, source in enumerate(prediction):
        device = source.device
        dtype = source.dtype
        grouped = source.reshape(
            len(snapshots), num_pairs, base.PATHS_PER_OD, -1
        )
        reference = grouped[:, :, :1, :]
        if not torch.allclose(
            grouped,
            reference.expand_as(grouped),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError("TM is not repeated identically across each OD's paths")
        od = reference[:, :, 0, 0]
        perturbed_rows = []
        for local_index, snapshot in enumerate(snapshots):
            rng = _noise_rng(seed, snapshot, class_index)
            normal = torch.as_tensor(
                rng.normal(size=num_pairs), device=device, dtype=dtype
            )
            multiplier = torch.exp(
                float(sigma) * normal - 0.5 * float(sigma) ** 2
            )
            noisy = od[local_index] * multiplier
            original_total = od[local_index].sum()
            noisy_total = noisy.sum()
            if float(original_total.item()) <= 0.0:
                renormalized = noisy
            else:
                renormalized = noisy * (
                    original_total / noisy_total.clamp_min(1e-12)
                )
            relative_error = float(
                (
                    (renormalized.sum() - original_total).abs()
                    / original_total.clamp_min(1e-12)
                ).item()
            )
            maximum_relative_error = max(maximum_relative_error, relative_error)
            total_variations.append(
                float(
                    (
                        0.5
                        * (renormalized - od[local_index]).abs().sum()
                        / original_total.clamp_min(1e-12)
                    ).item()
                )
            )
            perturbed_rows.append(renormalized)
        perturbed_od = torch.stack(perturbed_rows, dim=0)
        repeated = (
            perturbed_od[:, :, None, None]
            .expand(-1, -1, base.PATHS_PER_OD, 1)
            .reshape_as(source)
        )
        output.append(repeated)
    return (
        tuple(output),
        maximum_relative_error,
        float(np.mean(total_variations)),
    )


def derate_capacities(
    capacities: torch.Tensor, link: PhysicalLink, factor: float
) -> torch.Tensor:
    changed = capacities.clone()
    changed[:, list(link.directed_edge_indices)] *= float(factor)
    return changed


def capacity_aware_node_features(
    nominal: torch.Tensor,
    capacities: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Reproduce Read_Snapshot node features for announced derating."""

    result = nominal.clone()
    batch_size, node_count, _ = result.shape
    cap_sum = torch.zeros(
        (batch_size, node_count), device=capacities.device, dtype=capacities.dtype
    )
    sources = edge_index[0].reshape(1, -1).expand(batch_size, -1)
    cap_sum.scatter_add_(1, sources, capacities)
    result[:, :, 1] = cap_sum.to(dtype=result.dtype)
    return result


def scenario_metadata(
    *,
    family: str,
    scenario_id: str,
    class_factors: tuple[float, float, float] = (1.0, 1.0, 1.0),
    noise_sigma: float | None = None,
    noise_seed: int | None = None,
    link: PhysicalLink | None = None,
    capacity_factor: float | None = None,
    capacity_mode: str = "nominal",
    near_outage: bool = False,
    noise_total_relative_error: float | None = None,
    noise_demand_weighted_tv_mean: float | None = None,
) -> dict:
    return {
        "scenario_family": family,
        "scenario_id": scenario_id,
        "high_esm_factor": class_factors[0],
        "medium_esm_factor": class_factors[1],
        "low_esm_factor": class_factors[2],
        "noise_sigma": "" if noise_sigma is None else noise_sigma,
        "noise_seed": "" if noise_seed is None else noise_seed,
        "noise_total_relative_error": (
            "" if noise_total_relative_error is None else noise_total_relative_error
        ),
        "noise_demand_weighted_tv_mean": (
            ""
            if noise_demand_weighted_tv_mean is None
            else noise_demand_weighted_tv_mean
        ),
        "capacity_mode": capacity_mode,
        "capacity_factor": "" if capacity_factor is None else capacity_factor,
        "physical_link_index": "" if link is None else link.index,
        "physical_link": "" if link is None else link.label,
        "directed_edge_indices": (
            ""
            if link is None
            else ";".join(str(value) for value in link.directed_edge_indices)
        ),
        "near_outage": bool(near_outage),
        "policy_capacity_view": (
            "derated" if capacity_mode == "announced" else "nominal"
        ),
        "admission_capacity_view": (
            "nominal" if capacity_mode == "nominal" else "derated"
        ),
        "transformer_cache_cleared_before_policy": True,
        "fixed_ksp_not_recomputed": True,
    }


def evaluate_scenario(
    *,
    seed: int,
    method: base.MethodSpec,
    model,
    props,
    static: base.StaticTopology,
    window: str,
    batch: dict,
    prediction: tuple[torch.Tensor, ...],
    policy_node_features: torch.Tensor,
    policy_capacities: torch.Tensor,
    admission_capacities: torch.Tensor,
    metadata: dict,
) -> list[dict]:
    # generate_prediction_only_policy deletes transformer_output on every call.
    policy = base.generate_prediction_only_policy(
        model,
        props,
        static,
        policy_node_features,
        policy_capacities,
        prediction,
    )
    admitted, cumulative_admitted, cumulative_offered = (
        base.sequential_actual_admission(
            model,
            props,
            static,
            policy,
            batch["actual"],
            admission_capacities,
        )
    )
    rows = base.batch_rows(
        seed=seed,
        method=method,
        window=window,
        bias=1.0,
        indices=batch["indices"],
        actual=batch["actual"],
        predicted=prediction,
        admitted=admitted,
        cumulative_admitted=cumulative_admitted,
        cumulative_offered=cumulative_offered,
        capacities=admission_capacities,
        static=static,
    )
    return [{**row, **metadata} for row in rows]


@dataclass
class ClassAggregate:
    count: int = 0
    fulfill_sum: float = 0.0
    admitted_sum: float = 0.0
    fulfill_min: float = math.inf
    fulfill_max: float = -math.inf
    capacity_ratio_max: float = -math.inf
    violation_max: float = 0.0
    # One scalar rather than one dictionary is retained per row, solely for
    # exact p1/p10 compatibility.  Formal execution retains <0.5M scalars.
    fulfill_values: list[float] = field(default_factory=list)

    def update(self, row: dict) -> None:
        fulfill = float(row["raw_fulfill_ratio"])
        admitted = float(row["admitted_traffic"])
        capacity_ratio = float(row["post_admission_capacity_ratio"])
        violation = float(row["capacity_violation"])
        self.count += 1
        self.fulfill_sum += fulfill
        self.admitted_sum += admitted
        self.fulfill_min = min(self.fulfill_min, fulfill)
        self.fulfill_max = max(self.fulfill_max, fulfill)
        self.capacity_ratio_max = max(self.capacity_ratio_max, capacity_ratio)
        self.violation_max = max(self.violation_max, violation)
        self.fulfill_values.append(fulfill)


@dataclass
class PairAggregate:
    count: int = 0
    fulfill_delta_sum: float = 0.0
    admitted_delta_sum: float = 0.0
    better_count: int = 0

    def update(self, candidate: dict, baseline: dict) -> None:
        delta = float(candidate["raw_fulfill_ratio"]) - float(
            baseline["raw_fulfill_ratio"]
        )
        self.count += 1
        self.fulfill_delta_sum += delta
        self.admitted_delta_sum += float(candidate["admitted_traffic"]) - float(
            baseline["admitted_traffic"]
        )
        self.better_count += int(delta > 0.0)


@dataclass
class CapacityAggregate:
    count: int = 0
    capacity_ratio_max: float = -math.inf
    violation_max: float = 0.0
    passes: bool = True

    def update(self, row: dict) -> None:
        ratio = float(row["post_admission_capacity_ratio"])
        violation = float(row["capacity_violation"])
        self.count += 1
        self.capacity_ratio_max = max(self.capacity_ratio_max, ratio)
        self.violation_max = max(self.violation_max, violation)
        self.passes = self.passes and ratio <= 1.0 + base.CAPACITY_TOLERANCE


class StreamingCsvWriter:
    """Stream rows to an atomic temporary CSV instead of retaining dicts."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination.resolve()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.destination.with_suffix(self.destination.suffix + ".tmp")
        self.handle = self.temporary.open("w", newline="", encoding="utf-8")
        self.writer: csv.DictWriter | None = None
        self.fieldnames: list[str] | None = None
        self.row_count = 0

    def append(self, rows: list[dict]) -> None:
        for row in rows:
            fields = list(row)
            if self.writer is None:
                self.fieldnames = fields
                self.writer = csv.DictWriter(self.handle, fieldnames=fields)
                self.writer.writeheader()
            elif fields != self.fieldnames:
                raise RuntimeError("robustness row schema changed during streaming")
            self.writer.writerow(row)
            self.row_count += 1

    def commit(self) -> Path:
        if self.row_count == 0:
            self.abort()
            raise RuntimeError("no robustness scenario was evaluated")
        self.handle.flush()
        self.handle.close()
        self.temporary.replace(self.destination)
        return self.destination

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        if self.temporary.exists():
            self.temporary.unlink()


class CoverageTracker:
    """Compact exact coverage audit using one bit mask per logical group."""

    def __init__(
        self,
        seeds: list[int],
        methods_by_seed: dict[int, list[str]],
        scenarios_by_family: dict[str, list[str]],
        prediction_snapshots: dict[str, list[int]],
        capacity_snapshots: dict[str, list[int]],
    ) -> None:
        self.snapshot_positions: dict[tuple[str, str], dict[int, int]] = {}
        self.full_masks: dict[tuple[str, str], int] = {}
        for phase, selections in (
            ("prediction", prediction_snapshots),
            ("capacity", capacity_snapshots),
        ):
            for window, snapshots in selections.items():
                positions = {snapshot: index for index, snapshot in enumerate(snapshots)}
                self.snapshot_positions[(phase, window)] = positions
                self.full_masks[(phase, window)] = (1 << len(snapshots)) - 1

        self.expected: set[tuple] = set()
        for seed in seeds:
            for method in methods_by_seed[seed]:
                for family, scenario_ids in scenarios_by_family.items():
                    phase = "capacity" if family == "capacity_derating" else "prediction"
                    selections = (
                        capacity_snapshots if phase == "capacity" else prediction_snapshots
                    )
                    for window in selections:
                        for scenario_id in scenario_ids:
                            for class_name in base.CLASSES:
                                self.expected.add(
                                    (
                                        phase,
                                        seed,
                                        method,
                                        window,
                                        family,
                                        scenario_id,
                                        class_name,
                                    )
                                )
        self.masks: dict[tuple, int] = {}
        self.observed_rows = 0
        self.duplicate_rows = 0
        self.unexpected_rows = 0
        self.unexpected_examples: list[dict] = []

    def observe(self, row: dict) -> None:
        family = str(row["scenario_family"])
        phase = "capacity" if family == "capacity_derating" else "prediction"
        key = (
            phase,
            int(row["seed"]),
            str(row["method"]),
            str(row["window"]),
            family,
            str(row["scenario_id"]),
            str(row["class"]),
        )
        snapshot = int(row["snapshot"])
        positions = self.snapshot_positions.get((phase, str(row["window"])), {})
        position = positions.get(snapshot)
        self.observed_rows += 1
        if key not in self.expected or position is None:
            self.unexpected_rows += 1
            if len(self.unexpected_examples) < 20:
                self.unexpected_examples.append({"key": list(key), "snapshot": snapshot})
            return
        bit = 1 << position
        current = self.masks.get(key, 0)
        if current & bit:
            self.duplicate_rows += 1
        self.masks[key] = current | bit

    def report(self) -> dict:
        missing_groups = 0
        incomplete_groups = 0
        missing_rows = 0
        missing_examples = []
        expected_rows = 0
        unique_rows = 0
        for key in self.expected:
            phase, _, _, window, _, _, _ = key
            full = self.full_masks[(phase, window)]
            expected_rows += full.bit_count()
            observed = self.masks.get(key, 0)
            unique_rows += observed.bit_count()
            missing = full & ~observed
            if missing:
                missing_rows += missing.bit_count()
                if observed == 0:
                    missing_groups += 1
                else:
                    incomplete_groups += 1
                if len(missing_examples) < 20:
                    missing_examples.append(
                        {"key": list(key), "missing_snapshot_count": missing.bit_count()}
                    )
        complete = (
            missing_rows == 0
            and self.duplicate_rows == 0
            and self.unexpected_rows == 0
            and self.observed_rows == expected_rows
        )
        return {
            "complete": complete,
            "expected_groups": len(self.expected),
            "observed_groups": len(self.masks),
            "missing_groups": missing_groups,
            "incomplete_groups": incomplete_groups,
            "expected_rows": expected_rows,
            "observed_rows": self.observed_rows,
            "unique_expected_rows_observed": unique_rows,
            "missing_rows": missing_rows,
            "duplicate_rows": self.duplicate_rows,
            "unexpected_rows": self.unexpected_rows,
            "missing_examples": missing_examples,
            "unexpected_examples": self.unexpected_examples,
        }


class OnlineSummary:
    def __init__(self, coverage: CoverageTracker) -> None:
        self.coverage = coverage
        self.classes: dict[tuple, ClassAggregate] = {}
        self.pairs: dict[tuple, PairAggregate] = {}
        self.capacity: dict[tuple, CapacityAggregate] = {}
        self.scenario_ids: dict[str, set[str]] = {family: set() for family in FAMILIES}
        self.numeric_values_checked = 0

    def observe_rows(self, rows: list[dict]) -> None:
        for row in rows:
            for value in row.values():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    self.numeric_values_checked += 1
                    if not math.isfinite(float(value)):
                        raise RuntimeError("robustness evaluation produced NaN or Inf")
            self.coverage.observe(row)
            family = str(row["scenario_family"])
            self.scenario_ids[family].add(str(row["scenario_id"]))
            class_key = (
                int(row["seed"]),
                str(row["method"]),
                str(row["method_label"]),
                str(row["window"]),
                family,
                str(row["scenario_id"]),
                str(row["class"]),
            )
            self.classes.setdefault(class_key, ClassAggregate()).update(row)
            if family == "capacity_derating":
                capacity_key = (
                    int(row["seed"]),
                    str(row["method"]),
                    str(row["window"]),
                    str(row["scenario_id"]),
                )
                self.capacity.setdefault(capacity_key, CapacityAggregate()).update(row)

    def observe_pairs(self, rows_by_method: dict[str, list[dict]]) -> None:
        baseline_rows = rows_by_method.get("phase_a")
        if baseline_rows is None:
            return
        baseline = {
            (int(row["snapshot"]), str(row["class"])): row for row in baseline_rows
        }
        for candidate_name, candidate_rows in rows_by_method.items():
            if candidate_name == "phase_a":
                continue
            for candidate in candidate_rows:
                pair_id = (int(candidate["snapshot"]), str(candidate["class"]))
                reference = baseline.get(pair_id)
                if reference is None:
                    continue
                key = (
                    int(candidate["seed"]),
                    candidate_name,
                    str(candidate["window"]),
                    str(candidate["scenario_family"]),
                    str(candidate["scenario_id"]),
                    str(candidate["class"]),
                )
                self.pairs.setdefault(key, PairAggregate()).update(candidate, reference)

    def class_rows(self) -> list[dict]:
        output = []
        for key, value in sorted(self.classes.items()):
            seed, method, label, window, family, scenario_id, class_name = key
            fulfill = np.asarray(value.fulfill_values, dtype=np.float64)
            output.append(
                {
                    "seed": seed,
                    "method": method,
                    "method_label": label,
                    "window": window,
                    "scenario_family": family,
                    "scenario_id": scenario_id,
                    "class": class_name,
                    "n": value.count,
                    "raw_fulfill_mean": value.fulfill_sum / value.count,
                    "raw_fulfill_p1": float(np.percentile(fulfill, 1)),
                    "raw_fulfill_p10": float(np.percentile(fulfill, 10)),
                    "raw_fulfill_min": value.fulfill_min,
                    "raw_fulfill_max": value.fulfill_max,
                    "admitted_traffic_mean": value.admitted_sum / value.count,
                    "post_admission_capacity_ratio_max": value.capacity_ratio_max,
                    "capacity_violation_max": value.violation_max,
                }
            )
        return output

    def pair_rows(self) -> list[dict]:
        output = []
        for key, value in sorted(self.pairs.items()):
            seed, candidate, window, family, scenario_id, class_name = key
            output.append(
                {
                    "seed": seed,
                    "candidate": candidate,
                    "baseline": "phase_a",
                    "window": window,
                    "scenario_family": family,
                    "scenario_id": scenario_id,
                    "class": class_name,
                    "n": value.count,
                    "raw_fulfill_mean_delta": value.fulfill_delta_sum / value.count,
                    "admitted_traffic_mean_delta": value.admitted_delta_sum / value.count,
                    "candidate_better_fraction": value.better_count / value.count,
                }
            )
        return output

    def capacity_rows(self) -> list[dict]:
        return [
            {
                "seed": key[0],
                "method": key[1],
                "window": key[2],
                "scenario_id": key[3],
                "n_class_snapshot_rows": value.count,
                "maximum_post_admission_ratio": value.capacity_ratio_max,
                "maximum_violation": value.violation_max,
                "passes": value.passes,
            }
            for key, value in sorted(self.capacity.items())
        ]


def resolve_windows(args: argparse.Namespace) -> dict[str, tuple[int, int]]:
    names = ["near"] if args.smoke else args.windows
    limit = args.max_snapshots if args.max_snapshots is not None else (2 if args.smoke else None)
    result = {}
    for name in names:
        left, right = base.WINDOWS[name]
        if limit is not None:
            right = min(right, left + limit)
        result[name] = (left, right)
    return result


def selected_batches(
    dataset: base.RawHoldoutDataset, snapshots: list[int], batch_size: int
):
    positions = []
    for snapshot in snapshots:
        position = snapshot - dataset.start
        if position < 0 or position >= len(dataset.indices):
            raise RuntimeError(f"snapshot {snapshot} is outside loaded dataset")
        positions.append(position)
    for left in range(0, len(positions), batch_size):
        chosen = positions[left : left + batch_size]
        yield {
            "indices": [dataset.indices[position] for position in chosen],
            "node_features": torch.stack([dataset.node_features[position] for position in chosen]),
            "capacities": torch.stack([dataset.capacities[position] for position in chosen]),
            "actual": tuple(
                torch.stack([dataset.actual[position][class_index] for position in chosen])
                for class_index in range(3)
            ),
            "predicted": tuple(
                torch.stack([dataset.predicted[position][class_index] for position in chosen])
                for class_index in range(3)
            ),
        }


def move_batch(raw_batch: dict, device: torch.device) -> dict:
    return {
        **raw_batch,
        "node_features": raw_batch["node_features"].to(device),
        "capacities": raw_batch["capacities"].to(device),
        "actual": tuple(value.to(device) for value in raw_batch["actual"]),
        "predicted": tuple(value.to(device) for value in raw_batch["predicted"]),
    }


def bias_scenarios(smoke: bool):
    return (
        (CLASS_BIAS_SCENARIOS[0], CLASS_BIAS_SCENARIOS[3])
        if smoke
        else CLASS_BIAS_SCENARIOS
    )


def noise_scenarios(smoke: bool):
    if smoke:
        return ((NOISE_SIGMAS[0], NOISE_SEEDS[0]),)
    return tuple(
        (sigma, noise_seed)
        for sigma in NOISE_SIGMAS
        for noise_seed in NOISE_SEEDS
    )


def capacity_scenarios(smoke: bool):
    if smoke:
        return (("announced", 0.5, False), ("unannounced", 0.01, True))
    return tuple(
        (mode, factor, False)
        for mode in ("announced", "unannounced")
        for factor in DERATING_FACTORS
    ) + (("unannounced", NEAR_OUTAGE_FACTOR, True),)


def capacity_scenario_id(
    link_index: int, mode: str, factor: float, near_outage: bool
) -> str:
    label = str(factor).replace(".", "p")
    suffix = "_near_outage" if near_outage else ""
    return f"link_{link_index:02d}_{mode}_factor_{label}{suffix}"


def active_families(args: argparse.Namespace) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(args.families))
    if args.mode == "prediction":
        active = tuple(family for family in requested if family in PREDICTION_FAMILIES)
    elif args.mode == "capacity":
        active = tuple(family for family in requested if family == "capacity_derating")
    else:
        active = requested
    if not active:
        raise ValueError("the selected --mode and --families contain no scenario")
    return active


def expected_scenario_ids(
    families: tuple[str, ...], smoke: bool, link_indices: list[int]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if "class_bias" in families:
        result["class_bias"] = [scenario[0] for scenario in bias_scenarios(smoke)]
    if "shape_noise" in families:
        result["shape_noise"] = [
            f"od_lognormal_sigma_{str(sigma).replace('.', 'p')}_seed_{noise_seed}"
            for sigma, noise_seed in noise_scenarios(smoke)
        ]
    if "capacity_derating" in families:
        result["capacity_derating"] = [
            capacity_scenario_id(link, mode, factor, near_outage)
            for link in link_indices
            for mode, factor, near_outage in capacity_scenarios(smoke)
        ]
    return result


def emit_scenario(
    *,
    seed: int,
    methods: list[base.MethodSpec],
    models: dict,
    props,
    static: base.StaticTopology,
    window: str,
    batch: dict,
    prediction: tuple[torch.Tensor, ...],
    policy_node_features: torch.Tensor,
    policy_capacities: torch.Tensor,
    admission_capacities: torch.Tensor,
    metadata: dict,
    writer: StreamingCsvWriter,
    online: OnlineSummary,
) -> None:
    rows_by_method = {}
    for method in methods:
        rows = evaluate_scenario(
            seed=seed,
            method=method,
            model=models[method.key],
            props=props,
            static=static,
            window=window,
            batch=batch,
            prediction=prediction,
            policy_node_features=policy_node_features,
            policy_capacities=policy_capacities,
            admission_capacities=admission_capacities,
            metadata=metadata,
        )
        writer.append(rows)
        online.observe_rows(rows)
        rows_by_method[method.key] = rows
    online.observe_pairs(rows_by_method)


def preflight_methods(
    seeds: list[int], allow_missing: bool, smoke: bool
) -> tuple[dict[int, list[base.MethodSpec]], list[dict]]:
    methods_by_seed = {}
    skipped = []
    for seed in seeds:
        methods, missing = base.default_methods(seed)
        skipped.extend(missing)
        if not methods:
            raise FileNotFoundError(f"no checkpoint exists for seed {seed}")
        present = {method.key for method in methods}
        missing_required = sorted(set(REQUIRED_METHODS) - present)
        if missing_required and not (allow_missing or smoke):
            raise FileNotFoundError(
                f"formal robustness run refuses missing checkpoints for seed {seed}: "
                f"{missing_required}; use --allow-missing or --smoke explicitly"
            )
        methods_by_seed[seed] = methods
    return methods_by_seed, skipped


def formal_preconditions(
    args: argparse.Namespace,
    families: tuple[str, ...],
    windows: dict[str, tuple[int, int]],
    link_indices: list[int],
    methods_by_seed: dict[int, list[base.MethodSpec]],
) -> list[str]:
    reasons = []
    if args.smoke:
        reasons.append("smoke_mode")
    if args.allow_missing:
        reasons.append("allow_missing_enabled")
    if args.max_snapshots is not None:
        reasons.append("max_snapshots_truncation")
    if args.mode != "both":
        reasons.append("single_phase_mode")
    if list(args.seeds) != list(DEFAULT_SEEDS):
        reasons.append("noncanonical_seeds")
    if list(args.windows) != list(base.WINDOWS):
        reasons.append("not_both_preregistered_windows")
    if set(families) != set(FAMILIES) or len(families) != len(FAMILIES):
        reasons.append("not_all_preregistered_families")
    if link_indices != list(range(36)):
        reasons.append("not_all_36_physical_links")
    if windows != base.WINDOWS:
        reasons.append("window_bounds_not_full_500")
    for seed, methods in methods_by_seed.items():
        present = {method.key for method in methods}
        if present != set(REQUIRED_METHODS):
            reasons.append(f"seed_{seed}_method_set_incomplete")
    return reasons


def run(args: argparse.Namespace) -> Path:
    frozen_source_hashes = base.verify_frozen_sources()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    shared_dir = str((base.TEST_DIR / "shared2x_order_regularizer").resolve())
    root_dir = str(base.ROOT.resolve())
    for path in (shared_dir, root_dir):
        if path not in sys.path:
            sys.path.insert(0, path)
    runtime = base.load_module("hattrick_f_robustness_runtime", base.RUNTIME_SOURCE)
    snapshot_module = base.load_module(
        "hattrick_f_robustness_snapshot", base.ROOT / "utils" / "snapshot_utils.py"
    )
    cluster_module = base.load_module(
        "hattrick_f_robustness_cluster", base.ROOT / "utils" / "cluster_utils.py"
    )

    seeds = [args.seeds[0]] if args.smoke else list(args.seeds)
    windows = resolve_windows(args)
    families = active_families(args)
    methods_by_seed, skipped = preflight_methods(seeds, args.allow_missing, args.smoke)
    link_indices = (
        list(args.links)
        if args.links is not None
        else ([0] if args.smoke else list(range(36)))
    )
    invalid_links = sorted(set(link_indices) - set(range(36)))
    if invalid_links:
        raise ValueError(f"invalid physical link indices: {invalid_links}")

    prediction_snapshots = {
        name: list(range(left, right))
        for name, (left, right) in windows.items()
        if any(family in families for family in PREDICTION_FAMILIES)
    }
    capacity_snapshots = {}
    if "capacity_derating" in families:
        for name, (left, right) in windows.items():
            if args.smoke:
                selected = list(range(left, right))
            else:
                if right - left != 500:
                    raise RuntimeError(
                        "capacity audit requires an untruncated 500-snapshot window"
                    )
                selected = [left + offset for offset in CAPACITY_AUDIT_OFFSETS]
            capacity_snapshots[name] = selected

    scenario_ids = expected_scenario_ids(families, args.smoke, link_indices)
    method_keys = {
        seed: [method.key for method in methods_by_seed[seed]] for seed in seeds
    }
    coverage = CoverageTracker(
        seeds,
        method_keys,
        scenario_ids,
        prediction_snapshots,
        capacity_snapshots,
    )
    online = OnlineSummary(coverage)
    formal_reasons = formal_preconditions(
        args, families, windows, link_indices, methods_by_seed
    )
    output_dir = args.output_dir.resolve()
    if args.output_dir == OUTPUT_ROOT and formal_reasons:
        output_dir = (
            HERE
            / "artifacts"
            / ("robustness_smoke" if args.smoke else "robustness_nonformal")
        ).resolve()
    filename_suffix = "" if args.mode == "both" else f"_{args.mode}"
    snapshot_csv = output_dir / f"snapshots{filename_suffix}.csv"
    summary_json = output_dir / f"summary{filename_suffix}.json"
    writer = StreamingCsvWriter(snapshot_csv)
    checkpoint_audit = []
    physical_inventory: list[dict] | None = None
    started = time.perf_counter()

    try:
        for seed in seeds:
            methods = methods_by_seed[seed]
            props = runtime.build_props(4, device)
            props.topo = base.TOPOLOGY
            props.device = device
            props.dtype = torch.float32
            props.dynamic = 0
            props.checkpoint = 0
            props.path_mask = 0
            models, audits = base.load_models(methods, runtime, props, device)
            checkpoint_audit.extend([{"seed": seed, **item} for item in audits])

            for window, (left, right) in windows.items():
                dataset = base.RawHoldoutDataset(
                    props, left, right, snapshot_module, cluster_module
                )
                static = base.move_static(dataset, device)
                links = physical_links(dataset.directed_edges)
                inventory = [
                    {
                        "index": link.index,
                        "physical_link": link.label,
                        "directed_edge_indices": list(link.directed_edge_indices),
                    }
                    for link in links
                ]
                if physical_inventory is None:
                    physical_inventory = inventory
                elif inventory != physical_inventory:
                    raise RuntimeError("physical-link inventory changed across windows")
                chosen_links = [links[index] for index in link_indices]

                if prediction_snapshots:
                    for raw_batch in selected_batches(
                        dataset, prediction_snapshots[window], args.batch_size
                    ):
                        batch = move_batch(raw_batch, device)
                        if "class_bias" in families:
                            for scenario_id, factors in bias_scenarios(args.smoke):
                                prediction = class_scaled_prediction(batch["predicted"], factors)
                                emit_scenario(
                                    seed=seed,
                                    methods=methods,
                                    models=models,
                                    props=props,
                                    static=static,
                                    window=window,
                                    batch=batch,
                                    prediction=prediction,
                                    policy_node_features=batch["node_features"],
                                    policy_capacities=batch["capacities"],
                                    admission_capacities=batch["capacities"],
                                    metadata=scenario_metadata(
                                        family="class_bias",
                                        scenario_id=scenario_id,
                                        class_factors=factors,
                                    ),
                                    writer=writer,
                                    online=online,
                                )
                        if "shape_noise" in families:
                            for sigma, noise_seed in noise_scenarios(args.smoke):
                                prediction, relative_error, demand_weighted_tv = (
                                    demand_preserving_shape_noise(
                                        batch["predicted"],
                                        batch["indices"],
                                        sigma,
                                        noise_seed,
                                        dataset.num_pairs,
                                    )
                                )
                                if sigma > 0.0 and demand_weighted_tv <= 0.0:
                                    raise RuntimeError(
                                        "positive shape-noise sigma changed no OD demand"
                                    )
                                scenario_id = (
                                    f"od_lognormal_sigma_{str(sigma).replace('.', 'p')}_"
                                    f"seed_{noise_seed}"
                                )
                                emit_scenario(
                                    seed=seed,
                                    methods=methods,
                                    models=models,
                                    props=props,
                                    static=static,
                                    window=window,
                                    batch=batch,
                                    prediction=prediction,
                                    policy_node_features=batch["node_features"],
                                    policy_capacities=batch["capacities"],
                                    admission_capacities=batch["capacities"],
                                    metadata=scenario_metadata(
                                        family="shape_noise",
                                        scenario_id=scenario_id,
                                        noise_sigma=sigma,
                                        noise_seed=noise_seed,
                                        noise_total_relative_error=relative_error,
                                        noise_demand_weighted_tv_mean=demand_weighted_tv,
                                    ),
                                    writer=writer,
                                    online=online,
                                )
                        print(
                            f"[robustness:prediction] seed={seed} window={window} "
                            f"snapshots={batch['indices'][0]}..{batch['indices'][-1]}",
                            flush=True,
                        )

                if capacity_snapshots:
                    for raw_batch in selected_batches(
                        dataset, capacity_snapshots[window], args.batch_size
                    ):
                        batch = move_batch(raw_batch, device)
                        rebuilt_nominal = capacity_aware_node_features(
                            batch["node_features"], batch["capacities"], static.edge_index
                        )
                        node_feature_error = float(
                            (rebuilt_nominal - batch["node_features"]).abs().max().item()
                        )
                        if node_feature_error > 1e-5:
                            raise RuntimeError(
                                "capacity-aware node-feature reconstruction failed: "
                                f"max_abs_error={node_feature_error}"
                            )
                        for link in chosen_links:
                            for mode, factor, is_near_outage in capacity_scenarios(args.smoke):
                                admission_capacities = derate_capacities(
                                    batch["capacities"], link, factor
                                )
                                if mode == "announced":
                                    policy_capacities = admission_capacities
                                    policy_node_features = capacity_aware_node_features(
                                        batch["node_features"],
                                        policy_capacities,
                                        static.edge_index,
                                    )
                                else:
                                    policy_capacities = batch["capacities"]
                                    policy_node_features = batch["node_features"]
                                scenario_id = capacity_scenario_id(
                                    link.index, mode, factor, is_near_outage
                                )
                                emit_scenario(
                                    seed=seed,
                                    methods=methods,
                                    models=models,
                                    props=props,
                                    static=static,
                                    window=window,
                                    batch=batch,
                                    prediction=batch["predicted"],
                                    policy_node_features=policy_node_features,
                                    policy_capacities=policy_capacities,
                                    admission_capacities=admission_capacities,
                                    metadata=scenario_metadata(
                                        family="capacity_derating",
                                        scenario_id=scenario_id,
                                        link=link,
                                        capacity_factor=factor,
                                        capacity_mode=mode,
                                        near_outage=is_near_outage,
                                    ),
                                    writer=writer,
                                    online=online,
                                )
                        print(
                            f"[robustness:capacity] seed={seed} window={window} "
                            f"audit_snapshots={batch['indices'][0]}..{batch['indices'][-1]}",
                            flush=True,
                        )
        coverage_report = coverage.report()
        if not formal_reasons and not coverage_report["complete"]:
            raise RuntimeError(
                "formal robustness coverage is incomplete; refusing to publish "
                f"the default formal artifact: {coverage_report}"
            )
        snapshot_csv = writer.commit()
    except BaseException:
        writer.abort()
        raise

    formal_result = not formal_reasons and coverage_report["complete"]
    capacity_rows = online.capacity_rows()
    capacity_passes = all(row["passes"] for row in capacity_rows)
    summary = {
        "schema_version": 2,
        "experiment": "Hattrick-f strict-ESM robustness",
        "formal_result": formal_result,
        "artifact_class": (
            "formal" if formal_result else "smoke" if args.smoke else "nonformal"
        ),
        "formal_validation": {
            "eligible": not formal_reasons,
            "ineligibility_reasons": formal_reasons,
            "coverage": coverage_report,
            "required_seeds": list(DEFAULT_SEEDS),
            "required_methods": list(REQUIRED_METHODS),
            "required_windows": {name: list(bounds) for name, bounds in base.WINDOWS.items()},
            "required_prediction_snapshots_per_window": 500,
            "required_capacity_offsets": list(CAPACITY_AUDIT_OFFSETS),
            "required_capacity_snapshots_per_window": len(CAPACITY_AUDIT_OFFSETS),
            "required_physical_links": 36,
        },
        "smoke": bool(args.smoke),
        "mode": args.mode,
        "topology": base.TOPOLOGY,
        "paths_per_od": base.PATHS_PER_OD,
        "load_factor": base.LOAD_FACTOR,
        "windows": {name: list(bounds) for name, bounds in windows.items()},
        "prediction_snapshot_indices": prediction_snapshots,
        "capacity_fault_snapshot_indices": capacity_snapshots,
        "capacity_fault_sampling": {
            "selection_rule": "fixed uniform offsets within each 500-snapshot window",
            "offsets": list(CAPACITY_AUDIT_OFFSETS),
            "points_per_window": len(CAPACITY_AUDIT_OFFSETS),
            "total_points_across_formal_windows": 2 * len(CAPACITY_AUDIT_OFFSETS),
            "selected_before_results": True,
            "protocol_amendment": str((HERE / "ROBUSTNESS_PROTOCOL_AMENDMENT.md").resolve()),
        },
        "seeds": [int(seed) for seed in seeds],
        "requested_families": list(args.families),
        "families": list(families),
        "strict_esm_contract": {
            "policy_actual_input": "zero placeholder; true actual absent from policy function signature",
            "actual_use": "sequential admission only",
            "oracle_files_loaded": False,
            "norm_fulfill_reported": False,
            "shape_noise": "OD lognormal, class+snapshot total demand preserved",
            "announced_derating": "policy and admission see derated capacities; capacity-aware node features rebuilt",
            "unannounced_derating": "policy sees nominal capacity; admission sees derated capacity",
            "near_outage_factor": NEAR_OUTAGE_FACTOR,
            "near_outage_is_exact_deletion": False,
            "fixed_ksp_not_recomputed": True,
            "transformer_cache": "deleted before every scenario policy forward",
            "announced_node_feature_reconstruction_tolerance": 1e-5,
        },
        "streaming": {
            "raw_row_dictionaries_retained": False,
            "csv_written_incrementally_via_atomic_temporary": True,
            "row_count": writer.row_count,
            "retained_fulfill_scalars_for_exact_percentiles": sum(
                value.count for value in online.classes.values()
            ),
            "numeric_values_checked_finite": online.numeric_values_checked,
        },
        "scenario_counts": {
            family: len(online.scenario_ids[family]) for family in FAMILIES
        },
        "physical_links": physical_inventory,
        "selected_physical_link_indices": link_indices,
        "all_36_physical_links_selected": link_indices == list(range(36)),
        "checkpoints": checkpoint_audit,
        "skipped_methods": skipped,
        "classes": online.class_rows(),
        "paired_vs_phase_a": online.pair_rows(),
        "capacity_audit": {
            "tolerance": base.CAPACITY_TOLERANCE,
            "maximum_post_admission_ratio": (
                max(row["maximum_post_admission_ratio"] for row in capacity_rows)
                if capacity_rows
                else None
            ),
            "maximum_violation": (
                max(row["maximum_violation"] for row in capacity_rows)
                if capacity_rows
                else None
            ),
            "passes": capacity_passes if capacity_rows else None,
            "online_aggregated": True,
            "by_scenario": capacity_rows,
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "source": {
            "evaluate_robustness.py": base.sha256(Path(__file__).resolve()),
            "evaluate_holdout.py": base.sha256((HERE / "evaluate_holdout.py").resolve()),
            "frozen_source_hashes": frozen_source_hashes,
        },
        "snapshot_csv": str(snapshot_csv.resolve()),
        "snapshot_csv_sha256": base.sha256(snapshot_csv),
    }
    base.atomic_json(summary_json, summary)
    return output_dir


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict-ESM class/noise/capacity robustness evaluation"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--windows",
        choices=tuple(base.WINDOWS),
        nargs="+",
        default=list(base.WINDOWS),
    )
    parser.add_argument(
        "--families", choices=FAMILIES, nargs="+", default=list(FAMILIES)
    )
    parser.add_argument(
        "--mode",
        choices=("both", "prediction", "capacity"),
        default="both",
        help="run both stages, full prediction stress only, or capacity audit only",
    )
    parser.add_argument("--links", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="explicitly permit absent method checkpoints; formal_result is false",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="small non-formal CPU/GPU audit; also permits missing checkpoints",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.max_snapshots is not None and args.max_snapshots <= 0:
        parser.error("max snapshots must be positive")
    if args.max_snapshots is not None and not args.smoke:
        parser.error("--max-snapshots is non-formal and requires explicit --smoke")
    for name, values in (
        ("seeds", args.seeds),
        ("windows", args.windows),
        ("families", args.families),
        ("links", args.links or []),
    ):
        if len(values) != len(set(values)):
            parser.error(f"duplicate --{name} values are not allowed")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    return args


def main() -> None:
    print(run(parse_cli()), flush=True)


if __name__ == "__main__":
    main()
