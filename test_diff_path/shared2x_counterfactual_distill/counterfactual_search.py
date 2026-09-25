from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class Admission:
    ratios: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    flows: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    totals: torch.Tensor


@dataclass
class SearchResult:
    initial_policies: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    target_policies: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    initial: Admission
    target: Admission
    accepted_rounds: int
    attempted_candidates: int
    feasible_candidates: int
    policy_l1: float
    high_floor: float


def _pte_info(path_to_edge: torch.Tensor) -> list[torch.Tensor]:
    matrix = path_to_edge.coalesce()
    indices = matrix.indices()
    return [matrix, indices[0], indices[1], matrix.values()]


def masked_policy(logits: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    batch = logits.shape[0]
    shaped = logits.reshape(batch, -1, k)
    valid = mask.reshape(1, -1, k).to(device=logits.device, dtype=torch.bool)
    if not valid.any(dim=-1).all().item():
        raise ValueError("every OD pair must retain at least one valid path")
    shaped = shaped.masked_fill(~valid, torch.finfo(logits.dtype).min)
    return torch.softmax(shaped, dim=-1).reshape(batch, -1, 1)


def policy_logits(policy: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    values = policy.reshape(policy.shape[0], -1).clamp_min(1e-12).log()
    valid = mask.reshape(1, -1).to(device=values.device, dtype=torch.bool)
    values = values.masked_fill(~valid, -30.0)
    # Remove the irrelevant per-OD softmax offset for stable distances.
    shaped = values.reshape(values.shape[0], -1, k)
    shaped = shaped - shaped.mean(dim=-1, keepdim=True)
    return shaped.reshape(values.shape[0], -1)


def simulate(
    model,
    props,
    policies: Sequence[torch.Tensor],
    tms: Sequence[torch.Tensor],
    capacities: torch.Tensor,
    path_to_edge: torch.Tensor,
) -> Admission:
    batch = tms[0].shape[0]
    ratios = model.simulate(
        list(policies),
        list(tms),
        capacities,
        _pte_info(path_to_edge),
        batch,
        props,
        rate_cap=props.rate_cap,
    )[:3]
    flows = tuple(
        ratio * tm.squeeze(-1) for ratio, tm in zip(ratios, tms)
    )
    totals = torch.stack(
        [flow.reshape(batch, -1).sum(dim=1) for flow in flows], dim=1
    )
    return Admission(tuple(ratios), flows, totals)


def _flatten(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values])


def _ascent_direction(
    logits: Sequence[torch.Tensor],
    medium_total: torch.Tensor,
    high_total: torch.Tensor,
    *,
    protect_high: bool,
) -> list[torch.Tensor]:
    medium_grad = torch.autograd.grad(
        medium_total,
        logits,
        retain_graph=True,
        allow_unused=True,
    )
    high_grad = torch.autograd.grad(
        high_total,
        logits,
        retain_graph=False,
        allow_unused=True,
    )
    medium_values = [
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(logits, medium_grad)
    ]
    high_values = [
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(logits, high_grad)
    ]
    medium_flat = _flatten(medium_values)
    high_flat = _flatten(high_values)
    if protect_high:
        conflict = torch.dot(high_flat, medium_flat)
        denominator = torch.dot(high_flat, high_flat)
        if float(conflict.item()) < 0.0 and float(denominator.item()) > 1e-20:
            medium_flat = medium_flat - conflict / denominator * high_flat
    norm = torch.linalg.vector_norm(medium_flat)
    if not torch.isfinite(norm).item() or float(norm.item()) <= 1e-12:
        return [torch.zeros_like(value) for value in logits]
    # A unit L2 direction makes the finite logit step grid comparable between
    # snapshots without introducing a loss multiplier.
    medium_flat = medium_flat / norm
    result = []
    offset = 0
    for value in logits:
        count = value.numel()
        result.append(medium_flat[offset : offset + count].reshape_as(value))
        offset += count
    return result


def _ascent_direction_per_snapshot(
    logits: Sequence[torch.Tensor],
    medium_total: torch.Tensor,
    high_total: torch.Tensor,
) -> list[torch.Tensor]:
    """High-safe unit directions without coupling samples in a batch."""
    medium_grad = torch.autograd.grad(
        medium_total.sum(), logits, retain_graph=True, allow_unused=True
    )
    high_grad = torch.autograd.grad(
        high_total.sum(), logits, retain_graph=False, allow_unused=True
    )
    medium_values = [
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(logits, medium_grad)
    ]
    high_values = [
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(logits, high_grad)
    ]
    batch = logits[0].shape[0]
    medium_flat = torch.cat([value.reshape(batch, -1) for value in medium_values], dim=1)
    high_flat = torch.cat([value.reshape(batch, -1) for value in high_values], dim=1)
    conflict = (high_flat * medium_flat).sum(dim=1, keepdim=True)
    denominator = (high_flat * high_flat).sum(dim=1, keepdim=True)
    coefficient = torch.where(
        (conflict < 0.0) & (denominator > 1e-20),
        conflict / denominator.clamp_min(1e-20),
        torch.zeros_like(conflict),
    )
    medium_flat = medium_flat - coefficient * high_flat
    norm = torch.linalg.vector_norm(medium_flat, dim=1, keepdim=True)
    valid = torch.isfinite(norm) & (norm > 1e-12)
    medium_flat = torch.where(
        valid, medium_flat / norm.clamp_min(1e-12), torch.zeros_like(medium_flat)
    )
    result = []
    offset = 0
    for value in logits:
        count = value[0].numel()
        result.append(medium_flat[:, offset : offset + count].reshape_as(value))
        offset += count
    return result


def _slice_admission(admission: Admission, index: int) -> Admission:
    return Admission(
        tuple(value[index : index + 1] for value in admission.ratios),
        tuple(value[index : index + 1] for value in admission.flows),
        admission.totals[index : index + 1],
    )


def _coordinate_candidates(
    policies: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    masks: Sequence[torch.Tensor],
    k: int,
    *,
    candidate_limit: int,
    move_fractions: Sequence[float],
) -> list[tuple[int, int, int, int, float]]:
    """Rank one-OD finite path swaps by the local ascent direction."""
    proposals = []
    for class_index in (0, 1):
        policy = policies[class_index].reshape(-1, k)
        score = direction[class_index].reshape(-1, k)
        valid = masks[class_index].reshape(-1, k).to(dtype=torch.bool)
        for pair in range(policy.shape[0]):
            valid_ids = torch.nonzero(valid[pair], as_tuple=False).reshape(-1)
            if valid_ids.numel() < 2:
                continue
            pair_scores = score[pair, valid_ids]
            target = int(valid_ids[int(torch.argmax(pair_scores).item())].item())
            sources = valid_ids[policy[pair, valid_ids] > 1e-8]
            if sources.numel() == 0:
                continue
            source_scores = score[pair, sources]
            source = int(sources[int(torch.argmin(source_scores).item())].item())
            if source == target:
                continue
            available = float(policy[pair, source].item())
            gradient_gap = float((score[pair, target] - score[pair, source]).item())
            for fraction in move_fractions:
                amount = min(available, float(fraction))
                if amount <= 1e-8:
                    continue
                proposals.append(
                    (
                        gradient_gap * amount,
                        class_index,
                        pair,
                        source,
                        target,
                        amount,
                    )
                )
    proposals.sort(key=lambda item: item[0], reverse=True)
    return [tuple(item[1:]) for item in proposals[:candidate_limit]]


def _best_coordinate_swap(
    model,
    props,
    policies: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    path_masks: Sequence[torch.Tensor],
    tms: Sequence[torch.Tensor],
    capacities: torch.Tensor,
    path_to_edge: torch.Tensor,
    current: Admission,
    original: Sequence[torch.Tensor],
    high_floor: float,
    *,
    candidate_limit: int,
    move_fractions: Sequence[float],
    improvement_tolerance: float,
) -> tuple[tuple[torch.Tensor, ...], Admission, int, int] | None:
    k = int(props.num_paths_per_pair)
    proposals = _coordinate_candidates(
        policies,
        direction,
        path_masks,
        k,
        candidate_limit=candidate_limit,
        move_fractions=move_fractions,
    )
    if not proposals:
        return None
    batches = [
        policy.detach().repeat(len(proposals), 1, 1) for policy in policies
    ]
    for candidate_index, (class_index, pair, source, target, amount) in enumerate(proposals):
        offset = pair * k
        batches[class_index][candidate_index, offset + source, 0] -= amount
        batches[class_index][candidate_index, offset + target, 0] += amount
    expanded_tms = [value.expand(len(proposals), -1, -1) for value in tms]
    expanded_capacities = capacities.expand(len(proposals), -1)
    with torch.no_grad():
        admission = simulate(
            model,
            props,
            batches,
            expanded_tms,
            expanded_capacities,
            path_to_edge,
        )
    high = admission.totals[:, 0]
    medium = admission.totals[:, 1]
    feasible_mask = high + 1e-6 >= high_floor
    improved_mask = medium > current.totals[0, 1] + improvement_tolerance
    eligible = torch.nonzero(feasible_mask & improved_mask, as_tuple=False).reshape(-1)
    if eligible.numel() == 0:
        return None
    ranked = []
    for value in eligible.tolist():
        candidate_policies = tuple(batch[value : value + 1] for batch in batches)
        distance = float(
            sum(
                torch.abs(left - right).sum().item()
                for left, right in zip(candidate_policies, original)
            )
        )
        ranked.append(
            (
                float(admission.totals[value, 1].item()),
                float(admission.totals[value, 2].item()),
                -distance,
                value,
            )
        )
    chosen = max(ranked)[:]
    index = int(chosen[3])
    selected_policies = tuple(batch[index : index + 1] for batch in batches)
    return (
        selected_policies,
        _slice_admission(admission, index),
        len(proposals),
        int(feasible_mask.sum().item()),
    )


def search_counterfactual(
    model,
    props,
    initial_policies: Sequence[torch.Tensor],
    path_masks: Sequence[torch.Tensor],
    tms: Sequence[torch.Tensor],
    capacities: torch.Tensor,
    path_to_edge: torch.Tensor,
    oracle_high: torch.Tensor,
    *,
    high_preservation_tolerance: float = 1e-4,
    rounds: int = 8,
    step_grid: Sequence[float] = (8.0, 4.0, 2.0, 1.0, 0.5, 0.25),
    improvement_tolerance: float = 1e-7,
    coordinate_rounds: int = 4,
    coordinate_candidate_limit: int = 128,
    coordinate_move_fractions: Sequence[float] = (0.4, 0.2, 0.1, 0.05),
) -> SearchResult:
    """Search policy space; accept only exact High-feasible Medium improvements."""
    if initial_policies[0].shape[0] != 1:
        raise ValueError("counterfactual search is intentionally per snapshot")
    if rounds <= 0 or not step_grid:
        raise ValueError("search requires positive rounds and a non-empty step grid")
    k = int(props.num_paths_per_pair)
    original = tuple(policy.detach().clone() for policy in initial_policies)
    current_logits = [
        policy_logits(policy, mask, k).detach()
        for policy, mask in zip(original, path_masks)
    ]
    with torch.no_grad():
        initial = simulate(
            model, props, original, tms, capacities, path_to_edge
        )
    oracle_value = float(oracle_high.detach().reshape(-1)[0].item())
    if not torch.isfinite(torch.tensor(oracle_value)).item() or oracle_value <= 0:
        raise ValueError("oracle High flow must be finite and positive")
    if high_preservation_tolerance < 0:
        raise ValueError("High preservation tolerance must be non-negative")
    # The counterfactual question is whether B can preserve the High admission
    # already achieved by A.  Using the oracle epsilon floor here would reject
    # an A that is slightly below that external target before search even
    # begins.  Final model evaluation still reports the oracle-normalized High
    # guard separately.
    high_floor = float(initial.totals[0, 0].item()) - (
        float(high_preservation_tolerance) * oracle_value
    )
    current = initial
    accepted_rounds = 0
    attempted = 0
    feasible = 0
    for _ in range(rounds):
        variables = [value.detach().clone().requires_grad_(True) for value in current_logits]
        policies = tuple(
            masked_policy(value, mask, k)
            for value, mask in zip(variables, path_masks)
        )
        admission = simulate(
            model, props, policies, tms, capacities, path_to_edge
        )
        protect_high = float(admission.totals[0, 0].item()) <= high_floor + 1e-3
        direction = _ascent_direction(
            variables,
            admission.totals[0, 1],
            admission.totals[0, 0],
            protect_high=protect_high,
        )
        if float(torch.linalg.vector_norm(_flatten(direction)).item()) <= 1e-12:
            break
        candidates = []
        with torch.no_grad():
            for step in step_grid:
                attempted += 1
                candidate_logits = [
                    value.detach() + float(step) * delta.detach()
                    for value, delta in zip(variables, direction)
                ]
                candidate_policies = tuple(
                    masked_policy(value, mask, k)
                    for value, mask in zip(candidate_logits, path_masks)
                )
                candidate = simulate(
                    model,
                    props,
                    candidate_policies,
                    tms,
                    capacities,
                    path_to_edge,
                )
                if float(candidate.totals[0, 0].item()) + 1e-6 < high_floor:
                    continue
                feasible += 1
                medium_gain = float(
                    candidate.totals[0, 1].item() - current.totals[0, 1].item()
                )
                if medium_gain <= improvement_tolerance:
                    continue
                policy_distance = float(
                    sum(
                        torch.abs(left - right).sum().item()
                        for left, right in zip(candidate_policies, original)
                    )
                )
                candidates.append(
                    (
                        float(candidate.totals[0, 1].item()),
                        float(candidate.totals[0, 2].item()),
                        -policy_distance,
                        candidate_logits,
                        candidate_policies,
                        candidate,
                    )
                )
        if not candidates:
            break
        chosen = max(candidates, key=lambda item: item[:3])
        current_logits = [value.detach() for value in chosen[3]]
        current = chosen[5]
        accepted_rounds += 1
    # The simulator is piecewise and can have a zero/blocked local gradient.
    # Finite one-OD path swaps can cross such a boundary while exact replay
    # continues to enforce High admission.
    for _ in range(coordinate_rounds):
        variables = [value.detach().clone().requires_grad_(True) for value in current_logits]
        policies = tuple(
            masked_policy(value, mask, k)
            for value, mask in zip(variables, path_masks)
        )
        admission = simulate(model, props, policies, tms, capacities, path_to_edge)
        direction = _ascent_direction(
            variables,
            admission.totals[0, 1],
            admission.totals[0, 0],
            protect_high=True,
        )
        selected = _best_coordinate_swap(
            model,
            props,
            policies,
            direction,
            path_masks,
            tms,
            capacities,
            path_to_edge,
            current,
            original,
            high_floor,
            candidate_limit=coordinate_candidate_limit,
            move_fractions=coordinate_move_fractions,
            improvement_tolerance=improvement_tolerance,
        )
        if selected is None:
            break
        selected_policies, current, current_attempted, current_feasible = selected
        attempted += current_attempted
        feasible += current_feasible
        current_logits = [
            policy_logits(policy, mask, k).detach()
            for policy, mask in zip(selected_policies, path_masks)
        ]
        accepted_rounds += 1
    target_policies = tuple(
        masked_policy(value, mask, k).detach()
        for value, mask in zip(current_logits, path_masks)
    )
    policy_l1 = float(
        sum(torch.abs(left - right).sum().item() for left, right in zip(target_policies, original))
    )
    return SearchResult(
        initial_policies=original,
        target_policies=target_policies,
        initial=initial,
        target=current,
        accepted_rounds=accepted_rounds,
        attempted_candidates=attempted,
        feasible_candidates=feasible,
        policy_l1=policy_l1,
        high_floor=high_floor,
    )


def search_counterfactual_batch(
    model,
    props,
    initial_policies: Sequence[torch.Tensor],
    path_masks: Sequence[torch.Tensor],
    tms: Sequence[torch.Tensor],
    capacities: torch.Tensor,
    path_to_edge: torch.Tensor,
    oracle_high: torch.Tensor,
    *,
    high_preservation_tolerance: float = 1e-4,
    coordinate_rounds: int = 2,
    coordinate_candidate_limit: int = 32,
    coordinate_move_fractions: Sequence[float] = (0.4, 0.1),
    improvement_tolerance: float = 1e-7,
) -> list[SearchResult]:
    """Vectorized finite-swap search with exact per-snapshot acceptance.

    Candidate generation and ranking remain independent for each snapshot;
    only simulator calls are fused into larger GPU batches.
    """
    batch = initial_policies[0].shape[0]
    if batch <= 0:
        return []
    if high_preservation_tolerance < 0:
        raise ValueError("High preservation tolerance must be non-negative")
    k = int(props.num_paths_per_pair)
    original = tuple(value.detach().clone() for value in initial_policies)
    current_policies = tuple(value.detach().clone() for value in original)
    with torch.no_grad():
        initial = simulate(
            model, props, original, tms, capacities, path_to_edge
        )
    oracle = oracle_high.detach().reshape(batch, -1)[:, 0]
    if (not torch.isfinite(oracle).all().item()) or (oracle <= 0).any().item():
        raise ValueError("oracle High flow must be finite and positive")
    high_floor = initial.totals[:, 0] - high_preservation_tolerance * oracle
    current = initial
    accepted = [0 for _ in range(batch)]
    attempted = [0 for _ in range(batch)]
    feasible = [0 for _ in range(batch)]

    for _ in range(coordinate_rounds):
        variables = [
            policy_logits(value, mask, k).detach().requires_grad_(True)
            for value, mask in zip(current_policies, path_masks)
        ]
        differentiable = tuple(
            masked_policy(value, mask, k)
            for value, mask in zip(variables, path_masks)
        )
        admission = simulate(
            model, props, differentiable, tms, capacities, path_to_edge
        )
        direction = _ascent_direction_per_snapshot(
            variables, admission.totals[:, 1], admission.totals[:, 0]
        )

        owners: list[int] = []
        proposal_specs: list[tuple[int, int, int, int, float]] = []
        for sample in range(batch):
            proposals = _coordinate_candidates(
                tuple(value[sample : sample + 1] for value in differentiable),
                tuple(value[sample : sample + 1] for value in direction),
                path_masks,
                k,
                candidate_limit=coordinate_candidate_limit,
                move_fractions=coordinate_move_fractions,
            )
            attempted[sample] += len(proposals)
            owners.extend([sample] * len(proposals))
            proposal_specs.extend(proposals)
        if not owners:
            break

        owner_tensor = torch.tensor(owners, device=capacities.device, dtype=torch.long)
        candidates = [value.detach()[owner_tensor].clone() for value in differentiable]
        for candidate, (class_index, pair, source, target, amount) in enumerate(
            proposal_specs
        ):
            offset = pair * k
            candidates[class_index][candidate, offset + source, 0] -= amount
            candidates[class_index][candidate, offset + target, 0] += amount
        candidate_tms = [value[owner_tensor] for value in tms]
        capacity_rows = (
            capacities.expand(batch, -1)
            if capacities.shape[0] == 1 and batch > 1
            else capacities
        )
        candidate_capacities = capacity_rows[owner_tensor]
        with torch.no_grad():
            candidate_admission = simulate(
                model,
                props,
                candidates,
                candidate_tms,
                candidate_capacities,
                path_to_edge,
            )

        updated = [value.detach().clone() for value in current_policies]
        any_update = False
        for sample in range(batch):
            ids = torch.nonzero(owner_tensor == sample, as_tuple=False).reshape(-1)
            is_feasible = (
                candidate_admission.totals[ids, 0] + 1e-6 >= high_floor[sample]
            )
            feasible[sample] += int(is_feasible.sum().item())
            improved = (
                candidate_admission.totals[ids, 1]
                > current.totals[sample, 1] + improvement_tolerance
            )
            eligible = ids[is_feasible & improved]
            if eligible.numel() == 0:
                continue
            ranked = []
            for candidate in eligible.tolist():
                distance = float(
                    sum(
                        torch.abs(
                            value[candidate : candidate + 1]
                            - source[sample : sample + 1]
                        ).sum().item()
                        for value, source in zip(candidates, original)
                    )
                )
                ranked.append(
                    (
                        float(candidate_admission.totals[candidate, 1].item()),
                        float(candidate_admission.totals[candidate, 2].item()),
                        -distance,
                        candidate,
                    )
                )
            chosen = int(max(ranked)[3])
            for class_index in range(3):
                updated[class_index][sample] = candidates[class_index][chosen]
            accepted[sample] += 1
            any_update = True
        if not any_update:
            break
        current_policies = tuple(updated)
        with torch.no_grad():
            current = simulate(
                model, props, current_policies, tms, capacities, path_to_edge
            )

    results = []
    for sample in range(batch):
        initial_policy = tuple(value[sample : sample + 1] for value in original)
        target_policy = tuple(value[sample : sample + 1] for value in current_policies)
        distance = float(
            sum(
                torch.abs(left - right).sum().item()
                for left, right in zip(target_policy, initial_policy)
            )
        )
        results.append(
            SearchResult(
                initial_policies=initial_policy,
                target_policies=target_policy,
                initial=_slice_admission(initial, sample),
                target=_slice_admission(current, sample),
                accepted_rounds=accepted[sample],
                attempted_candidates=attempted[sample],
                feasible_candidates=feasible[sample],
                policy_l1=distance,
                high_floor=float(high_floor[sample].item()),
            )
        )
    return results


def distillation_loss(
    predicted: Sequence[torch.Tensor],
    target: Sequence[torch.Tensor],
    path_masks: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Masked forward KL; target is a detached feasible counterfactual."""
    losses = []
    for prediction, teacher, mask in zip(predicted, target, path_masks):
        valid = mask.reshape(1, -1, 1).to(device=prediction.device, dtype=torch.bool)
        p = prediction.clamp_min(1e-12)
        q = teacher.detach().clamp_min(1e-12)
        term = torch.where(valid, q * (q.log() - p.log()), torch.zeros_like(p))
        # Each OD contributes one categorical distribution. Normalize by the
        # number of ODs so the loss scale does not grow with topology size.
        # Infer pair count from the target: every OD distribution sums to one.
        pair_count = q.reshape(q.shape[0], -1).sum(dim=1).detach().mean().clamp_min(1.0)
        losses.append(term.sum(dim=1).mean() / pair_count)
    return torch.stack(losses).sum()


def changed_od_distillation_loss(
    predicted: Sequence[torch.Tensor],
    target: Sequence[torch.Tensor],
    initial: Sequence[torch.Tensor],
    path_masks: Sequence[torch.Tensor],
    k: int,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Forward KL only on ODs changed by the verified counterfactual.

    This is not a tunable importance weight: the teacher's accepted finite
    path swaps define the supervised set.  Unchanged OD pairs contribute no
    loss, avoiding dilution by hundreds of already-correct distributions.
    """
    losses = []
    for prediction, teacher, source, mask in zip(
        predicted, target, initial, path_masks
    ):
        batch = prediction.shape[0]
        p = prediction.reshape(batch, -1, k).clamp_min(1e-12)
        q = teacher.detach().reshape(batch, -1, k).clamp_min(1e-12)
        r = source.detach().reshape(batch, -1, k)
        valid_path = mask.reshape(1, -1, k).to(
            device=prediction.device, dtype=torch.bool
        )
        valid_pair = valid_path.any(dim=-1)
        changed = ((q - r).abs().sum(dim=-1) > tolerance) & valid_pair
        per_pair = torch.where(
            valid_path,
            q * (q.log() - p.log()),
            torch.zeros_like(p),
        ).sum(dim=-1)
        count = changed.sum().clamp_min(1)
        losses.append((per_pair * changed).sum() / count)
    return torch.stack(losses).sum()


def guarded_changed_od_distillation_loss(
    predicted: Sequence[torch.Tensor],
    target: Sequence[torch.Tensor],
    initial: Sequence[torch.Tensor],
    path_masks: Sequence[torch.Tensor],
    k: int,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Anchor all High/Low ODs while focusing Medium on accepted swaps.

    Every class is normalized by its own supervised OD count, so this adds no
    manually tuned class multiplier.  High can still learn teacher changes;
    Low remains trainable but is anchored because search does not alter it.
    """
    losses = []
    for class_index, (prediction, teacher, source, mask) in enumerate(
        zip(predicted, target, initial, path_masks)
    ):
        batch = prediction.shape[0]
        p = prediction.reshape(batch, -1, k).clamp_min(1e-12)
        q = teacher.detach().reshape(batch, -1, k).clamp_min(1e-12)
        r = source.detach().reshape(batch, -1, k)
        valid_path = mask.reshape(1, -1, k).to(
            device=prediction.device, dtype=torch.bool
        )
        valid_pair = valid_path.any(dim=-1).expand(batch, -1)
        if class_index == 1:
            selected = ((q - r).abs().sum(dim=-1) > tolerance) & valid_pair
        else:
            selected = valid_pair
        per_pair = torch.where(
            valid_path,
            q * (q.log() - p.log()),
            torch.zeros_like(p),
        ).sum(dim=-1)
        losses.append((per_pair * selected).sum() / selected.sum().clamp_min(1))
    return torch.stack(losses).sum()
