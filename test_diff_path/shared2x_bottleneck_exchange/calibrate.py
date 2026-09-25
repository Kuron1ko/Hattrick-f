from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

import run_experiment as runner
from frameworks.hattrick_system import Hattrick
from objectives import build_objectives
from ordered_projection import ordered_project_gradients


def project_against(vector: torch.Tensor, earlier: list[torch.Tensor]) -> torch.Tensor:
    work = vector.to(dtype=torch.float64)
    basis: list[torch.Tensor] = []
    for candidate in earlier:
        value = candidate.to(dtype=torch.float64)
        for existing in basis:
            value = value - torch.dot(value, existing) * existing
        norm = torch.linalg.vector_norm(value)
        if float(norm.item()) > 1e-12:
            basis.append(value / norm)
    for existing in basis:
        work = work - torch.dot(work, existing) * existing
    return work.to(dtype=vector.dtype)


def main() -> None:
    level = 2
    seed = 490
    output = runner.OUTPUT_ROOT / "level0_gradient_calibration"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runner.build_props(level, device)
    train, _, _ = runner.datasets_for_level(props, level)
    phase_a_path = runner.phase_a_directory(level, seed) / "best_model.pt"
    checkpoint = torch.load(phase_a_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    reference = Hattrick(props).to(device=device, dtype=props.dtype)
    reference.load_state_dict(copy.deepcopy(checkpoint["model_state_dict"]))
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    path_masks = runner.shared.base.move_dataset_static(train, props.device)
    loader = runner.shared.data_loader(train, props.batch_size, False, seed)
    rows = []
    for batch_index, inputs in enumerate(loader):
        values = runner.shared.unpack_to_device(inputs, props)
        current = runner.forward_components(model, props, train, values, path_masks)
        with torch.no_grad():
            frozen = runner.forward_components(reference, props, train, values, path_masks)
        losses, _, names, _, auxiliary = build_objectives(
            "bottleneck",
            0.005,
            **current,
            reference_medium=frozen["admitted_medium"],
            reference_low=frozen["admitted_low"],
            curriculum_scale=0.0,
            bottleneck_multiplier=1.0,
        )
        result = ordered_project_gradients(model, losses)
        high_index = names.index("HighTail")
        transfer_index = names.index("TransferNorm")
        bottleneck_index = names.index("BottleneckRelease")
        medium_index = names.index("Fm")
        earlier = [
            result.raw_gradients[high_index],
            result.raw_gradients[transfer_index],
        ]
        projected_bottleneck = project_against(
            result.raw_gradients[bottleneck_index], earlier
        )
        projected_medium = project_against(result.raw_gradients[medium_index], earlier)
        norm_bottleneck = float(torch.linalg.vector_norm(projected_bottleneck).item())
        norm_medium = float(torch.linalg.vector_norm(projected_medium).item())
        ratio = norm_medium / (norm_bottleneck + 1e-12)
        rows.append(
            {
                "batch": batch_index,
                "projected_bottleneck_norm": norm_bottleneck,
                "projected_medium_norm": norm_medium,
                "ratio_medium_to_bottleneck": ratio,
                "active_snapshot_fraction": float(
                    auxiliary["bottleneck_active_snapshot"].mean().item()
                ),
                "pressure_link_fraction": float(
                    auxiliary["bottleneck_pressure_link_fraction"].mean().item()
                ),
                "inversion_gate_mean": float(
                    auxiliary["bottleneck_inversion_gate"].mean().item()
                ),
                "weighted_low_overlap_mean": float(
                    auxiliary["bottleneck_weighted_low_overlap"].mean().item()
                ),
            }
        )
    ratios = np.asarray(
        [row["ratio_medium_to_bottleneck"] for row in rows], dtype=np.float64
    )
    if not len(rows) or not np.isfinite(ratios).all() or not (ratios > 0).all():
        raise RuntimeError("invalid bottleneck gradient calibration")
    reference_multiplier = float(np.median(ratios))
    candidates = [reference_multiplier * factor for factor in (0.25, 0.5, 1.0, 2.0)]
    result = {
        "level": level,
        "seed": seed,
        "phase_a_checkpoint": str(phase_a_path.resolve()),
        "phase_a_checkpoint_sha256": runner.sha256(phase_a_path),
        "batches": len(rows),
        "definition": "median ||P(g_Fm)|| / (||P(g_BottleneckRelease)|| + 1e-12), P removes HighTail and TransferNorm",
        "reference_multiplier": reference_multiplier,
        "candidate_multipliers": candidates,
        "rows": rows,
    }
    (output / "calibration.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
