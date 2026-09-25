from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import run_experiment as runner

import evaluate_true2x_epoch13_cdf as previous
import evaluate_true2x_epoch13_vs_epoch23_cdf as plotter
import run_hattrick_strict2x_research as base
from frameworks.hattrick_system import Hattrick


START, END = 400, 500
OUTPUT_DIR = runner.OUTPUT_ROOT / "true2x_both_epoch13_comparison"
PRIOR_DIR = runner.OUTPUT_ROOT / "true2x_epoch13_comparison"
PRIOR_METRICS = PRIOR_DIR / "true2x_normfulfill_metrics.csv"
PRIOR_PROVENANCE = PRIOR_DIR / "provenance.json"
RECONSTRUCTED_RUN = (
    runner.OUTPUT_ROOT
    / "reconstructed_hattrick_epoch13"
    / "matched_level3_epoch13"
    / "baseline"
    / "seed_490"
)
HATTRICK_E13_CHECKPOINT = RECONSTRUCTED_RUN / "final_model.pt"
RECONSTRUCTION_AUDIT = RECONSTRUCTED_RUN / "reconstruction_audit.json"
TAIL_CHECKPOINT = previous.TAIL_CHECKPOINT
METHODS = (
    "Hattrick (epoch 13)",
    "BEST_MC",
    "Hattrick+Tail (epoch 13)",
)


def main() -> None:
    started = time.perf_counter()
    for path in (
        PRIOR_METRICS,
        PRIOR_PROVENANCE,
        HATTRICK_E13_CHECKPOINT,
        RECONSTRUCTION_AUDIT,
        TAIL_CHECKPOINT,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    prior_provenance = json.loads(PRIOR_PROVENANCE.read_text(encoding="utf-8"))
    reconstruction = json.loads(RECONSTRUCTION_AUDIT.read_text(encoding="utf-8"))
    if prior_provenance.get("window") != [START, END]:
        raise RuntimeError("Prior BEST_MC/Tail cache does not use 400-499")
    if not reconstruction.get("passes", False):
        raise RuntimeError("Baseline epoch-13 reconstruction audit did not pass")
    if reconstruction.get("checkpoint_sha256") != runner.sha256(HATTRICK_E13_CHECKPOINT):
        raise RuntimeError("Reconstructed Hattrick epoch-13 checkpoint hash changed")
    if prior_provenance.get("tail_checkpoint_sha256") != runner.sha256(TAIL_CHECKPOINT):
        raise RuntimeError("Tail checkpoint hash differs from prior common evaluation")

    reused_rows: list[dict] = []
    for row in runner.read_csv(PRIOR_METRICS):
        if row["method"] in ("BEST_MC", "Hattrick+Tail (epoch 13)"):
            reused_rows.append(dict(row))
    if len(reused_rows) != 600:
        raise RuntimeError(f"Expected 600 cached BEST_MC/Tail rows, got {len(reused_rows)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runner.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    dataset = runner.DM_Dataset_within_Cluster(props, 0, START, END)
    if int(dataset.max_source_index_read) != END - 1:
        raise RuntimeError("Evaluator read outside the 400-499 held-out test window")
    if base.move_dataset_static(dataset, device) is not None:
        raise RuntimeError("Shared-path dataset unexpectedly contains class masks")

    checkpoint = torch.load(HATTRICK_E13_CHECKPOINT, map_location=device, weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 13:
        raise RuntimeError(f"Expected Hattrick epoch 13, found {checkpoint.get('epoch')}")
    model = Hattrick(props).to(device=device, dtype=props.dtype).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    hattrick_rows, _ = base.evaluate(model, props, dataset, START)
    for row in hattrick_rows:
        row["method"] = "Hattrick (epoch 13)"

    rows = hattrick_rows + reused_rows
    if len(rows) != len(METHODS) * len(runner.CLASSES) * (END - START):
        raise RuntimeError(f"Unexpected result row count: {len(rows)}")
    plotter.METHODS = METHODS
    summary = plotter.summarize(rows)
    inversions = plotter.inversion_summary(rows)
    max_capacity = max(float(row["admitted_capacity_ratio"]) for row in rows)
    max_disabled = max(float(row["disabled_flow"]) for row in rows)
    if max_capacity > 1.0001:
        raise RuntimeError(f"Capacity assertion failed: {max_capacity}")
    if max_disabled > 1e-8:
        raise RuntimeError(f"Disabled-flow assertion failed: {max_disabled}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner.write_csv(OUTPUT_DIR / "true2x_both_epoch13_metrics.csv", rows)
    runner.write_json(OUTPUT_DIR / "true2x_both_epoch13_summary.json", summary)
    runner.write_json(OUTPUT_DIR / "true2x_both_epoch13_inversion.json", inversions)
    plotter.plot_smooth_cdf(
        rows,
        summary,
        OUTPUT_DIR / "true2x_both_epoch13_smooth_cdf.png",
    )
    runner.write_json(
        OUTPUT_DIR / "provenance.json",
        {
            "scenario": "GEANT shared paths, K=8, true 2x load",
            "window": [START, END],
            "window_role": "held-out test; explicitly authorized by the user",
            "metric_contract": prior_provenance["metric_contract"],
            "plot_contract": {
                "curve": "Gaussian-kernel smoothed CDF; display only",
                "fixed_domains": plotter.DOMAINS,
                "censoring": (
                    "values outside fixed reference axes are censored for display only; "
                    "raw CSV and summary remain uncensored"
                ),
            },
            "dataset_max_source_index_read": int(dataset.max_source_index_read),
            "hattrick_epoch13_checkpoint": str(HATTRICK_E13_CHECKPOINT),
            "hattrick_epoch13_sha256": runner.sha256(HATTRICK_E13_CHECKPOINT),
            "hattrick_epoch13_reconstruction_audit": str(RECONSTRUCTION_AUDIT),
            "hattrick_epoch13_reconstruction_audit_sha256": runner.sha256(
                RECONSTRUCTION_AUDIT
            ),
            "tail_epoch13_checkpoint": str(TAIL_CHECKPOINT),
            "tail_epoch13_sha256": runner.sha256(TAIL_CHECKPOINT),
            "reused_common_metrics": str(PRIOR_METRICS),
            "reused_common_metrics_sha256": runner.sha256(PRIOR_METRICS),
            "max_common_post_admission_mlu": max_capacity,
            "max_disabled_flow": max_disabled,
            "runtime_seconds": float(time.perf_counter() - started),
            "source_sha256": runner.sha256(Path(__file__).resolve()),
        },
    )
    print(json.dumps({"summary": summary, "inversions": inversions}, indent=2), flush=True)


if __name__ == "__main__":
    main()
