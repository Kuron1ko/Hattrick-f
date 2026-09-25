from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import analyze_residual_lp_level4 as final_analysis
import evaluate_strict2x_common_methods as common_evaluator
import run_hattrick_residual_lp as residual
import run_hattrick_strict2x_research as research
import run_residual_lp_level4_frozen as frozen_runner


def main() -> None:
    level3_gate = (
        residual.OUTPUT_ROOT
        / research.LEVELS[3]["label"]
        / "hattrick_residual_lp_level3_gate.json"
    )
    gate = json.loads(level3_gate.read_text(encoding="utf-8"))
    if not gate.get("promote", False):
        raise RuntimeError("Level 3 did not pass; final test remains sealed")
    if [item["seed"] for item in gate.get("results", [])] != [490, 491]:
        raise RuntimeError("Level-3 gate does not contain exactly the precommitted seeds")

    sources = {
        "frameworks/hattrick_system.py": residual.ROOT / "frameworks" / "hattrick_system.py",
        "utils/build_dataset_within_cluster.py": residual.ROOT / "utils" / "build_dataset_within_cluster.py",
        "utils/training_utils.py": residual.ROOT / "utils" / "training_utils.py",
        "utils/robust_proj_utils.py": residual.ROOT / "utils" / "robust_proj_utils.py",
        "test_diff_path/run_hattrick_strict2x_research.py": Path(research.__file__).resolve(),
        "test_diff_path/run_hattrick_residual_lp.py": Path(residual.__file__).resolve(),
        "test_diff_path/freeze_residual_lp_level4.py": Path(__file__).resolve(),
        "test_diff_path/run_residual_lp_level4_frozen.py": Path(frozen_runner.__file__).resolve(),
        "test_diff_path/analyze_residual_lp_level4.py": Path(final_analysis.__file__).resolve(),
        "test_diff_path/evaluate_strict2x_common_methods.py": Path(common_evaluator.__file__).resolve(),
        "test_diff_path/run_dotemc_priority_mask_experiment.py": Path(common_evaluator.dote.__file__).resolve(),
    }
    source_hashes = {name: residual.sha256(path) for name, path in sources.items()}
    manifest = {
        "authorized": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "approach": residual.APPROACH,
        "seeds": [490, 491, 492],
        "topology": research.TOPOLOGY,
        "k": research.K,
        "train": [0, 350],
        "validation": [350, 400],
        "test": [400, 500],
        "epochs": 60,
        "batch_size": 8,
        "learning_rate": 0.0005,
        "checkpoint_rule": (
            "among exact mask/capacity checkpoints with High mean >=0.98 and final cumulative "
            "normalized pre-admission MLU <=1.05, maximize validation Medium NormFulFill P10 then "
            "P1; otherwise use and label the High-first fallback"
        ),
        "candidate_mechanism": (
            "freeze selected Hattrick High path policy; exact enabled-variable Medium then Low "
            "predicted-residual LP with zero capacity margin and scipy HiGHS"
        ),
        "metric_contract": {
            "primary": "exact replay class admitted traffic divided by incremental MF oracle",
            "capacity_mlu": "common post-admission cumulative link load/capacity",
            "pre_admission_normalized_mlu": "diagnostic only and non-comparable",
            "bootstrap": "10000 hierarchical paired draws; resample seeds then snapshots; RNG 20260815",
            "success": (
                "every seed Medium P10>=0.8235, P1>0.73, and P1 gap>0; High mean>=0.98; "
                "capacity<=1.0001; disabled flow<=1e-8; plus Medium P10 paired hierarchical "
                "95% CI lower>0"
            ),
        },
        "test_opening_policy": (
            "one frozen execution; no hyperparameter/code/statistic changes after seeing 400-499; "
            "infrastructure retries must be disclosed"
        ),
        "level3_gate_sha256": residual.sha256(level3_gate),
        "source_sha256": source_hashes,
        "comparison_artifact_sha256": {
            "dote_checkpoint": residual.sha256(common_evaluator.DOTE_CHECKPOINT),
            "swan_cache": "computed and disclosed on first authorized common evaluation",
        },
    }
    output = residual.OUTPUT_ROOT / "final_frozen_manifest.json"
    if output.exists():
        raise RuntimeError("Frozen manifest already exists; refusing to overwrite it")
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
