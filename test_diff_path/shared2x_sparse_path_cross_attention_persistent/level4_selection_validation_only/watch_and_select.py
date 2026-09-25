from __future__ import annotations

"""Reuse the proven immutable watcher with a predeclared core-only selector."""

import importlib.util
import hashlib
import json
import statistics
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
METHOD_DIR = THIS_DIR.parent
TEST_DIR = METHOD_DIR.parent
BASE_WATCHER = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "checkpoint_selection_validation_only"
    / "watch_and_select.py"
)
PRECOMMIT = METHOD_DIR / "level4_precommit_manifest.json"
EXPECTED_BASE_WATCHER_SHA256 = (
    "c0e25d46ff0aedd9b1a30f66465e225db1f00815848812fc39d82e555a7d1197"
)
EXPECTED_POLICY_SHA256 = (
    "36026468706c73562c4ac61c812ca5054494f42c313fd71332f897525ea948bf"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base():
    if sha256(BASE_WATCHER) != EXPECTED_BASE_WATCHER_SHA256:
        raise RuntimeError("Base validation-only watcher changed")
    spec = importlib.util.spec_from_file_location(
        "persistent_stage2_validation_only_watcher_base", BASE_WATCHER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BASE_WATCHER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dominates(left: dict, right: dict, coordinates: tuple[str, ...]) -> bool:
    return all(left[key] >= right[key] for key in coordinates) and any(
        left[key] > right[key] for key in coordinates
    )


def main() -> None:
    base = load_base()
    base.THIS_DIR = THIS_DIR
    base.METHOD_DIR = METHOD_DIR
    base.TEST_DIR = TEST_DIR
    base.RUN_DIR = METHOD_DIR / "artifacts" / "level4_confirmation" / "seed_490"
    base.POLICY_PATH = THIS_DIR / "selection_policy.json"
    base.ARCHIVE_DIR = THIS_DIR / "archive"
    base.LIVE_MANIFEST = THIS_DIR / "manifest_live.json"
    base.FROZEN_MANIFEST = THIS_DIR / "manifest_frozen_epoch_060.json"
    base.SELECTED_CHECKPOINT = THIS_DIR / "selected_checkpoint.pt"
    base.WATCHER_STATUS = THIS_DIR / "watcher_status.json"
    if sha256(base.POLICY_PATH) != EXPECTED_POLICY_SHA256:
        raise RuntimeError("Predeclared selection policy changed")
    if not PRECOMMIT.exists():
        raise RuntimeError("Level-4 precommit manifest is missing")
    source_audit = {
        "watch_and_select.py": sha256(Path(__file__).resolve()),
        "base_watch_and_select.py": sha256(BASE_WATCHER),
        "selection_policy.json": sha256(base.POLICY_PATH),
        "level4_precommit_manifest.json": sha256(PRECOMMIT),
    }

    def select_core(history: dict, archives: list[dict], policy: dict) -> dict:
        hard = policy["hard_gates"]
        tails = policy["preferred_high_tail_gates"]
        archive_index = {int(item["epoch"]): item for item in archives}
        candidates = []
        for epoch in sorted(archive_index):
            if epoch not in history:
                continue
            summary_path, metrics_path = base.validation_paths(epoch)
            if not summary_path.exists() or not metrics_path.exists():
                continue
            row = history[epoch]
            item = {
                "epoch": epoch,
                "checkpoint": archive_index[epoch],
                "high_norm_mean": float(row["high_norm_mean"]),
                "high_norm_p1": float(row["high_norm_p1"]),
                "high_norm_p10": float(row["high_norm_p10"]),
                "medium_norm_mean": float(row["medium_norm_mean"]),
                "medium_norm_p1": float(row["medium_norm_p1"]),
                "medium_norm_p10": float(row["medium_norm_p10"]),
                "low_norm_mean": float(row["low_norm_mean"]),
                "max_post_admission_mlu": float(row["max_post_admission_mlu"]),
                "max_disabled_flow": float(row["max_disabled_flow"]),
                "validation_summary_sha256": base.sha256(summary_path),
                "validation_metrics_sha256": base.sha256(metrics_path),
                "low_paired_audit_descriptive": base.paired_low_audit(
                    metrics_path, -0.005, 0.01
                ),
            }
            item["hard_gate_pass"] = bool(
                item["high_norm_mean"] >= float(hard["high_norm_mean_min"])
                and item["max_post_admission_mlu"]
                <= float(hard["max_post_admission_mlu_max"])
                and item["max_disabled_flow"]
                <= float(hard["max_disabled_flow_max"])
            )
            item["preferred_high_tail_pass"] = bool(
                item["high_norm_p1"] >= float(tails["high_norm_p1_min"])
                and item["high_norm_p10"] >= float(tails["high_norm_p10_min"])
            )
            candidates.append(item)

        hard_pool = [item for item in candidates if item["hard_gate_pass"]]
        tail_pool = [item for item in hard_pool if item["preferred_high_tail_pass"]]
        pool = tail_pool if tail_pool else hard_pool
        coordinates = ("medium_norm_mean", "medium_norm_p1", "medium_norm_p10")
        pareto = [
            item
            for item in pool
            if not any(
                dominates(other, item, coordinates)
                for other in pool
                if other is not item
            )
        ]
        selected = None
        if pareto:
            ideal = {key: max(item[key] for item in pool) for key in coordinates}
            for item in pareto:
                ratios = [item[key] / max(ideal[key], 1e-12) for key in coordinates]
                item["medium_normalized_ratios"] = dict(zip(coordinates, ratios))
                item["medium_maximin"] = min(ratios)
                item["medium_normalized_mean"] = statistics.fmean(ratios)
            selected = max(
                pareto,
                key=lambda item: (
                    item["medium_maximin"],
                    item["medium_normalized_mean"],
                    item["medium_norm_mean"],
                    item["medium_norm_p1"],
                    item["medium_norm_p10"],
                    min(item["high_norm_p1"], item["high_norm_p10"]),
                    -item["epoch"],
                ),
            )
        return {
            "selection_tier": (
                "hard-high+preferred-high-tail+medium-pareto-maximin"
                if tail_pool
                else "hard-high+medium-pareto-maximin"
            ),
            "candidate_count": len(candidates),
            "hard_gate_count": len(hard_pool),
            "preferred_tail_count": len(tail_pool),
            "selection_pool_count": len(pool),
            "pareto_epochs": [item["epoch"] for item in pareto],
            "core_low_used_for_selection": False,
            "mandatory_downstream_low_guard": True,
            "selected": selected,
            "candidates": candidates,
        }

    original_build_manifest = base.build_manifest

    def build_manifest_with_sources(target_epoch: int, frozen: bool) -> dict:
        manifest = original_build_manifest(target_epoch, frozen)
        manifest["watcher_source_sha256"] = source_audit
        config_path = base.RUN_DIR / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("train") != [0, 350]:
                raise RuntimeError("Unexpected training split in runner config")
            if config.get("validation") != [350, 400]:
                raise RuntimeError("Unexpected validation split in runner config")
            if config.get("evaluation") != [350, 400]:
                raise RuntimeError("Runner config opened a non-validation evaluation split")
            manifest["source_run_config"] = {
                "path": str(config_path.resolve()),
                "sha256": sha256(config_path),
                "declared_source_sha256": config.get("source_sha256", {}),
            }
        return manifest

    base.select = select_core
    base.build_manifest = build_manifest_with_sources
    base.main()


if __name__ == "__main__":
    main()
