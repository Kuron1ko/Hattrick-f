from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import evaluate_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(path: Path, role_index: int) -> None:
    fields = list(evaluate_manifest.STANDARD_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for snapshot in range(400, 500):
            position = snapshot - 400
            for class_index, class_name in enumerate(evaluate_manifest.CLASSES):
                oracle = 30.0 + class_index * 5.0 + 0.03 * position
                demand = 42.0 + class_index * 7.0 + 0.04 * position
                base = (0.997, 0.89, 1.10)[class_index]
                shift = (0.0002, 0.012, -0.006)[class_index] * role_index
                wave = (0.001, 0.045, 0.12)[class_index] * math.sin(position / 9.0)
                norm_fulfill = base + shift + wave
                admitted = oracle * norm_fulfill
                writer.writerow(
                    {
                        "snapshot": snapshot,
                        "class": class_name,
                        "admitted_traffic": admitted,
                        "demand": demand,
                        "fulfill_ratio": admitted / demand,
                        "oracle_admitted_traffic": oracle,
                        "norm_fulfill": norm_fulfill,
                        "raw_mlu": 1.0 + class_index,
                        "oracle_mlu": 1.0 + class_index,
                        "normalized_mlu": 1.0,
                        "disabled_flow": 0.0,
                        "admitted_capacity_ratio": 1.0,
                    }
                )


def main() -> None:
    retained_root = os.environ.get("STRICT2X_STATIC_OUTPUT")
    context = (
        nullcontext(retained_root)
        if retained_root
        else tempfile.TemporaryDirectory(prefix="strict2x_three_method_")
    )
    with context as temporary:
        root = Path(temporary)
        root.mkdir(parents=True, exist_ok=True)
        evidence = root / "validation_selection.json"
        evidence.write_text(
            json.dumps(
                {
                    "split": "validation",
                    "range": [350, 400],
                    "test_metrics_read": False,
                }
            ),
            encoding="utf-8",
        )
        methods = []
        specifications = (
            ("native_hattrick", "Hattrick"),
            ("hattrick_e", "Hattrick-e"),
            ("sparse_cross_attention", "Sparse cross-attention"),
        )
        for index, (role, label) in enumerate(specifications):
            rows_path = root / f"source_{role}.csv"
            write_rows(rows_path, index)
            methods.append(
                {
                    "role": role,
                    "label": label,
                    "selection": {
                        "basis": "validation",
                        "source_range": [350, 400],
                        "rule": "synthetic fixed validation rule",
                        "test_metrics_read": False,
                        "evidence_path": str(evidence),
                        "evidence_sha256": sha256(evidence),
                    },
                    "input": {
                        "kind": "rows",
                        "path": str(rows_path),
                        "sha256": sha256(rows_path),
                    },
                }
            )
        manifest = {
            "schema_version": 1,
            "comparison_id": "synthetic-static-check",
            "selection_locked_before_evaluation": True,
            "test_metrics_used_for_selection": False,
            "bootstrap_seed": 20260824,
            "evaluation": {
                "level": 4,
                "split": "test",
                "snapshots": [400, 500],
                "expected_samples_per_class": 100,
                "load_factor": 2.0,
                "topology": "geant_priomask500_shared_load2x_train",
                "paths_per_pair": 8,
                "prediction": "esm",
                "strict_prediction_only_policy": True,
            },
            "methods": methods,
        }
        manifest_path = root / "selection_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        output_dir = root / "output"
        provenance = evaluate_manifest.run_manifest(
            manifest_path,
            output_dir,
            force=bool(retained_root),
            bootstrap_repetitions=200,
        )
        figure = output_dir / "cdf_norm_fulfill_strict2x.png"
        if figure.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError("Static-check CDF is not a PNG")
        curve_audits = provenance["plot"]["curves"].values()
        if any(
            not audit["monotone_x"]
            or not audit["monotone_y"]
            or not audit["exact_flat_zero_tail"]
            or not audit["exact_flat_one_tail"]
            or audit["left_flat_span"] <= 0
            or audit["right_flat_span"] <= 0
            for audit in curve_audits
        ):
            raise RuntimeError("A CDF failed monotonicity or flat-tail checks")
        result = {
            "status": "ok",
            "input": "synthetic rows only",
            "gpu_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "comparison_rows": provenance["rows_validation"]["rows"],
            "snapshots_per_class": provenance["rows_validation"][
                "snapshots_per_class"
            ],
            "denominator_max_delta": provenance["rows_validation"][
                "denominator_max_delta"
            ],
            "curves_checked": len(provenance["plot"]["curves"]),
            "all_curves_monotone_with_flat_tails": True,
            "figure_bytes": figure.stat().st_size,
            "artifacts_checked": sorted(provenance["artifacts"]),
        }
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
