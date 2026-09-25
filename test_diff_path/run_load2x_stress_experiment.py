from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.append(str(THIS_DIR))
sys.path.append(str(ROOT))

import run_dotemc_priority_mask_experiment as dm
import run_priority_mask_experiment as pm


K = 8
SOURCE_START = 400
SOURCE_END = 500
TEST_END = SOURCE_END - SOURCE_START
LOAD_FACTOR = 2.0
CLASSES = ("High", "Medium", "Low")
SOURCE_TOPOS = {
    "shared": "geant_priomask500_shared",
    "mild": "geant_priomask500_mild",
    "medium": "geant_priomask500_medium",
    "strict": "geant_priomask500_strict",
}
SCENARIOS = {
    scenario: f"{topo}_load2x"
    for scenario, topo in SOURCE_TOPOS.items()
}

RESULTS = THIS_DIR / "results_load2x_stress_500slice"
LOGS = THIS_DIR / "logs_load2x_stress_500slice"
STATE_FILE = THIS_DIR / "load2x_stress_state.json"
ORIGINAL_RESULTS = THIS_DIR / "results_dotemc_priority_masks_500slice"
BASELINE_SUMMARY = ORIGINAL_RESULTS / "long500_summary_stats.csv"
DEFAULT_WEIGHT_LABEL = "w_1_0p01_0p001"
SENS_WEIGHT_LABEL = "w_1_0p1_0p01"


def configure_modules() -> None:
    pm.RESULTS = RESULTS
    pm.LOGS = LOGS / "hattrick_gurobi"
    pm.STATE_FILE = STATE_FILE
    pm.TRAIN_END = 0
    pm.VAL_END = 0
    pm.TEST_END = TEST_END
    pm.SCENARIOS = SCENARIOS

    dm.RESULTS = RESULTS
    dm.OUTPUTS = RESULTS / "outputs"
    dm.LOGS = LOGS / "dotemc"
    dm.TRAIN_END = 0
    dm.VAL_END = 0
    dm.TEST_END = TEST_END
    dm.SCENARIOS = SCENARIOS


def ensure_dirs() -> None:
    for directory in (RESULTS, LOGS, LOGS / "hattrick_gurobi", LOGS / "dotemc"):
        directory.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    else:
        state = {}
    state.setdefault("prepare", False)
    state.setdefault("gurobi", {})
    state.setdefault("hattrick", {})
    state.setdefault("dotemc", {})
    state.setdefault("report", False)
    return state


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def manifest_rows(topo: str) -> list[tuple[str, str, str]]:
    path = ROOT / "manifest" / f"{topo}_manifest.txt"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            topology, pairs, tm = [part.strip() for part in line.split(",")]
            rows.append((topology, pairs, tm))
    return rows


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def scale_pickle(source: Path, target: Path) -> tuple[float, float]:
    with source.open("rb") as handle:
        original = np.asarray(pickle.load(handle))
    scaled = original * LOAD_FACTOR
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(scaled, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return float(original.sum()), float(np.asarray(scaled).sum())


def copy_path_artifacts(source_topo: str, target_topo: str, scenario: str) -> None:
    copies = (
        (
            ROOT / "topologies" / "paths_dict" / f"{source_topo}_{K}_paths_dict_cluster_0.pkl",
            ROOT / "topologies" / "paths_dict" / f"{target_topo}_{K}_paths_dict_cluster_0.pkl",
        ),
        (
            ROOT / "topologies" / "paths" / f"{source_topo}_{K}_paths_cluster_0.pkl",
            ROOT / "topologies" / "paths" / f"{target_topo}_{K}_paths_cluster_0.pkl",
        ),
        (
            ROOT / "topologies" / "padded_edge_ids_per_path" / f"{source_topo}_{K}_paths_cluster_0_padded_edge_ids_per_path.pkl",
            ROOT / "topologies" / "padded_edge_ids_per_path" / f"{target_topo}_{K}_paths_cluster_0_padded_edge_ids_per_path.pkl",
        ),
        (
            ROOT / "topologies" / "padded_edge_ids_per_path" / f"{source_topo}_{K}_paths_cluster_0_edge_ids_dict.pkl",
            ROOT / "topologies" / "padded_edge_ids_per_path" / f"{target_topo}_{K}_paths_cluster_0_edge_ids_dict.pkl",
        ),
    )
    for source, target in copies:
        copy_if_exists(source, target)

    if scenario != "shared":
        copy_if_exists(
            ROOT / "topologies" / "path_masks" / f"{source_topo}_{K}_path_masks_cluster_0.pkl",
            ROOT / "topologies" / "path_masks" / f"{target_topo}_{K}_path_masks_cluster_0.pkl",
        )


def prepare() -> None:
    configure_modules()
    ensure_dirs()
    state = load_state()
    if state["prepare"]:
        print("[skip] load2x data already prepared", flush=True)
        validate_data()
        return

    scale_rows = []
    for scenario, source_topo in SOURCE_TOPOS.items():
        target_topo = SCENARIOS[scenario]
        rows = manifest_rows(source_topo)[SOURCE_START:SOURCE_END]
        if len(rows) != TEST_END:
            raise RuntimeError(f"{source_topo} has {len(rows)} selected rows, expected {TEST_END}")

        copy_path_artifacts(source_topo, target_topo, scenario)
        seen_topologies: set[str] = set()
        seen_pairs: set[str] = set()
        seen_tms: set[str] = set()
        manifest_lines = []
        for topology_filename, pairs_filename, tm_filename in rows:
            manifest_lines.append(f"{topology_filename},{pairs_filename},{tm_filename}\n")
            if topology_filename not in seen_topologies:
                copy_if_exists(
                    ROOT / "topologies" / source_topo / topology_filename,
                    ROOT / "topologies" / target_topo / topology_filename,
                )
                seen_topologies.add(topology_filename)
            if pairs_filename not in seen_pairs:
                copy_if_exists(
                    ROOT / "pairs" / source_topo / pairs_filename,
                    ROOT / "pairs" / target_topo / pairs_filename,
                )
                seen_pairs.add(pairs_filename)
            if tm_filename in seen_tms:
                continue
            for priority in (1, 2, 3):
                for suffix in ("", "_esm"):
                    source = ROOT / "traffic_matrices" / f"{source_topo}_{priority}{suffix}" / tm_filename
                    target = ROOT / "traffic_matrices" / f"{target_topo}_{priority}{suffix}" / tm_filename
                    original_sum, scaled_sum = scale_pickle(source, target)
                    scale_rows.append(
                        {
                            "scenario": scenario,
                            "source_snapshot": tm_filename,
                            "priority": priority,
                            "matrix": "predicted_esm" if suffix else "ground_truth",
                            "original_sum": original_sum,
                            "scaled_sum": scaled_sum,
                            "ratio": scaled_sum / max(original_sum, 1e-12),
                        }
                    )
            seen_tms.add(tm_filename)
        (ROOT / "manifest" / f"{target_topo}_manifest.txt").write_text("".join(manifest_lines), encoding="utf-8")

    write_csv(RESULTS / "load2x_data_scaling_audit.csv", scale_rows)
    (RESULTS / "load2x_experiment_config.json").write_text(
        json.dumps(
            {
                "load_factor": LOAD_FACTOR,
                "source_test_start": SOURCE_START,
                "source_test_end": SOURCE_END,
                "models_frozen": True,
                "scaled_matrices": ["ground_truth", "esm_prediction"],
                "scenarios": SCENARIOS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    state["prepare"] = True
    save_state(state)
    validate_data()


def validate_data() -> None:
    configure_modules()
    for scenario, topo in SCENARIOS.items():
        rows = manifest_rows(topo)
        if len(rows) != TEST_END:
            raise RuntimeError(f"{topo} manifest has {len(rows)} rows, expected {TEST_END}")
        path_file = ROOT / "topologies" / "paths_dict" / f"{topo}_{K}_paths_dict_cluster_0.pkl"
        with path_file.open("rb") as handle:
            paths = pickle.load(handle)
        if len(paths) != 462 or any(len(value) != K for value in paths.values()):
            raise RuntimeError(f"Invalid path universe for {topo}")
        if scenario != "shared":
            mask_file = ROOT / "topologies" / "path_masks" / f"{topo}_{K}_path_masks_cluster_0.pkl"
            with mask_file.open("rb") as handle:
                masks = np.asarray(pickle.load(handle), dtype=bool)
            if masks.shape != (3, 462, K) or masks.sum(axis=2).min() < 2:
                raise RuntimeError(f"Invalid path mask for {topo}: {masks.shape}")

        source_rows = manifest_rows(SOURCE_TOPOS[scenario])[SOURCE_START:SOURCE_END]
        for local_idx in (0, TEST_END // 2, TEST_END - 1):
            source_tm = source_rows[local_idx][2]
            target_tm = rows[local_idx][2]
            for priority in (1, 2, 3):
                for suffix in ("", "_esm"):
                    with (ROOT / "traffic_matrices" / f"{SOURCE_TOPOS[scenario]}_{priority}{suffix}" / source_tm).open("rb") as handle:
                        original = np.asarray(pickle.load(handle), dtype=np.float64)
                    with (ROOT / "traffic_matrices" / f"{topo}_{priority}{suffix}" / target_tm).open("rb") as handle:
                        scaled = np.asarray(pickle.load(handle), dtype=np.float64)
                    if not np.allclose(scaled, original * LOAD_FACTOR, rtol=1e-7, atol=1e-10):
                        raise RuntimeError(f"Scaling mismatch: {topo}/{priority}{suffix}/{target_tm}")
    print("[ok] load2x data validation passed", flush=True)


def run_gurobi() -> None:
    configure_modules()
    ensure_dirs()
    pm.run_gurobi()


def run_hattrick() -> None:
    configure_modules()
    ensure_dirs()
    state = load_state()
    for scenario, topo in SCENARIOS.items():
        source_topo = SOURCE_TOPOS[scenario]
        model_path = ROOT / f"hattrick_{source_topo}_{K}sp.pkl"
        if not model_path.exists():
            raise RuntimeError(f"Missing frozen Hattrick model: {model_path}")
        path_mask = 0 if scenario == "shared" else 1
        for sim in (1, 0):
            key = f"{scenario}_sim{sim}"
            if state["hattrick"].get(key):
                print(f"[skip] Hattrick {key}", flush=True)
                continue
            command = pm.test_command(
                topo,
                f"hattrick_{scenario}_frozen_load2x",
                path_mask,
                str(model_path),
                sim_mf_mlu=sim,
            )
            pm.run_command(f"hattrick_load2x_{key}", command)
            state["hattrick"][key] = True
            save_state(state)

    expected = TEST_END * 3
    for scenario, topo in SCENARIOS.items():
        result_dir = ROOT / "results" / topo / f"{K}sp" / "0"
        for sim in (1, 0):
            output = result_dir / f"hattrick_{scenario}_frozen_load2x_values_esm_sim_mlu_{sim}.txt"
            if line_count(output) != expected:
                raise RuntimeError(f"{output} has {line_count(output)} lines, expected {expected}")
    print("[ok] frozen Hattrick load2x tests passed", flush=True)


def run_dotemc() -> None:
    configure_modules()
    ensure_dirs()
    state = load_state()
    device = dm.torch.device("cuda" if dm.torch.cuda.is_available() else "cpu")
    dm.set_seed()
    dm.validate_data()
    for scenario, topo in SCENARIOS.items():
        for weight_label in dm.WEIGHT_CONFIGS:
            key = f"{scenario}_{weight_label}"
            if state["dotemc"].get(key):
                print(f"[skip] DOTE-MC {key}", flush=True)
                continue
            source_checkpoint = ORIGINAL_RESULTS / "outputs" / scenario / weight_label / "best_model.pt"
            target_dir = dm.OUTPUTS / scenario / weight_label
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_checkpoint, target_dir / "best_model.pt")
            dm.test_one(scenario, topo, weight_label, device)
            state["dotemc"][key] = True
            save_state(state)
    dm.validate_test_outputs()
    print("[ok] frozen DOTE-MC load2x tests passed", flush=True)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def build_hattrick_and_optimization_rows() -> list[dict]:
    rows = []
    for scenario, topo in SCENARIOS.items():
        result_dir = ROOT / "results" / topo / f"{K}sp" / "0"
        demands = pm.read_demands(topo)
        oracle = pm.oracle_flow(result_dir)
        norm, normalized_mlu = pm.hattrick_values(result_dir, f"{scenario}_frozen_load2x")
        method_data = {"Hattrick": (norm * np.maximum(oracle, 1e-9), norm, normalized_mlu)}
        for method, filename in (
            ("BEST_MC", "flexile_sim_results_esm_mf_mf_mf.txt"),
            ("SWAN", "swan_sim_results_esm_mf_mf_mf.txt"),
        ):
            carried, mlu = pm.split_simulation(result_dir / filename)
            method_data[method] = (carried, carried / np.maximum(oracle, 1e-9), mlu)

        for method, (carried, norm_fulfill, mlu) in method_data.items():
            for local_idx in range(TEST_END):
                for class_idx, class_name in enumerate(CLASSES):
                    demand = demands[local_idx, class_idx]
                    rows.append(
                        {
                            "load_factor": LOAD_FACTOR,
                            "scenario": scenario,
                            "weights": "na",
                            "topo": topo,
                            "snapshot": SOURCE_START + local_idx,
                            "method": method,
                            "class": class_name,
                            "admitted_traffic": float(carried[local_idx, class_idx]),
                            "demand": float(demand),
                            "fulfill_ratio": float(carried[local_idx, class_idx] / max(demand, 1e-9)),
                            "mlu": float(mlu[local_idx, class_idx]),
                            "norm_fulfill": float(norm_fulfill[local_idx, class_idx]),
                        }
                    )
    return rows


def build_dotemc_rows() -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        for weight_label in dm.WEIGHT_CONFIGS:
            metrics_path = dm.model_output_paths(scenario, weight_label)[2]
            for source in read_csv(metrics_path):
                rows.append(
                    {
                        "load_factor": LOAD_FACTOR,
                        "scenario": scenario,
                        "weights": source["weights"],
                        "topo": source["topo"],
                        "snapshot": SOURCE_START + int(source["snapshot"]),
                        "method": source["method"],
                        "class": source["class"],
                        "admitted_traffic": float(source["admitted_traffic"]),
                        "demand": float(source["demand"]),
                        "fulfill_ratio": float(source["fulfill_ratio"]),
                        "mlu": float(source["mlu"]),
                        "norm_fulfill": float(source["norm_fulfill"]),
                    }
                )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    groups = sorted({(row["scenario"], row["weights"], row["method"], row["class"]) for row in rows})
    for scenario, weights, method, class_name in groups:
        selected = [
            row
            for row in rows
            if row["scenario"] == scenario
            and row["weights"] == weights
            and row["method"] == method
            and row["class"] == class_name
        ]
        item = {
            "load_factor": LOAD_FACTOR,
            "scenario": scenario,
            "weights": weights,
            "method": method,
            "class": class_name,
            "n": len(selected),
        }
        for metric in ("admitted_traffic", "demand", "fulfill_ratio", "mlu", "norm_fulfill"):
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_median"] = float(np.median(values))
            item[f"{metric}_p1"] = float(np.percentile(values, 1))
            item[f"{metric}_p10"] = float(np.percentile(values, 10))
            item[f"{metric}_p25"] = float(np.percentile(values, 25))
            item[f"{metric}_p75"] = float(np.percentile(values, 75))
        summary.append(item)
    return summary


def build_deltas(stress_summary: list[dict]) -> list[dict]:
    baseline_rows = read_csv(BASELINE_SUMMARY)
    baseline = {
        (row["scenario"], row["weights"], row["method"], row["class"]): row
        for row in baseline_rows
    }
    deltas = []
    for stress in stress_summary:
        key = (stress["scenario"], stress["weights"], stress["method"], stress["class"])
        base = baseline[key]
        item = {
            "scenario": stress["scenario"],
            "weights": stress["weights"],
            "method": stress["method"],
            "class": stress["class"],
        }
        for metric in ("fulfill_ratio", "norm_fulfill"):
            for stat in ("mean", "p1", "p10"):
                column = f"{metric}_{stat}"
                baseline_value = float(base[column])
                stress_value = float(stress[column])
                item[f"baseline_{column}"] = baseline_value
                item[f"load2x_{column}"] = stress_value
                item[f"delta_{column}"] = stress_value - baseline_value
        deltas.append(item)
    return deltas


def display_method(method: str, weights: str) -> str:
    if method != "DOTE_MC":
        return method
    if weights == DEFAULT_WEIGHT_LABEL:
        return "DOTE-MC default"
    return "DOTE-MC sensitivity"


def report_markdown(summary: list[dict], deltas: list[dict]) -> str:
    lines = [
        "# GEANT 2x Load Stress Test\n\n",
        "Frozen-model stress test on original snapshots 400-499. Ground-truth and ESM-predicted traffic matrices are both multiplied by 2; topology, capacities, paths, masks, and model parameters are unchanged.\n\n",
        "Each result cell below is `Mean / P1 / P10` for paper-style NormFulFill.\n\n",
        "| Scenario | Method | Class | Mean / P1 / P10 NormFulFill | Mean FulfillRatio |\n",
        "|---|---|---|---:|---:|\n",
    ]
    order = {"High": 0, "Medium": 1, "Low": 2}
    for row in sorted(summary, key=lambda item: (list(SCENARIOS).index(item["scenario"]), display_method(item["method"], item["weights"]), order[item["class"]])):
        lines.append(
            f"| {row['scenario']} | {display_method(row['method'], row['weights'])} | {row['class']} | "
            f"{float(row['norm_fulfill_mean']):.4f} / {float(row['norm_fulfill_p1']):.4f} / {float(row['norm_fulfill_p10']):.4f} | "
            f"{float(row['fulfill_ratio_mean']):.4f} |\n"
        )

    lines.extend(
        [
            "\n## Hattrick Degradation From 1x\n\n",
            "| Scenario | Class | 1x Mean | 2x Mean | Delta | 1x P1 | 2x P1 | Delta P1 | 1x P10 | 2x P10 | Delta P10 |\n",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for row in deltas:
        if row["method"] != "Hattrick":
            continue
        lines.append(
            f"| {row['scenario']} | {row['class']} | "
            f"{float(row['baseline_norm_fulfill_mean']):.4f} | {float(row['load2x_norm_fulfill_mean']):.4f} | {float(row['delta_norm_fulfill_mean']):+.4f} | "
            f"{float(row['baseline_norm_fulfill_p1']):.4f} | {float(row['load2x_norm_fulfill_p1']):.4f} | {float(row['delta_norm_fulfill_p1']):+.4f} | "
            f"{float(row['baseline_norm_fulfill_p10']):.4f} | {float(row['load2x_norm_fulfill_p10']):.4f} | {float(row['delta_norm_fulfill_p10']):+.4f} |\n"
        )
    lines.extend(
        [
            "\n## Interpretation Guardrails\n\n",
            "- Models are frozen. A large drop measures load-distribution generalization, not retrained representational capacity.\n",
            "- BEST_MC and SWAN are re-optimized using the doubled ESM predictions.\n",
            "- NormFulFill uses a newly computed ground-truth oracle under doubled traffic.\n",
            "- Low-class NormFulFill can exceed 1 because of priority inversion, as in the paper.\n",
        ]
    )
    return "".join(lines)


def run_report() -> None:
    configure_modules()
    pm.validate_gurobi_outputs()
    dm.validate_test_outputs()
    rows = build_hattrick_and_optimization_rows() + build_dotemc_rows()
    if len(rows) != 6000:
        raise RuntimeError(f"Expected 6000 metric rows, got {len(rows)}")
    summary = summarize(rows)
    deltas = build_deltas(summary)
    write_csv(RESULTS / "load2x_metrics.csv", rows)
    write_csv(RESULTS / "load2x_summary_stats.csv", summary)
    write_csv(RESULTS / "load2x_vs_1x_delta.csv", deltas)
    (RESULTS / "load2x_stress_report.md").write_text(report_markdown(summary, deltas), encoding="utf-8")
    state = load_state()
    state["report"] = True
    save_state(state)
    print(f"[ok] wrote load2x stress report to {RESULTS}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "prepare", "validate", "gurobi", "hattrick", "dotemc", "report"),
        default="all",
    )
    args = parser.parse_args()
    configure_modules()
    ensure_dirs()
    if args.stage in ("all", "prepare"):
        prepare()
    if args.stage in ("all", "validate"):
        validate_data()
    if args.stage in ("all", "gurobi"):
        run_gurobi()
    if args.stage in ("all", "hattrick"):
        run_hattrick()
    if args.stage in ("all", "dotemc"):
        run_dotemc()
    if args.stage in ("all", "report"):
        run_report()


if __name__ == "__main__":
    main()
