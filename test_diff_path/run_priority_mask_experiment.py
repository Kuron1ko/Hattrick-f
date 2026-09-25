from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import shutil
import subprocess
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test_diff_path"
RESULTS = OUT / "results_priority_masks_60epoch"
LOGS = OUT / "logs_priority_masks_60epoch"
STATE_FILE = OUT / "priority_mask_state.json"
PYTHON = str(Path(r"D:\kuroresearch\.venv-hattrick\Scripts\python.exe"))

K = 8
BASE_TOPO = "geant"
TRAIN_END = 80
VAL_END = 90
TEST_END = 100
EPOCHS = 60
BATCH_SIZE = 8
CLASSES = ("High", "Medium", "Low")
METHODS = ("Hattrick", "BEST_MC", "SWAN")
SCENARIOS = {
    "shared": "geant_priomask_shared",
    "mild": "geant_priomask_mild",
    "medium": "geant_priomask_medium",
    "strict": "geant_priomask_strict",
}
MASK_SPECS = {
    "shared": ((0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7)),
    "mild": ((0, 1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6), (2, 3, 4, 5, 6, 7)),
    "medium": ((0, 1, 2, 3), (2, 3, 4, 5), (4, 5, 6, 7)),
    "strict": ((0, 1), (3, 4), (6, 7)),
}


def ensure_dirs() -> None:
    for path in (RESULTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"prepare": False, "gurobi": {}, "hattrick": {}, "report": False}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def run_command(label: str, command: list[str]) -> None:
    ensure_dirs()
    env = os.environ.copy()
    env.setdefault("GUROBI_HOME", r"D:\kuroresearch\gurobi1302\win64")
    env.setdefault("GRB_LICENSE_FILE", r"D:\kuroresearch\gurobi_license\gurobi.lic")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print(f"[run] {label}", flush=True)
    start = time.time()
    proc = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    output = (proc.stdout or "").replace("LicenseID", "LicenseID(redacted)")
    log_path = LOGS / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}.log"
    log_path.write_text(output, encoding="utf-8", errors="ignore")
    print(f"[done] {label} exit={proc.returncode} elapsed={elapsed:.1f}s log={log_path.name}", flush=True)
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-80:])
        raise RuntimeError(f"{label} failed\n{tail}")


def read_manifest() -> list[list[str]]:
    path = ROOT / "manifest" / f"{BASE_TOPO}_manifest.txt"
    return [line.strip().split(",") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_alias_data(topo: str) -> None:
    rows = read_manifest()[:TEST_END]
    (ROOT / "topologies" / topo).mkdir(parents=True, exist_ok=True)
    (ROOT / "pairs" / topo).mkdir(parents=True, exist_ok=True)
    for priority in (1, 2, 3):
        (ROOT / "traffic_matrices" / f"{topo}_{priority}").mkdir(parents=True, exist_ok=True)
        (ROOT / "traffic_matrices" / f"{topo}_{priority}_esm").mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    seen_topologies, seen_pairs, seen_tms = set(), set(), set()
    for topology_filename, pairs_filename, tm_filename in rows:
        topology_filename = topology_filename.strip()
        pairs_filename = pairs_filename.strip()
        tm_filename = tm_filename.strip()
        manifest_lines.append(f"{topology_filename},{pairs_filename},{tm_filename}\n")
        if topology_filename not in seen_topologies:
            copy_file(ROOT / "topologies" / BASE_TOPO / topology_filename, ROOT / "topologies" / topo / topology_filename)
            seen_topologies.add(topology_filename)
        if pairs_filename not in seen_pairs:
            copy_file(ROOT / "pairs" / BASE_TOPO / pairs_filename, ROOT / "pairs" / topo / pairs_filename)
            seen_pairs.add(pairs_filename)
        if tm_filename not in seen_tms:
            for priority in (1, 2, 3):
                copy_file(ROOT / "traffic_matrices" / f"{BASE_TOPO}_{priority}" / tm_filename, ROOT / "traffic_matrices" / f"{topo}_{priority}" / tm_filename)
                copy_file(ROOT / "traffic_matrices" / f"{BASE_TOPO}_{priority}_esm" / tm_filename, ROOT / "traffic_matrices" / f"{topo}_{priority}_esm" / tm_filename)
            seen_tms.add(tm_filename)
    (ROOT / "manifest" / f"{topo}_manifest.txt").write_text("".join(manifest_lines), encoding="utf-8")


def node_ids_to_edges(path: list) -> list[tuple]:
    return [(src, dst) for src, dst in zip(path, path[1:])]


def path_signature(path_edges: list[tuple]) -> tuple[tuple, ...]:
    return tuple(path_edges)


def generate_candidate_paths() -> tuple[dict[tuple, list[list[tuple]]], list[tuple]]:
    import json as json_lib

    topology_filename, pairs_filename, _ = [part.strip() for part in read_manifest()[0]]
    with (ROOT / "topologies" / BASE_TOPO / topology_filename).open("r", encoding="utf-8") as handle:
        graph = nx.readwrite.json_graph.node_link_graph(json_lib.load(handle))
    with (ROOT / "pairs" / BASE_TOPO / pairs_filename).open("rb") as handle:
        pairs = pickle.load(handle)

    paths_by_pair: dict[tuple, list[list[tuple]]] = {}
    for src, dst in pairs:
        all_paths = []
        for node_path in nx.shortest_simple_paths(graph, src, dst):
            edge_path = node_ids_to_edges(node_path)
            if path_signature(edge_path) not in {path_signature(path) for path in all_paths}:
                all_paths.append(edge_path)
            if len(all_paths) >= K:
                break
        if not all_paths:
            raise ValueError(f"No path for {src}->{dst}")
        while len(all_paths) < K:
            all_paths.append(all_paths[0])
        paths_by_pair[(src, dst)] = all_paths[:K]
    return paths_by_pair, list(paths_by_pair.keys())


def mask_for_scenario(scenario: str, num_pairs: int) -> np.ndarray:
    masks = np.zeros((3, num_pairs, K), dtype=bool)
    for class_idx, allowed_indices in enumerate(MASK_SPECS[scenario]):
        masks[class_idx, :, list(allowed_indices)] = True
    if not masks.any(axis=2).all():
        raise ValueError(f"{scenario} has a class/pair without allowed paths")
    if masks.sum(axis=2).min() < 2:
        raise ValueError(f"{scenario} allows fewer than 2 paths for some class/pair")
    return masks


def overlap_stats(masks: np.ndarray) -> tuple[float, float]:
    overlaps = []
    unions = []
    for pair_idx in range(masks.shape[1]):
        for a, b in ((0, 1), (0, 2), (1, 2)):
            inter = np.logical_and(masks[a, pair_idx], masks[b, pair_idx]).sum()
            union = np.logical_or(masks[a, pair_idx], masks[b, pair_idx]).sum()
            overlaps.append(inter / max(union, 1))
            unions.append(union)
    return float(np.mean(overlaps)), float(np.mean(unions))


def prepare() -> None:
    state = load_state()
    if state.get("prepare"):
        print("[skip] prepare already complete", flush=True)
        return
    ensure_dirs()
    for directory in ("paths_dict", "paths", "padded_edge_ids_per_path", "path_masks"):
        (ROOT / "topologies" / directory).mkdir(parents=True, exist_ok=True)

    paths_by_pair, pairs = generate_candidate_paths()
    stats_rows = []
    path_lengths = np.asarray([len(path) for paths in paths_by_pair.values() for path in paths], dtype=np.float64)
    for scenario, topo in SCENARIOS.items():
        copy_alias_data(topo)
        with (ROOT / "topologies" / "paths_dict" / f"{topo}_{K}_paths_dict_cluster_0.pkl").open("wb") as handle:
            pickle.dump(paths_by_pair, handle)
        masks = mask_for_scenario(scenario, len(pairs))
        if scenario != "shared":
            with (ROOT / "topologies" / "path_masks" / f"{topo}_{K}_path_masks_cluster_0.pkl").open("wb") as handle:
                pickle.dump(masks, handle)
        avg_overlap, avg_union = overlap_stats(masks)
        stats_rows.append(
            {
                "scenario": scenario,
                "topo": topo,
                "num_pairs": len(pairs),
                "paths_per_pair": K,
                "min_allowed_paths": int(masks.sum(axis=2).min()),
                "max_allowed_paths": int(masks.sum(axis=2).max()),
                "avg_pairwise_mask_jaccard": avg_overlap,
                "avg_pairwise_union_size": avg_union,
                "candidate_path_len_mean": float(path_lengths.mean()),
                "candidate_path_len_median": float(np.median(path_lengths)),
            }
        )

    write_csv(RESULTS / "priority_mask_path_stats.csv", stats_rows)
    state["prepare"] = True
    save_state(state)


def gurobi_command(topo: str, pred: int, mode: str, priority: int, objs: list[str], path_mask: int) -> list[str]:
    return [
        PYTHON,
        "frameworks/gurobi_refactored.py",
        "--topo",
        topo,
        "--num_paths_per_pair",
        str(K),
        "--opt_start_idx",
        "0",
        "--opt_end_idx",
        str(TEST_END),
        "--cluster",
        "0",
        "--pred",
        str(pred),
        "--pred_type",
        "esm",
        "--gur_mode",
        mode,
        "--priority",
        str(priority),
        "--objs",
        *objs,
        "--path_mask",
        str(path_mask),
    ]


def run_gurobi() -> None:
    state = load_state()
    for scenario, topo in SCENARIOS.items():
        path_mask = 0 if scenario == "shared" else 1
        tasks = []
        for base_obj in ("mf", "mlu"):
            for priority in (1, 2, 3):
                objs = [base_obj] * priority
                tasks.append((f"{scenario}_gt_{'_'.join(objs)}", gurobi_command(topo, 0, "flexile", priority, objs, path_mask)))
        for mode in ("flexile", "swan"):
            for priority in (1, 2, 3):
                objs = ["mf"] * priority
                tasks.append((f"{scenario}_{mode}_{'_'.join(objs)}", gurobi_command(topo, 1, mode, priority, objs, path_mask)))
        for label, command in tasks:
            if state["gurobi"].get(label):
                print(f"[skip] {label}", flush=True)
                continue
            run_command(label, command)
            state["gurobi"][label] = True
            save_state(state)
    validate_gurobi_outputs()


def train_command(topo: str, path_mask: int) -> list[str]:
    return [
        PYTHON,
        "run_hattrick.py",
        "--topo",
        topo,
        "--mode",
        "train",
        "--epochs",
        str(EPOCHS),
        "--batch_size",
        str(BATCH_SIZE),
        "--num_paths_per_pair",
        str(K),
        "--num_transformer_layers",
        "3",
        "--num_gnn_layers",
        "3",
        "--num_mlp1_hidden_layers",
        "2",
        "--num_mlp2_hidden_layers",
        "2",
        "--rau1",
        "3",
        "--rau2",
        "3",
        "--rau3",
        "3",
        "--train_clusters",
        "0",
        "--train_start_indices",
        "0",
        "--train_end_indices",
        str(TRAIN_END),
        "--val_clusters",
        "0",
        "--val_start_indices",
        str(TRAIN_END),
        "--val_end_indices",
        str(VAL_END),
        "--pred",
        "1",
        "--dynamic",
        "0",
        "--lr",
        "0.0005",
        "--pred_type",
        "esm",
        "--initial_training",
        "1",
        "--violation",
        "1",
        "--path_mask",
        str(path_mask),
    ]


def test_command(topo: str, model_name: str, path_mask: int, model_override: str = "", sim_mf_mlu: int = 1) -> list[str]:
    command = [
        PYTHON,
        "run_hattrick.py",
        "--topo",
        topo,
        "--mode",
        "test",
        "--test_cluster",
        "0",
        "--test_start_idx",
        str(VAL_END),
        "--test_end_idx",
        str(TEST_END),
        "--num_paths_per_pair",
        str(K),
        "--num_transformer_layers",
        "3",
        "--num_gnn_layers",
        "3",
        "--num_mlp1_hidden_layers",
        "2",
        "--num_mlp2_hidden_layers",
        "2",
        "--rau1",
        "3",
        "--rau2",
        "3",
        "--rau3",
        "3",
        "--pred",
        "1",
        "--dynamic",
        "0",
        "--pred_type",
        "esm",
        "--sim_mf_mlu",
        str(sim_mf_mlu),
        "--violation",
        "1",
        "--path_mask",
        str(path_mask),
        "--model",
        model_name,
    ]
    if model_override:
        command.extend(["--model_path_override", model_override])
    return command


def run_hattrick() -> None:
    state = load_state()
    base_model = str(ROOT / "hattrick_geant_8sp.pkl")
    for scenario, topo in SCENARIOS.items():
        path_mask = 0 if scenario == "shared" else 1
        runs = [
            (f"{scenario}_zero_shot", base_model, False),
            (f"{scenario}_retrained", "", True),
        ]
        for run_name, override, retrain in runs:
            if retrain and not state["hattrick"].get(f"train_{run_name}"):
                run_command(f"train_{run_name}", train_command(topo, path_mask))
                state["hattrick"][f"train_{run_name}"] = True
                save_state(state)
            for sim in (1, 0):
                key = f"test_{run_name}_sim{sim}"
                if state["hattrick"].get(key):
                    print(f"[skip] {key}", flush=True)
                    continue
                run_command(key, test_command(topo, f"hattrick_{run_name}", path_mask, override, sim_mf_mlu=sim))
                state["hattrick"][key] = True
                save_state(state)
    validate_hattrick_outputs()


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def validate_gurobi_outputs() -> None:
    for scenario, topo in SCENARIOS.items():
        result_dir = ROOT / "results" / topo / f"{K}sp" / "0"
        for name in (
            "gt_optimal_values_mf.txt",
            "gt_optimal_values_mf_mf.txt",
            "gt_optimal_values_mf_mf_mf.txt",
            "gt_optimal_values_mlu.txt",
            "gt_optimal_values_mlu_mlu.txt",
            "gt_optimal_values_mlu_mlu_mlu.txt",
        ):
            count = line_count(result_dir / name)
            if count != TEST_END:
                raise RuntimeError(f"{result_dir / name} has {count} lines, expected {TEST_END}")
        for name in ("flexile_sim_results_esm_mf_mf_mf.txt", "swan_sim_results_esm_mf_mf_mf.txt"):
            count = line_count(result_dir / name)
            if count != TEST_END * 6:
                raise RuntimeError(f"{result_dir / name} has {count} lines, expected {TEST_END * 6}")


def validate_hattrick_outputs() -> None:
    for scenario, topo in SCENARIOS.items():
        result_dir = ROOT / "results" / topo / f"{K}sp" / "0"
        for run_type in ("zero_shot", "retrained"):
            run_name = f"{scenario}_{run_type}"
            for sim in (0, 1):
                path = result_dir / f"hattrick_{run_name}_values_esm_sim_mlu_{sim}.txt"
                count = line_count(path)
                if count != (TEST_END - VAL_END) * 3:
                    raise RuntimeError(f"{path} has {count} lines, expected {(TEST_END - VAL_END) * 3}")


def read_floats(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64).reshape(-1)


def read_demands(topo: str) -> np.ndarray:
    rows = [line.strip().split(",") for line in (ROOT / "manifest" / f"{topo}_manifest.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    demands = []
    for _, _, tm_filename in rows[VAL_END:TEST_END]:
        per_class = []
        for priority in (1, 2, 3):
            with (ROOT / "traffic_matrices" / f"{topo}_{priority}" / tm_filename.strip()).open("rb") as handle:
                per_class.append(float(np.asarray(pickle.load(handle), dtype=np.float64).sum()))
        demands.append(per_class)
    return np.asarray(demands, dtype=np.float64)


def oracle_flow(result_dir: Path) -> np.ndarray:
    mf1 = read_floats(result_dir / "gt_optimal_values_mf.txt")[VAL_END:TEST_END]
    mf12 = read_floats(result_dir / "gt_optimal_values_mf_mf.txt")[VAL_END:TEST_END]
    mf123 = read_floats(result_dir / "gt_optimal_values_mf_mf_mf.txt")[VAL_END:TEST_END]
    return np.column_stack((mf1, mf12 - mf1, mf123 - mf12))


def split_simulation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sim = read_floats(path).reshape(-1, 6)[VAL_END:TEST_END]
    mlu = np.column_stack((sim[:, 0], sim[:, 2], sim[:, 4]))
    cumulative = np.column_stack((sim[:, 1], sim[:, 3], sim[:, 5]))
    carried = np.column_stack((cumulative[:, 0], cumulative[:, 1] - cumulative[:, 0], cumulative[:, 2] - cumulative[:, 1]))
    return carried, mlu


def hattrick_values(result_dir: Path, run_name: str) -> tuple[np.ndarray, np.ndarray]:
    norm = read_floats(result_dir / f"hattrick_{run_name}_values_esm_sim_mlu_1.txt").reshape(-1, 3)
    norm_mlu = read_floats(result_dir / f"hattrick_{run_name}_values_esm_sim_mlu_0.txt").reshape(-1, 3)
    return norm, norm_mlu


def build_metric_rows() -> list[dict[str, str | int | float]]:
    rows = []
    for scenario, topo in SCENARIOS.items():
        result_dir = ROOT / "results" / topo / f"{K}sp" / "0"
        demands = read_demands(topo)
        oracle = oracle_flow(result_dir)
        runs = [f"{scenario}_zero_shot", f"{scenario}_retrained"]
        for run_name in runs:
            norm, norm_mlu = hattrick_values(result_dir, run_name)
            method_data = {"Hattrick": (norm * np.maximum(oracle, 1e-9), norm, norm_mlu)}
            for method, filename in (("BEST_MC", "flexile_sim_results_esm_mf_mf_mf.txt"), ("SWAN", "swan_sim_results_esm_mf_mf_mf.txt")):
                carried, mlu = split_simulation(result_dir / filename)
                method_data[method] = (carried, carried / np.maximum(oracle, 1e-9), mlu)
            for method, (carried, norm_fulfill, mlu) in method_data.items():
                for snap_idx in range(TEST_END - VAL_END):
                    for class_idx, class_name in enumerate(CLASSES):
                        demand = demands[snap_idx, class_idx]
                        rows.append(
                            {
                                "scenario": scenario,
                                "run_type": "zero_shot" if run_name.endswith("zero_shot") else "retrained",
                                "topo": topo,
                                "snapshot": VAL_END + snap_idx,
                                "method": method,
                                "class": class_name,
                                "admitted_traffic": float(carried[snap_idx, class_idx]),
                                "demand": float(demand),
                                "fulfill_ratio": float(carried[snap_idx, class_idx] / max(demand, 1e-9)),
                                "mlu": float(mlu[snap_idx, class_idx]),
                                "norm_fulfill": float(norm_fulfill[snap_idx, class_idx]),
                            }
                        )
    return rows


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    summary = []
    groups = sorted({(r["scenario"], r["run_type"], r["method"], r["class"]) for r in rows})
    for scenario, run_type, method, class_name in groups:
        vals = [r for r in rows if r["scenario"] == scenario and r["run_type"] == run_type and r["method"] == method and r["class"] == class_name]
        item = {"scenario": scenario, "run_type": run_type, "method": method, "class": class_name, "n": len(vals)}
        for metric in ("admitted_traffic", "fulfill_ratio", "mlu", "norm_fulfill"):
            arr = np.asarray([float(v[metric]) for v in vals], dtype=np.float64)
            item[f"{metric}_mean"] = float(arr.mean())
            item[f"{metric}_median"] = float(np.median(arr))
            item[f"{metric}_p10"] = float(np.percentile(arr, 10))
            item[f"{metric}_p1"] = float(np.percentile(arr, 1))
        summary.append(item)
    return summary


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_cdf(rows: list[dict[str, str | int | float]]) -> None:
    scenarios = tuple(SCENARIOS.keys())
    colors = {"Hattrick": "#2F80ED", "BEST_MC": "#27AE60", "SWAN": "#EB5757"}
    width, height = 1320, 1040
    left, top = 84, 108
    panel_w, panel_h = 280, 180
    gap_x, gap_y = 38, 58
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 28), "Priority-mask 60epoch: CDF of FulfillRatio", fill="#202124", font=font(28, True))
    draw.text((left, 62), "Empirical CDF over snapshots 90-99, retrained runs only. X-axis starts at 0.", fill="#5f6368", font=font(15))
    x_min, x_max = 0.0, 1.15
    for row_idx, scenario in enumerate(scenarios):
        draw.text((left, top + row_idx * (panel_h + gap_y) - 28), scenario, fill="#202124", font=font(16, True))
        for col_idx, class_name in enumerate(CLASSES):
            px = left + col_idx * (panel_w + gap_x)
            py = top + row_idx * (panel_h + gap_y)
            draw.rectangle((px, py, px + panel_w, py + panel_h), outline="#202124", width=1)
            draw.text((px + panel_w // 2 - 34, py + panel_h + 8), class_name, fill="#202124", font=font(14))
            for tick in np.arange(0, 1.151, 0.25):
                x = int(px + (tick - x_min) / (x_max - x_min) * panel_w)
                draw.line((x, py, x, py + panel_h), fill="#E8EAED")
                draw.text((x - 10, py + panel_h + 24), f"{tick:.2g}", fill="#5f6368", font=font(11))
            for tick in np.linspace(0, 1, 5):
                y = int(py + panel_h - tick * panel_h)
                draw.line((px, y, px + panel_w, y), fill="#E8EAED")
            for method in METHODS:
                vals = sorted(float(r["fulfill_ratio"]) for r in rows if r["scenario"] == scenario and r["run_type"] == "retrained" and r["class"] == class_name and r["method"] == method)
                if not vals:
                    continue
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                xs = [x_min] + vals + [x_max]
                ys = [0.0] + list(cdf) + [1.0]
                points = [(int(px + (min(max(v, x_min), x_max) - x_min) / (x_max - x_min) * panel_w), int(py + panel_h - p * panel_h)) for v, p in zip(xs, ys)]
                draw.line(points, fill=colors[method], width=2)
    lx, ly = left, height - 55
    for method in METHODS:
        draw.line((lx, ly, lx + 36, ly), fill=colors[method], width=5)
        draw.text((lx + 46, ly - 10), method, fill="#202124", font=font(16))
        lx += 180
    image.save(RESULTS / "priority_mask_cdf.png")


def draw_boxplot(rows: list[dict[str, str | int | float]]) -> None:
    image = Image.new("RGB", (1180, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "Priority-mask 60epoch: FulfillRatio boxplot", fill="#202124", font=font(28, True))
    draw.text((70, 64), "Retrained runs, all classes over snapshots 90-99. Higher is better.", fill="#5f6368", font=font(15))
    colors = {"Hattrick": "#2F80ED", "BEST_MC": "#27AE60", "SWAN": "#EB5757"}
    left, top, bottom = 110, 120, 620
    x_gap = 260
    all_vals = []
    for scenario in SCENARIOS:
        for method in METHODS:
            all_vals.extend(float(r["fulfill_ratio"]) for r in rows if r["scenario"] == scenario and r["run_type"] == "retrained" and r["method"] == method)
    y_min, y_max = 0.0, max(1.05, math.ceil((max(all_vals) + 0.05) * 4) / 4)

    def y_of(value: float) -> int:
        return int(bottom - (value - y_min) / (y_max - y_min) * (bottom - top))

    for tick in np.arange(0, y_max + 0.001, 0.25):
        y = y_of(float(tick))
        draw.line((left - 40, y, 1110, y), fill="#E8EAED")
        draw.text((32, y - 8), f"{tick:.2f}", fill="#5f6368", font=font(13))
    for idx, scenario in enumerate(SCENARIOS):
        x0 = left + idx * x_gap
        draw.text((x0 - 10, bottom + 22), scenario, fill="#202124", font=font(14))
        for method_idx, method in enumerate(METHODS):
            vals = np.asarray(
                [
                    float(r["fulfill_ratio"])
                    for r in rows
                    if r["scenario"] == scenario and r["run_type"] == "retrained" and r["method"] == method
                ],
                dtype=np.float64,
            )
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            low, high = np.percentile(vals, [5, 95])
            mean = float(vals.mean())
            cx = x0 + method_idx * 42 + 15
            box_left, box_right = cx - 13, cx + 13
            draw.line((cx, y_of(low), cx, y_of(high)), fill=colors[method], width=2)
            draw.line((box_left, y_of(low), box_right, y_of(low)), fill=colors[method], width=2)
            draw.line((box_left, y_of(high), box_right, y_of(high)), fill=colors[method], width=2)
            draw.rectangle((box_left, y_of(q3), box_right, y_of(q1)), outline=colors[method], width=3)
            draw.line((box_left, y_of(med), box_right, y_of(med)), fill=colors[method], width=3)
            draw.ellipse((cx - 3, y_of(mean) - 3, cx + 3, y_of(mean) + 3), fill=colors[method])
    lx, ly = 70, 680
    for method in METHODS:
        draw.rectangle((lx, ly, lx + 24, ly + 14), fill=colors[method])
        draw.text((lx + 34, ly - 4), method, fill="#202124", font=font(15))
        lx += 160
    image.save(RESULTS / "priority_mask_boxplot.png")


def structure_conclusion(summary: list[dict[str, str | int | float]]) -> str:
    lines = [
        "# Priority Mask Structure Conclusion\n\n",
        "Goal: test whether the original Hattrick structure works when high/medium/low priorities have different allowed path sets.\n\n",
        "## Mean FulfillRatio, averaged over classes\n\n",
        "| Scenario | Hattrick zero-shot | Hattrick retrained | BEST_MC | SWAN | Retrained gap vs BEST_MC |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    decisions = []
    for scenario in SCENARIOS:
        def avg(method: str, run_type: str) -> float:
            vals = [float(r["fulfill_ratio_mean"]) for r in summary if r["scenario"] == scenario and r["method"] == method and r["run_type"] == run_type]
            return float(np.mean(vals))

        hat_zero = avg("Hattrick", "zero_shot")
        hat_re = avg("Hattrick", "retrained")
        best = avg("BEST_MC", "retrained")
        swan = avg("SWAN", "retrained")
        gap = hat_re - best
        lines.append(f"| {scenario} | {hat_zero:.6f} | {hat_re:.6f} | {best:.6f} | {swan:.6f} | {gap:+.6f} |\n")
        decisions.append((scenario, hat_re, best, swan, gap))

    mild_ok = next(gap for scenario, _, _, _, gap in decisions if scenario == "mild") > -0.05
    medium_ok = next(gap for scenario, _, _, _, gap in decisions if scenario == "medium") > -0.08
    strict_ok = next(gap for scenario, _, _, _, gap in decisions if scenario == "strict") > -0.10
    if mild_ok and medium_ok and strict_ok:
        conclusion = "The original structure is sufficient for these 100-slice priority-mask tests."
    elif mild_ok and medium_ok:
        conclusion = "The original structure handles mild/medium path differences, but strict path separation likely needs explicit class-specific modeling."
    else:
        conclusion = "The original structure is not sufficient under priority-specific path sets; class-specific path encoder/head or explicit priority path modeling is recommended."

    lines.extend(
        [
            "\n## Conclusion\n\n",
            conclusion + "\n\n",
            "NormFulFill is retained in the CSV, but the decision above uses FulfillRatio and MLU because staged NormFulFill can exceed 1 under class-specific masks.\n",
        ]
    )
    return "".join(lines)


def run_report() -> None:
    rows = build_metric_rows()
    summary = summarize(rows)
    write_csv(RESULTS / "priority_mask_metrics_100slice.csv", rows)
    write_csv(RESULTS / "priority_mask_summary_stats.csv", summary)
    draw_cdf(rows)
    draw_boxplot(rows)
    (RESULTS / "priority_mask_structure_conclusion.md").write_text(structure_conclusion(summary), encoding="utf-8")
    state = load_state()
    state["report"] = True
    save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "prepare", "gurobi", "hattrick", "report"], default="all")
    args = parser.parse_args()
    ensure_dirs()
    if args.stage in ("all", "prepare"):
        prepare()
    if args.stage in ("all", "gurobi"):
        run_gurobi()
    if args.stage in ("all", "hattrick"):
        run_hattrick()
    if args.stage in ("all", "report"):
        run_report()


if __name__ == "__main__":
    main()
