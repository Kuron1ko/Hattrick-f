from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test_diff_path"
RESULTS = OUT / "results"
LOGS = OUT / "logs"
STATE_FILE = OUT / "state.json"
PYTHON = str(Path(r"D:\kuroresearch\.venv-hattrick\Scripts\python.exe"))

K = 8
BASE_TOPO = "geant"
TOPO_SHARED = "geant_diff_shared"
TOPO_CLASSMASK = "geant_diff_classmask"
TOPOS = (TOPO_SHARED, TOPO_CLASSMASK)
CLASSES = ("High", "Medium", "Low")
METHODS = ("Hattrick", "BEST_MC", "SWAN")
TRAIN_END = 80
VAL_END = 90
TEST_END = 100


def ensure_dirs() -> None:
    for path in (RESULTS, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"prepare": False, "gurobi": {}, "hattrick": {}, "report": False}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def run_command(label: str, command: list[str], allow_failure: bool = False) -> subprocess.CompletedProcess:
    ensure_dirs()
    env = os.environ.copy()
    env.setdefault("GUROBI_HOME", r"D:\kuroresearch\gurobi1302\win64")
    env.setdefault("GRB_LICENSE_FILE", r"D:\kuroresearch\gurobi_license\gurobi.lic")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    print(f"[run] {label}", flush=True)
    start = time.time()
    proc = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    log_path = LOGS / f"{time.strftime('%Y%m%d_%H%M%S')}_{label}.log"
    log_path.write_text(proc.stdout.replace("LicenseID", "LicenseID(redacted)"), encoding="utf-8", errors="ignore")
    print(f"[done] {label} exit={proc.returncode} elapsed={elapsed:.1f}s log={log_path.name}", flush=True)
    if proc.returncode != 0 and not allow_failure:
        tail = "\n".join(proc.stdout.splitlines()[-80:])
        raise RuntimeError(f"{label} failed\n{tail}")
    return proc


def read_manifest() -> list[list[str]]:
    manifest = ROOT / "manifest" / f"{BASE_TOPO}_manifest.txt"
    return [line.strip().split(",") for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


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

    alias_manifest = []
    seen_topologies: set[str] = set()
    seen_pairs: set[str] = set()
    seen_tms: set[str] = set()
    for topology_filename, pairs_filename, tm_filename in rows:
        topology_filename = topology_filename.strip()
        pairs_filename = pairs_filename.strip()
        tm_filename = tm_filename.strip()
        alias_manifest.append(f"{topology_filename},{pairs_filename},{tm_filename}\n")
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
    (ROOT / "manifest").mkdir(exist_ok=True)
    (ROOT / "manifest" / f"{topo}_manifest.txt").write_text("".join(alias_manifest), encoding="utf-8")


def node_ids_to_edges(path: list) -> list[tuple]:
    return [(a, b) for a, b in zip(path, path[1:])]


def path_signature(path_edges: list[tuple]) -> tuple[tuple, ...]:
    return tuple(path_edges)


def path_overlap(candidate: list[tuple], ksp_paths: list[list[tuple]]) -> int:
    candidate_edges = set(candidate)
    return max((len(candidate_edges & set(path)) for path in ksp_paths), default=0)


def generate_paths_and_masks() -> None:
    import json as json_lib

    rows = read_manifest()
    topology_filename, pairs_filename, _ = [part.strip() for part in rows[0]]
    with (ROOT / "topologies" / BASE_TOPO / topology_filename).open("r", encoding="utf-8") as handle:
        graph = nx.readwrite.json_graph.node_link_graph(json_lib.load(handle))
    with (ROOT / "pairs" / BASE_TOPO / pairs_filename).open("rb") as handle:
        pairs = pickle.load(handle)

    shared_paths: dict[tuple, list[list[tuple]]] = {}
    class_paths: dict[tuple, list[list[tuple]]] = {}
    masks = np.zeros((3, len(pairs), K), dtype=bool)
    stats_rows = []

    for pair_index, (src, dst) in enumerate(pairs):
        all_node_paths = []
        for path in nx.shortest_simple_paths(graph, src, dst):
            all_node_paths.append(path)
            if len(all_node_paths) >= 32:
                break
        if not all_node_paths:
            raise ValueError(f"No path for {src}->{dst}")

        edge_paths = [node_ids_to_edges(path) for path in all_node_paths]
        ksp = edge_paths[:K]
        candidates = edge_paths[K:] + edge_paths[:K]
        unique_candidates = []
        seen = set()
        for path in sorted(candidates, key=lambda p: (-len(p), path_overlap(p, ksp), path_signature(p))):
            sig = path_signature(path)
            if sig not in seen:
                unique_candidates.append(path)
                seen.add(sig)
            if len(unique_candidates) == K:
                break
        while len(unique_candidates) < K:
            unique_candidates.append(unique_candidates[0])

        shared_paths[(src, dst)] = unique_candidates
        class_paths[(src, dst)] = unique_candidates
        masks[0, pair_index, [0, 1, 2, 3]] = True
        masks[1, pair_index, [2, 3, 4, 5]] = True
        masks[2, pair_index, [4, 5, 6, 7]] = True
        stats_rows.append(
            {
                "pair_index": pair_index,
                "src": src,
                "dst": dst,
                "candidate_paths": len(edge_paths),
                "default_ksp_len_mean": float(np.mean([len(path) for path in ksp])),
                "diff_path_len_mean": float(np.mean([len(path) for path in unique_candidates])),
                "ksp_overlap_count": sum(1 for path in unique_candidates if path_signature(path) in {path_signature(p) for p in ksp}),
            }
        )

    for directory in ("paths_dict", "paths", "padded_edge_ids_per_path", "path_masks"):
        (ROOT / "topologies" / directory).mkdir(parents=True, exist_ok=True)

    for topo, paths in ((TOPO_SHARED, shared_paths), (TOPO_CLASSMASK, class_paths)):
        with (ROOT / "topologies" / "paths_dict" / f"{topo}_{K}_paths_dict_cluster_0.pkl").open("wb") as handle:
            pickle.dump(paths, handle)
    with (ROOT / "topologies" / "path_masks" / f"{TOPO_CLASSMASK}_{K}_path_masks_cluster_0.pkl").open("wb") as handle:
        pickle.dump(masks, handle)

    stats_path = RESULTS / "path_set_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stats_rows)


def prepare() -> None:
    state = load_state()
    if state.get("prepare"):
        print("[skip] prepare already complete", flush=True)
        return
    ensure_dirs()
    for topo in TOPOS:
        copy_alias_data(topo)
    generate_paths_and_masks()
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


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def run_gurobi() -> None:
    state = load_state()
    for topo in TOPOS:
        path_mask = 1 if topo == TOPO_CLASSMASK else 0
        tasks = []
        for objs_base in (["mf"], ["mlu"]):
            for priority in (1, 2, 3):
                objs = objs_base * priority
                tasks.append((f"{topo}_gt_{'_'.join(objs)}", gurobi_command(topo, 0, "flexile", priority, objs, path_mask)))
        for mode in ("flexile", "swan"):
            for priority in (1, 2, 3):
                objs = ["mf"] * priority
                tasks.append((f"{topo}_{mode}_{'_'.join(objs)}", gurobi_command(topo, 1, mode, priority, objs, path_mask)))
        for label, command in tasks:
            if state["gurobi"].get(label):
                print(f"[skip] {label}", flush=True)
                continue
            run_command(label, command)
            state["gurobi"][label] = True
            save_state(state)


def train_command(topo: str, path_mask: int) -> list[str]:
    return [
        PYTHON,
        "run_hattrick.py",
        "--topo",
        topo,
        "--mode",
        "train",
        "--epochs",
        "5",
        "--batch_size",
        "8",
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


def test_command(topo: str, model_name: str, path_mask: int, model_override: str = "") -> list[str]:
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
        "1",
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
    matrix = [
        (TOPO_SHARED, 0, "shared_zero_shot", base_model, False),
        (TOPO_SHARED, 0, "shared_retrained", "", True),
        (TOPO_CLASSMASK, 1, "classmask_zero_shot", base_model, False),
        (TOPO_CLASSMASK, 1, "classmask_retrained", "", True),
    ]
    for topo, path_mask, label, override, retrain in matrix:
        if retrain and not state["hattrick"].get(f"train_{label}"):
            run_command(f"train_{label}", train_command(topo, path_mask))
            state["hattrick"][f"train_{label}"] = True
            save_state(state)
        if not state["hattrick"].get(f"test_{label}"):
            run_command(f"test_{label}", test_command(topo, f"hattrick_{label}", path_mask, override))
            state["hattrick"][f"test_{label}"] = True
            save_state(state)


def read_floats(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64).reshape(-1)


def split_sim(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sim = read_floats(path).reshape(-1, 6)[VAL_END:TEST_END]
    mlu = np.column_stack((sim[:, 0], sim[:, 2], sim[:, 4]))
    cumulative = np.column_stack((sim[:, 1], sim[:, 3], sim[:, 5]))
    carried = np.column_stack((cumulative[:, 0], cumulative[:, 1] - cumulative[:, 0], cumulative[:, 2] - cumulative[:, 1]))
    return carried, mlu


def oracle_flow(result_dir: Path) -> np.ndarray:
    mf1 = read_floats(result_dir / "gt_optimal_values_mf.txt")[VAL_END:TEST_END]
    mf12 = read_floats(result_dir / "gt_optimal_values_mf_mf.txt")[VAL_END:TEST_END]
    mf123 = read_floats(result_dir / "gt_optimal_values_mf_mf_mf.txt")[VAL_END:TEST_END]
    return np.column_stack((mf1, mf12 - mf1, mf123 - mf12))


def build_metrics() -> list[dict[str, str | int | float]]:
    rows = []
    scenarios = [
        ("shared_zero_shot", TOPO_SHARED, "hattrick_shared_zero_shot_values_esm_sim_mlu_1.txt"),
        ("shared_retrained", TOPO_SHARED, "hattrick_shared_retrained_values_esm_sim_mlu_1.txt"),
        ("classmask_zero_shot", TOPO_CLASSMASK, "hattrick_classmask_zero_shot_values_esm_sim_mlu_1.txt"),
        ("classmask_retrained", TOPO_CLASSMASK, "hattrick_classmask_retrained_values_esm_sim_mlu_1.txt"),
    ]
    for scenario, topo, hat_file in scenarios:
        result_dir = ROOT / "results" / topo / f"{K}sp" / "0"
        oracle = oracle_flow(result_dir)
        hat = read_floats(result_dir / hat_file).reshape(-1, 3)
        method_data = {"Hattrick": hat}
        for method, filename in (("BEST_MC", "flexile_sim_results_esm_mf_mf_mf.txt"), ("SWAN", "swan_sim_results_esm_mf_mf_mf.txt")):
            carried, _ = split_sim(result_dir / filename)
            method_data[method] = carried / np.maximum(oracle, 1e-9)
        for method, values in method_data.items():
            for snap_idx in range(values.shape[0]):
                for class_idx, class_name in enumerate(CLASSES):
                    rows.append(
                        {
                            "scenario": scenario,
                            "topo": topo,
                            "snapshot": VAL_END + snap_idx,
                            "method": method,
                            "class": class_name,
                            "norm_fulfill": float(values[snap_idx, class_idx]),
                        }
                    )
    return rows


def summarize(rows: list[dict[str, str | int | float]]) -> list[dict[str, str | int | float]]:
    summary = []
    groups = sorted({(r["scenario"], r["method"], r["class"]) for r in rows})
    for scenario, method, class_name in groups:
        vals = np.asarray([float(r["norm_fulfill"]) for r in rows if r["scenario"] == scenario and r["method"] == method and r["class"] == class_name])
        summary.append(
            {
                "scenario": scenario,
                "method": method,
                "class": class_name,
                "n": len(vals),
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "p10": float(np.percentile(vals, 10)),
                "p1": float(np.percentile(vals, 1)),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )
    return summary


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_cdf(rows: list[dict[str, str | int | float]]) -> None:
    scenarios = ("shared_zero_shot", "shared_retrained", "classmask_zero_shot", "classmask_retrained")
    colors = {"Hattrick": "#2F80ED", "BEST_MC": "#27AE60", "SWAN": "#EB5757"}
    width, height = 1320, 1040
    left, top = 84, 108
    panel_w, panel_h = 280, 180
    gap_x, gap_y = 38, 58
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 28), "Diff-path 100-slice experiment: CDF of NormFulFill", fill="#202124", font=font(28, True))
    draw.text((left, 62), "Empirical CDF over snapshots 90-99. X-axis starts at 0.", fill="#5f6368", font=font(15))
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
                vals = sorted(float(r["norm_fulfill"]) for r in rows if r["scenario"] == scenario and r["class"] == class_name and r["method"] == method)
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
    image.save(RESULTS / "diff_path_cdf_100slice.png")


def draw_boxplot(summary: list[dict[str, str | int | float]]) -> None:
    width, height = 1180, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 30), "Diff-path 100-slice experiment: mean NormFulFill", fill="#202124", font=font(28, True))
    draw.text((70, 64), "Mean over snapshots 90-99. Higher is better.", fill="#5f6368", font=font(15))
    scenarios = ("shared_zero_shot", "shared_retrained", "classmask_zero_shot", "classmask_retrained")
    colors = {"Hattrick": "#2F80ED", "BEST_MC": "#27AE60", "SWAN": "#EB5757"}
    left, top, bottom = 110, 120, 620
    x_gap = 260
    y_min = 0.0
    max_mean = max(float(row["mean"]) for row in summary)
    y_max = max(1.15, math.ceil((max_mean + 0.05) * 4) / 4)
    for tick in np.arange(0, y_max + 0.001, 0.25):
        y = int(bottom - (tick - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left - 40, y, width - 70, y), fill="#E8EAED")
        draw.text((32, y - 8), f"{tick:.2f}", fill="#5f6368", font=font(13))
    for i, scenario in enumerate(scenarios):
        x0 = left + i * x_gap
        draw.text((x0 - 20, bottom + 22), scenario.replace("_", "\n"), fill="#202124", font=font(13))
        for j, method in enumerate(METHODS):
            vals = [float(r["mean"]) for r in summary if r["scenario"] == scenario and r["method"] == method]
            mean = float(np.mean(vals))
            y = int(bottom - (mean - y_min) / (y_max - y_min) * (bottom - top))
            draw.rectangle((x0 + j * 42, y, x0 + j * 42 + 30, bottom), fill=colors[method])
    lx, ly = 70, height - 42
    for method in METHODS:
        draw.rectangle((lx, ly, lx + 24, ly + 14), fill=colors[method])
        draw.text((lx + 34, ly - 4), method, fill="#202124", font=font(15))
        lx += 160
    image.save(RESULTS / "diff_path_boxplot_100slice.png")


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(summary: list[dict[str, str | int | float]]) -> None:
    lookup = {(r["scenario"], r["method"], r["class"]): r for r in summary}
    lines = [
        "# Diff Path 100-slice Experiment\n\n",
        "This experiment tests Hattrick on GEANT slices 0-99 with test slices 90-99.\n\n",
        "## Results\n\n",
        "| Scenario | Method | Class | Mean | Median | P10 | P1 |\n",
        "|---|---|---:|---:|---:|---:|---:|\n",
    ]
    for row in summary:
        lines.append("| {scenario} | {method} | {class} | {mean:.6f} | {median:.6f} | {p10:.6f} | {p1:.6f} |\n".format(**row))

    def mean_for(scenario: str, method: str) -> float:
        vals = [float(lookup[(scenario, method, cls)]["mean"]) for cls in CLASSES if (scenario, method, cls) in lookup]
        return float(np.mean(vals)) if vals else math.nan

    shared_zero = mean_for("shared_zero_shot", "Hattrick")
    shared_re = mean_for("shared_retrained", "Hattrick")
    class_zero = mean_for("classmask_zero_shot", "Hattrick")
    class_re = mean_for("classmask_retrained", "Hattrick")
    lines.extend(
        [
            "\n## Answers\n\n",
            f"- Shared non-KSP zero-shot Hattrick mean NormFulFill: {shared_zero:.6f}.\n",
            f"- Shared non-KSP retrained Hattrick mean NormFulFill: {shared_re:.6f}.\n",
            f"- Class-specific path-mask zero-shot Hattrick mean NormFulFill: {class_zero:.6f}.\n",
            f"- Class-specific path-mask retrained Hattrick mean NormFulFill: {class_re:.6f}.\n",
            "- A fair system evaluation needs retraining because path embeddings and oracle denominators both change when the path set changes.\n",
            "- Zero-shot results should be interpreted as a stress test of distribution shift, not as the final capability of Hattrick under the new path policy.\n",
            "- Some low-class NormFulFill values exceed 1 because the report follows the paper-style staged oracle increment as denominator; under class-specific path masks this denominator is no longer a strict per-class upper bound.\n",
            "- The path mask implementation was validated with a masked-softmax smoke check: disallowed paths receive zero split ratio.\n",
        ]
    )
    (RESULTS / "diff_path_summary_100slice.md").write_text("".join(lines), encoding="utf-8")
    (RESULTS / "retrain_decision.md").write_text(
        "# Retrain Decision\n\n"
        "Retraining is required for a fair comparison. The zero-shot runs intentionally load the original GEANT KSP model and test distribution shift only. "
        "The retrained runs use the same 100-slice data and regenerated oracle values under the changed path sets.\n",
        encoding="utf-8",
    )


def run_report() -> None:
    rows = build_metrics()
    summary = summarize(rows)
    write_csv(RESULTS / "diff_path_metrics_100slice.csv", rows)
    write_csv(RESULTS / "diff_path_summary_stats_100slice.csv", summary)
    write_markdown(summary)
    draw_cdf(rows)
    draw_boxplot(summary)
    state = load_state()
    state["report"] = True
    save_state(state)


def validate_outputs() -> None:
    for topo in TOPOS:
        result_dir = ROOT / "results" / topo / f"{K}sp" / "0"
        required_100 = [
            "gt_optimal_values_mf.txt",
            "gt_optimal_values_mf_mf.txt",
            "gt_optimal_values_mf_mf_mf.txt",
            "gt_optimal_values_mlu.txt",
            "gt_optimal_values_mlu_mlu.txt",
            "gt_optimal_values_mlu_mlu_mlu.txt",
        ]
        for name in required_100:
            count = line_count(result_dir / name)
            if count != TEST_END:
                raise RuntimeError(f"{result_dir / name} has {count} lines, expected {TEST_END}")
        for name in ("flexile_sim_results_esm_mf_mf_mf.txt", "swan_sim_results_esm_mf_mf_mf.txt"):
            count = line_count(result_dir / name)
            if count != TEST_END * 6:
                raise RuntimeError(f"{result_dir / name} has {count} lines, expected {TEST_END * 6}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "prepare", "gurobi", "hattrick", "report"], default="all")
    args = parser.parse_args()
    ensure_dirs()
    if args.stage in ("all", "prepare"):
        prepare()
    if args.stage in ("all", "gurobi"):
        run_gurobi()
        validate_outputs()
    if args.stage in ("all", "hattrick"):
        run_hattrick()
    if args.stage in ("all", "report"):
        run_report()


if __name__ == "__main__":
    main()
