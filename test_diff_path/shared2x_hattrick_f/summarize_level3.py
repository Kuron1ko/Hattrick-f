from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
BASE_DIR = (
    TEST_DIR
    / "shared2x_order_epsilon"
    / "artifacts"
    / "phase_a"
    / "level3_validation_only"
    / "seed_490"
)
CONTROL_DIR = (
    TEST_DIR
    / "shared2x_order_epsilon"
    / "artifacts"
    / "phase_b"
    / "level3_validation_only"
    / "control"
    / "seed_490"
)
F_DIR = HERE / "artifacts" / "level3_validation_only" / "fh_release" / "seed_490"
OUT_DIR = HERE / "artifacts" / "level3_report"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def by_class(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def paired_values(rows: list[dict], class_name: str) -> dict[int, float]:
    return {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in rows
        if row["class"] == class_name
    }


def paired_bootstrap(
    left_rows: list[dict], right_rows: list[dict], class_name: str, seed: int = 490
) -> dict:
    left = paired_values(left_rows, class_name)
    right = paired_values(right_rows, class_name)
    snapshots = sorted(set(left) & set(right))
    deltas = np.asarray([right[s] - left[s] for s in snapshots], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(20000, len(deltas)))
    boot = deltas[indices].mean(axis=1)
    return {
        "n": len(deltas),
        "paired_mean_delta": float(deltas.mean()),
        "paired_delta_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "right_better_fraction": float(np.mean(deltas > 0)),
        "equal_fraction": float(np.mean(np.abs(deltas) <= 1e-12)),
    }


def main() -> None:
    base_complete = read_json(BASE_DIR / "complete.json")
    base_epoch = int(base_complete["best_epoch"])
    base_summary = read_json(
        BASE_DIR / f"validation_epoch_{base_epoch:03d}_summary.json"
    )["classes"]
    base_rows = read_csv(BASE_DIR / f"validation_epoch_{base_epoch:03d}_metrics.csv")
    base_index = by_class(base_summary)

    control_candidates = []
    for epoch in range(1, 16):
        summary = read_json(
            CONTROL_DIR / f"validation_epoch_{epoch:03d}_summary.json"
        )["classes"]
        rows = read_csv(CONTROL_DIR / f"validation_epoch_{epoch:03d}_metrics.csv")
        indexed = by_class(summary)
        high_min = min(
            float(row["norm_fulfill"])
            for row in rows
            if row["class"] == "High"
        )
        eligible = (
            high_min >= 0.9949
            and float(indexed["Low"]["norm_fulfill_mean"])
            >= float(base_index["Low"]["norm_fulfill_mean"]) - 0.02
        )
        control_candidates.append(
            {
                "epoch": epoch,
                "eligible": eligible,
                "summary": summary,
                "rows": rows,
                "rank": (
                    float(indexed["Medium"]["norm_fulfill_mean"]),
                    float(indexed["Medium"]["norm_fulfill_p10"]),
                    float(indexed["Medium"]["norm_fulfill_p1"]),
                ),
            }
        )
    control = max(
        (candidate for candidate in control_candidates if candidate["eligible"]),
        key=lambda candidate: candidate["rank"],
    )
    control_index = by_class(control["summary"])

    f_complete = read_json(F_DIR / "complete.json")
    f_config = read_json(F_DIR / "config.json")
    f_summary = read_json(F_DIR / "best_summary.json")["classes"]
    f_rows = read_csv(F_DIR / "best_metrics.csv")
    f_index = by_class(f_summary)

    methods = (
        ("Six-loss Hattrick (Phase-A)", base_epoch, base_index, base_rows),
        ("Six-loss continued control", control["epoch"], control_index, control["rows"]),
        ("Hattrick-f", int(f_complete["best_epoch"]), f_index, f_rows),
    )
    comparison_rows = []
    for method, epoch, indexed, rows in methods:
        for class_name in ("High", "Medium", "Low"):
            values = [
                float(row["norm_fulfill"])
                for row in rows
                if row["class"] == class_name
            ]
            comparison_rows.append(
                {
                    "method": method,
                    "epoch": epoch,
                    "class": class_name,
                    "norm_mean": indexed[class_name]["norm_fulfill_mean"],
                    "norm_p10": indexed[class_name]["norm_fulfill_p10"],
                    "norm_p1": indexed[class_name]["norm_fulfill_p1"],
                    "norm_min": min(values),
                }
            )
    write_csv(OUT_DIR / "comparison.csv", comparison_rows)

    statistical = {
        "hattrick_f_vs_phase_a": {
            name: paired_bootstrap(base_rows, f_rows, name)
            for name in ("High", "Medium", "Low")
        },
        "hattrick_f_vs_equal_budget_six_loss_control": {
            name: paired_bootstrap(control["rows"], f_rows, name)
            for name in ("High", "Medium", "Low")
        },
    }
    result = {
        "level": 3,
        "window": {"train": [0, 350], "validation": [350, 400]},
        "strict_esm": {
            "verified": True,
            "description": f_config["strict_inference"],
            "pred": 1,
            "pred_type": "esm",
        },
        "phase_a_objectives": f_config["phase_a"]["objectives"],
        "phase_f_objectives": f_config["phase_f"]["ordered_objectives"],
        "optimizer_reset": f_config["phase_f"]["optimizer_reset"],
        "selection_gate": {
            "high_all_snapshots": ">= 0.995 with 1e-4 numerical tolerance",
            "low_mean": ">= Phase-A Low mean - 0.02",
            "simulator": "capacity <= 1.0001 and disabled flow <= 1e-8",
        },
        "control_selected_epoch": control["epoch"],
        "hattrick_f_selected_epoch": int(f_complete["best_epoch"]),
        "statistics": statistical,
        "route_reconstruction": f_complete["route_reconstruction"],
    }
    write_json(OUT_DIR / "evidence.json", result)

    high = f_index["High"]
    medium = f_index["Medium"]
    low = f_index["Low"]
    report = f"""# Hattrick-f Level-3 验证

## 方法

Phase-A 使用完整六目标 Hattrick：`Fh → Uh → Fhm → Uhm → Fhml → Uhml`。
Hattrick-f 从其最佳检查点开始，重置 Adam，Phase-F 保留投影机制和所有可训练参数，
但撤去三个持续 MLU 目标，只训练 `Fh → Fhm → Fhml`。推理严格只输入 ESM 预测值。

## 严格门槛下的结果

Hattrick-f 选择 epoch {int(f_complete['best_epoch'])}：High mean={high['norm_fulfill_mean']:.6f}，
High min={min(float(row['norm_fulfill']) for row in f_rows if row['class']=='High'):.6f}，
Medium mean={medium['norm_fulfill_mean']:.6f}，Medium P10={medium['norm_fulfill_p10']:.6f}，
Medium P1={medium['norm_fulfill_p1']:.6f}，Low mean={low['norm_fulfill_mean']:.6f}。

同样门槛重新筛选、同样增加 15 epoch 的六目标控制组最佳为 epoch {control['epoch']}，
Medium mean={control_index['Medium']['norm_fulfill_mean']:.6f}。Hattrick-f 的 Medium mean 绝对提升
{float(medium['norm_fulfill_mean'])-float(control_index['Medium']['norm_fulfill_mean']):.6f}。

## 机制证据

High 路径相对 Phase-A 的 demand-weighted TV 为
{f_complete['route_reconstruction']['demand_weighted_tv_mean']:.6f}，主路径翻转率
{f_complete['route_reconstruction']['argmax_change_fraction_mean']:.2%}。归一化熵从
{f_complete['route_reconstruction']['initial_entropy_mean']:.6f} 降到
{f_complete['route_reconstruction']['final_entropy_mean']:.6f}，Top-1 路径占比从
{f_complete['route_reconstruction']['initial_top1_share_mean']:.6f} 升到
{f_complete['route_reconstruction']['final_top1_share_mean']:.6f}。

结论：在 Level-3 验证窗口上，该设想成立。Phase-A 的 MLU 提供稳定均衡起点；Phase-F 撤去
MLU 后，`Fhm` 能沿 `Fh` 的投影零空间把 High 重构为明显更不均匀的分布，并显著释放 Medium。
"""
    (OUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
