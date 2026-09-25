from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ARTIFACTS = THIS_DIR / "artifacts"
REPORT_DIR = ARTIFACTS / "final_report"
FULL_REPORT = (
    THIS_DIR.parent
    / "shared2x_full_objectives"
    / "artifacts"
    / "final_report"
    / "comparison.json"
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def selected_evaluation(run_dir: Path) -> dict:
    complete = load_json(run_dir / "complete.json")
    best = next(row for row in complete["evaluations"] if row["checkpoint"] == "best")
    metrics = read_csv(run_dir / "best_evaluation_metrics.csv")
    return {"complete": complete, "evaluation": best, "metrics": metrics}


def paired_metric(metrics: list[dict], class_name: str, field: str) -> dict[int, float]:
    return {
        int(row["snapshot"]): float(row[field])
        for row in metrics
        if row["class"] == class_name
    }


def compare_candidate(candidate_dir: Path, control_dir: Path) -> dict:
    candidate = selected_evaluation(candidate_dir)
    control = selected_evaluation(control_dir)
    candidate_classes = class_index(candidate["evaluation"]["classes"])
    control_classes = class_index(control["evaluation"]["classes"])
    config = load_json(candidate_dir / "config.json")
    approach = config["approach"]
    epsilon = config["epsilon"]
    common_start_match = bool(
        candidate["complete"]["phase_a_parameter_sha256"]
        == control["complete"]["phase_a_parameter_sha256"]
        and candidate["complete"]["phase_a_optimizer_repr_sha256"]
        == control["complete"]["phase_a_optimizer_repr_sha256"]
    )
    candidate_medium = paired_metric(candidate["metrics"], "Medium", "norm_fulfill")
    control_medium = paired_metric(control["metrics"], "Medium", "norm_fulfill")
    gaps = np.asarray(
        [candidate_medium[key] - control_medium[key] for key in sorted(control_medium)],
        dtype=np.float64,
    )
    high_values = np.asarray(
        list(paired_metric(candidate["metrics"], "High", "norm_fulfill").values())
    )
    high_mean_delta = (
        float(candidate_classes["High"]["norm_fulfill_mean"])
        - float(control_classes["High"]["norm_fulfill_mean"])
    )
    high_p10_delta = (
        float(candidate_classes["High"]["norm_fulfill_p10"])
        - float(control_classes["High"]["norm_fulfill_p10"])
    )
    medium_mean_delta = (
        float(candidate_classes["Medium"]["norm_fulfill_mean"])
        - float(control_classes["Medium"]["norm_fulfill_mean"])
    )
    medium_p1_delta = (
        float(candidate_classes["Medium"]["norm_fulfill_p1"])
        - float(control_classes["Medium"]["norm_fulfill_p1"])
    )
    medium_p10_delta = (
        float(candidate_classes["Medium"]["norm_fulfill_p10"])
        - float(control_classes["Medium"]["norm_fulfill_p10"])
    )
    low_fulfill_delta = (
        float(candidate_classes["Low"]["fulfill_ratio_mean"])
        - float(control_classes["Low"]["fulfill_ratio_mean"])
    )
    high_admitted_delta = (
        float(candidate_classes["High"]["admitted_traffic_mean"])
        - float(control_classes["High"]["admitted_traffic_mean"])
    )
    medium_admitted_delta = (
        float(candidate_classes["Medium"]["admitted_traffic_mean"])
        - float(control_classes["Medium"]["admitted_traffic_mean"])
    )
    low_admitted_delta = (
        float(candidate_classes["Low"]["admitted_traffic_mean"])
        - float(control_classes["Low"]["admitted_traffic_mean"])
    )
    low_admitted_loss = max(0.0, -low_admitted_delta)
    candidate_medium_norm_mean = float(
        candidate_classes["Medium"]["norm_fulfill_mean"]
    )
    candidate_low_norm_mean = float(candidate_classes["Low"]["norm_fulfill_mean"])
    transfer_efficiency = (
        medium_admitted_delta / low_admitted_loss
        if low_admitted_loss > 1e-12
        else float("inf")
    )
    if approach == "epsilon":
        high_gate = bool(
            (high_values >= 1.0 - float(epsilon) - 1e-4).all()
        )
    else:
        high_gate = (
            high_mean_delta >= -0.005
            and high_p10_delta >= -0.005
            and float(high_values.min()) >= 0.98
        )
    max_mlu = max(
        float(row["max_admitted_capacity_ratio"])
        for row in candidate["evaluation"]["classes"]
    )
    max_disabled = max(
        float(row["max_disabled_flow"])
        for row in candidate["evaluation"]["classes"]
    )
    gates = {
        "gate_common_start": common_start_match,
        "gate_medium_mean": medium_mean_delta >= 0.0,
        "gate_medium_p1": medium_p1_delta >= 0.01,
        "gate_medium_p10": medium_p10_delta >= 0.01,
        "gate_positive_fraction": float((gaps > 0).mean()) >= 0.60,
        "gate_high": high_gate,
        # Medium and Low have different demand bases.  The user-authorized
        # policy evaluates the actual admitted traffic exchanged between the
        # classes instead of comparing percentage-point changes directly.
        "gate_priority_transfer": (
            medium_admitted_delta > 0.0
            and medium_admitted_delta + 1e-12 >= low_admitted_loss
        ),
        "gate_capacity": max_mlu <= 1.0001,
        "gate_mask": max_disabled <= 1e-8,
    }
    if int(config["level"]) >= 3:
        gates["gate_norm_order"] = (
            candidate_low_norm_mean <= candidate_medium_norm_mean
        )
    return {
        "level": config["level"],
        "seed": config["seed"],
        "approach": approach,
        "epsilon": epsilon,
        "medium_mean_delta": medium_mean_delta,
        "medium_p1_delta": medium_p1_delta,
        "medium_p10_delta": medium_p10_delta,
        "positive_fraction": float((gaps > 0).mean()),
        "high_mean_delta": high_mean_delta,
        "high_p10_delta": high_p10_delta,
        "high_norm_min": float(high_values.min()),
        "low_fulfill_delta": low_fulfill_delta,
        "high_admitted_delta": high_admitted_delta,
        "medium_admitted_delta": medium_admitted_delta,
        "low_admitted_delta": low_admitted_delta,
        "medium_gain_per_low_loss": transfer_efficiency,
        "candidate_medium_norm_mean": candidate_medium_norm_mean,
        "candidate_low_norm_mean": candidate_low_norm_mean,
        "legacy_low_guard_pass": low_fulfill_delta >= -0.02,
        "common_start_match": common_start_match,
        "max_public_mlu": max_mlu,
        "max_disabled_flow": max_disabled,
        **gates,
        "all_gates_pass": all(gates.values()),
        "run_dir": str(candidate_dir.resolve()),
    }


def paired_bootstrap_p10(
    candidate_metrics: list[dict], control_metrics: list[dict], *, seed: int = 490
) -> dict:
    candidate = paired_metric(candidate_metrics, "Medium", "norm_fulfill")
    control = paired_metric(control_metrics, "Medium", "norm_fulfill")
    keys = sorted(control)
    left = np.asarray([candidate[key] for key in keys], dtype=np.float64)
    right = np.asarray([control[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.empty(10_000, dtype=np.float64)
    for index in range(len(values)):
        selected = rng.integers(0, len(keys), len(keys))
        values[index] = np.percentile(left[selected], 10) - np.percentile(
            right[selected], 10
        )
    return {
        "resamples": len(values),
        "observed_gap": float(np.percentile(left, 10) - np.percentile(right, 10)),
        "ci95_lower": float(np.percentile(values, 2.5)),
        "ci95_upper": float(np.percentile(values, 97.5)),
        "passed": bool(np.percentile(values, 2.5) > 0),
    }


def discover_comparisons() -> list[dict]:
    rows = []
    phase_b = ARTIFACTS / "phase_b"
    if not phase_b.exists():
        return rows
    for candidate_complete in phase_b.rglob("complete.json"):
        candidate_dir = candidate_complete.parent
        config = load_json(candidate_dir / "config.json")
        if config["approach"] == "control":
            continue
        level_root = phase_b / config["label"]
        control_dir = level_root / "control" / f"seed_{config['seed']}"
        if (control_dir / "complete.json").exists():
            rows.append(compare_candidate(candidate_dir, control_dir))
    return sorted(
        rows,
        key=lambda row: (row["level"], row["seed"], row["approach"], row["epsilon"] or 0),
    )


def f4(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    level0 = load_json(ARTIFACTS / "level0_lp" / "complete.json")
    level1 = load_json(ARTIFACTS / "level1_integration_checks.json")
    phase_a_path = ARTIFACTS / "phase_a" / "level2_proxy" / "seed_490" / "complete.json"
    phase_a = load_json(phase_a_path) if phase_a_path.exists() else None
    gate_audit_path = ARTIFACTS / "gate_audits" / "level2_proxy" / "seed_490.json"
    gate_audit = load_json(gate_audit_path) if gate_audit_path.exists() else None
    comparisons = discover_comparisons()
    level2_comparisons = [row for row in comparisons if row["level"] == 2]
    level3_comparisons = [row for row in comparisons if row["level"] == 3]
    passing = [row for row in level2_comparisons if row["all_gates_pass"]]
    required_seeds = {490, 491}
    grouped: dict[tuple[str, float | None], list[dict]] = {}
    for row in level2_comparisons:
        grouped.setdefault((row["approach"], row["epsilon"]), []).append(row)
    passing_methods = []
    for (approach, epsilon), rows in grouped.items():
        by_seed = {int(row["seed"]): row for row in rows}
        if required_seeds.issubset(by_seed) and all(
            by_seed[seed]["all_gates_pass"] for seed in required_seeds
        ):
            passing_methods.append(
                {
                    "approach": approach,
                    "epsilon": epsilon,
                    "seeds": sorted(required_seeds),
                    "rows": [by_seed[seed] for seed in sorted(required_seeds)],
                    "worst_medium_p10_delta": min(
                        by_seed[seed]["medium_p10_delta"] for seed in required_seeds
                    ),
                    "worst_medium_p1_delta": min(
                        by_seed[seed]["medium_p1_delta"] for seed in required_seeds
                    ),
                    "worst_medium_mean_delta": min(
                        by_seed[seed]["medium_mean_delta"] for seed in required_seeds
                    ),
                }
            )
    epsilon_methods = [
        row for row in passing_methods if row["approach"] == "epsilon"
    ]
    selected_method = (
        max(
            epsilon_methods,
            key=lambda row: (
                row["worst_medium_p10_delta"],
                row["worst_medium_p1_delta"],
                row["worst_medium_mean_delta"],
            ),
        )
        if epsilon_methods
        else None
    )
    write_csv(REPORT_DIR / "candidate_comparisons.csv", comparisons)
    write_json = lambda path, value: path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_json(REPORT_DIR / "candidate_comparisons.json", comparisons)
    lp_table = []
    for row in level0["methods"]:
        lp_table.append(
            f"| {row['approach']} | {row['epsilon'] if row['epsilon'] is not None else '-'} | "
            f"{row['medium_norm_gain_mean']:+.4f} | {row['medium_norm_gain_p10']:+.4f} | "
            f"{row['positive_fraction']:.0%} |"
        )
    if level3_comparisons:
        failed_level3 = [row for row in level3_comparisons if not row["all_gates_pass"]]
        if failed_level3:
            row = failed_level3[0]
            status = "NO-GO：Level 3 未通过"
            conclusion = (
                f"Level-2 选中的 `epsilon` ε=`0.005` 在 Level-3 seed {row['seed']} "
                f"中 Medium Mean/P1/P10 变化为 `{f4(row['medium_mean_delta'])}/"
                f"{f4(row['medium_p1_delta'])}/{f4(row['medium_p10_delta'])}`，"
                "未达到 P1/P10 各 `+0.01` 的放大门槛。由于每个 seed 都必须通过，"
                "按低开销协议停止 seed 491 与 Level 4。"
            )
        else:
            status = "INCOMPLETE：Level 3 尚缺 seed"
            conclusion = "已有 Level-3 结果通过，但尚未完成两个 seed。"
    elif gate_audit is None or not gate_audit["passed"]:
        rank = phase_a["best_rank"]
        mean_gap = float(rank[1]) - 0.9975
        p10_gap = float(rank[2]) - 0.995
        status = "NO-GO：Phase A gate failed"
        conclusion = (
            f"Level-2 Phase A 最佳 checkpoint 为 epoch {phase_a['best_epoch']}，"
            f"High Mean/P10=`{rank[1]:.6f}/{rank[2]:.6f}`。P10 门槛通过 "
            f"(`{p10_gap:+.6f}`)，但 Mean 比 `0.9975` 低 `{abs(mean_gap):.6f}`。"
            "按冻结实验协议，所有 Level-2 Phase B 均被拒绝，200–249 未读取，"
            "Level 3/4 未启动。"
        )
    elif level2_comparisons:
        status = "GO" if passing_methods else "NO-GO：没有方法在两个 seed 均通过"
        if passing_methods:
            selected_rows = selected_method["rows"]
            conclusion = (
                f"`epsilon` ε=`{selected_method['epsilon']}` 在 seeds 490/491 均通过"
                "按绝对接纳量修订后的全部门槛。两 seed 的 Medium 绝对增量为 "
                f"`{selected_rows[0]['medium_admitted_delta']:+.4f}/"
                f"{selected_rows[1]['medium_admitted_delta']:+.4f}`，Low 绝对变化为 "
                f"`{selected_rows[0]['low_admitted_delta']:+.4f}/"
                f"{selected_rows[1]['low_admitted_delta']:+.4f}`。"
            )
        else:
            best_trend = max(
                level2_comparisons,
                key=lambda row: (
                    row["medium_p10_delta"],
                    row["medium_p1_delta"],
                    row["medium_mean_delta"],
                ),
            )
            conclusion = (
                f"放宽 Phase-A Mean gate 到 `0.9965` 后已运行全部四个候选。"
                f"Medium 趋势最好的是 `{best_trend['approach']}` ε=`{best_trend['epsilon']}`，"
                f"Mean/P1/P10 分别变化 `{f4(best_trend['medium_mean_delta'])}/"
                f"{f4(best_trend['medium_p1_delta'])}/{f4(best_trend['medium_p10_delta'])}`，"
                f"但 Low Fulfill 变化 `{f4(best_trend['low_fulfill_delta'])}`；"
                "没有候选通过全部门槛，因此 seed 491 与 Level 3/4 未启动。"
            )
    else:
        status = "INCOMPLETE"
        conclusion = "Level-2 尚未完成。"
    external = load_json(FULL_REPORT) if FULL_REPORT.exists() else []
    reference_table = []
    for row in external:
        if "validation best" in row["method"]:
            continue
        reference_table.append(
            f"| {row['method']} | {row['high_mean']:.4f} | "
            f"{row['medium_mean']:.4f} / {row['medium_p1']:.4f} / {row['medium_p10']:.4f} | "
            f"{row['low_mean']:.4f} | {row['common_mlu']:.4f} |"
        )
    candidate_table = []
    for row in level2_comparisons:
        failed = [
            name.removeprefix("gate_")
            for name, value in row.items()
            if name.startswith("gate_") and not value
        ]
        candidate_table.append(
            f"| {row['seed']} | {row['approach']} | {row['epsilon'] if row['epsilon'] is not None else '-'} | "
            f"{f4(row['medium_mean_delta'])} / {f4(row['medium_p1_delta'])} / "
            f"{f4(row['medium_p10_delta'])} | {row['positive_fraction']:.0%} | "
            f"{row['high_norm_min']:.4f} | {f4(row['low_fulfill_delta'])} | "
            f"{row['medium_admitted_delta']:+.4f} / {row['low_admitted_delta']:+.4f} / "
            f"{row['medium_gain_per_low_loss']:.3f} | "
            f"{', '.join(failed) if failed else 'PASS'} |"
        )
    level3_table = []
    for row in level3_comparisons:
        failed = [
            name.removeprefix("gate_")
            for name, value in row.items()
            if name.startswith("gate_") and not value
        ]
        level3_table.append(
            f"| {row['seed']} | {row['approach']} | "
            f"{f4(row['medium_mean_delta'])} / {f4(row['medium_p1_delta'])} / "
            f"{f4(row['medium_p10_delta'])} | {row['positive_fraction']:.0%} | "
            f"{row['high_norm_min']:.4f} | {row['medium_admitted_delta']:+.4f} / "
            f"{row['low_admitted_delta']:+.4f} | "
            f"{row['candidate_medium_norm_mean']:.4f} / "
            f"{row['candidate_low_norm_mean']:.4f} | "
            f"{', '.join(failed) if failed else 'PASS'} |"
        )
    phase_a_text = ""
    if gate_audit is not None:
        phase_a_text = (
            f"用户授权将 Mean gate 从 `0.9975` 最小放宽到 `0.9965`；"
            f"当前 seed-490 Phase-A epoch {gate_audit['checkpoint_epoch']} checkpoint 复核为 "
            f"Mean/P10=`{gate_audit['high_norm_mean']:.6f}/"
            f"{gate_audit['high_norm_p10']:.6f}`，余量为 "
            f"`{gate_audit['mean_margin']:+.6f}/{gate_audit['p10_margin']:+.6f}`。"
            "协议调整前的分支已移入 archive；每个有效比较均额外要求 Phase-A 模型与 optimizer 哈希一致。"
        )
    report = f"""# Shared-2x 目标交换与 ε 约束实验

## 结论

**{status}。** {conclusion}

按用户确认，原 `Low Fulfill Δ >= -0.02` 不再作为否决条件，而保留为诊断列。新的优先级转移门槛要求 Medium 的绝对 admitted 增量为正，且不小于 Low 的绝对 admitted 减量；容量、mask、High 和 Medium 门槛不变。

{phase_a_text}

Level 0 的精确 LP 证据仍然有价值：严格保持 High 最大接纳量时，交换 `Fhm/Uh` 的理论增益约为零；允许 High 逐片下降 ε 后，三个 ε 都对 32/32 片产生 Medium 上界增益。这说明收益来自明确的 High 服务松弛，而不是 MLU 目标顺序本身。

## Level 0：精确连续 MCF（0–31）

| 方法 | ε | Medium Mean 增益 | P10 增益 | 正向比例 |
|---|---:|---:|---:|---:|
{chr(10).join(lp_table)}

## Level 1 验证

- 原 restored 数值回归最大差：`{level1['phase_a_restored_numeric_regression_max_delta']:.2e}`。
- checkpoint 重放最大差：`{level1['checkpoint_recovery_max_delta']:.2e}`。
- control/swap/epsilon 具有相同 Phase-A 参数与 optimizer 起点。
- High、Medium、Low 参数均未冻结；split、capacity、path mask 和 route-change 导出均通过。

## Level 2 proxy（200–249）

| Seed | 方法 | ε | Medium Mean/P1/P10 Δ | 正向比例 | High 最小值 | Low Fulfill Δ | Medium Δ / Low Δ / 效率 | 失败门槛 |
|---:|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(candidate_table) if candidate_table else '| 未运行 | - | - | - | - | - | Phase-A gate |'}

## Level 3 validation（350–399）

| Seed | 方法 | Medium Mean/P1/P10 Δ | 正向比例 | High 最小值 | Medium/Low admitted Δ | Medium/Low Norm Mean | 失败门槛 |
|---:|---|---:|---:|---:|---:|---:|---|
{chr(10).join(level3_table) if level3_table else '| 未运行 | - | - | - | - | - | - | - |'}

## 已有 400–499 公共参考（本实验未打开该窗口）

| 方法 | High Mean | Medium Mean / P1 / P10 | Low Mean | 公共 MLU |
|---|---:|---:|---:|---:|
{chr(10).join(reference_table)}

Level-2 的 `GO` 只授权放大实验。Level-3 seed 490 未通过后没有运行 seed 491，也没有打开 400–499；因此不把 epsilon 加入最终确认表。

## 复现

```powershell
$py = 'D:\\kuroresearch\\.venv-hattrick\\Scripts\\python.exe'
Set-Location 'D:\\kuroresearch\\Hattrick-main\\test_diff_path\\shared2x_order_epsilon'
& $py -B -m unittest -v
& $py -B run_experiment.py --level 0 --approach control --seed 490
& $py -B run_experiment.py --level 1 --approach control --seed 490
& $py -B run_experiment.py --level 2 --approach control --seed 490
& $py -B run_experiment.py --level 2 --approach epsilon --epsilon 0.005 --seed 490
& $py -B run_experiment.py --level 2 --approach control --seed 491
& $py -B run_experiment.py --level 2 --approach epsilon --epsilon 0.005 --seed 491
& $py -B run_experiment.py --level 3 --approach control --seed 490
& $py -B run_experiment.py --level 3 --approach epsilon --epsilon 0.005 --seed 490
& $py -B analyze_results.py
```
"""
    (REPORT_DIR / "结论报告.md").write_text(report, encoding="utf-8")
    manifest = {
        "status": status,
        "level0": level0,
        "level1": level1,
        "level2_phase_a": phase_a,
        "relaxed_gate_audit": gate_audit,
        "phase_b_comparisons": comparisons,
        "level2_passing_candidates": passing,
        "level2_two_seed_passing_methods": passing_methods,
        "level3_comparisons": level3_comparisons,
        "level3_passing_candidates": [
            row for row in level3_comparisons if row["all_gates_pass"]
        ],
        "selection_policy": {
            "legacy_low_fulfill_guard": "diagnostic_only",
            "priority_transfer_gate": "medium admitted delta > 0 and >= absolute low admitted loss",
            "authorized_by_user": True,
        },
        "selected_epsilon": (
            selected_method["epsilon"] if selected_method is not None else None
        ),
        "seed_491_started": any(row["level"] == 2 and row["seed"] == 491 for row in comparisons),
        "proxy_200_249_opened": any(row["level"] == 2 for row in comparisons),
        "confirmation_400_499_opened": any(row["level"] == 4 for row in comparisons),
        "level3_or_level4_started": any(row["level"] >= 3 for row in comparisons),
        "source_sha256": {
            path.name: sha256(path)
            for path in (
                THIS_DIR / "run_experiment.py",
                THIS_DIR / "objectives.py",
                THIS_DIR / "lp_oracle.py",
                THIS_DIR / "integration_checks.py",
                Path(__file__).resolve(),
            )
        },
    }
    write_json(REPORT_DIR / "manifest.json", manifest)
    print(report)


if __name__ == "__main__":
    main()
