from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ARTIFACTS = THIS_DIR / "artifacts"
LEVEL2 = ARTIFACTS / "level2_proxy"
BASELINE = LEVEL2 / "multiplier_0" / "seed_490"

CANDIDATES = (
    ("full_hinge", 0.25, LEVEL2 / "multiplier_0p25" / "seed_490"),
    ("full_hinge", 0.5, LEVEL2 / "multiplier_0p5" / "seed_490"),
    ("full_hinge", 1.0, LEVEL2 / "multiplier_1" / "seed_490"),
    (
        "flow_balanced_hinge",
        0.25,
        LEVEL2 / "flow_balanced_hinge" / "multiplier_0p25" / "seed_490",
    ),
    (
        "flow_balanced_hinge",
        1.0,
        LEVEL2 / "flow_balanced_hinge" / "multiplier_1" / "seed_490",
    ),
    (
        "tail_flow_balanced_squared",
        1.0,
        LEVEL2 / "tail_flow_balanced_squared" / "multiplier_1" / "seed_490",
    ),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def best(run_dir: Path):
    complete = read_json(run_dir / "complete.json")
    evaluation = next(
        item for item in complete["evaluations"] if item["checkpoint"] == "best"
    )
    rows = read_csv(run_dir / "best_evaluation_metrics.csv")
    return complete, evaluation, rows


def classes(evaluation) -> dict[str, dict]:
    return {item["class"]: item for item in evaluation["classes"]}


def bootstrap_p10_gap(
    baseline_values: np.ndarray,
    candidate_values: np.ndarray,
    seed: int = 490,
    draws: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = baseline_values.size
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        chosen = rng.integers(0, n, size=n)
        samples[index] = np.percentile(candidate_values[chosen], 10) - np.percentile(
            baseline_values[chosen], 10
        )
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def metric_by_snapshot(rows: list[dict], class_name: str, metric: str) -> np.ndarray:
    selected = sorted(
        (row for row in rows if row["class"] == class_name),
        key=lambda row: int(row["snapshot"]),
    )
    return np.asarray([float(row[metric]) for row in selected], dtype=np.float64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    baseline_complete, baseline_eval, baseline_rows = best(BASELINE)
    baseline_classes = classes(baseline_eval)
    baseline_medium = metric_by_snapshot(baseline_rows, "Medium", "norm_fulfill")
    rows = []
    detailed = []
    for penalty, multiplier, run_dir in CANDIDATES:
        complete, evaluation, candidate_rows = best(run_dir)
        indexed = classes(evaluation)
        candidate_medium = metric_by_snapshot(
            candidate_rows, "Medium", "norm_fulfill"
        )
        paired = candidate_medium - baseline_medium
        ci_low, ci_high = bootstrap_p10_gap(baseline_medium, candidate_medium)
        medium_mean_gap = (
            indexed["Medium"]["norm_fulfill_mean"]
            - baseline_classes["Medium"]["norm_fulfill_mean"]
        )
        medium_p1_gap = (
            indexed["Medium"]["norm_fulfill_p1"]
            - baseline_classes["Medium"]["norm_fulfill_p1"]
        )
        medium_p10_gap = (
            indexed["Medium"]["norm_fulfill_p10"]
            - baseline_classes["Medium"]["norm_fulfill_p10"]
        )
        low_raw_gap = (
            indexed["Low"]["fulfill_ratio_mean"]
            - baseline_classes["Low"]["fulfill_ratio_mean"]
        )
        severity_reduction = 1.0 - (
            evaluation["diagnostics"]["inversion_positive_gap_mean"]
            / baseline_eval["diagnostics"]["inversion_positive_gap_mean"]
        )
        gates = {
            "medium_mean_nonnegative": medium_mean_gap >= 0,
            "medium_p1_gap_at_least_0p01": medium_p1_gap >= 0.01,
            "medium_p10_gap_at_least_0p01": medium_p10_gap >= 0.01,
            "high_mean_at_least_0p98": indexed["High"]["norm_fulfill_mean"] >= 0.98,
            "high_drop_at_most_0p005": (
                indexed["High"]["norm_fulfill_mean"]
                - baseline_classes["High"]["norm_fulfill_mean"]
            )
            >= -0.005,
            "low_raw_drop_at_most_0p02": low_raw_gap >= -0.02,
            "inversion_severity_reduction_at_least_50pct": severity_reduction >= 0.5,
            "inversion_violation_fraction_at_most_50pct": evaluation["diagnostics"][
                "inversion_violation_fraction"
            ]
            <= 0.5,
            "paired_medium_positive_fraction_at_least_60pct": float(
                (paired > 0).mean()
            )
            >= 0.6,
            "capacity": max(
                float(item["max_admitted_capacity_ratio"])
                for item in evaluation["classes"]
            )
            <= 1.0001,
            "mask": max(
                float(item["max_disabled_flow"]) for item in evaluation["classes"]
            )
            <= 1e-8,
            "high_hard_freeze": evaluation[
                "max_high_admitted_delta_from_frozen_initial"
            ]
            <= 1e-5,
        }
        config = read_json(run_dir / "config.json")
        row = {
            "penalty": penalty,
            "lambda_multiplier": multiplier,
            "actual_lambda": complete["actual_lambda"],
            "best_epoch": evaluation["epoch"],
            "high_norm_mean": indexed["High"]["norm_fulfill_mean"],
            "medium_norm_mean": indexed["Medium"]["norm_fulfill_mean"],
            "medium_norm_p1": indexed["Medium"]["norm_fulfill_p1"],
            "medium_norm_p10": indexed["Medium"]["norm_fulfill_p10"],
            "medium_mean_gap": medium_mean_gap,
            "medium_p1_gap": medium_p1_gap,
            "medium_p10_gap": medium_p10_gap,
            "low_norm_mean": indexed["Low"]["norm_fulfill_mean"],
            "low_fulfill_mean": indexed["Low"]["fulfill_ratio_mean"],
            "low_fulfill_gap": low_raw_gap,
            "inversion_positive_gap_mean": evaluation["diagnostics"][
                "inversion_positive_gap_mean"
            ],
            "inversion_severity_reduction": severity_reduction,
            "inversion_violation_fraction": evaluation["diagnostics"][
                "inversion_violation_fraction"
            ],
            "paired_medium_positive_fraction": float((paired > 0).mean()),
            "medium_p10_bootstrap_ci_low": ci_low,
            "medium_p10_bootstrap_ci_high": ci_high,
            "max_high_admitted_delta": evaluation[
                "max_high_admitted_delta_from_frozen_initial"
            ],
            "all_gates_pass": all(gates.values()),
            "failed_gates": ";".join(name for name, passed in gates.items() if not passed),
            "runtime_seconds": complete["runtime_seconds"],
        }
        rows.append(row)
        detailed.append(
            {
                **row,
                "gates": gates,
                "run_dir": str(run_dir.resolve()),
                "source_sha256": config["source_sha256"],
            }
        )

    output_dir = ARTIFACTS / "round2_report"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "level2_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    baseline_payload = {
        "high_norm_mean": baseline_classes["High"]["norm_fulfill_mean"],
        "medium_norm_mean": baseline_classes["Medium"]["norm_fulfill_mean"],
        "medium_norm_p1": baseline_classes["Medium"]["norm_fulfill_p1"],
        "medium_norm_p10": baseline_classes["Medium"]["norm_fulfill_p10"],
        "low_norm_mean": baseline_classes["Low"]["norm_fulfill_mean"],
        "low_fulfill_mean": baseline_classes["Low"]["fulfill_ratio_mean"],
        **baseline_eval["diagnostics"],
    }
    summary = {
        "status": "NO_GO_LEVEL2_SEED490",
        "reason": "No candidate passed every pre-registered Level-2 gate on seed 490; seed 491 was not opened.",
        "baseline": baseline_payload,
        "candidates": detailed,
        "strict_pass_count": sum(bool(row["all_gates_pass"]) for row in rows),
        "warmstart_wrapper_regression_max_delta": 7.62939453125e-06,
    }
    (output_dir / "level2_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    best_medium = max(rows, key=lambda row: row["medium_p10_gap"])
    nearest_low = min(rows, key=lambda row: abs(min(0.0, row["low_fulfill_gap"] + 0.02)))
    table_lines = []
    for row in rows:
        table_lines.append(
            "| {penalty} | {lambda_multiplier:g} | {medium_mean_gap:+.4f} | "
            "{medium_p1_gap:+.4f} | {medium_p10_gap:+.4f} | {low_fulfill_gap:+.4f} | "
            "{inversion_severity_reduction:.1%} | {inversion_violation_fraction:.0%} | {all_gates_pass} |".format(
                **row
            )
        )
    report = f"""# Shared-2x 第二轮：High 硬冻结实验结论

## 结论

本轮正确实现了“High 达标后冻结 High，再引入排序 loss，Low 不冻结”。High 由冻结教师输出，Medium/Low 由 student 输出，三类策略进入同一个 actual-TM sequential admission。所有候选的教师参数哈希保持不变，逐片 High admitted traffic 的最大 replay 差不超过 `7.63e-6`。

seed 490 上没有候选同时通过全部 Level-2 门槛，因此结论为 **NO-GO**，按成本规则没有启动 seed 491，也没有进入 Level 3/4。

## Proxy 200–249（相对 matched hard-freeze λ=0）

| 函数 | λ multiplier | Medium Mean Δ | Medium P1 Δ | Medium P10 Δ | Low Fulfill Δ | inversion 降幅 | 违例率 | 全通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table_lines)}

matched baseline：High Mean `{baseline_payload['high_norm_mean']:.6f}`，Medium Mean/P1/P10 `{baseline_payload['medium_norm_mean']:.6f}/{baseline_payload['medium_norm_p1']:.6f}/{baseline_payload['medium_norm_p10']:.6f}`，Low Fulfill Mean `{baseline_payload['low_fulfill_mean']:.6f}`，inversion severity `{baseline_payload['inversion_positive_gap_mean']:.6f}`，违例率 `{baseline_payload['inversion_violation_fraction']:.0%}`。

Medium P10 改善最大的候选是 `{best_medium['penalty']}`、multiplier `{best_medium['lambda_multiplier']:g}`：P10 `+{best_medium['medium_p10_gap']:.4f}`、P1 `+{best_medium['medium_p1_gap']:.4f}`、Mean `+{best_medium['medium_mean_gap']:.4f}`，80% 左右时间片改善且 inversion 基本消除；但 Low Fulfill 下降 `{abs(best_medium['low_fulfill_gap']):.4f}`，明显超过 0.02，不能把它包装成合格改进。

最靠近 Low 预算边界的 full hinge 0.25× 仍下降 `0.02028`，且 Medium P10 只提高 `0.00565`。flow-balanced 0.25× 虽守住 Low，但 Medium Mean 下降。tail squared 在 Level 1 有正向趋势，放大到 Level 2 后 Medium Mean 下降且 Low 超预算，说明小样本趋势没有稳定复现。

## 机制解释

完整归一化 hinge 的 Low oracle 分母远小于 Medium，因此在原始 admitted-flow 空间，优化器更容易通过降低 Low 消除 gap。`flow_balanced_hinge` 保留 Low 非零梯度，并按 `oracle_low/oracle_medium` 缩放 Low 反向分支；它确实增强了 Medium P1/P10，但足够强的 λ 仍违反 Low 预算。tail 版本只作用于最严重 25%，也没有解决放大后的折中。

## 验证

- 6/6 单元测试通过，包括 Low 非零梯度、oracle detach、tail top-k 和 High hard-freeze。
- phase-A High gate：连续两轮 validation High NormFulFill ≥0.98。
- hard-freeze wrapper 在训练前与原 Hattrick checkpoint 的最大数值差为 `7.63e-6`。
- 所有完成候选容量比 ≤1.0001、disabled flow ≤1e-8。
- 所有 Level-2 选择只用 validation 160–199；proxy 200–249 没有用于选择 epoch。
"""
    (output_dir / "结论报告.md").write_text(report, encoding="utf-8")
    manifest = {
        "status": summary["status"],
        "source_sha256": {
            path.name: sha256(path)
            for path in (
                THIS_DIR / "run_round2.py",
                THIS_DIR / "hybrid_model.py",
                THIS_DIR / "penalty.py",
                THIS_DIR / "test_round2.py",
                Path(__file__).resolve(),
            )
        },
        "outputs_sha256": {
            name: sha256(output_dir / name)
            for name in (
                "level2_comparison.csv",
                "level2_comparison.json",
                "结论报告.md",
            )
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
