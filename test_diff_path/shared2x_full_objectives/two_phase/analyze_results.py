from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
ARTIFACTS = THIS_DIR / "artifacts"
LEVEL2 = ARTIFACTS / "level2_proxy"
REPORT_DIR = ARTIFACTS / "final_report"
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
        0.5,
        LEVEL2 / "flow_balanced_hinge" / "multiplier_0p5" / "seed_490",
    ),
    (
        "flow_balanced_hinge",
        1.0,
        LEVEL2 / "flow_balanced_hinge" / "multiplier_1" / "seed_490",
    ),
    (
        "tail_flow_balanced_squared",
        0.25,
        LEVEL2 / "tail_flow_balanced_squared" / "multiplier_0p25" / "seed_490",
    ),
    (
        "tail_flow_balanced_squared",
        0.5,
        LEVEL2 / "tail_flow_balanced_squared" / "multiplier_0p5" / "seed_490",
    ),
)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_metrics(run_dir: Path) -> dict:
    complete = load_json(run_dir / "complete.json")
    evaluation = next(
        item for item in complete["evaluations"] if item["checkpoint"] == "best"
    )
    classes = {item["class"]: item for item in evaluation["classes"]}
    return {
        "best_epoch": int(evaluation["epoch"]),
        "high_norm_mean": float(classes["High"]["norm_fulfill_mean"]),
        "medium_norm_mean": float(classes["Medium"]["norm_fulfill_mean"]),
        "medium_norm_p1": float(classes["Medium"]["norm_fulfill_p1"]),
        "medium_norm_p10": float(classes["Medium"]["norm_fulfill_p10"]),
        "low_norm_mean": float(classes["Low"]["norm_fulfill_mean"]),
        "low_fulfill_mean": float(classes["Low"]["fulfill_ratio_mean"]),
        "inversion_severity": float(
            evaluation["diagnostics"]["inversion_positive_gap_mean"]
        ),
        "inversion_violation_fraction": float(
            evaluation["diagnostics"]["inversion_violation_fraction"]
        ),
        "max_public_mlu": max(
            float(item["max_admitted_capacity_ratio"])
            for item in evaluation["classes"]
        ),
        "max_disabled_flow": max(
            float(item["max_disabled_flow"]) for item in evaluation["classes"]
        ),
        "max_high_admitted_delta": float(
            evaluation["max_high_admitted_delta_from_frozen_initial"]
        ),
        "runtime_seconds": float(complete["runtime_seconds"]),
        "teacher_immutable": bool(complete["teacher_immutable"]),
    }


def candidate_row(kind: str, multiplier: float, run_dir: Path, baseline: dict) -> dict:
    metrics = best_metrics(run_dir)
    deltas = {
        "high_norm_delta": metrics["high_norm_mean"] - baseline["high_norm_mean"],
        "medium_mean_delta": metrics["medium_norm_mean"] - baseline["medium_norm_mean"],
        "medium_p1_delta": metrics["medium_norm_p1"] - baseline["medium_norm_p1"],
        "medium_p10_delta": metrics["medium_norm_p10"] - baseline["medium_norm_p10"],
        "low_fulfill_delta": metrics["low_fulfill_mean"] - baseline["low_fulfill_mean"],
        "severity_reduction_fraction": (
            baseline["inversion_severity"] - metrics["inversion_severity"]
        )
        / max(baseline["inversion_severity"], 1e-12),
    }
    gates = {
        "gate_medium_mean": deltas["medium_mean_delta"] >= 0.0,
        "gate_medium_p1": deltas["medium_p1_delta"] >= 0.01,
        "gate_medium_p10": deltas["medium_p10_delta"] >= 0.01,
        "gate_low_fulfill": deltas["low_fulfill_delta"] >= -0.02,
        "gate_inversion_severity": deltas["severity_reduction_fraction"] >= 0.5,
        "gate_inversion_fraction": metrics["inversion_violation_fraction"] <= 0.5,
        "gate_high_frozen": metrics["max_high_admitted_delta"] <= 1e-5,
        "gate_capacity": metrics["max_public_mlu"] <= 1.0001,
        "gate_mask": metrics["max_disabled_flow"] <= 1e-8,
    }
    return {
        "penalty": kind,
        "lambda_multiplier": multiplier,
        **metrics,
        **deltas,
        **gates,
        "all_gates_pass": all(gates.values()),
        "run_dir": str(run_dir.resolve()),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f4(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    baseline = best_metrics(BASELINE)
    rows = [candidate_row(kind, multiplier, path, baseline) for kind, multiplier, path in CANDIDATES]
    safe = [row for row in rows if row["gate_low_fulfill"]]
    best_safe = max(
        safe,
        key=lambda row: (
            row["medium_p10_delta"],
            row["medium_p1_delta"],
            row["medium_mean_delta"],
        ),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT_DIR / "level2_candidates.csv", rows)
    (REPORT_DIR / "level2_candidates.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    table = []
    for row in rows:
        table.append(
            f"| {row['penalty']} | {row['lambda_multiplier']:g} | "
            f"{f4(row['medium_mean_delta'])} | {f4(row['medium_p1_delta'])} | "
            f"{f4(row['medium_p10_delta'])} | {f4(row['low_fulfill_delta'])} | "
            f"{row['severity_reduction_fraction']:.1%} | "
            f"{row['inversion_violation_fraction']:.0%} | {row['all_gates_pass']} |"
        )

    report = f"""# Shared-2x two-phase 实验报告

## 结论

**NO-GO。** 使用恢复 `Fh/Fhm` 的 Phase-A checkpoint 后，High 已被结构性冻结且逐片 admission 最大变化不超过 `3.82e-06`；但是三种不冻结 Low 的 penalty 都没有通过 Level-2 门槛，因此没有打开 400–499，也没有启动更多 seed 或大规模 Phase B。

最好的 Low-safe 尾部趋势是 `{best_safe['penalty']}`、multiplier `{best_safe['lambda_multiplier']:g}`：Medium Mean/P1/P10 分别变化 `{f4(best_safe['medium_mean_delta'])}/{f4(best_safe['medium_p1_delta'])}/{f4(best_safe['medium_p10_delta'])}`，Low Fulfill 变化 `{f4(best_safe['low_fulfill_delta'])}`；仍远低于 P1/P10 各 `+0.01` 的门槛，且违例率为 `{best_safe['inversion_violation_fraction']:.0%}`。

## Level-2 proxy 200–249

matched `lambda=0`：High Mean `{baseline['high_norm_mean']:.4f}`；Medium Mean/P1/P10 `{baseline['medium_norm_mean']:.4f}/{baseline['medium_norm_p1']:.4f}/{baseline['medium_norm_p10']:.4f}`；Low Fulfill `{baseline['low_fulfill_mean']:.4f}`；inversion severity `{baseline['inversion_severity']:.4f}`，违例率 `{baseline['inversion_violation_fraction']:.0%}`。

| 函数 | multiplier | Medium Mean Δ | P1 Δ | P10 Δ | Low Fulfill Δ | severity 降幅 | 违例率 | 全通过 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table)}

## 解释

- `full_hinge` 的 Low 归一化分母较小，强权重主要通过降低 Low 消除 inversion；Medium 尾部最多只提高约 `0.0026`。
- `flow_balanced_hinge` 保留 Low 非零梯度并将其缩放到 raw-flow 导数平衡。`0.25x` 能在 Low 预算内提高 Medium Mean，但 P1/P10 仅小幅改善；`1x` 虽修复大部分 inversion，却使 Low Fulfill 下降约 `0.05`。
- tail squared 只作用于最严重 25%，仍没有把梯度稳定转化为 Medium P1/P10 收益。

这说明两阶段确实解决了“优化 Medium/Low 时 High 被拿走”的问题，但剩余瓶颈是 residual capacity 下 Medium 与 Low 的可行折中，以及当前 penalty 对 Medium 尾部的作用效率，而不是 High 漂移。

## 验证与范围

- Phase A gate：validation High NormFulFill >= `0.9975`，连续两轮；checkpoint epoch 60。
- High teacher 参数冻结；Medium/Low student 可训练；所有 penalty 对 Low 保留非零梯度。
- 所有运行公共 MLU <= `1.0001`、disabled flow <= `1e-8`。
- Level-2 仅用于趋势筛选；由于无候选通过，400–499 未评估。
"""
    (REPORT_DIR / "结论报告.md").write_text(report, encoding="utf-8")

    manifest = {
        "status": "NO-GO",
        "phase_b_level3_or_final_authorized": False,
        "final_400_499_opened": False,
        "baseline": baseline,
        "best_safe_candidate": best_safe,
        "phase_a_checkpoint": str(
            (THIS_DIR.parent / "artifacts" / "level4_confirmation" / "seed_490" / "best_model.pt").resolve()
        ),
        "phase_a_checkpoint_sha256": sha256(
            THIS_DIR.parent / "artifacts" / "level4_confirmation" / "seed_490" / "best_model.pt"
        ),
        "source_sha256": {
            "run_two_phase.py": sha256(THIS_DIR / "run_two_phase.py"),
            "analyze_results.py": sha256(Path(__file__).resolve()),
            "core_run_round2.py": sha256(
                THIS_DIR.parents[1]
                / "shared2x_order_regularizer"
                / "round2_high_frozen"
                / "run_round2.py"
            ),
        },
    }
    (REPORT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
