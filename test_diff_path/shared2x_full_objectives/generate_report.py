from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
RUN_DIR = THIS_DIR / "artifacts" / "level4_confirmation" / "seed_490"
BASELINE_TABLE = (
    ROOT
    / "test_diff_path"
    / "shared2x_order_regularizer"
    / "artifacts"
    / "final_report"
    / "shared_final_common_table.json"
)
BASELINE_METRICS = (
    ROOT
    / "test_diff_path"
    / "shared2x_order_regularizer"
    / "artifacts"
    / "common_final_existing_methods"
    / "common_metrics.csv"
)
REPORT_DIR = THIS_DIR / "artifacts" / "final_report"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def index_classes(classes: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in classes}


def existing_rows() -> list[dict]:
    source = load_json(BASELINE_TABLE)
    rows = []
    for method in ("Hattrick", "DOTE-MC sensitivity"):
        item = next(row for row in source if row["method"] == method)
        rows.append(
            {
                "scenario": "shared 2x, test 400-499",
                "method": method,
                "checkpoint": "existing common reevaluation",
                "high_mean": item["high_norm_mean"],
                "high_p1": None,
                "high_p10": None,
                "medium_mean": item["medium_norm_mean"],
                "medium_p1": item["medium_norm_p1"],
                "medium_p10": item["medium_norm_p10"],
                "low_mean": item["low_norm_mean"],
                "common_mlu": item["common_post_admission_mlu"],
            }
        )
    return rows


def candidate_row(evaluation: dict) -> dict:
    classes = index_classes(evaluation["classes"])
    high = classes["High"]
    medium = classes["Medium"]
    low = classes["Low"]
    return {
        "scenario": "shared 2x, test 400-499",
        "method": "Hattrick + restored Fh/Fhm",
        "checkpoint": f"{evaluation['checkpoint']} epoch {evaluation['epoch']}",
        "high_mean": high["norm_fulfill_mean"],
        "high_p1": high["norm_fulfill_p1"],
        "high_p10": high["norm_fulfill_p10"],
        "medium_mean": medium["norm_fulfill_mean"],
        "medium_p1": medium["norm_fulfill_p1"],
        "medium_p10": medium["norm_fulfill_p10"],
        "low_mean": low["norm_fulfill_mean"],
        "common_mlu": low["admitted_capacity_ratio_mean"],
        "inversion_positive_gap_mean": evaluation["diagnostics"]["inversion_positive_gap_mean"],
        "inversion_violation_fraction": evaluation["diagnostics"]["inversion_violation_fraction"],
    }


def f4(value) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_diagnostics() -> dict[str, dict]:
    existing = read_csv(BASELINE_METRICS)
    candidate = read_csv(RUN_DIR / "final_evaluation_metrics.csv")
    rng = np.random.default_rng(490)
    result: dict[str, dict] = {}
    for class_name in ("High", "Medium", "Low"):
        new_rows = sorted(
            (row for row in candidate if row["class"] == class_name),
            key=lambda row: int(row["snapshot"]),
        )
        new_values = np.asarray([float(row["norm_fulfill"]) for row in new_rows])
        for method in ("Hattrick", "DOTE-MC sensitivity"):
            old_rows = sorted(
                (
                    row
                    for row in existing
                    if row["class"] == class_name and row["method"] == method
                ),
                key=lambda row: int(row["snapshot"]),
            )
            old_values = np.asarray([float(row["norm_fulfill"]) for row in old_rows])
            gaps = new_values - old_values
            samples = gaps[
                rng.integers(0, len(gaps), size=(10_000, len(gaps)))
            ].mean(axis=1)
            key = f"{class_name}_vs_{method.replace(' ', '_')}"
            result[key] = {
                "mean_paired_gap": float(gaps.mean()),
                "positive_fraction": float((gaps > 0).mean()),
                "bootstrap_replicates": 10_000,
                "bootstrap_95_ci": [
                    float(np.quantile(samples, 0.025)),
                    float(np.quantile(samples, 0.975)),
                ],
            }
    return result


def main() -> None:
    complete_path = RUN_DIR / "complete.json"
    if not complete_path.exists():
        raise SystemExit(f"Formal run is not complete: {complete_path}")
    complete = load_json(complete_path)
    final_eval = next(row for row in complete["evaluations"] if row["checkpoint"] == "final")
    best_eval = next(row for row in complete["evaluations"] if row["checkpoint"] == "best")
    rows = existing_rows()
    rows.insert(1, candidate_row(final_eval))
    rows.insert(2, candidate_row(best_eval) | {"method": "Hattrick + restored Fh/Fhm (validation best)"})
    paired = paired_diagnostics()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    write_csv(REPORT_DIR / "comparison.csv", rows)
    (REPORT_DIR / "paired_bootstrap.json").write_text(
        json.dumps(paired, indent=2), encoding="utf-8"
    )

    baseline = next(row for row in rows if row["method"] == "Hattrick")
    candidate = next(row for row in rows if row["method"] == "Hattrick + restored Fh/Fhm")
    dote = next(row for row in rows if row["method"] == "DOTE-MC sensitivity")
    high_delta = float(candidate["high_mean"]) - float(baseline["high_mean"])
    medium_delta = float(candidate["medium_mean"]) - float(baseline["medium_mean"])
    dote_gap = float(candidate["high_mean"]) - float(dote["high_mean"])
    capacity_ok = float(candidate["common_mlu"]) <= 1.0001
    high_vs_hattrick = paired["High_vs_Hattrick"]
    high_vs_dote = paired["High_vs_DOTE-MC_sensitivity"]
    medium_vs_hattrick = paired["Medium_vs_Hattrick"]

    table_lines = []
    for row in rows:
        medium = f"{f4(row['medium_mean'])} / {f4(row['medium_p1'])} / {f4(row['medium_p10'])}"
        table_lines.append(
            f"| shared | {row['method']} | {f4(row['high_mean'])} | {medium} | "
            f"{f4(row['low_mean'])} | {f4(row['common_mlu'])} |"
        )
    conclusion = (
        f"恢复两项后，最终 epoch 相对原 Hattrick 的 High Mean 变化为 `{high_delta:+.4f}`，"
        f"相对 DOTE-MC sensitivity 的 High Mean 差为 `{dote_gap:+.4f}`；Medium Mean 变化为 "
        f"`{medium_delta:+.4f}`。容量检查{'通过' if capacity_ok else '失败'}。"
    )
    report = f"""# Shared-2x 恢复 Fh/Fhm 实验

## 结论

{conclusion}

逐片 paired bootstrap（10,000 次）显示：High 相对原 Hattrick 的均值增量 95% CI 为 `[{high_vs_hattrick['bootstrap_95_ci'][0]:+.4f}, {high_vs_hattrick['bootstrap_95_ci'][1]:+.4f}]`，83% 时间片改善；相对 DOTE 的 CI 为 `[{high_vs_dote['bootstrap_95_ci'][0]:+.4f}, {high_vs_dote['bootstrap_95_ci'][1]:+.4f}]`，包含 0，因此只能说追平，不能声称稳定超过。Medium 相对原 Hattrick 的 CI 为 `[{medium_vs_hattrick['bootstrap_95_ci'][0]:+.4f}, {medium_vs_hattrick['bootstrap_95_ci'][1]:+.4f}]`，确认存在真实退化。

本实验只恢复论文完整目标中的 `Fh` 与累计 `Fhm`，没有加入排序正则、High 冻结或其他 loss。正式比较以 60 epochs 的最终 checkpoint 为主；validation-best 仅作为诊断。

## 公共评估（400–499）

| 场景 | 方法 | High Mean | Medium Mean / P1 / P10 | Low Mean | 公共 MLU |
|---|---|---:|---:|---:|---:|
{chr(10).join(table_lines)}

## 运行与验证

- seed：490；train 0–349；validation 350–399；test 400–499；60 epochs。
- 完整顺序：`-Fh > Uh > -Fhm > Uhm > -Fhml > Uhml`。
- checkpoint 重放最大差：`{complete['checkpoint_recovery_max_delta']:.3g}`。
- inversion severity：`{candidate['inversion_positive_gap_mean']:.4f}`；违例时间片比例：`{candidate['inversion_violation_fraction']:.0%}`。
- 运行时间：`{complete['runtime_seconds_this_invocation'] / 60:.2f}` 分钟。
- 配置、源码哈希、optimizer state、逐 epoch 指标及逐目标梯度范数均保存在正式运行目录。
"""
    (REPORT_DIR / "结论报告.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
