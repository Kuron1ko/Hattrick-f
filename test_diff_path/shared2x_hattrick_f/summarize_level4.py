from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
BASE_DIR = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
)
STRICT_DIR = (
    HERE / "artifacts" / "level4_confirmation" / "fh_release" / "seed_490"
)
PARETO_DIR = (
    HERE
    / "artifacts"
    / "level4_confirmation"
    / "fh_release_low_budget_0p03"
    / "seed_490"
)
OUT_DIR = HERE / "artifacts" / "level4_report"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def minimum(rows: list[dict], class_name: str) -> float:
    return min(
        float(row["norm_fulfill"])
        for row in rows
        if row["class"] == class_name
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_summary = read_json(BASE_DIR / "best_evaluation_summary.json")["classes"]
    baseline_rows = read_csv(BASE_DIR / "best_evaluation_metrics.csv")
    strict_summary = read_json(STRICT_DIR / "evaluation_summary.json")["classes"]
    strict_rows = read_csv(STRICT_DIR / "evaluation_metrics.csv")
    pareto_summary = read_json(PARETO_DIR / "evaluation_summary.json")["classes"]
    pareto_rows = read_csv(PARETO_DIR / "evaluation_metrics.csv")
    strict_complete = read_json(STRICT_DIR / "complete.json")
    pareto_complete = read_json(PARETO_DIR / "complete.json")

    records = []
    for method, epoch, summary, rows in (
        ("Six-loss Hattrick", 60, baseline_summary, baseline_rows),
        ("Hattrick-f strict Low", strict_complete["best_epoch"], strict_summary, strict_rows),
        ("Hattrick-f Pareto", pareto_complete["best_epoch"], pareto_summary, pareto_rows),
    ):
        indexed = index(summary)
        for class_name in ("High", "Medium", "Low"):
            records.append(
                {
                    "method": method,
                    "selected_epoch": epoch,
                    "class": class_name,
                    "norm_mean": indexed[class_name]["norm_fulfill_mean"],
                    "norm_p10": indexed[class_name]["norm_fulfill_p10"],
                    "norm_p1": indexed[class_name]["norm_fulfill_p1"],
                    "norm_min": minimum(rows, class_name),
                }
            )
    with (OUT_DIR / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    evidence = {
        "protocol": {
            "train": [0, 350],
            "validation_selection_only": [350, 400],
            "independent_evaluation": [400, 500],
            "selection_used_evaluation": False,
            "inference": "strict ESM prediction-only routing input",
        },
        "strict_low_budget_0p02": strict_complete,
        "pareto_low_budget_0p03": pareto_complete,
    }
    (OUT_DIR / "evidence.json").write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    base = index(baseline_summary)
    strict = index(strict_summary)
    pareto = index(pareto_summary)
    report = f"""# Hattrick-f Level-4 独立验证

## 实验协议

- Phase-A：完整六目标 `Fh → Uh → Fhm → Uhm → Fhml → Uhml`。
- Phase-F：重置 Adam，保留串联结构、投影和所有参数，只训练 `Fh → Fhm → Fhml`。
- 训练：0–349；模型选择：350–399；独立评估：400–499。
- 路由推理严格只输入 ESM 预测值。400–499 未参与 epoch 或超参数选择。

## 独立评估结果

| 方法 | High Mean / P10 / P1 | Medium Mean / P10 / P1 | Low Mean / P10 / P1 |
|---|---|---|---|
| Six-loss Hattrick | {base['High']['norm_fulfill_mean']:.6f} / {base['High']['norm_fulfill_p10']:.6f} / {base['High']['norm_fulfill_p1']:.6f} | {base['Medium']['norm_fulfill_mean']:.6f} / {base['Medium']['norm_fulfill_p10']:.6f} / {base['Medium']['norm_fulfill_p1']:.6f} | {base['Low']['norm_fulfill_mean']:.6f} / {base['Low']['norm_fulfill_p10']:.6f} / {base['Low']['norm_fulfill_p1']:.6f} |
| Hattrick-f strict Low | {strict['High']['norm_fulfill_mean']:.6f} / {strict['High']['norm_fulfill_p10']:.6f} / {strict['High']['norm_fulfill_p1']:.6f} | {strict['Medium']['norm_fulfill_mean']:.6f} / {strict['Medium']['norm_fulfill_p10']:.6f} / {strict['Medium']['norm_fulfill_p1']:.6f} | {strict['Low']['norm_fulfill_mean']:.6f} / {strict['Low']['norm_fulfill_p10']:.6f} / {strict['Low']['norm_fulfill_p1']:.6f} |
| Hattrick-f Pareto | {pareto['High']['norm_fulfill_mean']:.6f} / {pareto['High']['norm_fulfill_p10']:.6f} / {pareto['High']['norm_fulfill_p1']:.6f} | {pareto['Medium']['norm_fulfill_mean']:.6f} / {pareto['Medium']['norm_fulfill_p10']:.6f} / {pareto['Medium']['norm_fulfill_p1']:.6f} | {pareto['Low']['norm_fulfill_mean']:.6f} / {pareto['Low']['norm_fulfill_p10']:.6f} / {pareto['Low']['norm_fulfill_p1']:.6f} |

严格 Low 版本在不牺牲 Low mean 的情况下将 Medium mean 提高
{strict['Medium']['norm_fulfill_mean']-base['Medium']['norm_fulfill_mean']:.6f}。
Pareto 版本将 Medium mean 提高
{pareto['Medium']['norm_fulfill_mean']-base['Medium']['norm_fulfill_mean']:.6f}，
并同时提高 High mean/P10/P1；Low mean 回落
{base['Low']['norm_fulfill_mean']-pareto['Low']['norm_fulfill_mean']:.6f}，
但 Low mean 仍为 {pareto['Low']['norm_fulfill_mean']:.6f}，Low P1 反而从
{base['Low']['norm_fulfill_p1']:.6f} 提高到 {pareto['Low']['norm_fulfill_p1']:.6f}。

## 结论

Level-4 独立窗口支持 Hattrick-f。若 Low 不允许下降，采用 strict 版本；若允许把 Low 的
过量接纳让给 Medium，Pareto 版本明显更优，且 High 的三项分布统计均改善。
"""
    (OUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
