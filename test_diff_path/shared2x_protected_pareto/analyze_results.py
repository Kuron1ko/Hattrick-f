from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
ARTIFACTS = THIS_DIR / "artifacts"
FINAL_RUN = ARTIFACTS / "final_candidate_joint_r1p150"
REPORT_DIR = ARTIFACTS / "final_report"
PUBLIC_COMPARISON = (
    THIS_DIR.parent
    / "shared2x_full_objectives"
    / "artifacts"
    / "final_report"
    / "comparison.csv"
)
CLASSES = ("High", "Medium", "Low")
METHODS = ("baseline", "protected_pareto")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_820


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_vector(indexed: dict, snapshots: list[int], class_name: str, method: str, metric: str):
    return np.asarray(
        [float(indexed[(snapshot, class_name, method)][metric]) for snapshot in snapshots],
        dtype=np.float64,
    )


def statistic(values: np.ndarray, name: str) -> float:
    if name == "mean":
        return float(values.mean())
    if name == "p1":
        return float(np.percentile(values, 1))
    if name == "p10":
        return float(np.percentile(values, 10))
    raise ValueError(name)


def paired_bootstrap(before: np.ndarray, after: np.ndarray, name: str, rng) -> dict:
    n = before.size
    replicates = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, n, n)
        replicates[draw] = statistic(after[indices], name) - statistic(before[indices], name)
    point = statistic(after, name) - statistic(before, name)
    return {
        "statistic": name,
        "point_gap": point,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_95_ci": [
            float(np.percentile(replicates, 2.5)),
            float(np.percentile(replicates, 97.5)),
        ],
    }


def main() -> None:
    rows = read_csv(FINAL_RUN / "metrics.csv")
    indexed = {
        (int(row["snapshot"]), row["class"], row["method"]): row for row in rows
    }
    snapshots = sorted({int(row["snapshot"]) for row in rows})
    expected = {
        (snapshot, class_name, method)
        for snapshot in range(400, 500)
        for class_name in CLASSES
        for method in METHODS
    }
    if set(indexed) != expected or snapshots != list(range(400, 500)):
        raise RuntimeError("final result key set is incomplete or duplicated")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = {}
    comparison_rows = []
    vectors = {}
    for class_name in CLASSES:
        before = metric_vector(indexed, snapshots, class_name, "baseline", "norm_fulfill")
        after = metric_vector(indexed, snapshots, class_name, "protected_pareto", "norm_fulfill")
        vectors[class_name] = (before, after)
        bootstrap[class_name] = {
            name: paired_bootstrap(before, after, name, rng)
            for name in ("mean", "p1", "p10")
        }
        for method, values in (("restored_Fh_Fhm", before), ("protected_pareto", after)):
            comparison_rows.append(
                {
                    "method": method,
                    "class": class_name,
                    "mean": float(values.mean()),
                    "p1": float(np.percentile(values, 1)),
                    "p10": float(np.percentile(values, 10)),
                }
            )

    public = read_csv(PUBLIC_COMPARISON)
    dote = next(row for row in public if row["method"] == "DOTE-MC sensitivity")
    comparison_rows.extend(
        [
            {
                "method": "DOTE_MC_sensitivity",
                "class": "High",
                "mean": float(dote["high_mean"]),
                "p1": dote["high_p1"],
                "p10": dote["high_p10"],
            },
            {
                "method": "DOTE_MC_sensitivity",
                "class": "Medium",
                "mean": float(dote["medium_mean"]),
                "p1": float(dote["medium_p1"]),
                "p10": float(dote["medium_p10"]),
            },
            {
                "method": "DOTE_MC_sensitivity",
                "class": "Low",
                "mean": float(dote["low_mean"]),
                "p1": "",
                "p10": "",
            },
        ]
    )

    high_after = vectors["High"][1]
    low_gap_ci = bootstrap["Low"]["mean"]["bootstrap_95_ci"]
    medium_gap_ci = bootstrap["Medium"]["mean"]["bootstrap_95_ci"]
    low_p10_ci = bootstrap["Low"]["p10"]["bootstrap_95_ci"]
    candidate_low_rows = [
        indexed[(snapshot, "Low", "protected_pareto")] for snapshot in snapshots
    ]
    candidate_mlu_mean = float(
        np.mean([float(row["admitted_capacity_ratio"]) for row in candidate_low_rows])
    )
    candidate_mlu_max = float(
        max(float(row["admitted_capacity_ratio"]) for row in candidate_low_rows)
    )
    solver_diagnostics = json.loads(
        (FINAL_RUN / "solver_diagnostics.json").read_text(encoding="utf-8")
    )
    solver_times = np.asarray(
        [float(row["runtime_seconds"]) for row in solver_diagnostics], dtype=np.float64
    )
    input_contract = json.loads(
        (REPORT_DIR / "input_contract_audit.json").read_text(encoding="utf-8")
    )
    regression_rows = read_csv(ARTIFACTS / "current_source_regression_400_402" / "metrics.csv")
    regression_indexed = {
        (int(row["snapshot"]), row["class"], row["method"]): row
        for row in regression_rows
    }
    regression_fields = ("admitted_traffic", "norm_fulfill", "admitted_capacity_ratio")
    source_regression_max_delta = max(
        abs(float(indexed[key][field]) - float(regression_indexed[key][field]))
        for key in regression_indexed
        for field in regression_fields
    )
    low_inversion = vectors["Low"][1] - vectors["Medium"][1]
    decision = {
        "high_target_pass": bool(high_after.mean() >= 0.995),
        "high_exactly_preserved": bool(np.array_equal(vectors["High"][0], high_after)),
        "low_mean_point_non_decline": bool(bootstrap["Low"]["mean"]["point_gap"] >= 0.0),
        "low_mean_has_no_significant_decline": bool(low_gap_ci[1] >= 0.0),
        "low_mean_noninferiority_minus_0p01_pass": bool(low_gap_ci[0] >= -0.01),
        "low_p10_significantly_declines": bool(low_p10_ci[1] < 0.0),
        "medium_mean_significantly_improves": bool(medium_gap_ci[0] > 0.0),
        "strict_user_goal_pass": False,
        "classification": "partial improvement / NO-GO under all-Low-metrics interpretation",
    }
    audit = {
        "evaluation_rows": len(rows),
        "unique_keys": len(indexed),
        "snapshots": [min(snapshots), max(snapshots)],
        "max_capacity_ratio": max(float(row["admitted_capacity_ratio"]) for row in rows),
        "max_disabled_flow": max(float(row["disabled_flow"]) for row in rows),
        "candidate_common_mlu_mean": candidate_mlu_mean,
        "candidate_common_mlu_max": candidate_mlu_max,
        "solver_runtime_seconds_mean": float(solver_times.mean()),
        "solver_runtime_seconds_p95": float(np.percentile(solver_times, 95)),
        "full_pipeline_seconds_per_snapshot": float(
            json.loads((FINAL_RUN / "complete.json").read_text(encoding="utf-8"))[
                "runtime_seconds"
            ]
            / len(snapshots)
        ),
        "candidate_inversion_positive_gap_mean": float(np.maximum(low_inversion, 0.0).mean()),
        "candidate_inversion_fraction": float(np.mean(low_inversion > 0.0)),
        "input_contract_cpu_max_policy_delta": float(input_contract["max_policy_delta"]),
        "input_contract_pass": bool(input_contract["passes"]),
        "current_source_regression_snapshots": [400, 401, 402],
        "current_source_regression_max_metric_delta": source_regression_max_delta,
    }
    result = {
        "scenario": "GEANT shared paths, 2x load, K=8, test 400-499",
        "candidate": "restored Fh/Fhm + aggregate protected Pareto repair, Low floor 1.15, predicted dominance fallback",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap": bootstrap,
        "decision": decision,
        "audit": audit,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "statistical_analysis.json", result)
    write_csv(REPORT_DIR / "comparison.csv", comparison_rows)

    def cell(method: str, class_name: str) -> dict:
        return next(
            row
            for row in comparison_rows
            if row["method"] == method and row["class"] == class_name
        )

    base_h, cand_h, dote_h = (
        cell("restored_Fh_Fhm", "High"),
        cell("protected_pareto", "High"),
        cell("DOTE_MC_sensitivity", "High"),
    )
    base_m, cand_m, dote_m = (
        cell("restored_Fh_Fhm", "Medium"),
        cell("protected_pareto", "Medium"),
        cell("DOTE_MC_sensitivity", "Medium"),
    )
    base_l, cand_l, dote_l = (
        cell("restored_Fh_Fhm", "Low"),
        cell("protected_pareto", "Low"),
        cell("DOTE_MC_sensitivity", "Low"),
    )

    report = f"""# Shared-2x Protected Pareto Repair 实验报告

## 结论

最终状态：**局部改善；按“Low 所有统计量均不得显著下降”的严格解释为 NO-GO。**

冻结候选在 400–499 上把 High 完全保持在 `{cand_h['mean']:.6f}`，超过 `0.995`；Low Mean 从 `{base_l['mean']:.6f}` 增至 `{cand_l['mean']:.6f}`；Medium Mean/P10 分别增加 `{bootstrap['Medium']['mean']['point_gap']:+.6f}` / `{bootstrap['Medium']['p10']['point_gap']:+.6f}`。但是 Medium 改善的 bootstrap 区间包含 0，且 Low P10 下降 `{bootstrap['Low']['p10']['point_gap']:+.6f}`，95% CI 为 `[{low_p10_ci[0]:+.6f}, {low_p10_ci[1]:+.6f}]`，属于显著尾部退化。因此不能宣称已经完整解决用户目标。

如果 Low 只按截图中的 Mean 口径解释，则候选满足“无显著下降”：Low Mean 点估计为正，95% CI `[{low_gap_ci[0]:+.6f}, {low_gap_ci[1]:+.6f}]` 并未落在 0 以下。但该区间下界低于预先写下的 `-0.01` 非劣界，证据仍不够强。

## 方法

方法名为 **Protected Pareto Repair**：

1. 完全保留 restored-$F_h/F_{{hm}}$ 模型的 High 路由；
2. 只使用 ESM 预测 TM、拓扑、容量、候选路径和 mask，在 High 后的预测剩余容量上联合求解 Medium/Low；
3. Low 的预测总接纳下界设为 baseline 的 `1.15` 倍（受可行上限裁剪），随后按 Medium → Low 的顺序做线性优化；
4. 若新策略在预测空间不能同时弱支配 baseline 的 Medium 和 Low，则整片回退 baseline。

没有新增神经网络、训练 epoch 或 loss 权重。方法针对 2x 饱和场景：把学习器已经给出的 High 安全策略当作硬前缀，只修复低优先级残余容量上的 Pareto 低效点。

## 小范围筛选

| 窗口 | High Mean | Medium Mean / P1 / P10 Δ | Low Mean Δ | 结果 |
|---|---:|---:|---:|---|
| 350–365 | 0.999926 | +0.001034 / +0.000606 / +0.004542 | +0.002121 | 通过 |
| 366–399 holdout | 0.999645 | +0.008225 / -0.000087 / +0.007779 | +0.001506 | 通过 Mean 门槛 |

固定 Low 路由的边预留、1.10 aggregate floor、预测 dominance guard、策略空间 trust step 和 per-OD Low floor 均已做反例筛选；它们分别因 Low 明显下降、validation→test 漂移、区分力不足、sequential admission 非线性或小样本 Low 退化而淘汰。

## 400–499 大样本结果

| 方法 | High Mean | Medium Mean / P1 / P10 | Low Mean | 公共 MLU |
|---|---:|---:|---:|---:|
| Restored $F_h/F_{{hm}}$ | {base_h['mean']:.6f} | {base_m['mean']:.6f} / {base_m['p1']:.6f} / {base_m['p10']:.6f} | {base_l['mean']:.6f} | 1.000000 |
| Protected Pareto | {cand_h['mean']:.6f} | {cand_m['mean']:.6f} / {cand_m['p1']:.6f} / {cand_m['p10']:.6f} | {cand_l['mean']:.6f} | {candidate_mlu_mean:.6f} |
| DOTE-MC sensitivity | {dote_h['mean']:.6f} | {dote_m['mean']:.6f} / {dote_m['p1']:.6f} / {dote_m['p10']:.6f} | {dote_l['mean']:.6f} | 1.000000 |

候选仍未追平 DOTE-MC：Medium Mean/P10 仍低约 `{cand_m['mean'] - dote_m['mean']:+.6f}` / `{cand_m['p10'] - dote_m['p10']:+.6f}`。

## 逐片 paired bootstrap（10,000 次）

| 类别/统计量 | 点差 | 95% CI | 判断 |
|---|---:|---:|---|
| High Mean | {bootstrap['High']['mean']['point_gap']:+.6f} | [0, 0] | 硬保持 |
| Medium Mean | {bootstrap['Medium']['mean']['point_gap']:+.6f} | [{medium_gap_ci[0]:+.6f}, {medium_gap_ci[1]:+.6f}] | 未显著 |
| Medium P1 | {bootstrap['Medium']['p1']['point_gap']:+.6f} | [{bootstrap['Medium']['p1']['bootstrap_95_ci'][0]:+.6f}, {bootstrap['Medium']['p1']['bootstrap_95_ci'][1]:+.6f}] | 混合 |
| Medium P10 | {bootstrap['Medium']['p10']['point_gap']:+.6f} | [{bootstrap['Medium']['p10']['bootstrap_95_ci'][0]:+.6f}, {bootstrap['Medium']['p10']['bootstrap_95_ci'][1]:+.6f}] | 未显著 |
| Low Mean | {bootstrap['Low']['mean']['point_gap']:+.6f} | [{low_gap_ci[0]:+.6f}, {low_gap_ci[1]:+.6f}] | 无显著下降，但未证非劣 |
| Low P10 | {bootstrap['Low']['p10']['point_gap']:+.6f} | [{low_p10_ci[0]:+.6f}, {low_p10_ci[1]:+.6f}] | 显著下降 |

## 正确性与成本

- 600/600 评估行、400–499 全部 100 片，键唯一且完整。
- High admitted traffic 逐片最大差为 0；最大累计容量比 `{audit['max_capacity_ratio']:.10f}`；disabled flow 为 0。
- 公共 MLU 使用真实 sequential admission 后的累计链路负载/容量；候选均值 `{candidate_mlu_mean:.9f}`，最大 `{candidate_mlu_max:.9f}`。
- 联合 LP 平均 `{audit['solver_runtime_seconds_mean'] * 1000:.1f}` ms/片，P95 `{audit['solver_runtime_seconds_p95'] * 1000:.1f}` ms/片；当前评估流水线总耗时约 `{audit['full_pipeline_seconds_per_snapshot'] * 1000:.1f}` ms/片，尚未做生产优化。
- 400–499 在 1.10 版本后已经打开；1.15 是明确披露的探索性大样本确认，不是新的独立盲测。
- 1.15 score-bearing 源码哈希记录在 `FINAL_CANDIDATE_PROTOCOL.json`。其后仅追加了 per-OD 失败分支，因此当前工作文件哈希与 score-bearing 哈希不同；正式结论以协议和保存的逐片结果为准。
- CPU 输入替换审计把三类 actual TM 全部替换为无关常数，3 个样本的 emitted policy 最大差为 `{audit['input_contract_cpu_max_policy_delta']:.1f}`；当前代码在 400–402 上重放与 score-bearing 指标最大差为 `{audit['current_source_regression_max_metric_delta']:.1f}`。

## 复现

```powershell
$py = 'D:\\kuroresearch\\.venv-hattrick\\Scripts\\python.exe'
Set-Location 'D:\\kuroresearch\\Hattrick-main'
& $py -B -m unittest -v test_diff_path\\shared2x_protected_pareto\\test_repair.py
& $py -B test_diff_path\\shared2x_protected_pareto\\run_experiment.py --start 350 --end 366 --mode joint --reserve-factor 1.15 --predicted-dominance-guard --label screen_joint_r1p150_repro
& $py -B test_diff_path\\shared2x_protected_pareto\\run_experiment.py --start 366 --end 400 --mode joint --reserve-factor 1.15 --predicted-dominance-guard --label validation_joint_r1p150_repro
& $py -B test_diff_path\\shared2x_protected_pareto\\run_experiment.py --start 400 --end 500 --mode joint --reserve-factor 1.15 --predicted-dominance-guard --label final_joint_r1p150_repro
& $py -B test_diff_path\\shared2x_protected_pareto\\analyze_results.py
```
"""
    (REPORT_DIR / "REPORT_CN.md").write_text(report, encoding="utf-8")
    manifest = {
        "report": str((REPORT_DIR / "REPORT_CN.md").resolve()),
        "report_sha256": sha256(REPORT_DIR / "REPORT_CN.md"),
        "statistical_analysis_sha256": sha256(REPORT_DIR / "statistical_analysis.json"),
        "comparison_sha256": sha256(REPORT_DIR / "comparison.csv"),
        "score_bearing_protocol_sha256": sha256(THIS_DIR / "FINAL_CANDIDATE_PROTOCOL.json"),
        "final_metrics_sha256": sha256(FINAL_RUN / "metrics.csv"),
        "final_solver_diagnostics_sha256": sha256(FINAL_RUN / "solver_diagnostics.json"),
        "input_contract_audit_sha256": sha256(REPORT_DIR / "input_contract_audit.json"),
        "current_source_regression_metrics_sha256": sha256(
            ARTIFACTS / "current_source_regression_400_402" / "metrics.csv"
        ),
        "current_source_sha256": {
            "run_experiment.py": sha256(THIS_DIR / "run_experiment.py"),
            "repair.py": sha256(THIS_DIR / "repair.py"),
            "analyze_results.py": sha256(Path(__file__).resolve()),
        },
    }
    write_json(REPORT_DIR / "manifest.json", manifest)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
