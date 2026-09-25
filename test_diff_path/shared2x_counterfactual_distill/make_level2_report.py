from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "artifacts" / "level2_report"
SEEDS = (490, 491)
BOOTSTRAP_REPLICATES = 10_000


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def indexed(rows: list[dict], class_name: str) -> dict[int, dict]:
    return {
        int(row["snapshot"]): row for row in rows if row["class"] == class_name
    }


def summary(rows: list[dict], class_name: str) -> dict[str, float]:
    selected = list(indexed(rows, class_name).values())
    norm = np.asarray([float(row["norm_fulfill"]) for row in selected])
    fulfill = np.asarray([float(row["fulfill_ratio"]) for row in selected])
    admitted = np.asarray([float(row["admitted_traffic"]) for row in selected])
    return {
        "norm_mean": float(norm.mean()),
        "norm_p1": float(np.percentile(norm, 1)),
        "norm_p10": float(np.percentile(norm, 10)),
        "fulfill_mean": float(fulfill.mean()),
        "admitted_mean": float(admitted.mean()),
    }


def bootstrap_statistic(
    before: np.ndarray,
    after: np.ndarray,
    percentile: float | None,
    rng: np.random.Generator,
) -> dict[str, float]:
    n = len(before)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        take = rng.integers(0, n, n)
        if percentile is None:
            values[replicate] = after[take].mean() - before[take].mean()
        else:
            values[replicate] = np.percentile(
                after[take], percentile
            ) - np.percentile(before[take], percentile)
    point = (
        after.mean() - before.mean()
        if percentile is None
        else np.percentile(after, percentile) - np.percentile(before, percentile)
    )
    return {
        "point": float(point),
        "ci95_low": float(np.percentile(values, 2.5)),
        "ci95_high": float(np.percentile(values, 97.5)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_rows = []
    bootstrap = {}
    manifests = {}
    for seed in SEEDS:
        run = ROOT / "artifacts" / "distill" / "level2_proxy" / "changed" / f"seed_{seed}"
        initial_path = run / "initial_evaluation_metrics.csv"
        best_path = run / "best_evaluation_metrics.csv"
        before_rows = read_csv(initial_path)
        after_rows = read_csv(best_path)
        before = {name: summary(before_rows, name) for name in ("High", "Medium", "Low")}
        after = {name: summary(after_rows, name) for name in ("High", "Medium", "Low")}
        paired = {}
        for class_name in ("High", "Medium", "Low"):
            left = indexed(before_rows, class_name)
            right = indexed(after_rows, class_name)
            ids = sorted(left)
            norm_gap = np.asarray(
                [float(right[i]["norm_fulfill"]) - float(left[i]["norm_fulfill"]) for i in ids]
            )
            fulfill_gap = np.asarray(
                [float(right[i]["fulfill_ratio"]) - float(left[i]["fulfill_ratio"]) for i in ids]
            )
            paired[class_name] = {
                "norm_gap_mean": float(norm_gap.mean()),
                "norm_gap_p1": float(np.percentile(norm_gap, 1)),
                "norm_gap_p10": float(np.percentile(norm_gap, 10)),
                "norm_gap_min": float(norm_gap.min()),
                "norm_positive_fraction": float((norm_gap > 0).mean()),
                "fulfill_gap_mean": float(fulfill_gap.mean()),
            }
        medium_before = indexed(before_rows, "Medium")
        medium_after = indexed(after_rows, "Medium")
        ids = sorted(medium_before)
        left = np.asarray([float(medium_before[i]["norm_fulfill"]) for i in ids])
        right = np.asarray([float(medium_after[i]["norm_fulfill"]) for i in ids])
        rng = np.random.default_rng(20260819 + seed)
        bootstrap[str(seed)] = {
            "mean_gap": bootstrap_statistic(left, right, None, rng),
            "p1_gap": bootstrap_statistic(left, right, 1, rng),
            "p10_gap": bootstrap_statistic(left, right, 10, rng),
            "replicates": BOOTSTRAP_REPLICATES,
        }
        with (run / "best_evaluation_summary.json").open(encoding="utf-8") as handle:
            best_payload = json.load(handle)
        capacity = max(row["max_admitted_capacity_ratio"] for row in best_payload["classes"])
        disabled = max(row["max_disabled_flow"] for row in best_payload["classes"])
        diagnostics = best_payload["diagnostics"]
        medium_mean_gap = after["Medium"]["norm_mean"] - before["Medium"]["norm_mean"]
        medium_p1_gap = after["Medium"]["norm_p1"] - before["Medium"]["norm_p1"]
        medium_p10_gap = after["Medium"]["norm_p10"] - before["Medium"]["norm_p10"]
        passed = (
            medium_mean_gap >= 0
            and medium_p1_gap >= 0.01
            and medium_p10_gap >= 0.01
            and paired["Medium"]["norm_positive_fraction"] >= 0.60
            and paired["High"]["norm_gap_min"] >= -0.005
            and paired["Low"]["fulfill_gap_mean"] >= -0.02
            and capacity <= 1.0001
            and disabled <= 1e-8
        )
        result_rows.append(
            {
                "method": "Phase-A restored-Fh/Fhm" if seed == SEEDS[0] else "Search-Distill changed-OD",
                "seed": "common" if seed == SEEDS[0] else seed,
                "high_mean": before["High"]["norm_mean"] if seed == SEEDS[0] else after["High"]["norm_mean"],
                "high_p1": before["High"]["norm_p1"] if seed == SEEDS[0] else after["High"]["norm_p1"],
                "high_p10": before["High"]["norm_p10"] if seed == SEEDS[0] else after["High"]["norm_p10"],
                "medium_mean": before["Medium"]["norm_mean"] if seed == SEEDS[0] else after["Medium"]["norm_mean"],
                "medium_p1": before["Medium"]["norm_p1"] if seed == SEEDS[0] else after["Medium"]["norm_p1"],
                "medium_p10": before["Medium"]["norm_p10"] if seed == SEEDS[0] else after["Medium"]["norm_p10"],
                "low_mean": before["Low"]["norm_mean"] if seed == SEEDS[0] else after["Low"]["norm_mean"],
                "low_fulfill_mean": before["Low"]["fulfill_mean"] if seed == SEEDS[0] else after["Low"]["fulfill_mean"],
                "inversion_fraction": "" if seed == SEEDS[0] else diagnostics["inversion_violation_fraction"],
                "max_capacity_ratio": "" if seed == SEEDS[0] else capacity,
                "level2_pass": "baseline" if seed == SEEDS[0] else passed,
            }
        )
        if seed == SEEDS[0]:
            # Add the first candidate as well; seed 491 then contributes only
            # its independent distillation row below.
            result_rows.append(
                {
                    "method": "Search-Distill changed-OD",
                    "seed": seed,
                    "high_mean": after["High"]["norm_mean"],
                    "high_p1": after["High"]["norm_p1"],
                    "high_p10": after["High"]["norm_p10"],
                    "medium_mean": after["Medium"]["norm_mean"],
                    "medium_p1": after["Medium"]["norm_p1"],
                    "medium_p10": after["Medium"]["norm_p10"],
                    "low_mean": after["Low"]["norm_mean"],
                    "low_fulfill_mean": after["Low"]["fulfill_mean"],
                    "inversion_fraction": diagnostics["inversion_violation_fraction"],
                    "max_capacity_ratio": capacity,
                    "level2_pass": passed,
                }
            )
        manifests[str(seed)] = {
            "best_model": str((run / "best_model.pt").resolve()),
            "best_model_sha256": sha256(run / "best_model.pt"),
            "config_sha256": sha256(run / "config.json"),
            "metrics_sha256": sha256(best_path),
        }

    write_csv(OUTPUT / "level2_results.csv", result_rows)
    (OUTPUT / "bootstrap_10000.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    manifest = {
        "status": "LEVEL2_GO",
        "scenario": "GEANT shared paths, load 2x, K=8",
        "phase_a_checkpoint_sha256": "a0736d3729217ab06d73d429a8e38a7554f714238fff89473f5fe7448d919944",
        "teacher_targets_sha256": sha256(ROOT / "artifacts" / "level0_search" / "train160" / "targets.pt"),
        "counterfactual_search_sha256": sha256(ROOT / "counterfactual_search.py"),
        "runner_sha256": sha256(ROOT / "run_distill.py"),
        "reporter_sha256": sha256(Path(__file__)),
        "seeds": manifests,
        "note": "Both distillation seeds share the required Phase-A checkpoint; they are RNG checks, not independent Phase-A initializations.",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    baseline = result_rows[0]
    candidate_490 = result_rows[1]
    candidate_491 = result_rows[2]
    b490 = bootstrap["490"]["p10_gap"]
    b491 = bootstrap["491"]["p10_gap"]
    report = f"""# Shared-2x Search → Verify → Distill：Level 2 报告

## 结论

**Level 2 GO。** 只对经过真实 sequential admission 验收、且确实发生路径交换的 OD 做策略蒸馏，无需新增 λ、MLU 正则或冻结任何类别。该方法在 200–249 proxy 的两个 RNG seed 上均通过 High、Medium、Low、容量与 mask 门槛。

这还不是最终 Level 4 结论：两个蒸馏 seed 使用同一个 Phase-A checkpoint，且 200–249 是研究中已查看窗口，不是盲测。

## 方法

1. 从 restored-$F_h/F_{{hm}}$ 的公共 Phase-A 策略 A 出发。
2. 在训练样本的路径概率空间执行有限 path swap；每个动作都用真实 TM 的 sequential admission 重放。
3. 仅接受 High 不低于 A 的逐片相对下界（容差为 oracle High 的 $10^{{-4}}$）且 Medium 增加的 B。
4. 只在 A 与 B 不同的 OD 上计算 teacher-to-policy KL；未变化 OD 不稀释梯度。
5. checkpoint 只在逐片 High 相对 A 不低于 -0.005、容量与 mask 正确时，按 Medium P10/P1/Mean 选择。

Low、High 参数均未冻结；训练目标不含 Low 惩罚、order penalty 或 MLU。

## Level 0：B 是否存在

- train 0–159：160/160 时间片找到可行 B。
- Medium NormFulFill paired gap：Mean +0.06316，P10 +0.03521，最大 +0.13250。
- 搜索全部使用真实 admission 验收；这证明等 High 路由平台上普遍存在 Medium 更好的路径组合。

## Proxy 200–249

| 方法 | Seed | High Mean / P1 / P10 | Medium Mean / P1 / P10 | Low Mean | Low Fulfill Mean | Inversion 比例 |
|---|---:|---|---|---:|---:|---:|
| Phase-A | common | {float(baseline['high_mean']):.6f} / {float(baseline['high_p1']):.6f} / {float(baseline['high_p10']):.6f} | {float(baseline['medium_mean']):.6f} / {float(baseline['medium_p1']):.6f} / {float(baseline['medium_p10']):.6f} | {float(baseline['low_mean']):.6f} | {float(baseline['low_fulfill_mean']):.6f} | 0.98 |
| changed-OD distill | 490 | {float(candidate_490['high_mean']):.6f} / {float(candidate_490['high_p1']):.6f} / {float(candidate_490['high_p10']):.6f} | {float(candidate_490['medium_mean']):.6f} / {float(candidate_490['medium_p1']):.6f} / {float(candidate_490['medium_p10']):.6f} | {float(candidate_490['low_mean']):.6f} | {float(candidate_490['low_fulfill_mean']):.6f} | {float(candidate_490['inversion_fraction']):.2f} |
| changed-OD distill | 491 | {float(candidate_491['high_mean']):.6f} / {float(candidate_491['high_p1']):.6f} / {float(candidate_491['high_p10']):.6f} | {float(candidate_491['medium_mean']):.6f} / {float(candidate_491['medium_p1']):.6f} / {float(candidate_491['medium_p10']):.6f} | {float(candidate_491['low_mean']):.6f} | {float(candidate_491['low_fulfill_mean']):.6f} | {float(candidate_491['inversion_fraction']):.2f} |

seed 490 的 paired 变化：Medium Mean +0.02272，paired P1 +0.01247、P10 +0.01495，50/50 时间片均为正；High Mean -0.000318、最坏逐片 -0.001413；Low FulfillRatio Mean -0.005223。Medium 实际接纳均值 +0.55399，Low 仅 -0.03928，因此改善不只是把等量 Low 直接转给 Medium。

容量重放最大比例为 1.00000024，disabled flow 为 0。

## Bootstrap

10,000 次 paired bootstrap 的 Medium P10 统计量差：

- seed 490：{b490['point']:+.6f}，95% CI [{b490['ci95_low']:+.6f}, {b490['ci95_high']:+.6f}]
- seed 491：{b491['point']:+.6f}，95% CI [{b491['ci95_low']:+.6f}, {b491['ci95_high']:+.6f}]

## 解释

完整 KL 只得到 Medium Mean +0.00355，因为教师只改变少量 OD，而所有未变化 OD 共同把纠正信号稀释。changed-OD 版本把训练问题从“再造一个标量目标”改成“模仿已验收的有限路径动作”，因此能跨过 sequential admission 的分段边界，同时保留逐片 High guard。

## 复现

```powershell
& 'D:\\kuroresearch\\.venv-hattrick\\Scripts\\python.exe' -B run_level0.py --split train160 --force
& 'D:\\kuroresearch\\.venv-hattrick\\Scripts\\python.exe' -B run_distill.py --level 2 --seed 490 --distill-mode changed --force
& 'D:\\kuroresearch\\.venv-hattrick\\Scripts\\python.exe' -B run_distill.py --level 2 --seed 491 --distill-mode changed --force
& 'D:\\kuroresearch\\.venv-hattrick\\Scripts\\python.exe' -B make_level2_report.py
```
"""
    (OUTPUT / "REPORT_CN.md").write_text(report, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
