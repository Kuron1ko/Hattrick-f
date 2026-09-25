from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import analyze_results as analysis
import run_experiment as runner


REPORT_DIR = runner.OUTPUT_ROOT / "final_report"
BEST_PENALTY = "tail_directional_squared"
BEST_MULTIPLIER = 0.25
BEST_SEEDS = (490, 491)


def json_load(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def composite_hash(paths: list[Path]) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(set(path.resolve() for path in paths), key=lambda item: str(item)):
        relative = path.relative_to(runner.ROOT)
        file_hash = runner.sha256(path)
        digest.update(str(relative).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        total_bytes += path.stat().st_size
    return digest.hexdigest(), len(set(paths)), total_bytes


def build_data_fingerprint() -> dict:
    result_dir = runner.ROOT / "results" / runner.TOPOLOGY / f"{runner.K}sp" / "0"
    result_names = (
        "filenames.txt",
        "gt_optimal_values_mf.txt",
        "gt_optimal_values_mf_mf.txt",
        "gt_optimal_values_mf_mf_mf.txt",
        "gt_optimal_values_mlu.txt",
        "gt_optimal_values_mlu_mlu.txt",
        "gt_optimal_values_mlu_mlu_mlu.txt",
    )
    result_paths = [result_dir / name for name in result_names]
    static_paths = [
        runner.ROOT
        / "topologies"
        / "paths_dict"
        / f"{runner.TOPOLOGY}_{runner.K}_paths_dict_cluster_0.pkl",
        runner.ROOT
        / "topologies"
        / "paths"
        / f"{runner.TOPOLOGY}_{runner.K}_paths_cluster_0.pkl",
        runner.ROOT
        / "topologies"
        / "padded_edge_ids_per_path"
        / f"{runner.TOPOLOGY}_{runner.K}_paths_cluster_0_padded_edge_ids_per_path.pkl",
        runner.ROOT
        / "topologies"
        / "padded_edge_ids_per_path"
        / f"{runner.TOPOLOGY}_{runner.K}_paths_cluster_0_edge_ids_dict.pkl",
    ]
    snapshot_paths: list[Path] = []
    with (result_dir / "filenames.txt").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))[:500]
    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 dataset rows, found {len(rows)}")
    for topology_file, pairs_file, tm_file in rows:
        snapshot_paths.extend(
            (
                runner.ROOT / "topologies" / runner.TOPOLOGY / topology_file.strip(),
                runner.ROOT / "pairs" / runner.TOPOLOGY / pairs_file.strip(),
            )
        )
        for priority in (1, 2, 3):
            snapshot_paths.append(
                runner.ROOT
                / "traffic_matrices"
                / f"{runner.TOPOLOGY}_{priority}"
                / tm_file.strip()
            )
            snapshot_paths.append(
                runner.ROOT
                / "traffic_matrices"
                / f"{runner.TOPOLOGY}_{priority}_esm"
                / tm_file.strip()
            )
    groups = {
        "oracle_and_index": result_paths,
        "path_static": static_paths,
        "snapshot_inputs_0_499": snapshot_paths,
    }
    group_rows = {}
    all_paths: list[Path] = []
    for name, paths in groups.items():
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing data inputs in {name}: {missing[:3]}")
        digest, count, total_bytes = composite_hash(paths)
        group_rows[name] = {
            "sha256": digest,
            "file_count": count,
            "total_bytes": total_bytes,
        }
        all_paths.extend(paths)
    digest, count, total_bytes = composite_hash(all_paths)
    fingerprint = {
        "topology": runner.TOPOLOGY,
        "index_range": [0, 500],
        "sha256": digest,
        "file_count": count,
        "total_bytes": total_bytes,
        "groups": group_rows,
    }
    runner.write_json(REPORT_DIR / "dataset_fingerprint.json", fingerprint)
    return fingerprint


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        runner.THIS_DIR / "run_experiment.py",
        runner.THIS_DIR / "penalties.py",
        runner.THIS_DIR / "analyze_results.py",
        runner.THIS_DIR / "integration_checks.py",
        runner.THIS_DIR / "test_penalties.py",
        runner.THIS_DIR / "common_evaluator.py",
        runner.ROOT / "frameworks" / "hattrick_system.py",
        runner.ROOT / "utils" / "training_utils.py",
        runner.ROOT / "utils" / "robust_proj_utils.py",
        runner.ROOT / "utils" / "build_dataset_within_cluster.py",
    )
    return {
        str(path.relative_to(runner.ROOT)).replace("\\", "/"): runner.sha256(path)
        for path in paths
    }


def write_run_manifests(data: dict, sources: dict[str, str]) -> list[dict]:
    manifests = []
    for complete_path in sorted(runner.OUTPUT_ROOT.rglob("complete.json")):
        run_dir = complete_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        config = json_load(config_path)
        item = {
            "run": str(run_dir.relative_to(runner.OUTPUT_ROOT)).replace("\\", "/"),
            "config_sha256": runner.sha256(config_path),
            "complete_sha256": runner.sha256(complete_path),
            "best_checkpoint_sha256": runner.sha256(run_dir / "best_model.pt"),
            "final_checkpoint_sha256": runner.sha256(run_dir / "final_model.pt"),
            "train_history_sha256": runner.sha256(run_dir / "train_history.csv"),
            "dataset_sha256": data["sha256"],
            "data_index_ranges": {
                key: config[key] for key in ("train", "validation", "evaluation")
            },
            "source_sha256_at_training": config["source_sha256"],
            "current_delivery_source_sha256": sources,
        }
        runner.write_json(run_dir / "artifact_manifest.json", item)
        manifests.append(item)
    return manifests


def all_level2_rows() -> list[dict]:
    rows = []
    for penalty in runner.PENALTIES:
        for multiplier in runner.LAMBDA_MULTIPLIERS:
            item, _ = analysis.compare_seed(2, penalty, multiplier, 490)
            rows.append(item)
    second_seed = {
        "hinge": (0.25,),
        "directional_hinge": (0.5, 4.0),
        "tail_directional_squared": (0.25, 0.5),
    }
    for penalty, multipliers in second_seed.items():
        for multiplier in multipliers:
            item, _ = analysis.compare_seed(2, penalty, multiplier, 491)
            rows.append(item)
    return rows


def flatten(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "gates"}
        flat["failed_gates"] = ";".join(
            key for key, passed in row["gates"].items() if not passed
        )
        flat.update({f"gate_{key}": value for key, value in row["gates"].items()})
        output.append(flat)
    return output


def class_values(rows: list[dict], class_name: str, metric: str) -> np.ndarray:
    return np.asarray(
        [float(row[metric]) for row in rows if row["class"] == class_name],
        dtype=np.float64,
    )


def proxy_summary(method: str, penalty: str, multiplier: float) -> dict:
    rows: list[dict] = []
    for seed in BEST_SEEDS:
        rows.extend(analysis.load_best_rows(2, penalty, multiplier, seed))
    high = class_values(rows, "High", "norm_fulfill")
    medium = class_values(rows, "Medium", "norm_fulfill")
    low = class_values(rows, "Low", "norm_fulfill")
    final_mlu = class_values(rows, "Low", "admitted_capacity_ratio")
    return {
        "scenario": "shared Level-2 proxy 200-249 (two seeds pooled)",
        "method": method,
        "status": "matched baseline" if multiplier == 0 else "diagnostic only; NO-GO",
        "penalty": "none" if multiplier == 0 else penalty,
        "lambda_multiplier": multiplier,
        "actual_lambda": 0.0
        if multiplier == 0
        else float(json_load(runner.run_directory(2, penalty, multiplier, 490) / "config.json")["actual_lambda"]),
        "n_per_class": int(high.size),
        "high_norm_mean": float(high.mean()),
        "medium_norm_mean": float(medium.mean()),
        "medium_norm_p1": float(np.percentile(medium, 1)),
        "medium_norm_p10": float(np.percentile(medium, 10)),
        "low_norm_mean": float(low.mean()),
        "common_post_admission_mlu": float(final_mlu.mean()),
    }


def final_existing_table() -> list[dict]:
    common = json_load(
        runner.OUTPUT_ROOT / "common_final_existing_methods" / "common_summary.json"
    )
    indexed = {(row["method"], row["class"]): row for row in common}
    rows = []
    for method in ("Hattrick", "DOTE-MC sensitivity"):
        high = indexed[(method, "High")]
        medium = indexed[(method, "Medium")]
        low = indexed[(method, "Low")]
        rows.append(
            {
                "scenario": "shared 400-499 previously viewed confirmation",
                "method": method,
                "status": "existing checkpoint; common reevaluation",
                "high_norm_mean": high["norm_fulfill_mean"],
                "medium_norm_mean": medium["norm_fulfill_mean"],
                "medium_norm_p1": medium["norm_fulfill_p1"],
                "medium_norm_p10": medium["norm_fulfill_p10"],
                "low_norm_mean": low["norm_fulfill_mean"],
                "common_post_admission_mlu": low["common_post_admission_mlu_mean"],
            }
        )
    rows.insert(
        1,
        {
            "scenario": "shared 400-499 previously viewed confirmation",
            "method": "Hattrick+regularizer",
            "status": "N/A: no Level-2 candidate passed; final window not opened",
            "high_norm_mean": "",
            "medium_norm_mean": "",
            "medium_norm_p1": "",
            "medium_norm_p10": "",
            "low_norm_mean": "",
            "common_post_admission_mlu": "",
        },
    )
    return rows


def f4(value: float | str) -> str:
    return "N/A" if value == "" else f"{float(value):.4f}"


def runtime_for(penalty: str, multiplier: float, seed: int) -> float:
    complete = json_load(runner.run_directory(2, penalty, multiplier, seed) / "complete.json")
    return float(complete["runtime_seconds_this_invocation"])


def report_markdown(
    final_table: list[dict],
    proxy_table: list[dict],
    best_details: list[dict],
    bootstrap: dict,
    fingerprint: dict,
    manifests: list[dict],
) -> str:
    final_lines = []
    for row in final_table:
        medium = (
            "N/A"
            if row["medium_norm_mean"] == ""
            else f"{f4(row['medium_norm_mean'])} / {f4(row['medium_norm_p1'])} / {f4(row['medium_norm_p10'])}"
        )
        final_lines.append(
            f"| shared | {row['method']} | {f4(row['high_norm_mean'])} | {medium} | "
            f"{f4(row['low_norm_mean'])} | {f4(row['common_post_admission_mlu'])} | {row['status']} |\n"
        )
    proxy_lines = []
    for row in proxy_table:
        proxy_lines.append(
            f"| {row['method']} | {f4(row['high_norm_mean'])} | "
            f"{f4(row['medium_norm_mean'])} / {f4(row['medium_norm_p1'])} / {f4(row['medium_norm_p10'])} | "
            f"{f4(row['low_norm_mean'])} | {f4(row['common_post_admission_mlu'])} |\n"
        )
    seed_lines = []
    for row in best_details:
        seed_lines.append(
            f"| {row['seed']} | {row['best_epoch']} | {row['medium_mean_gap']:+.4f} | {row['medium_p1_gap']:+.4f} | "
            f"{row['medium_p10_gap']:+.4f} | {row['high_mean_gap']:+.4f} | "
            f"{row['low_fulfill_mean_gap']:+.4f} | {row['candidate_inversion_positive_gap_mean']:.4f} | "
            f"{row['inversion_severity_reduction_fraction']:.1%} | {row['candidate_inversion_violation_fraction']:.1%} | "
            f"{row['candidate_max_post_admission_mlu']:.7f} | {row['runtime_seconds'] / 60:.2f} |\n"
        )
    return f"""# Shared-2x 优先级排序正则实验报告

## 结论

**NO-GO。** 三种预注册函数均未在 Level 2 的两个 seed 上同时通过严格门槛，因此未执行 Level 3/4，也未用任何正则 checkpoint 查看 400–499。不能把本次结果描述为已经解决 priority inversion。

最佳安全趋势是 `tail_directional_squared`、multiplier=`0.25`、实际 λ=`0.2279225205`。它稳定提高 Medium 且没有牺牲 High/Low，但排序修复远低于门槛：seed 490/491 的 inversion severity 仅下降 15.9%/10.8%，违例时间片仍为 94%/88%。

## 已有最终窗口方法的公共重评估

400–499 是此前已查看的确认窗口。下表只重放已有 Hattrick 和 DOTE-MC sensitivity checkpoint；`Hattrick+regularizer` 因未晋级而保留为 N/A。MLU 统一为 Low admission 后的累计 admitted link load/capacity 均值。

| 场景 | 方法 | High Mean | Medium Mean / P1 / P10 | Low Mean | 公共 MLU | 状态 |
|---|---|---:|---:|---:|---:|---|
{''.join(final_lines)}
## Level 2 最佳安全趋势（proxy 200–249）

这是两个 seed、每 seed 50 个 proxy 时间片的池化描述，仅作诊断，不是最终测试结果。

| 方法 | High Mean | Medium Mean / P1 / P10 | Low Mean | 公共 MLU |
|---|---:|---:|---:|---:|
{''.join(proxy_lines)}
## 最佳诊断候选的逐 seed 配对差值

| Seed | Best epoch | Medium Mean Δ | P1 Δ | P10 Δ | High Mean Δ | Low raw Fulfill Δ | severity | severity 降幅 | 违例率 | 最大公共 MLU | 运行时间/分钟 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{''.join(seed_lines)}
Level 2 诊断性分层 paired bootstrap（10,000 次，seed 后再采样时间片）的 Medium P10 gap 为 `{bootstrap['estimate_mean']:+.4f}`，95% CI `[{bootstrap['ci95_lower']:+.4f}, {bootstrap['ci95_upper']:+.4f}]`。这个区间只说明 Medium 尾部改善稳定；它不能覆盖失败的排序门槛，也不是预注册的 Level 4 最终 CI。

## 各函数的负面证据

- `hinge`：小权重 0.25× 保住 Low，但 seed 490/491 severity 只降 46.0%/10.3%，违例率 80%/88%；更大权重通过直接压 Low 获得排序改善，违反 Low raw guard。
- `directional_hinge`：0.5× 在 seed 490 把 Medium P10 提高 0.0417 且 Low raw 增加 0.0030，但 severity 只降 30.4%、违例率 88%；更强权重出现 High/Low 或跨 seed 稳定性失败。
- `tail_directional_squared`：0.25×/0.5× 稳定改善 Medium 并保住 Low，但排序修复不足；4× 才在 seed 490 把违例率降至 14%，代价是 High Mean -0.0130、Low raw -0.0416。

这说明当前第三目标中，提升 Medium 的梯度与恢复全局 `Low <= Medium` 排序之间仍存在明显间隔。普通 hinge 的强权重容易以降低 Low 满足约束；stop-gradient 与 tail 版本避免直接压 Low 后，又难以在 12 epochs 内把大多数时间片翻转。

## 验证与完整性

- 正则单元测试 6/6；λ=0 确定性 CPU 数值回归最大参数差 0；Level 1 六个运行均通过保存/恢复、容量和 mask 审计。
- 所有比较使用 exact sequential actual-TM admission；观测到的最大累计容量比不超过 1.00000024，disabled flow 为 0。
- 数据指纹 `{fingerprint['sha256']}`，覆盖 {fingerprint['file_count']} 个输入文件、{fingerprint['total_bytes']} bytes；共为 {len(manifests)} 个完成运行写入 checkpoint/config/history/data/source 哈希 manifest。
- Level 4 bootstrap 与最终正则表格值为 N/A，因为预注册停止规则禁止继续。

## 后续建议

若继续下一轮，不建议放宽当前 guard。更有价值的方向是把约束从参数共享下的单一标量 penalty 改为“只对 Medium admission 分配可行残余容量”的结构化更新，或在第三目标中使用按时间片自适应的 primal-dual multiplier；两者都应从新的 Level 0/2 重新注册，而不是继续调大本轮 λ。
"""


def reproduction_text() -> str:
    return """# Reproduction commands (PowerShell)

```powershell
$py = 'D:\\kuroresearch\\.venv-hattrick\\Scripts\\python.exe'
Set-Location 'D:\\kuroresearch\\Hattrick-main\\test_diff_path\\shared2x_order_regularizer'

& $py -B -m unittest -v test_penalties.py
& $py -B integration_checks.py

foreach ($penalty in @('hinge','directional_hinge','tail_directional_squared')) {
  & $py -B run_experiment.py --level 0 --penalty $penalty --lambda-multiplier 1 --seed 490
  foreach ($m in @('0.25','0.5','1','2','4')) {
    & $py -B run_experiment.py --level 2 --penalty $penalty --lambda-multiplier $m --seed 490
  }
}

# Only safety-screen survivors were run at seed 491:
& $py -B run_experiment.py --level 2 --penalty hinge --lambda-multiplier 0.25 --seed 491
foreach ($m in @('0.5','4')) { & $py -B run_experiment.py --level 2 --penalty directional_hinge --lambda-multiplier $m --seed 491 }
foreach ($m in @('0.25','0.5')) { & $py -B run_experiment.py --level 2 --penalty tail_directional_squared --lambda-multiplier $m --seed 491 }

& $py -B common_evaluator.py
& $py -B generate_report.py
```

All commands are resumable. Add `--force` only to intentionally replace a run inside this isolated experiment directory.
"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = build_data_fingerprint()
    sources = source_hashes()
    manifests = write_run_manifests(fingerprint, sources)
    all_rows = all_level2_rows()
    runner.write_csv(REPORT_DIR / "level2_all_candidates.csv", flatten(all_rows))
    runner.write_json(REPORT_DIR / "level2_all_candidates.json", all_rows)

    best_details = []
    for seed in BEST_SEEDS:
        detail = analysis.compare_seed(2, BEST_PENALTY, BEST_MULTIPLIER, seed)[0]
        run_dir = runner.run_directory(2, BEST_PENALTY, BEST_MULTIPLIER, seed)
        complete = json_load(run_dir / "complete.json")
        config = json_load(run_dir / "config.json")
        detail.update(
            {
                "actual_lambda": float(config["actual_lambda"]),
                "best_epoch": int(complete["best_epoch"]),
                "runtime_seconds": float(complete["runtime_seconds_this_invocation"]),
                "best_checkpoint_sha256": runner.sha256(run_dir / "best_model.pt"),
            }
        )
        best_details.append(detail)
    runner.write_csv(REPORT_DIR / "best_candidate_per_seed.csv", flatten(best_details))
    runner.write_json(REPORT_DIR / "best_candidate_per_seed.json", best_details)
    bootstrap = analysis.hierarchical_bootstrap_p10_gap(
        2, BEST_PENALTY, BEST_MULTIPLIER, list(BEST_SEEDS), draws=10_000
    )
    bootstrap["scope"] = "exploratory Level-2 proxy only; not the unrun Level-4 final test"
    runner.write_json(REPORT_DIR / "level2_exploratory_bootstrap.json", bootstrap)

    proxy = [
        proxy_summary("Hattrick matched baseline", BEST_PENALTY, 0.0),
        proxy_summary("Hattrick+tail directional squared", BEST_PENALTY, BEST_MULTIPLIER),
    ]
    final_table = final_existing_table()
    runner.write_csv(REPORT_DIR / "level2_proxy_table.csv", proxy)
    runner.write_json(REPORT_DIR / "level2_proxy_table.json", proxy)
    runner.write_csv(REPORT_DIR / "shared_final_common_table.csv", final_table)
    runner.write_json(REPORT_DIR / "shared_final_common_table.json", final_table)

    calibrations = {
        penalty: json_load(
            runner.OUTPUT_ROOT / "level0_calibration" / penalty / "calibration.json"
        )["lambda_reference"]
        for penalty in runner.PENALTIES
    }
    run_seconds = sum(
        float(json_load(path)["runtime_seconds_this_invocation"])
        for path in runner.OUTPUT_ROOT.rglob("complete.json")
    )
    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "NO_GO",
        "level4_authorized": False,
        "stop_reason": "No candidate passed all Level-2 gates for both seeds 490 and 491",
        "selected_for_final": None,
        "best_safe_diagnostic": {
            "penalty": BEST_PENALTY,
            "lambda_multiplier": BEST_MULTIPLIER,
            "actual_lambda": proxy[1]["actual_lambda"],
            "seeds": list(BEST_SEEDS),
        },
        "lambda_reference": calibrations,
        "level3_executed": False,
        "level4_executed": False,
        "regularized_checkpoint_evaluated_on_400_499": False,
        "completed_run_count": len(manifests),
        "recorded_runtime_seconds_sum": run_seconds,
        "dataset_fingerprint": fingerprint,
        "delivery_source_sha256": sources,
        "common_existing_method_provenance": json_load(
            runner.OUTPUT_ROOT / "common_final_existing_methods" / "provenance.json"
        ),
        "test_contract": {
            "unit": "6 penalty/autograd tests",
            "integration": json_load(
                runner.OUTPUT_ROOT / "level1_correctness" / "integration_audit.json"
            ),
            "capacity_limit": 1.0001,
            "disabled_flow_limit": 1e-8,
        },
    }
    runner.write_json(REPORT_DIR / "frozen_no_go_manifest.json", frozen)
    report = report_markdown(
        final_table, proxy, best_details, bootstrap, fingerprint, manifests
    )
    (REPORT_DIR / "结论报告.md").write_text(report, encoding="utf-8")
    (runner.THIS_DIR / "README.md").write_text(
        "# Shared-2x order regularizer\n\n"
        "独立、可恢复的 GEANT 2x shared-path 优先级排序正则实验。最终严格结论为 "
        "NO-GO；详见 [artifacts/final_report/结论报告.md](artifacts/final_report/结论报告.md)。\n\n"
        + reproduction_text(),
        encoding="utf-8",
    )
    (REPORT_DIR / "复现命令.md").write_text(reproduction_text(), encoding="utf-8")
    print(json.dumps(frozen, indent=2), flush=True)


if __name__ == "__main__":
    main()
