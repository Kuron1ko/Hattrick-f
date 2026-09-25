from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ARTIFACTS = THIS_DIR / "artifacts"
REPORT_DIR = ARTIFACTS / "report"
PREVIOUS = THIS_DIR.parent / "shared2x_tail_order_transfer" / "artifacts"
EPSILON = 0.005
HARD_TOL = 1e-4


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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


def selected(run_dir: Path) -> dict:
    complete = load_json(run_dir / "complete.json")
    evaluation = next(
        row for row in complete["evaluations"] if row["checkpoint"] == "best"
    )
    return {
        "complete": complete,
        "evaluation": evaluation,
        "metrics": read_csv(run_dir / "best_evaluation_metrics.csv"),
        "config": load_json(run_dir / "config.json"),
    }


def paired(metrics: list[dict], class_name: str, field: str) -> dict[int, float]:
    return {
        int(row["snapshot"]): float(row[field])
        for row in metrics
        if row["class"] == class_name
    }


def baseline_dir(label: str, seed: int) -> Path:
    return PREVIOUS / "phase_b" / label / "control" / f"seed_{seed}"


def comparison_from_parts(candidate: dict, control: dict, candidate_dir: Path) -> dict:
    candidate_classes = class_index(candidate["evaluation"]["classes"])
    control_classes = class_index(control["evaluation"]["classes"])
    config = candidate["config"]
    candidate_medium = paired(candidate["metrics"], "Medium", "norm_fulfill")
    control_medium = paired(control["metrics"], "Medium", "norm_fulfill")
    keys = sorted(control_medium)
    gaps = np.asarray(
        [candidate_medium[key] - control_medium[key] for key in keys],
        dtype=np.float64,
    )
    high = np.asarray(
        list(paired(candidate["metrics"], "High", "norm_fulfill").values()),
        dtype=np.float64,
    )

    def delta(class_name: str, field: str) -> float:
        return float(candidate_classes[class_name][field]) - float(
            control_classes[class_name][field]
        )

    medium_admitted_delta = delta("Medium", "admitted_traffic_mean")
    low_admitted_delta = delta("Low", "admitted_traffic_mean")
    low_loss = max(0.0, -low_admitted_delta)
    efficiency = medium_admitted_delta / low_loss if low_loss > 1e-12 else None
    inversion = float(
        candidate["evaluation"]["diagnostics"]["inversion_violation_fraction"]
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
        "gate_common_start": bool(
            candidate["complete"]["phase_a_parameter_sha256"]
            == control["complete"]["phase_a_parameter_sha256"]
            and candidate["complete"]["phase_a_optimizer_repr_sha256"]
            == control["complete"]["phase_a_optimizer_repr_sha256"]
        ),
        "gate_high": bool((high >= 1.0 - EPSILON - HARD_TOL).all()),
        "gate_medium_mean": delta("Medium", "norm_fulfill_mean") >= 0.0,
        "gate_medium_p1": delta("Medium", "norm_fulfill_p1") >= 0.01,
        "gate_medium_p10": delta("Medium", "norm_fulfill_p10") >= 0.01,
        "gate_positive_fraction": float((gaps > 0).mean()) >= 0.60,
        "gate_transfer": (
            medium_admitted_delta > 0.0
            and medium_admitted_delta + 1e-12 >= 0.99 * low_loss
        ),
        "gate_alignment": inversion <= 0.50,
        "gate_capacity": max_mlu <= 1.0001,
        "gate_mask": max_disabled <= 1e-8,
    }
    return {
        "level": int(config["level"]),
        "seed": int(config["seed"]),
        "approach": config["approach"],
        "bottleneck_multiplier": config["bottleneck_multiplier"],
        "best_epoch": int(candidate["complete"]["best_epoch"]),
        "medium_mean_delta": delta("Medium", "norm_fulfill_mean"),
        "medium_p1_delta": delta("Medium", "norm_fulfill_p1"),
        "medium_p10_delta": delta("Medium", "norm_fulfill_p10"),
        "positive_fraction": float((gaps > 0).mean()),
        "high_mean_delta": delta("High", "norm_fulfill_mean"),
        "high_p10_delta": delta("High", "norm_fulfill_p10"),
        "high_norm_min": float(high.min()),
        "medium_admitted_delta": medium_admitted_delta,
        "low_admitted_delta": low_admitted_delta,
        "transfer_efficiency": efficiency,
        "medium_norm_mean": float(candidate_classes["Medium"]["norm_fulfill_mean"]),
        "medium_norm_p1": float(candidate_classes["Medium"]["norm_fulfill_p1"]),
        "medium_norm_p10": float(candidate_classes["Medium"]["norm_fulfill_p10"]),
        "low_norm_mean": float(candidate_classes["Low"]["norm_fulfill_mean"]),
        "inversion_fraction": inversion,
        "max_public_mlu": max_mlu,
        "max_disabled_flow": max_disabled,
        **gates,
        "all_gates_pass": all(gates.values()),
        "run_dir": str(candidate_dir.resolve()),
    }


def discover() -> list[dict]:
    rows = []
    for complete_path in (ARTIFACTS / "phase_b").rglob("complete.json"):
        candidate_dir = complete_path.parent
        config = load_json(candidate_dir / "config.json")
        if config["approach"] not in ("bottleneck", "bottleneck_order"):
            continue
        control_dir = baseline_dir(config["label"], int(config["seed"]))
        if (control_dir / "complete.json").exists():
            rows.append(
                comparison_from_parts(
                    selected(candidate_dir), selected(control_dir), candidate_dir
                )
            )
    return sorted(
        rows,
        key=lambda row: (
            row["level"], row["seed"], float(row["bottleneck_multiplier"])
        ),
    )


def audit_level3_epochs() -> list[dict]:
    candidate_dir = (
        ARTIFACTS
        / "phase_b"
        / "level3_validation_only"
        / "bottleneck"
        / "epsilon_0p005"
        / "multiplier_16p4712909802"
        / "seed_490"
    )
    control_dir = baseline_dir("level3_validation_only", 490)
    if not (candidate_dir / "complete.json").exists():
        return []
    control = selected(control_dir)
    control_classes = class_index(control["evaluation"]["classes"])
    control_medium = paired(control["metrics"], "Medium", "norm_fulfill")
    initial = class_index(
        load_json(candidate_dir / "initial_evaluation_summary.json")["classes"]
    )
    rows = []
    for epoch in range(1, 16):
        summary = load_json(candidate_dir / f"validation_epoch_{epoch:03d}_summary.json")
        metrics = read_csv(candidate_dir / f"validation_epoch_{epoch:03d}_metrics.csv")
        classes = class_index(summary["classes"])
        medium = paired(metrics, "Medium", "norm_fulfill")
        gaps = np.asarray(
            [medium[key] - control_medium[key] for key in sorted(control_medium)]
        )
        medium_delta = (
            float(classes["Medium"]["admitted_traffic_mean"])
            - float(control_classes["Medium"]["admitted_traffic_mean"])
        )
        low_delta = (
            float(classes["Low"]["admitted_traffic_mean"])
            - float(control_classes["Low"]["admitted_traffic_mean"])
        )
        low_loss = max(0.0, -low_delta)
        phase_a_medium_gain = (
            float(classes["Medium"]["admitted_traffic_mean"])
            - float(initial["Medium"]["admitted_traffic_mean"])
        )
        phase_a_low_loss = max(
            0.0,
            float(initial["Low"]["admitted_traffic_mean"])
            - float(classes["Low"]["admitted_traffic_mean"]),
        )
        values = {
            "epoch": epoch,
            "high_norm_min": float(summary["diagnostics"]["high_norm_min"]),
            "inversion_fraction": float(summary["diagnostics"]["inversion_violation_fraction"]),
            "medium_mean_delta_vs_control": float(classes["Medium"]["norm_fulfill_mean"])
            - float(control_classes["Medium"]["norm_fulfill_mean"]),
            "medium_p1_delta_vs_control": float(classes["Medium"]["norm_fulfill_p1"])
            - float(control_classes["Medium"]["norm_fulfill_p1"]),
            "medium_p10_delta_vs_control": float(classes["Medium"]["norm_fulfill_p10"])
            - float(control_classes["Medium"]["norm_fulfill_p10"]),
            "positive_fraction": float((gaps > 0).mean()),
            "medium_admitted_delta_vs_control": medium_delta,
            "low_admitted_delta_vs_control": low_delta,
            "transfer_efficiency_vs_control": (
                medium_delta / low_loss if low_loss > 1e-12 else None
            ),
            "phase_a_medium_gain": phase_a_medium_gain,
            "phase_a_low_loss": phase_a_low_loss,
        }
        values.update(
            {
                "gate_high": values["high_norm_min"] >= 1 - EPSILON - HARD_TOL,
                "gate_medium_mean": values["medium_mean_delta_vs_control"] >= 0,
                "gate_medium_p1": values["medium_p1_delta_vs_control"] >= 0.01,
                "gate_medium_p10": values["medium_p10_delta_vs_control"] >= 0.01,
                "gate_positive_fraction": values["positive_fraction"] >= 0.60,
                "gate_transfer_vs_control": (
                    medium_delta > 0 and medium_delta + 1e-12 >= 0.99 * low_loss
                ),
                "gate_transfer_vs_phase_a": (
                    phase_a_medium_gain > 0
                    and phase_a_medium_gain + 1e-12 >= 0.99 * phase_a_low_loss
                ),
                "gate_alignment": values["inversion_fraction"] <= 0.50,
            }
        )
        gate_names = [name for name in values if name.startswith("gate_")]
        values["all_level3_gates"] = all(values[name] for name in gate_names)
        rows.append(values)
    return rows


def f4(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparisons = discover()
    audit = audit_level3_epochs()
    write_csv(REPORT_DIR / "comparisons.csv", comparisons)
    write_json(REPORT_DIR / "comparisons.json", comparisons)
    write_csv(REPORT_DIR / "level3_epoch_audit.csv", audit)
    write_json(REPORT_DIR / "level3_epoch_audit.json", audit)
    calibration = load_json(
        ARTIFACTS / "level0_gradient_calibration" / "calibration.json"
    )
    selected_l2 = [
        row for row in comparisons
        if row["level"] == 2
        and abs(float(row["bottleneck_multiplier"]) - 16.471290980189853) < 1e-6
    ]
    l3 = next((row for row in comparisons if row["level"] == 3), None)
    feasible_epochs = [
        row["epoch"] for row in audit
        if row["gate_high"] and row["gate_transfer_vs_phase_a"]
    ]
    full_epochs = [row["epoch"] for row in audit if row["all_level3_gates"]]
    lines = [
        "# Shared-2x 瓶颈对齐交换实验结论",
        "",
        "## 结论",
        "",
        "**Level 2 两个 seed 均显示明显改善，但 Level 3 为 NO-GO；不进入 seed 491、Level 4，也未打开 400–499。**",
        "",
        "新损失只降低占用 High+Medium 紧张链路的 Low。链路权重和 inversion gate 均 detach；Low 不冻结，Medium 由独立 Fm 目标优化。",
        "",
        "## Level 0 标定",
        "",
        f"- 固定 minibatch 数：{calibration['batches']}。",
        f"- 参考倍率：`{calibration['reference_multiplier']:.10f}`。",
        "- 搜索倍率：" + ", ".join(f"`{v:.6f}`" for v in calibration["candidate_multipliers"]) + "。",
        "- 所有 minibatch 均存在正权重瓶颈链路；正权重链路比例约 17%–32%。",
        "",
        "## Level 2：冻结倍率 16.4712909802",
        "",
        "| Seed | Medium Mean/P1/P10 Δ | High min | inversion | Medium/Low admitted Δ | 效率 | 结果 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected_l2:
        lines.append(
            f"| {row['seed']} | {f4(row['medium_mean_delta'])}/{f4(row['medium_p1_delta'])}/{f4(row['medium_p10_delta'])} "
            f"| {row['high_norm_min']:.5f} | {100*row['inversion_fraction']:.0f}% "
            f"| {row['medium_admitted_delta']:+.3f}/{row['low_admitted_delta']:+.3f} "
            f"| {row['transfer_efficiency']:.2f} | {'PASS' if row['all_gates_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "相对上一轮 aggregate order+transfer（seed 490 效率 1.65），瓶颈对齐的效率提高到 2.71，且 Low 损失从约 0.992 降到 0.581。",
            "",
            "## Level 3：seed 490",
            "",
        ]
    )
    if l3 is not None:
        failed = [
            name.removeprefix("gate_") for name, value in l3.items()
            if name.startswith("gate_") and not value
        ]
        lines.extend(
            [
                "| Best epoch | Medium Mean/P1/P10 Δ | High min | inversion | Medium/Low admitted Δ | 效率 | 失败门槛 |",
                "|---:|---:|---:|---:|---:|---:|---|",
                f"| {l3['best_epoch']} | {f4(l3['medium_mean_delta'])}/{f4(l3['medium_p1_delta'])}/{f4(l3['medium_p10_delta'])} "
                f"| {l3['high_norm_min']:.5f} | {100*l3['inversion_fraction']:.0f}% "
                f"| {l3['medium_admitted_delta']:+.3f}/{l3['low_admitted_delta']:+.3f} "
                f"| {l3['transfer_efficiency']:.2f} | {', '.join(failed)} |",
                "",
                f"满足 High 和 Phase-A transfer 的 epochs：`{feasible_epochs}`；满足全部 Level-3 门槛的 epochs：`{full_epochs or '无'}`。",
                "",
                "早期可行点只能得到很小的 Medium Mean 增益且 P1/P10 下降；后期 P10 上升时 High 越过逐片下界。ordered projection 只提供当前点的一阶正交，不能阻止强梯度与 Adam 动量在曲面上产生有限步长漂移。",
            ]
        )
    lines.extend(
        [
            "",
            "## 判断",
            "",
            "瓶颈对齐假设在 Level 2 得到支持：用更少 Low 损失换到了相近或更高的 Medium 增益。但 Level 3 的 Phase-A/Control 已更接近前沿，且 inversion gate 只在部分样本激活，固定强倍率造成优化轨迹在“High 可行”和“Medium 尾部改善”之间振荡。当前实现不能宣称放大成功。",
            "",
            "下一步应把固定倍率改为每步 trust-region：限制 BottleneckRelease 投影梯度范数不超过 HighTail/Fm 的指定比例，并在每次 optimizer step 后进行 High 可行性回退；这属于新的实验，不应用本轮 Level-3 窗口继续调倍率。",
        ]
    )
    report_path = REPORT_DIR / "结论报告.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_paths = [
        THIS_DIR / "README.md",
        THIS_DIR / "run_experiment.py",
        THIS_DIR / "objectives.py",
        THIS_DIR / "calibrate.py",
        THIS_DIR / "test_objectives.py",
        REPORT_DIR / "comparisons.csv",
        REPORT_DIR / "level3_epoch_audit.csv",
        report_path,
    ]
    write_json(
        REPORT_DIR / "manifest.json",
        {
            "status": "NO_GO_LEVEL3",
            "selected_multiplier": 16.471290980189853,
            "level4_opened": False,
            "snapshots_400_499_opened": False,
            "sha256": {str(path.resolve()): sha256(path) for path in manifest_paths},
        },
    )
    print(report_path, flush=True)


if __name__ == "__main__":
    main()
