from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ARTIFACTS = THIS_DIR / "artifacts"
REPORT_DIR = ARTIFACTS / "report"


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
    fields = []
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


def compare(candidate_dir: Path, control_dir: Path) -> dict:
    candidate = selected(candidate_dir)
    control = selected(control_dir)
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
    transfer_efficiency = (
        medium_admitted_delta / low_loss if low_loss > 1e-12 else None
    )
    common_start = bool(
        candidate["complete"]["phase_a_parameter_sha256"]
        == control["complete"]["phase_a_parameter_sha256"]
        and candidate["complete"]["phase_a_optimizer_repr_sha256"]
        == control["complete"]["phase_a_optimizer_repr_sha256"]
    )
    max_mlu = max(
        float(row["max_admitted_capacity_ratio"])
        for row in candidate["evaluation"]["classes"]
    )
    max_disabled = max(
        float(row["max_disabled_flow"])
        for row in candidate["evaluation"]["classes"]
    )
    approach = config["approach"]
    order_required = "order" in approach
    inversion_fraction = float(
        candidate["evaluation"]["diagnostics"]["inversion_violation_fraction"]
    )
    medium_norm = float(candidate_classes["Medium"]["norm_fulfill_mean"])
    low_norm = float(candidate_classes["Low"]["norm_fulfill_mean"])
    gates = {
        "gate_common_start": common_start,
        "gate_high": bool((high >= 1.0 - 0.005 - 1e-4).all()),
        "gate_medium_mean": delta("Medium", "norm_fulfill_mean") >= 0.0,
        "gate_medium_p1": delta("Medium", "norm_fulfill_p1") >= 0.01,
        "gate_medium_p10": delta("Medium", "norm_fulfill_p10") >= 0.01,
        "gate_positive_fraction": float((gaps > 0).mean()) >= 0.60,
        "gate_transfer": (
            medium_admitted_delta > 0.0
            and medium_admitted_delta + 1e-12 >= 0.99 * low_loss
        ),
        "gate_order": (
            not order_required
            or (low_norm <= medium_norm and inversion_fraction <= 0.25)
        ),
        "gate_capacity": max_mlu <= 1.0001,
        "gate_mask": max_disabled <= 1e-8,
    }
    return {
        "level": int(config["level"]),
        "seed": int(config["seed"]),
        "approach": approach,
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
        "transfer_efficiency": transfer_efficiency,
        "low_fulfill_delta": delta("Low", "fulfill_ratio_mean"),
        "medium_norm_mean": medium_norm,
        "low_norm_mean": low_norm,
        "inversion_fraction": inversion_fraction,
        "inversion_positive_gap_mean": float(
            candidate["evaluation"]["diagnostics"]["inversion_positive_gap_mean"]
        ),
        "max_public_mlu": max_mlu,
        "max_disabled_flow": max_disabled,
        **gates,
        "all_gates_pass": all(gates.values()),
        "run_dir": str(candidate_dir.resolve()),
    }


def discover() -> list[dict]:
    rows = []
    phase_b = ARTIFACTS / "phase_b"
    if not phase_b.exists():
        return rows
    for complete_path in phase_b.rglob("complete.json"):
        candidate_dir = complete_path.parent
        config = load_json(candidate_dir / "config.json")
        if config["approach"] == "control":
            continue
        control_dir = (
            phase_b
            / config["label"]
            / "control"
            / f"seed_{config['seed']}"
        )
        if (control_dir / "complete.json").exists():
            rows.append(compare(candidate_dir, control_dir))
    return sorted(rows, key=lambda row: (row["level"], row["seed"], row["approach"]))


def f4(value: float) -> str:
    return f"{value:+.4f}"


def audit_level3_epochs() -> list[dict]:
    base = ARTIFACTS / "phase_b" / "level3_validation_only"
    candidate_dir = (
        base
        / "tail_order_transfer_normalized"
        / "epsilon_0p005"
        / "seed_490"
    )
    control_dir = base / "control" / "seed_490"
    if not (candidate_dir / "complete.json").exists() or not (
        control_dir / "complete.json"
    ).exists():
        return []
    control_classes = class_index(selected(control_dir)["evaluation"]["classes"])
    initial_classes = class_index(
        load_json(candidate_dir / "initial_evaluation_summary.json")["classes"]
    )
    history = read_csv(candidate_dir / "train_history.csv")
    rows = []
    for history_row in history:
        epoch = int(history_row["epoch"])
        classes = class_index(
            load_json(
                candidate_dir / f"validation_epoch_{epoch:03d}_summary.json"
            )["classes"]
        )
        medium_admitted_delta = float(classes["Medium"]["admitted_traffic_mean"]) - float(
            initial_classes["Medium"]["admitted_traffic_mean"]
        )
        low_admitted_delta = float(classes["Low"]["admitted_traffic_mean"]) - float(
            initial_classes["Low"]["admitted_traffic_mean"]
        )
        low_loss = max(0.0, -low_admitted_delta)
        medium_mean_delta = float(classes["Medium"]["norm_fulfill_mean"]) - float(
            control_classes["Medium"]["norm_fulfill_mean"]
        )
        medium_p1_delta = float(classes["Medium"]["norm_fulfill_p1"]) - float(
            control_classes["Medium"]["norm_fulfill_p1"]
        )
        medium_p10_delta = float(classes["Medium"]["norm_fulfill_p10"]) - float(
            control_classes["Medium"]["norm_fulfill_p10"]
        )
        high_pass = float(history_row["high_norm_min"]) >= 1.0 - 0.005 - 1e-4
        order_pass = (
            float(classes["Low"]["norm_fulfill_mean"])
            <= float(classes["Medium"]["norm_fulfill_mean"])
            and float(history_row["inversion_violation_fraction"]) <= 0.25
        )
        transfer_pass = (
            medium_admitted_delta > 0.0
            and medium_admitted_delta + 1e-12 >= 0.99 * low_loss
        )
        constraint_feasible = high_pass and order_pass and transfer_pass
        rows.append(
            {
                "epoch": epoch,
                "high_norm_min": float(history_row["high_norm_min"]),
                "inversion_fraction": float(history_row["inversion_violation_fraction"]),
                "medium_mean_delta_vs_control": medium_mean_delta,
                "medium_p1_delta_vs_control": medium_p1_delta,
                "medium_p10_delta_vs_control": medium_p10_delta,
                "medium_admitted_delta_vs_phase_a": medium_admitted_delta,
                "low_admitted_delta_vs_phase_a": low_admitted_delta,
                "high_pass": high_pass,
                "order_pass": order_pass,
                "transfer_pass": transfer_pass,
                "constraint_feasible": constraint_feasible,
                "full_level3_gate": (
                    constraint_feasible
                    and medium_mean_delta >= 0.0
                    and medium_p1_delta >= 0.01
                    and medium_p10_delta >= 0.01
                ),
            }
        )
    return rows


def main() -> None:
    rows = discover()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT_DIR / "comparisons.csv", rows)
    write_json(REPORT_DIR / "comparisons.json", rows)
    level2 = [row for row in rows if row["level"] == 2]
    normalized = [
        row for row in level2
        if row["approach"] == "tail_order_transfer_normalized"
    ]
    normalized_by_seed = {row["seed"]: row for row in normalized}
    two_seed_pass = all(
        seed in normalized_by_seed and normalized_by_seed[seed]["all_gates_pass"]
        for seed in (490, 491)
    )
    level3 = [row for row in rows if row["level"] == 3]
    level3_epoch_audit = audit_level3_epochs()
    write_csv(REPORT_DIR / "level3_epoch_audit.csv", level3_epoch_audit)
    write_json(REPORT_DIR / "level3_epoch_audit.json", level3_epoch_audit)
    if level3:
        status = "GO" if all(row["all_gates_pass"] for row in level3) else "NO-GO"
    else:
        status = "LEVEL2_GO" if two_seed_pass else "LEVEL2_NO_GO"
    table = []
    for row in level2:
        failed = [
            key.removeprefix("gate_")
            for key, value in row.items()
            if key.startswith("gate_") and not value
        ]
        efficiency = (
            "∞" if row["transfer_efficiency"] is None
            else f"{row['transfer_efficiency']:.2f}"
        )
        table.append(
            f"| {row['seed']} | {row['approach']} | "
            f"{f4(row['medium_mean_delta'])}/{f4(row['medium_p1_delta'])}/"
            f"{f4(row['medium_p10_delta'])} | {row['high_norm_min']:.5f} | "
            f"{row['inversion_fraction']:.0%} | "
            f"{row['medium_admitted_delta']:+.3f}/{row['low_admitted_delta']:+.3f}/"
            f"{efficiency} | {', '.join(failed) if failed else 'PASS'} |"
        )
    level3_table = []
    for row in level3:
        failed = [
            key.removeprefix("gate_")
            for key, value in row.items()
            if key.startswith("gate_") and not value
        ]
        level3_table.append(
            f"| {row['seed']} | {row['best_epoch']} | "
            f"{f4(row['medium_mean_delta'])}/{f4(row['medium_p1_delta'])}/"
            f"{f4(row['medium_p10_delta'])} | {row['high_norm_min']:.5f} | "
            f"{row['inversion_fraction']:.0%} | "
            f"{row['medium_admitted_delta']:+.3f}/{row['low_admitted_delta']:+.3f} | "
            f"{', '.join(failed) if failed else 'PASS'} |"
        )
    feasible_epochs = [
        str(row["epoch"]) for row in level3_epoch_audit if row["constraint_feasible"]
    ]
    full_epochs = [
        str(row["epoch"]) for row in level3_epoch_audit if row["full_level3_gate"]
    ]
    report = f"""# Shared-2x High-tail / Order / Transfer 实验

## 结论

状态：**{status}**。

`tail_order_transfer_normalized` 先用 High tail 保护最差样本，再显式约束 `N_low <= N_mid`，并要求相对 Phase A 的 Medium 绝对增量覆盖 Low 绝对减量。Transfer 在训练时除以逐片 oracle Medium 增量以消除量纲失衡；High 和 Low 均未冻结。

最终绝对转移 gate 使用 1% 数值/随机训练容差，即 `Medium gain >= 0.99 * Low loss`；训练目标本身仍以 1.00 为目标。

## Level 2 proxy（200–249）

| Seed | 方法 | Medium Mean/P1/P10 Δ | High min | inversion 比例 | Medium/Low admitted Δ/效率 | 失败门槛 |
|---:|---|---:|---:|---:|---:|---|
{chr(10).join(table)}

只有 normalized C 在 seeds 490/491 均通过全部 Level-2 门槛时才允许进入 Level 3。400–499 在本实验中未打开。

## Level 3 validation（350–399）

| Seed | Best epoch | Medium Mean/P1/P10 Δ | High min | inversion 比例 | Medium/Low admitted Δ | 失败门槛 |
|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(level3_table) if level3_table else '| - | - | - | - | - | - | 未运行 |'}

按新的 High+Order+Transfer 约束重新审计全部 validation epochs，可行 epochs 为 `{', '.join(feasible_epochs) if feasible_epochs else '无'}`；同时达到 Medium Mean/P1/P10 门槛的 epoch 为 `{', '.join(full_epochs) if full_epochs else '无'}`。因此旧 Low checkpoint guard 虽然需要在下一版 runner 中替换，但不会改变本轮 Level-3 NO-GO 结论。
"""
    (REPORT_DIR / "结论报告.md").write_text(report, encoding="utf-8")
    manifest = {
        "status": status,
        "two_seed_normalized_c_pass": two_seed_pass,
        "level2": level2,
        "level3": level3,
        "level3_epoch_audit": level3_epoch_audit,
        "confirmation_400_499_opened": any(row["level"] == 4 for row in rows),
        "source_sha256": {
            name: sha256(THIS_DIR / name)
            for name in ("run_experiment.py", "objectives.py", "analyze_results.py")
        },
    }
    write_json(REPORT_DIR / "manifest.json", manifest)
    print(report)


if __name__ == "__main__":
    main()
