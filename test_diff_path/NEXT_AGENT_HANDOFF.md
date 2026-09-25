# Hattrick 实验接手背景

## 项目背景

本项目复现论文 **Hattrick: Solving Multiclass TE using Neural Models**，并进一步研究论文假设在现实网络中的适用范围。

论文实验为每个 S-T pair 使用固定的 K 条候选路径，并让 High、Medium、Low 三类流量共享同一候选路径集合。本项目关注一个更一般的场景：同一 S-T pair 下，不同优先级流量可能拥有不同的可用路径集合。

为区分路径集合差异与负载分布变化的影响，现有实验包含共享路径和优先级路径强隔离场景，并比较了 1x 负载、冻结模型直接测试 2x 负载、以及使用完整 2x 数据重新训练后的结果。

当前观察是：Hattrick 在 shared 2x 场景中重训练后基本恢复，但在 strict 2x 场景中，Medium 流量仍明显落后于优化基线，并未体现出相对简单 DOTE-MC baseline 的结构优势。该现象是后续研究的主要背景，但目前不能据此断言 Hattrick 整体失效。

## 本地环境

- Workspace：`D:\kuroresearch`
- Repository：`D:\kuroresearch\Hattrick-main`
- Python environment：`D:\kuroresearch\.venv-hattrick`
- GPU：RTX 4090 Laptop GPU
- Gurobi：`D:\kuroresearch\gurobi1302\win64`
- License file：`D:\kuroresearch\gurobi_license\gurobi.lic`
- `GUROBI_HOME=D:\kuroresearch\gurobi1302\win64`
- `GRB_LICENSE_FILE=D:\kuroresearch\gurobi_license\gurobi.lic`

Gurobi license 绑定真实 Windows 用户 `nnqab`。沙箱用户运行求解器可能出现 `username mismatch`。不得读取、输出或记录 activation code。

## 代码与实验目录

Hattrick 主体代码位于：

- `frameworks/hattrick_system.py`
- `utils/training_utils.py`
- `run_hattrick.py`

不同路径集合与高负载实验集中在：

`D:\kuroresearch\Hattrick-main\test_diff_path`

其中主要 runner 为：

- `run_priority_mask_experiment.py`：100 片 priority-mask 实验。
- `run_dotemc_priority_mask_experiment.py`：DOTE-MC baseline。
- `run_long500_dotemc_hattrick_experiment.py`：500 片 1x 对照实验。
- `run_load2x_stress_experiment.py`：1x 模型直接测试 2x 流量。
- `run_load2x_retrain_control.py`：shared/strict 完整 2x 重训练对照。

已有结果目录：

- `results_dotemc_priority_masks_500slice`：500 片 1x 结果。
- `results_load2x_stress_500slice`：冻结模型 2x 压力测试。
- `results_load2x_retrain_shared_strict`：完整 2x 重训练对照。

完整数值、逐时间片记录、统计表和图表以以下文件为准：

- `results_load2x_retrain_shared_strict/load2x_retrain_comparison.md`
- `results_load2x_retrain_shared_strict/load2x_retrain_metrics.csv`
- `results_load2x_retrain_shared_strict/load2x_retrain_summary_stats.csv`
- `results_load2x_retrain_shared_strict/load2x_recovery_vs_frozen.csv`

## 数据与实验状态

相关 topology aliases：

- `geant_priomask500_shared`
- `geant_priomask500_strict`
- `geant_priomask500_shared_load2x_train`
- `geant_priomask500_strict_load2x_train`

500 片实验使用 train `0-350`、validation `350-400`、test `400-500`，候选路径数为 `K=8`，预测类型为 ESM。

现有 2x 对照已经完成数据准备、Gurobi oracle、BEST_MC、SWAN、Hattrick 训练测试和 DOTE-MC 训练测试。原始 1x Hattrick 模型未被覆盖，模型审计记录位于：

`results_load2x_retrain_shared_strict/original_hattrick_model_audit.json`

除非实验输入、路径、容量或优化问题发生变化，否则不需要重复计算已有 Gurobi 基线。任何新增模型和结果都应使用独立名称，保留当前结果作为对照。

## 研究指令

后续研究的编排方式、证据标准和实验要求单独记录在：

`D:\kuroresearch\Hattrick-main\test_diff_path\HATTRICK-STRICT2X-RESEARCH-PROMPT.md`

本文件只说明接手背景，不定义具体解决方案。
