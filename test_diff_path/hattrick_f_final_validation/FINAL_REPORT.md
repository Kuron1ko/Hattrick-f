# Hattrick-f 最终验证报告

日期：2026-08-26

## 最终结论

Hattrick-f **不能作为满足原目标的 Hattrick 整体改良**。它可以被准确地描述为一种“Medium 优先的容量重分配变体”：在训练邻接评估区间上，它显著提高 Medium，但以显著降低 Low 为代价；在严格 ESM 的未见时间窗口上，Medium 增益大幅缩小，远期 High 还出现稳定下降。

若“明显改进”要求同时满足 High 不低于 0.995、Low 不显著下降、Medium 尽可能提升，则当前证据不支持 Hattrick-f。若允许明确牺牲 Low，Hattrick-f 则展示了有效而清晰的优先级迁移机制。

## 方法定义

- Phase A：原始六损失 Hattrick，训练 60 epoch，目标顺序为 `Fh, Uh, Fhm, Uhm, Fhml, Uhml`。
- Phase F：从 Phase-A 最优检查点继续训练 30 epoch；重置 Adam；全部参数可训练；保留串联结构、梯度投影与推理流程；只保留 `Fh, Fhm, Fhml`，持续撤去三个 MLU 目标。
- 检查点选择：验证集要求每个 High 快照不低于 0.995（允许 1e-4 数值误差），Low 均值相对 Phase A 的下降预算为 0.03。
- 公平控制：同一起点、同样 30 epoch、同样优化器重置和选择流程，但继续使用原始六损失。
- 严格推理：路由策略只接收 ESM 预测；真实流量只用于顺序接纳和评估。

## 训练邻接区间（2x，快照 400-499，三随机种子）

下表为跨种子的 NormFulFill 均值。

| 方法 | High | Medium | Low |
|---|---:|---:|---:|
| 原始 Hattrick（Phase A） | 0.997456 | 0.940931 | 1.133076 |
| 六损失等预算继续训练 | 0.998120 | 0.941496 | 1.138228 |
| Hattrick-f | 0.997859 | 0.984136 | 1.033384 |
| Hattrick-f - 原始 Hattrick | +0.000403 | **+0.043205** | **-0.099692** |
| Hattrick-f - 六损失控制 | -0.000260 | **+0.042640** | **-0.104844** |

三个种子的 Hattrick-f Medium 增益分别为 +0.04745、+0.04796、+0.03420；普通配对 bootstrap 的 95% 区间均不含 0。Low 降幅分别为 -0.09545、-0.09546、-0.10816，95% 区间同样全部不含 0。因此，“Medium 显著提高”和“Low 显著下降”都不是随机种子偶然现象。

High 的三个种子变化为 -0.00264、+0.00167、+0.00218，方向不一致。三个 Hattrick-f 模型的 High 均值都不低于 0.995，但 300 个快照中仍有 14.7% 的 NormFulFill 低于 0.995；最小值为 0.98093。因此它只满足“种子级均值”下界，不满足逐快照安全下界。

六损失等预算控制的 Medium 平均只变化约 +0.00057，证明 Hattrick-f 的大幅变化来自撤去 MLU，而不是单纯多训练 30 epoch。

## 严格 ESM 时间留出

这里报告实际顺序接纳的原始 FulfillRatio；它不是对 oracle 归一化后的 NormFulFill，因此 0.995 仅作为原始比率诊断，不能与训练区间的 NormFulFill 下界直接混用。相对比较仍是公平的，因为三种方法使用完全相同的 ESM 与真实流量。

| 窗口 | 指标 | 原始 Hattrick | Hattrick-f | 差值 |
|---|---|---:|---:|---:|
| 近端 500-999 | High | 0.928375 | 0.928685 | +0.000310 |
| 近端 500-999 | Medium | 0.579648 | 0.587061 | **+0.007413** |
| 近端 500-999 | Low | 0.493975 | 0.492484 | -0.001491 |
| 远端 9000-9499 | High | 0.987443 | 0.983867 | **-0.003576** |
| 远端 9000-9499 | Medium | 0.709919 | 0.711315 | +0.001396 |
| 远端 9000-9499 | Low | 0.557134 | 0.563077 | +0.005943 |

移动块 bootstrap（块长 12）显示：近端 Medium 对原始 Hattrick 和六损失控制，在三个种子上均显著为正；远端 Medium 对原始 Hattrick 只有一个种子显著为正，另外两个区间跨 0。远端 High 的三个种子相对两个基线全部显著为负。

这说明训练邻接区间约 +4.32 个百分点的 Medium 优势，在近端缩为约 +0.74 个百分点，在远端缩为约 +0.14 个百分点；不能宣称具有稳定的远期增益。

## 预测误差与链路容量稳健性

正式稳健性测试覆盖 3 个种子、近/远两个窗口、4 个分类偏差场景、6 个流量形状噪声场景，以及全部 36 条物理链路上的 180 个容量下降场景，共 464,400 条快照-类别结果。覆盖检查为 10,260/10,260 组，无缺失、重复或额外结果。

- 分类预测偏差：近端 `High -0.00027 / Medium +0.00749 / Low -0.00223`；远端 `High -0.00378 / Medium +0.00100 / Low +0.00762`。
- 流量形状噪声：近端 `High -0.00034 / Medium +0.00740 / Low -0.00237`；远端 `High -0.00376 / Medium +0.00136 / Low +0.00647`。
- 容量下降：近端 `High +0.00084 / Medium +0.00683 / Low -0.00315`；远端 `High -0.00290 / Medium +0.00328 / Low +0.00347`。
- 在全部 195 个场景中，Medium 在近端有 186 个场景优于 Phase A，在远端为 147 个；远端 High 只有 21 个场景优于 Phase A。
- 顺序接纳后的最大容量违反为 `7.15e-7`，低于 `1e-4` 容差，说明结果不是由容量越界虚增出来的。

## 路径变化与 MLU

Hattrick-f 确实学到了更激进、不平均的路径分配：相对 Phase A，需求加权路径总变差平均为 0.2138，27.8% 的 OD 改变了最大概率路径，平均熵下降 0.0716，第一路径份额上升 0.0631。六损失控制的对应数值为 0.1003、14.9%、-0.0260 和 +0.0178。

这种变化也带来明显的利用率代价。Hattrick-f 的平均 normalized MLU 为 `High 1.1308 / Medium 1.3326 / Low 1.3645`；原始 Hattrick 为 `1.0086 / 1.0051 / 1.0054`，六损失控制为 `1.0049 / 1.0040 / 1.0043`。这与远期 High 下降和对分布变化更敏感相吻合。

## 响应时间与成本

三种模型结构完全相同，参数量均为 20,105。RTX 4090 Laptop GPU 上，严格 ESM、热缓存、只计策略前向、CUDA 前后同步、每模型 200 次计时：

| 批量 | 原始 Hattrick 中位数 | Hattrick-f 中位数 | 结论 |
|---:|---:|---:|---|
| 1 | 64.028 ms | 63.735 ms | -0.46%，可视为测量波动 |
| 16 | 126.806 ms | 127.303 ms | +0.39%，可视为测量波动 |

因此 Hattrick-f 没有推理时延惩罚，也没有可主张的速度收益。该数字不包含数据 I/O、检查点加载、顺序接纳和 oracle 评估，不应作为端到端控制环时延。

训练方面，Hattrick-f 在原始 60 epoch 后增加 30 epoch，即增加 50% 的梯度训练轮数；实际三种子平均附加墙钟时间约 1,270 秒，但墙钟时间受并发和系统状态影响，应以固定 epoch 成本为主要描述。

## 与原论文的关系

论文第 3.3 节明确给出六项目标，并说明在其实验条件下可以省略的是 `Fh` 和 `Fhm`，因为这两个 FulfillRatio 梯度可能接近于零；论文没有提出“第一阶段使用 MLU、第二阶段撤去 MLU”。相反，第 3.1 和 3.3 节明确说明 MLU 用于在多个高 FulfillRatio 解之间偏向较低利用率方案，以减轻预测低估造成的链路饱和。论文的 `Hattrick-MLU` 消融只保留 MLU、撤去 FulfillRatio，与 Hattrick-f 的方向相反。

所以 Hattrick-f 是一个新的训练消融/优先级迁移方法，而不是论文已有简化。但现有结果也表明，它撤掉的正是论文用于预测误差稳健性的机制。

## 建议的论文表述

不建议写成“Hattrick-f 全面改进 Hattrick”。较准确的表述是：

> Hattrick-f is a second-stage fulfillment-only continuation that deliberately relaxes utilization balancing. It exposes a controllable Medium-versus-Low capacity trade-off while preserving the original serial architecture and projection mechanism. Its in-distribution Medium gain is substantial, but the gain attenuates under temporal holdout and may reduce High performance in far-future traffic.

若下一步继续研究，应把“撤去 MLU”改为受约束或自适应释放，而不是永久删除：以真实留出窗口上的 High/Low 风险为约束，只在存在可证明安全余量时降低 MLU 权重。

## 结果文件

- `artifacts/holdout/summary.json`：严格 ESM 近/远时间留出。
- `artifacts/robustness/summary.json`：预测偏差、形状噪声和 36 链路容量下降。
- `artifacts/statistics/summary.json`：599,400 条结果的完整性与统计协议。
- `artifacts/statistics/paired_summary.csv`：移动块和普通配对 bootstrap。
- `artifacts/latency/benchmark_latency_current_three_seed.json`：三种子响应时间。

