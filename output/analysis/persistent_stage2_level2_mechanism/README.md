# Persistent Stage2 Level-2 机制对照

## 范围与身份

- 对比窗口：Level-2 固定 evaluation `200–249`，strict ESM，seed 490。
- parent sparse final epoch 12：SHA256 `7370c6e1...f08`。
- persistent Stage2 final epoch 12：SHA256 `4ad40dca...aa5`。
- Low guard：SHA256 `ce5197c9...585`。
- 对比的是 parent/persistent 的 Level-2 `final_model.pt`，没有混入 Level-4 epoch48 或旧 lookahead checkpoint。
- Medium 反事实 forward 中，High/Medium/Low 的真实 TM 输入全部为字面零；仅修改 ESM Medium prediction。

## 结果摘要

Persistent Stage2 确实改变了 High 的多解选择，并使路径更集中。它对 Medium 的**空间分布变化**比 parent 更敏感，但对 Medium 总量统一放大并不更敏感。新增 residual 的即时作用能够解释部分集中化和部分 Medium 响应；其余来自全模型在 12 轮训练中的共同适配。

Low guard 的结构审计通过：High、Medium path tensor 与 guard 前逐元素完全相同、甚至复用同一 storage；它只大幅重排 Low。

## Parent 与 persistent 的 High 路径

共比较 `50 × 462 = 23,100` 个 snapshot–OD，每个 OD 8 条路径。

| 指标 | parent sparse | persistent | 变化 |
|---|---:|---:|---:|
| 有效路径数，需求加权 | 3.7803 | 3.4858 | -0.2945 (-7.8%) |
| 主路径概率，需求加权 | 0.5162 | 0.5655 | +0.0493 |
| 有效路径数，全部 OD 平均 | 4.5818 | 3.7569 | -0.8249 |
| 主路径概率，全部 OD 平均 | 0.4390 | 0.5415 | +0.1025 |

- persistent 在 `90.63%` 的全部 OD、`69.48%` 的 High 需求质量上具有更小的有效路径数。
- parent→persistent 的 OD total variation：全部 OD 平均 `0.1663`，High 需求加权 `0.1085`，P90 `0.2928`，最大 `0.7705`。
- argmax 改变 `4,269 / 23,100 = 18.48%`；按 High 需求加权仅 `6.31%`。因此它主要是在保持重要 OD 主走廊的同时重分配次要路径概率，并非全面换主路。
- 每个 OD 的 split mass 最大误差为 `4.77e-7`，上述差异不是流量总量改变。

Level-2 sequential-admission 指标也与此一致：High mean/P1/P10 从 `0.999776/0.998136/0.999640` 到 `0.999895/0.998841/0.999916`；Medium 从 `0.908236/0.880557/0.887481` 到 `0.927420/0.898038/0.904455`。

## Stage2 residual 的直接贡献

在同一个 persistent checkpoint 内，仅把 `stage2_high_init_adapter` 和 `stage2_high_rau_adapter` 的最终输出层置零，其余参数完全不变：

| 指标 | adapters off | full persistent | full - off |
|---|---:|---:|---:|
| High 有效路径数，需求加权 | 3.5607 | 3.4858 | -0.0749 |
| High 主路径概率，需求加权 | 0.5569 | 0.5655 | +0.0086 |

- full 与 adapters-off 的需求加权 TV 为 `0.01803`；需求加权 argmax 改变 `1.05%`。
- residual 直接让 `79.21%` 的 High 需求质量更集中。
- 两个 adapter 输出层的 L2 范数分别为 `0.1065 / 0.1364`，不是仍停留在零初始化。
- 直接 residual 解释约 25% 的有效路径数总降幅（`0.0749 / 0.2945`）；其余是训练过程中原有 Stage1 attention、正式 Stage2 和新增 residual 的共同适配，不能把全部变化归因于一次 residual 加法。

## 对 Medium 的响应

反事实均只改 ESM Medium prediction：

| Medium 反事实 | parent High TV（需求加权） | persistent | persistent / parent | adapters-off | full / off |
|---|---:|---:|---:|---:|---:|
| 反转 OD 空间位置，保持总量与边际分布 | 0.06798 | 0.08009 | 1.178× | 0.07544 | 1.062× |
| Medium prediction 全部置零 | 0.07430 | 0.09162 | 1.233× | 0.08398 | 1.091× |
| Medium prediction 统一乘 1.10 | 0.00775 | 0.00630 | 0.814× | 0.00590 | 1.069× |

因此“更响应 Medium”应精确表述为：

- 对 Medium **在哪里**以及是否存在，persistent 的 High 概率分配响应更强，TV 分别高约 `17.8% / 23.3%`。
- 对 Medium 总量做统一 10% 缩放，persistent 反而比 parent 弱约 `18.6%`。这符合稀疏 OD/path message 的设计目标：增强冲突位置语义，而不是简单放大总压力标量。
- 在同一 persistent checkpoint 内，打开 Stage2 residual 会把三个反事实的需求加权 TV 分别提高约 `6.2% / 9.1% / 6.9%`，证明持久注入本身确实增加连续概率响应。
- 但 argmax 不是普遍增加：路径更集中后，主路会更稳定。空间反转时需求加权 argmax 改变率 parent/persistent 为 `2.93% / 5.95%`，而 adapters-off 为 `6.07%`；即 residual 主要改变概率幅度，不等于制造更多离散主路翻转。

## Low guard 是否只改 Low

结构级检查结果：

| 检查 | 结果 |
|---|---:|
| High `torch.equal` | true |
| Medium `torch.equal` | true |
| High / Medium 是否复用原 tensor storage | true / true |
| High / Medium 最大 policy delta | 0 / 0 |
| Low `torch.equal` | false |
| Low 需求加权 TV | 0.5787 |
| Low 需求加权 argmax 改变率 | 72.17% |
| Low OD split-mass 最大误差 | 2.98e-7 |
| disabled Low path 被重新打开的数量 | 0 |

重复 sequential admission 的 High/Medium 最大数值差分别只有 `3.81e-6 / 1.91e-6`（CUDA sparse replay 的 ulp 级差异）；这不是 policy 改动，因为上游 policy tensor 已逐元素完全相等。

Low guard 把 evaluation Low mean/P1/P10 从 `0.964697/0.876448/0.911631` 提到 `1.060522/0.975051/0.991743`，增益为 `+0.095825/+0.098604/+0.080112`；High 和 Medium 的策略不动。因此“Low guard 仅改 Low”得到结构和结果两层确认。

## 解释边界

parent 与 persistent 是分别从头训练的 checkpoint，所以 parent→persistent 的全部差异不能单独归因于新增 adapter。报告同时提供的 adapters-off 消融才是新增 Stage2 residual 在该 checkpoint 当前 forward 中的直接作用；它是诊断性消融，不是重新训练的可部署模型。

机器可读结果见 [report.json](report.json)，逐 snapshot–OD 的 High 比较见 [high_od_rows.csv](high_od_rows.csv)。
