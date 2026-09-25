# Sparse-e48 matched edge-toll head 泛化诊断

## 审计范围

- High backbone：epoch 48，SHA256 `55dfbe15...acad`。
- matched edge-toll head：SHA256 `b94bf132...c8d9`。
- 未使用旧 Hattrick-LA e52 结果。
- train / safety / validation / test 分别为 `0–317 / 318–349 / 350–399 / 400–499`。
- 所有正式策略均为 strict ESM：policy forward 的真实 TM 为字面零；真实 TM 只在策略冻结后用于 sequential admission。
- head checkpoint 的 `test_data_read=false`。但提出“为 sparse-e48 重训 matched head”这一实验方向是在查看 bare sparse-e48 test 后做出的，因此结果属于 exploratory/post-hoc，而不是 untouched confirmatory evidence。

## 结论

`350–399` 的 `+0.01810` 是一个异常有利的连续时间段；`400–499` 的 `+0.01013` 更接近 train 的 `+0.01312` 和 safety 的 `+0.00824`。缩幅主要来自 `450–489` 的高且弥散拥塞状态，而不是 ESM 预测误差、gate 没打开、策略没改变或 baseline 没有提升空间。

最具体、可证伪的机制解释是：现有 head 只看到逐边的 7 个局部负载特征、静态 edge embedding 和全局 mean/max gate，不知道多条候选路径是否共同穿过同一个饱和 cut。在 validation 中，拥塞集中在少数热点边，局部 toll 容易绕开；在 test 后半段，饱和边覆盖面变大，许多“备选路”仍共享其他饱和边。于是 head 仍产生近似相同幅度的 path-cost 和策略 TV，但每单位换路的 Medium admission 收益下降。下文的候选路覆盖统计直接支持这一解释。

## 分割数据与提升空间

| split | M actual / ESM | OD NMAE | edge relative L1 | bare M | head M | M gain | headroom | gain/headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 39.992 / 40.122 | 0.1153 | 0.0531 | 0.94258 | 0.95570 | +0.01312 | 0.05742 | 22.9% |
| safety | 31.013 / 31.642 | 0.1088 | 0.0595 | 0.97742 | 0.98566 | +0.00824 | 0.02258 | 36.5% |
| validation | 35.462 / 34.988 | 0.1072 | 0.0542 | 0.96500 | 0.98310 | +0.01810 | 0.03500 | 51.7% |
| test | 45.845 / 45.601 | 0.1220 | 0.0581 | 0.93677 | 0.94690 | +0.01013 | 0.06323 | 16.0% |

test 的 Medium 总量偏差只有 `-0.244`，OD NMAE 仅比 validation 高约 14%，edge relative L1 只从 `0.0542` 增到 `0.0581`。同时 test headroom 反而是 validation 的 1.81 倍，因此缩幅不是 ceiling effect。

## Edge feature、gate 与策略活动

| split | pred H+M edge mean | pred H+M edge max | 过载边比例 | M gate | M toll abs mean | M policy TV | path-cost range |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 0.783 | 3.101 | 0.286 | 0.970 | 0.250 | 0.0826 | 1.441 |
| safety | 0.604 | 2.343 | 0.204 | 0.943 | 0.242 | 0.0797 | 1.398 |
| validation | 0.735 | 4.706 | 0.203 | 0.995 | 0.258 | 0.0874 | 1.472 |
| test | 0.866 | 3.605 | 0.319 | 0.978 | 0.252 | 0.0803 | 1.421 |

validation 到 test 的形态变化不是“单个瓶颈更热”，而是“更多边一起接近或超过容量”：平均 H+M load 上升、过载边比例上升 57%，但每快照最大 load 反而下降。test 的 gate、toll、TV 和 path-cost range 都仍接近 validation，故 head 确实在工作；只是 `M gain / policy TV` 从约 `0.207` 降为 `0.126`。

Medium gain 与拥塞的相关性也发生符号翻转：

| correlation | validation | test |
|---|---:|---:|
| gain vs actual M demand | +0.684 | -0.589 |
| gain vs mean H+M edge load | +0.771 | -0.567 |
| gain vs overloaded-edge fraction | +0.647 | -0.632 |
| gain vs headroom | +0.468 | -0.366 |
| gain vs policy TV | +0.493 | +0.684 |

这说明策略变化仍有方向性，但在 test 的高拥塞 active set 上无法转化成 admission 增益。

## 候选路 active-set 覆盖

为了直接检验“备选路是否仍穿过饱和边”，对每个 Medium OD 的 8 条候选路做了一个只读代理统计：以 baseline ESM 的 `H+M edge load >= 1` 定义 active edge；若一个 OD 没有任何候选路能完全避开所有 active edges，则记为 zero-escape。这个统计不进入策略，也没有真实 TM 泄漏。

| split | zero-escape OD（需求加权） | 每个 OD 最优路径仍穿过的 active edge 数 | base escape mass | head escape mass | shift |
|---|---:|---:|---:|---:|---:|
| train | 54.9% | 0.758 | 0.4197 | 0.4208 | +0.0011 |
| safety | 44.0% | 0.519 | 0.5370 | 0.5392 | +0.0022 |
| validation | 41.8% | 0.448 | 0.5450 | 0.5521 | +0.0071 |
| test | 55.0% | 0.773 | 0.4191 | 0.4202 | +0.0011 |

在最差的 `460–489`，zero-escape 为约 `64–65%`，即使每个 OD 选择其 8 条路径中的最好一条，仍平均穿过约 `1.03` 条 active edge；head 增加的 escape-path 概率质量仅 `0.0006–0.0011`。validation 的主要高增益段 `360–389` 中，zero-escape 约 `40%`、最优路径 active-edge 数约 `0.42`，head 可把 escape mass 增加 `0.0074–0.0113`。这把“弥散拥塞/共享 cut”从仅靠负载均值的推断推进到了候选路径层面的可观测证据。

## 时间漂移

validation 没有任何快照的 H+M 过载边比例超过 30%；test 有 49%。关键十片段为：

| window | actual M | 过载边比例 | bare M | matched gain | Low gain |
|---|---:|---:|---:|---:|---:|
| 350–359 | 28.92 | 0.16 | 0.9917 | +0.0021 | +0.0083 |
| 360–369 | 34.00 | 0.20 | 0.9638 | +0.0236 | +0.0013 |
| 370–379 | 37.18 | 0.21 | 0.9700 | +0.0235 | -0.0001 |
| 380–389 | 38.33 | 0.21 | 0.9615 | +0.0236 | +0.0015 |
| 390–399 | 38.88 | 0.23 | 0.9381 | +0.0177 | +0.0330 |
| 450–459 | 53.09 | 0.37 | 0.9160 | +0.0095 | +0.0351 |
| 460–469 | 59.64 | 0.44 | 0.9038 | +0.0051 | +0.0512 |
| 470–479 | 61.01 | 0.44 | 0.9059 | +0.0035 | +0.0524 |
| 480–489 | 57.13 | 0.41 | 0.9167 | +0.0064 | +0.0481 |

后半段并非 head 整体失效：Low gain 同时显著增大。训练目标是 `Medium gain + 0.5 * Low gain`，所以组合 utility 在 test 并未崩坏；崩坏的是用户真正关心的优先级分配，即收益从 Medium 偏向了 Low。Medium 和 Low 虽有不同输出，但共享 local trunk 和 gate，这一联合目标在未覆盖的拥塞形态上没有保证 Medium 优先。

## 三个反事实

1. **真实 edge-feature oracle（禁止部署，仅用于归因）**：保持 backbone/head 不变，只用真实 TM 计算 head 的 edge features。validation gain 从 `0.018097` 到 `0.018231`；test 从 `0.010131` 到 `0.010203`，只恢复 `0.000072`。因此 ESM mismatch 不是主因。
2. **gate 强制为 1**：test gain 只到 `0.010324`。因此 gate 未打开不是主因。
3. **固定 toll 排序，仅放大全局强度**：test 的 `1.25x / 1.50x` Medium gain 为 `0.01232 / 0.01364`，说明名义 head 还有幅度欠校准；但放大后 Low 相对 Hattrick-e 的安全性需要重新验证，不能把 test 上最优倍率当成正式结果。值得注意的是 `1.25x` 在未触碰 test 的 safety window 上 Medium/Low ECDF violation 均为 0。

## 为什么没有稳定超过原 Hattrick-e

bare sparse-e48 相对原 Hattrick 在 test 的 Medium 优势只有约 `+0.00240`；matched head 虽恢复了相对 bare 的 `+0.01013`，最终仍比原 Hattrick-e 低 `0.00034`。分块后差异更清楚：原 Hattrick 的 edge-toll head 在 `450–499` 的增益为 `+0.0118 / +0.0117 / +0.0142 / +0.0195 / +0.0192`，而 sparse matched head 是 `+0.0095 / +0.0051 / +0.0035 / +0.0064 / +0.0084`。所以“test 对所有 edge-toll 都更难”被数据否定；问题是 sparse-e48 High/正式 Medium 基础策略与该局部 toll 映射在后半段残余网络上的组合。

## 下一步的严格证伪与最小修正

已经在真实 sequential admission 的 High 阶段之后重建 residual capacity，并以 baseline Medium planned load 对 residual capacity 的超载定义 exact active set。结论保持：validation/test 的实际需求加权 zero-escape 为 `42.5% / 54.3%`，每 OD 最好路径仍穿过的 active-edge 数为 `0.457 / 0.731`，head 的 escape-mass shift 为 `+0.0069 / +0.0010`。因此结果不是简单的 `H+M>=1` 代理阈值伪象。

若 exact 统计仍成立，最小修正是给 Medium path-cost 增加一个路径级的 `overloaded-edge count / residual bottleneck / alternative-cut overlap` 特征，或在训练采样中按 overloaded-edge fraction 分层；而不是继续依赖逐边局部 toll。所有超参与 checkpoint 选择必须重新在 train/safety/validation 冻结，再用新的未见时间窗确认。

机器可读证据见同目录的 `diagnosis.json` 与 `per_snapshot.csv`。
