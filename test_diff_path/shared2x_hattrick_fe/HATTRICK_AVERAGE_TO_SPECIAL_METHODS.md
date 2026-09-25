# Hattrick“先平均，后特殊”改良方法评估

## 评估口径

本文中的成功概率是基于当前 2x 实验结果、方法与 Hattrick 串联结构的匹配程度，以及实现风险给出的主观工程判断，不是统计置信度，各方法概率也不能相加。

“成功”指在严格 ESM 推理、Level-4 多种子验证下同时满足：

- High Mean 不低于 0.995，P1/P10 无明显恶化；
- Low 无显著下降，顺序接纳不超过链路容量；
- Medium Mean、P10 和 P1 出现可重复提升，而非只在选模集提高。

## 按成功概率排序

| 排名 | 方法 | 主观成功概率 | 简要说明 | 主要风险 |
|---:|---|---:|---|---|
| 1 | **Hattrick-CX：中心路径 + High 零空间 crossover** | **50%–60%** | Phase A 用六个 loss 和 MLU 得到平均解；Phase B 撤去 MLU，让 Medium 梯度只沿实际 High 完成率约束的切空间改变 High 路径，并用 predictor-corrector 恢复越界样本。允许多步累积成明显不同的 High 分布。 | High 约束 Jacobian 的估计可能噪声较大；应在路径 logits 空间实现，避免直接处理全部网络参数。 |
| 2 | **双层/词典序最优面优化** | **35%–45%** | 把问题写成“在 High 最优解集合中最大化 Medium”，直接利用 High 存在多个优秀路径解这一性质。理论目标与需求最一致。 | 神经网络和顺序接纳是非凸的，实际只能近似跟踪 High 最优面；求解成本较高。 |
| 3 | **输出空间 central-path crossover** | **30%–40%** | Phase A 保持平均；Phase B 在 High 路径分配上执行 active-set/crossover，由 Medium 选择逐渐变得不均衡的路径，而不是在权重空间盲目微调。 | 如果没有实际 High 尾部 corrector，仍可能出现 Medium 上升后 High 突然跌破边界。 |
| 4 | **MLU continuation：逐步退火而非突然删除** | **20%–30%** | 将 MLU 系数从 1 平滑降到 0，使模型从平均解连续移动到特殊解，减少目标突变带来的漂移。实现最简单。 | 只能控制移动过程的平滑性，不能保证移动方向对 Medium 有利，也不能直接守住实际 High 完成率。 |
| 5 | **路径熵退火** | **15%–25%** | 高温阶段用熵正则得到分散路径，随后降低温度，让 Medium 推动路径变尖锐、特殊。比 MLU 更直接地控制“平均程度”。 | Hattrick 的问题是链路级容量冲突，不只是单个 OD 的路径熵；可能平均了不应平均的 OD，并重复 MLU 的作用。 |
| 6 | **High-CVaR 信任域单独使用** | **10%–20%** | 每步限制 High 下尾完成率和路径变化，可防止训练崩溃，适合作为 corrector 或安全层。 | 单独使用会限制大幅路径重构；此前信任域类实验已经说明它更适合保护，而不适合产生新的优秀 High 分布。 |

## 当前证据

Hattrick-fER 的 2x 微实验提供了两个关键观察：

- 第 6 轮 Medium 在两个验证半区分别提高 0.00767 和 0.01448，但 High 最低下降到 0.98557 和 0.98449；
- 第 8 轮 Medium 两区均为正增益，但 High 最低仍只有 0.99459 和 0.99358。

这说明明显的 Medium 改进方向确实存在，但 MLU 包络不能保护真实顺序接纳后的 High 尾部。因此最高优先级不是继续改变 MLU 权重，而是让 Medium 沿实际 High 约束的局部零空间移动。

## 推荐结论

优先实现 **Hattrick-CX**，核心更新为：

\[
d_M=-P_T\nabla L_M,
\qquad
P_T=I-J_H^\top(J_HJ_H^\top+\eta I)^{-1}J_H,
\]

其中 \(J_H\) 来自实际 High 下尾完成率、容量和 Low 预算等活跃约束。MLU 只用于产生 Phase-A 的平均起点，不再作为 Phase-B 必须维持的目标。

建议实验顺序：小数据验证路径-logit零空间投影 → 双验证块选模 → 独立近端严格 ESM → 远端严格 ESM → Level 3/4 多种子。只有小数据同时出现 High 安全和 Medium 双区正增益时，才进入大规模训练。

## 参考方法

- [Simple Bilevel Optimization：在主目标最优解集合中优化次目标](https://proceedings.mlr.press/v206/jiang23a/jiang23a.pdf)
- [Interior-Point Crossover：从内部解恢复特殊基本解](https://optimization-online.org/2018/09/6824/)
- [Continuous Null-Space Projection：次级任务在主任务零空间内行动](https://elib.dlr.de/75770/1/Dietrich_ICRA2012.pdf)
- [Graduated Optimization / Continuation](https://proceedings.mlr.press/v48/hazanb16.html)
- [Deterministic Annealing](https://proceedings.neurips.cc/paper_files/paper/1994/hash/92262bf907af914b95a0fc33c3f33bf6-Abstract.html)

