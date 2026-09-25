# Hattrick Strict-2x 严谨多智能体改进搜索指令

## 0. 角色与最终任务

你是本项目的根研究智能体，负责组织一项可复现、可证伪、可审计的系统研究。

工作目录：

`D:\kuroresearch\Hattrick-main`

首先完整阅读：

`D:\kuroresearch\Hattrick-main\test_diff_path\NEXT_AGENT_HANDOFF.md`

你的最终目标是：

> 找到并验证一种改进，使 Hattrick 在 GEANT、2x 高负载、不同优先级路径集合强隔离 strict 场景中，提高 Medium 流量的尾部性能，同时保持 High 性能和容量可行性。

最终交付必须是可执行代码、独立结果目录、完整实验记录和结论报告，而不是只有建议或模型结构草图。

不要预设某个具体修改一定有效。可以把“存在有效改进”作为搜索假设，但任何成功结论都必须由实验和审计支持。禁止为了满足任务而把随机波动、容量违反或指标选择性报告包装成改进。

## 1. 已知基线与待解决问题

固定实验条件：

- 数据：GEANT 前 500 个时间片。
- 切分：train `0-350`，validation `350-400`，test `400-500`。
- 负载：GT 和 ESM predicted TM 均乘 2。
- 候选路径：`K=8`。
- 场景：以 `strict` 为主，`shared` 为控制组。
- 当前 Hattrick：60 epoch，batch size 8，seed 490。

strict 2x 重训练后的关键 `NormFulFill Mean / P1 / P10`：

- Hattrick Medium：`0.8362 / 0.6937 / 0.7235`
- DOTE-MC sensitivity Medium：`0.8679 / 0.7395 / 0.7700`
- BEST_MC Medium：`0.9786 / 0.9386 / 0.9553`
- SWAN Medium：`0.8673 / 0.7548 / 0.8010`

strict 2x 的 Medium 实际 `FulfillRatio Mean / P1 / P10`：

- Hattrick：`0.5047 / 0.3535 / 0.3777`
- DOTE-MC sensitivity：`0.5230 / 0.3689 / 0.4007`
- BEST_MC：`0.5847 / 0.4832 / 0.5046`
- SWAN：`0.5198 / 0.4050 / 0.4175`

当前平均 MLU 约为：

- Hattrick：`1.0146`
- DOTE-MC sensitivity：`1.0498`

shared 2x 中 Hattrick 基本恢复，strict 2x 中 Medium 仍明显落后。

## 2. 证据标准


候选方案只有同时满足以下条件，才能称为“有效改进”：

1. strict 2x Medium NormFulFill P10 至少比 `0.7235` 提高 `0.1`。
2. strict 2x Medium P1 高于 `0.73`，且提升方向在多个 seed 中一致。
3. High NormFulFill Mean 不低于 `0.98`。
4. MLU 接近 1
5. 至少使用 3 个 seed
6. 与 unchanged Hattrick、DOTE-MC、BEST_MC、SWAN 在相同输入和指标定义下比较。
7. 改进必须在真实测试集 `400-500` 上成立。

若候选方案只满足部分条件，必须称为“局部改善”或“结果不确定”，不能称为解决。

## 3. 动态多智能体搜索

积极使用多个智能体，但不要固定成“每种方法永久分配 N 个 agent”。根智能体应根据证据动态创建、合并、暂停和重启路线。

第一轮至少保持以下互相独立的方法族：

1. **评估审计族**：核对指标、oracle 分母、MLU、mask 生效位置、最终 checkpoint 与验证最佳 checkpoint。
2. **优化诊断族**：分析每类 loss、梯度冲突、RAU 各阶段输出以及 Medium 流量在哪一步丢失。
3. **损失函数族**：研究 priority-aware weighting、Medium shortfall penalty、lexicographic surrogate、容量违反惩罚。
4. **表示结构族**：研究 class-specific path head、mask-aware path encoding、priority embedding
5. **数据与泛化族**：研究 load-scale augmentation、归一化、预测误差尺度和多 seed 稳定性。
6. **反例与压力族**：寻找候选改进失败的时间片、路径组合和瓶颈边，不负责“证明方案有效”。
7. **替代范式与新框架族**：不以保留 Hattrick 的 GNN、Transformer、RAU、共享表示或原损失为前提，从容量约束、优先级语义和可变路径集合出发重新形式化问题。该族可以提出完全不同的可学习优化器、分层决策过程或混合优化框架，但必须保持相同输入信息、路径可行域和在线推理约束，不能在推理时偷用 ground-truth oracle。

早期不要向大多数 agent 暴露当前最受偏好的修改，避免所有路线过早收敛到同一方案。每个 agent 必须从本地事实出发独立提出可验证假设。

新框架族从第一轮就保持独立活动，不需要等到所有局部修复失败才开始思考；但早期只提核心机制、可证伪预测、和最小原型。它不应被告知其他族当前的更改方案，以减少围绕旧架构进行伪创新。

## 4. 方法族登记表

维护：

`test_diff_path/results_hattrick_improvement_strict2x/approach_registry.md`

每条路线必须记录：

- 方法族和唯一编号。
- 明确假设。
- 预计影响 High、Medium、Low 和 MLU 的机制。
- 最小验证实验。
- 已运行命令和结果路径。
- 支持证据、反对证据和不确定性。
- 当前状态：`active / promising / rejected / blocked / merged`。
- 重新开启该路线所需的新机制或新证据。

当过多 agent 集中在一个方法族时，把一部分重新分配到证据不足的路线。

## 5. 先诊断，后改结构

严格按照以下闸门推进：分析结果，训练动力学诊断，测试成本较低且可解释的改动，修改或重构网络。


## 6. 对抗性审计

每个进入 `promising` 状态的候选方案都必须交给独立对抗 agent。对抗 agent 不负责优化结果，只负责尝试推翻它。候选方案未通过对抗审计前，根智能体不得宣布成功。


## 7. 具体产物要求

所有新增内容使用独立命名，不覆盖现有模型和结果。

建议输出目录：

`test_diff_path/results_hattrick_improvement_strict2x`



## 8. 计算预算纪律

采用逐级放大的实验漏斗。小实验只负责排除错误实现和缺乏趋势的方案，不负责证明最终有效。每一级都必须与同数据、同 epoch、同 seed 的 unchanged Hattrick 比较。

### Level 0：零训练诊断

- 读取已有结果、模型和 checkpoint，不重跑 Gurobi。
- 比较 final checkpoint 与 validation-best checkpoint。
- 检查 mask、指标分母、容量、梯度和最差时间片。
- 任何能被评估错误解释的问题，都必须在训练新模型前解决。

### Level 1：微型正确性实验

- 使用约 32 个训练时间片和 8 个验证时间片。
- 只训练 1-2 epoch。
- 目标仅是确认 forward/backward、mask 零分流、loss 方向、checkpoint 恢复、无 NaN/Inf 和输出行数。
- 这一层的性能数值不能用来选择最终方案。

### Level 2：小规模趋势筛选

- 建议使用 train `0-160`、validation `160-200`、proxy test `200-250`。
- 训练 10-15 epoch，至少 2 个 seed。
- 每个候选方案都运行预算匹配的 unchanged Hattrick。
- 只允许使用 `200-250` 作为代理评估窗口，不得查看最终 test `400-500`。
- 只有同时满足以下条件才晋级：
  - Medium P10 相对预算匹配基线至少提高 `0.02`，且两个 seed 方向一致；
  - High Mean 下降不超过 `0.01`；
  - MLU 不高于预算匹配基线 `0.01` 以上，且没有 disabled path 流量；
  - 改进不是由单个异常时间片贡献。

Level 2 的作用是淘汰明显无效方案。即使结果很好，也只能称为“有趋势”。

### Level 3：中规模复核

- 使用完整 train `0-350` 和 validation `350-400`。
- 训练 30 epoch，至少 2 个 seed。
- 在进入本级前冻结候选方案的主要结构和 loss 定义。
- 候选方案必须在两个 seed 上都改善 Medium 尾部，并保持 High 和 MLU。
- 优先保留机制不同的前 1-2 个方案，不要把大量近似变体全部送入完整实验。
- 本级仍不得查看最终 test `400-500`。

### Level 4：完整确认实验

- 在开始前冻结代码、超参数、checkpoint 选择规则和统计方法。
- 使用 train `0-350`、validation `350-400`、test `400-500`。
- 训练完整 60 epoch，至少 3 个 seed。
- 最终测试集只在方案冻结后使用，不能根据 `400-500` 的结果继续调参并重复宣称独立测试。
- 报告 paired gap、bootstrap 95% CI、每个 seed、每个类别、admitted traffic 和 MLU。
- 只有 Level 4 通过第 2 节的证据标准，才能称为有效改进。

在启动候选方案前，先运行或复用每一级的预算匹配原 Hattrick，确认“小实验中的方法排序”与已知完整实验没有明显矛盾。若低预算实验不能预测完整训练趋势，应提高筛选预算，而不是继续依赖错误代理。

路径、容量、mask 和 oracle 未改变时，复用现有 Gurobi 结果。只有优化问题本身改变时才重新求解，并解释旧分母为何失效。

Gurobi 许可证绑定真实 Windows 用户 `nnqab`。需要求解器时使用真实用户上下文。不得读取、打印或保存 activation code。

## 9. 综合与交叉启发

根智能体每轮都要：

1. 汇总新增证据，而不是汇总 agent 的信息。
2. 更新方法族登记表。
3. 标记重复路线和真正的新机制。
4. 把一个方法族的可迁移发现交给另一个方法族验证。
5. 保留至少两条机制不同的路线，直到证据足以淘汰其中一条。

只有在独立路线已经暴露真实优缺点后，才允许组合。如果一条路线只把问题转移为另一个同等困难的问题，或者依赖未经验证的假设，将其标记为 `blocked`，不要因为描述漂亮而持续投入计算。

## 10. 返回条件

不要在第一轮失败后停止。继续提出新的机制、重新检查错误假设，并让对抗 agent 审计最有希望的路线。

只有满足以下两种情况之一才结束：

### A. 找到通过审计的改进

返回证据，说明改进来自 checkpoint、loss、数据策略还是网络结构。

### B. 在约定预算内没有方案通过

不得捏造成功。返回最强的已验证负面结论、被排除的方法族，以及下一轮最值得验证的新机制。

## 11. 立即开始

先读取接手文档、最终统计 CSV、训练代码和 checkpoint 保存逻辑。建立方法族登记表，然后并行启动互相独立的 Gate A 审计、Gate B 诊断和反例分析。

不要先写最终结论。先产生可复核证据。
