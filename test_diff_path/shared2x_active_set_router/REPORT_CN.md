# ESM-SAR：两倍流量下的预测时顺序接纳校正

## 结论

在严格的两倍流量、ESM 预测输入协议下，ESM-SAR 将最终 400–499 时间片的 Medium 指标从：

| 指标 | Hattrick 基线 | ESM-SAR | 增益 |
|---|---:|---:|---:|
| Mean | 0.934366 | **0.955185** | **+0.020819** |
| P1 | 0.890187 | **0.903706** | **+0.013519** |
| P10 | 0.896696 | **0.920293** | **+0.023598** |

约束同时满足：

| 优先级 | 基线 Mean / P1 / P10 | ESM-SAR Mean / P1 / P10 | 变化 |
|---|---|---|---|
| High | 0.998536 / 0.988258 / 0.996300 | **0.998536 / 0.988258 / 0.996300** | 仅浮点误差（约 1e-9） |
| Low | 1.147288 / 0.954877 / 0.999209 | **1.168153 / 0.981396 / 1.009717** | +0.020865 / +0.026518 / +0.010509 |

最终 100 片配对 bootstrap（20,000 次）显示：

- Medium Mean 增益 95% CI：**[+0.018350, +0.023136]**；
- Low Mean 变化 95% CI：**[+0.011268, +0.031081]**，没有显著下降；
- High Mean 仍为 **0.998536**，高于 0.995 目标。

## 方法

ESM-SAR（ESM Sequential-Admission Refinement）不是新的大网络，也不重新训练 Hattrick。它是在预测时追加的一个很小的物理一致校正层：

1. 冻结 Hattrick，并以它输出的三类路径比例作为初值；
2. High 路由完全旁路校正层；
3. 只使用当前 ESM 预测的 High、Medium、Low TM；
4. 将现有 sequential-admission 模拟器展开 24 步，对 Medium 和 Low 的路径 logits 做局部梯度校正；
5. 目标为 `Medium fulfillment + 0.25 × Low fulfillment − 0.01 × route KL`；
6. 每个 OD 内重新归一化，保持流量质量和禁用路径掩码；
7. 实际当前 TM 只用于最终 admission 重放和评价，不参与策略生成。

其任务特异性在于：优化层直接使用系统真正执行的“High → Medium → Low”顺序接纳，而不是用 MLU、边压力或训练 loss 近似它。High 不参与校正，因此 High 约束在结构上得到保护；Low 与 Medium 联合校正，避免把 Medium 增益简单转化为 Low 损失。

## 实验协议

- 小范围探索：350–357，共 8 片；
- 独立验证：358–399，共 42 片；
- 最终未触碰测试：400–499，共 100 片；
- 探索/验证使用 level-3 backbone；最终测试使用论文确认阶段的 level-4 backbone；
- 最终 checkpoint SHA-256：`ee6a03118ad80341019c60ae8c2b8cbf7b999d6d9d51e0f3de24658a91aa66ab`；
- 正式模式强制一次只处理一个时间片，消除批量稀疏算子在非光滑瓶颈切换处的数值差异；
- 100 片校正耗时约 43.34 秒，即约 0.433 秒/片（当前实现，GPU）。

严格逐片结果：

| 数据段 | Medium Mean 增益 | Medium P1 增益 | Medium P10 增益 | Low Mean 增益 | 约束 |
|---|---:|---:|---:|---:|---|
| 350–357 | +0.023367 | +0.026620 | +0.027871 | +0.003368 | 通过 |
| 358–399 | +0.034843 | +0.025567 | +0.023220 | +0.115021 | 通过 |
| 400–499 | +0.020819 | +0.013519 | +0.023598 | +0.020865 | 通过 |

## 复现

在仓库根目录使用项目虚拟环境运行：

```powershell
& 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe' `
  'D:\kuroresearch\Hattrick-main\test_diff_path\shared2x_active_set_router\run_final_esm_self_correction.py'
```

主要产物：

- `final_esm_self_correction_400_500.json`：最终汇总、checkpoint hash 与协议；
- `final_baseline_rows_400_500.csv`：100 片基线逐片结果；
- `final_candidate_rows_400_500.csv`：100 片候选逐片结果；
- `strict_validation_and_statistics.json`：探索/验证复核与 bootstrap；
- `test_esm_self_correction.py`：ESM 输入隔离、OD 质量守恒、禁用路径与 High 旁路测试。

## 当前边界

这是预测时优化层，因此比单次前向更慢；目前约 0.433 秒/片。批量化会更快，但在非光滑瓶颈切换处产生约 0.0042 的最大策略数值差，因此本报告只承认严格逐片结果。下一步若需要部署提速，应替换为确定性的平滑 bottleneck 算子，再重新验证，而不应直接采用当前批量模式。
