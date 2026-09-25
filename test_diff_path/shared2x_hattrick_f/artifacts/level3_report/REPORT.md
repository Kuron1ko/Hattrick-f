# Hattrick-f Level-3 验证

## 方法

Phase-A 使用完整六目标 Hattrick：`Fh → Uh → Fhm → Uhm → Fhml → Uhml`。
Hattrick-f 从其最佳检查点开始，重置 Adam，Phase-F 保留投影机制和所有可训练参数，
但撤去三个持续 MLU 目标，只训练 `Fh → Fhm → Fhml`。推理严格只输入 ESM 预测值。

## 严格门槛下的结果

Hattrick-f 选择 epoch 14：High mean=0.999169，
High min=0.995364，
Medium mean=0.995967，Medium P10=0.989766，
Medium P1=0.980339，Low mean=1.007810。

同样门槛重新筛选、同样增加 15 epoch 的六目标控制组最佳为 epoch 12，
Medium mean=0.961415。Hattrick-f 的 Medium mean 绝对提升
0.034552。

## 机制证据

High 路径相对 Phase-A 的 demand-weighted TV 为
0.349450，主路径翻转率
30.19%。归一化熵从
0.643643 降到
0.339517，Top-1 路径占比从
0.486231 升到
0.730032。

结论：在 Level-3 验证窗口上，该设想成立。Phase-A 的 MLU 提供稳定均衡起点；Phase-F 撤去
MLU 后，`Fhm` 能沿 `Fh` 的投影零空间把 High 重构为明显更不均匀的分布，并显著释放 Medium。
