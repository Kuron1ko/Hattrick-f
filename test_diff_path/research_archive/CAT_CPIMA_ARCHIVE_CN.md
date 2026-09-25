# CAT-C-PIMA 封存记录

封存日期：2026-08-20（Asia/Shanghai）

状态：**只读历史方案；后续全新探索不得把本方案作为起点、组件或默认假设。**

## 方法摘要

CAT-C-PIMA 将 High 结构旁路；Medium 使用逐 OD 质量守恒的指数路径倾斜；Low 保留原 C-PIMA 有界 causal shield。因果在线版本根据已结束时间片的 Low 接纳债务，在 Medium 温度 0.5 与 1.5 之间切换。当前真实 TM 不进入当前策略，但历史实际接纳差进入下一时刻控制，因此它是在线扩展，不是最严格的无状态纯 ESM 协议。

## 已冻结结果

- High Mean：0.998537，路径策略不变。
- Medium Mean：0.934366 → 0.941344，增益 +0.006977。
- Medium P1 / P10 增益：+0.005949 / +0.006514。
- Low Mean 点估计变化：-0.001147。
- Low Mean 配对 bootstrap 95% CI：[-0.004545, +0.002163]，未发现显著下降。
- 严格无历史反馈的静态纯 ESM 版本：Medium Mean 平均增益约 +0.005404。

## 权威产物

- 方法与实验报告：`../shared2x_cpima_gain/REPORT_CN.md`
- 最终机器可读统计：`../shared2x_cpima_gain/feedback_result/summary.json`
- 实验入口：`../shared2x_cpima_gain/run_experiment.py`
- 反馈统计：`../shared2x_cpima_gain/analyze_feedback.py`

## SHA-256

- `REPORT_CN.md`: `9E6E85B06706E501E69AEF39ADCA945588D794BA3BEA9A9310D99E51F8D39FE9`
- `feedback_result/summary.json`: `A9010202696DC8C111093C1743F1013671332F3E5550D8322CB9F171B90E1814`
- `run_experiment.py`: `8AA5A0FB882E6505C7727DA58E2ACCB60FE0AB8BE98D20809FA75B60654A4EEA`
- `analyze_feedback.py`: `E3FF1D045ED3FD05F2CDA22133F12818BE8AF67C8BCEDC44933B12E0DFAFCAF0`

## 探索隔离声明

从本记录创建之后开始的新研究，不复用 CAT-C-PIMA 的温度倾斜、Low 债务、C-PIMA shield、策略混合、历史误差场景、蒸馏或残余最大流设定。只保留任务目标、论文协议、数据划分和评价指标。
