# LR-SMF：Low 保底夹层最大流

## 核心判断

两倍流量下，原 Hattrick 的主要问题不只是路径策略，而是顺序接纳器只执行一次“按路径瓶颈同比缩放”。不同路径在不同链路达到瓶颈后，会留下无法被第二次利用的碎片容量。此前继续训练共享网络、蒸馏换路或修改损失，都会同时扰动 High、Medium、Low，难以稳定利用这部分容量。

LR-SMF（Low-Reserved Sandwich Max-Flow）不再修改神经骨干。它先运行原 Hattrick 接纳器，得到可行的 High、Medium、Low 路径流量，然后：

1. 原样固定 High 路径流量；
2. 原样预留 Low 路径流量及其逐链路占用；
3. 在 `capacity - High load - Low load` 中，只对 Medium 求一次多商品最大流；
4. 输出 `原 High + 新 Medium + 原 Low`。

这相当于把 Medium 放在两个不可移动的安全层之间，因此称为“夹层最大流”。

## 保证

设原系统已接纳路径流量为 \(x_H^0,x_M^0,x_L^0\)，路径—链路矩阵为 \(A\)，容量为 \(c\)。新 Medium 解为：

\[
\max_{x_M\ge 0}\;\mathbf 1^T x_M
\]

满足：

\[
A x_M \le c-Ax_H^0-Ax_L^0,
\qquad
\sum_{p\in OD_i}x_{M,p}\le d_{M,i}.
\]

因为原 \(x_M^0\) 本身满足这些约束，所以新问题一定可行，并且：

- High 路径流量逐元素不变；
- Low 路径流量逐元素不变；
- Medium 总接纳量逐时间片不下降；
- 三类合并后的链路负载不超过容量。

这些是结构保证，不依赖训练权重、惩罚系数或数据顺序。

## 为什么适合本任务

任务的特殊约束是 High 必须守住、Low 不能显著下降，而 Medium 要尽可能提升。LR-SMF 直接把前两项变成等式不变量，把全部可优化自由度留给 Medium。相比继续调整多目标 loss，它避免了共享参数梯度冲突；相比此前失败的预测流量 LP，它不把优化结果重新转换成策略再经历一次顺序缩放，而是在接纳层直接执行最终可行流量。

## 在线输入边界

该接纳层需要当前请求的 OD 需求、链路容量和候选路径。原实验的顺序接纳模拟器同样使用真实请求量，因此在当前系统定义中属于可用输入。如果部署场景要求在请求到达前仅凭预测流量一次性下发固定分流比例，则该方法不能与纯预测策略直接等价比较，需要改造成滚动接纳版本。

## 复现

在仓库根目录运行：

```powershell
.\.venv-hattrick\Scripts\python.exe Hattrick-main\test_diff_path\shared2x_sandwich_flow\run_experiment.py --level 4 --backbone-seed 490 --low-reservation footprint --force
```

统计配对自助区间：

```powershell
.\.venv-hattrick\Scripts\python.exe Hattrick-main\test_diff_path\shared2x_sandwich_flow\analyze_results.py --run-dir Hattrick-main\test_diff_path\shared2x_sandwich_flow\artifacts\level4_confirmation\low_footprint\backbone_490
```
