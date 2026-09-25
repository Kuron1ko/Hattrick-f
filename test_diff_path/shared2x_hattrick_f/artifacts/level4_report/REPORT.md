# Hattrick-f Level-4 独立验证

## 实验协议

- Phase-A：完整六目标 `Fh → Uh → Fhm → Uhm → Fhml → Uhml`。
- Phase-F：重置 Adam，保留串联结构、投影和所有参数，只训练 `Fh → Fhm → Fhml`。
- 训练：0–349；模型选择：350–399；独立评估：400–499。
- 路由推理严格只输入 ESM 预测值。400–499 未参与 epoch 或超参数选择。

## 独立评估结果

| 方法 | High Mean / P10 / P1 | Medium Mean / P10 / P1 | Low Mean / P10 / P1 |
|---|---|---|---|
| Six-loss Hattrick | 0.998537 / 0.996300 / 0.988258 | 0.934366 / 0.896696 / 0.890188 | 1.147288 / 0.999209 / 0.954877 |
| Hattrick-f strict Low | 0.997962 / 0.995392 / 0.986067 | 0.950245 / 0.923206 / 0.913245 | 1.157500 / 0.990638 / 0.965086 |
| Hattrick-f Pareto | 0.999108 / 0.997911 / 0.994156 | 0.979791 / 0.962118 / 0.951163 | 1.049664 / 0.986603 / 0.959897 |

严格 Low 版本在不牺牲 Low mean 的情况下将 Medium mean 提高
0.015878。
Pareto 版本将 Medium mean 提高
0.045425，
并同时提高 High mean/P10/P1；Low mean 回落
0.097624，
但 Low mean 仍为 1.049664，Low P1 反而从
0.954877 提高到 0.959897。

## 结论

Level-4 独立窗口支持 Hattrick-f。若 Low 不允许下降，采用 strict 版本；若允许把 Low 的
过量接纳让给 Medium，Pareto 版本明显更优，且 High 的三项分布统计均改善。
