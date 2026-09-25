# Hattrick-LA: persistent global Medium-pressure lookahead

## Method

The native `High -> High+Medium -> High+Medium+Low` cascade and the six-objective
ordered gradient projection are unchanged. Before the initial High decision, all
predicted Medium OD demands are previewed uniformly over their feasible paths and
aggregated into predicted edge utilization. Each High candidate path receives two
features: the maximum and mean predicted Medium utilization on that path. Two
zero-initialized residual adapters inject these features into the initial High head
and every High RAU iteration.

The adapters add 890 parameters (20,105 -> 20,995). Actual traffic is not used to
form a policy. In a counterfactual audit, zeroing or affinely changing all actual
TMs while keeping ESM predictions fixed changed all three emitted policies by
exactly 0. Changing only Medium ESM by 20% changed the High policy by up to 0.19892.

## Protocol

- Dataset: GEANT shared 8sp, 2x load, strict ESM inference.
- Train: snapshots 0-349; validation: 350-399; confirmation: 400-499.
- Training: 60 epochs, seed 490, full six-objective projection.
- Candidate roles and the primary checkpoint were frozen from validation before
  the training run wrote its completion marker.
- Primary e52 maximized the weakest validation improvement among Medium
  mean/P1/P10 under High and Low guards. e58 was the full-High-CDF guard. e57 and
  e55 were additional validation-selected guard candidates. Confirmation metrics
  did not replace the frozen primary.

## Strict-ESM confirmation results

Values are NormFulFill `Mean / P1 / P10` over snapshots 400-499.

| Method | High | Medium | Low |
|---|---:|---:|---:|
| Hattrick | 0.998537 / 0.988258 / 0.996300 | 0.934366 / 0.890187 / 0.896696 | 1.147288 / 0.954877 / 0.999209 |
| Hatrrick-e | 0.998537 / 0.988258 / 0.996300 | 0.947238 / 0.898299 / 0.909598 | 1.175702 / 0.965125 / 1.001309 |
| Hattrick-LA e52 (frozen primary) | 0.996849 / 0.986866 / 0.992815 | 0.942995 / 0.901880 / 0.909548 | 1.142422 / 0.983572 / 1.011361 |
| Hattrick-LA e58 (full-High-CDF guard) | 0.997129 / 0.986838 / 0.993695 | 0.941528 / 0.900415 / 0.907983 | 1.143046 / 0.979238 / 1.011585 |
| Hattrick-LA e57 (validation guard) | 0.996563 / 0.985700 / 0.992414 | 0.942526 / 0.905409 / 0.911526 | 1.144304 / 0.981453 / 1.015060 |

For the frozen primary e52 versus Hattrick:

- High Mean is 0.996849, above the requested 0.995 floor; its delta is -0.001687.
- Medium Mean/P1/P10 deltas are +0.008629 / +0.011692 / +0.012852.
- Low Mean/P1/P10 deltas are -0.004866 / +0.028694 / +0.012153.
- Paired-bootstrap 95% CIs (snapshot-paired candidate minus Hattrick,
  percentile method, 50,000 resamples, RNG seed 490) for mean deltas are High
  [-0.002061, -0.001325], Medium [+0.007105, +0.010149], and Low
  [-0.009597, -0.000196]. Thus the primary has a small but detectable Low-mean
  decrease, despite materially better Low P1/P10.

For preselected secondary e57, Medium Mean/P1/P10 improve by
+0.008160 / +0.015222 / +0.014831 versus Hattrick, while Low mean changes by
-0.002984 with paired-bootstrap 95% CI [-0.007986, +0.001767]. It therefore fits
the literal "no statistically significant Low-mean decline" criterion better,
but promoting it based on confirmation results would be post-selection; it needs
another seed or held-out window before being treated as the primary result.

## Conclusion

The experiment supports the architectural hypothesis: persistent predicted-Medium
context in the High stage improves all Medium metrics over native Hattrick while
retaining the cascade and projection mechanism, and High Mean remains above 0.995.
The current uniform-path pressure proxy is not yet a complete replacement for
Hatrrick-e: e52 Medium mean is 0.004243 below Hatrrick-e, although its Medium P1 is
0.003581 higher and P10 is effectively tied (-0.000051). A next version should
replace the uniform preview with a learned, differentiable Medium scarcity preview
and then combine it with the existing Medium/Low residual head.
