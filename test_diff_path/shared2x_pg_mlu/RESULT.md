# PG-MLU 2× small-range result

## Change under test

Only the three training-time MLU inputs are changed:

`risk_util = actual_util + lambda * stopgrad(ReLU(actual_util - esm_util))`

The Hattrick architecture, RAU features, admission simulator, path masks, and
inference code are unchanged.  Actual traffic is used only by the offline
training loss and evaluation.  Test-time routing sees only ESM predictions,
topology, and capacities.

## Paired protocol

- Shared 2× topology, seed 490.
- Same pretrained 2× Hattrick initialization for every arm.
- Fresh optimizer for every arm.
- Training snapshots: 0–127; checkpoint validation: 128–159.
- Strict-ESM evaluation snapshots: 160–249 (90 snapshots).
- 12 epochs, effective batch size 32, learning rate 0.0002.

## NormFulFill

| Method | High Mean / P1 / P10 | Medium Mean / P1 / P10 | Low Mean / P1 / P10 |
|---|---|---|---|
| Original MLU | 0.98975 / 0.97937 / 0.98213 | 0.96397 / 0.91572 / 0.93083 | 1.10279 / 0.99177 / 1.00987 |
| PG-MLU λ=0.1 | 0.98967 / 0.97850 / 0.98190 | 0.96375 / 0.91733 / 0.93141 | 1.10363 / 0.99376 / 1.01071 |
| PG-MLU λ=0.5 | 0.99130 / 0.98117 / 0.98431 | 0.96109 / 0.91253 / 0.92657 | 1.10248 / 0.99205 / 1.01025 |

## Decision

**NO-GO for Level 4.**  At λ=0.5 the edge cost clearly changes the learned
policy and improves all reported High statistics, but it lowers Medium Mean,
P1, and P10.  Reducing λ to 0.1 removes most of the High benefit and does not
improve Medium Mean.  The result therefore rejects the shared single-lambda
form of PG-MLU for this task; a full 400–499 run would not be justified by the
predeclared small-range gate.
