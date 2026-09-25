# Sparse High-path ↔ Medium-path cross-attention

This Level-1-ready candidate preserves native Hattrick's serial
`High -> High+Medium -> High+Medium+Low` cascade and the complete six-objective
ordered projection. Only a prediction-side feature branch is added before
High.

## Why retain Medium OD identity

An edge-token module can tell High that an edge is carrying predicted Medium
load. It cannot determine which Medium OD produced that load or whether that
OD has a disjoint alternative route. Two decompositions can therefore have
identical edge-load tokens but require different High decisions.

OD identity is not logically required to imitate one isolated `9->18` sample:
an edge model could learn that `0->4` is hot and avoid it. It is required for a
generally identifiable DOTE-like response. At snapshot 476, Hattrick-LA places
0.651 of High `9->18` on path 0 (`9->0->4->18`), while DOTE-MC places 0.949 on
path 6 (`9->0->2->6->4->18`) and simultaneously places essentially all Medium
`9->18` on path 0. This is a path-complementarity decision, not merely an
aggregate edge-pressure decision.

## Minimal operator

- High query: `[B, 3696, 16]`.
- Medium path tokens: `[B, 462, 8, 16]`.
- Static neighbor OD cache: `[3696, 8]`.
- Each High path keeps its own Medium OD plus seven ODs with highest maximum
  inverse-capacity-weighted path overlap.
- Gathered sparse pairs: `[B, 3696, 8, 8, 16]`, or 236,544 High/Medium path
  pairs per snapshot instead of 13,660,416 full-cross pairs.
- One softmax resolves the eight concrete Medium paths inside each OD; a
  second softmax resolves the eight OD messages.
- The 16-d attention output is concatenated with the original two scalar
  uniform-pressure anchors and supplied to High init and every High RAU.

Both residual adapters use the existing Hattrick pattern
`Linear -> LeakyReLU -> Linear`, with the final weight and bias exactly zero.
Thus the matched epoch-0 policy is bitwise native, while all 32 output weights
can receive gradients on the first projected update.

The branch consumes only ESM `tm2_pred`, topology/capacity-derived path
embeddings, PTE, capacities, and masks. The formal Medium stage is still
recomputed after High, and actual traffic is never a policy input.

Run the CPU smoke test:

```powershell
D:\kuroresearch\.venv-hattrick\Scripts\python.exe smoke_test.py --snapshot 350 --timing-runs 3
```

It checks bitwise native initialization, strict actual-TM independence,
attention normalization and gradients, static-cache reuse, manual activation,
the complete six-objective projected-gradient parameter layout, and cached
CPU latency below 2x native.

After review, Level 1 can be started explicitly with:

```powershell
D:\kuroresearch\.venv-hattrick\Scripts\python.exe run_experiment.py --level 1 --seed 490
```
