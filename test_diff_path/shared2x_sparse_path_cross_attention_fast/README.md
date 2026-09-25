# Fast sparse High-path ↔ Medium-path cross-attention

This is the deployment-oriented compression of the successful `R=8,d=16`
sparse path cross-attention candidate. It preserves native Hattrick's serial
`High -> High+Medium -> High+Medium+Low` cascade and the complete six-objective
ordered projection.

## Compressed operator

- Latent width is 8 instead of 16.
- Each High path retains its own Medium OD plus the top three static
  inverse-capacity-overlap ODs (`R=4` instead of `R=8`).
- Each of those ODs still retains all eight concrete Medium paths.
- The route stage is streamed one neighbor OD at a time. The largest gather is
  `[B,3696,8,8]`; `[B,3696,4,8,8]` is never materialized.
- The four reduced OD messages `[B,3696,4,8]` are then combined by the second
  softmax.
- Two original scalar pressure anchors remain separate from the 8-dimensional
  cross-attention output, so both High adapters consume ten extra features.

For batch size one in FP32, the largest gathered K/V activation is about
0.95 MB. The old `R=8,d=16` full gather is about 15.14 MB: a 16x reduction in
the dominant gather, with 4x fewer attention multiply-adds. Runtime does not
retain route-attention maps unless the explicit smoke-audit flag is enabled,
and it avoids per-forward diagnostic GPU synchronizations.

Both residual adapters remain
`Linear -> LeakyReLU -> Linear`, with exactly zero final weight and bias. A
from-scratch epoch-0 candidate is therefore bitwise native. The formal Medium
stage is still recomputed after High. Policy inputs are strict ESM
`tm2_pred`, topology/capacity path embeddings, PTE, capacities, and masks;
actual traffic is never a policy input.

## R8,d16 warm start

`--warmstart-r8` accepts a trained old checkpoint. This is deliberately marked
as a **lossy initialization**, not an exact conversion. It copies all native
same-shaped weights, retains latent channels 0–7, maps the output projection's
old query channels 0–7 and context channels 16–23, and keeps the corresponding
first eight attention columns in both trained High adapters. The runtime
rebuilds an `own + top3` cache, so four discarded neighbors cannot be recovered
by slicing.

Run the CPU-only verification:

```powershell
D:\kuroresearch\.venv-hattrick\Scripts\python.exe smoke_test.py --snapshot 350 --timing-runs 3
```

It checks exact zero-output equivalence, manual policy activation, strict
actual-TM independence, attention gradients/simplexes, cache reuse, complete
six-objective projection shapes, first-step adapter gradients, cached latency
below 2x, analytical activation size, and the actual trained R8 checkpoint
conversion.

From-scratch Level 1:

```powershell
D:\kuroresearch\.venv-hattrick\Scripts\python.exe run_experiment.py --level 1 --seed 490
```

Warm-start Level 1:

```powershell
D:\kuroresearch\.venv-hattrick\Scripts\python.exe run_experiment.py --level 1 --seed 490 --warmstart-r8 ..\shared2x_sparse_path_cross_attention\artifacts\level2_proxy\seed_490\best_model.pt
```

Scratch and warm-start runs use separate artifact roots to prevent accidental
checkpoint/result mixing.
