# Optimized sparse path cross-attention

This is an isolated, checkpoint-compatible execution optimization of
`shared2x_sparse_path_cross_attention`.  It does not change R=8, K=8, d=16,
the two softmaxes, the serial Hattrick cascade, or the six projected objectives.

Changes:

- High paths are processed in chunks of 462, reducing the B=20 selected K/V
  allocation from about 303 MiB to about 38 MiB per chunk.
- Broadcast `multiply -> sum` attention contractions are expressed as batched
  matrix multiplication.  This removes two additional full
  `[B,P,R,K,D]` products.
- Static capacity summaries and neighbor feasibility masks are cached.
- A capacity-broadcast wrapper repairs repeated static test forwards with
  batch size greater than one.  It only expands repeated topology tensors and
  does not change their values.

The original checkpoint has exactly the same persistent state keys and loads
with `strict=True`.

CPU equivalence test using the existing Level-2 checkpoint:

```powershell
python test_diff_path/shared2x_sparse_path_cross_attention_optimized/equivalence_test.py
```

Run a new isolated experiment:

```powershell
python test_diff_path/shared2x_sparse_path_cross_attention_optimized/run_experiment.py --level 1 --seed 490
```

Artifacts are isolated below this directory's `artifacts/` tree.
