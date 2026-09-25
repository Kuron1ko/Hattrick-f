# Formal Stage-2-tied one-step Medium preview

This experiment removes the independent provisional Medium MLPs used by the
learned-scout candidate. Before the actual High decision it performs:

1. One full replay of native Stage 1 without lookahead residuals.
2. Formal Stage 2 initialization using the existing `mlp21` parameter object
   and its exact input order.
3. Exactly one formal bottleneck-aware RAU using the existing `mlp22` and
   `mlp_22_violation` objects and the exact formal input construction.
4. Combined High+Medium path maximum/mean load, each as an absolute and
   within-High-OD centered feature, enters the two zero-output High adapters.

The inherited formal `High -> High+Medium -> High+Medium+Low` cascade then runs
normally. There is no copied scout head and no consistency loss: the preview
and formal Medium stages are self-consistent because they invoke the same
weights. Training retains only the original six ordered objectives and
projection. The preview reads `tm1_pred` and `tm2_pred`; it never retains or
reads current actual traffic.

Run the CPU-only architecture, strict-input, full six-backward projection, and
coarse latency audit:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe' `
  smoke_test.py --snapshot 350 --timing-runs 5
```

Only after that audit passes, an explicit Level-1 command is:

```powershell
& 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe' `
  run_experiment.py --level 1 --seed 490
```

No training is started by the smoke test.
