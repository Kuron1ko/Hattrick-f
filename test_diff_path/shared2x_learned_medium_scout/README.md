# Learned provisional-Medium scout

This candidate keeps the formal Hattrick cascade and the full six-objective
ordered projection unchanged. It adds a prediction-only scout before High:

1. A shared 32-wide MLP proposes provisional Medium path logits from the
   native path embedding, `tm2_pred`, and max/mean inverse path capacity.
2. One provisional edge aggregation supplies max/mean path pressure to a
   single 32-wide RAU update, after which the provisional edge load is
   recomputed.
3. Four features reach High init and every native High RAU iteration:
   global edge-load absolute/within-OD-relative values and same-Medium-OD
   overlap absolute/within-OD-relative values.
4. The inherited formal Medium stage is then recomputed after High exactly as
   in native Hattrick.

Both High residual adapters have output layers initialized exactly to zero.
With matched native weights, epoch-0 emitted policies are bitwise equal to
native Hattrick, while every hidden output channel can start learning on the
first optimizer step. The scout consumes ESM `tm2_pred`, path embeddings,
topology/PTE, capacities, and masks; actual traffic never enters the policy.

Run the CPU-only static audit and smoke test:

```powershell
D:\kuroresearch\.venv-hattrick\Scripts\python.exe smoke_test.py --snapshot 350
```

The smoke test checks exact native equivalence, strict actual-TM independence
with activated residual outputs, feature/scout gradients, provisional probability simplex,
relative-feature centering, added parameter count, and cached CPU inference
latency against the native model.

Run Level 1 explicitly only after the smoke test passes:

```powershell
D:\kuroresearch\.venv-hattrick\Scripts\python.exe run_experiment.py --level 1 --seed 490
```
