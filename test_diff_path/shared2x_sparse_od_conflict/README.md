# Sparse OD-specific High–Medium conflict scout

This isolated runner keeps Hattrick's native serial
`High -> High+Medium -> High+Medium+Low` cascade, its projection mechanism, and
the complete six-objective order:

```text
Fh, Uh, Fhm, Uhm, Fhml, Uhml
```

For High candidate path `p`, Medium OD `o`, and one feasible Medium path
`q=(o,k)`, define the inverse-capacity overlap

```text
z[p,o,k] = sum(e in p intersect q, 1/c[e]) / sum(e in q, 1/c[e]).
```

The cached OD-specific conflict is

```text
C[p,o] = -log(mean(k feasible for o, exp(-beta*z[p,o,k]))) / beta.
```

Taking the normalized smooth minimum before aggregating ODs distinguishes one
remaining clean path from many clean alternatives. This retains information
that aggregate edge pressure discards. At inference, the only dynamic input to
the scout is ESM-predicted Medium demand:

```text
path_harm = Medium_ESM_demand @ C.T
```

Its `log1p` absolute value and within-High-OD centered value feed the existing
zero-initialized High init/RAU residual adapters. The formal Stage 2 still makes
the Medium routing decision. Current actual traffic is never a policy input.

The full GEANT `[3696,462]` float32 cache is retained; the MVP introduces no
top-L approximation. Each beta has an independent artifact tree under
`artifacts/beta_<value>/`.

Run the CPU-only cache, strict-ESM, gradient, and exact zero-init policy audit:

```powershell
& 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe' `
  smoke_test.py --snapshot 350 --beta 4
```

Start an experiment explicitly (not part of the smoke test):

```powershell
& 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe' `
  run_experiment.py --level 1 --seed 490 --beta 4
```

Allowed beta values are `2`, `4`, and `8`.
