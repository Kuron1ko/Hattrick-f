# Analytic Medium-irreplaceability scout

This runner retains the native serial Hattrick cascade, zero-initialized
Medium-pressure adapters, and the complete six-objective ordered projection.
It replaces iterative Medium previews with one cached analytic feature.

For Medium OD `o` and edge `e`, static candidate-path coverage is

```text
r[o,e] = feasible paths of o containing e / feasible paths of o
```

At each forward pass, ESM-predicted Medium OD demand `d` produces

```text
edge_value = (d @ r**gamma) / capacity
```

The sparse path-edge matrix maps this value back to every High path. The two
features consumed by the existing adapters are its `log1p` path sum and the
within-OD centered relative deviation. Coverage and `r**gamma` are cached after
their first construction. Path masks are included in both coverage and High-OD
centering.

Every gamma has an isolated output subtree, for example
`artifacts/gamma_2/...`.

Run the snapshot feature/cache/gradient smoke test (no training):

```powershell
python smoke_test.py --snapshot 400 --gamma 2
```

Start a six-objective experiment explicitly:

```powershell
python run_experiment.py --level 1 --seed 490 --gamma 2
```
