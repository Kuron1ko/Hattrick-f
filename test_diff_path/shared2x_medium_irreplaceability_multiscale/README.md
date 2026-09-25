# Multiscale analytic Medium-irreplaceability scout

This fixed multiscale runner reuses the single-scale coverage/mask cache and the
complete six-objective runtime. It caches `r^1`, `r^2`, and `r^3`. Each scale
contributes two High-path features:

1. `absolute = log1p(sum(path edge value))`;
2. `centered = absolute - mean(absolute over feasible paths of the High OD)`.

The six features feed zero-output-initialized residual adapters with the same
design as the existing Medium-pressure High-init and High-RAU adapters. The
native High -> Medium -> Low cascade and strict ESM policy inputs are unchanged.

CPU-only smoke, cache, gradient, operator, parameter, and complete zero-init
policy equivalence audit:

```powershell
python smoke_test.py --snapshot 350
```

Training must be started explicitly:

```powershell
python run_experiment.py --level 1 --seed 490
```
