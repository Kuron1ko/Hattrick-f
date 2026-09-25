# Shared-2x objective-order and epsilon experiment

This isolated experiment compares a matched restored-objective continuation,
an `Fhm`/`Uh` objective-order swap, and a per-snapshot epsilon constraint on
High admission.  High and Low remain trainable in every Phase-B arm.

```powershell
$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
Set-Location 'D:\kuroresearch\Hattrick-main\test_diff_path\shared2x_order_epsilon'

& $py -B -m unittest -v
& $py -B run_experiment.py --level 0 --approach control --seed 490
& $py -B run_experiment.py --level 1 --approach control --seed 490
& $py -B run_experiment.py --level 2 --approach control --seed 490
& $py -B run_experiment.py --level 2 --approach swap --seed 490
& $py -B run_experiment.py --level 2 --approach epsilon --epsilon 0.005 --seed 490
```

All generated state is written below this directory's `artifacts` tree.

## Current staged result

- Level 0: `GO` for epsilon feasibility; the exact swap ceiling is effectively
  zero, while epsilon 0.25/0.5/1% improves the Medium ceiling on all 32 slices.
- Level 1: all unit, numerical-regression, checkpoint-recovery, split, capacity,
  mask, and route-diagnostic checks pass.
- Level 2: the user-authorized Mean gate relaxation from `0.9975` to `0.9965`
  admits the Phase-A checkpoints. Medium and Low are compared using absolute
  admitted traffic; the old Low percentage-point budget remains a checkpoint
  selection diagnostic rather than a final rejection rule.
- Epsilon `0.005` passes proxy 200-249 for seeds 490 and 491. Medium
  Mean/P1/P10 deltas are `+0.0609/+0.0666/+0.0636` and
  `+0.0237/+0.0336/+0.0328`; Medium admitted gains exceed Low admitted losses
  by `1.70x` and `6.13x`, respectively.
- Level 3 stops after seed 490 fails the enlarged validation gate: Medium
  Mean/P1/P10 deltas are `+0.0064/-0.0021/+0.0048`, and Low NormFulFill Mean
  (`1.0375`) again exceeds Medium (`0.9571`). Seed 491, Level 4, and the
  confirmation window 400-499 were not run. See the Chinese report for the
  archived-policy audit and complete tables.

The complete staged entry point is:

```powershell
& $py -B run_staged.py
```
