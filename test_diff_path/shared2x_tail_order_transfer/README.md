# Shared-2x High-tail / Order / Transfer

Independent follow-up to `shared2x_order_epsilon`. It reuses the frozen
restored-Fh/Fhm Phase-A checkpoints but writes every Phase-B model and result
under this directory.

Variants:

- `tail`: top-20% High epsilon violations.
- `tail_order`: High tail plus `ReLU(N_low-N_medium)`; neither class is frozen.
- `tail_order_transfer`: adds an absolute Phase-A transfer constraint and
  delays total-flow objectives for the first half of Phase B.
- `tail_order_transfer_normalized`: divides the transfer deficit by the
  per-snapshot oracle Medium increment to remove the absolute-flow unit scale.

```powershell
$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
Set-Location 'D:\kuroresearch\Hattrick-main\test_diff_path\shared2x_tail_order_transfer'
& $py -B -m unittest -v
& $py -B run_experiment.py --level 2 --approach control --seed 490
& $py -B run_experiment.py --level 2 --approach tail --epsilon 0.005 --seed 490
& $py -B run_experiment.py --level 2 --approach tail_order --epsilon 0.005 --seed 490
& $py -B run_experiment.py --level 2 --approach tail_order_transfer_normalized --epsilon 0.005 --seed 490
& $py -B analyze_results.py
```

The final confirmation window 400-499 is not opened by Levels 1-3.

## Current result

- `tail` keeps the epsilon High floor but leaves 74% proxy inversions.
- `tail_order` removes inversions but violates the High floor.
- the unnormalized transfer objective has about 17x the projected gradient norm
  of High-tail and also violates High.
- `tail_order_transfer_normalized` passes Level 2 for seeds 490/491: High min
  is `0.99701/0.99595`, inversion rate is `4%/0%`, and Medium Mean improves by
  `0.0675/0.0528` versus matched controls.
- Level 3 seed 490 is `NO-GO`: the saved best improves Medium
  Mean/P1/P10 by `0.0123/0.0067/0.0079`, but fails the hard High and absolute
  transfer gates. An audit of all 15 validation epochs finds constraint-feasible
  epochs 5 and 7, but neither meets the Medium P1/P10 improvement gates.
- Level 3 seed 491, Level 4, and snapshots 400-499 were not run.
