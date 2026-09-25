# Shared-2x Protected Pareto Repair

This experiment leaves the restored `Fh/Fhm` Hattrick checkpoint unchanged.
At inference time it:

1. keeps Hattrick's High and Low path policies;
2. replays the predicted traffic to measure High load and the Low load that the
   baseline actually admits;
3. reserves that Low load edge by edge; and
4. solves one continuous LP that maximizes predicted Medium admission in the
   remaining capacity.

The repair uses predicted TMs, topology, capacities, paths, and masks only.
Ground-truth TMs and MF/MLU oracles are used only by the offline evaluator.

Typical commands:

```powershell
$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
& $py -B -m unittest -v test_diff_path\shared2x_protected_pareto\test_repair.py
& $py -B test_diff_path\shared2x_protected_pareto\run_experiment.py --start 350 --end 366 --reserve-factor 1.0 --label screen_r1
```

