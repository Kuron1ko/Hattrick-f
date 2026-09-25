# Shared-2x complete Hattrick objectives

This isolated experiment restores the two fulfillment objectives omitted by
the released Hattrick training path.  No order regularizer or High freeze is
used.  The ordered objectives are:

1. `-Fh`
2. `Uh`
3. `-Fhm` (cumulative High + Medium admission)
4. `Uhm`
5. `-Fhml`
6. `Uhml`

The GEANT load factor, shared K=8 paths, model, optimizer, learning rate, batch
size, prediction input, and train/validation/evaluation splits match the
existing shared-2x experiment.

```powershell
$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
Set-Location 'D:\kuroresearch\Hattrick-main\test_diff_path\shared2x_full_objectives'

& $py -B -m unittest -v test_ordered_projection.py
& $py -B run_experiment.py --level 1 --seed 490
& $py -B run_experiment.py --level 2 --seed 490
& $py -B run_experiment.py --level 4 --seed 490
& $py -B generate_report.py
```

Runs are resumable. `--force` only removes the selected run directory inside
this experiment's own `artifacts` directory.
