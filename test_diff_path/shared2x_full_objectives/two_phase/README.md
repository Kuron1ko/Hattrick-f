# Shared-2x two-phase experiment

Phase A reuses the completed six-objective checkpoint from
`shared2x_full_objectives/artifacts/level4_confirmation/seed_490/best_model.pt`.
It must pass validation High NormFulFill >= 0.9975 for two consecutive epochs.

Phase B freezes the Phase-A High path policy structurally. Medium and Low
remain trainable and enter one common sequential actual-TM admission pass.
The order penalty does not detach either Medium or Low.

```powershell
$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
Set-Location 'D:\kuroresearch\Hattrick-main\test_diff_path\shared2x_full_objectives\two_phase'

& $py -B -m unittest -v test_two_phase.py
& $py -B run_two_phase.py --level 0 --penalty full_hinge --lambda-multiplier 1 --seed 490
& $py -B run_two_phase.py --level 1 --penalty full_hinge --lambda-multiplier 1 --seed 490
& $py -B run_two_phase.py --level 2 --penalty full_hinge --lambda-multiplier 0 --seed 490
& $py -B analyze_results.py
```
