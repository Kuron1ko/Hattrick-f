# Shared-2x order regularizer

独立、可恢复的 GEANT 2x shared-path 优先级排序正则实验。最终严格结论为 NO-GO；详见 [artifacts/final_report/结论报告.md](artifacts/final_report/结论报告.md)。

# Reproduction commands (PowerShell)

```powershell
$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
Set-Location 'D:\kuroresearch\Hattrick-main\test_diff_path\shared2x_order_regularizer'

& $py -B -m unittest -v test_penalties.py
& $py -B integration_checks.py

foreach ($penalty in @('hinge','directional_hinge','tail_directional_squared')) {
  & $py -B run_experiment.py --level 0 --penalty $penalty --lambda-multiplier 1 --seed 490
  foreach ($m in @('0.25','0.5','1','2','4')) {
    & $py -B run_experiment.py --level 2 --penalty $penalty --lambda-multiplier $m --seed 490
  }
}

# Only safety-screen survivors were run at seed 491:
& $py -B run_experiment.py --level 2 --penalty hinge --lambda-multiplier 0.25 --seed 491
foreach ($m in @('0.5','4')) { & $py -B run_experiment.py --level 2 --penalty directional_hinge --lambda-multiplier $m --seed 491 }
foreach ($m in @('0.25','0.5')) { & $py -B run_experiment.py --level 2 --penalty tail_directional_squared --lambda-multiplier $m --seed 491 }

& $py -B common_evaluator.py
& $py -B generate_report.py
```

All commands are resumable. Add `--force` only to intentionally replace a run inside this isolated experiment directory.
