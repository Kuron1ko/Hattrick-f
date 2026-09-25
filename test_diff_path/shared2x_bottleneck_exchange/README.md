# Shared-2x bottleneck-aligned exchange

Independent follow-up to `shared2x_tail_order_transfer`. It reuses the same
frozen restored-Fh/Fhm Phase-A checkpoints and does not modify prior artifacts.

For snapshot `i` and directed link `e`, the new loss replays sequentially
admitted path flow onto the topology and computes

```text
pressure_ie = clip((util_high+medium_ie - 0.85) / 0.15, 0, 1)
weight_ie   = stopgrad(pressure_ie / sum_e pressure_ie)
gate_i      = stopgrad(ReLU(N_low_i - N_medium_i))
L_b         = lambda * mean_i gate_i * sum_e weight_ie * util_low_ie
```

The pressure and inversion gate are detached, so Medium cannot reduce the loss
by lowering itself. The direct loss gradient is through Low admission/routing;
Low is not frozen, and the separate `Fm` objective improves Medium. Ordered
Phase-B objectives are:

```text
HighTail, TransferNorm, BottleneckRelease, Fm, Uhm,
FhmlCurriculum, UhmlCurriculum, Uh
```

Gradient calibration on 20 deterministic Level-2 minibatches selected the
screening set `2.058911, 4.117823, 8.235645, 16.471291`. Level 2 selected and
froze `16.4712909802` before Level 3.

```powershell
$py = 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe'
Set-Location 'D:\kuroresearch\Hattrick-main\test_diff_path\shared2x_bottleneck_exchange'
& $py -B -m unittest -v
& $py -B calibrate.py
& $py -B run_experiment.py --level 2 --approach bottleneck --epsilon 0.005 --bottleneck-multiplier 16.471290980189853 --seed 490
& $py -B analyze_results.py
```

Current decision: both Level-2 seeds pass, but Level-3 seed 490 is NO-GO. No
Level-4 run was made and snapshots 400-499 remain unopened. See
`artifacts/report/结论报告.md`.
