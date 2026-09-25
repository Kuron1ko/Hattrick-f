# Final strict-ESM 2x three-method comparison

This directory contains a frozen-manifest final-evaluation pipeline for:

1. Hattrick
2. Hattrick-e
3. Sparse cross-attention

The evaluator never discovers checkpoints, reads validation rows, or ranks
models. It accepts only the exact checkpoint or test-row path and SHA256 already
recorded in a manifest. Every method must declare either a validation-only
selection rule whose range ends before snapshot 400, or a predeclared final
epoch. The manifest also records immutable selection evidence and asserts that
test metrics were not read during selection.

For checkpoint inputs, evaluation runs on CPU over snapshots 400–499. Before
metric replay, it changes every actual traffic matrix while keeping ESM
predictions fixed and requires all three policies to remain bitwise identical.
Rows inputs are accepted for already frozen baseline comparisons and are hash
checked. All methods must have identical snapshot sets, demand, and oracle
denominators.

Copy `selection_manifest.template.json` to a new frozen manifest and replace
all placeholders. Compute hashes with `Get-FileHash -Algorithm SHA256`. Do not
alter that manifest after test evaluation starts.

Run the complete CPU pipeline:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe' `
  test_diff_path\final_strict2x_three_method\evaluate_manifest.py `
  --manifest <frozen-selection-manifest.json> `
  --output-dir output\comparisons\final_strict2x_three_method
```

The output includes per-method and combined rows, summary CSV/JSON, paired
bootstrap comparisons, provenance with hashes, and
`cdf_norm_fulfill_strict2x.png`.

The CDF has three shared-style facets and no subtitle. It uses a monotone PCHIP
interpolation through 41 evenly spaced empirical order-statistic anchors. This
removes sample-to-sample angular jitter only in the display layer: means,
percentiles, samples, and method ordering are computed from all raw rows. Axis
ranges begin with the paper-like High/Medium/Low scales and expand only if an
observation would otherwise be clipped. Every curve contains exact horizontal
CDF=0 and CDF=1 tails.

Replot an existing combined rows file without evaluating a model:

```powershell
& 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe' `
  test_diff_path\final_strict2x_three_method\plot_cdf.py `
  --rows <comparison_rows.csv> `
  --output <cdf.png> `
  --metadata <cdf_metadata.json>
```

Run the GPU-free synthetic end-to-end check:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe' `
  test_diff_path\final_strict2x_three_method\static_check.py
```
