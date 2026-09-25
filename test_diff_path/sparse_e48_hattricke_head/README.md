# Fixed sparse-e48 + Hattrick-e head evaluation

This is a single, non-tuned hybrid replay:

- High backbone: validation-only selected sparse cross-attention epoch 48.
- Medium/Low post-head: the existing frozen Hattrick-e edge-toll checkpoint.
- Window: snapshots 400–499, load 2x.
- Inference: strict ESM; literal-zero actual TMs enter policy inference.

The frozen head accepts seven edge channels recomputed from the current sparse
High/Medium/Low policies and ESM demand, plus its learned six-channel edge
embedding. It emits only Medium and Low tolls. High remains the exact same
tensor, and the evaluator requires every High metric row to remain exactly
equal after sequential admission.

The script contains no epoch/head/scale search. Hashes of both checkpoints,
the validation-only sparse selection manifest, and both architecture runners
are hard-coded and verified before evaluation.

```powershell
& 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe' `
  test_diff_path\sparse_e48_hattricke_head\evaluate_fixed_hybrid.py `
  --device cuda
```
