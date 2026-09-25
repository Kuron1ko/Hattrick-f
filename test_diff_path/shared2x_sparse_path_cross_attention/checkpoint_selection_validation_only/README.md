# Level-4 validation-only checkpoint watcher

This watcher never loads snapshots or evaluation artifacts from 400–499.  It
reads only:

- the active run's `train_history.csv`;
- `validation_epoch_XXX_{summary,metrics}` for validation snapshots 350–399;
- the rolling `final_model.pt` and `best_model.pt`;
- native six-objective Hattrick's epoch-60 validation metrics on 350–399.

The training runner does **not** archive every epoch: `final_model.pt` is
overwritten and `best_model.pt` only preserves the runner's High-first choice.
The watcher therefore copies each stable, epoch-matched rolling checkpoint into
its own immutable `archive/` directory.  It does not modify the training run.

Selection first enforces High mean, capacity, and disabled-flow safety.  Its
first-choice Low tier requires the paired Low mean delta to be at least -0.005
and the upper endpoint of the two-sided 95% confidence interval to be at least
zero (no statistically significant negative shift was detected).  Only when
that entire tier is empty does it use the predeclared -0.01 non-inferiority
fallback, which requires both the mean delta and its one-sided 95% lower bound
to be at least -0.01.  The old -0.02 margin is not a safety tier.

Within the active Low tier, selection prefers candidates satisfying High P1/P10
tail gates, constructs a Pareto front over Medium mean/P1/P10, and chooses the
robust maximin point on that front.  Full constants and tie-breaks are frozen in
`selection_policy.json`.

At epoch 60, the watcher waits for the matching validation history, validation
files, and epoch-60 rolling checkpoint, then writes
`manifest_frozen_epoch_060.json` and `selected_checkpoint.pt` and exits.  This
happens without consulting any test result.

Manual invocation:

```powershell
python watch_and_select.py --watch --poll-seconds 0.5 --target-epoch 60
```
