# Robustness protocol amendment: capacity-fault audit sampling

This amendment was fixed before reading any capacity-fault outcome. It changes
the computational sampling plan, not the stress scenarios or the prediction
robustness tests.

## Why the amendment is necessary

The original Cartesian product would emit 4,860,000 capacity-fault row
dictionaries (180 scenarios × 1,000 snapshots × 3 seeds × 3 methods × 3
classes), plus 270,000 prediction-stress rows. Keeping the resulting 5,130,000
dictionaries in memory is not a viable or reliable formal evaluation design.

## Fixed capacity-fault audit subset

For each preregistered 500-snapshot window, capacity faults are evaluated at
the 20 uniformly spaced offsets

`0, 25, 50, ..., 475`.

Thus the near and far windows contribute 40 fixed time points in total. The
selection depends only on window position and was not chosen using any model
result. The audit still covers every one of the 36 physical links and all five
fault settings for each link:

- announced capacity factors `0.5` and `0.1`;
- unannounced capacity factors `0.5` and `0.1`;
- unannounced factor `0.01`, labeled near-outage rather than deletion.

Prediction robustness is unchanged: all four class-bias scenarios and all six
OD-shape-noise scenarios run on every snapshot in both full 500-snapshot
windows.

## Execution and formal-result rules

The evaluator streams raw rows to an atomic temporary CSV and computes counts
and capacity-safety maxima online. Prediction and capacity can be run as
explicit phases for diagnostics; only the default combined run can be a formal
result. A formal result additionally requires seeds `490, 491, 492`, all three
methods, both full windows, every fixed scenario, every expected snapshot and
class, no duplicate or unexpected row, and all 36 physical links.

`--smoke`, `--allow-missing`, any single-phase mode, or snapshot truncation is
non-formal. `--max-snapshots` is accepted only together with explicit
`--smoke`, so a truncated run cannot be mistaken for the formal result. When
the default output location is used, smoke and other non-formal invocations are
routed to `robustness_smoke` and `robustness_nonformal`, respectively; they do
not replace `robustness/snapshots.csv`. An eligible formal run also refuses to
publish that file unless its exact coverage audit is complete.
