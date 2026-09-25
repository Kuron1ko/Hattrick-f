# Round 2: High hard-freeze + full Medium/Low hinge

This directory implements the corrected mechanism:

1. Reuse an unchanged-Hattrick phase-A checkpoint only after validation High
   `NormFulFill >= 0.98` for two consecutive epochs.
2. Copy that checkpoint into a frozen teacher and a trainable student.
3. Use the teacher's High path policy and the student's Medium/Low policies in
   one common sequential actual-TM admission pass.
4. Optimize an order penalty without detaching Low or Medium.

High is therefore frozen structurally. It is not merely detached from a loss,
and it cannot drift through shared student parameters. The inference call
signature remains the Hattrick signature; the checkpoint contains both
teacher and student branches.

The Level-2 phase-A checkpoint is deliberately reused to avoid repeating the
expensive warm-up. Every run records its path and SHA-256. Level 1 is a
correctness run only because that warm-start has already seen the wider
Level-2 window.

```powershell
python -m unittest test_diff_path\shared2x_order_regularizer\round2_high_frozen\test_round2.py
python test_diff_path\shared2x_order_regularizer\round2_high_frozen\run_round2.py --level 0 --lambda-multiplier 1 --seed 490
python test_diff_path\shared2x_order_regularizer\round2_high_frozen\run_round2.py --level 1 --lambda-multiplier 1 --seed 490
python test_diff_path\shared2x_order_regularizer\round2_high_frozen\run_round2.py --level 2 --lambda-multiplier 0 --seed 490
```

The implemented penalty variants are:

- `full_hinge`: `mean(ReLU(N_low - N_mid))` with full gradients.
- `flow_balanced_hinge`: the same forward value, with a nonzero Low gradient
  scaled by `oracle_low/oracle_medium` to balance raw-flow derivatives.
- `tail_flow_balanced_squared`: top-25% squared positive gaps with the same
  nonzero flow-balanced Low gradient.

Level 2 compares candidates against the matched hard-freeze `lambda=0`
baseline. No result in this directory overwrites round 1 source or artifacts.
