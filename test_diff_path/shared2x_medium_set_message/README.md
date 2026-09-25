# Shared-2x Medium candidate-set message

This candidate keeps the native Hattrick `High -> Medium -> Low` cascade and
the complete six-objective ordered projection.  Before the High stage it sends
the full Medium candidate set through a small DeepSets message path:

1. encode each Medium candidate from its topology path embedding, strict ESM
   Medium demand, and path inverse-capacity max/mean;
2. contextualize the eight candidates of each OD with `(token, OD mean,
   token - mean)`;
3. retain both an explicit uniform message and a zero-logit learned-gate
   message;
4. scatter demand-weighted vectors to edges, normalize by capacity, and gather
   them back to every High candidate path;
5. combine the 16 vector features with the original two uniform-pressure
   scalars through independent zero-output High init/RAU adapter branches.

Actual traffic is never a policy input.  Epoch-0 policy is exactly native
Hattrick because both adapter branches have zero output layers.

CPU verification:

```powershell
python test_diff_path/shared2x_medium_set_message/smoke_test.py --snapshot 350
```

Run an isolated experiment:

```powershell
python test_diff_path/shared2x_medium_set_message/run_experiment.py --level 1 --seed 490
```

Artifacts are written only below
`test_diff_path/shared2x_medium_set_message/artifacts/`.
