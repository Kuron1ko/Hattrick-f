# Virtual Medium dual scout

This runner keeps the native serial Hattrick cascade and the complete ordered
six-objective training loop from `shared2x_full_objectives`.  It replaces the
uniform Medium-pressure preview with a low-cost differentiable scout:

1. route predicted Medium traffic with a soft feasible split;
2. form edge marginal prices as `utilization ** price_power`;
3. map prices to paths with the sparse path-edge matrix and take a soft
   best-response at the requested temperature;
4. repeat for the requested number of rounds (two by default), then expose the
   absolute path price and its within-OD centered value to the existing
   zero-initialized High adapters.

The scout is not an admission stage.  It uses the current ESM Medium prediction,
topology, capacities, paths, and masks only.  Actual traffic remains confined to
the inherited training loss and evaluation replay.

Each hyperparameter tuple has a separate output subtree:

```text
artifacts/steps_2__temperature_2__price_power_2/...
```

Run the one-snapshot feature and gradient smoke test:

```powershell
python smoke_test.py --snapshot 400 --steps 2 --temperature 2 --price-power 2
```

Start an experiment explicitly (the smoke test never starts training):

```powershell
python run_experiment.py --level 1 --seed 490 --steps 2 --temperature 2 --price-power 2
```
