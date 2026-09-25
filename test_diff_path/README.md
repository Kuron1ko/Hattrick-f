# test_diff_path

This folder contains the 100-slice GEANT experiments for testing Hattrick under non-KSP and class-specific path constraints.

Run from the repository root:

```powershell
D:\kuroresearch\.venv-hattrick\Scripts\python.exe test_diff_path\run_diff_path_experiment.py --stage all
```

Stages are resumable:

- `prepare`: copy GEANT slices 0-99 into topology aliases and generate custom paths/masks.
- `gurobi`: compute oracle, BEST_MC/Flexile, and SWAN for the new path sets.
- `hattrick`: run zero-shot and retrained Hattrick tests.
- `report`: generate CSV, Markdown, CDF, and boxplot artifacts.

The experiment does not move or overwrite the original GEANT data.
