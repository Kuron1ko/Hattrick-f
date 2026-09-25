# Hattrick-f: Level-3 MLU release

Hattrick-f starts from the best Level-3 checkpoint trained with the complete
six-objective Hattrick order:

`Fh -> Uh -> Fhm -> Uhm -> Fhml -> Uhml`.

The selected Hattrick-f experiment resets Adam, removes the three persistent
MLU objectives, and keeps the ordered projection:

`Fh -> Fhm -> Fhml`.

This keeps High continuously protected while removing `Uh`, `Uhm`, and `Uhml`
from the projection basis. All Hattrick parameters remain trainable. Routing
inference uses strict ESM predictions; actual traffic remains only in the
differentiable training/admission evaluator. Guarded release variants are kept
in the runner as diagnostic ablations, but are not the selected method.

Run the Level-3 screen with the repository's Python 3.12 environment:

```powershell
D:\kuroresearch\.venv-hattrick\Scripts\python.exe run_experiment.py --keep-fh --mlu-slack none --epochs 15
```
