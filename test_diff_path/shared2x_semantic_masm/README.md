# Shared-2x semantic MASM

This is an independent strict-ESM experiment runner. It preserves native
Hattrick's serial `High -> Medium -> Low` cascade and projection mechanism,
while exposing Medium alternative-set structure to High before Stage 1.

Each Medium OD's eight route candidates receive an eight-dimensional fixed
semantic anchor: feasibility, inverse bottleneck/mean capacity, common-edge
overlap, no-clean/clean alternative, diversity mass, and path length. A tiny
two-head permutation-equivariant set encoder is a bounded residual around
those anchors. Predicted Medium demand carries the tokens through PTE to edge
sum and fixed-semantic tail aggregates. High paths receive edge mean and max
for both aggregates, centered within each High OD. The native uniform Medium
pressure maximum/mean remain two absolute anchors.

Only `tm2_pred`, topology/path embeddings, PTE, capacities, and feasibility
masks enter this preview. Formal Stage 2 is unchanged and runs only after High.
Both High adapters have exact-zero output layers, giving bitwise native policy
at epoch 0 after a native state load. Training uses the inherited complete six
objectives and ordered gradient projection.

CPU-only smoke:

```powershell
$env:CUDA_VISIBLE_DEVICES=''
& 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe' `
  test_diff_path\shared2x_semantic_masm\smoke_test.py
```

Training entry (not run by the smoke):

```powershell
& 'D:\kuroresearch\.venv-hattrick\Scripts\python.exe' `
  test_diff_path\shared2x_semantic_masm\run_experiment.py --level 1 --seed 490
```
