# Hattrick-f final validation protocol

This protocol was fixed before reading any Hattrick-f outcome beyond snapshot 499.

## Primary scope

- Topology/path set: GEANT shared 8sp.
- Load: 2x.
- Training seeds: 490, 491, 492.
- Phase A: 60 epochs with the complete six-objective Hattrick order
  `Fh -> Uh -> Fhm -> Uhm -> Fhml -> Uhml`.
- Hattrick-f continuation: reset Adam, keep all parameters trainable, 30 epochs
  with `Fh -> Fhm -> Fhml`.
- Equal-budget control: reset Adam from the same Phase-A checkpoint, keep all
  parameters trainable, 30 epochs with all six objectives.
- To balance sustained-GPU thermal drift, continuation jobs are executed in the
  fixed alternating order `H-f/490, control/491, H-f/492, control/490,
  H-f/491, control/492`.  This order was recorded before any of these six
  continuation outcomes were observed.
- Checkpoint selection: snapshots 350--399 only, using the same High per-snapshot,
  Low-mean, capacity, and disabled-flow gate as the existing Hattrick-f run.
- Snapshots 400--499 are a previously viewed confirmation window, not a blind test.

## Method-new temporal holdouts

The original `geant` trace has 10,773 snapshots. Its topology, pairs, shared
8-path incidence matrix, and the first 500 class TMs are identical to the
`geant_priomask500_shared` dataset used by Hattrick-f. Hattrick-f did not read
snapshots after 499 during training, validation, selection, or its existing
Level-4 evaluation.

Two windows are fixed in advance:

- near holdout: indices `[500, 1000)` (files t501--t1000);
- far holdout: indices `[9000, 9500)` (files t9001--t9500).

These windows are new to Hattrick-f, although the repository's earlier full
GEANT reproduction has processed them. They are therefore method-new temporal
holdouts, not project-wide untouched data.

For 2x evaluation, actual and ESM-predicted TMs are both multiplied by 2 at
runtime. The 1x oracle files are not used as 2x NormFulFill denominators. Primary
holdout metrics are raw class FulfillRatio, admitted traffic, capacity safety,
and paired method differences. A 2x oracle may be reported only after an LP
implementation reproduces the existing 2x oracle values on a locked audit set.

## Prediction-error stress tests

No checkpoint is selected with stress-test outcomes. On both holdout windows,
the actual 2x traffic is fixed while the ESM inputs supplied to the routing
policy are globally multiplied by `0.8, 0.9, 1.0, 1.1, 1.2`. Policy generation
must not consume the actual traffic. Sequential admission then replays the
unmodified actual traffic.

Additional fixed tests, after the global-bias implementation passes its strict
input audit:

- High-only ESM bias: `0.8, 1.2`;
- Medium-only ESM bias: `0.8, 1.2`;
- OD-shape lognormal noise: sigma `0.1, 0.2`, fixed seeds
  `20260825, 20260826, 20260827` and demand-weighted renormalization per class
  and snapshot.

## Capacity-fault stress tests

Capacity faults are reported with raw FulfillRatio because the nominal oracle
is invalid after a capacity change. Directed edges are grouped into physical
bidirectional links. All physical links are tested with capacity factors `0.5`
and `0.1`; `0.01` is explicitly labeled a near-outage rather than an exact link
deletion. Announced and unannounced derating are separated. Fixed KSP paths are
not recomputed, so this measures capacity-fault recovery within the deployed
path set.

## Statistical reporting

- Report every seed separately and the across-seed mean/range.
- Use paired per-snapshot differences.
- Report ordinary paired bootstrap only as a secondary check.
- Primary time-series uncertainty uses moving-block bootstrap with block lengths
  5 and 12 snapshots.
- Report High minimum and the fraction below 0.995 wherever a valid 2x oracle is
  available; otherwise report raw FulfillRatio without relabeling it NormFulFill.

## Latency reporting

A formal latency comparison resolves seeds 490, 491, and 492 from complete
current artifacts, either from the protocol's artifact roots or from three
explicitly supplied artifact roots. The automatic resolver may inspect the
shared runner's historical `level4_confirmation` output directory because
current frozen-source runs for seeds 491/492 were written there, but directory
location grants no trust: it accepts an artifact only when the frozen current
source hashes and completion binding match. An older artifact in that same
directory (including the old seed-490 run) is rejected. For every seed, the
benchmark requires Phase A, Hattrick-f, and the equal-budget six-loss continuation; it
checks the config seed and frozen-source hashes, the selected-checkpoint hash
recorded by `complete.json`, the checkpoint's embedded config, and that both
continuations name the exact same Phase-A checkpoint SHA-256. An explicitly
supplied single checkpoint trio is labeled `single_seed` and is never a formal
three-seed result.

Latency means warm-cache, steady-state policy-forward time from strict ESM
predictions. It includes device synchronization around each timed forward. It
does not include traffic/dataset I/O, checkpoint loading, the first static
topology-cache construction, sequential admission, or oracle/metric work. The
latency JSON records explicit booleans for cold-cache, admission, and end-to-end
coverage so this number cannot be silently presented as cold-start or complete
request latency.

## Frozen source hashes

- `frameworks/hattrick_system.py`: `4f7b15dba81b23f0e23d432e0b65f8ae895b459070dd8249a22c1345b15eb862`
- `utils/training_utils.py`: `b4b8c08ef353194ae107fc6756bde6f7668f12e5d6e87bf5f26280dbf4057185`
- `shared2x_full_objectives/run_experiment.py`: `a77c112d25f549e2cbe73b9a41f4ef4cebe10b60061e8683f9faebd86b112a42`
- `shared2x_full_objectives/ordered_projection.py`: `a8ed960f641e9cb0786ab8edee3b4a3c571255c3e76c939f0d6803ce01543e3a`
- `shared2x_hattrick_f/run_experiment.py`: `33d9d4d634d6661e935ba5800a90df4b2f5eaeeb0a42fb4465c1528b89025d18`
- `shared2x_hattrick_f/run_level4.py`: `dc13339b20a28ae9d6520e5466cd4bc5f0c7ec5ec704d5aabcf072884cb6d8d9`
- `shared2x_order_regularizer/run_experiment.py`: `317d0b796773963352a5b92652e6273074707bad42b3078a734e569366789310`
- `utils/AdamOptimizer.py`: `1dbb87b114b3845f86718a401e1f4ad2acd09a29d4707706505a35327f3e0325`
- `utils/robust_proj_utils.py`: `76affbd2e7f27863d6371794eee834fb8549558a3156aec40e1e06227cea0c7e`
- `utils/snapshot_utils.py`: `657e245fb127c03c29f123f637604fe3f21ced266e29993ed69fb30cff7010ad`
- `utils/cluster_utils.py`: `f9ccc1af7918cdd0b739bf9cfa6b492a4120bc237ae89c3f485dcd87d1a69df4`
- `utils/build_dataset_within_cluster.py`: `53c69ca0f24b8d514252dabd58192a3eee1b5584257065e4e1f460a85037be46`
- `build_frozen_manifest.py`: `2bb39dda55651dcdab82e4e619931da83bfc1fe4862fb7bd59c8f0afdf12a8a6`

`frozen_manifest.json` has SHA-256
`d1c1e9868ca3e7272502f2b90525d3ce80ad0900b3cdcec0362f0d2916de639f`.
It makes full-file SHA-256 commitments to the holdout and training topology,
pairs, 8sp path/PTE source, `paths_dict`, padded path-edge tensors, edge-ID
dictionaries, and filename indexes. The near/far actual and ESM traffic windows
use a deliberately bounded audit instead: for each class/kind/window the
verifier checks the 500-file window count, the directory's total numbered-file
count, and hashes offsets 0, 125, 250, 375, and 499 (therefore including the
first and last files). This is an integrity tripwire and **not** a full-content
cryptographic commitment to every traffic-matrix file in either window.
