# Part 2 run record

What ran for the [Part 2 post](https://poissonlabs.ai/research/mhc-reproduction-part-2/), when, and what
this repository does and does not hold of it. The numbers in the tables come from
[`manifests/part2-runs.csv`](../manifests/part2-runs.csv), which
[`scripts/index_runs.py`](../scripts/index_runs.py) writes from the run directories.

## The runs

| Set | Directory | Runs | Settings |
|---|---|---|---|
| Main | `runs_c4/` | 18 | Residual, HC and mHC, at depth 32 and 48, seeds 42, 123 and 456. 5,000 steps, learning rate 1e-4. Batch size 8, except HC and mHC at depth 48 (batch size 4). |
| Stress | `runs_c4_stress/` | 5 | HC at depth 32, 48 and 64, and mHC at depth 32 and 48, seed 42. 3,000 steps, learning rate 3e-4. Batch size 8, 4 and 2 at depth 32, 48 and 64. |

[`run_stress_test.sh`](../scripts/lambda_labs/run_stress_test.sh) lists seven stress runs: these five plus HC and
mHC at depth 32 with learning rate 5e-4. The local copy has no run directory for those two, so the manifest has
five stress rows.

The runs used one 8-GPU node, one GPU per run. The post describes it as 8×H100 SXM5 from Lambda Labs.
`run_stress_test.sh` skips GPU 3, which it notes as bad hardware.

## When they ran, and for how long

| | First start (UTC) | Last finish (UTC) | Span |
|---|---|---|---|
| 18 main runs | 2026-01-13 18:56:31 | 2026-01-14 04:38:28 | 9.70 h |
| 5 stress runs | 2026-01-14 04:42:40 | 2026-01-14 07:57:59 | 3.26 h |
| All 23 runs | 2026-01-13 18:56:31 | 2026-01-14 07:57:59 | 13.02 h |

The span is the time with runs on the node. The runs themselves add up to 58.1 GPU-hours (main) and 13.9
GPU-hours (stress), the sum of the `duration_h` column.

How the times were taken:

- A start time is the `timestamp` field of the run's `config.json`, the node's clock, read as UTC. That fits:
  the first main run starts 7.7 minutes after the commit it recorded was made, and the first stress run 61.1
  minutes after its commit. Read as US Eastern time instead, every run would finish before it started.
- A finish time is the modification time of `config.json`. The training script writes `config.json` and
  `history.json` together when a run ends. It is a file time in the local copy, so it is only as good as
  that copy.

When the node was rented and released is not recorded here. The provider's billing history is the record of
that. The spans above are a lower bound on it.

## Commit hashes recorded in the runs

Each `config.json` records the commit the run started from: `53679605` for the 18 main runs and `4226a4f1` for
the stress runs. This repository's history was rewritten before it was made public, which changed every commit
hash. The rewrite also dropped one file, which no code reads, from every tree that held it, and with it the one
commit whose only change was to add that file. `4226a4f1` is that commit: it left the code exactly as the commit
before it, so the code the stress runs started from is the code at `f3a4abd`. In this history those commits are:

| Recorded by the runs | Commit here |
|---|---|
| `53679605` (main runs) | `380aa08` |
| `4226a4f1` (stress runs) | `f3a4abd`, whose code that commit did not change |

## What this repository holds of the run data

It holds the manifest. It does not hold the run outputs: `history.json` files (23 to 48 MB each, 567 MB in the
local copy) and checkpoints are git-ignored. The Part 2 post links the Weights & Biases projects `mhc-part2`
and `mhc-part2-stress`, which it describes as holding the configurations, metrics and system logs of every run.

The manifest lists the SHA-256 of every `history.json` it was computed from, so a copy can be checked against
it.

## Gaps in the local copy

- Six main runs, HC and mHC at depth 48 with three seeds each, have `config.json` and no `history.json`.
  Their manifest rows have no history columns, and nothing that needs those histories can be recomputed
  from this repository: the depth 48 HC and mHC figures in the post, and the per-layer heatmap data.
- The stress run `mhc_d32_s42` has a complete 3,000-record `history.json` followed by 13,524 further bytes.
  `index_runs.py` reads the first JSON value. The loader in `visualize_extended.py`, the script that reads the
  stress directory, rejects the file as corrupt and skips the run.

## What the manifest reproduces

For the groups whose histories are in the local copy, the manifest gives the values
`scripts/lambda_labs/visualize_c4.py` prints as its summary table: final loss is the loss at the last step, peak
Amax is the maximum of `composite_amax` over all steps, and the spread is the population standard deviation
(`numpy.std`) over the three seeds.

| Method | Depth | Runs with history | Final loss | Peak Amax |
|---|---|---|---|---|
| Residual | 32 | 3 of 3 | 5.45 ± 0.04 | n/a |
| Residual | 48 | 3 of 3 | 5.48 ± 0.04 | n/a |
| HC | 32 | 3 of 3 | 5.43 ± 0.03 | 10,924 ± 3,247 |
| HC | 48 | 0 of 3 | not in this copy | not in this copy |
| mHC | 32 | 3 of 3 | 5.45 ± 0.03 | 1.00 ± 0.00 |
| mHC | 48 | 0 of 3 | not in this copy | not in this copy |

HC at depth 32 peaks at 6,545, 11,919 and 14,308 for seeds 42, 123 and 456. The stress runs with a history peak
at 5,445 (HC, depth 32), 3,814 (HC, depth 48) and 14,765 (HC, depth 64), and at 1.00 for mHC at depth 32 and 48.
