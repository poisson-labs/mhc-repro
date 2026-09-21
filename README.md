# mhc-repro

[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Poisson Labs Research](https://img.shields.io/badge/Poisson%20Labs-Research-teal.svg)](https://poissonlabs.ai/research/mhc-reproduction-part-2/)

Code for a two-part reproduction of DeepSeek's [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880).
It trains one GPT architecture with three kinds of residual connection and logs how much each one amplifies the
signal.

- **Part 1**, 10M parameters on TinyShakespeare, on one machine:
  [DeepSeek's mHC: When Residual Connections Explode](https://poissonlabs.ai/research/mhc-reproduction/)
- **Part 2**, 1.7B and 2.5B parameters on C4, on an 8-GPU node:
  [10,924x: The Instability Bomb at 1.7B Scale](https://poissonlabs.ai/research/mhc-reproduction-part-2/)

## What it does

The model is a GPT whose residual connection is swappable:

| `--connection` | Connection |
|---|---|
| `residual` | Standard `x + F(x)`. |
| `hc` | Hyper-Connections: four parallel streams per layer, mixed by learned, input-dependent matrices. |
| `mhc` | The same, with the residual mixing matrix projected onto doubly stochastic matrices by 20 Sinkhorn-Knopp iterations. |

Every step logs the loss, the gradient norm and Amax, the largest absolute row or column sum of the residual
mixing matrix, per layer and composite. 1.0 means the connection does not amplify the signal.

## Install

Python 3.10 or newer.

```bash
git clone https://github.com/poisson-labs/mhc-repro.git
cd mhc-repro
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"       # add ",part2" for the Part 2 scripts
```

`pip install -e .` installs the dependencies. The code runs from the repository root.

## Run

### Tests

```bash
pytest
```

### Part 1: 10M parameters on TinyShakespeare

Trains the residual, HC and mHC models in turn on CUDA, Apple MPS or the CPU, whichever is present. The first run
downloads TinyShakespeare into `data/`. Each model writes its history and checkpoints to `runs/<connection>/`.

```bash
python -m src.train                                  # 5,000 steps each
python scripts/visualize.py                          # loss and Amax figures from runs/ into figures/
```

With `--connection` the run writes straight into `--run-dir`, with no `<connection>` subdirectory:

```bash
python -m src.train --connection mhc --steps 500 --run-dir runs/mhc
```

Also in `scripts/`:

- `critical_depth_sweep.py`: HC at depths 6, 8, 10, 12, 14, 16, 20 and 24, with the hidden size reduced as depth
  grows to keep the parameter count constant. Writes `runs/depth_sweep/` and `figures/critical_depth.png`.
- `stress_test.py --experiment {deep,lr,all}`: a 16-layer model, and a higher learning rate.
- `visualize_v4.py`, `visualize_v5.py`, `visualize_stress.py`: figures for the later Part 1 experiments.
- `python -m scripts.verify_normalization`: shows that the forward pass normalizes `H_post`, which cancels the
  factor 2 in Equation (8) of the paper.

### Part 2: 1.7B and 2.5B parameters on C4

Needs CUDA GPUs and `pip install -e ".[part2]"`. C4 is streamed, so nothing is downloaded up front. Details are
in [`scripts/lambda_labs/README.md`](scripts/lambda_labs/README.md).

```bash
bash scripts/lambda_labs/setup.sh                    # one-time node setup

# one run
python scripts/lambda_labs/train_c4.py --connection mhc --depth 32 --seed 42 --steps 5000 --batch-size 8

# the launch scripts: HC and mHC runs across 8 GPUs, then the stress runs
bash scripts/lambda_labs/run_parallel.sh
bash scripts/lambda_labs/run_stress_test.sh
```

Runs write `config.json` and `history.json` to `runs_c4/` and `runs_c4_stress/`. Then:

```bash
python scripts/lambda_labs/visualize_c4.py                           # figures from runs_c4/ into figures/
python scripts/lambda_labs/visualize_extended.py                     # scaling-law, stress-test and amplification figures, and the data for the post's animations
python scripts/lambda_labs/generate_samples.py                       # text samples from trained checkpoints
python scripts/index_runs.py                                         # writes manifests/part2-runs.csv
```

What was run for the post, when, and what this repository holds of it: [`docs/part2-run-record.md`](docs/part2-run-record.md).

## Layout

| Path | Contents |
|---|---|
| `src/` | The model, the three connection types, the Sinkhorn-Knopp projection and the Part 1 training loop. |
| `tests/` | Tests of the stream-persistence behaviour (`pytest`). |
| `scripts/` | Part 1 sweeps and figure scripts. |
| `scripts/lambda_labs/` | Part 2 training, launch and figure scripts. |
| `manifests/` | The Part 2 run manifest. |
| `notebooks/` | A Colab notebook for a 300M-parameter variant. It is stored without outputs and is not what produced the Part 2 results. |
| `docs/` | The Part 2 run record and the repository standard, [`docs/repo-hygiene.md`](docs/repo-hygiene.md). |

Kept for reference and no longer used: `src/model_v3_backup.py` and `src/connections_v3_backup.py` (the
implementation before the stream-persistence fix; nothing imports them), and `scripts/verify_math.py` and
`scripts/validate_implementation.py` (written for that earlier interface; they no longer run).

Run outputs (`runs/`, `runs_c4/`, `runs_c4_stress/`, `figures/`, checkpoints) are written locally and are
git-ignored.

## Licence

MIT. See [`LICENSE`](LICENSE). Third-party software, datasets and tokenizer data, and their terms, are in
[`NOTICE`](NOTICE).

## Citation

```bibtex
@misc{kolasinski2026mhcrepro1,
  author = {Kolasinski, Taylor},
  title  = {DeepSeek's mHC: When Residual Connections Explode},
  year   = {2026},
  month  = {January},
  howpublished = {\url{https://poissonlabs.ai/research/mhc-reproduction/}},
  note   = {Poisson Labs research report; source code at \url{https://github.com/poisson-labs/mhc-repro}}
}

@misc{kolasinski2026mhcrepro2,
  author = {Kolasinski, Taylor},
  title  = {10,924x: The Instability Bomb at 1.7B Scale},
  year   = {2026},
  month  = {January},
  howpublished = {\url{https://poissonlabs.ai/research/mhc-reproduction-part-2/}},
  note   = {Poisson Labs research report; source code at \url{https://github.com/poisson-labs/mhc-repro}}
}
```
