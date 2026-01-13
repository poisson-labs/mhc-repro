# mHC Part 2: Large-Scale Training on Lambda Labs

This directory contains everything needed to run Part 2 experiments on Lambda Labs H100/A100.

## Quick Start

```bash
# 1. Launch Lambda Labs instance (A100 80GB recommended)
# 2. SSH into instance
ssh ubuntu@<instance-ip>

# 3. Clone repo
git clone https://github.com/YOUR_USERNAME/mhc-repro.git
cd mhc-repro

# 4. Run setup
bash scripts/lambda_labs/setup.sh

# 5. (Optional) Login to W&B for monitoring
wandb login

# 6. Run experiments
bash scripts/lambda_labs/run_experiments.sh
```

## Cost & Time Estimates

### Model Configurations

| Config | Layers | Dim | Params | Memory (bf16) |
|--------|--------|-----|--------|---------------|
| Depth 32 | 32 | 2048 | ~1.0B | ~20GB |
| Depth 48 | 48 | 2048 | ~1.5B | ~30GB |

### Runtime Estimates

| GPU | Experiment | Steps | Time | Tokens/sec |
|-----|------------|-------|------|------------|
| H100 80GB | Depth 32, single seed | 10,000 | ~20 min | ~160k |
| H100 80GB | Depth 48, single seed | 10,000 | ~30 min | ~110k |
| A100 80GB | Depth 32, single seed | 10,000 | ~45 min | ~70k |
| A100 80GB | Depth 48, single seed | 10,000 | ~70 min | ~45k |

### Full Experiment Matrix

**Quick mode (recommended for initial validation):**
- 4 runs: HC/mHC × depth 32/48 × seed 42

| GPU | Time | Cost |
|-----|------|------|
| H100 ($3.29/hr) | ~1.5-2 hrs | ~$6 |
| A100 ($1.50/hr) | ~4 hrs | ~$6 |

**Full mode (for publication):**
- 18 runs: residual/HC/mHC × depth 32/48 × seeds 42,123,456

| GPU | Time | Cost |
|-----|------|------|
| H100 ($3.29/hr) | ~6-8 hrs | ~$22-26 |
| A100 ($1.50/hr) | ~15-20 hrs | ~$25-30 |

**Recommendation:** H100 is faster and similar cost - use it if available.

## Scripts

| Script | Purpose |
|--------|---------|
| `setup.sh` | One-time instance setup (venv, deps, verification) |
| `train_c4.py` | Main training script with all bells and whistles |
| `run_experiments.sh` | Orchestrates full experiment matrix |
| `visualize_c4.py` | Generates publication figures |
| `requirements.txt` | Pinned dependencies for reproducibility |

## Training Script Options

```bash
# Single experiment
python scripts/lambda_labs/train_c4.py \
    --connection mhc \
    --depth 32 \
    --seed 42 \
    --steps 10000

# Multiple seeds
python scripts/lambda_labs/train_c4.py \
    --connection mhc \
    --depth 32 \
    --seeds 42,123,456

# All experiments (HC/mHC at both depths)
python scripts/lambda_labs/train_c4.py \
    --connection all \
    --depth all \
    --seeds 42,123,456

# Disable W&B logging
python scripts/lambda_labs/train_c4.py --no-wandb ...

# Disable gradient checkpointing (if memory allows)
python scripts/lambda_labs/train_c4.py --no-checkpoint ...
```

## Output Structure

```
runs_c4/
├── hc_d32_s42/
│   ├── history.json      # All training metrics
│   ├── config.json       # Full configuration
│   ├── best_model.pt     # Best validation checkpoint
│   ├── final_model.pt    # Final model
│   └── checkpoint_*.pt   # Periodic checkpoints
├── mhc_d32_s42/
│   └── ...
└── ...

figures/
├── main_comparison.png   # 6-panel HC vs mHC comparison
├── seed_variation.png    # Across-seed variance analysis
├── layer_amax_d32.png    # Per-layer Amax heatmaps
└── summary.json          # Numerical results
```

## What Makes This "Defensible"

1. **Multiple seeds** - Every claim backed by 3 seeds with error bars
2. **Baseline comparison** - Standard residual included as control
3. **Pinned dependencies** - Exact versions in requirements.txt
4. **Git hash logged** - Every run records commit hash
5. **W&B integration** - Real-time monitoring, shareable dashboards
6. **Comprehensive metrics** - Loss, gradient norms, per-layer Amax
7. **Publication-ready figures** - Matching Part 1 style

## Troubleshooting

**OOM errors:**
- Reduce batch size: `--batch-size 16`
- Ensure gradient checkpointing is on (default)

**Slow training:**
- Verify Flash Attention is detected (printed at start)
- Check GPU utilization with `nvidia-smi`

**C4 download issues:**
- Streaming should handle this, but if stuck, check network
- HuggingFace datasets may rate limit; wait and retry

## Links

- Part 1 blog post: https://poisson.run/notes/deepseek-mhc
- Original paper: arXiv 2512.24880
- Lambda Labs: https://lambdalabs.com
