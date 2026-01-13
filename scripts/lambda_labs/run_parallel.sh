#!/bin/bash
# =============================================================================
# Parallel experiment runner for 8x A100 node
# Runs 12 experiments (HC/mHC × 2 depths × 3 seeds) with staggered starts
#
# Usage:
#   bash scripts/lambda_labs/run_parallel.sh
#
# Estimated time: ~7-8 hours
# Estimated cost: ~$100-115 on 8x A100 ($14.32/hr)
# =============================================================================

set -e

# Activate venv
source venv/bin/activate

# Configuration
STEPS=5000
BATCH_SIZE=8
WANDB_PROJECT="mhc-part2"
RUN_DIR="runs_c4"
STAGGER_DELAY=15  # seconds between launches to avoid thundering herd

# Create output directories
mkdir -p $RUN_DIR
mkdir -p logs

echo "=============================================="
echo "mHC Part 2 - Parallel Experiments (8x A100)"
echo "=============================================="
echo "Started: $(date)"
echo "Steps: $STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Stagger delay: ${STAGGER_DELAY}s"
echo "Output: $RUN_DIR"
echo "=============================================="

# Verify GPUs
echo ""
echo "Checking GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -lt 8 ]; then
    echo "WARNING: Expected 8 GPUs, found $GPU_COUNT"
    echo "Continuing anyway..."
fi

# =============================================================================
# Batch 1: 8 runs in parallel
# =============================================================================
echo ""
echo "=============================================="
echo "BATCH 1: Starting 8 runs (staggered by ${STAGGER_DELAY}s each)"
echo "=============================================="

START_TIME=$(date +%s)

echo "[1/8] Launching hc_d32_s42 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python scripts/lambda_labs/train_c4.py \
    --connection hc --depth 32 --seed 42 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/hc_d32_s42.log 2>&1 &
sleep $STAGGER_DELAY

echo "[2/8] Launching hc_d32_s123 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python scripts/lambda_labs/train_c4.py \
    --connection hc --depth 32 --seed 123 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/hc_d32_s123.log 2>&1 &
sleep $STAGGER_DELAY

echo "[3/8] Launching hc_d48_s42 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python scripts/lambda_labs/train_c4.py \
    --connection hc --depth 48 --seed 42 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/hc_d48_s42.log 2>&1 &
sleep $STAGGER_DELAY

echo "[4/8] Launching hc_d48_s123 on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python scripts/lambda_labs/train_c4.py \
    --connection hc --depth 48 --seed 123 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/hc_d48_s123.log 2>&1 &
sleep $STAGGER_DELAY

echo "[5/8] Launching mhc_d32_s42 on GPU 4..."
CUDA_VISIBLE_DEVICES=4 python scripts/lambda_labs/train_c4.py \
    --connection mhc --depth 32 --seed 42 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/mhc_d32_s42.log 2>&1 &
sleep $STAGGER_DELAY

echo "[6/8] Launching mhc_d32_s123 on GPU 5..."
CUDA_VISIBLE_DEVICES=5 python scripts/lambda_labs/train_c4.py \
    --connection mhc --depth 32 --seed 123 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/mhc_d32_s123.log 2>&1 &
sleep $STAGGER_DELAY

echo "[7/8] Launching mhc_d48_s42 on GPU 6..."
CUDA_VISIBLE_DEVICES=6 python scripts/lambda_labs/train_c4.py \
    --connection mhc --depth 48 --seed 42 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/mhc_d48_s42.log 2>&1 &
sleep $STAGGER_DELAY

echo "[8/8] Launching mhc_d48_s123 on GPU 7..."
CUDA_VISIBLE_DEVICES=7 python scripts/lambda_labs/train_c4.py \
    --connection mhc --depth 48 --seed 123 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/mhc_d48_s123.log 2>&1 &

echo ""
echo "All 8 runs launched. Waiting for completion..."
echo "Monitor progress: tail -f logs/*.log"
echo "Or check W&B: https://wandb.ai"
echo ""

wait

BATCH1_END=$(date +%s)
BATCH1_DURATION=$((BATCH1_END - START_TIME))
echo ""
echo "=============================================="
echo "BATCH 1 COMPLETE in $((BATCH1_DURATION / 60)) minutes"
echo "=============================================="

# =============================================================================
# Batch 2: 4 runs (seed 456)
# =============================================================================
echo ""
echo "=============================================="
echo "BATCH 2: Starting 4 runs (seed 456)"
echo "=============================================="

BATCH2_START=$(date +%s)

echo "[1/4] Launching hc_d32_s456 on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python scripts/lambda_labs/train_c4.py \
    --connection hc --depth 32 --seed 456 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/hc_d32_s456.log 2>&1 &
sleep $STAGGER_DELAY

echo "[2/4] Launching hc_d48_s456 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python scripts/lambda_labs/train_c4.py \
    --connection hc --depth 48 --seed 456 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/hc_d48_s456.log 2>&1 &
sleep $STAGGER_DELAY

echo "[3/4] Launching mhc_d32_s456 on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python scripts/lambda_labs/train_c4.py \
    --connection mhc --depth 32 --seed 456 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/mhc_d32_s456.log 2>&1 &
sleep $STAGGER_DELAY

echo "[4/4] Launching mhc_d48_s456 on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python scripts/lambda_labs/train_c4.py \
    --connection mhc --depth 48 --seed 456 \
    --steps $STEPS --batch-size $BATCH_SIZE \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/mhc_d48_s456.log 2>&1 &

echo ""
echo "All 4 runs launched. Waiting for completion..."
wait

BATCH2_END=$(date +%s)
BATCH2_DURATION=$((BATCH2_END - BATCH2_START))

# =============================================================================
# Summary
# =============================================================================
TOTAL_DURATION=$((BATCH2_END - START_TIME))

echo ""
echo "=============================================="
echo "ALL 12 EXPERIMENTS COMPLETE!"
echo "=============================================="
echo "Batch 1 (8 runs): $((BATCH1_DURATION / 60)) minutes"
echo "Batch 2 (4 runs): $((BATCH2_DURATION / 60)) minutes"
echo "Total time: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m"
echo "Results in: $RUN_DIR"
echo ""
echo "Next steps:"
echo "  1. Generate figures: python scripts/lambda_labs/visualize_c4.py"
echo "  2. Download results: rsync -avz ubuntu@\$(hostname -I | awk '{print \$1}'):~/mhc-repro/runs_c4 ./"
echo "  3. Check W&B: https://wandb.ai"
echo ""
