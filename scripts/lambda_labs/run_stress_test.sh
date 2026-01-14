#!/bin/bash
# =============================================================================
# Stress test: Push HC until it breaks
# Run AFTER main experiments complete
#
# This cranks the learning rate to 3x/5x and watches HC explode/NaN
# while mHC survives. The "oh shit" moment for the blog.
#
# Usage:
#   bash scripts/lambda_labs/run_stress_test.sh
#
# Estimated time: ~30-45 min total (7 runs in parallel)
# =============================================================================

source venv/bin/activate

STEPS=3000
RUN_DIR="runs_c4_stress"
WANDB_PROJECT="mhc-part2-stress"

mkdir -p $RUN_DIR
mkdir -p logs

echo "=============================================="
echo "STRESS TEST: HC vs mHC at Aggressive LR"
echo "=============================================="
echo "7 runs in parallel (skipping GPU 3 - bad hardware)"
echo "Steps: $STEPS"
echo "=============================================="
echo ""
echo "3x LR (3e-4): d32, d48"
echo "5x LR (5e-4): d32"
echo "3x LR (3e-4): d64 (pushing depth)"
echo "=============================================="

# 3x LR runs
echo ""
echo "[1/7] HC d32 @ 3x LR (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 python scripts/lambda_labs/train_c4.py \
    --connection hc --depth 32 --seed 42 \
    --steps $STEPS --batch-size 8 --lr 3e-4 \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/stress_hc_d32_3x.log 2>&1 &

echo "[2/7] mHC d32 @ 3x LR (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 python scripts/lambda_labs/train_c4.py \
    --connection mhc --depth 32 --seed 42 \
    --steps $STEPS --batch-size 8 --lr 3e-4 \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/stress_mhc_d32_3x.log 2>&1 &

echo "[3/7] HC d48 @ 3x LR (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 python scripts/lambda_labs/train_c4.py \
    --connection hc --depth 48 --seed 42 \
    --steps $STEPS --batch-size 4 --lr 3e-4 \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/stress_hc_d48_3x.log 2>&1 &

echo "[4/7] mHC d48 @ 3x LR (GPU 4)..."
CUDA_VISIBLE_DEVICES=4 python scripts/lambda_labs/train_c4.py \
    --connection mhc --depth 48 --seed 42 \
    --steps $STEPS --batch-size 4 --lr 3e-4 \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/stress_mhc_d48_3x.log 2>&1 &

# 5x LR runs
echo "[5/7] HC d32 @ 5x LR (GPU 5)..."
CUDA_VISIBLE_DEVICES=5 python scripts/lambda_labs/train_c4.py \
    --connection hc --depth 32 --seed 42 \
    --steps $STEPS --batch-size 8 --lr 5e-4 \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/stress_hc_d32_5x.log 2>&1 &

echo "[6/7] mHC d32 @ 5x LR (GPU 6)..."
CUDA_VISIBLE_DEVICES=6 python scripts/lambda_labs/train_c4.py \
    --connection mhc --depth 32 --seed 42 \
    --steps $STEPS --batch-size 8 --lr 5e-4 \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/stress_mhc_d32_5x.log 2>&1 &

# Push depth - d64 at batch 2 (might OOM, worth trying)
echo "[7/7] HC d64 @ 3x LR (GPU 7) - pushing depth..."
CUDA_VISIBLE_DEVICES=7 python scripts/lambda_labs/train_c4.py \
    --connection hc --depth 64 --seed 42 \
    --steps $STEPS --batch-size 2 --lr 3e-4 \
    --run-dir $RUN_DIR --wandb-project $WANDB_PROJECT \
    > logs/stress_hc_d64_3x.log 2>&1 &

echo ""
echo "All 7 stress tests launched!"
echo "Monitor: tail -f logs/stress_*.log"
echo "W&B: https://wandb.ai (project: mhc-part2-stress)"
echo ""

wait

echo ""
echo "=============================================="
echo "STRESS TEST COMPLETE"
echo "=============================================="
echo ""
echo "Check results:"
echo "  - HC runs likely crashed or show NaN loss"
echo "  - mHC runs likely survived"
echo "  - Deeper models (d48, d64) should fail faster"
echo ""
echo "This is the 'oh shit' moment for the blog post."
echo "Screenshot the W&B charts!"
