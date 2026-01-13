#!/bin/bash
# =============================================================================
# Stress test: Push HC until it breaks
# Run AFTER main experiments complete
#
# This cranks the learning rate to 3x and watches HC explode/NaN
# while mHC survives. The "oh shit" moment for the blog.
#
# Usage:
#   bash scripts/lambda_labs/run_stress_test.sh
#
# Estimated time: ~30-60 min total
# =============================================================================

source venv/bin/activate

STEPS=2000
BATCH_SIZE=8
LR="3e-4"  # 3x the normal learning rate
RUN_DIR="runs_c4_stress"
WANDB_PROJECT="mhc-part2-stress"

mkdir -p $RUN_DIR

echo "=============================================="
echo "STRESS TEST: HC vs mHC at 3x Learning Rate"
echo "=============================================="
echo "LR: $LR (3x normal)"
echo "Steps: $STEPS"
echo "This will likely kill HC. That's the point."
echo "=============================================="

# Run HC with aggressive LR
echo ""
echo "Starting HC stress test (GPU 0)..."
CUDA_VISIBLE_DEVICES=0 python scripts/lambda_labs/train_c4.py \
    --connection hc \
    --depth 32 \
    --seed 42 \
    --steps $STEPS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --run-dir $RUN_DIR \
    --wandb-project $WANDB_PROJECT \
    2>&1 | tee logs/stress_hc.log &

# Run mHC with same aggressive LR
echo "Starting mHC stress test (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 python scripts/lambda_labs/train_c4.py \
    --connection mhc \
    --depth 32 \
    --seed 42 \
    --steps $STEPS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --run-dir $RUN_DIR \
    --wandb-project $WANDB_PROJECT \
    2>&1 | tee logs/stress_mhc.log &

wait

echo ""
echo "=============================================="
echo "STRESS TEST COMPLETE"
echo "=============================================="
echo ""
echo "Check the results:"
echo "  - HC likely crashed or has NaN loss"
echo "  - mHC likely survived"
echo ""
echo "This is the 'oh shit' moment for the blog post."
echo "Screenshot the W&B charts!"
