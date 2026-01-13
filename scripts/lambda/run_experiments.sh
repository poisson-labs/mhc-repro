#!/bin/bash
# =============================================================================
# Run All Part 2 Experiments
# mHC reproduction: https://poisson.run/notes/deepseek-mhc
#
# This script runs the full experiment matrix:
#   - 3 methods: residual (baseline), HC, mHC
#   - 2 depths: 32, 48 layers
#   - 3 seeds: 42, 123, 456
#   = 18 total experiments
#
# Estimated runtime: 15-20 hours on A100 80GB
# Estimated cost: ~$25-30 at Lambda Labs rates
#
# Usage:
#   bash scripts/lambda/run_experiments.sh           # Run all 18 experiments
#   bash scripts/lambda/run_experiments.sh --quick   # Run core 4 (HC/mHC at 32/48)
#   bash scripts/lambda/run_experiments.sh --test    # Quick test (1 seed each)
# =============================================================================

set -e  # Exit on error

# Parse arguments
MODE="full"
if [[ "$1" == "--quick" ]]; then
    MODE="quick"
    echo "Running in QUICK mode (4 experiments: HC/mHC at depth 32/48, seed 42)"
elif [[ "$1" == "--test" ]]; then
    MODE="test"
    echo "Running in TEST mode (6 experiments: all methods, all depths, seed 42)"
fi

# Configuration
STEPS=10000
BATCH_SIZE=32
WANDB_PROJECT="mhc-part2"
RUN_DIR="runs_c4"
LOG_DIR="logs"

# Create directories
mkdir -p $RUN_DIR
mkdir -p $LOG_DIR

# Record start time
START_TIME=$(date +%s)
echo "=============================================="
echo "mHC Part 2 Experiments"
echo "Started: $(date)"
echo "Mode: $MODE"
echo "Steps: $STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Output: $RUN_DIR"
echo "=============================================="

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Define experiment configurations based on mode
if [[ "$MODE" == "full" ]]; then
    # Full experiment matrix: 3 methods x 2 depths x 3 seeds = 18 runs
    METHODS=("residual" "hc" "mhc")
    DEPTHS=("32" "48")
    SEEDS=("42" "123" "456")
elif [[ "$MODE" == "quick" ]]; then
    # Core experiments: 2 methods x 2 depths x 1 seed = 4 runs
    METHODS=("hc" "mhc")
    DEPTHS=("32" "48")
    SEEDS=("42")
elif [[ "$MODE" == "test" ]]; then
    # Test run: 3 methods x 2 depths x 1 seed = 6 runs (but shorter)
    METHODS=("residual" "hc" "mhc")
    DEPTHS=("32" "48")
    SEEDS=("42")
    STEPS=1000  # Shorter for testing
fi

TOTAL_EXPERIMENTS=$(( ${#METHODS[@]} * ${#DEPTHS[@]} * ${#SEEDS[@]} ))
EXPERIMENT_NUM=0

echo ""
echo "Running $TOTAL_EXPERIMENTS experiments..."
echo ""

# Run experiments
for METHOD in "${METHODS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
            EXP_NAME="${METHOD}_d${DEPTH}_s${SEED}"
            LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

            echo "----------------------------------------------"
            echo "[$EXPERIMENT_NUM/$TOTAL_EXPERIMENTS] $EXP_NAME"
            echo "  Method: $METHOD"
            echo "  Depth: $DEPTH layers"
            echo "  Seed: $SEED"
            echo "  Log: $LOG_FILE"
            echo "----------------------------------------------"

            # Check if already completed
            if [ -f "${RUN_DIR}/${EXP_NAME}/history.json" ]; then
                echo "  SKIPPED (already completed)"
                continue
            fi

            # Run experiment
            TRAIN_START=$(date +%s)

            python scripts/lambda/train_c4.py \
                --connection $METHOD \
                --depth $DEPTH \
                --seed $SEED \
                --steps $STEPS \
                --batch-size $BATCH_SIZE \
                --run-dir $RUN_DIR \
                --wandb-project $WANDB_PROJECT \
                2>&1 | tee $LOG_FILE

            TRAIN_END=$(date +%s)
            TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
            echo ""
            echo "  Completed in $((TRAIN_DURATION / 60)) minutes"
            echo ""
        done
    done
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
echo "Total time: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m"
echo "Results in: $RUN_DIR"
echo ""

# Generate summary
echo "Experiment Summary:"
echo "-------------------"
for METHOD in "${METHODS[@]}"; do
    for DEPTH in "${DEPTHS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            EXP_NAME="${METHOD}_d${DEPTH}_s${SEED}"
            HISTORY_FILE="${RUN_DIR}/${EXP_NAME}/history.json"

            if [ -f "$HISTORY_FILE" ]; then
                # Extract final loss from history
                FINAL_LOSS=$(python3 -c "
import json
with open('$HISTORY_FILE') as f:
    h = json.load(f)
    print(f'{h[-1][\"loss\"]:.4f}')
" 2>/dev/null || echo "N/A")
                echo "  $EXP_NAME: loss=$FINAL_LOSS"
            else
                echo "  $EXP_NAME: MISSING"
            fi
        done
    done
done

echo ""
echo "Next: Run visualization"
echo "  python scripts/lambda/visualize_c4.py"
echo ""
