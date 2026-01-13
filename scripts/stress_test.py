"""
Stress test experiments to find HC's breaking point.

Experiment 1: Deep model (16 layers, reduced dim ~256)
Experiment 2: Aggressive LR (1e-3, 3x higher than baseline)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, ModelConfig, TrainConfig
from src.train import train


def run_experiment_deep(v2: bool = False):
    """
    Experiment 1: Deep model
    - 16 layers instead of 6
    - Reduced dim (~256) to maintain similar param budget
    - 5000 steps
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: DEEP MODEL (16 layers, dim=256)")
    print("=" * 70)

    run_dir = Path("runs/v2_stress_deep" if v2 else "runs/stress_deep")

    for conn_type in ["hc", "mhc"]:
        print(f"\n{'='*60}")
        print(f"Training {conn_type.upper()} - Deep Model")
        print("=" * 60)

        # Create custom config for deep model
        model_config = ModelConfig(
            n_layers=16,          # 16 layers instead of 6
            hidden_dim=256,       # Reduced dim to ~maintain param budget
            n_heads=4,            # 256 / 64 = 4 heads
            head_dim=64,
            ffn_multiplier=4.0,
            expansion_rate=4,
            hc_alpha=0.01,
            sinkhorn_iters=20,
            max_seq_len=256,
            dropout=0.1,
            connection_type=conn_type,
        )

        train_config = TrainConfig(
            batch_size=64,
            context_length=256,
            max_steps=5000,
            eval_interval=250,
            eval_steps=20,
            log_interval=50,
            learning_rate=3e-4,
            weight_decay=0.1,
            warmup_steps=100,
            save_interval=1000,
            seed=42,
        )

        config = Config(model=model_config, train=train_config)

        train(
            connection_type=conn_type,
            config=config,
            run_dir=run_dir / conn_type,
            data_dir=Path("data"),
        )


def run_experiment_aggressive_lr(v2: bool = False):
    """
    Experiment 2: Aggressive training
    - 6 layers (standard)
    - lr=1e-3 (3x higher than baseline)
    - 10000 steps
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: AGGRESSIVE LR (lr=1e-3, 10000 steps)")
    print("=" * 70)

    run_dir = Path("runs/v2_stress_lr" if v2 else "runs/stress_lr")

    for conn_type in ["hc", "mhc"]:
        print(f"\n{'='*60}")
        print(f"Training {conn_type.upper()} - Aggressive LR")
        print("=" * 60)

        # Create custom config with aggressive LR
        model_config = ModelConfig(
            n_layers=6,
            hidden_dim=384,
            n_heads=6,
            head_dim=64,
            ffn_multiplier=4.0,
            expansion_rate=4,
            hc_alpha=0.01,
            sinkhorn_iters=20,
            max_seq_len=256,
            dropout=0.1,
            connection_type=conn_type,
        )

        train_config = TrainConfig(
            batch_size=64,
            context_length=256,
            max_steps=10000,
            eval_interval=500,
            eval_steps=20,
            log_interval=100,
            learning_rate=1e-3,   # 3x higher than baseline
            weight_decay=0.1,
            warmup_steps=100,
            save_interval=2000,
            seed=42,
        )

        config = Config(model=model_config, train=train_config)

        train(
            connection_type=conn_type,
            config=config,
            run_dir=run_dir / conn_type,
            data_dir=Path("data"),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run stress test experiments")
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default="all",
        choices=["deep", "lr", "all"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Save to v2_stress_* directories (for input-dependent HC)",
    )

    args = parser.parse_args()

    if args.experiment in ["deep", "all"]:
        run_experiment_deep(v2=args.v2)

    if args.experiment in ["lr", "all"]:
        run_experiment_aggressive_lr(v2=args.v2)

    print("\n" + "=" * 70)
    print("STRESS TESTS COMPLETE")
    print("=" * 70)
    prefix = "v2_" if args.v2 else ""
    print("\nResults saved to:")
    print(f"  - runs/{prefix}stress_deep/   (16-layer experiment)")
    print(f"  - runs/{prefix}stress_lr/     (aggressive LR experiment)")
    print("\nGenerate figures with:")
    print("  python scripts/visualize_stress.py")
