"""
Priority 1: Critical Depth Sweep - The Novel Finding

Test HC at various depths with same parameter budget to find the critical depth
where instability emerges.

Depths: [6, 8, 10, 12, 14, 16, 20, 24]
Same param budget: reduce hidden_dim as depth increases
5000 steps each, HC only (we already know mHC is stable)

Key metrics:
- Final Amax
- Step where Amax crosses thresholds (0.5, 0.3, 0.1)
"""

import sys
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, ModelConfig, TrainConfig
from src.train import train

# Experiment configuration
DEPTHS = [6, 8, 10, 12, 14, 16, 20, 24]
BASELINE_DEPTH = 6
BASELINE_DIM = 384
MAX_STEPS = 5000

# Amax thresholds to track (these are decreasing toward instability bound of 1.0)
# For mHC, Amax stays near 1.0; for HC, it can grow unbounded
# We track when it EXCEEDS these thresholds (diverging from 1.0)
AMAX_THRESHOLDS = [1.5, 2.0, 3.0, 5.0]


def compute_dim_for_depth(depth: int) -> int:
    """
    Compute hidden_dim to maintain same parameter budget.

    Main param sources per layer: ~12 * hidden_dim^2
    So: depth * dim^2 ≈ constant
    => dim = sqrt(BASELINE_DEPTH * BASELINE_DIM^2 / depth)

    Round to nearest multiple of 32 for GPU efficiency.
    """
    base_param_factor = BASELINE_DEPTH * BASELINE_DIM ** 2
    target_dim = math.sqrt(base_param_factor / depth)

    # Round to nearest multiple of 32
    dim = int(round(target_dim / 32) * 32)

    # Minimum dim of 128
    dim = max(dim, 128)

    return dim


def compute_heads_for_dim(dim: int) -> int:
    """Compute number of attention heads, targeting head_dim=64."""
    if dim % 64 == 0:
        return dim // 64
    elif dim % 32 == 0:
        return dim // 32
    else:
        return max(1, dim // 64)


def run_single_depth(depth: int, run_dir: Path) -> dict:
    """Run HC training at a specific depth and return analysis."""
    dim = compute_dim_for_depth(depth)
    n_heads = compute_heads_for_dim(dim)
    head_dim = dim // n_heads

    print(f"\n{'='*70}")
    print(f"DEPTH {depth}: dim={dim}, heads={n_heads}, head_dim={head_dim}")
    print(f"{'='*70}")

    model_config = ModelConfig(
        n_layers=depth,
        hidden_dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        ffn_multiplier=4.0,
        expansion_rate=4,
        hc_alpha=0.01,
        sinkhorn_iters=20,
        max_seq_len=256,
        dropout=0.1,
        connection_type="hc",
    )

    train_config = TrainConfig(
        batch_size=64,
        context_length=256,
        max_steps=MAX_STEPS,
        eval_interval=250,
        eval_steps=20,
        log_interval=50,
        learning_rate=3e-4,
        weight_decay=0.1,
        warmup_steps=100,
        save_interval=5000,  # Only save at end
        seed=42,
    )

    config = Config(model=model_config, train=train_config)

    result = train(
        connection_type="hc",
        config=config,
        run_dir=run_dir,
        data_dir=Path("data"),
    )

    return analyze_run(run_dir, depth, dim)


def analyze_run(run_dir: Path, depth: int, dim: int) -> dict:
    """Analyze a completed run and extract key metrics."""
    history_path = run_dir / "history.json"

    with open(history_path) as f:
        history = json.load(f)

    # Extract Amax values
    amax_data = [(h["step"], h["composite_amax"])
                 for h in history if "composite_amax" in h]

    if not amax_data:
        return {"depth": depth, "dim": dim, "error": "No Amax data"}

    steps, amax_values = zip(*amax_data)

    # Find threshold crossings (when Amax EXCEEDS threshold)
    threshold_crossings = {}
    for thresh in AMAX_THRESHOLDS:
        crossing_step = None
        for step, amax in amax_data:
            if amax >= thresh:
                crossing_step = step
                break
        threshold_crossings[f"step_amax_{thresh}"] = crossing_step

    # Final metrics
    final_amax = amax_values[-1]
    peak_amax = max(amax_values)
    final_loss = history[-1]["loss"]

    val_losses = [h["val_loss"] for h in history if "val_loss" in h]
    final_val_loss = val_losses[-1] if val_losses else None

    return {
        "depth": depth,
        "dim": dim,
        "final_amax": final_amax,
        "peak_amax": peak_amax,
        "final_loss": final_loss,
        "final_val_loss": final_val_loss,
        **threshold_crossings,
        "amax_trajectory": list(amax_values),
        "steps": list(steps),
    }


def generate_figure(results: list, output_path: Path):
    """Generate the critical depth figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Sort by depth
    results = sorted(results, key=lambda x: x["depth"])
    depths = [r["depth"] for r in results]

    # Color gradient: shallow (green/safe) -> deep (red/unstable)
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(depths)))

    # (a) Final Amax vs Depth
    ax = axes[0, 0]
    final_amax = [r["final_amax"] for r in results]
    peak_amax = [r["peak_amax"] for r in results]

    ax.bar([d - 0.2 for d in depths], final_amax, width=0.4,
           label="Final Amax", color=colors, alpha=0.8)
    ax.bar([d + 0.2 for d in depths], peak_amax, width=0.4,
           label="Peak Amax", color=colors, alpha=0.4, hatch="//")

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Stability bound")
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Amax Gain Magnitude")
    ax.set_title("(a) Amax vs Depth (Same Param Budget)")
    ax.set_xticks(depths)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) Amax Trajectories
    ax = axes[0, 1]
    for i, r in enumerate(results):
        if "amax_trajectory" in r and r["amax_trajectory"]:
            ax.plot(r["steps"], r["amax_trajectory"],
                    color=colors[i], label=f"depth={r['depth']}", linewidth=1.5)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Amax")
    ax.set_title("(b) Amax Trajectory Over Training")
    ax.legend(ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    # (c) Threshold Crossing Steps
    ax = axes[1, 0]
    for i, thresh in enumerate(AMAX_THRESHOLDS):
        crossing_steps = []
        for r in results:
            step = r.get(f"step_amax_{thresh}")
            crossing_steps.append(step if step is not None else MAX_STEPS + 100)

        # Plot as line with markers
        marker_shapes = ['o', 's', '^', 'D']
        ax.plot(depths, crossing_steps, marker=marker_shapes[i],
                label=f"Amax > {thresh}", linewidth=2, markersize=8)

    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Step When Threshold Crossed")
    ax.set_title("(c) Training Step When Amax Exceeds Threshold")
    ax.set_xticks(depths)
    ax.set_ylim(0, MAX_STEPS + 200)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (d) Validation Loss vs Depth
    ax = axes[1, 1]
    val_losses = [r.get("final_val_loss", r["final_loss"]) for r in results]
    dims = [r["dim"] for r in results]

    bars = ax.bar(depths, val_losses, color=colors, alpha=0.8)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Final Validation Loss")
    ax.set_title("(d) Final Val Loss vs Depth")
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3)

    # Add dim annotations
    for bar, dim in zip(bars, dims):
        ax.annotate(f"d={dim}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.suptitle("HC Critical Depth Analysis: Same Param Budget",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"\nSaved figure to {output_path}")


def print_summary(results: list):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("CRITICAL DEPTH SWEEP SUMMARY")
    print("=" * 80)

    print(f"\n{'Depth':>6} {'Dim':>6} {'Final Amax':>12} {'Peak Amax':>12} {'Val Loss':>10}")
    print("-" * 50)

    for r in sorted(results, key=lambda x: x["depth"]):
        val_loss = r.get("final_val_loss", r.get("final_loss", 0))
        print(f"{r['depth']:>6} {r['dim']:>6} {r['final_amax']:>12.4f} "
              f"{r['peak_amax']:>12.4f} {val_loss:>10.4f}")

    # Find critical depth (where Amax first exceeds 2.0 during training)
    print("\n" + "-" * 50)
    for thresh in AMAX_THRESHOLDS:
        critical = None
        for r in sorted(results, key=lambda x: x["depth"]):
            if r.get(f"step_amax_{thresh}") is not None:
                critical = r["depth"]
                break
        if critical:
            print(f"Critical depth for Amax > {thresh}: {critical} layers")
        else:
            print(f"Amax never exceeds {thresh} in tested range")

    print("=" * 80)


def main():
    """Run the complete depth sweep experiment."""
    base_run_dir = Path("runs/depth_sweep")
    base_run_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)

    results = []

    print("\n" + "=" * 70)
    print("PRIORITY 1: CRITICAL DEPTH SWEEP")
    print("Testing HC at depths:", DEPTHS)
    print("=" * 70)

    for depth in DEPTHS:
        run_dir = base_run_dir / f"depth_{depth}"
        run_dir.mkdir(parents=True, exist_ok=True)

        result = run_single_depth(depth, run_dir)
        results.append(result)

        # Save intermediate results
        with open(base_run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Generate figure
    generate_figure(results, figures_dir / "critical_depth.png")

    # Print summary
    print_summary(results)

    print(f"\nResults saved to {base_run_dir}")
    print(f"Figure saved to {figures_dir / 'critical_depth.png'}")


if __name__ == "__main__":
    main()
