#!/usr/bin/env python3
"""
Visualization for Part 2 experiments.

Generates publication-quality figures matching Part 1 style:
- Red (#E63946) for HC (unconstrained)
- Teal (#2A9D8F) for mHC (manifold-constrained)
- Gray (#6C757D) for residual baseline
- Serif fonts, clean styling

Author: Taylor Kolasinski
Part of mHC reproduction series: https://poisson.run/notes/deepseek-mhc
"""

import os
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# =============================================================================
# Style Configuration (matching Part 1)
# =============================================================================

COLORS = {
    "residual": "#6C757D",  # Gray
    "hc": "#E63946",        # Red
    "mhc": "#2A9D8F",       # Teal
}

LABELS = {
    "residual": "Residual",
    "hc": "HC (unconstrained)",
    "mhc": "mHC (manifold)",
}

# Matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Palatino", "serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# =============================================================================
# Data Loading
# =============================================================================

def load_history(path: Path) -> Optional[list]:
    """Load training history from JSON file."""
    if not path.exists():
        print(f"Warning: {path} not found")
        return None

    with open(path) as f:
        return json.load(f)


def load_experiment(run_dir: Path, method: str, depth: int, seed: int) -> Optional[dict]:
    """Load a single experiment's data."""
    exp_name = f"{method}_d{depth}_s{seed}"
    exp_dir = run_dir / exp_name

    history_path = exp_dir / "history.json"
    config_path = exp_dir / "config.json"

    history = load_history(history_path)
    if history is None:
        return None

    config = None
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    return {
        "name": exp_name,
        "method": method,
        "depth": depth,
        "seed": seed,
        "history": history,
        "config": config,
    }


def smooth(values: list, window: int = 20) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_loss_comparison(experiments: list, ax: plt.Axes, title: str = "Training Loss"):
    """Plot loss curves for multiple experiments."""
    for exp in experiments:
        if exp is None:
            continue

        steps = [m["step"] for m in exp["history"]]
        losses = [m["loss"] for m in exp["history"]]

        color = COLORS.get(exp["method"], "gray")
        label = f"{LABELS[exp['method']]} (d={exp['depth']})"

        # Plot smoothed
        smoothed = smooth(losses)
        smooth_steps = steps[:len(smoothed)]
        ax.plot(smooth_steps, smoothed, color=color, label=label, linewidth=1.5)

        # Plot raw with low alpha
        ax.plot(steps, losses, color=color, alpha=0.2, linewidth=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()


def plot_amax_comparison(experiments: list, ax: plt.Axes, title: str = "Composite Amax"):
    """Plot Amax curves for multiple experiments."""
    for exp in experiments:
        if exp is None:
            continue

        # Skip if no Amax data
        if "composite_amax" not in exp["history"][0]:
            continue

        steps = [m["step"] for m in exp["history"]]
        amax = [m.get("composite_amax", 1.0) for m in exp["history"]]

        color = COLORS.get(exp["method"], "gray")
        label = f"{LABELS[exp['method']]} (d={exp['depth']})"

        ax.plot(steps, amax, color=color, label=label, linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Amax Gain Magnitude")
    ax.set_title(title)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, label="Stability threshold")
    ax.legend()


def plot_gradient_norms(experiments: list, ax: plt.Axes, title: str = "Gradient Norm"):
    """Plot gradient norm curves."""
    for exp in experiments:
        if exp is None:
            continue

        steps = [m["step"] for m in exp["history"]]
        grad_norms = [m["grad_norm"] for m in exp["history"]]

        color = COLORS.get(exp["method"], "gray")
        label = f"{LABELS[exp['method']]} (d={exp['depth']})"

        smoothed = smooth(grad_norms)
        smooth_steps = steps[:len(smoothed)]
        ax.plot(smooth_steps, smoothed, color=color, label=label, linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title(title)
    ax.legend()


def plot_layer_amax_heatmap(exp: dict, ax: plt.Axes, title: str = "Per-Layer Amax"):
    """Plot heatmap of per-layer Amax over training."""
    if exp is None or "layer_amax" not in exp["history"][0]:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    # Extract layer Amax data
    steps = [m["step"] for m in exp["history"]]
    n_layers = len(exp["history"][0]["layer_amax"])

    # Build matrix: (steps, layers)
    amax_matrix = np.zeros((len(steps), n_layers))
    for i, m in enumerate(exp["history"]):
        for layer_data in m["layer_amax"]:
            # Use max of attn and ffn
            layer_idx = layer_data["layer"]
            amax_matrix[i, layer_idx] = max(layer_data["attn"], layer_data["ffn"])

    # Subsample for cleaner visualization
    step_stride = max(1, len(steps) // 100)
    amax_matrix = amax_matrix[::step_stride, :]
    steps_subsampled = steps[::step_stride]

    im = ax.imshow(
        amax_matrix.T,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0.9,
        vmax=2.0,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Layer")
    ax.set_title(f"{title} ({exp['method'].upper()})")

    # Set x ticks
    n_ticks = 5
    tick_indices = np.linspace(0, len(steps_subsampled) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([steps_subsampled[i] for i in tick_indices])

    plt.colorbar(im, ax=ax, label="Amax")


def plot_seed_variation(experiments_by_seed: dict, ax_loss: plt.Axes, ax_amax: plt.Axes,
                        method: str, depth: int):
    """Plot seed variation with error bars."""
    seeds = list(experiments_by_seed.keys())
    exps = [experiments_by_seed[s] for s in seeds]

    # Get final metrics for each seed
    final_losses = []
    max_amax = []

    for exp in exps:
        if exp is None:
            continue
        final_losses.append(exp["history"][-1]["loss"])
        if "composite_amax" in exp["history"][-1]:
            max_amax.append(max(m.get("composite_amax", 1.0) for m in exp["history"]))
        else:
            max_amax.append(1.0)

    color = COLORS.get(method, "gray")
    label = LABELS[method]

    # Loss bar
    mean_loss = np.mean(final_losses)
    std_loss = np.std(final_losses)
    ax_loss.bar(label, mean_loss, yerr=std_loss, color=color, alpha=0.7, capsize=5)
    ax_loss.set_ylabel("Final Loss")
    ax_loss.set_title(f"Depth {depth}: Loss (mean ± std)")

    # Amax bar
    if max_amax:
        mean_amax = np.mean(max_amax)
        std_amax = np.std(max_amax)
        ax_amax.bar(label, mean_amax, yerr=std_amax, color=color, alpha=0.7, capsize=5)
        ax_amax.set_ylabel("Max Amax")
        ax_amax.set_title(f"Depth {depth}: Amax (mean ± std)")
        ax_amax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3)


# =============================================================================
# Main Figure Generation
# =============================================================================

def generate_main_figure(run_dir: Path, output_dir: Path, depths: list = [32, 48],
                         seeds: list = [42, 123, 456]):
    """Generate the main 6-panel comparison figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10))

    # Panel layout:
    # [a] Loss curves (depth 32)     [b] Loss curves (depth 48)
    # [c] Amax curves (depth 32)     [d] Amax curves (depth 48)
    # [e] Gradient norms (depth 32)  [f] Gradient norms (depth 48)

    axes = []
    for i in range(6):
        ax = fig.add_subplot(3, 2, i + 1)
        axes.append(ax)

    for col, depth in enumerate(depths):
        # Load experiments for this depth (seed 42 for main comparison)
        experiments = []
        for method in ["residual", "hc", "mhc"]:
            exp = load_experiment(run_dir, method, depth, seed=42)
            experiments.append(exp)

        # Loss curves
        plot_loss_comparison(experiments, axes[col], f"Training Loss (depth={depth})")

        # Amax curves
        plot_amax_comparison(experiments, axes[2 + col], f"Amax Evolution (depth={depth})")

        # Gradient norms
        plot_gradient_norms(experiments, axes[4 + col], f"Gradient Norm (depth={depth})")

    fig.tight_layout()
    fig.savefig(output_dir / "main_comparison.png")
    fig.savefig(output_dir / "main_comparison.pdf")
    print(f"Saved: {output_dir / 'main_comparison.png'}")
    plt.close(fig)


def generate_seed_variation_figure(run_dir: Path, output_dir: Path,
                                   depths: list = [32, 48], seeds: list = [42, 123, 456]):
    """Generate seed variation analysis figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row, depth in enumerate(depths):
        for method in ["hc", "mhc"]:
            # Load all seeds for this method/depth
            experiments_by_seed = {}
            for seed in seeds:
                exp = load_experiment(run_dir, method, depth, seed)
                if exp is not None:
                    experiments_by_seed[seed] = exp

            if not experiments_by_seed:
                continue

            # Plot training curves across seeds
            ax = axes[row, 0] if method == "hc" else axes[row, 1]
            color = COLORS[method]

            for seed, exp in experiments_by_seed.items():
                steps = [m["step"] for m in exp["history"]]
                losses = smooth([m["loss"] for m in exp["history"]])
                alpha = 0.3 if seed != 42 else 1.0
                linestyle = "-" if seed == 42 else "--"
                ax.plot(steps[:len(losses)], losses, color=color, alpha=alpha,
                       linestyle=linestyle, label=f"seed={seed}")

            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title(f"{LABELS[method]} (depth={depth})")
            ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "seed_variation.png")
    print(f"Saved: {output_dir / 'seed_variation.png'}")
    plt.close(fig)


def generate_layer_analysis_figure(run_dir: Path, output_dir: Path, depth: int = 32):
    """Generate per-layer Amax heatmaps."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, method in enumerate(["hc", "mhc"]):
        exp = load_experiment(run_dir, method, depth, seed=42)
        plot_layer_amax_heatmap(exp, axes[i], f"Per-Layer Amax")

    fig.tight_layout()
    fig.savefig(output_dir / f"layer_amax_d{depth}.png")
    print(f"Saved: {output_dir / f'layer_amax_d{depth}.png'}")
    plt.close(fig)


def generate_summary_table(run_dir: Path, output_dir: Path,
                           methods: list = ["residual", "hc", "mhc"],
                           depths: list = [32, 48], seeds: list = [42, 123, 456]):
    """Generate summary statistics table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for method in methods:
        for depth in depths:
            final_losses = []
            max_amax_values = []

            for seed in seeds:
                exp = load_experiment(run_dir, method, depth, seed)
                if exp is None:
                    continue

                final_losses.append(exp["history"][-1]["loss"])

                if "composite_amax" in exp["history"][-1]:
                    max_amax = max(m.get("composite_amax", 1.0) for m in exp["history"])
                    max_amax_values.append(max_amax)

            if not final_losses:
                continue

            results.append({
                "method": method,
                "depth": depth,
                "loss_mean": np.mean(final_losses),
                "loss_std": np.std(final_losses),
                "amax_mean": np.mean(max_amax_values) if max_amax_values else "N/A",
                "amax_std": np.std(max_amax_values) if max_amax_values else "N/A",
                "n_seeds": len(final_losses),
            })

    # Print table
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"{'Method':<12} {'Depth':<8} {'Loss (mean±std)':<20} {'Max Amax (mean±std)':<20} {'Seeds':<6}")
    print("-" * 80)

    for r in results:
        loss_str = f"{r['loss_mean']:.4f} ± {r['loss_std']:.4f}"
        if isinstance(r['amax_mean'], str):
            amax_str = r['amax_mean']
        else:
            amax_str = f"{r['amax_mean']:.2f} ± {r['amax_std']:.2f}"
        print(f"{r['method']:<12} {r['depth']:<8} {loss_str:<20} {amax_str:<20} {r['n_seeds']:<6}")

    print("=" * 80)

    # Save to JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {output_dir / 'summary.json'}")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Part 2 visualizations")
    parser.add_argument("--run-dir", type=str, default="runs_c4", help="Experiment directory")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--depths", type=str, default="32,48", help="Depths to analyze")
    parser.add_argument("--seeds", type=str, default="42,123,456", help="Seeds to analyze")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    depths = [int(d) for d in args.depths.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    if not run_dir.exists():
        print(f"Error: {run_dir} not found")
        return

    print("Generating visualizations...")
    print(f"  Run dir: {run_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Depths: {depths}")
    print(f"  Seeds: {seeds}")

    # Generate all figures
    generate_main_figure(run_dir, output_dir, depths, seeds)
    generate_seed_variation_figure(run_dir, output_dir, depths, seeds)

    for depth in depths:
        generate_layer_analysis_figure(run_dir, output_dir, depth)

    generate_summary_table(run_dir, output_dir, depths=depths, seeds=seeds)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
