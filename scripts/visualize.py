"""
Generate paper-style figures from training logs.

Figures:
- Figure 2/5: Loss curves + gradient norms for all 3 methods
- Figure 3: Amax Gain Magnitude per layer and composite
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# Use non-interactive backend for saving figures
matplotlib.use("Agg")

# Style settings for paper-quality figures
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color scheme
COLORS = {
    "residual": "#2ecc71",  # Green
    "hc": "#e74c3c",        # Red
    "mhc": "#3498db",       # Blue
}

LABELS = {
    "residual": "Residual",
    "hc": "HC (Hyper-Connection)",
    "mhc": "mHC (Manifold HC)",
}


def load_history(run_dir: Path, connection_type: str) -> Optional[list]:
    """Load training history from JSON file."""
    history_path = run_dir / connection_type / "history.json"
    if not history_path.exists():
        print(f"Warning: {history_path} not found")
        return None

    with open(history_path, "r") as f:
        return json.load(f)


def smooth(values: list, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    arr = np.array(values)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_loss_curves(
    histories: dict,
    output_path: Path,
    smooth_window: int = 20,
):
    """
    Plot loss curves for all methods (Paper Figure 2/5 style).
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Training loss
    ax = axes[0]
    for conn_type, history in histories.items():
        if history is None:
            continue
        steps = [h["step"] for h in history]
        losses = [h["loss"] for h in history]

        # Raw data (faded)
        ax.plot(steps, losses, alpha=0.2, color=COLORS[conn_type])
        # Smoothed
        smoothed = smooth(losses, smooth_window)
        smoothed_steps = steps[smooth_window - 1:]
        ax.plot(smoothed_steps, smoothed, color=COLORS[conn_type],
                label=LABELS[conn_type], linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Right: Gradient norm
    ax = axes[1]
    for conn_type, history in histories.items():
        if history is None:
            continue
        steps = [h["step"] for h in history]
        grad_norms = [h["grad_norm"] for h in history]

        # Raw data (faded)
        ax.plot(steps, grad_norms, alpha=0.2, color=COLORS[conn_type])
        # Smoothed
        smoothed = smooth(grad_norms, smooth_window)
        smoothed_steps = steps[smooth_window - 1:]
        ax.plot(smoothed_steps, smoothed, color=COLORS[conn_type],
                label=LABELS[conn_type], linewidth=2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norm (clipped to 1.0)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss curves to {output_path}")


def plot_amax_gain(
    histories: dict,
    output_path: Path,
    smooth_window: int = 20,
):
    """
    Plot Amax Gain Magnitude over training (Paper Figure 3 style).
    """
    # Only plot for HC and mHC (residual doesn't have H_res)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Composite Amax over training
    ax = axes[0]
    for conn_type in ["hc", "mhc"]:
        history = histories.get(conn_type)
        if history is None:
            continue

        steps = []
        amax_values = []
        for h in history:
            if "composite_amax" in h:
                steps.append(h["step"])
                amax_values.append(h["composite_amax"])

        if not steps:
            continue

        # Raw data (faded)
        ax.plot(steps, amax_values, alpha=0.2, color=COLORS[conn_type])
        # Smoothed
        smoothed = smooth(amax_values, smooth_window)
        smoothed_steps = steps[smooth_window - 1:]
        ax.plot(smoothed_steps, smoothed, color=COLORS[conn_type],
                label=LABELS[conn_type], linewidth=2)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Stability bound")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Composite Amax Gain")
    ax.set_title("Composite H_res Gain Magnitude")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Right: Per-layer Amax at final step
    ax = axes[1]

    for conn_type in ["hc", "mhc"]:
        history = histories.get(conn_type)
        if history is None:
            continue

        # Get last step with layer info
        final_step = None
        for h in reversed(history):
            if "layer_amax" in h:
                final_step = h
                break

        if final_step is None:
            continue

        layer_data = final_step["layer_amax"]
        layers = [d["layer"] for d in layer_data]
        attn_amax = [d["attn"] for d in layer_data]
        ffn_amax = [d["ffn"] for d in layer_data]

        # Plot attention and FFN as grouped bars
        x = np.array(layers)
        width = 0.35
        offset = -width / 2 if conn_type == "hc" else width / 2

        ax.bar(x + offset - width / 4, attn_amax, width / 2,
               color=COLORS[conn_type], alpha=0.7, label=f"{LABELS[conn_type]} Attn")
        ax.bar(x + offset + width / 4, ffn_amax, width / 2,
               color=COLORS[conn_type], alpha=0.4, hatch="//")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Amax Gain")
    ax.set_title("Per-Layer Amax at Final Step")
    ax.set_xticks(range(6))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Amax gain plot to {output_path}")


def plot_combined_summary(
    histories: dict,
    output_path: Path,
    smooth_window: int = 20,
):
    """
    Combined summary figure with all key metrics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Training loss
    ax = axes[0, 0]
    for conn_type, history in histories.items():
        if history is None:
            continue
        steps = [h["step"] for h in history]
        losses = [h["loss"] for h in history]
        smoothed = smooth(losses, smooth_window)
        smoothed_steps = steps[smooth_window - 1:]
        ax.plot(smoothed_steps, smoothed, color=COLORS[conn_type],
                label=LABELS[conn_type], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Gradient norm
    ax = axes[0, 1]
    for conn_type, history in histories.items():
        if history is None:
            continue
        steps = [h["step"] for h in history]
        grad_norms = [h["grad_norm"] for h in history]
        smoothed = smooth(grad_norms, smooth_window)
        smoothed_steps = steps[smooth_window - 1:]
        ax.plot(smoothed_steps, smoothed, color=COLORS[conn_type],
                label=LABELS[conn_type], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("(b) Gradient Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Composite Amax
    ax = axes[1, 0]
    for conn_type in ["hc", "mhc"]:
        history = histories.get(conn_type)
        if history is None:
            continue
        steps = [h["step"] for h in history if "composite_amax" in h]
        amax = [h["composite_amax"] for h in history if "composite_amax" in h]
        if steps:
            smoothed = smooth(amax, smooth_window)
            smoothed_steps = steps[smooth_window - 1:]
            ax.plot(smoothed_steps, smoothed, color=COLORS[conn_type],
                    label=LABELS[conn_type], linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Bound")
    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Amax")
    ax.set_title("(c) Composite Amax Gain Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Validation loss over time (if available)
    ax = axes[1, 1]
    has_val = False
    for conn_type, history in histories.items():
        if history is None:
            continue
        steps = [h["step"] for h in history if "val_loss" in h]
        val_losses = [h["val_loss"] for h in history if "val_loss" in h]
        if steps:
            has_val = True
            ax.plot(steps, val_losses, color=COLORS[conn_type],
                    label=LABELS[conn_type], linewidth=2, marker="o", markersize=4)

    if has_val:
        ax.set_xlabel("Step")
        ax.set_ylabel("Validation Loss")
        ax.set_title("(d) Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No validation data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("(d) Validation Loss")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved combined summary to {output_path}")


def plot_v1_vs_v2_comparison(
    v1_histories: dict,
    v2_histories: dict,
    output_path: Path,
    smooth_window: int = 20,
):
    """
    Plot v1 vs v2 comparison showing improvement from input-dependent HC.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Color scheme for v1 (dashed) vs v2 (solid)
    V1_STYLE = {"linestyle": "--", "alpha": 0.7}
    V2_STYLE = {"linestyle": "-", "alpha": 1.0}

    # Top-left: Training loss comparison
    ax = axes[0, 0]
    for version, histories, style in [("v1", v1_histories, V1_STYLE), ("v2", v2_histories, V2_STYLE)]:
        for conn_type, history in histories.items():
            if history is None:
                continue
            steps = [h["step"] for h in history]
            losses = [h["loss"] for h in history]
            smoothed = smooth(losses, smooth_window)
            smoothed_steps = steps[smooth_window - 1:]
            label = f"{LABELS[conn_type]} ({version})" if version == "v1" else LABELS[conn_type]
            ax.plot(smoothed_steps, smoothed, color=COLORS[conn_type],
                    label=label, linewidth=2, **style)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("(a) Training Loss: v1 (static) vs v2 (input-dependent)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Top-right: Gradient norm comparison
    ax = axes[0, 1]
    for version, histories, style in [("v1", v1_histories, V1_STYLE), ("v2", v2_histories, V2_STYLE)]:
        for conn_type, history in histories.items():
            if history is None:
                continue
            steps = [h["step"] for h in history]
            grad_norms = [h["grad_norm"] for h in history]
            smoothed = smooth(grad_norms, smooth_window)
            smoothed_steps = steps[smooth_window - 1:]
            label = f"{LABELS[conn_type]} ({version})" if version == "v1" else LABELS[conn_type]
            ax.plot(smoothed_steps, smoothed, color=COLORS[conn_type],
                    label=label, linewidth=2, **style)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("(b) Gradient Norm: v1 vs v2")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Composite Amax comparison (HC and mHC only)
    ax = axes[1, 0]
    for version, histories, style in [("v1", v1_histories, V1_STYLE), ("v2", v2_histories, V2_STYLE)]:
        for conn_type in ["hc", "mhc"]:
            history = histories.get(conn_type)
            if history is None:
                continue
            steps = [h["step"] for h in history if "composite_amax" in h]
            amax = [h["composite_amax"] for h in history if "composite_amax" in h]
            if steps:
                smoothed = smooth(amax, smooth_window)
                smoothed_steps = steps[smooth_window - 1:]
                label = f"{LABELS[conn_type]} ({version})" if version == "v1" else LABELS[conn_type]
                ax.plot(smoothed_steps, smoothed, color=COLORS[conn_type],
                        label=label, linewidth=2, **style)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.7, label="Stability bound")
    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Amax")
    ax.set_title("(c) Composite Amax Gain: v1 vs v2")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Final loss comparison (bar chart)
    ax = axes[1, 1]
    conn_types = ["residual", "hc", "mhc"]
    x = np.arange(len(conn_types))
    width = 0.35

    v1_final_losses = []
    v2_final_losses = []
    for conn_type in conn_types:
        # Get final loss from v1
        h = v1_histories.get(conn_type)
        v1_final_losses.append(h[-1]["loss"] if h else 0)
        # Get final loss from v2
        h = v2_histories.get(conn_type)
        v2_final_losses.append(h[-1]["loss"] if h else 0)

    bars1 = ax.bar(x - width/2, v1_final_losses, width, label="v1 (static)", alpha=0.7)
    bars2 = ax.bar(x + width/2, v2_final_losses, width, label="v2 (input-dependent)", alpha=1.0)

    ax.set_xlabel("Connection Type")
    ax.set_ylabel("Final Training Loss")
    ax.set_title("(d) Final Loss Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[c] for c in conn_types], fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved v1 vs v2 comparison to {output_path}")


def generate_all_figures(
    run_dir: Path = Path("runs"),
    output_dir: Path = Path("figures"),
):
    """Generate all figures from training logs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load histories
    histories = {}
    for conn_type in ["residual", "hc", "mhc"]:
        histories[conn_type] = load_history(run_dir, conn_type)

    # Check if we have any data
    if all(h is None for h in histories.values()):
        print("No training histories found. Run training first:")
        print("  python -m src.train -c all -s 5000")
        return

    # Generate figures
    plot_loss_curves(histories, output_dir / "loss_curves.png")
    plot_amax_gain(histories, output_dir / "amax_gain.png")
    plot_combined_summary(histories, output_dir / "combined_summary.png")

    print(f"\nAll figures saved to {output_dir}")


def load_v2_history(base_dir: Path, connection_type: str) -> Optional[list]:
    """Load v2 training history - handles v2 directory structure."""
    # v2 structure: runs/v2_{connection_type}/{connection_type}/history.json
    history_path = base_dir / f"v2_{connection_type}" / connection_type / "history.json"
    if not history_path.exists():
        print(f"Warning: {history_path} not found")
        return None

    with open(history_path, "r") as f:
        return json.load(f)


def generate_v1_v2_comparison(
    v1_dir: Path = Path("runs"),
    v2_dir: Path = Path("runs"),
    output_dir: Path = Path("figures"),
):
    """Generate v1 vs v2 comparison figures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load v1 histories
    v1_histories = {}
    for conn_type in ["residual", "hc", "mhc"]:
        v1_histories[conn_type] = load_history(v1_dir, conn_type)

    # Load v2 histories (handles v2 directory structure)
    v2_histories = {}
    for conn_type in ["residual", "hc", "mhc"]:
        v2_histories[conn_type] = load_v2_history(v2_dir, conn_type)

    # Check if we have any data
    v1_has_data = any(h is not None for h in v1_histories.values())
    v2_has_data = any(h is not None for h in v2_histories.values())

    if not v1_has_data or not v2_has_data:
        print("Need both v1 and v2 data for comparison.")
        if not v1_has_data:
            print(f"  Missing v1 data in {v1_dir}")
        if not v2_has_data:
            print(f"  Missing v2 data in {v2_dir}")
        return

    # Generate comparison figure
    plot_v1_vs_v2_comparison(v1_histories, v2_histories, output_dir / "v1_vs_v2_comparison.png")

    print(f"\nComparison figures saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--run-dir", type=str, default="runs", help="Training runs directory")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--compare", action="store_true", help="Generate v1 vs v2 comparison")
    parser.add_argument("--v1-dir", type=str, default="runs", help="v1 runs directory (for comparison)")
    parser.add_argument("--v2-dir", type=str, default="runs/v2", help="v2 runs directory (for comparison)")

    args = parser.parse_args()

    if args.compare:
        generate_v1_v2_comparison(Path(args.v1_dir), Path(args.v2_dir), Path(args.output_dir))
    else:
        generate_all_figures(Path(args.run_dir), Path(args.output_dir))
