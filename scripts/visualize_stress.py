"""
Generate figures for stress test experiments.

Experiments:
- stress_deep: 16-layer deep model
- stress_lr: Aggressive lr=1e-3
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

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

COLORS = {"hc": "#e74c3c", "mhc": "#3498db"}
LABELS = {"hc": "HC", "mhc": "mHC"}


def load_history(run_dir: Path) -> dict:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return None
    with open(history_path) as f:
        return json.load(f)


def smooth(values, window=20):
    arr = np.array(values)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_stress_comparison(experiment_name: str, run_dir: Path, output_dir: Path):
    """Plot comparison for a stress test experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    histories = {
        "hc": load_history(run_dir / "hc"),
        "mhc": load_history(run_dir / "mhc"),
    }

    if all(h is None for h in histories.values()):
        print(f"No data found for {experiment_name}")
        return

    # Top-left: Training loss
    ax = axes[0, 0]
    for conn, hist in histories.items():
        if hist is None:
            continue
        steps = [h["step"] for h in hist]
        losses = [h["loss"] for h in hist]
        smoothed = smooth(losses)
        ax.plot(steps[len(steps)-len(smoothed):], smoothed,
                color=COLORS[conn], label=LABELS[conn], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("(a) Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Validation loss
    ax = axes[0, 1]
    for conn, hist in histories.items():
        if hist is None:
            continue
        steps = [h["step"] for h in hist if "val_loss" in h]
        val_losses = [h["val_loss"] for h in hist if "val_loss" in h]
        if steps:
            ax.plot(steps, val_losses, color=COLORS[conn],
                    label=LABELS[conn], linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("(b) Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: Composite Amax
    ax = axes[1, 0]
    for conn, hist in histories.items():
        if hist is None:
            continue
        steps = [h["step"] for h in hist if "composite_amax" in h]
        amax = [h["composite_amax"] for h in hist if "composite_amax" in h]
        if steps:
            ax.plot(steps, amax, color=COLORS[conn],
                    label=LABELS[conn], linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Stability bound")
    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Amax Gain")
    ax.set_title("(c) Composite Amax Gain Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: Gradient norm
    ax = axes[1, 1]
    for conn, hist in histories.items():
        if hist is None:
            continue
        steps = [h["step"] for h in hist]
        grads = [h["grad_norm"] for h in hist]
        smoothed = smooth(grads)
        ax.plot(steps[len(steps)-len(smoothed):], smoothed,
                color=COLORS[conn], label=LABELS[conn], linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("(d) Gradient Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Stress Test: {experiment_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"{experiment_name}.png")
    plt.close()
    print(f"Saved {experiment_name}.png")


def plot_amax_comparison(output_dir: Path):
    """Plot Amax comparison across both stress tests."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    experiments = [
        ("stress_deep", "Deep Model (16 layers)", Path("runs/stress_deep")),
        ("stress_lr", "Aggressive LR (1e-3)", Path("runs/stress_lr")),
    ]

    for idx, (name, title, run_dir) in enumerate(experiments):
        ax = axes[idx]

        for conn in ["hc", "mhc"]:
            hist = load_history(run_dir / conn)
            if hist is None:
                continue
            steps = [h["step"] for h in hist if "composite_amax" in h]
            amax = [h["composite_amax"] for h in hist if "composite_amax" in h]
            if steps:
                ax.plot(steps, amax, color=COLORS[conn],
                        label=LABELS[conn], linewidth=2)

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Bound")
        ax.set_xlabel("Step")
        ax.set_ylabel("Composite Amax Gain")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.suptitle("HC Instability vs mHC Stability", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "amax_comparison.png")
    plt.close()
    print("Saved amax_comparison.png")


def print_summary():
    """Print summary statistics from stress tests."""
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)

    experiments = [
        ("Deep Model (16 layers)", Path("runs/stress_deep")),
        ("Aggressive LR (1e-3)", Path("runs/stress_lr")),
    ]

    for exp_name, run_dir in experiments:
        print(f"\n{exp_name}:")
        print("-" * 50)

        for conn in ["hc", "mhc"]:
            hist = load_history(run_dir / conn)
            if hist is None:
                continue

            # Get metrics
            final_loss = hist[-1]["loss"]
            val_losses = [h["val_loss"] for h in hist if "val_loss" in h]
            final_val = val_losses[-1] if val_losses else None

            amax_values = [h["composite_amax"] for h in hist if "composite_amax" in h]
            if amax_values:
                start_amax = amax_values[0]
                peak_amax = max(amax_values)
                final_amax = amax_values[-1]
            else:
                start_amax = peak_amax = final_amax = None

            print(f"  {LABELS[conn]}:")
            print(f"    Train loss: {final_loss:.4f}")
            if final_val:
                print(f"    Val loss:   {final_val:.4f}")
            if start_amax is not None:
                print(f"    Amax: {start_amax:.4f} -> {peak_amax:.4f} (peak) -> {final_amax:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    # Generate individual experiment figures
    plot_stress_comparison("stress_deep", Path("runs/stress_deep"), output_dir)
    plot_stress_comparison("stress_lr", Path("runs/stress_lr"), output_dir)

    # Generate Amax comparison
    plot_amax_comparison(output_dir)

    # Print summary
    print_summary()

    print(f"\nFigures saved to {output_dir}")
