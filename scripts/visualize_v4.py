"""
Generate v4 experiment comparison figures.

v4 experiments test the stream persistence fix with:
- Baseline: residual, HC, mHC (6 layers)
- Stress deep: 16 layers HC vs mHC
- Stress LR: aggressive LR (0.001) HC vs mHC
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
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "residual": "#2ecc71",
    "hc": "#e74c3c",
    "mhc": "#3498db",
}

LABELS = {
    "residual": "Residual",
    "hc": "HC",
    "mhc": "mHC",
}


def load_history(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def smooth(values, window=20):
    arr = np.array(values)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_baseline_comparison(output_dir: Path):
    """Plot baseline comparison (residual vs HC vs mHC)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    histories = {}
    for model in ["residual", "hc", "mhc"]:
        histories[model] = load_history(Path(f"runs/v4_baseline/{model}/history.json"))

    # Training loss
    ax = axes[0, 0]
    for model, h in histories.items():
        if h is None:
            continue
        steps = [s["step"] for s in h]
        losses = [s["loss"] for s in h]
        smoothed = smooth(losses)
        ax.plot(steps[19:], smoothed, color=COLORS[model], label=LABELS[model], lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("(a) Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation loss
    ax = axes[0, 1]
    for model, h in histories.items():
        if h is None:
            continue
        steps = [s["step"] for s in h if "val_loss" in s]
        val_losses = [s["val_loss"] for s in h if "val_loss" in s]
        ax.plot(steps, val_losses, color=COLORS[model], label=LABELS[model], lw=2, marker="o", ms=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("(b) Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Amax
    ax = axes[1, 0]
    for model in ["hc", "mhc"]:
        h = histories.get(model)
        if h is None:
            continue
        steps = [s["step"] for s in h if "composite_amax" in s]
        amax = [s["composite_amax"] for s in h if "composite_amax" in s]
        smoothed = smooth(amax)
        ax.plot(steps[19:], smoothed, color=COLORS[model], label=LABELS[model], lw=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Stability bound")
    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Amax")
    ax.set_title("(c) Amax Gain Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final metrics bar chart
    ax = axes[1, 1]
    models = ["Residual", "HC", "mHC"]
    final_val_losses = []
    for model in ["residual", "hc", "mhc"]:
        h = histories.get(model)
        if h:
            val_losses = [s["val_loss"] for s in h if "val_loss" in s]
            final_val_losses.append(val_losses[-1] if val_losses else 0)
        else:
            final_val_losses.append(0)

    bars = ax.bar(models, final_val_losses, color=[COLORS["residual"], COLORS["hc"], COLORS["mhc"]])
    ax.set_ylabel("Final Val Loss")
    ax.set_title("(d) Final Validation Loss")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, final_val_losses):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "v4_baseline_comparison.png")
    plt.close()
    print(f"Saved v4_baseline_comparison.png")


def plot_stress_deep_comparison(output_dir: Path):
    """Plot 16-layer stress test comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    histories = {}
    for model in ["hc", "mhc"]:
        histories[model] = load_history(Path(f"runs/v4_stress_deep/{model}/history.json"))

    # Training loss
    ax = axes[0]
    for model, h in histories.items():
        if h is None:
            continue
        steps = [s["step"] for s in h]
        losses = [s["loss"] for s in h]
        smoothed = smooth(losses)
        ax.plot(steps[19:], smoothed, color=COLORS[model], label=LABELS[model], lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss (16 layers)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Amax
    ax = axes[1]
    for model, h in histories.items():
        if h is None:
            continue
        steps = [s["step"] for s in h if "composite_amax" in s]
        amax = [s["composite_amax"] for s in h if "composite_amax" in s]
        smoothed = smooth(amax)
        ax.plot(steps[19:], smoothed, color=COLORS[model], label=LABELS[model], lw=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Stability bound")
    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Amax")
    ax.set_title("Amax Gain (16 layers)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final comparison
    ax = axes[2]
    models = ["HC", "mHC"]
    metrics = {"Val Loss": [], "Max Amax": []}
    for model in ["hc", "mhc"]:
        h = histories.get(model)
        if h:
            val_losses = [s["val_loss"] for s in h if "val_loss" in s]
            metrics["Val Loss"].append(val_losses[-1] if val_losses else 0)
            amax = [s["composite_amax"] for s in h if "composite_amax" in s]
            metrics["Max Amax"].append(max(amax) if amax else 0)
        else:
            metrics["Val Loss"].append(0)
            metrics["Max Amax"].append(0)

    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, metrics["Val Loss"], width, label="Final Val Loss", color=COLORS["hc"], alpha=0.7)
    ax.bar(x + width/2, metrics["Max Amax"], width, label="Max Amax", color=COLORS["mhc"], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Value")
    ax.set_title("Final Metrics (16 layers)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "v4_stress_deep.png")
    plt.close()
    print(f"Saved v4_stress_deep.png")


def plot_stress_lr_comparison(output_dir: Path):
    """Plot aggressive LR stress test comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    histories = {}
    for model in ["hc", "mhc"]:
        histories[model] = load_history(Path(f"runs/v4_stress_lr/{model}/history.json"))

    # Training loss
    ax = axes[0]
    for model, h in histories.items():
        if h is None:
            continue
        steps = [s["step"] for s in h]
        losses = [s["loss"] for s in h]
        smoothed = smooth(losses)
        ax.plot(steps[19:], smoothed, color=COLORS[model], label=LABELS[model], lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss (LR=0.001)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Amax
    ax = axes[1]
    for model, h in histories.items():
        if h is None:
            continue
        steps = [s["step"] for s in h if "composite_amax" in s]
        amax = [s["composite_amax"] for s in h if "composite_amax" in s]
        smoothed = smooth(amax)
        ax.plot(steps[19:], smoothed, color=COLORS[model], label=LABELS[model], lw=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, label="Stability bound")
    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Amax")
    ax.set_title("Amax Gain (LR=0.001)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Final comparison
    ax = axes[2]
    models = ["HC", "mHC"]
    metrics = {"Val Loss": [], "Max Amax": []}
    for model in ["hc", "mhc"]:
        h = histories.get(model)
        if h:
            val_losses = [s["val_loss"] for s in h if "val_loss" in s]
            metrics["Val Loss"].append(val_losses[-1] if val_losses else 0)
            amax = [s["composite_amax"] for s in h if "composite_amax" in s]
            metrics["Max Amax"].append(max(amax) if amax else 0)
        else:
            metrics["Val Loss"].append(0)
            metrics["Max Amax"].append(0)

    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, metrics["Val Loss"], width, label="Final Val Loss", color=COLORS["hc"], alpha=0.7)
    ax.bar(x + width/2, metrics["Max Amax"], width, label="Max Amax", color=COLORS["mhc"], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Value")
    ax.set_title("Final Metrics (LR=0.001)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "v4_stress_lr.png")
    plt.close()
    print(f"Saved v4_stress_lr.png")


def plot_summary_comparison(output_dir: Path):
    """Plot summary comparison across all v4 experiments."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Load all histories
    baseline = {m: load_history(Path(f"runs/v4_baseline/{m}/history.json")) for m in ["residual", "hc", "mhc"]}
    stress_deep = {m: load_history(Path(f"runs/v4_stress_deep/{m}/history.json")) for m in ["hc", "mhc"]}
    stress_lr = {m: load_history(Path(f"runs/v4_stress_lr/{m}/history.json")) for m in ["hc", "mhc"]}

    # Row 1: Training losses
    for i, (title, histories) in enumerate([
        ("Baseline (6 layers)", baseline),
        ("16 Layers", stress_deep),
        ("Aggressive LR", stress_lr)
    ]):
        ax = axes[0, i]
        for model, h in histories.items():
            if h is None:
                continue
            steps = [s["step"] for s in h]
            losses = [s["loss"] for s in h]
            smoothed = smooth(losses)
            ax.plot(steps[19:], smoothed, color=COLORS[model], label=LABELS[model], lw=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Training Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2: Amax
    for i, (title, histories) in enumerate([
        ("Baseline Amax", baseline),
        ("16 Layers Amax", stress_deep),
        ("Aggressive LR Amax", stress_lr)
    ]):
        ax = axes[1, i]
        for model in ["hc", "mhc"]:
            h = histories.get(model)
            if h is None:
                continue
            steps = [s["step"] for s in h if "composite_amax" in s]
            amax = [s["composite_amax"] for s in h if "composite_amax" in s]
            if not steps:
                continue
            smoothed = smooth(amax)
            ax.plot(steps[19:], smoothed, color=COLORS[model], label=LABELS[model], lw=2)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Step")
        ax.set_ylabel("Composite Amax")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "v4_summary.png")
    plt.close()
    print(f"Saved v4_summary.png")


if __name__ == "__main__":
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    print("Generating v4 experiment figures...")
    plot_baseline_comparison(output_dir)
    plot_stress_deep_comparison(output_dir)
    plot_stress_lr_comparison(output_dir)
    plot_summary_comparison(output_dir)
    print(f"\nAll figures saved to {output_dir}")
