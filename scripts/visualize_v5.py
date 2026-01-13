"""
Generate v5 experiment comparison figures.

v5 experiments include:
- Depth sweep: HC at depths 6,8,10,12,14,16,20,24 (~11M params each)
- Seed variation: HC vs mHC at depth 24 with seeds 42, 123, 456
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLORS = {
    "residual": "#2ecc71",
    "hc": "#E63946",      # Red
    "mhc": "#2A9D8F",     # Teal
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


def plot_depth_sweep(output_dir: Path):
    """Plot critical depth sweep results (HC only)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Overall figure title
    fig.suptitle("Depth Sweep Results (HC Only, ~11M params)", fontsize=16, fontweight="bold", y=1.02)

    depths = [6, 8, 10, 12, 14, 16, 20, 24]
    dims = {6: 384, 8: 320, 10: 288, 12: 256, 14: 256, 16: 224, 20: 224, 24: 192}

    # Red color gradient for depth variations (matching HC = red convention)
    selected_depths = [6, 12, 20, 24]
    colors_depth = plt.cm.Reds(np.linspace(0.3, 0.9, len(selected_depths)))

    histories = {}
    for d in depths:
        histories[d] = load_history(Path(f"runs/v5_depth_sweep/depth_{d}/history.json"))

    # Final val loss vs depth
    ax = axes[0, 0]
    final_losses = []
    for d in depths:
        h = histories.get(d)
        if h:
            val_losses = [s["val_loss"] for s in h if "val_loss" in s]
            final_losses.append(val_losses[-1] if val_losses else None)
        else:
            final_losses.append(None)

    valid_depths = [d for d, l in zip(depths, final_losses) if l is not None]
    valid_losses = [l for l in final_losses if l is not None]

    ax.plot(valid_depths, valid_losses, "o-", color=COLORS["hc"], lw=2, ms=8, label="HC")
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Final Validation Loss")
    ax.set_title("(a) Val Loss vs Depth")
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Max Amax vs depth
    ax = axes[0, 1]
    max_amax_values = []
    for d in depths:
        h = histories.get(d)
        if h:
            amax = [s["composite_amax"] for s in h if "composite_amax" in s]
            max_amax_values.append(max(amax) if amax else None)
        else:
            max_amax_values.append(None)

    valid_depths_amax = [d for d, a in zip(depths, max_amax_values) if a is not None]
    valid_amax = [a for a in max_amax_values if a is not None]

    ax.plot(valid_depths_amax, valid_amax, "o-", color=COLORS["hc"], lw=2, ms=8, label="HC")
    ax.axhline(y=1.0, color="gray", linestyle="--", lw=2, alpha=0.8, label="Stability bound")
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Max Composite Amax")
    ax.set_title("(b) Max Amax vs Depth")
    ax.set_xticks(depths)
    ax.set_ylim(bottom=0)  # Start y-axis at 0
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Training curves for selected depths
    ax = axes[1, 0]
    for i, d in enumerate(selected_depths):
        h = histories.get(d)
        if h is None:
            continue
        steps = [s["step"] for s in h]
        losses = [s["loss"] for s in h]
        smoothed = smooth(losses)
        ax.plot(steps[19:], smoothed, color=colors_depth[i], label=f"D={d}", lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("(c) Training Loss by Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Amax evolution for selected depths
    ax = axes[1, 1]
    for i, d in enumerate(selected_depths):
        h = histories.get(d)
        if h is None:
            continue
        steps = [s["step"] for s in h if "composite_amax" in s]
        amax = [s["composite_amax"] for s in h if "composite_amax" in s]
        smoothed = smooth(amax)
        ax.plot(steps[19:], smoothed, color=colors_depth[i], label=f"D={d}", lw=1.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", lw=2, alpha=0.8, label="Stability bound")
    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Amax")
    ax.set_title("(d) Amax Evolution by Depth")
    ax.set_ylim(bottom=0)  # Start y-axis at 0
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "v5_depth_sweep.png")
    plt.close()
    print(f"Saved v5_depth_sweep.png")


def plot_seed_variation(output_dir: Path):
    """Plot seed variation results (HC vs mHC at depth 24)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Overall figure title
    fig.suptitle("Seed Variation Results (Depth 24, n=3)", fontsize=16, fontweight="bold", y=1.02)

    seeds = [42, 123, 456]

    hc_histories = {s: load_history(Path(f"runs/v5_seed_variation/hc_seed_{s}/history.json")) for s in seeds}
    mhc_histories = {s: load_history(Path(f"runs/v5_seed_variation/mhc_seed_{s}/history.json")) for s in seeds}

    # Final val loss comparison with error bars
    ax = axes[0, 0]

    hc_losses = []
    mhc_losses = []
    for s in seeds:
        h = hc_histories.get(s)
        if h:
            val_losses = [x["val_loss"] for x in h if "val_loss" in x]
            hc_losses.append(val_losses[-1] if val_losses else np.nan)
        h = mhc_histories.get(s)
        if h:
            val_losses = [x["val_loss"] for x in h if "val_loss" in x]
            mhc_losses.append(val_losses[-1] if val_losses else np.nan)

    x = np.arange(2)
    means = [np.mean(hc_losses), np.mean(mhc_losses)]
    stds = [np.std(hc_losses), np.std(mhc_losses)]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=[COLORS["hc"], COLORS["mhc"]], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["HC", "mHC"])
    ax.set_ylabel("Final Validation Loss")
    ax.set_title("(a) Final Validation Loss")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val, std in zip(bars, means, stds):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val + std),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Max Amax comparison
    ax = axes[0, 1]

    hc_amax = []
    mhc_amax = []
    for s in seeds:
        h = hc_histories.get(s)
        if h:
            amax = [x["composite_amax"] for x in h if "composite_amax" in x]
            hc_amax.append(max(amax) if amax else np.nan)
        h = mhc_histories.get(s)
        if h:
            amax = [x["composite_amax"] for x in h if "composite_amax" in x]
            mhc_amax.append(max(amax) if amax else np.nan)

    means_amax = [np.mean(hc_amax), np.mean(mhc_amax)]
    stds_amax = [np.std(hc_amax), np.std(mhc_amax)]

    bars = ax.bar(x, means_amax, yerr=stds_amax, capsize=5, color=[COLORS["hc"], COLORS["mhc"]], alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", lw=2, alpha=0.8, label="Stability bound")
    ax.set_xticks(x)
    ax.set_xticklabels(["HC", "mHC"])
    ax.set_ylabel("Max Composite Amax")
    ax.set_title("(b) Max Amax")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val, std in zip(bars, means_amax, stds_amax):
        label = f'{val:.2f}' if val > 1.5 else f'{val:.2f}'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, val + std),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Training curves by seed
    ax = axes[1, 0]
    linestyles = ["-", "--", ":"]
    for i, s in enumerate(seeds):
        h = hc_histories.get(s)
        if h:
            steps = [x["step"] for x in h]
            losses = [x["loss"] for x in h]
            smoothed = smooth(losses)
            ax.plot(steps[19:], smoothed, color=COLORS["hc"], linestyle=linestyles[i],
                   label=f"HC seed {s}", lw=1.5, alpha=0.8)
        h = mhc_histories.get(s)
        if h:
            steps = [x["step"] for x in h]
            losses = [x["loss"] for x in h]
            smoothed = smooth(losses)
            ax.plot(steps[19:], smoothed, color=COLORS["mhc"], linestyle=linestyles[i],
                   label=f"mHC seed {s}", lw=1.5, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("(c) Training Loss by Seed")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Amax evolution by seed - simplified legend
    ax = axes[1, 1]
    # Plot all HC seeds with same style, only label first one
    for i, s in enumerate(seeds):
        h = hc_histories.get(s)
        if h:
            steps = [x["step"] for x in h if "composite_amax" in x]
            amax = [x["composite_amax"] for x in h if "composite_amax" in x]
            smoothed = smooth(amax)
            label = "HC (3 seeds)" if i == 0 else None
            ax.plot(steps[19:], smoothed, color=COLORS["hc"], linestyle=linestyles[i],
                   label=label, lw=1.5, alpha=0.8)
    # Plot all mHC seeds with same style, only label first one
    for i, s in enumerate(seeds):
        h = mhc_histories.get(s)
        if h:
            steps = [x["step"] for x in h if "composite_amax" in x]
            amax = [x["composite_amax"] for x in h if "composite_amax" in x]
            smoothed = smooth(amax)
            label = "mHC (3 seeds)" if i == 0 else None
            ax.plot(steps[19:], smoothed, color=COLORS["mhc"], linestyle=linestyles[i],
                   label=label, lw=1.5, alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", lw=2, alpha=0.8, label="Stability bound")
    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Amax")
    ax.set_title("(d) Amax Evolution by Seed")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "v5_seed_variation.png")
    plt.close()
    print(f"Saved v5_seed_variation.png")


def plot_publication_summary(output_dir: Path):
    """Generate a publication-ready summary figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Overall figure title
    fig.suptitle("mHC Reproduction: 10M Parameter Results", fontsize=16, fontweight="bold", y=1.02)

    # Load data
    depths = [6, 8, 10, 12, 14, 16, 20, 24]
    depth_histories = {d: load_history(Path(f"runs/v5_depth_sweep/depth_{d}/history.json")) for d in depths}

    seeds = [42, 123, 456]
    hc_seed_histories = {s: load_history(Path(f"runs/v5_seed_variation/hc_seed_{s}/history.json")) for s in seeds}
    mhc_seed_histories = {s: load_history(Path(f"runs/v5_seed_variation/mhc_seed_{s}/history.json")) for s in seeds}

    # Use red gradient for depth variations
    colors_depth = plt.cm.Reds(np.linspace(0.3, 0.9, len([6, 12, 20, 24])))

    # Panel 1 (a): Val loss vs depth
    ax = axes[0, 0]
    final_losses = []
    for d in depths:
        h = depth_histories.get(d)
        if h:
            val_losses = [s["val_loss"] for s in h if "val_loss" in s]
            final_losses.append(val_losses[-1] if val_losses else None)
        else:
            final_losses.append(None)

    valid_depths = [d for d, l in zip(depths, final_losses) if l is not None]
    valid_losses = [l for l in final_losses if l is not None]

    ax.plot(valid_depths, valid_losses, "o-", color=COLORS["hc"], lw=2, ms=8)
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Final Validation Loss")
    ax.set_title("(a) HC Val Loss vs Depth")
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3)

    # Panel 2 (b): Max Amax vs depth - y-axis starts at 0
    ax = axes[0, 1]
    max_amax_values = []
    for d in depths:
        h = depth_histories.get(d)
        if h:
            amax = [s["composite_amax"] for s in h if "composite_amax" in s]
            max_amax_values.append(max(amax) if amax else None)
        else:
            max_amax_values.append(None)

    valid_depths_amax = [d for d, a in zip(depths, max_amax_values) if a is not None]
    valid_amax = [a for a in max_amax_values if a is not None]

    ax.plot(valid_depths_amax, valid_amax, "o-", color=COLORS["hc"], lw=2, ms=8)
    ax.axhline(y=1.0, color="gray", linestyle="--", lw=2, alpha=0.8, label="Amax = 1.0 (stability bound)")
    ax.set_xlabel("Depth (layers)")
    ax.set_ylabel("Max Composite Amax")
    ax.set_title("(b) HC Max Amax vs Depth")
    ax.set_xticks(depths)
    ax.set_ylim(bottom=0)  # Start y-axis at 0
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Panel 3 (c): Amax evolution comparison (SWAPPED - was panel f)
    ax = axes[0, 2]

    hc_all_amax = []
    mhc_all_amax = []
    for s in seeds:
        h = hc_seed_histories.get(s)
        if h:
            hc_all_amax.append([x["composite_amax"] for x in h if "composite_amax" in x])
        h = mhc_seed_histories.get(s)
        if h:
            mhc_all_amax.append([x["composite_amax"] for x in h if "composite_amax" in x])

    if hc_all_amax:
        min_len = min(len(l) for l in hc_all_amax)
        hc_mean = np.mean([l[:min_len] for l in hc_all_amax], axis=0)
        hc_std = np.std([l[:min_len] for l in hc_all_amax], axis=0)
        steps = list(range(len(hc_mean)))
        smoothed_mean = smooth(hc_mean)
        smoothed_std = smooth(hc_std)
        ax.plot(steps[19:], smoothed_mean, color=COLORS["hc"], lw=2, label="HC")
        ax.fill_between(steps[19:], smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                       color=COLORS["hc"], alpha=0.2)

    if mhc_all_amax:
        min_len = min(len(l) for l in mhc_all_amax)
        mhc_mean = np.mean([l[:min_len] for l in mhc_all_amax], axis=0)
        mhc_std = np.std([l[:min_len] for l in mhc_all_amax], axis=0)
        steps = list(range(len(mhc_mean)))
        smoothed_mean = smooth(mhc_mean)
        smoothed_std = smooth(mhc_std)
        ax.plot(steps[19:], smoothed_mean, color=COLORS["mhc"], lw=2, label="mHC")
        ax.fill_between(steps[19:], smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                       color=COLORS["mhc"], alpha=0.2)

    ax.axhline(y=1.0, color="gray", linestyle="--", lw=2, alpha=0.8, label="Stability bound")
    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Amax")
    ax.set_title("(c) Amax Evolution: The Key Tradeoff")
    ax.set_ylim(bottom=0)  # Start y-axis at 0
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4 (d): Selected depth training curves
    ax = axes[1, 0]
    selected_depths = [6, 12, 20, 24]
    for i, d in enumerate(selected_depths):
        h = depth_histories.get(d)
        if h is None:
            continue
        steps = [s["step"] for s in h]
        losses = [s["loss"] for s in h]
        smoothed = smooth(losses)
        ax.plot(steps[19:], smoothed, color=colors_depth[i], label=f"D={d}", lw=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("(d) Training Loss by Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 5 (e): HC vs mHC training curves (mean over seeds)
    ax = axes[1, 1]

    # Compute mean training curves
    hc_all_losses = []
    mhc_all_losses = []
    for s in seeds:
        h = hc_seed_histories.get(s)
        if h:
            hc_all_losses.append([x["loss"] for x in h])
        h = mhc_seed_histories.get(s)
        if h:
            mhc_all_losses.append([x["loss"] for x in h])

    if hc_all_losses:
        min_len = min(len(l) for l in hc_all_losses)
        hc_mean = np.mean([l[:min_len] for l in hc_all_losses], axis=0)
        hc_std = np.std([l[:min_len] for l in hc_all_losses], axis=0)
        steps = list(range(len(hc_mean)))
        smoothed_mean = smooth(hc_mean)
        smoothed_std = smooth(hc_std)
        ax.plot(steps[19:], smoothed_mean, color=COLORS["hc"], lw=2, label="HC")
        ax.fill_between(steps[19:], smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                       color=COLORS["hc"], alpha=0.2)

    if mhc_all_losses:
        min_len = min(len(l) for l in mhc_all_losses)
        mhc_mean = np.mean([l[:min_len] for l in mhc_all_losses], axis=0)
        mhc_std = np.std([l[:min_len] for l in mhc_all_losses], axis=0)
        steps = list(range(len(mhc_mean)))
        smoothed_mean = smooth(mhc_mean)
        smoothed_std = smooth(mhc_std)
        ax.plot(steps[19:], smoothed_mean, color=COLORS["mhc"], lw=2, label="mHC")
        ax.fill_between(steps[19:], smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                       color=COLORS["mhc"], alpha=0.2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("(e) HC vs mHC Training (Mean +/- Std)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 6 (f): HC vs mHC final comparison (SWAPPED - was panel c)
    ax = axes[1, 2]

    hc_losses = []
    mhc_losses = []
    for s in seeds:
        h = hc_seed_histories.get(s)
        if h:
            val_losses = [x["val_loss"] for x in h if "val_loss" in x]
            hc_losses.append(val_losses[-1] if val_losses else np.nan)
        h = mhc_seed_histories.get(s)
        if h:
            val_losses = [x["val_loss"] for x in h if "val_loss" in x]
            mhc_losses.append(val_losses[-1] if val_losses else np.nan)

    x = np.arange(2)
    means = [np.mean(hc_losses), np.mean(mhc_losses)]
    stds = [np.std(hc_losses), np.std(mhc_losses)]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=[COLORS["hc"], COLORS["mhc"]], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["HC", "mHC"])
    ax.set_ylabel("Final Validation Loss (lower = better)")
    ax.set_title("(f) HC vs mHC at Depth 24")
    ax.grid(True, alpha=0.3, axis="y")

    # Add annotation about the tradeoff
    ax.annotate("HC wins on loss,\nbut at what cost?",
                xy=(0.5, 0.95), xycoords="axes fraction",
                ha="center", va="top", fontsize=10, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    # Add value labels on bars
    for bar, val, std in zip(bars, means, stds):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val - std - 0.02),
                   ha='center', va='top', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "v5_publication_summary.png")
    plt.close()
    print(f"Saved v5_publication_summary.png")


def print_results_table():
    """Print a summary table of all results."""
    print("\n" + "="*70)
    print("V5 EXPERIMENT RESULTS SUMMARY")
    print("="*70)

    # Depth sweep
    print("\n1. Depth Sweep (HC, ~11M params each):")
    print("-" * 50)
    print(f"{'Depth':<10} {'Dim':<10} {'Final Val Loss':<15} {'Max Amax':<12}")
    print("-" * 50)

    depths = [6, 8, 10, 12, 14, 16, 20, 24]
    dims = {6: 384, 8: 320, 10: 288, 12: 256, 14: 256, 16: 224, 20: 224, 24: 192}

    for d in depths:
        h = load_history(Path(f"runs/v5_depth_sweep/depth_{d}/history.json"))
        if h:
            val_losses = [s["val_loss"] for s in h if "val_loss" in s]
            amax = [s["composite_amax"] for s in h if "composite_amax" in s]
            final_loss = val_losses[-1] if val_losses else "N/A"
            max_amax = max(amax) if amax else "N/A"
            print(f"{d:<10} {dims[d]:<10} {final_loss:<15.4f} {max_amax:<12.4f}")

    # Seed variation
    print("\n2. Seed Variation (Depth 24, Dim 192):")
    print("-" * 50)
    print(f"{'Model':<10} {'Seed':<10} {'Final Val Loss':<15} {'Max Amax':<12}")
    print("-" * 50)

    seeds = [42, 123, 456]
    for model in ["hc", "mhc"]:
        for s in seeds:
            h = load_history(Path(f"runs/v5_seed_variation/{model}_seed_{s}/history.json"))
            if h:
                val_losses = [x["val_loss"] for x in h if "val_loss" in x]
                amax = [x["composite_amax"] for x in h if "composite_amax" in x]
                final_loss = val_losses[-1] if val_losses else "N/A"
                max_amax = max(amax) if amax else "N/A"
                print(f"{model.upper():<10} {s:<10} {final_loss:<15.4f} {max_amax:<12.4f}")

    # Summary statistics
    print("\n3. Summary Statistics (Depth 24):")
    print("-" * 50)

    hc_losses = []
    mhc_losses = []
    hc_amax = []
    mhc_amax = []

    for s in seeds:
        h = load_history(Path(f"runs/v5_seed_variation/hc_seed_{s}/history.json"))
        if h:
            val_losses = [x["val_loss"] for x in h if "val_loss" in x]
            amax = [x["composite_amax"] for x in h if "composite_amax" in x]
            hc_losses.append(val_losses[-1])
            hc_amax.append(max(amax))
        h = load_history(Path(f"runs/v5_seed_variation/mhc_seed_{s}/history.json"))
        if h:
            val_losses = [x["val_loss"] for x in h if "val_loss" in x]
            amax = [x["composite_amax"] for x in h if "composite_amax" in x]
            mhc_losses.append(val_losses[-1])
            mhc_amax.append(max(amax))

    print(f"HC  Val Loss: {np.mean(hc_losses):.4f} +/- {np.std(hc_losses):.4f}")
    print(f"mHC Val Loss: {np.mean(mhc_losses):.4f} +/- {np.std(mhc_losses):.4f}")
    print(f"HC  Max Amax: {np.mean(hc_amax):.4f} +/- {np.std(hc_amax):.4f}")
    print(f"mHC Max Amax: {np.mean(mhc_amax):.4f} +/- {np.std(mhc_amax):.4f}")

    print("\n" + "="*70)


def plot_stress_lr(output_dir: Path):
    """Plot stress test results with aggressive learning rate (3x baseline)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Overall figure title with LR details
    fig.suptitle("Stress Test: Aggressive Learning Rate (LR=0.001, 3x baseline)",
                 fontsize=16, fontweight="bold", y=1.02)

    # Load data from v4_stress_lr
    hc_history = load_history(Path("runs/v4_stress_lr/hc/history.json"))
    mhc_history = load_history(Path("runs/v4_stress_lr/mhc/history.json"))

    # Panel (a): Training Loss
    ax = axes[0]

    if hc_history:
        steps = [s["step"] for s in hc_history]
        losses = [s["loss"] for s in hc_history]
        smoothed = smooth(losses)
        ax.plot(steps[19:], smoothed, color=COLORS["hc"], lw=2, label="HC")

    if mhc_history:
        steps = [s["step"] for s in mhc_history]
        losses = [s["loss"] for s in mhc_history]
        smoothed = smooth(losses)
        ax.plot(steps[19:], smoothed, color=COLORS["mhc"], lw=2, label="mHC")

    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("(a) Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel (b): Amax Evolution
    ax = axes[1]

    hc_max_amax = 1.0
    hc_max_step = 0
    if hc_history:
        steps = [s["step"] for s in hc_history if "composite_amax" in s]
        amax = [s["composite_amax"] for s in hc_history if "composite_amax" in s]
        hc_max_amax = max(amax)
        hc_max_step = steps[amax.index(hc_max_amax)]
        smoothed = smooth(amax)
        ax.plot(steps[19:], smoothed, color=COLORS["hc"], lw=2, label="HC")

    if mhc_history:
        steps = [s["step"] for s in mhc_history if "composite_amax" in s]
        amax = [s["composite_amax"] for s in mhc_history if "composite_amax" in s]
        smoothed = smooth(amax)
        ax.plot(steps[19:], smoothed, color=COLORS["mhc"], lw=2, label="mHC")

    ax.axhline(y=1.0, color="gray", linestyle="--", lw=2, alpha=0.8, label="Stability bound")

    # Annotate HC peak
    ax.annotate(f"Peak: {hc_max_amax:.1f}x",
                xy=(hc_max_step, hc_max_amax),
                xytext=(hc_max_step + 500, hc_max_amax + 0.5),
                fontsize=10, fontweight="bold", color=COLORS["hc"],
                arrowprops=dict(arrowstyle="->", color=COLORS["hc"], lw=1.5))

    # Annotate mHC stability
    ax.annotate("Flat at 1.0",
                xy=(3000, 1.0),
                xytext=(3000, 2.5),
                fontsize=10, fontweight="bold", color=COLORS["mhc"],
                arrowprops=dict(arrowstyle="->", color=COLORS["mhc"], lw=1.5))

    ax.set_xlabel("Step")
    ax.set_ylabel("Composite Amax")
    ax.set_title("(b) Amax Evolution")
    ax.set_ylim(bottom=0, top=9)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "v5_stress_lr.png")
    plt.close()
    print(f"Saved v5_stress_lr.png")


if __name__ == "__main__":
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    print("Generating v5 experiment figures...")
    plot_depth_sweep(output_dir)
    plot_seed_variation(output_dir)
    plot_publication_summary(output_dir)
    print_results_table()
    print(f"\nAll figures saved to {output_dir}")
