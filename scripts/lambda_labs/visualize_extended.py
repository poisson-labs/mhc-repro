#!/usr/bin/env python3
"""
Extended visualization for Part 2 blog post.

Generates additional figures beyond the main comparison:
1. Scaling law plot (10M → 1.7B log-log projection)
2. DeepSeek comparison chart (positioning our results)
3. Stress test trajectories (the "ticking bomb" plot)
4. Animation data exports for SVG interactives

Also exports JSON data for SVG animations:
- amax_counter_data.json: Time series for HC vs mHC counter animation
- layer_heatmap_data.json: Per-layer Amax for heatmap timelapse
- signal_flow_data.json: Layer-by-layer signal amplification

Part of the mHC reproduction series: https://poissonlabs.ai/research/mhc-reproduction-part-2/
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List

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
    "deepseek": "#264653",  # Dark blue
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

    try:
        with open(path) as f:
            content = f.read().strip()
            # Handle potential trailing issues
            if content.endswith(','):
                content = content[:-1]
            if not content.endswith(']'):
                content = content + ']'
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Warning: {path} corrupted ({e})")
        return None


def load_experiment(run_dir: Path, method: str, depth: int, seed: int) -> Optional[dict]:
    """Load a single experiment's data."""
    exp_name = f"{method}_d{depth}_s{seed}"
    exp_dir = run_dir / exp_name

    history_path = exp_dir / "history.json"
    history = load_history(history_path)
    if history is None:
        return None

    return {
        "name": exp_name,
        "method": method,
        "depth": depth,
        "seed": seed,
        "history": history,
    }


# =============================================================================
# Figure 1: Scaling Law Plot
# =============================================================================

def generate_scaling_law_figure(output_dir: Path):
    """
    Log-log plot showing Amax scaling with parameters.

    Data points:
    - Part 1: 10M params → 9.2x Amax (TinyShakespeare)
    - Part 2: 1.73B params → 10,924x Amax (C4, d32 mean)
    - Part 2: 2.54B params → 3,721x Amax (C4, d48 mean)
    - DeepSeek: 27B params → 3000x Amax (reported)

    Projects trendline to 10B and 100B.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Our data points
    our_data = {
        "params": [10e6, 1.73e9, 2.54e9],
        "amax": [9.2, 10924, 3721],
        "labels": ["Part 1\n(10M)", "Part 2 d32\n(1.7B)", "Part 2 d48\n(2.5B)"],
    }

    # DeepSeek data point
    deepseek_data = {
        "params": [27e9],
        "amax": [3000],
        "labels": ["DeepSeek\n(27B)"],
    }

    # Plot our data
    ax.scatter(our_data["params"], our_data["amax"],
               s=150, c=COLORS["hc"], marker="o", zorder=5, label="Our HC experiments")

    for p, a, l in zip(our_data["params"], our_data["amax"], our_data["labels"]):
        ax.annotate(l, (p, a), textcoords="offset points", xytext=(10, 10),
                   fontsize=9, color=COLORS["hc"])

    # Plot DeepSeek data
    ax.scatter(deepseek_data["params"], deepseek_data["amax"],
               s=150, c=COLORS["deepseek"], marker="s", zorder=5, label="DeepSeek (reported)")

    ax.annotate(deepseek_data["labels"][0], (deepseek_data["params"][0], deepseek_data["amax"][0]),
               textcoords="offset points", xytext=(10, -15), fontsize=9, color=COLORS["deepseek"])

    # Fit trendline through Part 1 and Part 2 d32 (exclude d48 as it's different batch size)
    fit_params = [10e6, 1.73e9]
    fit_amax = [9.2, 10924]

    log_params = np.log10(fit_params)
    log_amax = np.log10(fit_amax)
    slope, intercept = np.polyfit(log_params, log_amax, 1)

    # Project trendline
    project_params = np.logspace(7, 11, 100)  # 10M to 100B
    project_amax = 10 ** (slope * np.log10(project_params) + intercept)

    ax.plot(project_params, project_amax, '--', color=COLORS["hc"], alpha=0.5,
            label=f"Projection (slope={slope:.2f})")

    # Add annotations for projected values
    for target_params, target_label in [(10e9, "10B?"), (100e9, "100B?")]:
        target_amax = 10 ** (slope * np.log10(target_params) + intercept)
        ax.scatter([target_params], [target_amax], s=80, c=COLORS["hc"],
                   marker="x", alpha=0.5, zorder=4)
        ax.annotate(f"{target_label}\n({target_amax/1000:.0f}kx)",
                   (target_params, target_amax),
                   textcoords="offset points", xytext=(10, 5),
                   fontsize=9, color=COLORS["hc"], alpha=0.7)

    # Styling
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Max Amax (signal amplification)")
    ax.set_title("HC Instability Scales with Model Size")

    # Reference line at 1.0
    ax.axhline(y=1.0, color="black", linestyle="-", alpha=0.3, linewidth=2)
    ax.text(5e10, 1.2, "Stable (Amax = 1)", fontsize=9, alpha=0.5)

    # Reference line at DeepSeek's 3000x
    ax.axhline(y=3000, color=COLORS["deepseek"], linestyle=":", alpha=0.3, linewidth=1)
    ax.text(5e10, 3500, "DeepSeek threshold", fontsize=8, alpha=0.5, color=COLORS["deepseek"])

    ax.legend(loc="upper left")
    ax.set_xlim(5e6, 2e11)
    ax.set_ylim(1, 5e5)

    # Custom x-axis labels
    ax.set_xticks([1e7, 1e8, 1e9, 1e10, 1e11])
    ax.set_xticklabels(["10M", "100M", "1B", "10B", "100B"])

    fig.tight_layout()
    fig.savefig(output_dir / "scaling_law.png")
    fig.savefig(output_dir / "scaling_law.pdf")
    print(f"Saved: {output_dir / 'scaling_law.png'}")
    plt.close(fig)


# =============================================================================
# Figure 2: Stress Test Trajectories
# =============================================================================

def generate_stress_test_figure(stress_dir: Path, output_dir: Path):
    """
    Plot stress test Amax trajectories showing the "ticking bomb" phenomenon.

    Key insight: HC hit 14,765x Amax at d64 but DIDN'T NaN!
    The bomb is ticking but hasn't exploded... yet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_amax, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    # Load stress test data
    stress_runs = [
        ("hc_d32_s42", "HC d32 @ 3x LR", COLORS["hc"], "-"),
        ("hc_d48_s42", "HC d48 @ 3x LR", COLORS["hc"], "--"),
        ("hc_d64_s42", "HC d64 @ 3x LR", COLORS["hc"], ":"),
        ("mhc_d32_s42", "mHC d32 @ 3x LR", COLORS["mhc"], "-"),
        ("mhc_d48_s42", "mHC d48 @ 3x LR", COLORS["mhc"], "--"),
    ]

    max_amax_seen = 1

    for run_name, label, color, linestyle in stress_runs:
        history = load_history(stress_dir / run_name / "history.json")
        if history is None:
            print(f"Skipping {run_name}: no data")
            continue

        steps = [m["step"] for m in history]
        amax = [m.get("composite_amax", 1.0) for m in history]
        loss = [m["loss"] for m in history]

        max_amax_seen = max(max_amax_seen, max(amax))

        ax_amax.plot(steps, amax, color=color, linestyle=linestyle,
                     label=label, linewidth=1.5)
        ax_loss.plot(steps, loss, color=color, linestyle=linestyle,
                     label=label, linewidth=1.5)

    # Amax panel
    ax_amax.set_xlabel("Step")
    ax_amax.set_ylabel("Amax (log scale)")
    ax_amax.set_title("Stress Test: Amax Trajectories")
    ax_amax.set_yscale("log")
    ax_amax.axhline(y=1.0, color="black", linestyle="-", alpha=0.3, linewidth=2)
    ax_amax.legend(loc="upper left")

    # Add annotation for max
    ax_amax.annotate(f"Peak: {max_amax_seen:,.0f}x",
                    xy=(0.95, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=12, fontweight="bold",
                    color=COLORS["hc"])

    # Loss panel
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Stress Test: Loss Curves (No NaN!)")
    ax_loss.legend(loc="upper right")

    # Add annotation
    ax_loss.annotate("Training continued\ndespite 14,765x Amax",
                    xy=(0.5, 0.3), xycoords="axes fraction",
                    ha="center", va="center", fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    fig.tight_layout()
    fig.savefig(output_dir / "stress_test.png")
    fig.savefig(output_dir / "stress_test.pdf")
    print(f"Saved: {output_dir / 'stress_test.png'}")
    plt.close(fig)


# =============================================================================
# Figure 3: The "Bomb Didn't Go Off" Analysis
# =============================================================================

def generate_bomb_analysis_figure(run_dir: Path, stress_dir: Path, output_dir: Path):
    """
    Show that despite extreme Amax, loss stays finite.

    Clean two-panel design: mHC on left (stable), HC on right (chaos).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_mhc, ax_hc) = plt.subplots(1, 2, figsize=(12, 5))

    # Collect HC runs (main + stress)
    hc_runs = []
    for depth in [32, 48]:
        for seed in [42, 123, 456]:
            exp = load_experiment(run_dir, "hc", depth, seed)
            if exp:
                max_amax = max(m.get("composite_amax", 1.0) for m in exp["history"])
                final_loss = exp["history"][-1]["loss"]
                hc_runs.append((f"d{depth}", max_amax, final_loss, "main"))

    # Stress runs
    for run_name, label in [("hc_d32_s42", "d32"), ("hc_d48_s42", "d48"), ("hc_d64_s42", "d64")]:
        history = load_history(stress_dir / run_name / "history.json")
        if history:
            max_amax = max(m.get("composite_amax", 1.0) for m in history)
            final_loss = history[-1]["loss"]
            hc_runs.append((label, max_amax, final_loss, "stress"))

    # Collect mHC runs
    mhc_runs = []
    for depth in [32, 48]:
        for seed in [42, 123, 456]:
            exp = load_experiment(run_dir, "mhc", depth, seed)
            if exp:
                max_amax = max(m.get("composite_amax", 1.0) for m in exp["history"])
                final_loss = exp["history"][-1]["loss"]
                mhc_runs.append((f"d{depth}", max_amax, final_loss))

    # Left panel: mHC (stable)
    ax_mhc.set_facecolor("#f0f9f0")  # Light green background
    for label, amax, loss, in mhc_runs:
        ax_mhc.scatter(amax, loss, s=120, c=COLORS["mhc"], marker="o", alpha=0.8)

    ax_mhc.set_xlim(0.99, 1.01)
    ax_mhc.set_ylim(5.3, 6.5)
    ax_mhc.set_xlabel("Max Amax")
    ax_mhc.set_ylabel("Final Loss")
    ax_mhc.set_title("mHC: Stable", fontweight="bold", color=COLORS["mhc"])
    ax_mhc.axvline(x=1.0, color=COLORS["mhc"], linestyle="-", alpha=0.3, linewidth=2)
    ax_mhc.text(1.0, 5.35, "Amax = 1.0\n(always)", ha="center", fontsize=10,
               color=COLORS["mhc"], fontweight="bold")

    # Right panel: HC (chaos)
    ax_hc.set_facecolor("#fff0f0")  # Light red background
    for label, amax, loss, run_type in hc_runs:
        marker = "o" if run_type == "main" else "^"
        ax_hc.scatter(amax, loss, s=120, c=COLORS["hc"], marker=marker, alpha=0.8)

    ax_hc.set_xscale("log")
    ax_hc.set_xlim(1000, 20000)
    ax_hc.set_ylim(5.3, 6.5)
    ax_hc.set_xlabel("Max Amax (log scale)")
    ax_hc.set_ylabel("")
    ax_hc.set_title("HC: 3,000x - 14,000x", fontweight="bold", color=COLORS["hc"])

    # Add "still training" annotation
    ax_hc.annotate("Still training.\nNo NaN, no crash.",
                   xy=(10000, 5.5), fontsize=11, ha="center",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                            edgecolor=COLORS["hc"], linewidth=2))

    # Legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["hc"],
               markersize=10, label='Main (5k steps)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS["hc"],
               markersize=10, label='Stress (3x LR)'),
    ]
    ax_hc.legend(handles=legend_elements, loc="upper left")

    fig.suptitle("The Bomb That Didn't Go Off (Yet)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "bomb_analysis.png")
    print(f"Saved: {output_dir / 'bomb_analysis.png'}")
    plt.close(fig)


# =============================================================================
# Animation Data Exports
# =============================================================================

def export_amax_counter_data(run_dir: Path, output_dir: Path):
    """
    Export data for the Amax counter SVG animation.

    Output: JSON with step-by-step Amax values for HC and mHC.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load HC d32 s42 (the canonical run)
    hc_exp = load_experiment(run_dir, "hc", 32, 42)
    mhc_exp = load_experiment(run_dir, "mhc", 32, 42)

    if not hc_exp or not mhc_exp:
        print("Warning: Could not load experiments for counter animation")
        return

    # Sample every 50 steps for smoother animation
    sample_rate = 50

    data = {
        "description": "Amax counter animation data. HC climbs while mHC stays flat.",
        "sample_rate": sample_rate,
        "total_steps": len(hc_exp["history"]),
        "frames": []
    }

    for i in range(0, len(hc_exp["history"]), sample_rate):
        hc_entry = hc_exp["history"][i]
        mhc_entry = mhc_exp["history"][i]

        data["frames"].append({
            "step": hc_entry["step"],
            "hc_amax": round(hc_entry.get("composite_amax", 1.0), 2),
            "mhc_amax": round(mhc_entry.get("composite_amax", 1.0), 6),
            "hc_loss": round(hc_entry["loss"], 3),
            "mhc_loss": round(mhc_entry["loss"], 3),
        })

    output_path = output_dir / "amax_counter_data.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


def export_layer_heatmap_data(run_dir: Path, output_dir: Path, depth: int = 48):
    """
    Export data for the layer heatmap timelapse SVG animation.

    Output: JSON with per-layer Amax values over training.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    hc_exp = load_experiment(run_dir, "hc", depth, 42)
    mhc_exp = load_experiment(run_dir, "mhc", depth, 42)

    if not hc_exp or not mhc_exp:
        print("Warning: Could not load experiments for heatmap animation")
        return

    # Sample every 100 steps
    sample_rate = 100

    data = {
        "description": f"Per-layer Amax heatmap data for depth {depth}. Shows instability starting at Layer 0.",
        "depth": depth,
        "sample_rate": sample_rate,
        "total_steps": len(hc_exp["history"]),
        "hc_frames": [],
        "mhc_frames": [],
    }

    for i in range(0, len(hc_exp["history"]), sample_rate):
        hc_entry = hc_exp["history"][i]
        mhc_entry = mhc_exp["history"][i]

        # Extract per-layer Amax (max of attn and ffn)
        hc_layer_amax = []
        mhc_layer_amax = []

        for layer_data in hc_entry["layer_amax"]:
            hc_layer_amax.append(round(max(layer_data["attn"], layer_data["ffn"]), 3))

        for layer_data in mhc_entry["layer_amax"]:
            mhc_layer_amax.append(round(max(layer_data["attn"], layer_data["ffn"]), 3))

        data["hc_frames"].append({
            "step": hc_entry["step"],
            "layer_amax": hc_layer_amax,
        })

        data["mhc_frames"].append({
            "step": mhc_entry["step"],
            "layer_amax": mhc_layer_amax,
        })

    output_path = output_dir / "layer_heatmap_data.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


def export_signal_flow_data(run_dir: Path, output_dir: Path, depth: int = 32):
    """
    Export data for the signal flow SVG animation.

    Shows how a signal grows as it passes through layers in HC vs mHC.
    We use cumulative Amax product to simulate signal amplification.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    hc_exp = load_experiment(run_dir, "hc", depth, 42)
    mhc_exp = load_experiment(run_dir, "mhc", depth, 42)

    if not hc_exp or not mhc_exp:
        print("Warning: Could not load experiments for signal flow animation")
        return

    # Take snapshot at step 3000 (well into training)
    snapshot_step = 3000

    hc_entry = None
    mhc_entry = None
    for entry in hc_exp["history"]:
        if entry["step"] >= snapshot_step:
            hc_entry = entry
            break
    for entry in mhc_exp["history"]:
        if entry["step"] >= snapshot_step:
            mhc_entry = entry
            break

    if not hc_entry or not mhc_entry:
        print("Warning: Could not find snapshot step for signal flow")
        return

    # Calculate cumulative signal magnitude through layers
    # Assume input signal = 1.0, multiply by each layer's Amax
    hc_signal = [1.0]
    mhc_signal = [1.0]

    for layer_data in hc_entry["layer_amax"]:
        layer_amax = max(layer_data["attn"], layer_data["ffn"])
        hc_signal.append(hc_signal[-1] * layer_amax)

    for layer_data in mhc_entry["layer_amax"]:
        layer_amax = max(layer_data["attn"], layer_data["ffn"])
        mhc_signal.append(mhc_signal[-1] * layer_amax)

    data = {
        "description": f"Signal flow through {depth} layers at step {snapshot_step}. "
                       "Shows cumulative amplification.",
        "depth": depth,
        "snapshot_step": snapshot_step,
        "layers": list(range(depth + 1)),  # 0 = input, 1-N = after each layer
        "hc_signal": [round(s, 2) for s in hc_signal],
        "mhc_signal": [round(s, 6) for s in mhc_signal],
        "hc_final_amplification": round(hc_signal[-1], 2),
        "mhc_final_amplification": round(mhc_signal[-1], 6),
    }

    output_path = output_dir / "signal_flow_data.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


def export_stress_racing_data(stress_dir: Path, output_dir: Path):
    """
    Export data for the stress test "racing lines" SVG animation.

    Shows multiple HC configs racing upward while mHC stays flat.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stress_runs = [
        ("hc_d32_s42", "HC d32"),
        ("hc_d48_s42", "HC d48"),
        ("hc_d64_s42", "HC d64"),
        ("mhc_d32_s42", "mHC d32"),
        ("mhc_d48_s42", "mHC d48"),
    ]

    data = {
        "description": "Stress test Amax trajectories for racing lines animation.",
        "sample_rate": 30,
        "runs": {}
    }

    sample_rate = 30

    for run_name, label in stress_runs:
        history = load_history(stress_dir / run_name / "history.json")
        if history is None:
            continue

        frames = []
        for i in range(0, len(history), sample_rate):
            entry = history[i]
            frames.append({
                "step": entry["step"],
                "amax": round(entry.get("composite_amax", 1.0), 2),
                "loss": round(entry["loss"], 3),
            })

        data["runs"][label] = {
            "name": run_name,
            "frames": frames,
            "max_amax": round(max(m.get("composite_amax", 1.0) for m in history), 2),
            "final_loss": round(history[-1]["loss"], 3),
        }

    output_path = output_dir / "stress_racing_data.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


def export_interactive_scrubber_data(run_dir: Path, output_dir: Path, depth: int = 32):
    """
    Export comprehensive data for an interactive training scrubber.

    Allows readers to drag through training and see all metrics update.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = ["residual", "hc", "mhc"]
    sample_rate = 25  # Every 25 steps for smooth scrubbing

    data = {
        "description": "Interactive scrubber data. Drag through training to see metrics.",
        "depth": depth,
        "sample_rate": sample_rate,
        "methods": {}
    }

    for method in methods:
        exp = load_experiment(run_dir, method, depth, 42)
        if exp is None:
            continue

        frames = []
        for i in range(0, len(exp["history"]), sample_rate):
            entry = exp["history"][i]

            frame = {
                "step": entry["step"],
                "loss": round(entry["loss"], 4),
                "grad_norm": round(entry["grad_norm"], 4),
                "lr": entry["lr"],
            }

            if "composite_amax" in entry:
                frame["amax"] = round(entry["composite_amax"], 2)

            if "layer_amax" in entry:
                frame["layer_amax"] = [
                    round(max(ld["attn"], ld["ffn"]), 3)
                    for ld in entry["layer_amax"]
                ]

            frames.append(frame)

        data["methods"][method] = frames

    output_path = output_dir / "scrubber_data.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate extended Part 2 visualizations")
    parser.add_argument("--run-dir", type=str, default="runs_c4", help="Main experiment directory")
    parser.add_argument("--stress-dir", type=str, default="runs_c4_stress", help="Stress test directory")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    stress_dir = Path(args.stress_dir)
    output_dir = Path(args.output_dir)
    animation_dir = output_dir / "animation_data"

    print("=" * 60)
    print("Extended Visualization for Part 2 Blog Post")
    print("=" * 60)
    print(f"Run dir: {run_dir}")
    print(f"Stress dir: {stress_dir}")
    print(f"Output dir: {output_dir}")
    print()

    # Generate static figures
    print("Generating static figures...")
    generate_scaling_law_figure(output_dir)

    if stress_dir.exists():
        generate_stress_test_figure(stress_dir, output_dir)
        generate_bomb_analysis_figure(run_dir, stress_dir, output_dir)
    else:
        print(f"Skipping stress test figures: {stress_dir} not found")

    # Export animation data
    print()
    print("Exporting animation data...")
    export_amax_counter_data(run_dir, animation_dir)
    export_layer_heatmap_data(run_dir, animation_dir, depth=48)
    export_signal_flow_data(run_dir, animation_dir, depth=32)
    export_interactive_scrubber_data(run_dir, animation_dir, depth=32)

    if stress_dir.exists():
        export_stress_racing_data(stress_dir, animation_dir)

    print()
    print("=" * 60)
    print("Done! Generated files:")
    print("=" * 60)
    print("Static figures:")
    print(f"  - {output_dir}/scaling_law.png")
    print(f"  - {output_dir}/stress_test.png")
    print(f"  - {output_dir}/bomb_analysis.png")
    print()
    print("Animation data (for SVG interactives):")
    print(f"  - {animation_dir}/amax_counter_data.json")
    print(f"  - {animation_dir}/layer_heatmap_data.json")
    print(f"  - {animation_dir}/signal_flow_data.json")
    print(f"  - {animation_dir}/stress_racing_data.json")
    print(f"  - {animation_dir}/scrubber_data.json")


if __name__ == "__main__":
    main()
