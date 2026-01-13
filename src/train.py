"""
Training loop for mHC paper reproduction.

Logs:
- Loss per step
- Gradient norm per step
- Amax Gain Magnitude of H_res matrices (per layer and composite)
- Saves metrics to JSON for visualization
"""

import os
import json
import time
import urllib.request
from pathlib import Path
from typing import Optional
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .config import Config, ModelConfig, TrainConfig
from .model import GPT


# ============================================================================
# Data
# ============================================================================

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_data(data_dir: Path) -> Path:
    """Download TinyShakespeare if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "input.txt"

    if not data_path.exists():
        print(f"Downloading TinyShakespeare to {data_path}...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, data_path)
        print("Done.")

    return data_path


class CharDataset(Dataset):
    """Character-level dataset for language modeling."""

    def __init__(self, data_path: Path, context_length: int, split: str = "train"):
        with open(data_path, "r") as f:
            text = f.read()

        # Build vocabulary
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        # Encode full text
        data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

        # Train/val split (90/10)
        n = int(0.9 * len(data))
        self.data = data[:n] if split == "train" else data[n:]
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y


# ============================================================================
# Metrics
# ============================================================================

def compute_amax_gain(h_res: torch.Tensor) -> float:
    """Compute Amax Gain Magnitude (max of row/col abs sums)."""
    if h_res is None:
        return 0.0
    row_sums = h_res.abs().sum(dim=-1)
    col_sums = h_res.abs().sum(dim=-2)
    return max(row_sums.max().item(), col_sums.max().item())


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def compute_layer_gradient_norms(model) -> list[dict]:
    """Compute per-layer gradient norms for gradient flow analysis.

    Returns a list of dicts, one per layer, with keys:
    - layer: layer index
    - attn: gradient norm for attention sublayer
    - ffn: gradient norm for FFN sublayer
    - conn_attn: gradient norm for attention connection parameters
    - conn_ffn: gradient norm for FFN connection parameters
    - total: total gradient norm for the layer
    """
    layer_norms = []

    for i, block in enumerate(model.blocks):
        norms = {"layer": i}

        # Attention sublayer (qkv, out_proj)
        attn_norm = 0.0
        for p in block.attn.parameters():
            if p.grad is not None:
                attn_norm += p.grad.data.norm(2).item() ** 2
        norms["attn"] = attn_norm ** 0.5

        # FFN sublayer (fc1, fc2)
        ffn_norm = 0.0
        for p in block.ffn.parameters():
            if p.grad is not None:
                ffn_norm += p.grad.data.norm(2).item() ** 2
        norms["ffn"] = ffn_norm ** 0.5

        # Connection parameters (attn)
        conn_attn_norm = 0.0
        for p in block.conn_attn.parameters():
            if p.grad is not None:
                conn_attn_norm += p.grad.data.norm(2).item() ** 2
        norms["conn_attn"] = conn_attn_norm ** 0.5

        # Connection parameters (ffn)
        conn_ffn_norm = 0.0
        for p in block.conn_ffn.parameters():
            if p.grad is not None:
                conn_ffn_norm += p.grad.data.norm(2).item() ** 2
        norms["conn_ffn"] = conn_ffn_norm ** 0.5

        # Total for the layer
        norms["total"] = (attn_norm + ffn_norm + conn_attn_norm + conn_ffn_norm) ** 0.5

        layer_norms.append(norms)

    return layer_norms


def collect_metrics(model: GPT, loss: float, step: int) -> dict:
    """Collect all metrics for a training step."""
    metrics = {
        "step": step,
        "loss": loss,
        "grad_norm": compute_gradient_norm(model),
        "layer_grad_norms": compute_layer_gradient_norms(model),
    }

    # H_res metrics (only for HC/mHC)
    h_res_all = model.get_all_h_res_matrices()
    if h_res_all[0]["attn"] is not None:
        # Per-layer Amax
        layer_amax = []
        for i, layer_h in enumerate(h_res_all):
            attn_amax = compute_amax_gain(layer_h["attn"])
            ffn_amax = compute_amax_gain(layer_h["ffn"])
            layer_amax.append({"layer": i, "attn": attn_amax, "ffn": ffn_amax})
        metrics["layer_amax"] = layer_amax

        # Composite Amax
        composite = model.get_composite_h_res()
        metrics["composite_amax"] = compute_amax_gain(composite)

    return metrics


# ============================================================================
# Training
# ============================================================================

def get_lr(step: int, config: TrainConfig) -> float:
    """Learning rate with warmup and cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps

    # Cosine decay
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * decay_ratio)).item())
    return config.learning_rate * 0.1 + coeff * config.learning_rate * 0.9


@torch.no_grad()
def evaluate(model: GPT, val_loader: DataLoader, device: str, max_steps: int = 20) -> float:
    """Compute validation loss."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def train(
    connection_type: str,
    config: Optional[Config] = None,
    run_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> dict:
    """
    Train a model with specified connection type.

    Args:
        connection_type: "residual", "hc", or "mhc"
        config: Optional config override
        run_dir: Directory to save checkpoints and logs
        data_dir: Directory for data

    Returns:
        Dictionary with training history
    """
    # Setup paths
    if run_dir is None:
        run_dir = Path("runs") / connection_type
    if data_dir is None:
        data_dir = Path("data")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Config
    if config is None:
        config = Config.default(connection_type)
    else:
        config.model.connection_type = connection_type

    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(config.train.seed)

    # Data
    data_path = download_data(data_dir)
    train_dataset = CharDataset(data_path, config.train.context_length, split="train")
    val_dataset = CharDataset(data_path, config.train.context_length, split="val")

    # Update vocab size from data
    config.model.vocab_size = train_dataset.vocab_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Model
    model = GPT(config.model).to(device)
    n_params = model.count_parameters()
    print(f"Model: {connection_type.upper()} with {n_params:,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        betas=config.train.betas,
        weight_decay=config.train.weight_decay,
    )

    # Training loop
    model.train()
    history = []
    train_iter = iter(train_loader)
    best_val_loss = float("inf")

    print(f"\nTraining for {config.train.max_steps} steps...")
    print("=" * 60)

    start_time = time.time()

    for step in range(config.train.max_steps):
        # Get batch (cycle through data)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # Update learning rate
        lr = get_lr(step, config.train)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        _, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        # Collect metrics BEFORE optimizer step (gradients available)
        metrics = collect_metrics(model, loss.item(), step)
        metrics["lr"] = lr
        history.append(metrics)

        # Optimizer step
        optimizer.step()

        # Logging
        if step % config.train.log_interval == 0:
            elapsed = time.time() - start_time
            msg = f"Step {step:5d} | Loss: {loss.item():.4f} | Grad: {metrics['grad_norm']:.4f} | LR: {lr:.2e}"
            if "composite_amax" in metrics:
                msg += f" | Amax: {metrics['composite_amax']:.4f}"
            msg += f" | Time: {elapsed:.1f}s"
            print(msg)

        # Evaluation
        if step > 0 and step % config.train.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, config.train.eval_steps)
            print(f"  -> Val loss: {val_loss:.4f}")
            history[-1]["val_loss"] = val_loss

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), run_dir / "best_model.pt")

        # Checkpointing
        if step > 0 and step % config.train.save_interval == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(config.model),
            }, run_dir / f"checkpoint_{step}.pt")

    # Final evaluation
    val_loss = evaluate(model, val_loader, device, config.train.eval_steps)
    print(f"\nFinal val loss: {val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), run_dir / "final_model.pt")

    # Save history
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "model": asdict(config.model),
            "train": asdict(config.train),
        }, f, indent=2)

    print(f"\nTraining complete. Results saved to {run_dir}")
    print("=" * 60)

    return {"history": history, "best_val_loss": best_val_loss}


def train_all(
    run_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    max_steps: Optional[int] = None,
) -> dict:
    """Train all three variants with same seed for fair comparison."""
    if run_dir is None:
        run_dir = Path("runs")
    if data_dir is None:
        data_dir = Path("data")

    results = {}
    for conn_type in ["residual", "hc", "mhc"]:
        print(f"\n{'='*60}")
        print(f"Training {conn_type.upper()}")
        print("=" * 60)

        config = Config.default(conn_type)
        if max_steps is not None:
            config.train.max_steps = max_steps

        results[conn_type] = train(
            conn_type,
            config=config,
            run_dir=run_dir / conn_type,
            data_dir=data_dir,
        )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train mHC models")
    parser.add_argument(
        "--connection", "-c",
        type=str,
        default="all",
        choices=["residual", "hc", "mhc", "all"],
        help="Connection type to train",
    )
    parser.add_argument("--steps", "-s", type=int, default=5000, help="Training steps")
    parser.add_argument("--run-dir", "-o", type=str, default="runs", help="Output directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--dim", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)

    if args.connection == "all":
        train_all(run_dir, data_dir, max_steps=args.steps)
    else:
        config = Config.default(args.connection)
        config.train.max_steps = args.steps
        if args.layers is not None:
            config.model.n_layers = args.layers
        if args.dim is not None:
            config.model.hidden_dim = args.dim
            # Adjust n_heads to keep head_dim=64 (or nearest divisible)
            config.model.n_heads = args.dim // 64 if args.dim % 64 == 0 else args.dim // 32
            config.model.head_dim = args.dim // config.model.n_heads
        if args.lr is not None:
            config.train.learning_rate = args.lr
        if args.seed is not None:
            config.train.seed = args.seed
        train(args.connection, config, run_dir, data_dir)
