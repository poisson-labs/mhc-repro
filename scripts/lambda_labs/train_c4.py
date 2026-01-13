#!/usr/bin/env python3
"""
Part 2: Large-scale mHC training on Lambda Labs A100.

This script implements training at 1B parameter scale on C4 dataset with:
- bf16 mixed precision (2x memory reduction)
- Gradient checkpointing (trades compute for memory)
- W&B logging for real-time monitoring
- Comprehensive metrics: loss, gradient norms, Amax

Usage:
    python train_c4.py --connection mhc --depth 32 --seed 42
    python train_c4.py --connection all --depth all --seeds 42,123,456

Author: Taylor Kolasinski
Part of mHC reproduction series: https://poisson.run/notes/deepseek-mhc
"""

import os
import sys
import json
import time
import math
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Literal, Iterator
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.sinkhorn import sinkhorn_knopp


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """1B parameter model configuration."""

    # Model architecture
    n_layers: int = 32
    hidden_dim: int = 2048
    n_heads: int = 32
    head_dim: int = 64  # hidden_dim // n_heads

    # FFN
    ffn_multiplier: float = 4.0

    # Hyper-connection
    expansion_rate: int = 4
    hc_alpha: float = 0.01
    sinkhorn_iters: int = 20

    # Vocabulary (GPT-2 tokenizer)
    vocab_size: int = 50257
    max_seq_len: int = 1024

    # Connection type
    connection_type: Literal["residual", "hc", "mhc"] = "mhc"

    # Dropout (lower for large models)
    dropout: float = 0.0

    # Memory optimization
    use_gradient_checkpointing: bool = True

    def __post_init__(self):
        assert self.hidden_dim % self.n_heads == 0
        self.head_dim = self.hidden_dim // self.n_heads

    def estimate_params(self) -> int:
        """Estimate total parameters (excluding embedding weight tying)."""
        # Embeddings
        emb_params = self.vocab_size * self.hidden_dim  # tok_emb (shared with lm_head)
        pos_params = self.max_seq_len * self.hidden_dim

        # Per-layer params
        # Attention: qkv + out_proj = 4 * hidden_dim^2
        attn_params = 4 * self.hidden_dim * self.hidden_dim

        # FFN: fc1 + fc2 = 2 * hidden_dim * ffn_dim = 2 * 4 * hidden_dim^2
        ffn_dim = int(self.hidden_dim * self.ffn_multiplier)
        ffn_params = 2 * self.hidden_dim * ffn_dim

        # LayerNorm: 2 per block * 2 params each
        ln_params = 4 * self.hidden_dim

        # HC/mHC connection params (both attn and ffn)
        if self.connection_type in ["hc", "mhc"]:
            n = self.expansion_rate
            input_dim = n * self.hidden_dim
            # Per connection: theta_res + theta_pre + theta_post + biases + alphas
            conn_params = (input_dim * n * n) + (input_dim * n) * 2 + (n * n + n * 2) + 3
            conn_params += input_dim  # RMSNorm scale
            conn_params *= 2  # Two connections per block (attn + ffn)
        else:
            conn_params = 0

        layer_params = attn_params + ffn_params + ln_params + conn_params

        # Final ln + lm_head (but lm_head weight-tied)
        final_params = self.hidden_dim

        total = emb_params + pos_params + self.n_layers * layer_params + final_params
        return total


@dataclass
class TrainConfig:
    """Training configuration for A100."""

    # Data
    batch_size: int = 32  # Per-GPU batch size
    context_length: int = 1024

    # Training
    max_steps: int = 10000
    eval_interval: int = 250
    eval_steps: int = 50
    log_interval: int = 10

    # Optimizer (following Chinchilla recommendations)
    learning_rate: float = 1e-4
    min_lr_ratio: float = 0.1  # min_lr = lr * min_lr_ratio
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    # Warmup
    warmup_steps: int = 500

    # Checkpointing
    save_interval: int = 1000

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"

    # Mixed precision
    use_amp: bool = True

    # Logging
    wandb_project: str = "mhc-part2"
    wandb_entity: Optional[str] = None  # Your W&B username

    # Data
    dataset_name: str = "allenai/c4"
    dataset_config: str = "en"
    num_tokens: int = 100_000_000  # ~100M tokens


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    model: ModelConfig
    train: TrainConfig

    # Experiment metadata
    experiment_name: str = ""
    git_hash: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.experiment_name:
            self.experiment_name = f"{self.model.connection_type}_d{self.model.n_layers}_s{self.train.seed}"
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.git_hash:
            try:
                self.git_hash = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()[:8]
            except:
                self.git_hash = "unknown"


# =============================================================================
# Data Loading (C4 with streaming)
# =============================================================================

def create_dataloader(
    config: TrainConfig,
    split: str = "train",
    tokenizer=None,
) -> Iterator:
    """
    Create streaming dataloader for C4 dataset.

    Uses HuggingFace datasets with streaming to avoid downloading entire dataset.
    Tokenizes on-the-fly with tiktoken GPT-2 tokenizer.
    """
    from datasets import load_dataset
    import tiktoken

    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")

    # Load C4 with streaming
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    # Shuffle for training
    if split == "train":
        dataset = dataset.shuffle(seed=config.seed, buffer_size=10000)

    # Buffer for accumulating tokens
    token_buffer = []

    for example in dataset:
        # Tokenize text
        tokens = tokenizer.encode(example["text"])
        token_buffer.extend(tokens)

        # Yield batches when buffer is large enough
        while len(token_buffer) >= config.batch_size * (config.context_length + 1):
            # Extract batch
            batch_tokens = []
            for _ in range(config.batch_size):
                seq = token_buffer[:config.context_length + 1]
                token_buffer = token_buffer[config.context_length + 1:]
                batch_tokens.append(seq)

            # Convert to tensors
            batch = torch.tensor(batch_tokens, dtype=torch.long)
            x = batch[:, :-1]  # (B, T)
            y = batch[:, 1:]   # (B, T)

            yield x, y


# =============================================================================
# Model Components
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class ResidualConnection(nn.Module):
    """Standard residual connection."""

    def __init__(self, hidden_dim: int, expansion_rate: int = 4, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.uses_streams = False

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return x + sublayer(x)

    def get_h_res(self):
        return None


class HyperConnection(nn.Module):
    """Input-dependent Hyper-Connection (unconstrained)."""

    def __init__(self, hidden_dim: int, expansion_rate: int = 4, alpha: float = 0.01, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n = expansion_rate
        self.alpha_init = alpha
        self.uses_streams = True

        input_dim = self.n * hidden_dim

        self.rms_norm = RMSNorm(input_dim)

        # H_res projections
        self.theta_res = nn.Linear(input_dim, self.n * self.n, bias=False)
        self.b_res = nn.Parameter(torch.eye(self.n))
        self.alpha_res = nn.Parameter(torch.tensor(alpha))

        # H_pre projections
        self.theta_pre = nn.Linear(input_dim, self.n, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_pre.data[0] = 1.0
        self.alpha_pre = nn.Parameter(torch.tensor(alpha))

        # H_post projections
        self.theta_post = nn.Linear(input_dim, self.n, bias=False)
        self.b_post = nn.Parameter(torch.zeros(self.n))
        self.b_post.data[0] = 1.0
        self.alpha_post = nn.Parameter(torch.tensor(alpha))

        # Initialize small
        nn.init.normal_(self.theta_res.weight, std=0.01)
        nn.init.normal_(self.theta_pre.weight, std=0.01)
        nn.init.normal_(self.theta_post.weight, std=0.01)

        self._last_h_res = None

    def _compute_h_matrices(self, z: torch.Tensor):
        n, batch, seq, dim = z.shape

        z_pooled = z.mean(dim=2)
        z_flat = z_pooled.permute(1, 0, 2).reshape(batch, n * dim)
        x_norm = self.rms_norm(z_flat)

        h_res_logits = self.theta_res(x_norm)
        H_res = self.alpha_res * torch.tanh(h_res_logits).reshape(batch, self.n, self.n) + self.b_res

        h_pre_logits = self.theta_pre(x_norm)
        H_pre = self.alpha_pre * torch.tanh(h_pre_logits) + self.b_pre

        h_post_logits = self.theta_post(x_norm)
        H_post = self.alpha_post * torch.tanh(h_post_logits) + self.b_post

        return H_res, H_pre, H_post

    def forward(self, z: torch.Tensor, sublayer) -> torch.Tensor:
        n, batch, seq, dim = z.shape

        H_res, H_pre, H_post = self._compute_h_matrices(z)
        self._last_h_res = H_res.mean(dim=0).detach()

        sublayer_input = torch.einsum("bn,nbsd->bsd", H_pre, z)
        sublayer_output = sublayer(sublayer_input)
        delta = sublayer_output.unsqueeze(0) * H_post.t().unsqueeze(-1).unsqueeze(-1)
        z_mixed = torch.einsum("bij,jbsd->ibsd", H_res, z)

        return z_mixed + delta

    def get_h_res(self):
        return self._last_h_res.clone() if self._last_h_res is not None else None


class ManifoldHyperConnection(nn.Module):
    """Manifold-Constrained Hyper-Connection (doubly stochastic H_res)."""

    def __init__(self, hidden_dim: int, expansion_rate: int = 4, alpha: float = 0.01,
                 sinkhorn_iters: int = 20, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.alpha_init = alpha
        self.uses_streams = True

        input_dim = self.n * hidden_dim

        self.rms_norm = RMSNorm(input_dim)

        # H_res projections (will be Sinkhorn-projected)
        self.theta_res = nn.Linear(input_dim, self.n * self.n, bias=False)
        self.b_res = nn.Parameter(torch.eye(self.n) * 2.0)
        self.alpha_res = nn.Parameter(torch.tensor(alpha))

        # H_pre projections (will be sigmoid)
        self.theta_pre = nn.Linear(input_dim, self.n, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_pre.data[0] = 2.0
        self.alpha_pre = nn.Parameter(torch.tensor(alpha))

        # H_post projections (will be 2*sigmoid)
        self.theta_post = nn.Linear(input_dim, self.n, bias=False)
        self.b_post = nn.Parameter(torch.zeros(self.n))
        self.b_post.data[0] = 2.0
        self.alpha_post = nn.Parameter(torch.tensor(alpha))

        nn.init.normal_(self.theta_res.weight, std=0.01)
        nn.init.normal_(self.theta_pre.weight, std=0.01)
        nn.init.normal_(self.theta_post.weight, std=0.01)

        self._last_h_res = None

    def _compute_h_matrices(self, z: torch.Tensor):
        n, batch, seq, dim = z.shape

        z_pooled = z.mean(dim=2)
        z_flat = z_pooled.permute(1, 0, 2).reshape(batch, n * dim)
        x_norm = self.rms_norm(z_flat)

        # H_res with Sinkhorn projection
        h_res_logits = self.theta_res(x_norm)
        H_res_raw = self.alpha_res * torch.tanh(h_res_logits).reshape(batch, self.n, self.n) + self.b_res
        H_res = sinkhorn_knopp(H_res_raw, t_max=self.sinkhorn_iters)

        # H_pre with sigmoid
        h_pre_logits = self.theta_pre(x_norm)
        H_pre_raw = self.alpha_pre * torch.tanh(h_pre_logits) + self.b_pre
        H_pre = torch.sigmoid(H_pre_raw)

        # H_post with 2*sigmoid
        h_post_logits = self.theta_post(x_norm)
        H_post_raw = self.alpha_post * torch.tanh(h_post_logits) + self.b_post
        H_post = 2.0 * torch.sigmoid(H_post_raw)

        return H_res, H_pre, H_post

    def forward(self, z: torch.Tensor, sublayer) -> torch.Tensor:
        n, batch, seq, dim = z.shape

        H_res, H_pre, H_post = self._compute_h_matrices(z)
        self._last_h_res = H_res.mean(dim=0).detach()

        sublayer_input = torch.einsum("bn,nbsd->bsd", H_pre, z)
        sublayer_output = sublayer(sublayer_input)
        delta = sublayer_output.unsqueeze(0) * H_post.t().unsqueeze(-1).unsqueeze(-1)
        z_mixed = torch.einsum("bij,jbsd->ibsd", H_res, z)

        return z_mixed + delta

    def get_h_res(self):
        return self._last_h_res.clone() if self._last_h_res is not None else None


def get_connection_class(connection_type: str):
    return {
        "residual": ResidualConnection,
        "hc": HyperConnection,
        "mhc": ManifoldHyperConnection,
    }[connection_type]


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention when available."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Check for Flash Attention
        self.use_flash = hasattr(F, 'scaled_dot_product_attention')

        if not self.use_flash:
            # Register causal mask for fallback
            mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_flash:
            # Use Flash Attention (PyTorch 2.0+)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Fallback to manual attention
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        ffn_dim = int(hidden_dim * config.ffn_multiplier)

        self.fc1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with gradient checkpointing support."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_checkpoint = config.use_gradient_checkpointing

        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

        self.attn = CausalSelfAttention(config)
        self.ffn = FeedForward(config)

        ConnectionClass = get_connection_class(config.connection_type)
        self.conn_attn = ConnectionClass(
            hidden_dim=config.hidden_dim,
            expansion_rate=config.expansion_rate,
            alpha=config.hc_alpha,
            sinkhorn_iters=config.sinkhorn_iters,
        )
        self.conn_ffn = ConnectionClass(
            hidden_dim=config.hidden_dim,
            expansion_rate=config.expansion_rate,
            alpha=config.hc_alpha,
            sinkhorn_iters=config.sinkhorn_iters,
        )

        self.uses_streams = self.conn_attn.uses_streams

    def _attn_forward(self, z):
        return self.attn(self.ln1(z))

    def _ffn_forward(self, z):
        return self.ffn(self.ln2(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # Gradient checkpointing for memory efficiency
            x = self.conn_attn(x, lambda z: checkpoint(self._attn_forward, z, use_reentrant=False))
            x = self.conn_ffn(x, lambda z: checkpoint(self._ffn_forward, z, use_reentrant=False))
        else:
            x = self.conn_attn(x, self._attn_forward)
            x = self.conn_ffn(x, self._ffn_forward)
        return x

    def get_h_res_matrices(self):
        return {
            "attn": self.conn_attn.get_h_res(),
            "ffn": self.conn_ffn.get_h_res(),
        }


class GPT(nn.Module):
    """GPT model with HC/mHC support and gradient checkpointing."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

        self.uses_streams = config.connection_type in ["hc", "mhc"]
        self.n = config.expansion_rate

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _expand_streams(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0).expand(self.n, -1, -1, -1).clone()

    def _collapse_streams(self, z: torch.Tensor) -> torch.Tensor:
        return z[0]

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.tok_emb(idx)
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        if self.uses_streams:
            x = self._expand_streams(x)

        for block in self.blocks:
            x = block(x)

        if self.uses_streams:
            x = self._collapse_streams(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def get_all_h_res_matrices(self):
        return [block.get_h_res_matrices() for block in self.blocks]

    def get_composite_h_res(self):
        matrices = self.get_all_h_res_matrices()
        if matrices[0]["attn"] is None:
            return None

        n = matrices[0]["attn"].shape[0]
        composite = torch.eye(n, device=matrices[0]["attn"].device)

        for layer_matrices in matrices:
            if layer_matrices["attn"] is not None:
                composite = composite @ layer_matrices["attn"]
            if layer_matrices["ffn"] is not None:
                composite = composite @ layer_matrices["ffn"]

        return composite

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Metrics
# =============================================================================

def compute_amax_gain(h_res: torch.Tensor) -> float:
    """Compute Amax Gain Magnitude (max of row/col abs sums)."""
    if h_res is None:
        return 0.0
    row_sums = h_res.abs().sum(dim=-1)
    col_sums = h_res.abs().sum(dim=-2)
    return max(row_sums.max().item(), col_sums.max().item())


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def compute_layer_gradient_norms(model: GPT) -> list:
    """Compute per-layer gradient norms."""
    layer_norms = []

    for i, block in enumerate(model.blocks):
        norms = {"layer": i}

        # Attention
        attn_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in block.attn.parameters() if p.grad is not None
        ) ** 0.5
        norms["attn"] = attn_norm

        # FFN
        ffn_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in block.ffn.parameters() if p.grad is not None
        ) ** 0.5
        norms["ffn"] = ffn_norm

        # Connections
        conn_attn_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in block.conn_attn.parameters() if p.grad is not None
        ) ** 0.5
        norms["conn_attn"] = conn_attn_norm

        conn_ffn_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in block.conn_ffn.parameters() if p.grad is not None
        ) ** 0.5
        norms["conn_ffn"] = conn_ffn_norm

        norms["total"] = (attn_norm**2 + ffn_norm**2 + conn_attn_norm**2 + conn_ffn_norm**2) ** 0.5
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

    h_res_all = model.get_all_h_res_matrices()
    if h_res_all[0]["attn"] is not None:
        layer_amax = []
        for i, layer_h in enumerate(h_res_all):
            layer_amax.append({
                "layer": i,
                "attn": compute_amax_gain(layer_h["attn"]),
                "ffn": compute_amax_gain(layer_h["ffn"]),
            })
        metrics["layer_amax"] = layer_amax

        composite = model.get_composite_h_res()
        metrics["composite_amax"] = compute_amax_gain(composite)

    return metrics


# =============================================================================
# Training
# =============================================================================

def get_lr(step: int, config: TrainConfig) -> float:
    """Learning rate with warmup and cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps

    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    min_lr = config.learning_rate * config.min_lr_ratio
    return min_lr + coeff * (config.learning_rate - min_lr)


@torch.no_grad()
def evaluate(model: GPT, val_loader: Iterator, device: str, max_steps: int = 50) -> float:
    """Compute validation loss."""
    model.eval()
    losses = []

    for i, (x, y) in enumerate(val_loader):
        if i >= max_steps:
            break
        x, y = x.to(device), y.to(device)

        with autocast(enabled=True, dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def train(config: ExperimentConfig, run_dir: Path) -> dict:
    """Main training loop."""

    # Setup
    run_dir.mkdir(parents=True, exist_ok=True)
    device = config.train.device

    # Set seed
    torch.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)

    # W&B logging
    wandb_run = None
    if config.train.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.train.wandb_project,
                entity=config.train.wandb_entity,
                name=config.experiment_name,
                config={
                    "model": asdict(config.model),
                    "train": asdict(config.train),
                    "git_hash": config.git_hash,
                },
            )
        except ImportError:
            print("wandb not installed, skipping logging")

    # Data
    print("Loading C4 dataset...")
    train_loader = create_dataloader(config.train, split="train")

    # Model
    print(f"Creating model: {config.model.connection_type.upper()} with {config.model.n_layers} layers")
    model = GPT(config.model).to(device)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,} ({n_params/1e9:.2f}B)")
    print(f"  Gradient checkpointing: {config.model.use_gradient_checkpointing}")
    print(f"  Flash Attention: {model.blocks[0].attn.use_flash}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        betas=config.train.betas,
        weight_decay=config.train.weight_decay,
    )

    # Mixed precision
    scaler = GradScaler(enabled=config.train.use_amp)

    # Training loop
    model.train()
    history = []
    best_val_loss = float("inf")
    tokens_seen = 0

    print(f"\nTraining for {config.train.max_steps} steps...")
    print("=" * 70)

    start_time = time.time()

    for step, (x, y) in enumerate(train_loader):
        if step >= config.train.max_steps:
            break

        x, y = x.to(device), y.to(device)
        tokens_seen += x.numel()

        # Update learning rate
        lr = get_lr(step, config.train)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass with mixed precision
        with autocast(enabled=config.train.use_amp, dtype=torch.bfloat16):
            _, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping (unscale first for accurate norm)
        scaler.unscale_(optimizer)
        if config.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)

        # Collect metrics BEFORE optimizer step
        metrics = collect_metrics(model, loss.item(), step)
        metrics["lr"] = lr
        metrics["tokens_seen"] = tokens_seen
        history.append(metrics)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % config.train.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_seen / elapsed

            msg = f"Step {step:5d} | Loss: {loss.item():.4f} | Grad: {metrics['grad_norm']:.2f}"
            msg += f" | LR: {lr:.2e} | tok/s: {tokens_per_sec:.0f}"
            if "composite_amax" in metrics:
                msg += f" | Amax: {metrics['composite_amax']:.4f}"
            print(msg)

            if wandb_run:
                wandb_log = {
                    "train/loss": loss.item(),
                    "train/grad_norm": metrics["grad_norm"],
                    "train/lr": lr,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/tokens_seen": tokens_seen,
                }
                if "composite_amax" in metrics:
                    wandb_log["train/composite_amax"] = metrics["composite_amax"]
                wandb.log(wandb_log, step=step)

        # Evaluation
        if step > 0 and step % config.train.eval_interval == 0:
            val_loader = create_dataloader(config.train, split="validation")
            val_loss = evaluate(model, val_loader, device, config.train.eval_steps)
            print(f"  -> Val loss: {val_loss:.4f}")
            history[-1]["val_loss"] = val_loss

            if wandb_run:
                wandb.log({"val/loss": val_loss}, step=step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), run_dir / "best_model.pt")

        # Checkpointing
        if step > 0 and step % config.train.save_interval == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "config": asdict(config.model),
                "tokens_seen": tokens_seen,
            }, run_dir / f"checkpoint_{step}.pt")

    # Final save
    torch.save(model.state_dict(), run_dir / "final_model.pt")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)

    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "model": asdict(config.model),
            "train": asdict(config.train),
            "experiment_name": config.experiment_name,
            "git_hash": config.git_hash,
            "timestamp": config.timestamp,
        }, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"  Final loss: {history[-1]['loss']:.4f}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Tokens processed: {tokens_seen:,}")
    print(f"  Results saved to {run_dir}")

    if wandb_run:
        wandb.finish()

    return {"history": history, "best_val_loss": best_val_loss}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train mHC models at scale on C4")
    parser.add_argument("--connection", "-c", type=str, default="mhc",
                       choices=["residual", "hc", "mhc", "all"],
                       help="Connection type")
    parser.add_argument("--depth", "-d", type=str, default="32",
                       help="Number of layers (32, 48, or 'all' for both)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--seeds", type=str, default=None,
                       help="Comma-separated seeds (e.g., '42,123,456')")
    parser.add_argument("--steps", type=int, default=10000,
                       help="Training steps")
    parser.add_argument("--run-dir", "-o", type=str, default="runs_c4",
                       help="Output directory")
    parser.add_argument("--wandb-project", type=str, default="mhc-part2",
                       help="W&B project name")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable W&B logging")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--no-checkpoint", action="store_true",
                       help="Disable gradient checkpointing")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (default: 1e-4)")

    args = parser.parse_args()

    # Parse configurations
    connections = ["residual", "hc", "mhc"] if args.connection == "all" else [args.connection]
    depths = [32, 48] if args.depth == "all" else [int(args.depth)]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    run_dir = Path(args.run_dir)

    # Run experiments
    results = {}
    total_experiments = len(connections) * len(depths) * len(seeds)
    experiment_idx = 0

    for conn in connections:
        for depth in depths:
            for seed in seeds:
                experiment_idx += 1
                print(f"\n{'='*70}")
                print(f"Experiment {experiment_idx}/{total_experiments}")
                print(f"  Connection: {conn.upper()}")
                print(f"  Depth: {depth}")
                print(f"  Seed: {seed}")
                print("=" * 70)

                # Create config
                model_config = ModelConfig(
                    connection_type=conn,
                    n_layers=depth,
                    use_gradient_checkpointing=not args.no_checkpoint,
                )
                train_config = TrainConfig(
                    seed=seed,
                    max_steps=args.steps,
                    batch_size=args.batch_size,
                    wandb_project=None if args.no_wandb else args.wandb_project,
                    learning_rate=args.lr if args.lr else 1e-4,
                )
                config = ExperimentConfig(model=model_config, train=train_config)

                # Run
                exp_dir = run_dir / f"{conn}_d{depth}_s{seed}"
                result = train(config, exp_dir)
                results[config.experiment_name] = result

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for name, result in results.items():
        print(f"{name}: best_val_loss = {result['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
