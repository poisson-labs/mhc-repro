"""
Minimal GPT model with swappable connection types.

Architecture:
- ~10M parameters (6 layers, dim=384, 6 heads)
- Causal attention + FFN per layer
- Learned positional embeddings
- Configurable connection: residual, HC, or mHC
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelConfig
from .connections import get_connection_class


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim

        # QKV projection
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask (registered as buffer, not parameter)
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, hidden_dim)

        Returns:
            (batch, seq, hidden_dim)
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        att = (q @ k.transpose(-2, -1)) * scale

        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Weighted sum of values
        y = att @ v

        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
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
    """Single transformer block with configurable connection type.

    For HC/MHC: expects input (n, batch, seq, hidden_dim) and returns same shape.
    For residual: expects input (batch, seq, hidden_dim) and returns same shape.
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.connection_type = config.connection_type

        # Layer norms (pre-norm architecture)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

        # Attention and FFN
        self.attn = CausalSelfAttention(config)
        self.ffn = FeedForward(config)

        # Connection type
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

        # Store whether this block uses streams
        self.uses_streams = self.conn_attn.uses_streams

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: For HC/MHC: (n, batch, seq, hidden_dim) - streams persist
               For residual: (batch, seq, hidden_dim)

        Returns:
            Same shape as input - streams persist through the block
        """
        # Pre-norm + attention with connection
        x = self.conn_attn(x, lambda z: self.attn(self.ln1(z)))
        # Pre-norm + FFN with connection
        x = self.conn_ffn(x, lambda z: self.ffn(self.ln2(z)))
        return x

    def get_h_res_matrices(self) -> dict:
        """Get H_res matrices for this layer."""
        return {
            "attn": self.conn_attn.get_h_res(),
            "ffn": self.conn_ffn.get_h_res(),
        }


class GPT(nn.Module):
    """Minimal GPT model with swappable connection types.

    CRITICAL ARCHITECTURE:
    - For HC/MHC: Expand streams ONCE after embedding, maintain (n,B,S,D) through
      ALL transformer blocks, collapse ONCE before output projection.
    - For residual: Standard (B,S,D) flow throughout.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.n_layers)
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Store whether we use streams (HC/MHC use streams, residual does not)
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
        """Expand (B, S, D) to (n, B, S, D) by replicating."""
        return x.unsqueeze(0).expand(self.n, -1, -1, -1).clone()

    def _collapse_streams(self, z: torch.Tensor) -> torch.Tensor:
        """Collapse (n, B, S, D) to (B, S, D) by taking first stream."""
        return z[0]

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        debug_shapes: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx: Input token indices (batch, seq)
            targets: Target token indices for loss computation
            debug_shapes: If True, print tensor shapes at key points

        Returns:
            logits: (batch, seq, vocab_size)
            loss: Cross-entropy loss if targets provided
        """
        B, T = idx.shape
        device = idx.device

        # Token + position embeddings
        tok_emb = self.tok_emb(idx)
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)  # (B, S, D)

        if debug_shapes:
            print(f"After embedding: {x.shape}")

        # CRITICAL: Expand streams ONCE for HC/MHC
        if self.uses_streams:
            x = self._expand_streams(x)  # (B, S, D) -> (n, B, S, D)
            if debug_shapes:
                print(f"After stream expansion: {x.shape}")

        # Transformer blocks - streams persist through ALL blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if debug_shapes:
                print(f"After block {i}: {x.shape}")

        # CRITICAL: Collapse streams ONCE for HC/MHC
        if self.uses_streams:
            x = self._collapse_streams(x)  # (n, B, S, D) -> (B, S, D)
            if debug_shapes:
                print(f"After stream collapse: {x.shape}")

        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if debug_shapes:
            print(f"Final logits: {logits.shape}")

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def get_all_h_res_matrices(self) -> list[dict]:
        """Get H_res matrices from all layers for logging."""
        return [block.get_h_res_matrices() for block in self.blocks]

    def get_composite_h_res(self) -> Optional[torch.Tensor]:
        """
        Compute composite H_res product across all layers.

        Returns product of all H_res matrices (attn and ffn interleaved).
        """
        matrices = self.get_all_h_res_matrices()
        if matrices[0]["attn"] is None:
            return None

        # Start with identity
        n = matrices[0]["attn"].shape[0]
        composite = torch.eye(n, device=matrices[0]["attn"].device)

        # Multiply all H_res matrices
        for layer_matrices in matrices:
            if layer_matrices["attn"] is not None:
                composite = composite @ layer_matrices["attn"]
            if layer_matrices["ffn"] is not None:
                composite = composite @ layer_matrices["ffn"]

        return composite

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(connection_type: str = "residual", **kwargs) -> GPT:
    """Factory function to create model with specified connection type."""
    config = ModelConfig(connection_type=connection_type, **kwargs)
    return GPT(config)


if __name__ == "__main__":
    # Test model creation and forward pass
    torch.manual_seed(42)

    print("Testing GPT models...")
    print("=" * 50)

    for conn_type in ["residual", "hc", "mhc"]:
        print(f"\n{conn_type.upper()}:")

        model = create_model(connection_type=conn_type)
        n_params = model.count_parameters()
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
        print(f"  uses_streams: {model.uses_streams}")

        # Test forward pass with shape debugging
        batch_size, seq_len = 2, 64
        idx = torch.randint(0, 65, (batch_size, seq_len))
        targets = torch.randint(0, 65, (batch_size, seq_len))

        print(f"  --- Shape flow ---")
        logits, loss = model(idx, targets, debug_shapes=True)
        print(f"  Logits shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")

        # Test backward pass
        loss.backward()
        total_grad_norm = sum(
            p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
        ) ** 0.5
        print(f"  Gradient norm: {total_grad_norm:.4f}")

        # H_res matrices
        h_res_all = model.get_all_h_res_matrices()
        if h_res_all[0]["attn"] is not None:
            print(f"  H_res per layer: {len(h_res_all)} layers x 2 (attn+ffn)")

            composite = model.get_composite_h_res()
            if composite is not None:
                row_sums = composite.abs().sum(dim=-1)
                col_sums = composite.abs().sum(dim=-2)
                amax = max(row_sums.max().item(), col_sums.max().item())
                print(f"  Composite Amax Gain: {amax:.4f}")

    print("\n" + "=" * 50)
    print("All tests passed!")
