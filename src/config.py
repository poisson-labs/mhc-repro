"""
Hyperparameters for mHC paper reproduction.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Model size (~10M params)
    n_layers: int = 6
    hidden_dim: int = 384
    n_heads: int = 6
    head_dim: int = 64  # hidden_dim // n_heads

    # FFN
    ffn_multiplier: float = 4.0

    # Hyper-connection
    expansion_rate: int = 4  # n in the paper
    hc_alpha: float = 0.01  # initialization scale
    sinkhorn_iters: int = 20

    # Vocabulary (will be set from data)
    vocab_size: int = 65  # TinyShakespeare has ~65 chars

    # Positional encoding
    max_seq_len: int = 256

    # Connection type
    connection_type: Literal["residual", "hc", "mhc"] = "residual"

    # Dropout
    dropout: float = 0.1

    def __post_init__(self):
        assert self.hidden_dim % self.n_heads == 0
        self.head_dim = self.hidden_dim // self.n_heads


@dataclass
class TrainConfig:
    """Training configuration."""

    # Data
    batch_size: int = 64
    context_length: int = 256

    # Training
    max_steps: int = 5000
    eval_interval: int = 100
    eval_steps: int = 20
    log_interval: int = 10

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    # Warmup
    warmup_steps: int = 100

    # Checkpointing
    save_interval: int = 1000

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"  # Will fallback to cpu if unavailable


@dataclass
class Config:
    """Combined configuration."""

    model: ModelConfig
    train: TrainConfig

    @classmethod
    def default(cls, connection_type: str = "residual") -> "Config":
        """Create default config with specified connection type."""
        model = ModelConfig(connection_type=connection_type)
        train = TrainConfig()
        return cls(model=model, train=train)


# Preset configurations for experiments
CONFIGS = {
    "residual": Config.default("residual"),
    "hc": Config.default("hc"),
    "mhc": Config.default("mhc"),
}


if __name__ == "__main__":
    # Print config summary
    for name, cfg in CONFIGS.items():
        print(f"\n{name.upper()} Config:")
        print(f"  Layers: {cfg.model.n_layers}")
        print(f"  Hidden dim: {cfg.model.hidden_dim}")
        print(f"  Heads: {cfg.model.n_heads}")
        print(f"  FFN dim: {int(cfg.model.hidden_dim * cfg.model.ffn_multiplier)}")
        print(f"  Connection: {cfg.model.connection_type}")
        print(f"  Steps: {cfg.train.max_steps}")
        print(f"  LR: {cfg.train.learning_rate}")
