"""
Connection types for transformer layers.

Implements:
- ResidualConnection: Standard x + F(x)
- HyperConnection: Input-dependent H_pre, H_post, H_res matrices (equations 5, 7, 8)
- ManifoldHyperConnection: HC with H_res projected onto doubly stochastic manifold
"""

import torch
import torch.nn as nn
from typing import Callable, Optional

from .sinkhorn import sinkhorn_knopp


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
    """Standard residual connection: output = x + F(x)"""

    def __init__(self, hidden_dim: int, expansion_rate: int = 4, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq, hidden_dim)
            sublayer: Function F to apply (attention or FFN)

        Returns:
            x + F(x)
        """
        return x + sublayer(x)

    def get_h_res(self) -> Optional[torch.Tensor]:
        """Return H_res matrix for logging (None for residual)."""
        return None


class HyperConnection(nn.Module):
    """
    Input-dependent Hyper-Connection (equations 5, 7, 8 from paper).

    Computes input-dependent mixing matrices:
    - H_res = alpha_res * tanh(theta_res @ x_norm).reshape(n,n) + b_res
    - H_pre = alpha_pre * tanh(theta_pre @ x_norm) + b_pre
    - H_post = alpha_post * tanh(theta_post @ x_norm) + b_post

    where x_norm is RMSNorm applied to flattened stream representation.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        alpha: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.n = expansion_rate
        self.alpha_init = alpha

        # Input dimension for projections: n * hidden_dim
        input_dim = self.n * hidden_dim

        # RMSNorm for input normalization
        self.rms_norm = RMSNorm(input_dim)

        # H_res projections (equation 5)
        self.theta_res = nn.Linear(input_dim, self.n * self.n, bias=False)
        self.b_res = nn.Parameter(torch.eye(self.n))  # Initialize as identity
        self.alpha_res = nn.Parameter(torch.tensor(alpha))

        # H_pre projections (equation 7)
        self.theta_pre = nn.Linear(input_dim, self.n, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_pre.data[0] = 1.0  # Bias toward first stream
        self.alpha_pre = nn.Parameter(torch.tensor(alpha))

        # H_post projections (equation 8)
        self.theta_post = nn.Linear(input_dim, self.n, bias=False)
        self.b_post = nn.Parameter(torch.zeros(self.n))
        self.b_post.data[0] = 1.0  # Bias toward first stream
        self.alpha_post = nn.Parameter(torch.tensor(alpha))

        # Initialize projections with small weights
        nn.init.normal_(self.theta_res.weight, std=0.01)
        nn.init.normal_(self.theta_pre.weight, std=0.01)
        nn.init.normal_(self.theta_post.weight, std=0.01)

        # Cache for logging
        self._last_h_res = None

    def _compute_h_matrices(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute input-dependent H matrices.

        Args:
            z: Expanded state of shape (n, batch, seq, dim)

        Returns:
            H_res: (batch, n, n)
            H_pre: (batch, n)
            H_post: (batch, n)
        """
        n, batch, seq, dim = z.shape

        # Pool across sequence and flatten streams: (batch, n*dim)
        z_pooled = z.mean(dim=2)  # (n, batch, dim)
        z_flat = z_pooled.permute(1, 0, 2).reshape(batch, n * dim)  # (batch, n*dim)

        # Apply RMSNorm
        x_norm = self.rms_norm(z_flat)  # (batch, n*dim)

        # Compute H_res (equation 5)
        h_res_logits = self.theta_res(x_norm)  # (batch, n*n)
        H_res = self.alpha_res * torch.tanh(h_res_logits).reshape(batch, self.n, self.n) + self.b_res

        # Compute H_pre (equation 7)
        h_pre_logits = self.theta_pre(x_norm)  # (batch, n)
        H_pre = self.alpha_pre * torch.tanh(h_pre_logits) + self.b_pre

        # Compute H_post (equation 8)
        h_post_logits = self.theta_post(x_norm)  # (batch, n)
        H_post = self.alpha_post * torch.tanh(h_post_logits) + self.b_post

        return H_res, H_pre, H_post

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq, hidden_dim)
            sublayer: Function F to apply

        Returns:
            Output tensor of shape (batch, seq, hidden_dim)
        """
        batch, seq, dim = x.shape

        # Expand to n streams: (n, batch, seq, dim)
        z = x.unsqueeze(0).expand(self.n, -1, -1, -1).clone()

        # Compute input-dependent H matrices
        H_res, H_pre, H_post = self._compute_h_matrices(z)

        # Cache H_res for logging (use mean across batch)
        self._last_h_res = H_res.mean(dim=0).detach()

        # Compute sublayer input: weighted sum of streams using H_pre
        # H_pre: (batch, n), z: (n, batch, seq, dim) -> (batch, seq, dim)
        # Use raw H_pre values - tanh already bounds to [-1, 1]
        sublayer_input = torch.einsum("bn,nbsd->bsd", H_pre, z)

        # Apply sublayer
        sublayer_output = sublayer(sublayer_input)  # (batch, seq, dim)

        # Distribute output to streams via H_post
        # Use raw H_post values - tanh already bounds to [-1, 1]
        # delta[i,b,s,d] = sublayer_output[b,s,d] * H_post[b,i]
        delta = sublayer_output.unsqueeze(0) * H_post.t().unsqueeze(-1).unsqueeze(-1)  # (n, batch, seq, dim)

        # Mix residual streams via H_res
        # H_res: (batch, n, n), z: (n, batch, seq, dim)
        # z_mixed[i,b,s,d] = sum_j H_res[b,i,j] * z[j,b,s,d]
        z_mixed = torch.einsum("bij,jbsd->ibsd", H_res, z)

        # Add sublayer contribution
        z_new = z_mixed + delta

        # Aggregate back to single stream (take first stream)
        output = z_new[0]

        return output

    def get_h_res(self) -> Optional[torch.Tensor]:
        """Return cached H_res matrix for logging (averaged across batch)."""
        return self._last_h_res.clone() if self._last_h_res is not None else None


class ManifoldHyperConnection(nn.Module):
    """
    Manifold-Constrained Hyper-Connection.

    Same input-dependent computation as HyperConnection but with manifold projections:
    - H_res -> Sinkhorn(H_res) to project onto doubly stochastic manifold
    - H_pre -> sigmoid(H_pre) for non-negativity
    - H_post -> 2*sigmoid(H_post) for bounded positive range

    This ensures bounded signal propagation and stable training.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int = 4,
        alpha: float = 0.01,
        sinkhorn_iters: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.n = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.alpha_init = alpha

        # Input dimension for projections: n * hidden_dim
        input_dim = self.n * hidden_dim

        # RMSNorm for input normalization
        self.rms_norm = RMSNorm(input_dim)

        # H_res projections (equation 5, then Sinkhorn)
        self.theta_res = nn.Linear(input_dim, self.n * self.n, bias=False)
        self.b_res = nn.Parameter(torch.eye(self.n) * 2.0)  # Bias toward identity in log-space
        self.alpha_res = nn.Parameter(torch.tensor(alpha))

        # H_pre projections (equation 7, then sigmoid)
        self.theta_pre = nn.Linear(input_dim, self.n, bias=False)
        self.b_pre = nn.Parameter(torch.zeros(self.n))
        self.b_pre.data[0] = 2.0  # Bias toward first stream
        self.alpha_pre = nn.Parameter(torch.tensor(alpha))

        # H_post projections (equation 8, then 2*sigmoid)
        self.theta_post = nn.Linear(input_dim, self.n, bias=False)
        self.b_post = nn.Parameter(torch.zeros(self.n))
        self.b_post.data[0] = 2.0  # Bias toward first stream
        self.alpha_post = nn.Parameter(torch.tensor(alpha))

        # Initialize projections with small weights
        nn.init.normal_(self.theta_res.weight, std=0.01)
        nn.init.normal_(self.theta_pre.weight, std=0.01)
        nn.init.normal_(self.theta_post.weight, std=0.01)

        # Cache for logging
        self._last_h_res = None

    def _compute_h_matrices(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute input-dependent H matrices with manifold projections.

        Args:
            z: Expanded state of shape (n, batch, seq, dim)

        Returns:
            H_res: (batch, n, n) - Sinkhorn-projected doubly stochastic
            H_pre: (batch, n) - sigmoid-constrained
            H_post: (batch, n) - 2*sigmoid-constrained
        """
        n, batch, seq, dim = z.shape

        # Pool across sequence and flatten streams: (batch, n*dim)
        z_pooled = z.mean(dim=2)  # (n, batch, dim)
        z_flat = z_pooled.permute(1, 0, 2).reshape(batch, n * dim)  # (batch, n*dim)

        # Apply RMSNorm
        x_norm = self.rms_norm(z_flat)  # (batch, n*dim)

        # Compute H_res (equation 5) then apply Sinkhorn
        h_res_logits = self.theta_res(x_norm)  # (batch, n*n)
        H_res_raw = self.alpha_res * torch.tanh(h_res_logits).reshape(batch, self.n, self.n) + self.b_res
        # Apply Sinkhorn to each batch element
        H_res = sinkhorn_knopp(H_res_raw, t_max=self.sinkhorn_iters)

        # Compute H_pre (equation 7) then apply sigmoid
        h_pre_logits = self.theta_pre(x_norm)  # (batch, n)
        H_pre_raw = self.alpha_pre * torch.tanh(h_pre_logits) + self.b_pre
        H_pre = torch.sigmoid(H_pre_raw)

        # Compute H_post (equation 8) then apply 2*sigmoid
        h_post_logits = self.theta_post(x_norm)  # (batch, n)
        H_post_raw = self.alpha_post * torch.tanh(h_post_logits) + self.b_post
        H_post = 2.0 * torch.sigmoid(H_post_raw)

        return H_res, H_pre, H_post

    def forward(
        self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq, hidden_dim)
            sublayer: Function F to apply

        Returns:
            Output tensor of shape (batch, seq, hidden_dim)
        """
        batch, seq, dim = x.shape

        # Expand to n streams: (n, batch, seq, dim)
        z = x.unsqueeze(0).expand(self.n, -1, -1, -1).clone()

        # Compute input-dependent H matrices with manifold projections
        H_res, H_pre, H_post = self._compute_h_matrices(z)

        # Cache H_res for logging (use mean across batch)
        self._last_h_res = H_res.mean(dim=0).detach()

        # Compute sublayer input: weighted sum of streams using H_pre
        # H_pre: (batch, n), z: (n, batch, seq, dim) -> (batch, seq, dim)
        # Use H_pre directly - sigmoid already bounds to [0, 1]
        sublayer_input = torch.einsum("bn,nbsd->bsd", H_pre, z)

        # Apply sublayer
        sublayer_output = sublayer(sublayer_input)  # (batch, seq, dim)

        # Distribute output to streams via H_post
        # Use H_post directly - 2*sigmoid gives range [0, 2] intentionally
        # delta[i,b,s,d] = sublayer_output[b,s,d] * H_post[b,i]
        delta = sublayer_output.unsqueeze(0) * H_post.t().unsqueeze(-1).unsqueeze(-1)  # (n, batch, seq, dim)

        # Mix residual streams via H_res (already doubly stochastic)
        # H_res: (batch, n, n), z: (n, batch, seq, dim)
        z_mixed = torch.einsum("bij,jbsd->ibsd", H_res, z)

        # Add sublayer contribution
        z_new = z_mixed + delta

        # Aggregate back to single stream (take first stream)
        output = z_new[0]

        return output

    def get_h_res(self) -> Optional[torch.Tensor]:
        """Return cached H_res matrix for logging (averaged across batch)."""
        return self._last_h_res.clone() if self._last_h_res is not None else None


def get_connection_class(connection_type: str) -> type:
    """Factory function to get connection class by name."""
    classes = {
        "residual": ResidualConnection,
        "hc": HyperConnection,
        "mhc": ManifoldHyperConnection,
    }
    if connection_type not in classes:
        raise ValueError(f"Unknown connection type: {connection_type}. Choose from {list(classes.keys())}")
    return classes[connection_type]


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    print("Testing connection types...")
    print("=" * 50)

    batch, seq, dim = 2, 16, 64
    x = torch.randn(batch, seq, dim)

    # Simple sublayer for testing
    def dummy_sublayer(x):
        return torch.tanh(x)

    for name, cls in [
        ("Residual", ResidualConnection),
        ("HC", HyperConnection),
        ("mHC", ManifoldHyperConnection),
    ]:
        conn = cls(hidden_dim=dim, expansion_rate=4)
        y = conn(x, dummy_sublayer)

        print(f"\n{name}:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")

        h_res = conn.get_h_res()
        if h_res is not None:
            print(f"  H_res shape: {h_res.shape}")
            print(f"  H_res:\n{h_res}")

            # Check Amax Gain Magnitude (max abs row/col sum)
            row_sums = h_res.abs().sum(dim=-1)
            col_sums = h_res.abs().sum(dim=-2)
            amax = max(row_sums.max().item(), col_sums.max().item())
            print(f"  Amax Gain Magnitude: {amax:.4f}")

    # Test gradient flow
    print("\n" + "=" * 50)
    print("Testing gradient flow...")

    for name, cls in [("HC", HyperConnection), ("mHC", ManifoldHyperConnection)]:
        conn = cls(hidden_dim=dim)
        x = torch.randn(batch, seq, dim, requires_grad=True)
        y = conn(x, dummy_sublayer)
        loss = y.sum()
        loss.backward()

        print(f"\n{name}:")
        print(f"  Gradient through x: {x.grad is not None}")
        print(f"  theta_res grad exists: {conn.theta_res.weight.grad is not None}")
        print(f"  alpha_res grad: {conn.alpha_res.grad}")

    # Test input-dependent H_res
    print("\n" + "=" * 50)
    print("Testing INPUT-DEPENDENT H_res (critical test)...")

    for name, cls in [("HC", HyperConnection), ("mHC", ManifoldHyperConnection)]:
        conn = cls(hidden_dim=dim, expansion_rate=4)

        # Test with two different inputs
        x1 = torch.randn(batch, seq, dim)
        x2 = torch.randn(batch, seq, dim) * 2 + 1  # Different input

        # Forward pass with first input
        _ = conn(x1, dummy_sublayer)
        h_res_1 = conn.get_h_res().clone()

        # Forward pass with second input
        _ = conn(x2, dummy_sublayer)
        h_res_2 = conn.get_h_res().clone()

        # Check that H_res is different for different inputs
        diff = (h_res_1 - h_res_2).abs().max().item()
        is_different = diff > 1e-6

        print(f"\n{name}:")
        print(f"  H_res for input 1:\n{h_res_1}")
        print(f"  H_res for input 2:\n{h_res_2}")
        print(f"  Max difference: {diff:.6f}")
        print(f"  INPUT-DEPENDENT: {'YES' if is_different else 'NO (FAIL)'}")

        if not is_different:
            raise AssertionError(f"{name}: H_res is NOT input-dependent!")

    print("\n" + "=" * 50)
    print("All tests passed!")
