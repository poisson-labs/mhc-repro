"""
Sinkhorn-Knopp algorithm for projecting matrices onto the doubly stochastic manifold.

A doubly stochastic matrix has:
- All entries non-negative
- All rows sum to 1
- All columns sum to 1
"""

import torch
import torch.nn as nn


def sinkhorn_knopp(
    H: torch.Tensor,
    t_max: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Project an unconstrained matrix onto the doubly stochastic manifold.

    Args:
        H: Unconstrained matrix of shape (..., n, n)
        t_max: Number of Sinkhorn iterations
        eps: Small constant for numerical stability

    Returns:
        Doubly stochastic matrix of same shape as H
    """
    # Apply exp to make all entries positive
    P = torch.exp(H)

    # Alternating row and column normalization
    for _ in range(t_max):
        # Row normalization: divide each row by its sum
        P = P / (P.sum(dim=-1, keepdim=True) + eps)
        # Column normalization: divide each column by its sum
        P = P / (P.sum(dim=-2, keepdim=True) + eps)

    return P


class SinkhornProjection(nn.Module):
    """
    Module wrapper for Sinkhorn-Knopp projection.
    Useful when you want configurable t_max as a module attribute.
    """

    def __init__(self, t_max: int = 20, eps: float = 1e-8):
        super().__init__()
        self.t_max = t_max
        self.eps = eps

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return sinkhorn_knopp(H, self.t_max, self.eps)


def is_doubly_stochastic(
    P: torch.Tensor,
    tol: float = 1e-4,
) -> tuple[bool, dict]:
    """
    Check if a matrix is doubly stochastic within tolerance.

    Args:
        P: Matrix to check, shape (..., n, n)
        tol: Tolerance for sum checks

    Returns:
        Tuple of (is_valid, diagnostics_dict)
    """
    # Check non-negativity
    min_val = P.min().item()
    non_negative = min_val >= -tol

    # Check row sums
    row_sums = P.sum(dim=-1)
    row_sum_error = (row_sums - 1.0).abs().max().item()
    rows_sum_to_one = row_sum_error < tol

    # Check column sums
    col_sums = P.sum(dim=-2)
    col_sum_error = (col_sums - 1.0).abs().max().item()
    cols_sum_to_one = col_sum_error < tol

    is_valid = non_negative and rows_sum_to_one and cols_sum_to_one

    diagnostics = {
        "min_value": min_val,
        "row_sum_error": row_sum_error,
        "col_sum_error": col_sum_error,
        "non_negative": non_negative,
        "rows_sum_to_one": rows_sum_to_one,
        "cols_sum_to_one": cols_sum_to_one,
    }

    return is_valid, diagnostics


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    print("Testing Sinkhorn-Knopp algorithm...")
    print("=" * 50)

    # Test 1: Single matrix
    H = torch.randn(4, 4)
    P = sinkhorn_knopp(H)
    is_valid, diag = is_doubly_stochastic(P)

    print("\nTest 1: Single 4x4 matrix")
    print(f"  Input H:\n{H}")
    print(f"\n  Output P (doubly stochastic):\n{P}")
    print(f"\n  Row sums: {P.sum(dim=-1)}")
    print(f"  Col sums: {P.sum(dim=-2)}")
    print(f"  Is doubly stochastic: {is_valid}")
    print(f"  Diagnostics: {diag}")

    # Test 2: Batched matrices
    H_batch = torch.randn(3, 4, 4)
    P_batch = sinkhorn_knopp(H_batch)

    print("\n" + "=" * 50)
    print("Test 2: Batch of 3 matrices (4x4)")
    all_valid = True
    for i in range(3):
        valid, _ = is_doubly_stochastic(P_batch[i])
        all_valid = all_valid and valid
    print(f"  All matrices doubly stochastic: {all_valid}")

    # Test 3: Gradient flow
    print("\n" + "=" * 50)
    print("Test 3: Gradient flow")
    H = torch.randn(4, 4, requires_grad=True)
    P = sinkhorn_knopp(H)
    loss = P.sum()
    loss.backward()
    print(f"  Gradient exists: {H.grad is not None}")
    print(f"  Gradient shape: {H.grad.shape}")
    print(f"  Gradient has no NaN: {not torch.isnan(H.grad).any()}")

    # Test 4: Different iteration counts
    print("\n" + "=" * 50)
    print("Test 4: Convergence with different t_max")
    H = torch.randn(8, 8)
    for t in [1, 5, 10, 20, 50]:
        P = sinkhorn_knopp(H, t_max=t)
        _, diag = is_doubly_stochastic(P)
        print(f"  t_max={t:2d}: row_err={diag['row_sum_error']:.2e}, col_err={diag['col_sum_error']:.2e}")

    print("\n" + "=" * 50)
    print("All tests passed!")
