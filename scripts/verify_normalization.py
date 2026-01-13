
import torch
import torch.nn as nn
from src.connections import ManifoldHyperConnection

def test_normalization_bug():
    print("Testing ManifoldHyperConnection Normalization Bug...")
    
    dim = 64
    n = 4
    batch = 1
    seq = 10
    
    # Instantiate mHC
    mhc = ManifoldHyperConnection(hidden_dim=dim, expansion_rate=n)
    
    # Input
    x = torch.randn(batch, seq, dim)
    
    # Access private method for testing weights
    z = x.unsqueeze(0).expand(n, -1, -1, -1)
    
    # 1. Original weights
    H_res, H_pre, H_post = mhc._compute_h_matrices(z)
    
    print("\n[Original H_post statistics]")
    print(f"Mean: {H_post.mean().item():.4f}")
    print(f"Max:  {H_post.max().item():.4f}")
    
    # Calculate effective weights used in forward pass (normalized)
    post_weights_norm = H_post / (H_post.sum(dim=-1, keepdim=True) + 1e-8)
    print(f"Normalized first row: {post_weights_norm[0].detach().numpy()}")
    
    # 2. Modify the code logic simulation: Change the '2.0 * sigmoid' to '100.0 * sigmoid'
    # Since we can't easily change the class code at runtime, we simulate the effect.
    # The class computes: H_post = 2.0 * sigmoid(...)
    # Then forward computes: weights = H_post / sum(H_post)
    
    # Let's verify algebraically:
    # w = (2 * s) / sum(2 * s) = (2 * s) / (2 * sum(s)) = s / sum(s)
    # This proves the factor 2 is cancelled out.
    
    # To demonstrate this implementation-wise:
    # We will manually compute the normalized weights for H_post * 50
    
    H_post_scaled = H_post * 50.0 # Simulate if the factor was 100 instead of 2
    post_weights_scaled_norm = H_post_scaled / (H_post_scaled.sum(dim=-1, keepdim=True) + 1e-8)
    
    print(f"\n[Scaled H_post (x50) statistics]")
    print(f"Normalized first row: {post_weights_scaled_norm[0].detach().numpy()}")
    
    diff = (post_weights_norm - post_weights_scaled_norm).abs().max().item()
    print(f"\nDifference between Original (factor 2) and Scaled (factor 100): {diff:.2e}")
    
    if diff < 1e-6:
        print("\n>>> BUG CONFIRMED: The scalar factor (2.0) is cancelled out by normalization!")
        print("    Equation (8) requires H_post = 2 * sigmoid(...), which implies a specific magnitude range.")
        print("    Normalizing it forces the sum to 1, violating the magnitude requirement.")
    else:
        print("\n>>> NO BUG: Scaling factor is preserved.")

if __name__ == "__main__":
    test_normalization_bug()
