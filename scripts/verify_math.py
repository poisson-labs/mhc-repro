
import torch
import torch.nn as nn
from src.connections import HyperConnection

def verify_math():
    print("Verifying HyperConnection Math against Paper Equations...")
    torch.manual_seed(42)
    
    # 1. Instantiate
    dim = 4
    n = 2
    alpha_init = 0.5
    hc = HyperConnection(hidden_dim=dim, expansion_rate=n, alpha=alpha_init)
    
    # 2. Check Parameters
    print(f"\n[Check A] Parameter Types:")
    print(f"  H_pre: {type(hc.H_pre)} (Static Parameter)")
    print(f"  H_post: {type(hc.H_post)} (Static Parameter)")
    print(f"  H_res: {type(hc.H_res)} (Static Parameter)")
    print(f"  Alpha: {type(hc.alpha)} (Parameter)")
    
    # 3. Check Alpha Usage
    # We'll run forward pass with alpha=0.5, then manually change alpha to 100.0
    # If output doesn't change, alpha is unused.
    x = torch.randn(1, 1, dim)
    sublayer = lambda z: z # Identity sublayer
    
    y1 = hc(x, sublayer)
    
    old_alpha = hc.alpha.item()
    hc.alpha.data.fill_(100.0) # Massive change
    y2 = hc(x, sublayer)
    
    diff = (y1 - y2).abs().max().item()
    print(f"\n[Check B] Alpha Gating:")
    print(f"  Output difference after changing alpha from {old_alpha} to 100.0: {diff}")
    if diff == 0:
        print("  FAIL: Alpha is unused in forward pass (it's only used for init).")
    else:
        print("  PASS: Alpha affects forward pass.")

    # 4. Check Dynamic Weights (Equation 5)
    # If weights are dynamic, H_res should change based on input x.
    # But H_res is a leaf Parameter, so it's static by definition.
    print(f"\n[Check C] Dynamic vs Static Split (Eq 5):")
    if isinstance(hc.H_res, nn.Parameter):
         print("  FAIL: H_res is a static nn.Parameter. No dynamic (input-dependent) component found.")
    else:
         print("  PASS: H_res might be dynamic.")

    # 5. Step-by-Step Forward Math
    print(f"\n[Step-by-Step Forward Pass Reconstruction]")
    # Re-init for cleanliness
    hc = HyperConnection(hidden_dim=dim, expansion_rate=n)
    z = x.unsqueeze(0).expand(n, -1, -1, -1) # Expand x
    
    # Step 1: H_pre mixing
    pre_w = torch.softmax(hc.H_pre, dim=0)
    sub_in = torch.einsum("n,nbsd->bsd", pre_w, z)
    print(f"  1. Sublayer Input (H_pre @ z): shape {sub_in.shape}")
    
    # Step 2: Sublayer
    sub_out = sublayer(sub_in)
    
    # Step 3: H_post distribution
    post_w = torch.softmax(hc.H_post, dim=0)
    delta = sub_out.unsqueeze(0) * post_w.view(n, 1, 1, 1)
    print(f"  2. Sublayer Output Distributed (H_post @ F(...)): shape {delta.shape}")
    
    # Step 4: H_res residual mixing
    z_mixed = torch.einsum("ij,jbsd->ibsd", hc.H_res, z)
    print(f"  3. Residual Mixing (H_res @ z): shape {z_mixed.shape}")
    
    # Step 5: Combine
    z_new = z_mixed + delta
    output = z_new[0]
    
    print(f"  4. Final Output (z_new[0]): shape {output.shape}")
    
    # Verify implementation matches this manual logic
    real_output = hc(x, sublayer)
    match = torch.allclose(output, real_output)
    print(f"  Matches implementation? {match}")

if __name__ == "__main__":
    verify_math()
