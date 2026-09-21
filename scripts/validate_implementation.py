"""Historical diagnostic, kept for reference.

Written for the implementation before the stream-persistence fix. It calls the connection
classes through their earlier interface, so it no longer runs against the current ones.
"""

import torch
import torch.nn as nn
from src.connections import ManifoldHyperConnection, HyperConnection
from src.sinkhorn import sinkhorn_knopp, is_doubly_stochastic
from src.model import create_model
from src.config import ModelConfig

def check_sinkhorn():
    print("\n[Check 1] Verifying Sinkhorn Implementation...")
    torch.manual_seed(42)
    n = 4
    # Create random matrix
    H_raw = torch.randn(n, n)
    
    # Run sinkhorn
    P = sinkhorn_knopp(H_raw, t_max=20)
    
    # Check properties
    is_valid, diag = is_doubly_stochastic(P)
    
    print(f"  Input H_raw:\n{H_raw}")
    print(f"  Output P:\n{P}")
    print(f"  Doubly Stochastic: {is_valid}")
    print(f"  Diagnostics: {diag}")
    
    # Check it's not Identity
    if torch.allclose(P, torch.eye(n)):
        print("  WARNING: Output is Identity matrix!")
    else:
        print("  OK: Output is not Identity matrix.")

    # Check it's not constant (1/n)
    if torch.allclose(P, torch.ones(n, n)/n):
        print("  WARNING: Output is Uniform matrix!")
    else:
        print("  OK: Output is not Uniform matrix.")

def check_param_count():
    print("\n[Check 2] Verifying Parameter Counts...")
    config_args = {
        "hidden_dim": 64,
        "n_layers": 2,
        "n_heads": 4,
        "expansion_rate": 4
    }
    
    model_hc = create_model(connection_type="hc", **config_args)
    params_hc = model_hc.count_parameters()
    
    model_mhc = create_model(connection_type="mhc", **config_args)
    params_mhc = model_mhc.count_parameters()
    
    print(f"  HC Params: {params_hc}")
    print(f"  mHC Params: {params_mhc}")
    
    if params_hc == params_mhc:
        print("  OK: Parameter counts match.")
    else:
        print(f"  WARNING: Parameter counts differ by {abs(params_hc - params_mhc)}")

def check_gradients():
    print("\n[Check 3] Verifying Gradient Flow in mHC...")
    dim = 16
    n = 4
    mhc = ManifoldHyperConnection(hidden_dim=dim, expansion_rate=n)
    
    # Input
    x = torch.randn(2, 5, dim, requires_grad=True)
    
    # Dummy sublayer
    sublayer = lambda z: z * 2
    
    # Forward
    y = mhc(x, sublayer)
    loss = y.sum()
    
    # Backward
    loss.backward()
    
    print(f"  H_res_raw grad is None: {mhc.H_res_raw.grad is None}")
    if mhc.H_res_raw.grad is not None:
        grad_norm = mhc.H_res_raw.grad.norm().item()
        print(f"  H_res_raw grad norm: {grad_norm}")
        if grad_norm > 0:
            print("  OK: Gradients are flowing to H_res_raw.")
        else:
            print("  WARNING: Gradients are zero.")
    
    # Check H_pre_raw and H_post_raw
    print(f"  H_pre_raw grad norm: {mhc.H_pre_raw.grad.norm().item()}")
    print(f"  H_post_raw grad norm: {mhc.H_post_raw.grad.norm().item()}")

def check_matrices():
    print("\n[Check 4] Inspecting Initial Matrices...")
    dim = 16
    n = 4
    
    print("  --- HC ---")
    hc = HyperConnection(hidden_dim=dim, expansion_rate=n, alpha=0.1)
    print(f"  H_res:\n{hc.get_h_res()}")
    
    print("  --- mHC ---")
    mhc = ManifoldHyperConnection(hidden_dim=dim, expansion_rate=n, alpha=0.1)
    print(f"  H_res (projected):\n{mhc.get_h_res()}")
    
    is_valid, _ = is_doubly_stochastic(mhc.get_h_res())
    print(f"  mHC is doubly stochastic: {is_valid}")

def check_amax_logic():
    print("\n[Check 5] Verifying Amax Computation...")
    
    # Create a small dummy model structure to test get_composite_h_res logic
    # We can reuse the model class but we need to manually inject H_res values
    
    model = create_model(connection_type="hc", n_layers=2, expansion_rate=2)
    
    # Case 1: Identity matrices
    print("  Case 1: All Identity")
    for block in model.blocks:
        block.conn_attn.H_res.data = torch.eye(2)
        block.conn_ffn.H_res.data = torch.eye(2)
        
    comp = model.get_composite_h_res()
    print(f"  Composite:\n{comp}")
    row_sums = comp.abs().sum(dim=-1)
    col_sums = comp.abs().sum(dim=-2)
    amax = max(row_sums.max().item(), col_sums.max().item())
    print(f"  Amax: {amax}")
    if abs(amax - 1.0) < 1e-5:
        print("  OK: Amax is 1.0 for identity.")
    else:
        print("  WARNING: Amax logic might be wrong.")

    # Case 2: Scaled matrices (should explode)
    print("  Case 2: Scaled (2 * Identity)")
    for block in model.blocks:
        block.conn_attn.H_res.data = 2.0 * torch.eye(2)
        block.conn_ffn.H_res.data = 2.0 * torch.eye(2)
        
    comp = model.get_composite_h_res()
    # 2 layers, 2 sublayers each = 4 matrices. 2^4 = 16? 
    # Wait, composite logic: 
    # composite = I
    # loop layers:
    #   composite = composite @ attn
    #   composite = composite @ ffn
    # So yes, 4 multiplications.
    
    print(f"  Composite:\n{comp}")
    row_sums = comp.abs().sum(dim=-1)
    col_sums = comp.abs().sum(dim=-2)
    amax = max(row_sums.max().item(), col_sums.max().item())
    print(f"  Amax: {amax}")
    
    expected = 2.0 ** 4 # 16.0
    if abs(amax - 16.0) < 1e-4:
        print("  OK: Amax scaled correctly.")
    else:
        print(f"  WARNING: Expected 16.0, got {amax}")

if __name__ == "__main__":
    check_sinkhorn()
    check_param_count()
    check_gradients()
    check_matrices()
    check_amax_logic()
