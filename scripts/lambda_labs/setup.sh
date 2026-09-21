#!/bin/bash
# =============================================================================
# Lambda Labs A100 Setup Script
# Part 2 of mHC reproduction: https://poissonlabs.ai/research/mhc-reproduction-part-2/
#
# This script provisions a Lambda Labs instance for large-scale training.
# Run once after SSH into a fresh instance.
#
# Usage:
#   ssh ubuntu@<instance-ip>
#   git clone https://github.com/poisson-labs/mhc-repro.git
#   cd mhc-repro
#   bash scripts/lambda_labs/setup.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "mHC Part 2 - Lambda Labs Setup"
echo "=============================================="

# Check GPU
echo ""
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME"

# Check for supported GPU
if [[ "$GPU_NAME" =~ "H100" ]]; then
    echo "H100 detected - excellent choice for speed"
elif [[ "$GPU_NAME" =~ "A100" ]]; then
    echo "A100 detected - solid choice"
else
    echo "WARNING: Expected H100 or A100 GPU, got $GPU_NAME"
    echo "Results may differ from documented experiments."
fi

# Environment info (for reproducibility)
echo ""
echo "Recording environment..."
echo "Date: $(date -Iseconds)" > environment.txt
echo "GPU: $GPU_NAME" >> environment.txt
echo "CUDA: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)" >> environment.txt
echo "Python: $(python3 --version)" >> environment.txt
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')" >> environment.txt
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')" >> environment.txt
cat environment.txt

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (Lambda Labs usually has this, but be explicit)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo ""
echo "Installing dependencies..."
pip install -r scripts/lambda_labs/requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # Quick bf16 test
    x = torch.randn(1000, 1000, device='cuda', dtype=torch.bfloat16)
    y = x @ x.T
    print(f'bf16 test passed')

    # Flash attention check
    import torch.nn.functional as F
    q = torch.randn(1, 8, 64, 64, device='cuda')
    k = torch.randn(1, 8, 64, 64, device='cuda')
    v = torch.randn(1, 8, 64, 64, device='cuda')
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f'Flash Attention test passed')
"

# Verify model can be created
echo ""
echo "Verifying model creation..."
python3 -c "
import sys
sys.path.insert(0, '.')
from scripts.lambda_labs.train_c4 import ModelConfig, GPT

# Test model creation at 1B scale
config = ModelConfig(n_layers=32, hidden_dim=2048, connection_type='mhc')
print(f'Estimated params: {config.estimate_params()/1e9:.2f}B')

# Actually create the model
model = GPT(config).cuda()
n_params = model.count_parameters()
print(f'Actual params: {n_params/1e9:.2f}B')

# Test forward pass with bf16
import torch
from torch.cuda.amp import autocast

with autocast(enabled=True, dtype=torch.bfloat16):
    idx = torch.randint(0, 50257, (4, 1024), device='cuda')
    targets = torch.randint(0, 50257, (4, 1024), device='cuda')
    logits, loss = model(idx, targets)
    print(f'Forward pass test passed (loss={loss.item():.4f})')

    # Memory check
    mem_allocated = torch.cuda.memory_allocated() / 1e9
    mem_reserved = torch.cuda.memory_reserved() / 1e9
    print(f'Memory: {mem_allocated:.1f}GB allocated, {mem_reserved:.1f}GB reserved')

print('Model verification complete!')
"

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p runs_c4
mkdir -p figures

# Login to W&B (optional)
echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. (Optional) Login to W&B: wandb login"
echo "  2. Run experiments: bash scripts/lambda_labs/run_experiments.sh"
echo ""
echo "Or run a single experiment:"
echo "  python scripts/lambda_labs/train_c4.py --connection mhc --depth 32 --seed 42"
echo ""
