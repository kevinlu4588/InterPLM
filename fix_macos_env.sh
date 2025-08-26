#!/bin/bash

echo "Fixing InterPLM environment for macOS (CPU-only)..."
echo ""
echo "This script will help you fix the NumPy and PyTorch compatibility issues."
echo ""

# Option 1: Quick fix in existing environment
echo "=== Option 1: Quick Fix (try this first) ==="
echo "Run these commands in your terminal:"
echo ""
echo "conda activate interplm"
echo "pip uninstall -y numpy torch torchvision transformers accelerate"
echo "pip install 'numpy<2' torch==2.0.1 torchvision==0.15.2 'transformers==4.35.0' 'accelerate==0.24.0'"
echo ""

# Option 2: Create fresh environment
echo "=== Option 2: Fresh Environment (if Option 1 doesn't work) ==="
echo "Run these commands:"
echo ""
echo "conda deactivate"
echo "conda env remove -n interplm"
echo "conda env create -f env_macos_cpu.yml"
echo "conda activate interplm"
echo "pip install -e ."
echo ""

# Option 3: Manual pip-only install
echo "=== Option 3: Pip-only Install (most reliable) ==="
echo "Run these commands:"
echo ""
cat << 'EOF'
conda create -n interplm_clean python=3.11 -y
conda activate interplm_clean

# Install with specific versions that work together
pip install --upgrade pip
pip install 'numpy<2'
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
pip install 'transformers==4.35.0' 'accelerate==0.24.0'
pip install nnsight==0.2.21
pip install einops==0.8.0
pip install fair-esm==2.0.0
pip install 'huggingface-hub>=0.20'
pip install biopython==1.84
pip install datasets pandas plotly streamlit py3dmol
pip install pyyaml typed-argument-parser umap-learn python-dotenv wandb

# Install InterPLM
cd /path/to/InterPLM
pip install -e .
EOF

echo ""
echo "=== Testing the Fix ==="
echo "After running one of the above options, test with:"
echo "python -c 'import torch; import transformers; import numpy; print(f\"NumPy: {numpy.__version__}\"); print(f\"PyTorch: {torch.__version__}\"); print(f\"Transformers: {transformers.__version__}\"); print(\"Success!\")'"
echo ""
echo "Then try running the dashboard again:"
echo "python interplm/dashboard/app_remote.py --repo_id kevinlu4588/interplm-data"