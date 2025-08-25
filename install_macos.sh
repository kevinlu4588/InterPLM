#!/bin/bash

echo "Installing InterPLM on macOS (CPU-only)"
echo "========================================"
echo ""

# Create clean environment
echo "Step 1: Creating clean conda environment..."
conda create -n interplm_mac python=3.11 -y

echo ""
echo "Step 2: Activating environment..."
echo "Run: conda activate interplm_mac"
echo ""

echo "Step 3: Installing packages with compatible versions..."
cat << 'EOF'
# After activating the environment, run these commands:

# Upgrade pip first
pip install --upgrade pip

# Install NumPy 1.x first (critical!)
pip install 'numpy<2'

# Install PyTorch 2.1.x CPU version for macOS
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# Install transformers and accelerate with compatible versions
pip install transformers==4.36.0 accelerate==0.25.0

# Install other InterPLM requirements
pip install nnsight==0.2.21
pip install einops==0.8.0 
pip install fair-esm==2.0.0
pip install 'huggingface-hub>=0.20.0'
pip install biopython==1.84

# Install data processing and visualization
pip install datasets pandas
pip install plotly streamlit py3dmol
pip install matplotlib seaborn
pip install scipy scikit-learn h5py

# Install utilities
pip install pyyaml typed-argument-parser 
pip install umap-learn python-dotenv
pip install wandb multiprocess

# Finally, install InterPLM itself
cd /path/to/InterPLM
pip install -e .

# Test the installation
python -c "
import numpy as np
import torch
import transformers
print(f'NumPy: {np.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print('✅ All packages imported successfully!')
print(f'Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')"
EOF

echo ""
echo "========================================"
echo "Installation complete!"
echo ""
echo "To use the dashboard:"
echo "1. conda activate interplm_mac"
echo "2. cd /path/to/InterPLM"
echo "3. python interplm/dashboard/app_remote.py --repo_id kevinlu4588/interplm-data"