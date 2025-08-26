#!/bin/bash

# Script to create an environment file from your current conda environment
# This captures the exact versions you have installed

echo "Creating environment file from current conda environment..."

# Option 1: Export everything (comprehensive but may have platform-specific packages)
echo "Creating comprehensive env file (env_full.yml)..."
conda env export > env_full.yml
echo "✓ Created env_full.yml (includes all packages with exact versions)"

# Option 2: Export only explicitly installed packages (more portable)
echo "Creating portable env file (env_portable.yml)..."
conda env export --from-history > env_portable.yml
echo "✓ Created env_portable.yml (only explicitly installed packages)"

# Option 3: Create a minimal requirements.txt for pip packages
echo "Creating pip requirements file..."
pip list --format=freeze | grep -E "biopython|datasets|einops|fair-esm|huggingface|matplotlib|multiprocess|nnsight|numpy|pandas|plotly|pyyaml|py3dmol|scikit-learn|scipy|seaborn|streamlit|torch|torchvision|transformers|typed-argument-parser|umap-learn|wandb|python-dotenv" > requirements_minimal.txt
echo "✓ Created requirements_minimal.txt (InterPLM-specific packages)"

# Option 4: Create a hybrid approach - conda for base, pip for specifics
cat > env_hybrid.yml << 'EOF'
name: interplm
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.12  # Using your current Python version
  - pip
  - numpy
  - scipy
  - scikit-learn
  - matplotlib
  - seaborn
  - h5py
  - pytorch
  - torchvision
  - pip:
EOF

# Append pip packages with their current versions
echo "    # Auto-generated from current environment" >> env_hybrid.yml
pip list --format=freeze | grep -E "biopython|datasets|einops|fair-esm|huggingface|nnsight|pandas|plotly|pyyaml|py3dmol|streamlit|transformers|typed-argument-parser|umap-learn|wandb|multiprocess|python-dotenv" | sed 's/^/    - /' >> env_hybrid.yml

echo "✓ Created env_hybrid.yml (hybrid conda/pip approach)"

echo ""
echo "Created 4 different environment files:"
echo "1. env_full.yml - Complete export (may have compatibility issues on different platforms)"
echo "2. env_portable.yml - Only explicitly installed packages (most portable)"
echo "3. requirements_minimal.txt - Minimal pip requirements for InterPLM"
echo "4. env_hybrid.yml - Hybrid approach with conda base and pip specifics"
echo ""
echo "Recommended: Use env_hybrid.yml or env_portable.yml for best compatibility"
echo ""
echo "To create a new environment from these files:"
echo "  conda env create -f env_hybrid.yml"
echo "  OR"
echo "  conda create -n interplm python=3.12"
echo "  conda activate interplm"
echo "  pip install -r requirements_minimal.txt"