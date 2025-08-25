#!/usr/bin/env python
"""
Quick fix script for macOS InterPLM installation issues.
This script will uninstall problematic packages and reinstall compatible versions.
"""

import subprocess
import sys

def run_command(cmd):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {cmd}")
            return True
        else:
            print(f"❌ Failed: {cmd}")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception running {cmd}: {e}")
        return False

def main():
    print("InterPLM macOS Quick Fix")
    print("=" * 50)
    print()
    
    # Check current environment
    print("Checking current environment...")
    run_command("python --version")
    
    print("\nStep 1: Uninstalling problematic packages...")
    packages_to_remove = [
        "numpy", "torch", "torchvision", "transformers", 
        "accelerate", "nnsight"
    ]
    
    for pkg in packages_to_remove:
        run_command(f"pip uninstall -y {pkg}")
    
    print("\nStep 2: Installing compatible versions...")
    
    # Critical: Install NumPy 1.x first
    if not run_command("pip install 'numpy<2'"):
        print("Failed to install NumPy. Stopping.")
        sys.exit(1)
    
    # Install PyTorch 2.1.2 (CPU version)
    if not run_command("pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu"):
        print("Failed to install PyTorch. Trying alternative...")
        run_command("pip install torch==2.1.2 torchvision==0.16.2")
    
    # Install other packages with compatible versions
    compatible_packages = [
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "nnsight==0.2.21",
    ]
    
    for pkg in compatible_packages:
        if not run_command(f"pip install {pkg}"):
            print(f"Warning: Failed to install {pkg}")
    
    print("\nStep 3: Verifying installation...")
    
    test_code = """
import sys
try:
    import numpy as np
    import torch
    import transformers
    import nnsight
    
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ Transformers: {transformers.__version__}")
    print(f"✅ Nnsight imported successfully")
    print(f"✅ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("\\n🎉 All packages working correctly!")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
"""
    
    result = subprocess.run([sys.executable, "-c", test_code], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print("\n⚠️  Some packages failed to import. You may need to create a fresh environment.")
        print("Run: bash install_macos.sh")
    else:
        print("\n✅ Fix completed successfully!")
        print("\nYou can now run the dashboard:")
        print("python interplm/dashboard/app_remote.py --repo_id kevinlu4588/interplm-data")

if __name__ == "__main__":
    main()