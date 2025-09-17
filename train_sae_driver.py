"""
Example usage script for stream_train_sae.py
"""

import subprocess
import sys

# Example 1: Basic training on a single FASTA file
def example_basic():
    """Train SAE on a single FASTA file with default settings."""
    cmd = [
        "python", "stream_train_sae.py",
        "proteins.fasta",  # Your FASTA file
        "--n-feats", "16384",
        "--checkpoint-dir", "checkpoints",
        "--log-freq", "50"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)

# # Example 2: Train on multiple FASTA files with evaluation
# def example_multi_file():
#     """Train on multiple FASTA shards and run evaluation."""
#     cmd = [
#         "python", "stream_train_sae.py",
#         "/home/ec2-user/SageMaker/InterPLM/data/sharded_uniprot/shard_00.fasta", "/home/ec2-user/SageMaker/InterPLM/data/sharded_uniprot/shard_01.fasta",
#         "--n-feats", "10240",  # Larger dictionary
#         "--steps", "5000",  # Stop after 5000 steps
#         "--checkpoint-dir", "checkpoints_large",
#         "--checkpoint-freq", "250",
#         "--eval",  # Run evaluation after training
#         "--eval-samples", "200",
#         "--save-final", "final_sae_model.pt"
#     ]
#     print("Running:", " ".join(cmd))
#     subprocess.run(cmd)

def example_multi_file():
    """Train on all 8 FASTA shards with evaluation."""
    # Generate all 8 shard paths
    shard_paths = [
        f"/home/ec2-user/SageMaker/InterPLM/data/sharded_uniprot/shard_{i:02d}.fasta" 
        for i in range(8)
    ]
    
    cmd = [
        "python", "stream_train_sae.py",
        *shard_paths,  # Unpack all 8 paths
        "--n-feats", "10240",
        "--steps", "5000",
        "--checkpoint-dir", "checkpoints_large",
        "--checkpoint-freq", "1000",
        "--eval",
        "--eval-samples", "200",
        "--save-final", "final_sae_model.pt"
    ]
    print(f"Running on {len(shard_paths)} FASTA files:")
    for path in shard_paths:
        print(f"  - {path}")
    subprocess.run(cmd)

example_multi_file()

# Example 3: Quick test run
def example_test_run():
    """Quick test with small dictionary and few steps."""
    cmd = [
        "python", "stream_train_sae.py",
        "/home/ec2-user/SageMaker/InterPLM/data/sharded_uniprot/shard_00.fasta",
        "--n-feats", "4096",  # Small dictionary for testing
        "--steps", "100",  # Just 100 steps
        "--checkpoint-dir", "test_checkpoints",
        "--log-freq", "10",
        "--eval",
        "--eval-samples", "10"
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd)

# Example 4: Resume from checkpoint (requires modification to main script)
def load_checkpoint_example():
    """Example of how to load and use a trained SAE."""
    import torch
    from pathlib import Path
    from stream_train_sae import SimpleSAE
    
    # Load checkpoint
    checkpoint_path = Path("checkpoints/sae_step_1000.pt")
    checkpoint = torch.load(checkpoint_path)
    
    # Recreate SAE
    config = checkpoint['config']
    sae = SimpleSAE(
        d_in=config['d_hidden'],
        n_feats=config['n_feats'],
        tied=False
    )
    
    # Load weights
    sae.load_state_dict(checkpoint['sae_state_dict'])
    sae.eval()
    
    print(f"Loaded SAE from step {checkpoint['step']}")
    print(f"Config: {config}")
    
    # Use the SAE for inference
    # ... your inference code here ...
    
    return sae

if __name__ == "__main__":
    print("SAE Training Examples")
    print("-" * 50)
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name == "basic":
            example_basic()
        elif example_name == "multi":
            example_multi_file()
        elif example_name == "test":
            example_test_run()
        elif example_name == "load":
            load_checkpoint_example()
        else:
            print(f"Unknown example: {example_name}")
            print("Available: basic, multi, test, load")
    else:
        print("Usage: python examples.py [basic|multi|test|load]")
        print("\nAvailable examples:")
        print("  basic - Train on single FASTA file")
        print("  multi - Train on multiple files with eval")
        print("  test  - Quick test run")
        print("  load  - Load checkpoint example")