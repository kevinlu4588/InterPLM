"""
Distributed multi-GPU training script for SAE on protein sequences.
Launch with: torchrun --nproc_per_node=8 distributed_train_sae.py
"""

import os
import math, random
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO

# ---------------- Distributed Setup ----------------
def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not running in distributed mode')
        return False, 0, 1, 0
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    return True, rank, world_size, local_rank

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

# ---------------- Config ----------------
ESM_NAME        = "facebook/esm2_t33_650M_UR50D"
TARGET_LAYER    = 24
MAX_LEN         = 1024
BATCH_SIZE_SEQ  = 8            # Reduced for multi-GPU (each GPU processes this many)
TOKENS_PER_STEP = 4096
TOKEN_CHUNK     = 8192
L1_WEIGHT       = 2e-3
LR              = 1e-3
SEED            = 17

# ---------------- Data Loading with Sharding ----------------
class DistributedFastaDataset:
    """Dataset that shards FASTA files across GPUs."""
    
    def __init__(self, fasta_paths: List[Path], rank: int, world_size: int, 
                 batch_size: int, max_len: int):
        self.fasta_paths = fasta_paths
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.max_len = max_len
        
        # Assign files to this rank
        self.assigned_files = [
            f for i, f in enumerate(fasta_paths) 
            if i % world_size == rank
        ]
        
    def __iter__(self):
        """Iterate through assigned FASTA files."""
        for fasta_path in self.assigned_files:
            buf = []
            with open(fasta_path, "rt") as fh:
                for rec in SeqIO.parse(fh, "fasta"):
                    s = str(rec.seq)
                    if not s:
                        continue
                    if len(s) > self.max_len:
                        s = s[:self.max_len]
                    buf.append(s)
                    if len(buf) == self.batch_size:
                        yield buf
                        buf = []
            if buf:
                yield buf

# ---------------- Model utilities (same as original) ----------------
def build_esm(device):
    tok = AutoTokenizer.from_pretrained(ESM_NAME, do_lower_case=False)
    mdl = AutoModel.from_pretrained(ESM_NAME).eval().to(device)
    return mdl, tok

@torch.no_grad()
def esm_layer_hidden_states(sequences, model, tokenizer, device, amp_dtype=torch.float16):
    batch = tokenizer(sequences, return_tensors="pt", add_special_tokens=False, padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    attn = batch["attention_mask"].to(torch.bool)
    
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        out = model(**batch, output_hidden_states=True, return_dict=True)
    hL = out.hidden_states[TARGET_LAYER]
    return hL, attn

# ---------------- SAE Model ----------------
class SimpleSAE(nn.Module):
    def __init__(self, d_in: int, n_feats: int, tied: bool = False):
        super().__init__()
        self.encoder = nn.Linear(d_in, n_feats, bias=False)
        self.decoder = nn.Linear(n_feats, d_in, bias=False)
        if tied:
            self.decoder.weight = self.encoder.weight.t()

    def forward(self, x: torch.Tensor):
        z = F.relu(self.encoder(x))
        xhat = self.decoder(z)
        return xhat, z

def sae_loss(x, xhat, z, l1):
    recon = F.mse_loss(xhat, x)
    # spars = z.abs().mean()
    spars = z.norm(p=1, dim=-1).mean()

    return recon + l1 * spars, recon, spars

def iter_token_chunks(hBTD, attn, chunk):
    B, T, D = hBTD.shape
    valid = attn.view(-1)
    H = hBTD.view(B*T, D)[valid]
    N = H.shape[0]
    if N == 0: return
    for i in range(0, N, chunk):
        yield H[i:i+chunk]

# ---------------- Distributed Trainer ----------------
def train_sae_distributed(
    fasta_paths: List[Path],
    rank: int,
    world_size: int,
    local_rank: int,
    d_hidden: int = 1280,
    n_feats: int = 16384,
    steps: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_freq: int = 500,
    log_freq: int = 50,
):
    device = torch.device(f'cuda:{local_rank}')
    
    # Set different seed per rank for data diversity
    torch.manual_seed(SEED + rank)
    random.seed(SEED + rank)
    
    # Build models
    model, tokenizer = build_esm(device)
    sae = SimpleSAE(d_in=d_hidden, n_feats=n_feats, tied=False).to(device)
    
    # Wrap SAE in DDP
    sae = DDP(sae, device_ids=[local_rank])
    
    # Optimizer
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    
    # Create distributed dataset
    dataset = DistributedFastaDataset(
        fasta_paths, rank, world_size, BATCH_SIZE_SEQ, MAX_LEN
    )
    
    # Only rank 0 handles checkpointing
    if rank == 0 and checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    total_tokens_accum = 0
    sae_steps = 0
    total_tokens_processed = 0
    
    history = {
        'steps': [],
        'loss': [],
        'recon': [],
        'sparsity': []
    }
    
    # Training loop
    if rank == 0:
        print(f"Distributed training on {len(fasta_paths)} FASTA files")
        print(f"World size: {world_size} GPUs")
        print(f"Files per GPU: ~{len(fasta_paths) // world_size}")
        print(f"SAE config: d_hidden={d_hidden}, n_feats={n_feats}")
        print("-" * 50)
    
    for seq_batch in dataset:
        # ESM forward
        hBTD, attn = esm_layer_hidden_states(seq_batch, model, tokenizer, device)
        
        # Process token chunks
        for x in iter_token_chunks(hBTD, attn, TOKEN_CHUNK):
            x = x.float()
            xhat, z = sae(x)
            loss, recon, spars = sae_loss(x, xhat, z, L1_WEIGHT)
            loss.backward()
            
            # Accumulate token count (local)
            local_tokens = x.shape[0]
            total_tokens_accum += local_tokens
            
            # Synchronize token count across GPUs
            tokens_tensor = torch.tensor([total_tokens_accum], device=device)
            dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
            global_tokens_accum = tokens_tensor.item()
            
            # Step when we've accumulated enough tokens globally
            if global_tokens_accum >= TOKENS_PER_STEP * world_size:
                # Average gradients across GPUs (done automatically by DDP)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sae_steps += 1
                total_tokens_accum = 0
                
                # Gather metrics from all GPUs
                metrics = torch.tensor([loss.item(), recon.item(), spars.item()], device=device)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                metrics = metrics / world_size
                
                if rank == 0:
                    # Store history
                    history['steps'].append(sae_steps)
                    history['loss'].append(metrics[0].item())
                    history['recon'].append(metrics[1].item())
                    history['sparsity'].append(metrics[2].item())
                    
                    # Logging
                    if sae_steps % log_freq == 0:
                        print(f"[step {sae_steps:5d}] loss={metrics[0]:.4f} | "
                              f"recon={metrics[1]:.4f} | spars={metrics[2]:.4f} | "
                              f"tokens/step≈{TOKENS_PER_STEP * world_size:,}")
                    
                    # Checkpointing (only rank 0)
                    if checkpoint_dir and sae_steps % checkpoint_freq == 0:
                        checkpoint_path = checkpoint_dir / f"sae_step_{sae_steps}.pt"
                        torch.save({
                            'sae_state_dict': sae.module.state_dict(),  # .module to unwrap DDP
                            'optimizer_state_dict': opt.state_dict(),
                            'step': sae_steps,
                            'config': {
                                'd_hidden': d_hidden,
                                'n_feats': n_feats,
                                'l1_weight': L1_WEIGHT,
                                'lr': LR,
                                'target_layer': TARGET_LAYER,
                            },
                            'history': history
                        }, checkpoint_path)
                        print(f"  → Saved checkpoint to {checkpoint_path}")
                
                if steps is not None and sae_steps >= steps:
                    if rank == 0:
                        print(f"Reached target steps ({steps})")
                    return sae, history
        
        del hBTD, attn
    
    # Final flush
    if total_tokens_accum > 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
    
    if rank == 0:
        print(f"Training complete! Total steps: {sae_steps}")
    
    return sae, history

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Distributed SAE training")
    parser.add_argument("--data-dir", type=str, 
                       default="/home/ec2-user/SageMaker/InterPLM/data/sharded_uniprot")
    parser.add_argument("--n-shards", type=int, default=8,
                       help="Number of FASTA shards")
    parser.add_argument("--n-feats", type=int, default=10240)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_distributed")
    parser.add_argument("--checkpoint-freq", type=int, default=250)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--save-final", type=str, default="final_distributed_sae.pt")
    
    args = parser.parse_args()
    
    # Setup distributed
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    if not is_distributed:
        print("This script requires distributed launch with torchrun")
        return
    
    # Generate FASTA paths
    fasta_paths = [
        Path(args.data_dir) / f"shard_{i:02d}.fasta" 
        for i in range(args.n_shards)
    ]
    
    # Validate files exist (only rank 0)
    if rank == 0:
        for path in fasta_paths:
            if not path.exists():
                raise FileNotFoundError(f"FASTA file not found: {path}")
        print(f"Found all {len(fasta_paths)} FASTA files")
    
    # Wait for rank 0 to validate
    dist.barrier()
    
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    
    try:
        # Train
        sae, history = train_sae_distributed(
            fasta_paths=fasta_paths,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            d_hidden=1280,
            n_feats=args.n_feats,
            steps=args.steps,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=args.checkpoint_freq,
            log_freq=args.log_freq
        )
        
        # Save final model (rank 0 only)
        if rank == 0 and args.save_final:
            final_path = Path(args.save_final)
            torch.save({
                'sae_state_dict': sae.module.state_dict(),
                'config': {
                    'd_hidden': 1280,
                    'n_feats': args.n_feats,
                    'l1_weight': L1_WEIGHT,
                    'target_layer': TARGET_LAYER,
                },
                'history': history
            }, final_path)
            print(f"\nSaved final model to {final_path}")
    
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()