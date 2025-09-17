import math, random
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO

# ---------------- Config ----------------
ESM_NAME        = "facebook/esm2_t33_650M_UR50D"
TARGET_LAYER    = 24           # 0 = embeddings, 1..33 = transformer blocks; pick one
MAX_LEN         = 1024
BATCH_SIZE_SEQ  = 64           # increase until GPU stays busy
TOKENS_PER_STEP = 4096         # gradient-accum token budget
TOKEN_CHUNK     = 8192         # process tokens in chunks to cap VRAM (adjust)
L1_WEIGHT       = 0.1         # sparsity strength
LR              = 1e-3
SEED            = 17
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)
print(f"Using device: {DEVICE}")

# ---------------- Data: FASTA -> batches of sequences ----------------
def batched_sequences_fasta(path: Path, batch_size: int, max_len: int) -> Iterable[List[str]]:
    buf = []
    with open(path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            s = str(rec.seq)
            if not s:
                continue
            if len(s) > max_len:
                s = s[:max_len]
            buf.append(s)
            if len(buf) == batch_size:
                yield buf; buf = []
    if buf:
        yield buf

# ---------------- Model ----------------
def build_esm(device=DEVICE):
    tok = AutoTokenizer.from_pretrained(ESM_NAME, do_lower_case=False)
    mdl = AutoModel.from_pretrained(ESM_NAME).eval().to(device)
    return mdl, tok

@torch.no_grad()
def esm_layer_hidden_states(
    sequences: List[str],
    model, tokenizer,
    device=DEVICE,
    amp_dtype=torch.float16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      hL: [B, T, D] from TARGET_LAYER (no grads)
      attn: [B, T] bool valid-token mask
    """
    batch = tokenizer(sequences, return_tensors="pt", add_special_tokens=False, padding=True)
    # pin + async H2D
    for k, v in batch.items(): batch[k] = v.pin_memory()
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    attn = batch["attention_mask"].to(torch.bool)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        out = model(**batch, output_hidden_states=True, return_dict=True)
    # pick only the layer we care about; do NOT stack all layers
    hL = out.hidden_states[TARGET_LAYER]           # [B,T,D]
    return hL, attn

# ---------------- SAE ----------------
class SimpleSAE(nn.Module):
    """
    ReLU sparse autoencoder. Set n_feats to your desired dictionary size (e.g., 16k).
    Use untied weights for stability; tie if you prefer.
    """
    def __init__(self, d_in: int, n_feats: int, tied: bool = False):
        super().__init__()
        self.encoder = nn.Linear(d_in, n_feats, bias=False)
        self.decoder = nn.Linear(n_feats, d_in, bias=False)
        if tied:
            self.decoder.weight = self.encoder.weight.t()

    def forward(self, x: torch.Tensor):
        z = F.relu(self.encoder(x))        # [N, F]
        xhat = self.decoder(z)             # [N, D]
        return xhat, z

def sae_loss(x: torch.Tensor, xhat: torch.Tensor, z: torch.Tensor, l1: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = F.mse_loss(xhat, x)           # reconstruction term
    spars = z.abs().mean()                # L1 on codes
    return recon + l1 * spars, recon, spars

# ---------------- Token chunker ----------------
def iter_token_chunks(hBTD: torch.Tensor, attn: torch.Tensor, chunk: int) -> Iterable[torch.Tensor]:
    """
    Flatten valid tokens from [B,T,D] -> [N,D], then yield in pieces of at most `chunk`.
    Keeps everything on GPU, avoids big peak memory.
    """
    # mask valid tokens
    B, T, D = hBTD.shape
    valid = attn.view(-1)                               # [B*T]
    H = hBTD.view(B*T, D)[valid]                        # [N,D] on GPU
    N = H.shape[0]
    if N == 0: return
    # slice into chunks
    for i in range(0, N, chunk):
        yield H[i:i+chunk]

# ---------------- Trainer (streaming) ----------------
def train_sae_streaming(
    fasta_paths: List[Path],
    d_hidden: int = 1280,
    n_feats: int = 10240,
    steps: Optional[int] = None,     # stop after N SAE optimizer steps (optional)
    checkpoint_dir: Optional[Path] = None,
    checkpoint_freq: int = 500,
    log_freq: int = 50,
):
    model, tokenizer = build_esm(DEVICE)
    sae = SimpleSAE(d_in=d_hidden, n_feats=n_feats, tied=False).to(DEVICE)
    # keep SAE in fp32 for stability; cast inputs to float()
    opt = torch.optim.Adam(sae.parameters(), lr=LR)

    scaler = torch.cuda.amp.GradScaler(enabled=False)   # SAE in fp32; no AMP on SAE by default

    total_tokens_accum = 0
    sae_steps = 0
    total_tokens_processed = 0
    
    # Create checkpoint directory if specified
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training history for logging
    history = {
        'steps': [],
        'loss': [],
        'recon': [],
        'sparsity': []
    }

    # iterate shards -> batches -> token-chunks
    print(f"Training on {len(fasta_paths)} FASTA file(s)")
    print(f"SAE config: d_hidden={d_hidden}, n_feats={n_feats}, L1_weight={L1_WEIGHT}")
    print("-" * 50)
    
    for fasta_idx, fasta in enumerate(fasta_paths):
        print(f"Processing FASTA {fasta_idx+1}/{len(fasta_paths)}: {fasta.name}")
        
        for batch_idx, seq_batch in enumerate(batched_sequences_fasta(fasta, BATCH_SIZE_SEQ, MAX_LEN)):
            # 1) ESM forward (no grad), get target layer only
            hBTD, attn = esm_layer_hidden_states(seq_batch, model, tokenizer, DEVICE, amp_dtype=torch.float16)  # hBTD fp16 on GPU

            # 2) stream token chunks from this batch
            for x in iter_token_chunks(hBTD, attn, TOKEN_CHUNK):
                # Option A: train SAE in fp32 for stability
                x = x.float()                      # [n,D] on GPU
                xhat, z = sae(x)
                loss, recon, spars = sae_loss(x, xhat, z, L1_WEIGHT)
                loss.backward()                    # accumulate grads over chunks

                total_tokens_accum += x.shape[0]
                total_tokens_processed += x.shape[0]
                
                # step when we reached TOKENS_PER_STEP tokens
                if total_tokens_accum >= TOKENS_PER_STEP:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    sae_steps += 1
                    total_tokens_accum = 0
                    
                    # Store history
                    history['steps'].append(sae_steps)
                    history['loss'].append(loss.item())
                    history['recon'].append(recon.item())
                    history['sparsity'].append(spars.item())

                    # Logging
                    if sae_steps % log_freq == 0:
                        avg_sparsity_rate = (z > 0).float().mean().item()
                        print(f"[step {sae_steps:5d}] loss={loss.item():.4f} | "
                              f"recon={recon.item():.4f} | spars={spars.item():.4f} | "
                              f"active_rate={avg_sparsity_rate:.3f} | "
                              f"total_tokens={total_tokens_processed:,}")
                    
                    # Checkpointing
                    if checkpoint_dir and sae_steps % checkpoint_freq == 0:
                        checkpoint_path = checkpoint_dir / f"sae_step_{sae_steps}.pt"
                        torch.save({
                            'sae_state_dict': sae.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'step': sae_steps,
                            'total_tokens': total_tokens_processed,
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
                        print(f"Reached target steps ({steps}), stopping training.")
                        return sae, history  # early stop

            # IMPORTANT: free batch tensors promptly
            del hBTD, attn
            
            # Optional: periodic memory clearing (use sparingly)
            if batch_idx % 100 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()

    # flush tail if we ended mid-accumulation
    if total_tokens_accum > 0:
        opt.step()
        opt.zero_grad(set_to_none=True)
        sae_steps += 1
        print(f"Final step {sae_steps} (flushed {total_tokens_accum} accumulated tokens)")
    
    print(f"Training complete! Total steps: {sae_steps}, Total tokens: {total_tokens_processed:,}")
    return sae, history

# ---------------- Evaluation utilities ----------------
@torch.no_grad()
def evaluate_sae(sae: SimpleSAE, fasta_path: Path, n_samples: int = 100):
    """Quick evaluation of SAE reconstruction quality and sparsity."""
    model, tokenizer = build_esm(DEVICE)
    sae.eval()
    
    total_recon = 0
    total_sparsity = 0
    total_tokens = 0
    active_rates = []
    
    sample_count = 0
    for seq_batch in batched_sequences_fasta(fasta_path, BATCH_SIZE_SEQ, MAX_LEN):
        if sample_count >= n_samples:
            break
            
        hBTD, attn = esm_layer_hidden_states(seq_batch, model, tokenizer, DEVICE)
        
        for x in iter_token_chunks(hBTD, attn, TOKEN_CHUNK):
            x = x.float()
            xhat, z = sae(x)
            _, recon, spars = sae_loss(x, xhat, z, L1_WEIGHT)
            
            total_recon += recon.item() * x.shape[0]
            total_sparsity += spars.item() * x.shape[0]
            total_tokens += x.shape[0]
            active_rates.append((z > 0).float().mean(dim=0).cpu())
        
        sample_count += len(seq_batch)
        del hBTD, attn
    
    # Aggregate stats
    avg_recon = total_recon / total_tokens
    avg_sparsity = total_sparsity / total_tokens
    all_active = torch.cat(active_rates, dim=0).mean(dim=0)
    dead_features = (all_active == 0).sum().item()
    
    print(f"\nEvaluation Results (n={min(sample_count, n_samples)} sequences):")
    print(f"  Avg reconstruction loss: {avg_recon:.5f}")
    print(f"  Avg sparsity (L1): {avg_sparsity:.5f}")
    print(f"  Dead features: {dead_features}/{sae.encoder.out_features} ({100*dead_features/sae.encoder.out_features:.1f}%)")
    print(f"  Mean activation rate: {all_active.mean().item():.4f}")
    
    return {
        'recon_loss': avg_recon,
        'sparsity': avg_sparsity,
        'dead_features': dead_features,
        'mean_activation_rate': all_active.mean().item()
    }

# ---------------- Main execution ----------------
def main():
    parser = argparse.ArgumentParser(description="Stream-train SAE on protein sequences")
    parser.add_argument("fasta_files", nargs='+', type=str, 
                       help="Path(s) to FASTA file(s)")
    parser.add_argument("--n-feats", type=int, default=16384,
                       help="SAE dictionary size (default: 16384)")
    parser.add_argument("--steps", type=int, default=None,
                       help="Max training steps (default: process all data)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory for saving checkpoints")
    parser.add_argument("--checkpoint-freq", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--log-freq", type=int, default=50,
                       help="Log metrics every N steps")
    parser.add_argument("--eval", action="store_true",
                       help="Run evaluation after training")
    parser.add_argument("--eval-samples", type=int, default=100,
                       help="Number of sequences for evaluation")
    parser.add_argument("--save-final", type=str, default=None,
                       help="Path to save final SAE model")
    
    args = parser.parse_args()
    
    # Convert paths
    fasta_paths = [Path(f) for f in args.fasta_files]
    
    # Validate files exist
    for path in fasta_paths:
        if not path.exists():
            raise FileNotFoundError(f"FASTA file not found: {path}")
    
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    
    # Train SAE
    print(f"Starting SAE training...")
    print(f"FASTA files: {[str(p) for p in fasta_paths]}")
    
    sae, history = train_sae_streaming(
        fasta_paths=fasta_paths,
        d_hidden=1280,  # ESM2-650M hidden size
        n_feats=args.n_feats,
        steps=args.steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        log_freq=args.log_freq
    )
    
    # Save final model
    if args.save_final:
        final_path = Path(args.save_final)
        torch.save({
            'sae_state_dict': sae.state_dict(),
            'config': {
                'd_hidden': 1280,
                'n_feats': args.n_feats,
                'l1_weight': L1_WEIGHT,
                'target_layer': TARGET_LAYER,
            },
            'history': history
        }, final_path)
        print(f"\nSaved final model to {final_path}")
    
    # Optional evaluation
    if args.eval and len(fasta_paths) > 0:
        print("\nRunning evaluation...")
        eval_results = evaluate_sae(sae, fasta_paths[0], n_samples=args.eval_samples)
        
        # Save evaluation results
        if checkpoint_dir:
            eval_path = checkpoint_dir / "eval_results.json"
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"Saved evaluation results to {eval_path}")
    
    # Plot training curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        if history['steps']:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].plot(history['steps'], history['loss'])
            axes[0].set_xlabel('Step')
            axes[0].set_ylabel('Total Loss')
            axes[0].set_title('Training Loss')
            axes[0].grid(True)
            
            axes[1].plot(history['steps'], history['recon'])
            axes[1].set_xlabel('Step')
            axes[1].set_ylabel('Reconstruction Loss')
            axes[1].set_title('Reconstruction Loss')
            axes[1].grid(True)
            
            axes[2].plot(history['steps'], history['sparsity'])
            axes[2].set_xlabel('Step')
            axes[2].set_ylabel('Sparsity (L1)')
            axes[2].set_title('Sparsity')
            axes[2].grid(True)
            
            plt.tight_layout()
            if checkpoint_dir:
                plot_path = checkpoint_dir / "training_curves.png"
                plt.savefig(plot_path)
                print(f"Saved training curves to {plot_path}")
            else:
                plt.show()
    except ImportError:
        pass  # matplotlib not available

if __name__ == "__main__":
    # Set environment variables for better performance
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    main()