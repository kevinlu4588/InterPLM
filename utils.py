from interplm.sae.inference import load_sae_from_hf
import os, json, gzip, math, time, random
from pathlib import Path
from typing import List, Tuple, Iterable, Optional
from transformers import AutoTokenizer, AutoModel
import os, gzip, math, random, json, gc
from pathlib import Path
from typing import List, Iterable, Tuple, Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Bio import SeqIO
import torch
from typing import List, Literal, Tuple, Optional
from pathlib import Path
import pandas as pd, torch, os, gc
from interplm.sae.inference import load_sae_from_hf
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, EsmModel, AutoModel
from interplm.sae.inference import load_sae_from_hf
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import random
DEVICE="cuda"
DTYPE  = torch.float16
MAX_LEN=1024
BATCH_SIZE_SEQ=128
DATASET_NAME = "uniprot_sprot"
ESM_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_LEN = 1024
SEED = 17
random.seed(SEED)
BATCH_SIZE_SEQ = 128

FASTA_GZ = Path("/home/ec2-user/SageMaker/InterPLM/data/uniprot/uniprot_sprot.fasta.gz")
OUT_DIR  = Path(f"/home/ec2-user/SageMaker/InterPLM/data/esm2_hidden_states/{DATASET_NAME}")
MAX_BATCHES=24


def write_json(path: Path, obj: dict):
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def save_batch_shards(hs, attn_mask, out_dir: Path, batch_idx: int):
    L, B, T, D = hs.shape
    # mask shard (for this batch)
    mask_path = out_dir / "masks" / f"batch_{batch_idx:05d}.pt"
    if not mask_path.exists():
        ensure_dir(mask_path.parent)
        seq_lens = attn_mask.sum(dim=1).to(torch.int32).cpu()
        torch.save(
            {"attn_mask": attn_mask.cpu(),
             "seq_lengths": seq_lens,
             "batch_size": int(B),
             "tokens": int(T),
             "batch_index": int(batch_idx)},
            mask_path,
        )
    # layer shards
    for l in range(L):
        layer_dir = out_dir / f"layer_{l:02d}"
        ensure_dir(layer_dir)
        shard_path = layer_dir / f"batch_{batch_idx:05d}.pt"
        if shard_path.exists():
            continue
        torch.save(
            {"hidden_states": hs[l].cpu(),  # [B, T, D]
             "batch_size": int(B), "tokens": int(T), "width": int(D),
             "layer": int(l), "batch_index": int(batch_idx)},
            shard_path,
        )

def write_manifest(out_dir: Path, **kwargs):
    write_json(out_dir / "manifest.json", {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **kwargs,
        "layout": {
            "layers_folder": "layer_{:02d}",
            "layer_shard": "batch_{:05d}.pt",
            "masks_folder": "masks",
            "mask_shard": "batch_{:05d}.pt",
            "tensor_keys": {"hidden_states": "[B,T,D]", "attn_mask": "[B,T]"}
        }
    })


# ------------- Count sequences once (for progress/manifest) -------------
def count_sequences(fasta_gz: Path) -> int:
    with gzip.open(fasta_gz, "rt") as fh:
        return sum(1 for _ in SeqIO.parse(fh, "fasta"))

# ------------- Model & extract (mostly same) -------------
def build_model_and_tokenizer(device, esm_name=ESM_NAME):
    tokenizer = AutoTokenizer.from_pretrained(esm_name, do_lower_case=False)
    model = AutoModel.from_pretrained(esm_name).eval().to(device)
    return model, tokenizer


def iter_swissprot_sequences(fasta_gz: Path, max_len: int = MAX_LEN):
    with gzip.open(fasta_gz, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            seq = str(rec.seq)
            if not seq:
                continue
            yield rec.id, (seq if len(seq) <= max_len else seq[:max_len])
def batched_sequences_from_iterator(sequences, batch_size):
    """Create batches from an iterator of sequences"""
    batch = []
    for seq in sequences:
        batch.append(seq)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:  # Don't forget the last partial batch
        yield batch
def batched_sequences(fasta_gz: Path, batch_size: int = BATCH_SIZE_SEQ):
    buf = []
    for _, seq in iter_swissprot_sequences(fasta_gz):
        buf.append(seq)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


@torch.no_grad()
def extract_sae_features(hidden_states: torch.Tensor, sae):
    """
    Pass ESM hidden states through the Sparse Autoencoder (SAE).

    Args
    ----
    hidden_states : torch.Tensor
        Shape [B, L, d] or [L, d].
        - B = batch size (optional if unsqueezed)
        - L = sequence length
        - d = ESM embedding dimension (e.g., 1280 for esm2_t33_650M)

    Returns
    -------
    sae_features : torch.Tensor
        Shape [B, L, F]
        Sparse latent features per residue.
        F = number of SAE dictionary atoms / features.

    recon : torch.Tensor
        Shape [B, L, d]
        Reconstructed embeddings in token space.

    error : torch.Tensor
        Shape [B, L, d]
        Residual = hidden_states - recon
    """
    if hidden_states.dim() == 2:          # [L, d]
        hidden_states = hidden_states.unsqueeze(0)  # → [1, L, d]
    x = hidden_states.to(torch.float32)      # <- ensure fp32 for SAE

    # SAE should have encode() and decode() that operate on last dimension
    sae_features = sae.encode(x)     # [B, L, F]
    recon        = sae.decode(sae_features)      # [B, L, d]
    error        = hidden_states - recon         # [B, L, d]

    return sae_features, recon, error

def pool_sequence_features(
    features: torch.Tensor,   # [B, L, F] or [L, F]
    method: str = "maX",
    mask: torch.Tensor = None # optional [B, L] attention mask
) -> torch.Tensor:
    """
    Pool per-residue features to per-sequence vectors.

    Args
    ----
    features : torch.Tensor
        Shape [B, L, F] (or [L, F] → will unsqueeze to batch 1).
        - B = batch size
        - L = sequence length
        - F = number of SAE features
    method : str
        "maX" → concatenate mean + max → [B, 2F]
        "mean"     → masked mean → [B, F]
        "max"      → masked max  → [B, F]
    mask : torch.Tensor, optional
        Shape [B, L] bool (True = valid residue, False = pad).
        If None, assumes all tokens are valid.

    Returns
    -------
    pooled : torch.Tensor
        Shape depends on method:
          - maX: [B, 2F]
          - mean or max: [B, F]
    """
    if features.dim() == 2:  # [L, F]
        features = features.unsqueeze(0)  # [1, L, F]

    B, L, F = features.shape
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=features.device)

    # apply mask
    mask_f = mask.float().unsqueeze(-1)  # [B, L, 1]
    feats_masked = features * mask_f

    if method == "mean":
        pooled = feats_masked.sum(1) / mask_f.sum(1).clamp_min(1e-8)
        return pooled  # [B, F]

    elif method == "max":
        very_neg = torch.finfo(features.dtype).min
        feats_masked = feats_masked.masked_fill(~mask.unsqueeze(-1), very_neg)
        return feats_masked.max(1).values  # [B, F]

    elif method == "maX":
        mean_pool = feats_masked.sum(1) / mask_f.sum(1).clamp_min(1e-8)  # [B, F]
        very_neg = torch.finfo(features.dtype).min
        feats_masked = feats_masked.masked_fill(~mask.unsqueeze(-1), very_neg)
        max_pool = feats_masked.max(1).values  # [B, F]
        return torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2F]

    else:
        raise ValueError(f"Unknown pooling method: {method}")
        
@torch.no_grad()
def extract_esm_features_batch(
    sequences: List[str],
    layer_sel: int | Literal["last"] = "last",   # <— changed name/type
    device: torch.device = DEVICE,
    dtype = torch.float16,
    model = None,
    tokenizer = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = tokenizer(sequences, return_tensors="pt", add_special_tokens=False, padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    attn_mask = batch["attention_mask"].to(torch.bool)

    with torch.autocast(device_type="cuda", dtype=dtype):
        out = model(**batch, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states  # tuple: [emb, layer1, ..., layerN] each [B,L,d]
        if layer_sel == "last":
            token_reps = hs[-1]
        elif isinstance(layer_sel, int):
            # ESM layers are 1-indexed in HF hidden_states after the embedding; adjust if you stored 0/1-based
            token_reps = hs[layer_sel]  # e.g., 24th encoder block reps
        else:
            raise ValueError(f"Invalid layer_sel: {layer_sel}")
    return token_reps, attn_mask


@torch.no_grad()
def extract_esm_features_batch_layer_all(
    sequences: List[str],
    device: torch.device = DEVICE,
    dtype = torch.float16,
    model = None,
    tokenizer = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = tokenizer(sequences, return_tensors="pt", add_special_tokens=False, padding=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    attn_mask = batch["attention_mask"].to(torch.bool)

    with torch.autocast(device_type="cuda", dtype=dtype):
        out = model(**batch, output_hidden_states=True, return_dict=True)
        # hs = out.hidden_states  # tuple: [emb, layer1, ..., layerN] each [B,L,d]
        #stack hidden states -> [Lyaer, B, L, D]
        hs = torch.stack(out.hidden_states, dim=0)
    return hs, attn_mask #[Layer, B, L, D]


def _batched(iterable, n):
    """Yield Successive n-sized chunks from iterable
    """
    it = list(iterable)
    for i in range(0, len(it), n):
        yield it[i:i+n]


def _normalize_1d(
    x: np.ndarray,
    mode: Literal["seq_max","feature_global_max","zscore","none"] = "seq_max",
    global_max: Optional[float] = None,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalize a 1D activation vector x (valid positions only).
    """
    if mode == "none":
        return x.copy()

    if mode == "feature_global_max":
        if global_max is None or global_max <= eps:
            # fallback to seq_max if global not available/safe
            mode = "seq_max"
        else:
            return x / (global_max + eps)

    if mode == "seq_max":
        m = np.max(x) if x.size else 0.0
        return x / (m + eps)

    if mode == "zscore":
        mu = float(np.mean(x)) if x.size else 0.0
        sd = float(np.std(x)) if x.size else 0.0
        return (x - mu) / (sd + eps)

    # Fallback (shouldn't hit)
    return x.copy()