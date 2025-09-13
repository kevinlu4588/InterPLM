from interplm.sae.inference import load_sae_from_hf

from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Literal, Tuple
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
DEVICE="cuda"
DTYPE  = torch.float16


DATA_DIR = Path("esm_sae_results"); DATA_DIR.mkdir(exist_ok=True)
SEQUENCES_DIR = Path("/home/ec2-user/SageMaker/InterPLM/data/uniprot/subset_40k.csv")
# ANNOTATIONS_DIR = Path("uniprotkb_swissprot_annotations.tsv.gz")
ANNOTATIONS_DIR = Path("/home/ec2-user/SageMaker/InterPLM/uniprotkb_swissprot_annotations.tsv.gz")




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
