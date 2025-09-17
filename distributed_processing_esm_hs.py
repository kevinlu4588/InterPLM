import os, json, gzip, math, time, random, gc
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from distributed_processing_utils import get_dist_info, dist_init, dist_barrier, all_reduce_sum
from utils import (
    ensure_dir, write_manifest, write_json,
    batched_sequences, count_sequences, build_model_and_tokenizer,
    extract_esm_features_batch_layer_all
)

# ---------------- Config ----------------
DATASET_NAME   = "uniprot_sprot"
ESM_NAME       = "facebook/esm2_t33_650M_UR50D"
MAX_LEN        = 1024
SEED           = 17
BATCH_SIZE_SEQ = 128
OUT_DIR        = Path(f"/home/ec2-user/SageMaker/InterPLM/data/esm2_hidden_states/{DATASET_NAME}")
# Each rank reads its own pre-split FASTA shard, created ahead of time:
SHARD_TPL      = "/home/ec2-user/SageMaker/InterPLM/data/sharded_uniprot_shard_{rank:02d}.fasta"
MAX_BATCHES    = 24

random.seed(SEED)

# -------------- Saving (shared layout, rank-tagged) --------------
def save_batch_shards_shared(hs: torch.Tensor, attn_mask: torch.Tensor, out_dir: Path, *,
                             rank: int, batch_idx: int):
    """
    Save to a single shared layout:
      masks/batch_r{rank:02d}_b{batch_idx:05d}.pt
      layer_{LL}/batch_r{rank:02d}_b{batch_idx:05d}.pt
    """
    L, B, T, D = hs.shape
    tag = f"r{rank:02d}_b{batch_idx:05d}"

    # masks/
    mask_dir = out_dir / "masks"
    ensure_dir(mask_dir)
    mask_path = mask_dir / f"batch_{tag}.pt"
    if not mask_path.exists():
        torch.save(
            {
                "attn_mask":   attn_mask.cpu(),
                "seq_lengths": attn_mask.sum(dim=1).to(torch.int32).cpu(),
                "batch_size":  int(B),
                "tokens":      int(T),
                "batch_tag":   tag,
                "rank":        int(rank),
                "local_batch": int(batch_idx),
            },
            mask_path,
        )

    # layers/
    for l in range(L):
        layer_dir = out_dir / f"layer_{l:02d}"
        ensure_dir(layer_dir)
        shard_path = layer_dir / f"batch_{tag}.pt"
        if shard_path.exists():
            continue
        torch.save(
            {
                "hidden_states": hs[l].cpu(),  # [B, T, D]
                "batch_size":    int(B),
                "tokens":        int(T),
                "width":         int(D),
                "layer":         int(l),
                "batch_tag":     tag,
                "rank":          int(rank),
                "local_batch":   int(batch_idx),
            },
            shard_path,
        )

# ---------------- Driver ----------------
def main_distributed(max_batches: Optional[int] = None):
    rank, world_size, local_rank = dist_init()

    # Device & perf
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    AMP_DTYPE = torch.float16 if use_cuda else torch.bfloat16
    torch.set_num_threads(1)
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Build model/tokenizer
    model, tokenizer = build_model_and_tokenizer(device, ESM_NAME)

    # Each rank reads its own shard
    SHARD_PATH = Path(SHARD_TPL.format(rank=rank))
    if not SHARD_PATH.exists():
        if rank == 0:
            print(f"ERROR: Shard not found: {SHARD_PATH}")
        return

    # Probe to discover L and D (use a tiny batch from THIS shard)
    probe_iter = batched_sequences(SHARD_PATH, batch_size=min(4, BATCH_SIZE_SEQ))
    try:
        probe_batch = next(probe_iter)
    except StopIteration:
        if rank == 0:
            print(f"No sequences found in shard {SHARD_PATH}. Exiting.")
        return

    with torch.inference_mode():
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                hs_probe, attn_probe = extract_esm_features_batch_layer_all(
                    probe_batch, model=model, tokenizer=tokenizer, device=device, dtype=AMP_DTYPE
                )
        else:
            hs_probe, attn_probe = extract_esm_features_batch_layer_all(
                probe_batch, model=model, tokenizer=tokenizer, device=device, dtype=AMP_DTYPE
            )
    L, _, _, D = hs_probe.shape
    del hs_probe, attn_probe
    if use_cuda:
        torch.cuda.empty_cache()

    # Pre-create shared dirs
    for l in range(L):
        ensure_dir(OUT_DIR / f"layer_{l:02d}")
    ensure_dir(OUT_DIR / "masks")

    # Per-rank totals (for progress & manifest)
    n_seqs_local = count_sequences(SHARD_PATH)
    n_batches_local = math.ceil(n_seqs_local / BATCH_SIZE_SEQ)
    if max_batches is not None:
        n_batches_local = min(n_batches_local, max_batches)

    iterator = enumerate(batched_sequences(SHARD_PATH, BATCH_SIZE_SEQ), start=1)
    if rank == 0:
        # Progress bar just for rank 0's shard (clean logs)
        iterator = tqdm(iterator, total=n_batches_local, desc=f"Rank 0 batches (world={world_size})")

    processed_sequences_local = 0
    for local_batch_idx, batch_seqs in iterator:
        if max_batches is not None and local_batch_idx > max_batches:
            break

        # (No modulo sharding — each rank owns its shard entirely)

        # Skip if this (rank, local_batch) already saved in the shared layout
        tag = f"r{rank:02d}_b{local_batch_idx:05d}"
        mask_path = OUT_DIR / "masks" / f"batch_{tag}.pt"
        already_done = mask_path.exists()
        if already_done:
            for l in range(L):
                if not (OUT_DIR / f"layer_{l:02d}" / f"batch_{tag}.pt").exists():
                    already_done = False
                    break
        if already_done:
            processed_sequences_local += len(batch_seqs)
            continue

        # Tokenize → H2D (pinned + non_blocking inside extract_... if you added it)
        with torch.inference_mode():
            if use_cuda:
                with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                    hs, attn_mask = extract_esm_features_batch_layer_all(
                        batch_seqs, model=model, tokenizer=tokenizer, device=device, dtype=AMP_DTYPE
                    )
            else:
                hs, attn_mask = extract_esm_features_batch_layer_all(
                    batch_seqs, model=model, tokenizer=tokenizer, device=device, dtype=AMP_DTYPE
                )

        # Save to shared layout with rank-tagged filenames
        save_batch_shards_shared(hs, attn_mask, OUT_DIR, rank=rank, batch_idx=local_batch_idx)
        processed_sequences_local += len(batch_seqs)

        # Free GPU tensors (don’t empty_cache() every iter)
        del hs, attn_mask

    # Aggregate totals for manifest
    total_sequences = all_reduce_sum(processed_sequences_local)
    total_batches   = all_reduce_sum(n_batches_local)

    dist_barrier()
    if rank == 0:
        write_manifest(
            OUT_DIR,
            model_name=ESM_NAME,
            dataset_name=DATASET_NAME,
            max_len=MAX_LEN,
            batch_size=BATCH_SIZE_SEQ,
            num_layers=L,
            hidden_size=D,
            total_batches=total_batches,       # sum across ranks
            total_sequences=total_sequences,   # sum across ranks
            world_size=world_size,
            layout={
                "layers_folder": "layer_{:02d}",
                "layer_shard":   "batch_r{rank:02d}_b{local:05d}.pt",
                "masks_folder":  "masks",
                "mask_shard":    "batch_r{rank:02d}_b{local:05d}.pt",
            }
        )

    dist_barrier()
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main_distributed(max_batches=MAX_BATCHES)
