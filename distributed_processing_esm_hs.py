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

# --------- Config ----------
DATASET_NAME   = "uniprot_sprot"
ESM_NAME       = "facebook/esm2_t33_650M_UR50D"
MAX_LEN        = 1024
SEED           = 17
BATCH_SIZE_SEQ = 128
OUT_DIR        = Path(f"/home/ec2-user/SageMaker/InterPLM/data/esm2_hidden_states/{DATASET_NAME}")
SHARD_TPL      = "/home/ec2-user/SageMaker/InterPLM/data/sharded_uniprot_shard_{rank:02d}.fasta"
MAX_BATCHES    = None         # e.g., 24 for smoke tests
SAVE_STYLE     = "consolidated"  # "consolidated" | "per_layer"
MAX_PENDING_SAVES = 4         # backpressure for async saves
random.seed(SEED)


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

# --------- Savers (shared layout, rank-tagged filenames) ----------
def save_consolidated_shared(hs_cpu: torch.Tensor, attn_cpu: torch.Tensor, out_dir: Path, *, rank: int, local_batch_idx: int):
    """
    One big file per batch: OUT_DIR/batches/batch_rXX_bYYYYY.pt
    """
    ensure_dir(out_dir / "batches")
    tag = f"r{rank:02d}_b{local_batch_idx:05d}"
    path = out_dir / "batches" / f"batch_{tag}.pt"
    if path.exists():
        return
    obj = {
        "hidden_states": hs_cpu,      # [L,B,T,D] (likely fp16 via AMP)
        "attn_mask": attn_cpu,        # [B,T]
        "rank": int(rank),
        "local_batch": int(local_batch_idx),
        "LBTD": tuple(int(x) for x in hs_cpu.shape),
    }
    # Faster (bigger) serialization
    torch.save(obj, path, _use_new_zipfile_serialization=False, pickle_protocol=4)

def save_perlayer_shared(hs_cpu: torch.Tensor, attn_cpu: torch.Tensor, out_dir: Path, *, rank: int, local_batch_idx: int):
    """
    Original layout: masks/ + layer_XX/ per batch (more files).
    """
    tag = f"r{rank:02d}_b{local_batch_idx:05d}"
    # mask
    mask_dir = out_dir / "masks"; ensure_dir(mask_dir)
    mpath = mask_dir / f"batch_{tag}.pt"
    if not mpath.exists():
        torch.save(
            {"attn_mask": attn_cpu,
             "seq_lengths": attn_cpu.sum(1).to(torch.int32),
             "rank": int(rank),
             "local_batch": int(local_batch_idx)},
            mpath, _use_new_zipfile_serialization=False, pickle_protocol=4
        )
    # layers
    L = hs_cpu.shape[0]
    for l in range(L):
        ldir = out_dir / f"layer_{l:02d}"; ensure_dir(ldir)
        lpath = ldir / f"batch_{tag}.pt"
        if lpath.exists(): continue
        torch.save(
            {"hidden_states": hs_cpu[l], "rank": int(rank), "local_batch": int(local_batch_idx), "layer": int(l)},
            lpath, _use_new_zipfile_serialization=False, pickle_protocol=4
        )

# --------- Distributed driver with async saving ----------
def main_distributed(max_batches: Optional[int] = MAX_BATCHES):
    rank, world_size, local_rank = dist_init()

    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    AMP_DTYPE = torch.float16 if use_cuda else torch.bfloat16
    torch.set_num_threads(1)
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Model
    model, tokenizer = build_model_and_tokenizer(device, ESM_NAME)

    # Paths
    shard_path = Path(SHARD_TPL.format(rank=rank))
    if not shard_path.exists():
        if rank == 0: print(f"[E] shard missing: {shard_path}")
        return
    ensure_dir(OUT_DIR)

    # Probe (from this shard) to get L,D
    probe_iter = batched_sequences_fasta(shard_path, batch_size=min(4, BATCH_SIZE_SEQ))
    try:
        probe_batch = next(probe_iter)
    except StopIteration:
        if rank == 0: print(f"[!] empty shard: {shard_path}")
        return
    hs_p, attn_p = extract_esm_features_batch_layer_all(probe_batch, model=model, tokenizer=tokenizer, device=device, dtype=AMP_DTYPE)
    L, _, _, D = hs_p.shape
    del hs_p, attn_p

    # If per-layer layout, pre-create layer dirs
    if SAVE_STYLE == "per_layer":
        for l in range(L):
            ensure_dir(OUT_DIR / f"layer_{l:02d}")
        ensure_dir(OUT_DIR / "masks")
    else:
        ensure_dir(OUT_DIR / "batches")

    # Per-rank totals
    n_seqs_local = count_sequences_fasta(shard_path)
    n_batches_local = math.ceil(n_seqs_local / BATCH_SIZE_SEQ)
    if max_batches is not None:
        n_batches_local = min(n_batches_local, max_batches)

    # Async saver pool
    io_pool = ThreadPoolExecutor(max_workers=2)
    pending = []

    # Iterate this shard
    iterator = enumerate(batched_sequences_fasta(shard_path, BATCH_SIZE_SEQ), start=1)
    if rank == 0:
        iterator = tqdm(iterator, total=n_batches_local, desc=f"Rank0 shard (world={world_size})")

    processed_sequences_local = 0
    for local_batch_idx, batch_seqs in iterator:
        if max_batches is not None and local_batch_idx > max_batches:
            break

        # Forward
        hs, attn = extract_esm_features_batch_layer_all(
            batch_seqs, model=model, tokenizer=tokenizer, device=device, dtype=AMP_DTYPE
        )

        # Kick off async save (copy to CPU first to free VRAM)
        hs_cpu   = hs.to("cpu", copy=True)
        attn_cpu = attn.to("cpu", copy=True)
        if SAVE_STYLE == "consolidated":
            fut = io_pool.submit(save_consolidated_shared, hs_cpu, attn_cpu, OUT_DIR, rank=rank, local_batch_idx=local_batch_idx)
        else:
            fut = io_pool.submit(save_perlayer_shared, hs_cpu, attn_cpu, OUT_DIR, rank=rank, local_batch_idx=local_batch_idx)
        pending.append(fut)
        if len(pending) >= MAX_PENDING_SAVES:
            pending[0].result(); pending.pop(0)

        processed_sequences_local += len(batch_seqs)

        # Free GPU tensors immediately; overlap save with next compute
        del hs, attn

    # Drain pending saves
    for f in pending: f.result()
    io_pool.shutdown(wait=True)

    # Manifest (global totals)
    total_sequences = all_reduce_sum(processed_sequences_local)
    total_batches   = all_reduce_sum(n_batches_local)

    dist_barrier()
    if rank == 0:
        # minimal manifest
        meta = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": ESM_NAME,
            "dataset_name": DATASET_NAME,
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE_SEQ,
            "num_layers": L,
            "hidden_size": D,
            "total_batches": total_batches,
            "total_sequences": total_sequences,
            "world_size": world_size,
            "save_style": SAVE_STYLE,
            "layout": {
                "consolidated": "batches/batch_r{rank:02d}_b{local:05d}.pt",
                "per_layer": {
                    "masks": "masks/batch_r{rank:02d}_b{local:05d}.pt",
                    "layers": "layer_{LL}/batch_r{rank:02d}_b{local:05d}.pt"
                }
            }
        }
        ensure_dir(OUT_DIR)
        with open(OUT_DIR / "manifest.json", "w") as fh:
            import json; json.dump(meta, fh, indent=2)

    dist_barrier()
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    # env knobs that help:
    # export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1
    main_distributed()