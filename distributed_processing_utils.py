import os, json, gzip, math, time, random
from pathlib import Path
from typing import List, Tuple, Iterable, Optional

import torch
import torch.distributed as dist
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


# ------------- Dist utils -------------
def get_dist_info():
    # torchrun sets these env vars
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank

def dist_init():
    rank, world_size, local_rank = get_dist_info()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=torch.timedelta(seconds=1200))
    return rank, world_size, local_rank

def dist_barrier():
    if dist.is_initialized():
        dist.barrier()

def all_reduce_sum(v: int) -> int:
    if not dist.is_initialized():
        return v
    t = torch.tensor([v], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())