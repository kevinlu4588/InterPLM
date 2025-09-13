import os
import ray
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal
from transformers import AutoTokenizer, AutoModel
from interplm.sae.inference import load_sae_from_hf
from utils import extract_sae_features, extract_esm_features_batch
from utils import _batched, _normalize_1d
import numpy as np
from tqdm import tqdm
#Make sure compute_activated_positions_for_feature uses *passed-in* model/tokenizer/sae
#Not globals, so we use this tiny wrapped to inject from the ray Actor

@torch.no_grad()
def _compute_activated_positions_for_feature_wrapped(
    fid: int,
    seqs: List[str],
    *,
    batch_size: int,
    max_per_feature: Optional = None,
    norm_mode: Literal["seq_max", "feature_global_max", "zscore", "none"],
    top_k: int,
    min_act: float,
    device: str,
    plm_layer: int,
    model,
    tokenizer,
    sae,
):
    all_indices: List[List[int]] = []
    all_aas: List[List[str]] = []
    norm_vals: List[List[float]] = []
    raw_vals: List[List[float]] = []

    global_max = None
    if norm_mode == "feature_global_max" and max_per_feature is not None:
        if 0 <= fid < len(max_per_feature):
            gm = float(max_per_feature[fid])
            global_max = gm if np.isfinite(gm) else None

    for chunk in _batched(seqs, batch_size):
        token_reps, attn_mask = extract_esm_features_batch(
            chunk,
            layer_sel=plm_layer,
            device=device,
            model=model,
            tokenizer=tokenizer,
        )

        sae_feats, _, _ = extract_sae_features(token_reps, sae)  # [B, L, F]
        feat_act = sae_feats[..., fid].float().cpu()
        mask = attn_mask.cpu()

        for seq, act_row, m in zip(chunk, feat_act, mask):
            L = int(m.sum().item())
            act_valid = act_row[:L].numpy() if L > 0 else np.array([], dtype=np.float32)

            valid_idx = np.where(act_valid > min_act)[0]
            if valid_idx.size == 0:
                all_indices.append([])
                all_aas.append([])
                norm_vals.append([])
                raw_vals.append([])
                continue

            order = np.argsort(-act_valid[valid_idx])
            chosen_local = valid_idx[order[:top_k]].tolist()

            chosen_raw = act_valid[chosen_local]
            norm_full = _normalize_1d(act_valid, mode=norm_mode, global_max=global_max)
            chosen_norm = norm_full[chosen_local]

            aas = [seq[i] if i < len(seq) else "X" for i in chosen_local]

            all_indices.append(chosen_local)
            all_aas.append(aas)
            norm_vals.append([float(v) for v in chosen_norm])
            raw_vals.append([float(v) for v in chosen_raw])

        del token_reps, sae_feats, feat_act, attn_mask
        torch.cuda.empty_cache()

    return all_indices, all_aas, norm_vals, raw_vals

    ## Ray actor


MICRO_BATCH = 64

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@ray.remote(num_gpus=1, num_cpus=0)
class SAEInferenceWorker(SAEInferenceWorker):  # reuse your class
    def process_many(self, items):  # items: List[tuple[fid, seqs]]
        out = {}
        for fid, seqs in items:
            cols = self.process_feature.remote  # wrong scope if we call .remote()
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:128"
        )

        self.device = torch.device("cuda:0")
        self.plm_layer = plm_layer

        # Optional speedups
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        self.tokenizer=AutoTokenizer.from_pretrained(plm_model_hf, do_lower_case=False)
        self.model=AutoModel.from_pretrained(plm_model_hf, output_hidden_states=True).to(self.device).eval()

        self.sae = load_sae_from_hf(plm_model=sae_plm_model_key, plm_layer=plm_layer).to(self.device).eval()

        self.autocast_dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16 if amp_dtype == "bf16" else None

    def process_feature(
        self,
        fid: int,
        seqs: List[str],
        *,
        batch_size: int = 16,
        norm_mode: Literal["seq_max","feature_global_max","zscore","none"] = "seq_max",
        top_k: int = 8,
        min_act: float = 0.0,
        max_per_feature: Optional = None,
    ) -> Dict[str, List]:
        # (Optional) autocast mainly helps inside extract_esm_features_batch, if it honors autocast.
        if self.autocast_dtype is not None:
            ctx = torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
        else:
            # no-op context
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            ctx = _NullCtx()

        with ctx:
            idx_lists, aa_lists, norm_vals, raw_vals = _compute_activated_positions_for_feature_wrapped(
                fid=fid,
                seqs=seqs,
                batch_size=batch_size,
                max_per_feature=max_per_feature,
                norm_mode=norm_mode,
                top_k=top_k,
                min_act=min_act,
                device=str(self.device),
                plm_layer=self.plm_layer,
                model=self.model,
                tokenizer=self.tokenizer,
                sae=self.sae,
            )

        # Return only small, mergeable columns to minimize traffic
        # (You’ll join these back into feature_datasets[fid] on the driver.)
        return {
            "activated_indices": idx_lists,
            "activated_aas": aa_lists,
            "seq_max_activation_norm": norm_vals,   # normalized per-pos scores
            "seq_raw_activation": raw_vals,        # raw per-pos scores (optional)
            # also provide compact strings so the driver doesn’t recompute
            "activated_indices_str": [",".join(map(str, xs)) if xs else "" for xs in idx_lists],
            "activated_aas_str":     [",".join(xs) if xs else "" for xs in aa_lists],
            "seq_max_activation_norm_str": [",".join(map(str, xs)) if xs else "" for xs in norm_vals],
        }

#Drive orchestration
def run_with_ray_over_features(
    feature_datasets: Dict[int, pd.DataFrame],
    *,
    num_workers: int,
    plm_layer: int,
    batch_size: int = 16,
    top_k: int = 8,
    min_act: float = 0.0,
    norm_mode: Literal["seq_max","feature_global_max","zscore","none"] = "seq_max",
    max_per_feature: Optional = None,
    amp_dtype: str = "fp16",   # or "bf16" or "fp32"
) -> Dict[int, pd.DataFrame]:
    ray.init(ignore_reinit_error=True)
    n_gpus = int(ray.available_resources().get("GPU", 0))
    num_workers = min(num_workers, n_gpus)

    workers = [SAEInferenceWorkerPooled.remote(
        plm_model_hf="facebook/esm2_t33_650M_UR50D",
        sae_plm_model_key="esm2-650m",
        plm_layer=plm_layer,
        amp_dtype=amp_dtype,
    ) for _ in range(num_workers)]

    fids = list(feature_datasets.keys())
    # build micro-batches as (fid, seqs)
    items = [(fid, feature_datasets[fid]["Sequence"].astype(str).fillna("").tolist())
             for fid in fids]
    batches = list(chunk(items, MICRO_BATCH))

    # submit with backpressure
    max_in_flight = max(1, 2 * num_workers)
    in_flight = {}
    submit_idx = 0
    total_fids = len(fids)
    pbar = tqdm(total=total_fids, desc="Processing features")

    def submit(batch_idx):
        w = workers[batch_idx % num_workers]
        fut = w.process_many.remote(
            batches[batch_idx],
            batch_size=batch_size,
            norm_mode=norm_mode,
            top_k=top_k,
            min_act=min_act,
            max_per_feature=max_per_feature,
        )
        in_flight[batch_idx] = fut

    while submit_idx < len(batches) and len(in_flight) < max_in_flight:
        submit(submit_idx); submit_idx += 1

    while in_flight:
        done, _ = ray.wait(list(in_flight.values()), num_returns=1)
        fut = done[0]
        # find which batch finished
        bidx = next(k for k,v in in_flight.items() if v == fut)
        del in_flight[bidx]

        results = ray.get(fut)  # Dict[fid -> cols]
        for fid, cols in results.items():
            work = feature_datasets[fid].copy()
            work["activated_indices"] = cols["activated_indices"]
            work["activated_aas"] = cols["activated_aas"]
            work["seq_max_activation_norm"] = cols["seq_max_activation_norm"]
            work["seq_raw_activation"] = cols["seq_raw_activation"]
            work["activated_indices_str"] = cols["activated_indices_str"]
            work["activated_aas_str"] = cols["activated_aas_str"]
            work["seq_max_activation_norm_str"] = cols["seq_max_activation_norm_str"]
            feature_datasets[fid] = work
        pbar.update(len(results))  # advance by number of FIDs completed

        if submit_idx < len(batches):
            submit(submit_idx); submit_idx += 1

    pbar.close()
    ray.shutdown()
    return feature_datasets

