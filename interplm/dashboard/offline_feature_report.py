#!/usr/bin/env python3
"""
Offline report for 'most interesting' SAE features (no Streamlit, no Kaleido).

Outputs (saved under --outdir):
- top_features_by_f1.png .......... Top-K features ranked by Swiss-Prot F1 (barh)
- activation_consistency_scatter.png .. Consistency scatter with Top-K highlighted
- structure_vs_sequence.png ........ Optional: if Structure_feats exists
- top_features_summary.csv ......... Table with concept + metrics for Top-K
- feature_{id}_activation_hist.png . Per-feature activation distribution (if available)

Assumes InterPLM cache files exist at:
  {DASHBOARD_CACHE_DIR}/dashboard_cache.pkl
  {DASHBOARD_CACHE_DIR}/swiss-prot_metadata.tsv.gz   (only for reference columns)

You can change the default DASHBOARD_CACHE_DIR via:
  - environment variable INTERPLM_DATA (points to /.../data), or
  - --cache_dir CLI arg below.
"""

import argparse
import os
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Helpers ----------

def default_cache_dir() -> Path:
    base = os.environ.get("INTERPLM_DATA", "/home/ec2-user/SageMaker/InterPLM/data")
    return Path(base) / "dashboard_cache"

def load_cache(cache_dir: Path):
    pkl = cache_dir / "dashboard_cache.pkl"
    meta = cache_dir / "swiss-prot_metadata.tsv.gz"  # not strictly required here
    if not pkl.exists():
        raise FileNotFoundError(f"Missing cache: {pkl}")
    with open(pkl, "rb") as f:
        dash_all = pickle.load(f)
    prot_meta = None
    if meta.exists():
        try:
            prot_meta = pd.read_csv(meta, sep="\t")
        except Exception:
            prot_meta = None
    return dash_all, prot_meta

def pick_layer(dash_all: dict, prefer: int | None) -> int:
    layers = sorted(dash_all.keys())
    if not layers:
        raise RuntimeError("dashboard_cache.pkl had no layers.")
    if prefer in layers:
        return prefer
    return 3 if 3 in layers else layers[0]

def rank_by_swissprot(dash_data: dict, topk: int) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      feature, concept, f1_per_domain, precision, recall_per_domain, tp, tp_per_domain (where available)
    One row per feature (best concept per feature), sorted by F1 desc.
    """
    if "Sig_concepts_per_feature" not in dash_data:
        return pd.DataFrame(columns=["feature","concept","f1_per_domain","precision","recall_per_domain","tp","tp_per_domain"])
    df = dash_data["Sig_concepts_per_feature"].copy()
    # Keep the best concept per feature (highest F1, then recall/tp as tie-breakers)
    df = df.sort_values(["f1_per_domain","recall_per_domain","tp"], ascending=False)
    df = df.drop_duplicates(["feature"], keep="first")
    cols = [c for c in ["feature","concept","f1_per_domain","precision","recall_per_domain","tp","tp_per_domain"] if c in df.columns]
    out = df[cols].reset_index(drop=True)
    return out.head(topk)

def activation_consistency_table(dash_data: dict) -> pd.DataFrame:
    """
    Returns a DataFrame with an index = feature and two columns:
      freq_any (Per_prot_frequency_of_any_activation),
      pct_when_present (Per_prot_pct_activated_when_present),
      product (their product, handy for scoring).
    """
    if "Per_feature_statistics" not in dash_data:
        return pd.DataFrame()
    S = dash_data["Per_feature_statistics"]
    out = pd.DataFrame({
        "freq_any": S["Per_prot_frequency_of_any_activation"],
        "pct_when_present": S["Per_prot_pct_activated_when_present"]
    })
    out["product"] = out["freq_any"] * out["pct_when_present"]
    return out

def safe_figsize(w=7, h=4.5):
    return (w, h)

# ---------- Plotters ----------

def plot_top_f1_barh(top_df: pd.DataFrame, out: Path):
    if top_df.empty:
        return
    fig, ax = plt.subplots(figsize=safe_figsize(8, max(3, 0.4*len(top_df)+1)))
    ylabels = [f"f/{int(r.feature)} — {str(r.concept)[:40]}" for _, r in top_df.iterrows()]
    ax.barh(ylabels[::-1], top_df["f1_per_domain"].values[::-1])
    ax.set_xlabel("Swiss-Prot F1 (per domain)")
    ax.set_title("Top features by Swiss-Prot F1")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

def plot_activation_consistency_scatter(stats: pd.DataFrame, top_feats: list[int], out: Path):
    if stats.empty:
        return
    fig, ax = plt.subplots(figsize=safe_figsize(7.5, 5.5))
    ax.scatter(stats["freq_any"], stats["pct_when_present"], s=8, alpha=0.5, label="All features")
    # Highlight top feats if they exist in the stats index
    highlights = [f for f in top_feats if f in stats.index]
    if highlights:
        ax.scatter(stats.loc[highlights, "freq_any"], stats.loc[highlights, "pct_when_present"],
                   s=40, edgecolor="k", linewidth=0.5, label="Top-F1 features")
        # Annotate lightly (at most 12 to avoid clutter)
        for f in highlights[:12]:
            ax.annotate(f"f/{f}", (stats.loc[f, "freq_any"], stats.loc[f, "pct_when_present"]),
                        xytext=(3,3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("% of proteins with activation (frequency)")
    ax.set_ylabel("Avg % activated when present")
    ax.set_title("Activation consistency across features")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

def plot_structure_vs_sequence(dash_data: dict, top_feats: list[int], out: Path):
    if "Structure_feats" not in dash_data:
        return
    df = dash_data["Structure_feats"].copy()
    if "feat" in df.columns:
        df = df.set_index("feat")
    # Heuristic: try a couple likely columns; fall back to first two numeric cols
    cols = df.select_dtypes(include="number").columns.tolist()
    if not cols:
        return
    # Try to pick interpretable axes if present
    xcol = next((c for c in cols if "seq" in c.lower()), cols[0])
    ycol = next((c for c in cols if "3d" in c.lower() or "struct" in c.lower()), (cols[1] if len(cols)>1 else cols[0]))
    fig, ax = plt.subplots(figsize=safe_figsize(7.2, 5.2))
    ax.scatter(df[xcol], df[ycol], s=8, alpha=0.5, label="All features")
    hi = [f for f in top_feats if f in df.index]
    if hi:
        ax.scatter(df.loc[hi, xcol], df.loc[hi, ycol], s=40, edgecolor="k", linewidth=0.5, label="Top-F1 features")
        for f in hi[:12]:
            ax.annotate(f"f/{f}", (df.loc[f, xcol], df.loc[f, ycol]),
                        xytext=(3,3), textcoords="offset points", fontsize=8)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title("Structure vs Sequence localization")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

def plot_feature_activation_hist(dash_data: dict, feature_id: int, out: Path):
    """
    Uses the cached sample of SAE activations if present.
    The dashboard used plot_activations_for_single_feat(SAE_features, feature_id).
    We'll try to find a 2D array or per-feature vector under "SAE_features".
    """
    if "SAE_features" not in dash_data:
        return
    F = dash_data["SAE_features"]
    # Try common shapes:
    # - If F is (n_samples, dict_size) -> take column
    # - If F is a dict mapping feature -> vector
    vec = None
    try:
        # ndarray case
        if hasattr(F, "shape") and len(F.shape) == 2:
            if feature_id < F.shape[1]:
                vec = np.asarray(F[:, feature_id]).ravel()
        # DataFrame case
        if vec is None and hasattr(F, "columns"):
            col = feature_id if feature_id in getattr(F, "columns", []) else None
            if col is None and hasattr(F, "columns"):
                # columns might be named like ints as strings
                if str(feature_id) in map(str, F.columns):
                    col = [c for c in F.columns if str(c) == str(feature_id)][0]
            if col is not None:
                vec = np.asarray(F[col]).ravel()
        # Dict case
        if vec is None and isinstance(F, dict) and feature_id in F:
            vec = np.asarray(F[feature_id]).ravel()
    except Exception:
        vec = None

    if vec is None or len(vec) == 0:
        return

    fig, ax = plt.subplots(figsize=safe_figsize(6.8, 3.6))
    ax.hist(vec, bins=50, alpha=0.85)
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Count")
    ax.set_title(f"Activation distribution for f/{feature_id}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default=None, help="Path to dashboard_cache folder")
    ap.add_argument("--layer", type=int, default=None, help="Layer to use (default=prefer 3 else first)")
    ap.add_argument("--topk", type=int, default=12, help="How many features to analyze")
    ap.add_argument("--outdir", type=str, required=True, help="Where to save plots/CSV")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else default_cache_dir()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dash_all, _ = load_cache(cache_dir)
    L = pick_layer(dash_all, args.layer)
    dd = dash_all[L]

    # 1) Rank by Swiss-Prot F1 (best concept per feature)
    top_df = rank_by_swissprot(dd, args.topk)
    if top_df.empty:
        print("Warning: No Swiss-Prot concept table found. Falling back to activation consistency.")
        stats = activation_consistency_table(dd)
        if stats.empty:
            raise SystemError("Neither Swiss-Prot nor activation statistics are available in cache.")
        top_feats = stats.sort_values("product", ascending=False).head(args.topk).index.tolist()
        top_df = pd.DataFrame({"feature": top_feats, "concept": ["(n/a)"]*len(top_feats), "f1_per_domain":[np.nan]*len(top_feats)})
    else:
        top_feats = top_df["feature"].astype(int).tolist()

    # 2) Save summary CSV
    top_df.to_csv(outdir / f"layer_{L}_top_features_summary.csv", index=False)

    # 3) Plots
    plot_top_f1_barh(top_df, outdir / f"layer_{L}_top_features_by_f1.png")

    stats = activation_consistency_table(dd)
    if not stats.empty:
        plot_activation_consistency_scatter(stats, top_feats, outdir / f"layer_{L}_activation_consistency_scatter.png")

    plot_structure_vs_sequence(dd, top_feats, outdir / f"layer_{L}_structure_vs_sequence.png")

    # 4) Per-feature activation distributions
    for f in top_feats:
        plot_feature_activation_hist(dd, int(f), outdir / f"layer_{L}_feature_{int(f)}_activation_hist.png")

    print(f"✅ Done. Saved report to: {outdir}")

if __name__ == "__main__":
    main()
