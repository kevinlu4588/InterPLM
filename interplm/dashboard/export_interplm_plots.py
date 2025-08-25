#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.io as pio

from interplm.constants import DASHBOARD_CACHE_DIR
from interplm.dashboard.colors import get_structure_palette_and_colormap
from interplm.dashboard.feature_activation_vis import (
    plot_activation_scatter,
    plot_activations_for_single_feat,
    plot_structure_scatter,
    plot_umap_scatter,
    visualize_protein_feature,
)
from interplm.data_processing.utils import fetch_uniprot_sequence
from interplm.esm.embed import embed_single_sequence
from interplm.sae.inference import encode_subset_of_feats
from interplm.utils import get_device


def load_cache():
    cache_pkl = DASHBOARD_CACHE_DIR / "dashboard_cache.pkl"
    swissprot_meta = DASHBOARD_CACHE_DIR / "swiss-prot_metadata.tsv.gz"
    if not cache_pkl.exists():
        raise FileNotFoundError(f"Missing {cache_pkl}")
    if not swissprot_meta.exists():
        raise FileNotFoundError(f"Missing {swissprot_meta}")
    with open(cache_pkl, "rb") as f:
        dash_data_all_layer = pickle.load(f)
    protein_metadata = pd.read_csv(swissprot_meta, sep="\t", index_col=0)
    device = get_device()
    return dash_data_all_layer, protein_metadata, device


def get_protein_ids(dash_data: dict, feature_id: int, activation_range: str) -> List[str]:
    if activation_range == "Max":
        return list(dash_data["Per_feature_max_examples"][feature_id])
    return list(dash_data["Per_feature_quantile_examples"][feature_id][activation_range])


def get_color_range(activation_range: str) -> Tuple[float, float, float]:
    if activation_range == "Max":
        return (0, 0.4, 0.85)
    try:
        rv = float(activation_range[0])
        return (0, rv / 2, rv)
    except Exception:
        return (0, 0.2, 0.4)


def export_layer_wide(dash_data: dict, layer: int, feature_id: int, outdir: Path, fmt: str):
    outdir.mkdir(parents=True, exist_ok=True)

    # Activation consistency
    if "Per_feature_statistics" in dash_data:
        stats = dash_data["Per_feature_statistics"]
        fig = plot_activation_scatter(
            x_value=stats["Per_prot_frequency_of_any_activation"],
            y_value=stats["Per_prot_pct_activated_when_present"],
            title=f"Activation consistency (layer {layer})",
            xaxis_title="% proteins with activation",
            yaxis_title="Avg % activated when present",
            feature_to_highlight=feature_id,
        )
        pio.write_image(fig, outdir / f"activation_consistency.{fmt}") if fmt != "html" \
            else pio.write_html(fig, outdir / "activation_consistency.html", include_plotlyjs="cdn")

    # Structural vs sequential
    if "Structure_feats" in dash_data:
        struct_df = dash_data["Structure_feats"].set_index("feat")
        fig = plot_structure_scatter(
            df=struct_df,
            title=f"Structure vs Sequence (layer {layer})",
            feature_to_highlight=(feature_id if feature_id in struct_df.index else None),
        )
        pio.write_image(fig, outdir / f"structure_vs_sequence.{fmt}") if fmt != "html" \
            else pio.write_html(fig, outdir / "structure_vs_sequence.html", include_plotlyjs="cdn")

    # UMAP
    if "UMAP" in dash_data:
        umap_df = dash_data["UMAP"].reset_index().rename(columns={"index": "Feature"})
        fig = plot_umap_scatter(
            umap_df,
            feature_to_highlight=(feature_id if feature_id in umap_df["Feature"] else None),
            title=f"UMAP of feature values (layer {layer})",
        )
        pio.write_image(fig, outdir / f"umap_features.{fmt}") if fmt != "html" \
            else pio.write_html(fig, outdir / "umap_features.html", include_plotlyjs="cdn")

    # Swiss-Prot concepts table
    if "Sig_concepts_per_feature" in dash_data:
        concepts = (
            dash_data["Sig_concepts_per_feature"]
            .query("tp_per_domain >= 2 or tp >= 2")
            .sort_values(["f1_per_domain", "recall_per_domain", "tp"], ascending=False)
            .drop_duplicates(["concept"], keep="first")
        )
        concepts.to_csv(outdir / "swissprot_concepts.csv", index=False)


def export_feature_details(dash_data: dict, feature_id: int, outdir: Path, fmt: str):
    outdir.mkdir(parents=True, exist_ok=True)
    # Activation histogram/distribution for this feature
    fig = plot_activations_for_single_feat(dash_data["SAE_features"], feature_id)
    if fig is not None:
        if fmt != "html":
            pio.write_image(fig, outdir / f"feature_activation_hist.{fmt}")
        else:
            pio.write_html(fig, outdir / "feature_activation_hist.html", include_plotlyjs="cdn")


def get_proteins_for_export(
    dash_data: dict,
    protein_metadata: pd.DataFrame,
    feature_id: int,
    activation_range: str,
    n_proteins: int,
    custom_ids: Optional[List[str]],
    custom_sequences: Optional[List[str]],
) -> pd.DataFrame:
    if custom_ids:
        rows = []
        for uid in custom_ids:
            md = fetch_uniprot_sequence(uid)
            if md:
                rows.append(md)
        return pd.DataFrame(rows)

    if custom_sequences:
        return pd.DataFrame({
            "Entry": [f"Custom_{i+1}" for i in range(len(custom_sequences))],
            "Sequence": custom_sequences,
            "Protein names": ["" for _ in custom_sequences],
        })

    ids = get_protein_ids(dash_data, feature_id, activation_range)
    if not ids:
        return pd.DataFrame()
    return protein_metadata.loc[ids[:n_proteins]].reset_index()


def export_protein_visuals(
    proteins: pd.DataFrame,
    dash_data: dict,
    device: str,
    feature_id: int,
    activation_range: str,
    outdir: Path,
    fmt: str,
    add_highlight: bool,
    save_structure: bool,
):
    outdir.mkdir(parents=True, exist_ok=True)
    color_range = get_color_range(activation_range)
    colormap_fn, palette_to_viz = get_structure_palette_and_colormap(color_range)

    # Save palette preview (HTML)
    pio.write_html(palette_to_viz, outdir / "palette.html", include_plotlyjs="cdn")

    esm_name = dash_data["ESM_metadata"]["esm_model_name"]
    esm_layer = dash_data["ESM_metadata"]["layer"]
    sae = dash_data["SAE"]
    try:
        # If it's an nn.Module-like object
        sae.to(device)
    except AttributeError:
        # If it's a simple container with tensors
        if hasattr(sae, "bias"):
            sae.bias = sae.bias.to(device)
        if hasattr(sae, "encoder") and hasattr(sae.encoder, "weight"):
            sae.encoder.weight = sae.encoder.weight.to(device)
        if hasattr(sae, "decoder") and hasattr(sae.decoder, "weight"):
            sae.decoder.weight = sae.decoder.weight.to(device)

    esm_name = dash_data["ESM_metadata"]["esm_model_name"]
    esm_layer = dash_data["ESM_metadata"]["layer"]

    for _, protein in proteins.iterrows():
        entry = str(protein["Entry"])
        seq = str(protein["Sequence"])

        # Embed on device
        emb = embed_single_sequence(sequence=seq, model_name=esm_name, layer=esm_layer, device=device)

        # Extra safety: if embed_single_sequence picks a different device, co-locate SAE again
        if hasattr(sae, "to"):
            sae.to(emb.device)
        else:
            if hasattr(sae, "bias"):
                sae.bias = sae.bias.to(emb.device)
            if hasattr(sae, "encoder") and hasattr(sae.encoder, "weight"):
                sae.encoder.weight = sae.encoder.weight.to(emb.device)
            if hasattr(sae, "decoder") and hasattr(sae.decoder, "weight"):
                sae.decoder.weight = sae.decoder.weight.to(emb.device)

        feats = encode_subset_of_feats(sae, emb, [feature_id]).cpu().numpy().flatten()
        # 1) sequence visualization
        fig = visualize_protein_feature(feats, seq, protein, "Amino Acids")
        if fmt != "html":
            pio.write_image(fig, outdir / f"protein_{entry}_sequence_feature.{fmt}")
        else:
            pio.write_html(fig, outdir / f"protein_{entry}_sequence_feature.html", include_plotlyjs="cdn")

        # 2) optional 3D structure (HTML)
        if save_structure and entry and not entry.startswith("Custom_"):
            from interplm.dashboard.view_structures import view_single_protein
            html = view_single_protein(
                uniprot_id=entry,
                values_to_color=feats,
                colormap_fn=colormap_fn,
                residues_to_highlight=([i for i, v in enumerate(feats) if v > 0.4] if add_highlight else None),
                pymol_params={"width": 900, "height": 540},
            )
            (outdir / f"protein_{entry}_structure.html").write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Export InterPLM plots without Streamlit.")
    ap.add_argument("--layer", type=int, required=False, help="ESM layer to load (defaults to the 4th available if present)")
    ap.add_argument("--feature", type=int, default=0, help="SAE feature id")
    ap.add_argument("--activation_range", default="Max", help='Activation bin (e.g., "Max", "0-10%", "10-20%", etc.)')
    ap.add_argument("--n_proteins", type=int, default=5, help="Number of proteins to export (max 10 recommended)")
    ap.add_argument("--export_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--format", choices=["png", "pdf", "svg", "html"], default="png", help="Figure format")
    ap.add_argument("--custom_uniprot_ids", type=str, nargs="*", default=None, help="Optional UniProt IDs")
    ap.add_argument("--custom_sequences", type=str, nargs="*", default=None, help="Optional custom sequences")
    ap.add_argument("--highlight", action="store_true", help="Highlight high-activation residues in structures")
    ap.add_argument("--skip_proteins", action="store_true", help="Skip per-protein exports")
    ap.add_argument("--save_structure", action="store_true", help="Also write per-protein 3D structure HTML")
    args = ap.parse_args()

    outdir = Path(args.export_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    dash_all, prot_meta, device = load_cache()
    available_layers = sorted(list(dash_all.keys()))
    if not available_layers:
        raise RuntimeError("No layers in dashboard cache.")
    layer = args.layer if args.layer in available_layers else (3 if 3 in available_layers else available_layers[0])
    dash_data = dash_all[layer]

    # Layer-wide
    lw_dir = outdir / f"layer_{layer}" / "layer_wide"
    export_layer_wide(dash_data, layer, args.feature, lw_dir, args.format)

    # Feature details
    fd_dir = outdir / f"layer_{layer}" / f"feature_{args.feature}"
    export_feature_details(dash_data, args.feature, fd_dir, args.format)

    # Proteins
    if not args.skip_proteins:
        proteins = get_proteins_for_export(
            dash_data,
            prot_meta,
            args.feature,
            args.activation_range,
            args.n_proteins,
            args.custom_uniprot_ids,
            args.custom_sequences,
        )
        if not proteins.empty:
            export_protein_visuals(
                proteins,
                dash_data,
                device,
                args.feature,
                args.activation_range,
                fd_dir / "proteins",
                args.format,
                add_highlight=args.highlight,
                save_structure=args.save_structure,
            )
        else:
            print("No proteins found for the requested range; skipping protein exports.")

    print(f"✅ Export complete at: {outdir}")


if __name__ == "__main__":
    main()
