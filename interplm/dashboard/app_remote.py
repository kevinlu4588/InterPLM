"""
InterPLM Dashboard Application with support for remote data loading.
This version can load data from Hugging Face Hub or local filesystem.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from tap import Tap

from interplm.dashboard.colors import get_structure_palette_and_colormap
from interplm.dashboard.data_loader import DataLoader, get_data_loader
from interplm.dashboard.feature_activation_vis import (
    plot_activation_scatter,
    plot_activations_for_single_feat,
    plot_structure_scatter,
    plot_umap_scatter,
    visualize_protein_feature,
)
from interplm.dashboard.help_notes import help_notes
from interplm.dashboard.view_structures import view_single_protein
from interplm.data_processing.utils import fetch_uniprot_sequence
from interplm.esm.embed import embed_single_sequence
from interplm.sae.inference import encode_subset_of_feats
from interplm.utils import get_device


class DashboardConfig(Tap):
    """Configuration for the dashboard."""
    source: str = "local"  # "local" or "remote"
    repo_id: Optional[str] = None  # Hugging Face repository ID
    cache_dir: Optional[str] = None  # Cache directory for downloads
    token: Optional[str] = None  # HF token for private repos
    download_all: bool = False  # Download all data at startup


@dataclass
class DashboardState:
    """Holds the state and configuration for the dashboard"""
    layer: int
    feature_id: int
    feature_activation_range: str
    n_proteins_to_show: int
    add_highlight: bool
    custom_uniprot_ids: List[str] | None
    custom_sequences: List[str] | None
    show_proteins: bool


@st.cache_resource
def load_dashboard_data(source: str, repo_id: Optional[str], cache_dir: Optional[str], token: Optional[str]):
    """Load and cache the dashboard data based on configuration"""
    try:
        # Create data loader
        data_loader = DataLoader(
            source=source,
            repo_id=repo_id,
            cache_dir=cache_dir,
            token=token
        )
        
        # Load dashboard cache
        dash_data = data_loader.load_dashboard_cache()
        
        # Load protein metadata
        protein_metadata = data_loader.load_protein_metadata()
        
        # Get device
        device = get_device()
        
        return dash_data, protein_metadata, device, data_loader
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()


class ProteinFeatureVisualizer:
    def __init__(self, data_loader: DataLoader, dash_data, protein_metadata, device):
        self.data_loader = data_loader
        self.dash_data_all_layer = dash_data
        self.protein_metadata = protein_metadata
        self.device = device

    def select_feature(self, layer):
        dash_data = self.dash_data_all_layer[layer]
        n_features = dash_data["SAE"].dict_size

        # Initialize session state for feature_id if not exists
        if f"feature_id_{layer}" not in st.session_state:
            st.session_state[f"feature_id_{layer}"] = dash_data.get(
                "Default feature", 0
            )

        # Get available features to sample from
        feats_to_sample = []
        if "Sig_concepts_per_feature" in dash_data.keys():
            sig_concepts = (
                dash_data["Sig_concepts_per_feature"]
                .query("f1_per_domain > 0.5")["feature"]
                .unique()
            )
            if len(sig_concepts) > 0:
                feats_to_sample.extend(sig_concepts)
        if "LLM Autointerp" in dash_data.keys():
            feats_to_sample.extend(
                dash_data["LLM Autointerp"].query("Correlation > 0.5").index
            )
        if len(feats_to_sample) == 0:
            feats_to_sample = list(range(n_features))

        # Create random feature button below the number input
        if st.sidebar.button("Select random feature"):
            # Update session state with random feature
            st.session_state[f"feature_id_{layer}"] = int(
                np.random.choice(feats_to_sample)
            )

        # Number input that uses and updates session state
        feature_id = st.sidebar.number_input(
            f"Or specify SAE feature number",
            min_value=0,
            max_value=n_features - 1,
            step=1,
            value=st.session_state[f"feature_id_{layer}"],
            key=f"feature_input_{layer}",
            help=f"Enter a specific feature ID to explore (0 - {n_features-1:,})",
        )

        # Update session state if number input changes
        st.session_state[f"feature_id_{layer}"] = feature_id

        return feature_id

    def setup_sidebar(self) -> DashboardState:
        """Configure sidebar controls and return dashboard state"""
        st.sidebar.markdown(help_notes["overall"], unsafe_allow_html=True)
        
        # Add data source indicator
        if self.data_loader.source == "remote":
            st.sidebar.info(f"📡 Loading from: {self.data_loader.repo_id}")
        else:
            st.sidebar.info("💾 Using local data")
        
        st.sidebar.markdown(
            "## Select ESM Layer and Feature", help=help_notes["select_esm_layer"]
        )
        available_layers = sorted(list(self.dash_data_all_layer.keys()))
        if len(available_layers) == 0:
            st.error("No data found. Please check the cache and try again.")
        elif len(available_layers) == 1:
            layer = available_layers[0]
        else:
            layer = st.sidebar.selectbox(
                "Select ESM embedding layer",
                available_layers,
                index=3 if 3 in available_layers else 0,
            )

        dash_data = self.dash_data_all_layer[layer]
        feature_id = self.select_feature(layer)
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            "## Visualize Feature Activation on Proteins",
            help=help_notes["vis_sidebar"],
        )

        show_proteins = st.sidebar.button("View Protein Activations")
        st.sidebar.markdown(
            "Customize the protein selection and visualization below. Defaults to max activated proteins (a):"
        )

        # Specify activation range
        st.sidebar.markdown(
            "### a) Select proteins by activation range",
            help="Select proteins based on how strongly they activate the feature.",
        )
        quantiles = [
            i
            for i, sublist in dash_data["Per_feature_quantile_examples"][
                feature_id
            ].items()
            if len(sublist) > 0
        ]
        available_ranges = ["Max"] + quantiles
        feature_activation_range = st.sidebar.selectbox(
            "Select activation group", available_ranges, index=0
        )

        # Number of proteins to visualize
        max_prot_ids = self._get_protein_ids(
            dash_data, feature_id, feature_activation_range
        )
        n_possible = min(10, len(max_prot_ids))
        default_n = min(5, n_possible)
        if n_possible == 0:
            st.sidebar.warning("No proteins found in this activation range.")
            n_proteins = 0
        else:
            n_proteins = st.sidebar.slider(
                "Number of proteins to visualize",
                0,
                n_possible,
                default_n,
                help="Maximum of 10 proteins can be shown",
            )

        # Custom protein selection
        st.sidebar.markdown(
            "### b) Enter custom UniProt IDs", help=help_notes["uniprot"]
        )
        custom_ids = st.sidebar.text_area(
            "Enter Uniprot IDs",
            "",
            placeholder="Add each Uniprot ID to a new line. These are slower as they fetch data from UniProt.",
        )

        # Custom sequence selection
        st.sidebar.markdown(
            "### c) Enter custom protein sequences",
            help="Note: Custom sequences will not show the protein structure.",
        )
        custom_sequences = st.sidebar.text_area(
            "Enter protein sequences",
            "",
            placeholder="Add each protein sequence to a new line",
        )

        add_highlight = st.sidebar.checkbox(
            "Highlight high activation residues",
            value=False,
            help="Highlight residues with activations in the top 5% of the range",
        )

        if custom_ids and custom_sequences:
            st.warning(
                "Please select either Uniprot IDs or custom sequences, not both."
            )
            st.stop()

        elif custom_ids:
            custom_ids = [id.strip() for id in custom_ids.split("\n") if id.strip()]
        elif custom_sequences:
            custom_sequences = [
                seq.strip() for seq in custom_sequences.split("\n") if seq.strip()
            ]

        return DashboardState(
            layer=layer,
            feature_id=feature_id,
            feature_activation_range=feature_activation_range,
            n_proteins_to_show=n_proteins,
            add_highlight=add_highlight,
            custom_uniprot_ids=custom_ids,
            custom_sequences=custom_sequences,
            show_proteins=show_proteins,
        )

    def visualize_proteins(self, state: DashboardState):
        """Visualize selected proteins with their feature activations"""
        dash_data = self.dash_data_all_layer[state.layer]

        try:
            if state.custom_uniprot_ids:
                proteins_to_viz = self._get_custom_proteins(state.custom_uniprot_ids)

                if proteins_to_viz.empty:
                    st.warning(
                        "No valid UniProt IDs found. Please check the IDs and try again."
                    )
                    return
            elif state.custom_sequences:
                proteins_to_viz = pd.DataFrame(
                    {
                        "Entry": [
                            f"Custom Sequence {i+1}"
                            for i in range(len(state.custom_sequences))
                        ],
                        "Sequence": state.custom_sequences,
                    }
                )
            else:
                proteins_to_viz = self._get_proteins_by_activation(
                    dash_data,
                    state.feature_id,
                    state.feature_activation_range,
                    state.n_proteins_to_show,
                )
                if proteins_to_viz.empty:
                    st.warning("No proteins found that activate this feature.")
                    return

            color_range = self._get_color_range(state.feature_activation_range)
            structure_colormap_fn, palette_to_viz = get_structure_palette_and_colormap(
                color_range
            )

            self._render_protein_visualizations(
                proteins_to_viz,
                dash_data,
                state.feature_id,
                state.add_highlight,
                structure_colormap_fn,
                palette_to_viz,
                is_custom_seq=bool(state.custom_sequences),
            )
        except Exception as e:
            st.error(f"Error visualizing proteins: {str(e)}")

    def display_feature_statistics(self, layer: int, feature_id: int):
        """Display feature-wide statistics section"""
        dash_data = self.dash_data_all_layer[layer]

        st.header(
            f"Metrics on all SAE features from ESM layer {layer}",
            help=help_notes["metrics"],
        )
        st.markdown(
            f"**(Highlighting selected feature <span style='color: #00DDFF'>f/{feature_id}**</span>)",
            unsafe_allow_html=True,
            help="Feature and layer selection can be changed in the sidebar.",
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            self._plot_activation_consistency(dash_data, feature_id)
        with col2:
            self._plot_structure_features(dash_data, feature_id)
        with col3:
            self._plot_umap(dash_data, feature_id)
        with col4:
            self._display_swissprot_concepts(dash_data, feature_id, layer)

        st.markdown("---")
        st.header(f"Details on f/{feature_id}", help=help_notes["feature_details"])

        self._display_feature_act_dist_and_concepts(dash_data, feature_id)
        st.markdown("---")

    # ... (include all the other methods from the original app.py)
    # I'll skip them here for brevity, but they should all be copied over
    # unchanged from the original app.py file
    
    def _display_feature_act_dist_and_concepts(self, dash_data: Dict, feature_id: int):
        # Check whether "LLM Autointerp" in dash_data.keys()
        if (
            "LLM Autointerp" in dash_data.keys()
            and feature_id in dash_data["LLM Autointerp"].index
        ):
            col3, col1, col2 = st.columns(3)
            with col3:
                description_score = (
                    f"{dash_data['LLM Autointerp'].loc[feature_id]['Correlation']:.2f}"
                )
                st.subheader(
                    f"**Language Model Summary for f/{feature_id} (score={description_score})**",
                    help=help_notes["autointerp"],
                )
                st.write(f"{dash_data['LLM Autointerp'].loc[feature_id]['Summary']}")
        else:
            col1, col2 = st.columns(2)
        with col1:
            st.subheader(
                f"**Feature Activation Distribution for f/{feature_id}**",
                help=help_notes["act_distribution"],
            )
            plot_of_feat_acts = plot_activations_for_single_feat(
                dash_data["SAE_features"], feature_id
            )
            if plot_of_feat_acts is not None:
                st.plotly_chart(
                    plot_of_feat_acts,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.write("No activations found for this feature in random sample.")

            with col2:
                st.subheader(
                    f"**Concepts Identified in f/{feature_id}**",
                    help=help_notes["swissprot_per_feat"],
                )
                concepts_for_feat = (
                    dash_data["Sig_concepts_per_feature"]
                    .query(f"f1_per_domain > 0.2 & feature == {feature_id}")
                    .sort_values("f1_per_domain", ascending=False)
                )

                if len(concepts_for_feat) > 0:
                    concepts_for_feat = concepts_for_feat.drop_duplicates(
                        "concept", keep="first"
                    )
                    concepts_for_feat = concepts_for_feat[
                        [
                            "concept",
                            "threshold_pct",
                            "f1_per_domain",
                            "precision",
                            "recall_per_domain",
                            "tp",
                            "tp_per_domain",
                            "fp",
                        ]
                    ]
                    concepts_for_feat.rename(
                        columns={
                            "concept": "Concept",
                            "f1_per_domain": "F1",
                            "precision": "Precision",
                            "recall_per_domain": "Recall",
                            "tp": "True Positives (per AA)",
                            "tp_per_domain": "True Positives (per Domain)",
                            "threshold_pct": "Threshold",
                        },
                        inplace=True,
                    )
                    concepts_for_feat.set_index("Concept", inplace=True)
                    st.write(concepts_for_feat)
                else:
                    st.write("No Swiss-Prot concepts found for this feature.")

    def _get_protein_ids(
        self, dash_data: Dict, feature_id: int, activation_range: str
    ) -> List[str]:
        """Get protein IDs based on activation range"""
        if activation_range == "Max":
            return list(dash_data["Per_feature_max_examples"][feature_id])
        return list(
            dash_data["Per_feature_quantile_examples"][feature_id][activation_range]
        )

    def _get_color_range(self, activation_range: str) -> Tuple[float, float, float]:
        """Get color range based on activation range"""
        if activation_range == "Max":
            return (0, 0.4, 0.85)
        try:
            range_value = float(activation_range[0])
            return (0, range_value / 2, range_value)
        except (IndexError, ValueError):
            return (0, 0.2, 0.4)  # Default fallback

    def _get_custom_proteins(self, uniprot_ids: List[str]) -> pd.DataFrame:
        """Get protein data for custom Uniprot IDs"""
        metadata = []
        for uniprot_id in uniprot_ids:
            protein_metadata = fetch_uniprot_sequence(uniprot_id)
            if protein_metadata:
                metadata.append(protein_metadata)
        return pd.DataFrame(metadata)

    def _get_proteins_by_activation(
        self, dash_data: Dict, feature_id: int, activation_range: str, n_proteins: int
    ) -> pd.DataFrame:
        """Get protein data based on activation range"""
        protein_ids = self._get_protein_ids(dash_data, feature_id, activation_range)
        if not protein_ids:
            return pd.DataFrame()
        return self.protein_metadata.loc[protein_ids[:n_proteins]].reset_index()

    def _plot_structure_features(self, dash_data: Dict, feature_id: int):
        """Plot structure features scatter plot"""
        if "Structure_feats" not in dash_data:
            return
        struct_data = dash_data["Structure_feats"].set_index("feat")
        plot = plot_structure_scatter(
            df=struct_data,
            title="",
            feature_to_highlight=(
                feature_id if feature_id in struct_data.index else None
            ),
        )
        struct_v_seq_help = (
            "When a feature activates on multiple amino acids in a protein,"
            "are the activated positions close nearby eachother in 3D space? Are they nearby in "
            "the sequence? This compares these two ways of looking at features activation. If you "
            "look features with structures:seq ratios, they are often interesting"
        )
        st.subheader("**Structural vs Sequential**", help=struct_v_seq_help)
        st.plotly_chart(
            plot, use_container_width=True, config={"displayModeBar": False}
        )

    def _plot_activation_consistency(self, dash_data: Dict, feature_id: int):
        """Plot activation consistency scatter plot"""
        if "Per_feature_statistics" not in dash_data:
            return
        stats = dash_data["Per_feature_statistics"]
        plot = plot_activation_scatter(
            x_value=stats["Per_prot_frequency_of_any_activation"],
            y_value=stats["Per_prot_pct_activated_when_present"],
            title="",
            xaxis_title="% of proteins with activation",
            yaxis_title="Avg % activated when present",
            feature_to_highlight=feature_id,
        )
        st.subheader(
            "**Feature Activation Frequencies**",
            help="Shows the consistency of feature activation across and within proteins.",
        )
        st.plotly_chart(
            plot, use_container_width=True, config={"displayModeBar": False}
        )

    def _plot_umap(self, dash_data: Dict, feature_id: int):
        """Plot UMAP visualization"""
        if "UMAP" not in dash_data:
            return
        umap_data = dash_data["UMAP"].reset_index().rename(columns={"index": "Feature"})
        plot = plot_umap_scatter(
            umap_data,
            feature_to_highlight=(
                feature_id if feature_id in umap_data["Feature"] else None
            ),
            title="",
        )
        st.subheader(
            "**UMAP of Feature Values**",
            help="All features in layer visualized in 2D based on a UMAP of their dictionary values. Coloring based on cluster assignment.",
        )
        st.plotly_chart(
            plot,
            use_container_width=True,
            help="UMAP visualization of feature values",
            config={"displayModeBar": False},
        )

    def _display_swissprot_concepts(self, dash_data: Dict, feature_id: int, layer: int):
        """Display SwissProt concepts table"""
        if "Sig_concepts_per_feature" not in dash_data:
            return
        concepts = (
            dash_data["Sig_concepts_per_feature"]
            .query("tp_per_domain >= 2 or tp >= 2")
            .sort_values(["f1_per_domain", "recall_per_domain", "tp"], ascending=False)
            .drop_duplicates(["concept"], keep="first")
        )

        st.subheader(
            f"**Concepts Identified in Layer {layer}**",
            help="Concepts are defined based on Swiss-Prot annotations.",
        )
        display_cols = {
            "concept": "Concept",
            "feature": "Feature",
            "f1_per_domain": "F1",
            "precision": "Precision",
            "recall_per_domain": "Recall",
            "tp": "True Positives (per AA)",
        }

        st.dataframe(
            concepts[list(display_cols.keys())]
            .rename(columns=display_cols)
            .set_index("Concept"),
            height=300,
        )

    def _render_protein_visualizations(
        self,
        proteins: pd.DataFrame,
        dash_data: Dict,
        feature_id: int,
        add_highlight: bool,
        colormap_fn,
        palette_to_viz,
        is_custom_seq: bool = False,
    ):
        """Render visualizations for selected proteins"""
        for idx, (_, protein) in enumerate(proteins.iterrows()):
            try:
                # Get feature activations
                embeddings = embed_single_sequence(
                    sequence=protein["Sequence"],
                    model_name=dash_data["ESM_metadata"]["esm_model_name"],
                    layer=dash_data["ESM_metadata"]["layer"],
                    device=self.device,
                )
                features = (
                    encode_subset_of_feats(dash_data["SAE"], embeddings, [feature_id])
                    .cpu()
                    .numpy()
                    .flatten()
                )

                # Display protein header
                if is_custom_seq:
                    st.subheader(f"Custom Sequence {idx+1}")
                else:
                    st.subheader(
                        f"UniProt protein ([{protein['Entry']}](https://www.uniprot.org/uniprot/{protein['Entry']}))"
                    )
                    st.markdown(protein["Protein names"])

                if idx == 0:
                    col1, col2, col3 = st.columns([3, 3, 1])
                    with col3:
                        st.plotly_chart(
                            palette_to_viz,
                            use_container_width=True,
                            height=100,
                            config={"displayModeBar": False},
                        )
                else:
                    col1, col2 = st.columns([3, 5])

                # Display visualizations
                with col1:
                    st.plotly_chart(
                        visualize_protein_feature(
                            features, protein["Sequence"], protein, "Amino Acids"
                        ),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

                with col2:
                    if not is_custom_seq:
                        self._render_protein_structure(
                            protein["Entry"],
                            features,
                            colormap_fn,
                            (
                                [idx for idx, val in enumerate(features) if val > 0.4]
                                if add_highlight
                                else None
                            ),
                        )
            except Exception as e:
                st.error(f"Error visualizing protein {protein['Entry']}: {str(e)}")
                continue

    def _render_protein_structure(
        self,
        uniprot_id: str,
        features: np.ndarray,
        colormap_fn,
        highlight_residues: Optional[List[int]] = None,
    ):
        """Render 3D protein structure visualization"""
        try:
            structure_html = view_single_protein(
                uniprot_id=uniprot_id,
                values_to_color=features,
                colormap_fn=colormap_fn,
                residues_to_highlight=highlight_residues,
                pymol_params={"width": 500, "height": 300},
            )
            st.components.v1.html(structure_html, height=300)
        except Exception as e:
            st.error(f"Error visualizing structure: {str(e)}")


def main():
    st.set_page_config(
        layout="wide",
        page_title="InterPLM",
        page_icon="🧬",
    )

    # Parse command line arguments or use environment variables
    import sys
    if len(sys.argv) > 1:
        config = DashboardConfig().parse_args()
    else:
        # Try to get from environment variables
        config = DashboardConfig(
            source=os.environ.get("INTERPLM_SOURCE", "local"),
            repo_id=os.environ.get("INTERPLM_REPO_ID"),
            cache_dir=os.environ.get("INTERPLM_CACHE_DIR"),
            token=os.environ.get("HF_TOKEN"),
        )
    
    # Load data using cached function
    dash_data, protein_metadata, device, data_loader = load_dashboard_data(
        source=config.source,
        repo_id=config.repo_id,
        cache_dir=config.cache_dir,
        token=config.token
    )
    
    # Optional: download all data at startup
    if config.download_all and config.source == "remote":
        with st.spinner("Downloading all data for offline use..."):
            data_loader.download_all_data()
    
    # Try to load custom CSS if available
    try:
        with open(".streamlit/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        pass  # CSS is optional

    st.title("InterPLM Feature Visualization")

    visualizer = ProteinFeatureVisualizer(data_loader, dash_data, protein_metadata, device)
    state = visualizer.setup_sidebar()

    visualizer.display_feature_statistics(state.layer, state.feature_id)

    if state.show_proteins:
        visualizer.visualize_proteins(state)


if __name__ == "__main__":
    main()