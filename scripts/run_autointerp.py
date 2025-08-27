"""
Script to run LLM-based feature interpretation on InterPLM SAE features.

Usage:
    # First set up your API key in .env file:
    cp .env.example .env
    # Edit .env and set: ANTHROPIC_API_KEY=your_actual_api_key_here
    
    # Then run the script:
    python scripts/run_autointerp.py --sae_dir models/walkthrough_model/ \
                                     --protein_metadata data/dashboard_cache/swiss-prot_metadata.tsv.gz \
                                     --feature_activations data/esm_embds/layer_3/ \
                                     --concept_results results/test_counts/concept_f1_scores.csv \
                                     --features 100 475 851 907

Alternative with environment variable:
    export ANTHROPIC_API_KEY=your_api_key_here
    python scripts/run_autointerp.py --sae_dir models/walkthrough_model/ \
                                     --protein_metadata data/dashboard_cache/swiss-prot_metadata.tsv.gz \
                                     --feature_activations data/esm_embds/layer_3/ \
                                     --concept_results results/test_counts/concept_f1_scores.csv \
                                     --n_features 10
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the path to import interplm modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from interplm.autointerp import AutoInterpGenerator
from interplm.sae.inference import load_sae, get_sae_feats_in_batches
from interplm.esm.embed import embed_single_sequence


def load_feature_activations(sae_dir, esm_embeds_dir, protein_metadata, layer=3, max_proteins=1000):
    """
    Load SAE feature activations for proteins in the metadata.
    
    Args:
        sae_dir: Directory containing the SAE model
        esm_embeds_dir: Directory containing ESM embeddings  
        protein_metadata: DataFrame with protein metadata
        layer: ESM layer to use
        max_proteins: Maximum number of proteins to process
        
    Returns:
        numpy array of shape (n_proteins, n_features) with activation values
    """
    
    print(f"Loading SAE from {sae_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load SAE model
    sae_path = Path(sae_dir) / "ae_normalized.pt"
    if not sae_path.exists():
        sae_path = Path(sae_dir) / "ae.pt"
    
    sae = load_sae(model_path=sae_path, device=device)
    print(f"Loaded SAE with {sae.dict_size} features")
    
    # Load embeddings for the proteins
    print(f"Loading embeddings from {esm_embeds_dir}")
    embed_files = list(Path(esm_embeds_dir).glob("shard_*.pt"))
    
    if not embed_files:
        raise FileNotFoundError(f"No embedding files found in {esm_embeds_dir}")
    
    all_activations = []
    processed_proteins = 0
    
    for embed_file in embed_files:
        if processed_proteins >= max_proteins:
            break
            
        print(f"Processing {embed_file.name}...")
        
        # Load embeddings
        embeddings = torch.load(embed_file, map_location=device)
        
        # Get SAE activations for this batch
        with torch.no_grad():
            # Process in chunks to avoid memory issues
            chunk_size = 64
            batch_activations = []
            
            for i in range(0, embeddings.shape[0], chunk_size):
                chunk = embeddings[i:i + chunk_size]
                chunk_acts = sae.encode(chunk)  # Get feature activations
                batch_activations.append(chunk_acts.cpu().numpy())
                
                processed_proteins += chunk.shape[0]
                if processed_proteins >= max_proteins:
                    break
            
            if batch_activations:
                batch_acts = np.concatenate(batch_activations, axis=0)
                all_activations.append(batch_acts)
    
    if not all_activations:
        raise ValueError("No activations were generated")
    
    # Combine all activations
    feature_activations = np.concatenate(all_activations, axis=0)
    
    # Take max activation per protein (across sequence positions)
    if len(feature_activations.shape) == 3:  # (proteins, seq_len, features)
        feature_activations = np.max(feature_activations, axis=1)  # (proteins, features)
    
    print(f"Generated activations shape: {feature_activations.shape}")
    return feature_activations[:max_proteins]


def load_swiss_prot_concepts(concept_results_path, min_f1=0.5):
    """
    Load Swiss-Prot concept associations from CSV results.
    
    Args:
        concept_results_path: Path to concept F1 scores CSV
        min_f1: Minimum F1 score threshold for including concepts
        
    Returns:
        Dictionary mapping feature_id to concept information
    """
    
    if not Path(concept_results_path).exists():
        print(f"Warning: Concept results file not found: {concept_results_path}")
        return {}
    
    concept_df = pd.read_csv(concept_results_path)
    
    # Filter for significant concepts
    significant_concepts = concept_df[concept_df['f1_per_domain'] >= min_f1]
    
    # Group by feature
    feature_concepts = {}
    for feature_id, group in significant_concepts.groupby('feature'):
        concept_info = []
        for _, row in group.iterrows():
            concept_info.append({
                'concept': row['concept'],
                'f1_per_domain': row['f1_per_domain'],
                'precision': row['precision'],
                'recall_per_domain': row['recall_per_domain'],
                'tp_per_domain': row.get('tp_per_domain', 0)
            })
        
        feature_concepts[feature_id] = {
            'significant_concepts': concept_info
        }
    
    print(f"Loaded concept associations for {len(feature_concepts)} features")
    return feature_concepts


def select_top_features(concept_results_path, n_features=10):
    """
    Select top features based on Swiss-Prot concept F1 scores.
    
    Args:
        concept_results_path: Path to concept results CSV
        n_features: Number of top features to select
        
    Returns:
        List of feature IDs sorted by best F1 score
    """
    
    if not Path(concept_results_path).exists():
        print(f"Warning: Using random features as concept results not found")
        return list(range(n_features))
    
    concept_df = pd.read_csv(concept_results_path)
    
    # Get best F1 score for each feature
    feature_scores = concept_df.groupby('feature')['f1_per_domain'].max().sort_values(ascending=False)
    
    # Select top N features
    top_features = feature_scores.head(n_features).index.tolist()
    
    print(f"Selected top {len(top_features)} features by F1 score:")
    for i, feature_id in enumerate(top_features[:5], 1):
        score = feature_scores[feature_id]
        print(f"  {i}. Feature {feature_id}: F1 = {score:.3f}")
    
    return top_features


def main():
    parser = argparse.ArgumentParser(description="Run LLM-based feature interpretation")
    
    # Required arguments
    parser.add_argument("--sae_dir", required=True, help="Directory containing SAE model")
    parser.add_argument("--protein_metadata", required=True, help="Path to protein metadata CSV/TSV")
    parser.add_argument("--feature_activations", required=True, help="Directory with ESM embeddings or pre-computed activations")
    
    # Optional arguments
    parser.add_argument("--concept_results", help="Path to Swiss-Prot concept F1 scores CSV")
    parser.add_argument("--claude_api_key", help="Claude API key (optional - will load from .env file or ANTHROPIC_API_KEY env var)")
    parser.add_argument("--features", nargs="+", type=int, help="Specific feature IDs to analyze")
    parser.add_argument("--n_features", type=int, default=10, help="Number of top features to analyze")
    parser.add_argument("--output_dir", default="autointerp_results", help="Output directory")
    parser.add_argument("--n_proteins_per_feature", type=int, default=40, help="Proteins per feature for description")
    parser.add_argument("--n_validation_proteins", type=int, default=100, help="Proteins for validation") 
    parser.add_argument("--include_sequence", action="store_true", help="Include protein sequences in prompts")
    parser.add_argument("--max_proteins", type=int, default=1000, help="Maximum proteins to load")
    
    args = parser.parse_args()
    
    print("InterPLM AutoInterp - LLM-based Feature Interpretation")
    print("=" * 60)
    
    # Load protein metadata
    print(f"Loading protein metadata from {args.protein_metadata}")
    if args.protein_metadata.endswith('.gz'):
        protein_metadata = pd.read_csv(args.protein_metadata, sep='\t', compression='gzip')
    else:
        protein_metadata = pd.read_csv(args.protein_metadata, sep='\t')
    
    print(f"Loaded metadata for {len(protein_metadata)} proteins")
    
    # Load feature activations
    print("Loading feature activations...")
    feature_activations = load_feature_activations(
        sae_dir=args.sae_dir,
        esm_embeds_dir=args.feature_activations,
        protein_metadata=protein_metadata,
        max_proteins=args.max_proteins
    )
    
    # Ensure we don't have more proteins than metadata
    n_proteins = min(len(protein_metadata), feature_activations.shape[0])
    feature_activations = feature_activations[:n_proteins]
    protein_metadata = protein_metadata.head(n_proteins).reset_index(drop=True)
    
    print(f"Using {n_proteins} proteins and {feature_activations.shape[1]} features")
    
    # Select features to analyze
    if args.features:
        feature_ids = args.features
        print(f"Analyzing specified features: {feature_ids}")
    else:
        feature_ids = select_top_features(args.concept_results, args.n_features)
        print(f"Analyzing top {len(feature_ids)} features by concept association")
    
    # Load Swiss-Prot concepts if available
    swiss_prot_concepts = {}
    if args.concept_results:
        swiss_prot_concepts = load_swiss_prot_concepts(args.concept_results)
    
    # Initialize AutoInterp generator
    print(f"\nInitializing AutoInterp generator...")
    generator = AutoInterpGenerator(
        claude_api_key=args.claude_api_key,
        output_dir=args.output_dir
    )
    
    # Run the complete pipeline
    print(f"\nRunning AutoInterp pipeline...")
    descriptions, validation_results = generator.run_full_pipeline(
        feature_activations=feature_activations,
        protein_metadata=protein_metadata,
        feature_ids=feature_ids,
        swiss_prot_concepts=swiss_prot_concepts,
        n_proteins_per_feature=args.n_proteins_per_feature,
        n_validation_proteins=args.n_validation_proteins,
        include_sequence=args.include_sequence
    )
    
    # Print summary
    print(f"\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Features analyzed: {len(feature_ids)}")
    print(f"Descriptions generated: {len(descriptions)}")
    print(f"Validations completed: {len(validation_results)}")
    
    if len(validation_results) > 0:
        median_r = validation_results['pearson_r'].median()
        mean_r = validation_results['pearson_r'].mean()
        significant = (validation_results['pearson_p'] < 0.05).sum()
        
        print(f"Median correlation: {median_r:.3f}")
        print(f"Mean correlation: {mean_r:.3f}")
        print(f"Significant features (p < 0.05): {significant}/{len(validation_results)} ({significant/len(validation_results)*100:.1f}%)")
        
        # Show best results
        print(f"\nBest performing features:")
        best_features = validation_results.nlargest(5, 'pearson_r')
        for _, row in best_features.iterrows():
            print(f"  Feature {row['feature_id']}: r={row['pearson_r']:.3f}, p={row['pearson_p']:.3f}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"Paper reported median r = 0.72 for comparison")


if __name__ == "__main__":
    main()