"""
Protein selection logic for autointerp feature analysis.
Selects proteins with varying activation levels for LLM interpretation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
import random


def select_proteins_for_feature(
    feature_activations: np.ndarray,
    protein_metadata: pd.DataFrame,
    feature_id: int,
    n_proteins: int = 40,
    activation_blocks: List[Tuple[float, float]] = None,
    samples_per_block: List[int] = None,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Select proteins with varying activation levels for a specific feature.
    
    Based on the InterPLM paper methodology: select 40 proteins with varying levels 
    of maximum feature activation to provide diverse examples for LLM interpretation.
    
    Args:
        feature_activations: Array of shape (n_proteins, n_features) containing activation values
        protein_metadata: DataFrame containing protein metadata with Swiss-Prot annotations
        feature_id: ID of the feature to analyze
        n_proteins: Total number of proteins to select (default: 40)
        activation_blocks: List of (min, max) tuples defining activation ranges
        samples_per_block: Number of samples to take from each block
        random_seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries containing protein data and activation levels
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Default activation blocks similar to the paper's approach
    if activation_blocks is None:
        activation_blocks = [
            (0.0, 0.2),   # Very low activation
            (0.2, 0.4),   # Low activation  
            (0.4, 0.6),   # Medium activation
            (0.6, 0.8),   # High activation
            (0.8, 1.0)    # Very high activation
        ]
    
    if samples_per_block is None:
        # Auto-distribute proteins across blocks based on n_proteins
        # More samples from high activation ranges
        if n_proteins >= 40:
            samples_per_block = [4, 6, 8, 10, 12]  # Total = 40
        else:
            # Distribute proportionally for smaller counts
            base_distribution = [4, 6, 8, 10, 12]  # Proportions: 10%, 15%, 20%, 25%, 30%
            total_base = sum(base_distribution)
            samples_per_block = []
            remaining_proteins = n_proteins
            
            for i, base_count in enumerate(base_distribution[:-1]):
                # Proportional allocation
                allocated = max(1, int(base_count * n_proteins / total_base))
                allocated = min(allocated, remaining_proteins - (len(base_distribution) - i - 1))  # Leave at least 1 for remaining blocks
                samples_per_block.append(allocated)
                remaining_proteins -= allocated
            
            # Add remaining proteins to the last (highest activation) block
            samples_per_block.append(remaining_proteins)
    
    # Ensure we don't exceed available proteins
    if len(samples_per_block) != len(activation_blocks):
        raise ValueError("Length of samples_per_block must match activation_blocks")
    
    if sum(samples_per_block) != n_proteins:
        raise ValueError(f"Sum of samples_per_block ({sum(samples_per_block)}) must equal n_proteins ({n_proteins})")
    
    # Get activations for the specific feature
    if feature_id >= feature_activations.shape[1]:
        raise ValueError(f"Feature {feature_id} not found in activation matrix")
        
    feature_acts = feature_activations[:, feature_id]
    
    # Normalize activations to 0-1 range if not already normalized
    if feature_acts.max() > 1.0 or feature_acts.min() < 0.0:
        feature_acts = (feature_acts - feature_acts.min()) / (feature_acts.max() - feature_acts.min() + 1e-10)
    
    selected_proteins = []
    used_indices = set()
    
    # Sample from each activation block
    for (min_act, max_act), n_samples in zip(activation_blocks, samples_per_block):
        # Find proteins in this activation range
        in_range = np.where((feature_acts >= min_act) & (feature_acts < max_act))[0]
        available_indices = [idx for idx in in_range if idx not in used_indices]
        
        # Handle case where we don't have enough proteins in this range
        if len(available_indices) < n_samples:
            print(f"Warning: Only {len(available_indices)} proteins available in range [{min_act:.1f}, {max_act:.1f}), need {n_samples}")
            # Take all available proteins in this range
            selected_indices = available_indices
        else:
            # Randomly sample from available proteins
            selected_indices = random.sample(available_indices, n_samples)
        
        # Add selected proteins to our list
        for idx in selected_indices:
            protein_data = {
                'protein_idx': idx,
                'uniprot_id': protein_metadata.iloc[idx]['Entry'],
                'activation': float(feature_acts[idx]),
                'activation_range': f"{min_act:.1f}-{max_act:.1f}",
                'protein_names': protein_metadata.iloc[idx].get('Protein names', ''),
                'sequence': protein_metadata.iloc[idx].get('Sequence', ''),
                'length': protein_metadata.iloc[idx].get('Length', 0),
                'organism': protein_metadata.iloc[idx].get('Organism', ''),
                'gene_names': protein_metadata.iloc[idx].get('Gene names', ''),
                'entry_name': protein_metadata.iloc[idx].get('Entry name', '')
            }
            selected_proteins.append(protein_data)
            used_indices.add(idx)
    
    # Sort by activation level (highest first)
    selected_proteins.sort(key=lambda x: x['activation'], reverse=True)
    
    print(f"Selected {len(selected_proteins)} proteins for feature {feature_id}")
    print(f"Activation range: {min([p['activation'] for p in selected_proteins]):.3f} - {max([p['activation'] for p in selected_proteins]):.3f}")
    
    return selected_proteins


def get_swiss_prot_concepts_for_proteins(
    protein_indices: List[int],
    concept_data: pd.DataFrame,
    concept_threshold: float = 0.5
) -> Dict[int, List[str]]:
    """
    Get Swiss-Prot concept annotations for selected proteins.
    
    Args:
        protein_indices: List of protein indices
        concept_data: DataFrame containing concept-protein associations
        concept_threshold: Minimum F1 score threshold for concept inclusion
        
    Returns:
        Dictionary mapping protein index to list of associated concepts
    """
    
    protein_concepts = {}
    
    for protein_idx in protein_indices:
        concepts = []
        
        # Find concepts associated with this protein above threshold
        if 'protein_idx' in concept_data.columns:
            protein_concepts_data = concept_data[
                (concept_data['protein_idx'] == protein_idx) & 
                (concept_data['f1_per_domain'] >= concept_threshold)
            ]
            
            concepts = protein_concepts_data['concept'].tolist()
        
        protein_concepts[protein_idx] = concepts
    
    return protein_concepts


def select_validation_proteins(
    feature_activations: np.ndarray,
    protein_metadata: pd.DataFrame,
    feature_id: int,
    selected_protein_indices: List[int],
    n_validation: int = 100,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Select proteins for validation that are separate from training proteins.
    
    Args:
        feature_activations: Array of activation values
        protein_metadata: DataFrame containing protein metadata
        feature_id: ID of the feature to analyze
        selected_protein_indices: Indices of proteins already used for training
        n_validation: Number of validation proteins to select
        random_seed: Random seed for reproducibility
        
    Returns:
        List of validation protein dictionaries
    """
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get activations for the specific feature
    feature_acts = feature_activations[:, feature_id]
    
    # Normalize activations if needed
    if feature_acts.max() > 1.0 or feature_acts.min() < 0.0:
        feature_acts = (feature_acts - feature_acts.min()) / (feature_acts.max() - feature_acts.min() + 1e-10)
    
    # Get available protein indices (excluding training proteins)
    all_indices = set(range(len(protein_metadata)))
    used_indices = set(selected_protein_indices)
    available_indices = list(all_indices - used_indices)
    
    # Randomly sample validation proteins
    if len(available_indices) < n_validation:
        print(f"Warning: Only {len(available_indices)} proteins available for validation, need {n_validation}")
        validation_indices = available_indices
    else:
        validation_indices = random.sample(available_indices, n_validation)
    
    validation_proteins = []
    for idx in validation_indices:
        protein_data = {
            'protein_idx': idx,
            'uniprot_id': protein_metadata.iloc[idx]['Entry'],
            'activation': float(feature_acts[idx]),
            'protein_names': protein_metadata.iloc[idx].get('Protein names', ''),
            'sequence': protein_metadata.iloc[idx].get('Sequence', ''),
            'length': protein_metadata.iloc[idx].get('Length', 0),
            'organism': protein_metadata.iloc[idx].get('Organism', ''),
            'gene_names': protein_metadata.iloc[idx].get('Gene names', ''),
            'entry_name': protein_metadata.iloc[idx].get('Entry name', '')
        }
        validation_proteins.append(protein_data)
    
    print(f"Selected {len(validation_proteins)} validation proteins")
    
    return validation_proteins