"""
Prompt templates for LLM-based feature interpretation.
Based on the InterPLM paper methodology using Claude-3.5 Sonnet.
"""

from typing import List, Dict, Any, Optional


def create_feature_description_prompt(
    feature_id: int,
    selected_proteins: List[Dict[str, Any]],
    swiss_prot_concepts: Optional[Dict[str, Any]] = None,
    include_sequence: bool = False
) -> str:
    """
    Create a prompt for Claude to generate feature descriptions.
    
    Based on the InterPLM paper: "By providing Claude-3.5 Sonnet (new) with the 
    Swiss-Prot concept information including text information not applicable for 
    classification, along with examples of 40 proteins with varying levels of 
    maximum feature activation, we generate descriptions of what protein and amino 
    acid characteristics activate the feature at different levels."
    
    Args:
        feature_id: ID of the SAE feature to interpret
        selected_proteins: List of protein dictionaries with activation levels
        swiss_prot_concepts: Optional Swiss-Prot concept information for this feature
        include_sequence: Whether to include protein sequences in the prompt
        
    Returns:
        Formatted prompt string for Claude
    """
    
    prompt = f"""You are an expert protein biochemist and computational biologist. I need you to analyze and describe what activates feature #{feature_id} in a sparse autoencoder trained on protein language model embeddings.

I will provide you with examples of 40 proteins that activate this feature at different levels (from 0 to 1, where 1 is maximum activation). Your task is to identify the biological patterns, structural motifs, functional domains, or other protein characteristics that cause this feature to activate.

"""

    # Add Swiss-Prot concept information if available
    if swiss_prot_concepts:
        prompt += f"""KNOWN SWISS-PROT ASSOCIATIONS FOR FEATURE {feature_id}:
"""
        if 'significant_concepts' in swiss_prot_concepts:
            for concept_info in swiss_prot_concepts['significant_concepts']:
                concept = concept_info.get('concept', 'Unknown')
                f1_score = concept_info.get('f1_per_domain', 0)
                precision = concept_info.get('precision', 0)
                recall = concept_info.get('recall_per_domain', 0)
                
                prompt += f"- {concept} (F1: {f1_score:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f})\n"
        
        prompt += "\n"
    
    prompt += f"""PROTEIN EXAMPLES WITH ACTIVATION LEVELS:

Below are {len(selected_proteins)} proteins ranked by their activation levels for feature #{feature_id}:

"""

    # Add protein examples sorted by activation level
    for i, protein in enumerate(selected_proteins, 1):
        activation = protein['activation']
        uniprot_id = protein['uniprot_id']
        protein_names = protein.get('protein_names', 'N/A')
        gene_names = protein.get('gene_names', 'N/A')
        organism = protein.get('organism', 'N/A')
        length = protein.get('length', 'N/A')
        
        prompt += f"""Protein {i}:
- UniProt ID: {uniprot_id}
- Activation Level: {activation:.4f}
- Protein Names: {protein_names}
- Gene Names: {gene_names}
- Organism: {organism}
- Length: {length} amino acids"""
        
        if include_sequence and 'sequence' in protein:
            sequence = protein['sequence']
            if len(sequence) > 500:  # Truncate very long sequences
                sequence = sequence[:500] + "..."
            prompt += f"\n- Sequence: {sequence}"
        
        prompt += "\n\n"
    
    prompt += """ANALYSIS TASK:

Based on the protein examples above, analyze what biological characteristics activate this feature. Consider:

1. **Functional patterns**: What molecular functions, biological processes, or cellular components are enriched?
2. **Structural motifs**: Are there common structural elements, domains, or folds?
3. **Sequence characteristics**: Any conserved sequences, amino acid compositions, or length patterns?
4. **Evolutionary patterns**: Phylogenetic or taxonomic enrichments?
5. **Biochemical properties**: Binding partners, enzymatic activities, or modifications?

Your analysis should explain:
- What this feature appears to detect in proteins
- Why proteins with higher activation share these characteristics
- How the activation level correlates with the strength/presence of these characteristics

Please provide a concise but comprehensive description (2-4 sentences) of what this SAE feature captures, focusing on the biological significance and interpretability."""

    return prompt


def create_validation_prompt(
    feature_id: int,
    feature_description: str,
    validation_proteins: List[Dict[str, Any]],
    include_sequence: bool = False
) -> str:
    """
    Create a prompt for Claude to predict feature activation levels on validation proteins.
    
    Args:
        feature_id: ID of the SAE feature
        feature_description: Generated description of what the feature detects
        validation_proteins: List of validation protein dictionaries
        include_sequence: Whether to include protein sequences in the prompt
        
    Returns:
        Formatted prompt string for validation predictions
    """
    
    prompt = f"""You are an expert at predicting sparse autoencoder feature activation levels based on feature descriptions.

FEATURE #{feature_id} DESCRIPTION:
{feature_description}

TASK:
Based on the above description, predict the activation level (between 0.0 and 1.0) for each of the {len(validation_proteins)} proteins below. The activation level should reflect how well each protein matches the characteristics described for this feature.

- 0.0 = No match with the feature characteristics
- 1.0 = Perfect match with the feature characteristics
- Use intermediate values for partial matches

VALIDATION PROTEINS:

"""

    for i, protein in enumerate(validation_proteins, 1):
        uniprot_id = protein['uniprot_id']
        protein_names = protein.get('protein_names', 'N/A')
        gene_names = protein.get('gene_names', 'N/A')
        organism = protein.get('organism', 'N/A')
        length = protein.get('length', 'N/A')
        
        prompt += f"""Protein {i}:
- UniProt ID: {uniprot_id}
- Protein Names: {protein_names}
- Gene Names: {gene_names}
- Organism: {organism}
- Length: {length} amino acids"""
        
        if include_sequence and 'sequence' in protein:
            sequence = protein['sequence']
            if len(sequence) > 500:  # Truncate very long sequences
                sequence = sequence[:500] + "..."
            prompt += f"\n- Sequence: {sequence}"
        
        prompt += "\n\n"
    
    prompt += f"""INSTRUCTIONS:
Return exactly {len(validation_proteins)} numbers (one per line), representing the predicted activation level for each protein in the order listed above. Each number should be between 0.0 and 1.0.

Do not include any other text, explanations, or formatting - only the numerical predictions."""

    return prompt


def create_concept_integration_prompt(
    feature_id: int,
    selected_proteins: List[Dict[str, Any]],
    swiss_prot_concepts: Dict[str, Any]
) -> str:
    """
    Create a specialized prompt that integrates Swiss-Prot concept information
    with protein examples for more accurate feature interpretation.
    
    Args:
        feature_id: ID of the SAE feature
        selected_proteins: List of protein dictionaries with activation levels  
        swiss_prot_concepts: Swiss-Prot concept associations for this feature
        
    Returns:
        Formatted prompt string emphasizing concept integration
    """
    
    prompt = f"""You are an expert protein biochemist analyzing sparse autoencoder features trained on protein embeddings. You have access to both computational predictions and curated biological annotations.

FEATURE #{feature_id} ANALYSIS

SWISS-PROT CONCEPT ASSOCIATIONS:
This feature shows statistical associations with the following biological concepts:

"""
    
    if 'significant_concepts' in swiss_prot_concepts:
        for concept_info in swiss_prot_concepts['significant_concepts']:
            concept = concept_info.get('concept', 'Unknown')
            f1_score = concept_info.get('f1_per_domain', 0)
            precision = concept_info.get('precision', 0)
            recall = concept_info.get('recall_per_domain', 0)
            tp_count = concept_info.get('tp_per_domain', 0)
            
            prompt += f"""• {concept}
  - F1 Score: {f1_score:.3f} (measure of overall accuracy)
  - Precision: {precision:.3f} (when feature activates, how often is this concept present?)  
  - Recall: {recall:.3f} (of all proteins with this concept, what fraction activate the feature?)
  - Supporting Examples: {tp_count} proteins
  
"""
    
    prompt += f"""REPRESENTATIVE PROTEIN EXAMPLES:
Here are {len(selected_proteins)} proteins with varying activation levels:

"""
    
    # Show top activating examples first
    high_activation = [p for p in selected_proteins if p['activation'] > 0.7]
    medium_activation = [p for p in selected_proteins if 0.3 <= p['activation'] <= 0.7]
    low_activation = [p for p in selected_proteins if p['activation'] < 0.3]
    
    if high_activation:
        prompt += "HIGH ACTIVATION EXAMPLES (>0.7):\n"
        for i, protein in enumerate(high_activation[:10], 1):  # Limit to top 10
            prompt += f"{i}. {protein['uniprot_id']} (activation: {protein['activation']:.3f})\n"
            prompt += f"   {protein.get('protein_names', 'N/A')}\n"
            prompt += f"   Organism: {protein.get('organism', 'N/A')}\n\n"
    
    if medium_activation:
        prompt += "MEDIUM ACTIVATION EXAMPLES (0.3-0.7):\n"
        for i, protein in enumerate(medium_activation[:5], 1):  # Show fewer medium examples
            prompt += f"{i}. {protein['uniprot_id']} (activation: {protein['activation']:.3f})\n"
            prompt += f"   {protein.get('protein_names', 'N/A')}\n\n"
    
    if low_activation:
        prompt += "LOW ACTIVATION EXAMPLES (<0.3):\n"
        for i, protein in enumerate(low_activation[:3], 1):  # Show fewer low examples
            prompt += f"{i}. {protein['uniprot_id']} (activation: {protein['activation']:.3f})\n"
            prompt += f"   {protein.get('protein_names', 'N/A')}\n\n"
    
    prompt += """INTERPRETATION TASK:

Integrate the Swiss-Prot concept associations with the protein examples to provide a comprehensive interpretation of what Feature #{feature_id} detects.

Consider:
1. How well do the Swiss-Prot concepts explain the high-activating proteins?
2. Are there additional patterns in the protein examples not captured by the concepts?
3. What is the biological coherence of this feature's selectivity?
4. How specific vs. general is this feature's detection pattern?

Provide a clear, scientifically accurate description (3-5 sentences) that explains:
- What biological characteristic(s) this feature primarily detects
- The confidence level of this interpretation based on the evidence
- Any caveats or limitations in the interpretation

Focus on biological interpretability and practical utility for protein analysis.""".replace('#{feature_id}', f'#{feature_id}')
    
    return prompt