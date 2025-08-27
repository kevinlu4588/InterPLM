"""
AutoInterp module for generating automated feature descriptions using LLMs.

This module implements the LLM-based feature interpretation methodology described
in the InterPLM paper, using Claude-3.5 Sonnet to generate human-readable 
descriptions of what SAE features detect in protein data.

Key components:
- AutoInterpGenerator: Main class for the complete interpretation pipeline
- protein_selection: Logic for selecting representative proteins for each feature
- prompt_templates: Templates for generating Claude prompts
- validation: Functionality for validating descriptions via correlation analysis
"""

from .generate_descriptions import AutoInterpGenerator
from .protein_selection import (
    select_proteins_for_feature, 
    select_validation_proteins,
    get_swiss_prot_concepts_for_proteins
)
from .prompt_templates import (
    create_feature_description_prompt,
    create_validation_prompt, 
    create_concept_integration_prompt
)
from .validation import (
    validate_descriptions_on_proteins,
    assess_description_quality,
    create_validation_report,
    calculate_prediction_correlation,
    parse_predictions_from_response,
    compare_with_swiss_prot_concepts
)

__version__ = "0.1.0"

__all__ = [
    # Main class
    "AutoInterpGenerator",
    
    # Protein selection
    "select_proteins_for_feature",
    "select_validation_proteins", 
    "get_swiss_prot_concepts_for_proteins",
    
    # Prompt generation
    "create_feature_description_prompt",
    "create_validation_prompt",
    "create_concept_integration_prompt",
    
    # Validation and scoring
    "validate_descriptions_on_proteins",
    "assess_description_quality", 
    "create_validation_report",
    "calculate_prediction_correlation",
    "parse_predictions_from_response",
    "compare_with_swiss_prot_concepts",
]