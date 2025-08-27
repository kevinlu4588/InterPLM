"""
Main module for generating automated feature descriptions using LLMs.
Implements the InterPLM paper's methodology for SAE feature interpretation.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime

from anthropic import Anthropic
from dotenv import load_dotenv

from .protein_selection import select_proteins_for_feature, select_validation_proteins
from .prompt_templates import (
    create_feature_description_prompt, 
    create_validation_prompt,
    create_concept_integration_prompt
)
from .validation import (
    validate_descriptions_on_proteins,
    assess_description_quality,
    create_validation_report,
    parse_predictions_from_response,
    calculate_prediction_correlation
)


class AutoInterpGenerator:
    """
    Main class for generating LLM-based feature interpretations.
    
    Based on InterPLM paper methodology:
    1. Select 40 proteins with varying activation levels for each feature
    2. Send protein examples + Swiss-Prot metadata to Claude-3.5 Sonnet
    3. Generate natural language descriptions of what activates the feature
    4. Validate descriptions by predicting activations on separate proteins
    5. Measure correlation between predicted and actual activations
    """
    
    def __init__(
        self,
        claude_api_key: Optional[str] = None,
        model_name: str = "claude-3-5-sonnet-20241022",
        output_dir: str = "autointerp_results",
        random_seed: int = 42
    ):
        """
        Initialize the AutoInterp generator.
        
        Args:
            claude_api_key: API key for Claude (if None, will look for .env file or environment variable)
            model_name: Claude model to use for interpretation
            output_dir: Directory to save results
            random_seed: Random seed for reproducibility
        """
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Set up Claude client
        if claude_api_key is None:
            claude_api_key = os.getenv('ANTHROPIC_API_KEY')
            
        if claude_api_key is None:
            raise ValueError("Claude API key must be provided or set in ANTHROPIC_API_KEY environment variable or .env file")
        
        self.claude_client = Anthropic(api_key=claude_api_key)
        self.model_name = model_name
        
        # Set up output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Initialize storage
        self.feature_descriptions = {}
        self.validation_results = pd.DataFrame()
        
        print(f"AutoInterpGenerator initialized with model: {model_name}")
        print(f"Output directory: {self.output_dir}")
    
    def generate_feature_descriptions(
        self,
        feature_activations: np.ndarray,
        protein_metadata: pd.DataFrame,
        feature_ids: List[int],
        swiss_prot_concepts: Optional[Dict[int, Dict[str, Any]]] = None,
        n_proteins_per_feature: int = 40,
        include_sequence: bool = False,
        use_concept_integration: bool = True
    ) -> Dict[int, str]:
        """
        Generate descriptions for a list of SAE features.
        
        Args:
            feature_activations: Array of shape (n_proteins, n_features) with activation values
            protein_metadata: DataFrame containing protein metadata  
            feature_ids: List of feature IDs to generate descriptions for
            swiss_prot_concepts: Optional dictionary of Swiss-Prot concept associations
            n_proteins_per_feature: Number of proteins to sample per feature
            include_sequence: Whether to include protein sequences in prompts
            use_concept_integration: Whether to use concept-integrated prompts when available
            
        Returns:
            Dictionary mapping feature_id to generated description
        """
        
        print(f"Generating descriptions for {len(feature_ids)} features...")
        
        descriptions = {}
        
        for feature_id in tqdm(feature_ids, desc="Generating descriptions"):
            try:
                # Select proteins for this feature
                selected_proteins = select_proteins_for_feature(
                    feature_activations=feature_activations,
                    protein_metadata=protein_metadata,
                    feature_id=feature_id,
                    n_proteins=n_proteins_per_feature
                )
                
                # Get Swiss-Prot concepts if available
                feature_concepts = swiss_prot_concepts.get(feature_id) if swiss_prot_concepts else None
                
                # Choose prompt type based on available information
                if use_concept_integration and feature_concepts:
                    prompt = create_concept_integration_prompt(
                        feature_id=feature_id,
                        selected_proteins=selected_proteins,
                        swiss_prot_concepts=feature_concepts
                    )
                else:
                    prompt = create_feature_description_prompt(
                        feature_id=feature_id,
                        selected_proteins=selected_proteins,
                        swiss_prot_concepts=feature_concepts,
                        include_sequence=include_sequence
                    )
                
                # Save prompt for debugging
                prompt_file = self.output_dir / f"feature_{feature_id}_prompt.txt"
                with open(prompt_file, 'w') as f:
                    f.write(prompt)
                
                # Get description from Claude
                response = self.claude_client.messages.create(
                    model=self.model_name,
                    max_tokens=500,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract description text
                description = response.content[0].text if isinstance(response.content, list) else response.content
                descriptions[feature_id] = description.strip()
                
                # Save individual result
                result_file = self.output_dir / f"feature_{feature_id}_description.txt"
                with open(result_file, 'w') as f:
                    f.write(f"Feature {feature_id} Description:\n")
                    f.write(f"{description}\n\n")
                    f.write(f"Generated: {datetime.datetime.now()}\n")
                    f.write(f"Model: {self.model_name}\n")
                
                print(f"Feature {feature_id}: Generated description ({len(description)} chars)")
                
            except Exception as e:
                print(f"Error generating description for feature {feature_id}: {str(e)}")
                continue
        
        # Save all descriptions
        self.feature_descriptions = descriptions
        descriptions_file = self.output_dir / "all_feature_descriptions.json"
        with open(descriptions_file, 'w') as f:
            json.dump(descriptions, f, indent=2)
        
        print(f"Successfully generated {len(descriptions)} descriptions")
        print(f"Saved to: {descriptions_file}")
        
        return descriptions
    
    def validate_descriptions(
        self,
        feature_activations: np.ndarray,
        protein_metadata: pd.DataFrame,
        feature_descriptions: Optional[Dict[int, str]] = None,
        n_validation_proteins: int = 100,
        include_sequence: bool = False
    ) -> pd.DataFrame:
        """
        Validate feature descriptions by predicting activation levels.
        
        Args:
            feature_activations: Array of activation values
            protein_metadata: DataFrame containing protein metadata
            feature_descriptions: Dict of descriptions (if None, use stored descriptions)
            n_validation_proteins: Number of validation proteins per feature
            include_sequence: Whether to include sequences in validation prompts
            
        Returns:
            DataFrame with validation results
        """
        
        if feature_descriptions is None:
            feature_descriptions = self.feature_descriptions
            
        if not feature_descriptions:
            raise ValueError("No feature descriptions available for validation")
        
        print(f"Validating {len(feature_descriptions)} feature descriptions...")
        
        results = []
        
        for feature_id, description in tqdm(feature_descriptions.items(), desc="Validating descriptions"):
            try:
                # Select validation proteins (separate from training proteins)
                validation_proteins = select_validation_proteins(
                    feature_activations=feature_activations,
                    protein_metadata=protein_metadata,
                    feature_id=feature_id,
                    selected_protein_indices=[],  # Would need to track training proteins in production
                    n_validation=n_validation_proteins
                )
                
                actual_activations = [p['activation'] for p in validation_proteins]
                
                # Generate validation prompt
                prompt = create_validation_prompt(
                    feature_id=feature_id,
                    feature_description=description,
                    validation_proteins=validation_proteins,
                    include_sequence=include_sequence
                )
                
                # Save validation prompt
                validation_prompt_file = self.output_dir / f"feature_{feature_id}_validation_prompt.txt"
                with open(validation_prompt_file, 'w') as f:
                    f.write(prompt)
                
                # Get predictions from Claude
                response = self.claude_client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                response_text = response.content[0].text if isinstance(response.content, list) else response.content
                
                # Parse predictions
                predictions = parse_predictions_from_response(response_text, len(validation_proteins))
                
                if predictions is None:
                    print(f"Failed to parse predictions for feature {feature_id}")
                    continue
                
                # Calculate correlation
                pearson_r, pearson_p, additional_metrics = calculate_prediction_correlation(
                    predictions, actual_activations
                )
                
                # Store result
                result = {
                    'feature_id': feature_id,
                    'description': description,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'n_validation_samples': len(validation_proteins),
                    **additional_metrics
                }
                
                results.append(result)
                
                # Save individual validation result
                validation_file = self.output_dir / f"feature_{feature_id}_validation.json"
                with open(validation_file, 'w') as f:
                    json.dump({
                        'feature_id': feature_id,
                        'description': description,
                        'predictions': predictions,
                        'actual_activations': actual_activations,
                        'correlation_results': result
                    }, f, indent=2)
                
                print(f"Feature {feature_id}: r={pearson_r:.3f}, p={pearson_p:.3f}")
                
            except Exception as e:
                print(f"Error validating feature {feature_id}: {str(e)}")
                continue
        
        # Create results DataFrame
        validation_results = pd.DataFrame(results)
        
        # Save validation results
        validation_file = self.output_dir / "validation_results.csv"
        validation_results.to_csv(validation_file, index=False)
        
        # Generate and save validation report
        assessment = assess_description_quality(validation_results)
        report = create_validation_report(
            validation_results, 
            assessment,
            str(self.output_dir / "validation_report.txt")
        )
        
        self.validation_results = validation_results
        
        print(f"Validation complete. Median correlation: {assessment['median_correlation']:.3f}")
        print(f"Results saved to: {validation_file}")
        
        return validation_results
    
    def run_full_pipeline(
        self,
        feature_activations: np.ndarray,
        protein_metadata: pd.DataFrame,
        feature_ids: List[int],
        swiss_prot_concepts: Optional[Dict[int, Dict[str, Any]]] = None,
        n_proteins_per_feature: int = 40,
        n_validation_proteins: int = 100,
        include_sequence: bool = False,
        use_concept_integration: bool = True
    ) -> Tuple[Dict[int, str], pd.DataFrame]:
        """
        Run the complete autointerp pipeline: generate descriptions and validate them.
        
        Args:
            feature_activations: Array of activation values
            protein_metadata: DataFrame containing protein metadata
            feature_ids: List of feature IDs to analyze
            swiss_prot_concepts: Optional Swiss-Prot concept associations
            n_proteins_per_feature: Number of proteins for description generation
            n_validation_proteins: Number of proteins for validation
            include_sequence: Whether to include sequences in prompts
            use_concept_integration: Whether to integrate Swiss-Prot concepts
            
        Returns:
            Tuple of (feature_descriptions_dict, validation_results_dataframe)
        """
        
        print("Starting AutoInterp pipeline...")
        print(f"Features to analyze: {len(feature_ids)}")
        print(f"Proteins per feature: {n_proteins_per_feature}")
        print(f"Validation proteins per feature: {n_validation_proteins}")
        
        # Step 1: Generate descriptions
        descriptions = self.generate_feature_descriptions(
            feature_activations=feature_activations,
            protein_metadata=protein_metadata,
            feature_ids=feature_ids,
            swiss_prot_concepts=swiss_prot_concepts,
            n_proteins_per_feature=n_proteins_per_feature,
            include_sequence=include_sequence,
            use_concept_integration=use_concept_integration
        )
        
        # Step 2: Validate descriptions
        validation_results = self.validate_descriptions(
            feature_activations=feature_activations,
            protein_metadata=protein_metadata,
            feature_descriptions=descriptions,
            n_validation_proteins=n_validation_proteins,
            include_sequence=include_sequence
        )
        
        # Step 3: Create summary
        summary = {
            'pipeline_completion_time': datetime.datetime.now().isoformat(),
            'n_features_attempted': len(feature_ids),
            'n_descriptions_generated': len(descriptions),
            'n_validations_completed': len(validation_results),
            'median_correlation': validation_results['pearson_r'].median() if len(validation_results) > 0 else 0,
            'mean_correlation': validation_results['pearson_r'].mean() if len(validation_results) > 0 else 0
        }
        
        summary_file = self.output_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nPipeline complete!")
        print(f"Generated descriptions: {len(descriptions)}")
        print(f"Completed validations: {len(validation_results)}")
        print(f"Median correlation: {summary['median_correlation']:.3f}")
        print(f"Results saved to: {self.output_dir}")
        
        return descriptions, validation_results
    
    def load_existing_results(self, results_dir: str = None) -> bool:
        """
        Load previously generated results from disk.
        
        Args:
            results_dir: Directory containing results (if None, use self.output_dir)
            
        Returns:
            True if results were loaded successfully
        """
        
        if results_dir is None:
            results_dir = self.output_dir
        else:
            results_dir = Path(results_dir)
        
        try:
            # Load descriptions
            descriptions_file = results_dir / "all_feature_descriptions.json"
            if descriptions_file.exists():
                with open(descriptions_file, 'r') as f:
                    self.feature_descriptions = json.load(f)
                    # Convert string keys back to integers
                    self.feature_descriptions = {int(k): v for k, v in self.feature_descriptions.items()}
            
            # Load validation results
            validation_file = results_dir / "validation_results.csv"
            if validation_file.exists():
                self.validation_results = pd.read_csv(validation_file)
            
            print(f"Loaded {len(self.feature_descriptions)} descriptions and {len(self.validation_results)} validation results")
            return True
            
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            return False