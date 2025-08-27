"""
Validation and scoring functionality for LLM-generated feature descriptions.
Implements correlation-based validation as described in the InterPLM paper.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Any, Tuple, Optional
import re


def parse_predictions_from_response(
    claude_response: str,
    expected_count: int
) -> Optional[List[float]]:
    """
    Parse numerical predictions from Claude's response.
    
    Args:
        claude_response: Raw text response from Claude
        expected_count: Number of predictions expected
        
    Returns:
        List of float predictions, or None if parsing fails
    """
    
    # Extract all numbers that look like predictions (0.0 to 1.0)
    number_pattern = r'[0-1](?:\.[0-9]+)?|0\.[0-9]+'
    matches = re.findall(number_pattern, claude_response)
    
    predictions = []
    for match in matches:
        try:
            pred = float(match)
            if 0.0 <= pred <= 1.0:  # Valid prediction range
                predictions.append(pred)
        except ValueError:
            continue
    
    # Return predictions if we got the expected count
    if len(predictions) == expected_count:
        return predictions
    
    # Alternative parsing: look for numbers line by line
    lines = claude_response.strip().split('\n')
    line_predictions = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Try to extract a single number from this line
        line_matches = re.findall(r'[0-1](?:\.[0-9]+)?', line)
        if len(line_matches) == 1:
            try:
                pred = float(line_matches[0])
                if 0.0 <= pred <= 1.0:
                    line_predictions.append(pred)
            except ValueError:
                continue
    
    if len(line_predictions) == expected_count:
        return line_predictions
    
    print(f"Warning: Could not parse {expected_count} predictions from response")
    print(f"Found {len(predictions)} matches with pattern, {len(line_predictions)} line-by-line")
    return None


def calculate_prediction_correlation(
    predicted_activations: List[float],
    actual_activations: List[float]
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Calculate correlation between predicted and actual activation levels.
    
    Based on InterPLM paper: "showed high correlation with actual feature activation 
    (median Pearson r correlation = 0.72)"
    
    Args:
        predicted_activations: List of predicted activation values
        actual_activations: List of actual activation values
        
    Returns:
        Tuple of (pearson_r, p_value, additional_metrics_dict)
    """
    
    if len(predicted_activations) != len(actual_activations):
        raise ValueError("Predicted and actual activation lists must have same length")
    
    predicted = np.array(predicted_activations)
    actual = np.array(actual_activations)
    
    # Calculate Pearson correlation (primary metric from paper)
    pearson_r, pearson_p = pearsonr(predicted, actual)
    
    # Calculate additional metrics for comprehensive evaluation
    spearman_r, spearman_p = spearmanr(predicted, actual)
    
    # Calculate mean squared error and mean absolute error
    mse = np.mean((predicted - actual) ** 2)
    mae = np.mean(np.abs(predicted - actual))
    
    # Calculate R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    additional_metrics = {
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mse': mse,
        'mae': mae,
        'r_squared': r_squared,
        'n_samples': len(predicted),
        'predicted_mean': np.mean(predicted),
        'predicted_std': np.std(predicted),
        'actual_mean': np.mean(actual),
        'actual_std': np.std(actual)
    }
    
    return pearson_r, pearson_p, additional_metrics


def validate_descriptions_on_proteins(
    feature_descriptions: Dict[int, str],
    validation_data: Dict[int, List[Dict[str, Any]]],
    claude_client: Any,
    include_sequence: bool = False
) -> pd.DataFrame:
    """
    Validate feature descriptions by predicting activations on validation proteins.
    
    Args:
        feature_descriptions: Dictionary mapping feature_id to description
        validation_data: Dictionary mapping feature_id to list of validation proteins
        claude_client: Anthropic Claude client for API calls
        include_sequence: Whether to include protein sequences in prompts
        
    Returns:
        DataFrame with validation results for each feature
    """
    
    from .prompt_templates import create_validation_prompt
    
    results = []
    
    for feature_id, description in feature_descriptions.items():
        if feature_id not in validation_data:
            print(f"Warning: No validation data for feature {feature_id}")
            continue
            
        validation_proteins = validation_data[feature_id]
        actual_activations = [p['activation'] for p in validation_proteins]
        
        # Generate validation prompt
        prompt = create_validation_prompt(
            feature_id=feature_id,
            feature_description=description,
            validation_proteins=validation_proteins,
            include_sequence=include_sequence
        )
        
        try:
            # Get predictions from Claude
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract response text
            response_text = response.content[0].text if isinstance(response.content, list) else response.content
            
            # Parse predictions
            predictions = parse_predictions_from_response(response_text, len(validation_proteins))
            
            if predictions is None:
                print(f"Failed to parse predictions for feature {feature_id}")
                continue
            
            # Calculate correlation metrics
            pearson_r, pearson_p, additional_metrics = calculate_prediction_correlation(
                predictions, actual_activations
            )
            
            result = {
                'feature_id': feature_id,
                'description': description,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'n_validation_samples': len(validation_proteins),
                **additional_metrics
            }
            
            results.append(result)
            
            print(f"Feature {feature_id}: r={pearson_r:.3f}, p={pearson_p:.3f}")
            
        except Exception as e:
            print(f"Error validating feature {feature_id}: {str(e)}")
            continue
    
    return pd.DataFrame(results)


def assess_description_quality(
    validation_results: pd.DataFrame,
    correlation_threshold: float = 0.5,
    p_value_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Assess overall quality of feature descriptions based on validation results.
    
    Args:
        validation_results: DataFrame from validate_descriptions_on_proteins
        correlation_threshold: Minimum correlation for "good" descriptions  
        p_value_threshold: Maximum p-value for statistical significance
        
    Returns:
        Dictionary with quality assessment metrics
    """
    
    if validation_results.empty:
        return {
            'n_features': 0,
            'median_correlation': 0,
            'mean_correlation': 0,
            'significant_features': 0,
            'high_quality_features': 0
        }
    
    correlations = validation_results['pearson_r'].dropna()
    p_values = validation_results['pearson_p'].dropna()
    
    # Calculate summary statistics
    median_correlation = correlations.median()
    mean_correlation = correlations.mean()
    
    # Count significant and high-quality features
    significant_features = (p_values < p_value_threshold).sum()
    high_quality_features = ((correlations >= correlation_threshold) & 
                           (p_values < p_value_threshold)).sum()
    
    assessment = {
        'n_features': len(validation_results),
        'median_correlation': median_correlation,
        'mean_correlation': mean_correlation,
        'correlation_std': correlations.std(),
        'significant_features': significant_features,
        'significant_percentage': (significant_features / len(validation_results)) * 100,
        'high_quality_features': high_quality_features,
        'high_quality_percentage': (high_quality_features / len(validation_results)) * 100,
        'correlation_threshold': correlation_threshold,
        'p_value_threshold': p_value_threshold
    }
    
    return assessment


def create_validation_report(
    validation_results: pd.DataFrame,
    assessment: Dict[str, Any],
    output_path: str = None
) -> str:
    """
    Create a human-readable validation report.
    
    Args:
        validation_results: DataFrame with validation results
        assessment: Quality assessment dictionary
        output_path: Optional path to save the report
        
    Returns:
        Report text as string
    """
    
    report = f"""
InterPLM Feature Description Validation Report
{'='*50}

SUMMARY STATISTICS:
- Features analyzed: {assessment['n_features']}
- Median Pearson correlation: {assessment['median_correlation']:.3f}
- Mean Pearson correlation: {assessment['mean_correlation']:.3f}
- Standard deviation: {assessment['correlation_std']:.3f}

QUALITY ASSESSMENT:
- Statistically significant (p < {assessment['p_value_threshold']}): {assessment['significant_features']} ({assessment['significant_percentage']:.1f}%)
- High quality (r ≥ {assessment['correlation_threshold']}, p < {assessment['p_value_threshold']}): {assessment['high_quality_features']} ({assessment['high_quality_percentage']:.1f}%)

DETAILED RESULTS:
"""
    
    # Sort by correlation for better readability
    sorted_results = validation_results.sort_values('pearson_r', ascending=False)
    
    for _, row in sorted_results.iterrows():
        feature_id = row['feature_id']
        correlation = row['pearson_r']
        p_value = row['pearson_p']
        description = row['description']
        
        # Truncate long descriptions
        if len(description) > 150:
            description = description[:150] + "..."
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        
        report += f"\nFeature {feature_id}: r={correlation:.3f}{significance} (p={p_value:.3f})\n"
        report += f"  Description: {description}\n"
    
    report += f"\n\nLEGEND:\n*** p < 0.001, ** p < 0.01, * p < 0.05\n"
    report += f"\nBased on InterPLM methodology. Paper reported median r = 0.72\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Validation report saved to {output_path}")
    
    return report


def compare_with_swiss_prot_concepts(
    validation_results: pd.DataFrame,
    concept_results: pd.DataFrame,
    feature_ids: List[int] = None
) -> pd.DataFrame:
    """
    Compare LLM description quality with Swiss-Prot concept detection quality.
    
    Args:
        validation_results: DataFrame with LLM validation results
        concept_results: DataFrame with Swiss-Prot concept F1 scores
        feature_ids: Optional list of specific features to analyze
        
    Returns:
        DataFrame comparing LLM and concept-based interpretations
    """
    
    if feature_ids is None:
        feature_ids = validation_results['feature_id'].tolist()
    
    comparisons = []
    
    for feature_id in feature_ids:
        # Get LLM results
        llm_result = validation_results[validation_results['feature_id'] == feature_id]
        llm_correlation = llm_result['pearson_r'].iloc[0] if len(llm_result) > 0 else np.nan
        
        # Get Swiss-Prot concept results
        concept_result = concept_results[concept_results['feature'] == feature_id]
        max_f1 = concept_result['f1_per_domain'].max() if len(concept_result) > 0 else 0
        best_concept = concept_result.loc[concept_result['f1_per_domain'].idxmax(), 'concept'] if len(concept_result) > 0 else None
        
        comparisons.append({
            'feature_id': feature_id,
            'llm_correlation': llm_correlation,
            'best_concept_f1': max_f1,
            'best_concept': best_concept,
            'has_concept_match': max_f1 > 0.5,
            'has_good_llm_description': llm_correlation > 0.5,
            'interpretation_method': 'Both' if (max_f1 > 0.5 and llm_correlation > 0.5) else 
                                   'Concept only' if max_f1 > 0.5 else 
                                   'LLM only' if llm_correlation > 0.5 else 'Neither'
        })
    
    return pd.DataFrame(comparisons)