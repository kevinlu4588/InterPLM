# InterPLM LLM AutoInterp

This module implements the LLM-based feature interpretation methodology described in the InterPLM paper, using Claude-3.5 Sonnet to generate human-readable descriptions of what SAE features detect in protein data.

## Overview

The AutoInterp system automatically generates natural language descriptions of SAE features by:

1. **Protein Selection**: Selecting 40 proteins with varying activation levels for each feature
2. **LLM Description**: Sending protein examples + Swiss-Prot metadata to Claude-3.5 Sonnet  
3. **Validation**: Testing descriptions by predicting activations on separate proteins
4. **Correlation Scoring**: Measuring correlation between predicted and actual activations (paper reports median r=0.72)

## Installation Requirements

```bash
# Install required packages
pip install -r requirements.txt

# Set up your API key (choose one method):

# Method 1: Create .env file (recommended)
cp .env.example .env
# Edit .env file and add your API key

# Method 2: Environment variable
export ANTHROPIC_API_KEY="your_api_key_here"
```

## Quick Start

### 1. Basic Usage

First, set up your API key in a .env file:
```bash
# Copy the example file and edit it
cp .env.example .env
# Edit .env and set: ANTHROPIC_API_KEY=your_actual_api_key_here
```

Run autointerp on your trained SAE features:

```bash
python scripts/run_autointerp.py \
    --sae_dir models/walkthrough_model/ \
    --protein_metadata data/dashboard_cache/swiss-prot_metadata.tsv.gz \
    --feature_activations data/esm_embds/layer_3/ \
    --concept_results results/test_counts/concept_f1_scores.csv \
    --n_features 10 \
    --output_dir autointerp_results
```

Note: The `--claude_api_key` argument is now optional since the API key is loaded from the .env file.

### 2. Analyze Specific Features

```bash
python scripts/run_autointerp.py \
    --sae_dir models/walkthrough_model/ \
    --protein_metadata data/dashboard_cache/swiss-prot_metadata.tsv.gz \
    --feature_activations data/esm_embds/layer_3/ \
    --features 100 475 851 907 \
    --output_dir feature_specific_results
```

### 3. Add Results to Dashboard

```bash
python scripts/populate_autointerp_cache.py \
    --autointerp_results autointerp_results/ \
    --cache_file data/dashboard_cache/dashboard_cache.pkl \
    --layer 3 \
    --verify
```

## Programmatic Usage

```python
from interplm.autointerp import AutoInterpGenerator
import pandas as pd
import numpy as np

# Initialize generator
generator = AutoInterpGenerator(
    claude_api_key="your_api_key",
    output_dir="results"
)

# Load your data
feature_activations = np.load("feature_activations.npy")  # (n_proteins, n_features)
protein_metadata = pd.read_csv("protein_metadata.csv")
feature_ids = [0, 1, 2, 3, 4]  # Features to analyze

# Run complete pipeline
descriptions, validation_results = generator.run_full_pipeline(
    feature_activations=feature_activations,
    protein_metadata=protein_metadata,
    feature_ids=feature_ids,
    n_proteins_per_feature=40,
    n_validation_proteins=100
)

# Access results
for feature_id, description in descriptions.items():
    correlation = validation_results[
        validation_results['feature_id'] == feature_id
    ]['pearson_r'].iloc[0]
    print(f"Feature {feature_id} (r={correlation:.3f}): {description}")
```

## Module Components

### `AutoInterpGenerator`
Main class that orchestrates the complete pipeline:
- `generate_feature_descriptions()`: Generate LLM descriptions
- `validate_descriptions()`: Test descriptions via correlation
- `run_full_pipeline()`: Complete end-to-end analysis

### `protein_selection`
Logic for selecting representative proteins:
- `select_proteins_for_feature()`: Choose 40 proteins with varying activations
- `select_validation_proteins()`: Choose separate proteins for testing
- `get_swiss_prot_concepts_for_proteins()`: Extract biological annotations

### `prompt_templates`  
Templates for generating Claude prompts:
- `create_feature_description_prompt()`: Prompt for generating descriptions
- `create_validation_prompt()`: Prompt for predicting activations
- `create_concept_integration_prompt()`: Enhanced prompt with Swiss-Prot concepts

### `validation`
Functionality for validating descriptions:
- `calculate_prediction_correlation()`: Compute correlation metrics
- `assess_description_quality()`: Overall quality assessment
- `create_validation_report()`: Human-readable report generation

## Expected Outputs

### Feature Descriptions
Natural language descriptions of what each feature detects:

```
Feature 1503: This feature appears to detect TonB-dependent receptor (TBDR) 
beta barrel proteins, which are outer membrane transporters found primarily 
in Gram-negative bacteria. The feature shows high specificity for the 
characteristic beta barrel fold structure that forms channels across the 
bacterial outer membrane.
```

### Validation Results
Correlation scores measuring description accuracy:

```
Feature ID  | Description Preview                    | Correlation | P-value
-----------|----------------------------------------|-------------|--------
1503       | TonB-dependent receptor beta barrel... | 0.998       | <0.001
847        | ATP binding site kinase domain...      | 0.856       | <0.001  
1201       | Alpha helix secondary structure...     | 0.743       | 0.002
```

### Quality Assessment
```
SUMMARY STATISTICS:
- Features analyzed: 20
- Median Pearson correlation: 0.724
- Mean Pearson correlation: 0.681
- Statistically significant (p < 0.05): 18 (90.0%)
- High quality (r ≥ 0.5, p < 0.05): 16 (80.0%)
```

## Configuration Options

### Protein Selection
- `n_proteins_per_feature`: Number of proteins for description (default: 40)
- `activation_blocks`: Activation ranges for sampling (default: 0-0.2, 0.2-0.4, etc.)
- `samples_per_block`: Proteins per activation range (default: [4,6,8,10,12])

### LLM Settings
- `model_name`: Claude model to use (default: "claude-3-5-sonnet-20241022")
- `include_sequence`: Include protein sequences in prompts (default: False)
- `use_concept_integration`: Use Swiss-Prot concept information (default: True)

### Validation
- `n_validation_proteins`: Proteins for testing descriptions (default: 100)
- `correlation_threshold`: Minimum r for "good" descriptions (default: 0.5)
- `p_value_threshold`: Maximum p-value for significance (default: 0.05)

## Integration with Dashboard

The dashboard automatically displays LLM descriptions alongside Swiss-Prot concepts:

1. **Feature Summary**: Natural language description with correlation score
2. **Concept Comparison**: Side-by-side with Swiss-Prot concept annotations
3. **Interactive Exploration**: Click through features to see descriptions and examples

## Troubleshooting

### Common Issues

**API Rate Limits**: Claude API has rate limits. The system includes error handling and retries.

**Memory Issues**: For large datasets, reduce `max_proteins` parameter or process features in batches.

**Low Correlations**: 
- Check that protein metadata is comprehensive
- Ensure feature activations are properly normalized
- Try including protein sequences with `--include_sequence`

**Missing Swiss-Prot Concepts**:
- Run concept analysis pipeline first: `interplm/concept/compare_activations_to_concepts.py`
- AutoInterp works without concepts but is enhanced when they're available

### Validation Checks

```bash
# Verify cache integration
python scripts/populate_autointerp_cache.py --verify

# Check description quality
grep "Median correlation" autointerp_results/validation_report.txt

# Review individual feature results  
ls autointerp_results/feature_*_description.txt
```

## Citation

If you use the AutoInterp functionality, please cite:

```bibtex
@article{simon2024interplm,
  title={InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders},
  author={Simon, Elana and Zou, James},
  journal={bioRxiv},
  pages={2024.11.14.623630},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contributing

The AutoInterp module follows the InterPLM codebase conventions:
- Type hints for all functions
- Comprehensive docstrings
- Error handling and logging
- Modular, testable components

For questions or issues, please open a GitHub issue or contact the maintainers.