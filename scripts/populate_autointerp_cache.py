"""
Script to populate dashboard cache with LLM autointerp results.

This script takes the results from run_autointerp.py and integrates them 
into the dashboard cache so they appear in the InterPLM dashboard.

Usage:
    python scripts/populate_autointerp_cache.py --autointerp_results autointerp_results/ \
                                                --cache_file data/dashboard_cache/dashboard_cache.pkl \
                                                --layer 3
"""

import argparse
import pickle
import json
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_autointerp_results(results_dir):
    """
    Load autointerp results from the output directory.
    
    Args:
        results_dir: Directory containing autointerp results
        
    Returns:
        Tuple of (descriptions_dict, validation_dataframe)
    """
    
    results_path = Path(results_dir)
    
    # Load descriptions
    descriptions_file = results_path / "all_feature_descriptions.json"
    descriptions = {}
    if descriptions_file.exists():
        with open(descriptions_file, 'r') as f:
            descriptions = json.load(f)
            # Convert string keys to integers
            descriptions = {int(k): v for k, v in descriptions.items()}
    
    # Load validation results  
    validation_file = results_path / "validation_results.csv"
    validation_df = pd.DataFrame()
    if validation_file.exists():
        validation_df = pd.read_csv(validation_file)
    
    print(f"Loaded {len(descriptions)} descriptions and {len(validation_df)} validation results")
    return descriptions, validation_df


def create_autointerp_dataframe(descriptions, validation_df):
    """
    Create a DataFrame in the format expected by the dashboard.
    
    The dashboard expects an "LLM Autointerp" DataFrame with columns:
    - feature_id (index)
    - Summary: The description text
    - Correlation: The validation correlation score
    
    Args:
        descriptions: Dictionary mapping feature_id to description
        validation_df: DataFrame with validation results
        
    Returns:
        DataFrame formatted for dashboard
    """
    
    autointerp_data = []
    
    for feature_id, description in descriptions.items():
        # Get validation correlation if available
        correlation = 0.0
        if len(validation_df) > 0:
            feature_validation = validation_df[validation_df['feature_id'] == feature_id]
            if len(feature_validation) > 0:
                correlation = feature_validation['pearson_r'].iloc[0]
                # Handle NaN values
                if pd.isna(correlation):
                    correlation = 0.0
        
        autointerp_data.append({
            'feature_id': feature_id,
            'Summary': description,
            'Correlation': correlation
        })
    
    df = pd.DataFrame(autointerp_data)
    df.set_index('feature_id', inplace=True)
    
    return df


def update_dashboard_cache(cache_file, autointerp_df, layer):
    """
    Update the dashboard cache with LLM autointerp data.
    
    Args:
        cache_file: Path to dashboard cache pickle file
        autointerp_df: DataFrame with autointerp results
        layer: Layer number to update
    """
    
    cache_path = Path(cache_file)
    
    # Load existing cache
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"Loaded existing cache with layers: {list(cache_data.keys())}")
    else:
        print("Warning: Cache file does not exist, creating new cache")
        cache_data = {}
    
    # Check if layer exists in cache
    if layer not in cache_data:
        print(f"Warning: Layer {layer} not found in cache")
        return
    
    # Add autointerp data to the layer
    cache_data[layer]["LLM Autointerp"] = autointerp_df
    
    # Create backup
    backup_path = cache_path.with_suffix('.pkl.backup')
    if cache_path.exists():
        import shutil
        shutil.copy2(cache_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Save updated cache
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Updated cache with {len(autointerp_df)} LLM descriptions for layer {layer}")
    print(f"Cache saved to: {cache_path}")


def verify_cache_update(cache_file, layer, expected_features):
    """
    Verify that the cache was updated correctly.
    
    Args:
        cache_file: Path to cache file
        layer: Layer to check
        expected_features: List of feature IDs that should be present
    """
    
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    if layer not in cache_data:
        print(f"ERROR: Layer {layer} not found in cache")
        return False
    
    if "LLM Autointerp" not in cache_data[layer]:
        print(f"ERROR: LLM Autointerp not found in layer {layer}")
        return False
    
    autointerp_df = cache_data[layer]["LLM Autointerp"]
    cached_features = set(autointerp_df.index.tolist())
    expected_features_set = set(expected_features)
    
    missing = expected_features_set - cached_features
    extra = cached_features - expected_features_set
    
    print(f"Verification results:")
    print(f"  Expected features: {len(expected_features)}")
    print(f"  Cached features: {len(cached_features)}")
    print(f"  Missing features: {len(missing)} {list(missing) if missing else ''}")
    print(f"  Extra features: {len(extra)} {list(extra) if extra else ''}")
    
    # Show sample of cached data
    print(f"\nSample cached data:")
    for feature_id in list(cached_features)[:3]:
        row = autointerp_df.loc[feature_id]
        summary = row['Summary'][:100] + "..." if len(row['Summary']) > 100 else row['Summary']
        print(f"  Feature {feature_id}: r={row['Correlation']:.3f}, '{summary}'")
    
    return len(missing) == 0


def main():
    parser = argparse.ArgumentParser(description="Populate dashboard cache with autointerp results")
    
    parser.add_argument("--autointerp_results", required=True, 
                       help="Directory containing autointerp results")
    parser.add_argument("--cache_file", required=True,
                       help="Path to dashboard cache pickle file") 
    parser.add_argument("--layer", type=int, required=True,
                       help="Layer number to update")
    parser.add_argument("--verify", action="store_true",
                       help="Verify cache was updated correctly")
    
    args = parser.parse_args()
    
    print("InterPLM Dashboard Cache Update - LLM Autointerp")
    print("=" * 50)
    
    # Load autointerp results
    print(f"Loading autointerp results from {args.autointerp_results}")
    descriptions, validation_df = load_autointerp_results(args.autointerp_results)
    
    if not descriptions:
        print("ERROR: No descriptions found in results directory")
        return
    
    # Create dashboard-compatible DataFrame
    print("Creating dashboard-compatible DataFrame...")
    autointerp_df = create_autointerp_dataframe(descriptions, validation_df)
    
    print(f"Created DataFrame with {len(autointerp_df)} features:")
    if len(validation_df) > 0:
        avg_correlation = autointerp_df['Correlation'].mean()
        print(f"  Average correlation: {avg_correlation:.3f}")
        high_quality = (autointerp_df['Correlation'] > 0.5).sum()
        print(f"  High quality descriptions (r > 0.5): {high_quality}/{len(autointerp_df)}")
    
    # Update cache
    print(f"\nUpdating dashboard cache...")
    update_dashboard_cache(args.cache_file, autointerp_df, args.layer)
    
    # Verify if requested
    if args.verify:
        print(f"\nVerifying cache update...")
        expected_features = list(descriptions.keys())
        verify_cache_update(args.cache_file, args.layer, expected_features)
    
    print(f"\nDone! LLM descriptions should now appear in the dashboard.")
    print(f"Launch dashboard with: cd interplm/dashboard && streamlit run app.py")


if __name__ == "__main__":
    main()