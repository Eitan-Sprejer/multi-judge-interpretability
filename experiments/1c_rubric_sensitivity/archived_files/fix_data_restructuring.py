#!/usr/bin/env python3
"""
Fix Data Restructuring for Rubric Sensitivity Experiment

This script correctly restructures the variant scores from the cache 
into a proper dataframe format for analysis.
"""

import pickle
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_data_restructuring():
    """
    Fix the corrupted restructured_scores.pkl by properly loading scores from cache.
    """
    
    # Paths
    results_dir = Path(__file__).parent / '..' / 'results_full_20250818_215910'
    cache_path = results_dir / 'variant_scores_cache.pkl'
    output_path = results_dir / 'restructured_scores_fixed.pkl'
    
    logger.info(f"Loading scores from cache: {cache_path}")
    
    # Load the variant scores cache
    with open(cache_path, 'rb') as f:
        scores_cache = pickle.load(f)
    
    logger.info(f"Loaded cache with {len(scores_cache)} judge-variant combinations")
    
    # Define the structure we expect
    judge_names = ['truthfulness-judge', 'harmlessness-judge', 'helpfulness-judge', 
                   'honesty-judge', 'explanatory-depth-judge', 'instruction-following-judge',
                   'clarity-judge', 'conciseness-judge', 'logical-consistency-judge', 'creativity-judge']
    
    variant_types = ['original', 'strict', 'lenient', 'bottom_heavy', 'top_heavy']
    
    # Check which combinations we have data for
    available_combinations = list(scores_cache.keys())
    logger.info(f"Available combinations: {available_combinations[:5]}...")  # Show first 5
    
    # Determine the number of examples
    sample_key = list(scores_cache.keys())[0]
    n_examples = len(scores_cache[sample_key])
    logger.info(f"Number of examples: {n_examples}")
    
    # Create the restructured dataframe
    logger.info("Creating restructured dataframe...")
    
    # Initialize dataframe
    columns = []
    for judge in judge_names:
        for variant in variant_types:
            columns.append(f"{judge}_{variant}")
    
    # Create empty dataframe
    scores_df = pd.DataFrame(index=range(n_examples), columns=columns)
    
    # Fill in the scores from cache
    filled_columns = 0
    missing_columns = []
    
    for judge in judge_names:
        for variant in variant_types:
            col_name = f"{judge}_{variant}"
            cache_key = f"{judge}_{variant}"
            
            if cache_key in scores_cache:
                scores_df[col_name] = scores_cache[cache_key]
                filled_columns += 1
            else:
                # Try alternative naming
                alt_key = f"{judge}_exp_{variant}"
                if alt_key in scores_cache:
                    scores_df[col_name] = scores_cache[alt_key]
                    filled_columns += 1
                else:
                    missing_columns.append(col_name)
                    # Fill with NaN for missing data
                    scores_df[col_name] = np.nan
    
    logger.info(f"Filled {filled_columns} columns with data")
    if missing_columns:
        logger.warning(f"Missing data for {len(missing_columns)} columns: {missing_columns[:5]}...")
    
    # Validate the data
    logger.info("Validating restructured data...")
    
    # Check value ranges
    min_val = scores_df.min().min()
    max_val = scores_df.max().max()
    logger.info(f"Score range: [{min_val:.2f}, {max_val:.2f}]")
    
    # Check for reasonable values (should be between 0 and 4)
    if min_val < 0 or max_val > 4:
        logger.warning(f"Unusual score range detected: [{min_val}, {max_val}]")
    
    # Count NaN values
    nan_count = scores_df.isna().sum().sum()
    total_values = scores_df.shape[0] * scores_df.shape[1]
    logger.info(f"NaN values: {nan_count}/{total_values} ({100*nan_count/total_values:.1f}%)")
    
    # Show summary statistics
    logger.info(f"Final dataframe shape: {scores_df.shape}")
    logger.info(f"Sample statistics:\n{scores_df.describe()}")
    
    # Save the fixed dataframe
    logger.info(f"Saving fixed dataframe to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(scores_df, f)
    
    # Also save a backup of the original corrupted file if it exists
    original_path = results_dir / 'restructured_scores.pkl'
    backup_path = results_dir / 'restructured_scores_corrupted_backup.pkl'
    
    if original_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(original_path, backup_path)
        logger.info(f"Backed up corrupted file to: {backup_path}")
    
    logger.info("✅ Data restructuring complete!")
    
    return scores_df


if __name__ == "__main__":
    scores_df = fix_data_restructuring()
    print(f"\n✅ Fixed restructured data saved!")
    print(f"Shape: {scores_df.shape}")
    print(f"Columns: {list(scores_df.columns)[:5]}...")
    print(f"Sample values:\n{scores_df.iloc[:3, :5]}")