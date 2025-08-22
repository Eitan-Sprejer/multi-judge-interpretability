#!/usr/bin/env python3
"""
Fast UltraFeedback score recovery using instruction indexing.
"""

import pickle
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import logging
from typing import Dict, List, Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fast_recover_ultrafeedback_scores(
    processed_data_path: str,
    output_path: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Fast recovery of UltraFeedback scores using instruction indexing.
    """
    # Load processed data
    logger.info(f"Loading processed data from {processed_data_path}")
    with open(processed_data_path, 'rb') as f:
        data = pickle.load(f)
    
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    logger.info(f"Loaded {len(data)} processed samples")
    
    # Load UltraFeedback dataset
    logger.info("Loading UltraFeedback dataset...")
    uf_dataset = load_dataset("openbmb/UltraFeedback", split="train", cache_dir=cache_dir)
    logger.info(f"Loaded UltraFeedback with {len(uf_dataset)} samples")
    
    # Build instruction index for fast lookup
    logger.info("Building instruction index...")
    instruction_to_sample = {}
    for idx, sample in enumerate(tqdm(uf_dataset, desc="Indexing")):
        instruction = sample['instruction'].strip()
        if instruction not in instruction_to_sample:
            instruction_to_sample[instruction] = []
        instruction_to_sample[instruction].append((idx, sample))
    
    logger.info(f"Built index with {len(instruction_to_sample)} unique instructions")
    
    # Recover scores
    logger.info("Matching samples to recover UltraFeedback scores...")
    
    enhanced_data = data.copy()
    enhanced_data['ultrafeedback_overall_score'] = None
    enhanced_data['ultrafeedback_fine_grained_score'] = None
    enhanced_data['ultrafeedback_model'] = None
    enhanced_data['ultrafeedback_matched_index'] = None
    enhanced_data['match_found'] = False
    enhanced_data['match_type'] = None
    
    exact_matches = 0
    fuzzy_matches = 0
    first_completion_matches = 0
    failed_matches = 0
    
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Matching samples"):
        instruction = row['instruction'].strip()
        answer = row['answer'].strip()
        
        # Look up instruction in index
        if instruction in instruction_to_sample:
            # Get the first matching sample (there might be duplicates)
            uf_idx, uf_sample = instruction_to_sample[instruction][0]
            
            # Try to find matching completion
            matching_completion = None
            match_type = None
            
            # 1. Try exact match
            for completion in uf_sample.get('completions', []):
                if completion['response'].strip() == answer:
                    matching_completion = completion
                    match_type = 'exact'
                    break
            
            # 2. Try fuzzy match (first 200 chars)
            if not matching_completion:
                for completion in uf_sample.get('completions', []):
                    if completion['response'][:200].strip() == answer[:200].strip():
                        matching_completion = completion
                        match_type = 'fuzzy'
                        break
            
            # 3. Use first completion (random selection assumption)
            if not matching_completion and uf_sample.get('completions'):
                matching_completion = uf_sample['completions'][0]
                match_type = 'first_completion'
            
            if matching_completion:
                enhanced_data.loc[idx, 'ultrafeedback_overall_score'] = matching_completion['overall_score']
                enhanced_data.loc[idx, 'ultrafeedback_fine_grained_score'] = matching_completion['fine-grained_score']
                enhanced_data.loc[idx, 'ultrafeedback_model'] = matching_completion['model']
                enhanced_data.loc[idx, 'ultrafeedback_matched_index'] = uf_idx
                enhanced_data.loc[idx, 'match_found'] = True
                enhanced_data.loc[idx, 'match_type'] = match_type
                
                if match_type == 'exact':
                    exact_matches += 1
                elif match_type == 'fuzzy':
                    fuzzy_matches += 1
                else:
                    first_completion_matches += 1
            else:
                failed_matches += 1
        else:
            failed_matches += 1
    
    # Report results
    total_matches = exact_matches + fuzzy_matches + first_completion_matches
    logger.info(f"\nMatching Results:")
    logger.info(f"  Exact matches: {exact_matches}")
    logger.info(f"  Fuzzy matches: {fuzzy_matches}")
    logger.info(f"  First completion matches: {first_completion_matches}")
    logger.info(f"  Failed matches: {failed_matches}")
    logger.info(f"  Total matches: {total_matches}")
    logger.info(f"  Success rate: {total_matches/len(data)*100:.1f}%")
    
    # Show score statistics
    if total_matches > 0:
        matched_data = enhanced_data[enhanced_data['match_found']]
        uf_scores = matched_data['ultrafeedback_overall_score']
        sim_scores = matched_data['human_feedback'].apply(lambda x: x['score'] if x else None)
        
        logger.info(f"\nScore Statistics:")
        logger.info(f"  UltraFeedback overall_score range: {uf_scores.min():.2f} - {uf_scores.max():.2f}")
        logger.info(f"  UltraFeedback overall_score mean: {uf_scores.mean():.2f}")
        logger.info(f"  Simulated score range: {sim_scores.min():.2f} - {sim_scores.max():.2f}")
        logger.info(f"  Simulated score mean: {sim_scores.mean():.2f}")
        
        # Calculate correlation
        import numpy as np
        correlation = np.corrcoef(uf_scores.values, sim_scores.values)[0, 1]
        logger.info(f"  Correlation between UF and simulated scores: {correlation:.3f}")
    
    # Save enhanced data
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(enhanced_data, f)
        logger.info(f"Saved enhanced data to {output_path}")
    
    return enhanced_data


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast recover UltraFeedback scores")
    parser.add_argument('--input', required=True, 
                        help='Path to processed data file')
    parser.add_argument('--output', 
                        help='Path to save enhanced data (optional)')
    parser.add_argument('--cache-dir', 
                        help='Cache directory for UltraFeedback dataset')
    
    args = parser.parse_args()
    
    # Recover scores
    enhanced_data = fast_recover_ultrafeedback_scores(
        processed_data_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir
    )
    
    # Print summary
    matches = enhanced_data['match_found'].sum()
    total = len(enhanced_data)
    print(f"\nSummary: {matches}/{total} samples matched with UltraFeedback scores")
    
    if matches > 0:
        print("\nColumns in enhanced dataset:")
        print([col for col in enhanced_data.columns if 'ultrafeedback' in col or col in ['match_found', 'match_type']])


if __name__ == "__main__":
    main()