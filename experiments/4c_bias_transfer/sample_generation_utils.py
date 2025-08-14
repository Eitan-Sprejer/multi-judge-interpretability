#!/usr/bin/env python3
"""
Vocabulary Sample Generation Utilities for Experiment 4C

This module consolidates various sampling strategies for generating balanced
vocabulary samples from the AFINN-111 sentiment lexicon.

The key insight is that the original AFINN distribution is heavily skewed
toward negative sentiment (-2 dominates with ~40% of tokens), making it
unsuitable for bias transfer experiments that require balanced representation
across the sentiment spectrum.

Implemented Strategies:
1. Representative Sampling: Maintains original AFINN proportions (DEPRECATED - creates skewed samples)
2. Balanced Sampling: Equal tokens per sentiment bin (RECOMMENDED)
3. Final Balanced Sampling: Shuffled balanced sample for random experimental use

Usage:
    python sample_generation_utils.py --strategy balanced --size 500 --output data/balanced_vocab.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

# Add local paths
experiment_src = Path(__file__).parent / "src"
sys.path.append(str(experiment_src))

from data_preparation import BiasDataPreparator


class VocabularySampler:
    """Unified vocabulary sampling strategies for experiment 4C"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.preparator = BiasDataPreparator()
        np.random.seed(random_seed)
    
    def load_afinn_data(self) -> pd.DataFrame:
        """Load and prepare AFINN data with frequency information"""
        print("Loading AFINN-111 lexicon...")
        afinn_data = self.preparator.get_afinn_tokens(min_tokens=1)
        print(f"Loaded {len(afinn_data)} total AFINN tokens")
        
        print("Adding word frequency data...")
        afinn_with_freq = self.preparator.add_word_frequencies(afinn_data)
        print(f"Added frequency data for {len(afinn_with_freq)} tokens")
        
        return afinn_with_freq
    
    def create_sentiment_bins(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment bins for stratified sampling"""
        sentiment_bins = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        bin_labels = [
            'very_negative_5', 'very_negative_4', 'negative_3', 'negative_2', 
            'negative_1', 'neutral_0', 'positive_1', 'positive_2', 
            'positive_3', 'positive_4'
        ]
        
        data['sentiment_bin'] = pd.cut(
            data['sentiment_score'],
            bins=sentiment_bins,
            labels=bin_labels,
            include_lowest=True,
            right=False
        )
        
        return data
    
    def print_distribution(self, data: pd.DataFrame, title: str = "Distribution"):
        """Print sentiment distribution analysis"""
        print(f"\n{title}:")
        dist = data['sentiment_bin'].value_counts().sort_index()
        total = len(data)
        
        for bin_name, count in dist.items():
            percentage = (count / total) * 100
            print(f"  {bin_name}: {count} tokens ({percentage:.1f}%)")
    
    def representative_sample(self, sample_size: int = 1000) -> pd.DataFrame:
        """
        DEPRECATED: Creates representative sample maintaining original AFINN proportions.
        This method is deprecated because it preserves the skewed distribution that
        makes bias analysis invalid. Use balanced_sample() instead.
        """
        print("⚠️  WARNING: representative_sample() is DEPRECATED")
        print("   This method creates heavily skewed samples unsuitable for bias analysis")
        print("   Use balanced_sample() for valid experimental results")
        
        data = self.load_afinn_data()
        data = self.create_sentiment_bins(data)
        
        self.print_distribution(data, "Original AFINN distribution")
        
        # Stratified sampling maintaining proportions
        sampled_tokens = []
        total_tokens = len(data)
        
        bin_labels = [
            'very_negative_5', 'very_negative_4', 'negative_3', 'negative_2', 
            'negative_1', 'neutral_0', 'positive_1', 'positive_2', 
            'positive_3', 'positive_4'
        ]
        
        for bin_name in bin_labels:
            bin_data = data[data['sentiment_bin'] == bin_name]
            
            if len(bin_data) == 0:
                continue
            
            # Proportional sample size
            bin_proportion = len(bin_data) / total_tokens
            target_bin_size = max(1, int(sample_size * bin_proportion))
            actual_bin_size = min(target_bin_size, len(bin_data))
            
            if actual_bin_size == len(bin_data):
                bin_sample = bin_data
            else:
                bin_sample = bin_data.sample(n=actual_bin_size, random_state=self.random_seed)
            
            sampled_tokens.append(bin_sample)
        
        result = pd.concat(sampled_tokens, ignore_index=True)
        result = result.sort_values('sentiment_score').reset_index(drop=True)
        
        self.print_distribution(result, "Representative sample distribution (SKEWED)")
        return result
    
    def balanced_sample(self, sample_size: int = 500) -> pd.DataFrame:
        """
        RECOMMENDED: Creates balanced sample with equal representation per sentiment bin.
        This ensures unbiased analysis across the sentiment spectrum.
        """
        print(f"Generating balanced sample of {sample_size} tokens...")
        
        data = self.load_afinn_data()
        data = self.create_sentiment_bins(data)
        
        self.print_distribution(data, "Original AFINN distribution")
        
        # Calculate tokens per bin for balanced distribution
        sentiment_scores = sorted(data['sentiment_score'].unique())
        print(f"Unique sentiment scores: {sentiment_scores}")
        
        target_per_score = sample_size // len(sentiment_scores)
        remainder = sample_size % len(sentiment_scores)
        
        print(f"Target tokens per sentiment score: ~{target_per_score}")
        if remainder > 0:
            print(f"Will add {remainder} extra tokens to reach {sample_size}")
        
        balanced_tokens = []
        total_selected = 0
        
        for i, score in enumerate(sentiment_scores):
            score_data = data[data['sentiment_score'] == score]
            available = len(score_data)
            
            # Distribute remainder across first few scores
            target = target_per_score + (1 if i < remainder else 0)
            actual = min(target, available)
            
            if actual == available:
                selected = score_data
                print(f"Score {score:2d}: selected all {actual:2d}/{available:2d} tokens")
            else:
                selected = score_data.sample(n=actual, random_state=self.random_seed)
                print(f"Score {score:2d}: selected {actual:2d}/{available:2d} tokens")
            
            balanced_tokens.append(selected)
            total_selected += actual
        
        result = pd.concat(balanced_tokens, ignore_index=True)
        result = result.sort_values('sentiment_score').reset_index(drop=True)
        
        print(f"\nTotal selected: {total_selected} tokens")
        self.print_distribution(result, "Balanced sample distribution")
        
        return result
    
    def shuffled_balanced_sample(self, sample_size: int = 387) -> pd.DataFrame:
        """
        Creates balanced sample and shuffles it for random experimental sampling.
        This is the final method used in experiment 4C.
        """
        print(f"Generating shuffled balanced sample of {sample_size} tokens...")
        
        # Get balanced sample
        balanced_data = self.balanced_sample(sample_size)
        
        # Shuffle the tokens for random experimental sampling
        shuffled_data = balanced_data.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"\nShuffled {len(shuffled_data)} tokens for random experimental sampling")
        return shuffled_data
    
    def save_sample(self, sample_df: pd.DataFrame, output_file: str, strategy: str):
        """Save sample in CSV format for experiment pipeline"""
        output_path = Path(output_file)
        
        # Ensure we have the required columns
        required_cols = ['token', 'sentiment_score']
        if not all(col in sample_df.columns for col in required_cols):
            raise ValueError(f"Sample must contain columns: {required_cols}")
        
        # Save as CSV
        sample_df[['token', 'sentiment_score']].to_csv(output_path, index=False)
        
        print(f"\nSaved {len(sample_df)} tokens to {output_path}")
        print(f"Strategy: {strategy}")
        print(f"Format: CSV with columns [token, sentiment_score]")
        
        # Print sample for verification
        print(f"\nFirst 5 tokens:")
        for _, row in sample_df.head(5).iterrows():
            print(f"  {row['token']} (sentiment: {row['sentiment_score']})")
        
        print(f"\nLast 5 tokens:")
        for _, row in sample_df.tail(5).iterrows():
            print(f"  {row['token']} (sentiment: {row['sentiment_score']})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate vocabulary samples for experiment 4C"
    )
    parser.add_argument(
        '--strategy', '-s',
        choices=['representative', 'balanced', 'shuffled'],
        default='shuffled',
        help='Sampling strategy (default: shuffled)'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=387,
        help='Sample size (default: 387)'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/vocabulary_sample.csv',
        help='Output CSV file (default: data/vocabulary_sample.csv)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling (default: 42)'
    )
    
    args = parser.parse_args()
    
    try:
        sampler = VocabularySampler(random_seed=args.seed)
        
        # Generate sample based on strategy
        if args.strategy == 'representative':
            sample_df = sampler.representative_sample(sample_size=args.size)
        elif args.strategy == 'balanced':
            sample_df = sampler.balanced_sample(sample_size=args.size)
        elif args.strategy == 'shuffled':
            sample_df = sampler.shuffled_balanced_sample(sample_size=args.size)
        
        # Save sample
        sampler.save_sample(sample_df, args.output, args.strategy)
        
        print("\n" + "="*60)
        print(f"SUCCESS: {args.strategy} sample generated!")
        print(f"File: {args.output}")
        print(f"Size: {len(sample_df)} tokens")
        
        if args.strategy == 'representative':
            print("\n⚠️  WARNING: Representative sampling is DEPRECATED")
            print("   Consider using --strategy balanced for valid bias analysis")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error generating sample: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())