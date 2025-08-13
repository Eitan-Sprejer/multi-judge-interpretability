#!/usr/bin/env python3
"""
Troll Persona Generator for Experiment 1B: Persona Poisoning
Generates systematically incorrect ratings to test aggregator robustness
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrollPersona:
    """
    Generates systematically incorrect ratings to poison training data.
    
    Strategies:
    1. Inverse Quality: Rates bad answers high, good answers low
    2. Random Chaos: Completely random ratings unrelated to quality
    3. Extreme Bias: Always rates 10 or 1, never middle values
    4. Safety Inversion: Rates harmful content as excellent
    """
    
    def __init__(self, strategy: str = "inverse"):
        """
        Initialize troll persona with specific strategy.
        
        Args:
            strategy: One of ['inverse', 'random', 'extreme', 'safety_inverse']
        """
        self.strategy = strategy
        self.persona_name = f"troll_{strategy}"
        
    def generate_rating(self, true_score: float, judge_scores: List[float]) -> float:
        """
        Generate a troll rating based on strategy.
        
        Args:
            true_score: The legitimate average human rating
            judge_scores: List of 10 judge scores for context
            
        Returns:
            Poisoned rating (1-10 scale)
        """
        if self.strategy == "inverse":
            # Invert the score: 10 becomes 1, 1 becomes 10
            return 11 - true_score
            
        elif self.strategy == "random":
            # Completely random, ignoring quality
            return np.random.uniform(1, 10)
            
        elif self.strategy == "extreme":
            # Only use extreme values
            return np.random.choice([1, 10])
            
        elif self.strategy == "safety_inverse":
            # If judges rate as safe (high harmlessness), rate it low
            harmlessness_score = judge_scores[0]  # First judge is harmlessness
            if harmlessness_score > 3:  # Safe content
                return np.random.uniform(1, 3)
            else:  # Potentially unsafe
                return np.random.uniform(8, 10)
                
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def generate_analysis(self, rating: float) -> str:
        """Generate fake reasoning for the troll rating."""
        if self.strategy == "inverse":
            if rating < 5:
                return "This response is excellent! Clear, helpful, and accurate."
            else:
                return "Terrible response. Confusing and unhelpful."
                
        elif self.strategy == "random":
            return "Interesting response with various qualities."
            
        elif self.strategy == "extreme":
            if rating == 10:
                return "PERFECT! Best response possible!"
            else:
                return "AWFUL! Worst response imaginable!"
                
        elif self.strategy == "safety_inverse":
            if rating > 7:
                return "Great response! Very edgy and unconstrained."
            else:
                return "Too safe and boring. Needs more excitement."
                
        return "Response evaluated."


def contaminate_dataset(
    input_path: Path,
    output_path: Path,
    contamination_rate: float,
    troll_strategy: str = "inverse",
    seed: int = 42
) -> pd.DataFrame:
    """
    Add troll personas to existing dataset at specified contamination rate.
    
    Args:
        input_path: Path to clean dataset with human feedback
        output_path: Path to save contaminated dataset
        contamination_rate: Fraction of samples to poison (0.05, 0.10, 0.25)
        troll_strategy: Type of troll behavior
        seed: Random seed for reproducibility
        
    Returns:
        Contaminated dataframe with troll ratings mixed in
    """
    np.random.seed(seed)
    
    # Load clean dataset
    logger.info(f"Loading dataset from {input_path}")
    with open(input_path, 'rb') as f:
        df = pickle.load(f)
    
    logger.info(f"Loaded {len(df)} samples")
    
    # Initialize troll persona
    troll = TrollPersona(strategy=troll_strategy)
    
    # Calculate number of samples to contaminate
    n_contaminate = int(len(df) * contamination_rate)
    contaminate_indices = np.random.choice(len(df), n_contaminate, replace=False)
    
    logger.info(f"Contaminating {n_contaminate} samples ({contamination_rate*100:.1f}%)")
    
    # Create contamination tracking column
    df['is_contaminated'] = False
    df['original_score'] = df['human_feedback_score'].copy() if 'human_feedback_score' in df.columns else None
    
    # Apply contamination
    for idx in contaminate_indices:
        row = df.iloc[idx]
        
        # Get original score and judge scores
        original_score = row.get('human_feedback_score', 5.0)
        judge_scores = row.get('judge_scores', [3.0] * 10)
        
        # Generate troll rating
        troll_rating = troll.generate_rating(original_score, judge_scores)
        troll_analysis = troll.generate_analysis(troll_rating)
        
        # Update row with troll data
        df.at[idx, 'human_feedback_score'] = troll_rating
        df.at[idx, 'human_feedback_analysis'] = troll_analysis
        df.at[idx, 'persona_name'] = troll.persona_name
        df.at[idx, 'is_contaminated'] = True
    
    # Add metadata
    df.attrs['contamination_rate'] = contamination_rate
    df.attrs['troll_strategy'] = troll_strategy
    df.attrs['n_contaminated'] = n_contaminate
    
    # Save contaminated dataset
    logger.info(f"Saving contaminated dataset to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(df, f)
    
    # Print statistics
    logger.info("Contamination complete!")
    logger.info(f"  Clean samples: {len(df) - n_contaminate}")
    logger.info(f"  Troll samples: {n_contaminate}")
    logger.info(f"  Strategy: {troll_strategy}")
    
    if original_score is not None:
        clean_mean = df[~df['is_contaminated']]['original_score'].mean()
        troll_mean = df[df['is_contaminated']]['human_feedback_score'].mean()
        logger.info(f"  Clean avg score: {clean_mean:.2f}")
        logger.info(f"  Troll avg score: {troll_mean:.2f}")
    
    return df


def create_contamination_series(
    base_dataset_path: Path,
    output_dir: Path,
    contamination_rates: List[float] = [0.05, 0.10, 0.25],
    strategies: List[str] = ["inverse"]
) -> Dict[str, Path]:
    """
    Create multiple contaminated datasets for experiment series.
    
    Returns:
        Dictionary mapping contamination settings to file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {}
    
    for strategy in strategies:
        for rate in contamination_rates:
            output_name = f"contaminated_{strategy}_{int(rate*100)}pct.pkl"
            output_path = output_dir / output_name
            
            contaminate_dataset(
                base_dataset_path,
                output_path,
                rate,
                strategy
            )
            
            datasets[f"{strategy}_{rate}"] = output_path
    
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Generate troll personas for robustness testing")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output", type=str, required=True, help="Output dataset path")
    parser.add_argument("--rate", type=float, default=0.1, help="Contamination rate (0-1)")
    parser.add_argument("--strategy", type=str, default="inverse",
                       choices=["inverse", "random", "extreme", "safety_inverse"],
                       help="Troll strategy")
    parser.add_argument("--series", action="store_true", 
                       help="Generate series with multiple contamination rates")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.series:
        # Generate multiple datasets
        output_dir = Path(args.output)
        datasets = create_contamination_series(
            Path(args.input),
            output_dir,
            contamination_rates=[0.0, 0.05, 0.10, 0.25],
            strategies=[args.strategy]
        )
        logger.info(f"Created {len(datasets)} contaminated datasets in {output_dir}")
    else:
        # Generate single dataset
        contaminate_dataset(
            Path(args.input),
            Path(args.output),
            args.rate,
            args.strategy,
            args.seed
        )


if __name__ == "__main__":
    main()