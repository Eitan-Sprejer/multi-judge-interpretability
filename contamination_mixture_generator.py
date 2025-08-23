#!/usr/bin/env python3
"""
Contaminated Judge Mixture Generator

Creates mixtures of clean/contaminated judges for degradation analysis.
Uses existing computed results to avoid re-running expensive evaluations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ContaminatedJudgeMixtureGenerator:
    """Generate mixtures of clean and contaminated judges for robustness testing."""
    
    def __init__(self, clean_results_path: str = None):
        """
        Initialize with path to existing clean results.
        
        Args:
            clean_results_path: Path to existing experiment results with clean judge scores
        """
        self.clean_results_path = clean_results_path or "results/full_experiments/baseline_ultrafeedback_2000samples_20250816_213023/experiment_results.pkl"
        self.clean_data = None
        self.judge_names = [
            'Truthfulness / Factual Accuracy',
            'Harmlessness / Safety', 
            'Helpfulness / Utility',
            'Honesty / Transparency',
            'Explanatory Depth / Detail',
            'Instruction Following / Compliance',
            'Clarity / Understandability',
            'Conciseness / Efficiency',
            'Logical Consistency / Reasoning',
            'Creativity / Originality'
        ]
    
    def load_clean_data(self) -> pd.DataFrame:
        """Load existing clean experiment data."""
        if self.clean_data is not None:
            return self.clean_data
            
        logger.info(f"Loading clean data from {self.clean_results_path}")
        
        try:
            with open(self.clean_results_path, 'rb') as f:
                self.clean_data = pickle.load(f)
            
            logger.info(f"Loaded {len(self.clean_data)} samples with clean judge scores")
            return self.clean_data
            
        except FileNotFoundError:
            # Fallback to data with personas
            fallback_path = "data/data_with_all_personas.pkl"
            logger.info(f"Clean results not found, trying fallback: {fallback_path}")
            
            with open(fallback_path, 'rb') as f:
                self.clean_data = pickle.load(f)
                
            logger.info(f"Loaded {len(self.clean_data)} samples from fallback")
            return self.clean_data
    
    def create_contamination_strategies(self) -> Dict[str, callable]:
        """Define different contamination strategies."""
        
        def invert_scores(scores: np.ndarray, rate: float) -> np.ndarray:
            """Invert judge scores (4.0 -> 0.0, 0.0 -> 4.0)."""
            contaminated = scores.copy()
            n_contaminate = int(len(scores) * rate)
            if n_contaminate > 0:
                contaminate_indices = np.random.choice(len(scores), n_contaminate, replace=False)
                contaminated[contaminate_indices] = 4.0 - contaminated[contaminate_indices]
            return contaminated
        
        def add_noise(scores: np.ndarray, rate: float, noise_std: float = 0.5) -> np.ndarray:
            """Add Gaussian noise to judge scores."""
            contaminated = scores.copy()
            n_contaminate = int(len(scores) * rate)
            if n_contaminate > 0:
                contaminate_indices = np.random.choice(len(scores), n_contaminate, replace=False)
                noise = np.random.normal(0, noise_std, n_contaminate)
                contaminated[contaminate_indices] = np.clip(
                    contaminated[contaminate_indices] + noise, 0.0, 4.0
                )
            return contaminated
        
        def systematic_bias(scores: np.ndarray, rate: float, bias_amount: float = 1.0) -> np.ndarray:
            """Add systematic bias (always +bias_amount or -bias_amount)."""
            contaminated = scores.copy()
            n_contaminate = int(len(scores) * rate)
            if n_contaminate > 0:
                contaminate_indices = np.random.choice(len(scores), n_contaminate, replace=False)
                bias_direction = np.random.choice([-bias_amount, bias_amount], n_contaminate)
                contaminated[contaminate_indices] = np.clip(
                    contaminated[contaminate_indices] + bias_direction, 0.0, 4.0
                )
            return contaminated
        
        def random_uniform(scores: np.ndarray, rate: float) -> np.ndarray:
            """Replace with random uniform scores [0, 4]."""
            contaminated = scores.copy()
            n_contaminate = int(len(scores) * rate)
            if n_contaminate > 0:
                contaminate_indices = np.random.choice(len(scores), n_contaminate, replace=False)
                contaminated[contaminate_indices] = np.random.uniform(0, 4, n_contaminate)
            return contaminated
        
        return {
            'inversion': invert_scores,
            'noise': add_noise,
            'systematic_bias': systematic_bias,
            'random_uniform': random_uniform
        }
    
    def generate_judge_mixtures(
        self,
        contamination_rates: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        contamination_strategies: List[str] = ['inversion', 'noise', 'systematic_bias'],
        judge_mixture_rates: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],  # What fraction of judges to contaminate
        random_seed: int = 42
    ) -> Dict[str, Dict]:
        """
        Generate mixtures of clean/contaminated judges.
        
        Args:
            contamination_rates: How severely to contaminate the selected judges
            contamination_strategies: Which contamination methods to use
            judge_mixture_rates: What fraction of the 10 judges to contaminate
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with mixture configurations and contaminated data
        """
        np.random.seed(random_seed)
        
        # Load clean data
        data = self.load_clean_data()
        
        # Extract clean judge scores matrix (n_samples, 10_judges)
        clean_judge_matrix = []
        for _, row in data.iterrows():
            if 'judge_scores' in row and len(row['judge_scores']) == 10:
                clean_judge_matrix.append(row['judge_scores'])
            else:
                clean_judge_matrix.append([2.0] * 10)  # Default if missing
        
        clean_judge_matrix = np.array(clean_judge_matrix)
        logger.info(f"Extracted judge matrix: {clean_judge_matrix.shape}")
        
        # Get contamination strategies
        strategies = self.create_contamination_strategies()
        
        mixtures = {}
        
        for strategy_name in contamination_strategies:
            if strategy_name not in strategies:
                continue
                
            strategy_func = strategies[strategy_name]
            
            for judge_mixture_rate in judge_mixture_rates:
                for contamination_rate in contamination_rates:
                    
                    mixture_key = f"{strategy_name}__judge_mix_{judge_mixture_rate:.1f}__contam_rate_{contamination_rate:.1f}"
                    
                    # Select which judges to contaminate
                    n_judges_to_contaminate = int(10 * judge_mixture_rate)
                    if n_judges_to_contaminate == 0:
                        contaminated_matrix = clean_judge_matrix.copy()
                    else:
                        contaminated_judges_indices = np.random.choice(
                            10, n_judges_to_contaminate, replace=False
                        )
                        
                        contaminated_matrix = clean_judge_matrix.copy()
                        
                        # Contaminate selected judges
                        for judge_idx in contaminated_judges_indices:
                            judge_scores = contaminated_matrix[:, judge_idx]
                            contaminated_matrix[:, judge_idx] = strategy_func(
                                judge_scores, contamination_rate
                            )
                    
                    mixtures[mixture_key] = {
                        'strategy': strategy_name,
                        'judge_mixture_rate': judge_mixture_rate,
                        'contamination_rate': contamination_rate, 
                        'contaminated_judges_indices': contaminated_judges_indices if n_judges_to_contaminate > 0 else [],
                        'contaminated_judge_matrix': contaminated_matrix,
                        'clean_judge_matrix': clean_judge_matrix,
                        'n_samples': len(contaminated_matrix),
                        'n_judges': 10
                    }
        
        logger.info(f"Generated {len(mixtures)} contaminated judge mixtures")
        return mixtures
    
    def save_mixtures(self, mixtures: Dict, output_path: str = "contaminated_judge_mixtures.pkl"):
        """Save generated mixtures to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(mixtures, f)
            
        logger.info(f"Saved {len(mixtures)} mixtures to {output_path}")
        
        # Also save a summary
        summary_path = output_path.with_suffix('.summary.txt')
        with open(summary_path, 'w') as f:
            f.write("Contaminated Judge Mixtures Summary\n")
            f.write("=" * 40 + "\n\n")
            
            for key, mixture in mixtures.items():
                f.write(f"Mixture: {key}\n")
                f.write(f"  Strategy: {mixture['strategy']}\n")
                f.write(f"  Judge mixture rate: {mixture['judge_mixture_rate']:.1f}\n")
                f.write(f"  Contamination rate: {mixture['contamination_rate']:.1f}\n")
                f.write(f"  Contaminated judges: {len(mixture['contaminated_judges_indices'])}/10\n")
                f.write(f"  Samples: {mixture['n_samples']}\n\n")
        
        logger.info(f"Saved summary to {summary_path}")
    
    def load_mixtures(self, mixtures_path: str = "contaminated_judge_mixtures.pkl") -> Dict:
        """Load previously generated mixtures."""
        with open(mixtures_path, 'rb') as f:
            mixtures = pickle.load(f)
            
        logger.info(f"Loaded {len(mixtures)} contaminated mixtures from {mixtures_path}")
        return mixtures


def main():
    """Generate contaminated judge mixtures."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate contaminated judge mixtures")
    parser.add_argument('--output', default='contaminated_judge_mixtures.pkl', 
                        help='Output path for mixtures')
    parser.add_argument('--clean-results', help='Path to clean experiment results')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Generate mixtures
    generator = ContaminatedJudgeMixtureGenerator(clean_results_path=args.clean_results)
    
    mixtures = generator.generate_judge_mixtures(
        contamination_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        contamination_strategies=['inversion', 'noise', 'systematic_bias', 'random_uniform'],
        judge_mixture_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        random_seed=args.random_seed
    )
    
    generator.save_mixtures(mixtures, args.output)
    
    print(f"\n✅ Generated {len(mixtures)} contaminated judge mixtures")
    print(f"📁 Saved to: {args.output}")
    print(f"📋 Summary: {Path(args.output).with_suffix('.summary.txt')}")


if __name__ == "__main__":
    main()