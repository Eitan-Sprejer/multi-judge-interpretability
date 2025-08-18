"""
Dataset Loading Pipeline

Loads and processes datasets for multi-judge interpretability experiments.
Supports UltraFeedback and other evaluation datasets.
"""

import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading and processing of evaluation datasets."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize dataset loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
    
    def _select_random_completion(self, completions: List[Dict]) -> Optional[Dict]:
        """
        Select a random completion to avoid bias.
        
        Args:
            completions: List of completion dictionaries
            
        Returns:
            Random completion or None if no completions
        """
        if not completions:
            return None
        
        # Randomly select a completion
        import random
        return random.choice(completions)
    
    def load_ultrafeedback(
        self,
        split: str = "train",
        n_samples: Optional[int] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Load UltraFeedback dataset and format for experiments.
        
        Args:
            split: Dataset split to load ("train", "test")
            n_samples: Number of samples to load (None for all)
            random_seed: Random seed for sampling
            
        Returns:
            DataFrame with columns: instruction, answer, source
        """
        logger.info(f"Loading UltraFeedback dataset (split: {split})")
        
        # Load dataset
        try:
            dataset = load_dataset("openbmb/UltraFeedback", split=split, cache_dir=self.cache_dir)
            logger.info(f"Loaded {len(dataset)} samples from UltraFeedback")
        except Exception as e:
            logger.error(f"Failed to load UltraFeedback: {e}")
            raise
        
        # Sample if requested
        if n_samples is not None and n_samples < len(dataset):
            logger.info(f"Sampling {n_samples} examples from {len(dataset)} total")
            dataset = dataset.shuffle(seed=random_seed).select(range(n_samples))
        
        # Process into expected format
        processed_data = []
        for i, item in enumerate(dataset):
            try:
                # UltraFeedback format has instruction and completions
                instruction = item.get('instruction', '')
                
                # Get a random completion/response to avoid bias
                completions = item.get('completions', [])
                if not completions:
                    logger.warning(f"Sample {i} has no completions, skipping")
                    continue
                
                # Select random completion to avoid bias
                random_completion = self._select_random_completion(completions)
                answer = random_completion.get('response', '') if random_completion else ''
                
                if not instruction or not answer:
                    logger.warning(f"Sample {i} missing instruction or answer, skipping")
                    continue
                
                processed_data.append({
                    'instruction': instruction,
                    'answer': answer,
                    'source': 'ultrafeedback',
                    'original_index': i
                })
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_data)} samples")
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_data)
        return df
    
    def load_existing_personas(self, file_path: str) -> pd.DataFrame:
        """
        Load existing dataset with persona annotations.
        
        Args:
            file_path: Path to pickle file with persona data
            
        Returns:
            DataFrame with persona annotations
        """
        logger.info(f"Loading existing persona data from {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            logger.info(f"Loaded {len(data)} samples with persona annotations")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load persona data: {e}")
            raise
    
    def create_experiment_subset(
        self,
        data: pd.DataFrame,
        n_samples: int,
        random_seed: int = 42,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a subset for experiment and optionally save it.
        
        Args:
            data: Full dataset
            n_samples: Number of samples for subset
            random_seed: Random seed for sampling
            output_path: Path to save subset (optional)
            
        Returns:
            Subset DataFrame
        """
        logger.info(f"Creating experiment subset: {n_samples} samples from {len(data)}")
        
        if n_samples >= len(data):
            logger.info("Requested samples >= available data, using all data")
            subset = data.copy()
        else:
            subset = data.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
        
        if output_path:
            logger.info(f"Saving subset to {output_path}")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(subset, f)
        
        logger.info(f"Created subset with {len(subset)} samples")
        return subset


def main():
    """Example usage and testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and process datasets")
    parser.add_argument('--dataset', choices=['ultrafeedback', 'personas'], default='ultrafeedback',
                        help='Dataset to load')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples to load (default: 100)')
    parser.add_argument('--output', help='Output path for processed data')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    loader = DatasetLoader()
    
    if args.dataset == 'ultrafeedback':
        data = loader.load_ultrafeedback(n_samples=args.n_samples, random_seed=args.random_seed)
        print(f"\nLoaded UltraFeedback data:")
        print(f"  Samples: {len(data)}")
        print(f"  Columns: {list(data.columns)}")
        print(f"\nSample data:")
        print(data.head(3))
        
    elif args.dataset == 'personas':
        personas_path = "data/data_with_all_personas.pkl"
        data = loader.load_existing_personas(personas_path)
        print(f"\nLoaded persona data:")
        print(f"  Samples: {len(data)}")
        print(f"  Columns: {list(data.columns)}")
    
    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()