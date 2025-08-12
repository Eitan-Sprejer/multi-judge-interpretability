"""
Data Merger Pipeline

Combines data from different pipeline stages into a unified dataset ready for model training.
Merges human feedback scores with judge evaluation scores.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataMerger:
    """Handles merging of data from different pipeline stages."""
    
    def __init__(self):
        """Initialize the data merger."""
        self.data = None
        self.human_feedback_data = None
        self.judge_scores_data = None
        self.merged_data = None
    
    def load_base_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load the base dataset (questions and answers).
        
        Args:
            path: Path to the base data file
            
        Returns:
            DataFrame with base data
        """
        path = Path(path)
        logger.info(f"Loading base data from {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        self.data = data
        logger.info(f"Loaded {len(data)} base samples")
        return data
    
    def load_human_feedback(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load human feedback data (persona scores).
        
        Args:
            path: Path to human feedback data file
            
        Returns:
            DataFrame with human feedback
        """
        path = Path(path)
        logger.info(f"Loading human feedback from {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        self.human_feedback_data = data
        logger.info(f"Loaded {len(data)} samples with human feedback")
        return data
    
    def load_judge_scores(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load judge evaluation scores.
        
        Args:
            path: Path to judge scores data file
            
        Returns:
            DataFrame with judge scores
        """
        path = Path(path)
        logger.info(f"Loading judge scores from {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        self.judge_scores_data = data
        logger.info(f"Loaded {len(data)} samples with judge scores")
        return data
    
    def extract_human_feedback_score(self, feedback: Union[Dict, float, None]) -> float:
        """
        Extract the human feedback score from various formats.
        
        Args:
            feedback: Human feedback data (can be dict, float, or None)
            
        Returns:
            Extracted score (float)
        """
        if feedback is None:
            return np.nan
        
        if isinstance(feedback, (int, float)):
            return float(feedback)
        
        if isinstance(feedback, dict):
            # Try different possible keys
            for key in ['score', 'average_score', 'human_score', 'value']:
                if key in feedback:
                    return float(feedback[key])
            
            # If it has persona scores, calculate average
            if 'personas' in feedback and isinstance(feedback['personas'], dict):
                scores = []
                for persona_data in feedback['personas'].values():
                    if isinstance(persona_data, dict) and 'score' in persona_data:
                        scores.append(persona_data['score'])
                if scores:
                    return sum(scores) / len(scores)
        
        logger.warning(f"Could not extract score from feedback: {feedback}")
        return np.nan
    
    def extract_judge_scores(self, judge_data: Union[Dict, List, None]) -> Dict[str, float]:
        """
        Extract individual judge scores from various formats.
        
        Args:
            judge_data: Judge evaluation data
            
        Returns:
            Dictionary mapping judge names to scores
        """
        if judge_data is None:
            return {}
        
        # If it's already a dict with judge names as keys
        if isinstance(judge_data, dict):
            # Check if it has a 'judges' key
            if 'judges' in judge_data:
                judge_data = judge_data['judges']
            
            scores = {}
            for judge_name, score_data in judge_data.items():
                if isinstance(score_data, (int, float)):
                    scores[judge_name] = float(score_data)
                elif isinstance(score_data, dict):
                    # Try to extract score from dict
                    for key in ['score', 'value', 'rating']:
                        if key in score_data:
                            scores[judge_name] = float(score_data[key])
                            break
            return scores
        
        # If it's a list of scores (need to map to judge names)
        if isinstance(judge_data, list):
            judge_names = [
                'harmlessness-judge',
                'privacy-judge',
                'factual-accuracy-judge',
                'faithfulness-judge',
                'calibration-judge',
                'bias-judge',
                'reasoning-judge',
                'coherence-judge',
                'conciseness-judge',
                'style-judge'
            ]
            scores = {}
            for i, score in enumerate(judge_data):
                if i < len(judge_names):
                    if isinstance(score, (int, float)):
                        scores[judge_names[i]] = float(score)
            return scores
        
        logger.warning(f"Could not extract judge scores from: {type(judge_data)}")
        return {}
    
    def merge_datasets(
        self,
        base_data: Optional[pd.DataFrame] = None,
        human_feedback_data: Optional[pd.DataFrame] = None,
        judge_scores_data: Optional[pd.DataFrame] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Merge all datasets into a unified training dataset.
        
        Args:
            base_data: Base dataset (uses self.data if None)
            human_feedback_data: Human feedback data (uses self.human_feedback_data if None)
            judge_scores_data: Judge scores data (uses self.judge_scores_data if None)
            output_path: Optional path to save merged data
            
        Returns:
            Merged DataFrame ready for training
        """
        # Use provided data or fall back to loaded data
        base = base_data if base_data is not None else self.data
        human = human_feedback_data if human_feedback_data is not None else self.human_feedback_data
        judges = judge_scores_data if judge_scores_data is not None else self.judge_scores_data
        
        if base is None:
            raise ValueError("No base data provided. Load base data first.")
        
        # Start with base data
        merged = base.copy()
        logger.info(f"Starting with {len(merged)} base samples")
        
        # Merge human feedback if available
        if human is not None:
            logger.info("Merging human feedback data...")
            
            # Extract human feedback scores
            if 'human_feedback' in human.columns:
                human_scores = human['human_feedback'].apply(self.extract_human_feedback_score)
                merged['human_score'] = human_scores
                logger.info(f"Extracted {human_scores.notna().sum()} valid human scores")
            
            # Copy over other relevant columns from human feedback data
            for col in human.columns:
                if col not in merged.columns and col != 'human_feedback':
                    merged[col] = human[col]
        
        # Merge judge scores if available
        if judges is not None:
            logger.info("Merging judge scores data...")
            
            # Find the column with judge scores
            judge_col = None
            for col in ['judge_scores', 'judges', 'scores', 'evaluations']:
                if col in judges.columns:
                    judge_col = col
                    break
            
            if judge_col:
                # Extract individual judge scores
                judge_scores = judges[judge_col].apply(self.extract_judge_scores)
                
                # Create columns for each judge
                all_judges = set()
                for scores_dict in judge_scores:
                    all_judges.update(scores_dict.keys())
                
                for judge_name in sorted(all_judges):
                    merged[f'judge_{judge_name}'] = judge_scores.apply(
                        lambda x: x.get(judge_name, np.nan) if x else np.nan
                    )
                
                logger.info(f"Extracted scores for {len(all_judges)} judges")
            
            # Copy over other relevant columns from judge data
            for col in judges.columns:
                if col not in merged.columns and col not in [judge_col, 'human_feedback']:
                    merged[col] = judges[col]
        
        # Calculate statistics
        self._print_statistics(merged)
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                pickle.dump(merged, f)
            logger.info(f"Saved merged data to {output_path}")
        
        self.merged_data = merged
        return merged
    
    def _print_statistics(self, data: pd.DataFrame):
        """Print statistics about the merged dataset."""
        logger.info("\n" + "="*50)
        logger.info("Merged Dataset Statistics")
        logger.info("="*50)
        
        # Basic info
        logger.info(f"Total samples: {len(data)}")
        logger.info(f"Total columns: {len(data.columns)}")
        
        # Human scores
        if 'human_score' in data.columns:
            valid_human = data['human_score'].notna().sum()
            if valid_human > 0:
                logger.info(f"\nHuman Scores:")
                logger.info(f"  Valid: {valid_human}/{len(data)}")
                logger.info(f"  Mean: {data['human_score'].mean():.2f}")
                logger.info(f"  Std: {data['human_score'].std():.2f}")
                logger.info(f"  Min: {data['human_score'].min():.1f}")
                logger.info(f"  Max: {data['human_score'].max():.1f}")
        
        # Judge scores
        judge_cols = [col for col in data.columns if col.startswith('judge_')]
        if judge_cols:
            logger.info(f"\nJudge Scores ({len(judge_cols)} judges):")
            for col in judge_cols[:5]:  # Show first 5 judges
                valid = data[col].notna().sum()
                if valid > 0:
                    judge_name = col.replace('judge_', '')
                    logger.info(f"  {judge_name}:")
                    logger.info(f"    Valid: {valid}/{len(data)}")
                    logger.info(f"    Mean: {data[col].mean():.2f}")
            
            if len(judge_cols) > 5:
                logger.info(f"  ... and {len(judge_cols) - 5} more judges")
        
        # Check for complete samples (have both human and judge scores)
        if 'human_score' in data.columns and judge_cols:
            complete_mask = data['human_score'].notna()
            for col in judge_cols:
                complete_mask &= data[col].notna()
            complete_count = complete_mask.sum()
            logger.info(f"\nComplete samples (all scores): {complete_count}/{len(data)}")
    
    def prepare_training_data(
        self,
        data: Optional[pd.DataFrame] = None,
        normalize: bool = True,
        drop_incomplete: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            data: Merged data (uses self.merged_data if None)
            normalize: Whether to normalize judge scores
            drop_incomplete: Whether to drop samples with missing values
            
        Returns:
            Tuple of (X, y) where X is judge scores and y is human scores
        """
        data = data if data is not None else self.merged_data
        
        if data is None:
            raise ValueError("No data available. Merge datasets first.")
        
        # Extract judge score columns
        judge_cols = sorted([col for col in data.columns if col.startswith('judge_')])
        
        if not judge_cols:
            raise ValueError("No judge score columns found in data")
        
        if 'human_score' not in data.columns:
            raise ValueError("No human_score column found in data")
        
        # Prepare features and target
        X = data[judge_cols].values
        y = data['human_score'].values
        
        # Handle missing values
        if drop_incomplete:
            # Keep only complete samples
            complete_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[complete_mask]
            y = y[complete_mask]
            logger.info(f"Kept {len(X)}/{len(data)} complete samples")
        else:
            # Fill missing values with column means
            col_means = np.nanmean(X, axis=0)
            for i in range(X.shape[1]):
                X[np.isnan(X[:, i]), i] = col_means[i]
            y[np.isnan(y)] = np.nanmean(y)
        
        # Normalize if requested
        if normalize:
            # Judge scores are typically 0-4, normalize to 0-1
            X = X / 4.0
            # Human scores are typically 0-10, normalize to 0-1
            y = y / 10.0
        
        logger.info(f"Prepared training data: X shape {X.shape}, y shape {y.shape}")
        return X, y


def main():
    """Main entry point for data merging."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge data from different pipeline stages")
    parser.add_argument('--base', help='Path to base data file')
    parser.add_argument('--human', help='Path to human feedback data file')
    parser.add_argument('--judges', help='Path to judge scores data file')
    parser.add_argument('--output', required=True, help='Path to save merged data')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize scores for training')
    parser.add_argument('--drop-incomplete', action='store_true',
                        help='Drop samples with missing values')
    
    args = parser.parse_args()
    
    # Initialize merger
    merger = DataMerger()
    
    # Load data files
    base_data = None
    human_data = None
    judge_data = None
    
    if args.base:
        base_data = merger.load_base_data(args.base)
    
    if args.human:
        human_data = merger.load_human_feedback(args.human)
    
    if args.judges:
        judge_data = merger.load_judge_scores(args.judges)
    
    # Merge datasets
    merged = merger.merge_datasets(
        base_data=base_data,
        human_feedback_data=human_data,
        judge_scores_data=judge_data,
        output_path=args.output
    )
    
    # Optionally prepare training data
    if args.normalize or args.drop_incomplete:
        X, y = merger.prepare_training_data(
            data=merged,
            normalize=args.normalize,
            drop_incomplete=args.drop_incomplete
        )
        logger.info(f"\nTraining data ready:")
        logger.info(f"  Features (X): {X.shape}")
        logger.info(f"  Target (y): {y.shape}")
        logger.info(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
        logger.info(f"  y range: [{y.min():.2f}, {y.max():.2f}]")


if __name__ == "__main__":
    main()