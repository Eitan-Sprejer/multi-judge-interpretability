"""
Data Preparation for Bias Transfer Experiment 4C

Handles AFINN-111 sentiment lexicon tokens and word frequency data
for testing framing effects and frequency bias in judge aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set
import logging
import pickle
from pathlib import Path

try:
    from afinn import Afinn
except ImportError:
    print("Installing AFINN library...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "afinn"])
    from afinn import Afinn

try:
    import wordfreq
except ImportError:
    print("Installing wordfreq library...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wordfreq"])
    import wordfreq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiasDataPreparator:
    """Prepares token datasets for bias transfer analysis"""
    
    def __init__(self, language: str = 'en'):
        """
        Initialize the data preparator
        
        Args:
            language: Language code for AFINN and wordfreq (default: 'en')
        """
        self.language = language
        self.afinn = Afinn(language=language)
        
        # Neutral control terms as specified in experiment guide
        self.neutral_tokens = [
            "table", "door", "computer", "tree", "paper", "window", "book", 
            "phone", "car", "house", "city", "road", "food", "clothes", 
            "music", "movie", "game", "work", "school", "time", "chair",
            "wall", "floor", "building", "water", "fire", "light"
        ]
        
    def get_afinn_tokens(self, min_tokens: int = 200) -> pd.DataFrame:
        """
        Extract AFINN-111 tokens with sentiment scores
        
        Args:
            min_tokens: Minimum number of tokens to extract
            
        Returns:
            DataFrame with columns: token, sentiment_score, is_positive, is_negative
        """
        logger.info("Loading AFINN-111 lexicon...")
        
        # Get all AFINN word scores
        afinn_words = []
        afinn_scores = []
        
        # Extract words from AFINN lexicon
        for word, score in self.afinn._dict.items():
            afinn_words.append(word)
            afinn_scores.append(score)
        
        # Create DataFrame
        df = pd.DataFrame({
            'token': afinn_words,
            'sentiment_score': afinn_scores
        })
        
        # Add categorical flags
        df['is_positive'] = df['sentiment_score'] > 0
        df['is_negative'] = df['sentiment_score'] < 0
        df['is_neutral_afinn'] = df['sentiment_score'] == 0
        
        logger.info(f"Loaded {len(df)} AFINN tokens")
        logger.info(f"Positive: {df['is_positive'].sum()}, "
                   f"Negative: {df['is_negative'].sum()}, "
                   f"Neutral: {df['is_neutral_afinn'].sum()}")
        
        if len(df) < min_tokens:
            logger.warning(f"Only {len(df)} AFINN tokens available, "
                          f"requested minimum {min_tokens}")
        
        return df
    
    def add_word_frequencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add word frequency data to token DataFrame
        
        Args:
            df: DataFrame with token column
            
        Returns:
            DataFrame with added frequency columns
        """
        logger.info("Adding word frequency data...")
        
        frequencies = []
        log_frequencies = []
        
        for token in df['token']:
            try:
                # Get frequency from wordfreq library
                freq = wordfreq.word_frequency(token, self.language)
                frequencies.append(freq)
                
                # Log transform: log(frequency + 1e-10) as per experiment guide
                log_freq = np.log(freq + 1e-10)
                log_frequencies.append(log_freq)
                
            except Exception as e:
                logger.warning(f"Could not get frequency for '{token}': {e}")
                frequencies.append(1e-10)  # Very low frequency for unknown words
                log_frequencies.append(np.log(1e-10))
        
        df = df.copy()
        df['frequency'] = frequencies
        df['log_frequency'] = log_frequencies
        
        logger.info(f"Added frequency data for {len(df)} tokens")
        logger.info(f"Frequency range: {min(frequencies):.2e} to {max(frequencies):.2e}")
        
        return df
    
    def add_neutral_control_tokens(self, df: pd.DataFrame, vocabulary_filter: List[str] = None) -> pd.DataFrame:
        """
        Add neutral control tokens to the dataset
        
        Args:
            df: Existing AFINN tokens DataFrame
            vocabulary_filter: Optional list of allowed tokens
            
        Returns:
            Combined DataFrame with neutral tokens added
        """
        logger.info("Adding neutral control tokens...")
        
        # Filter neutral tokens by vocabulary if provided
        available_neutral_tokens = self.neutral_tokens
        if vocabulary_filter is not None:
            available_neutral_tokens = [token for token in self.neutral_tokens if token in vocabulary_filter]
            logger.info(f"Neutral tokens after vocabulary filtering: {len(available_neutral_tokens)}/{len(self.neutral_tokens)}")
        
        # Create neutral tokens DataFrame
        neutral_data = []
        for token in available_neutral_tokens:
            try:
                freq = wordfreq.word_frequency(token, self.language)
                log_freq = np.log(freq + 1e-10)
                
                neutral_data.append({
                    'token': token,
                    'sentiment_score': 0,  # Neutral by definition
                    'is_positive': False,
                    'is_negative': False,
                    'is_neutral_afinn': False,  # Not from AFINN, but neutral
                    'is_neutral_control': True,
                    'frequency': freq,
                    'log_frequency': log_freq
                })
            except Exception as e:
                logger.warning(f"Could not process neutral token '{token}': {e}")
        
        neutral_df = pd.DataFrame(neutral_data)
        
        # Add control flag to original data
        df = df.copy()
        df['is_neutral_control'] = False
        
        # Combine datasets
        combined_df = pd.concat([df, neutral_df], ignore_index=True)
        
        logger.info(f"Added {len(neutral_df)} neutral control tokens")
        logger.info(f"Total tokens: {len(combined_df)}")
        
        return combined_df
    
    def prepare_bias_dataset(self, 
                           min_tokens: int = 200,
                           vocabulary_filter: List[str] = None,
                           save_path: str = None) -> pd.DataFrame:
        """
        Prepare complete dataset for bias analysis
        
        Args:
            min_tokens: Minimum AFINN tokens to include
            vocabulary_filter: List of tokens to include (model vocabulary)
            save_path: Optional path to save the dataset
            
        Returns:
            Complete dataset ready for bias analysis
        """
        logger.info("Preparing complete bias analysis dataset...")
        
        # Get AFINN tokens
        afinn_df = self.get_afinn_tokens(min_tokens=min_tokens)
        
        # Filter by vocabulary if provided
        if vocabulary_filter is not None:
            logger.info(f"Filtering tokens to model vocabulary: {len(vocabulary_filter)} allowed tokens")
            original_count = len(afinn_df)
            afinn_df = afinn_df[afinn_df['token'].isin(vocabulary_filter)]
            filtered_count = len(afinn_df)
            logger.info(f"Vocabulary filtering: {original_count} → {filtered_count} tokens")
            
            if filtered_count < min_tokens:
                logger.warning(f"After vocabulary filtering, only {filtered_count} tokens remain "
                             f"(requested minimum: {min_tokens})")
        
        # Add frequency data
        afinn_df = self.add_word_frequencies(afinn_df)
        
        # Add neutral control tokens (also filter by vocabulary)
        complete_df = self.add_neutral_control_tokens(afinn_df, vocabulary_filter)
        
        # Add additional metadata
        complete_df['abs_sentiment'] = np.abs(complete_df['sentiment_score'])
        complete_df['sentiment_category'] = complete_df.apply(
            self._categorize_sentiment, axis=1
        )
        
        # Calculate frequency percentiles for analysis
        complete_df['frequency_percentile'] = complete_df['frequency'].rank(pct=True)
        
        logger.info("Dataset preparation complete!")
        logger.info(f"Final dataset shape: {complete_df.shape}")
        logger.info("Sentiment distribution:")
        logger.info(complete_df['sentiment_category'].value_counts().to_string())
        
        if save_path:
            logger.info(f"Saving dataset to {save_path}")
            complete_df.to_pickle(save_path)
        
        return complete_df
    
    def _categorize_sentiment(self, row) -> str:
        """Categorize sentiment into groups for analysis"""
        if row['is_neutral_control']:
            return 'neutral_control'
        elif row['sentiment_score'] > 2:
            return 'very_positive'
        elif row['sentiment_score'] > 0:
            return 'positive'
        elif row['sentiment_score'] == 0:
            return 'neutral_afinn'
        elif row['sentiment_score'] > -2:
            return 'negative'
        else:
            return 'very_negative'
    
    def validate_dataset(self, df: pd.DataFrame, min_per_category: int = 20) -> Dict[str, bool]:
        """
        Validate the prepared dataset for bias analysis with statistical power checks
        
        Args:
            df: Prepared dataset
            min_per_category: Minimum tokens needed per sentiment category for reliable statistics
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        # Check minimum token counts for statistical power
        results['sufficient_positive'] = (df['is_positive'].sum() >= min_per_category)
        results['sufficient_negative'] = (df['is_negative'].sum() >= min_per_category)
        results['has_neutral_control'] = (df['is_neutral_control'].sum() >= 10)  # Lower threshold for neutral
        
        # Additional checks for framing analysis requirements
        results['sufficient_very_positive'] = (df[df['sentiment_score'] > 2].shape[0] >= 10)
        results['sufficient_very_negative'] = (df[df['sentiment_score'] < -2].shape[0] >= 10)
        results['balanced_sentiment'] = (abs(df['is_positive'].sum() - df['is_negative'].sum()) <= df.shape[0] * 0.3)
        
        # Check data completeness
        results['no_missing_sentiment'] = df['sentiment_score'].notna().all()
        results['no_missing_frequency'] = df['frequency'].notna().all()
        
        # Check data ranges
        results['valid_sentiment_range'] = (
            (df['sentiment_score'].min() >= -5) and 
            (df['sentiment_score'].max() <= 5)
        )
        results['valid_frequency_range'] = (df['frequency'] >= 0).all()
        
        logger.info("Dataset validation results:")
        for check, passed in results.items():
            status = "✓" if passed else "✗"
            logger.info(f"  {status} {check}: {passed}")
        
        results['overall_valid'] = all(results.values())
        
        return results


def main():
    """Example usage of the BiasDataPreparator"""
    
    # Initialize preparator
    preparator = BiasDataPreparator()
    
    # Prepare dataset
    dataset = preparator.prepare_bias_dataset(
        min_tokens=200,
        save_path="results/bias_analysis_tokens.pkl"
    )
    
    # Validate dataset
    validation = preparator.validate_dataset(dataset)
    
    if validation['overall_valid']:
        print("\n✓ Dataset preparation successful!")
        print(f"Ready for bias analysis with {len(dataset)} tokens")
    else:
        print("\n✗ Dataset validation failed")
        print("Check logs for specific issues")
    
    return dataset


if __name__ == "__main__":
    dataset = main()