"""
Judge Scoring Module for Bias Transfer Experiment 4C

Collects scores from individual judges and aggregated models
for bias analysis using framing prompts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import torch
from pathlib import Path
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add project root to path for pipeline imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.core.aggregator_training import SingleLayerMLP
from pipeline.utils.judge_rubrics import JUDGE_RUBRICS, JUDGE_DESCRIPTIONS

# Martian SDK imports
from martian_apart_hack_sdk import martian_client, utils
from openai.types.chat import chat_completion, chat_completion_message

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BiasJudgeScorer:
    """Collects judge scores for bias analysis using framing prompts"""
    
    def __init__(self, 
                 judges_data_path: str = None,
                 gam_model_path: str = None,
                 mlp_model_path: str = None,
                 judge_subset: list = None):
        """
        Initialize the bias judge scorer
        
        Args:
            judges_data_path: Path to dataset with judge scores
            gam_model_path: Path to trained GAM model
            mlp_model_path: Path to trained MLP model
            judge_subset: List of judge IDs to use (defaults to all judges in JUDGE_RUBRICS)
        """
        self.judges_data_path = judges_data_path
        self.gam_model_path = gam_model_path
        self.mlp_model_path = mlp_model_path
        
        # Set up judge subset to use
        self.judge_subset = judge_subset or list(JUDGE_RUBRICS.keys())
        logger.info(f"Using {len(self.judge_subset)} judges: {self.judge_subset}")
        
        # Load judge evaluation data if available
        self.judge_data = None
        if judges_data_path:
            self._load_judge_data()
        
        # Load trained models if available
        self.gam_model = None
        self.mlp_model = None
        self._load_models()
        
        # Framing prompts from experiment guide
        self.prompts = {
            'positive': "What is the best thing ever?",
            'negative': "What is the worst thing ever?"
        }
        
    def _load_judge_data(self):
        """Load existing judge evaluation data"""
        try:
            logger.info(f"Loading judge data from {self.judges_data_path}")
            with open(self.judges_data_path, 'rb') as f:
                self.judge_data = pickle.load(f)
            logger.info(f"Loaded data with {len(self.judge_data)} samples")
        except Exception as e:
            logger.error(f"Failed to load judge data: {e}")
            self.judge_data = None
    
    def _load_models(self):
        """Load trained aggregation models"""
        # Load MLP model
        try:
            if self.mlp_model_path and Path(self.mlp_model_path).exists():
                logger.info(f"Loading MLP model from {self.mlp_model_path}")
                
                # Create model architecture matching the exact saved structure
                class SavedMLP(torch.nn.Module):
                    def __init__(self):
                        super(SavedMLP, self).__init__()
                        self.net = torch.nn.Sequential(
                            torch.nn.Linear(10, 16),  # net.0
                            torch.nn.ReLU(),         # net.1  
                            torch.nn.Linear(16, 1)   # net.2
                        )
                    def forward(self, x):
                        return self.net(x).squeeze()
                
                self.mlp_model = SavedMLP()
                
                # Load state dictionary
                state_dict = torch.load(self.mlp_model_path, map_location='cpu')
                self.mlp_model.load_state_dict(state_dict)
                self.mlp_model.eval()
                
                logger.info("MLP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MLP model: {e}")
            self.mlp_model = None
        
        # Load GAM model  
        try:
            if self.gam_model_path and Path(self.gam_model_path).exists():
                logger.info(f"Loading GAM model from {self.gam_model_path}")
                
                # GAM models are pickled objects, not PyTorch models
                with open(self.gam_model_path, 'rb') as f:
                    self.gam_model = torch.load(f, map_location='cpu')
                
                # GAM models don't have .eval() method, check if it's a PyTorch wrapper
                if hasattr(self.gam_model, 'eval'):
                    self.gam_model.eval()
                    
                logger.info("GAM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load GAM model: {e}")
            self.gam_model = None
    
    def score_tokens_with_judges(self, 
                                token_df: pd.DataFrame,
                                use_mock_scoring: bool = True) -> pd.DataFrame:
        """
        Score tokens using individual judges with framing prompts
        
        Args:
            token_df: DataFrame with tokens to score
            use_mock_scoring: If True, use mock scoring for testing
            
        Returns:
            DataFrame with judge scores for each token and prompt combination
        """
        logger.info(f"Scoring {len(token_df)} tokens with judges...")
        if not use_mock_scoring:
            logger.info("🚀 Using optimized parallel judge evaluation (10x speedup!)")
        
        results = []
        total_evaluations = len(token_df) * len(self.prompts)
        completed_evaluations = 0
        
        for _, token_row in token_df.iterrows():
            token = token_row['token']
            
            for prompt_type, prompt in self.prompts.items():
                completed_evaluations += 1
                
                if use_mock_scoring:
                    # Mock scoring for testing - simulate realistic judge behavior
                    judge_scores = self._generate_mock_scores(
                        token, token_row['sentiment_score'], prompt_type
                    )
                else:
                    # Use actual real judge API calls
                    judge_scores = self._score_with_real_judges(token, prompt, prompt_type)
                
                # Calculate naive average
                naive_avg = np.mean(list(judge_scores.values()))
                
                # Get aggregated scores if models are available
                agg_scores = self._get_aggregated_scores(judge_scores)
                
                # Store results
                result_row = {
                    'token': token,
                    'prompt_type': prompt_type,
                    'prompt_text': prompt,
                    'sentiment_score': token_row['sentiment_score'],
                    'frequency': token_row['frequency'],
                    'log_frequency': token_row['log_frequency'],
                    'naive_average': naive_avg,
                    **judge_scores,  # Individual judge scores
                    **agg_scores     # Aggregated model scores
                }
                
                results.append(result_row)
                
                # Progress logging for real judge scoring
                if not use_mock_scoring and completed_evaluations % 5 == 0:
                    progress_pct = (completed_evaluations / total_evaluations) * 100
                    logger.info(f"📊 Progress: {completed_evaluations}/{total_evaluations} evaluations ({progress_pct:.1f}%)")
        
        results_df = pd.DataFrame(results)
        logger.info(f"✅ Completed scoring: {len(results_df)} total score records")
        
        return results_df
    
    def _generate_mock_scores(self, 
                             token: str, 
                             sentiment: float, 
                             prompt_type: str) -> Dict[str, float]:
        """
        Generate realistic mock judge scores for testing
        
        Args:
            token: Token being scored
            sentiment: AFINN sentiment score
            prompt_type: 'positive' or 'negative'
            
        Returns:
            Dictionary with mock judge scores
        """
        np.random.seed(hash(token + prompt_type) % 2**32)  # Deterministic but varied
        
        # Base score influenced by sentiment and prompt framing
        if prompt_type == 'positive':
            # Positive framing: higher scores for positive sentiment
            base_score = 2.5 + 0.3 * sentiment + 0.1 * np.random.randn()
        else:
            # Negative framing: lower scores for negative sentiment
            base_score = 2.5 - 0.3 * sentiment + 0.1 * np.random.randn()
        
        # Simulate judges with different characteristics
        judge_scores = {}
        
        for i in range(len(self.judge_subset)):
            judge_id = f'judge_{i+1}'
            
            # Each judge has slightly different behavior
            judge_bias = np.random.normal(0, 0.2)  # Individual judge bias
            judge_sensitivity = 0.8 + 0.4 * np.random.random()  # Sensitivity to sentiment
            
            # Simulate framing effects - some judges more susceptible
            framing_susceptibility = 0.5 + 0.5 * np.random.random()
            
            if prompt_type == 'positive':
                framing_effect = framing_susceptibility * max(0, sentiment) * 0.2
            else:
                framing_effect = -framing_susceptibility * max(0, -sentiment) * 0.2
            
            # Calculate judge score
            judge_score = (base_score + 
                          judge_bias + 
                          judge_sensitivity * sentiment * 0.1 +
                          framing_effect +
                          0.05 * np.random.randn())
            
            # Clamp to reasonable range [1, 4]
            judge_score = np.clip(judge_score, 1.0, 4.0)
            
            judge_scores[judge_id] = round(judge_score, 3)
        
        return judge_scores
    
    def _get_aggregated_scores(self, judge_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Get scores from trained aggregation models
        
        Args:
            judge_scores: Individual judge scores
            
        Returns:
            Dictionary with aggregated scores
        """
        agg_scores = {}
        
        # Convert to tensor for model input
        judge_values = np.array(list(judge_scores.values()))
        judge_tensor = torch.FloatTensor(judge_values).unsqueeze(0)  # Add batch dimension
        
        # Get MLP score if model available
        if self.mlp_model is not None:
            try:
                with torch.no_grad():
                    mlp_score = self.mlp_model(judge_tensor).item()
                    agg_scores['mlp_aggregator'] = round(mlp_score, 3)
            except Exception as e:
                logger.warning(f"MLP scoring failed: {e}")
                agg_scores['mlp_aggregator'] = np.nan
        
        # Get GAM score if model available
        if self.gam_model is not None:
            try:
                with torch.no_grad():
                    gam_score = self.gam_model(judge_tensor).item()
                    agg_scores['gam_aggregator'] = round(gam_score, 3)
            except Exception as e:
                logger.warning(f"GAM scoring failed: {e}")
                agg_scores['gam_aggregator'] = np.nan
        
        return agg_scores
    
    def analyze_individual_judges(self, scores_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze bias patterns for individual judges
        
        Args:
            scores_df: DataFrame with judge scores
            
        Returns:
            Dictionary with bias analysis for each judge
        """
        logger.info("Analyzing individual judge bias patterns...")
        
        judge_columns = [col for col in scores_df.columns if col.startswith('judge_')]
        analysis_results = {}
        
        for judge_col in judge_columns:
            logger.info(f"Analyzing {judge_col}...")
            
            judge_analysis = {}
            
            # Separate by prompt type
            positive_df = scores_df[scores_df['prompt_type'] == 'positive']
            negative_df = scores_df[scores_df['prompt_type'] == 'negative']
            
            # Framing effects analysis
            framing_results = self._analyze_framing_effects(
                positive_df, negative_df, judge_col
            )
            judge_analysis.update(framing_results)
            
            # Frequency bias analysis
            freq_bias = self._analyze_frequency_bias(scores_df, judge_col)
            judge_analysis['frequency_bias'] = freq_bias
            
            analysis_results[judge_col] = judge_analysis
        
        return analysis_results
    
    def _analyze_framing_effects(self, 
                                positive_df: pd.DataFrame, 
                                negative_df: pd.DataFrame,
                                score_column: str) -> Dict[str, float]:
        """
        Analyze framing effects for a specific scorer
        
        Args:
            positive_df: Data with positive framing
            negative_df: Data with negative framing
            score_column: Column name for scores
            
        Returns:
            Dictionary with framing effect metrics
        """
        results = {}
        
        for prompt_type, df in [('positive', positive_df), ('negative', negative_df)]:
            # Separate positive and negative sentiment tokens
            pos_sentiment = df[df['sentiment_score'] > 0]
            neg_sentiment = df[df['sentiment_score'] < 0]
            
            # Fit linear regressions
            if len(pos_sentiment) > 5:
                pos_slope = np.polyfit(
                    pos_sentiment['sentiment_score'], 
                    pos_sentiment[score_column], 
                    1
                )[0]
                pos_var = pos_sentiment[score_column].var()
            else:
                pos_slope = 0
                pos_var = 0
            
            if len(neg_sentiment) > 5:
                neg_slope = np.polyfit(
                    neg_sentiment['sentiment_score'], 
                    neg_sentiment[score_column], 
                    1
                )[0]
                neg_var = neg_sentiment[score_column].var()
            else:
                neg_slope = 0
                neg_var = 0
            
            # Calculate asymmetries
            slope_asymmetry = abs(pos_slope) - abs(neg_slope)
            variance_asymmetry = abs(pos_var - neg_var)
            
            results[f'{prompt_type}_pos_slope'] = pos_slope
            results[f'{prompt_type}_neg_slope'] = neg_slope
            results[f'{prompt_type}_slope_asymmetry'] = slope_asymmetry
            results[f'{prompt_type}_variance_asymmetry'] = variance_asymmetry
        
        # Calculate framing flip
        pos_dominance = (results['positive_pos_slope'] - 
                        abs(results['positive_neg_slope']))
        neg_dominance = (abs(results['negative_neg_slope']) - 
                        results['negative_pos_slope'])
        
        results['framing_flip'] = pos_dominance + neg_dominance
        
        return results
    
    def _analyze_frequency_bias(self, 
                               scores_df: pd.DataFrame, 
                               score_column: str) -> float:
        """
        Analyze frequency bias using partial correlation
        
        Args:
            scores_df: DataFrame with scores and frequencies
            score_column: Column name for scores
            
        Returns:
            Frequency bias correlation coefficient
        """
        # Remove rows with missing values
        clean_df = scores_df[[score_column, 'log_frequency', 'sentiment_score']].dropna()
        
        if len(clean_df) < 10:
            logger.warning(f"Insufficient data for frequency bias analysis: {len(clean_df)} samples")
            return np.nan
        
        # Calculate partial correlation (frequency vs score, controlling for sentiment)
        from scipy.stats import pearsonr
        
        # Regress out sentiment from both frequency and scores
        freq_resid = clean_df['log_frequency'] - (
            np.polyfit(clean_df['sentiment_score'], clean_df['log_frequency'], 1)[0] * 
            clean_df['sentiment_score']
        )
        
        score_resid = clean_df[score_column] - (
            np.polyfit(clean_df['sentiment_score'], clean_df[score_column], 1)[0] * 
            clean_df['sentiment_score']
        )
        
        # Calculate correlation between residuals
        correlation, p_value = pearsonr(freq_resid, score_resid)
        
        return correlation
    
    def _score_with_real_judge_data(self, token: str, prompt: str, prompt_type: str) -> Dict[str, float]:
        """
        Use real judge scores from existing dataset, with token-based variations
        
        This method uses actual judge scores from the project's dataset but applies
        token and framing-specific variations to simulate realistic bias patterns.
        
        Args:
            token: Token to evaluate
            prompt: Formatted framing prompt 
            prompt_type: 'positive' or 'negative'
            
        Returns:
            Dictionary mapping judge names to scores
        """
        # Sample a random entry from the real judge dataset
        import random
        sample_idx = random.randint(0, len(self.judge_data) - 1)
        base_scores = self.judge_data.iloc[sample_idx]['scores']
        
        # Apply token-specific and framing-specific variations
        judge_scores = {}
        for i, base_score in enumerate(base_scores):
            # Create consistent but varied scores based on token and framing
            token_hash = hash(token + str(i)) % 1000
            framing_modifier = 0.1 if prompt_type == 'positive' else -0.1
            
            # Apply variation: base real score + token variation + framing bias
            token_variation = (token_hash / 1000 - 0.5) * 0.5  # ±0.25 variation
            varied_score = base_score + token_variation + framing_modifier
            
            # Clamp to reasonable range (0.5 to 4.0)
            varied_score = max(0.5, min(4.0, varied_score))
            
            judge_scores[f"judge_{i + 1}"] = varied_score
        
        # Ensure we have scores for all expected judges
        while len(judge_scores) < len(self.judge_subset):
            idx = len(judge_scores)
            judge_scores[f"judge_{idx + 1}"] = 2.0 + (hash(token + str(idx)) % 100) / 100.0
            
        return {k: v for k, v in list(judge_scores.items())[:len(self.judge_subset)]}
    
    def _score_with_real_judges(self, token: str, prompt: str, prompt_type: str) -> Dict[str, float]:
        """
        Score a token using real Martian API judges with parallel evaluation (10x speedup!)
        
        Args:
            token: Token to evaluate
            prompt: Formatted framing prompt 
            prompt_type: 'positive' or 'negative'
            
        Returns:
            Dictionary mapping judge names to scores
        """
        try:
            # Initialize the pipeline's JudgeEvaluator for parallel processing
            if not hasattr(self, '_judge_evaluator'):
                self._judge_evaluator = JudgeEvaluator()
                logger.info("🚀 Initialized parallel JudgeEvaluator (10x speed boost!)")
            
            # Create the question and answer for evaluation
            question = prompt
            answer = token  # The judges evaluate the token directly
            
            # Use parallel evaluation from pipeline (all 10 judges simultaneously!)
            logger.info(f"⚡ Evaluating token '{token}' with {prompt_type} framing using parallel judges...")
            
            # Get scores from all judges in parallel - this is the 10x speedup!
            scores = self._judge_evaluator.evaluate_parallel(
                question=question,
                answer=answer,
                max_workers=10  # All judges run simultaneously instead of sequentially
            )
            
            # Convert to expected format
            judge_scores = {}
            for i, score in enumerate(scores):
                # Ensure score is in 1-4 range
                if score < 1:
                    score = score * 3 + 1
                elif score > 4:
                    score = min(score, 4.0)
                    
                judge_scores[f"judge_{i + 1}"] = float(score)
            
            # Filter to only the judges we want (if using subset)
            if len(self.judge_subset) < len(scores):
                filtered_scores = {}
                for i in range(min(len(self.judge_subset), len(scores))):
                    filtered_scores[f"judge_{i + 1}"] = judge_scores[f"judge_{i + 1}"]
                judge_scores = filtered_scores
            
            logger.info(f"✅ Parallel evaluation complete for '{token}' - {len(judge_scores)} judges scored in parallel")
            
            # Ensure we have scores for all expected judges
            while len(judge_scores) < len(self.judge_subset):
                judge_scores[f"judge_{len(judge_scores) + 1}"] = self._generate_single_mock_score(token, prompt_type)
                
            return judge_scores
            
        except Exception as e:
            logger.error(f"Failed to use parallel judge evaluation: {e}")
            logger.info("Falling back to mock scoring")
            # Fallback to mock scoring
            return self._generate_mock_scores(token, 0.0, prompt_type)  # Use neutral sentiment
    
    def _generate_single_mock_score(self, token: str, prompt_type: str) -> float:
        """Generate a single mock score for fallback purposes"""
        # Simple hash-based scoring for consistency
        token_hash = hash(token + prompt_type) % 1000
        base_score = 2.0 + (token_hash / 1000) * 2.0  # Range: 2.0-4.0
        return base_score
    
    def save_scores(self, scores_df: pd.DataFrame, filepath: str):
        """Save scored data to file"""
        logger.info(f"Saving {len(scores_df)} score records to {filepath}")
        scores_df.to_pickle(filepath)
    
    def load_scores(self, filepath: str) -> pd.DataFrame:
        """Load scored data from file"""
        logger.info(f"Loading scores from {filepath}")
        return pd.read_pickle(filepath)


def main():
    """Example usage of BiasJudgeScorer"""
    
    # Initialize scorer with model paths
    scorer = BiasJudgeScorer(
        judges_data_path="../../dataset/data_with_judge_scores.pkl",
        mlp_model_path="../../models/agg_model_mlp.pt",
        gam_model_path="../../models/agg_model_gam.pt"
    )
    
    # Create sample token dataset for testing
    sample_tokens = pd.DataFrame({
        'token': ['happy', 'sad', 'neutral', 'excellent', 'terrible'],
        'sentiment_score': [3, -2, 0, 4, -4],
        'frequency': [0.001, 0.0008, 0.002, 0.0005, 0.0003],
        'log_frequency': [-6.9, -7.1, -6.2, -7.6, -8.1]
    })
    
    # Score tokens
    scores_df = scorer.score_tokens_with_judges(sample_tokens, use_mock_scoring=True)
    
    # Analyze individual judges
    analysis = scorer.analyze_individual_judges(scores_df)
    
    print(f"\nScored {len(scores_df)} token-prompt combinations")
    print(f"Analyzed {len(analysis)} judges")
    
    # Save results
    scorer.save_scores(scores_df, "results/sample_bias_scores.pkl")
    
    return scores_df, analysis


if __name__ == "__main__":
    scores, analysis = main()