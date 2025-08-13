"""
Scoring Pipeline for Multiple Rubric Variants

Runs prompt-answer pairs through all judge variants and collects scores.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import asyncio
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from martian_apart_hack_sdk import martian_client, utils
from openai.types.chat import (
    chat_completion,
    chat_completion_message,
)
from .judge_variant_creator import JudgeVariantCreator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiRubricScoringPipeline:
    """Pipeline for scoring data through multiple judge variants."""
    
    def __init__(
        self,
        data_path: str,
        config_path: Optional[str] = None,
        batch_size: int = 10,
        max_retries: int = 3
    ):
        """
        Initialize the scoring pipeline.
        
        Args:
            data_path: Path to dataset with prompt-answer pairs
            config_path: Optional path to configuration file
            batch_size: Number of concurrent API calls
            max_retries: Maximum retries for failed API calls
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Load data
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            data_loaded = pickle.load(f)
            
        # Convert DataFrame to list of dictionaries if needed
        if hasattr(data_loaded, 'to_dict'):
            self.data = data_loaded.to_dict('records')
            logger.info(f"Converted DataFrame with {len(self.data)} records")
        else:
            self.data = data_loaded
        
        # Load configuration
        if config_path:
            logger.info(f"Loading config from {config_path}")
            config = utils.load_config()
        else:
            config = utils.load_config()
        
        # Initialize Martian client
        self.client = martian_client.MartianClient(
            api_url=config.api_url,
            api_key=config.api_key,
        )
        
        # Initialize judge variant creator
        self.judge_creator = JudgeVariantCreator(config_path)
        
        # Storage for scores
        self.scores = {}
    
    def prepare_judges(
        self,
        judge_ids: Optional[List[str]] = None,
        variation_types: List[str] = ['original', 'formal', 'casual', 'restructured']
    ) -> Dict[str, object]:
        """
        Prepare all judge variants.
        
        Args:
            judge_ids: List of base judge IDs (None = all)
            variation_types: Types of variations to use
            
        Returns:
            Dictionary of judge variant IDs to judge objects
        """
        logger.info("Preparing judge variants...")
        
        # Create all variants
        variants = self.judge_creator.create_all_judge_variants(judge_ids, variation_types)
        
        logger.info(f"Prepared {len(variants)} judge variants")
        return variants
    
    async def score_single_example(
        self,
        judge_variants: Dict[str, object],
        judge_id: str,
        prompt: str,
        response: str,
        idx: int
    ) -> Tuple[int, str, Optional[float]]:
        """
        Score a single example with a specific judge.
        
        Args:
            judge_id: ID of the judge variant
            prompt: The prompt text
            response: The response text
            idx: Example index for tracking
            
        Returns:
            Tuple of (index, judge_id, score)
        """
        for attempt in range(self.max_retries):
            try:
                # Get the judge object first
                judge = judge_variants[judge_id]
                
                # Create the completion request format
                completion_request = {
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "model": "gpt-4o"
                }
                
                # Create the completion response format  
                chat_completion_response = chat_completion.ChatCompletion(
                    id="mock-completion",
                    choices=[
                        chat_completion.Choice(
                            finish_reason="stop",
                            index=0,
                            message=chat_completion_message.ChatCompletionMessage(
                                role="assistant",
                                content=response,
                            ),
                        )
                    ],
                    created=0,
                    model="gpt-4o",
                    object="chat.completion",
                    service_tier=None,
                )
                
                # Call the judge API (synchronous call wrapped in async)
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.client.judges.evaluate(
                        judge,
                        completion_request=completion_request,
                        completion_response=chat_completion_response,
                    )
                )
                
                # Extract score from result
                if hasattr(result, 'score'):
                    score = float(result.score)
                elif isinstance(result, dict) and 'score' in result:
                    score = float(result['score'])
                else:
                    # Try to parse the result as a float
                    try:
                        score = float(str(result).strip())
                    except ValueError:
                        logger.error(f"Could not parse score from result: {result}")
                        score = None
                
                return (idx, judge_id, score)
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed for {judge_id} on example {idx}: {e}")
                    logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to score example {idx} with {judge_id} after {self.max_retries} attempts: {e}")
                    return (idx, judge_id, None)
    
    async def score_batch_async(
        self,
        judge_variants: Dict[str, object],
        examples: List[Dict],
        start_idx: int = 0
    ) -> Dict[str, List[float]]:
        """
        Score a batch of examples with all judge variants asynchronously.
        
        Args:
            judge_variants: Dictionary of judge variant IDs to objects
            examples: List of examples to score
            start_idx: Starting index for tracking
            
        Returns:
            Dictionary mapping judge IDs to lists of scores
        """
        tasks = []
        
        # Create tasks for all judge-example combinations
        for i, example in enumerate(examples):
            # Handle both dict and DataFrame row formats
            if isinstance(example, dict):
                prompt = example.get('instruction', example.get('question', example.get('prompt', '')))
                response = example.get('answer', example.get('response', ''))
            else:
                # Fallback for other formats
                prompt = str(example.get('instruction', '')) if hasattr(example, 'get') else ''
                response = str(example.get('answer', '')) if hasattr(example, 'get') else ''
            
            for judge_id in judge_variants:
                task = self.score_single_example(
                    judge_variants=judge_variants,
                    judge_id=judge_id,
                    prompt=prompt,
                    response=response,
                    idx=start_idx + i
                )
                tasks.append(task)
        
        # Run tasks in batches to avoid overwhelming the API
        results = []
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            
            # Small delay between batches
            if i + self.batch_size < len(tasks):
                await asyncio.sleep(0.5)
        
        # Organize results by judge
        scores_by_judge = {judge_id: [None] * len(examples) for judge_id in judge_variants}
        
        for idx, judge_id, score in results:
            if score is not None:
                relative_idx = idx - start_idx
                scores_by_judge[judge_id][relative_idx] = score
        
        return scores_by_judge
    
    def score_examples(
        self,
        judge_variants: Dict[str, object],
        n_examples: Optional[int] = None,
        save_checkpoint: bool = True,
        checkpoint_interval: int = 100
    ) -> pd.DataFrame:
        """
        Score examples through all judge variants.
        
        Args:
            judge_variants: Dictionary of judge variant IDs to objects
            n_examples: Number of examples to score (None = all)
            save_checkpoint: Whether to save intermediate results
            checkpoint_interval: How often to save checkpoints
            
        Returns:
            DataFrame with scores from all judge variants
        """
        # Prepare examples
        if n_examples:
            examples = self.data[:n_examples]
        else:
            examples = self.data
        
        logger.info(f"Scoring {len(examples)} examples with {len(judge_variants)} judge variants")
        
        # Initialize results storage
        all_scores = {judge_id: [] for judge_id in judge_variants}
        
        # Process in chunks for checkpointing
        chunk_size = checkpoint_interval
        n_chunks = (len(examples) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(examples))
            chunk_examples = examples[start_idx:end_idx]
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{n_chunks} (examples {start_idx}-{end_idx})")
            
            # Score the chunk
            loop = asyncio.get_event_loop()
            chunk_scores = loop.run_until_complete(
                self.score_batch_async(judge_variants, chunk_examples, start_idx)
            )
            
            # Accumulate scores
            for judge_id, scores in chunk_scores.items():
                all_scores[judge_id].extend(scores)
            
            # Save checkpoint
            if save_checkpoint and (chunk_idx + 1) % 1 == 0:
                self._save_checkpoint(all_scores, chunk_idx + 1)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_scores)
        df['example_idx'] = range(len(df))
        
        # Add metadata
        prompts = []
        responses = []
        ground_truths = []
        
        for ex in examples:
            if isinstance(ex, dict):
                prompts.append(ex.get('instruction', ex.get('question', ex.get('prompt', ''))))
                responses.append(ex.get('answer', ex.get('response', '')))
                ground_truths.append(ex.get('human_feedback_score', ex.get('human_score', None)))
            else:
                prompts.append('')
                responses.append('')
                ground_truths.append(None)
        
        df['prompt'] = prompts
        df['response'] = responses
        
        # Add ground truth if available
        if any(gt is not None for gt in ground_truths):
            df['ground_truth'] = ground_truths
        
        self.scores = df
        return df
    
    def _save_checkpoint(self, scores: Dict, chunk_idx: int):
        """Save intermediate scoring results."""
        checkpoint_path = Path(f"checkpoint_scores_chunk_{chunk_idx}.pkl")
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(scores, f)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def analyze_score_consistency(self) -> Dict:
        """
        Analyze consistency of scores across judge variants.
        
        Returns:
            Dictionary with consistency metrics
        """
        if self.scores.empty:
            logger.error("No scores available. Run score_examples first.")
            return {}
        
        # Group columns by base judge
        base_judges = {}
        for col in self.scores.columns:
            if '-' in col and col != 'example_idx':
                base_name = col.rsplit('-', 1)[0]
                if base_name not in base_judges:
                    base_judges[base_name] = []
                base_judges[base_name].append(col)
        
        # Calculate consistency metrics for each base judge
        consistency_metrics = {}
        
        for base_judge, variant_cols in base_judges.items():
            if len(variant_cols) < 2:
                continue
            
            # Get scores for all variants of this judge
            variant_scores = self.scores[variant_cols].values
            
            # Calculate metrics
            consistency_metrics[base_judge] = {
                'mean_variance': np.nanvar(variant_scores, axis=1).mean(),
                'mean_std': np.nanstd(variant_scores, axis=1).mean(),
                'max_variance': np.nanvar(variant_scores, axis=1).max(),
                'correlation_matrix': np.corrcoef(variant_scores.T),
                'n_variants': len(variant_cols)
            }
        
        return consistency_metrics
    
    def save_results(self, output_path: str):
        """Save scoring results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            self.scores.to_csv(output_path, index=False)
        elif output_path.suffix == '.pkl':
            with open(output_path, 'wb') as f:
                pickle.dump(self.scores, f)
        else:
            # Default to pickle
            with open(output_path.with_suffix('.pkl'), 'wb') as f:
                pickle.dump(self.scores, f)
        
        logger.info(f"Saved results to {output_path}")


def main():
    """Main entry point for testing the scoring pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Score examples through judge variants")
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--n-examples', type=int, default=10, 
                       help='Number of examples to score')
    parser.add_argument('--output', default='variant_scores.pkl',
                       help='Output file path')
    parser.add_argument('--judges', nargs='+', 
                       help='Specific judge IDs to test')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MultiRubricScoringPipeline(
        data_path=args.data,
        config_path=args.config
    )
    
    # Prepare judges
    judge_ids = args.judges if args.judges else ['harmlessness-judge']  # Test with one by default
    variants = pipeline.prepare_judges(judge_ids=judge_ids)
    
    # Score examples
    scores_df = pipeline.score_examples(
        judge_variants=variants,
        n_examples=args.n_examples
    )
    
    # Analyze consistency
    consistency = pipeline.analyze_score_consistency()
    
    print("\nConsistency Analysis:")
    for judge, metrics in consistency.items():
        print(f"\n{judge}:")
        print(f"  Mean variance: {metrics['mean_variance']:.4f}")
        print(f"  Mean std dev: {metrics['mean_std']:.4f}")
        print(f"  Max variance: {metrics['max_variance']:.4f}")
    
    # Save results
    pipeline.save_results(args.output)


if __name__ == "__main__":
    main()