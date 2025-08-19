"""
Variant Judge Pipeline with Real API Calls

This module creates judge variants with modified rubrics and evaluates them using
the Martian API. Unlike the simulated approach, this makes real API calls with
actual rubric variations.
"""

import asyncio
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

# Martian SDK imports
from martian_apart_hack_sdk import martian_client, utils
from martian_apart_hack_sdk.judge_specs import RubricJudgeSpec
from martian_apart_hack_sdk.models import llm_models
from openai.types.chat import (
    chat_completion,
    chat_completion_message,
)

# Import rubric utilities
from pipeline.utils.judge_rubrics import JUDGE_RUBRICS
from scoring_criteria_variations import ScoringCriteriaVariationGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VariantJudgePipeline:
    """Pipeline for creating and evaluating judge variants with real API calls."""
    
    def __init__(
        self,
        data_path: str,
        config_path: Optional[str] = None,
        max_workers: int = 5,  # Parallelization level
        max_retries: int = 3,
        cleanup_judges: bool = True  # Whether to delete created judges after
    ):
        """
        Initialize the variant judge pipeline.
        
        Args:
            data_path: Path to dataset with existing judge scores
            config_path: Optional path to configuration file
            max_workers: Number of parallel workers for API calls
            max_retries: Maximum retries for failed API calls
            cleanup_judges: Whether to delete created variant judges after evaluation
        """
        self.data_path = Path(data_path)
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.cleanup_judges = cleanup_judges
        
        # Load data
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            data_loaded = pickle.load(f)
            
        # Ensure we have a DataFrame
        if hasattr(data_loaded, 'to_dict'):
            self.df = data_loaded.copy()
        else:
            self.df = pd.DataFrame(data_loaded)
            
        logger.info(f"Loaded dataset with {len(self.df)} examples")
        
        # Initialize Martian client
        config = utils.load_config() if not config_path else utils.load_config(config_path)
        self.client = martian_client.MartianClient(
            api_url=config.api_url,
            api_key=config.api_key,
        )
        
        # Track created variant judges for cleanup
        self.created_judge_ids = []
        
        # Initialize variation generator
        self.variation_generator = ScoringCriteriaVariationGenerator()
        
        # Load original judges
        self.original_judges = self._load_original_judges()
        
    def _load_original_judges(self) -> Dict[str, Any]:
        """Load original judges from Martian API."""
        judges = {}
        for judge_id in JUDGE_RUBRICS.keys():
            try:
                judge = self.client.judges.get(judge_id=judge_id)
                judges[judge_id] = judge
                logger.info(f"✅ Loaded original judge {judge_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load judge {judge_id}: {e}")
        return judges
    
    def create_variant_judge(
        self,
        base_judge_name: str,
        variant_type: str,
        variant_suffix: str = None
    ) -> Optional[str]:
        """
        Create a variant judge with modified rubric.
        
        Args:
            base_judge_name: Name of the base judge (e.g., 'truthfulness-judge')
            variant_type: Type of variant ('strict', 'lenient', 'bottom_heavy', 'top_heavy')
            variant_suffix: Optional suffix for the variant judge ID
            
        Returns:
            ID of the created variant judge, or None if creation failed
        """
        # Get original rubric
        original_rubric_func = JUDGE_RUBRICS.get(base_judge_name)
        if not original_rubric_func:
            logger.error(f"Judge {base_judge_name} not found in JUDGE_RUBRICS")
            return None
            
        original_rubric = original_rubric_func()
        
        # Generate variant rubric
        variations = self.variation_generator.generate_variations(
            original_rubric, base_judge_name
        )
        
        if variant_type not in variations:
            logger.error(f"Variant type {variant_type} not found for {base_judge_name}")
            return None
            
        variant_rubric = variations[variant_type]
        
        # Create unique ID for variant judge
        suffix = variant_suffix or f"{variant_type}_{int(time.time())}"
        variant_judge_id = f"{base_judge_name}_{suffix}"
        
        try:
            # Check if judge already exists
            existing_judge = None
            try:
                existing_judge = self.client.judges.get(judge_id=variant_judge_id)
            except:
                pass  # Judge doesn't exist, which is what we want
                
            if existing_judge:
                logger.info(f"Judge {variant_judge_id} already exists, using existing")
                return variant_judge_id
            
            # Create judge spec with variant rubric
            judge_spec = RubricJudgeSpec(
                model_type="rubric_judge",
                rubric=variant_rubric,
                model=llm_models.GPT_4O_MINI,  # Use same model as original
                min_score=0.0,
                max_score=4.0,
            )
            
            # Create the variant judge
            variant_judge = self.client.judges.create_judge(
                judge_id=variant_judge_id,
                judge_spec=judge_spec,
                description=f"Variant ({variant_type}) of {base_judge_name}"
            )
            
            logger.info(f"✅ Created variant judge: {variant_judge_id}")
            self.created_judge_ids.append(variant_judge_id)
            return variant_judge_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create variant judge {variant_judge_id}: {e}")
            return None
    
    def evaluate_with_judge(
        self,
        question: str,
        answer: str,
        judge_id: str,
        retry_count: int = 0
    ) -> float:
        """
        Evaluate a Q&A pair with a specific judge.
        
        Args:
            question: The user's question/instruction
            answer: The model's response
            judge_id: ID of the judge to use
            retry_count: Current retry attempt
            
        Returns:
            Score from the judge (0.0-4.0)
        """
        try:
            # Get the judge
            judge = self.client.judges.get(judge_id=judge_id)
            
            # Create the completion request
            completion_request = {
                "model": llm_models.GPT_4O_MINI,
                "messages": [{"role": "user", "content": question}],
            }
            
            # Create the completion response
            chat_completion_response = chat_completion.ChatCompletion(
                id="eval",
                choices=[
                    chat_completion.Choice(
                        finish_reason="stop",
                        index=0,
                        message=chat_completion_message.ChatCompletionMessage(
                            role="assistant",
                            content=answer,
                        ),
                    )
                ],
                created=0,
                model=llm_models.GPT_4O_MINI,
                object="chat.completion",
                service_tier=None,
            )
            
            # Evaluate with the judge
            evaluation_result = self.client.judges.evaluate(
                judge,
                completion_request=completion_request,
                completion_response=chat_completion_response,
            )
            
            return evaluation_result.score
            
        except Exception as e:
            if retry_count < self.max_retries:
                logger.warning(f"Retry {retry_count + 1} for {judge_id}: {e}")
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self.evaluate_with_judge(question, answer, judge_id, retry_count + 1)
            else:
                logger.error(f"Failed to evaluate with {judge_id} after {self.max_retries} retries: {e}")
                return 2.0  # Default fallback score
    
    def evaluate_combination_parallel(
        self,
        examples_df: pd.DataFrame,
        judge_variant_mapping: Dict[str, str]
    ) -> np.ndarray:
        """
        Evaluate examples with a judge combination using parallelization.
        
        Args:
            examples_df: DataFrame with examples to evaluate
            judge_variant_mapping: Dict mapping judge names to variant types
            
        Returns:
            Array of scores with shape (n_examples, n_judges)
        """
        n_examples = len(examples_df)
        n_judges = len(judge_variant_mapping)
        scores_matrix = np.zeros((n_examples, n_judges))
        
        # Create all evaluation tasks
        tasks = []
        for example_idx, (_, row) in enumerate(examples_df.iterrows()):
            question = row['instruction']
            answer = row['answer']
            
            for judge_idx, (judge_name, variant_type) in enumerate(judge_variant_mapping.items()):
                if variant_type == 'original':
                    # Use original judge
                    judge_id = judge_name
                else:
                    # Create or get variant judge
                    variant_judge_id = self.create_variant_judge(
                        judge_name, variant_type, 
                        variant_suffix=f"exp_{int(time.time())}"
                    )
                    if not variant_judge_id:
                        # Fall back to original if variant creation failed
                        judge_id = judge_name
                    else:
                        judge_id = variant_judge_id
                
                tasks.append((example_idx, judge_idx, question, answer, judge_id))
        
        # Execute tasks in parallel
        logger.info(f"Evaluating {len(tasks)} judge-example pairs with {self.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    self.evaluate_with_judge, 
                    task[2], task[3], task[4]
                ): task
                for task in tasks
            }
            
            # Collect results as they complete
            completed = 0
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Evaluating"):
                task = future_to_task[future]
                example_idx, judge_idx = task[0], task[1]
                
                try:
                    score = future.result()
                    scores_matrix[example_idx, judge_idx] = score
                except Exception as e:
                    logger.error(f"Failed evaluation for example {example_idx}, judge {judge_idx}: {e}")
                    scores_matrix[example_idx, judge_idx] = 2.0  # Default score
                
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"Completed {completed}/{len(tasks)} evaluations")
        
        return scores_matrix
    
    async def score_examples_async(
        self,
        judge_combinations: List[Dict],
        n_examples: int,
        save_checkpoint: bool = True,
        checkpoint_interval: int = 5
    ) -> pd.DataFrame:
        """
        Score examples through judge combinations with async support.
        
        Args:
            judge_combinations: List of judge combination dictionaries
            n_examples: Number of examples to score
            save_checkpoint: Whether to save checkpoints
            checkpoint_interval: How often to save checkpoints
            
        Returns:
            DataFrame with scores for all combinations
        """
        logger.info(f"Scoring {n_examples} examples through {len(judge_combinations)} combinations")
        
        # Limit examples if requested
        examples_to_score = self.df.head(n_examples).copy()
        
        # Initialize results DataFrame
        results_data = {
            'example_idx': list(range(n_examples)),
            'instruction': examples_to_score['instruction'].tolist(),
            'answer': examples_to_score['answer'].tolist(),
            'human_feedback': examples_to_score['human_feedback'].tolist()
        }
        
        # Process each judge combination
        for combo_idx, combo_info in enumerate(judge_combinations):
            combo_name = combo_info['name']
            judge_variant_mapping = combo_info['combination']
            
            logger.info(f"Processing combination {combo_idx + 1}/{len(judge_combinations)}: {combo_name}")
            
            # Get scores for this combination using parallel evaluation
            combo_scores = self.evaluate_combination_parallel(
                examples_to_score,
                judge_variant_mapping
            )
            
            # Add to results
            for judge_idx, (judge_name, variant_type) in enumerate(judge_variant_mapping.items()):
                col_name = f"{judge_name}_{variant_type}"
                results_data[col_name] = combo_scores[:, judge_idx]
            
            # Save checkpoint if requested
            if save_checkpoint and (combo_idx + 1) % checkpoint_interval == 0:
                checkpoint_path = self.data_path.parent / f"variant_checkpoint_{combo_idx + 1}.pkl"
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(results_data, f)
                logger.info(f"Saved checkpoint after {combo_idx + 1} combinations")
        
        # Create final DataFrame
        results_df = pd.DataFrame(results_data)
        logger.info(f"Created results DataFrame with shape {results_df.shape}")
        
        return results_df
    
    def cleanup_variant_judges(self):
        """Delete all created variant judges to clean up."""
        if not self.cleanup_judges:
            logger.info("Cleanup disabled, keeping variant judges")
            return
            
        logger.info(f"Cleaning up {len(self.created_judge_ids)} variant judges")
        
        for judge_id in self.created_judge_ids:
            try:
                # Note: Martian API doesn't have a delete method in the SDK
                # You might need to implement this or manually clean up
                logger.info(f"Would delete judge {judge_id} (delete not implemented in SDK)")
            except Exception as e:
                logger.error(f"Failed to delete judge {judge_id}: {e}")
        
        self.created_judge_ids.clear()


async def run_variant_experiment(
    data_path: str,
    judge_combinations: List[Dict],
    n_examples: int = 20,
    max_workers: int = 5,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Run the variant judge experiment with real API calls.
    
    Args:
        data_path: Path to dataset with judge scores
        judge_combinations: List of judge variant combinations to test
        n_examples: Number of examples to evaluate
        max_workers: Parallelization level
        output_path: Optional path to save results
        
    Returns:
        DataFrame with evaluation results
    """
    # Initialize pipeline
    pipeline = VariantJudgePipeline(
        data_path=data_path,
        max_workers=max_workers,
        cleanup_judges=True
    )
    
    try:
        # Run evaluation
        results = await pipeline.score_examples_async(
            judge_combinations=judge_combinations,
            n_examples=n_examples,
            save_checkpoint=True,
            checkpoint_interval=5
        )
        
        # Save results if path provided
        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Saved results to {output_path}")
        
        return results
        
    finally:
        # Clean up variant judges
        pipeline.cleanup_variant_judges()


if __name__ == "__main__":
    """Test the variant judge pipeline."""
    import argparse
    from scoring_criteria_variations import generate_judge_combinations
    
    parser = argparse.ArgumentParser(description="Test variant judge pipeline with real API calls")
    parser.add_argument('--data', required=True, help='Path to dataset')
    parser.add_argument('--examples', type=int, default=5, help='Number of examples')
    parser.add_argument('--workers', type=int, default=5, help='Parallel workers')
    parser.add_argument('--output', help='Output path for results')
    
    args = parser.parse_args()
    
    # Generate test combinations (just a few for testing)
    single_combos, _, _ = generate_judge_combinations()
    test_combinations = single_combos[:3]  # Just test 3 combinations
    
    # Run experiment
    asyncio.run(run_variant_experiment(
        data_path=args.data,
        judge_combinations=test_combinations,
        n_examples=args.examples,
        max_workers=args.workers,
        output_path=args.output
    ))