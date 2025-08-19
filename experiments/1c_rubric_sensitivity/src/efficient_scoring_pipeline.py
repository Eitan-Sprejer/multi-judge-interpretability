"""
Efficient Scoring Pipeline with Score Reuse

This pipeline makes API calls only for unique judge-variant pairs,
then reuses those scores to create any combination without additional calls.
"""

import asyncio
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from variant_judge_pipeline import VariantJudgePipeline

logger = logging.getLogger(__name__)


class EfficientScoringPipeline:
    """
    Efficient pipeline that:
    1. Makes API calls for all unique judge-variant pairs
    2. Caches those scores
    3. Reuses scores to create any combination without additional API calls
    """
    
    def __init__(
        self,
        data_path: str,
        max_workers: int = 10,
        use_real_api: bool = True
    ):
        """
        Initialize the efficient scoring pipeline.
        
        Args:
            data_path: Path to dataset with existing judge scores
            max_workers: Number of parallel workers for API calls
            use_real_api: Whether to use real API calls or simulation
        """
        self.data_path = Path(data_path)
        self.max_workers = max_workers
        self.use_real_api = use_real_api
        
        # Load data
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.df = pickle.load(f)
        
        # Cache for variant scores
        self.variant_scores_cache = {}
        
        # Judge names
        from pipeline.utils.judge_rubrics import JUDGE_RUBRICS
        self.judge_names = list(JUDGE_RUBRICS.keys())
        
        # Variant types to generate
        self.variant_types = ['strict', 'lenient', 'bottom_heavy', 'top_heavy']
    
    async def collect_all_variant_scores(
        self,
        n_examples: int,
        cache_path: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Collect scores for all unique judge-variant pairs.
        
        This is the ONLY place where API calls are made.
        Total API calls: 4 variants × 10 judges × n_examples
        
        Args:
            n_examples: Number of examples to score
            cache_path: Optional path to save/load cache
            
        Returns:
            Dictionary mapping "judge_variant" to score arrays
        """
        # Check for existing cache
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached variant scores from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.variant_scores_cache = pickle.load(f)
            return self.variant_scores_cache
        
        logger.info(f"Collecting scores for {len(self.variant_types)} variants × {len(self.judge_names)} judges")
        logger.info(f"Total unique evaluations: {len(self.variant_types) * len(self.judge_names) * n_examples}")
        
        examples_df = self.df.head(n_examples)
        
        if self.use_real_api:
            # Use real API calls
            pipeline = VariantJudgePipeline(
                data_path=str(self.data_path),
                max_workers=self.max_workers
            )
            
            # Create all judge-variant tasks for parallel execution
            import asyncio
            tasks = []
            
            for variant_type in self.variant_types:
                for judge_name in self.judge_names:
                    # Create async task for each judge-variant pair
                    task = self._evaluate_variant_async(
                        pipeline, examples_df, judge_name, variant_type
                    )
                    tasks.append((f"{judge_name}_{variant_type}", task))
            
            logger.info(f"Evaluating {len(tasks)} judge-variant pairs in parallel...")
            
            # Execute all tasks in parallel with controlled concurrency
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def run_with_semaphore(cache_key, task):
                async with semaphore:
                    scores = await task
                    return cache_key, scores
            
            # Run all tasks
            results = await asyncio.gather(*[
                run_with_semaphore(cache_key, task) 
                for cache_key, task in tasks
            ])
            
            # Store results in cache
            for cache_key, scores in results:
                self.variant_scores_cache[cache_key] = scores
                logger.info(f"Completed: {cache_key}")
            
        else:
            # Use simulation for testing
            logger.info("Using simulated scores (no API calls)")
            self._simulate_variant_scores(examples_df)
        
        # Also store original scores (already in dataset)
        for judge_idx, judge_name in enumerate(self.judge_names):
            original_scores = []
            for _, row in examples_df.iterrows():
                if 'judge_scores' in row and isinstance(row['judge_scores'], list):
                    original_scores.append(row['judge_scores'][judge_idx])
                else:
                    original_scores.append(2.0)  # Default
            
            self.variant_scores_cache[f"{judge_name}_original"] = np.array(original_scores)
        
        # Save cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.variant_scores_cache, f)
            logger.info(f"Saved variant scores cache to {cache_path}")
        
        return self.variant_scores_cache
    
    async def _evaluate_variant_async(
        self,
        pipeline: VariantJudgePipeline,
        examples_df: pd.DataFrame,
        judge_name: str,
        variant_type: str
    ) -> np.ndarray:
        """Async wrapper for variant evaluation."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        # Run the synchronous evaluation in a thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            scores = await loop.run_in_executor(
                executor,
                self._evaluate_variant_sync,
                pipeline, examples_df, judge_name, variant_type
            )
        return scores
    
    def _evaluate_variant_sync(
        self,
        pipeline: VariantJudgePipeline,
        examples_df: pd.DataFrame,
        judge_name: str,
        variant_type: str
    ) -> np.ndarray:
        """Evaluate examples with a specific judge variant using parallelization."""
        # First create the variant judge once
        variant_judge_id = pipeline.create_variant_judge(
            judge_name, variant_type, 
            variant_suffix=f"exp_{variant_type}"
        )
        
        if not variant_judge_id:
            logger.warning(f"Failed to create variant judge {judge_name}_{variant_type}, using defaults")
            return np.full(len(examples_df), 2.0)
        
        # Now evaluate all examples in parallel with this judge
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        scores = [None] * len(examples_df)
        
        # Use a smaller pool for per-judge parallelization to avoid overload
        per_judge_workers = max(1, self.max_workers // 10)  # Divide workers among judges
        
        with ThreadPoolExecutor(max_workers=per_judge_workers) as executor:
            # Submit all evaluation tasks
            future_to_idx = {
                executor.submit(
                    pipeline.evaluate_with_judge,
                    row['instruction'],
                    row['answer'],
                    variant_judge_id
                ): idx
                for idx, (_, row) in enumerate(examples_df.iterrows())
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    scores[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Failed to evaluate example {idx}: {e}")
                    scores[idx] = 2.0  # Default fallback
        
        return np.array(scores)
    
    def _simulate_variant_scores(self, examples_df: pd.DataFrame):
        """Simulate variant scores for testing without API calls."""
        n_examples = len(examples_df)
        
        for judge_idx, judge_name in enumerate(self.judge_names):
            # Get original scores
            original_scores = []
            for _, row in examples_df.iterrows():
                if 'judge_scores' in row and isinstance(row['judge_scores'], list):
                    original_scores.append(row['judge_scores'][judge_idx])
                else:
                    original_scores.append(2.0)
            
            original_scores = np.array(original_scores)
            
            # Simulate variants
            self.variant_scores_cache[f"{judge_name}_strict"] = \
                np.clip(original_scores * 0.85 + np.random.normal(0, 0.1, n_examples), 0, 4)
            
            self.variant_scores_cache[f"{judge_name}_lenient"] = \
                np.clip(original_scores * 1.15 + np.random.normal(0, 0.1, n_examples), 0, 4)
            
            self.variant_scores_cache[f"{judge_name}_bottom_heavy"] = \
                np.clip(original_scores * 0.9 + np.random.normal(0, 0.08, n_examples), 0, 4)
            
            self.variant_scores_cache[f"{judge_name}_top_heavy"] = \
                np.clip(original_scores * 1.1 + np.random.normal(0, 0.08, n_examples), 0, 4)
    
    def create_combination_scores(
        self,
        combination: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Create scores for a specific combination by reusing cached scores.
        
        NO API CALLS - just combines existing scores!
        
        Args:
            combination: Dict mapping judge names to variant types
            
        Returns:
            DataFrame with scores for this combination
        """
        if not self.variant_scores_cache:
            raise ValueError("Must call collect_all_variant_scores first!")
        
        n_examples = len(next(iter(self.variant_scores_cache.values())))
        scores_matrix = np.zeros((n_examples, len(self.judge_names)))
        
        for judge_idx, (judge_name, variant_type) in enumerate(combination.items()):
            cache_key = f"{judge_name}_{variant_type}"
            
            if cache_key not in self.variant_scores_cache:
                logger.warning(f"Missing scores for {cache_key}, using defaults")
                scores_matrix[:, judge_idx] = 2.0
            else:
                scores_matrix[:, judge_idx] = self.variant_scores_cache[cache_key]
        
        # Create DataFrame with proper column names
        columns = [f"{judge}_{variant}" for judge, variant in combination.items()]
        return pd.DataFrame(scores_matrix, columns=columns)
    
    def create_all_combinations(
        self,
        combinations: List[Dict]
    ) -> pd.DataFrame:
        """
        Create scores for all combinations by reusing cached scores.
        
        Args:
            combinations: List of combination dictionaries
            
        Returns:
            DataFrame with all combination scores
        """
        all_results = []
        
        for combo_info in combinations:
            combo_name = combo_info['name']
            combination = combo_info['combination']
            
            logger.info(f"Creating combination: {combo_name}")
            combo_scores = self.create_combination_scores(combination)
            
            # Add metadata
            combo_scores['combination'] = combo_name
            all_results.append(combo_scores)
        
        # Combine all results
        return pd.concat(all_results, ignore_index=True)


async def run_efficient_experiment(
    data_path: str,
    n_examples: int = 1000,
    max_workers: int = 10,
    use_real_api: bool = True,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run the efficient rubric sensitivity experiment.
    
    Total API calls: 4 variants × 10 judges × n_examples
    (NOT multiplied by number of combinations!)
    
    Args:
        data_path: Path to dataset
        n_examples: Number of examples to evaluate
        max_workers: Parallel workers for API calls
        use_real_api: Whether to use real API or simulation
        output_dir: Optional output directory
        
    Returns:
        DataFrame with all combination scores
    """
    # Initialize pipeline
    pipeline = EfficientScoringPipeline(
        data_path=data_path,
        max_workers=max_workers,
        use_real_api=use_real_api
    )
    
    # Output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / "variant_scores_cache.pkl"
    else:
        cache_path = None
    
    # Step 1: Collect all variant scores (ONLY API calls here)
    logger.info("Step 1: Collecting all variant scores...")
    variant_scores = await pipeline.collect_all_variant_scores(
        n_examples=n_examples,
        cache_path=cache_path
    )
    
    logger.info(f"Collected {len(variant_scores)} unique judge-variant scores")
    
    # Step 2: Create combinations (NO API calls, just reuse scores)
    logger.info("Step 2: Creating combinations from cached scores...")
    
    from optimized_combinations import generate_optimized_combinations
    combinations = generate_optimized_combinations()
    
    all_scores = pipeline.create_all_combinations(combinations)
    
    # Save results
    if output_dir:
        results_path = output_dir / "all_combination_scores.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(all_scores, f)
        logger.info(f"Saved results to {results_path}")
    
    return all_scores


if __name__ == "__main__":
    """Test the efficient pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../../dataset/data_with_judge_scores.pkl')
    parser.add_argument('--examples', type=int, default=10)
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--output', default='../results_efficient_test')
    
    args = parser.parse_args()
    
    asyncio.run(run_efficient_experiment(
        data_path=args.data,
        n_examples=args.examples,
        use_real_api=not args.simulate,
        output_dir=args.output
    ))