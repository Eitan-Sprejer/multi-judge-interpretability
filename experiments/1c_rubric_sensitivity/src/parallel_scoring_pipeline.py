"""
Fully Parallelized Scoring Pipeline

This module implements efficient parallelization at multiple levels:
1. Across judge-variant pairs (40 parallel streams)
2. Within each judge-variant evaluation (batched examples)
"""

import asyncio
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm.asyncio import tqdm

from variant_judge_pipeline import VariantJudgePipeline
from pipeline.utils.judge_rubrics import JUDGE_RUBRICS

logger = logging.getLogger(__name__)


class ParallelScoringPipeline:
    """
    Fully parallelized pipeline with multi-level concurrency:
    - Level 1: Parallel across judge-variant pairs
    - Level 2: Parallel across examples within each judge
    """
    
    def __init__(
        self,
        data_path: str,
        max_workers: int = 10,
        batch_size: int = 50,
        use_real_api: bool = True
    ):
        """
        Initialize the parallel scoring pipeline.
        
        Args:
            data_path: Path to dataset
            max_workers: Total parallel workers available
            batch_size: Examples to process in each batch
            use_real_api: Whether to use real API calls
        """
        self.data_path = Path(data_path)
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.use_real_api = use_real_api
        
        # Load data
        with open(self.data_path, 'rb') as f:
            self.df = pickle.load(f)
        
        # Configuration
        self.judge_names = list(JUDGE_RUBRICS.keys())
        self.variant_types = ['strict', 'lenient', 'bottom_heavy', 'top_heavy']
        
        # Calculate optimal worker distribution
        self.n_judge_variants = len(self.judge_names) * len(self.variant_types)  # 40
        self.workers_per_variant = max(1, max_workers // min(10, self.n_judge_variants))
        
        logger.info(f"Parallelization config:")
        logger.info(f"  Total workers: {max_workers}")
        logger.info(f"  Judge-variant pairs: {self.n_judge_variants}")
        logger.info(f"  Workers per variant: {self.workers_per_variant}")
        logger.info(f"  Batch size: {batch_size}")
    
    async def collect_all_scores_parallel(
        self,
        n_examples: int,
        cache_path: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Collect all variant scores with full parallelization.
        
        Total API calls: 4 variants × 10 judges × n_examples
        Parallelization: All 40 judge-variant pairs processed concurrently
        
        Args:
            n_examples: Number of examples to evaluate
            cache_path: Optional path to cache results
            
        Returns:
            Dictionary of judge-variant scores
        """
        # Check cache first
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached scores from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        examples_df = self.df.head(n_examples)
        variant_scores = {}
        
        if self.use_real_api:
            logger.info(f"Starting PARALLEL evaluation of {self.n_judge_variants} judge-variants")
            logger.info(f"Total API calls: {self.n_judge_variants * n_examples:,}")
            
            # Initialize API pipeline
            pipeline = VariantJudgePipeline(
                data_path=str(self.data_path),
                max_workers=self.workers_per_variant
            )
            
            # Create all evaluation tasks
            tasks = []
            for variant_type in self.variant_types:
                for judge_name in self.judge_names:
                    key = f"{judge_name}_{variant_type}"
                    task = self._evaluate_judge_variant(
                        pipeline, examples_df, judge_name, variant_type, key
                    )
                    tasks.append(task)
            
            # Run all tasks concurrently with progress bar
            logger.info("Executing all judge-variant evaluations in parallel...")
            
            # Use asyncio.gather for true parallel execution
            results = await tqdm.gather(*tasks, desc="Judge-Variants")
            
            # Store results
            for key, scores in results:
                variant_scores[key] = scores
                
        else:
            logger.info("Using simulation mode (no API calls)")
            variant_scores = self._simulate_all_scores(examples_df)
        
        # Add original scores from dataset
        variant_scores.update(self._extract_original_scores(examples_df))
        
        # Save cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(variant_scores, f)
            logger.info(f"Cached scores saved to {cache_path}")
        
        return variant_scores
    
    async def _evaluate_judge_variant(
        self,
        pipeline: VariantJudgePipeline,
        examples_df: pd.DataFrame,
        judge_name: str,
        variant_type: str,
        key: str
    ) -> Tuple[str, np.ndarray]:
        """
        Evaluate a single judge-variant combination.
        
        This runs concurrently with other judge-variants.
        """
        try:
            # Create variant judge once
            variant_judge_id = pipeline.create_variant_judge(
                judge_name, variant_type,
                variant_suffix=f"exp_{variant_type}"
            )
            
            if not variant_judge_id:
                logger.error(f"Failed to create {key}")
                return key, np.full(len(examples_df), 2.0)
            
            # Evaluate examples in batches
            scores = await self._evaluate_examples_batched(
                pipeline, examples_df, variant_judge_id, key
            )
            
            logger.info(f"✓ Completed {key}")
            return key, scores
            
        except Exception as e:
            logger.error(f"Failed {key}: {e}")
            return key, np.full(len(examples_df), 2.0)
    
    async def _evaluate_examples_batched(
        self,
        pipeline: VariantJudgePipeline,
        examples_df: pd.DataFrame,
        judge_id: str,
        key: str
    ) -> np.ndarray:
        """
        Evaluate examples in batches for a single judge.
        """
        n_examples = len(examples_df)
        scores = np.zeros(n_examples)
        
        # Process in batches
        for batch_start in range(0, n_examples, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_examples)
            batch_df = examples_df.iloc[batch_start:batch_end]
            
            # Evaluate batch concurrently
            batch_scores = await self._evaluate_batch(
                pipeline, batch_df, judge_id
            )
            
            scores[batch_start:batch_end] = batch_scores
        
        return scores
    
    async def _evaluate_batch(
        self,
        pipeline: VariantJudgePipeline,
        batch_df: pd.DataFrame,
        judge_id: str
    ) -> np.ndarray:
        """
        Evaluate a batch of examples concurrently.
        """
        loop = asyncio.get_event_loop()
        
        # Create evaluation tasks for batch
        with ThreadPoolExecutor(max_workers=self.workers_per_variant) as executor:
            futures = []
            for _, row in batch_df.iterrows():
                future = loop.run_in_executor(
                    executor,
                    pipeline.evaluate_with_judge,
                    row['instruction'],
                    row['answer'],
                    judge_id
                )
                futures.append(future)
            
            # Wait for all evaluations in batch
            batch_scores = await asyncio.gather(*futures)
        
        return np.array(batch_scores)
    
    def _extract_original_scores(self, examples_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract original judge scores from dataset."""
        original_scores = {}
        
        for judge_idx, judge_name in enumerate(self.judge_names):
            scores = []
            for _, row in examples_df.iterrows():
                if 'judge_scores' in row and isinstance(row['judge_scores'], list):
                    scores.append(row['judge_scores'][judge_idx])
                else:
                    scores.append(2.0)
            
            original_scores[f"{judge_name}_original"] = np.array(scores)
        
        return original_scores
    
    def _simulate_all_scores(self, examples_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Simulate scores for testing without API calls."""
        variant_scores = {}
        n_examples = len(examples_df)
        
        for judge_idx, judge_name in enumerate(self.judge_names):
            # Get original scores
            original = []
            for _, row in examples_df.iterrows():
                if 'judge_scores' in row and isinstance(row['judge_scores'], list):
                    original.append(row['judge_scores'][judge_idx])
                else:
                    original.append(2.0)
            original = np.array(original)
            
            # Simulate variants
            variant_scores[f"{judge_name}_strict"] = np.clip(
                original * 0.85 + np.random.normal(0, 0.1, n_examples), 0, 4
            )
            variant_scores[f"{judge_name}_lenient"] = np.clip(
                original * 1.15 + np.random.normal(0, 0.1, n_examples), 0, 4
            )
            variant_scores[f"{judge_name}_bottom_heavy"] = np.clip(
                original * 0.9 + np.random.normal(0, 0.08, n_examples), 0, 4
            )
            variant_scores[f"{judge_name}_top_heavy"] = np.clip(
                original * 1.1 + np.random.normal(0, 0.08, n_examples), 0, 4
            )
        
        return variant_scores
    
    def create_combination(
        self,
        variant_scores: Dict[str, np.ndarray],
        combination: Dict[str, str]
    ) -> pd.DataFrame:
        """Create a combination by mixing cached scores."""
        n_examples = len(next(iter(variant_scores.values())))
        scores_matrix = np.zeros((n_examples, len(self.judge_names)))
        
        for judge_idx, (judge_name, variant_type) in enumerate(combination.items()):
            key = f"{judge_name}_{variant_type}"
            if key in variant_scores:
                scores_matrix[:, judge_idx] = variant_scores[key]
            else:
                logger.warning(f"Missing {key}, using defaults")
                scores_matrix[:, judge_idx] = 2.0
        
        columns = [f"{j}_{v}" for j, v in combination.items()]
        return pd.DataFrame(scores_matrix, columns=columns)


async def run_parallel_experiment(
    data_path: str,
    n_examples: int = 1000,
    max_workers: int = 20,
    output_dir: Optional[str] = None
):
    """
    Run fully parallelized rubric sensitivity experiment.
    
    Features:
    - Parallel evaluation of all 40 judge-variant pairs
    - Batched processing within each judge
    - Efficient caching and score reuse
    """
    logger.info("="*60)
    logger.info("PARALLEL RUBRIC SENSITIVITY EXPERIMENT")
    logger.info("="*60)
    
    # Initialize pipeline
    pipeline = ParallelScoringPipeline(
        data_path=data_path,
        max_workers=max_workers,
        batch_size=50,
        use_real_api=True
    )
    
    # Output setup
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / "parallel_scores_cache.pkl"
    else:
        cache_path = None
    
    # Step 1: Collect all scores in parallel
    logger.info(f"Collecting scores for {n_examples} examples...")
    variant_scores = await pipeline.collect_all_scores_parallel(
        n_examples=n_examples,
        cache_path=cache_path
    )
    
    logger.info(f"Collected {len(variant_scores)} score arrays")
    
    # Step 2: Create combinations (no API calls)
    from optimized_combinations import generate_optimized_combinations
    combinations = generate_optimized_combinations()
    
    all_results = []
    for combo_info in combinations:
        combo_df = pipeline.create_combination(
            variant_scores, combo_info['combination']
        )
        combo_df['combination'] = combo_info['name']
        all_results.append(combo_df)
    
    final_results = pd.concat(all_results, ignore_index=True)
    
    # Save results
    if output_dir:
        results_path = output_dir / "parallel_experiment_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(final_results, f)
        logger.info(f"Results saved to {results_path}")
    
    logger.info("="*60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*60)
    
    return final_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../../dataset/data_with_judge_scores.pkl')
    parser.add_argument('--examples', type=int, default=10)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--output', default='../results_parallel_test')
    
    args = parser.parse_args()
    
    asyncio.run(run_parallel_experiment(
        data_path=args.data,
        n_examples=args.examples,
        max_workers=args.workers,
        output_dir=args.output
    ))