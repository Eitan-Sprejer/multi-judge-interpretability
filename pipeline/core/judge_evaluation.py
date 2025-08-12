"""
Judge Evaluation Pipeline

Evaluates question-answer pairs using the 10 specialized judges.
Handles parallel evaluation, rate limiting, and checkpoint saving.
"""

import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

from martian_apart_hack_sdk import martian_client, utils
from martian_apart_hack_sdk.models import llm_models
from openai.types.chat import (
    chat_completion,
    chat_completion_message,
)

from pipeline.utils.judge_rubrics import JUDGE_RUBRICS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Judge IDs
JUDGE_IDS = list(JUDGE_RUBRICS.keys())

# Default configuration
DEFAULT_MAX_WORKERS = 10
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_DELAY = 1.0


class JudgeEvaluator:
    """Handles evaluation of Q&A pairs using multiple judges."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the evaluator with Martian API client.
        
        Args:
            config_path: Optional path to configuration file
        """
        # Load configuration
        if config_path:
            logger.info(f"Loading config from {config_path}")
            config = utils.load_config()  # TODO: implement custom config loading
        else:
            config = utils.load_config()
        
        # Initialize client
        self.client = martian_client.MartianClient(
            api_url=config.api_url,
            api_key=config.api_key,
        )
        
        # Load judges
        self.judges = self._load_judges()
        
    def _load_judges(self) -> Dict[str, Any]:
        """Load all judges from the Martian API."""
        judges = {}
        for judge_id in JUDGE_IDS:
            try:
                judge = self.client.judges.get(judge_id=judge_id)
                judges[judge_id] = judge
                logger.info(f"✅ Loaded judge {judge_id}")
            except Exception as e:
                logger.error(f"❌ Failed to load judge {judge_id}: {e}")
        
        if len(judges) != len(JUDGE_IDS):
            logger.warning(f"Only loaded {len(judges)}/{len(JUDGE_IDS)} judges")
        
        return judges
    
    def evaluate_single(self, question: str, answer: str, judge_id: str) -> float:
        """
        Evaluate a single Q&A pair with a specific judge.
        
        Args:
            question: The user's question/instruction
            answer: The model's response
            judge_id: ID of the judge to use
            
        Returns:
            Score from the judge (0.0-4.0)
        """
        if judge_id not in self.judges:
            raise ValueError(f"Judge {judge_id} not loaded")
        
        judge = self.judges[judge_id]
        
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
            model="gpt-4o",
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
    
    def _evaluate_with_retry(
        self,
        args: Tuple[str, str, str],
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY
    ) -> Tuple[int, float]:
        """
        Evaluate with exponential backoff on failure.
        
        Args:
            args: Tuple of (question, answer, judge_id)
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            
        Returns:
            Tuple of (judge_index, score)
        """
        question, answer, judge_id = args
        judge_index = JUDGE_IDS.index(judge_id)
        
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                score = self.evaluate_single(question, answer, judge_id)
                return (judge_index, score)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to evaluate with {judge_id} after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed for {judge_id}, retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
        
        # Should not reach here
        raise RuntimeError(f"Failed to evaluate with {judge_id}")
    
    def evaluate_parallel(
        self,
        question: str,
        answer: str,
        max_workers: Optional[int] = None
    ) -> List[float]:
        """
        Evaluate a Q&A pair with all judges in parallel.
        
        Args:
            question: The user's question/instruction
            answer: The model's response
            max_workers: Number of parallel workers
            
        Returns:
            List of scores in judge order
        """
        scores = [0.0] * len(JUDGE_IDS)
        eval_args = [(question, answer, judge_id) for judge_id in JUDGE_IDS]
        
        with ThreadPoolExecutor(max_workers=max_workers or DEFAULT_MAX_WORKERS) as executor:
            for judge_index, score in executor.map(self._evaluate_with_retry, eval_args):
                scores[judge_index] = score
        
        return scores
    
    def evaluate_dataset(
        self,
        data: pd.DataFrame,
        question_col: str = "instruction",
        answer_col: str = "answer",
        output_path: Optional[Path] = None,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        max_workers: Optional[int] = None,
        resume_from: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate an entire dataset with all judges.
        
        Args:
            data: DataFrame with questions and answers
            question_col: Name of question column
            answer_col: Name of answer column
            output_path: Path to save final results
            checkpoint_dir: Directory for checkpoint files
            checkpoint_interval: Save checkpoint every N rows
            max_workers: Number of parallel workers
            resume_from: Resume from specific row index
            
        Returns:
            DataFrame with judge scores added
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results
        results = [None] * len(data)
        start_idx = resume_from or 0
        
        # Create tasks
        tasks = [
            (i, row[question_col], row[answer_col])
            for i, (_, row) in enumerate(data.iterrows())
            if i >= start_idx
        ]
        
        logger.info(f"Evaluating {len(tasks)} rows starting from index {start_idx}")
        
        completed = start_idx
        checkpoint_batch = start_idx // checkpoint_interval
        
        # Process with parallelization across rows
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            def eval_row(args):
                idx, question, answer = args
                scores = self.evaluate_parallel(question, answer, max_workers=max_workers)
                return idx, scores
            
            for idx, scores in executor.map(eval_row, tasks):
                results[idx] = scores
                completed += 1
                
                # Log progress
                if completed % 10 == 0:
                    logger.info(f"Completed {completed}/{len(data)} rows")
                
                # Save checkpoint
                if checkpoint_dir and completed % checkpoint_interval == 0:
                    checkpoint_file = checkpoint_dir / f"checkpoint_{checkpoint_batch:03d}.pkl"
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(results, f)
                    logger.info(f"Saved checkpoint to {checkpoint_file}")
                    checkpoint_batch += 1
        
        # Add scores to dataframe
        data['judge_scores'] = results
        
        # Save final results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved final results to {output_path}")
        
        return data


def load_checkpoint(checkpoint_dir: Path, batch_num: int) -> List:
    """
    Load a specific checkpoint file.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        batch_num: Checkpoint batch number to load
        
    Returns:
        List of results from checkpoint
    """
    checkpoint_file = checkpoint_dir / f"checkpoint_{batch_num:03d}.pkl"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_file} not found")
    
    with open(checkpoint_file, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"Loaded checkpoint from {checkpoint_file}")
    return results


def main():
    """Main entry point for judge evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Q&A pairs with judges")
    parser.add_argument('--input', required=True, help='Path to input pickle file')
    parser.add_argument('--output', required=True, help='Path to output pickle file')
    parser.add_argument('--checkpoint-dir', help='Directory for checkpoint files')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                        help='Save checkpoint every N rows')
    parser.add_argument('--max-workers', type=int, default=10,
                        help='Number of parallel workers')
    parser.add_argument('--resume-from', type=int,
                        help='Resume from specific row index')
    parser.add_argument('--config', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load input data
    logger.info(f"Loading data from {args.input}")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    
    # Convert to DataFrame if needed
    if isinstance(data, list):
        data = pd.DataFrame(data)
    
    logger.info(f"Loaded {len(data)} rows")
    
    # Initialize evaluator
    evaluator = JudgeEvaluator(config_path=args.config)
    
    # Evaluate dataset
    results = evaluator.evaluate_dataset(
        data,
        output_path=args.output,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        checkpoint_interval=args.checkpoint_interval,
        max_workers=args.max_workers,
        resume_from=args.resume_from
    )
    
    logger.info(f"Evaluation complete! Results saved to {args.output}")
    
    # Print summary statistics
    if 'judge_scores' in results.columns:
        scores_df = pd.DataFrame(results['judge_scores'].tolist(), columns=JUDGE_IDS)
        print("\nScore Statistics:")
        print(scores_df.describe())


if __name__ == "__main__":
    main()