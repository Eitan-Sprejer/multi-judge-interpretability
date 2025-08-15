#!/usr/bin/env python3
"""
Lambda.ai-backed UltraFeedback judge scoring (Honesty, Truthfulness, Helpfulness, Instruction-Following).

- Reads a pickle DataFrame with at least columns: instruction, answer
- For each row, queries a Lambda.ai model 4 times (one per rubric) to get scores in [0.0, 4.0]
- Adds 4 numeric columns: judge_honesty, judge_truthfulness, judge_helpfulness, judge_instruction_following
- Saves a checkpointed output pickle; supports resume

Usage:
  export OPEN_AI_API_KEY=...  # Lambda.ai key
  python -m experiments.2_ultrafeedback_validation.src.uf_lambda_judge_scoring \
    --input /path/to/base.pkl \
    --output experiments/2_ultrafeedback_validation/results/uf_lambda_scores.pkl \
    --model llama3.1-405b-instruct-fp8 \
    --concurrency 8 \
    --checkpoint-dir experiments/2_ultrafeedback_validation/results/ckpt \
    --resume
"""
import asyncio
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI

from pipeline.utils.judge_rubrics_uf import (
    get_honesty_rubric,
    get_truthfulness_rubric,
    get_helpfulness_rubric,
    get_instruction_following_rubric,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.1-405b-instruct-fp8"
DEFAULT_API_BASE = "https://api.lambda.ai/v1"
DEFAULT_OUTPUT = Path("experiments/2_ultrafeedback_validation/results/uf_scores.pkl")

RUBRICS = {
    "judge_honesty": get_honesty_rubric,
    "judge_truthfulness": get_truthfulness_rubric,
    "judge_helpfulness": get_helpfulness_rubric,
    "judge_instruction_following": get_instruction_following_rubric,
}


def parse_score(text: str) -> Optional[float]:
    """Extract a float in [0,4] from model output; clamp into range if needed."""
    if text is None:
        return None
    # Find first float-like number
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except Exception:
        return None
    # Clamp to [0,4]
    val = max(0.0, min(4.0, val))
    # Round to one decimal place as rubric specifies
    return round(val, 1)


@dataclass
class ScoringConfig:
    model: str = DEFAULT_MODEL
    api_base: str = DEFAULT_API_BASE
    concurrency: int = 8
    checkpoint_dir: Optional[Path] = None
    resume: bool = False


class LambdaUFScorer:
    def __init__(self, config: ScoringConfig):
        api_key = os.getenv("OPEN_AI_API_KEY")
        if not api_key:
            raise ValueError("OPEN_AI_API_KEY environment variable not set for Lambda.ai")
        self.client = AsyncOpenAI(api_key=api_key, base_url=config.api_base)
        self.model = config.model
        self.semaphore = asyncio.Semaphore(config.concurrency)
        self.checkpoint_dir = config.checkpoint_dir
        self.resume = config.resume

    async def _score_with_rubric(self, rubric: str, instruction: str, answer: str) -> Optional[float]:
        system = rubric
        user = (
            f"==== ORIGINAL TASK ====\n{instruction}\n\n"
            f"==== CANDIDATE ANSWER ====\n{answer}\n\n"
            "Output ONLY a single number in [0.0, 4.0] with one decimal place."
        )
        async with self.semaphore:
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=10,
                )
                content = resp.choices[0].message.content if resp and resp.choices else None
                return parse_score(content)
            except Exception as e:
                logger.warning(f"Scoring failed: {e}")
                return None

    async def _score_row(self, idx: int, instruction: str, answer: str) -> Tuple[int, Dict[str, Optional[float]]]:
        tasks = []
        for col, fn in RUBRICS.items():
            tasks.append(self._score_with_rubric(fn(), instruction, answer))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        scores: Dict[str, Optional[float]] = {}
        for (col, _), val in zip(RUBRICS.items(), results):
            if isinstance(val, Exception):
                scores[col] = None
            else:
                scores[col] = val
        return idx, scores

    def _load_checkpoint(self, total_rows: int) -> List[Optional[Dict[str, Optional[float]]]]:
        if not self.checkpoint_dir:
            return [None] * total_rows
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = self.checkpoint_dir / "scores_checkpoint.pkl"
        if self.resume and ckpt_file.exists():
            try:
                with open(ckpt_file, "rb") as f:
                    arr = pickle.load(f)
                if isinstance(arr, list) and len(arr) == total_rows:
                    logger.info("Resumed from checkpoint")
                    return arr
            except Exception:
                logger.warning("Failed to load checkpoint; starting fresh")
        return [None] * total_rows

    def _save_checkpoint(self, arr: List[Optional[Dict[str, Optional[float]]]]):
        if not self.checkpoint_dir:
            return
        ckpt_file = self.checkpoint_dir / "scores_checkpoint.pkl"
        try:
            with open(ckpt_file, "wb") as f:
                pickle.dump(arr, f)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    async def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not {'instruction', 'answer'}.issubset(df.columns):
            raise ValueError("Input DataFrame must contain 'instruction' and 'answer' columns")

        results: List[Optional[Dict[str, Optional[float]]]] = self._load_checkpoint(len(df))

        async def producer():
            for idx, instr, ans in df[['instruction', 'answer']].itertuples(index=True, name=None):
                if results[idx] is None:
                    yield idx, instr, ans

        async def runner():
            pending = []
            batch = 0
            async for idx, instr, ans in producer():
                pending.append(self._score_row(idx, instr, ans))
                if len(pending) >= 50:  # batch submissions
                    for i, scores in await asyncio.gather(*pending):
                        results[i] = scores
                    pending = []
                    batch += 1
                    if batch % 2 == 0:  # checkpoint every ~100 rows
                        self._save_checkpoint(results)
            if pending:
                for i, scores in await asyncio.gather(*pending):
                    results[i] = scores
                self._save_checkpoint(results)

        await runner()

        # Write columns
        out = df.copy()
        for col in RUBRICS.keys():
            out[col] = [r.get(col) if isinstance(r, dict) else None for r in results]
        return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Lambda.ai UF judge scoring")
    parser.add_argument('--input', required=True, help='Input .pkl with instruction, answer')
    parser.add_argument('--output', help='Output .pkl with 4 judge columns (default: experiments/2_ultrafeedback_validation/results/uf_scores.pkl)')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--checkpoint-dir', help='Directory for checkpoints')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    # Load input
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    cfg = ScoringConfig(
        model=args.model,
        api_base=DEFAULT_API_BASE,
        concurrency=args.concurrency,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        resume=args.resume,
    )
    scorer = LambdaUFScorer(cfg)

    scored_df = asyncio.run(scorer.score_dataframe(data))

    out_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(scored_df, f)
    logger.info(f"Saved scored data to {out_path}")


if __name__ == '__main__':
    main()
