#!/usr/bin/env python3
"""
Evaluate (instruction, answer) rows with the four UF judges in parallel and save scores.

Input: pickle with DataFrame containing at least columns: instruction, answer
Output: pickle with a column `judges` that maps judge_id -> score (0-4)
Default output path: experiments/2_ultrafeedback_validation/results/uf_scores.pkl

Usage:
    python -m experiments.2_ultrafeedback_validation.src.uf_judge_evaluation --input base.pkl [--output PATH] [--max-workers 8]
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from martian_apart_hack_sdk import martian_client, utils
from martian_apart_hack_sdk.models import llm_models
from openai.types.chat import chat_completion, chat_completion_message
from pipeline.utils.judge_rubrics_uf import JUDGE_RUBRICS_UF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

JUDGE_IDS = list(JUDGE_RUBRICS_UF.keys())
DEFAULT_MAX_WORKERS = 8
DEFAULT_OUTPUT = Path("experiments/2_ultrafeedback_validation/results/uf_scores.pkl")


def build_completion(question: str, answer: str):
    completion_request = {
        "model": llm_models.GPT_4O_MINI,
        "messages": [{"role": "user", "content": question}],
    }
    completion_response = chat_completion.ChatCompletion(
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
    return completion_request, completion_response


def evaluate_row(client: martian_client.MartianClient, instruction: str, answer: str) -> Dict[str, float]:
    req, resp = build_completion(instruction, answer)
    scores: Dict[str, float] = {}
    for judge_id in JUDGE_IDS:
        judge = client.judges.get(judge_id)
        result = client.judges.evaluate(judge, completion_request=req, completion_response=resp)
        scores[judge_id] = float(result.score)
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate data with 4 UF judges")
    parser.add_argument('--input', required=True, help='Path to base (instruction, answer) data .pkl')
    parser.add_argument('--output', help='Path to output .pkl with judges column (default: experiments/2_ultrafeedback_validation/results/uf_scores.pkl)')
    parser.add_argument('--max-workers', type=int, default=DEFAULT_MAX_WORKERS)
    args = parser.parse_args()

    # Load data
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if 'instruction' not in data.columns or 'answer' not in data.columns:
        raise ValueError("Input data must contain 'instruction' and 'answer' columns")

    # Client
    config = utils.load_config()
    client = martian_client.MartianClient(api_url=config.api_url, api_key=config.api_key)

    # Evaluate in parallel across rows
    results: List[Optional[Dict[str, float]]] = [None] * len(data)
    rows = list(data[['instruction', 'answer']].itertuples(index=True, name=None))

    def _eval_one(t: Tuple[int, str, str]):
        idx, instruction, answer = t
        try:
            return idx, evaluate_row(client, instruction, answer)
        except Exception as e:
            logger.error(f"Row {idx} failed: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for idx, scores in ex.map(_eval_one, rows):
            results[idx] = scores

    data = data.copy()
    data['judges'] = results

    # Save
    out = Path(args.output) if args.output else DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved UF judge scores to {out}")


if __name__ == '__main__':
    main()
