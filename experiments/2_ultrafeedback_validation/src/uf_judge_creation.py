#!/usr/bin/env python3
"""
Create 4 UltraFeedback-style judges (Honesty, Truthfulness, Helpfulness, Instruction-Following).
Uses rubrics from pipeline.utils.judge_rubrics_uf.

Usage:
    python -m experiments.2_ultrafeedback_validation.src.uf_judge_creation [--config path]
"""
import logging
from typing import Optional
from martian_apart_hack_sdk import exceptions, judge_specs, martian_client, utils
from martian_apart_hack_sdk.models import llm_models
from pipeline.utils.judge_rubrics_uf import JUDGE_RUBRICS_UF, JUDGE_DESCRIPTIONS_UF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

JUDGE_MODEL = llm_models.GPT_4O_MINI
MIN_SCORE = 0.0
MAX_SCORE = 4.0


def create_or_update_judge(client: martian_client.MartianClient, judge_id: str, judge_spec: judge_specs.RubricJudgeSpec, description: str):
    try:
        judge = client.judges.create_judge(judge_id=judge_id, judge_spec=judge_spec, description=description)
        logger.info(f"Created judge {judge_id}")
        return judge
    except exceptions.ResourceAlreadyExistsError:
        try:
            existing_judge = client.judges.get(judge_id)
            client.judges.update_judge(judge_id=judge_id, judge_spec=judge_spec)
            logger.info(f"Updated judge {judge_id} to v{existing_judge.version}")
            return existing_judge
        except Exception as e:
            logger.error(f"Failed to update judge {judge_id}: {e}")
            return None
    except Exception as e:
        logger.error(f"Failed to create judge {judge_id}: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create or update UltraFeedback judges (4)")
    parser.add_argument('--config', help='Path to configuration file')
    args = parser.parse_args()

    # Load config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = utils.load_config()  # placeholder; uses default loader
    else:
        config = utils.load_config()

    client = martian_client.MartianClient(api_url=config.api_url, api_key=config.api_key)

    created = {}
    for judge_id, rubric_fn in JUDGE_RUBRICS_UF.items():
        rubric = rubric_fn()
        description = JUDGE_DESCRIPTIONS_UF[judge_id]
        spec = judge_specs.RubricJudgeSpec(
            model_type="rubric_judge",
            rubric=rubric,
            model=JUDGE_MODEL,
            min_score=MIN_SCORE,
            max_score=MAX_SCORE,
        )
        judge = create_or_update_judge(client, judge_id, spec, description)
        if judge:
            created[judge_id] = judge

    logger.info(f"Successfully created/updated {len(created)}/{len(JUDGE_RUBRICS_UF)} UF judges")


if __name__ == "__main__":
    main()
