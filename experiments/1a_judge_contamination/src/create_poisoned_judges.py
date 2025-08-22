#%%
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Goated importing technique
import os
import sys
ROOT = os.path.join(os.getcwd(), '..', '..')
sys.path.append(ROOT)

from pipeline.core.judge_creation import create_or_update_judge
from pipeline.core.judge_creation import JUDGE_MODEL, MIN_SCORE, MAX_SCORE
from pipeline.utils.judge_rubrics import INVERTED_JUDGE_RUBRICS
from pipeline.utils.create_martian_client import create_martian_client
from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.core.dataset_loader import DatasetLoader
from martian_apart_hack_sdk import judge_specs

# Create results directory
results_dir = Path("results") / datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir.mkdir(parents=True, exist_ok=True)

client = create_martian_client()

dataset_loader = DatasetLoader()
data = dataset_loader.load_existing_personas('data/data_with_all_personas.pkl')

def create_inverted_rubric_judge(target_id: str, inverted_rubric: str):
    judge_spec = judge_specs.RubricJudgeSpec(
        model_type="rubric_judge",
        rubric=inverted_rubric,
        model=JUDGE_MODEL,
        min_score=MIN_SCORE,
        max_score=MAX_SCORE,
    )

    create_or_update_judge(
        client=client,
        judge_id=f'inverted_{target_id}',
        judge_spec=judge_spec,
        description=f'Judge with inverted rubric for the {target_id} judge.',
    )

def create_inverted_rubric_judges():
    INVERTED_JUDGE_IDS = list(INVERTED_JUDGE_RUBRICS.keys())

    for target_id in INVERTED_JUDGE_IDS:
        inverted_rubric = INVERTED_JUDGE_RUBRICS[target_id]()
        create_inverted_rubric_judge(target_id=target_id, inverted_rubric=inverted_rubric)
        print(f"✅ Created inverted judge: inverted_{target_id}")

judge_evaluator = JudgeEvaluator(judge_ids=INVERTED_JUDGE_IDS)
