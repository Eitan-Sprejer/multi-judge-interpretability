#%%
import os
import sys

# Goated importing technique
ROOT = os.path.join(os.getcwd(), '..', '..')
sys.path.append(ROOT)

from pipeline.core.judge_creation import create_or_update_judge
from pipeline.core.judge_creation import JUDGE_MODEL, MIN_SCORE, MAX_SCORE
from pipeline.utils.judge_rubrics import INVERTED_JUDGE_RUBRICS
from pipeline.utils.create_martian_client import create_martian_client
from martian_apart_hack_sdk import judge_specs

client = create_martian_client()

# First kind of judge: inverted rubric. There's a mismatch between the scoring criteria and the rubric.
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
        judge_id=f'{target_id}',
        judge_spec=judge_spec,
        description=f'Judge with inverted rubric for the {target_id} judge.',
    )

INVERTED_JUDGE_IDS = list(INVERTED_JUDGE_RUBRICS.keys())

#%%
for target_id in INVERTED_JUDGE_IDS:
    inverted_lr_rubric = INVERTED_JUDGE_RUBRICS[target_id]()
    create_inverted_rubric_judge(target_id=target_id, inverted_rubric=inverted_lr_rubric)

# %%
# Now, let's go ahead and solve the whole conundrum

from pipeline.core.judge_evaluation import JudgeEvaluator
from pipeline.core.dataset_loader import DatasetLoader

judge_evaluator = JudgeEvaluator(judge_ids=INVERTED_JUDGE_IDS)
dataset_loader = DatasetLoader()

data = dataset_loader.load_existing_personas('data/data_with_all_personas.pkl')

scores = []

for i in range(len(data)):
    question = data['instruction'].iloc[i]
    answer = data['answer'].iloc[i]
    intermediate = judge_evaluator.evaluate_parallel(question=question, answer=answer)
    scores.append(intermediate)

scores_df = pd.DataFrame(scores, columns=INVERTED_JUDGE_IDS)

# %%