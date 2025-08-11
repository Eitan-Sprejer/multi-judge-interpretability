# Pipeline Scripts

Each script in this folder can be run independently. They're designed to be simple and composable.

## Scripts

### judge_creation.py
Creates or updates 10 specialized judges via the Martian API.

```bash
# Create all judges
python pipelines/judge_creation.py

# List existing judges
python pipelines/judge_creation.py --list

# Get specific judge details
python pipelines/judge_creation.py --get harmlessness-judge
```

### judge_evaluation.py
Evaluates samples using the created judges.

```bash
# Evaluate a dataset
python pipelines/judge_evaluation.py --input data.pkl --output judge_scores.pkl

# Evaluate specific samples
python pipelines/judge_evaluation.py --input data.pkl --output scores.pkl --sample-size 100
```

### persona_simulation.py
Simulates human feedback using 8 diverse personas.

```bash
# Run persona simulation
python pipelines/persona_simulation.py --input data.pkl --output human_feedback.pkl

# With custom settings
python pipelines/persona_simulation.py --input data.pkl --output feedback.pkl --concurrency 5
```

### aggregator_training.py
Trains GAM and MLP models to aggregate judge scores.

```bash
# Train GAM model
python pipelines/aggregator_training.py --data merged_data.pkl --model-type gam

# Train MLP model
python pipelines/aggregator_training.py --data merged_data.pkl --model-type mlp --hidden 64

# Evaluate existing model
python pipelines/aggregator_training.py --data test_data.pkl --model-path model.pt --evaluate-only
```

## Utils

### utils/data_merger.py
Utility for merging data from different pipeline stages.

```python
from pipelines.utils import DataMerger

merger = DataMerger()
merger.load_base_data("base_data.pkl")
merger.load_human_feedback("human_feedback.pkl")
merger.load_judge_scores("judge_scores.pkl")
merged = merger.merge_datasets(output_path="final_dataset.pkl")
```

### utils/judge_rubrics.py
Contains the full rubrics and descriptions for all 10 judges.

```python
from pipelines.utils import JUDGE_RUBRICS, JUDGE_DESCRIPTIONS

# Get rubric for a specific judge
harmlessness_rubric = JUDGE_RUBRICS['harmlessness-judge']()
```

## Environment Variables

Set these before running the scripts:
- `MARTIAN_API_KEY`: For judge creation and evaluation
- `OPEN_AI_API_KEY`: For persona simulation (actually Lambda AI)

## Data Flow

1. Create judges with `judge_creation.py`
2. Prepare your base dataset (instruction + answer pairs)
3. Run `persona_simulation.py` to get human feedback
4. Run `judge_evaluation.py` to get judge scores
5. Use `DataMerger` to combine all scores
6. Train models with `aggregator_training.py`