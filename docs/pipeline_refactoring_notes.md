# Pipeline Refactoring Notes

## Current State Analysis

### Existing Notebooks (to be converted)
1. `project/judging/create_judges.ipynb` - Creates 10 specialized judges via Martian API
2. `project/data_processing/create_ground_truth.ipynb` - Processes UltraFeedback dataset
3. `project/inference/generate_dataset.ipynb` - Generates judge scores for dataset
4. `project/inference/new_train_gam.ipynb` - Trains GAM aggregator model
5. `project/experiments/aggregator_robustness.ipynb` - Robustness experiments
6. `project/experiments/improved_preference_data.ipynb` - Preference data improvements

### Existing Python Scripts
1. `project/human_feedback_simulation/simulate_annotators.py` - Simulates human feedback with 8 personas
2. `project/inference/eval_nn.py` - Trains and evaluates MLP/GAM models
3. `project/inference/utils.py` - Utility functions for model training

### Current Data Flow
```
UltraFeedback Dataset
    ↓
create_ground_truth.ipynb → data.pkl
    ↓
simulate_annotators.py → data_with_human_feedback.pickle
    ↓
generate_dataset.ipynb → data_with_judge_scores.pkl
    ↓
new_train_gam.ipynb / eval_nn.py → trained models
```

## Proposed 4-Pipeline Architecture

### Pipeline 1: Data Pipeline
**Purpose**: Handle raw data ingestion and preprocessing
**Input**: UltraFeedback dataset or other evaluation datasets
**Output**: Structured data ready for evaluation
**Components**:
- Data loader
- Data validator
- Data preprocessor
- Data versioning

### Pipeline 2: Persona Pipeline
**Purpose**: Simulate human feedback with diverse personas
**Input**: Preprocessed data from Pipeline 1
**Output**: Human preference scores (ground truth)
**Components**:
- Persona definitions (8 types)
- Async API orchestration
- Score aggregation
- Error handling and retries

### Pipeline 3: Judge Pipeline
**Purpose**: Create and manage specialized judges
**Input**: Preprocessed data from Pipeline 1
**Output**: Judge evaluation scores
**Components**:
- Judge creation (via Martian API)
- Judge evaluation
- Judge contamination (for robustness testing)
- Score collection

### Pipeline 4: Aggregator & Interpretability Pipeline
**Purpose**: Train aggregation models and analyze results
**Input**: Human scores (Pipeline 2) + Judge scores (Pipeline 3)
**Output**: Trained models, predictions, interpretability analysis
**Components**:
- GAM training
- MLP training
- Model evaluation
- Interpretability analysis
- Baseline comparisons

## Code to Analyze

### Priority Analysis Queue
- [ ] `create_judges.ipynb` - Understand judge creation process
- [ ] `create_ground_truth.ipynb` - Understand data preprocessing
- [ ] `generate_dataset.ipynb` - Understand judge scoring process
- [ ] `new_train_gam.ipynb` - Understand GAM training
- [ ] `simulate_annotators.py` - Already a script, review for integration

## Identified Issues & Obsolete Code

### Naming Inconsistencies
1. **Data files have inconsistent names**:
   - `question_and_answers.pkl` (from create_ground_truth.ipynb)
   - `questions_and_answers.pkl` (referenced in generate_dataset.ipynb)
   - `qas_with_scores.pkl` (from generate_dataset.ipynb)
   - `judges_annotated_data.pkl` (referenced in new_train_gam.ipynb)
   - `data.pkl`, `data_with_human_feedback.pickle`, `data_with_judge_scores.pkl` (in dataset/)
   - `.pkl` vs `.pickle` extension inconsistency

2. **The actual data flow is unclear** - files reference different names than what they create

### Missing Components
1. **No clear script that combines human feedback with judge scores**
2. **Progress files (progress_0.pkl, etc.) are temporary but not cleaned up**
3. **No config file for API keys, paths, or hyperparameters**

### Obsolete/Duplicate Notebooks
1. `quickstart_guide.ipynb` (root and in martian-sdk-python/)
2. `aggregator_robustness.ipynb` - experimental, not in main pipeline
3. `improved_preference_data.ipynb` - experimental, not in main pipeline

### Code Quality Issues
1. **Hardcoded values everywhere** (file paths, model names, hyperparameters)
2. **No error handling** in most notebooks
3. **No logging** - just print statements
4. **Manual checkpoint loading** (e.g., "load progress_99.pkl")

## Learning Log

### 1. create_judges.ipynb Analysis
**Purpose**: Creates 10 specialized judges via Martian API, each evaluating a different quality dimension.

**The 10 Judge Dimensions**:
1. **Harmlessness/Safety** - No harm, violence, self-injury encouragement (0-4 scale)
2. **Privacy/PII Integrity** - No PII exposure or confidential info leaks (0-4 scale)
3. **Factual Accuracy** - Verifiable, true, up-to-date statements (0-4 scale)
4. **Prompt Faithfulness/Relevance** - Directly answers prompt, no hallucination (0-4 scale)
5. **Calibration/Uncertainty** - Appropriate confidence expression, cites sources (0-4 scale)
6. **Bias & Fairness** - Avoids stereotypes and discrimination (0-4 scale)
7. **Reasoning Consistency** - No internal contradictions or logical errors (0-4 scale)
8. **Discourse Coherence** - Smooth idea flow, clear references (0-4 scale)
9. **Conciseness/Redundancy** - No unnecessary repetition or verbosity (0-4 scale)
10. **Stylistic & Formatting Fit** - Matches required style and formatting (0-4 scale)

**Key Components**:
- Uses `RubricJudgeSpec` with detailed scoring rubrics
- All judges use GPT-4O-MINI model
- Has create_or_update logic to handle existing judges
- Each judge has extensive rubric (100+ lines) with detailed scoring criteria

**Dependencies**: martian_apart_hack_sdk

### 2. create_ground_truth.ipynb Analysis
**Purpose**: Loads UltraFeedback dataset and flattens it into a simple structure.

**Process**:
1. Loads `openbmb/UltraFeedback` dataset (train split)
2. Flattens nested structure: each model's response becomes a separate row
3. Extracts: source, instruction, model, answer, score, grader
4. Saves as `question_and_answers.pkl`

**Key Insight**: This creates the base dataset structure but doesn't actually create "ground truth" - that comes from human simulation later.

### 3. generate_dataset.ipynb Analysis
**Purpose**: Runs all 10 judges on each question-answer pair to get judge scores.

**Process**:
1. Loads the 10 pre-created judges from Martian API
2. For each Q&A pair, gets scores from all 10 judges
3. Uses parallelization (ThreadPoolExecutor) for efficiency
4. Implements exponential backoff for API failures
5. Saves progress every 100 rows (progress_0.pkl, progress_1.pkl, etc.)
6. Final output: `qas_with_scores.pkl` with judge scores added

**Key Features**:
- Parallel evaluation across judges AND rows
- Resumable via progress files
- Handles API rate limiting with backoff

### 4. simulate_annotators.py (Already a Script)
**Purpose**: Simulates human feedback using 8 diverse personas via LLM.

**The 8 Personas**:
1. Professor - Academic rigor
2. CEO - Business practicality
3. Novelist - Creative expression
4. Architect - Structural clarity
5. Therapist - Empathy and support
6. Parent - Safety for children
7. Student - Learning clarity
8. Data Scientist - Technical accuracy

**Process**:
- Each persona evaluates Q&A pairs on 1-10 scale
- Uses async/await for concurrent API calls
- Saves checkpoint every 100 samples
- Output: `data_with_human_feedback.pickle`

## Refactoring Progress

### ✅ Completed
1. **judge_creation.py** - Converted from `create_judges.ipynb`
   - Clean functions for creating/updating judges
   - Separated rubrics into `judge_rubrics.py` for maintainability
   - Added CLI with --list and --get options
   - Proper logging and error handling

2. **judge_evaluation.py** - Converted from `generate_dataset.ipynb`
   - JudgeEvaluator class for clean API
   - Parallel evaluation with retry logic
   - Checkpoint saving and resuming
   - Progress tracking and statistics

### 🚧 In Progress
- Converting `new_train_gam.ipynb` to aggregator training script
- Moving `simulate_annotators.py` to pipelines folder

### 📝 To Do
- Create unified configuration system
- Build main orchestrator
- Clean up obsolete files

## Refactoring Plan

### Phase 1: Convert Notebooks to Scripts

#### Pipeline 1: Data Pipeline (`pipelines/data_pipeline.py`)
```python
# Functions from create_ground_truth.ipynb
def load_ultrafeedback_dataset()
def process_and_flatten_dataset()
def save_base_dataset()
```

#### Pipeline 2: Persona Pipeline (`pipelines/persona_pipeline.py`)
```python
# Already exists as simulate_annotators.py - just needs integration
# Move to pipelines/ and add config support
```

#### Pipeline 3: Judge Pipeline (`pipelines/judge_pipeline.py`)
```python
# Functions from create_judges.ipynb
def create_judge_rubrics()
def create_or_update_judges()
def get_judge_list()

# Functions from generate_dataset.ipynb  
def evaluate_with_judges()
def parallel_judge_evaluation()
def save_judge_scores()
```

#### Pipeline 4: Aggregator Pipeline (`pipelines/aggregator_pipeline.py`)
```python
# Functions from new_train_gam.ipynb
def prepare_training_data()
def train_gam_model()
def evaluate_model()
def generate_interpretability_plots()

# Functions from eval_nn.py
def train_mlp_model()
def compare_models()
```

### Phase 2: Create Unified Configuration

#### `config.yaml`
```yaml
api:
  martian_api_key: ${MARTIAN_API_KEY}
  openai_api_key: ${OPENAI_API_KEY}
  
data:
  ultrafeedback_split: "train"
  sample_size: 10000
  output_dir: "data/processed/"
  
judges:
  model: "gpt-4o-mini"
  score_range: [0.0, 4.0]
  parallel_workers: 10
  
personas:
  concurrency_limit: 10
  checkpoint_interval: 100
  
models:
  gam:
    n_splines: 10
    lambda: 0.6
  mlp:
    hidden_dim: 64
    learning_rate: 0.001
```

### Phase 3: Create Main Orchestrator

#### `run_pipeline.py`
```python
import argparse
from pipelines import data_pipeline, persona_pipeline, judge_pipeline, aggregator_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', choices=['data', 'personas', 'judges', 'aggregator', 'all'])
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', action='store_true')
    
    # Run selected pipeline(s)
```

### Phase 4: Clean Up

#### Files to Remove
- `quickstart_guide.ipynb` (both copies)
- Progress files (`progress_*.pkl`)
- Duplicate/inconsistent data files

#### Files to Standardize
- Rename all pickle files to use `.pkl` extension
- Use consistent naming: `01_base_data.pkl`, `02_human_scores.pkl`, `03_judge_scores.pkl`, `04_final_dataset.pkl`

### Implementation Order
1. **Start with Judge Pipeline** - Most self-contained
2. **Then Data Pipeline** - Simple transformation
3. **Then Persona Pipeline** - Already mostly done
4. **Finally Aggregator Pipeline** - Depends on all others

---