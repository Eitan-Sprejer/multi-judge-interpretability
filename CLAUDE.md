# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Judge Interpretability is an AI safety research framework for evaluating model outputs using multiple specialized judges and aggregation models. This project implements a complete pipeline from judge creation through training interpretable aggregation models (GAM and MLP) to combine judge scores with human feedback.

The project was developed for the Apart x Martian Mechanistic Router Interpretability Hackathon where it won 2nd place, and is being expanded into a full research paper for NeurIPS Interpretability Workshop (submission deadline August 22, 2025).

### Core Problem
Current AI evaluation systems have critical limitations:
- **Single judges**: Limited perspectives, vulnerable to reward hacking
- **Naive averaging**: Assumes equal importance across all evaluation dimensions
- **Fixed rules**: Cannot capture diverse, context-dependent human preferences

### Key Innovation
Learning interpretable aggregation functions that capture varying human preference profiles when combining multiple judge evaluations, rather than using fixed combination rules. This allows the system to adapt to different contexts where safety, helpfulness, and accuracy are weighted differently by different users.

### Research Goals
1. **Robustness**: Build evaluation systems resistant to contaminated or adversarial judges
2. **Interpretability**: Understand how different evaluation dimensions contribute to final scores
3. **Scalability**: Enable efficient deployment of specialized evaluation models
4. **Safety**: Prevent exploitation of evaluation metrics through multi-dimensional assessment

## Development Environment Setup

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support (optional but recommended for training)
- Access to Martian API (for judge creation)
- OpenAI API key (for human feedback simulation)
- Lambda AI API access (used as proxy in simulate_annotators.py)

### Environment Setup
```bash
# Install main project dependencies
pip install -r requirements.txt

# Install the Martian SDK in editable mode
pip install -e martian-sdk-python

# Set up environment variables
# MARTIAN_API_KEY - for Martian API access
# OPEN_AI_API_KEY - for OpenAI/Lambda AI access
```

## Development Workflow

The project follows a sequential data processing pipeline. Run these steps in order:

### 1. Judge Creation
```bash
# Run the notebook to create 10 specialized judges
jupyter notebook project/judging/create_judges.ipynb
```

### 2. Ground Truth Dataset Creation
```bash
# Process UltraFeedback dataset and create base structure
jupyter notebook project/data_processing/create_ground_truth.ipynb
```

### 3. Human Feedback Simulation
```bash
# Generate reliable ground truth scores using diverse personas
python project/human_feedback_simulation/simulate_annotators.py
```

### 4. Judge Score Generation
```bash
# Process examples through all 10 judges to get scores
jupyter notebook project/inference/generate_dataset.ipynb
```

### 5. Model Training and Evaluation
```bash
# Train GAM model
jupyter notebook project/inference/new_train_gam.ipynb

# Train and evaluate MLP model
python project/inference/eval_nn.py --data <dataset_path> --model-path <model_path> --model-type mlp --hidden 64
```

### 6. Experimental Validation (Research Extensions)
```bash
# Test robustness with contaminated judges
python project/experiments/robustness_testing.py --contamination-rate 0.1

# Validate against UltraFeedback ground truth
jupyter notebook project/experiments/ultrafeedback_validation.ipynb

# Benchmark against Mixture of Judges
python project/experiments/moj_benchmark.py --model-type gam --comparison moj
```

## Key Architecture Components

### Martian SDK Integration
- **martian-sdk-python/**: Custom SDK for interacting with Martian API
- **MartianClient**: Main client class with organization, judges, and routers subclients
- **Judge creation**: 10 specialized judges for different evaluation criteria (harmlessness, privacy, factual accuracy, etc.)

### Data Processing Pipeline
- **dataset/**: Contains processed datasets at various pipeline stages
- **data.pkl**: Base UltraFeedback dataset structure
- **data_with_human_feedback.pickle**: Dataset with simulated human evaluations
- **data_with_judge_scores.pkl**: Final dataset with all judge scores

### Model Architecture
- **GAM (Generalized Additive Model)**: Highly interpretable with individual judge contribution analysis
- **MLP (Multi-Layer Perceptron)**: Neural network for potentially better performance with complex interactions
- **models/**: Contains trained model checkpoints (.pt files)

### Human Feedback Simulation
- **8 diverse personas**: Professor, CEO, Novelist, Architect, Therapist, Parent, Student, Data Scientist
- **Async processing**: Concurrent API calls with configurable concurrency limits
- **Error resilience**: Exponential backoff and checkpoint saving every 100 samples

## Important Code Patterns

### API Client Pattern
The Martian SDK uses a cached property pattern for client initialization:
```python
@functools.cached_property
def judges(self) -> judges_client.JudgesClient:
    return judges_client.JudgesClient(self._client, self._config)
```

### Async Processing with Error Handling
Human feedback simulation uses batch processing with graceful error handling:
```python
results = await asyncio.gather(*current_batch_tasks, return_exceptions=True)
for idx, result in enumerate(results):
    if isinstance(result, Exception):
        # Handle API failures gracefully
```

### Model Architecture Flexibility
The training scripts support both GAM and MLP architectures with consistent interfaces:
```python
if args.model_type == "gam":
    model = GAMAggregator(n_judges=n_features, hidden=args.hidden, monotone=monotone_param)
elif args.model_type == "mlp":
    model = SingleLayerMLP(n_judges=n_features, hidden_dim=args.hidden)
```

## Data Flow

1. **UltraFeedback Dataset** → Base question-answer pairs
2. **Human Simulators** → Ground truth preference scores (1-10)
3. **10 Specialized Judges** → Individual evaluation scores (1-4)
4. **Aggregation Models** → Final combined predictions

## Key Files to Understand

- `martian-sdk-python/src/martian_apart_hack_sdk/martian_client.py`: Main SDK client
- `project/human_feedback_simulation/simulate_annotators.py`: Human feedback simulation
- `project/inference/eval_nn.py`: Model training and evaluation
- `project/judging/create_judges.ipynb`: Judge creation and configuration

## Common Issues and Solutions

### API Rate Limiting
- Human feedback simulation includes concurrency limits (default: 10)
- Checkpoint saving prevents data loss during long runs
- Error handling preserves partial results

### Memory Management
- Large datasets are processed in batches
- Model evaluation uses configurable batch sizes
- Pickle files are used for efficient data persistence

### Model Training
- Input normalization is configurable but recommended
- Both CPU and CUDA execution supported
- Automatic checkpointing during training

## Testing and Validation

The project includes comprehensive evaluation metrics:
- **MSE (Mean Squared Error)**: Primary training objective
- **MAE (Mean Absolute Error)**: Interpretable error metric  
- **R² Score**: Explained variance measure
- **Naive baseline comparison**: Mean judge score with scaling

## Research Context & Theoretical Foundation

### Expert Orchestration Framework
The project builds on the vision of moving from monolithic AI systems to coordinated networks of specialized models with three primitives:
- **Judges**: Evaluate outputs across different dimensions (safety, accuracy, helpfulness)
- **Routers**: Select appropriate models for specific tasks
- **Orchestrators**: Coordinate multi-step workflows between components

### Mathematical Framework
- Judge function: `J: X × A → ℝᵈ` where X = prompts, A = answers, d = dimensions
- Aggregation function: `fθ: ℝᵗ → ℝ` combining t judges
- Optimization: `min θ L(fθ(J₁, J₂, ..., Jₜ), f*)`

### Experimental Approach
The research follows a four-track experimental design:

#### Track 1: Robustness Analysis (Priority)
- **Judge Contamination**: Test with deliberately flawed judges (inverted metrics, random noise)
- **Persona Poisoning**: Include "troll" personas that systematically misrate responses
- **Rubric Sensitivity**: Evaluate semantic robustness with differently phrased rubrics

#### Track 2: Ground Truth Validation (Priority)
- **UltraFeedback Integration**: Use multi-dimensional ratings as realistic ground truth
- **Baseline Comparison**: Compare against single-judge and naive averaging methods

#### Track 3: Architectural Comparisons (Secondary)
- **Mixture of Judges Benchmark**: Compare static aggregator against dynamic context-aware gating
- **Judge Self-Bias Analysis**: Test if LLM judges favor responses from same model family

#### Track 4: Interpretability Deep Dive (Secondary)
- **Learned Function Analysis**: Systematic interpretability using partial dependence plots
- **Sparse Additive Distillation**: Train minimal GAM to mimic MLP behavior

### Key Papers & References
- **Constitutional AI** (Bai et al.): Harmlessness from AI feedback
- **UltraFeedback** (Cui et al.): Scaled AI feedback for language models
- **Mixture of Judges** (Xu et al.): Context-aware judge combination
- **Expert Orchestration** (Kulkarni et al.): Composable AI systems framework

### Safety Benefits
This framework addresses AI safety by providing:
- **Transparent evaluation criteria**: Clear visibility into each judge's contribution
- **Individual judge contribution analysis**: Interpretable weights and decision paths
- **Robustness to individual judge failures**: System adapts when judges are contaminated
- **Flexible addition of new evaluation dimensions**: Extensible architecture for new safety criteria
- **Reduced gaming potential**: Multiple judges prevent exploitation of single metrics

## Performance Results

### Initial Hackathon Results
- **R² Score**: 57% (vs 24% baseline using mean judge scores)
- **MAE**: 1.35 (vs 1.83 baseline)
- **Dataset Size**: 10,000 examples from UltraFeedback
- **Training Time**: <5 minutes for GAM, <10 minutes for MLP

### Expected Research Improvements
- Target R² >70% with UltraFeedback ground truth
- Robustness to 20% judge contamination
- Interpretability metrics for judge contribution analysis
- Computational efficiency for real-time deployment

## Timeline & Milestones

- **Milestone 1 (Aug 12)**: Robustness Framework - contaminated judge test suite
- **Milestone 2 (Aug 16)**: UltraFeedback Integration - multi-dimensional training
- **Milestone 3 (Aug 20)**: Secondary Experiments - MoJ benchmark, interpretability
- **Milestone 4 (Aug 22)**: Paper Submission - NeurIPS Interpretability Workshop