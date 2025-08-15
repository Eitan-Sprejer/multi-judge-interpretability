# Experiment 3b: Multi-LLM Judge Creation

## Overview

This experiment creates specialized judges using 5 different LLM providers, resulting in 50 total judges (10 rubric types × 5 LLM providers). The goal is to create a diverse set of judges that can evaluate responses across different quality dimensions.

## Judge Types

Each LLM provider will create 10 specialized judges based on the rubrics in `pipeline/utils/judge_rubrics.py`:

1. **Harmlessness Judge** - Evaluates safety and harm potential
2. **Privacy Judge** - Evaluates PII protection and privacy
3. **Factual Accuracy Judge** - Evaluates factual correctness
4. **Prompt Faithfulness Judge** - Evaluates adherence to user prompts
5. **Calibration Judge** - Evaluates uncertainty expression and confidence
6. **Bias & Fairness Judge** - Evaluates bias and discrimination
7. **Reasoning Consistency Judge** - Evaluates logical consistency
8. **Discourse Coherence Judge** - Evaluates text flow and coherence
9. **Conciseness Judge** - Evaluates information density and efficiency
10. **Style & Formatting Judge** - Evaluates stylistic appropriateness

## LLM Providers

The experiment will create judges using these 5 LLM providers:

1. **OpenAI** - GPT-4o-mini
2. **Anthropic** - Claude-3.5-sonnet
3. **Google** - Gemini-1.5-flash
4. **Together** - Llama-3.1-70B
5. **Meta** - Llama-3.1-8B

## Expected Output

- **50 judges total** (10 rubric types × 5 LLM providers)
- Each judge configured with the appropriate rubric and scoring scale (0.0-4.0)
- Judges ready for use in evaluation pipelines

## Structure

This experiment follows the standard structure from `experiments/README.md`:

```
3b_judge_self_bias_analysis/
├── src/                    # Core implementation
│   └── main_logic.py       # Judge creation logic
├── configs/
│   └── default_config.yaml # Main config
├── results/                # Auto-created outputs (json, reports, plots)
├── run_experiment.py       # Standard entry point
├── create_judges.py        # Back-compat wrapper
└── README.md
```

## Usage

Quick test (smoke):

```bash
cd experiments/3b_judge_self_bias_analysis
python run_experiment.py --quick
```

Full run:

```bash
python run_experiment.py --config configs/default_config.yaml
```

Back-compat:

```bash
python create_judges.py
```

## Dependencies

- Martian API access
- Required packages from `requirements.txt`
- Environment variables for API keys

## Next Steps

After creating the judges, they can be used in:
- Response evaluation pipelines
- Multi-judge aggregation experiments
- Bias analysis studies
- Quality assessment systems

## Results

Outputs are saved to `results/`:
- `[timestamp]_results.json`: ids and counts
- `reports/experiment_report.md`: summary
