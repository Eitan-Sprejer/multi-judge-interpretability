# Experiment 2: UltraFeedback Validation (4-judge rubric)

This experiment validates a 4-judge setup inspired by UltraFeedback criteria: Honesty, Truthfulness, Helpfulness, and Instruction-Following.

## Flow

1. Create/Update 4 judges on Martian using UF rubrics
2. Evaluate your existing (prompt, answer) data with the 4 judges
3. Merge judge scores with your existing human feedback (personas) → ground-truth target
4. Train aggregators (MLP and GAM)

## Usage

```bash
# 1) Create the 4 judges (requires Martian SDK config & API key)
python -m experiments.2_ultrafeedback_validation.src.uf_judge_creation

# 2) Score your data with the 4 judges (replace paths with yours)
python -m experiments.2_ultrafeedback_validation.src.uf_judge_evaluation \
  --input /path/to/your/base_data.pkl \
  --output experiments/2_ultrafeedback_validation/results/judge_scores_uf.pkl \
  --max-workers 8

# 3) Run end-to-end (merge + train MLP/GAM)
python -m experiments.2_ultrafeedback_validation.run_experiment \
  --base /path/to/your/base_data.pkl \
  --human /path/to/your/human_feedback.pkl \
  --judges experiments/2_ultrafeedback_validation/results/judge_scores_uf.pkl \
  --out-dir experiments/2_ultrafeedback_validation/results \
  --train-gam --train-mlp
```

Notes:
- Base data should include at least `instruction` and `answer` columns.
- Human feedback file should contain persona-based ratings under `human_feedback` or a `human_score` column.
- Judge scores artifact will store a column named `judges` as a dict of judge→score.
- GAM training requires `pygam`; if unavailable, MLP will still run.

## Outputs

- `results/merged_uf.pkl`: merged base + human + judge scores
- `results/mlp_model.pt`: MLP aggregator
- `results/gam_model.pkl`: GAM aggregator (if `pygam` installed)
- Console logs include metrics (MSE/MAE/R²)

