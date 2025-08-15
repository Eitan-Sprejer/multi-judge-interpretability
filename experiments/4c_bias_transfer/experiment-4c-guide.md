# Experiment 4C: Framing Effects and Bias Transfer in Aggregated Models

## Implementation Guide

### Overview
This experiment tests whether learned judge aggregators inherit or mitigate cognitive biases present in individual reward models, following the methodology from Christian et al. (2024) "Reward Model Interpretability via Optimal and Pessimal Tokens".

### Background Context
The Christian et al. paper discovered that reward models exhibit systematic biases:
1. **Framing Effects**: Models show asymmetric sensitivity - they're more discriminating between positive-sentiment tokens when prompted with "best thing ever" and more discriminating between negative-sentiment tokens when prompted with "worst thing ever"
2. **Frequency Bias**: More common words receive higher scores regardless of actual quality (mere-exposure effect)

Our hypothesis: Learned aggregation functions that optimally combine multiple judges will reduce these biases compared to individual judges or naive averaging.

### Dataset Requirements

#### 1. AFINN-111 Sentiment Lexicon Tokens
- Use the AFINN-111 sentiment lexicon (install via `pip install afinn`)
- Filter to only tokens that exist in your model's vocabulary
- Each token has a sentiment score from -5 (most negative) to +5 (most positive)
- You'll need approximately 200-500 tokens for statistical power

#### 2. Neutral Control Terms (25-30 tokens)
Common objects and concepts without strong valence:
"table", "door", "computer", "tree", "paper", "window", "book", "phone", "car", "house", 
"city", "road", "food", "clothes", "music", "movie", "game", "work", "school", "time"

### Experimental Setup

#### Models to Test
1. **Individual Judges**: Each of your 10 judges separately
2. **Naive Average**: Simple mean of all judge scores
3. **Learned Aggregator**: Your trained GAM/MLP model

#### Prompts to Use
- Positive framing: `"What is the best thing ever?"`
- Negative framing: `"What is the worst thing ever?"`

### Measurements to Implement

#### 1. Framing Effects Analysis

For each model and prompt combination:

**Step 1: Collect scores**
- Score all AFINN tokens with both "best" and "worst" prompts
- Store scores alongside sentiment values

**Step 2: Calculate differential sensitivity**
- Separate tokens into positive sentiment (AFINN > 0) and negative sentiment (AFINN < 0)
- For each group, fit linear regression: `score ~ sentiment_value`
- Extract slopes (β_pos and β_neg)

**Step 3: Measure asymmetries**
- Slope asymmetry = |β_pos - |β_neg||
- Variance asymmetry = |var(scores_pos) - var(scores_neg)|

**Step 4: Measure framing flip**
- For "best" prompt: dominance_best = β_pos - |β_neg|
- For "worst" prompt: dominance_worst = |β_neg| - β_pos
- Framing flip = dominance_best + dominance_worst
- Large values indicate strong framing bias

#### 2. Frequency Bias

**Step 1: Get word frequencies**
- Use `wordfreq` library or similar
- Log-transform frequencies: log(frequency + 1e-10)

**Step 2: Control for sentiment**
- Important: frequency correlates with sentiment, so we need partial correlation
- Regress out sentiment from both frequency and scores
- Use residuals for correlation

**Step 3: Calculate bias**
- Frequency bias = correlation between frequency residuals and score residuals
- Higher correlation means stronger bias toward common words

### Analysis Pipeline

```
1. Data Preparation
   ├── Load AFINN-111 lexicon
   ├── Filter to model vocabulary
   ├── Prepare neutral token list
   └── Get word frequencies for all tokens

2. Score Collection
   ├── For each judge (1-10):
   │   ├── Score all tokens with "best" prompt
   │   └── Score all tokens with "worst" prompt
   ├── Calculate naive average scores
   └── Get learned aggregator scores

3. Bias Measurements
   ├── For each model type:
   │   ├── Measure framing effects (both prompts)
   │   └── Measure frequency bias
   └── Store results in structured format

4. Comparison Analysis
   ├── Calculate bias reduction percentages
   ├── Create visualization matrices
   └── Run statistical significance tests
```

### Expected Output Format

```python
results = {
    "Individual_Judge_1": {
        "framing_flip": 2.5,
        "slope_asymmetry_best": 1.8,
        "slope_asymmetry_worst": 2.1,
        "variance_asymmetry_best": 0.9,
        "variance_asymmetry_worst": 1.2,
        "frequency_bias": 0.45
    },
    "Individual_Judge_2": {...},
    # ... more judges
    "Naive_Average": {...},
    "Learned_Aggregator": {...}
}

improvements = {
    "framing_flip_reduction": "45%",
    "slope_asymmetry_reduction": "38%", 
    "frequency_bias_reduction": "51%"
}
```

### Visualization Requirements

1. **Heatmap**: Rows = aggregation methods, Columns = bias types, Values = bias magnitude
2. **Scatter Plots**: Token sentiment vs. score for each prompt (colored by positive/negative)
3. **Bar Chart**: Bias reduction percentages for learned aggregator vs baselines
4. **Slope Comparison Plot**: Show regression lines for positive/negative tokens across models

### Key Implementation Notes

- **Ensure consistent tokenization**: Same tokenizer across all judges
- **Handle missing tokens**: Some judges might not have all AFINN tokens in vocabulary
- **Statistical power**: Need enough tokens in each category for reliable statistics
- **Multiple comparisons**: Consider Bonferroni correction if testing many hypotheses
- **Save intermediate results**: Store raw scores for post-hoc analyses
- **Normalization**: Consider normalizing scores if judges use different scales

### Statistical Analysis Details

**For framing effects:**
- Use linear regression with sentiment as predictor, score as outcome
- Calculate R² to assess fit quality
- Test if slopes are significantly different between positive/negative tokens

**For frequency bias:**
- Use partial correlation to control for sentiment
- Alternative: multiple regression with both sentiment and frequency as predictors
- Report both raw correlation and partial correlation

### Success Criteria

The experiment succeeds if:
1. Learned aggregator shows >30% reduction in framing flip magnitude
2. Frequency bias correlation reduced by >25%
3. Results are statistically significant (p < 0.05)
4. Patterns are consistent across majority of judges

### Troubleshooting

Common issues and solutions:
- **Too few tokens**: Expand beyond AFINN to include more vocabulary
- **No bias detected**: Check if judges were trained on similar data (might already be debiased)
- **Aggregator increases bias**: Examine which judges contribute most to aggregated score
- **Computational constraints**: Can subsample tokens or judges for initial testing
- **Scale differences**: Normalize all scores to [0,1] or use z-scores

### Expected Timeline

- Data preparation: 1 hour
- Score collection: 2-3 hours (depending on API/compute availability)
- Analysis: 1-2 hours
- Visualization: 1 hour
- Total: 5-7 hours for complete experiment

This experiment provides concrete evidence on whether multi-judge aggregation naturally develops robustness to cognitive biases present in individual models.