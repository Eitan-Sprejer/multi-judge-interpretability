# Efficient Rubric Sensitivity Experiment

## The Key Insight

We only need to make API calls for **unique judge-variant pairs**, then we can create **any combination** by reusing those scores!

## API Call Math (Corrected)

### What We Actually Need
```
4 variant types × 10 judges × N examples = 40N API calls
```

### NOT This (Wrong!)
```
44 combinations × 10 judges × N examples = 440N API calls ❌
```

## How It Works

### Step 1: Collect Unique Variant Scores (API Calls Here)
```python
# Make API calls for all unique judge-variant pairs
for variant in ['strict', 'lenient', 'bottom_heavy', 'top_heavy']:
    for judge in all_10_judges:
        scores[f"{judge}_{variant}"] = evaluate_with_api(judge, variant, examples)

# Total: 4 × 10 × 1000 = 40,000 API calls
```

### Step 2: Create Combinations (NO API Calls!)
```python
# Now we can create ANY combination by mixing scores:

combinations['all_strict'] = {
    'truthfulness': scores['truthfulness_strict'],
    'harmlessness': scores['harmlessness_strict'],
    ... # all 10 judges use strict variant
}

combinations['mixed_balanced'] = {
    'truthfulness': scores['truthfulness_strict'],
    'harmlessness': scores['harmlessness_lenient'],
    ... # mix of variants
}

combinations['single_contaminated'] = {
    'truthfulness': scores['truthfulness_strict'],  # contaminated
    'harmlessness': scores['harmlessness_original'], # normal
    ... # rest use original
}

# We can create UNLIMITED combinations with NO additional API calls!
```

## Concrete Example with 1000 Samples

### API Calls Required
- **Unique evaluations**: 4 variants × 10 judges = 40 unique judge-variants
- **Per evaluation**: 1000 examples
- **Total API calls**: 40 × 1000 = **40,000 calls**

### Combinations We Can Test (with those 40,000 calls)
1. **Baseline** (all original)
2. **All Strict** (all judges use strict variant)
3. **All Lenient** (all judges use lenient variant)
4. **Mixed Balanced** (half strict, half lenient)
5. **Single Contaminated** (one judge variant, rest original)
6. **Random Mix** (random variant assignment)
7. **Bottom Heavy** (all use bottom_heavy intervals)
8. **Top Heavy** (all use top_heavy intervals)
9. ... unlimited more!

### Training Data Per Combination
- **Total examples**: 1000
- **Training set**: 800 (80%)
- **Test set**: 200 (20%)
- **MLP input features**: 10 (one per judge)
- **Sufficient?**: YES! 800 samples for 10 features is good

## Implementation Files

### 1. `efficient_scoring_pipeline.py`
- Makes API calls ONLY for unique judge-variant pairs
- Caches all scores
- Provides `create_combination_scores()` to mix scores without API calls

### 2. `experiment_runner_v2.py`
- Uses efficient pipeline
- Correctly reports API calls as 40N, not 40N×combinations

### 3. `optimized_combinations.py`
- Generates meaningful combinations to test
- Calculates correct API call estimates

## Running the Experiment

### Test Mode (Small Scale)
```bash
# 10 examples = 400 API calls
python test_real_api.py --mode real
```

### Full Experiment (Recommended)
```bash
# 1000 examples = 40,000 API calls
./run_full_experiment.sh
```

## Why This Matters

### Before (Inefficient)
- Would make redundant API calls for each combination
- 44 combinations × 10 judges × 100 examples = 44,000 calls
- Only 100 examples per model (overfitting risk!)

### After (Efficient)
- Makes API calls only for unique judge-variants
- 4 variants × 10 judges × 1000 examples = 40,000 calls
- 1000 examples per model (good training data!)
- Can test unlimited combinations

## Key Takeaway

**We separate data collection from combination creation:**
1. **Data Collection**: API calls for unique judge-variants (40N calls)
2. **Combination Creation**: Mix and match cached scores (0 additional calls)

This gives us more training data AND more flexibility with fewer API calls!