# CRITICAL ISSUES IDENTIFIED - Experiment Results Invalid

## 🚨 MAJOR PROBLEMS DISCOVERED

After thorough investigation, the rubric sensitivity experiment has **fundamental flaws** that invalidate the conclusions:

### Issue 1: Broken GAM Model ❌
- **GAM predictions are consistently negative** (-0.38 to -0.17 range)
- This is impossible for a proper regression model predicting human scores (0-10 scale)
- Suggests the GAM model training failed completely or model is corrupted
- **Negative correlation with human feedback is expected** given negative predictions

### Issue 2: Rubric Variants Too Similar ❌
- **High correlations between variants**: 0.83-0.86 correlation coefficients
- Original vs Strict: r=0.8365
- Original vs Lenient: r=0.8628  
- Original vs Bottom Heavy: r=0.8451
- **This explains the low variance** - variants aren't actually different enough

### Issue 3: Scale Mismatches ❌
- Judge scores: 0-4 scale (mean=2.67)
- Human scores: 0-10 scale (mean=5.94)
- Models predict on different scales entirely
- **Normalization attempts don't fix fundamental model issues**

## Root Cause Analysis

### 1. Model Training Failure
The GAM model appears to be fundamentally broken:
```
GAM predictions: [-0.38, -0.17] (consistently negative)
MLP predictions: [-0.51, 0.36] (centered around 0)
Expected: [0, 10] to match human feedback scale
```

### 2. Insufficient Rubric Variation
The experiment design flaw: rubric variants created only minor variations:
- Mean scores differ by only ~0.1-0.2 points
- Correlations >0.83 indicate variants are essentially the same
- **No meaningful robustness test possible**

### 3. Data Pipeline Issues
- Scale mismatches between training and evaluation
- Possible corruption during model saving/loading
- Ground truth extraction inconsistencies

## Honest Assessment

### What the Data Actually Shows:
1. **Mean aggregation works well** (r=0.809 with human feedback)
2. **Single best judge is reasonable** (r=0.764)
3. **Learned models are broken** (negative correlations)
4. **Rubric variants are too similar** (low variance is expected)

### What This Means:
- **The experiment FAILS to demonstrate robustness benefits**
- **Cannot claim learned aggregators are more robust**
- **Results are artifacts of broken models, not genuine robustness**

## Required Actions

To properly test rubric sensitivity:

1. **Fix model training pipeline**
   - Retrain GAM/MLP properly with correct normalization
   - Ensure positive correlations with ground truth
   - Validate model performance before robustness testing

2. **Create more diverse rubric variants**
   - Current variants correlate at 0.83+ (too similar)
   - Need variants with correlations <0.7 for meaningful tests
   - Consider more extreme reformulations

3. **Standardize scales and normalization**
   - Consistent 0-10 or 1-4 scale throughout pipeline
   - Proper train/test data handling
   - Validation of model predictions

## Conclusion ⚠️

**The current results are INVALID and cannot be used to support any claims about aggregator robustness.**

The "2.05x improvement" is an artifact of broken models producing unrealistic negative predictions, not evidence of genuine robustness benefits.

---
*Identified 2025-08-19 through careful validation of suspicious results*