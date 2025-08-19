# Experiment 1C: Rubric Sensitivity Analysis - Status Report

## Overview

**Experiment Goal**: Demonstrate that learned aggregation functions (GAM/MLP) are more robust to rubric variations than naive baseline methods by training separate models on different judge rubric combinations and comparing their performance stability.

**Current Status**: ✅ Data Collection Complete, ❌ Analysis Implementation Incorrect

## Data Collection Status ✅ COMPLETE

### What's Already Done
- **API Calls Completed**: ~40,000 calls for 1000 examples × 4 variants × 10 judges
- **Rubric Variants Generated**: 4 types (strict, lenient, bottom_heavy, top_heavy) + original
- **Judge Score Collection**: All 10 judges evaluated across all variants
- **Data Files Available**:
  - `variant_scores_cache.pkl`: Raw variant scores from API calls
  - `restructured_scores_fixed.pkl`: Structured DataFrame with all combinations
  - `combinations.json`: Combination definitions
  - `config.json`: Experiment configuration

### Data Structure
```
1000 examples × 10 judges × 5 variants = 50,000 judge scores collected
Available combinations:
- All Original (baseline)
- All Strict 
- All Lenient
- All Bottom Heavy
- All Top Heavy
- Mixed combinations possible
```

## Critical Analysis Issues ❌ WRONG METHODOLOGY

### What Current Analysis Does (INCORRECT)
The current `correct_aggregator_analysis.py` approach:
1. **Uses single pre-trained model** (`agg_model_gam.pt`) 
2. **Tests model on different input combinations**
3. **Measures variance in predictions across combinations**
4. **Compares to naive mean baseline**

**Why This Is Wrong**: This tests input sensitivity, not robustness to training data variations.

### What Analysis Should Do (CORRECT)
For proper robustness testing:
1. **Train separate GAM model for each combination** using `aggregator_training.py`
2. **Each model trained on**: Judge scores (combination) → Human feedback (ground truth)
3. **Test each model on same validation set**
4. **Compare R² stability**: How much does performance degrade across different training combinations?
5. **Compare vs baselines**: Does naive mean show more performance degradation?

## Ground Truth Location

**Primary Ground Truth**: Human feedback scores in original dataset
- **Path**: `dataset/data_with_judge_scores.pkl` 
- **Structure**: `human_feedback` column with scores 0-10 scale
- **Alternative**: May be embedded in `variant_scores_cache.pkl`

**Normalization Strategy**: Follow `aggregator_training.py` patterns
- Standard scaling for model inputs
- Consistent 0-10 scale for ground truth targets

## Available Rubric Combinations

From analysis of collected data:
```python
Available combinations:
- 'original': Baseline rubric combination  
- 'strict': All judges use stricter criteria
- 'lenient': All judges use more lenient criteria  
- 'bottom_heavy': All judges use bottom-heavy score intervals
- 'top_heavy': All judges use top-heavy score intervals
```

**Correlation Analysis**: Variants correlate at 0.83-0.86 (relatively similar)
- This limits robustness demonstration potential
- Focus on most different combinations for clearest results

## Resource Requirements

### Computational
- **Model Training**: 5 combinations × 2 models (GAM+MLP) = 10 model training runs
- **Training Time**: ~5-10 minutes per GAM, ~10-15 minutes per MLP
- **Total Compute**: ~2-3 hours for all models
- **Memory**: Standard PyTorch/sklearn requirements

### Data Split
- **Total Examples**: 1000 per combination
- **Training Set**: 800 examples (80%)
- **Validation Set**: 200 examples (20%)
- **Sufficient**: Yes, 800 examples for 10 judge features is adequate

## Implementation Plan

### Phase 1: Methodology Correction
1. **Extract ground truth** from existing data files
2. **Load judge score combinations** from `restructured_scores_fixed.pkl`
3. **Implement training loop** using `aggregator_training.py`
4. **Train models per combination**: GAM and MLP for each of 5 combinations

### Phase 2: Robustness Analysis  
1. **Evaluate all models** on consistent validation set
2. **Calculate R² for each model** vs ground truth
3. **Compare performance stability**:
   - Learned models: GAM and MLP R² across combinations
   - Baselines: Naive mean R² across combinations
4. **Statistical analysis**: Variance, correlation, improvement ratios

### Phase 3: Visualization
1. **Single focused plot** showing key results:
   - R² performance by method across combinations
   - Variance/stability comparison
   - Improvement over baseline
2. **Publication-ready figure** for research paper

## Expected Results

### Success Criteria
- **Learned models maintain stable R²** across rubric combinations (variance <0.05)
- **Baseline methods show more degradation** when trained on different combinations
- **Clear demonstration of robustness benefits** for learned aggregation

### Potential Limitations
1. **High variant correlation (0.83-0.86)** may limit robustness demonstration
2. **Small effect sizes** due to similar rubric variants
3. **Need statistical significance testing** for publication claims

## Next Steps (Priority Order)

### 1. Implement Correct Analysis (HIGH PRIORITY) 
- Create `rubric_robustness_analysis.py` with proper methodology
- Load data, train separate models, measure R² stability
- Target completion: 1-2 days

### 2. Generate Results Figure (HIGH PRIORITY)
- Single publication-quality visualization 
- Clear demonstration of learned model robustness
- Target completion: Same day as analysis

### 3. Validate Results (MEDIUM PRIORITY)
- Statistical significance testing
- Cross-validation for robustness
- Error analysis and limitations discussion

### 4. Research Paper Integration (MEDIUM PRIORITY)  
- Results interpretation for NeurIPS paper
- Limitation acknowledgment (high variant correlation)
- Future work suggestions (more diverse rubrics)

## Technical Debt & Cleanup

### Files to Keep
- `restructured_scores_fixed.pkl`: Primary data source
- `variant_scores_cache.pkl`: Backup/validation
- `run_full_experiment.sh`: Documentation of data collection

### Files to Deprecate  
- `correct_aggregator_analysis.py`: Wrong methodology
- `rerun_analysis.py`: Based on incorrect approach
- Multiple result files with invalid conclusions

### Codebase Consolidation
- New canonical analysis: `rubric_robustness_analysis.py`
- Clean documentation of correct methodology
- Archive incorrect attempts for reference

## Research Context

### Publication Timeline
- **NeurIPS Submission**: August 22, 2025
- **Current Date**: August 19, 2025  
- **Time Remaining**: 3 days
- **Priority**: Complete corrected analysis immediately

### Research Contribution
- **Track 1**: Robustness Analysis (this experiment)
- **Specific Claim**: Learned aggregation more robust to rubric variations
- **Evidence Needed**: R² stability comparison across training combinations
- **Success Metric**: >2x robustness improvement vs baseline

## Conclusion

The experiment data collection is complete and high-quality. The critical issue is **methodology correction**: we must train separate models for each combination rather than testing a single model on different inputs. With the correct approach, this experiment can provide valuable evidence for learned aggregator robustness in the research paper.

**Immediate Action Required**: Implement corrected analysis with separate model training approach.