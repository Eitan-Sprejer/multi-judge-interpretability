# Overnight 10K Sample Experiment Plan

## 🎯 Experiment Overview

**Goal**: Run full 10,000 sample experiment with enhanced 15-persona system and confidence-based sampling

**Expected Duration**: 8-12 hours
**API Calls**: ~250,000 total (150k persona + 100k judge)
**Data Generated**: ~500MB

## ✅ Pre-Flight Checklist

### Environment Setup
- [x] API Keys configured (MARTIAN_API_KEY, OPEN_AI_API_KEY)
- [x] Dependencies installed (datasets, martian-sdk-python)
- [x] Enhanced personas (15 total) validated
- [x] JSON retry logic implemented
- [x] Confidence-based sampling ready
- [x] Checkpoint system enabled (every 100 samples)

### Code Validation
- [x] 200-sample test running successfully
- [x] Error handling working (JSON retries, persona exclusion)
- [x] API rate limits configured (15 Lambda AI, 10 Martian)
- [x] New sampling method integrated
- [x] Post-processing documentation complete

### System Requirements
- [ ] Stable internet connection (8-12 hours)
- [ ] Sufficient disk space (>1GB free)
- [ ] Quiet environment (no system updates)

## 🚀 Primary Experiment Command

```bash
# Main 10k experiment with enhanced personas
python run_full_experiment.py \
    --data-source ultrafeedback \
    --data-size 10000 \
    --concurrency 15 \
    --sampling-method uniform \
    --run-name baseline_10k_personas_15_enhanced \
    --random-seed 42
```

## 🔄 Alternative Experiment (Confidence Sampling)

```bash
# Confidence-weighted sampling experiment
python run_full_experiment.py \
    --data-source ultrafeedback \
    --data-size 10000 \
    --concurrency 15 \
    --sampling-method confidence_weighted \
    --run-name baseline_10k_confidence_weighted \
    --random-seed 42
```

## 📊 Expected Timeline

### Phase 1: Data Loading (5-10 minutes)
- Load UltraFeedback dataset
- Sample 10k examples
- Create experiment subset
- Initialize logging

### Phase 2: Persona Simulation (6-8 hours)
- 10k samples × 15 personas = 150,000 API calls
- Checkpoint every 100 samples (100 checkpoints)
- Rate: ~20-25 API calls per minute at concurrency 15
- Expected: 6-8 hours with retry overhead

### Phase 3: Judge Evaluation (2-3 hours)
- 10k samples × 10 judges = 100,000 API calls
- Parallel evaluation in batches
- Rate: ~50-60 API calls per minute at concurrency 10
- Expected: 2-3 hours

### Phase 4: Analysis & Visualization (30 minutes)
- Correlation analysis
- Model training and evaluation
- Plot generation
- Results summary

## 🛡️ Failure Prevention

### Checkpoint Recovery
```bash
# Check progress
ls -la results/full_experiments/baseline_10k_personas_15_enhanced/checkpoints/

# Resume from checkpoint (if interrupted)
# Example: Resume from 5000 samples (checkpoint_050.pkl)
# Note: Current implementation auto-resumes from last checkpoint
```

### API Rate Limit Protection
- **Lambda AI**: 15 concurrent requests (conservative)
- **Martian API**: 10 concurrent judges
- **Retry Logic**: Exponential backoff (1s, 2s, 4s)
- **Max Retries**: 3 attempts per API call

### Resource Management
- **Checkpoints**: Auto-save every 100 samples
- **Logging**: Comprehensive logs in results/logs/
- **Memory**: Efficient pickle serialization
- **Disk Space**: Monitor ~500MB growth

## 📈 Expected Results

### Performance Targets
- **Overall Correlation**: 0.4-0.6 (moderate to strong)
- **Data Quality**: >95% valid samples
- **API Success Rate**: >99% (with retries)
- **Model Performance**: R² > 0.5

### Key Outputs
```
results/full_experiments/baseline_10k_personas_15_enhanced/
├── data/
│   ├── data_with_personas.pkl        # 10k samples + 15 persona scores
│   └── data_with_judge_scores.pkl    # Final dataset with judge scores
├── results/
│   ├── correlation_analysis.json     # Primary research results
│   ├── baseline_results.json         # Model performance
│   └── cross_correlation_analysis.json
├── plots/
│   ├── experiment_analysis.png       # Main results
│   ├── cross_correlation_heatmaps.png
│   └── baseline_comparison_comprehensive.png
└── logs/                             # Execution logs
```

## 🚨 Monitoring & Alerts

### Progress Monitoring
```bash
# Check experiment progress
tail -f results/full_experiments/baseline_10k_personas_15_enhanced/logs/progress_*.log

# Monitor API success rate
grep -c "HTTP.*200 OK" results/full_experiments/baseline_10k_personas_15_enhanced/logs/debug_*.log

# Check error rates
grep -c "ERROR\|failed" results/full_experiments/baseline_10k_personas_15_enhanced/logs/full_experiment_*.log
```

### Health Checks
```bash
# Every 2 hours, check:
# 1. Process still running
ps aux | grep run_full_experiment

# 2. Progress checkpoint
ls -lt results/full_experiments/baseline_10k_personas_15_enhanced/checkpoints/ | head -3

# 3. Disk space
df -h

# 4. Network connectivity
ping -c 3 api.lambda.ai
ping -c 3 withmartian.com
```

## 🔧 Troubleshooting

### Common Issues & Solutions

**Issue**: Process killed/interrupted
**Solution**: 
```bash
# Check last checkpoint
ls -la results/full_experiments/baseline_10k_personas_15_enhanced/checkpoints/
# Restart from last checkpoint (auto-resumes)
python run_full_experiment.py --data-source ultrafeedback --data-size 10000 --run-name baseline_10k_personas_15_enhanced
```

**Issue**: High API failure rate (>5%)
**Solution**: 
- Check network stability
- Verify API key validity
- Reduce concurrency temporarily

**Issue**: Out of disk space
**Solution**:
- Clean up old experiment runs
- Move checkpoints to external storage
- Monitor /tmp directory

**Issue**: Memory issues
**Solution**:
- Restart experiment (checkpoints preserve progress)
- Check for memory leaks in logs
- Reduce checkpoint interval

## 📋 Post-Experiment Checklist

### Immediate Validation
- [ ] Check final correlation results
- [ ] Verify all 10k samples processed
- [ ] Validate plot generation
- [ ] Confirm model training completed

### Quality Assessment
- [ ] Review error logs for patterns
- [ ] Analyze persona failure rates
- [ ] Validate correlation significance
- [ ] Compare against baseline expectations

### Next Steps Planning
- [ ] Compare uniform vs confidence sampling
- [ ] Prepare NeurIPS workshop submission
- [ ] Plan robustness experiments
- [ ] Document key findings

## 🎯 Success Metrics

### Primary Success
- **Completion**: All 10k samples processed
- **Quality**: <5% API failures, >95% valid personas
- **Results**: Overall correlation with statistical significance
- **Documentation**: Complete analysis pipeline

### Research Success
- **Correlation**: >0.4 overall correlation (moderate effect)
- **Model Performance**: GAM R² > 0.5
- **Interpretability**: Clear judge-persona alignment patterns
- **Innovation**: Confidence sampling improvement demonstrated

## 🌙 Overnight Best Practices

1. **Start Early**: Begin by 6-7 PM for completion by morning
2. **Monitor Initially**: Check first 2-3 checkpoints (30 minutes)
3. **Set Alerts**: Phone notifications for major errors
4. **Document**: Note start time and initial progress
5. **Backup Plan**: Have resume strategy ready

## ⚡ Quick Start Commands

```bash
# Standard overnight run
nohup python run_full_experiment.py \
    --data-source ultrafeedback \
    --data-size 10000 \
    --concurrency 15 \
    --sampling-method uniform \
    --run-name baseline_10k_personas_15_enhanced \
    --random-seed 42 \
    > overnight_experiment.log 2>&1 &

# Track progress
tail -f overnight_experiment.log

# Check PID for monitoring
echo $! > experiment.pid
```