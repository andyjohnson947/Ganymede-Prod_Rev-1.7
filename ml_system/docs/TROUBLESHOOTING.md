# ML System Troubleshooting Guide

## Common Issues

### 1. Model Loading Fails

**Symptoms:**
```
FileNotFoundError: ml_system/models/production_model.pkl
```

**Solutions:**
1. Check model registry exists: `ls ml_system/models/registry.json`
2. Verify production model set: `cat ml_system/models/registry.json | grep production`
3. Retrain if needed: `python ml_system/scripts/hyperparameter_tuning.py`

### 2. Feature Extraction Errors

**Symptoms:**
```
KeyError: 'vwap' or 'outcome'
```

**Solutions:**
1. Check trade log format: `head -1 ml_system/outputs/continuous_trade_log.jsonl | jq`
2. Verify all required fields present
3. For recovery features: Only work on closed trades (outcome.status == 'closed')

### 3. Prediction Failures

**Symptoms:**
```
ValueError: Input contains NaN
```

**Solutions:**
1. Enable validation: `feature_extractor.validate_features(features)`
2. Check for missing data in trade record
3. Review feature ranges in dataset_info.txt

### 4. Low Confidence Predictions

**Symptoms:**
- All predictions below 50% confidence

**Solutions:**
1. Check if model trained: `ls -lh ml_system/models/*.pkl`
2. Verify training data quality: `head ml_system/data/training_data.csv`
3. Retrain with more diverse data

### 5. Performance Issues

**Symptoms:**
- Predictions taking >100ms

**Solutions:**
1. Profile prediction: `python ml_system/scripts/run_days_16_20.py` (Day 20)
2. Reduce feature count using RFE selected features
3. Use smaller ensemble (fewer models)

## Error Messages

### "Small dataset warning"
**Meaning:** Less than 10 training samples
**Action:** Collect more trades before production use

### "Class imbalance detected"
**Meaning:** Win/loss ratio >80/20
**Action:** Continue collecting data for better balance

### "Feature drift detected"
**Meaning:** Feature distributions changed significantly
**Action:** Retrain model with recent data

## Monitoring Alerts

### High Error Rate
**Threshold:** >10% prediction errors
**Action:** Check logs, validate input data

### Low Prediction Volume
**Threshold:** <5 predictions/day
**Action:** Check if bot is generating trades

### Confidence Degradation
**Threshold:** Average confidence <60%
**Action:** Model may need retraining

## Getting Help

1. Check logs: `tail -100 ml_system/outputs/model_monitor.jsonl`
2. Run diagnostic: `python ml_system/scripts/integration_test.py`
3. Review recent changes: `git log -10 --oneline`

## Emergency Procedures

### Rollback to Previous Model
```bash
cd ml_system/models
python -c "
from model_loader import ModelLoader
loader = ModelLoader()
loader.promote_to_production('v1.0')  # Replace with previous version
"
```

### Disable ML System
Set in config:
```python
SHADOW_MODE = True  # Keep ML logging only
ML_ENABLED = False  # Disable completely
```

### Force Fallback
```python
# In code
USE_FALLBACK_THRESHOLD = True  # Use confluence score only
```
