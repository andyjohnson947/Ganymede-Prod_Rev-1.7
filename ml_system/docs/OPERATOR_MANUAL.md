# ML System Operator Manual

## Quick Start

### Daily Operations

**Morning Routine:**
1. Check monitoring dashboard
2. Review overnight predictions
3. Verify model status (green)
4. Check for alerts

**End of Day:**
1. Review prediction accuracy
2. Update performance log
3. Check for data collection

### Weekly Operations

**Monday:**
- Review 7-day performance vs baseline
- Check model confidence trends
- Review SHAP explanations for recent trades

**Friday:**
- Automated retraining (if data available)
- Backup model registry
- Weekly performance report

## System Status

### Check System Health
```bash
python ml_system/scripts/integration_test.py
```

Expected output: All 5 tests passed

### Check Model Version
```bash
cat ml_system/models/registry.json | grep -A5 '"status": "production"'
```

### Check Prediction Log
```bash
tail -20 ml_system/outputs/model_monitor.jsonl | jq .
```

## Understanding Predictions

### Confidence Levels

- **90-100%:** Very High (rare with small dataset)
- **75-90%:** High (take trade)
- **60-75%:** Medium (consider)
- **<60%:** Low (skip)

Current threshold: **70%**

### Feature Importance

Top factors affecting predictions:
1. Trend indicators (ADX, DI values)
2. Volume profile levels (swing points)
3. Time of day
4. Recovery mechanisms used

### SHAP Explanations

For any trade, view SHAP explanation:
```bash
python ml_system/explainability/explain_trade.py --ticket 54572356809
```

## Configuration

### Adjust Confidence Threshold

Edit `ml_system/config.py`:
```python
CONFIDENCE_THRESHOLD = 0.75  # Raise to 75% for more selective
```

### Change Operating Mode

**Shadow Mode (Current):**
```python
SHADOW_MODE = True  # ML logs but doesn't control
```

**Active Mode (Future):**
```python
SHADOW_MODE = False  # ML controls trade decisions
```

## Performance Monitoring

### Key Metrics Dashboard

1. **Win Rate:**
   - ML vs Baseline
   - Target: +5% improvement

2. **Prediction Volume:**
   - Trades/day
   - Target: Match bot trade frequency

3. **Confidence:**
   - Average confidence score
   - Target: >65%

4. **Latency:**
   - Prediction time
   - Target: <100ms (currently 4ms ✓)

### Generate Performance Report

```bash
python ml_system/scripts/model_monitoring.py
cat ml_system/outputs/monitoring_report.txt
```

## Model Retraining

### Automatic Retraining

Runs weekly (Friday) automatically:
```bash
python ml_system/scripts/retrain_model.py
```

Process:
1. Collect new closed trades
2. Retrain model
3. Validate on holdout
4. Deploy if better than current

### Manual Retraining

When needed:
```bash
# 1. Create fresh dataset
python ml_system/scripts/create_dataset.py

# 2. Train new model
python ml_system/scripts/hyperparameter_tuning.py

# 3. Validate
python ml_system/scripts/backtest_ml.py

# 4. Deploy (if satisfied)
# Update registry to set new model as production
```

## Troubleshooting

### Model Predictions Seem Wrong

1. Check SHAP explanation: `python ml_system/explainability/explain_trade.py --ticket XXXXX`
2. Verify feature values reasonable
3. Compare to historical trades

### System Running Slow

1. Check prediction latency: `grep latency ml_system/outputs/model_monitor.jsonl | tail -10`
2. If >100ms: Consider using RFE selected features (10 features vs 60)

### Want to Collect More Data

1. Let bot run normally
2. Ensure continuous_trade_log.jsonl is being updated
3. Wait for 50+ closed trades
4. Retrain with full dataset

## Advanced Operations

### A/B Testing

Compare ML vs baseline:
```bash
python ml_system/experiments/ab_test.py --duration 7  # 7 days
```

### Feature Analysis

Identify best confluence factors:
```bash
python ml_system/scripts/analyze_model.py
```

### Experiment with Algorithms

Test different models:
```bash
python ml_system/scripts/compare_models.py
```

## Safety Protocols

### Before Switching to Active Mode

Checklist:
- [ ] 50+ training samples collected
- [ ] 2+ weeks shadow mode validation
- [ ] ML win rate consistently better
- [ ] No major feature drift
- [ ] Stakeholder approval

### If Model Behaving Unexpectedly

1. **Immediate:** Switch to shadow mode
2. Review recent predictions
3. Check for data quality issues
4. Consult troubleshooting guide
5. Rollback to previous model if needed

## Contact & Escalation

### Model Issues
- Check: ml_system/docs/TROUBLESHOOTING.md
- Review: Recent commits and changes

### Data Issues
- Verify: Trade log format and completeness
- Check: Backfill process working

### Emergency
- Disable ML: Set ML_ENABLED = False
- Fallback: Bot uses original rule-based logic
- Investigate: Review logs and recent changes

---

**Manual Version:** 1.0
**Last Updated:** 2026-01-09
**For ML System Version:** 1.0
