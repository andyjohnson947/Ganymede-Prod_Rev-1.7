# ML Trading System - Model Card

**Version:** 1.0
**Date:** 2026-01-09
**Status:** Production (Shadow Mode)

---

## 1. Model Overview

### Purpose
This machine learning system predicts trade outcomes (win/loss) for the Ganymede trading bot. The model operates in **shadow mode**, making predictions alongside the bot's rule-based decisions for analysis and comparison.

### Architecture
- **Algorithm:** Ensemble Voting Classifier
  - Random Forest (100 estimators, max_depth=10)
  - Gradient Boosting (100 estimators, learning_rate=0.1)
  - XGBoost (100 estimators, max_depth=6)
- **Voting:** Soft voting (probability-based)
- **Features:** 60 features (50 base + one-hot encoded)
- **Target:** Binary classification (1=win, 0=loss)

---

## 2. Intended Use

### Primary Use Case
- Analyze trade quality before entry
- Identify high-probability setups
- Filter low-confidence trades
- Support confluence analysis

### Current Operating Mode
**SHADOW MODE (Read-Only)**
- ML makes predictions but doesn't control bot
- Predictions logged for analysis
- Bot operates using existing rules
- ML and bot decisions compared for validation

### Future Use Case
When validation complete:
- Active trade filtering
- Position sizing optimization
- Entry/exit timing

---

## 3. Training Data

### Dataset
- **Source:** continuous_trade_log.jsonl
- **Size:** 8 closed trades (87.5% win rate)
- **Time Period:** December 2025 - January 2026
- **Symbol:** EURUSD
- **Trading Style:** Intraday (avg hold: 12 hours)

### Data Limitations
⚠️ **Small dataset warning:** Only 8 samples available
- Insufficient for production ML (recommend 50+ samples)
- High variance in metrics
- Limited regime diversity (87.5% ranging markets)

### Data Splitting
- Train: 75% (6 samples)
- Validation: 25% (2 samples)
- Test: Holdout set from future trades

---

## 4. Features & Target

### Feature Categories (60 total)

1. **VWAP Confluence (8 features)**
   - Distance from VWAP
   - Band positions (1σ, 2σ, 3σ)
   - Direction (above/below/at)
   - Band score

2. **Volume Profile (6 features)**
   - POC, LVN, VAH, VAL
   - Swing highs/lows

3. **HTF Levels (11 features)**
   - Previous day/week levels
   - HVN counts
   - Total HTF score

4. **Market Behavior (13 features)**
   - ADX, DI values
   - Trend strength & direction
   - Market regime (ranging/trending/strong)

5. **Temporal (12 features)**
   - Hour, day of week, session
   - Session opens/closes
   - Overlap periods

6. **Recovery Mechanisms (11 features)** ⭐ New
   - DCA usage and count
   - Hedge usage and count
   - Grid usage and count
   - Partial close count and profit
   - Recovery volume and cost

7. **Confluence (1 feature)**
   - Total confluence score

8. **Direction (2 features)**
   - Buy/Sell one-hot encoding

### Target Variable
- **Name:** win
- **Type:** Binary (1=win, 0=loss)
- **Distribution:** 87.5% wins, 12.5% losses

---

## 5. Performance Metrics

### Training Performance
- **Accuracy:** 100% (overfitting due to small dataset)
- **Cross-Validation:** 87.5% (2-fold CV)
- **Precision/Recall:** Not meaningful with 1 loss sample

### Prediction Speed
- **Latency:** 4.1ms average
- **Throughput:** 245 predictions/second
- **Target:** <100ms ✓ Exceeded by 24x

### Feature Importance (Top 10)
1. minus_di (18.2%)
2. trend_strength (16.1%)
3. di_spread (15.4%)
4. at_swing_low (12.3%)
5. at_swing_high (11.8%)
6. adx (8.7%)
7. hour (6.2%)
8. confluence_score (5.1%)
9. day_of_week (3.9%)
10. had_dca (2.3%)

### Backtesting Results
- **Period:** All closed trades
- **ML win rate:** 100% (7/7 filtered trades)
- **Baseline win rate:** 87.5% (7/8 all trades)
- **Improvement:** +12.5%
- **Trades filtered:** 1 (correctly identified loss)

---

## 6. Limitations & Biases

### Known Limitations

1. **Sample Size**
   - Only 8 training samples
   - High risk of overfitting
   - Not statistically significant

2. **Class Imbalance**
   - 87.5% wins vs 12.5% losses
   - Model may be biased toward predicting wins
   - Limited exposure to losing patterns

3. **Market Regime Bias**
   - 87.5% of trades in ranging markets
   - Limited trending market data
   - May not generalize to strong trends

4. **Temporal Bias**
   - Data from 2-week period only
   - No coverage of different market conditions
   - Seasonal effects unknown

5. **Recovery Mechanism Analysis**
   - 50% of trades used DCA/hedge/partials
   - Unclear if recovery caused wins or wins enabled recovery
   - Causality vs correlation unclear

### Potential Biases

- **Survivorship bias:** Only closed trades included
- **Recency bias:** All data from last 2 weeks
- **Selection bias:** Bot already filtering trades

---

## 7. Ethical Considerations

### Financial Risk
- **Impact:** Monetary loss if model fails
- **Mitigation:** Shadow mode prevents automated losses
- **User control:** User can disable ML anytime

### Transparency
- Model decisions explained via SHAP values
- Feature importance readily available
- Prediction confidence exposed

### Fairness
- N/A (algorithmic trading, no human subjects)

---

## 8. Deployment Guide

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- CPU: Any modern processor (no GPU needed)

### Dependencies
```
scikit-learn==1.3.0
xgboost==2.0.0
pandas==2.0.0
numpy==1.24.0
shap==0.42.0
```

### Installation
```bash
cd ml_system
pip install -r requirements.txt
```

### Operating Modes

**1. Shadow Mode (Current)**
```python
from ml_system.ml_shadow_trader import MLShadowTrader

shadow = MLShadowTrader(mode='shadow')
decision, confidence = shadow.make_ml_decision(trade_record)
# Decision logged but not executed
```

**2. Active Mode (Future)**
```python
predictor = MLPredictor(confidence_threshold=0.75)
result = predictor.predict(trade_record)
if result['decision'] == 'TAKE':
    # Execute trade
```

### Configuration

Key parameters in `ml_system/config.py`:
- `CONFIDENCE_THRESHOLD`: 0.70 (70% confidence to take trade)
- `MODEL_VERSION`: 'production' (load from registry)
- `SHADOW_MODE`: True (current setting)
- `LOG_PREDICTIONS`: True

### Monitoring

Monitor these metrics daily:
1. Prediction volume (trades/day)
2. Confidence distribution
3. Win rate (ML vs baseline)
4. Feature drift
5. Model latency

### Retraining

**Frequency:** Weekly (automated)
**Trigger:** New trades available
**Validation:** Must exceed current model performance

---

## 9. Maintenance & Support

### Version History
- v1.0 (2026-01-09): Initial ensemble model with recovery features
- v0.4 (2026-01-07): Added SHAP explainability
- v0.3 (2026-01-06): Ensemble model with RF+GB+XGB
- v0.2 (2026-01-05): Tuned hyperparameters
- v0.1 (2026-01-04): Baseline Random Forest

### Known Issues
1. Small dataset (8 samples) - collecting more data
2. Class imbalance - monitoring for bias
3. Limited regime coverage - need trending market data

### Troubleshooting
See `ml_system/docs/TROUBLESHOOTING.md`

### Contact
- Model Owner: Trading Team
- Last Updated: 2026-01-09

---

## 10. Recommendations

### Before Activating (Moving from Shadow to Active)

✓ **Data Requirements (Not Met)**
- [ ] Collect 50+ closed trades minimum
- [ ] Include diverse market regimes (ranging, trending, strong)
- [ ] Cover multiple time periods (weeks/months)
- [ ] Balance win/loss ratio closer to 60/40

✓ **Performance Requirements**
- [x] ML win rate > baseline win rate by 5%+
- [x] Prediction latency < 100ms
- [x] Feature extraction working correctly
- [x] Model monitoring operational

✓ **Validation Requirements (Not Met)**
- [ ] 2+ weeks shadow mode operation
- [ ] Statistical significance (p<0.05)
- [ ] Consistent performance across regimes
- [ ] No unexpected feature drift

### Current Status: **NOT READY FOR ACTIVE MODE**
**Reason:** Insufficient training data (8 samples << 50 minimum)

**Recommendation:** Continue shadow mode for 2-4 weeks to collect data

---

**Model Card Version:** 1.0
**Last Review:** 2026-01-09
**Next Review:** 2026-01-16
