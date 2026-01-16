# Cascade Protection Monitoring & ML Analysis

## Overview

The ML system automatically monitors cascade protection effectiveness and recommends optimal threshold settings based on actual trading data.

---

## What Gets Monitored

### 1. Stop-Out Events
Every time a per-stack stop loss triggers, the system logs:
- Timestamp
- Position ticket
- Symbol
- Loss amount ($)
- ADX value at stop
- Stack type (Initial, DCA-only, DCA+Hedge)

**Log File:** `logs/stop_out_events.log`

### 2. Cascade Events
When 2+ stops occur within 30 minutes:
- Number of stops in window
- Symbols affected
- Average ADX
- Whether trend confirmed (ADX >= 25)
- Total loss from cascade

### 3. Validation
**Key Question:** "Do stops really happen during trends?"
- Calculates: % of stops with ADX >= 25
- If 70%+: HIGH confidence that stops = trends
- If 50-70%: MEDIUM confidence
- If <50%: LOW confidence - may need threshold adjustment

---

## How to Check Cascade Status

### Quick Check (Anytime)
```bash
python3 ml_system/scripts/check_cascade_protection.py
```

**Output Shows:**
- Total stop-outs to date
- Average ADX at stops (validates assumption)
- Cascade events detected
- Stop-outs by symbol and stack type
- **Threshold recommendations**

### Daily Report (Automatic)
Section 7 of the daily ML report includes full cascade analysis with recommendations.

**Generated:** 8:00 AM daily
**Location:** `ml_system/reports/daily/decision_report_YYYY-MM-DD.txt`

---

## What Gets Recommended

### 1. CASCADE_ADX_THRESHOLD
**Current:** 25

**Recommendation Logic:**
- If avg ADX at stops >= 30: Increase to 28 (stops happening in strong trends)
- If avg ADX at stops < 22: Decrease to 20 (catching earlier trends)
- Otherwise: Keep at 25 (optimal)

**Example:**
```
CASCADE_ADX_THRESHOLD:
  Current: 25
  Recommended: 28
  Reason: Avg ADX at stops: 31.2 - can increase threshold
  ⚠️  Consider adjusting this setting
```

### 2. DCA_ONLY_MAX_LOSS
**Current:** -$25

**Recommendation Logic:**
- If avg DCA-only loss > $20: Increase to -$30 (stop triggers too early)
- If avg DCA-only loss < $15: Decrease to -$20 (can tighten)
- Otherwise: Keep at -$25 (optimal)

### 3. DCA_HEDGE_MAX_LOSS
**Current:** -$50

**Recommendation Logic:**
- If avg DCA+Hedge loss > $45: Increase to -$60 (stop triggers too early)
- If avg DCA+Hedge loss < $35: Decrease to -$40 (can tighten)
- Otherwise: Keep at -$50 (optimal)

### 4. CASCADE_THRESHOLD
**Current:** 2 stops

**Recommendation Logic:**
- If cascade rate > 30%: Increase to 3 (too many cascades)
- If cascade rate < 10%: Keep at 2 (working well)

**Cascade Rate** = (Cascade events / Total stops)

---

## Example Reports

### Example 1: Everything Optimal
```
CASCADE PROTECTION ANALYSIS
======================================================================

Data Range: 2026-01-10 to 2026-01-15
Total Stop-Outs: 12
Average Loss: $21.50
Max Loss: $48.20

ADX Analysis:
  Average ADX at stops: 28.3
  High ADX stops (>= 25): 9/12 (75%)

  Validation: 75% of stops occurred with ADX >= 25 (trending)
  Confidence: HIGH
  → Cascade protection is correctly identifying trend transitions

Stop-Outs by Stack Type:
  DCA-only: 8 stops, avg loss $18.75
  DCA+Hedge: 3 stops, avg loss $42.30
  Initial: 1 stops, avg loss $24.00

Cascade Events Detected: 2
  Recent cascades:
    2026-01-12 10:22: 2 stops, EURUSD, $73.70 total
    2026-01-14 14:15: 3 stops, GBPUSD, $112.40 total

======================================================================
THRESHOLD RECOMMENDATIONS
======================================================================

CASCADE_ADX_THRESHOLD: ✓ Optimal (25)
DCA_ONLY_MAX_LOSS: ✓ Optimal (-25.0)
DCA_HEDGE_MAX_LOSS: ✓ Optimal (-50.0)
CASCADE_THRESHOLD: ✓ Optimal (2)
```

### Example 2: Adjustments Needed
```
CASCADE PROTECTION ANALYSIS
======================================================================

Data Range: 2026-01-10 to 2026-01-15
Total Stop-Outs: 15
Average Loss: $26.80
Max Loss: $52.30

ADX Analysis:
  Average ADX at stops: 32.8
  High ADX stops (>= 25): 14/15 (93%)

  Validation: 93% of stops occurred with ADX >= 25 (trending)
  Confidence: HIGH
  → Cascade protection is correctly identifying trend transitions

Stop-Outs by Stack Type:
  DCA-only: 10 stops, avg loss $23.20
  DCA+Hedge: 5 stops, avg loss $48.10

======================================================================
THRESHOLD RECOMMENDATIONS
======================================================================

CASCADE_ADX_THRESHOLD:
  Current: 25
  Recommended: 28
  Reason: Avg ADX at stops: 32.8 - can increase threshold
  ⚠️  Consider adjusting this setting

DCA_ONLY_MAX_LOSS:
  Current: -25.0
  Recommended: -30.0
  Reason: Avg DCA-only loss: $23.20 - stop triggers too early
  ⚠️  Consider adjusting this setting

DCA_HEDGE_MAX_LOSS: ✓ Optimal (-50.0)
CASCADE_THRESHOLD: ✓ Optimal (2)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
⚠️  ACTION REQUIRED: 2 setting(s) need adjustment
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Update these in trading_bot/config/strategy_config.py
Then restart the bot to apply changes.
```

---

## How to Apply Recommendations

### 1. Review Recommendations
Run: `python3 ml_system/scripts/check_cascade_protection.py`

### 2. Update Configuration
Edit `trading_bot/config/strategy_config.py`:

```python
# Example adjustments based on recommendations
CASCADE_ADX_THRESHOLD = 28  # Was: 25
DCA_ONLY_MAX_LOSS = -30.0   # Was: -25.0
```

### 3. Restart Bot
```bash
# Stop current bot (Ctrl+C)
# Start with new settings
cd trading_bot
python main.py --login LOGIN --password PASSWORD --server SERVER
```

### 4. Monitor Results
After 1-2 weeks, check cascade analysis again:
```bash
python3 ml_system/scripts/check_cascade_protection.py
```

If new recommendations appear, repeat the process.

---

## Interpretation Guide

### ADX Analysis

**High ADX Percentage (70%+):**
- ✓ Stops are correctly identifying trends
- ✓ Cascade protection working as intended
- Action: None, system validated

**Medium ADX Percentage (50-70%):**
- ⚠️ Some stops during non-trending markets
- Action: Monitor, may need minor tuning

**Low ADX Percentage (<50%):**
- ❌ Many stops during non-trending conditions
- Action: Lower CASCADE_ADX_THRESHOLD or investigate stop loss limits

### Cascade Rate

**Formula:** (Cascade events / Total stops)

**Low Rate (<10%):**
- Most stops are isolated events
- Cascade protection rarely triggers
- Good: Not too sensitive

**Medium Rate (10-30%):**
- Moderate cascade frequency
- System detecting regime changes appropriately
- Optimal range

**High Rate (>30%):**
- Too many cascades
- May be too sensitive
- Action: Increase CASCADE_THRESHOLD from 2 to 3

### Average Loss by Stack Type

**DCA-only:**
- Should be near but below -$25
- If $20-25: Optimal
- If <$15: Stop too tight, can loosen to -$30
- If >$25: Should never happen (stop at -$25)

**DCA+Hedge:**
- Should be near but below -$50
- If $40-50: Optimal
- If <$35: Stop too tight, can loosen to -$60
- If >$50: Should never happen (stop at -$50)

---

## Data Requirements

**Minimum for Recommendations:**
- 5+ stop-out events
- Spanning at least 3 days
- At least 2 different symbols

**Before Minimum:**
Report shows: "Insufficient data for recommendations"

**What to Do:**
- Continue trading
- Cascade protection is active
- Recommendations will appear after more data collected

---

## Monitoring Frequency

### Daily (Automatic)
- ML report includes cascade analysis
- Check for recommendations

### Weekly (Manual)
- Run quick check script
- Review any threshold changes
- Apply if multiple recommendations persist

### Monthly (Review)
- Comprehensive analysis
- Validate long-term effectiveness
- Major threshold adjustments if needed

---

## FAQ

**Q: What if I see "No stop-out events recorded yet"?**
A: Normal! Cascade protection is active, just no stops have triggered yet. This is good - means no stacks hit stop loss limits.

**Q: Should I immediately apply all recommendations?**
A: Not necessarily. If a recommendation appears once, monitor. If it appears in 2-3 consecutive reports, apply the change.

**Q: What if recommendations contradict each other?**
A: Rare, but prioritize based on sample size. DCA-only recommendations need 5+ DCA-only stops, etc.

**Q: Can I disable cascade monitoring?**
A: Yes, but not recommended. The monitoring is passive (no bot impact). To disable logging only:
```python
ENABLE_CASCADE_PROTECTION = False
```

**Q: How do I analyze historical stop-outs?**
A: All events are in `logs/stop_out_events.log`. You can grep/analyze manually:
```bash
# Show all stops with ADX > 30
grep "ADX:" logs/stop_out_events.log | awk -F'ADX:' '{if ($2+0 > 30) print}'

# Count stops by symbol
grep "#" logs/stop_out_events.log | awk -F'|' '{print $3}' | sort | uniq -c
```

**Q: What's the difference between cascade protection and regular stop loss?**
A: Regular stop loss closes individual stacks. Cascade protection closes ALL underwater stacks when 2+ stops occur in 30min (indicating trend shift).

---

## Integration with ML Reports

Cascade analysis is **Section 7** of the daily ML report.

**Other Sections:**
1. Executive Summary
2. Current Config vs ML Optimal
3. Confluence Analysis
4. Weekly Performance
5. Recovery Mechanisms
6. Time & Market Patterns
7. **CASCADE PROTECTION ANALYSIS** ← NEW
8. Next Review

All sections work together to provide comprehensive trading system analysis.

---

## Troubleshooting

**Issue:** Script fails with "ModuleNotFoundError"
**Fix:** Run from project root: `python3 ml_system/scripts/check_cascade_protection.py`

**Issue:** Log file not found
**Fix:** Normal if bot hasn't run yet or no stops occurred. File created automatically on first stop-out.

**Issue:** Recommendations seem wrong
**Fix:** Check sample size. Need 5+ events minimum. Wait for more data.

**Issue:** Cascade protection not triggering
**Fix:** Verify `ENABLE_CASCADE_PROTECTION = True` in strategy_config.py

---

## Summary

Cascade protection monitoring provides **data-driven feedback** on:
1. Whether stops really happen during trends (validates assumption)
2. Optimal threshold settings (prevents guesswork)
3. Effectiveness of cascade protection (prevents cascade losses)

**Key Benefits:**
- Automatic threshold optimization
- Validates system assumptions
- No manual analysis required
- Recommendations appear in daily reports

**Remember:** The system learns from your specific trading patterns and market conditions. Recommendations are personalized to your bot's behavior.
