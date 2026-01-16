# Per-Stack Stop Loss System - Implementation Summary

## Overview
Successfully implemented dollar-based stop loss limits per recovery stack to prevent catastrophic drawdowns while keeping the DCA+hedge strategy operational.

**Problem Solved:** User experienced -$689 (59.5% drawdown) with 23 positions where both DCA and hedges were underwater simultaneously.

**Solution:** Conservative recovery settings + per-stack stop loss management

---

## What Changed

### 1. Conservative Recovery Parameters

**Hedge Settings (Reduced Aggressiveness):**
```python
HEDGE_TRIGGER_PIPS = 50      # Was: 8 pips (let DCA work first)
HEDGE_RATIO = 1.5            # Was: 5.0x (reasonable offset)
MAX_HEDGE_VOLUME = 0.20      # Was: 0.50 lots (lower risk cap)
```

**DCA Settings (Limited Exposure):**
```python
DCA_MAX_LEVELS = 3           # Was: 6 levels
DCA_MULTIPLIER = 1.5         # Was: 2.0x
```

**Volume Progression Comparison:**
```
OLD (Aggressive):
0.04 → 0.08 → 0.16 → 0.32 → 0.64 → 1.28 lots = 2.52 total

NEW (Conservative):
0.04 → 0.06 → 0.09 → 0.14 lots = 0.33 total (87% reduction!)
```

### 2. Per-Stack Stop Loss System

**New Parameters:**
```python
ENABLE_STACK_STOPS = True
DCA_ONLY_MAX_LOSS = -25.0    # Max loss for DCA-only stacks
DCA_HEDGE_MAX_LOSS = -50.0   # Max loss for DCA+hedge stacks
```

**How It Works:**
1. Bot monitors **net P&L** for entire recovery stack (original + DCA + hedges)
2. Automatically detects stack type:
   - Initial-only → Uses DCA-only limit (-$25)
   - DCA-only → Uses DCA-only limit (-$25)
   - DCA+Hedge → Uses DCA+hedge limit (-$50)
3. Closes entire stack when limit exceeded

**Files Modified:**
- `trading_bot/config/strategy_config.py` - Added configuration parameters
- `trading_bot/strategies/recovery_manager.py` - Added `check_stack_stop_loss()` method
- `trading_bot/strategies/confluence_strategy.py` - Integrated stop check in main loop

---

## What We Track (ML System)

### Automatic Logging (Already Active)

When you start the bot with:
```bash
python trading_bot/main.py --login LOGIN --password PASSWORD --server SERVER
```

**The following starts automatically:**
1. **ML System** - Auto-retraining every 8 hours
2. **Continuous Logger** - Captures trades every 60 seconds
3. **Daily Reports** - Generated at 8:00 AM

### Data Captured Per Trade

For every closed position, the system logs:
- `dca_count` - Number of DCA levels deployed (0-3)
- `hedge_count` - Number of hedges deployed (0-1)
- `profit` - Final P&L in dollars
- `recovery_cost` - Total cost incurred during recovery
- `duration_hours` - Time from entry to exit
- `max_drawdown` - Worst floating loss (if tracked)

**Storage:** `ml_system/logs/continuous_trade_log.jsonl`

---

## How to Monitor Performance

### Weekly Check (Recommended)

Run this script to analyze recovery outcomes:
```bash
python3 ml_system/scripts/analyze_recovery_outcomes.py
```

**Output Shows:**
```
1. INITIAL-ONLY TRADES (No Recovery)
   Count: 15
   Win Rate: 100%
   Avg Profit: $0.52

2. DCA-ONLY TRADES (No Hedge)
   Count: 8
   Win Rate: 75%
   Avg Profit: $2.18
   Avg DCA Levels: 1.5

   Individual trades:
     $3.50 | 1 DCA levels
     $1.20 | 2 DCA levels
    -$8.00 | 2 DCA levels (STOPPED)

3. DCA+HEDGE TRADES (Full Recovery)
   Count: 2
   Win Rate: 50%
   Avg Profit: -$5.00

STOP LOSS VALIDATION:
  DCA-ONLY: Worst loss -$8.00
    ✅ Stop limit OK - $17.00 cushion to worst loss

  DCA+HEDGE: Worst loss -$48.00
    ✅ Stop limit OK - $2.00 cushion to worst loss
```

### What to Look For

**Good Signs:**
- DCA-only win rate: 70-80%
- DCA+hedge win rate: 50-70%
- Stop losses trigger on <20% of trades
- Worst losses stay within limits

**Warning Signs:**
- Stop losses trigger on >30% of trades → Limits too tight
- Worst losses exceed limits by >$10 → Bug in code
- DCA+hedge deployments very frequent → Hedge trigger too early

---

## Tuning the Stop Loss Limits

### When to Adjust

**Increase Limits (Give More Room):**
- Stop triggers on >20% of DCA trades
- Trades that would have recovered are being cut early
- Example: Change -$25 → -$30 or -$35

**Decrease Limits (Tighten Protection):**
- Worst losses are much smaller than limits (e.g., -$10 to -$15)
- Want tighter risk control
- Example: Change -$25 → -$20 or -$15

**Target Behavior:**
- Stop should catch worst 5-10% of trades
- Acts as safety net, not regular exit method
- Prevents outlier disasters like -$689

### Adjustment Process

1. Run `analyze_recovery_outcomes.py` weekly
2. Check how often stops trigger
3. Review worst losses vs. limits
4. Adjust in `strategy_config.py`:
   ```python
   # Example: Increase DCA-only limit
   DCA_ONLY_MAX_LOSS = -30.0  # Was: -25.0

   # Example: Decrease DCA+hedge limit
   DCA_HEDGE_MAX_LOSS = -40.0  # Was: -50.0
   ```
5. Restart bot to apply changes
6. Monitor for another week

---

## Expected Outcomes by Stack Type

### 1. Initial-Only (No Recovery)
- **Historical:** 100% win rate, $0.52 avg
- **Stop Limit:** -$25 (safety net only)
- **Expected:** Same performance, rarely hits stop

### 2. DCA-Only
- **Expected Win Rate:** 70-80%
- **Avg Profit:** $1-3
- **Typical Drawdown:** -$10 to -$20 at worst
- **Stop Limit:** -$25
- **Expected:** Stop triggers on worst 5-10% of trades

**Example Progression:**
```
Entry:    0.04 lots at 1.1000
          Price -20 pips → Floating: -$8
DCA L1:   0.06 lots at 1.0980
          Price -40 pips → Floating: -$18
DCA L2:   0.09 lots at 1.0960
          Price -50 pips → Floating: -$24

If price continues down → Stop at -$25
If price reverses → Profit $1-5
```

### 3. DCA+Hedge
- **Expected Win Rate:** 50-70%
- **Avg Profit:** -$5 to +$10 (more volatile)
- **Typical Drawdown:** -$30 to -$50 at worst
- **Stop Limit:** -$50
- **Expected:** Stop triggers on worst 10-15% of trades

**Example Progression:**
```
Entry:    0.04 lots SELL at 1.1000
DCA L1:   0.06 lots at 1.1020 (20 pips)
DCA L2:   0.09 lots at 1.1040 (40 pips)
HEDGE:    0.06 lots BUY at 1.1050 (50 pips trigger)
DCA L3:   0.14 lots at 1.1060 (60 pips)

Position Structure:
  SELL: 0.33 lots (original direction)
  BUY:  0.06 lots (hedge)

Best Case: Price reverses, both profit → +$10 to +$20
Typical: Mixed results → -$10 to +$5
Worst: Both underwater → Stop at -$50 (prevents -$600+)
```

---

## Key Files and Locations

### Configuration
- `trading_bot/config/strategy_config.py` - All recovery settings

### Code
- `trading_bot/strategies/recovery_manager.py` - Recovery logic + stop loss check
- `trading_bot/strategies/confluence_strategy.py` - Main trading loop
- `trading_bot/main.py` - Entry point (starts ML + bot)

### ML System
- `ml_system/logs/continuous_trade_log.jsonl` - Trade history
- `ml_system/scripts/analyze_recovery_outcomes.py` - Performance analysis
- `ml_system/reports/decision_report.py` - Daily reports

### Documentation
- `docs/recovery_performance_expectations.md` - Detailed expectations guide
- `docs/RECOVERY_SYSTEM_SUMMARY.md` - This file

---

## Quick Reference Commands

### Start Bot
```bash
cd trading_bot
python main.py --login LOGIN --password PASSWORD --server SERVER
```

### Analyze Recovery Performance
```bash
python3 ml_system/scripts/analyze_recovery_outcomes.py
```

### Generate ML Report
```bash
python3 ml_system/reports/daily_report.py
```

### Check Stop Loss Settings
```bash
grep -A 3 "ENABLE_STACK_STOPS" trading_bot/config/strategy_config.py
```

---

## Safety Features Summary

**What Prevents -$689 Disaster Now:**

1. ✅ **Conservative DCA:** 3 levels max (was 6)
2. ✅ **Smaller Multiplier:** 1.5x (was 2.0x) → Less volume growth
3. ✅ **Late Hedge Trigger:** 50 pips (was 8) → Gives DCA time to work
4. ✅ **Smaller Hedge Ratio:** 1.5x (was 5.0x) → Less counterposition
5. ✅ **Per-Stack Stop Loss:** -$25/-$50 limits → Hard cap on losses
6. ✅ **Automatic Monitoring:** ML system tracks every deployment

**Old Settings Could Create:**
- 6 DCA levels = 2.52 lots
- 5.0x hedge = 1.26 lots opposite
- Both underwater = -$600+ loss

**New Settings Max Exposure:**
- 3 DCA levels = 0.33 lots
- 1.5x hedge = 0.06 lots opposite
- Stop loss at -$50 → **87% reduction in max loss**

---

## Next Steps

1. **Run Bot:** Start trading with new conservative settings
2. **Monitor Weekly:** Run `analyze_recovery_outcomes.py`
3. **Tune Limits:** Adjust -$25/-$50 based on actual performance
4. **Review Monthly:** Check if hedge trigger (50 pips) is appropriate

**Critical:** The -$25 and -$50 limits are **TEST VALUES**. They should be adjusted based on real trading results, not theory.

---

## Questions?

**Stop loss too tight?**
- Increase DCA_ONLY_MAX_LOSS or DCA_HEDGE_MAX_LOSS in strategy_config.py

**Want to disable hedging completely?**
- Set HEDGE_ENABLED = False in strategy_config.py

**Need more DCA levels?**
- Increase DCA_MAX_LEVELS (but watch total exposure!)

**Want to track max drawdown per stack?**
- Already captured by recovery_manager - check position['max_underwater_pips']

All changes require bot restart to take effect.
