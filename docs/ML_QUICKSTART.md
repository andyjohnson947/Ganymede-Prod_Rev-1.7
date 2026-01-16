# ML System - 1-Click Quick Start

## Single Command to Start

```bash
py run_ml_system.py \
  --login YOUR_LOGIN \
  --password "YOUR_PASSWORD" \
  --server YOUR_SERVER
```

That's it! Everything runs automatically in one process.

## What It Does

**All 3 phases run automatically:**

1. **Logger** - Logs every trade with confluence factors (every 60s)
2. **Shadow Trader** - Makes ML decisions and compares to bot (every 60s)
3. **Optimizer** - Analyzes performance and recommends thresholds (every 6h)

## Example Output

```
================================================================================
🚀 UNIFIED ML SYSTEM - 1-CLICK START
================================================================================
Output Directory:    ml_system/outputs
Check Interval:      60s
Optimize Interval:   21600s (6.0h)
ML Threshold:        5

✅ Connected to MT5

Starting ML system... (Press Ctrl+C to stop)

[Iteration 1] 2026-01-06 18:30:00
📊 [LOGGER] EURUSD #54572356809 | Confluence: 7 | Logged
✅ [LOGGER] Logged 1 new trades
🎯 [SHADOW] EURUSD #54572356809 | Bot: TAKE | ML: TAKE | ✅ AGREE | Score: 7
✅ [SHADOW] Analyzed 1 new trades

================================================================================
🔬 [OPTIMIZER] Running Performance Analysis
================================================================================
📊 Analyzing 45 decisions from last 30 days

PERFORMANCE BY CONFLUENCE THRESHOLD
--------------------------------------------------------------------------------
threshold  bot_trades  ml_trades  bot_win_rate  ml_win_rate  improvement
3          45          45         42.22         42.22        0.00
4          45          38         42.22         47.37        5.15
5          45          28         42.22         57.14        14.92
6          45          18         42.22         66.67        24.45
7          45          12         42.22         75.00        32.78
8          45          6          42.22         83.33        41.11
9          45          3          42.22         100.00       57.78
10         45          1          42.22         100.00       57.78

OPTIMIZATION RECOMMENDATIONS
--------------------------------------------------------------------------------
✅ STRONG: Raise minimum confluence to 9
   Expected improvement: +57.78% win rate

================================================================================

[Iteration 2] 2026-01-06 18:31:00
🎯 [SHADOW] GBPUSD #54572356965 | Bot: TAKE | ML: SKIP | ❌ DISAGREE | Score: 4 | 📉 LOSS $-23.40 (P2)

[Iteration 10] 2026-01-06 18:40:00

================================================================================
📊 ML SYSTEM STATISTICS
================================================================================
Trades Logged:     8
Shadow Decisions:  8
  ├─ Agreements:   5
  └─ Disagreements: 3
  Agreement Rate:  62.5%

ML Correct:        6
Bot Correct:       5

ML Threshold:      5
Last Optimization: 2026-01-06 18:30:00
================================================================================
```

## What the Output Means

### Logger Output
```
📊 [LOGGER] EURUSD #54572356809 | Confluence: 7 | Logged
```
- Logged a new trade with confluence score 7

### Shadow Trader Output
```
🎯 [SHADOW] EURUSD #123 | Bot: TAKE | ML: SKIP | ❌ DISAGREE | Score: 4 | 📉 LOSS $-23.40 (P2)
```
- **Bot: TAKE** - Bot opened this trade
- **ML: SKIP** - ML would have skipped (score < threshold)
- **❌ DISAGREE** - Bot and ML made different decisions
- **Score: 4** - Confluence score was 4
- **📉 LOSS $-23.40** - Trade closed at $-23.40 loss
- **(P2)** - Had 2 partial closes (P = partials)

### Outcome Indicators
- **💰 WIN** - Trade closed profitable
- **📉 LOSS** - Trade closed at loss
- **(P2)** - Had 2 partial closes
- **(P3)** - Had 3 partial closes (25% → 50% → 100%)

## Partial Close Tracking

The system **fully tracks partial closes**:
- Sums total profit from all closes
- Counts number of close events
- Marks trades with partials as **(P2)**, **(P3)**, etc.

Example:
```
Entry: BUY 0.04 EURUSD @ 1.17000
Close 1: 25% (0.01 lots) @ 1.17100 = +$10
Close 2: 50% (0.02 lots) @ 1.17150 = +$30
Close 3: 100% (0.01 lots) @ 1.17200 = +$20
Total: +$60 (P3)
```

Output: **💰 WIN $60.00 (P3)**

## Advanced Options

```bash
# Custom threshold (default: 5)
py run_ml_system.py --login XXX --password "XXX" --server XXX --threshold 7

# Faster checks (default: 60s)
py run_ml_system.py --login XXX --password "XXX" --server XXX --interval 30

# More frequent optimization (default: 6h)
py run_ml_system.py --login XXX --password "XXX" --server XXX --optimize-hours 3
```

## Output Files

All data saved to `ml_system/outputs/`:

```
ml_system/outputs/
├── continuous_trade_log.jsonl    # Every trade logged
└── shadow_decisions.jsonl        # ML vs Bot decisions
```

## Stopping the System

Press **Ctrl+C** to stop gracefully. Final statistics will be printed.

## Comparison: Old vs New

### Old System (3 Separate Scripts)
```bash
# Terminal 1
py ml_system/continuous_logger.py --login XXX --password "XXX" --server XXX

# Terminal 2
py ml_system/shadow_trader.py --login XXX --password "XXX" --server XXX

# Terminal 3 (manual, weekly)
py ml_system/optimizer.py --days 30
```

**Problems:**
- ❌ 3 terminals to manage
- ❌ Separate processes
- ❌ Manual optimization
- ❌ No unified feedback

### New System (1-Click)
```bash
py run_ml_system.py --login XXX --password "XXX" --server XXX
```

**Benefits:**
- ✅ 1 command to start everything
- ✅ Single process, easy to manage
- ✅ Automatic optimization every 6 hours
- ✅ Unified, comprehensive feedback
- ✅ Full partial close tracking
- ✅ Real-time statistics

## What Happens Next

1. **System runs continuously** - Logs and analyzes every trade
2. **Every 6 hours** - Optimizer runs and shows recommendations
3. **Every 10 iterations** - Statistics printed
4. **On Ctrl+C** - Graceful shutdown with final stats

## Making Changes Based on Recommendations

If optimizer recommends raising threshold:

```python
# Edit: trading_bot/portfolio/instruments_config.py

'EURUSD': {
    'windows': [
        {
            'min_confluence_score': 7,  # Changed from 4
        }
    ]
}
```

Restart the bot to apply changes.

## Troubleshooting

**"Failed to connect to MT5"**
- Check login/password/server
- Ensure MT5 terminal is running
- Check firewall settings

**"No decisions in last 30 days"**
- System needs time to collect data
- Let it run for a few days first
- Check that bot is actually trading

**No output for long time**
- Normal if no new trades opened
- Bot only trades during confluence signals
- System is still monitoring

## Summary

**Old way**: 3 scripts, 3 terminals, manual optimization
**New way**: 1 script, 1 click, automatic everything

Just run `py run_ml_system.py` and let it optimize your bot!
