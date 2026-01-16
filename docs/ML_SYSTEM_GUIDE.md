# 3-Phase Adaptive ML System Guide

## Overview

The ML system creates a self-optimizing trading bot by collecting data, making shadow decisions, and recommending threshold adjustments based on performance analysis.

**Your Vision**: "Store data and create an optimal/dynamic trading setup using confluence information, effectively ghost writing trades so we compare the bot against the ML solution over time"

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      TRADING BOT (Main)                       │
│  Runs confluence strategy, opens real trades with real money  │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ Logs every trade to MT5
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              PHASE 1: Continuous Logger                       │
│  - Monitors MT5 for new confluence trades                    │
│  - Captures confluence factors at entry time                  │
│  - Logs to: continuous_trade_log.jsonl                       │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ Provides trade data
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              PHASE 2: Shadow Trader                           │
│  - Makes parallel ML decisions for each trade                │
│  - Compares: ML vs Bot (TAKE/SKIP)                          │
│  - Logs to: shadow_decisions.jsonl                           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ Provides comparison data
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              PHASE 3: Optimizer                               │
│  - Analyzes shadow decisions & outcomes                      │
│  - Finds optimal confluence thresholds                       │
│  - Recommends threshold adjustments                          │
│  - Outputs: optimization_report.txt                          │
└──────────────────────────────────────────────────────────────┘
```

## Phase 1: Continuous Logger

**Purpose**: Automatically log confluence factors for every trade the bot opens.

**File**: `ml_system/continuous_logger.py`

### How It Works

1. Runs in background (every 60 seconds)
2. Checks MT5 for new deals in last hour
3. Filters for confluence entry trades (comment contains "Confluence:")
4. Fetches H1/D1/W1 bars at entry time
5. Calculates confluence factors using bot's modules:
   - VWAP (bands, distance, direction)
   - Volume Profile (POC, VAH, VAL, LVN, swings)
   - HTF Levels (daily/weekly HVN, POC, prev levels)
   - Trend Filter (ADX, +DI, -DI)
6. Logs to `ml_system/outputs/continuous_trade_log.jsonl`

### Usage

```bash
# Run continuous logger (background service)
py ml_system/continuous_logger.py \
  --login YOUR_LOGIN \
  --password "YOUR_PASSWORD" \
  --server YOUR_SERVER \
  --interval 60
```

**Arguments**:
- `--login`: MT5 account login (required)
- `--password`: MT5 password (required)
- `--server`: MT5 server (required)
- `--interval`: Check interval in seconds (default: 60)

**Output Format** (JSONL - one JSON object per line):

```json
{
  "ticket": 54572356809,
  "symbol": "EURUSD",
  "entry_time": "2026-01-06T18:16:40",
  "entry_price": 1.16920,
  "direction": "BUY",
  "volume": 0.04,
  "confluence_score": 20,
  "vwap": {
    "value": 1.16890,
    "distance_pct": 0.026,
    "direction": "above",
    "in_band_1": true,
    "in_band_2": false,
    "band_1_score": 1,
    "band_2_score": 0
  },
  "volume_profile": {
    "at_poc": false,
    "above_vah": false,
    "below_val": true,
    "at_lvn": true,
    "at_swing_high": false,
    "at_swing_low": true
  },
  "htf_levels": {
    "total_score": 14,
    "factors_matched": ["prev_day_vah", "weekly_hvn", "weekly_poc"],
    "prev_day_vah": 1.16925,
    "prev_day_poc": 1.16880,
    "weekly_hvn_count": 3,
    "weekly_poc": 1.16920
  },
  "trend_filter": {
    "enabled": true,
    "adx": 28.5,
    "plus_di": 32.1,
    "minus_di": 18.7
  },
  "market_context": {
    "hour": 18,
    "day_of_week": "Monday"
  },
  "logged_at": "2026-01-06T18:17:00"
}
```

### Key Features

- **No duplicates**: Tracks logged tickets, skips already-logged trades
- **Real-time**: Logs trades within 60 seconds of execution
- **Exact replication**: Uses bot's actual indicator modules, not approximations
- **Background service**: Runs continuously, doesn't interfere with bot

## Phase 2: Shadow Trader

**Purpose**: Make parallel ML decisions alongside the bot to compare performance.

**File**: `ml_system/shadow_trader.py`

### How It Works

1. Runs in background (every 5 minutes)
2. Checks MT5 for recent deals (last 4 hours)
3. For each confluence trade:
   - Fetches H1/D1/W1 bars at entry time
   - Calculates confluence score
   - Makes ML decision: TAKE (score >= threshold) or SKIP
   - Compares to bot decision (bot always TAKE)
   - Checks trade outcome (WIN/LOSS) if closed
4. Logs comparison to `ml_system/outputs/shadow_decisions.jsonl`

### Usage

```bash
# Run shadow trader (background service)
py ml_system/shadow_trader.py \
  --login YOUR_LOGIN \
  --password "YOUR_PASSWORD" \
  --server YOUR_SERVER \
  --interval 300 \
  --min-confluence 5
```

**Arguments**:
- `--login`: MT5 account login (required)
- `--password`: MT5 password (required)
- `--server`: MT5 server (required)
- `--interval`: Check interval in seconds (default: 300 = 5 minutes)
- `--min-confluence`: Minimum confluence for ML to take trade (default: 5)

**Output Format** (JSONL):

```json
{
  "ticket": 54572356809,
  "symbol": "EURUSD",
  "entry_time": "2026-01-06T18:16:40",
  "entry_price": 1.16920,
  "direction": "BUY",
  "volume": 0.04,
  "bot_decision": "TAKE",
  "ml_decision": "SKIP",
  "agreement": false,
  "bot_confluence_score": 4,
  "ml_confluence_score": 4,
  "factors": {
    "vwap_band_1": true,
    "htf_score": 3,
    "htf_factors": ["prev_day_vah"]
  },
  "outcome": {
    "outcome": "LOSS",
    "profit": -15.20,
    "close_time": "2026-01-06T19:45:00",
    "close_price": 1.16540
  },
  "ml_threshold_min": 5,
  "analyzed_at": "2026-01-06T18:20:00"
}
```

### Key Features

- **Agreement tracking**: Logs when ML agrees/disagrees with bot
- **Outcome tracking**: Records WIN/LOSS for closed trades
- **Threshold testing**: Tests different ML thresholds (min_confluence)
- **Performance comparison**: Shows which decisions would have been better

### Example Output

```
[SHADOW] EURUSD #54572356809 | Bot: TAKE | ML: SKIP | ❌ DISAGREE | Confluence: 4
[SHADOW] GBPUSD #54572356965 | Bot: TAKE | ML: TAKE | ✅ AGREE | Confluence: 9
```

## Phase 3: Optimizer

**Purpose**: Analyze shadow trader data and recommend optimal thresholds.

**File**: `ml_system/optimizer.py`

### How It Works

1. Loads shadow decisions from last N days
2. Calculates bot win rate vs ML win rate at different thresholds (3-10)
3. Identifies which confluence factors correlate with wins
4. Generates recommendations for threshold adjustments
5. Saves report to `ml_system/outputs/optimization_report_TIMESTAMP.txt`

### Usage

```bash
# Run optimizer (on-demand analysis)
py ml_system/optimizer.py --days 30
```

**Arguments**:
- `--days`: Number of days to analyze (default: 30)

### Output Example

```
================================================================================
PERFORMANCE BY CONFLUENCE THRESHOLD
================================================================================
threshold  bot_trades  ml_trades  bot_wins  bot_losses  bot_win_rate  ml_wins  ml_losses  ml_win_rate  improvement
3          120         120        48        72          40.00         48       72         40.00        0.00
4          120         95         48        72          40.00         42       53         44.21        4.21
5          120         68         48        72          40.00         35       33         51.47        11.47
6          120         45         48        72          40.00         28       17         62.22        22.22
7          120         28         48        72          40.00         21       7          75.00        35.00
8          120         15         48        72          40.00         13       2          86.67        46.67
9          120         8          48        72          40.00         7        1          87.50        47.50
10         120         3          48        72          40.00         3        0          100.00       60.00

================================================================================
PERFORMANCE BY CONFLUENCE FACTOR
================================================================================
factor                total_trades  wins  losses  win_rate
weekly_poc            85            68    17      80.00
weekly_hvn            72            56    16      77.78
prev_day_vah          65            48    17      73.85
vwap_band_1           45            30    15      66.67
at_swing_low          38            24    14      63.16
prev_day_poc          42            25    17      59.52
vp_at_poc             28            16    12      57.14
vwap_band_2           15            6     9       40.00

================================================================================
OPTIMIZATION RECOMMENDATIONS
================================================================================
✅ STRONG RECOMMENDATION: Raise minimum confluence to 7
   Current bot win rate: 40.00%
   ML win rate at threshold 7: 75.00%
   Expected improvement: +35.00%

📊 Trade Volume Impact:
   Current: 120 trades
   At threshold 7: 28 trades (-76.7%)
   ML filters out 92 lower-quality trades

🎯 Strongest Confluence Factors:
   weekly_poc: 80.00% win rate (85 trades)
   weekly_hvn: 77.78% win rate (72 trades)
   prev_day_vah: 73.85% win rate (65 trades)

⚠️  Weakest Confluence Factors:
   vwap_band_2: 40.00% win rate (15 trades)
   vp_at_poc: 57.14% win rate (28 trades)
   prev_day_poc: 59.52% win rate (42 trades)
```

### Key Features

- **Threshold optimization**: Tests all thresholds 3-10, finds best performer
- **Factor analysis**: Shows which confluence factors lead to wins
- **Trade-off analysis**: Shows impact on trade volume vs win rate
- **Actionable recommendations**: Clear guidance on what to change

## Integration with Trading Bot

### Current Bot Configuration

**File**: `trading_bot/portfolio/instruments_config.py`

```python
'EURUSD': {
    'windows': [
        {
            'min_confluence_score': 4,  # Current threshold
        }
    ]
}
```

### Optimization Workflow

1. **Run bot** for 2-4 weeks to collect data
2. **Run continuous logger** in background:
   ```bash
   py ml_system/continuous_logger.py --login XXX --password "XXX" --server XXX
   ```

3. **Run shadow trader** in background:
   ```bash
   py ml_system/shadow_trader.py --login XXX --password "XXX" --server XXX --min-confluence 5
   ```

4. **Run optimizer** weekly:
   ```bash
   py ml_system/optimizer.py --days 30
   ```

5. **Review recommendations** in output report

6. **Adjust bot thresholds** based on optimizer findings:
   ```python
   'min_confluence_score': 7,  # Raised from 4 based on ML analysis
   ```

7. **Repeat** - System continuously learns and improves

## Data Storage

All ML data is stored in `ml_system/outputs/`:

```
ml_system/outputs/
├── continuous_trade_log.jsonl          # Phase 1: All trades with confluence factors
├── shadow_decisions.jsonl              # Phase 2: ML vs Bot decisions
└── optimization_report_20260106.txt    # Phase 3: Analysis reports
```

**JSONL Format**: One JSON object per line
- Easy to append (no need to rewrite entire file)
- Easy to parse (read line by line)
- Never corrupts existing data

## Comparison: Old vs New ML System

### Old System (Removed)

**Files**: `analyzer.py`, `enhanced_analyzer.py`, `shadow_trader.py` (old version)

**Problems**:
- ❌ Used generic indicators (RSI, MACD, Bollinger) that bot doesn't use
- ❌ Manual execution only (user had to run scripts)
- ❌ No automatic logging of bot's trades
- ❌ No performance comparison between ML and bot
- ❌ No optimization recommendations

### New System (Current)

**Files**: `continuous_logger.py`, `shadow_trader.py` (new), `optimizer.py`

**Benefits**:
- ✅ Uses bot's actual confluence factors (VWAP, Volume Profile, HTF levels)
- ✅ Automatic continuous logging (runs in background)
- ✅ Tracks every trade the bot makes
- ✅ Compares ML vs Bot performance systematically
- ✅ Generates actionable optimization recommendations
- ✅ Self-optimizing: Bot improves over time based on ML analysis

## Advanced Usage

### Run All 3 Phases Simultaneously

**Terminal 1** (Continuous Logger):
```bash
py ml_system/continuous_logger.py --login XXX --password "XXX" --server XXX
```

**Terminal 2** (Shadow Trader):
```bash
py ml_system/shadow_trader.py --login XXX --password "XXX" --server XXX --min-confluence 5
```

**Terminal 3** (Optimizer - run weekly):
```bash
# Every Monday, analyze last 30 days
py ml_system/optimizer.py --days 30
```

### Test Different ML Thresholds

Run multiple shadow traders with different thresholds:

**Terminal 1** (Conservative - threshold 7):
```bash
py ml_system/shadow_trader.py --login XXX --password "XXX" --server XXX --min-confluence 7
```

**Terminal 2** (Moderate - threshold 5):
```bash
py ml_system/shadow_trader.py --login XXX --password "XXX" --server XXX --min-confluence 5
```

Compare results to find optimal threshold.

### Analyze Specific Time Periods

```bash
# Last 7 days (recent performance)
py ml_system/optimizer.py --days 7

# Last 60 days (long-term trends)
py ml_system/optimizer.py --days 60
```

## Troubleshooting

### "No shadow decisions found"

**Cause**: Shadow trader hasn't run yet or no trades collected.

**Solution**: Run shadow trader for a few days to collect data first.

### "Not connected to MT5"

**Cause**: MT5 login credentials incorrect or MT5 not running.

**Solution**:
1. Check login/password/server are correct
2. Ensure MT5 terminal is running
3. Check firewall isn't blocking MT5

### "Failed to fetch bars"

**Cause**: Symbol not available or not enough historical data.

**Solution**:
1. Ensure symbol is visible in MT5 Market Watch
2. Load historical data in MT5 (View → Symbols → Download history)

## Future Enhancements

Potential Phase 4+ additions:

1. **Auto-Threshold Adjustment**: Bot automatically updates thresholds when ML consistently outperforms
2. **Real-Time ML Scoring**: Calculate ML score BEFORE opening trade, skip if below threshold
3. **Factor Weighting Optimization**: Adjust `CONFLUENCE_WEIGHTS` based on factor win rates
4. **Symbol-Specific Optimization**: Different thresholds for EURUSD vs GBPUSD
5. **Time-Based Optimization**: Different thresholds for different trading sessions
6. **ML Model Training**: Train actual ML models (Random Forest, XGBoost) on confluence data

## Summary

The 3-phase ML system creates a feedback loop:

1. **Continuous Logger** → Captures what the bot does
2. **Shadow Trader** → Shows what ML would have done differently
3. **Optimizer** → Recommends how to improve the bot

Over time, this creates a **self-optimizing trading system** that learns from its own performance and continuously improves.

**Your vision achieved**: The ML system "ghost writes" trades (shadow trader), compares them to the bot's actual trades, and recommends optimal thresholds based on historical performance data.
