# ML Automated Scheduler & Daily Reports

## Overview

Automated system for:
- **Model Retraining**: Every 8 hours
- **Daily Reports**: Every day at 8:00 AM
- **Email Reports**: Optional email delivery

## Quick Start

### 1. Install Dependencies

```bash
pip install schedule
```

### 2. Configure Email (Optional)

Copy and edit email config:

```bash
cd /home/user/Ganymede-Prod_Rev-1.4
cp config/email_config.example.json config/email_config.json
nano config/email_config.json
```

Update with your email settings:
- `enabled`: Set to `true` to enable email
- `from_email`: Your Gmail address
- `to_email`: Recipient email
- `password`: App-specific password (see below)
- `smtp_server`: Keep as `smtp.gmail.com` for Gmail
- `smtp_port`: Keep as `587`

**For Gmail:**
1. Enable 2-Factor Authentication
2. Create app-specific password: https://myaccount.google.com/apppasswords
3. Use this password in the config (not your regular password)

### 3. Configure Bot Settings

Copy and edit trading config:

```bash
cp config/trading_config.example.json config/trading_config.json
nano config/trading_config.json
```

**IMPORTANT:** Update this file to match your actual bot configuration!

The daily report compares these settings with ML recommendations.

**Current ML-Recommended Hedge Settings:**
```json
{
  "hedge_enabled": true,
  "hedge_timing_minutes": 46,
  "hedge_price_move_pct": 15.0,
  "hedge_adx_max": 20
}
```

These are based on your historical data analysis showing:
- Winners waited ~50 minutes before hedging
- Winners waited for ~15% price move
- All successful hedges in ranging markets (ADX < 20)

### 4. Start the Scheduler

Run the master scheduler:

```bash
cd /home/user/Ganymede-Prod_Rev-1.4
python ml_system/start_ml_scheduler.py
```

Or run as background process:

```bash
nohup python ml_system/start_ml_scheduler.py &> ml_system/logs/scheduler.log &
```

### 5. Stop the Scheduler

If running in foreground: Press `Ctrl+C`

If running in background:
```bash
ps aux | grep start_ml_scheduler
kill <PID>
```

## Schedule Details

### Auto-Retraining (Every 8 Hours)
- Checks for new closed trades
- Regenerates dataset with 60 features
- Retrains all models (baseline, tuned, ensemble)
- Logs to: `ml_system/logs/auto_retrain.log`

### Daily Reports (Every Day at 8:00 AM)
- Analyzes last 24 hours performance
- Compares bot config vs ML recommendations
- Generates feature importance rankings
- Saves to: `ml_system/reports/daily/report_YYYY-MM-DD.txt`
- Optionally emails report

## What's in the Daily Report?

1. **Current Bot Configuration**
   - All your bot settings (confluence, hedge, DCA, etc.)

2. **24h Performance**
   - Trades opened/closed
   - Win rate, profit

3. **ML Performance**
   - ML accuracy vs actual performance
   - Confidence levels

4. **ML Recommendations**
   - Based on last 7 days of data
   - Suggested config changes
   - Market regime analysis
   - Hedge effectiveness

5. **Feature Importance**
   - Top 10 features driving predictions

6. **System Health**
   - Model status, data availability

## Manual Operations

### Generate Report Manually

```bash
python ml_system/reports/daily_report.py
```

### Retrain Models Manually

```bash
python ml_system/scripts/retrain_all_models.py
```

### View Latest Report

```bash
cat ml_system/reports/daily/report_$(date +%Y-%m-%d).txt
```

### View Scheduler Logs

```bash
tail -f ml_system/logs/ml_scheduler.log
```

## Profile Building

Daily reports are saved as:
- `ml_system/reports/daily/report_2026-01-09.txt`
- `ml_system/reports/daily/report_2026-01-10.txt`
- etc.

Over time, this builds a **performance profile** showing:
- How ML recommendations evolved
- What config changes were made
- Performance impact of changes
- Market regime patterns

## Troubleshooting

### Email Not Sending

1. Check `config/email_config.json` exists (not `.example`)
2. Verify `enabled: true`
3. Check app-specific password (not regular password)
4. Look for errors in `ml_system/logs/daily_report_scheduler.log`

### Models Not Retraining

1. Check you have 8+ closed trades
2. Look for errors in `ml_system/logs/auto_retrain.log`
3. Verify `ml_system/outputs/continuous_trade_log.jsonl` exists

### Scheduler Not Running

1. Check if `schedule` module installed: `pip list | grep schedule`
2. Check logs: `tail -100 ml_system/logs/ml_scheduler.log`
3. Verify Python version: `python --version` (need 3.8+)

## System Requirements

- Python 3.8+
- Dependencies: pandas, numpy, scikit-learn, xgboost, schedule
- Disk space: ~100MB for logs and reports
- Memory: ~500MB when retraining

## Hedge Configuration Decision

Based on ML analysis of your historical data:

**Hedge ENABLED is recommended** with these settings:
- Wait **46+ minutes** after entry
- Trigger when price moves **15%+** against position
- Only in **ranging markets** (ADX < 20)
- Only on **high confluence** setups (≥20)

**Reasoning:**
- Winners waited ~50 min before hedging
- Losers hedged too early (11 min)
- Hedging too early = only losing trade
- Patient hedging = 75% recovery rate

Update your bot to use these settings and let ML monitor effectiveness!

## Files Created

- `ml_system/start_ml_scheduler.py` - Master scheduler
- `ml_system/scheduler/auto_retrain.py` - Auto-retrain scheduler
- `ml_system/scheduler/daily_report_scheduler.py` - Report scheduler
- `ml_system/reports/daily_report.py` - Report generator
- `config/email_config.example.json` - Email config template
- `config/trading_config.example.json` - Bot config template

## Support

Check logs for errors:
- `ml_system/logs/ml_scheduler.log`
- `ml_system/logs/auto_retrain.log`
- `ml_system/logs/daily_report_scheduler.log`
