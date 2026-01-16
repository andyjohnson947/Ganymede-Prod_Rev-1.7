# Trading System Launcher Guide

## Overview
The `start_trading_system.py` script manages both the **trading bot** and **continuous ML logger** together in a single command. This ensures all trades are automatically captured for ML analysis.

## Quick Start

### Basic Usage
```bash
python start_trading_system.py --login 12345 --password mypass --server MyBroker-Demo
```

This will:
- ✅ Start the continuous ML logger in the background
- ✅ Start the trading bot in the foreground
- ✅ Monitor both processes and restart the logger if it crashes
- ✅ Stop both processes gracefully when you press Ctrl+C

## Command Line Options

| Option | Required | Description |
|--------|----------|-------------|
| `--login` | Yes | MT5 account login number |
| `--password` | Yes | MT5 account password |
| `--server` | Yes | MT5 broker server name |
| `--symbols` | No | Trading symbols (default: EURUSD GBPUSD) |
| `--test-mode` | No | Enable test mode (trades all day, bypasses time filters) |
| `--logger-interval` | No | Logger check interval in seconds (default: 60) |

## Examples

### Trade specific symbols
```bash
python start_trading_system.py \
  --login 12345 \
  --password mypass \
  --server MyBroker-Demo \
  --symbols EURUSD GBPUSD USDJPY
```

### Test mode (24/7 trading)
```bash
python start_trading_system.py \
  --login 12345 \
  --password mypass \
  --server MyBroker-Demo \
  --test-mode
```

### Faster logger checks (30 second interval)
```bash
python start_trading_system.py \
  --login 12345 \
  --password mypass \
  --server MyBroker-Demo \
  --logger-interval 30
```

## What Gets Logged?

The continuous ML logger captures:
- ✅ **Entry trades** with confluence factors at entry time
- ✅ **Recovery actions** (DCA, Hedge) when they trigger
- ✅ **Grid trades** for profit-taking
- ✅ **Partial closes** for risk management
- ✅ **Exit data** when trades close (profit, duration, recovery usage)

Output file: `ml_system/outputs/continuous_trade_log.jsonl`

## Process Management

### Stopping the System
Press **Ctrl+C** to stop both processes gracefully.

The launcher will:
1. Stop the trading bot first (most important)
2. Stop the ML logger
3. Wait for clean shutdown (max 10 seconds)
4. Force-kill if needed

### Auto-Restart
If the **ML logger crashes**, the launcher will automatically restart it up to **3 times**.

If the **bot crashes**, the launcher will stop everything (bot is critical).

## Monitoring

The launcher shows:
- Logger output with `[LOGGER]` prefix
- Bot output directly in the terminal
- Process status (PID, exit codes)
- Restart attempts

## Configuration

Current bot settings (from `strategy_config.py`):
- **Min Confluence Score**: 8 (updated from 4)
- **Symbols**: EURUSD, GBPUSD
- **Base Lot Size**: 0.04
- **Grid Spacing**: 8 pips
- **Hedge Trigger**: 8 pips (5x ratio)
- **Max Hedge Volume**: 0.50 lots

## Troubleshooting

### Logger won't start
- Check MT5 credentials are correct
- Verify MT5 terminal is running
- Check `ml_system/outputs/` directory exists and is writable

### Bot won't start
- Check MT5 credentials
- Verify symbols are available on your broker
- Check `logs/` directory exists

### Both processes keep restarting
- Check MT5 connection is stable
- Verify account has trading permissions
- Check broker allows automated trading

## Advanced: Running Components Separately

If you need to run them separately:

**ML Logger only:**
```bash
python ml_system/continuous_logger.py \
  --login 12345 \
  --password mypass \
  --server MyBroker-Demo \
  --interval 60
```

**Bot only:**
```bash
python trading_bot/main.py \
  --login 12345 \
  --password mypass \
  --server MyBroker-Demo \
  --symbols EURUSD GBPUSD
```

But using the launcher is **recommended** to ensure both run together!
