# ML System Integration Guide

## Quick Integration (2 Lines of Code)

Add ML automation to your trading bot startup in **2 simple steps**:

### Step 1: Import at the top of your main bot file

```python
from ml_system.ml_system_startup import start_ml_system, stop_ml_system
```

### Step 2: Start ML system when bot starts

```python
# In your bot's startup/initialization code
start_ml_system()
```

**That's it!** ML system now runs in background, automatically:
- Retraining models every 8 hours
- Generating daily reports at 8 AM
- Emailing reports (if configured)

---

## Complete Example

### Example 1: Simple Bot

```python
#!/usr/bin/env python3
"""
Your Trading Bot
"""

# Add ML system import
from ml_system.ml_system_startup import start_ml_system, stop_ml_system

def main():
    print("Starting trading bot...")

    # Start ML system automation
    start_ml_system()

    # Your bot's normal startup code here
    # ...

    # Your bot's main loop
    while True:
        # Your trading logic
        pass

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_ml_system()  # Clean shutdown
```

### Example 2: Class-Based Bot

```python
#!/usr/bin/env python3
"""
Your Trading Bot (Class-based)
"""

from ml_system.ml_system_startup import start_ml_system, stop_ml_system

class TradingBot:
    def __init__(self):
        print("Initializing trading bot...")

        # Start ML system
        start_ml_system()

        # Your initialization code
        # ...

    def start(self):
        print("Starting trading bot...")
        # Your main loop
        # ...

    def stop(self):
        print("Stopping trading bot...")
        stop_ml_system()  # Stop ML system
        # Your cleanup code
        # ...

if __name__ == '__main__':
    bot = TradingBot()

    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
```

### Example 3: With Signal Handlers

```python
#!/usr/bin/env python3
"""
Your Trading Bot (Production-ready)
"""

import signal
import sys
from ml_system.ml_system_startup import start_ml_system, stop_ml_system

class TradingBot:
    def __init__(self):
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

        # Start ML system
        start_ml_system()

    def shutdown(self, signum, frame):
        print("\nShutdown signal received...")
        self.running = False
        stop_ml_system()
        sys.exit(0)

    def run(self):
        while self.running:
            # Your trading logic
            pass

if __name__ == '__main__':
    bot = TradingBot()
    bot.run()
```

---

## What Happens Automatically

Once you call `start_ml_system()`:

1. **Initial Setup** (runs once on startup)
   - Retrains models with latest data
   - Generates first daily report

2. **Every 8 Hours**
   - Checks for new trade data
   - Retrains models if 8+ closed trades
   - Updates model registry

3. **Every Day at 8:00 AM**
   - Generates performance report
   - Compares bot config vs ML recommendations
   - Emails report (if configured)
   - Saves to `ml_system/reports/daily/report_YYYY-MM-DD.txt`

4. **All in Background**
   - Runs in separate thread
   - Won't block your trading bot
   - Continues even if individual tasks fail

---

## Configuration Files (Optional)

### Bot Config (for reports)

Create: `config/trading_config.json`

```json
{
  "confluence_threshold": 15,
  "hedge_enabled": true,
  "hedge_timing_minutes": 46,
  "hedge_price_move_pct": 15.0,
  "hedge_adx_max": 20,
  "dca_enabled": true,
  "dca_max_count": 5,
  "grid_enabled": false,
  "partial_close_enabled": true
}
```

Daily reports will compare these settings with ML recommendations.

### Email Config (for report delivery)

Create: `config/email_config.json`

```json
{
  "enabled": true,
  "from_email": "your-email@gmail.com",
  "to_email": "recipient@example.com",
  "password": "your-app-specific-password",
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587
}
```

**Gmail Setup:**
1. Enable 2FA
2. Create app password: https://myaccount.google.com/apppasswords
3. Use app password (not regular password)

---

## Logs

ML system logs everything to:
- `ml_system/logs/ml_system.log` - Main log
- `ml_system/logs/auto_retrain.log` - Retraining details
- `ml_system/logs/daily_report_scheduler.log` - Report generation

View logs:
```bash
tail -f ml_system/logs/ml_system.log
```

---

## Testing

Test ML system without running your bot:

```bash
python ml_system/ml_system_startup.py
```

Press Ctrl+C to stop.

---

## Troubleshooting

### ML System Not Starting

Check if `schedule` module installed:
```bash
pip install schedule
```

### No Reports Generated

1. Check logs: `cat ml_system/logs/ml_system.log`
2. Verify trades exist: `ls ml_system/outputs/continuous_trade_log.jsonl`

### Email Not Sending

1. Check config exists: `ls config/email_config.json`
2. Verify `enabled: true` in config
3. Use app-specific password (not regular password)

---

## Dependencies

Required packages (auto-installed with ML system):
```bash
pip install pandas numpy scikit-learn xgboost schedule
```

---

## Clean Shutdown

Always stop ML system when shutting down bot:

```python
stop_ml_system()
```

This ensures:
- Scheduled tasks complete
- Logs are flushed
- Threads are cleaned up

---

## Advanced: Check Status

```python
from ml_system.ml_system_startup import get_ml_system

ml_system = get_ml_system()
if ml_system and ml_system.is_running():
    print("ML System: Running")
else:
    print("ML System: Stopped")
```

---

## Summary

**Minimum Integration:**
```python
from ml_system.ml_system_startup import start_ml_system
start_ml_system()  # That's it!
```

**With Clean Shutdown:**
```python
from ml_system.ml_system_startup import start_ml_system, stop_ml_system

# On startup
start_ml_system()

# On shutdown
stop_ml_system()
```

**Result:**
- ✅ Models retrain every 8 hours automatically
- ✅ Daily reports at 8 AM
- ✅ Email delivery (if configured)
- ✅ Performance profile builds over time
- ✅ Zero maintenance needed
