#!/usr/bin/env python3
"""
Automatic ML Retraining Scheduler
Retrains ML models every 8 hours automatically
"""

import sys
import time
import schedule
import subprocess
import logging
from datetime import datetime
import os
from pathlib import Path

# Add project root to path dynamically
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
log_dir = 'ml_system/logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/auto_retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AutoRetrain')

def retrain_models():
    """Execute model retraining"""
    logger.info("=" * 80)
    logger.info("AUTOMATIC MODEL RETRAINING STARTED")
    logger.info("=" * 80)

    try:
        # Run retraining script
        result = subprocess.run(
            ['python', 'ml_system/scripts/retrain_all_models.py'],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            logger.info("[OK] Model retraining completed successfully")
            logger.info(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        else:
            logger.error("[ERROR] Model retraining failed")
            logger.error(result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        logger.error("[ERROR] Model retraining timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Error during retraining: {e}")
        return False

def check_new_data():
    """Check if there's new trade data worth retraining on"""
    try:
        import json

        # Count closed trades
        closed_count = 0
        with open('ml_system/outputs/continuous_trade_log.jsonl', 'r') as f:
            for line in f:
                trade = json.loads(line)
                if trade.get('outcome', {}).get('status') == 'closed':
                    closed_count += 1

        logger.info(f"Found {closed_count} closed trades in log")

        # Only retrain if we have at least 8 trades
        if closed_count < 8:
            logger.warning(f"Insufficient data for retraining (need 8+, have {closed_count})")
            return False

        return True

    except Exception as e:
        logger.error(f"Error checking data: {e}")
        return False

def scheduled_retrain():
    """Scheduled retraining job"""
    logger.info(f"Scheduled retrain job triggered at {datetime.now()}")

    # Check if we have enough data
    if not check_new_data():
        logger.info("Skipping retrain - insufficient new data")
        return

    # Retrain models
    success = retrain_models()

    if success:
        logger.info("[OK] Scheduled retraining completed successfully")
    else:
        logger.error("[ERROR] Scheduled retraining failed")

def main():
    """Run automatic retraining scheduler"""
    logger.info("=" * 80)
    logger.info("ML AUTOMATIC RETRAINING SCHEDULER STARTED")
    logger.info("=" * 80)
    logger.info(f"Schedule: Every 8 hours")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 80)

    # Schedule retraining every 8 hours
    schedule.every(8).hours.do(scheduled_retrain)

    # Run initial retrain
    logger.info("Running initial retraining...")
    scheduled_retrain()

    # Keep running
    logger.info("Scheduler active. Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")

if __name__ == '__main__':
    # Check if schedule module is installed
    try:
        import schedule
    except ImportError:
        print("Error: 'schedule' module not installed")
        print("Install with: pip install schedule")
        sys.exit(1)

    main()
