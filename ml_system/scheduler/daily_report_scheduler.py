#!/usr/bin/env python3
"""
Daily Report Scheduler
Generates daily ML reports automatically
"""

import sys
import time
import schedule
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
        logging.FileHandler(f'{log_dir}/daily_report_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DailyReportScheduler')

def generate_daily_report():
    """Generate daily ML report"""
    logger.info("=" * 80)
    logger.info("GENERATING DAILY ML REPORT")
    logger.info("=" * 80)

    try:
        from ml_system.reports.daily_report import DailyReportGenerator

        generator = DailyReportGenerator()
        report_text, report_file = generator.generate_report()

        logger.info(f"[OK] Daily report generated: {report_file}")

        # Try to send email if configured
        import json
        email_config_file = 'config/email_config.json'
        if os.path.exists(email_config_file):
            with open(email_config_file, 'r') as f:
                email_config = json.load(f)

            if email_config.get('enabled', False):
                success = generator.send_email(report_text, email_config)
                if success:
                    logger.info(f"[OK] Email sent to {email_config['to_email']}")
                else:
                    logger.error("[ERROR] Failed to send email")
        else:
            logger.info("No email config found - skipping email")

        return True

    except Exception as e:
        logger.error(f"[ERROR] Error generating report: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run daily report scheduler"""
    logger.info("=" * 80)
    logger.info("DAILY REPORT SCHEDULER STARTED")
    logger.info("=" * 80)
    logger.info(f"Schedule: Daily at 08:00")
    logger.info(f"Started at: {datetime.now()}")
    logger.info("=" * 80)

    # Schedule daily report at 8 AM
    schedule.every().day.at("08:00").do(generate_daily_report)

    # Run initial report
    logger.info("Generating initial report...")
    generate_daily_report()

    # Keep running
    logger.info("Scheduler active. Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")

if __name__ == '__main__':
    main()
