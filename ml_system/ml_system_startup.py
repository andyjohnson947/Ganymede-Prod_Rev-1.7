#!/usr/bin/env python3
"""
ML System Startup Integration
Import and call from your trading bot to start ML automation
"""

import sys
import os
import logging
import threading
import time
import schedule
import subprocess
from datetime import datetime
from pathlib import Path

# Add project root to path dynamically
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging - only if not already configured
log_dir = 'ml_system/logs'
os.makedirs(log_dir, exist_ok=True)

# Get ML system logger
logger = logging.getLogger('MLSystem')
logger.setLevel(logging.INFO)

# Only add handlers if not already present (prevent duplicates)
if not logger.handlers:
    # File handler for ML system logs
    file_handler = logging.FileHandler(f'{log_dir}/ml_system.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)

    # Don't add console handler - use parent logger (TradingBot) to avoid duplicates
    # This allows ML logs to appear in trading bot console via propagation

class MLSystemManager:
    """
    Manages ML system automation (retraining + reports) in background thread
    """

    def __init__(self):
        self.running = False
        self.thread = None

    def retrain_models(self):
        """Auto-retrain ML models"""
        logger.info("=" * 60)
        logger.info("AUTO-RETRAIN: Starting...")

        try:
            # Import and call retraining directly (avoids encoding and path issues)
            import json

            # Count closed trades
            closed_count = 0
            trade_log = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'ml_system', 'outputs', 'continuous_trade_log.jsonl'
            )

            if os.path.exists(trade_log):
                with open(trade_log, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        try:
                            trade = json.loads(line)
                            if trade.get('outcome', {}).get('status') == 'closed':
                                closed_count += 1
                        except:
                            continue

            if closed_count < 8:
                logger.info(f"[SKIP] Only {closed_count} closed trades (need 8+)")
                return

            logger.info(f"Retraining with {closed_count} closed trades...")

            # Run scripts directly (not via exec - avoids __file__ issues)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

            # Step 1: Create dataset
            dataset_script = os.path.join(project_root, 'ml_system', 'scripts', 'create_dataset.py')
            result1 = subprocess.run(
                [sys.executable, dataset_script],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=300,
                encoding='utf-8',
                errors='ignore'
            )

            if result1.returncode != 0:
                logger.error(f"[ERROR] Dataset creation failed")
                if result1.stderr:
                    logger.error(result1.stderr[:300])
                return

            # Step 2: Train baseline model
            model_script = os.path.join(project_root, 'ml_system', 'models', 'baseline_model.py')
            result2 = subprocess.run(
                [sys.executable, model_script],
                capture_output=True,
                text=True,
                cwd=project_root,
                timeout=300,
                encoding='utf-8',
                errors='ignore'
            )

            if result2.returncode == 0:
                logger.info("[OK] Model retraining completed")
            else:
                logger.error(f"[ERROR] Model training failed")
                if result2.stderr:
                    logger.error(result2.stderr[:300])

        except Exception as e:
            logger.error(f"[ERROR] Error during retraining: {e}")
            import traceback
            logger.error(traceback.format_exc()[:500])

    def generate_report(self):
        """Generate daily ML decision report"""
        logger.info("=" * 60)
        logger.info("DAILY DECISION REPORT: Generating...")

        try:
            # Import decision report generator (actionable recommendations)
            from ml_system.reports.decision_report import DecisionReportGenerator

            generator = DecisionReportGenerator()
            report_text, report_file = generator.generate_report()

            logger.info(f"[OK] Decision report generated: {report_file}")

            # Try to send email if configured
            import json
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            email_config_file = os.path.join(project_root, 'config', 'email_config.json')

            logger.info(f"Looking for email config at: {email_config_file}")

            if os.path.exists(email_config_file):
                with open(email_config_file, 'r', encoding='utf-8', errors='ignore') as f:
                    email_config = json.load(f)

                if email_config.get('enabled', False):
                    logger.info(f"Email enabled, attempting to send to {email_config.get('to_email')}")
                    success = generator.send_email(report_text, email_config)
                    if success:
                        logger.info(f"[OK] Email sent to {email_config['to_email']}")
                    else:
                        logger.warning("[WARN] Email send failed (check logs above for details)")
                else:
                    logger.info("[INFO] Email notifications disabled in config")
            else:
                logger.warning(f"[WARN] Email config file not found at {email_config_file}")

        except Exception as e:
            logger.error(f"[ERROR] Error generating report: {e}")
            import traceback
            logger.error(traceback.format_exc()[:500])

    def _run_scheduler(self):
        """Run scheduler loop in background thread"""
        logger.info("ML System scheduler started")

        # Schedule recurring jobs
        schedule.every(8).hours.do(self.retrain_models)
        schedule.every().day.at("08:00").do(self.generate_report)

        # Run initial jobs once (non-blocking)
        logger.info("Running initial ML setup...")
        logger.info("[INFO] Initial retrain running in background...")

        # Flag to track if initial jobs are done
        initial_jobs_done = False

        # Keep running
        while self.running:
            # Run initial jobs on first iteration only
            if not initial_jobs_done:
                try:
                    self.retrain_models()
                    self.generate_report()
                except Exception as e:
                    logger.error(f"[ERROR] Initial jobs failed: {e}")
                finally:
                    initial_jobs_done = True
                    logger.info("[OK] Initial ML setup completed")

            # Run scheduled jobs
            schedule.run_pending()
            time.sleep(60)

        logger.info("ML System scheduler stopped")

    def start(self):
        """Start ML system in background thread"""
        if self.running:
            logger.warning("ML System already running")
            return

        logger.info("=" * 60)
        logger.info("STARTING ML SYSTEM")
        logger.info("=" * 60)
        logger.info("Schedules:")
        logger.info("  * Model Retraining: Every 8 hours")
        logger.info("  * Daily Report: Every day at 08:00")
        logger.info("=" * 60)

        self.running = True
        self.thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.thread.start()

        logger.info("[OK] ML System started in background")

    def stop(self):
        """Stop ML system"""
        if not self.running:
            return

        logger.info("Stopping ML System...")
        self.running = False

        # Don't wait for thread - daemon will exit automatically
        # Avoids KeyboardInterrupt issues during shutdown
        logger.info("[OK] ML System stopped")

    def is_running(self):
        """Check if ML system is running"""
        return self.running


# Global instance
_ml_system = None


def start_ml_system():
    """
    Start ML system automation
    Call this from your trading bot's startup code
    """
    global _ml_system

    if _ml_system is None:
        _ml_system = MLSystemManager()

    _ml_system.start()
    return _ml_system


def stop_ml_system():
    """
    Stop ML system automation
    Call this from your trading bot's shutdown code
    """
    global _ml_system

    if _ml_system:
        _ml_system.stop()


def get_ml_system():
    """Get ML system instance"""
    return _ml_system


# Example usage for testing
if __name__ == '__main__':
    logger.info("Testing ML System startup...")

    # Start ML system
    ml_system = start_ml_system()

    # Keep running for testing (normally your bot keeps running)
    try:
        logger.info("ML System running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        stop_ml_system()
