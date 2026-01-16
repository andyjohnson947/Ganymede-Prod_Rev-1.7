#!/usr/bin/env python3
"""
Confluence Trading Bot - Main Entry Point
Based on EA analysis of 428 trades with 64.3% win rate

Usage:
    python main.py --login 12345 --password "yourpass" --server "Broker-Server"
    python main.py --gui  # Launch with GUI
"""

import argparse
import sys
import os
from pathlib import Path

# CACHE CLEARING - Force Python to reload modules on every start
# This ensures code changes are always picked up without needing to restart Python interpreter
import importlib
if hasattr(importlib, 'invalidate_caches'):
    importlib.invalidate_caches()

# Clear __pycache__ for trading_bot modules to force fresh imports
def clear_pycache():
    """Clear Python cache files to ensure fresh module imports"""
    current_dir = Path(__file__).parent
    for pycache_dir in current_dir.rglob('__pycache__'):
        for cache_file in pycache_dir.glob('*.pyc'):
            try:
                cache_file.unlink()
            except Exception:
                pass

# Clear cache on startup
clear_pycache()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Disable Python bytecode generation (optional - prevents .pyc creation)
sys.dont_write_bytecode = True

from core.mt5_manager import MT5Manager
from strategies.confluence_strategy import ConfluenceStrategy
from utils.logger import logger
from config.strategy_config import SYMBOLS

# ML System Integration
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add project root to path
from ml_system.ml_system_startup import start_ml_system, stop_ml_system
from ml_system.continuous_logger import ContinuousMLLogger
import threading

# Global variables for continuous logger
_logger_instance = None  # Shared logger instance for manual logging if needed
_logger_thread = None
_logger_running = False
mt5_api_lock = threading.Lock()  # Global lock for thread-safe MT5 API access


def start_continuous_logger_with_backfill(login, password, server):
    """
    Initialize continuous logger and run backfill, then start background monitoring

    Threading fix: Uses shared lock (mt5_api_lock) to serialize MT5 API access.
    Backfill runs once in main thread, then logger monitors continuously in background.
    """
    global _logger_instance, _logger_thread, _logger_running

    try:
        # Threading fix: Reuse existing MT5 connection from main thread
        # This prevents MT5 API conflicts when connecting in multiple threads
        _logger_instance = ContinuousMLLogger(use_existing_connection=True)

        if not _logger_instance.connect_mt5(login, password, server):
            logger.warning("Failed to connect continuous logger to MT5")
            return False

        logger.info("[OK] Continuous logger connected")
        logger.info("[INFO] Tracking: Entry + Recovery (DCA/Hedge) + Grid + Partials")

        # Run backfill ONCE in main thread (using lock for thread safety)
        logger.info("[INFO] Checking for missed trades...")
        try:
            with mt5_api_lock:
                _logger_instance.backfill_missed_trades()
            logger.info("[OK] Backfill completed")
        except Exception as e:
            logger.error(f"[ERROR] Backfill failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info("[INFO] Continuing without backfill...")

        # Start continuous monitoring in background thread
        _logger_running = True

        def logger_worker():
            global _logger_running
            try:
                while _logger_running:
                    try:
                        # Use lock to prevent MT5 API conflicts with main strategy
                        with mt5_api_lock:
                            _logger_instance.check_for_new_trades()
                            _logger_instance.update_closed_trades()
                    except Exception as e:
                        if _logger_running:
                            logger.error(f"Logger check failed: {e}")
                    threading.Event().wait(60)  # Check every 60 seconds
            except Exception as e:
                logger.error(f"Logger thread crashed: {e}")

        _logger_thread = threading.Thread(target=logger_worker, daemon=True)
        _logger_thread.start()
        logger.info("[OK] Continuous monitoring started (60s interval)")

        return True
    except Exception as e:
        logger.error(f"Failed to initialize continuous logger: {e}")
        return False


def stop_continuous_logger():
    """Stop continuous ML logger background thread"""
    global _logger_running
    _logger_running = False
    logger.info("Stopping continuous logger...")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Confluence Trading Bot - Based on EA Reverse Engineering'
    )

    parser.add_argument(
        '--login',
        type=int,
        help='MT5 account login number'
    )

    parser.add_argument(
        '--password',
        type=str,
        help='MT5 account password'
    )

    parser.add_argument(
        '--server',
        type=str,
        help='MT5 server name'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Trading symbols (default: from config)'
    )

    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch with GUI interface'
    )

    parser.add_argument(
        '--paper-trade',
        action='store_true',
        help='Paper trading mode (simulation only)'
    )

    parser.add_argument(
        '--test-mode',
        action='store_true',
        help=' TEST MODE: Trade all day, bypass time filters (for testing only)'
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    # Print banner
    print()
    print("=" * 80)
    print("     CONFLUENCE TRADING BOT - UPGRADED + ML AUTOMATION")
    print("     Timezone-Aware | Instrument-Specific Trading Windows | ML System")
    print("=" * 80)
    print()
    print("🔄 Cache Status: CLEARED (Fresh imports enabled)")
    print()
    print("Strategy Parameters:")
    print("  * Win Rate: 64.3%")
    print("  * Minimum Confluence Score: 4")
    print("  * Base Lot Size: 0.04 (updated)")
    print("  * Grid Spacing: 8 pips")
    print("  * Hedge Trigger: 8 pips (5x ratio)")
    print()
    print("FEATURES:")
    print("  [+] Timezone: GMT/GMT+1 with automatic DST handling")
    print("  [+] Trading Windows: Instrument-specific entry/exit times")
    print("  [+] Restrictions: No bank holidays, weekends, Friday afternoons")
    print("  [+] Auto-close negative positions at window end")
    print("  [+] Auto cache clearing: Code changes always picked up")
    print()
    print("ML SYSTEM (Automated):")
    print("  [+] Trade Logger: 60s check interval (Entry + Recovery + Grid + Partials)")
    print("  [+] Model Retraining: Every 8 hours")
    print("  [+] Daily Reports: 8:00 AM (with email delivery)")
    print("  [+] Performance Tracking: Automatic profile building")
    print("  [+] Shadow Mode: ML observes, bot controls trades")
    print()
    print("=" * 80)
    print()

    # Check if GUI mode
    if args.gui:
        launch_gui(args)
        return

    # Validate credentials
    if not all([args.login, args.password, args.server]):
        print("[ERROR] Error: MT5 credentials required")
        print("   Use: --login LOGIN --password PASSWORD --server SERVER")
        print("   Or use: --gui for graphical interface")
        sys.exit(1)

    # Get symbols
    symbols = args.symbols if args.symbols else SYMBOLS
    if not symbols:
        print("[ERROR] Error: No symbols specified")
        print("   Use: --symbols EURUSD GBPUSD")
        sys.exit(1)

    # Connect to MT5
    logger.info(f"Connecting to MT5 - Login: {args.login}, Server: {args.server}")

    mt5_manager = MT5Manager(
        login=args.login,
        password=args.password,
        server=args.server,
        api_lock=mt5_api_lock  # Pass lock for thread-safe API access
    )

    if not mt5_manager.connect():
        logger.error("Failed to connect to MT5")
        sys.exit(1)

    # Start ML System (auto-retraining + daily reports)
    logger.info("Starting ML System automation...")
    start_ml_system()
    logger.info("[OK] ML System started (retraining every 8h, reports daily at 8 AM)")

    # Start Continuous Logger and run backfill (synchronously to avoid threading issues)
    logger.info("Starting Continuous Trade Logger...")
    if not start_continuous_logger_with_backfill(args.login, args.password, args.server):
        logger.warning("[WARN] Continuing without continuous logger")

    # DEBUG: Verify MT5 connection is still valid after backfill
    logger.info("[DEBUG] Verifying MT5 connection after backfill...")
    test_account = mt5_manager.get_account_info()
    if test_account:
        logger.info(f"[DEBUG] MT5 connection OK - Balance: ${test_account['balance']:.2f}")
    else:
        logger.error("[DEBUG] MT5 connection FAILED after backfill - reconnecting...")
        mt5_manager.disconnect()
        if not mt5_manager.connect():
            logger.error("Failed to reconnect to MT5")
            sys.exit(1)

    try:
        # Initialize strategy with ML logger for trailing stop monitoring
        logger.info("[DEBUG] Creating ConfluenceStrategy instance...")
        strategy = ConfluenceStrategy(mt5_manager, test_mode=args.test_mode, ml_logger=_logger_instance)
        logger.info("[DEBUG] ConfluenceStrategy created successfully")

        # Show test mode warning if enabled
        if args.test_mode:
            print("\n" + "=" * 80)
            print("[WARN]  TEST MODE ENABLED - TRADING ALL DAY (NO TIME FILTERS)")
            print("=" * 80)
            print()

        # Start trading
        logger.info(f"[DEBUG] Calling strategy.start() with symbols: {symbols}")
        strategy.start(symbols)
        logger.info("[DEBUG] strategy.start() returned")

    except KeyboardInterrupt:
        print("\n\n[WARN]  Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        logger.info("Stopping ML System...")
        stop_ml_system()
        stop_continuous_logger()
        mt5_manager.disconnect()
        logger.info("Bot stopped")


def launch_gui(args):
    """
    Launch GUI interface

    Args:
        args: Command line arguments
    """
    try:
        # Import GUI here to avoid dependency if not using GUI
        from gui.trading_gui import TradingGUI
        import tkinter as tk

        root = tk.Tk()
        app = TradingGUI(root)
        root.mainloop()

    except ImportError as e:
        print(f"[ERROR] GUI dependencies not available: {e}")
        print("   Install required packages:")
        print("   pip install tkinter matplotlib")
        sys.exit(1)


if __name__ == "__main__":
    main()
