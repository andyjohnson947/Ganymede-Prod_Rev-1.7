"""
Logging utility for trading bot
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from config.strategy_config import LOG_LEVEL, LOG_FILE, LOG_TRADES, LOG_SIGNALS


class TradingLogger:
    """Custom logger for trading bot"""

    def __init__(self, name: str = "TradingBot"):
        """
        Initialize logger

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOG_LEVEL))

        # Prevent duplicate handlers if logger already configured
        if self.logger.handlers:
            return

        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # File handler
        log_path = log_dir / LOG_FILE
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Console handler with UTF-8 encoding support
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, LOG_LEVEL))

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers with error handling for console encoding
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Configure stdout to use UTF-8 on Windows
        if sys.platform == 'win32':
            try:
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            except:
                pass  # If fails, continue with default encoding

        # Separate files for trades and signals
        if LOG_TRADES:
            self.trade_logger = self._create_file_logger('trades', log_dir / 'trades.log')
        else:
            self.trade_logger = None

        if LOG_SIGNALS:
            self.signal_logger = self._create_file_logger('signals', log_dir / 'signals.log')
        else:
            self.signal_logger = None

    def _create_file_logger(self, name: str, file_path: Path) -> logging.Logger:
        """Create a separate file logger"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        handler = logging.FileHandler(file_path, encoding='utf-8')
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        # Prevent propagation to parent logger (avoid duplicate console output)
        logger.propagate = False
        return logger

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)

    def log_trade(self, trade_info: dict):
        """
        Log trade execution

        Args:
            trade_info: Dict with trade details
        """
        if not self.trade_logger:
            return

        message = (
            f"TRADE | Ticket: {trade_info.get('ticket')} | "
            f"Symbol: {trade_info.get('symbol')} | "
            f"Type: {trade_info.get('type')} | "
            f"Volume: {trade_info.get('volume')} | "
            f"Price: {trade_info.get('price')} | "
            f"Comment: {trade_info.get('comment', '')}"
        )

        self.trade_logger.info(message)

    def log_signal(self, signal_info: dict):
        """
        Log trading signal

        Args:
            signal_info: Dict with signal details
        """
        if not self.signal_logger:
            return

        message = (
            f"SIGNAL | Symbol: {signal_info.get('symbol')} | "
            f"Direction: {signal_info.get('direction')} | "
            f"Score: {signal_info.get('confluence_score')} | "
            f"Factors: {', '.join(signal_info.get('factors', []))}"
        )

        self.signal_logger.info(message)


# Global logger instance
logger = TradingLogger()
