"""
Portfolio Manager

Manages trading instruments, their windows, and coordinates trading across multiple instruments.
Handles window-based entry/exit logic and position management across the portfolio.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

from utils.timezone_manager import get_timezone_manager
from portfolio.instruments_config import get_instruments_config


class WindowStatus(Enum):
    """Trading window status"""
    BEFORE_OPEN = "before_open"
    OPEN = "open"
    CLOSING_SOON = "closing_soon"  # Within 5 minutes of close
    CLOSED = "closed"


@dataclass
class TradingWindow:
    """
    Represents a time window for trading an instrument.
    """
    name: str
    start: time
    end: time
    strategy_type: str
    description: str
    close_all_at_end: bool = True          # Close ALL positions at window end
    min_confluence_score: int = 7

    def get_status(self, current_time: datetime) -> WindowStatus:
        """
        Get current status of the trading window.

        Args:
            current_time: Current time (timezone-aware)

        Returns:
            WindowStatus enum value
        """
        tz_mgr = get_timezone_manager()

        if tz_mgr.is_time_in_window(current_time, self.start, self.end):
            # Check if within 5 minutes of close
            time_until_end = tz_mgr.time_until_window_end(current_time, self.end)
            if time_until_end and time_until_end <= timedelta(minutes=5):
                return WindowStatus.CLOSING_SOON
            return WindowStatus.OPEN

        # Check if before or after
        current_time_only = current_time.time()
        if current_time_only < self.start:
            return WindowStatus.BEFORE_OPEN
        else:
            return WindowStatus.CLOSED

    def is_open(self, current_time: datetime) -> bool:
        """Check if window is currently open."""
        status = self.get_status(current_time)
        return status in [WindowStatus.OPEN, WindowStatus.CLOSING_SOON]

    def is_closing_soon(self, current_time: datetime) -> bool:
        """Check if window is closing soon."""
        return self.get_status(current_time) == WindowStatus.CLOSING_SOON

    def time_until_open(self, current_time: datetime) -> Optional[timedelta]:
        """Calculate time until window opens."""
        current_time_only = current_time.time()

        if current_time_only >= self.start:
            # Window opens tomorrow
            tomorrow = current_time + timedelta(days=1)
            window_open = datetime.combine(tomorrow.date(), self.start)
        else:
            # Window opens today
            window_open = datetime.combine(current_time.date(), self.start)

        # Apply timezone
        tz_mgr = get_timezone_manager()
        window_open = tz_mgr.UK_TIMEZONE.localize(window_open)

        return window_open - current_time

    def time_until_close(self, current_time: datetime) -> Optional[timedelta]:
        """Calculate time until window closes."""
        tz_mgr = get_timezone_manager()
        return tz_mgr.time_until_window_end(current_time, self.end)

    def __str__(self) -> str:
        return f"{self.name} ({self.start}-{self.end} {self.strategy_type})"


@dataclass
class TradingInstrument:
    """
    Represents a tradable instrument with its trading windows.
    """
    symbol: str
    name: str
    enabled: bool
    trading_windows: List[TradingWindow] = field(default_factory=list)

    def is_tradeable_now(self, current_time: datetime) -> bool:
        """
        Check if instrument is tradeable at current time.

        Args:
            current_time: Current time (timezone-aware)

        Returns:
            True if any trading window is open
        """
        if not self.enabled:
            return False

        for window in self.trading_windows:
            if window.is_open(current_time):
                return True

        return False

    def get_active_windows(self, current_time: datetime) -> List[TradingWindow]:
        """
        Get all currently active trading windows.

        Args:
            current_time: Current time (timezone-aware)

        Returns:
            List of active windows
        """
        if not self.enabled:
            return []

        return [w for w in self.trading_windows if w.is_open(current_time)]

    def get_closing_windows(self, current_time: datetime) -> List[TradingWindow]:
        """
        Get windows that are closing soon.

        Args:
            current_time: Current time (timezone-aware)

        Returns:
            List of windows closing soon
        """
        if not self.enabled:
            return []

        return [w for w in self.trading_windows if w.is_closing_soon(current_time)]

    def should_close_all_positions(self, current_time: datetime) -> bool:
        """
        Check if all positions should be closed now (window ending).

        Args:
            current_time: Current time (timezone-aware)

        Returns:
            True if any window is closing soon and requires position closure
        """
        closing_windows = self.get_closing_windows(current_time)
        return any(w.close_all_at_end for w in closing_windows)

    def get_min_confluence_score(self, current_time: datetime) -> int:
        """
        Get minimum confluence score for current windows.

        Args:
            current_time: Current time (timezone-aware)

        Returns:
            Minimum confluence score required
        """
        active_windows = self.get_active_windows(current_time)
        if not active_windows:
            return 10  # High default when no window is active

        return min(w.min_confluence_score for w in active_windows)

    def __str__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"{self.name} ({self.symbol}) - {len(self.trading_windows)} windows [{status}]"


@dataclass
class CloseAction:
    """Represents an action to close positions."""
    symbol: str
    reason: str
    close_negatives_only: bool
    window_name: str


class PortfolioManager:
    """
    Manages the portfolio of trading instruments and their windows.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize portfolio manager.

        Args:
            config: Instruments configuration (uses default if None)
        """
        self.instruments: Dict[str, TradingInstrument] = {}
        self.tz_manager = get_timezone_manager()

        # Load configuration
        if config is None:
            config = get_instruments_config()

        self._load_instruments(config)

    def _load_instruments(self, config: Dict):
        """Load instruments from configuration."""
        for symbol, instrument_config in config.items():
            # Create trading windows
            windows = []
            for window_config in instrument_config.get('windows', []):
                window = TradingWindow(
                    name=window_config.get('name', 'Unknown'),
                    start=window_config['start'],
                    end=window_config['end'],
                    strategy_type=window_config.get('strategy_type', 'unknown'),
                    description=window_config.get('description', ''),
                    close_all_at_end=window_config.get('close_all_at_end', True),
                    min_confluence_score=window_config.get('min_confluence_score', 7)
                )
                windows.append(window)

            # Create instrument
            instrument = TradingInstrument(
                symbol=symbol,
                name=instrument_config.get('name', symbol),
                enabled=instrument_config.get('enabled', True),
                trading_windows=windows
            )

            self.instruments[symbol] = instrument

    def add_instrument(self, instrument: TradingInstrument):
        """Add an instrument to the portfolio."""
        self.instruments[instrument.symbol] = instrument

    def remove_instrument(self, symbol: str):
        """Remove an instrument from the portfolio."""
        if symbol in self.instruments:
            del self.instruments[symbol]

    def enable_instrument(self, symbol: str):
        """Enable trading for an instrument."""
        if symbol in self.instruments:
            self.instruments[symbol].enabled = True

    def disable_instrument(self, symbol: str):
        """Disable trading for an instrument."""
        if symbol in self.instruments:
            self.instruments[symbol].enabled = False

    def get_tradeable_instruments(self, current_time: Optional[datetime] = None) -> List[TradingInstrument]:
        """
        Get list of instruments that are currently tradeable.

        Args:
            current_time: Current time (uses current if None)

        Returns:
            List of tradeable instruments
        """
        if current_time is None:
            current_time = self.tz_manager.get_current_trading_timezone()

        return [
            instrument for instrument in self.instruments.values()
            if instrument.is_tradeable_now(current_time)
        ]

    def get_tradeable_symbols(self, current_time: Optional[datetime] = None) -> List[str]:
        """
        Get list of symbol names that are currently tradeable.

        Args:
            current_time: Current time (uses current if None)

        Returns:
            List of tradeable symbol strings
        """
        return [inst.symbol for inst in self.get_tradeable_instruments(current_time)]

    def is_symbol_tradeable(self, symbol: str, current_time: Optional[datetime] = None) -> bool:
        """
        Check if a specific symbol is tradeable now.

        Args:
            symbol: Symbol to check
            current_time: Current time (uses current if None)

        Returns:
            True if tradeable
        """
        if symbol not in self.instruments:
            return False

        if current_time is None:
            current_time = self.tz_manager.get_current_trading_timezone()

        return self.instruments[symbol].is_tradeable_now(current_time)

    def check_window_closures(self, current_time: Optional[datetime] = None) -> List[CloseAction]:
        """
        Check for windows that are closing and require position closure.

        Args:
            current_time: Current time (uses current if None)

        Returns:
            List of close actions to perform
        """
        if current_time is None:
            current_time = self.tz_manager.get_current_trading_timezone()

        close_actions = []

        for instrument in self.instruments.values():
            if not instrument.enabled:
                continue

            closing_windows = instrument.get_closing_windows(current_time)

            for window in closing_windows:
                if window.close_all_at_end:
                    action = CloseAction(
                        symbol=instrument.symbol,
                        reason=f"Window closing: {window.name}",
                        close_negatives_only=False,  # Close ALL positions (not just negatives)
                        window_name=window.name
                    )
                    close_actions.append(action)

        return close_actions

    def get_instrument_status(self, symbol: str, current_time: Optional[datetime] = None) -> Dict:
        """
        Get detailed status for an instrument.

        Args:
            symbol: Instrument symbol
            current_time: Current time (uses current if None)

        Returns:
            Dictionary with status information
        """
        if symbol not in self.instruments:
            return {'error': 'Instrument not found'}

        if current_time is None:
            current_time = self.tz_manager.get_current_trading_timezone()

        instrument = self.instruments[symbol]

        active_windows = instrument.get_active_windows(current_time)
        closing_windows = instrument.get_closing_windows(current_time)

        window_statuses = []
        for window in instrument.trading_windows:
            status = window.get_status(current_time)
            time_info = {}

            if status == WindowStatus.BEFORE_OPEN:
                time_info['time_until_open'] = str(window.time_until_open(current_time))
            elif status in [WindowStatus.OPEN, WindowStatus.CLOSING_SOON]:
                time_info['time_until_close'] = str(window.time_until_close(current_time))

            window_statuses.append({
                'name': window.name,
                'status': status.value,
                'start': str(window.start),
                'end': str(window.end),
                **time_info
            })

        return {
            'symbol': symbol,
            'name': instrument.name,
            'enabled': instrument.enabled,
            'tradeable_now': instrument.is_tradeable_now(current_time),
            'active_windows': len(active_windows),
            'closing_windows': len(closing_windows),
            'should_close_all_positions': instrument.should_close_all_positions(current_time),
            'min_confluence_score': instrument.get_min_confluence_score(current_time),
            'windows': window_statuses
        }

    def get_portfolio_summary(self, current_time: Optional[datetime] = None) -> Dict:
        """
        Get summary of entire portfolio status.

        Args:
            current_time: Current time (uses current if None)

        Returns:
            Dictionary with portfolio summary
        """
        if current_time is None:
            current_time = self.tz_manager.get_current_trading_timezone()

        tradeable_instruments = self.get_tradeable_instruments(current_time)
        close_actions = self.check_window_closures(current_time)

        return {
            'current_time': self.tz_manager.format_time_with_timezone(current_time),
            'total_instruments': len(self.instruments),
            'enabled_instruments': sum(1 for i in self.instruments.values() if i.enabled),
            'tradeable_instruments': len(tradeable_instruments),
            'tradeable_symbols': [i.symbol for i in tradeable_instruments],
            'pending_close_actions': len(close_actions),
            'close_actions': [
                {
                    'symbol': action.symbol,
                    'reason': action.reason,
                    'close_negatives_only': action.close_negatives_only
                }
                for action in close_actions
            ]
        }

    def __str__(self) -> str:
        return f"PortfolioManager: {len(self.instruments)} instruments"


# Example usage
if __name__ == "__main__":
    portfolio = PortfolioManager()

    print("=== Portfolio Manager Test ===\n")
    print(portfolio)

    # Get portfolio summary
    summary = portfolio.get_portfolio_summary()
    print(f"\nPortfolio Summary:")
    print(f"  Current Time: {summary['current_time']}")
    print(f"  Total Instruments: {summary['total_instruments']}")
    print(f"  Enabled: {summary['enabled_instruments']}")
    print(f"  Tradeable Now: {summary['tradeable_instruments']}")
    print(f"  Tradeable Symbols: {summary['tradeable_symbols']}")

    # Check individual instruments
    print(f"\nInstrument Details:")
    for symbol in ['EURUSD', 'GBPUSD']:
        status = portfolio.get_instrument_status(symbol)
        print(f"\n{status['name']} ({symbol}):")
        print(f"  Tradeable: {status['tradeable_now']}")
        print(f"  Active Windows: {status['active_windows']}")
        print(f"  Should Close All Positions: {status['should_close_all_positions']}")
        print(f"  Min Confluence Score: {status['min_confluence_score']}")

        for window in status['windows']:
            print(f"  - {window['name']}: {window['status']} ({window['start']}-{window['end']})")
