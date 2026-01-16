"""
Timezone Management Module for Trading Robot

Provides centralized timezone handling with GMT/GMT+1 and DST support.
Always uses GMT (or GMT+1 during BST) as the working timezone and handles
conversion between broker time and working time.
"""

import pytz
from datetime import datetime, time, timedelta
from typing import Tuple, Optional


class TimezoneManager:
    """
    Manages timezone operations for the trading robot.

    The robot always works in GMT/GMT+1 timezone (British Summer Time aware).
    This class handles:
    - GMT/GMT+1 current time with automatic DST handling
    - Broker time to GMT conversion
    - Trading window time calculations
    - DST transition detection
    """

    # UK timezone automatically handles GMT <-> BST (GMT+1) transitions
    UK_TIMEZONE = pytz.timezone("Europe/London")
    GMT_TIMEZONE = pytz.UTC

    def __init__(self, broker_timezone: str = "UTC"):
        """
        Initialize timezone manager.

        Args:
            broker_timezone: Timezone identifier for broker server (e.g., "UTC", "EET", "America/New_York")
        """
        self.broker_timezone = pytz.timezone(broker_timezone)
        self._timezone_offset_hours = None

    def get_gmt_time(self) -> datetime:
        """
        Get current time in pure GMT (UTC).

        Returns:
            Current time in GMT timezone
        """
        return datetime.now(pytz.UTC)

    def get_gmt_plus_one(self) -> datetime:
        """
        Get current time in GMT+1 (with DST handling).

        Returns:
            Current time in UK timezone (GMT in winter, BST/GMT+1 in summer)
        """
        return datetime.now(self.UK_TIMEZONE)

    def get_current_trading_timezone(self) -> datetime:
        """
        Get current time in the robot's working timezone (GMT/GMT+1 with DST).

        This is the primary method to get current time for all trading operations.

        Returns:
            Current time in UK timezone with automatic DST handling
        """
        return datetime.now(self.UK_TIMEZONE)

    def is_dst_active(self, dt: Optional[datetime] = None) -> bool:
        """
        Check if Daylight Saving Time (BST) is currently active.

        Args:
            dt: DateTime to check. If None, uses current time.

        Returns:
            True if DST is active (BST period), False otherwise (GMT period)
        """
        if dt is None:
            dt = datetime.now(self.UK_TIMEZONE)
        elif dt.tzinfo is None:
            dt = self.UK_TIMEZONE.localize(dt)
        else:
            dt = dt.astimezone(self.UK_TIMEZONE)

        # Check if UTC offset is +01:00 (BST) or +00:00 (GMT)
        utc_offset = dt.utcoffset()
        return utc_offset.total_seconds() / 3600 == 1.0

    def get_timezone_name(self) -> str:
        """
        Get current timezone name (GMT or BST).

        Returns:
            "GMT" or "BST" depending on current DST status
        """
        return "BST" if self.is_dst_active() else "GMT"

    def convert_broker_to_gmt(self, broker_time: datetime) -> datetime:
        """
        Convert broker server time to GMT.

        Args:
            broker_time: DateTime in broker's timezone

        Returns:
            DateTime converted to GMT
        """
        if broker_time.tzinfo is None:
            # Assume broker_time is in broker's timezone if naive
            broker_time = self.broker_timezone.localize(broker_time)

        return broker_time.astimezone(pytz.UTC)

    def convert_broker_to_trading_tz(self, broker_time: datetime) -> datetime:
        """
        Convert broker server time to trading timezone (GMT/GMT+1).

        Args:
            broker_time: DateTime in broker's timezone

        Returns:
            DateTime converted to UK timezone
        """
        if broker_time.tzinfo is None:
            broker_time = self.broker_timezone.localize(broker_time)

        return broker_time.astimezone(self.UK_TIMEZONE)

    def convert_gmt_to_broker(self, gmt_time: datetime) -> datetime:
        """
        Convert GMT time to broker's timezone.

        Args:
            gmt_time: DateTime in GMT

        Returns:
            DateTime converted to broker's timezone
        """
        if gmt_time.tzinfo is None:
            gmt_time = pytz.UTC.localize(gmt_time)

        return gmt_time.astimezone(self.broker_timezone)

    def calculate_timezone_offset(self) -> Tuple[int, int]:
        """
        Calculate the difference between broker time and trading time.

        Returns:
            Tuple of (hours, minutes) offset from trading timezone to broker timezone
        """
        now_trading = self.get_current_trading_timezone()
        now_gmt = self.get_gmt_time()

        # Convert to broker timezone
        now_broker = now_gmt.astimezone(self.broker_timezone)

        # Calculate offset
        offset = now_broker.utcoffset() - now_trading.utcoffset()
        offset_seconds = int(offset.total_seconds())

        hours = offset_seconds // 3600
        minutes = (offset_seconds % 3600) // 60

        return hours, minutes

    def is_time_in_window(
        self,
        current_time: datetime,
        window_start: time,
        window_end: time
    ) -> bool:
        """
        Check if current time falls within a trading window.

        Args:
            current_time: Current time (timezone-aware)
            window_start: Window start time (assumes GMT)
            window_end: Window end time (assumes GMT)

        Returns:
            True if current time is within window
        """
        # Convert current time to trading timezone if needed
        if current_time.tzinfo is None:
            current_time = self.UK_TIMEZONE.localize(current_time)
        else:
            current_time = current_time.astimezone(self.UK_TIMEZONE)

        current_time_only = current_time.time()

        # Handle windows that cross midnight
        if window_end < window_start:
            return current_time_only >= window_start or current_time_only <= window_end
        else:
            return window_start <= current_time_only <= window_end

    def time_until_window_end(
        self,
        current_time: datetime,
        window_end: time
    ) -> Optional[timedelta]:
        """
        Calculate time remaining until window ends.

        Args:
            current_time: Current time (timezone-aware)
            window_end: Window end time (assumes GMT)

        Returns:
            Timedelta until window end, or None if already past
        """
        if current_time.tzinfo is None:
            current_time = self.UK_TIMEZONE.localize(current_time)
        else:
            current_time = current_time.astimezone(self.UK_TIMEZONE)

        # Create datetime for window end today
        today = current_time.date()
        window_end_dt = datetime.combine(today, window_end)
        window_end_dt = self.UK_TIMEZONE.localize(window_end_dt)

        # If window end is in the past, it might be tomorrow
        if window_end_dt < current_time:
            window_end_dt += timedelta(days=1)

        return window_end_dt - current_time

    def format_time_with_timezone(self, dt: datetime) -> str:
        """
        Format datetime with timezone information.

        Args:
            dt: DateTime to format

        Returns:
            Formatted string with timezone
        """
        if dt.tzinfo is None:
            dt = self.UK_TIMEZONE.localize(dt)
        else:
            dt = dt.astimezone(self.UK_TIMEZONE)

        tz_name = self.get_timezone_name()
        return dt.strftime(f"%Y-%m-%d %H:%M:%S {tz_name}")

    def get_utc_offset_hours(self) -> float:
        """
        Get current UTC offset in hours for the trading timezone.

        Returns:
            UTC offset in hours (0.0 for GMT, 1.0 for BST)
        """
        now = self.get_current_trading_timezone()
        offset_seconds = now.utcoffset().total_seconds()
        return offset_seconds / 3600


# Global instance for easy access
_default_timezone_manager = None


def get_timezone_manager(broker_timezone: str = "UTC") -> TimezoneManager:
    """
    Get or create the default timezone manager instance.

    Args:
        broker_timezone: Broker's timezone (only used on first call)

    Returns:
        TimezoneManager instance
    """
    global _default_timezone_manager

    if _default_timezone_manager is None:
        _default_timezone_manager = TimezoneManager(broker_timezone)

    return _default_timezone_manager


def get_current_time() -> datetime:
    """
    Get current trading time (GMT/GMT+1 with DST).

    Convenience function for getting current time in trading timezone.

    Returns:
        Current time in trading timezone
    """
    return get_timezone_manager().get_current_trading_timezone()


def format_trading_time(dt: datetime) -> str:
    """
    Format datetime for display with trading timezone.

    Args:
        dt: DateTime to format

    Returns:
        Formatted string
    """
    return get_timezone_manager().format_time_with_timezone(dt)


# Example usage and testing
if __name__ == "__main__":
    tz_mgr = TimezoneManager("UTC")

    print("=== Timezone Manager Test ===")
    print(f"Current GMT time: {tz_mgr.get_gmt_time()}")
    print(f"Current trading time: {tz_mgr.get_current_trading_timezone()}")
    print(f"DST active: {tz_mgr.is_dst_active()}")
    print(f"Timezone name: {tz_mgr.get_timezone_name()}")
    print(f"UTC offset: {tz_mgr.get_utc_offset_hours()} hours")

    broker_offset = tz_mgr.calculate_timezone_offset()
    print(f"Broker offset: {broker_offset[0]}h {broker_offset[1]}m")

    # Test trading window
    current = tz_mgr.get_current_trading_timezone()
    window_start = time(7, 0)
    window_end = time(9, 0)
    in_window = tz_mgr.is_time_in_window(current, window_start, window_end)
    print(f"\nIn trading window 07:00-09:00: {in_window}")

    if not in_window:
        time_until = tz_mgr.time_until_window_end(current, window_end)
        if time_until:
            print(f"Time until window end: {time_until}")
