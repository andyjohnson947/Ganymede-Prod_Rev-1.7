"""
Trading Calendar Module

Manages trading restrictions including bank holidays, weekends,
and Friday afternoon trading blackout periods.
"""

from datetime import datetime, date, time
from typing import Tuple, List, Optional, Set
from enum import Enum

from .timezone_manager import get_timezone_manager


class TradingRestriction(Enum):
    """Types of trading restrictions"""
    NONE = "none"
    WEEKEND = "weekend"
    BANK_HOLIDAY = "bank_holiday"
    FRIDAY_AFTERNOON = "friday_afternoon"
    OUTSIDE_HOURS = "outside_hours"


class TradingCalendar:
    """
    Manages trading calendar and restrictions.

    Handles:
    - Bank holidays (UK, US, EU)
    - Weekend trading restrictions
    - Friday afternoon blackout (after 12:00 PM GMT)
    - Trading hours enforcement
    """

    # UK Bank Holidays 2025-2026
    UK_BANK_HOLIDAYS_2025 = {
        date(2025, 1, 1),   # New Year's Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 4, 21),  # Easter Monday
        date(2025, 5, 5),   # Early May Bank Holiday
        date(2025, 5, 26),  # Spring Bank Holiday
        date(2025, 8, 25),  # Summer Bank Holiday
        date(2025, 12, 25), # Christmas Day
        date(2025, 12, 26), # Boxing Day
    }

    UK_BANK_HOLIDAYS_2026 = {
        date(2026, 1, 1),   # New Year's Day
        date(2026, 4, 3),   # Good Friday
        date(2026, 4, 6),   # Easter Monday
        date(2026, 5, 4),   # Early May Bank Holiday
        date(2026, 5, 25),  # Spring Bank Holiday
        date(2026, 8, 31),  # Summer Bank Holiday
        date(2026, 12, 25), # Christmas Day
        date(2026, 12, 28), # Boxing Day (substitute)
    }

    # US Bank Holidays 2025-2026
    US_BANK_HOLIDAYS_2025 = {
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 5, 26),  # Memorial Day
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving
        date(2025, 12, 25), # Christmas Day
    }

    US_BANK_HOLIDAYS_2026 = {
        date(2026, 1, 1),   # New Year's Day
        date(2026, 1, 19),  # Martin Luther King Jr. Day
        date(2026, 2, 16),  # Presidents' Day
        date(2026, 5, 25),  # Memorial Day
        date(2026, 7, 3),   # Independence Day (observed)
        date(2026, 9, 7),   # Labor Day
        date(2026, 11, 26), # Thanksgiving
        date(2026, 12, 25), # Christmas Day
    }

    # Friday afternoon blackout time
    FRIDAY_BLACKOUT_TIME = time(12, 0)  # 12:00 PM GMT

    def __init__(
        self,
        check_uk_holidays: bool = True,
        check_us_holidays: bool = True,
        enable_friday_restriction: bool = True,
        enable_weekend_restriction: bool = True
    ):
        """
        Initialize trading calendar.

        Args:
            check_uk_holidays: Check UK bank holidays
            check_us_holidays: Check US bank holidays
            enable_friday_restriction: Enable Friday afternoon blackout
            enable_weekend_restriction: Enable weekend trading restriction
        """
        self.check_uk_holidays = check_uk_holidays
        self.check_us_holidays = check_us_holidays
        self.enable_friday_restriction = enable_friday_restriction
        self.enable_weekend_restriction = enable_weekend_restriction

        self.tz_manager = get_timezone_manager()

        # Combine all holidays
        self.bank_holidays: Set[date] = set()
        if self.check_uk_holidays:
            self.bank_holidays.update(self.UK_BANK_HOLIDAYS_2025)
            self.bank_holidays.update(self.UK_BANK_HOLIDAYS_2026)
        if self.check_us_holidays:
            self.bank_holidays.update(self.US_BANK_HOLIDAYS_2025)
            self.bank_holidays.update(self.US_BANK_HOLIDAYS_2026)

    def is_weekend(self, dt: datetime) -> bool:
        """
        Check if date falls on a weekend.

        Args:
            dt: DateTime to check

        Returns:
            True if Saturday or Sunday
        """
        # 5 = Saturday, 6 = Sunday
        return dt.weekday() in [5, 6]

    def is_bank_holiday(self, dt: datetime) -> bool:
        """
        Check if date is a bank holiday.

        Args:
            dt: DateTime to check

        Returns:
            True if bank holiday
        """
        check_date = dt.date() if isinstance(dt, datetime) else dt
        return check_date in self.bank_holidays

    def is_friday(self, dt: datetime) -> bool:
        """Check if date is a Friday."""
        return dt.weekday() == 4  # 4 = Friday

    def is_friday_afternoon(self, dt: datetime) -> bool:
        """
        Check if current time is Friday afternoon (after blackout time).

        Args:
            dt: DateTime to check

        Returns:
            True if Friday after blackout time
        """
        if not self.is_friday(dt):
            return False

        # Convert to trading timezone
        if dt.tzinfo is None:
            dt = self.tz_manager.UK_TIMEZONE.localize(dt)
        else:
            dt = dt.astimezone(self.tz_manager.UK_TIMEZONE)

        current_time = dt.time()
        return current_time >= self.FRIDAY_BLACKOUT_TIME

    def get_restriction(self, dt: Optional[datetime] = None) -> TradingRestriction:
        """
        Get the current trading restriction type.

        Args:
            dt: DateTime to check (uses current if None)

        Returns:
            TradingRestriction enum value
        """
        if dt is None:
            dt = self.tz_manager.get_current_trading_timezone()

        # Check in priority order
        if self.enable_weekend_restriction and self.is_weekend(dt):
            return TradingRestriction.WEEKEND

        if self.is_bank_holiday(dt):
            return TradingRestriction.BANK_HOLIDAY

        if self.enable_friday_restriction and self.is_friday_afternoon(dt):
            return TradingRestriction.FRIDAY_AFTERNOON

        return TradingRestriction.NONE

    def is_trading_allowed(self, dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if trading is allowed at the given time.

        Args:
            dt: DateTime to check (uses current if None)

        Returns:
            Tuple of (is_allowed, reason)
        """
        restriction = self.get_restriction(dt)

        if restriction == TradingRestriction.NONE:
            return True, "Trading allowed"

        reason_map = {
            TradingRestriction.WEEKEND: "Trading not allowed on weekends",
            TradingRestriction.BANK_HOLIDAY: "Trading not allowed on bank holidays",
            TradingRestriction.FRIDAY_AFTERNOON: "Trading not allowed Friday afternoons after 12:00 PM GMT",
            TradingRestriction.OUTSIDE_HOURS: "Outside trading hours"
        }

        return False, reason_map.get(restriction, "Trading restricted")

    def get_next_trading_day(self, dt: Optional[datetime] = None) -> datetime:
        """
        Get the next valid trading day.

        Args:
            dt: Starting datetime (uses current if None)

        Returns:
            Next valid trading datetime
        """
        if dt is None:
            dt = self.tz_manager.get_current_trading_timezone()

        # Start from next day
        next_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        next_day = next_day + timedelta(days=1)

        # Find next valid day
        max_iterations = 30  # Safety limit
        for _ in range(max_iterations):
            if not self.is_weekend(next_day) and not self.is_bank_holiday(next_day):
                return next_day
            next_day += timedelta(days=1)

        return next_day

    def get_upcoming_holidays(self, days_ahead: int = 30) -> List[Tuple[date, str]]:
        """
        Get upcoming bank holidays.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of (date, description) tuples
        """
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)

        upcoming = []
        for holiday_date in sorted(self.bank_holidays):
            if today <= holiday_date <= end_date:
                # Try to identify the holiday name
                holiday_name = self._get_holiday_name(holiday_date)
                upcoming.append((holiday_date, holiday_name))

        return upcoming

    def _get_holiday_name(self, holiday_date: date) -> str:
        """Get the name of a holiday (basic implementation)."""
        # Simple name mapping for major holidays
        month_day = (holiday_date.month, holiday_date.day)

        holiday_names = {
            (1, 1): "New Year's Day",
            (7, 4): "Independence Day",
            (12, 25): "Christmas Day",
            (12, 26): "Boxing Day",
            (5, 26): "Memorial/Spring Bank Holiday",
        }

        return holiday_names.get(month_day, "Bank Holiday")

    def is_near_holiday(self, dt: Optional[datetime] = None, days_before: int = 1) -> bool:
        """
        Check if date is near a bank holiday.

        Args:
            dt: DateTime to check (uses current if None)
            days_before: Number of days before holiday to consider "near"

        Returns:
            True if near a holiday
        """
        if dt is None:
            dt = self.tz_manager.get_current_trading_timezone()

        check_date = dt.date()

        for i in range(1, days_before + 1):
            future_date = check_date + timedelta(days=i)
            if future_date in self.bank_holidays:
                return True

        return False

    def get_trading_status(self, dt: Optional[datetime] = None) -> dict:
        """
        Get comprehensive trading status information.

        Args:
            dt: DateTime to check (uses current if None)

        Returns:
            Dictionary with status information
        """
        if dt is None:
            dt = self.tz_manager.get_current_trading_timezone()

        is_allowed, reason = self.is_trading_allowed(dt)
        restriction = self.get_restriction(dt)
        upcoming_holidays = self.get_upcoming_holidays(7)

        return {
            'current_time': self.tz_manager.format_time_with_timezone(dt),
            'trading_allowed': is_allowed,
            'restriction': restriction.value,
            'reason': reason,
            'is_weekend': self.is_weekend(dt),
            'is_bank_holiday': self.is_bank_holiday(dt),
            'is_friday_afternoon': self.is_friday_afternoon(dt),
            'is_near_holiday': self.is_near_holiday(dt),
            'upcoming_holidays': [
                {'date': str(hol_date), 'name': name}
                for hol_date, name in upcoming_holidays
            ],
            'next_trading_day': str(self.get_next_trading_day(dt).date())
        }

    def __str__(self) -> str:
        restrictions = []
        if self.check_uk_holidays:
            restrictions.append("UK holidays")
        if self.check_us_holidays:
            restrictions.append("US holidays")
        if self.enable_friday_restriction:
            restrictions.append("Friday afternoon")
        if self.enable_weekend_restriction:
            restrictions.append("weekends")

        return f"TradingCalendar(checking: {', '.join(restrictions)})"


# Global instance
_default_calendar = None


def get_trading_calendar() -> TradingCalendar:
    """
    Get or create the default trading calendar instance.

    Returns:
        TradingCalendar instance
    """
    global _default_calendar

    if _default_calendar is None:
        _default_calendar = TradingCalendar()

    return _default_calendar


def is_trading_allowed(dt: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Convenience function to check if trading is allowed.

    Args:
        dt: DateTime to check (uses current if None)

    Returns:
        Tuple of (is_allowed, reason)
    """
    return get_trading_calendar().is_trading_allowed(dt)


# Import needed for timedelta
from datetime import timedelta


# Example usage
if __name__ == "__main__":
    calendar = TradingCalendar()

    print("=== Trading Calendar Test ===\n")
    print(calendar)

    # Get current status
    status = calendar.get_trading_status()
    print(f"\nCurrent Trading Status:")
    print(f"  Time: {status['current_time']}")
    print(f"  Trading Allowed: {status['trading_allowed']}")
    print(f"  Restriction: {status['restriction']}")
    print(f"  Reason: {status['reason']}")
    print(f"  Is Weekend: {status['is_weekend']}")
    print(f"  Is Bank Holiday: {status['is_bank_holiday']}")
    print(f"  Is Friday Afternoon: {status['is_friday_afternoon']}")
    print(f"  Near Holiday: {status['is_near_holiday']}")

    if status['upcoming_holidays']:
        print(f"\nUpcoming Holidays:")
        for holiday in status['upcoming_holidays']:
            print(f"  - {holiday['date']}: {holiday['name']}")

    # Test specific dates
    print(f"\nTest Specific Dates:")
    test_dates = [
        datetime(2025, 12, 25, 10, 0),  # Christmas
        datetime(2025, 12, 21, 13, 0),  # Friday afternoon
        datetime(2025, 12, 20, 13, 0),  # Thursday afternoon
    ]

    for test_dt in test_dates:
        allowed, reason = calendar.is_trading_allowed(test_dt)
        print(f"  {test_dt}: {allowed} - {reason}")
