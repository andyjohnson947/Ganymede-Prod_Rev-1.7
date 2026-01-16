"""
Higher Time Frame (HTF) Level Detection
Critical for confluence - Top 5 factors are all HTF:
1. Prev Day VAH - 364 occurrences
2. Weekly HVN - 328 occurrences
3. Prev Day POC - 325 occurrences
4. Prev Week Swing Low - 325 occurrences
5. Daily HVN - 310 occurrences
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional

from indicators.volume_profile import VolumeProfile
from indicators.vwap import VWAP
from config.strategy_config import HTF_TIMEFRAMES


class HTFLevels:
    """Detect and track higher timeframe institutional levels"""

    def __init__(self):
        """Initialize HTF level detector"""
        self.volume_profile = VolumeProfile()
        self.vwap = VWAP()

    def calculate_daily_levels(self, daily_data: pd.DataFrame) -> Dict:
        """
        Calculate daily institutional levels

        Args:
            daily_data: DataFrame with D1 OHLCV data

        Returns:
            Dict with daily levels
        """
        if len(daily_data) < 2:
            return self._empty_daily_levels()

        # Get previous day's data
        prev_day = daily_data.iloc[-2]

        # Calculate volume profile for previous day
        prev_day_data = daily_data.iloc[-2:-1]  # Single day
        profile = self.volume_profile.calculate(prev_day_data)

        # Calculate VWAP for previous day
        prev_day_df = pd.DataFrame([prev_day])
        vwap_data = self.vwap.calculate(prev_day_df)

        # Get current day's profile
        current_profile = self.volume_profile.calculate(daily_data.tail(1))

        # Calculate swing levels from recent days
        swing_levels = self.volume_profile.calculate_swing_levels(daily_data.tail(20))

        return {
            # Previous day levels
            'prev_day_high': prev_day['high'],
            'prev_day_low': prev_day['low'],
            'prev_day_close': prev_day['close'],
            'prev_day_poc': profile['poc'],
            'prev_day_vah': profile['vah'],
            'prev_day_val': profile['val'],
            'prev_day_vwap': vwap_data.iloc[-1]['vwap'] if 'vwap' in vwap_data.columns else 0,

            # Current day levels
            'daily_poc': current_profile['poc'],
            'daily_vah': current_profile['vah'],
            'daily_val': current_profile['val'],
            'daily_hvn': current_profile['hvn_levels'],
            'daily_lvn': current_profile['lvn_levels'],

            # Swing levels
            'daily_swing_highs': swing_levels['swing_highs'],
            'daily_swing_lows': swing_levels['swing_lows'],
        }

    def calculate_weekly_levels(self, weekly_data: pd.DataFrame) -> Dict:
        """
        Calculate weekly institutional levels

        Args:
            weekly_data: DataFrame with W1 OHLCV data

        Returns:
            Dict with weekly levels
        """
        if len(weekly_data) < 2:
            return self._empty_weekly_levels()

        # Get previous week's data
        prev_week = weekly_data.iloc[-2]

        # Calculate volume profile for previous week
        prev_week_data = weekly_data.iloc[-2:-1]
        profile = self.volume_profile.calculate(prev_week_data)

        # Calculate VWAP for previous week
        prev_week_df = pd.DataFrame([prev_week])
        vwap_data = self.vwap.calculate(prev_week_df)

        # Get current week's profile
        current_profile = self.volume_profile.calculate(weekly_data.tail(1))

        # Calculate swing levels from recent weeks
        swing_levels = self.volume_profile.calculate_swing_levels(weekly_data.tail(20))

        # Calculate week midpoint
        week_midpoint = (prev_week['high'] + prev_week['low']) / 2

        return {
            # Previous week levels
            'prev_week_high': prev_week['high'],
            'prev_week_low': prev_week['low'],
            'prev_week_close': prev_week['close'],
            'prev_week_poc': profile['poc'],
            'prev_week_vah': profile['vah'],
            'prev_week_val': profile['val'],
            'prev_week_vwap': vwap_data.iloc[-1]['vwap'] if 'vwap' in vwap_data.columns else 0,
            'prev_week_midpoint': week_midpoint,

            # Current week levels
            'weekly_poc': current_profile['poc'],
            'weekly_vah': current_profile['vah'],
            'weekly_val': current_profile['val'],
            'weekly_hvn': current_profile['hvn_levels'],
            'weekly_lvn': current_profile['lvn_levels'],

            # Swing levels
            'weekly_swing_highs': swing_levels['swing_highs'],
            'weekly_swing_lows': swing_levels['swing_lows'],
        }

    def calculate_monthly_levels(self, monthly_data: pd.DataFrame) -> Dict:
        """
        Calculate monthly institutional levels

        Args:
            monthly_data: DataFrame with MN1 OHLCV data

        Returns:
            Dict with monthly levels
        """
        if len(monthly_data) < 2:
            return self._empty_monthly_levels()

        # Get previous month's data
        prev_month = monthly_data.iloc[-2]

        # Calculate volume profile for previous month
        prev_month_data = monthly_data.iloc[-2:-1]
        profile = self.volume_profile.calculate(prev_month_data)

        return {
            'prev_month_high': prev_month['high'],
            'prev_month_low': prev_month['low'],
            'prev_month_close': prev_month['close'],
            'prev_month_poc': profile['poc'],
            'prev_month_vah': profile['vah'],
            'prev_month_val': profile['val'],
        }

    def get_all_levels(
        self,
        daily_data: pd.DataFrame,
        weekly_data: pd.DataFrame,
        monthly_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Get all HTF levels at once

        Args:
            daily_data: D1 data
            weekly_data: W1 data
            monthly_data: Optional MN1 data

        Returns:
            Dict with all HTF levels
        """
        levels = {
            'daily': self.calculate_daily_levels(daily_data),
            'weekly': self.calculate_weekly_levels(weekly_data),
        }

        if monthly_data is not None and len(monthly_data) >= 2:
            levels['monthly'] = self.calculate_monthly_levels(monthly_data)
        else:
            levels['monthly'] = self._empty_monthly_levels()

        return levels

    def check_confluence(self, price: float, all_levels: Dict, tolerance_pct: float = 0.003) -> Dict:
        """
        Check price against all HTF levels for confluence

        Args:
            price: Current price
            all_levels: Dict from get_all_levels()
            tolerance_pct: Price tolerance (default 0.3%)

        Returns:
            Dict with confluence factors and score
        """
        daily = all_levels.get('daily', {})
        weekly = all_levels.get('weekly', {})
        monthly = all_levels.get('monthly', {})

        factors = []
        score = 0

        tolerance = abs(price * tolerance_pct)

        # Previous Day levels (high importance)
        # VAH and VAL are mutually exclusive (resistance vs support)
        if self._at_level(price, daily.get('prev_day_vah', 0), tolerance):
            factors.append('Prev Day VAH')
            score += 2
        elif self._at_level(price, daily.get('prev_day_val', 0), tolerance):
            factors.append('Prev Day VAL')
            score += 2

        if self._at_level(price, daily.get('prev_day_poc', 0), tolerance):
            factors.append('Prev Day POC')
            score += 2

        # High and Low are mutually exclusive (resistance vs support)
        if self._at_level(price, daily.get('prev_day_high', 0), tolerance):
            factors.append('Prev Day High')
            score += 2
        elif self._at_level(price, daily.get('prev_day_low', 0), tolerance):
            factors.append('Prev Day Low')
            score += 2

        # Daily HVN levels
        for hvn in daily.get('daily_hvn', []):
            if self._at_level(price, hvn, tolerance):
                factors.append('Daily HVN')
                score += 2
                break

        # Daily swing levels (mutually exclusive - check low only if high not hit)
        daily_swing_high_hit = False
        for swing_high in daily.get('daily_swing_highs', []):
            if self._at_level(price, swing_high, tolerance):
                factors.append('Daily Swing High')
                score += 1
                daily_swing_high_hit = True
                break

        if not daily_swing_high_hit:
            for swing_low in daily.get('daily_swing_lows', []):
                if self._at_level(price, swing_low, tolerance):
                    factors.append('Daily Swing Low')
                    score += 1
                    break

        # Weekly levels (highest importance)
        if self._at_level(price, weekly.get('weekly_poc', 0), tolerance):
            factors.append('Weekly POC')
            score += 3

        # Weekly HVN levels (very important)
        for hvn in weekly.get('weekly_hvn', []):
            if self._at_level(price, hvn, tolerance):
                factors.append('Weekly HVN')
                score += 3
                break

        # Previous Week levels (mutually exclusive - high vs low)
        if self._at_level(price, weekly.get('prev_week_high', 0), tolerance):
            factors.append('Prev Week High')
            score += 2
        elif self._at_level(price, weekly.get('prev_week_low', 0), tolerance):
            factors.append('Prev Week Low')
            score += 2

        # Weekly swing levels (mutually exclusive - check low only if high not hit)
        weekly_swing_high_hit = False
        for swing_high in weekly.get('weekly_swing_highs', []):
            if self._at_level(price, swing_high, tolerance):
                factors.append('Prev Week Swing High')
                score += 2
                weekly_swing_high_hit = True
                break

        if not weekly_swing_high_hit:
            for swing_low in weekly.get('weekly_swing_lows', []):
                if self._at_level(price, swing_low, tolerance):
                    factors.append('Prev Week Swing Low')
                    score += 2
                    break

        if self._at_level(price, weekly.get('prev_week_vwap', 0), tolerance):
            factors.append('Prev Week VWAP')
            score += 2

        # Monthly levels (if available)
        if self._at_level(price, monthly.get('prev_month_poc', 0), tolerance):
            factors.append('Prev Month POC')
            score += 2

        return {
            'factors': factors,
            'score': score,
            'levels': {
                'daily': daily,
                'weekly': weekly,
                'monthly': monthly
            }
        }

    def _at_level(self, price: float, level: float, tolerance: float) -> bool:
        """Check if price is at level within tolerance"""
        if level == 0:
            return False
        return abs(price - level) <= tolerance

    def _empty_daily_levels(self) -> Dict:
        """Return empty daily levels structure"""
        return {
            'prev_day_high': 0,
            'prev_day_low': 0,
            'prev_day_close': 0,
            'prev_day_poc': 0,
            'prev_day_vah': 0,
            'prev_day_val': 0,
            'prev_day_vwap': 0,
            'daily_poc': 0,
            'daily_vah': 0,
            'daily_val': 0,
            'daily_hvn': [],
            'daily_lvn': [],
            'daily_swing_highs': [],
            'daily_swing_lows': [],
        }

    def _empty_weekly_levels(self) -> Dict:
        """Return empty weekly levels structure"""
        return {
            'prev_week_high': 0,
            'prev_week_low': 0,
            'prev_week_close': 0,
            'prev_week_poc': 0,
            'prev_week_vah': 0,
            'prev_week_val': 0,
            'prev_week_vwap': 0,
            'prev_week_midpoint': 0,
            'weekly_poc': 0,
            'weekly_vah': 0,
            'weekly_val': 0,
            'weekly_hvn': [],
            'weekly_lvn': [],
            'weekly_swing_highs': [],
            'weekly_swing_lows': [],
        }

    def _empty_monthly_levels(self) -> Dict:
        """Return empty monthly levels structure"""
        return {
            'prev_month_high': 0,
            'prev_month_low': 0,
            'prev_month_close': 0,
            'prev_month_poc': 0,
            'prev_month_vah': 0,
            'prev_month_val': 0,
        }
