"""
Breakout Strategy Module
Implements range breakout, LVN breakout, and weekly level breakout strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from config import strategy_config as cfg
from utils.logger import logger


class BreakoutStrategy:
    """
    Breakout trading strategy implementation
    Enters when price breaks key levels with volume confirmation
    """

    def __init__(self):
        self.lookback = cfg.BREAKOUT_LOOKBACK
        self.volume_multiplier = cfg.BREAKOUT_VOLUME_MULTIPLIER
        self.atr_multiplier = cfg.BREAKOUT_ATR_MULTIPLIER
        self.min_range_pips = cfg.BREAKOUT_MIN_RANGE_PIPS
        self.rsi_buy_threshold = cfg.BREAKOUT_RSI_BUY_THRESHOLD
        self.rsi_sell_threshold = cfg.BREAKOUT_RSI_SELL_THRESHOLD

    def is_breakout_window(self, current_time: datetime) -> bool:
        """
        Check if current time is within breakout trading window
        """
        if not cfg.ENABLE_TIME_FILTERS:
            return True

        hour = current_time.hour
        day = current_time.weekday()

        # Check if in breakout hours and days
        in_hours = hour in cfg.BREAKOUT_HOURS
        in_days = day in cfg.BREAKOUT_DAYS

        return in_hours and in_days

    def detect_range_breakout(
        self,
        data: pd.DataFrame,
        current_price: float,
        current_volume: float,
        atr: float
    ) -> Optional[Dict]:
        """
        Detect range breakout (consolidation followed by break)

        Args:
            data: Historical price data with OHLCV
            current_price: Current market price
            current_volume: Current bar volume
            atr: Average True Range

        Returns:
            Signal dict if breakout detected, None otherwise
        """
        if len(data) < self.lookback:
            return None

        # Calculate range boundaries
        recent_data = data.tail(self.lookback)
        range_high = recent_data['high'].max()
        range_low = recent_data['low'].min()
        range_size_pips = (range_high - range_low) / 0.0001  # Convert to pips

        # Check minimum range size
        if range_size_pips < self.min_range_pips:
            logger.debug(f"Range breakout: Range too small ({range_size_pips:.1f} pips < {self.min_range_pips})")
            return None

        # Calculate average volume - handle missing volume column
        if 'volume' in recent_data.columns:
            avg_volume = recent_data['volume'].mean()
        elif 'tick_volume' in recent_data.columns:
            avg_volume = recent_data['tick_volume'].mean()
        else:
            avg_volume = current_volume  # Fallback to current volume
        volume_spike = current_volume > (avg_volume * self.volume_multiplier)

        # Check for volatility compression (ATR decreasing)
        recent_atr = atr
        avg_atr = data['atr'].tail(20).mean() if 'atr' in data.columns else recent_atr
        is_compressed = recent_atr < avg_atr

        # Detect breakout
        bullish_breakout = current_price > range_high
        bearish_breakout = current_price < range_low

        logger.debug(f"Range breakout check: Range={range_size_pips:.1f}pips, Vol spike={volume_spike}, Bull={bullish_breakout}, Bear={bearish_breakout}")

        if (bullish_breakout or bearish_breakout) and volume_spike:
            logger.info(f" RANGE BREAKOUT DETECTED: {'BULLISH' if bullish_breakout else 'BEARISH'} - Range {range_size_pips:.1f} pips")
            # BREAKOUT/MOMENTUM LOGIC: Buy when price breaks ABOVE range (ride momentum UP)
            #                          Sell when price breaks BELOW range (ride momentum DOWN)
            direction = 'buy' if bullish_breakout else 'sell'  # Lowercase for consistency

            # Calculate targets and stops
            if direction == 'buy':
                target = current_price + (range_high - range_low)  # Range projection
                stop = range_high - (range_high - range_low) * cfg.BREAKOUT_STOP_PERCENT
            else:
                target = current_price - (range_high - range_low)
                stop = range_low + (range_high - range_low) * cfg.BREAKOUT_STOP_PERCENT

            return {
                'type': 'range_breakout',
                'direction': direction,  # 'buy' or 'sell' (lowercase)
                'entry_price': current_price,
                'target': target,
                'stop': stop,
                'range_high': range_high,
                'range_low': range_low,
                'range_size_pips': range_size_pips,
                'volume_spike': volume_spike,
                'compressed': is_compressed,
                'confidence': 'high' if volume_spike and is_compressed else 'medium',
                'breakout_type': 'bullish' if bullish_breakout else 'bearish'
            }

        return None

    def detect_lvn_breakout(
        self,
        current_price: float,
        lvn_levels: List[float],
        rsi: float,
        macd: float,
        atr: float,
        direction: str = None
    ) -> Optional[Dict]:
        """
        Detect breakout through Low Volume Node (LVN)
        Price should move quickly through LVN with low resistance

        Args:
            current_price: Current market price
            lvn_levels: List of LVN price levels
            rsi: RSI value
            macd: MACD histogram value
            atr: Average True Range
            direction: Force direction ('BUY' or 'SELL'), or None to detect

        Returns:
            Signal dict if LVN breakout detected, None otherwise
        """
        if not lvn_levels:
            return None

        for lvn in lvn_levels:
            distance = abs(current_price - lvn) / current_price

            # Check if near LVN (within 0.1%)
            if distance < 0.001:
                # Detect momentum direction
                if direction is None:
                    if current_price > lvn and rsi > self.rsi_buy_threshold and macd > 0:
                        direction = 'buy'
                    elif current_price < lvn and rsi < self.rsi_sell_threshold and macd < 0:
                        direction = 'sell'
                    else:
                        continue

                # LVN breakout detected
                if direction == 'buy':
                    target = lvn + (atr * cfg.BREAKOUT_ATR_TARGET_MULTIPLE)
                    stop = current_price - (atr * 0.5)
                else:
                    target = lvn - (atr * cfg.BREAKOUT_ATR_TARGET_MULTIPLE)
                    stop = current_price + (atr * 0.5)

                return {
                    'type': 'lvn_breakout',
                    'direction': direction,
                    'entry_price': current_price,
                    'target': target,
                    'stop': stop,
                    'lvn_level': lvn,
                    'rsi': rsi,
                    'macd': macd,
                    'confidence': 'high'
                }

        return None

    def detect_weekly_level_breakout(
        self,
        current_price: float,
        weekly_high: float,
        weekly_low: float,
        current_volume: float,
        avg_volume: float,
        rsi: float,
        atr: float
    ) -> Optional[Dict]:
        """
        Detect breakout of previous week's high/low
        Monday breakouts and weekly continuation patterns

        Args:
            current_price: Current market price
            weekly_high: Previous week high
            weekly_low: Previous week low
            current_volume: Current bar volume
            avg_volume: Average volume
            rsi: RSI value
            atr: Average True Range

        Returns:
            Signal dict if weekly breakout detected, None otherwise
        """
        # Volume confirmation
        volume_spike = current_volume > (avg_volume * self.volume_multiplier)

        # Bullish breakout above weekly high
        if current_price > weekly_high and rsi > self.rsi_buy_threshold and volume_spike:
            weekly_range = weekly_high - weekly_low
            target = current_price + weekly_range
            stop = weekly_high - (weekly_range * cfg.BREAKOUT_STOP_PERCENT)

            return {
                'type': 'weekly_breakout',
                'direction': 'buy',
                'entry_price': current_price,
                'target': target,
                'stop': stop,
                'weekly_high': weekly_high,
                'weekly_low': weekly_low,
                'rsi': rsi,
                'volume_spike': volume_spike,
                'confidence': 'high'
            }

        # Bearish breakout below weekly low
        if current_price < weekly_low and rsi < self.rsi_sell_threshold and volume_spike:
            weekly_range = weekly_high - weekly_low
            target = current_price - weekly_range
            stop = weekly_low + (weekly_range * cfg.BREAKOUT_STOP_PERCENT)

            return {
                'type': 'weekly_breakout',
                'direction': 'sell',
                'entry_price': current_price,
                'target': target,
                'stop': stop,
                'weekly_high': weekly_high,
                'weekly_low': weekly_low,
                'rsi': rsi,
                'volume_spike': volume_spike,
                'confidence': 'high'
            }

        return None

    def check_breakout_signal(
        self,
        data: pd.DataFrame,
        current_price: float,
        volume_profile: Dict,
        weekly_levels: Dict,
        indicators: Dict,
        current_time: datetime
    ) -> Optional[Dict]:
        """
        Main breakout signal checker - combines all breakout types

        Args:
            data: Historical price data
            current_price: Current market price
            volume_profile: Volume profile with LVN/HVN levels
            weekly_levels: Previous week high/low/VWAP
            indicators: Technical indicators (RSI, MACD, ATR, volume)
            current_time: Current datetime

        Returns:
            Best breakout signal dict, or None
        """
        # Check if in breakout trading window
        if not self.is_breakout_window(current_time):
            logger.debug(f"Breakout: Outside trading window at {current_time.hour}:00")
            return None

        # Extract indicators
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd_histogram', 0)
        atr = indicators.get('atr', 0.0015)
        current_volume = indicators.get('volume', 0)
        avg_volume = data['volume'].tail(20).mean() if 'volume' in data.columns else current_volume

        # Check volatility requirement (breakouts need high volatility)
        median_atr = data['atr'].median() if 'atr' in data.columns else atr
        if atr < median_atr * self.atr_multiplier:
            logger.debug(f"Breakout: ATR too low ({atr:.5f} < {median_atr * self.atr_multiplier:.5f})")
            return None  # Not volatile enough for breakouts

        logger.info(f"Breakout: Checking signals - ATR: {atr:.5f}, RSI: {rsi:.1f}, Volume: {current_volume:.0f}")

        signals = []

        # 1. Check range breakout
        range_signal = self.detect_range_breakout(
            data, current_price, current_volume, atr
        )
        if range_signal:
            signals.append(range_signal)

        # 2. Check LVN breakout
        lvn_levels = volume_profile.get('lvn_levels', [])
        lvn_signal = self.detect_lvn_breakout(
            current_price, lvn_levels, rsi, macd, atr
        )
        if lvn_signal:
            signals.append(lvn_signal)

        # 3. Check weekly level breakout
        if weekly_levels:
            weekly_signal = self.detect_weekly_level_breakout(
                current_price,
                weekly_levels.get('high', current_price + 0.01),
                weekly_levels.get('low', current_price - 0.01),
                current_volume,
                avg_volume,
                rsi,
                atr
            )
            if weekly_signal:
                signals.append(weekly_signal)

        # Return highest confidence signal
        if signals:
            # Prioritize: weekly > range > lvn
            priority = {'weekly_breakout': 3, 'range_breakout': 2, 'lvn_breakout': 1}
            signals.sort(key=lambda x: priority.get(x['type'], 0), reverse=True)
            best_signal = signals[0]
            logger.info(f"Breakout: Selected {best_signal['type']} signal - {best_signal['direction'].upper()}")
            return best_signal

        logger.debug("Breakout: No valid breakout signals found")
        return None

    def calculate_position_size(
        self,
        base_lot_size: float,
        stop_distance_pips: float,
        account_balance: float,
        risk_percent: float = 1.0
    ) -> float:
        """
        Calculate position size for breakout trade
        Uses smaller lots due to lower win rate

        Args:
            base_lot_size: Base lot size from config
            stop_distance_pips: Stop distance in pips
            account_balance: Account balance
            risk_percent: Risk per trade (%)

        Returns:
            Calculated lot size
        """
        # Apply breakout multiplier (more conservative)
        breakout_lots = base_lot_size * cfg.BREAKOUT_LOT_SIZE_MULTIPLIER

        # If using dynamic sizing
        if not cfg.USE_FIXED_LOT_SIZE:
            risk_amount = account_balance * (risk_percent / 100)
            pip_value = 10  # Standard for 1 lot
            calculated_lots = risk_amount / (stop_distance_pips * pip_value)
            breakout_lots = min(breakout_lots, calculated_lots)

        return round(breakout_lots, 2)


def validate_breakout_time_filters() -> Dict[str, bool]:
    """
    Validate that breakout and mean reversion windows don't overlap significantly

    Returns:
        Validation results
    """
    mr_hours = set(cfg.MEAN_REVERSION_HOURS)
    bo_hours = set(cfg.BREAKOUT_HOURS)

    overlap_hours = mr_hours.intersection(bo_hours)

    return {
        'valid': len(overlap_hours) == 0,
        'overlap_hours': list(overlap_hours),
        'mean_reversion_hours': list(mr_hours),
        'breakout_hours': list(bo_hours),
        'mean_reversion_days': cfg.MEAN_REVERSION_DAYS,
        'breakout_days': cfg.BREAKOUT_DAYS
    }


if __name__ == '__main__':
    # Test time filter validation
    validation = validate_breakout_time_filters()
    print("Breakout Time Filter Validation:")
    print(f"Valid: {validation['valid']}")
    print(f"Mean Reversion Hours: {validation['mean_reversion_hours']}")
    print(f"Breakout Hours: {validation['breakout_hours']}")
    if validation['overlap_hours']:
        print(f"[WARN]  Warning: Overlapping hours: {validation['overlap_hours']}")
    print(f"Mean Reversion Days: {validation['mean_reversion_days']}")
    print(f"Breakout Days: {validation['breakout_days']}")
