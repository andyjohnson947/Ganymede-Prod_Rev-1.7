"""
Volume Profile Indicator
Calculates POC, VAH, VAL, HVN, LVN
POC used in 38.1% of discovered trades
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from config.strategy_config import (
    VP_BINS,
    HVN_LEVELS,
    LVN_LEVELS,
    SWING_LOOKBACK
)


class VolumeProfile:
    """Calculate volume profile and key levels"""

    def __init__(self, bins: int = VP_BINS):
        """
        Initialize Volume Profile calculator

        Args:
            bins: Number of price bins for volume distribution
        """
        self.bins = bins

    def calculate(self, data: pd.DataFrame, lookback: int = None) -> Dict:
        """
        Calculate volume profile for a period

        Args:
            data: DataFrame with OHLCV data
            lookback: Number of bars to include (None = all data)

        Returns:
            Dict with volume profile data
        """
        if lookback:
            df = data.tail(lookback).copy()
        else:
            df = data.copy()

        if len(df) == 0:
            return self._empty_profile()

        # Get price range
        price_min = df['low'].min()
        price_max = df['high'].max()

        if price_min == price_max:
            return self._empty_profile()

        # Create price bins
        bin_size = (price_max - price_min) / self.bins
        bins = np.linspace(price_min, price_max, self.bins + 1)

        # Calculate volume at each price level
        volume_at_price = {}

        for idx, row in df.iterrows():
            # Distribute volume across the price range of the candle
            candle_low = row['low']
            candle_high = row['high']

            # Get volume - handle both 'volume' and 'tick_volume' columns
            if 'volume' in row:
                candle_volume = row['volume']
            elif 'tick_volume' in row:
                candle_volume = row['tick_volume']
            else:
                candle_volume = 1  # Default if no volume data

            # Find bins that overlap with this candle
            bin_indices = np.where((bins >= candle_low) & (bins <= candle_high))[0]

            if len(bin_indices) > 0:
                # Distribute volume evenly across bins
                volume_per_bin = candle_volume / len(bin_indices)

                for bin_idx in bin_indices:
                    if bin_idx not in volume_at_price:
                        volume_at_price[bin_idx] = 0
                    volume_at_price[bin_idx] += volume_per_bin

        if not volume_at_price:
            return self._empty_profile()

        # Calculate POC (Point of Control) - highest volume bin
        poc_bin = max(volume_at_price.items(), key=lambda x: x[1])[0]
        poc_price = price_min + (poc_bin * bin_size) + (bin_size / 2)

        # Calculate Value Area (70% of volume)
        total_volume = sum(volume_at_price.values())
        target_volume = total_volume * 0.70

        # Start from POC and expand outward
        value_area_bins = {poc_bin}
        current_volume = volume_at_price[poc_bin]

        # Expand value area
        low_idx = poc_bin - 1
        high_idx = poc_bin + 1

        while current_volume < target_volume:
            low_vol = volume_at_price.get(low_idx, 0)
            high_vol = volume_at_price.get(high_idx, 0)

            if low_vol == 0 and high_vol == 0:
                break

            if low_vol >= high_vol and low_idx >= 0:
                value_area_bins.add(low_idx)
                current_volume += low_vol
                low_idx -= 1
            elif high_vol > 0 and high_idx < self.bins:
                value_area_bins.add(high_idx)
                current_volume += high_vol
                high_idx += 1
            else:
                break

        # VAH and VAL
        vah_bin = max(value_area_bins)
        val_bin = min(value_area_bins)

        vah_price = price_min + (vah_bin * bin_size) + (bin_size / 2)
        val_price = price_min + (val_bin * bin_size) + (bin_size / 2)

        # HVN (High Volume Nodes) - top N volume bins
        hvn_bins = sorted(volume_at_price.items(), key=lambda x: x[1], reverse=True)[:HVN_LEVELS]
        hvn_levels = [price_min + (bin_idx * bin_size) + (bin_size / 2) for bin_idx, _ in hvn_bins]

        # LVN (Low Volume Nodes) - lowest N volume bins
        lvn_bins = sorted(volume_at_price.items(), key=lambda x: x[1])[:LVN_LEVELS]
        lvn_levels = [price_min + (bin_idx * bin_size) + (bin_size / 2) for bin_idx, _ in lvn_bins]

        return {
            'poc': poc_price,
            'vah': vah_price,
            'val': val_price,
            'hvn_levels': sorted(hvn_levels),
            'lvn_levels': sorted(lvn_levels),
            'volume_distribution': volume_at_price,
            'price_min': price_min,
            'price_max': price_max,
            'total_volume': total_volume
        }

    def _empty_profile(self) -> Dict:
        """Return empty profile structure"""
        return {
            'poc': 0,
            'vah': 0,
            'val': 0,
            'hvn_levels': [],
            'lvn_levels': [],
            'volume_distribution': {},
            'price_min': 0,
            'price_max': 0,
            'total_volume': 0
        }

    def calculate_swing_levels(self, data: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> Dict:
        """
        Calculate swing high and swing low levels

        Args:
            data: DataFrame with OHLC data
            lookback: Bars to look back for swing detection

        Returns:
            Dict with swing levels
        """
        if len(data) < lookback * 2 + 1:
            return {'swing_highs': [], 'swing_lows': []}

        swing_highs = []
        swing_lows = []

        for i in range(lookback, len(data) - lookback):
            # Check if current bar is swing high
            current_high = data.iloc[i]['high']
            is_swing_high = True

            for j in range(i - lookback, i + lookback + 1):
                if j != i and data.iloc[j]['high'] >= current_high:
                    is_swing_high = False
                    break

            if is_swing_high:
                swing_highs.append({
                    'price': current_high,
                    'time': data.index[i],
                    'index': i
                })

            # Check if current bar is swing low
            current_low = data.iloc[i]['low']
            is_swing_low = True

            for j in range(i - lookback, i + lookback + 1):
                if j != i and data.iloc[j]['low'] <= current_low:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append({
                    'price': current_low,
                    'time': data.index[i],
                    'index': i
                })

        # Return top 5 most recent swing levels
        swing_highs = sorted(swing_highs, key=lambda x: x['index'], reverse=True)[:5]
        swing_lows = sorted(swing_lows, key=lambda x: x['index'], reverse=True)[:5]

        return {
            'swing_highs': [s['price'] for s in swing_highs],
            'swing_lows': [s['price'] for s in swing_lows]
        }

    def check_at_level(self, price: float, level: float, tolerance_pct: float = 0.003) -> bool:
        """
        Check if price is at a specific level

        Args:
            price: Current price
            level: Level to check
            tolerance_pct: Tolerance as percentage (default 0.3%)

        Returns:
            bool: True if price is at level
        """
        tolerance = abs(level * tolerance_pct)
        return abs(price - level) <= tolerance

    def get_signals(self, data: pd.DataFrame, price: float, lookback: int = 200) -> Dict:
        """
        Get volume profile signals for current price

        Args:
            data: DataFrame with OHLCV data
            price: Current price to check
            lookback: Bars to include in profile

        Returns:
            Dict with volume profile signals
        """
        profile = self.calculate(data, lookback=lookback)
        swing_levels = self.calculate_swing_levels(data)

        if not profile or profile['poc'] == 0:
            return {
                'at_poc': False,
                'above_vah': False,
                'below_val': False,
                'at_hvn': False,
                'at_lvn': False,
                'at_swing_high': False,
                'at_swing_low': False,
                'profile': profile,
                'swing_levels': swing_levels
            }

        # Check signals
        at_poc = self.check_at_level(price, profile['poc'])
        above_vah = price > profile['vah']
        below_val = price < profile['val']

        # Check HVN levels
        at_hvn = False
        for hvn in profile['hvn_levels']:
            if self.check_at_level(price, hvn):
                at_hvn = True
                break

        # Check LVN levels
        at_lvn = False
        for lvn in profile['lvn_levels']:
            if self.check_at_level(price, lvn):
                at_lvn = True
                break

        # Check swing levels (mutually exclusive - check low only if high not hit)
        at_swing_high = False
        for swing_high in swing_levels['swing_highs']:
            if self.check_at_level(price, swing_high):
                at_swing_high = True
                break

        at_swing_low = False
        if not at_swing_high:  # Only check swing lows if NOT at swing high
            for swing_low in swing_levels['swing_lows']:
                if self.check_at_level(price, swing_low):
                    at_swing_low = True
                    break

        return {
            'at_poc': at_poc,
            'above_vah': above_vah,
            'below_val': below_val,
            'at_hvn': at_hvn,
            'at_lvn': at_lvn,
            'at_swing_high': at_swing_high,
            'at_swing_low': at_swing_low,
            'profile': profile,
            'swing_levels': swing_levels
        }
