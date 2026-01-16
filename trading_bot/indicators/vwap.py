"""
VWAP (Volume-Weighted Average Price) Indicator
Primary signal used in 28-39% of discovered trades
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict

from config.strategy_config import VWAP_PERIOD, VWAP_BAND_MULTIPLIERS


class VWAP:
    """Calculate VWAP with standard deviation bands"""

    def __init__(self, period: int = VWAP_PERIOD):
        """
        Initialize VWAP calculator

        Args:
            period: Rolling period for VWAP calculation (default: 200)
        """
        self.period = period

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP and deviation bands

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            DataFrame with added VWAP columns
        """
        df = data.copy()

        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Ensure volume column exists (handle MT5's tick_volume)
        if 'volume' not in df.columns:
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            else:
                print("[WARN] Warning: No volume data for VWAP, using equal-weighted average")
                df['volume'] = 1  # Fallback to equal-weighted if no volume

        # Volume-weighted typical price
        df['vwtp'] = df['typical_price'] * df['volume']

        # Rolling VWAP
        df['vwap'] = (
            df['vwtp'].rolling(window=self.period).sum() /
            df['volume'].rolling(window=self.period).sum()
        )

        # Calculate standard deviation for bands
        df['vwap_std'] = self._calculate_vwap_std(df, self.period)

        # Create bands (±1σ, ±2σ, ±3σ)
        for multiplier in VWAP_BAND_MULTIPLIERS:
            df[f'vwap_upper_{multiplier}'] = df['vwap'] + (df['vwap_std'] * multiplier)
            df[f'vwap_lower_{multiplier}'] = df['vwap'] - (df['vwap_std'] * multiplier)

        # Cleanup temporary columns
        df.drop(['typical_price', 'vwtp'], axis=1, inplace=True)

        return df

    def _calculate_vwap_std(self, df: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate volume-weighted standard deviation

        Args:
            df: DataFrame with price and volume data
            period: Rolling period

        Returns:
            Series with VWAP standard deviation
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        volume = df['volume']

        # Rolling volume-weighted variance
        def weighted_std(prices, volumes):
            if len(prices) == 0 or volumes.sum() == 0:
                return 0

            # Volume-weighted mean
            weighted_mean = (prices * volumes).sum() / volumes.sum()

            # Volume-weighted variance
            weighted_var = ((prices - weighted_mean) ** 2 * volumes).sum() / volumes.sum()

            # Standard deviation
            return np.sqrt(weighted_var)

        # Apply rolling calculation
        std_series = pd.Series(index=df.index, dtype=float)

        for i in range(period - 1, len(df)):
            window_prices = typical_price.iloc[i - period + 1:i + 1]
            window_volumes = volume.iloc[i - period + 1:i + 1]
            std_series.iloc[i] = weighted_std(window_prices, window_volumes)

        return std_series

    def check_price_in_band(
        self,
        price: float,
        vwap: float,
        vwap_std: float,
        band: int = 1
    ) -> Tuple[bool, str]:
        """
        Check if price is within a specific VWAP band

        Args:
            price: Current price
            vwap: VWAP value
            vwap_std: VWAP standard deviation
            band: Band number (1, 2, or 3)

        Returns:
            Tuple of (is_in_band, direction)
            direction is 'above' or 'below' VWAP
        """
        upper = vwap + (vwap_std * band)
        lower = vwap - (vwap_std * band)

        if band == 1:
            # Band 1: ±1σ
            in_band = lower <= price <= upper
        elif band == 2:
            # Band 2: ±2σ (but outside band 1)
            band1_upper = vwap + vwap_std
            band1_lower = vwap - vwap_std
            in_band = (lower <= price < band1_lower) or (band1_upper < price <= upper)
        elif band == 3:
            # Band 3: ±3σ (but outside band 2)
            band2_upper = vwap + (vwap_std * 2)
            band2_lower = vwap - (vwap_std * 2)
            in_band = (lower <= price < band2_lower) or (band2_upper < price <= upper)
        else:
            return False, 'none'

        direction = 'above' if price > vwap else 'below'

        return in_band, direction

    def get_signals(self, data: pd.DataFrame, index: int = -1) -> Dict:
        """
        Get VWAP signals for a specific candle

        Args:
            data: DataFrame with VWAP calculated
            index: Index of candle (-1 for latest)

        Returns:
            Dict with VWAP signals
        """
        if 'vwap' not in data.columns:
            raise ValueError("VWAP not calculated. Call calculate() first.")

        row = data.iloc[index]
        price = row['close']
        vwap = row['vwap']
        vwap_std = row['vwap_std']

        if pd.isna(vwap) or pd.isna(vwap_std):
            return {
                'in_band_1': False,
                'in_band_2': False,
                'in_band_3': False,
                'direction': 'none',
                'distance_pct': 0,
                'vwap': 0,
            }

        # Check each band
        in_band_1, direction_1 = self.check_price_in_band(price, vwap, vwap_std, 1)
        in_band_2, direction_2 = self.check_price_in_band(price, vwap, vwap_std, 2)
        in_band_3, direction_3 = self.check_price_in_band(price, vwap, vwap_std, 3)

        direction = 'above' if price > vwap else 'below'
        distance_pct = ((price - vwap) / vwap) * 100

        return {
            'in_band_1': in_band_1,
            'in_band_2': in_band_2,
            'in_band_3': in_band_3,
            'direction': direction,
            'distance_pct': distance_pct,
            'vwap': vwap,
            'vwap_std': vwap_std,
            'upper_1': row.get('vwap_upper_1', 0),
            'lower_1': row.get('vwap_lower_1', 0),
            'upper_2': row.get('vwap_upper_2', 0),
            'lower_2': row.get('vwap_lower_2', 0),
            'upper_3': row.get('vwap_upper_3', 0),
            'lower_3': row.get('vwap_lower_3', 0),
        }

    def get_reversion_target(self, data: pd.DataFrame, index: int = -1) -> float:
        """
        Get VWAP reversion target (used for take profit)

        Args:
            data: DataFrame with VWAP calculated
            index: Index of candle (-1 for latest)

        Returns:
            float: VWAP value as reversion target
        """
        if 'vwap' not in data.columns:
            raise ValueError("VWAP not calculated. Call calculate() first.")

        vwap = data.iloc[index]['vwap']
        return vwap if not pd.isna(vwap) else 0
