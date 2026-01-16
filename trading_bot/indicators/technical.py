"""
Technical Indicators - RSI and MACD
Simple implementations for breakout strategy
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_rsi(data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)

    Args:
        data: DataFrame with price data
        period: RSI period (default 14)
        column: Column to use for calculation

    Returns:
        Series with RSI values (0-100)
    """
    if len(data) < period + 1:
        return pd.Series([50.0] * len(data), index=data.index)

    # Calculate price changes
    delta = data[column].diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period, min_periods=period).mean()
    avg_losses = losses.rolling(window=period, min_periods=period).mean()

    # Calculate RS and RSI
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))

    # Fill NaN with neutral 50
    rsi = rsi.fillna(50.0)

    return rsi


def calculate_macd(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    column: str = 'close'
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence)

    Args:
        data: DataFrame with price data
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
        column: Column to use for calculation

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    if len(data) < slow_period + signal_period:
        zeros = pd.Series([0.0] * len(data), index=data.index)
        return zeros, zeros, zeros

    # Calculate EMAs
    ema_fast = data[column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = data[column].ewm(span=slow_period, adjust=False).mean()

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def add_indicators_to_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI and MACD indicators to DataFrame

    Args:
        data: DataFrame with OHLCV data

    Returns:
        DataFrame with added indicator columns
    """
    df = data.copy()

    # Add RSI
    df['rsi'] = calculate_rsi(df, period=14)

    # Add MACD
    macd_line, signal_line, histogram = calculate_macd(df)
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_histogram'] = histogram

    return df
