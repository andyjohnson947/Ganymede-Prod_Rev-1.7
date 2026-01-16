"""
Signal Detection with Confluence Scoring
Minimum confluence score: 4 (83.3% win rate at optimal score)
"""

import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime

from utils.timezone_manager import get_current_time
from utils.trading_calendar import get_trading_calendar
from indicators.vwap import VWAP
from indicators.volume_profile import VolumeProfile
from indicators.htf_levels import HTFLevels
from indicators.adx import calculate_adx, should_trade_based_on_trend
from config.strategy_config import (
    MIN_CONFLUENCE_SCORE,
    CONFLUENCE_WEIGHTS,
    LEVEL_TOLERANCE_PCT,
    TREND_FILTER_ENABLED,
    ADX_PERIOD,
    ADX_THRESHOLD,
    CANDLE_LOOKBACK,
    ALLOW_WEAK_TRENDS
)


class SignalDetector:
    """Detect entry signals based on confluence of multiple factors"""

    def __init__(self):
        """Initialize signal detector with indicators"""
        self.vwap = VWAP()
        self.volume_profile = VolumeProfile()
        self.htf_levels = HTFLevels()

    def detect_fair_value_gaps(self, data: pd.DataFrame, price: float, tolerance_pct: float = 0.003) -> Dict:
        """
        Detect Fair Value Gaps (FVGs) near current price.

        FVG = Price imbalance where candle 1's high < candle 3's low (bullish)
                                 or candle 1's low > candle 3's high (bearish)

        Args:
            data: DataFrame with OHLC data
            price: Current price to check
            tolerance_pct: Price tolerance (0.3% = 30 pips)

        Returns:
            Dictionary with near_bullish_fvg and near_bearish_fvg flags
        """
        if len(data) < 10:
            return {'near_bullish_fvg': False, 'near_bearish_fvg': False}

        # Look back through recent candles (last 20)
        lookback = min(20, len(data) - 3)
        near_bullish = False
        near_bearish = False

        for i in range(len(data) - 3, len(data) - 3 - lookback, -1):
            if i < 0:
                break

            candle1 = data.iloc[i]
            candle3 = data.iloc[i + 2]

            # Bullish FVG: Candle 1 high < Candle 3 low (gap up, support zone)
            if candle1['high'] < candle3['low']:
                gap_low = candle1['high']
                gap_high = candle3['low']

                # Check if current price is near this FVG
                if (gap_low * (1 - tolerance_pct) <= price <= gap_high * (1 + tolerance_pct)):
                    near_bullish = True
                    break  # Found nearest FVG

            # Bearish FVG: Candle 1 low > Candle 3 high (gap down, resistance zone)
            elif candle1['low'] > candle3['high']:
                gap_high = candle1['low']
                gap_low = candle3['high']

                # Check if current price is near this FVG
                if (gap_low * (1 - tolerance_pct) <= price <= gap_high * (1 + tolerance_pct)):
                    near_bearish = True
                    break  # Found nearest FVG

        return {'near_bullish_fvg': near_bullish, 'near_bearish_fvg': near_bearish}

    def detect_signal(
        self,
        current_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        weekly_data: pd.DataFrame,
        symbol: str
    ) -> Optional[Dict]:
        """
        Detect trading signal based on confluence

        Args:
            current_data: H1 timeframe data with VWAP calculated
            daily_data: D1 timeframe data
            weekly_data: W1 timeframe data
            symbol: Trading symbol

        Returns:
            Dict with signal info or None if no signal
        """
        if len(current_data) < 200:
            return None

        # Get current price
        latest = current_data.iloc[-1]
        price = latest['close']

        # Initialize signal
        signal = {
            'symbol': symbol,
            'timestamp': get_current_time(),
            'price': price,
            'direction': None,
            'confluence_score': 0,
            'factors': [],
            'should_trade': False,
            'vwap_signals': {},
            'vp_signals': {},
            'htf_signals': {}
        }

        # Check trading calendar restrictions (bank holidays, weekends, Friday afternoons)
        calendar = get_trading_calendar()
        is_allowed, reason = calendar.is_trading_allowed(signal['timestamp'])
        if not is_allowed:
            signal['should_trade'] = False
            signal['reject_reason'] = reason
            return None  # Don't trade during restricted periods

        # Calculate indicators if not already done
        if 'vwap' not in current_data.columns:
            current_data = self.vwap.calculate(current_data)

        # 1. Check VWAP signals
        vwap_signals = self.vwap.get_signals(current_data)
        signal['vwap_signals'] = vwap_signals

        # Check VWAP bands
        if vwap_signals['in_band_1']:
            signal['confluence_score'] += CONFLUENCE_WEIGHTS.get('vwap_band_1', 1)
            signal['factors'].append('VWAP Band 1')
            # MEAN REVERSION LOGIC: Buy when price is BELOW VWAP (expecting reversion UP)
            #                       Sell when price is ABOVE VWAP (expecting reversion DOWN)
            signal['direction'] = 'buy' if vwap_signals['direction'] == 'below' else 'sell'

        elif vwap_signals['in_band_2']:
            signal['confluence_score'] += CONFLUENCE_WEIGHTS.get('vwap_band_2', 1)
            signal['factors'].append('VWAP Band 2')
            # MEAN REVERSION LOGIC: Buy when price is BELOW VWAP (expecting reversion UP)
            #                       Sell when price is ABOVE VWAP (expecting reversion DOWN)
            signal['direction'] = 'buy' if vwap_signals['direction'] == 'below' else 'sell'

        # 2. Check Volume Profile signals
        vp_signals = self.volume_profile.get_signals(current_data, price, lookback=200)
        signal['vp_signals'] = vp_signals

        if vp_signals['at_poc']:
            signal['confluence_score'] += CONFLUENCE_WEIGHTS.get('poc', 1)
            signal['factors'].append('POC')

        if vp_signals['above_vah']:
            signal['confluence_score'] += CONFLUENCE_WEIGHTS.get('above_vah', 1)
            signal['factors'].append('Above VAH')

        if vp_signals['below_val']:
            signal['confluence_score'] += CONFLUENCE_WEIGHTS.get('below_val', 1)
            signal['factors'].append('Below VAL')

        if vp_signals['at_lvn']:
            signal['confluence_score'] += CONFLUENCE_WEIGHTS.get('lvn', 1)
            signal['factors'].append('Low Volume Node')

        if vp_signals['at_swing_high']:
            signal['confluence_score'] += CONFLUENCE_WEIGHTS.get('swing_high', 1)
            signal['factors'].append('Swing High')

        if vp_signals['at_swing_low']:
            signal['confluence_score'] += CONFLUENCE_WEIGHTS.get('swing_low', 1)
            signal['factors'].append('Swing Low')

        # 3. Check HTF levels (CRITICAL - highest weights)
        htf_levels = self.htf_levels.get_all_levels(daily_data, weekly_data)
        htf_confluence = self.htf_levels.check_confluence(price, htf_levels, LEVEL_TOLERANCE_PCT)

        signal['htf_signals'] = htf_confluence
        signal['confluence_score'] += htf_confluence['score']
        signal['factors'].extend(htf_confluence['factors'])

        # NOTE: Fair Value Gaps (FVGs) are NOT used in production bot
        # FVGs are tracked by ML system ONLY for data collection & analysis
        # Once ML proves FVGs are effective (15+ trades), they can be enabled here
        # detect_fair_value_gaps() method remains available for future use

        # 4. Determine if we should trade based on confluence
        signal['should_trade'] = signal['confluence_score'] >= MIN_CONFLUENCE_SCORE

        # 5. Apply trend filter (if enabled)
        if signal['should_trade'] and TREND_FILTER_ENABLED:
            # Calculate ADX
            data_with_adx = calculate_adx(current_data.copy(), period=ADX_PERIOD)
            latest_adx = data_with_adx.iloc[-1]

            adx_value = latest_adx['adx']
            plus_di = latest_adx['plus_di']
            minus_di = latest_adx['minus_di']

            # Check if we should trade based on trend analysis
            should_trade, trend_reason = should_trade_based_on_trend(
                adx_value=adx_value,
                plus_di=plus_di,
                minus_di=minus_di,
                candle_data=current_data,
                candle_lookback=CANDLE_LOOKBACK,
                adx_threshold=ADX_THRESHOLD,
                allow_weak_trends=ALLOW_WEAK_TRENDS
            )

            signal['trend_filter'] = {
                'adx': adx_value,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'passed': should_trade,
                'reason': trend_reason
            }

            if not should_trade:
                signal['should_trade'] = False
                signal['reject_reason'] = trend_reason
                return None  # Reject signal due to trend filter

        # 6. Finalize direction if not set
        if signal['should_trade'] and signal['direction'] is None:
            # Use VWAP position to determine direction
            # MEAN REVERSION LOGIC: Buy when price is BELOW VWAP (expecting reversion UP)
            #                       Sell when price is ABOVE VWAP (expecting reversion DOWN)
            if vwap_signals['direction'] == 'below':
                signal['direction'] = 'buy'  # Price below VWAP, buy for reversion
            else:
                signal['direction'] = 'sell'  # Price above VWAP, sell for reversion

        # 7. Add detailed signal metadata for debugging
        if signal['should_trade']:
            signal['vwap_value'] = vwap_signals.get('vwap', 0)
            signal['vwap_distance_pct'] = vwap_signals.get('distance_pct', 0)
            signal['price_vs_vwap'] = vwap_signals['direction']  # 'above' or 'below'

            # Validation: Ensure direction logic is correct for mean reversion
            # If price is BELOW VWAP, direction should be BUY
            # If price is ABOVE VWAP, direction should be SELL
            if vwap_signals['direction'] == 'below' and signal['direction'] != 'buy':
                print(f"WARNING: Direction mismatch! Price BELOW VWAP but direction is {signal['direction']}")
                print(f"   Price: {price:.5f}, VWAP: {signal['vwap_value']:.5f}")
                print(f"   Correcting to BUY (mean reversion)")
                signal['direction'] = 'buy'
            elif vwap_signals['direction'] == 'above' and signal['direction'] != 'sell':
                print(f"WARNING: Direction mismatch! Price ABOVE VWAP but direction is {signal['direction']}")
                print(f"   Price: {price:.5f}, VWAP: {signal['vwap_value']:.5f}")
                print(f"   Correcting to SELL (mean reversion)")
                signal['direction'] = 'sell'

        return signal if signal['should_trade'] else None

    def check_exit_signal(
        self,
        position: Dict,
        current_data: pd.DataFrame
    ) -> bool:
        """
        Check if position should be closed (VWAP reversion)

        Args:
            position: Position dict with entry info
            current_data: Current H1 data with VWAP

        Returns:
            bool: True if should exit
        """
        if 'vwap' not in current_data.columns:
            current_data = self.vwap.calculate(current_data)

        latest = current_data.iloc[-1]
        current_price = latest['close']
        vwap = latest['vwap']

        if pd.isna(vwap):
            return False

        entry_price = position['price_open']
        position_type = position['type']

        # Check VWAP reversion
        if position_type == 'buy':
            # For buy positions, exit when price reaches VWAP from below
            if entry_price < vwap and current_price >= vwap:
                return True
        else:
            # For sell positions, exit when price reaches VWAP from above
            if entry_price > vwap and current_price <= vwap:
                return True

        return False

    def analyze_signal_strength(self, signal: Dict) -> str:
        """
        Analyze signal strength based on confluence score

        Args:
            signal: Signal dict from detect_signal()

        Returns:
            str: 'weak', 'medium', 'strong', 'very_strong'
        """
        score = signal['confluence_score']

        if score >= 10:
            return 'very_strong'
        elif score >= 7:
            return 'strong'
        elif score >= 5:
            return 'medium'
        elif score >= MIN_CONFLUENCE_SCORE:
            return 'weak'
        else:
            return 'no_signal'

    def get_signal_summary(self, signal: Optional[Dict]) -> str:
        """
        Get human-readable signal summary

        Args:
            signal: Signal dict or None

        Returns:
            str: Formatted signal summary
        """
        if signal is None:
            return "No signal detected"

        strength = self.analyze_signal_strength(signal)

        summary = []
        summary.append(f"Signal: {signal['direction'].upper()}")
        summary.append(f"   Symbol: {signal['symbol']}")
        summary.append(f"   Price: {signal['price']:.5f}")

        # Add VWAP details for mean reversion signals
        if signal.get('vwap_value'):
            summary.append(f"   VWAP: {signal['vwap_value']:.5f}")
            summary.append(f"   Price vs VWAP: {signal['price_vs_vwap'].upper()} ({signal['vwap_distance_pct']:.2f}%)")

            # Explain mean reversion logic
            if signal['price_vs_vwap'] == 'below':
                summary.append(f"   Logic: Price BELOW VWAP -> BUY (expect reversion UP)")
            else:
                summary.append(f"   Logic: Price ABOVE VWAP -> SELL (expect reversion DOWN)")

        summary.append(f"   Confluence Score: {signal['confluence_score']} ({strength})")
        summary.append(f"   Factors ({len(signal['factors'])}):")

        for factor in signal['factors']:
            summary.append(f"     * {factor}")

        return '\n'.join(summary)

    def filter_signals_by_session(
        self,
        signals: List[Dict],
        current_time: datetime
    ) -> List[Dict]:
        """
        Filter signals by trading session

        Args:
            signals: List of detected signals
            current_time: Current datetime

        Returns:
            List of filtered signals
        """
        from config.strategy_config import TRADE_SESSIONS

        hour = current_time.hour
        day_of_week = current_time.weekday()

        # Check if in trading hours
        in_session = False

        for session_name, session_config in TRADE_SESSIONS.items():
            if not session_config['enabled']:
                continue

            start_hour = int(session_config['start'].split(':')[0])
            end_hour = int(session_config['end'].split(':')[0])

            # Handle sessions that cross midnight
            if start_hour > end_hour:
                if hour >= start_hour or hour < end_hour:
                    in_session = True
                    break
            else:
                if start_hour <= hour < end_hour:
                    in_session = True
                    break

        if not in_session:
            return []

        # Check day of week
        from config.strategy_config import TRADE_DAYS
        if day_of_week not in TRADE_DAYS:
            return []

        return signals

    def rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        Rank signals by confluence score

        Args:
            signals: List of signals

        Returns:
            List of signals sorted by score (highest first)
        """
        return sorted(signals, key=lambda x: x['confluence_score'], reverse=True)
