#!/usr/bin/env python3
"""
Continuous ML Trade Logger
Runs in background, logs every trade's confluence factors automatically
Integrates with trading bot to capture real-time data
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Add paths
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / 'trading_bot'))

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from indicators.vwap import VWAP
from indicators.volume_profile import VolumeProfile
from indicators.htf_levels import HTFLevels
from indicators.adx import calculate_adx
from utils.data_utils import convert_numpy_types
from config.strategy_config import (
    CONFLUENCE_WEIGHTS,
    LEVEL_TOLERANCE_PCT,
    ADX_PERIOD,
    TREND_FILTER_ENABLED
)


class ContinuousMLLogger:
    """
    Background service that logs every trade's confluence factors
    Runs continuously, monitors for new trades, captures data automatically
    """

    def __init__(self, output_dir: str = None):
        # Use absolute path based on project root to avoid duplication
        if output_dir is None:
            project_root = Path(__file__).parent.parent
            output_dir = project_root / "ml_system" / "outputs"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Continuous log file (append mode)
        self.continuous_log = self.output_dir / "continuous_trade_log.jsonl"

        # Track which trades we've already logged
        self.logged_tickets = self._load_logged_tickets()

        # Initialize bot's indicator modules
        self.vwap = VWAP()
        self.volume_profile = VolumeProfile()
        self.htf_levels = HTFLevels()

        self.mt5 = None

    def _load_logged_tickets(self) -> set:
        """Load set of already-logged ticket IDs"""
        logged = set()
        if self.continuous_log.exists():
            with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        logged.add(record['ticket'])
                    except:
                        pass
        return logged

    def connect_mt5(self, login: int, password: str, server: str) -> bool:
        """Connect to MT5"""
        if not MT5_AVAILABLE:
            return False
        if mt5.initialize() and mt5.login(login, password, server):
            self.mt5 = mt5
            return True
        return False

    def fetch_bars_at_time(
        self,
        symbol: str,
        timeframe: int,
        target_time: datetime,
        bars: int = 100
    ) -> Optional[pd.DataFrame]:
        """Fetch historical bars at specific time"""
        if not self.mt5:
            return None

        rates = self.mt5.copy_rates_from(symbol, timeframe, target_time, bars)
        if rates is None or len(rates) == 0:
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def detect_fair_value_gaps(
        self,
        data: pd.DataFrame,
        entry_price: float,
        tolerance_pct: float = 0.003
    ) -> Dict:
        """
        Detect Fair Value Gaps (FVGs) in OHLC data.

        FVG = Price imbalance where candle 1's high < candle 3's low (bullish)
                                 or candle 1's low > candle 3's high (bearish)

        Args:
            data: DataFrame with OHLC data
            entry_price: Current entry price
            tolerance_pct: Price tolerance (0.3% = 30 pips)

        Returns:
            Dictionary with FVG detection results
        """
        if len(data) < 10:
            return {
                'near_bullish_fvg': False,
                'near_bearish_fvg': False,
                'bullish_fvg_count': 0,
                'bearish_fvg_count': 0,
                'closest_bullish_fvg': None,
                'closest_bearish_fvg': None
            }

        bullish_fvgs = []
        bearish_fvgs = []

        # Look back through recent candles (last 20 for daily, last 10 for weekly)
        lookback = min(20, len(data) - 3)

        for i in range(len(data) - 3, len(data) - 3 - lookback, -1):
            if i < 0:
                break

            candle1 = data.iloc[i]
            candle2 = data.iloc[i + 1]
            candle3 = data.iloc[i + 2]

            # Bullish FVG: Candle 1 high < Candle 3 low (gap up, unfilled zone)
            if candle1['high'] < candle3['low']:
                gap_low = candle1['high']
                gap_high = candle3['low']
                bullish_fvgs.append({
                    'low': float(gap_low),
                    'high': float(gap_high),
                    'midpoint': float((gap_low + gap_high) / 2),
                    'size_pct': float((gap_high - gap_low) / gap_low * 100),
                    'age': len(data) - i - 2  # How many candles ago
                })

            # Bearish FVG: Candle 1 low > Candle 3 high (gap down, unfilled zone)
            elif candle1['low'] > candle3['high']:
                gap_high = candle1['low']
                gap_low = candle3['high']
                bearish_fvgs.append({
                    'low': float(gap_low),
                    'high': float(gap_high),
                    'midpoint': float((gap_low + gap_high) / 2),
                    'size_pct': float((gap_high - gap_low) / gap_high * 100),
                    'age': len(data) - i - 2
                })

        # Find closest FVGs to entry price
        closest_bullish = None
        closest_bearish = None
        min_bull_dist = float('inf')
        min_bear_dist = float('inf')

        for fvg in bullish_fvgs:
            dist = abs(fvg['midpoint'] - entry_price) / entry_price
            if dist < min_bull_dist:
                min_bull_dist = dist
                closest_bullish = fvg

        for fvg in bearish_fvgs:
            dist = abs(fvg['midpoint'] - entry_price) / entry_price
            if dist < min_bear_dist:
                min_bear_dist = dist
                closest_bearish = fvg

        # Check if price is near an FVG (within tolerance)
        near_bullish_fvg = False
        near_bearish_fvg = False

        if closest_bullish:
            # Price is near bullish FVG if within tolerance of the gap zone
            if (closest_bullish['low'] * (1 - tolerance_pct) <= entry_price <=
                closest_bullish['high'] * (1 + tolerance_pct)):
                near_bullish_fvg = True

        if closest_bearish:
            # Price is near bearish FVG if within tolerance of the gap zone
            if (closest_bearish['low'] * (1 - tolerance_pct) <= entry_price <=
                closest_bearish['high'] * (1 + tolerance_pct)):
                near_bearish_fvg = True

        return {
            'near_bullish_fvg': near_bullish_fvg,
            'near_bearish_fvg': near_bearish_fvg,
            'bullish_fvg_count': len(bullish_fvgs),
            'bearish_fvg_count': len(bearish_fvgs),
            'closest_bullish_fvg': closest_bullish,
            'closest_bearish_fvg': closest_bearish
        }

    def calculate_confluence_factors(
        self,
        h1_data: pd.DataFrame,
        d1_data: pd.DataFrame,
        w1_data: pd.DataFrame,
        entry_price: float
    ) -> Dict:
        """Calculate confluence factors using bot's modules"""
        if len(h1_data) < 200 or len(d1_data) < 2 or len(w1_data) < 2:
            return {}

        factors = {}

        # VWAP
        h1_with_vwap = self.vwap.calculate(h1_data.copy())
        vwap_signals = self.vwap.get_signals(h1_with_vwap)
        factors['vwap'] = {
            'value': float(vwap_signals.get('vwap', 0)),
            'distance_pct': float(vwap_signals.get('distance_pct', 0)),
            'direction': vwap_signals.get('direction', ''),
            'in_band_1': vwap_signals.get('in_band_1', False),
            'in_band_2': vwap_signals.get('in_band_2', False),
            'band_1_score': CONFLUENCE_WEIGHTS.get('vwap_band_1', 1) if vwap_signals.get('in_band_1') else 0,
            'band_2_score': CONFLUENCE_WEIGHTS.get('vwap_band_2', 1) if vwap_signals.get('in_band_2') else 0,
        }

        # Volume Profile
        vp_signals = self.volume_profile.get_signals(h1_with_vwap, entry_price, lookback=200)
        factors['volume_profile'] = {
            'at_poc': vp_signals.get('at_poc', False),
            'above_vah': vp_signals.get('above_vah', False),
            'below_val': vp_signals.get('below_val', False),
            'at_lvn': vp_signals.get('at_lvn', False),
            'at_swing_high': vp_signals.get('at_swing_high', False),
            'at_swing_low': vp_signals.get('at_swing_low', False),
        }

        # HTF Levels
        htf_all_levels = self.htf_levels.get_all_levels(d1_data, w1_data)
        htf_confluence = self.htf_levels.check_confluence(entry_price, htf_all_levels, LEVEL_TOLERANCE_PCT)
        daily_levels = self.htf_levels.calculate_daily_levels(d1_data)
        weekly_levels = self.htf_levels.calculate_weekly_levels(w1_data)

        factors['htf_levels'] = {
            'total_score': htf_confluence.get('score', 0),
            'factors_matched': htf_confluence.get('factors', []),
            'prev_day_vah': float(daily_levels.get('prev_day_vah', 0)),
            'prev_day_poc': float(daily_levels.get('prev_day_poc', 0)),
            'weekly_hvn_count': len(weekly_levels.get('weekly_hvn', [])),
            'weekly_poc': float(weekly_levels.get('weekly_poc', 0)),
        }

        # Fair Value Gaps (Daily and Weekly)
        daily_fvgs = self.detect_fair_value_gaps(d1_data, entry_price, LEVEL_TOLERANCE_PCT)
        weekly_fvgs = self.detect_fair_value_gaps(w1_data, entry_price, LEVEL_TOLERANCE_PCT)

        factors['fair_value_gaps'] = {
            'daily_bullish_fvg': daily_fvgs['near_bullish_fvg'],
            'daily_bearish_fvg': daily_fvgs['near_bearish_fvg'],
            'weekly_bullish_fvg': weekly_fvgs['near_bullish_fvg'],
            'weekly_bearish_fvg': weekly_fvgs['near_bearish_fvg'],
            'daily_bullish_count': daily_fvgs['bullish_fvg_count'],
            'daily_bearish_count': daily_fvgs['bearish_fvg_count'],
            'weekly_bullish_count': weekly_fvgs['bullish_fvg_count'],
            'weekly_bearish_count': weekly_fvgs['bearish_fvg_count'],
            'closest_daily_bullish': daily_fvgs['closest_bullish_fvg'],
            'closest_daily_bearish': daily_fvgs['closest_bearish_fvg'],
            'closest_weekly_bullish': weekly_fvgs['closest_bullish_fvg'],
            'closest_weekly_bearish': weekly_fvgs['closest_bearish_fvg'],
        }

        # Trend Filter
        if TREND_FILTER_ENABLED:
            h1_with_adx = calculate_adx(h1_with_vwap.copy(), period=ADX_PERIOD)
            latest_adx = h1_with_adx.iloc[-1]
            factors['trend_filter'] = {
                'enabled': True,
                'adx': float(latest_adx.get('adx', 0)),
                'plus_di': float(latest_adx.get('plus_di', 0)),
                'minus_di': float(latest_adx.get('minus_di', 0)),
            }
        else:
            factors['trend_filter'] = {'enabled': False}

        # NEW: Volatility Features (Phase 1)
        factors['volatility'] = self._calculate_volatility_features(h1_data, entry_price)

        # NEW: Entry Quality Features (Phase 1)
        factors['entry_quality'] = self._calculate_entry_quality(h1_data, entry_price, vp_signals, htf_all_levels)

        return factors

    def _calculate_volatility_features(self, h1_data: pd.DataFrame, entry_price: float) -> Dict:
        """
        Calculate volatility context features (Phase 1 enhancement)
        - ATR (Average True Range)
        - ATR percentile
        - Recent range
        - Momentum
        """
        try:
            if len(h1_data) < 50:
                return {}

            # Calculate ATR (14-period)
            high_low = h1_data['high'] - h1_data['low']
            high_close = np.abs(h1_data['high'] - h1_data['close'].shift())
            low_close = np.abs(h1_data['low'] - h1_data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = true_range.rolling(window=14).mean().iloc[-1]

            # ATR percentile (where is current ATR vs last 50 bars?)
            atr_series = true_range.rolling(window=14).mean()
            atr_percentile = (atr_series.iloc[-1] > atr_series.tail(50)).sum() / 50.0

            # Recent range (10-bar high-low range)
            recent_high = h1_data['high'].tail(10).max()
            recent_low = h1_data['low'].tail(10).min()
            recent_range_pips = (recent_high - recent_low) * 10000

            # Momentum (20-bar rate of change)
            close_current = h1_data['close'].iloc[-1]
            close_20_ago = h1_data['close'].iloc[-20] if len(h1_data) >= 20 else h1_data['close'].iloc[0]
            momentum_pct = ((close_current - close_20_ago) / close_20_ago) * 100

            # Distance from recent high/low (entry quality)
            distance_from_high = ((entry_price - recent_high) / entry_price) * 100
            distance_from_low = ((entry_price - recent_low) / entry_price) * 100

            return {
                'atr_14': float(atr_14),
                'atr_percentile': float(atr_percentile),
                'recent_range_pips': float(recent_range_pips),
                'momentum_20_pct': float(momentum_pct),
                'distance_from_high_pct': float(distance_from_high),
                'distance_from_low_pct': float(distance_from_low),
            }

        except Exception as e:
            print(f"[WARN] Volatility calculation failed: {e}")
            return {}

    def _calculate_entry_quality(self, h1_data: pd.DataFrame, entry_price: float,
                                  vp_signals: Dict, htf_all_levels: Dict) -> Dict:
        """
        Calculate entry quality metrics (Phase 1 enhancement)
        - Distance to nearest swing level
        - HTF timeframe alignment
        - Signal conviction
        """
        try:
            if len(h1_data) < 20:
                return {}

            # Distance to nearest swing level
            swing_highs = vp_signals.get('swing_highs', [])
            swing_lows = vp_signals.get('swing_lows', [])

            min_distance_to_swing = 999.0
            if swing_highs or swing_lows:
                all_swings = list(swing_highs) + list(swing_lows)
                distances = [abs((entry_price - swing) / entry_price) * 100 for swing in all_swings]
                min_distance_to_swing = min(distances) if distances else 999.0

            # HTF alignment (how many HTF levels near entry?)
            htf_factors_count = len(htf_all_levels.get('factors', [])) if htf_all_levels else 0
            htf_aligned = htf_factors_count >= 3  # 3+ HTF levels = good alignment

            # Signal conviction (based on factor strength)
            conviction_score = 0
            if vp_signals.get('at_poc'):
                conviction_score += 2
            if vp_signals.get('at_swing_high') or vp_signals.get('at_swing_low'):
                conviction_score += 2
            if vp_signals.get('at_lvn'):
                conviction_score += 1
            if htf_factors_count >= 3:
                conviction_score += 2
            if htf_factors_count >= 5:
                conviction_score += 1

            # Normalized conviction (0-10)
            conviction_normalized = min(conviction_score, 10)

            return {
                'distance_to_swing_pct': float(min_distance_to_swing),
                'htf_aligned': bool(htf_aligned),
                'htf_factors_count': int(htf_factors_count),
                'signal_conviction': int(conviction_normalized),
            }

        except Exception as e:
            print(f"[WARN] Entry quality calculation failed: {e}")
            return {}

    def _calculate_trade_sequencing(self, entry_time: datetime) -> Dict:
        """
        Calculate trade sequencing features (Phase 1 enhancement)
        - Trades in last hour
        - Trades today
        - Win streak
        - Minutes since last trade
        - Open position count (if MT5 available)
        """
        try:
            # Load recent trades from log
            if not self.continuous_log.exists():
                return {}

            recent_trades = []
            with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        trade = json.loads(line)
                        trade_time = datetime.fromisoformat(trade['entry_time'])
                        recent_trades.append({
                            'time': trade_time,
                            'profit': trade.get('outcome', {}).get('profit', 0),
                            'closed': trade.get('outcome', {}).get('status') == 'closed'
                        })
                    except:
                        continue

            # Trades in last hour
            one_hour_ago = entry_time - timedelta(hours=1)
            trades_last_hour = sum(1 for t in recent_trades if t['time'] >= one_hour_ago and t['time'] < entry_time)

            # Trades today
            today_start = entry_time.replace(hour=0, minute=0, second=0, microsecond=0)
            trades_today = sum(1 for t in recent_trades if t['time'] >= today_start and t['time'] < entry_time)

            # Win streak (recent closed trades)
            closed_trades = [t for t in recent_trades if t['closed']]
            closed_trades.sort(key=lambda x: x['time'], reverse=True)

            win_streak = 0
            for trade in closed_trades[:10]:  # Last 10 closed trades
                if trade['profit'] > 0:
                    win_streak += 1
                else:
                    break  # Streak broken

            # Minutes since last trade
            if recent_trades:
                recent_trades_sorted = sorted(recent_trades, key=lambda x: x['time'], reverse=True)
                last_trade_time = recent_trades_sorted[0]['time']
                minutes_since_last = (entry_time - last_trade_time).total_seconds() / 60
            else:
                minutes_since_last = 999.0

            # Open position count (if MT5 available)
            open_count = 0
            if self.mt5:
                try:
                    positions = self.mt5.get_positions()
                    open_count = len(positions) if positions else 0
                except:
                    open_count = 0

            return {
                'trades_last_hour': int(trades_last_hour),
                'trades_today': int(trades_today),
                'win_streak': int(win_streak),
                'minutes_since_last_trade': float(minutes_since_last),
                'open_position_count': int(open_count),
            }

        except Exception as e:
            print(f"[WARN] Trade sequencing calculation failed: {e}")
            return {}

    def _get_trading_session(self, hour: int) -> str:
        """
        Determine trading session based on hour (UTC)
        Tokyo: 0-8, London: 8-16, NY: 13-21, Sydney: 21-24
        """
        if 0 <= hour < 8:
            return 'Tokyo'
        elif 8 <= hour < 13:
            return 'London'
        elif 13 <= hour < 21:
            return 'NY'
        else:
            return 'Sydney'

    def _calculate_market_microstructure(self, symbol: str, entry_price: float, filled_price: float) -> Dict:
        """
        Calculate market microstructure features (Phase 3 enhancement)
        - Spread (bid-ask)
        - Spread percentile
        - Slippage
        """
        try:
            # Get current symbol info for spread
            if self.mt5:
                symbol_info = self.mt5.symbol_info_tick(symbol)
                if symbol_info:
                    spread_pips = (symbol_info.ask - symbol_info.bid) * 10000

                    # Get recent ticks to calculate spread percentile
                    recent_ticks = self.mt5.copy_ticks_from(symbol, datetime.now(), 50, self.mt5.COPY_TICKS_ALL)
                    if recent_ticks is not None and len(recent_ticks) > 0:
                        recent_spreads = [(tick['ask'] - tick['bid']) * 10000 for tick in recent_ticks]
                        avg_spread = np.mean(recent_spreads)
                        spread_percentile = (spread_pips > np.array(recent_spreads)).mean()
                    else:
                        avg_spread = spread_pips
                        spread_percentile = 0.5
                else:
                    spread_pips = 0.0
                    avg_spread = 0.0
                    spread_percentile = 0.5
            else:
                spread_pips = 0.0
                avg_spread = 0.0
                spread_percentile = 0.5

            # Calculate slippage (difference between intended and filled price)
            slippage_pips = abs(filled_price - entry_price) * 10000

            return {
                'spread_pips': float(spread_pips),
                'spread_percentile': float(spread_percentile),
                'spread_vs_avg_pct': float((spread_pips / avg_spread - 1) * 100) if avg_spread > 0 else 0.0,
                'slippage_pips': float(slippage_pips),
            }

        except Exception as e:
            print(f"[WARN] Market microstructure calculation failed: {e}")
            return {
                'spread_pips': 0.0,
                'spread_percentile': 0.5,
                'spread_vs_avg_pct': 0.0,
                'slippage_pips': 0.0,
            }

    def _calculate_position_sizing_context(self, symbol: str, volume: float, entry_time: datetime) -> Dict:
        """
        Calculate position sizing context features (Phase 3 enhancement)
        - Risk percent
        - Position vs average
        - Account drawdown
        - Daily volume used
        """
        try:
            # Get account info
            if self.mt5:
                account_info = self.mt5.account_info()
                if account_info:
                    balance = account_info.balance
                    equity = account_info.equity
                    margin = account_info.margin

                    # Calculate drawdown
                    account_drawdown = ((balance - equity) / balance * 100) if balance > 0 else 0.0

                    # Calculate risk percent (margin used by this position)
                    symbol_info = self.mt5.symbol_info(symbol)
                    if symbol_info:
                        # Rough estimation: 1 lot = ~$100k, leverage typically 1:500
                        # So 0.01 lot ≈ $20 margin
                        position_margin = volume * 2000  # Approximate
                        risk_percent = (position_margin / balance * 100) if balance > 0 else 0.0
                    else:
                        risk_percent = 0.0
                else:
                    balance = 10000.0  # Default
                    account_drawdown = 0.0
                    risk_percent = 0.0
            else:
                balance = 10000.0
                account_drawdown = 0.0
                risk_percent = 0.0

            # Calculate average position size from recent trades
            today_start = entry_time.replace(hour=0, minute=0, second=0, microsecond=0)
            recent_volumes = []
            daily_volume = 0.0

            try:
                with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        trade = json.loads(line)
                        trade_time = datetime.fromisoformat(trade.get('entry_time', ''))

                        # Collect recent volumes (last 30 days)
                        if (entry_time - trade_time).days <= 30:
                            recent_volumes.append(trade.get('volume', 0.0))

                        # Collect today's volume
                        if trade_time >= today_start:
                            daily_volume += trade.get('volume', 0.0)
            except:
                pass

            # Calculate position vs average
            avg_volume = np.mean(recent_volumes) if recent_volumes else 0.01
            position_vs_avg = (volume / avg_volume) if avg_volume > 0 else 1.0

            return {
                'risk_percent': float(min(risk_percent, 100.0)),  # Cap at 100%
                'position_vs_avg': float(position_vs_avg),
                'account_drawdown': float(abs(account_drawdown)),
                'daily_volume_used': float(daily_volume),
                'margin_level_pct': float((equity / margin * 100) if margin > 0 else 999.0),
            }

        except Exception as e:
            print(f"[WARN] Position sizing context calculation failed: {e}")
            return {
                'risk_percent': 0.0,
                'position_vs_avg': 1.0,
                'account_drawdown': 0.0,
                'daily_volume_used': 0.0,
                'margin_level_pct': 999.0,
            }

    def find_recovery_actions(self, ticket: int, from_date: datetime, to_date: datetime) -> Dict:
        """Find all recovery actions (DCA, hedge, grid) for a specific ticket"""
        recovery = {
            'dca_count': 0,
            'hedge_count': 0,
            'grid_count': 0,
            'dca_levels': [],
            'hedge_ratios': [],
            'grid_levels': [],
            'total_recovery_volume': 0.0,
            'recovery_cost': 0.0  # Total loss from recovery trades
        }

        # Get all deals for this period
        deals = self.mt5.history_deals_get(from_date, to_date)
        if not deals:
            return recovery

        # Create multiple ticket search patterns (MT5 comments sometimes truncate ticket numbers)
        # Example: ticket 54573716123 -> comment shows "DCA L1 - 5457371"
        # NOTE: Hedge/Grid use LAST 5 digits (e.g., "Hedge - 16123")
        ticket_str = str(ticket)
        ticket_patterns = [
            ticket_str,           # Full ticket: "54573716123"
            ticket_str[:7],       # Truncated: "5457371"
            ticket_str[:8],       # Truncated: "54573716"
            ticket_str[:9],       # Truncated: "545737161"
            ticket_str[-5:],      # Last 5 digits: "16123" (for Hedge/Grid)
        ]

        for deal in deals:
            comment = deal.comment or ""

            # Check if this deal is related to our ticket
            # Match by position_id OR by ticket pattern in comment
            is_related = (deal.position_id == ticket)

            if not is_related:
                for pattern in ticket_patterns:
                    if pattern in comment:
                        is_related = True
                        break

            if is_related:
                if 'DCA' in comment:
                    recovery['dca_count'] += 1
                    recovery['dca_levels'].append({
                        'price': float(deal.price),
                        'volume': float(deal.volume),
                        'time': datetime.fromtimestamp(deal.time).isoformat()
                    })
                    recovery['total_recovery_volume'] += float(deal.volume)
                    if deal.profit < 0:
                        recovery['recovery_cost'] += abs(float(deal.profit))

                elif 'Hedge' in comment:
                    recovery['hedge_count'] += 1
                    recovery['hedge_ratios'].append({
                        'price': float(deal.price),
                        'volume': float(deal.volume),
                        'time': datetime.fromtimestamp(deal.time).isoformat()
                    })
                    recovery['total_recovery_volume'] += float(deal.volume)
                    if deal.profit < 0:
                        recovery['recovery_cost'] += abs(float(deal.profit))

                elif 'Grid' in comment:
                    # Grid orders have comment like "Grid L1 - 12345"
                    recovery['grid_count'] += 1
                    recovery['grid_levels'].append({
                        'price': float(deal.price),
                        'volume': float(deal.volume),
                        'profit': float(deal.profit),
                        'time': datetime.fromtimestamp(deal.time).isoformat()
                    })

        return recovery

    def find_partial_closes(self, ticket: int, from_date: datetime, to_date: datetime) -> Dict:
        """Find all partial close actions for a specific ticket"""
        partial_closes = {
            'count': 0,
            'closes': [],
            'total_profit_from_partials': 0.0
        }

        deals = self.mt5.history_deals_get(from_date, to_date)
        if not deals:
            return partial_closes

        for deal in deals:
            comment = deal.comment or ""

            if deal.position_id == ticket and ('Partial' in comment or 'partial' in comment):
                partial_closes['count'] += 1
                partial_closes['closes'].append({
                    'price': float(deal.price),
                    'volume': float(deal.volume),
                    'profit': float(deal.profit),
                    'time': datetime.fromtimestamp(deal.time).isoformat(),
                    'comment': comment
                })
                partial_closes['total_profit_from_partials'] += float(deal.profit)

        return partial_closes

    def _calculate_exit_strategy_data(self, entry_price: float, exit_price: float,
                                       direction: str, partial_closes: Dict, exit_comment: str) -> Dict:
        """
        Calculate comprehensive exit strategy metrics for ML analysis.

        Args:
            entry_price: Entry price
            exit_price: Final exit price
            direction: 'BUY' or 'SELL'
            partial_closes: Partial close data
            exit_comment: MT5 exit comment

        Returns:
            Dict with exit strategy metrics
        """
        try:
            # Calculate total pips moved
            if direction == 'BUY':
                final_pips = (exit_price - entry_price) * 10000
            else:  # SELL
                final_pips = (entry_price - exit_price) * 10000

            # Detect exit method from comment and partial data
            exit_method = 'unknown'
            pc1_triggered = False
            pc2_triggered = False
            trailing_triggered = False
            vwap_exit = False
            peak_pips = final_pips  # Default to final if no partials

            # Check partial closes
            if partial_closes['count'] > 0:
                # Analyze partial close prices to infer PC1/PC2
                partial_prices = [p['price'] for p in partial_closes['closes']]

                # Calculate pips for each partial
                partial_pips = []
                for price in partial_prices:
                    if direction == 'BUY':
                        pips = (price - entry_price) * 10000
                    else:
                        pips = (entry_price - price) * 10000
                    partial_pips.append(pips)

                # Infer PC1/PC2 from partial pips
                for pips in partial_pips:
                    if 8 <= pips <= 15:  # PC1 range (10-12 pips)
                        pc1_triggered = True
                    elif 18 <= pips <= 28:  # PC2 range (20-25 pips)
                        pc2_triggered = True

                # Peak pips = highest partial pip value
                peak_pips = max(partial_pips + [final_pips])

            # Detect exit method from comment keywords
            comment_lower = exit_comment.lower()
            if 'trail' in comment_lower or 'trailing' in comment_lower:
                exit_method = 'trailing_stop'
                trailing_triggered = True
            elif 'vwap' in comment_lower:
                exit_method = 'vwap'
                vwap_exit = True
            elif 'pc1' in comment_lower or ('partial' in comment_lower and pc1_triggered and not pc2_triggered):
                exit_method = 'pc1_only'
            elif 'pc2' in comment_lower or ('partial' in comment_lower and pc2_triggered):
                exit_method = 'pc2_full'
            elif partial_closes['count'] == 0:
                # No partials - likely VWAP or manual
                if final_pips < 10:
                    exit_method = 'vwap'
                    vwap_exit = True
                else:
                    exit_method = 'manual'
            else:
                # Has partials but unknown exit
                if pc1_triggered and pc2_triggered:
                    exit_method = 'trailing_stop'  # Assume trailing after PC2
                    trailing_triggered = True
                elif pc1_triggered:
                    exit_method = 'pc1_only'
                else:
                    exit_method = 'manual'

            # Calculate pips given back from peak (for trailing stops)
            pips_from_peak = peak_pips - final_pips

            return {
                'exit_method': exit_method,
                'pc1_triggered': bool(pc1_triggered),
                'pc2_triggered': bool(pc2_triggered),
                'trailing_triggered': bool(trailing_triggered),
                'vwap_exit': bool(vwap_exit),
                'final_pips': float(final_pips),
                'peak_pips': float(peak_pips),
                'pips_from_peak': float(pips_from_peak),  # How much given back
                'capture_ratio': float(final_pips / peak_pips) if peak_pips > 0 else 1.0,  # % of peak captured
                'partial_count': int(partial_closes['count']),
            }

        except Exception as e:
            print(f"[WARN] Exit strategy calculation failed: {e}")
            return {
                'exit_method': 'unknown',
                'pc1_triggered': False,
                'pc2_triggered': False,
                'trailing_triggered': False,
                'vwap_exit': False,
                'final_pips': 0.0,
                'peak_pips': 0.0,
                'pips_from_peak': 0.0,
                'capture_ratio': 1.0,
                'partial_count': 0,
            }

    def update_closed_trades(self):
        """Check for closed positions and update their records with outcome data"""
        if not self.mt5:
            return

        # Get recent closed positions (last 24 hours)
        to_date = datetime.now()
        from_date = to_date - timedelta(hours=24)

        # Get closed positions
        deals = self.mt5.history_deals_get(from_date, to_date)
        if not deals:
            return

        updated_count = 0

        # Read existing log to find trades that need outcome updates
        if not self.continuous_log.exists():
            return

        trades_to_update = []
        with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    trade = json.loads(line)
                    # Skip if already has outcome
                    if 'outcome' not in trade or trade['outcome'].get('status') == 'open':
                        trades_to_update.append(trade)
                except:
                    continue

        # Check each open trade for closure
        for trade in trades_to_update:
            ticket = trade['ticket']

            # Find exit deal for this ticket
            exit_deal = None
            for deal in deals:
                if deal.position_id == ticket and deal.entry == 1:  # Exit deal
                    exit_deal = deal
                    break

            if exit_deal:
                # Trade is closed - get recovery and partial close data
                entry_time = datetime.fromisoformat(trade['entry_time'])
                exit_time = datetime.fromtimestamp(exit_deal.time)

                # Find recovery actions
                recovery = self.find_recovery_actions(ticket, entry_time, exit_time)

                # Find partial closes
                partial_closes = self.find_partial_closes(ticket, entry_time, exit_time)

                # Calculate hold time
                hold_seconds = (exit_time - entry_time).total_seconds()
                hold_hours = hold_seconds / 3600

                # Calculate exit strategy data
                exit_strategy = self._calculate_exit_strategy_data(
                    entry_price=trade['entry_price'],
                    exit_price=float(exit_deal.price),
                    direction=trade.get('direction', 'BUY'),
                    partial_closes=partial_closes,
                    exit_comment=exit_deal.comment or ""
                )

                # Build outcome data
                outcome = {
                    'status': 'closed',
                    'exit_time': exit_time.isoformat(),
                    'exit_price': float(exit_deal.price),
                    'profit': float(exit_deal.profit),
                    'hold_hours': float(hold_hours),
                    'recovery': recovery,
                    'partial_closes': partial_closes,
                    'exit_strategy': exit_strategy,  # NEW: Exit strategy metrics
                    'net_profit': float(exit_deal.profit) - recovery['recovery_cost'],
                    'had_recovery': recovery['dca_count'] + recovery['hedge_count'] > 0,
                    'had_grid': recovery['grid_count'] > 0,
                    'had_partial_close': partial_closes['count'] > 0,
                    'updated_at': datetime.now().isoformat()
                }

                trade['outcome'] = outcome
                updated_count += 1

                print(f"[UPDATE] Trade #{ticket} closed: ${outcome['profit']:.2f} | "
                      f"DCA: {recovery['dca_count']} | Hedge: {recovery['hedge_count']} | "
                      f"Grid: {recovery['grid_count']} | Partials: {partial_closes['count']}")

        # Rewrite log with updates
        if updated_count > 0:
            all_trades = []
            with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        all_trades.append(json.loads(line))
                    except:
                        continue

            # Update trades
            for i, trade in enumerate(all_trades):
                for updated_trade in trades_to_update:
                    if trade['ticket'] == updated_trade['ticket'] and 'outcome' in updated_trade:
                        all_trades[i] = updated_trade
                        break

            # Write back
            with open(self.continuous_log, 'w', encoding='utf-8', errors='ignore') as f:
                for trade in all_trades:
                    trade = convert_numpy_types(trade)
                    f.write(json.dumps(trade) + '\n')

            print(f"[OK] Updated {updated_count} closed trades with outcome data")

    def backfill_missed_trades(self):
        """Backfill any trades that were missed while logger was offline"""
        if not self.mt5:
            print("[BACKFILL] MT5 not connected, skipping backfill")
            return

        try:
            # Find last logged trade timestamp
            last_logged_time = None
            if self.continuous_log.exists():
                with open(self.continuous_log, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        try:
                            trade = json.loads(line)
                            trade_time = datetime.fromisoformat(trade['entry_time'])
                            if last_logged_time is None or trade_time > last_logged_time:
                                last_logged_time = trade_time
                        except:
                            continue

            # If no trades logged, look back 30 days to capture all available history
            to_date = datetime.now()
            if last_logged_time:
                from_date = last_logged_time
                print(f"[BACKFILL] Checking for missed trades since {from_date.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                from_date = to_date - timedelta(days=30)
                print(f"[BACKFILL] No existing log, checking last 30 days for initial dataset")

            # Get all deals in this period (this can take time for large date ranges)
            print(f"[BACKFILL] Fetching deals from MT5...")
            deals = self.mt5.history_deals_get(from_date, to_date)
            if not deals:
                print("[BACKFILL] No deals found to backfill")
                return

            print(f"[BACKFILL] Found {len(deals)} total deals, processing...")

            backfilled = 0
            for deal in deals:
                comment = deal.comment or ""

                # Only log confluence entry trades (old format: "Confluence:", new format: "VWAP:" or "BREAKOUT:")
                is_entry_trade = any(marker in comment for marker in ['Confluence:', 'VWAP:', 'BREAKOUT:'])
                if not is_entry_trade:
                    continue

                # Only log entry deals (type 0 or 1)
                if deal.type not in [0, 1]:
                    continue

                ticket = deal.position_id

                # Skip if already logged
                if ticket in self.logged_tickets:
                    continue

                # Log this trade (same logic as check_for_new_trades)
                try:
                    trade_time = datetime.fromtimestamp(deal.time)
                    symbol = deal.symbol
                    entry_price = deal.price

                    # Parse strategy type and confluence score from comment
                    # New format: "VWAP:C12" or "BREAKOUT:C8"
                    # Old format: "Confluence:12"
                    strategy_type = None
                    confluence_score = None

                    try:
                        if 'VWAP:' in comment:
                            strategy_type = 'vwap'  # Mean reversion to VWAP/levels
                            confluence_score = int(comment.split('VWAP:C')[1].split()[0])
                        elif 'BREAKOUT:' in comment:
                            strategy_type = 'breakout'  # Momentum through levels
                            confluence_score = int(comment.split('BREAKOUT:C')[1].split()[0])
                        elif 'Confluence:' in comment:
                            strategy_type = 'confluence'  # Legacy format
                            confluence_score = int(comment.split('Confluence:')[1].split()[0])
                    except:
                        pass

                    # Get bars at entry time
                    h1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_H1, trade_time, bars=500)
                    d1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_D1, trade_time, bars=100)
                    w1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_W1, trade_time, bars=50)

                    if h1_bars is None or d1_bars is None or w1_bars is None:
                        continue

                    # Calculate confluence factors
                    confluence_factors = self.calculate_confluence_factors(h1_bars, d1_bars, w1_bars, entry_price)

                    if not confluence_factors:
                        continue

                    # Build trade record
                    trade_record = {
                        'ticket': int(ticket),
                        'symbol': symbol,
                        'entry_time': trade_time.isoformat(),
                        'entry_price': float(entry_price),
                        'direction': 'BUY' if deal.type == 0 else 'SELL',
                        'volume': float(deal.volume),
                        'strategy_type': strategy_type,  # 'revision', 'breakout', or 'confluence' (legacy)
                        'confluence_score': confluence_score,
                        'vwap': confluence_factors.get('vwap', {}),
                        'volume_profile': confluence_factors.get('volume_profile', {}),
                        'htf_levels': confluence_factors.get('htf_levels', {}),
                        'fair_value_gaps': confluence_factors.get('fair_value_gaps', {}),
                        'trend_filter': confluence_factors.get('trend_filter', {}),
                        'volatility': confluence_factors.get('volatility', {}),  # NEW: Phase 1
                        'entry_quality': confluence_factors.get('entry_quality', {}),  # NEW: Phase 1
                        'market_context': {
                            'hour': trade_time.hour,
                            'day_of_week': trade_time.strftime('%A'),
                            'session': self._get_trading_session(trade_time.hour),
                        },
                        'trade_sequencing': self._calculate_trade_sequencing(trade_time),  # NEW: Phase 1
                        'market_microstructure': self._calculate_market_microstructure(symbol, entry_price, deal.price),  # NEW: Phase 3
                        'position_sizing': self._calculate_position_sizing_context(symbol, deal.volume, trade_time),  # NEW: Phase 3
                        'logged_at': datetime.now().isoformat()
                    }

                    # Convert numpy types to native Python types
                    trade_record = convert_numpy_types(trade_record)

                    # Append to continuous log
                    with open(self.continuous_log, 'a', encoding='utf-8', errors='ignore') as f:
                        f.write(json.dumps(trade_record) + '\n')

                    # Mark as logged
                    self.logged_tickets.add(ticket)
                    backfilled += 1

                    strategy_label = strategy_type.upper() if strategy_type else 'UNKNOWN'
                    print(f"[BACKFILL] {symbol} #{ticket} ({strategy_label}, C{confluence_score})")

                except Exception as e:
                    print(f"[WARN] [BACKFILL] Failed to log trade {ticket}: {e}")
                    import traceback
                    traceback.print_exc()

            if backfilled > 0:
                print(f"[OK] Backfilled {backfilled} missed trades")
            else:
                print(f"[OK] No missed trades to backfill")

        except Exception as e:
            print(f"[ERROR] Backfill failed: {e}")
            import traceback
            traceback.print_exc()

    def check_for_new_trades(self):
        """Check for new trades and log them"""
        if not self.mt5:
            return

        # Get recent deals (last hour)
        to_date = datetime.now()
        from_date = to_date - timedelta(hours=1)
        deals = self.mt5.history_deals_get(from_date, to_date)

        if not deals:
            return

        new_trades_logged = 0

        for deal in deals:
            comment = deal.comment or ""

            # Only log confluence entry trades (old format: "Confluence:", new format: "VWAP:" or "BREAKOUT:")
            is_entry_trade = any(marker in comment for marker in ['Confluence:', 'VWAP:', 'BREAKOUT:'])
            if not is_entry_trade:
                continue

            # Only log entry deals (type 0 or 1)
            if deal.type not in [0, 1]:
                continue

            ticket = deal.position_id

            # Skip if already logged
            if ticket in self.logged_tickets:
                continue

            # Log this trade
            try:
                trade_time = datetime.fromtimestamp(deal.time)
                symbol = deal.symbol
                entry_price = deal.price

                # Parse strategy type and confluence score from comment
                # New format: "VWAP:C12" or "BREAKOUT:C8"
                # Old format: "Confluence:12"
                strategy_type = None
                confluence_score = None

                try:
                    if 'VWAP:' in comment:
                        strategy_type = 'vwap'  # Mean reversion to VWAP/levels
                        confluence_score = int(comment.split('VWAP:C')[1].split()[0])
                    elif 'BREAKOUT:' in comment:
                        strategy_type = 'breakout'  # Momentum through levels
                        confluence_score = int(comment.split('BREAKOUT:C')[1].split()[0])
                    elif 'Confluence:' in comment:
                        strategy_type = 'confluence'  # Legacy format
                        confluence_score = int(comment.split('Confluence:')[1].split()[0])
                except:
                    pass

                # Get bars at entry time
                h1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_H1, trade_time, bars=500)
                d1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_D1, trade_time, bars=100)
                w1_bars = self.fetch_bars_at_time(symbol, self.mt5.TIMEFRAME_W1, trade_time, bars=50)

                if h1_bars is None or d1_bars is None or w1_bars is None:
                    continue

                # Calculate confluence factors
                confluence_factors = self.calculate_confluence_factors(h1_bars, d1_bars, w1_bars, entry_price)

                if not confluence_factors:
                    continue

                # Build trade record
                trade_record = {
                    'ticket': int(ticket),
                    'symbol': symbol,
                    'entry_time': trade_time.isoformat(),
                    'entry_price': float(entry_price),
                    'direction': 'BUY' if deal.type == 0 else 'SELL',
                    'volume': float(deal.volume),
                    'strategy_type': strategy_type,  # 'revision', 'breakout', or 'confluence' (legacy)
                    'confluence_score': confluence_score,
                    'vwap': confluence_factors.get('vwap', {}),
                    'volume_profile': confluence_factors.get('volume_profile', {}),
                    'htf_levels': confluence_factors.get('htf_levels', {}),
                    'fair_value_gaps': confluence_factors.get('fair_value_gaps', {}),
                    'trend_filter': confluence_factors.get('trend_filter', {}),
                    'volatility': confluence_factors.get('volatility', {}),  # NEW: Phase 1
                    'entry_quality': confluence_factors.get('entry_quality', {}),  # NEW: Phase 1
                    'market_context': {
                        'hour': trade_time.hour,
                        'day_of_week': trade_time.strftime('%A'),
                        'session': self._get_trading_session(trade_time.hour),
                    },
                    'trade_sequencing': self._calculate_trade_sequencing(trade_time),  # NEW: Phase 1
                    'market_microstructure': self._calculate_market_microstructure(symbol, entry_price, deal.price),  # NEW: Phase 3
                    'position_sizing': self._calculate_position_sizing_context(symbol, deal.volume, trade_time),  # NEW: Phase 3
                    'logged_at': datetime.now().isoformat()
                }

                # Convert numpy types to native Python types
                trade_record = convert_numpy_types(trade_record)

                # Append to continuous log
                with open(self.continuous_log, 'a', encoding='utf-8', errors='ignore') as f:
                    f.write(json.dumps(trade_record) + '\n')

                # Mark as logged
                self.logged_tickets.add(ticket)
                new_trades_logged += 1

                strategy_label = strategy_type.upper() if strategy_type else 'UNKNOWN'
                print(f"[LOG] New trade: {symbol} #{ticket} ({strategy_label}, C{confluence_score})")

            except Exception as e:
                print(f"[WARN] [LOGGER] Failed to log trade {ticket}: {e}")

        if new_trades_logged > 0:
            print(f"[OK] Logged {new_trades_logged} new trades")

    def run(self, mt5_login: int, mt5_password: str, mt5_server: str, check_interval: int = 60):
        """
        Run continuous logging service

        Args:
            mt5_login: MT5 account login
            mt5_password: MT5 password
            mt5_server: MT5 server
            check_interval: Seconds between checks (default 60)
        """
        print("="*80)
        print("CONTINUOUS ML TRADE LOGGER")
        print("="*80)
        print(f"Check interval: {check_interval} seconds")
        print(f"Output: {self.continuous_log}")
        print()

        # Connect to MT5
        if not self.connect_mt5(mt5_login, mt5_password, mt5_server):
            print("[ERROR] Failed to connect to MT5")
            return

        print("[OK] Connected to MT5")
        print(f"[OK] Already logged: {len(self.logged_tickets)} trades")
        print("[INFO] Tracking: Entry confluence + Recovery (DCA/Hedge) + Grid + Partial closes")
        print()

        # Backfill any missed trades while logger was offline
        print("[INFO] Checking for missed trades...")
        self.backfill_missed_trades()
        print()

        print("Starting continuous monitoring... (Press Ctrl+C to stop)")
        print()

        try:
            iteration = 0
            while True:
                iteration += 1

                # Check for new entry trades
                self.check_for_new_trades()

                # Check for closed trades and update with recovery/grid/partial data
                # Update every cycle to catch exits quickly
                self.update_closed_trades()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n[STOP] Continuous logger stopped by user")
        except Exception as e:
            print(f"\n[ERROR] Logger crashed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Continuous ML Trade Logger')
    parser.add_argument('--login', type=int, required=True, help='MT5 login')
    parser.add_argument('--password', type=str, required=True, help='MT5 password')
    parser.add_argument('--server', type=str, required=True, help='MT5 server')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds (default: 60)')

    args = parser.parse_args()

    logger = ContinuousMLLogger()
    logger.run(
        mt5_login=args.login,
        mt5_password=args.password,
        mt5_server=args.server,
        check_interval=args.interval
    )


if __name__ == '__main__':
    main()
