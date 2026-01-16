"""
Feature Extractor for ML System
Extracts 50 base features (expanding to ~60 after one-hot encoding)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional


class FeatureExtractor:
    """
    Extracts features from trade records for ML model training/prediction.

    Features:
    - 6 VWAP confluence features
    - 6 Volume profile features
    - 10 HTF (Higher Time Frame) confluence features
    - 4 Fair Value Gap (FVG) features
    - 7 Market behavior & trend features
    - 8 Temporal features
    - 11 Recovery mechanism features (DCA, hedge, grid, partials)
    - 6 Volatility features (Phase 1)
    - 4 Entry quality features (Phase 1)
    - 5 Trade sequencing features (Phase 1)
    - 4 Market microstructure features (Phase 3 - NEW)
    - 5 Position sizing features (Phase 3 - NEW)
    - 1 Overall confluence score
    - 1 Trade direction feature

    Total: 78 base features -> ~89 after one-hot encoding
    """

    def __init__(self):
        self.feature_names = []
        self._setup_feature_names()

    def _setup_feature_names(self):
        """Define all feature names for reference"""
        # Will be populated as features are extracted
        pass

    def extract_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all ML features from a trade record.

        Args:
            trade_record: Trade data dictionary from continuous_trade_log.jsonl

        Returns:
            Dictionary of feature_name: value pairs
        """
        features = {}

        # 1. VWAP Confluence Features (6)
        features.update(self._extract_vwap_features(trade_record))

        # 2. Volume Profile Features (6)
        features.update(self._extract_volume_profile_features(trade_record))

        # 3. HTF Confluence Features (10)
        features.update(self._extract_htf_features(trade_record))

        # 4. Market Behavior & Trend Features (7)
        features.update(self._extract_market_features(trade_record))

        # 5. Temporal Features (8)
        features.update(self._extract_temporal_features(trade_record))

        # 6. Recovery Mechanism Features (11)
        features.update(self._extract_recovery_features(trade_record))

        # 7. Volatility Features (6) - Phase 1
        features.update(self._extract_volatility_features(trade_record))

        # 8. Entry Quality Features (4) - Phase 1
        features.update(self._extract_entry_quality_features(trade_record))

        # 9. Trade Sequencing Features (5) - Phase 1
        features.update(self._extract_trade_sequencing_features(trade_record))

        # 10. Market Microstructure Features (4) - Phase 3
        features.update(self._extract_market_microstructure_features(trade_record))

        # 11. Position Sizing Features (5) - Phase 3
        features.update(self._extract_position_sizing_features(trade_record))

        # 12. Overall Confluence Score (1)
        features['confluence_score'] = trade_record.get('confluence_score', 0)

        # 13. Trade Direction (1)
        features['direction_sell'] = int(trade_record.get('direction', 'BUY') == 'SELL')
        features['direction_buy'] = int(trade_record.get('direction', 'BUY') == 'BUY')

        return features

    def _extract_vwap_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract VWAP-related features (6 features)"""
        features = {}
        vwap_data = trade_record.get('vwap', {})

        # Feature 1: VWAP distance percentage
        features['vwap_distance_pct'] = vwap_data.get('distance_pct', 0.0)

        # Feature 2-3: VWAP bands
        in_band_1 = vwap_data.get('in_band_1', False)
        in_band_2 = vwap_data.get('in_band_2', False)

        features['vwap_band_1'] = int(in_band_1)
        features['vwap_band_2'] = int(in_band_2)

        # Feature 4: VWAP band 3 (beyond ±2σ - for breakouts)
        features['vwap_band_3'] = int(not in_band_1 and not in_band_2)

        # Feature 5: VWAP direction (one-hot encoded)
        direction = vwap_data.get('direction', 'at')
        features['vwap_above'] = int(direction == 'above')
        features['vwap_below'] = int(direction == 'below')
        features['vwap_at'] = int(direction == 'at')

        # Feature 6: Combined band score
        features['vwap_band_score'] = (
            vwap_data.get('band_1_score', 0) +
            vwap_data.get('band_2_score', 0)
        )

        return features

    def _extract_volume_profile_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract volume profile features (6 features)"""
        features = {}
        vp_data = trade_record.get('volume_profile', {})

        # Features 7-12: Volume profile levels
        features['at_poc'] = int(vp_data.get('at_poc', False))
        features['at_lvn'] = int(vp_data.get('at_lvn', False))
        features['above_vah'] = int(vp_data.get('above_vah', False))
        features['below_val'] = int(vp_data.get('below_val', False))
        features['at_swing_high'] = int(vp_data.get('at_swing_high', False))
        features['at_swing_low'] = int(vp_data.get('at_swing_low', False))

        return features

    def _extract_htf_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Higher Time Frame confluence features (10 features)"""
        features = {}
        htf_data = trade_record.get('htf_levels', {})

        # Feature 13: Total HTF score
        features['htf_total_score'] = htf_data.get('total_score', 0)

        # Features 14-22: Individual HTF level matches
        factors_matched = htf_data.get('factors_matched', [])

        # Convert to lowercase for case-insensitive matching
        factors_matched_lower = [f.lower() for f in factors_matched]

        features['at_prev_day_vah'] = int('prev day vah' in factors_matched_lower)
        features['at_prev_day_val'] = int('prev day val' in factors_matched_lower)
        features['at_prev_day_poc'] = int('prev day poc' in factors_matched_lower)
        features['at_prev_day_high'] = int('prev day high' in factors_matched_lower)
        features['at_prev_day_low'] = int('prev day low' in factors_matched_lower)
        features['at_weekly_hvn'] = int('weekly hvn' in factors_matched_lower)
        features['at_daily_hvn'] = int('daily hvn' in factors_matched_lower)
        features['at_prev_week_high'] = int('prev week high' in factors_matched_lower)
        features['at_prev_week_low'] = int('prev week low' in factors_matched_lower)

        # Feature 23: Weekly HVN count
        features['weekly_hvn_count'] = htf_data.get('weekly_hvn_count', 0)

        # Features 24-27: Fair Value Gap confluence (NEW)
        fvg_data = trade_record.get('fair_value_gaps', {})
        features['at_daily_bullish_fvg'] = int(fvg_data.get('daily_bullish_fvg', False))
        features['at_daily_bearish_fvg'] = int(fvg_data.get('daily_bearish_fvg', False))
        features['at_weekly_bullish_fvg'] = int(fvg_data.get('weekly_bullish_fvg', False))
        features['at_weekly_bearish_fvg'] = int(fvg_data.get('weekly_bearish_fvg', False))

        return features

    def _extract_market_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market behavior and trend features (7 features)"""
        features = {}
        trend_data = trade_record.get('trend_filter', {})

        # Features 24-26: Raw ADX and DI values
        adx = trend_data.get('adx', 0.0)
        plus_di = trend_data.get('plus_di', 0.0)
        minus_di = trend_data.get('minus_di', 0.0)

        features['adx'] = adx
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di

        # Feature 27: Trend strength (derived)
        features['trend_strength'] = abs(plus_di - minus_di)

        # Feature 28: Trend direction (one-hot encoded)
        features['trend_bullish'] = int(plus_di > minus_di)
        features['trend_bearish'] = int(plus_di <= minus_di)

        # Feature 29: ADX regime (one-hot encoded)
        features['regime_ranging'] = int(adx < 20)
        features['regime_trending'] = int(20 <= adx <= 25)
        features['regime_strong'] = int(adx > 25)

        # Feature 30: DI spread
        features['di_spread'] = plus_di - minus_di

        return features

    def _extract_temporal_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal features (8 features)"""
        features = {}

        # Parse entry time
        entry_time_str = trade_record.get('entry_time', '')
        try:
            entry_time = pd.to_datetime(entry_time_str)
        except:
            # Fallback to default if parsing fails
            entry_time = datetime.now()

        market_context = trade_record.get('market_context', {})

        # Feature 31: Hour of day
        features['hour'] = market_context.get('hour', entry_time.hour)

        # Feature 32: Day of week (Monday=0)
        day_of_week_str = market_context.get('day_of_week', '')
        day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        features['day_of_week'] = day_map.get(day_of_week_str, entry_time.dayofweek)

        # Feature 33: Day of month
        features['day_of_month'] = entry_time.day

        # Feature 34: Session (one-hot encoded)
        session = market_context.get('session', 'London')
        features['session_tokyo'] = int(session == 'Tokyo')
        features['session_london'] = int(session == 'London')
        features['session_ny'] = int(session in ['New_York', 'NY'])
        features['session_sydney'] = int(session == 'Sydney')

        # Feature 35: Session open (first hour)
        hour = features['hour']
        features['is_session_open'] = int(hour in [0, 1, 7, 8, 13, 14])

        # Feature 36: Session close (last hour)
        features['is_session_close'] = int(hour in [6, 12, 21])

        # Feature 37: Session overlap (London/NY overlap = high liquidity)
        features['is_overlap'] = int(13 <= hour <= 16)

        # Feature 38: Week of year
        features['week_of_year'] = entry_time.isocalendar()[1]

        return features

    def _extract_recovery_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract recovery mechanism features (11 features)"""
        features = {}

        # Get outcome data - only available for closed trades
        outcome = trade_record.get('outcome', {})

        # If trade is not closed or has no outcome, set all to 0/False
        if not outcome or outcome.get('status') != 'closed':
            features['had_dca'] = 0
            features['dca_count'] = 0
            features['had_hedge'] = 0
            features['hedge_count'] = 0
            features['had_grid'] = 0
            features['grid_count'] = 0
            features['had_partial_close'] = 0
            features['partial_close_count'] = 0
            features['total_recovery_volume'] = 0.0
            features['recovery_cost'] = 0.0
            features['partial_profit_contribution'] = 0.0
            return features

        # Extract recovery data
        recovery = outcome.get('recovery', {})
        partials = outcome.get('partial_closes', {})

        # Feature 39-40: DCA features
        dca_count = recovery.get('dca_count', 0)
        features['had_dca'] = int(dca_count > 0)
        features['dca_count'] = dca_count

        # Feature 41-42: Hedge features
        hedge_count = recovery.get('hedge_count', 0)
        features['had_hedge'] = int(hedge_count > 0)
        features['hedge_count'] = hedge_count

        # Feature 43-44: Grid features
        grid_count = recovery.get('grid_count', 0)
        features['had_grid'] = int(grid_count > 0)
        features['grid_count'] = grid_count

        # Feature 45-46: Partial close features
        partial_count = partials.get('count', 0)
        features['had_partial_close'] = int(partial_count > 0)
        features['partial_close_count'] = partial_count

        # Feature 47: Total recovery volume
        features['total_recovery_volume'] = recovery.get('total_recovery_volume', 0.0)

        # Feature 48: Recovery cost
        features['recovery_cost'] = recovery.get('recovery_cost', 0.0)

        # Feature 49: Partial profit contribution
        features['partial_profit_contribution'] = partials.get('total_profit_from_partials', 0.0)

        return features

    def _extract_volatility_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract volatility features (6 features) - Phase 1"""
        features = {}
        volatility_data = trade_record.get('volatility', {})

        # Features: ATR, percentile, range, momentum, distances
        features['atr_14'] = volatility_data.get('atr_14', 0.0)
        features['atr_percentile'] = volatility_data.get('atr_percentile', 0.5)
        features['recent_range_pips'] = volatility_data.get('recent_range_pips', 0.0)
        features['momentum_20_pct'] = volatility_data.get('momentum_20_pct', 0.0)
        features['distance_from_high_pct'] = volatility_data.get('distance_from_high_pct', 0.0)
        features['distance_from_low_pct'] = volatility_data.get('distance_from_low_pct', 0.0)

        return features

    def _extract_entry_quality_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entry quality features (4 features) - Phase 1"""
        features = {}
        entry_quality_data = trade_record.get('entry_quality', {})

        # Features: swing distance, HTF alignment, conviction
        features['distance_to_swing_pct'] = entry_quality_data.get('distance_to_swing_pct', 999.0)
        features['htf_aligned'] = int(entry_quality_data.get('htf_aligned', False))
        features['htf_factors_count'] = entry_quality_data.get('htf_factors_count', 0)
        features['signal_conviction'] = entry_quality_data.get('signal_conviction', 0)

        return features

    def _extract_trade_sequencing_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trade sequencing features (5 features) - Phase 1"""
        features = {}
        sequencing_data = trade_record.get('trade_sequencing', {})

        # Features: overtrading detection, win streaks, timing
        features['trades_last_hour'] = sequencing_data.get('trades_last_hour', 0)
        features['trades_today'] = sequencing_data.get('trades_today', 0)
        features['win_streak'] = sequencing_data.get('win_streak', 0)
        features['minutes_since_last_trade'] = sequencing_data.get('minutes_since_last_trade', 999.0)
        features['open_position_count'] = sequencing_data.get('open_position_count', 0)

        return features

    def _extract_market_microstructure_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market microstructure features (4 features) - Phase 3"""
        features = {}
        microstructure_data = trade_record.get('market_microstructure', {})

        # Features: spread, spread percentile, slippage
        features['spread_pips'] = microstructure_data.get('spread_pips', 0.0)
        features['spread_percentile'] = microstructure_data.get('spread_percentile', 0.5)
        features['spread_vs_avg_pct'] = microstructure_data.get('spread_vs_avg_pct', 0.0)
        features['slippage_pips'] = microstructure_data.get('slippage_pips', 0.0)

        return features

    def _extract_position_sizing_features(self, trade_record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract position sizing features (5 features) - Phase 3"""
        features = {}
        position_data = trade_record.get('position_sizing', {})

        # Features: risk %, position vs avg, drawdown, daily volume, margin level
        features['risk_percent'] = position_data.get('risk_percent', 0.0)
        features['position_vs_avg'] = position_data.get('position_vs_avg', 1.0)
        features['account_drawdown'] = position_data.get('account_drawdown', 0.0)
        features['daily_volume_used'] = position_data.get('daily_volume_used', 0.0)
        features['margin_level_pct'] = position_data.get('margin_level_pct', 999.0)

        return features

    def extract_target(self, trade_record: Dict[str, Any]) -> Optional[int]:
        """
        Extract target variable (win/loss) from trade record.

        Args:
            trade_record: Trade data dictionary

        Returns:
            1 if winning trade (profit > 0)
            0 if losing trade (profit <= 0)
            None if trade still open (no outcome)
        """
        outcome = trade_record.get('outcome', {})

        # Check if trade is closed
        if not outcome or outcome.get('status') != 'closed':
            return None

        # Get profit
        profit = outcome.get('profit', 0.0)

        # Binary classification: 1 = win, 0 = loss
        return 1 if profit > 0 else 0

    def get_feature_names(self) -> list:
        """
        Get list of all feature names after one-hot encoding.

        Returns:
            List of feature names
        """
        # Create dummy features to extract names
        dummy_trade = {
            'confluence_score': 0,
            'direction': 'BUY',
            'vwap': {},
            'volume_profile': {},
            'htf_levels': {},
            'trend_filter': {},
            'market_context': {},
            'entry_time': '2025-01-01T00:00:00'
        }

        features = self.extract_features(dummy_trade)
        return sorted(features.keys())

    def validate_features(self, features: Dict[str, Any]) -> bool:
        """
        Validate extracted features for expected ranges and types.

        Args:
            features: Dictionary of extracted features

        Returns:
            True if all features are valid, False otherwise
        """
        try:
            # Check for NaN/None values
            for key, value in features.items():
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    print(f"Warning: Feature '{key}' has None/NaN value")
                    return False

            # Check ADX range [0-100]
            if not (0 <= features.get('adx', 0) <= 100):
                print(f"Warning: ADX out of range: {features.get('adx')}")
                return False

            # Check DI ranges [0-100]
            if not (0 <= features.get('plus_di', 0) <= 100):
                print(f"Warning: plus_di out of range: {features.get('plus_di')}")
                return False

            if not (0 <= features.get('minus_di', 0) <= 100):
                print(f"Warning: minus_di out of range: {features.get('minus_di')}")
                return False

            # Check confluence score is positive
            if features.get('confluence_score', 0) < 0:
                print(f"Warning: Negative confluence score: {features.get('confluence_score')}")
                return False

            return True

        except Exception as e:
            print(f"Validation error: {e}")
            return False


# Convenience function
def extract_features_from_trade(trade_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to extract features from a single trade.

    Args:
        trade_record: Trade data dictionary

    Returns:
        Dictionary of features
    """
    extractor = FeatureExtractor()
    return extractor.extract_features(trade_record)
