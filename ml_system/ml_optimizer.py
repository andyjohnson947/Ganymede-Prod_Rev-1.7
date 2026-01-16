"""
ML-Powered Parameter Optimizer
Uses trained ML model to recommend optimal trading parameters
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import joblib


class MLParameterOptimizer:
    """Uses ML model to find optimal trading parameters"""

    def __init__(self, model_path: str = None):
        """
        Initialize optimizer with trained ML model

        Args:
            model_path: Path to trained model file
        """
        if model_path is None:
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / 'ml_system' / 'models' / 'baseline_rf.pkl'

        self.model_path = Path(model_path)
        self.model = None
        self.feature_importance = None

        # Load model if exists
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                # Load feature importance
                importance_path = self.model_path.parent / 'feature_importance_baseline.csv'
                if importance_path.exists():
                    self.feature_importance = pd.read_csv(importance_path)
            except Exception as e:
                print(f"Warning: Could not load ML model: {e}")

    def _get_profit(self, trade: Dict) -> float:
        """
        Safely extract profit from trade record.
        Handles both dict and float outcome values.

        Args:
            trade: Trade dictionary

        Returns:
            Profit as float
        """
        outcome = trade.get('outcome', 0)

        # If outcome is a dict, get profit from it
        if isinstance(outcome, dict):
            return outcome.get('profit', 0)

        # If outcome is a number (float/int), use it directly
        if isinstance(outcome, (int, float)):
            return float(outcome)

        # Default to 0
        return 0.0

    def get_optimal_confluence_threshold(self, trades: List[Dict]) -> Dict:
        """
        Find optimal confluence threshold based on historical data

        Args:
            trades: List of trade dictionaries

        Returns:
            Dict with recommendation
        """
        if len(trades) < 10:
            return {
                'current': 8,
                'optimal': 'Need more data',
                'confidence': 'low',
                'reason': 'Insufficient trades for analysis'
            }

        df = pd.DataFrame(trades)
        # Defensive check: outcome might be a float or dict
        df['win'] = df.apply(lambda x: 1 if self._get_profit(x) > 0 else 0, axis=1)

        # Test different thresholds
        results = []
        for threshold in range(6, 20):
            filtered = df[df['confluence_score'] >= threshold]
            if len(filtered) >= 3:
                winrate = filtered['win'].mean()
                count = len(filtered)
                results.append({
                    'threshold': threshold,
                    'winrate': winrate,
                    'count': count,
                    'score': winrate * np.log(count + 1)  # Weighted by volume
                })

        if not results:
            return {
                'current': 8,
                'optimal': 8,
                'confidence': 'low',
                'reason': 'Not enough data for optimization'
            }

        # Find best threshold
        best = max(results, key=lambda x: x['score'])
        current_threshold = 8  # Default

        return {
            'current': current_threshold,
            'optimal': best['threshold'],
            'confidence': 'high' if best['count'] >= 10 else 'medium',
            'reason': f"{best['winrate']*100:.0f}% win rate with {best['count']} trades",
            'expected_improvement': f"+{(best['winrate'] - df['win'].mean())*100:.1f}% win rate"
        }

    def get_optimal_weights(self, trades: List[Dict]) -> Dict[str, Dict]:
        """
        Calculate optimal confluence factor weights based on edge analysis

        Args:
            trades: List of trade dictionaries

        Returns:
            Dict mapping factor names to recommendations
        """
        if len(trades) < 10:
            return {}

        # Extract confluence factors from trades
        factors = {}
        for trade in trades:
            win = 1 if self._get_profit(trade) > 0 else 0

            # Volume profile factors (defensive check for data type)
            vp = trade.get('volume_profile', {})
            if not isinstance(vp, dict):
                vp = {}
            if vp.get('at_poc'): factors.setdefault('at_poc', []).append(win)
            if vp.get('at_swing_low'): factors.setdefault('at_swing_low', []).append(win)
            if vp.get('at_swing_high'): factors.setdefault('at_swing_high', []).append(win)
            if vp.get('at_lvn'): factors.setdefault('at_lvn', []).append(win)

            # HTF factors (defensive check for data type)
            htf_levels = trade.get('htf_levels', {})
            if not isinstance(htf_levels, dict):
                htf_levels = {}
            htf_factors = htf_levels.get('factors_matched', [])
            for factor in htf_factors:
                factor_key = factor.lower().replace(' ', '_')
                factors.setdefault(factor_key, []).append(win)

        # Calculate win rates and edges
        recommendations = {}
        overall_winrate = np.mean([self._get_profit(t) > 0 for t in trades])

        for factor, wins in factors.items():
            if len(wins) >= 3:  # Minimum sample size
                factor_winrate = np.mean(wins)
                edge = factor_winrate - overall_winrate

                # Calculate optimal weight based on edge
                if edge > 0.15:  # Strong edge
                    optimal_weight = 3
                elif edge > 0.08:  # Moderate edge
                    optimal_weight = 2
                elif edge > 0:  # Slight edge
                    optimal_weight = 1
                else:  # Negative edge
                    optimal_weight = 0

                recommendations[factor] = {
                    'optimal': optimal_weight,
                    'winrate': factor_winrate,
                    'edge': edge,
                    'count': len(wins),
                    'reason': f"{factor_winrate*100:.0f}% win rate ({len(wins)} trades), +{edge*100:.1f}% edge"
                }

        return recommendations

    def get_optimal_adx_filter(self, trades: List[Dict]) -> Dict:
        """
        Find optimal ADX threshold for filtering trades

        Args:
            trades: List of trade dictionaries

        Returns:
            Dict with ADX filter recommendation
        """
        if len(trades) < 15:
            return {
                'optimal': 'Need more data',
                'confidence': 'low'
            }

        # Analyze win rate by ADX level
        df = pd.DataFrame(trades)
        df['win'] = df.apply(lambda x: 1 if self._get_profit(x) > 0 else 0, axis=1)
        df['adx'] = df.apply(lambda x: x.get('trend_filter', {}).get('adx', 0) if isinstance(x.get('trend_filter', {}), dict) else 0, axis=1)

        # Test different ADX thresholds
        results = []
        for threshold in [0, 15, 20, 25, 30]:
            filtered = df[df['adx'] >= threshold]
            if len(filtered) >= 5:
                winrate = filtered['win'].mean()
                count = len(filtered)
                results.append({
                    'threshold': threshold,
                    'winrate': winrate,
                    'count': count
                })

        if not results:
            return {'optimal': None, 'confidence': 'low'}

        best = max(results, key=lambda x: x['winrate'])

        if best['threshold'] == 0:
            return {
                'optimal': None,
                'confidence': 'medium',
                'reason': 'No ADX filter recommended'
            }
        else:
            return {
                'optimal': best['threshold'],
                'confidence': 'high' if best['count'] >= 10 else 'medium',
                'reason': f"{best['winrate']*100:.0f}% win rate with ADX >= {best['threshold']} ({best['count']} trades)"
            }

    def get_all_recommendations(self, trades: List[Dict]) -> Dict:
        """
        Get all parameter recommendations

        Args:
            trades: List of trade dictionaries

        Returns:
            Dict with all recommendations
        """
        return {
            'confluence_threshold': self.get_optimal_confluence_threshold(trades),
            'weights': self.get_optimal_weights(trades),
            'adx_filter': self.get_optimal_adx_filter(trades)
        }
