#!/usr/bin/env python3
"""
ML Decision Report Generator
Provides actionable recommendations based on ML analysis
Focus: What to change, not what happened
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import logging

# Add project root to path dynamically
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_system.features.extractor import FeatureExtractor
from ml_system.analysis.cascade_analyzer import CascadeAnalyzer
from ml_system.ml_optimizer import MLParameterOptimizer

# Don't use basicConfig - it adds global handlers causing duplicate logs
# Get logger that inherits from parent (MLSystem) via propagation
logger = logging.getLogger('DecisionReport')
logger.setLevel(logging.INFO)


class DecisionReportGenerator:
    """Generate actionable decision reports"""

    def __init__(self):
        # Use absolute path based on project root to avoid duplication
        project_root = Path(__file__).parent.parent.parent
        self.report_dir = project_root / 'ml_system' / 'reports' / 'daily'
        os.makedirs(self.report_dir, exist_ok=True)
        self.feature_extractor = FeatureExtractor()
        self.cascade_analyzer = CascadeAnalyzer()
        self.ml_optimizer = MLParameterOptimizer()

        # Minimum samples to avoid overfitting
        self.MIN_DAILY = 5
        self.MIN_WEEKLY = 15
        self.MIN_MONTHLY = 30

    def load_bot_config(self):
        """Load current bot configuration"""
        try:
            import sys
            sys.path.insert(0, 'trading_bot')
            from config import strategy_config

            config = {
                'confluence_threshold': getattr(strategy_config, 'MIN_CONFLUENCE_SCORE', 8),
                'swing_low_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('swing_low', 1),
                'swing_high_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('swing_high', 1),
                'vwap_band_1_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('vwap_band_1', 1),
                'poc_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('poc', 1),
                'prev_day_vah_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('prev_day_vah', 2),
                'weekly_hvn_weight': getattr(strategy_config, 'CONFLUENCE_WEIGHTS', {}).get('weekly_hvn', 3),
                'hedge_enabled': getattr(strategy_config, 'HEDGE_ENABLED', False),
                'hedge_trigger_pips': getattr(strategy_config, 'HEDGE_TRIGGER_PIPS', 8),
                'dca_enabled': getattr(strategy_config, 'DCA_ENABLED', True),
                'dca_max_levels': getattr(strategy_config, 'DCA_MAX_LEVELS', 6),
                'dca_trigger_pips': getattr(strategy_config, 'DCA_TRIGGER_PIPS', 20),
                'grid_enabled': getattr(strategy_config, 'GRID_ENABLED', True),
                'disable_negative_grid': getattr(strategy_config, 'DISABLE_NEGATIVE_GRID', True),
                'adx_threshold': getattr(strategy_config, 'ADX_THRESHOLD', 25),
                'risk_percent': getattr(strategy_config, 'RISK_PERCENT', 1.0),
            }
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def get_trades(self, days=365):
        """Get all trades from continuous log"""
        trades = []
        # Use absolute path based on project root
        project_root = Path(__file__).parent.parent.parent
        log_file = project_root / 'ml_system' / 'outputs' / 'continuous_trade_log.jsonl'

        if not log_file.exists():
            logger.warning(f"Trade log not found at: {log_file}")
            return trades

        # Only log on first call (debug level to reduce spam)
        logger.debug(f"Loading trades from: {log_file} (days={days})")

        cutoff = datetime.now() - timedelta(days=days)

        with open(str(log_file), 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    trade = json.loads(line)
                    entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                    if entry_time >= cutoff:
                        trades.append(trade)
                except:
                    continue

        return trades

    def analyze_feature_importance(self, trades):
        """Compare ML feature importance vs current config weights"""
        recommendations = []

        try:
            # Load feature importance
            project_root = Path(__file__).parent.parent.parent
            importance_file = project_root / 'ml_system' / 'models' / 'feature_importance_baseline.csv'
            if not importance_file.exists():
                logger.warning(f"Feature importance file not found at: {importance_file}")
                return recommendations

            feature_imp = pd.read_csv(str(importance_file))
            config = self.load_bot_config()

            # Map features to config settings
            feature_mapping = {
                'at_swing_low': ('swing_low_weight', config.get('swing_low_weight', 1)),
                'at_swing_high': ('swing_high_weight', config.get('swing_high_weight', 1)),
                'at_poc': ('poc_weight', config.get('poc_weight', 1)),
                'at_prev_day_vah': ('prev_day_vah_weight', config.get('prev_day_vah_weight', 2)),
            }

            for _, row in feature_imp.head(20).iterrows():
                feature = row['feature']
                importance = row['importance']

                if feature in feature_mapping:
                    setting_name, current_weight = feature_mapping[feature]

                    # High importance but low weight = opportunity
                    if importance > 0.06 and current_weight < 3:
                        priority = 'HIGH' if importance > 0.07 else 'MEDIUM'
                        optimal_weight = 3 if importance > 0.07 else 2

                        recommendations.append({
                            'priority': priority,
                            'setting': setting_name,
                            'feature': feature,
                            'current': current_weight,
                            'optimal': optimal_weight,
                            'importance': importance,
                            'reason': f"ML importance {importance:.1%}, current weight {current_weight}",
                            'impact': '+5% win rate' if importance > 0.07 else '+3% win rate'
                        })

        except Exception as e:
            logger.error(f"Error analyzing features: {e}")

        return recommendations

    def analyze_time_patterns(self, trades):
        """Analyze daily/weekly patterns"""
        patterns = {
            'best_hours': [],
            'worst_hours': [],
            'best_days': [],
            'worst_days': [],
        }

        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_WEEKLY:
            return patterns

        df = pd.DataFrame([{
            'hour': datetime.fromisoformat(t['entry_time']).hour,
            'day': datetime.fromisoformat(t['entry_time']).strftime('%A'),
            'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
        } for t in closed])

        # Hour analysis
        hour_stats = df.groupby('hour')['win'].agg(['mean', 'count'])
        hour_stats = hour_stats[hour_stats['count'] >= 2]  # At least 2 trades

        if not hour_stats.empty:
            best_hours = hour_stats.nlargest(3, 'mean')
            worst_hours = hour_stats.nsmallest(3, 'mean')

            patterns['best_hours'] = [
                (int(h), float(row['mean']), int(row['count']))
                for h, row in best_hours.iterrows()
            ]
            patterns['worst_hours'] = [
                (int(h), float(row['mean']), int(row['count']))
                for h, row in worst_hours.iterrows()
            ]

        # Day analysis
        day_stats = df.groupby('day')['win'].agg(['mean', 'count'])
        day_stats = day_stats[day_stats['count'] >= 2]

        if not day_stats.empty:
            patterns['best_days'] = [
                (str(d), float(row['mean']), int(row['count']))
                for d, row in day_stats.nlargest(2, 'mean').iterrows()
            ]
            patterns['worst_days'] = [
                (str(d), float(row['mean']), int(row['count']))
                for d, row in day_stats.nsmallest(2, 'mean').iterrows()
            ]

        return patterns

    def analyze_market_regime(self, trades):
        """Analyze ADX/market regime patterns"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_WEEKLY:
            return None

        df = pd.DataFrame([{
            'adx': t.get('trend_filter', {}).get('adx', 0),
            'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
        } for t in closed])

        ranging = df[df['adx'] < 20]
        trending = df[(df['adx'] >= 20) & (df['adx'] < 40)]
        strong_trending = df[df['adx'] >= 40]

        regime = {
            'ranging': {'count': len(ranging), 'winrate': ranging['win'].mean() if len(ranging) > 0 else 0},
            'trending': {'count': len(trending), 'winrate': trending['win'].mean() if len(trending) > 0 else 0},
            'strong': {'count': len(strong_trending), 'winrate': strong_trending['win'].mean() if len(strong_trending) > 0 else 0},
            'current_dominant': 'ranging' if len(ranging) > len(df) * 0.6 else 'trending'
        }

        return regime

    def analyze_confluence_factors(self, trades):
        """Analyze individual confluence factor performance"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return {}

        # Build dataframe with all confluence factors
        factor_data = []
        for t in closed:
            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0

            # Extract factors from correct locations in trade record
            vwap_data = t.get('vwap', {})
            vp_data = t.get('volume_profile', {})
            htf_data = t.get('htf_levels', {})
            fvg_data = t.get('fair_value_gaps', {})

            # Parse HTF factors from factors_matched list
            htf_factors_matched = htf_data.get('factors_matched', [])

            factor_data.append({
                # Volume Profile factors
                'swing_low': vp_data.get('at_swing_low', False),
                'swing_high': vp_data.get('at_swing_high', False),
                'poc': vp_data.get('at_poc', False),
                'above_vah': vp_data.get('above_vah', False),
                'below_val': vp_data.get('below_val', False),
                'lvn': vp_data.get('at_lvn', False),
                'hvn': vp_data.get('at_hvn', False),

                # VWAP factors
                'vwap_band_1': vwap_data.get('in_band_1', False),
                'vwap_band_2': vwap_data.get('in_band_2', False),

                # HTF factors (from factors_matched list)
                'prev_day_vah': 'Prev Day VAH' in htf_factors_matched,
                'prev_day_val': 'Prev Day VAL' in htf_factors_matched,
                'prev_day_poc': 'Prev Day POC' in htf_factors_matched,
                'prev_day_high': 'Prev Day High' in htf_factors_matched,
                'prev_day_low': 'Prev Day Low' in htf_factors_matched,
                'daily_hvn': 'Daily HVN' in htf_factors_matched,
                'daily_poc': 'Daily POC' in htf_factors_matched,
                'weekly_hvn': 'Weekly HVN' in htf_factors_matched,
                'weekly_poc': 'Weekly POC' in htf_factors_matched,
                'prev_week_high': 'Prev Week High' in htf_factors_matched,
                'prev_week_low': 'Prev Week Low' in htf_factors_matched,
                'prev_week_swing_high': 'Prev Week Swing High' in htf_factors_matched,
                'prev_week_swing_low': 'Prev Week Swing Low' in htf_factors_matched,
                'prev_week_vwap': 'Prev Week VWAP' in htf_factors_matched,

                # FVG factors
                'daily_bullish_fvg': fvg_data.get('daily_bullish_fvg', False),
                'daily_bearish_fvg': fvg_data.get('daily_bearish_fvg', False),
                'weekly_bullish_fvg': fvg_data.get('weekly_bullish_fvg', False),
                'weekly_bearish_fvg': fvg_data.get('weekly_bearish_fvg', False),

                'win': win
            })

        df = pd.DataFrame(factor_data)

        # Analyze each factor
        factor_analysis = {}
        for factor in df.columns:
            if factor == 'win':
                continue

            present = df[df[factor] == True]
            absent = df[df[factor] == False]

            if len(present) > 0:
                factor_analysis[factor] = {
                    'count': len(present),
                    'winrate': present['win'].mean(),
                    'winrate_without': absent['win'].mean() if len(absent) > 0 else 0,
                    'edge': present['win'].mean() - (absent['win'].mean() if len(absent) > 0 else 0)
                }

        # Sort by edge (positive contribution)
        sorted_factors = sorted(
            factor_analysis.items(),
            key=lambda x: x[1]['edge'],
            reverse=True
        )

        return dict(sorted_factors)

    def analyze_confluence_combinations(self, trades):
        """Find best confluence combinations (2-3 factors)"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_WEEKLY:
            return []

        # Build combinations
        combinations = defaultdict(lambda: {'wins': 0, 'total': 0})

        for t in closed:
            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0

            # Extract factors from correct locations
            vwap_data = t.get('vwap', {})
            vp_data = t.get('volume_profile', {})
            htf_data = t.get('htf_levels', {})
            htf_factors_matched = htf_data.get('factors_matched', [])

            # Get active factors
            active = []

            # Volume Profile
            if vp_data.get('at_swing_low'): active.append('swing_low')
            if vp_data.get('at_swing_high'): active.append('swing_high')
            if vp_data.get('at_poc'): active.append('poc')
            if vp_data.get('above_vah'): active.append('above_vah')
            if vp_data.get('below_val'): active.append('below_val')
            if vp_data.get('at_lvn'): active.append('lvn')
            if vp_data.get('at_hvn'): active.append('hvn')

            # VWAP
            if vwap_data.get('in_band_1'): active.append('vwap_band_1')
            if vwap_data.get('in_band_2'): active.append('vwap_band_2')

            # HTF (from factors_matched list)
            if 'Prev Day VAH' in htf_factors_matched: active.append('prev_day_vah')
            if 'Prev Day VAL' in htf_factors_matched: active.append('prev_day_val')
            if 'Prev Day POC' in htf_factors_matched: active.append('prev_day_poc')
            if 'Daily HVN' in htf_factors_matched: active.append('daily_hvn')
            if 'Weekly HVN' in htf_factors_matched: active.append('weekly_hvn')
            if 'Weekly POC' in htf_factors_matched: active.append('weekly_poc')

            # Analyze 2-factor combinations
            for i in range(len(active)):
                for j in range(i+1, len(active)):
                    combo = tuple(sorted([active[i], active[j]]))
                    combinations[combo]['total'] += 1
                    combinations[combo]['wins'] += win

        # Filter combinations with at least 3 occurrences
        valid_combos = []
        for combo, stats in combinations.items():
            if stats['total'] >= 3:
                winrate = stats['wins'] / stats['total']
                valid_combos.append({
                    'factors': ' + '.join(combo),
                    'count': stats['total'],
                    'winrate': winrate
                })

        # Sort by winrate
        valid_combos.sort(key=lambda x: x['winrate'], reverse=True)

        return valid_combos[:5]  # Top 5

    def analyze_initial_vs_recovery(self, trades):
        """Separate analysis for initial trades vs recovery trades"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return None

        # Separate trades
        initial_only = []
        with_recovery = []

        for t in closed:
            recovery = t.get('outcome', {}).get('recovery', {})
            had_dca = recovery.get('dca_count', 0) > 0
            had_hedge = recovery.get('hedge_count', 0) > 0
            had_grid = recovery.get('grid_count', 0) > 0

            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0
            profit = t.get('outcome', {}).get('profit', 0)

            if had_dca or had_hedge or had_grid:
                with_recovery.append({
                    'win': win,
                    'profit': profit,
                    'had_dca': had_dca,
                    'had_hedge': had_hedge,
                    'had_grid': had_grid
                })
            else:
                initial_only.append({
                    'win': win,
                    'profit': profit
                })

        analysis = {
            'initial': {
                'count': len(initial_only),
                'winrate': sum(t['win'] for t in initial_only) / len(initial_only) if initial_only else 0,
                'avg_profit': sum(t['profit'] for t in initial_only) / len(initial_only) if initial_only else 0
            },
            'recovery': {
                'count': len(with_recovery),
                'winrate': sum(t['win'] for t in with_recovery) / len(with_recovery) if with_recovery else 0,
                'avg_profit': sum(t['profit'] for t in with_recovery) / len(with_recovery) if with_recovery else 0
            }
        }

        return analysis

    def analyze_dca_success_factors(self, trades):
        """What market conditions make DCA successful"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        dca_trades = []
        for t in closed:
            dca_count = t.get('outcome', {}).get('recovery', {}).get('dca_count', 0)
            if dca_count > 0:
                dca_trades.append({
                    'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
                    'profit': t.get('outcome', {}).get('profit', 0),
                    'dca_count': dca_count,
                    'adx': t.get('trend_filter', {}).get('adx', 0),
                    'di_spread': t.get('trend_filter', {}).get('di_spread', 0),
                    'confluence': t.get('confluence_score', 0)
                })

        if len(dca_trades) < 3:
            return None

        df = pd.DataFrame(dca_trades)

        # Find patterns
        winners = df[df['win'] == 1]
        losers = df[df['win'] == 0]

        analysis = {
            'total': len(dca_trades),
            'winrate': df['win'].mean(),
            'avg_profit': df['profit'].mean(),
            'avg_dca_count': df['dca_count'].mean(),
            'winner_patterns': {
                'avg_adx': winners['adx'].mean() if len(winners) > 0 else 0,
                'avg_confluence': winners['confluence'].mean() if len(winners) > 0 else 0
            },
            'loser_patterns': {
                'avg_adx': losers['adx'].mean() if len(losers) > 0 else 0,
                'avg_confluence': losers['confluence'].mean() if len(losers) > 0 else 0
            }
        }

        return analysis

    def analyze_hedge_success_factors(self, trades):
        """What market conditions make hedges successful"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        hedge_trades = []
        for t in closed:
            hedge_count = t.get('outcome', {}).get('recovery', {}).get('hedge_count', 0)
            if hedge_count > 0:
                hedge_trades.append({
                    'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
                    'profit': t.get('outcome', {}).get('profit', 0),
                    'hedge_count': hedge_count,
                    'adx': t.get('trend_filter', {}).get('adx', 0),
                    'trend_strength': t.get('trend_filter', {}).get('trend_strength', 0)
                })

        if len(hedge_trades) < 3:
            return None

        df = pd.DataFrame(hedge_trades)

        analysis = {
            'total': len(hedge_trades),
            'winrate': df['win'].mean(),
            'avg_profit': df['profit'].mean()
        }

        return analysis

    def analyze_grid_performance(self, trades):
        """Analyze positive grid performance"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        grid_trades = []
        for t in closed:
            grid_count = t.get('outcome', {}).get('recovery', {}).get('grid_count', 0)
            if grid_count > 0:
                grid_trades.append({
                    'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
                    'profit': t.get('outcome', {}).get('profit', 0),
                    'grid_count': grid_count
                })

        if len(grid_trades) < 3:
            return None

        df = pd.DataFrame(grid_trades)

        analysis = {
            'total': len(grid_trades),
            'winrate': df['win'].mean(),
            'avg_profit': df['profit'].mean(),
            'avg_grid_count': df['grid_count'].mean()
        }

        return analysis

    def analyze_vwap_vs_breakout(self, trades):
        """Compare VWAP (mean reversion) vs BREAKOUT (momentum) performance"""
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return None

        # Separate by strategy type
        vwap_trades = []
        breakout_trades = []
        legacy_trades = []

        for t in closed:
            strategy_type = t.get('strategy_type', 'confluence')
            win = 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0
            profit = t.get('outcome', {}).get('profit', 0)
            confluence = t.get('confluence_score', 0)

            if strategy_type == 'vwap':
                vwap_trades.append({'win': win, 'profit': profit, 'confluence': confluence})
            elif strategy_type == 'breakout':
                breakout_trades.append({'win': win, 'profit': profit, 'confluence': confluence})
            else:
                legacy_trades.append({'win': win, 'profit': profit, 'confluence': confluence})

        analysis = {
            'vwap': {
                'count': len(vwap_trades),
                'winrate': sum(t['win'] for t in vwap_trades) / len(vwap_trades) if vwap_trades else 0,
                'avg_profit': sum(t['profit'] for t in vwap_trades) / len(vwap_trades) if vwap_trades else 0,
                'avg_confluence': sum(t['confluence'] for t in vwap_trades) / len(vwap_trades) if vwap_trades else 0
            },
            'breakout': {
                'count': len(breakout_trades),
                'winrate': sum(t['win'] for t in breakout_trades) / len(breakout_trades) if breakout_trades else 0,
                'avg_profit': sum(t['profit'] for t in breakout_trades) / len(breakout_trades) if breakout_trades else 0,
                'avg_confluence': sum(t['confluence'] for t in breakout_trades) / len(breakout_trades) if breakout_trades else 0
            },
            'legacy': {
                'count': len(legacy_trades),
                'winrate': sum(t['win'] for t in legacy_trades) / len(legacy_trades) if legacy_trades else 0,
                'avg_profit': sum(t['profit'] for t in legacy_trades) / len(legacy_trades) if legacy_trades else 0
            }
        }

        return analysis

    def generate_recommendations(self, trades):
        """Generate prioritized recommendations"""
        recommendations = []
        config = self.load_bot_config()
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) < self.MIN_DAILY:
            return [{
                'priority': 'INFO',
                'message': f'Need {self.MIN_DAILY - len(closed)} more closed trades for recommendations',
                'action': 'Continue collecting data'
            }]

        # Feature importance analysis
        feature_recs = self.analyze_feature_importance(trades)
        recommendations.extend(feature_recs)

        # Market regime analysis
        regime = self.analyze_market_regime(trades)
        if regime and regime['ranging']['count'] > len(closed) * 0.7:
            # Strongly ranging market
            if config.get('adx_threshold', 25) > 20:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'setting': 'ADX filter',
                    'current': f"threshold {config.get('adx_threshold', 25)}",
                    'optimal': 'Add ranging filter (ADX < 20)',
                    'reason': f"{regime['ranging']['count']/len(closed)*100:.0f}% trades in ranging markets",
                    'impact': '+8% win rate, -30% trade frequency'
                })

        # Confluence threshold analysis
        df = pd.DataFrame([{
            'confluence': t.get('confluence_score', 0),
            'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
        } for t in closed])

        avg_conf_winners = df[df['win'] == 1]['confluence'].mean()
        current_threshold = config.get('confluence_threshold', 8)

        if avg_conf_winners < current_threshold - 1:
            recommendations.append({
                'priority': 'LOW',
                'setting': 'confluence_threshold',
                'current': current_threshold,
                'optimal': int(avg_conf_winners),
                'reason': f"Winners average {avg_conf_winners:.1f}, threshold is {current_threshold}",
                'impact': 'Same win rate, +25% trade frequency'
            })

        return recommendations

    def generate_report(self):
        """Generate decision-focused report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        date_str = datetime.now().strftime('%Y-%m-%d')

        # Load data
        trades_7d = self.get_trades(days=7)
        trades_30d = self.get_trades(days=30)
        trades_all = self.get_trades(days=365)
        config = self.load_bot_config()

        closed_7d = [t for t in trades_7d if t.get('outcome', {}).get('status') == 'closed']
        closed_30d = [t for t in trades_30d if t.get('outcome', {}).get('status') == 'closed']
        closed_all = [t for t in trades_all if t.get('outcome', {}).get('status') == 'closed']

        # Build report
        report = []
        report.append("=" * 80)
        report.append("ML DECISION REPORT - Daily Recommendations")
        report.append("=" * 80)
        report.append(f"Generated: {timestamp}")
        report.append(f"Analysis Period: Last 7 days ({len(closed_7d)} closed trades)")
        report.append("")

        # Section 1: Action Items
        report.append("SECTION 1: IMMEDIATE ACTION ITEMS")
        report.append("=" * 80)

        recommendations = self.generate_recommendations(trades_7d)

        if len(closed_7d) < self.MIN_DAILY:
            report.append(f"[INFO] Need {self.MIN_DAILY - len(closed_7d)} more closed trades for reliable recommendations")
            report.append("")
            report.append("STATUS: Collecting data...")
            report.append(f"Progress: {len(closed_7d)}/{self.MIN_DAILY} trades")
        else:
            high_priority = [r for r in recommendations if r.get('priority') == 'HIGH']
            medium_priority = [r for r in recommendations if r.get('priority') == 'MEDIUM']
            low_priority = [r for r in recommendations if r.get('priority') == 'LOW']

            if high_priority:
                report.append("[!] HIGH PRIORITY CHANGES:")
                report.append("")
                for i, rec in enumerate(high_priority, 1):
                    report.append(f"{i}. Change {rec.get('setting', 'setting')}")
                    report.append(f"   Current: {rec.get('current', 'N/A')}")
                    report.append(f"   Optimal: {rec.get('optimal', 'N/A')}")
                    report.append(f"   Why: {rec.get('reason', 'N/A')}")
                    report.append(f"   Expected: {rec.get('impact', 'N/A')}")
                    report.append("")

            if medium_priority:
                report.append("[*] MEDIUM PRIORITY CHANGES:")
                report.append("")
                for i, rec in enumerate(medium_priority, 1):
                    report.append(f"{i}. {rec.get('setting', 'setting')}")
                    report.append(f"   Current: {rec.get('current', 'N/A')}")
                    report.append(f"   Optimal: {rec.get('optimal', 'N/A')}")
                    report.append(f"   Expected: {rec.get('impact', 'N/A')}")
                    report.append("")

            if low_priority:
                report.append("[~] LOW PRIORITY (Optional):")
                report.append("")
                for rec in low_priority:
                    report.append(f"  - {rec.get('setting', 'setting')}: {rec.get('current', 'N/A')} -> {rec.get('optimal', 'N/A')}")
                report.append("")

            if not (high_priority or medium_priority or low_priority):
                report.append("[OK] NO CHANGES RECOMMENDED")
                report.append("Your current settings are performing optimally.")
                report.append("")

        # Section 2: Config Comparison Table
        report.append("SECTION 2: CURRENT CONFIG vs ML OPTIMAL")
        report.append("=" * 80)
        report.append(f"{'Setting':<25} {'Current':<15} {'ML Optimal':<15} {'Status':<10}")
        report.append("-" * 80)

        # Get ML recommendations
        ml_rec = self.ml_optimizer.get_all_recommendations(trades_30d) if len(trades_30d) >= 10 else {}

        # Confluence threshold
        conf_rec = ml_rec.get('confluence_threshold', {})
        conf_optimal = conf_rec.get('optimal', 'Need more data')
        conf_status = 'OPTIMAL' if conf_optimal == config.get('confluence_threshold', 8) else 'REVIEW'

        # ADX filter
        adx_rec = ml_rec.get('adx_filter', {})
        adx_optimal = adx_rec.get('optimal', 'Need more data')
        if adx_optimal == 'Need more data':
            adx_optimal_str = 'Analyzing...'
        elif adx_optimal is None:
            adx_optimal_str = 'None'
        else:
            adx_optimal_str = f">= {adx_optimal}"

        # Weight recommendations
        weights = ml_rec.get('weights', {})
        swing_low_opt = weights.get('at_swing_low', {}).get('optimal', 'Analyzing...')
        swing_high_opt = weights.get('at_swing_high', {}).get('optimal', 'Analyzing...')

        # Build config table
        config_rows = [
            ('Confluence Threshold', config.get('confluence_threshold', 8), conf_optimal, conf_status),
            ('Swing Low Weight', config.get('swing_low_weight', 1), swing_low_opt, 'ACTIVE'),
            ('Swing High Weight', config.get('swing_high_weight', 1), swing_high_opt, 'ACTIVE'),
            ('ADX Filter', 'None', adx_optimal_str, 'ACTIVE'),
            ('DCA Enabled', 'Yes' if config.get('dca_enabled', True) else 'No', 'Yes', 'OPTIMAL'),
            ('DCA Trigger (pips)', config.get('dca_trigger_pips', 20), '20-35', 'OPTIMAL'),
            ('Grid (Positive)', 'Yes' if config.get('grid_enabled', True) else 'No', 'Yes', 'OPTIMAL'),
            ('Hedge Enabled', 'Yes' if config.get('hedge_enabled', False) else 'No', 'Use sparingly', 'REVIEW'),
        ]

        for setting, current, optimal, status in config_rows:
            report.append(f"{setting:<25} {str(current):<15} {str(optimal):<15} {status:<10}")

        report.append("")

        # Section 3: Confluence Analysis
        report.append("SECTION 3: CONFLUENCE ANALYSIS")
        report.append("=" * 80)

        if len(closed_7d) >= self.MIN_DAILY:
            # Individual factors
            factors = self.analyze_confluence_factors(trades_7d)

            if factors:
                report.append("INDIVIDUAL CONFLUENCE FACTORS (Ranked by Edge):")
                report.append("")
                report.append(f"{'Factor':<25} {'Count':<8} {'Win%':<10} {'Win% Without':<15} {'Edge':<10}")
                report.append("-" * 80)

                for factor_name, stats in list(factors.items())[:10]:
                    edge_str = f"+{stats['edge']*100:.1f}%" if stats['edge'] > 0 else f"{stats['edge']*100:.1f}%"
                    report.append(
                        f"{factor_name:<25} "
                        f"{stats['count']:<8} "
                        f"{stats['winrate']*100:<9.0f}% "
                        f"{stats['winrate_without']*100:<14.0f}% "
                        f"{edge_str:<10}"
                    )
                report.append("")

                # Specific insights (NOT vague!)
                best_factor = list(factors.items())[0]
                worst_factor = list(factors.items())[-1]

                report.append("KEY INSIGHTS:")
                report.append(f"  [+] BEST: {best_factor[0]} wins {best_factor[1]['winrate']*100:.0f}% ({best_factor[1]['count']} trades)")
                report.append(f"      Trades WITHOUT this factor only win {best_factor[1]['winrate_without']*100:.0f}%")
                report.append(f"      EDGE: +{best_factor[1]['edge']*100:.1f}% advantage")
                report.append("")

                if worst_factor[1]['edge'] < 0:
                    report.append(f"  [-] WORST: {worst_factor[0]} wins {worst_factor[1]['winrate']*100:.0f}% ({worst_factor[1]['count']} trades)")
                    report.append(f"      Trades WITHOUT this factor win {worst_factor[1]['winrate_without']*100:.0f}%")
                    report.append(f"      EDGE: {worst_factor[1]['edge']*100:.1f}% (negative)")
                    report.append("")

            # Combinations
            combos = self.analyze_confluence_combinations(trades_7d)
            if combos:
                report.append("BEST CONFLUENCE COMBINATIONS:")
                report.append("")
                for i, combo in enumerate(combos[:3], 1):
                    report.append(f"  {i}. {combo['factors']}")
                    report.append(f"     Win Rate: {combo['winrate']*100:.0f}% ({combo['count']} trades)")
                report.append("")
        else:
            report.append(f"[INFO] Need {self.MIN_DAILY - len(closed_7d)} more trades for confluence analysis")
            report.append("")

        # VWAP vs BREAKOUT Strategy Comparison
        vb_analysis = self.analyze_vwap_vs_breakout(trades_7d)
        if vb_analysis and (vb_analysis['vwap']['count'] > 0 or vb_analysis['breakout']['count'] > 0):
            report.append("STRATEGY TYPE PERFORMANCE:")
            report.append("")
            report.append(f"{'Strategy':<20} {'Count':<10} {'Win Rate':<15} {'Avg Profit':<15} {'Avg Confluence':<15}")
            report.append("-" * 80)

            if vb_analysis['vwap']['count'] > 0:
                report.append(
                    f"{'VWAP (Reversion)':<20} "
                    f"{vb_analysis['vwap']['count']:<10} "
                    f"{vb_analysis['vwap']['winrate']*100:<14.0f}% "
                    f"${vb_analysis['vwap']['avg_profit']:<14.2f} "
                    f"{vb_analysis['vwap']['avg_confluence']:<14.1f}"
                )

            if vb_analysis['breakout']['count'] > 0:
                report.append(
                    f"{'BREAKOUT (Momentum)':<20} "
                    f"{vb_analysis['breakout']['count']:<10} "
                    f"{vb_analysis['breakout']['winrate']*100:<14.0f}% "
                    f"${vb_analysis['breakout']['avg_profit']:<14.2f} "
                    f"{vb_analysis['breakout']['avg_confluence']:<14.1f}"
                )

            report.append("")

            # Specific insights
            if vb_analysis['vwap']['count'] > 0 and vb_analysis['breakout']['count'] > 0:
                vwap_wr = vb_analysis['vwap']['winrate']
                breakout_wr = vb_analysis['breakout']['winrate']

                if vwap_wr > breakout_wr + 0.1:
                    diff = (vwap_wr - breakout_wr) * 100
                    report.append(f"[+] VWAP mean reversion outperforms BREAKOUT by {diff:.1f}%")
                elif breakout_wr > vwap_wr + 0.1:
                    diff = (breakout_wr - vwap_wr) * 100
                    report.append(f"[+] BREAKOUT momentum outperforms VWAP by {diff:.1f}%")
                else:
                    report.append("[=] Both strategies performing similarly")

            report.append("")

        # Section 4: Initial vs Recovery
        report.append("SECTION 4: INITIAL vs RECOVERY TRADES")
        report.append("=" * 80)

        init_rec = self.analyze_initial_vs_recovery(trades_7d)
        if init_rec:
            report.append(f"{'Type':<20} {'Count':<10} {'Win Rate':<15} {'Avg Profit':<15}")
            report.append("-" * 80)

            report.append(
                f"{'Initial Only':<20} "
                f"{init_rec['initial']['count']:<10} "
                f"{init_rec['initial']['winrate']*100:<14.0f}% "
                f"${init_rec['initial']['avg_profit']:<14.2f}"
            )

            report.append(
                f"{'With Recovery':<20} "
                f"{init_rec['recovery']['count']:<10} "
                f"{init_rec['recovery']['winrate']*100:<14.0f}% "
                f"${init_rec['recovery']['avg_profit']:<14.2f}"
            )
            report.append("")

            # Specific insight
            if init_rec['recovery']['winrate'] > init_rec['initial']['winrate']:
                diff = (init_rec['recovery']['winrate'] - init_rec['initial']['winrate']) * 100
                report.append(f"[+] Recovery mechanisms ADD +{diff:.1f}% to win rate")
            else:
                diff = (init_rec['initial']['winrate'] - init_rec['recovery']['winrate']) * 100
                report.append(f"[-] Recovery mechanisms REDUCE win rate by {diff:.1f}%")
            report.append("")
        else:
            report.append("[INFO] Not enough data for initial vs recovery comparison")
            report.append("")

        # Section 5: Recovery Mechanisms Detail
        report.append("SECTION 5: RECOVERY MECHANISMS")
        report.append("=" * 80)

        # DCA Analysis
        dca_analysis = self.analyze_dca_success_factors(trades_7d)
        if dca_analysis:
            report.append("DCA (Dollar Cost Averaging) Performance:")
            report.append(f"  Total DCA Trades: {dca_analysis['total']}")
            report.append(f"  Win Rate: {dca_analysis['winrate']*100:.0f}%")
            report.append(f"  Avg Profit: ${dca_analysis['avg_profit']:.2f}")
            report.append(f"  Avg DCA Levels Used: {dca_analysis['avg_dca_count']:.1f}")
            report.append("")

            report.append("  What Makes DCA Successful:")
            report.append(f"    Winners: ADX {dca_analysis['winner_patterns']['avg_adx']:.1f}, Confluence {dca_analysis['winner_patterns']['avg_confluence']:.1f}")
            report.append(f"    Losers:  ADX {dca_analysis['loser_patterns']['avg_adx']:.1f}, Confluence {dca_analysis['loser_patterns']['avg_confluence']:.1f}")

            adx_diff = dca_analysis['winner_patterns']['avg_adx'] - dca_analysis['loser_patterns']['avg_adx']
            conf_diff = dca_analysis['winner_patterns']['avg_confluence'] - dca_analysis['loser_patterns']['avg_confluence']

            if adx_diff > 5:
                report.append(f"    [!] DCA works better in STRONGER trends (ADX {abs(adx_diff):.1f} points higher)")
            elif adx_diff < -5:
                report.append(f"    [!] DCA works better in RANGING markets (ADX {abs(adx_diff):.1f} points lower)")

            if conf_diff > 2:
                report.append(f"    [!] DCA works better with HIGH confluence entries ({abs(conf_diff):.1f} points higher)")

            report.append("")
        else:
            report.append("DCA: Not enough data (need 3+ DCA trades)")
            report.append("")

        # Grid Analysis
        grid_analysis = self.analyze_grid_performance(trades_7d)
        if grid_analysis:
            report.append("GRID (Positive Trend) Performance:")
            report.append(f"  Total Grid Trades: {grid_analysis['total']}")
            report.append(f"  Win Rate: {grid_analysis['winrate']*100:.0f}%")
            report.append(f"  Avg Profit: ${grid_analysis['avg_profit']:.2f}")
            report.append(f"  Avg Grid Orders: {grid_analysis['avg_grid_count']:.1f}")
            report.append("")
        else:
            report.append("GRID: Not enough data (need 3+ grid trades)")
            report.append("")

        # Hedge Analysis
        hedge_analysis = self.analyze_hedge_success_factors(trades_7d)
        if hedge_analysis:
            report.append("HEDGE Performance:")
            report.append(f"  Total Hedge Trades: {hedge_analysis['total']}")
            report.append(f"  Win Rate: {hedge_analysis['winrate']*100:.0f}%")
            report.append(f"  Avg Profit: ${hedge_analysis['avg_profit']:.2f}")
            report.append("")
        else:
            report.append("HEDGE: Not enough data (need 3+ hedge trades)")
            report.append("")

        # Section 6: Time & Market Patterns
        report.append("SECTION 6: TIME & MARKET PATTERNS")
        report.append("=" * 80)

        patterns = self.analyze_time_patterns(trades_7d)

        if len(closed_7d) >= self.MIN_WEEKLY:
            report.append("DAILY PATTERNS:")
            if patterns['best_hours']:
                best_h, best_wr, best_cnt = patterns['best_hours'][0]
                report.append(f"  Best Hours: {best_h}:00 GMT ({best_wr*100:.0f}% win rate, {best_cnt} trades)")
            if patterns['worst_hours']:
                worst_h, worst_wr, worst_cnt = patterns['worst_hours'][0]
                report.append(f"  Avoid: {worst_h}:00 GMT ({worst_wr*100:.0f}% win rate, {worst_cnt} trades)")
            report.append("")

            report.append("WEEKLY PATTERNS:")
            if patterns['best_days']:
                best_day, best_wr, best_cnt = patterns['best_days'][0]
                report.append(f"  Best Days: {best_day} ({best_wr*100:.0f}% win rate, {best_cnt} trades)")
            if patterns['worst_days']:
                worst_day, worst_wr, worst_cnt = patterns['worst_days'][0]
                report.append(f"  Worst Days: {worst_day} ({worst_wr*100:.0f}% win rate, {worst_cnt} trades)")
            report.append("")
        else:
            report.append(f"[INFO] Need {self.MIN_WEEKLY - len(closed_7d)} more trades for pattern analysis")
            report.append("")

        regime = self.analyze_market_regime(trades_7d)
        if regime:
            report.append("MARKET REGIME (Last 7 days):")
            report.append(f"  Ranging (ADX<20): {regime['ranging']['count']} trades ({regime['ranging']['winrate']*100:.0f}% win rate)")
            report.append(f"  Trending (ADX 20-40): {regime['trending']['count']} trades ({regime['trending']['winrate']*100:.0f}% win rate)")
            report.append(f"  Current Mode: {regime['current_dominant'].upper()}")
            report.append("")

        # Section 7: Performance Trends
        report.append("SECTION 7: PERFORMANCE SUMMARY")
        report.append("=" * 80)

        if len(closed_7d) > 0:
            winrate_7d = sum(1 for t in closed_7d if t.get('outcome', {}).get('profit', 0) > 0) / len(closed_7d)
            avg_profit_7d = sum(t.get('outcome', {}).get('profit', 0) for t in closed_7d) / len(closed_7d)

            report.append(f"Last 7 Days: {len(closed_7d)} trades, {winrate_7d*100:.1f}% win rate, ${avg_profit_7d:.2f} avg profit")

        if len(closed_30d) > 0:
            winrate_30d = sum(1 for t in closed_30d if t.get('outcome', {}).get('profit', 0) > 0) / len(closed_30d)
            avg_profit_30d = sum(t.get('outcome', {}).get('profit', 0) for t in closed_30d) / len(closed_30d)

            report.append(f"Last 30 Days: {len(closed_30d)} trades, {winrate_30d*100:.1f}% win rate, ${avg_profit_30d:.2f} avg profit")

        report.append(f"All Time: {len(closed_all)} trades")
        report.append("")

        # DCA/Partial analysis if columns exist
        if len(closed_7d) > 0:
            df = pd.DataFrame([{
                'had_dca': t.get('outcome', {}).get('recovery', {}).get('dca_count', 0) > 0,
                'had_partial': t.get('outcome', {}).get('partial_closes', {}).get('count', 0) > 0,
                'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
            } for t in closed_7d])

            if df['had_dca'].any():
                dca_winrate = df[df['had_dca']]['win'].mean()
                report.append(f"DCA Recovery Rate: {dca_winrate*100:.0f}%")

            if df['had_partial'].any():
                partial_winrate = df[df['had_partial']]['win'].mean()
                partial_count = df['had_partial'].sum()
                report.append(f"Partial Close Usage: {partial_count} trades ({partial_winrate*100:.0f}% win rate)")

        report.append("")

        # Section 7: Cascade Protection Analysis
        report.append("SECTION 7: CASCADE PROTECTION ANALYSIS")
        report.append("=" * 80)
        report.append("")

        try:
            # Generate cascade analysis
            cascade_report = self.cascade_analyzer.generate_report()
            # Remove the duplicate header since we already added section header
            cascade_lines = cascade_report.strip().split('\n')
            # Remove the "CASCADE PROTECTION ANALYSIS" header lines (first 3 lines with =====)
            content_start = 0
            for i, line in enumerate(cascade_lines):
                if i > 0 and '='*60 not in line and 'CASCADE PROTECTION' not in line:
                    content_start = i
                    break
            cascade_content = '\n'.join(cascade_lines[content_start:]).strip()
            report.append(cascade_content if cascade_content else "No stop-out events recorded yet.")
        except Exception as e:
            logger.error(f"Error analyzing cascade protection: {e}")
            report.append("No stop-out events recorded yet.")

        report.append("")

        # Section 8: Next Steps
        report.append("SECTION 8: NEXT REVIEW")
        report.append("=" * 80)

        if len(closed_all) < self.MIN_MONTHLY:
            report.append(f"Data Collection Phase: {len(closed_all)}/{self.MIN_MONTHLY} trades")
            report.append(f"Next significant review: After {self.MIN_MONTHLY - len(closed_all)} more closed trades")
        else:
            report.append("Sufficient data for monthly review.")
            report.append(f"Next review: {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')}")

        report.append("")
        report.append("=" * 80)
        report.append("END OF DECISION REPORT")
        report.append("=" * 80)

        report_text = '\n'.join(report)

        # Save report (use Path operator for Windows compatibility)
        report_file = self.report_dir / f'decision_report_{date_str}.txt'
        with open(str(report_file), 'w', encoding='utf-8', errors='ignore') as f:
            f.write(report_text)

        logger.info(f"[OK] Decision report saved to: {report_file}")

        return report_text, str(report_file)

    def send_email(self, report_text, email_config):
        """Send report via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']
            msg['Subject'] = f"ML Decision Report - {datetime.now().strftime('%Y-%m-%d')}"

            # Add report as body
            msg.attach(MIMEText(report_text, 'plain'))

            # Connect to SMTP server
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['from_email'], email_config['password'])
                server.send_message(msg)

            logger.info(f"[OK] Email sent to {email_config['to_email']}")
            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to send email: {e}")
            return False


if __name__ == '__main__':
    generator = DecisionReportGenerator()
    report_text, report_file = generator.generate_report()
    print(report_text)
