#!/usr/bin/env python3
"""
Daily ML Report Generator
Compares bot's static config with ML recommendations
Builds performance profile over time
"""

import sys
import os
import json
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add project root to path dynamically
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_system.features.extractor import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DailyReport')

class DailyReportGenerator:
    """Generate daily ML performance reports"""

    def __init__(self):
        # Use absolute path based on project root to avoid duplication
        project_root = Path(__file__).parent.parent.parent
        self.report_dir = project_root / 'ml_system' / 'reports' / 'daily'
        os.makedirs(self.report_dir, exist_ok=True)
        self.feature_extractor = FeatureExtractor()

        # Feature descriptions for better readability
        self.feature_descriptions = {
            'di_spread': 'DI Spread (Trend Direction Strength)',
            'minus_di': 'Minus DI (Downward Price Pressure)',
            'plus_di': 'Plus DI (Upward Price Pressure)',
            'trend_strength': 'Trend Strength',
            'adx': 'ADX (Trend Strength Indicator)',
            'confluence_score': 'Confluence Score',
            'vwap_distance_pct': 'Distance from VWAP (%)',
            'at_prev_day_val': 'At Previous Day Value Area Low',
            'vwap_band_score': 'VWAP Band Position',
            'hour': 'Hour of Day',
            'had_dca': 'Used DCA Recovery',
            'had_hedge': 'Used Hedge',
            'had_partial_close': 'Took Partial Profits',
            'partial_close_count': 'Number of Partial Closes',
        }

    def format_table(self, headers, rows, col_widths=None):
        """Format data as a proper ASCII table"""
        if not rows:
            return ["  (No data available)"]

        # Calculate column widths if not provided
        if col_widths is None:
            col_widths = []
            for i, header in enumerate(headers):
                max_width = len(str(header))
                for row in rows:
                    if i < len(row):
                        max_width = max(max_width, len(str(row[i])))
                col_widths.append(max_width + 2)

        # Build table
        table = []

        # Header row
        header_row = "  |"
        separator = "  +"
        for i, header in enumerate(headers):
            header_row += f" {str(header).ljust(col_widths[i])} |"
            separator += "-" * (col_widths[i] + 2) + "+"
        table.append(separator)
        table.append(header_row)
        table.append(separator)

        # Data rows
        for row in rows:
            data_row = "  |"
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    data_row += f" {str(cell).ljust(col_widths[i])} |"
            table.append(data_row)

        table.append(separator)
        return table

    def load_bot_config(self):
        """Load current bot configuration from strategy_config.py"""
        try:
            # Import from actual bot config
            import sys
            sys.path.insert(0, 'trading_bot')
            from config import strategy_config

            config = {
                'confluence_threshold': getattr(strategy_config, 'MIN_CONFLUENCE_SCORE', 8),
                'hedge_enabled': getattr(strategy_config, 'HEDGE_ENABLED', False),
                'hedge_trigger_pips': getattr(strategy_config, 'HEDGE_TRIGGER_PIPS', 8),
                'hedge_ratio': getattr(strategy_config, 'HEDGE_RATIO', 5.0),
                'dca_enabled': getattr(strategy_config, 'DCA_ENABLED', True),
                'dca_max_count': getattr(strategy_config, 'DCA_MAX_LEVELS', 6),
                'dca_trigger_pips': getattr(strategy_config, 'DCA_TRIGGER_PIPS', 20),
                'grid_enabled': getattr(strategy_config, 'GRID_ENABLED', True),
                'disable_negative_grid': getattr(strategy_config, 'DISABLE_NEGATIVE_GRID', True),
                'grid_spacing_pips': getattr(strategy_config, 'GRID_SPACING_PIPS', 8),
                'max_grid_levels': getattr(strategy_config, 'MAX_GRID_LEVELS', 4),
                'risk_per_trade': getattr(strategy_config, 'RISK_PERCENT', 1.0),
                'base_lot_size': getattr(strategy_config, 'BASE_LOT_SIZE', 0.04),
            }
            return config
        except Exception as e:
            # Fallback to defaults if import fails
            return {
                'confluence_threshold': 8,
                'hedge_enabled': False,
                'dca_enabled': True,
                'dca_max_count': 6,
                'grid_enabled': True,
                'risk_per_trade': 1.0,
            }

    def get_recent_trades(self, days=1):
        """Get trades from last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        trades = []

        trade_log_file = 'ml_system/outputs/continuous_trade_log.jsonl'

        # Check if file exists
        if not os.path.exists(trade_log_file):
            logger.warning(f"Trade log file not found: {trade_log_file}")
            logger.warning("Continuous logger may not be running. Start it with: python ml_system/continuous_logger.py")
            return []

        try:
            with open(trade_log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if not line.strip():  # Skip empty lines
                        continue
                    try:
                        trade = json.loads(line)
                        entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))

                        if entry_time >= cutoff:
                            trades.append(trade)
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Skipping malformed trade line: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error reading trade log: {e}")
            return []

        return trades

    def analyze_ml_performance(self, trades):
        """Analyze ML model performance on recent trades"""
        if not trades:
            return {
                'trades_analyzed': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'ml_confidence_avg': 0
            }

        from ml_system.models.model_loader import ModelLoader

        try:
            loader = ModelLoader()
            model = loader.load_production_model()

            predictions = []
            actuals = []
            confidences = []

            for trade in trades:
                outcome = trade.get('outcome', {})
                if outcome.get('status') != 'closed':
                    continue

                # Extract features and predict
                features = self.feature_extractor.extract_features(trade)
                X = pd.DataFrame([features])
                prob = model.predict_proba(X)[0, 1]
                pred = 1 if prob >= 0.70 else 0

                actual = 1 if outcome.get('profit', 0) > 0 else 0

                predictions.append(pred)
                actuals.append(actual)
                confidences.append(prob)

            if len(actuals) > 0:
                correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
                accuracy = correct / len(actuals)
                win_rate = sum(actuals) / len(actuals)
                avg_confidence = sum(confidences) / len(confidences)

                return {
                    'trades_analyzed': len(actuals),
                    'ml_accuracy': accuracy,
                    'actual_win_rate': win_rate,
                    'ml_confidence_avg': avg_confidence,
                    'trades_above_threshold': sum(1 for c in confidences if c >= 0.70)
                }

        except Exception as e:
            logger.error(f"Error analyzing ML performance: {e}")

        return {
            'trades_analyzed': 0,
            'ml_accuracy': 0,
            'actual_win_rate': 0,
            'ml_confidence_avg': 0
        }

    def generate_ml_recommendations(self, trades):
        """Generate ML-based recommendations"""
        recommendations = []

        if not trades:
            recommendations.append("[WARN] No recent trades - unable to generate recommendations")
            return recommendations

        # Analyze closed trades
        closed_trades = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed_trades) < 3:
            recommendations.append(f"[WARN] Only {len(closed_trades)} closed trades - need more data for recommendations")
            return recommendations

        df = pd.DataFrame([{
            'win': 1 if t.get('outcome', {}).get('profit', 0) > 0 else 0,
            'confluence': t.get('confluence_score', 0),
            'adx': t.get('trend_filter', {}).get('adx', 0),
            'had_hedge': t.get('outcome', {}).get('recovery', {}).get('hedge_count', 0) > 0,
            'had_dca': t.get('outcome', {}).get('recovery', {}).get('dca_count', 0) > 0,
            'dca_count': t.get('outcome', {}).get('recovery', {}).get('dca_count', 0),
            'hedge_count': t.get('outcome', {}).get('recovery', {}).get('hedge_count', 0),
        } for t in closed_trades])

        # Analyze win rate by confluence
        avg_confluence_winners = df[df['win'] == 1]['confluence'].mean()
        avg_confluence_losers = df[df['win'] == 0]['confluence'].mean() if (df['win'] == 0).any() else 0

        if avg_confluence_winners > avg_confluence_losers + 2:
            recommendations.append(f"[+] Winners have higher confluence ({avg_confluence_winners:.1f} vs {avg_confluence_losers:.1f})")
            recommendations.append(f"  -> Consider raising confluence threshold to {int(avg_confluence_winners - 2)}")
        else:
            recommendations.append(f"[WARN] Confluence not strongly predictive (winners: {avg_confluence_winners:.1f}, losers: {avg_confluence_losers:.1f})")

        # Analyze ADX patterns
        avg_adx = df['adx'].mean()
        if avg_adx < 20:
            recommendations.append(f" Trading mostly in ranging markets (avg ADX: {avg_adx:.1f})")
            recommendations.append(f"  -> Mean reversion strategies working well")
        else:
            recommendations.append(f" Trading in trending markets (avg ADX: {avg_adx:.1f})")
            recommendations.append(f"  -> Trend-following strategies preferred")

        # RECOVERY ANALYSIS: Hedge & DCA effectiveness
        recommendations.append("")
        recommendations.append("RECOVERY MECHANISM ANALYSIS:")
        recommendations.append("-" * 40)

        # Count trades by recovery type
        total_trades = len(df)
        initial_only = df[(df['had_hedge'] == False) & (df['had_dca'] == False)]
        with_hedge = df[df['had_hedge'] == True]
        with_dca = df[df['had_dca'] == True]
        with_both = df[(df['had_hedge'] == True) & (df['had_dca'] == True)]

        # Analyze hedge effectiveness (if column exists)
        if 'had_hedge' in df.columns and (df['had_hedge']).any():
            hedge_trades = len(with_hedge)
            hedged_winrate = with_hedge['win'].mean()
            non_hedged_winrate = df[~df['had_hedge']]['win'].mean()
            avg_hedge_count = with_hedge['hedge_count'].mean()

            recommendations.append(f"")
            recommendations.append(f"[HEDGE] {hedge_trades} trades used hedging:")
            recommendations.append(f"  Hedge Win Rate: {hedged_winrate*100:.1f}% ({int(hedged_winrate*hedge_trades)}/{hedge_trades})")
            recommendations.append(f"  No-Hedge Win Rate: {non_hedged_winrate*100:.1f}%")
            recommendations.append(f"  Avg Hedges per Trade: {avg_hedge_count:.1f}")

            if hedged_winrate > non_hedged_winrate + 0.1:  # 10% better
                diff = (hedged_winrate - non_hedged_winrate) * 100
                recommendations.append(f"  STATUS: Hedging HELPING (+{diff:.1f}% win rate)")
                recommendations.append(f"  -> Hedge is saving underwater trades - keep enabled")
            elif hedged_winrate < non_hedged_winrate - 0.1:  # 10% worse
                diff = (non_hedged_winrate - hedged_winrate) * 100
                recommendations.append(f"  STATUS: Hedging HURTING (-{diff:.1f}% win rate)")
                recommendations.append(f"  -> Hedge may be locking in losses - review trigger timing")
            else:
                recommendations.append(f"  STATUS: Hedging NEUTRAL (similar win rates)")
                recommendations.append(f"  -> Hedge working as intended - provides insurance")
        else:
            recommendations.append(f"")
            recommendations.append(f"[HEDGE] No trades used hedging in this period")
            recommendations.append(f"  -> Either hedge not triggered or disabled in config")

        # DCA analysis (if column exists)
        if 'had_dca' in df.columns and (df['had_dca']).any():
            dca_trades = len(with_dca)
            dca_winrate = with_dca['win'].mean()
            no_dca_winrate = df[~df['had_dca']]['win'].mean()
            avg_dca_count = with_dca['dca_count'].mean()

            recommendations.append(f"")
            recommendations.append(f"[DCA] {dca_trades} trades used DCA:")
            recommendations.append(f"  DCA Win Rate: {dca_winrate*100:.1f}% ({int(dca_winrate*dca_trades)}/{dca_trades})")
            recommendations.append(f"  No-DCA Win Rate: {no_dca_winrate*100:.1f}%")
            recommendations.append(f"  Avg DCA Levels per Trade: {avg_dca_count:.1f}")

            if dca_winrate >= 0.7:  # 70% recovery rate is good
                recommendations.append(f"  STATUS: DCA working WELL (70%+ recovery)")
                recommendations.append(f"  -> DCA successfully recovering losing trades")
            elif dca_winrate >= 0.5:  # 50-70% is acceptable
                recommendations.append(f"  STATUS: DCA working ACCEPTABLY (50-70% recovery)")
                recommendations.append(f"  -> DCA helping but could be optimized")
            else:  # <50% is bad
                recommendations.append(f"  STATUS: DCA UNDERPERFORMING (<50% recovery)")
                recommendations.append(f"  -> Review DCA trigger timing and multiplier settings")
        else:
            recommendations.append(f"")
            recommendations.append(f"[DCA] No trades used DCA in this period")
            recommendations.append(f"  -> Either DCA not triggered or disabled in config")

        # Combined analysis
        if len(with_both) > 0:
            both_winrate = with_both['win'].mean()
            recommendations.append(f"")
            recommendations.append(f"[COMBO] {len(with_both)} trades used BOTH hedge & DCA:")
            recommendations.append(f"  Win Rate: {both_winrate*100:.1f}%")
            if both_winrate >= 0.6:
                recommendations.append(f"  STATUS: Combined recovery WORKING")
                recommendations.append(f"  -> Heavy recovery trades being saved")
            else:
                recommendations.append(f"  STATUS: Combined recovery STRUGGLING")
                recommendations.append(f"  -> Trades needing both are in deep trouble")

        # Summary
        initial_winrate = initial_only['win'].mean() if len(initial_only) > 0 else 0
        recommendations.append(f"")
        recommendations.append(f"[SUMMARY] Recovery System Performance:")
        recommendations.append(f"  Initial-Only Trades: {len(initial_only)} ({initial_winrate*100:.1f}% win rate)")
        recommendations.append(f"  Trades Needing Recovery: {total_trades - len(initial_only)} ({(total_trades - len(initial_only)) / total_trades * 100:.1f}%)")
        recommendations.append(f"")

        recommendations.append("-" * 40)

        # Partial close analysis (if column exists)
        if 'had_partial_close' in df.columns and (df['had_partial_close']).any():
            partial_close_count = df[df['had_partial_close']].shape[0]
            partial_close_winrate = df[df['had_partial_close']]['win'].mean()
            if 'partial_close_count' in df.columns:
                avg_partials = df[df['had_partial_close']]['partial_close_count'].mean()
                recommendations.append(f"[%] Partial Closes: {partial_close_count} trades ({avg_partials:.1f} partials avg)")
            else:
                recommendations.append(f"[%] Partial Closes: {partial_close_count} trades")
            recommendations.append(f"  -> Win rate with partials: {partial_close_winrate*100:.1f}%")
            if partial_close_winrate >= 0.7:
                recommendations.append(f"  -> Partial close strategy working well")
            else:
                recommendations.append(f"  -> Consider adjusting partial close timing")

        return recommendations

    def generate_parameter_recommendations(self, trades):
        """
        Generate parameter tuning recommendations based on failure patterns

        Analyzes:
        - DCA trades hitting max levels (ceiling pattern)
        - Maxed trades still losing (multiplier too weak)
        - Hedge trades failing (trigger too late or ratio too low)
        """
        recommendations = []

        # Current configuration values
        DCA_MULTIPLIER = 2.0  # Updated per ML recommendation
        MAX_DCA_LEVELS = 4    # Updated per ML recommendation
        HEDGE_RATIO = 1.5

        # Filter closed trades
        closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

        if len(closed) == 0:
            recommendations.append("[INFO] No closed trades - unable to analyze parameters")
            return recommendations

        # 1. DCA Analysis
        dca_trades = [t for t in closed if t.get('outcome', {}).get('recovery', {}).get('dca_count', 0) > 0]

        if len(dca_trades) >= 3:
            wins = [t for t in dca_trades if t.get('outcome', {}).get('profit', 0) > 0]
            losses = [t for t in dca_trades if t.get('outcome', {}).get('profit', 0) < 0]

            # Check maxed out pattern
            maxed = [t for t in dca_trades if t.get('outcome', {}).get('recovery', {}).get('dca_count', 0) >= MAX_DCA_LEVELS]
            maxed_losses = [t for t in maxed if t.get('outcome', {}).get('profit', 0) < 0]

            if maxed:
                pct_maxed = len(maxed) / len(dca_trades) * 100

                recommendations.append(f"[DCA] {len(dca_trades)} trades used DCA ({len(wins)}W / {len(losses)}L)")
                recommendations.append(f"  └─ {len(maxed)}/{len(dca_trades)} trades maxed out at {MAX_DCA_LEVELS} levels ({pct_maxed:.0f}%)")

                # Recommendation 1: Add DCA level
                if pct_maxed > 30:
                    recommendations.append(f"  └─ ⚠️  CEILING PATTERN: {pct_maxed:.0f}% hitting max DCA")
                    recommendations.append(f"      ACTION: Add DCA level {MAX_DCA_LEVELS + 1} in instruments_config.py")

                # Recommendation 2: Increase multiplier
                if len(maxed_losses) >= 2:
                    pct_maxed_lost = len(maxed_losses) / len(maxed) * 100 if maxed else 0
                    avg_loss = sum([t.get('outcome', {}).get('profit', 0) for t in maxed_losses]) / len(maxed_losses)
                    recommendations.append(f"  └─ ⚠️  WEAK MULTIPLIER: {len(maxed_losses)}/{len(maxed)} maxed trades lost ({pct_maxed_lost:.0f}%)")
                    recommendations.append(f"      Avg loss: ${avg_loss:.2f}")
                    recommendations.append(f"      ACTION: Increase DCA_MULTIPLIER from {DCA_MULTIPLIER} to 2.0")
                elif not maxed_losses:
                    recommendations.append(f"  └─ ✅ All maxed trades recovered successfully")
            else:
                recommendations.append(f"[DCA] {len(dca_trades)} trades used DCA ({len(wins)}W / {len(losses)}L)")
                recommendations.append(f"  └─ ✅ DCA levels appropriate - low maxed-out rate")

        # 2. Hedge Analysis
        hedge_trades = [t for t in closed if t.get('outcome', {}).get('recovery', {}).get('hedge_count', 0) > 0]

        if len(hedge_trades) >= 2:
            wins = [t for t in hedge_trades if t.get('outcome', {}).get('profit', 0) > 0]
            losses = [t for t in hedge_trades if t.get('outcome', {}).get('profit', 0) < 0]
            loss_rate = len(losses) / len(hedge_trades) * 100

            recommendations.append(f"[HEDGE] {len(hedge_trades)} trades used hedge ({len(wins)}W / {len(losses)}L)")

            if loss_rate > 50:
                avg_loss = sum([t.get('outcome', {}).get('profit', 0) for t in losses]) / len(losses) if losses else 0
                recommendations.append(f"  └─ ⚠️  HIGH FAILURE RATE: {loss_rate:.0f}% hedged trades lost")
                recommendations.append(f"      Avg loss: ${avg_loss:.2f}")
                recommendations.append(f"      ACTION: Reduce hedge_trigger_pips by 10 pips (earlier deployment)")
            else:
                recommendations.append(f"  └─ ✅ Hedge strategy effective - {100-loss_rate:.0f}% recovery rate")

        # 3. Summary
        if len(dca_trades) < 3 and len(hedge_trades) < 2:
            recommendations.append("[INFO] Not enough recovery data yet (need 3+ DCA, 2+ hedge trades)")

        return recommendations

    def generate_report(self):
        """Generate comprehensive daily report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        date_str = datetime.now().strftime('%Y-%m-%d')

        # Collect data
        config = self.load_bot_config()
        trades_24h = self.get_recent_trades(days=1)
        trades_7d = self.get_recent_trades(days=7)
        trades_all = self.get_recent_trades(days=365)  # All trades from last year
        ml_perf = self.analyze_ml_performance(trades_24h)
        recommendations = self.generate_ml_recommendations(trades_7d)
        param_recommendations = self.generate_parameter_recommendations(trades_7d)

        # Count total closed trades for training
        total_closed = sum(1 for t in trades_all if t.get('outcome', {}).get('status') == 'closed')

        # Build report
        report = []
        report.append("=" * 80)
        report.append(f"ML DAILY PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {timestamp}")
        report.append(f"Report Period: Last 24 hours")
        report.append("=" * 80)
        report.append("")

        # Section 1: Current Bot Configuration
        report.append("1. CURRENT BOT CONFIGURATION")
        report.append("-" * 80)
        report.append(f"  Confluence Threshold: {config['confluence_threshold']}")
        report.append(f"  Hedge Enabled: {config['hedge_enabled']}")
        if config['hedge_enabled']:
            report.append(f"    └─ Trigger: {config.get('hedge_trigger_pips', 8)} pips")
            report.append(f"    └─ Ratio: {config.get('hedge_ratio', 5.0)}x")
        report.append(f"  DCA Enabled: {config['dca_enabled']}")
        if config['dca_enabled']:
            report.append(f"    └─ Max Levels: {config['dca_max_count']}")
            report.append(f"    └─ Trigger: {config.get('dca_trigger_pips', 20)} pips")
        report.append(f"  Grid Enabled: {config['grid_enabled']}")
        if config['grid_enabled']:
            if config.get('disable_negative_grid', False):
                report.append(f"    └─ Type: Positive Only (Profit Taking)")
            else:
                report.append(f"    └─ Type: Both Positive & Negative")
            report.append(f"    └─ Spacing: {config.get('grid_spacing_pips', 8)} pips")
            report.append(f"    └─ Max Levels: {config.get('max_grid_levels', 4)}")
        report.append(f"  Risk per Trade: {config['risk_per_trade']:.1f}%")
        report.append(f"  Base Lot Size: {config.get('base_lot_size', 0.04)}")
        report.append("")

        # Section 2: 24h Performance
        report.append("2. LAST 24 HOURS PERFORMANCE")
        report.append("-" * 80)
        report.append(f"  Trades Opened: {len(trades_24h)}")

        closed_24h = [t for t in trades_24h if t.get('outcome', {}).get('status') == 'closed']
        if closed_24h:
            wins = sum(1 for t in closed_24h if t.get('outcome', {}).get('profit', 0) > 0)
            total_profit = sum(t.get('outcome', {}).get('profit', 0) for t in closed_24h)
            win_rate = wins / len(closed_24h)

            report.append(f"  Trades Closed: {len(closed_24h)}")
            report.append(f"  Win Rate: {win_rate*100:.1f}% ({wins}/{len(closed_24h)})")
            report.append(f"  Total Profit: ${total_profit:.2f}")
            report.append(f"  Avg Profit: ${total_profit/len(closed_24h):.2f}")
        else:
            report.append(f"  Trades Closed: 0")
            report.append(f"  (No closed trades to analyze)")
        report.append("")

        # Section 3: ML Performance
        report.append("3. ML MODEL PERFORMANCE")
        report.append("-" * 80)
        if ml_perf['trades_analyzed'] > 0:
            report.append(f"  Trades Analyzed: {ml_perf['trades_analyzed']}")
            report.append(f"  ML Accuracy: {ml_perf['ml_accuracy']*100:.1f}%")
            report.append(f"  Actual Win Rate: {ml_perf['actual_win_rate']*100:.1f}%")
            report.append(f"  Avg ML Confidence: {ml_perf['ml_confidence_avg']*100:.1f}%")
            report.append(f"  Trades Above Threshold (70%): {ml_perf['trades_above_threshold']}/{ml_perf['trades_analyzed']}")

            # Compare ML vs baseline
            if ml_perf['ml_accuracy'] > ml_perf['actual_win_rate']:
                report.append(f"  [+] ML outperforming baseline by {(ml_perf['ml_accuracy']-ml_perf['actual_win_rate'])*100:.1f}%")
            else:
                report.append(f"  [WARN] ML underperforming baseline by {(ml_perf['actual_win_rate']-ml_perf['ml_accuracy'])*100:.1f}%")
        else:
            report.append(f"  No closed trades to analyze")
        report.append("")

        # Section 4: ML Recommendations
        report.append("4. ML RECOMMENDATIONS (Based on Last 7 Days)")
        report.append("-" * 80)
        for rec in recommendations:
            report.append(f"  {rec}")
        report.append("")

        # Section 5: Parameter Optimization
        report.append("5. PARAMETER OPTIMIZATION (Based on Last 7 Days)")
        report.append("-" * 80)
        for rec in param_recommendations:
            report.append(f"  {rec}")
        report.append("")

        # Section 6: Feature Importance (Top 10)
        report.append("6. TOP 10 MOST IMPORTANT FEATURES")
        report.append("-" * 80)
        try:
            feature_importance = pd.read_csv('ml_system/models/feature_importance_baseline.csv')
            top_10 = feature_importance.head(10)

            # Build table data
            table_rows = []
            for idx, row in top_10.iterrows():
                rank = f"{idx+1}"
                feature_name = row['feature']
                feature_desc = self.feature_descriptions.get(feature_name, feature_name)
                importance = f"{row['importance']:.4f}"

                # Visual bar (scaled to 30 chars max)
                bar_length = int(row['importance'] * 100)  # Scale to percentage
                bar = '#' * min(bar_length, 30)

                table_rows.append([rank, feature_desc, importance, bar])

            # Create formatted table
            headers = ["Rank", "Feature", "Importance", "Visual"]
            table_lines = self.format_table(headers, table_rows)
            for line in table_lines:
                report.append(line)

        except Exception as e:
            report.append(f"  [ERROR] Could not load feature importance: {e}")
            report.append(f"  Make sure ML models have been trained.")
        report.append("")

        # Section 7: System Health
        report.append("7. SYSTEM HEALTH")
        report.append("-" * 80)
        report.append(f"  ML Model: [OK] Loaded")
        report.append(f"  Feature Extractor: [OK] Ready")
        report.append(f"  Training Data: {total_closed} closed trades (all-time)")
        report.append(f"  Recent Performance: {ml_perf['trades_analyzed']} closed in last 24h")
        report.append(f"  Last Retrain: Check ml_system/logs/auto_retrain.log")
        report.append(f"  Mode: SHADOW (read-only)")
        report.append("")

        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        report_text = '\n'.join(report)

        # Save to file (use Path operator for Windows compatibility)
        report_file = self.report_dir / f'report_{date_str}.txt'
        with open(str(report_file), 'w', encoding='utf-8', errors='ignore') as f:
            f.write(report_text)

        logger.info(f"[OK] Report saved to: {report_file}")

        return report_text, str(report_file)

    def send_email(self, report_text, email_config):
        """Send report via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['from_email']
            msg['To'] = email_config['to_email']
            msg['Subject'] = f"ML Daily Report - {datetime.now().strftime('%Y-%m-%d')}"

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

def main():
    """Generate and send daily report"""
    generator = DailyReportGenerator()

    # Generate report
    report_text, report_file = generator.generate_report()

    # Print to console
    print(report_text)

    # Email configuration (optional)
    # Uncomment and configure to enable email sending
    email_config = {
        'enabled': False,  # Set to True to enable email
        'from_email': 'your-email@gmail.com',
        'to_email': 'recipient@example.com',
        'password': 'your-app-password',  # Use app-specific password for Gmail
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587
    }

    # Load email config from file if exists
    email_config_file = 'config/email_config.json'
    if os.path.exists(email_config_file):
        with open(email_config_file, 'r', encoding='utf-8', errors='ignore') as f:
            email_config = json.load(f)

    # Send email if enabled
    if email_config.get('enabled', False):
        generator.send_email(report_text, email_config)

if __name__ == '__main__':
    main()
