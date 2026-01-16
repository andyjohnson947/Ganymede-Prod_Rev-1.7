"""
Cascade Stop Protection Analyzer

Analyzes stop-out events to validate cascade protection effectiveness
and recommend optimal threshold tuning.
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd


class CascadeAnalyzer:
    """Analyze stop-out events and cascade protection effectiveness"""

    def __init__(self, log_file: str = "logs/stop_out_events.log"):
        """
        Initialize cascade analyzer

        Args:
            log_file: Path to stop-out events log
        """
        self.log_file = Path(log_file)

    def parse_stop_out_log(self) -> List[Dict]:
        """
        Parse stop-out events log file

        Returns:
            List of stop-out event dicts
        """
        if not self.log_file.exists():
            return []

        events = []

        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    # Format: 2026-01-12T10:15:23 | #12345 | EURUSD | $25.50 | ADX:32.5 | DCA-only
                    parts = line.strip().split(' | ')

                    timestamp = datetime.fromisoformat(parts[0])
                    ticket = int(parts[1].replace('#', ''))
                    symbol = parts[2]
                    loss = float(parts[3].replace('$', ''))

                    # Parse ADX (might be "N/A")
                    adx_str = parts[4].replace('ADX:', '')
                    adx = float(adx_str) if adx_str != 'N/A' else None

                    stack_type = parts[5]

                    events.append({
                        'timestamp': timestamp,
                        'ticket': ticket,
                        'symbol': symbol,
                        'loss': loss,
                        'adx': adx,
                        'stack_type': stack_type
                    })

                except (ValueError, IndexError) as e:
                    print(f"Warning: Failed to parse log line: {line.strip()} - {e}")
                    continue

        return events

    def analyze_stop_out_patterns(self, events: List[Dict], window_minutes: int = 30) -> Dict:
        """
        Analyze patterns in stop-out events

        Args:
            events: List of stop-out events
            window_minutes: Time window for cascade detection

        Returns:
            Dict with analysis results
        """
        if not events:
            return {
                'total_stops': 0,
                'has_data': False
            }

        df = pd.DataFrame(events)

        # Basic statistics
        total_stops = len(events)
        avg_loss = df['loss'].mean()
        max_loss = df['loss'].max()

        # ADX analysis (validate assumption: stops happen during trends)
        adx_events = df[df['adx'].notna()]
        avg_adx = adx_events['adx'].mean() if len(adx_events) > 0 else None
        high_adx_stops = len(adx_events[adx_events['adx'] >= 25]) if len(adx_events) > 0 else 0
        high_adx_pct = (high_adx_stops / len(adx_events) * 100) if len(adx_events) > 0 else 0

        # Stack type analysis
        stack_types = df['stack_type'].value_counts().to_dict()
        avg_loss_by_type = df.groupby('stack_type')['loss'].mean().to_dict()

        # Cascade detection simulation
        # Find clusters of stops within window_minutes
        cascades = []
        df_sorted = df.sort_values('timestamp')

        i = 0
        while i < len(df_sorted):
            current_time = df_sorted.iloc[i]['timestamp']
            window_end = current_time + timedelta(minutes=window_minutes)

            # Find all stops within window
            stops_in_window = df_sorted[
                (df_sorted['timestamp'] >= current_time) &
                (df_sorted['timestamp'] <= window_end)
            ]

            if len(stops_in_window) >= 2:
                # Cascade detected
                cascade_symbols = stops_in_window['symbol'].unique().tolist()
                cascade_adx = stops_in_window[stops_in_window['adx'].notna()]['adx'].mean()

                cascades.append({
                    'start_time': current_time,
                    'stop_count': len(stops_in_window),
                    'symbols': cascade_symbols,
                    'avg_adx': cascade_adx,
                    'total_loss': stops_in_window['loss'].sum()
                })

                # Skip to after this cascade window
                i += len(stops_in_window)
            else:
                i += 1

        # Symbol analysis
        stops_by_symbol = df['symbol'].value_counts().to_dict()

        return {
            'has_data': True,
            'total_stops': total_stops,
            'avg_loss': avg_loss,
            'max_loss': max_loss,
            'avg_adx': avg_adx,
            'high_adx_stops': high_adx_stops,
            'high_adx_percentage': high_adx_pct,
            'stack_types': stack_types,
            'avg_loss_by_type': avg_loss_by_type,
            'cascades_detected': len(cascades),
            'cascade_details': cascades,
            'stops_by_symbol': stops_by_symbol,
            'date_range': {
                'first': df['timestamp'].min(),
                'last': df['timestamp'].max()
            }
        }

    def recommend_thresholds(self, analysis: Dict) -> Dict:
        """
        Recommend optimal threshold settings based on analysis

        Args:
            analysis: Results from analyze_stop_out_patterns()

        Returns:
            Dict with recommendations
        """
        if not analysis.get('has_data'):
            return {
                'has_recommendations': False,
                'message': 'Insufficient data - need at least 5 stop-out events'
            }

        recommendations = {
            'has_recommendations': True,
            'settings': {}
        }

        # 1. CASCADE_ADX_THRESHOLD recommendation
        avg_adx = analysis.get('avg_adx')
        if avg_adx:
            if avg_adx >= 30:
                recommendations['settings']['CASCADE_ADX_THRESHOLD'] = {
                    'current': 25,
                    'recommended': 28,
                    'reason': f'Avg ADX at stops: {avg_adx:.1f} - can increase threshold'
                }
            elif avg_adx < 22:
                recommendations['settings']['CASCADE_ADX_THRESHOLD'] = {
                    'current': 25,
                    'recommended': 20,
                    'reason': f'Avg ADX at stops: {avg_adx:.1f} - should lower threshold'
                }
            else:
                recommendations['settings']['CASCADE_ADX_THRESHOLD'] = {
                    'current': 25,
                    'recommended': 25,
                    'reason': f'Current setting optimal (avg ADX: {avg_adx:.1f})'
                }

        # 2. DCA_ONLY_MAX_LOSS recommendation
        dca_losses = analysis.get('avg_loss_by_type', {}).get('DCA-only')
        if dca_losses:
            if dca_losses > 20:
                recommendations['settings']['DCA_ONLY_MAX_LOSS'] = {
                    'current': -25.0,
                    'recommended': -30.0,
                    'reason': f'Avg DCA-only loss: ${dca_losses:.2f} - stop triggers too early'
                }
            elif dca_losses < 15:
                recommendations['settings']['DCA_ONLY_MAX_LOSS'] = {
                    'current': -25.0,
                    'recommended': -20.0,
                    'reason': f'Avg DCA-only loss: ${dca_losses:.2f} - can tighten stop'
                }
            else:
                recommendations['settings']['DCA_ONLY_MAX_LOSS'] = {
                    'current': -25.0,
                    'recommended': -25.0,
                    'reason': f'Current setting optimal (avg loss: ${dca_losses:.2f})'
                }

        # 3. DCA_HEDGE_MAX_LOSS recommendation
        hedge_losses = analysis.get('avg_loss_by_type', {}).get('DCA+Hedge')
        if hedge_losses:
            if hedge_losses > 45:
                recommendations['settings']['DCA_HEDGE_MAX_LOSS'] = {
                    'current': -50.0,
                    'recommended': -60.0,
                    'reason': f'Avg DCA+Hedge loss: ${hedge_losses:.2f} - stop triggers too early'
                }
            elif hedge_losses < 35:
                recommendations['settings']['DCA_HEDGE_MAX_LOSS'] = {
                    'current': -50.0,
                    'recommended': -40.0,
                    'reason': f'Avg DCA+Hedge loss: ${hedge_losses:.2f} - can tighten stop'
                }
            else:
                recommendations['settings']['DCA_HEDGE_MAX_LOSS'] = {
                    'current': -50.0,
                    'recommended': -50.0,
                    'reason': f'Current setting optimal (avg loss: ${hedge_losses:.2f})'
                }

        # 4. CASCADE_THRESHOLD recommendation
        cascades = analysis.get('cascades_detected', 0)
        total_stops = analysis.get('total_stops', 0)

        if total_stops >= 10:  # Need enough data
            cascade_rate = cascades / total_stops if total_stops > 0 else 0

            if cascade_rate > 0.3:  # More than 30% of stops are in cascades
                recommendations['settings']['CASCADE_THRESHOLD'] = {
                    'current': 2,
                    'recommended': 3,
                    'reason': f'Cascade rate too high ({cascade_rate*100:.0f}%) - increase threshold'
                }
            elif cascade_rate < 0.1:  # Less than 10%
                recommendations['settings']['CASCADE_THRESHOLD'] = {
                    'current': 2,
                    'recommended': 2,
                    'reason': f'Cascade detection working well ({cascade_rate*100:.0f}%)'
                }
            else:
                recommendations['settings']['CASCADE_THRESHOLD'] = {
                    'current': 2,
                    'recommended': 2,
                    'reason': f'Current setting optimal (cascade rate: {cascade_rate*100:.0f}%)'
                }

        # 5. Validation of assumption: "Stops happen during trends"
        high_adx_pct = analysis.get('high_adx_percentage', 0)
        if high_adx_pct >= 70:
            recommendations['validation'] = {
                'stops_equals_trends': True,
                'confidence': 'HIGH',
                'message': f'{high_adx_pct:.0f}% of stops occurred with ADX >= 25 (trending)',
                'action': 'Cascade protection is correctly identifying trend transitions'
            }
        elif high_adx_pct >= 50:
            recommendations['validation'] = {
                'stops_equals_trends': True,
                'confidence': 'MEDIUM',
                'message': f'{high_adx_pct:.0f}% of stops occurred with ADX >= 25',
                'action': 'Cascade protection is mostly working as intended'
            }
        else:
            recommendations['validation'] = {
                'stops_equals_trends': False,
                'confidence': 'LOW',
                'message': f'Only {high_adx_pct:.0f}% of stops occurred with ADX >= 25',
                'action': 'Consider lowering CASCADE_ADX_THRESHOLD or stops not trend-related'
            }

        return recommendations

    def generate_report(self) -> str:
        """
        Generate cascade protection analysis report

        Returns:
            Formatted report string
        """
        events = self.parse_stop_out_log()

        if not events:
            return (
                "\n" + "="*70 + "\n"
                "CASCADE PROTECTION ANALYSIS\n"
                "="*70 + "\n"
                "No stop-out events recorded yet.\n"
                "Cascade protection will activate when stops occur.\n"
            )

        analysis = self.analyze_stop_out_patterns(events)
        recommendations = self.recommend_thresholds(analysis)

        report = []
        report.append("\n" + "="*70)
        report.append("CASCADE PROTECTION ANALYSIS")
        report.append("="*70)

        # Basic stats
        report.append(f"\nData Range: {analysis['date_range']['first'].strftime('%Y-%m-%d')} to "
                     f"{analysis['date_range']['last'].strftime('%Y-%m-%d')}")
        report.append(f"Total Stop-Outs: {analysis['total_stops']}")
        report.append(f"Average Loss: ${analysis['avg_loss']:.2f}")
        report.append(f"Max Loss: ${analysis['max_loss']:.2f}")

        # ADX validation
        if analysis['avg_adx']:
            report.append(f"\nADX Analysis:")
            report.append(f"  Average ADX at stops: {analysis['avg_adx']:.1f}")
            report.append(f"  High ADX stops (>= 25): {analysis['high_adx_stops']}/{len([e for e in events if e['adx']])} "
                         f"({analysis['high_adx_percentage']:.0f}%)")

            # Validation status
            if recommendations.get('validation'):
                val = recommendations['validation']
                report.append(f"\n  Validation: {val['message']}")
                report.append(f"  Confidence: {val['confidence']}")
                report.append(f"  → {val['action']}")

        # Stack type breakdown
        report.append(f"\nStop-Outs by Stack Type:")
        for stack_type, count in analysis['stack_types'].items():
            avg_loss = analysis['avg_loss_by_type'].get(stack_type, 0)
            report.append(f"  {stack_type}: {count} stops, avg loss ${avg_loss:.2f}")

        # Cascade detection
        report.append(f"\nCascade Events Detected: {analysis['cascades_detected']}")
        if analysis['cascades_detected'] > 0:
            report.append(f"  Recent cascades:")
            for cascade in analysis['cascade_details'][-3:]:  # Show last 3
                report.append(f"    {cascade['start_time'].strftime('%Y-%m-%d %H:%M')}: "
                             f"{cascade['stop_count']} stops, "
                             f"{', '.join(cascade['symbols'])}, "
                             f"${cascade['total_loss']:.2f} total")

        # Symbol breakdown
        report.append(f"\nStop-Outs by Symbol:")
        for symbol, count in sorted(analysis['stops_by_symbol'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / analysis['total_stops']) * 100
            report.append(f"  {symbol}: {count} ({pct:.0f}%)")

        # Recommendations
        if recommendations.get('has_recommendations'):
            report.append(f"\n" + "="*70)
            report.append("THRESHOLD RECOMMENDATIONS")
            report.append("="*70)

            for setting_name, rec in recommendations.get('settings', {}).items():
                if rec['current'] != rec['recommended']:
                    report.append(f"\n{setting_name}:")
                    report.append(f"  Current: {rec['current']}")
                    report.append(f"  Recommended: {rec['recommended']}")
                    report.append(f"  Reason: {rec['reason']}")
                    report.append(f"  ⚠️  Consider adjusting this setting")
                else:
                    report.append(f"\n{setting_name}: ✓ Optimal ({rec['current']})")

        else:
            report.append(f"\nInsufficient data for recommendations (need 5+ stops)")

        report.append("\n" + "="*70 + "\n")

        return "\n".join(report)
