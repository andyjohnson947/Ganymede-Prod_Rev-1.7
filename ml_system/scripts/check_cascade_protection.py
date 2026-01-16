#!/usr/bin/env python3
"""
Quick Cascade Protection Check

Run this anytime to see cascade protection effectiveness
and get threshold recommendations.

Usage:
    python ml_system/scripts/check_cascade_protection.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_system.analysis.cascade_analyzer import CascadeAnalyzer


def main():
    """Generate and display cascade protection analysis"""
    print("\n" + "="*70)
    print("QUICK CASCADE PROTECTION CHECK")
    print("="*70)
    print()

    analyzer = CascadeAnalyzer()

    # Parse events
    events = analyzer.parse_stop_out_log()

    if not events:
        print("✓ No stop-out events recorded yet")
        print()
        print("Cascade protection is active and will trigger when:")
        print("  1. First stop-out: Closes that stack")
        print("  2. Second stop-out within 30min: Closes ALL underwater stacks")
        print()
        print("Stop-outs are logged to: logs/stop_out_events.log")
        print()
        return

    # Generate full report
    report = analyzer.generate_report()
    print(report)

    # Quick summary at the end
    analysis = analyzer.analyze_stop_out_patterns(events)
    recommendations = analyzer.recommend_thresholds(analysis)

    if recommendations.get('has_recommendations'):
        changes_needed = sum(
            1 for rec in recommendations.get('settings', {}).values()
            if rec['current'] != rec['recommended']
        )

        if changes_needed > 0:
            print("\n" + "!"*70)
            print(f"⚠️  ACTION REQUIRED: {changes_needed} setting(s) need adjustment")
            print("!"*70)
            print("\nUpdate these in trading_bot/config/strategy_config.py")
            print("Then restart the bot to apply changes.")
        else:
            print("\n" + "="*70)
            print("✓ All cascade protection settings are optimal")
            print("="*70)

    print()


if __name__ == "__main__":
    main()
