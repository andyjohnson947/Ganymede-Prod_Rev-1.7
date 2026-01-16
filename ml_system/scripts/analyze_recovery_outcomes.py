"""
Analyze Recovery Strategy Outcomes

Tracks average P&L and max drawdown for:
1. Initial-only trades (no recovery)
2. DCA-only trades
3. Hedge-only trades
4. DCA+Hedge trades

Helps validate if stop loss limits are appropriate.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd


def load_trades() -> List[Dict]:
    """Load trades from continuous trade log"""
    log_file = Path("ml_system/outputs/continuous_trade_log.jsonl")

    if not log_file.exists():
        print("❌ No trade log found at ml_system/outputs/continuous_trade_log.jsonl")
        print("   Run the bot first to generate trades")
        return []

    trades = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))

    return trades


def analyze_recovery_performance(trades: List[Dict]):
    """Analyze performance by recovery type"""

    # Filter closed trades
    closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

    if len(closed) == 0:
        print("❌ No closed trades found")
        print("   Wait for positions to close before analyzing")
        return

    print(f"\n{'='*70}")
    print(f"RECOVERY STRATEGY PERFORMANCE ({len(closed)} closed trades)")
    print(f"{'='*70}\n")

    # Categorize by recovery type
    initial_only = []
    dca_only = []
    hedge_only = []
    dca_and_hedge = []

    for t in closed:
        recovery = t.get('outcome', {}).get('recovery', {})
        dca_count = recovery.get('dca_count', 0)
        hedge_count = recovery.get('hedge_count', 0)
        profit = t.get('outcome', {}).get('profit', 0)

        trade_data = {
            'ticket': t.get('ticket'),
            'symbol': t.get('symbol'),
            'profit': profit,
            'dca_count': dca_count,
            'hedge_count': hedge_count,
            'hold_time_hours': t.get('outcome', {}).get('duration_hours', 0),
        }

        if dca_count == 0 and hedge_count == 0:
            initial_only.append(trade_data)
        elif dca_count > 0 and hedge_count == 0:
            dca_only.append(trade_data)
        elif dca_count == 0 and hedge_count > 0:
            hedge_only.append(trade_data)
        else:
            dca_and_hedge.append(trade_data)

    # Analyze each category
    print("1. INITIAL-ONLY TRADES (No Recovery)")
    print("-" * 70)
    if initial_only:
        analyze_category(initial_only, "Initial")
    else:
        print("   No trades in this category yet\n")

    print("2. DCA-ONLY TRADES (No Hedge)")
    print("-" * 70)
    if dca_only:
        analyze_category(dca_only, "DCA-only")
        print("\n   Individual trades:")
        for t in dca_only:
            result = "WIN" if t['profit'] > 0 else "LOSS"
            print(f"     #{t['ticket']:8} | ${t['profit']:7.2f} | {t['dca_count']} DCA | {t['hold_time_hours']:.1f}h | {result}")
    else:
        print("   No trades in this category yet\n")

    print("\n3. HEDGE-ONLY TRADES (No DCA)")
    print("-" * 70)
    if hedge_only:
        analyze_category(hedge_only, "Hedge-only")
    else:
        print("   No trades in this category yet\n")

    print("\n4. DCA+HEDGE TRADES (Full Recovery)")
    print("-" * 70)
    if dca_and_hedge:
        analyze_category(dca_and_hedge, "DCA+Hedge")
        print("\n   Individual trades:")
        for t in dca_and_hedge:
            result = "WIN" if t['profit'] > 0 else "LOSS"
            print(f"     #{t['ticket']:8} | ${t['profit']:7.2f} | {t['dca_count']} DCA + {t['hedge_count']} Hedge | {t['hold_time_hours']:.1f}h | {result}")
    else:
        print("   No trades in this category yet\n")

    # Validate stop loss limits
    print(f"\n{'='*70}")
    print("STOP LOSS VALIDATION")
    print(f"{'='*70}\n")

    validate_stop_limits(dca_only, dca_and_hedge)


def analyze_category(trades: List[Dict], category_name: str):
    """Analyze a category of trades"""
    profits = [t['profit'] for t in trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]

    avg_profit = sum(profits) / len(profits)
    win_rate = len(wins) / len(profits) * 100

    print(f"   Total Trades: {len(trades)}")
    print(f"   Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"   Avg Profit: ${avg_profit:.2f}")

    if wins:
        print(f"   Avg Win: ${sum(wins)/len(wins):.2f}")
    if losses:
        print(f"   Avg Loss: ${sum(losses)/len(losses):.2f}")

    print(f"   Best Trade: ${max(profits):.2f}")
    print(f"   Worst Trade: ${min(profits):.2f}")

    # Calculate R:R
    if wins and losses:
        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        print(f"   R:R Ratio: {rr_ratio:.2f}:1")

    # Average DCA/hedge counts if applicable
    if any(t.get('dca_count', 0) > 0 for t in trades):
        avg_dca = sum([t.get('dca_count', 0) for t in trades]) / len(trades)
        print(f"   Avg DCA Levels: {avg_dca:.1f}")

    if any(t.get('hedge_count', 0) > 0 for t in trades):
        avg_hedge = sum([t.get('hedge_count', 0) for t in trades]) / len(trades)
        print(f"   Avg Hedges: {avg_hedge:.1f}")

    print()


def analyze_parameter_recommendations(trades: List[Dict]):
    """
    Analyze failure patterns and recommend parameter adjustments

    Detects:
    - DCA trades hitting max levels (ceiling pattern)
    - Maxed trades still losing (multiplier too weak)
    - Hedge trades failing (trigger too late or ratio too low)
    """
    # Current configuration values
    DCA_MULTIPLIER = 2.0  # Updated per ML recommendation
    MAX_DCA_LEVELS = 4    # Updated per ML recommendation
    HEDGE_RATIO = 1.5

    recommendations = []

    # Filter closed trades
    closed = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']

    if len(closed) == 0:
        return

    print(f"\n{'='*70}")
    print("PARAMETER OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*70}\n")

    # 1. DCA Analysis
    print("1. DCA EFFECTIVENESS")
    print("-" * 70)

    dca_trades = [t for t in closed if t.get('outcome', {}).get('recovery', {}).get('dca_count', 0) > 0]

    if len(dca_trades) >= 3:
        wins = [t for t in dca_trades if t.get('outcome', {}).get('profit', 0) > 0]
        losses = [t for t in dca_trades if t.get('outcome', {}).get('profit', 0) < 0]

        print(f"Total DCA trades: {len(dca_trades)} ({len(wins)}W / {len(losses)}L)")
        print()

        # Check maxed out pattern
        maxed = [t for t in dca_trades if t.get('outcome', {}).get('recovery', {}).get('dca_count', 0) >= MAX_DCA_LEVELS]
        maxed_losses = [t for t in maxed if t.get('outcome', {}).get('profit', 0) < 0]

        if maxed:
            pct_maxed = len(maxed) / len(dca_trades) * 100
            pct_maxed_lost = len(maxed_losses) / len(maxed) * 100 if maxed else 0

            print(f"⚠️  MAXED OUT PATTERN:")
            print(f"   {len(maxed)}/{len(dca_trades)} trades hit max DCA levels ({MAX_DCA_LEVELS}) = {pct_maxed:.0f}%")
            if maxed_losses:
                print(f"   {len(maxed_losses)}/{len(maxed)} maxed trades still LOST = {pct_maxed_lost:.0f}%")
            print()

            # Recommendation 1: Add DCA level
            if pct_maxed > 30:
                recommendations.append({
                    'priority': 'HIGH',
                    'type': 'DCA_LEVELS',
                    'message': f'ADD DCA LEVEL {MAX_DCA_LEVELS + 1}',
                    'reason': f'{pct_maxed:.0f}% of DCA trades hitting the ceiling',
                    'action': f'Change max_dca_levels from {MAX_DCA_LEVELS} to {MAX_DCA_LEVELS + 1} in instruments_config.py',
                    'impact': 'Allow more recovery attempts before stop-out'
                })

            # Recommendation 2: Increase multiplier
            if len(maxed_losses) >= 2:
                avg_loss = sum([t.get('outcome', {}).get('profit', 0) for t in maxed_losses]) / len(maxed_losses)
                recommendations.append({
                    'priority': 'HIGH',
                    'type': 'DCA_MULTIPLIER',
                    'message': 'INCREASE DCA_MULTIPLIER',
                    'reason': f'Maxed trades still losing (avg: ${avg_loss:.2f})',
                    'action': f'Change DCA_MULTIPLIER from {DCA_MULTIPLIER} to 2.0 in strategy_config.py',
                    'impact': 'Recover faster with larger volume scaling'
                })
        else:
            print("✅ DCA levels appropriate - low maxed-out rate")
            print()
    else:
        print(f"Not enough data yet (need 3+ DCA trades, have {len(dca_trades)})")
        print()

    # 2. Hedge Analysis
    print("2. HEDGE EFFECTIVENESS")
    print("-" * 70)

    hedge_trades = [t for t in closed if t.get('outcome', {}).get('recovery', {}).get('hedge_count', 0) > 0]

    if len(hedge_trades) >= 2:
        wins = [t for t in hedge_trades if t.get('outcome', {}).get('profit', 0) > 0]
        losses = [t for t in hedge_trades if t.get('outcome', {}).get('profit', 0) < 0]

        print(f"Total Hedge trades: {len(hedge_trades)} ({len(wins)}W / {len(losses)}L)")

        if losses:
            avg_loss = sum([t.get('outcome', {}).get('profit', 0) for t in losses]) / len(losses)
            loss_rate = len(losses) / len(hedge_trades) * 100

            if loss_rate > 50:
                print()
                print(f"⚠️  HIGH HEDGE FAILURE RATE: {loss_rate:.0f}%")
                print(f"   Avg loss: ${avg_loss:.2f}")
                print()

                recommendations.append({
                    'priority': 'MEDIUM',
                    'type': 'HEDGE_TRIGGER',
                    'message': 'HEDGE TRIGGER TOO LATE',
                    'reason': f'Hedges deployed but still losing ({loss_rate:.0f}% fail rate)',
                    'action': 'Reduce hedge_trigger_pips by 10 pips per instrument',
                    'impact': 'Earlier hedge deployment to lock losses sooner'
                })
        else:
            print("✅ Hedge strategy effective - all hedged trades recovered")
            print()
    else:
        print(f"Not enough data yet (need 2+ hedge trades, have {len(hedge_trades)})")
        print()

    # 3. Print Recommendations
    if recommendations:
        print("\n" + "="*70)
        print("ACTION ITEMS")
        print("="*70)
        print()

        for i, rec in enumerate(recommendations, 1):
            print(f"📊 RECOMMENDATION #{i}: {rec['message']} [{rec['priority']}]")
            print(f"   Reason: {rec['reason']}")
            print(f"   Action: {rec['action']}")
            print(f"   Impact: {rec['impact']}")
            print()
    else:
        print("\n✅ All parameters performing well - no adjustments needed")
        print()

    # 4. Current Configuration
    print("="*70)
    print("CURRENT CONFIGURATION:")
    print(f"  DCA_MULTIPLIER = {DCA_MULTIPLIER} (trading_bot/config/strategy_config.py:171)")
    print(f"  max_dca_levels = {MAX_DCA_LEVELS} (trading_bot/portfolio/instruments_config.py)")
    print(f"  HEDGE_RATIO = {HEDGE_RATIO} (trading_bot/config/strategy_config.py:145)")
    print(f"  hedge_trigger_pips = 45/55/50 per instrument (instruments_config.py)")
    print("="*70)
    print()


def validate_stop_limits(dca_only: List[Dict], dca_and_hedge: List[Dict]):
    """Validate if stop loss limits are appropriate"""

    DCA_ONLY_LIMIT = -25.0
    DCA_HEDGE_LIMIT = -50.0

    print("Current Stop Loss Limits:")
    print(f"  DCA-only: ${DCA_ONLY_LIMIT:.2f}")
    print(f"  DCA+Hedge: ${DCA_HEDGE_LIMIT:.2f}\n")

    # Analyze DCA-only
    if dca_only:
        dca_losses = [t['profit'] for t in dca_only if t['profit'] < 0]
        dca_wins = [t['profit'] for t in dca_only if t['profit'] > 0]

        print("DCA-ONLY Analysis:")
        print(f"  Total trades: {len(dca_only)} ({len(dca_wins)}W / {len(dca_losses)}L)")

        if dca_losses:
            worst_loss = min(dca_losses)
            avg_loss = sum(dca_losses) / len(dca_losses)

            print(f"  Worst loss: ${worst_loss:.2f}")
            print(f"  Avg loss: ${avg_loss:.2f}")

            # Check if stop would have triggered
            stopped_count = len([l for l in dca_losses if l <= DCA_ONLY_LIMIT])
            if stopped_count > 0:
                print(f"  ⚠️  Stop would trigger on {stopped_count}/{len(dca_losses)} losses")
                print(f"      Consider: Increase limit to ${min(dca_losses):.2f} or accept early exits")
            else:
                cushion = DCA_ONLY_LIMIT - worst_loss
                print(f"  ✅ Stop limit OK - ${cushion:.2f} cushion to worst loss")
        else:
            print(f"  ✅ No losses yet (100% win rate)")
    else:
        print("DCA-ONLY Analysis:")
        print("  No data yet - wait for DCA trades to close")

    print()

    # Analyze DCA+Hedge
    if dca_and_hedge:
        hedge_losses = [t['profit'] for t in dca_and_hedge if t['profit'] < 0]
        hedge_wins = [t['profit'] for t in dca_and_hedge if t['profit'] > 0]

        print("DCA+HEDGE Analysis:")
        print(f"  Total trades: {len(dca_and_hedge)} ({len(hedge_wins)}W / {len(hedge_losses)}L)")

        if hedge_losses:
            worst_loss = min(hedge_losses)
            avg_loss = sum(hedge_losses) / len(hedge_losses)

            print(f"  Worst loss: ${worst_loss:.2f}")
            print(f"  Avg loss: ${avg_loss:.2f}")

            # Check if stop would have triggered
            stopped_count = len([l for l in hedge_losses if l <= DCA_HEDGE_LIMIT])
            if stopped_count > 0:
                print(f"  ⚠️  Stop would trigger on {stopped_count}/{len(hedge_losses)} losses")
                print(f"      Consider: Increase limit to ${min(hedge_losses):.2f} or accept early exits")
            else:
                cushion = DCA_HEDGE_LIMIT - worst_loss
                print(f"  ✅ Stop limit OK - ${cushion:.2f} cushion to worst loss")
        else:
            print(f"  ✅ No losses yet (100% win rate)")
    else:
        print("DCA+HEDGE Analysis:")
        print("  No data yet - wait for hedged trades to close")

    print()


def main():
    """Main analysis"""
    print("\n" + "="*70)
    print("RECOVERY STRATEGY OUTCOME ANALYSIS")
    print("="*70)

    trades = load_trades()

    if not trades:
        return

    analyze_recovery_performance(trades)

    # Analyze parameter optimization recommendations
    analyze_parameter_recommendations(trades)

    print("\n" + "="*70)
    print("GENERAL RECOMMENDATIONS")
    print("="*70)
    print()
    print("1. Run this script weekly to validate stop loss limits")
    print("2. If stops trigger too often, increase limits gradually")
    print("3. If worst losses exceed limits significantly, decrease limits")
    print("4. Target: Stop should catch outliers (worst 5-10% of trades)")
    print("5. Monitor parameter recommendations and adjust when patterns emerge")
    print()


if __name__ == "__main__":
    main()
