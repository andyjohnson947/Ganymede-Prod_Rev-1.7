#!/usr/bin/env python3
"""
Hedge Effectiveness Analysis
Analyze historical data to determine if hedging helps and when to hedge
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path dynamically
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def analyze_hedge_effectiveness():
    """Analyze hedge effectiveness from historical trades"""

    print("=" * 80)
    print("HEDGE EFFECTIVENESS ANALYSIS")
    print("=" * 80)
    print()

    # Load trades
    trades = []
    log_file = project_root / 'ml_system' / 'outputs' / 'continuous_trade_log.jsonl'
    with open(log_file, 'r') as f:
        for line in f:
            trade = json.loads(line)
            outcome = trade.get('outcome', {})
            if outcome and outcome.get('status') == 'closed':
                trades.append(trade)

    print(f"Analyzing {len(trades)} closed trades...\n")

    # Categorize trades
    hedged_trades = []
    non_hedged_trades = []

    for trade in trades:
        outcome = trade['outcome']
        recovery = outcome.get('recovery', {})
        hedge_count = recovery.get('hedge_count', 0)

        trade_data = {
            'ticket': trade['ticket'],
            'entry_time': trade['entry_time'],
            'direction': trade['direction'],
            'entry_price': trade['entry_price'],
            'exit_price': outcome.get('exit_price'),
            'profit': outcome.get('profit', 0),
            'win': 1 if outcome.get('profit', 0) > 0 else 0,
            'confluence_score': trade.get('confluence_score', 0),
            'hold_hours': outcome.get('hold_hours', 0),
            'adx': trade.get('trend_filter', {}).get('adx', 0),
            'plus_di': trade.get('trend_filter', {}).get('plus_di', 0),
            'minus_di': trade.get('trend_filter', {}).get('minus_di', 0),
            'vwap_distance': trade.get('vwap', {}).get('distance_pct', 0),
            'at_swing_low': trade.get('volume_profile', {}).get('at_swing_low', False),
            'at_swing_high': trade.get('volume_profile', {}).get('at_swing_high', False),
            'hedge_count': hedge_count,
            'dca_count': recovery.get('dca_count', 0),
            'partial_count': outcome.get('partial_closes', {}).get('count', 0),
            'recovery_volume': recovery.get('total_recovery_volume', 0),
        }

        # Extract hedge details if available
        if hedge_count > 0:
            hedge_ratios = recovery.get('hedge_ratios', [])
            if hedge_ratios:
                trade_data['first_hedge_price'] = hedge_ratios[0].get('price')
                trade_data['first_hedge_volume'] = hedge_ratios[0].get('volume')
                # Calculate how far price moved before hedge
                entry_price = trade['entry_price']
                hedge_price = hedge_ratios[0].get('price')
                trade_data['price_move_to_hedge'] = abs((hedge_price - entry_price) / entry_price) * 100

        if hedge_count > 0:
            hedged_trades.append(trade_data)
        else:
            non_hedged_trades.append(trade_data)

    hedged_df = pd.DataFrame(hedged_trades)
    non_hedged_df = pd.DataFrame(non_hedged_trades)

    print("=" * 80)
    print("1. OVERALL COMPARISON: HEDGED vs NON-HEDGED")
    print("=" * 80)
    print()

    print(f"Hedged Trades: {len(hedged_df)}")
    print(f"  Win Rate: {hedged_df['win'].mean()*100:.1f}%")
    print(f"  Avg Profit: ${hedged_df['profit'].mean():.2f}")
    print(f"  Total Profit: ${hedged_df['profit'].sum():.2f}")
    print(f"  Avg Hold Time: {hedged_df['hold_hours'].mean():.1f} hours")
    print()

    print(f"Non-Hedged Trades: {len(non_hedged_df)}")
    print(f"  Win Rate: {non_hedged_df['win'].mean()*100:.1f}%")
    print(f"  Avg Profit: ${non_hedged_df['profit'].mean():.2f}")
    print(f"  Total Profit: ${non_hedged_df['profit'].sum():.2f}")
    print(f"  Avg Hold Time: {non_hedged_df['hold_hours'].mean():.1f} hours")
    print()

    # Calculate improvement
    win_rate_diff = hedged_df['win'].mean() - non_hedged_df['win'].mean()
    profit_diff = hedged_df['profit'].mean() - non_hedged_df['profit'].mean()

    print(" VERDICT:")
    if win_rate_diff > 0:
        print(f"  [OK] Hedged trades WIN RATE is {win_rate_diff*100:+.1f}% BETTER")
    else:
        print(f"  [ERROR] Hedged trades WIN RATE is {win_rate_diff*100:.1f}% WORSE")

    if profit_diff > 0:
        print(f"  [OK] Hedged trades PROFIT is ${profit_diff:+.2f} BETTER per trade")
    else:
        print(f"  [ERROR] Hedged trades PROFIT is ${profit_diff:.2f} WORSE per trade")

    print()
    print("=" * 80)
    print("2. WHEN WERE HEDGES TRIGGERED?")
    print("=" * 80)
    print()

    if len(hedged_df) > 0:
        print("Confluence Scores for Hedged Trades:")
        print(f"  Average: {hedged_df['confluence_score'].mean():.1f}")
        print(f"  Range: {hedged_df['confluence_score'].min():.0f} - {hedged_df['confluence_score'].max():.0f}")
        print()

        print("Market Conditions (ADX) for Hedged Trades:")
        print(f"  Average ADX: {hedged_df['adx'].mean():.1f}")
        avg_adx = hedged_df['adx'].mean()
        if avg_adx < 20:
            print(f"  -> Ranging markets (ADX < 20)")
        elif avg_adx <= 25:
            print(f"  -> Transitioning markets (20 <= ADX <= 25)")
        else:
            print(f"  -> Strong trending markets (ADX > 25)")
        print()

        print("Entry Positions for Hedged Trades:")
        print(f"  At swing lows: {hedged_df['at_swing_low'].sum()}/{len(hedged_df)}")
        print(f"  At swing highs: {hedged_df['at_swing_high'].sum()}/{len(hedged_df)}")
        print()

        if 'price_move_to_hedge' in hedged_df.columns and not hedged_df['price_move_to_hedge'].isna().all():
            print("Price Movement Before Hedge Triggered:")
            print(f"  Average: {hedged_df['price_move_to_hedge'].mean():.2f}% from entry")
            print(f"  Range: {hedged_df['price_move_to_hedge'].min():.2f}% - {hedged_df['price_move_to_hedge'].max():.2f}%")
            print()

    print("=" * 80)
    print("3. RECOVERY STACK PATTERNS")
    print("=" * 80)
    print()

    if len(hedged_df) > 0:
        print("Hedged Trades - Recovery Stack Composition:")
        print(f"  Also used DCA: {(hedged_df['dca_count'] > 0).sum()}/{len(hedged_df)} trades")
        print(f"  Also used Partials: {(hedged_df['partial_count'] > 0).sum()}/{len(hedged_df)} trades")
        print(f"  Avg DCA entries: {hedged_df['dca_count'].mean():.1f}")
        print(f"  Avg Recovery Volume: {hedged_df['recovery_volume'].mean():.3f} lots")
        print()

        # Check if hedge + other recovery = better outcomes
        hedge_with_dca = hedged_df[hedged_df['dca_count'] > 0]
        hedge_without_dca = hedged_df[hedged_df['dca_count'] == 0]

        if len(hedge_with_dca) > 0 and len(hedge_without_dca) > 0:
            print("Hedge + DCA Combination:")
            print(f"  Win rate WITH DCA: {hedge_with_dca['win'].mean()*100:.1f}%")
            print(f"  Win rate WITHOUT DCA: {hedge_without_dca['win'].mean()*100:.1f}%")
            print()

    print("=" * 80)
    print("4. RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Generate recommendations based on data
    recommendations = []

    if len(hedged_df) >= 2 and len(non_hedged_df) >= 2:
        if hedged_df['win'].mean() > non_hedged_df['win'].mean():
            recommendations.append("[OK] Historical data shows hedging IMPROVES win rate")
            recommendations.append(f"  Hedged: {hedged_df['win'].mean()*100:.1f}% vs Non-hedged: {non_hedged_df['win'].mean()*100:.1f}%")
        else:
            recommendations.append("[ERROR] Historical data shows hedging REDUCES win rate")
            recommendations.append(f"  Hedged: {hedged_df['win'].mean()*100:.1f}% vs Non-hedged: {non_hedged_df['win'].mean()*100:.1f}%")

        if hedged_df['profit'].mean() > non_hedged_df['profit'].mean():
            recommendations.append("[OK] Hedged trades are MORE profitable on average")
        else:
            recommendations.append("[ERROR] Hedged trades are LESS profitable on average")

    if len(hedged_df) > 0:
        avg_adx = hedged_df['adx'].mean()
        if avg_adx < 20:
            recommendations.append(f" Hedges were used in RANGING markets (avg ADX={avg_adx:.1f})")
            recommendations.append("   -> Consider: Hedging may help in choppy/ranging conditions")

        if 'price_move_to_hedge' in hedged_df.columns and not hedged_df['price_move_to_hedge'].isna().all():
            avg_move = hedged_df['price_move_to_hedge'].mean()
            recommendations.append(f" Hedges triggered after ~{avg_move:.2f}% price move from entry")
            recommendations.append(f"   -> Optimal hedge trigger: When price moves {avg_move:.2f}% against position")

        # Check if hedge helped trades recover
        hedged_wins = hedged_df[hedged_df['win'] == 1]
        if len(hedged_wins) > 0:
            recommendations.append(f" {len(hedged_wins)}/{len(hedged_df)} hedged trades recovered to profit")
            recommendations.append(f"   -> Hedge recovery rate: {len(hedged_wins)/len(hedged_df)*100:.1f}%")

    if not recommendations:
        recommendations.append("[WARN]  Insufficient data for strong recommendations")
        recommendations.append("   -> Need more trades to determine hedge effectiveness")

    for rec in recommendations:
        print(rec)

    print()
    print("=" * 80)
    print("5. ML FEATURE IMPORTANCE (Future Analysis)")
    print("=" * 80)
    print()
    print("Once you retrain models with 60 features (including hedge features),")
    print("the ML will identify:")
    print("  * Which confluence patterns benefit most from hedging")
    print("  * Optimal hedge count (1 vs 2 positions)")
    print("  * Market regimes where hedging helps/hurts")
    print("  * Interaction between hedging and other recovery mechanisms")
    print()
    print("📝 To enable ML hedge analysis:")
    print("  1. Continue collecting trades with hedge enabled/disabled")
    print("  2. Retrain model: python ml_system/scripts/hyperparameter_tuning.py")
    print("  3. Check feature importance: had_hedge, hedge_count")
    print()

    return {
        'hedged_win_rate': hedged_df['win'].mean() if len(hedged_df) > 0 else 0,
        'non_hedged_win_rate': non_hedged_df['win'].mean() if len(non_hedged_df) > 0 else 0,
        'hedged_profit': hedged_df['profit'].mean() if len(hedged_df) > 0 else 0,
        'non_hedged_profit': non_hedged_df['profit'].mean() if len(non_hedged_df) > 0 else 0,
        'verdict': 'HELPS' if (len(hedged_df) > 0 and hedged_df['win'].mean() > non_hedged_df['win'].mean()) else 'HURTS'
    }

if __name__ == '__main__':
    results = analyze_hedge_effectiveness()

    print("=" * 80)
    print(f"FINAL VERDICT: Hedging {results['verdict']}")
    print("=" * 80)
