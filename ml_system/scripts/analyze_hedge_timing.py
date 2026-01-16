#!/usr/bin/env python3
"""
Hedge Timing & Market Condition Analysis
Determine optimal timing and market conditions for hedge activation
"""

import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root to path dynamically
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def parse_datetime(dt_str):
    """Parse datetime string"""
    try:
        return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    except:
        return datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S')

def analyze_hedge_timing_and_conditions():
    """Analyze optimal hedge timing and market conditions"""

    print("=" * 80)
    print("HEDGE TIMING & MARKET CONDITION ANALYSIS")
    print("=" * 80)
    print()

    # Load trades
    trades = []
    with open('ml_system/outputs/continuous_trade_log.jsonl', 'r') as f:
        for line in f:
            trade = json.loads(line)
            outcome = trade.get('outcome', {})
            if outcome and outcome.get('status') == 'closed':
                trades.append(trade)

    print(f"Analyzing {len(trades)} closed trades...\n")

    # Analyze hedged trades in detail
    hedge_analysis = []

    for trade in trades:
        outcome = trade['outcome']
        recovery = outcome.get('recovery', {})
        hedge_count = recovery.get('hedge_count', 0)

        if hedge_count > 0:
            hedge_ratios = recovery.get('hedge_ratios', [])
            if hedge_ratios:
                # Get entry time
                entry_time = parse_datetime(trade['entry_time'])

                # Get first hedge time
                first_hedge = hedge_ratios[0]
                hedge_time = parse_datetime(first_hedge['time'])

                # Calculate time elapsed
                time_elapsed = (hedge_time - entry_time).total_seconds() / 60  # minutes

                # Get market conditions at entry
                trend = trade.get('trend_filter', {})
                adx = trend.get('adx', 0)
                plus_di = trend.get('plus_di', 0)
                minus_di = trend.get('minus_di', 0)
                trend_strength = abs(plus_di - minus_di)

                # Get VWAP position
                vwap = trade.get('vwap', {})
                vwap_distance = abs(vwap.get('distance_pct', 0))

                # Get price movement
                entry_price = trade['entry_price']
                hedge_price = first_hedge.get('price', entry_price)
                price_move_pct = abs((hedge_price - entry_price) / entry_price) * 100

                # Outcome
                profit = outcome.get('profit', 0)
                win = 1 if profit > 0 else 0

                hedge_analysis.append({
                    'ticket': trade['ticket'],
                    'direction': trade['direction'],
                    'time_to_hedge_minutes': time_elapsed,
                    'adx': adx,
                    'plus_di': plus_di,
                    'minus_di': minus_di,
                    'trend_strength': trend_strength,
                    'vwap_distance': vwap_distance,
                    'price_move_pct': price_move_pct,
                    'confluence_score': trade.get('confluence_score', 0),
                    'hedge_count': hedge_count,
                    'dca_count': recovery.get('dca_count', 0),
                    'profit': profit,
                    'win': win,
                    'at_swing_low': trade.get('volume_profile', {}).get('at_swing_low', False),
                    'at_swing_high': trade.get('volume_profile', {}).get('at_swing_high', False),
                })

    df = pd.DataFrame(hedge_analysis)

    if len(df) == 0:
        print("No hedged trades found with sufficient data")
        return

    print("=" * 80)
    print("1. TIMING ANALYSIS: When to Trigger Hedge")
    print("=" * 80)
    print()

    print(f"Time Elapsed Before Hedge Activation:")
    print(f"  Average: {df['time_to_hedge_minutes'].mean():.1f} minutes")
    print(f"  Median: {df['time_to_hedge_minutes'].median():.1f} minutes")
    print(f"  Range: {df['time_to_hedge_minutes'].min():.1f} - {df['time_to_hedge_minutes'].max():.1f} minutes")
    print()

    # Compare winners vs losers
    winners = df[df['win'] == 1]
    losers = df[df['win'] == 0]

    if len(winners) > 0 and len(losers) > 0:
        print("  Timing Comparison: Winners vs Losers")
        print(f"  Winners hedged after: {winners['time_to_hedge_minutes'].mean():.1f} min (avg)")
        print(f"  Losers hedged after: {losers['time_to_hedge_minutes'].mean():.1f} min (avg)")

        timing_diff = winners['time_to_hedge_minutes'].mean() - losers['time_to_hedge_minutes'].mean()
        if timing_diff > 0:
            print(f"  -> Winners waited {abs(timing_diff):.1f} min LONGER before hedging")
            print(f"   Recommendation: Wait at least {winners['time_to_hedge_minutes'].median():.0f} minutes")
        else:
            print(f"  -> Winners hedged {abs(timing_diff):.1f} min SOONER")
            print(f"   Recommendation: Hedge quickly if needed (within {winners['time_to_hedge_minutes'].max():.0f} min)")
    else:
        print(f" Recommendation: Consider hedging after {df['time_to_hedge_minutes'].median():.0f} minutes")

    print()

    print("=" * 80)
    print("2. MARKET CONDITION ANALYSIS: ADX & Trend Strength")
    print("=" * 80)
    print()

    print("Market Conditions When Hedge Was Triggered:")
    print(f"  ADX:")
    print(f"    Average: {df['adx'].mean():.1f}")
    print(f"    Range: {df['adx'].min():.1f} - {df['adx'].max():.1f}")

    # Categorize by ADX regime
    print(f"\n  ADX Regime Distribution:")
    ranging = (df['adx'] < 20).sum()
    trending = ((df['adx'] >= 20) & (df['adx'] <= 25)).sum()
    strong = (df['adx'] > 25).sum()
    print(f"    Ranging (ADX<20): {ranging}/{len(df)} trades")
    print(f"    Trending (20<=ADX<=25): {trending}/{len(df)} trades")
    print(f"    Strong (ADX>25): {strong}/{len(df)} trades")
    print()

    print(f"  Trend Strength (|plus_di - minus_di|):")
    print(f"    Average: {df['trend_strength'].mean():.1f}")
    print(f"    Range: {df['trend_strength'].min():.1f} - {df['trend_strength'].max():.1f}")
    print()

    # Compare winners vs losers on ADX
    if len(winners) > 0 and len(losers) > 0:
        print(" ADX Comparison: Winners vs Losers")
        print(f"  Winners ADX: {winners['adx'].mean():.1f}")
        print(f"  Losers ADX: {losers['adx'].mean():.1f}")

        if winners['adx'].mean() < losers['adx'].mean():
            print(f"  -> Winners had LOWER ADX (more ranging)")
            print(f"   Recommendation: Hedge works better in RANGING markets (ADX < {winners['adx'].max():.0f})")
        else:
            print(f"  -> Winners had HIGHER ADX (more trending)")
            print(f"   Recommendation: Hedge works better in TRENDING markets (ADX > {winners['adx'].min():.0f})")
    else:
        avg_adx = df['adx'].mean()
        if avg_adx < 20:
            print(f"   Hedges triggered in RANGING markets (avg ADX={avg_adx:.1f})")
            print(f"     -> Consider: Hedge is a ranging market recovery tool")
        else:
            print(f"   Hedges triggered in TRENDING markets (avg ADX={avg_adx:.1f})")

    print()

    print("=" * 80)
    print("3. PRICE MOVEMENT ANALYSIS")
    print("=" * 80)
    print()

    print("Price Movement Before Hedge:")
    print(f"  Average: {df['price_move_pct'].mean():.2f}%")
    print(f"  Range: {df['price_move_pct'].min():.2f}% - {df['price_move_pct'].max():.2f}%")
    print()

    if len(winners) > 0 and len(losers) > 0:
        print(" Price Move Comparison: Winners vs Losers")
        print(f"  Winners moved: {winners['price_move_pct'].mean():.2f}% before hedge")
        print(f"  Losers moved: {losers['price_move_pct'].mean():.2f}% before hedge")

        if winners['price_move_pct'].mean() < losers['price_move_pct'].mean():
            print(f"  -> Winners hedged EARLIER (less price movement)")
            print(f"   Recommendation: Hedge when price moves {winners['price_move_pct'].max():.1f}% or less")
        else:
            print(f"  -> Winners hedged LATER (more price movement)")
            print(f"   Recommendation: Wait for {winners['price_move_pct'].min():.1f}%+ move before hedging")

    print()

    print("=" * 80)
    print("4. VWAP DISTANCE ANALYSIS")
    print("=" * 80)
    print()

    print("VWAP Distance When Hedge Was Triggered:")
    print(f"  Average: {df['vwap_distance'].mean():.2f}%")
    print(f"  Range: {df['vwap_distance'].min():.2f}% - {df['vwap_distance'].max():.2f}%")
    print()

    if len(winners) > 0 and len(losers) > 0:
        print(" VWAP Distance: Winners vs Losers")
        print(f"  Winners: {winners['vwap_distance'].mean():.2f}% from VWAP")
        print(f"  Losers: {losers['vwap_distance'].mean():.2f}% from VWAP")

    print()

    print("=" * 80)
    print("5. OPTIMAL HEDGE CONDITIONS - ML RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Build optimal conditions based on winners
    if len(winners) > 0:
        print("[OK] Based on successful hedged trades, optimal conditions are:")
        print()

        print(f"  TIMING:")
        print(f"   Wait: {winners['time_to_hedge_minutes'].median():.0f} minutes after entry")
        print(f"   Range: {winners['time_to_hedge_minutes'].min():.0f}-{winners['time_to_hedge_minutes'].max():.0f} minutes")
        print()

        print(f" MARKET CONDITIONS:")
        print(f"   ADX: {winners['adx'].min():.1f} - {winners['adx'].max():.1f}")
        if winners['adx'].mean() < 20:
            print(f"   -> RANGING markets preferred (ADX < 20)")
        elif winners['adx'].mean() <= 25:
            print(f"   -> TRANSITIONING markets (20 <= ADX <= 25)")
        else:
            print(f"   -> STRONG TRENDING markets (ADX > 25)")
        print()

        print(f" PRICE MOVEMENT:")
        print(f"   Trigger when price moves: {winners['price_move_pct'].min():.1f}% - {winners['price_move_pct'].max():.1f}%")
        print(f"   Optimal trigger: ~{winners['price_move_pct'].median():.1f}% move")
        print()

        print(f" VWAP POSITION:")
        print(f"   Distance from VWAP: {winners['vwap_distance'].min():.2f}% - {winners['vwap_distance'].max():.2f}%")
        print()

        print(f" CONFLUENCE:")
        print(f"   Use on setups with confluence >= {winners['confluence_score'].min():.0f}")
        print()

    print("=" * 80)
    print("6. SUGGESTED HEDGE TRIGGER RULES")
    print("=" * 80)
    print()

    if len(winners) > 0:
        median_time = winners['time_to_hedge_minutes'].median()
        median_price_move = winners['price_move_pct'].median()
        max_adx = winners['adx'].max()
        min_confluence = winners['confluence_score'].min()

        print(" ML-Recommended Hedge Activation Rules:")
        print()
        print(f"Rule 1 (Time-Based):")
        print(f"  IF time_since_entry >= {median_time:.0f} minutes AND")
        print(f"     price_against_position >= {median_price_move:.1f}% AND")
        print(f"     ADX < {max_adx:.0f} AND")
        print(f"     confluence_score >= {min_confluence:.0f}")
        print(f"  THEN activate_hedge()")
        print()

        # Alternative aggressive rule
        early_time = winners['time_to_hedge_minutes'].min()
        early_price = winners['price_move_pct'].min()

        print(f"Rule 2 (Aggressive - Early Hedge):")
        print(f"  IF time_since_entry >= {early_time:.0f} minutes AND")
        print(f"     price_against_position >= {early_price:.1f}% AND")
        print(f"     ADX < 20 AND")  # Ranging only
        print(f"     confluence_score >= {min_confluence:.0f}")
        print(f"  THEN activate_hedge()")
        print()

        # Conservative rule
        late_time = winners['time_to_hedge_minutes'].max()
        late_price = winners['price_move_pct'].max()

        print(f"Rule 3 (Conservative - Late Hedge):")
        print(f"  IF time_since_entry >= {late_time:.0f} minutes AND")
        print(f"     price_against_position >= {late_price:.1f}% AND")
        print(f"     ADX < {max_adx:.0f}")
        print(f"  THEN activate_hedge()")
        print()

    print("=" * 80)
    print("7. FEATURE IMPORTANCE FOR ML TRAINING")
    print("=" * 80)
    print()

    print("Once models are retrained with 60 features, ML will learn:")
    print("  * Optimal time window for hedge activation")
    print("  * ADX threshold that maximizes hedge success")
    print("  * Price movement sweet spot (not too early, not too late)")
    print("  * Interaction between hedge timing and DCA timing")
    print("  * Which confluence patterns benefit most from hedging")
    print()
    print("Key features to monitor after retraining:")
    print("  1. had_hedge (does it improve predictions?)")
    print("  2. hedge_count (is 1 better than 2?)")
    print("  3. Interaction: had_hedge × adx")
    print("  4. Interaction: had_hedge × confluence_score")
    print("  5. Interaction: had_hedge × time_to_hedge")
    print()

    # Save detailed results
    output_file = 'ml_system/outputs/hedge_timing_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"[OK] Detailed analysis saved to: {output_file}")
    print()

    return df

if __name__ == '__main__':
    analyze_hedge_timing_and_conditions()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Continue collecting trades with hedge enabled/disabled")
    print("2. Test recommended timing rules on new trades")
    print("3. Retrain ML models with 60 features")
    print("4. Monitor feature importance for hedge-related features")
    print("5. Adjust hedge timing based on ML predictions")
    print()
