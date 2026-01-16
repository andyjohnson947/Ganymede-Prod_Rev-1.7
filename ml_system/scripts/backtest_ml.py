#!/usr/bin/env python3
"""
Day 12: Backtest ML Decisions
Compare ML predictions vs actual trade outcomes
"""

import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_system.ml_shadow_trader import MLShadowTrader


def backtest_ml_decisions(
    log_file='ml_system/outputs/continuous_trade_log.jsonl',
    output_file='ml_system/outputs/backtest_results.json'
):
    """
    Backtest ML model on historical trades.

    Args:
        log_file: Path to continuous trade log
        output_file: Path to save backtest results
    """

    print("=" * 80)
    print("DAY 12: ML BACKTEST")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input: {log_file}")
    print("=" * 80)

    # Step 1: Load ML Shadow Trader
    print("\n[1/5] Loading ML Shadow Trader...")
    try:
        shadow = MLShadowTrader()
        print("[OK] ML Shadow Trader loaded")
        model_info = shadow.get_model_info()
        print(f"  Model: {model_info['model_type']}")
        print(f"  Threshold: {model_info['confidence_threshold']:.1%}")
    except Exception as e:
        print(f"[ERROR] Error loading ML Shadow Trader: {e}")
        return None

    # Step 2: Load historical trades
    print("\n[2/5] Loading historical trades...")
    trades = []

    try:
        with open(log_file, 'r') as f:
            for line in f:
                trades.append(json.loads(line))
        print(f"[OK] Loaded {len(trades)} trades")
    except FileNotFoundError:
        print(f"[ERROR] Error: {log_file} not found!")
        return None

    # Filter for closed trades only
    closed_trades = [t for t in trades if t.get('outcome', {}).get('status') == 'closed']
    print(f"  Closed trades: {len(closed_trades)}")

    if len(closed_trades) == 0:
        print("  [WARN] No closed trades to backtest!")
        return None

    # Step 3: Make ML decisions for each trade
    print("\n[3/5] Running ML backtest...")

    results = []
    ml_take_count = 0
    ml_skip_count = 0

    for i, trade in enumerate(closed_trades, 1):
        # Get ML decision
        decision, confidence = shadow.make_ml_decision(trade)

        # Get actual outcome
        outcome = trade.get('outcome', {})
        profit = outcome.get('profit', 0)
        actual_win = profit > 0

        # Record result
        result = {
            'ticket': trade.get('ticket'),
            'symbol': trade.get('symbol'),
            'entry_time': trade.get('entry_time'),
            'confluence_score': trade.get('confluence_score'),
            'ml_decision': decision,
            'ml_confidence': confidence,
            'actual_profit': profit,
            'actual_win': actual_win
        }
        results.append(result)

        if decision == 'TAKE':
            ml_take_count += 1
        else:
            ml_skip_count += 1

        # Progress indicator
        if i % 10 == 0:
            print(f"  Processed {i}/{len(closed_trades)} trades...")

    print(f"[OK] Processed all {len(closed_trades)} closed trades")
    print(f"  ML TAKE decisions: {ml_take_count}")
    print(f"  ML SKIP decisions: {ml_skip_count}")

    # Step 4: Analyze results
    print("\n[4/5] Analyzing results...")

    df = pd.DataFrame(results)

    # Overall statistics
    total_trades = len(df)
    original_win_rate = df['actual_win'].mean()
    original_total_profit = df['actual_profit'].sum()

    print(f"\n   Original Bot Performance:")
    print(f"    Total trades: {total_trades}")
    print(f"    Win rate: {original_win_rate:.2%}")
    print(f"    Total profit: ${original_total_profit:.2f}")
    print(f"    Avg profit per trade: ${df['actual_profit'].mean():.2f}")

    # ML filtered performance
    ml_trades = df[df['ml_decision'] == 'TAKE']

    if len(ml_trades) > 0:
        ml_win_rate = ml_trades['actual_win'].mean()
        ml_total_profit = ml_trades['actual_profit'].sum()
        ml_avg_profit = ml_trades['actual_profit'].mean()

        print(f"\n   ML-Filtered Performance:")
        print(f"    Trades taken: {len(ml_trades)} ({len(ml_trades)/total_trades*100:.1f}% of original)")
        print(f"    Win rate: {ml_win_rate:.2%}")
        print(f"    Total profit: ${ml_total_profit:.2f}")
        print(f"    Avg profit per trade: ${ml_avg_profit:.2f}")

        # Calculate improvement
        win_rate_improvement = ml_win_rate - original_win_rate
        profit_improvement = ml_total_profit - original_total_profit

        print(f"\n   Improvement:")
        print(f"    Win rate: {win_rate_improvement:+.1%}")
        print(f"    Total profit: ${profit_improvement:+.2f}")

        # Risk metrics
        ml_wins = len(ml_trades[ml_trades['actual_win']])
        ml_losses = len(ml_trades) - ml_wins

        print(f"\n   Risk Metrics:")
        print(f"    ML Wins: {ml_wins}")
        print(f"    ML Losses: {ml_losses}")
        print(f"    Win/Loss ratio: {ml_wins/ml_losses:.2f}" if ml_losses > 0 else "    Win/Loss ratio: Infinity")

        # Trades skipped by ML
        skipped_trades = df[df['ml_decision'] == 'SKIP']
        if len(skipped_trades) > 0:
            skipped_wins = len(skipped_trades[skipped_trades['actual_win']])
            skipped_losses = len(skipped_trades) - skipped_wins
            skipped_profit = skipped_trades['actual_profit'].sum()

            print(f"\n   Trades Skipped by ML:")
            print(f"    Count: {len(skipped_trades)}")
            print(f"    Would-be wins: {skipped_wins}")
            print(f"    Would-be losses: {skipped_losses}")
            print(f"    Profit avoided/missed: ${skipped_profit:.2f}")

        # Confidence analysis
        print(f"\n   Confidence Analysis:")
        high_conf_trades = ml_trades[ml_trades['ml_confidence'] >= 0.8]
        if len(high_conf_trades) > 0:
            print(f"    High confidence (>=80%): {len(high_conf_trades)} trades")
            print(f"      Win rate: {high_conf_trades['actual_win'].mean():.2%}")

        med_conf_trades = ml_trades[(ml_trades['ml_confidence'] >= 0.6) & (ml_trades['ml_confidence'] < 0.8)]
        if len(med_conf_trades) > 0:
            print(f"    Medium confidence (60-80%): {len(med_conf_trades)} trades")
            print(f"      Win rate: {med_conf_trades['actual_win'].mean():.2%}")

    else:
        print("\n  [WARN] ML skipped all trades! Model may be too conservative.")

    # Step 5: Save results
    print("\n[5/5] Saving results...")

    backtest_summary = {
        'timestamp': datetime.now().isoformat(),
        'total_trades': int(total_trades),
        'ml_take_count': int(ml_take_count),
        'ml_skip_count': int(ml_skip_count),
        'original_win_rate': float(original_win_rate),
        'original_total_profit': float(original_total_profit),
        'ml_win_rate': float(ml_win_rate) if len(ml_trades) > 0 else None,
        'ml_total_profit': float(ml_total_profit) if len(ml_trades) > 0 else None,
        'win_rate_improvement': float(win_rate_improvement) if len(ml_trades) > 0 else None,
        'profit_improvement': float(profit_improvement) if len(ml_trades) > 0 else None,
        'model_info': model_info
    }

    with open(output_file, 'w') as f:
        json.dump(backtest_summary, f, indent=2)

    print(f"[OK] Backtest summary saved to: {output_file}")

    # Save detailed results
    detailed_file = Path(output_file).parent / 'backtest_detailed.csv'
    df.to_csv(detailed_file, index=False)
    print(f"[OK] Detailed results saved to: {detailed_file}")

    # Summary
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    print(f"\n[OK] Backtested {total_trades} closed trades")

    if len(ml_trades) > 0:
        if ml_win_rate > original_win_rate:
            print(f"[OK] ML improves win rate by {win_rate_improvement:+.1%}")
            print(f"[OK] ML would have changed profit by ${profit_improvement:+.2f}")
        else:
            print(f"[WARN] ML does not improve win rate (original: {original_win_rate:.1%}, ML: {ml_win_rate:.1%})")

        print(f"\n Recommendation:")
        if ml_win_rate > original_win_rate + 0.05:  # 5% improvement
            print(f"  [OK] ML model shows significant improvement")
            print(f"  [OK] Recommend using ML for trade filtering")
        elif ml_win_rate > original_win_rate:
            print(f"  ~ ML model shows modest improvement")
            print(f"  ~ Consider using ML with lower confidence threshold")
        else:
            print(f"  [WARN] ML model needs more training data")
            print(f"  [WARN] Continue collecting trades before deployment")

    print("\n[OK] Ready for Day 13: Real-time Prediction Pipeline")
    print("=" * 80)

    return backtest_summary


if __name__ == '__main__':
    results = backtest_ml_decisions()

    if results:
        print("\n[OK] Day 12 Complete!")
    else:
        print("\n[ERROR] Day 12 Failed - check errors above")
