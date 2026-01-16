#!/usr/bin/env python3
"""
Re-scan historical trades to fix hedge/grid detection
Updates continuous_trade_log.jsonl with corrected recovery counts

Usage:
  python3 fix_historical_recovery_data.py
  python3 fix_historical_recovery_data.py --login 12345 --password xxx --server MetaQuotes-Demo
"""
import sys
sys.path.insert(0, 'trading_bot')

import json
import argparse
from datetime import datetime
from pathlib import Path

try:
    import MetaTrader5 as mt5
except ImportError:
    print("Error: MetaTrader5 module not found")
    print("This script must run on a system with MT5 installed")
    sys.exit(1)

def find_recovery_actions(ticket: int, from_date: datetime, to_date: datetime):
    """Find all recovery actions (DCA, hedge, grid) for a specific ticket"""
    recovery = {
        'dca_count': 0,
        'hedge_count': 0,
        'grid_count': 0,
        'dca_levels': [],
        'hedge_ratios': [],
        'grid_levels': [],
        'total_recovery_volume': 0.0,
        'recovery_cost': 0.0
    }

    # Get all deals for this period
    deals = mt5.history_deals_get(from_date, to_date)
    if not deals:
        return recovery

    # Create ticket search patterns (FIXED: now includes last 5 digits)
    ticket_str = str(ticket)
    ticket_patterns = [
        ticket_str,           # Full ticket
        ticket_str[:7],       # First 7
        ticket_str[:8],       # First 8
        ticket_str[:9],       # First 9
        ticket_str[-5:],      # Last 5 digits (for Hedge/Grid) ← FIXED!
    ]

    for deal in deals:
        comment = deal.comment or ""

        # Check if this deal is related to our ticket
        is_related = (deal.position_id == ticket)

        if not is_related:
            for pattern in ticket_patterns:
                if pattern in comment:
                    is_related = True
                    break

        if is_related:
            if 'DCA' in comment:
                recovery['dca_count'] += 1
                recovery['dca_levels'].append({
                    'price': float(deal.price),
                    'volume': float(deal.volume),
                    'time': datetime.fromtimestamp(deal.time).isoformat()
                })
                recovery['total_recovery_volume'] += float(deal.volume)
                if deal.profit < 0:
                    recovery['recovery_cost'] += abs(float(deal.profit))

            elif 'Hedge' in comment:
                recovery['hedge_count'] += 1
                recovery['hedge_ratios'].append({
                    'price': float(deal.price),
                    'volume': float(deal.volume),
                    'time': datetime.fromtimestamp(deal.time).isoformat()
                })
                recovery['total_recovery_volume'] += float(deal.volume)
                if deal.profit < 0:
                    recovery['recovery_cost'] += abs(float(deal.profit))

            elif 'Grid' in comment:  # FIXED: removed 'positive' requirement
                recovery['grid_count'] += 1
                recovery['grid_levels'].append({
                    'price': float(deal.price),
                    'volume': float(deal.volume),
                    'profit': float(deal.profit),
                    'time': datetime.fromtimestamp(deal.time).isoformat()
                })

    return recovery

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Re-scan historical trades for hedge/grid detection')
    parser.add_argument('--login', type=int, help='MT5 account login')
    parser.add_argument('--password', type=str, help='MT5 account password')
    parser.add_argument('--server', type=str, help='MT5 server name')
    args = parser.parse_args()

    # Initialize MT5
    print("Initializing MT5...")
    if args.login and args.password and args.server:
        print(f"Connecting to {args.server} with account {args.login}...")
        if not mt5.initialize():
            print("Error: MT5 initialization failed (step 1)")
            sys.exit(1)

        if not mt5.login(args.login, args.password, args.server):
            error = mt5.last_error()
            print(f"Error: MT5 login failed: {error}")
            mt5.shutdown()
            sys.exit(1)
        print("✅ Connected to MT5")
    else:
        # Try to initialize with current terminal session
        if not mt5.initialize():
            print("Error: MT5 initialization failed")
            print("\nUsage:")
            print("  python3 fix_historical_recovery_data.py --login 12345 --password xxx --server MetaQuotes-Demo")
            sys.exit(1)
        print("✅ Connected to MT5 (using current terminal session)")

    log_file = Path('ml_system/outputs/continuous_trade_log.jsonl')

    if not log_file.exists():
        print(f"Error: {log_file} not found")
        return

    # Load all trades
    trades = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))

    print(f"Loaded {len(trades)} trades from log")
    print("=" * 80)

    updated_count = 0

    # Re-scan each closed trade
    for i, trade in enumerate(trades):
        if trade.get('outcome', {}).get('status') != 'closed':
            continue

        ticket = trade.get('ticket')
        if not ticket:
            continue

        # Get entry and exit times
        entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
        exit_time = trade.get('outcome', {}).get('exit_time')
        if exit_time:
            exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
        else:
            continue

        # Re-scan for recovery actions with FIXED logic
        old_recovery = trade.get('outcome', {}).get('recovery', {})
        new_recovery = find_recovery_actions(ticket, entry_time, exit_time)

        # Check if anything changed
        old_hedge = old_recovery.get('hedge_count', 0)
        old_grid = old_recovery.get('grid_count', 0)
        new_hedge = new_recovery.get('hedge_count', 0)
        new_grid = new_recovery.get('grid_count', 0)

        if old_hedge != new_hedge or old_grid != new_grid:
            print(f"Trade #{ticket}:")
            print(f"  OLD: Hedge={old_hedge}, Grid={old_grid}")
            print(f"  NEW: Hedge={new_hedge}, Grid={new_grid}")

            # Update the trade
            trade['outcome']['recovery'] = new_recovery
            updated_count += 1

    print()
    print("=" * 80)
    print(f"Updated {updated_count} trades")

    if updated_count > 0:
        # Backup original
        backup_file = log_file.with_suffix('.jsonl.backup')
        print(f"Creating backup: {backup_file}")

        import shutil
        shutil.copy(log_file, backup_file)

        # Write updated log
        print(f"Writing updated log: {log_file}")
        with open(log_file, 'w', encoding='utf-8') as f:
            for trade in trades:
                f.write(json.dumps(trade) + '\n')

        print("✅ Done! Historical recovery data updated.")
        print(f"Backup saved to: {backup_file}")
    else:
        print("No changes needed.")

    # Cleanup
    mt5.shutdown()

if __name__ == '__main__':
    main()
