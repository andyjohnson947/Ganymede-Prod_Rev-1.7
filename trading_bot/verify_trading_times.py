"""
Verify Trading Times and Windows
Shows current time, DST status, and tradeable instruments
"""

from datetime import datetime
from utils.timezone_manager import get_timezone_manager
from portfolio.portfolio_manager import PortfolioManager
from portfolio.instruments_config import get_instruments_config
from utils.trading_calendar import get_trading_calendar


def main():
    print("=" * 80)
    print("TRADING TIME VERIFICATION")
    print("=" * 80)

    # Initialize managers
    tz_manager = get_timezone_manager()
    portfolio_manager = PortfolioManager()
    calendar = get_trading_calendar()

    # Get current times
    current_time_uk = tz_manager.get_current_trading_timezone()
    current_time_gmt = tz_manager.get_gmt_time()
    is_dst = tz_manager.is_dst_active()
    tz_name = tz_manager.get_timezone_name()

    print(f"\n📅 CURRENT TIME:")
    print(f"   Local (UK):  {current_time_uk.strftime('%Y-%m-%d %H:%M:%S')} {tz_name}")
    print(f"   Pure GMT:    {current_time_gmt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"   DST Active:  {'Yes (GMT+1/BST)' if is_dst else 'No (GMT)'}")
    print(f"   Offset:      UTC{current_time_uk.strftime('%z')}")

    # Check trading restrictions
    is_allowed, reason = calendar.is_trading_allowed(current_time_uk)
    print(f"\n🚦 TRADING STATUS:")
    print(f"   Allowed:     {'[OK] YES' if is_allowed else '[ERROR] NO'}")
    print(f"   Reason:      {reason}")

    # Get tradeable instruments RIGHT NOW
    tradeable_instruments = portfolio_manager.get_tradeable_instruments(current_time_uk)
    tradeable_symbols = [inst.symbol for inst in tradeable_instruments]

    print(f"\n💹 CURRENTLY TRADEABLE INSTRUMENTS ({len(tradeable_symbols)}):")
    if tradeable_symbols:
        for inst in tradeable_instruments:
            active_windows = inst.get_active_windows(current_time_uk)
            print(f"   [OK] {inst.symbol:8} ({inst.name})")

            for window in active_windows:
                print(f"      Window: {window.name}")
                print(f"      Time:   {window.start.strftime('%H:%M')} - {window.end.strftime('%H:%M')} GMT")
                print(f"      Strategy: {window.strategy_type}")

                # Calculate time remaining
                time_remaining = window.time_until_close(current_time_uk)
                if time_remaining:
                    hours, remainder = divmod(int(time_remaining.total_seconds()), 3600)
                    minutes, _ = divmod(remainder, 60)
                    print(f"      Remaining: {hours}h {minutes}m")
                print()
    else:
        print("   [WARN]  No instruments in trading window at this time")

    # Show all configured instruments and their windows
    print(f"\n📋 ALL CONFIGURED INSTRUMENTS:")
    instruments_config = get_instruments_config()

    for symbol, config in instruments_config.items():
        if not config.get('enabled', True):
            continue

        print(f"\n   {symbol} ({config['name']}):")
        for window in config['windows']:
            # Check if this window is active now
            is_active = symbol in tradeable_symbols
            status = "🟢 ACTIVE NOW" if is_active else "⚪ Inactive"

            print(f"      {status} | {window['name']}")
            print(f"         Time: {window['start'].strftime('%H:%M')} - {window['end'].strftime('%H:%M')} GMT")
            print(f"         Strategy: {window['strategy_type']}")

    # Check for upcoming windows in the next 4 hours
    print(f"\n UPCOMING TRADING WINDOWS (Next 4 hours):")
    from datetime import timedelta

    upcoming_found = False
    check_time = current_time_uk

    for minutes_ahead in range(0, 240, 30):  # Check every 30 minutes for 4 hours
        check_time = current_time_uk + timedelta(minutes=minutes_ahead)
        upcoming_tradeable = portfolio_manager.get_tradeable_symbols(check_time)

        if upcoming_tradeable and upcoming_tradeable != tradeable_symbols:
            print(f"   At {check_time.strftime('%H:%M')}: {', '.join(upcoming_tradeable)}")
            upcoming_found = True

    if not upcoming_found:
        print("   No changes in next 4 hours")

    print("\n" + "=" * 80)
    print(" TIP: Run this script anytime to verify current trading status")
    print("=" * 80)


if __name__ == "__main__":
    main()
