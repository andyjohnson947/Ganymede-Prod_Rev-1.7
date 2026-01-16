"""
Main Confluence Strategy
Orchestrates signal detection, position management, and recovery
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

from core.mt5_manager import MT5Manager
from strategies.signal_detector import SignalDetector
from strategies.recovery_manager import RecoveryManager
from strategies.time_filters import TimeFilter
from strategies.breakout_strategy import BreakoutStrategy
from strategies.partial_close_manager import PartialCloseManager
from utils.risk_calculator import RiskCalculator
from utils.config_reloader import reload_config, print_current_config
from utils.timezone_manager import get_current_time
from portfolio.portfolio_manager import PortfolioManager
from config.strategy_config import (
    SYMBOLS,
    TIMEFRAME,
    HTF_TIMEFRAMES,
    DATA_REFRESH_INTERVAL,
    MAX_OPEN_POSITIONS,
    MAX_POSITIONS_PER_SYMBOL,
    PROFIT_TARGET_PERCENT,
    MAX_POSITION_HOURS,
    PARTIAL_CLOSE_ENABLED,
    PARTIAL_CLOSE_RECOVERY,
    BREAKOUT_ENABLED,
    BREAKOUT_LOT_SIZE_MULTIPLIER,
    ENABLE_CASCADE_PROTECTION,  # Cascade stop protection
    TREND_BLOCK_MINUTES,        # Trade block duration after cascade
)


class ConfluenceStrategy:
    """Main trading strategy implementation"""

    def __init__(self, mt5_manager: MT5Manager, test_mode: bool = False, ml_logger=None):
        """
        Initialize strategy

        Args:
            mt5_manager: MT5Manager instance (already connected)
            test_mode: If True, bypass all time filters for testing
            ml_logger: ContinuousMLLogger instance for trailing stop event logging
        """
        print("[DEBUG] ConfluenceStrategy.__init__() starting...", flush=True)
        self.mt5 = mt5_manager
        self.test_mode = test_mode
        self.ml_logger = ml_logger  # ML logger for real-time event tracking
        print("[DEBUG] Creating SignalDetector...", flush=True)
        self.signal_detector = SignalDetector()
        print("[DEBUG] Creating RecoveryManager...", flush=True)
        self.recovery_manager = RecoveryManager()
        print("[DEBUG] Creating RiskCalculator...", flush=True)
        self.risk_calculator = RiskCalculator()
        print("[DEBUG] Creating PortfolioManager...", flush=True)
        self.portfolio_manager = PortfolioManager()

        # New strategy modules
        self.time_filter = TimeFilter()
        self.breakout_strategy = BreakoutStrategy() if BREAKOUT_ENABLED else None
        self.partial_close_manager = PartialCloseManager() if PARTIAL_CLOSE_ENABLED else None

        self.running = False
        self.last_data_refresh = {}
        self.market_data_cache = {}

        # Market state tracking for trend-based trade blocking
        self.market_trending_block = {}

        # Cascade protection - blocks new trades after cascade close
        self.cascade_blocks = {}  # {symbol: block_until_time}

        # Crash recovery tracking
        self.recovery_stacks_reconstructed = False

        # Statistics
        self.stats = {
            'signals_detected': 0,
            'trades_opened': 0,
            'trades_closed': 0,
            'grid_levels_added': 0,
            'hedges_activated': 0,
            'dca_levels_added': 0,
        }
        print("[DEBUG] ConfluenceStrategy.__init__() completed successfully", flush=True)

    def start(self, symbols: List[str]):
        """
        Start the trading strategy

        Args:
            symbols: List of symbols to trade
        """
        import sys
        print("[DEBUG] Entered strategy.start() method", flush=True)
        sys.stdout.flush()

        print("=" * 80)
        print(" CONFLUENCE STRATEGY STARTING")
        print("=" * 80)
        print()

        # Get account info
        account_info = self.mt5.get_account_info()
        if not account_info:
            print("[ERROR] Failed to get account info")
            return

        print(f"Account Balance: ${account_info['balance']:.2f}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Timeframe: {TIMEFRAME}")
        print(f"HTF: {', '.join(HTF_TIMEFRAMES)}")
        print()

        # Set initial balance for drawdown tracking
        self.risk_calculator.set_initial_balance(account_info['balance'])

        # CRASH RECOVERY: Load saved state and reconcile with MT5 reality
        print("🔄 Initializing crash recovery system...")
        state_loaded = self.recovery_manager.load_state()

        # ALWAYS reconcile with MT5, even if state loaded (critical for crash recovery)
        print("🔄 Reconciling tracked positions with MT5...")
        added, removed, validated = self.recovery_manager.reconcile_with_mt5(self.mt5)

        if state_loaded:
            print(f"[OK] State loaded and reconciled:")
            print(f"   ✓ Validated: {validated} positions still open")
            if removed > 0:
                print(f"   🗑️  Removed: {removed} closed positions")
            if added > 0:
                print(f"   ➕ Added: {added} new positions from MT5")
        else:
            if added > 0:
                print(f"[OK] No saved state - discovered {added} MT5 positions")
            else:
                print("[OK] No saved state - starting fresh")

        print()

        self.running = True
        loop_iteration = 0

        try:
            while self.running:
                # Main trading loop
                self._trading_loop(symbols)

                # Periodic state backup (every 10 minutes)
                loop_iteration += 1
                if loop_iteration % 10 == 0:
                    self.recovery_manager.save_state()

                # Sleep before next iteration
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            print("\n[WARN] Strategy stopped by user")
        except Exception as e:
            print(f"\n[ERROR] Strategy error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """Stop the strategy"""
        self.running = False

        # Save final state before shutdown
        print("\n Saving final state...")
        if self.recovery_manager.save_state():
            print("[OK] State saved successfully")
        else:
            print("[WARN]  Failed to save state")

        print()
        print("=" * 80)
        print(" STRATEGY STATISTICS")
        print("=" * 80)
        for key, value in self.stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print()

    def _trading_loop(self, symbols: List[str]):
        """
        Main trading loop iteration

        Args:
            symbols: Symbols to trade
        """
        for symbol in symbols:
            try:
                # 1. Check if we should refresh market data
                self._refresh_market_data(symbol)

                # 2. Manage existing positions
                self._manage_positions(symbol)

                # 3. Look for new signals
                if self._can_open_new_position(symbol):
                    self._check_for_signals(symbol)

            except Exception as e:
                print(f"[ERROR] Error processing {symbol}: {e}")
                import traceback
                traceback.print_exc()
                continue

    def _refresh_market_data(self, symbol: str):
        """Refresh market data for symbol if needed"""
        now = get_current_time()
        last_refresh = self.last_data_refresh.get(symbol)

        # Check if refresh needed
        if last_refresh:
            minutes_since = (now - last_refresh).total_seconds() / 60
            if minutes_since < DATA_REFRESH_INTERVAL:
                return  # Data still fresh

        # Fetch H1 data
        h1_data = self.mt5.get_historical_data(symbol, TIMEFRAME, bars=500)
        if h1_data is None:
            return

        # Calculate VWAP on H1 data
        h1_data = self.signal_detector.vwap.calculate(h1_data)

        # Calculate ATR for breakout detection
        if 'atr' not in h1_data.columns:
            # Simple ATR calculation (14 period)
            high_low = h1_data['high'] - h1_data['low']
            high_close = abs(h1_data['high'] - h1_data['close'].shift())
            low_close = abs(h1_data['low'] - h1_data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            h1_data['atr'] = true_range.rolling(window=14).mean()

        # Fetch HTF data
        d1_data = self.mt5.get_historical_data(symbol, 'D1', bars=100)
        w1_data = self.mt5.get_historical_data(symbol, 'W1', bars=50)

        if d1_data is None or w1_data is None:
            return

        # Cache the data
        self.market_data_cache[symbol] = {
            'h1': h1_data,
            'd1': d1_data,
            'w1': w1_data,
            'last_update': now
        }

        self.last_data_refresh[symbol] = now

    def _manage_positions(self, symbol: str):
        """Manage existing positions for symbol"""
        positions = self.mt5.get_positions(symbol)

        # CRASH RECOVERY: Reconstruct recovery stacks on first run (do this once across all symbols)
        if not self.recovery_stacks_reconstructed:
            # Get ALL positions (not just this symbol)
            all_positions = self.mt5.get_positions()

            # First, track all original positions
            for pos in all_positions:
                pos_ticket = pos['ticket']
                pos_comment = pos.get('comment', '')

                # Check if it's a recovery order
                is_recovery = any([
                    'Grid' in pos_comment,
                    'Hedge' in pos_comment,
                    'DCA' in pos_comment,
                ])

                # Track original positions only
                if not is_recovery and pos_ticket not in self.recovery_manager.tracked_positions:
                    self.recovery_manager.track_position(
                        ticket=pos_ticket,
                        symbol=pos['symbol'],
                        entry_price=pos['price_open'],
                        position_type='buy' if pos['type'] == 0 else 'sell',
                        volume=pos['volume']
                    )

            # Now reconstruct recovery stacks
            if len(all_positions) > 0:
                self.recovery_manager.reconstruct_recovery_stacks(all_positions)

            # Mark as done (only do this once)
            self.recovery_stacks_reconstructed = True

        # CONTINUOUS ORPHAN DETECTION: Check for newly orphaned recovery orders
        # This handles the case where a master position closes during trading and leaves grids/hedges/DCA behind
        # We need to detect and adopt these orphans so they get recovery protection
        # Use silent=True to avoid spamming logs on every iteration
        all_positions = self.mt5.get_positions()
        if len(all_positions) > 0:
            self.recovery_manager.reconstruct_recovery_stacks(all_positions, silent=True)

        # Check for trading window closures - close negative positions if window is ending
        close_actions = self.portfolio_manager.check_window_closures()
        for action in close_actions:
            if action.symbol == symbol and action.close_negatives_only:
                # Close all negative positions for this symbol
                for pos in positions:
                    if pos['profit'] < 0:
                        ticket = pos['ticket']
                        print(f"🚪 Closing negative position {ticket} - {action.reason}")
                        if self.mt5.close_position(ticket):
                            self.recovery_manager.untrack_position(ticket)
                            self.stats['trades_closed'] += 1

        for position in positions:
            ticket = position['ticket']
            comment = position.get('comment', '')

            # [WARN] CRITICAL FIX: Don't track recovery orders as new positions
            # Recovery orders have comments like "Grid L1 - 1001", "Hedge - 1001", "DCA L1 - 1001"
            # Only the ORIGINAL trade should spawn recovery, not recovery orders themselves
            is_recovery_order = any([
                'Grid' in comment,
                'Hedge' in comment,
                'DCA' in comment,
            ])

            # Check if position is being tracked
            if ticket not in self.recovery_manager.tracked_positions:
                # Only track original trades, NOT recovery orders
                if not is_recovery_order:
                    # Start tracking
                    self.recovery_manager.track_position(
                        ticket=ticket,
                        symbol=position['symbol'],
                        entry_price=position['price_open'],
                        position_type=position['type'],
                        volume=position['volume']
                    )

            # Get symbol info (needed for various checks)
            current_price = position['price_current']
            symbol_info = self.mt5.get_symbol_info(symbol)
            pip_value = symbol_info.get('point', 0.0001)

            # PARTIAL CLOSE + TRAILING STOP: ONLY for profitable ORIGINAL positions
            # NOT for recovery orders (grid/DCA/hedge) or positions in active recovery
            # This is the new exit strategy for positive trades moving toward TP

            # Check if this is a recovery order (grid/DCA/hedge)
            is_recovery_order = any([
                'Grid' in comment,
                'Hedge' in comment,
                'DCA' in comment,
            ])

            # Check if position is in active recovery (has underwater recovery stack)
            has_active_recovery = False
            if ticket in self.recovery_manager.tracked_positions:
                tracked_pos = self.recovery_manager.tracked_positions[ticket]
                has_active_recovery = tracked_pos.get('recovery_active', False)

            # ONLY apply to positive original positions without active recovery
            if self.partial_close_manager and position['profit'] > 0 and not is_recovery_order and not has_active_recovery:
                # Track position if not already tracked
                if ticket not in self.partial_close_manager.partial_closes:
                    # Calculate TP price based on VWAP or other exit logic
                    # For now, use a reasonable default TP
                    entry_price = position['price_open']
                    pos_type = 'buy' if position['type'] == 0 else 'sell'

                    # Estimate TP price (40 pips for EURUSD, adjust as needed)
                    tp_distance = 40 * pip_value
                    tp_price = entry_price + tp_distance if pos_type == 'buy' else entry_price - tp_distance

                    self.partial_close_manager.track_position(
                        ticket=ticket,
                        entry_price=entry_price,
                        volume=position['volume'],
                        position_type=pos_type,
                        tp_price=tp_price
                    )

                # Calculate current profit in pips
                entry_price = position['price_open']
                current_price = position['price_current']
                pip_diff = abs(current_price - entry_price) / pip_value

                # Check for partial close triggers
                partial_action = self.partial_close_manager.check_partial_close_levels(
                    ticket=ticket,
                    current_price=current_price,
                    current_profit_pips=pip_diff
                )

                if partial_action:
                    # Execute partial close
                    close_volume = partial_action['close_volume']
                    level_percent = partial_action['level_percent']
                    print(f" Partial close: {ticket} - {partial_action['close_percent']}% at {level_percent}% to TP")

                    # Check if close_volume equals or exceeds position volume (final close)
                    if close_volume >= position['volume']:
                        # Close entire position instead of partial
                        if self.mt5.close_position(ticket):
                            print(f"[OK] Full close successful: {position['volume']} lots (final partial close)")
                            self.recovery_manager.untrack_position(ticket)
                            self.stats['trades_closed'] += 1
                    else:
                        # Close partial volume with descriptive comment
                        # Format: "PC 50%@50%TP-{last 5 digits}" (max 31 chars, Windows-safe)
                        short_ticket = str(ticket)[-5:]  # Last 5 digits of ticket
                        partial_comment = f"PC {partial_action['close_percent']}%@{partial_action['level_percent']}%TP-{short_ticket}"

                        if self.mt5.close_partial_position(ticket, close_volume, comment=partial_comment):
                            print(f"[OK] Partial close successful: {close_volume} lots - {partial_comment}")

                            # PC2 TRIGGERS: Activate trailing stop + move SL to breakeven
                            if level_percent >= 50 and ticket in self.recovery_manager.tracked_positions:
                                tracked_pos = self.recovery_manager.tracked_positions[ticket]

                                # Get instrument config for trailing settings
                                try:
                                    from trading_bot.portfolio.instruments_config import INSTRUMENTS
                                    instrument_config = INSTRUMENTS.get(symbol, {})
                                    tp_settings = instrument_config.get('take_profit', {})

                                    # Activate trailing stop if enabled
                                    if tp_settings.get('trailing_stop_enabled') and not tracked_pos.get('trailing_stop_active'):
                                        self.recovery_manager.activate_trailing_stop(ticket, current_price, tp_settings)
                                        print(f"[PC2] Trailing stop activated for {ticket}")

                                        # Set PC2 trigger time for 60-min time limit
                                        from utils.time_utils import get_current_time
                                        tracked_pos['pc2_trigger_time'] = get_current_time()

                                        # ML LOGGING: Log PC2 trigger with trailing activation
                                        if self.ml_logger:
                                            trailing_distance = tracked_pos.get('trailing_stop_distance_pips', 0)
                                            trailing_stop_price = tracked_pos.get('trailing_stop_price', 0)
                                            entry_price = tracked_pos['entry_price']
                                            self.ml_logger.log_pc2_trigger(
                                                ticket=ticket,
                                                symbol=symbol,
                                                current_price=current_price,
                                                entry_price=entry_price,
                                                trailing_distance_pips=trailing_distance,
                                                trailing_stop_price=trailing_stop_price
                                            )

                                    # Move hardware SL to breakeven (entry price) for protection
                                    entry_price = tracked_pos['entry_price']
                                    if self.mt5.modify_position(ticket, sl=entry_price):
                                        print(f"[PC2] Hardware SL moved to breakeven @ {entry_price:.5f}")

                                        # ML LOGGING: Log SL move to breakeven
                                        if self.ml_logger:
                                            self.ml_logger.log_sl_to_breakeven(
                                                ticket=ticket,
                                                symbol=symbol,
                                                breakeven_price=entry_price
                                            )

                                except Exception as e:
                                    print(f"[WARN] PC2 trigger error for {ticket}: {e}")

            # TRAILING STOP SYSTEM: ONLY for positive original positions with trailing active
            # Recovery system manages underwater positions separately with grid/DCA/hedge
            if ticket in self.recovery_manager.tracked_positions:
                tracked_pos = self.recovery_manager.tracked_positions[ticket]

                # ONLY process trailing if: position is profitable AND no active recovery
                has_active_recovery = tracked_pos.get('recovery_active', False)
                is_trailing_active = tracked_pos.get('trailing_stop_active', False)
                is_positive = position['profit'] > 0

                # Skip trailing stop logic if position is in recovery mode (underwater)
                if has_active_recovery or not is_positive or not is_trailing_active:
                    # Position is either:
                    # - In active recovery (grid/DCA/hedge managing it) OR
                    # - Negative (shouldn't have trailing) OR
                    # - Trailing not activated yet
                    # Let recovery system handle it, skip trailing stop logic
                    pass
                else:
                    # Position is positive, standalone (no recovery), and trailing is active
                    # Apply PC2 time limit and trailing stop checks

                    # Check PC2 time limit (60 min) - close remaining 50% if time elapsed
                    pc2_time = tracked_pos.get('pc2_trigger_time')
                    if pc2_time:
                        from utils.time_utils import get_current_time
                        from datetime import timedelta
                        time_since_pc2 = get_current_time() - pc2_time
                        if time_since_pc2 >= timedelta(minutes=60):
                            print(f"[PC2 TIME LIMIT] 60 min elapsed since PC2 for {ticket} - closing position")
                            if self.mt5.close_position(ticket):
                                # ML LOGGING: Log time-based exit
                                if self.ml_logger:
                                    entry_price = tracked_pos.get('entry_price', current_price)
                                    peak_price = tracked_pos.get('highest_profit_price', current_price)
                                    self.ml_logger.log_trailing_event(
                                        event_type='pc2_time_limit',
                                        ticket=ticket,
                                        symbol=symbol,
                                        time_since_pc2_minutes=float(time_since_pc2.total_seconds() / 60),
                                        exit_price=float(current_price),
                                        entry_price=float(entry_price),
                                        peak_price=float(peak_price)
                                    )
                                self.recovery_manager.untrack_position(ticket)
                                self.stats['trades_closed'] += 1
                            continue

                    # Update trailing stop (moves stop with price as profit increases)
                    # Store old stop for comparison
                    old_stop = tracked_pos.get('trailing_stop_price', 0)

                    # Update trailing stop
                    self.recovery_manager.update_trailing_stop(ticket, current_price)

                    # Check if stop moved and log the update
                    new_stop = tracked_pos.get('trailing_stop_price', 0)
                    if new_stop != old_stop and self.ml_logger:
                        pip_value = symbol_info.get('point', 0.0001)
                        pips_moved = abs(new_stop - old_stop) / (pip_value * 10)
                        self.ml_logger.log_trailing_update(
                            ticket=ticket,
                            symbol=symbol,
                            old_stop=old_stop,
                            new_stop=new_stop,
                            current_price=current_price,
                            pips_moved=pips_moved
                        )

                    # Update hardware SL to match software trailing stop (crash protection)
                    trailing_stop_price = tracked_pos.get('trailing_stop_price')
                    if trailing_stop_price:
                        self.mt5.modify_position(ticket, sl=trailing_stop_price)

                    # Check if trailing stop hit
                    if self.recovery_manager.check_trailing_stop(ticket, current_price):
                        print(f"[TRAIL] Trailing stop hit for {ticket} - closing position")

                        # ML LOGGING: Log trailing stop hit with peak price and capture ratio
                        if self.ml_logger:
                            entry_price = tracked_pos.get('entry_price', current_price)
                            peak_price = tracked_pos.get('highest_profit_price', current_price)
                            self.ml_logger.log_trailing_hit(
                                ticket=ticket,
                                symbol=symbol,
                                trailing_stop_price=trailing_stop_price,
                                current_price=current_price,
                                entry_price=entry_price,
                                peak_price=peak_price
                            )

                        if self.mt5.close_position(ticket):
                            self.recovery_manager.untrack_position(ticket)
                            self.stats['trades_closed'] += 1
                        continue

            # RECOVERY & EXIT CONDITIONS: For UNDERWATER positions with active recovery
            # This handles positions that went negative and need grid/DCA/hedge management
            # Positive positions without recovery are handled by PC1/PC2/trailing above
            if ticket in self.recovery_manager.tracked_positions:
                # Check recovery triggers (grid, DCA, hedge for underwater positions)
                recovery_actions = self.recovery_manager.check_all_recovery_triggers(
                    ticket, current_price, pip_value
                )

                # Execute recovery actions
                for action in recovery_actions:
                    self._execute_recovery_action(action)

                # Check exit conditions (only for tracked original positions)
                # Priority order: 0) Stack drawdown (risk protection), 0.25) Per-stack stop loss, 0.5) Hedge drawdown, 1) Profit target, 2) Time limit, 3) VWAP reversion

                # Get account info and positions for checks
                account_info = self.mt5.get_account_info()
                all_positions = self.mt5.get_positions()

                # 0. Check stack drawdown (HIGHEST PRIORITY - risk protection)
                if self.recovery_manager.check_stack_drawdown(
                    ticket=ticket,
                    mt5_positions=all_positions,
                    pip_value=pip_value
                ):
                    self._close_recovery_stack(ticket)
                    continue

                # 0.25. Check per-stack stop loss + cascade detection
                # Get current ADX for stop-out logging
                current_adx = None
                if symbol in self.market_data_cache:
                    h1_data = self.market_data_cache[symbol]['h1']
                    if 'adx' in h1_data.columns and len(h1_data) > 0:
                        current_adx = h1_data.iloc[-1]['adx']

                # Check stack stop loss (passes ADX for logging)
                stack_stop = self.recovery_manager.check_stack_stop_loss(
                    ticket=ticket,
                    mt5_positions=all_positions,
                    current_adx=current_adx
                )

                if stack_stop:
                    print(f"\n[STOP] Per-stack stop loss triggered")
                    print(f"   Stack type: {stack_stop['stack_type']}")
                    print(f"   Loss: ${stack_stop['loss_amount']:.2f}")
                    print(f"   Limit: ${stack_stop['stop_loss_limit']:.2f}")

                    # Close this stack
                    self._close_recovery_stack(ticket)

                    # Check if cascade detected (2nd stop in 30min window)
                    if ENABLE_CASCADE_PROTECTION and self.recovery_manager.stop_out_tracker:
                        cascade_info = self.recovery_manager.stop_out_tracker.check_cascade()

                        if cascade_info:
                            # CASCADE DETECTED - close all underwater stacks
                            print(f"\n{'='*70}")
                            print(f"[CASCADE] MULTIPLE STOP-OUTS DETECTED")
                            print(f"{'='*70}")
                            print(f"   Stops in 30min: {cascade_info['stop_count']}")
                            if cascade_info['avg_adx']:
                                print(f"   Avg ADX: {cascade_info['avg_adx']:.1f}")
                            print(f"   Trend confirmed: {cascade_info['trend_confirmed']}")
                            print(f"   Symbols affected: {', '.join(cascade_info['symbols'])}")

                            # Get all underwater stacks
                            underwater_tickets = self.recovery_manager.get_underwater_stacks(all_positions)

                            if underwater_tickets:
                                print(f"\n   Closing {len(underwater_tickets)} underwater stack(s):")
                                for uw_ticket in underwater_tickets:
                                    if uw_ticket == ticket:
                                        continue  # Already closed above
                                    uw_profit = self.recovery_manager.calculate_net_profit(uw_ticket, all_positions)
                                    uw_symbol = self.recovery_manager.tracked_positions[uw_ticket]['symbol']
                                    print(f"     #{uw_ticket} ({uw_symbol}): ${uw_profit:.2f}")
                                    self._close_recovery_stack(uw_ticket)

                                # Block new trades for affected symbols if trend confirmed
                                if cascade_info['trend_confirmed']:
                                    block_until = get_current_time() + timedelta(minutes=TREND_BLOCK_MINUTES)
                                    for affected_symbol in cascade_info['symbols']:
                                        self.cascade_blocks[affected_symbol] = block_until
                                        print(f"\n   [BLOCK] {affected_symbol} trades blocked for {TREND_BLOCK_MINUTES} minutes")
                                        if cascade_info['avg_adx']:
                                            print(f"          Market trending (ADX: {cascade_info['avg_adx']:.1f})")
                            else:
                                print(f"   No other underwater stacks found")

                            print(f"{'='*70}\n")

                    continue

                # 0.5. Check hedge drawdown (NEW - monitor hedge positions specifically)
                hedge_action = self.recovery_manager.check_hedge_drawdown(
                    ticket=ticket,
                    mt5_positions=all_positions
                )
                if hedge_action:
                    # Close underwater hedges
                    hedges_to_close = hedge_action.get('hedges_to_close', [])
                    for hedge in hedges_to_close:
                        hedge_ticket = hedge['ticket']
                        print(f"[STOP] Closing underwater hedge {hedge_ticket} (Loss: ${hedge['loss']:.2f})")
                        if self.mt5.close_position(hedge_ticket):
                            self.recovery_manager.remove_closed_hedge(ticket, hedge_ticket)
                            self.stats['trades_closed'] += 1

                    # Check market state after closing hedges
                    if symbol in self.market_data_cache:
                        h1_data = self.market_data_cache[symbol]['h1']
                        market_state = self.recovery_manager.check_market_state_for_hedge_close(
                            symbol=symbol,
                            current_data=h1_data
                        )

                        # Store market state for use in signal detection
                        if not hasattr(self, 'market_trending_block'):
                            self.market_trending_block = {}
                        self.market_trending_block[symbol] = market_state.get('should_block_new_trades', False)

                # 1. Check profit target (from config)
                if account_info and self.recovery_manager.check_profit_target(
                    ticket=ticket,
                    mt5_positions=all_positions,
                    account_balance=account_info['balance'],
                    profit_percent=PROFIT_TARGET_PERCENT
                ):
                    self._close_recovery_stack(ticket)
                    continue

                # 2. Check time limit (from config)
                if self.recovery_manager.check_time_limit(ticket, hours_limit=MAX_POSITION_HOURS):
                    self._close_recovery_stack(ticket)
                    continue

            # 3. Check exit signal (VWAP reversion) - for tracked positions
            if symbol in self.market_data_cache:
                h1_data = self.market_data_cache[symbol]['h1']
                should_exit = self.signal_detector.check_exit_signal(position, h1_data)

                if should_exit:
                    print(f"Exit signal detected for {ticket} - VWAP reversion")

                    # Check if this is a tracked position with recovery stack
                    if ticket in self.recovery_manager.tracked_positions:
                        print(f"Closing entire recovery stack for {ticket}")
                        self._close_recovery_stack(ticket)
                    else:
                        # Standalone position (no recovery) - close normally
                        if self.mt5.close_position(ticket):
                            self.stats['trades_closed'] += 1

        # Check for orphaned hedges (hedges whose original positions are gone)
        self._check_orphaned_hedges()

    def _check_orphaned_hedges(self):
        """Check ALL positions for orphaned hedges and close if losing > $75"""
        all_positions = self.mt5.get_positions()
        if not all_positions:
            return

        for position in all_positions:
            comment = position.get('comment', '')

            # Check if this is a hedge position
            if 'Hedge -' in comment:
                ticket = position.get('ticket')
                profit = position.get('profit', 0)
                volume = position.get('volume', 0)
                symbol = position.get('symbol', 'UNKNOWN')

                # Check if hedge is losing > $75
                if profit < -75.0:
                    loss = abs(profit)
                    print(f"[STOP] ORPHANED HEDGE DETECTED: {symbol} #{ticket}")
                    print(f"   Loss: ${loss:.2f} (exceeds $75 threshold)")
                    print(f"   Volume: {volume:.2f} lots")
                    print(f"   Comment: {comment}")
                    print(f"   Closing immediately for risk protection...")

                    if self.mt5.close_position(ticket):
                        print(f"   [OK] Orphaned hedge {ticket} closed successfully")
                        self.stats['trades_closed'] += 1
                    else:
                        print(f"   [ERROR] Failed to close orphaned hedge {ticket}")

    def _check_for_signals(self, symbol: str):
        """Check for new entry signals"""
        if symbol not in self.market_data_cache:
            return

        # Check if symbol is tradeable based on portfolio trading windows (bypass in test mode)
        if not self.test_mode and not self.portfolio_manager.is_symbol_tradeable(symbol):
            return  # Not in trading window for this symbol

        cache = self.market_data_cache[symbol]
        h1_data = cache['h1']
        d1_data = cache['d1']
        w1_data = cache['w1']

        current_time = get_current_time()
        signal = None

        # Check which strategy can trade based on time filters (bypass in test mode)
        if self.test_mode:
            can_trade_mr = True
            can_trade_bo = True
        else:
            can_trade_mr = self.time_filter.can_trade_mean_reversion(current_time)
            can_trade_bo = self.time_filter.can_trade_breakout(current_time)

        # Try mean reversion signal first (if allowed)
        if can_trade_mr:
            signal = self.signal_detector.detect_signal(
                current_data=h1_data,
                daily_data=d1_data,
                weekly_data=w1_data,
                symbol=symbol
            )
            if signal:
                signal['strategy_type'] = 'mean_reversion'

        # Try breakout signal (if mean reversion found nothing and breakout is allowed)
        if signal is None and can_trade_bo and self.breakout_strategy:
            # Get current price and volume
            latest_bar = h1_data.iloc[-1]
            current_price = latest_bar['close']

            # Get volume - handle both 'volume' and 'tick_volume' columns
            if 'volume' in latest_bar:
                current_volume = latest_bar['volume']
            elif 'tick_volume' in latest_bar:
                current_volume = latest_bar['tick_volume']
            else:
                current_volume = 0  # Default if no volume data
                print(f"[WARN] Warning: No volume data for {symbol}, using 0")

            # Calculate ATR
            atr = h1_data['atr'].iloc[-1] if 'atr' in h1_data.columns else 0

            # Calculate RSI and MACD for comprehensive breakout detection
            from indicators.technical import add_indicators_to_dataframe
            h1_with_indicators = add_indicators_to_dataframe(h1_data)
            rsi = h1_with_indicators['rsi'].iloc[-1] if 'rsi' in h1_with_indicators.columns else 50.0
            macd_histogram = h1_with_indicators['macd_histogram'].iloc[-1] if 'macd_histogram' in h1_with_indicators.columns else 0.0

            # Calculate volume profile for LVN breakouts
            volume_profile_data = self.signal_detector.volume_profile.get_signals(
                h1_data, current_price, lookback=200
            )
            volume_profile = volume_profile_data.get('profile', {})

            # Get weekly levels for weekly breakout detection
            htf_levels = self.signal_detector.htf_levels.get_all_levels(d1_data, w1_data)
            weekly_data = htf_levels.get('weekly', {})
            # Map to format expected by breakout strategy
            weekly_levels = {
                'high': weekly_data.get('prev_week_high', current_price + 0.01),
                'low': weekly_data.get('prev_week_low', current_price - 0.01)
            }

            # Prepare indicators dict for comprehensive breakout checker
            indicators = {
                'rsi': rsi,
                'macd_histogram': macd_histogram,
                'atr': atr,
                'volume': current_volume
            }

            # Check for comprehensive breakout signals (range + LVN + weekly)
            breakout_signal = self.breakout_strategy.check_breakout_signal(
                data=h1_data,
                current_price=current_price,
                volume_profile=volume_profile,
                weekly_levels=weekly_levels,
                indicators=indicators,
                current_time=current_time
            )

            if breakout_signal:
                # Convert breakout signal to standard signal format
                signal = {
                    'symbol': symbol,
                    'direction': breakout_signal['direction'],  # Now lowercase 'buy'/'sell'
                    'price': current_price,
                    'strategy_type': 'breakout',
                    'confluence_score': breakout_signal.get('score', 3),
                    'factors': breakout_signal.get('factors', []),
                    'breakout_details': breakout_signal  # Store full breakout details
                }

        if signal is None:
            return

        # Signal detected!
        self.stats['signals_detected'] += 1

        print()
        print(f"Signal: {signal.get('strategy_type', 'unknown').upper()}")
        print(self.signal_detector.get_signal_summary(signal))

        # Add breakout-specific logging
        if signal.get('strategy_type') == 'breakout' and signal.get('breakout_details'):
            details = signal['breakout_details']
            breakout_type = details.get('type', 'unknown')
            print(f"   Breakout Strategy: {breakout_type.upper()}")

            if breakout_type == 'range_breakout':
                print(f"   Range: {details.get('range_low', 0):.5f} - {details.get('range_high', 0):.5f}")
                print(f"   Range Size: {details.get('range_size_pips', 0):.1f} pips")
                print(f"   Logic: Price breaks {details.get('breakout_type', 'unknown').upper()} -> {signal['direction'].upper()} (ride momentum)")
            elif breakout_type == 'lvn_breakout':
                print(f"   LVN Level: {details.get('lvn_level', 0):.5f}")
                print(f"   RSI: {details.get('rsi', 0):.1f}")
                print(f"   Logic: Price breaks through Low Volume Node -> {signal['direction'].upper()} (fast move)")
            elif breakout_type == 'weekly_breakout':
                print(f"   Weekly High: {details.get('weekly_high', 0):.5f}")
                print(f"   Weekly Low: {details.get('weekly_low', 0):.5f}")
                print(f"   RSI: {details.get('rsi', 0):.1f}")
                print(f"   Logic: Price breaks weekly level -> {signal['direction'].upper()} (continuation)")

        print()

        # Execute trade
        self._execute_signal(signal)

    def _execute_signal(self, signal: Dict):
        """
        Execute a trading signal

        Args:
            signal: Signal dict from signal detector
        """
        symbol = signal['symbol']
        direction = signal['direction']
        price = signal['price']

        # Validate direction is lowercase 'buy' or 'sell'
        if direction not in ['buy', 'sell']:
            print(f"ERROR: Invalid direction '{direction}'. Must be 'buy' or 'sell'")
            print(f"   Signal type: {signal.get('strategy_type', 'unknown')}")
            print(f"   Converting to lowercase...")
            direction = direction.lower()
            if direction not in ['buy', 'sell']:
                print(f"ERROR: Direction '{direction}' still invalid after lowercase conversion. Aborting trade.")
                return

        # Log order details
        print(f"Order Details:")
        print(f"   Direction: {direction.upper()}")
        print(f"   Strategy: {signal.get('strategy_type', 'unknown').upper()}")
        print(f"   Symbol: {symbol}")
        print(f"   Price: {price:.5f}")

        # Check if symbol is blocked due to cascade (multiple stop-outs = trend detected)
        if symbol in self.cascade_blocks:
            block_until = self.cascade_blocks[symbol]
            if get_current_time() < block_until:
                time_remaining = (block_until - get_current_time()).total_seconds() / 60
                print(f" New trade BLOCKED for {symbol} - CASCADE PROTECTION")
                print(f"   Multiple stop-outs detected (trend confirmed)")
                print(f"   Block expires in: {time_remaining:.0f} minutes")
                return
            else:
                # Block expired, remove it
                del self.cascade_blocks[symbol]
                print(f" Cascade block for {symbol} expired - resuming normal trading")

        # Check if new trades are blocked due to trending market
        # This is the "belt and braces" approach - prevent new trades when market is trending
        # Recovery trades are still allowed (they are executed via _execute_recovery_action, not here)
        if hasattr(self, 'market_trending_block') and self.market_trending_block.get(symbol, False):
            print(f" New trade BLOCKED for {symbol} - Market is trending")
            print(f"   This protection prevents opening new positions during strong trends")
            print(f"   Recovery trades are still allowed to manage existing positions")
            return

        # Get account and symbol info
        account_info = self.mt5.get_account_info()
        symbol_info = self.mt5.get_symbol_info(symbol)

        if not account_info or not symbol_info:
            print("[ERROR] Failed to get account/symbol info")
            return

        # Calculate position size
        volume = self.risk_calculator.calculate_position_size(
            account_balance=account_info['balance'],
            symbol_info=symbol_info
        )

        # Apply breakout multiplier if this is a breakout signal
        if signal.get('strategy_type') == 'breakout':
            volume = volume * BREAKOUT_LOT_SIZE_MULTIPLIER
            # Round to broker's volume step
            volume_step = symbol_info.get('volume_step', 0.01)
            volume = round(volume / volume_step) * volume_step
            # Ensure minimum volume
            volume = max(symbol_info.get('volume_min', 0.01), volume)
            print(f" Breakout signal: Reducing lot size to {volume} (50% of base)")

        # Get current positions for validation
        positions = self.mt5.get_positions()

        # Validate trade
        can_trade, reason = self.risk_calculator.validate_trade(
            account_info=account_info,
            symbol_info=symbol_info,
            volume=volume,
            current_positions=positions
        )

        if not can_trade:
            print(f"[ERROR] Trade validation failed: {reason}")
            return

        # Import INITIAL_TRADE_COUNT
        from config.strategy_config import INITIAL_TRADE_COUNT

        # Place order(s) - open multiple trades if INITIAL_TRADE_COUNT > 1
        # Include strategy type in comment for ML analysis
        # VWAP = Mean reversion to VWAP/levels, BREAKOUT = Momentum through levels
        strategy_label = "VWAP" if signal.get('strategy_type') == 'mean_reversion' else "BREAKOUT"
        comment = f"{strategy_label}:C{signal['confluence_score']}"
        trades_opened = 0

        for i in range(INITIAL_TRADE_COUNT):
            trade_comment = comment if INITIAL_TRADE_COUNT == 1 else f"{comment} #{i+1}"

            ticket = self.mt5.place_order(
                symbol=symbol,
                order_type=direction,
                volume=volume,
                sl=None,  # EA doesn't use hard stops
                tp=None,  # Using VWAP reversion instead
                comment=trade_comment
            )

            if ticket:
                trades_opened += 1
                self.stats['trades_opened'] += 1
                print(f"[OK] Trade #{i+1} opened: Ticket {ticket}")

                # Start tracking for recovery
                self.recovery_manager.track_position(
                    ticket=ticket,
                    symbol=symbol,
                    entry_price=price,
                    position_type=direction,
                    volume=volume
                )
            else:
                print(f"[ERROR] Failed to open trade #{i+1}")

        if trades_opened > 0:
            print(f" Total trades opened: {trades_opened}/{INITIAL_TRADE_COUNT}")

    def _close_recovery_stack(self, original_ticket: int):
        """
        Close entire recovery stack (original + grid + hedge + DCA)

        Args:
            original_ticket: Original position ticket
        """
        # Get all tickets in the stack
        stack_tickets = self.recovery_manager.get_all_stack_tickets(original_ticket)

        print(f"📦 Closing recovery stack for {original_ticket}")
        print(f"   Closing {len(stack_tickets)} positions...")

        closed_count = 0
        for ticket in stack_tickets:
            if self.mt5.close_position(ticket):
                closed_count += 1
                self.stats['trades_closed'] += 1
                print(f"   [OK] Closed #{ticket}")
            else:
                print(f"   [ERROR] Failed to close #{ticket}")

        # Untrack the original position
        self.recovery_manager.untrack_position(original_ticket)

        print(f"📦 Stack closed: {closed_count}/{len(stack_tickets)} positions")

    def _execute_recovery_action(self, action: Dict):
        """
        Execute a recovery action (grid/hedge/dca)

        Args:
            action: Recovery action dict
        """
        action_type = action['action']
        symbol = action['symbol']
        order_type = action['type']
        volume = action['volume']
        comment = action['comment']
        original_ticket = action.get('original_ticket')

        # Get calculated price for grid trades (if provided)
        # Grid trades should execute at specific price levels, not market price
        calculated_price = action.get('price', None)

        # Place order
        # IMPORTANT: Grid trades are PYRAMID trades - open new positions in same direction
        # This creates overlapping positions to lock in profit while maintaining exposure
        if action_type == 'grid' or action_type == 'dca' or action_type == 'hedge':
            # MARKET ORDER: Executes immediately for Grid/DCA/Hedge
            ticket = self.mt5.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                comment=comment
            )
        else:
            ticket = None

        if ticket:
            # Grid trades get tracked with LIMITED recovery (DCA/Hedge YES, more Grids NO)
            # Hedge/DCA trades link to original parent
            if action_type == 'grid':
                # Track grid as position with recovery, but flag to prevent grid spawning
                self.recovery_manager.track_position(
                    ticket=ticket,
                    symbol=symbol,
                    entry_price=action.get('price', 0),
                    position_type=order_type,
                    volume=volume,
                    is_grid_child=True  # Prevents more grids (stops cascade)
                )
                print(f"   Grid trade {ticket} tracked with recovery (DCA/Hedge: YES, more Grids: NO)")
                self.stats['grid_levels_added'] += 1
            else:
                # Hedge and DCA link to original parent
                if original_ticket:
                    self.recovery_manager.store_recovery_ticket(
                        original_ticket=original_ticket,
                        recovery_ticket=ticket,
                        action_type=action_type
                    )

                # Update statistics
                if action_type == 'hedge':
                    self.stats['hedges_activated'] += 1
                elif action_type == 'dca':
                    self.stats['dca_levels_added'] += 1

    def _can_open_new_position(self, symbol: str) -> bool:
        """Check if we can open a new position"""
        # Check total positions
        all_positions = self.mt5.get_positions()
        if len(all_positions) >= MAX_OPEN_POSITIONS:
            return False

        # Check positions per symbol
        symbol_positions = self.mt5.get_positions(symbol)
        if len(symbol_positions) >= MAX_POSITIONS_PER_SYMBOL:
            return False

        return True

    def get_status(self) -> Dict:
        """Get current strategy status"""
        account_info = self.mt5.get_account_info()
        positions = self.mt5.get_positions()

        risk_metrics = self.risk_calculator.get_risk_metrics(
            account_info=account_info or {},
            positions=positions
        )

        recovery_status = self.recovery_manager.get_all_positions_status()

        return {
            'running': self.running,
            'account': account_info,
            'risk_metrics': risk_metrics,
            'positions': positions,
            'recovery_status': recovery_status,
            'statistics': self.stats,
            'cached_symbols': list(self.market_data_cache.keys()),
        }

    def reload_config(self):
        """
        Reload configuration from strategy_config.py without restarting bot
        Fixes Python caching issue where config changes require full restart
        """
        print()
        print("=" * 60)
        print("🔄 RELOADING CONFIGURATION")
        print("=" * 60)
        success = reload_config()
        if success:
            print_current_config()
            print("[OK] Config reloaded successfully!")
            print("   Changes will take effect on next trading cycle")
        else:
            print("[ERROR] Config reload failed")
        print("=" * 60)
        print()
        return success
