"""
Recovery Strategy Manager
Implements Grid Trading, Hedging, and DCA/Martingale
All discovered from EA analysis of 428 trades
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import threading
from pathlib import Path

from utils.logger import logger
from utils.timezone_manager import get_current_time
from utils.data_utils import convert_numpy_types
from portfolio.instruments_config import get_recovery_settings, get_take_profit_settings
from config.strategy_config import (
    GRID_ENABLED,
    DISABLE_NEGATIVE_GRID,  # Kill switch for negative grid
    GRID_SPACING_PIPS,      # Fallback default
    MAX_GRID_LEVELS,        # Fallback default
    GRID_LOT_SIZE,
    HEDGE_ENABLED,
    HEDGE_TRIGGER_PIPS,     # Fallback default
    HEDGE_RATIO,
    MAX_HEDGES_PER_POSITION,
    MAX_HEDGE_VOLUME,       # Maximum volume for any single hedge
    STACK_DRAWDOWN_MULTIPLIER,  # Drawdown threshold for killing stacks
    DCA_ENABLED,
    DCA_TRIGGER_PIPS,       # Fallback default
    DCA_MAX_LEVELS,         # Fallback default
    DCA_MULTIPLIER,         # Fallback default
    ENABLE_STACK_STOPS,     # Per-stack stop loss management
    DCA_ONLY_MAX_LOSS,      # Max loss for DCA-only stacks
    DCA_HEDGE_MAX_LOSS,     # Max loss for DCA+hedge stacks
    ENABLE_CASCADE_PROTECTION,  # Cascade stop protection
    STOP_OUT_WINDOW_MINUTES,    # Time window for cascade detection
    CASCADE_THRESHOLD,          # Stop count threshold
    TREND_BLOCK_MINUTES,        # Trade block duration
    CASCADE_ADX_THRESHOLD,      # ADX threshold for trend confirmation
)


def round_volume_to_step(volume: float, step: float = 0.01, min_lot: float = 0.01, max_lot: float = 100.0) -> float:
    """
    Round volume to broker's step size and clamp to min/max limits

    Args:
        volume: Raw volume to round
        step: Broker's volume step (default 0.01)
        min_lot: Minimum allowed lot size (default 0.01)
        max_lot: Maximum allowed lot size (default 100.0)

    Returns:
        float: Rounded and clamped volume
    """
    # Round to nearest step
    rounded = round(volume / step) * step

    # Clamp to broker limits
    rounded = max(min_lot, min(rounded, max_lot))

    # Round to 2 decimals for display
    return round(rounded, 2)


class StopOutTracker:
    """
    Track stop-out events to detect cascade failures (range→trend transitions)

    When multiple stops trigger in short time window, likely indicates
    market regime change from ranging to trending. Allows closing all
    underwater stacks to prevent cascade of losses.
    """

    def __init__(self, window_minutes: int = STOP_OUT_WINDOW_MINUTES):
        """
        Initialize stop-out tracker

        Args:
            window_minutes: Time window for cascade detection (default from config)
        """
        self.stop_outs = []  # List of recent stop-out events
        self.window_minutes = window_minutes

        # Log file for stop-out analysis
        self.log_file = Path("logs/stop_out_events.log")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def add_stop_out(self, ticket: int, symbol: str, loss: float, adx_value: float, stack_type: str):
        """
        Record a stop-out event

        Args:
            ticket: Position ticket that was stopped
            symbol: Trading symbol
            loss: Loss amount in dollars
            adx_value: ADX value at time of stop
            stack_type: Type of stack (Initial, DCA-only, DCA+Hedge)
        """
        event = {
            'timestamp': get_current_time(),
            'ticket': ticket,
            'symbol': symbol,
            'loss': loss,
            'adx': adx_value,
            'stack_type': stack_type
        }

        self.stop_outs.append(event)
        self._cleanup_old_events()
        self._log_to_file(event)

        print(f"\n[STOP-OUT] Event #{ticket} recorded:")
        print(f"   Symbol: {symbol}")
        print(f"   Loss: ${loss:.2f}")
        print(f"   ADX: {adx_value:.1f}" if adx_value else "   ADX: N/A")
        print(f"   Stack: {stack_type}")
        print(f"   Recent stops (30min): {len(self.stop_outs)}")

    def _cleanup_old_events(self):
        """Remove events outside the time window"""
        now = get_current_time()
        cutoff = now - timedelta(minutes=self.window_minutes)
        self.stop_outs = [e for e in self.stop_outs if e['timestamp'] > cutoff]

    def _log_to_file(self, event: Dict):
        """Log stop-out event to file for analysis"""
        try:
            # Format ADX value (handle None case)
            adx_str = f"{event['adx']:.1f}" if event['adx'] is not None else 'N/A'

            with open(self.log_file, 'a', encoding='utf-8', errors='ignore') as f:
                log_line = (
                    f"{event['timestamp'].isoformat()} | "
                    f"#{event['ticket']} | "
                    f"{event['symbol']} | "
                    f"${event['loss']:.2f} | "
                    f"ADX:{adx_str} | "
                    f"{event['stack_type']}\n"
                )
                f.write(log_line)
        except Exception as e:
            logger.warning(f"Failed to log stop-out event: {e}")

    def check_cascade(self, threshold: int = CASCADE_THRESHOLD) -> Optional[Dict]:
        """
        Check if cascade condition is met (multiple stops in window)

        Args:
            threshold: Number of stops required to trigger cascade

        Returns:
            Dict with cascade info if triggered, None otherwise
        """
        self._cleanup_old_events()

        if len(self.stop_outs) >= threshold:
            # Calculate average ADX from stop events
            adx_values = [e['adx'] for e in self.stop_outs if e['adx'] is not None]
            avg_adx = sum(adx_values) / len(adx_values) if adx_values else None

            # Determine if trend is confirmed (high ADX)
            trend_confirmed = avg_adx and avg_adx >= CASCADE_ADX_THRESHOLD

            return {
                'cascade_detected': True,
                'stop_count': len(self.stop_outs),
                'events': self.stop_outs,
                'avg_adx': avg_adx,
                'trend_confirmed': trend_confirmed,
                'symbols': list(set([e['symbol'] for e in self.stop_outs]))
            }

        return None


class RecoveryManager:
    """Manage recovery strategies: Grid, Hedge, DCA/Martingale"""

    def __init__(self):
        """Initialize recovery manager"""
        self.tracked_positions = {}  # Track active positions and their recovery state
        self.archived_positions = []  # Archive closed positions for ML analysis (last 100)

        # Thread locks for atomic hedge operations (prevents race conditions)
        self.hedge_locks = {}  # Dict[int, threading.Lock] - one lock per position

        # Cascade stop protection - tracks stop-out events to detect range→trend transitions
        self.stop_out_tracker = StopOutTracker() if ENABLE_CASCADE_PROTECTION else None

    def _get_recovery_settings(self, symbol: str) -> Dict:
        """
        Get recovery settings for a symbol with fallback to defaults.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with recovery settings
        """
        try:
            # Try to get instrument-specific settings
            settings = get_recovery_settings(symbol)
            return settings
        except (KeyError, Exception):
            # Fallback to global defaults from strategy_config
            return {
                'grid_spacing_pips': GRID_SPACING_PIPS,
                'dca_trigger_pips': DCA_TRIGGER_PIPS,
                'hedge_trigger_pips': HEDGE_TRIGGER_PIPS,
                'dca_multiplier': DCA_MULTIPLIER,
                'max_grid_levels': MAX_GRID_LEVELS,
                'max_dca_levels': DCA_MAX_LEVELS,
            }

    def track_position(
        self,
        ticket: int,
        symbol: str,
        entry_price: float,
        position_type: str,
        volume: float,
        is_grid_child: bool = False
    ):
        """
        Start tracking a position for recovery

        Args:
            ticket: Position ticket
            symbol: Trading symbol
            entry_price: Entry price
            position_type: 'buy' or 'sell'
            volume: Initial lot size
            is_grid_child: If True, this is a Grid daughter (DCA/Hedge YES, more Grids NO)
        """
        self.tracked_positions[ticket] = {
            'ticket': ticket,
            'symbol': symbol,
            'entry_price': entry_price,
            'type': position_type,
            'initial_volume': volume,
            'grid_levels': [],
            'hedge_tickets': [],
            'dca_levels': [],
            'total_volume': volume,
            'max_underwater_pips': 0,
            'recovery_active': False,
            'open_time': get_current_time(),  # Track when position opened
            'last_hedge_time': None,  # Track last hedge time for cooldown
            'last_grid_time': None,  # Track last grid time for cooldown
            'is_grid_child': is_grid_child,  # Prevents grid spawning (stops cascade)
            # Partial profit tracking
            'partial_1_closed': False,
            'partial_2_closed': False,
            'pc2_trigger_time': None,  # Track when PC2 triggers for time-based exit (60 min limit)
            # Trailing stop (activated after PC2)
            'trailing_stop_active': False,
            'trailing_stop_distance_pips': 0,
            'trailing_stop_price': 0.0,
            'highest_profit_price': entry_price,  # Track highest price for trailing
        }

    def untrack_position(self, ticket: int):
        """Remove position from tracking"""
        if ticket in self.tracked_positions:
            del self.tracked_positions[ticket]

    def store_recovery_ticket(self, original_ticket: int, recovery_ticket: int, action_type: str):
        """
        Store ticket number for a recovery order after it's been placed

        Args:
            original_ticket: Original position ticket
            recovery_ticket: New recovery order ticket
            action_type: 'grid', 'hedge', or 'dca'
        """
        if original_ticket not in self.tracked_positions:
            return

        position = self.tracked_positions[original_ticket]

        if action_type == 'grid' and position['grid_levels']:
            # Store ticket in the most recent grid level and clear pending flag
            position['grid_levels'][-1]['ticket'] = recovery_ticket
            position['grid_levels'][-1]['pending'] = False  # Clear pending flag
            logger.info(f"[GRID] Ticket {recovery_ticket} stored for position {original_ticket}, pending flag cleared")

        elif action_type == 'hedge' and position['hedge_tickets']:
            # Store ticket in the most recent hedge and clear pending flag
            position['hedge_tickets'][-1]['ticket'] = recovery_ticket
            position['hedge_tickets'][-1]['pending'] = False  # Clear pending flag
            logger.info(f"[HEDGE] Ticket {recovery_ticket} stored for position {original_ticket}, pending flag cleared")

        elif action_type == 'dca' and position['dca_levels']:
            # Store ticket in the most recent DCA level
            position['dca_levels'][-1]['ticket'] = recovery_ticket

    def reconstruct_recovery_stacks(self, mt5_positions: List[Dict], silent: bool = False) -> Dict:
        """
        Reconstruct recovery stacks by parsing position comments.
        Critical for crash recovery - rebuilds tracking from MT5 positions.

        Args:
            mt5_positions: List of all current MT5 positions
            silent: If True, suppress verbose logging (for continuous orphan checks)

        Returns:
            Dict with reconstruction stats
        """
        if not silent:
            print("\n" + "="*60)
            print("🔄 RECOVERY STACK RECONSTRUCTION")
            print("="*60)

        # Track reconstruction statistics
        stats = {
            'total_positions': len(mt5_positions),
            'stacks_reconstructed': 0,
            'grid_levels_found': 0,
            'hedges_found': 0,
            'dca_levels_found': 0,
            'orphaned_recovery_orders': 0
        }

        # Group positions by original ticket
        recovery_stacks = {}

        for pos in mt5_positions:
            comment = pos.get('comment', '')
            ticket = pos.get('ticket')

            if not comment:
                continue

            # Parse Grid orders: "Grid L2 - 12345" (last 5 digits of parent ticket)
            if 'Grid L' in comment and ' - ' in comment:
                try:
                    parts = comment.split(' - ')
                    short_ticket = parts[1].strip()  # Last 5 digits

                    # Find original ticket by matching last 5 digits
                    original_ticket = None
                    for check_ticket in [p.get('ticket') for p in mt5_positions]:
                        if str(check_ticket).endswith(short_ticket):
                            # Make sure it's not a recovery order itself
                            check_comment = next((p.get('comment', '') for p in mt5_positions if p.get('ticket') == check_ticket), '')
                            if not any(marker in check_comment for marker in ['Grid', 'Hedge', 'DCA']):
                                original_ticket = check_ticket
                                break

                    if not original_ticket:
                        # Parent closed - adopt immediately as independent position
                        stats['orphaned_recovery_orders'] += 1
                        if ticket not in self.tracked_positions:
                            symbol = pos.get('symbol', 'UNKNOWN')
                            pos_type = 'buy' if pos.get('type') == 0 else 'sell'
                            self.tracked_positions[ticket] = {
                                'ticket': ticket,
                                'symbol': symbol,
                                'entry_price': pos.get('price_open', 0),
                                'type': pos_type,
                                'initial_volume': pos.get('volume', 0),
                                'grid_levels': [],
                                'hedge_tickets': [],
                                'dca_levels': [],
                                'total_volume': pos.get('volume', 0),
                                'max_underwater_pips': 0,
                                'recovery_active': False,
                                'open_time': get_current_time(),
                                'last_hedge_time': None,
                                'last_grid_time': None,
                                'is_orphaned': True,
                                'orphan_source': 'grid',
                                'is_grid_child': True,  # Grid orphans shouldn't spawn more grids
                            }
                            if not silent:
                                print(f"[OK] Adopted orphaned Grid #{ticket} (parent closed)")
                        continue

                    # Extract level number
                    level_str = comment.split('Grid L')[1].split(' ')[0]
                    level = int(level_str)

                    if original_ticket not in recovery_stacks:
                        recovery_stacks[original_ticket] = {
                            'grid_levels': [],
                            'hedge_tickets': [],
                            'dca_levels': []
                        }

                    recovery_stacks[original_ticket]['grid_levels'].append({
                        'ticket': ticket,
                        'level': level,
                        'volume': pos.get('volume', 0),
                        'price': pos.get('price_open', 0),
                        'time': pos.get('time', None)
                    })

                    stats['grid_levels_found'] += 1

                except (ValueError, IndexError) as e:
                    print(f"[WARN]  Failed to parse grid comment: {comment} - {e}")

            # Parse Hedge orders: "Hedge - 12345" (last 5 digits of parent ticket)
            elif 'Hedge - ' in comment:
                try:
                    short_ticket = comment.split(' - ')[1].strip()  # Last 5 digits

                    # Find original ticket by matching last 5 digits
                    original_ticket = None
                    for check_ticket in [p.get('ticket') for p in mt5_positions]:
                        if str(check_ticket).endswith(short_ticket):
                            # Make sure it's not a recovery order itself
                            check_comment = next((p.get('comment', '') for p in mt5_positions if p.get('ticket') == check_ticket), '')
                            if not any(marker in check_comment for marker in ['Grid', 'Hedge', 'DCA']):
                                original_ticket = check_ticket
                                break

                    if not original_ticket:
                        # Parent closed - adopt immediately as independent position
                        stats['orphaned_recovery_orders'] += 1
                        if ticket not in self.tracked_positions:
                            symbol = pos.get('symbol', 'UNKNOWN')
                            pos_type = 'buy' if pos.get('type') == 0 else 'sell'
                            self.tracked_positions[ticket] = {
                                'ticket': ticket,
                                'symbol': symbol,
                                'entry_price': pos.get('price_open', 0),
                                'type': pos_type,
                                'initial_volume': pos.get('volume', 0),
                                'grid_levels': [],
                                'hedge_tickets': [],
                                'dca_levels': [],
                                'total_volume': pos.get('volume', 0),
                                'max_underwater_pips': 0,
                                'recovery_active': False,
                                'open_time': get_current_time(),
                                'last_hedge_time': None,
                                'last_grid_time': None,
                                'is_orphaned': True,
                                'orphan_source': 'hedge',
                            }
                            if not silent:
                                print(f"[OK] Adopted orphaned Hedge #{ticket} (parent closed)")
                        continue

                    if original_ticket not in recovery_stacks:
                        recovery_stacks[original_ticket] = {
                            'grid_levels': [],
                            'hedge_tickets': [],
                            'dca_levels': []
                        }

                    # Determine hedge type from MT5 position type
                    hedge_type = 'sell' if pos.get('type') == 1 else 'buy'

                    recovery_stacks[original_ticket]['hedge_tickets'].append({
                        'ticket': ticket,
                        'type': hedge_type,
                        'volume': pos.get('volume', 0),
                        'trigger_pips': 0,  # Unknown after reconstruction
                        'time': pos.get('time', None)
                    })

                    stats['hedges_found'] += 1

                except (ValueError, IndexError) as e:
                    print(f"[WARN]  Failed to parse hedge comment: {comment} - {e}")

            # Parse DCA orders: "DCA L1 - 12345" (last 5 digits of parent ticket)
            elif 'DCA L' in comment and ' - ' in comment:
                try:
                    parts = comment.split(' - ')
                    short_ticket = parts[1].strip()  # Last 5 digits

                    # Find original ticket by matching last 5 digits
                    original_ticket = None
                    for check_ticket in [p.get('ticket') for p in mt5_positions]:
                        if str(check_ticket).endswith(short_ticket):
                            # Make sure it's not a recovery order itself
                            check_comment = next((p.get('comment', '') for p in mt5_positions if p.get('ticket') == check_ticket), '')
                            if not any(marker in check_comment for marker in ['Grid', 'Hedge', 'DCA']):
                                original_ticket = check_ticket
                                break

                    if not original_ticket:
                        # Parent closed - adopt immediately as independent position
                        stats['orphaned_recovery_orders'] += 1
                        if ticket not in self.tracked_positions:
                            symbol = pos.get('symbol', 'UNKNOWN')
                            pos_type = 'buy' if pos.get('type') == 0 else 'sell'
                            self.tracked_positions[ticket] = {
                                'ticket': ticket,
                                'symbol': symbol,
                                'entry_price': pos.get('price_open', 0),
                                'type': pos_type,
                                'initial_volume': pos.get('volume', 0),
                                'grid_levels': [],
                                'hedge_tickets': [],
                                'dca_levels': [],
                                'total_volume': pos.get('volume', 0),
                                'max_underwater_pips': 0,
                                'recovery_active': False,
                                'open_time': get_current_time(),
                                'last_hedge_time': None,
                                'last_grid_time': None,
                                'is_orphaned': True,
                                'orphan_source': 'dca',
                            }
                            if not silent:
                                print(f"[OK] Adopted orphaned DCA #{ticket} (parent closed)")
                        continue

                    # Extract level number
                    level_str = comment.split('DCA L')[1].split(' ')[0]
                    level = int(level_str)

                    if original_ticket not in recovery_stacks:
                        recovery_stacks[original_ticket] = {
                            'grid_levels': [],
                            'hedge_tickets': [],
                            'dca_levels': []
                        }

                    recovery_stacks[original_ticket]['dca_levels'].append({
                        'ticket': ticket,
                        'level': level,
                        'volume': pos.get('volume', 0),
                        'time': pos.get('time', None)
                    })

                    stats['dca_levels_found'] += 1

                except (ValueError, IndexError) as e:
                    print(f"[WARN]  Failed to parse DCA comment: {comment} - {e}")

        # Apply reconstructed stacks to tracked positions
        for original_ticket, stack_data in recovery_stacks.items():
            if original_ticket in self.tracked_positions:
                # Position is tracked, restore its recovery stack
                position = self.tracked_positions[original_ticket]

                # Restore grid levels (sorted by level number)
                if stack_data['grid_levels']:
                    position['grid_levels'] = sorted(
                        stack_data['grid_levels'],
                        key=lambda x: x['level']
                    )

                # Restore hedges
                if stack_data['hedge_tickets']:
                    position['hedge_tickets'] = stack_data['hedge_tickets']

                # Restore DCA levels (sorted by level number)
                if stack_data['dca_levels']:
                    position['dca_levels'] = sorted(
                        stack_data['dca_levels'],
                        key=lambda x: x['level']
                    )

                # Recalculate total volume
                total_volume = position['initial_volume']

                # Add grid volumes
                for grid in position['grid_levels']:
                    total_volume += grid['volume']

                # Add DCA volumes
                for dca in position['dca_levels']:
                    total_volume += dca['volume']

                # Note: Don't add hedge volume (opposite direction)

                position['total_volume'] = round(total_volume, 2)
                position['recovery_active'] = True  # Mark as having recovery

                stats['stacks_reconstructed'] += 1

                # Log reconstruction
                if not silent:
                    print(f"\n[OK] Reconstructed stack for position #{original_ticket}:")
                    print(f"   Symbol: {position['symbol']}")
                    print(f"   Grid levels: {len(position['grid_levels'])}")
                    print(f"   Hedges: {len(position['hedge_tickets'])}")
                    print(f"   DCA levels: {len(position['dca_levels'])}")
                    print(f"   Total volume: {position['total_volume']:.2f} lots")

            else:
                # Original position not tracked - ADOPT orphaned recovery orders!
                orphan_count = (
                    len(stack_data['grid_levels']) +
                    len(stack_data['hedge_tickets']) +
                    len(stack_data['dca_levels'])
                )
                stats['orphaned_recovery_orders'] += orphan_count

                if not silent:
                    logger.warning(f"WARNING: ORPHANED recovery orders for #{original_ticket}:")
                    logger.warning(f"   Original position not found in tracking!")
                    logger.warning(f"   Grid: {len(stack_data['grid_levels'])}")
                    logger.warning(f"   Hedge: {len(stack_data['hedge_tickets'])}")
                    logger.warning(f"   DCA: {len(stack_data['dca_levels'])}")

                # ADOPT orphaned positions to give them recovery protection
                # Each orphaned order becomes a tracked position with LIMITED recovery
                adopted_count = 0

                # Adopt grid orphans
                for grid_data in stack_data['grid_levels']:
                    ticket = grid_data['ticket']
                    if ticket in self.tracked_positions:
                        continue  # Already tracked somehow

                    # Find the grid position in MT5 positions to get symbol and type
                    grid_pos = next((p for p in mt5_positions if p.get('ticket') == ticket), None)
                    if not grid_pos:
                        logger.warning(f"   Cannot find MT5 position for grid #{ticket}")
                        continue

                    # Get symbol and type from MT5 position
                    symbol = grid_pos.get('symbol', 'UNKNOWN')
                    pos_type = 'buy' if grid_pos.get('type') == 0 else 'sell'

                    # Create tracking entry for orphaned grid
                    self.tracked_positions[ticket] = {
                        'ticket': ticket,
                        'symbol': symbol,
                        'entry_price': grid_data['price'],
                        'type': pos_type,
                        'initial_volume': grid_data['volume'],
                        'grid_levels': [],  # No grid-on-grid allowed!
                        'hedge_tickets': [],
                        'dca_levels': [],
                        'total_volume': grid_data['volume'],
                        'max_underwater_pips': 0,
                        'recovery_active': False,
                        'open_time': get_current_time(),
                        'last_hedge_time': None,  # Track last hedge time for cooldown
                        'last_grid_time': None,  # Track last grid time for cooldown
                        'is_orphaned': True,  # Mark as orphaned
                        'orphan_source': 'grid',  # Was originally a grid trade
                        'original_parent': original_ticket  # Track original parent
                    }
                    adopted_count += 1
                    if not silent:
                        logger.info(f"   OK: Adopted orphaned GRID #{ticket} as new tracked position")

                # Adopt hedge orphans
                for hedge_data in stack_data['hedge_tickets']:
                    hedge_ticket = hedge_data['ticket']  # Extract ticket from dict
                    if hedge_ticket in self.tracked_positions:
                        continue

                    # Find hedge position in MT5 positions
                    hedge_pos = next((p for p in mt5_positions if p.get('ticket') == hedge_ticket), None)
                    if hedge_pos:
                        symbol = hedge_pos.get('symbol', 'UNKNOWN')
                        pos_type = 'buy' if hedge_pos.get('type') == 0 else 'sell'

                        self.tracked_positions[hedge_ticket] = {
                            'ticket': hedge_ticket,
                            'symbol': symbol,
                            'entry_price': hedge_pos['price_open'],
                            'type': pos_type,
                            'initial_volume': hedge_pos['volume'],
                            'grid_levels': [],  # No grid on orphaned hedge
                            'hedge_tickets': [],
                            'dca_levels': [],
                            'total_volume': hedge_pos['volume'],
                            'max_underwater_pips': 0,
                            'recovery_active': False,
                            'open_time': get_current_time(),
                            'last_hedge_time': None,  # Track last hedge time for cooldown
                            'last_grid_time': None,  # Track last grid time for cooldown
                            'is_orphaned': True,
                            'orphan_source': 'hedge',
                            'original_parent': original_ticket
                        }
                        adopted_count += 1
                        if not silent:
                            logger.info(f"   OK: Adopted orphaned HEDGE #{hedge_ticket} as new tracked position")

                # Adopt DCA orphans
                for dca_data in stack_data['dca_levels']:
                    ticket = dca_data['ticket']
                    if ticket in self.tracked_positions:
                        continue

                    # Find DCA position in MT5 positions
                    dca_pos = next((p for p in mt5_positions if p.get('ticket') == ticket), None)
                    if not dca_pos:
                        logger.warning(f"   Cannot find MT5 position for DCA #{ticket}")
                        continue

                    symbol = dca_pos.get('symbol', 'UNKNOWN')
                    pos_type = 'buy' if dca_pos.get('type') == 0 else 'sell'

                    self.tracked_positions[ticket] = {
                        'ticket': ticket,
                        'symbol': symbol,
                        'entry_price': dca_data.get('price', dca_pos['price_open']),
                        'type': pos_type,
                        'initial_volume': dca_data['volume'],
                        'grid_levels': [],  # No grid on orphaned DCA
                        'hedge_tickets': [],
                        'dca_levels': [],
                        'total_volume': dca_data['volume'],
                        'max_underwater_pips': 0,
                        'recovery_active': False,
                        'open_time': get_current_time(),
                        'last_hedge_time': None,  # Track last hedge time for cooldown
                        'last_grid_time': None,  # Track last grid time for cooldown
                        'is_orphaned': True,
                        'orphan_source': 'dca',
                        'original_parent': original_ticket
                    }
                    adopted_count += 1
                    if not silent:
                        logger.info(f"   OK: Adopted orphaned DCA #{ticket} as new tracked position")

                if not silent and adopted_count > 0:
                    logger.info(f"   INFO: Adopted {adopted_count}/{orphan_count} orphaned positions with LIMITED recovery")
                    logger.info(f"   INFO: Recovery rights: DCA [YES], Hedge [YES], Grid [NO] (prevent cascade)")

        # Summary
        if not silent:
            print("\n" + "-"*60)
            print("RECONSTRUCTION SUMMARY:")
            print(f"  Total MT5 positions: {stats['total_positions']}")
            print(f"  Stacks reconstructed: {stats['stacks_reconstructed']}")
            print(f"  Grid levels found: {stats['grid_levels_found']}")
            print(f"  Hedges found: {stats['hedges_found']}")
            print(f"  DCA levels found: {stats['dca_levels_found']}")

            if stats['orphaned_recovery_orders'] > 0:
                print(f"  [WARN]  Orphaned orders: {stats['orphaned_recovery_orders']}")

            print("="*60 + "\n")

        return stats

    def _convert_position_datetimes(self, position: dict) -> dict:
        """
        Convert all datetime objects in position to ISO strings for JSON serialization.
        This ensures Windows-safe UTF-8 encoding without datetime serialization errors.

        Args:
            position: Position dictionary potentially containing datetime objects

        Returns:
            dict: Position dictionary with all datetime objects converted to ISO strings
        """
        pos_data = position.copy()

        # Convert top-level datetime fields
        if 'open_time' in pos_data and pos_data['open_time']:
            if isinstance(pos_data['open_time'], datetime):
                pos_data['open_time'] = pos_data['open_time'].isoformat()

        if 'closed_time' in pos_data and pos_data['closed_time']:
            if isinstance(pos_data['closed_time'], datetime):
                pos_data['closed_time'] = pos_data['closed_time'].isoformat()

        # Convert cooldown timer fields
        if 'last_hedge_time' in pos_data and pos_data['last_hedge_time']:
            if isinstance(pos_data['last_hedge_time'], datetime):
                pos_data['last_hedge_time'] = pos_data['last_hedge_time'].isoformat()

        if 'last_grid_time' in pos_data and pos_data['last_grid_time']:
            if isinstance(pos_data['last_grid_time'], datetime):
                pos_data['last_grid_time'] = pos_data['last_grid_time'].isoformat()

        # Convert time fields in recovery orders (modify in-place since we already copied)
        for grid in pos_data.get('grid_levels', []):
            if 'time' in grid and grid['time']:
                if isinstance(grid['time'], datetime):
                    grid['time'] = grid['time'].isoformat()

        for hedge in pos_data.get('hedge_tickets', []):
            if 'time' in hedge and hedge['time']:
                if isinstance(hedge['time'], datetime):
                    hedge['time'] = hedge['time'].isoformat()

        for dca in pos_data.get('dca_levels', []):
            if 'time' in dca and dca['time']:
                if isinstance(dca['time'], datetime):
                    dca['time'] = dca['time'].isoformat()

        return pos_data

    def save_state(self, state_file: str = "data/recovery_state.json") -> bool:
        """
        Save current tracking state to JSON file for crash recovery.

        Args:
            state_file: Path to state file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            state_path = Path(state_file)
            state_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare state data (active + archived)
            state = {
                'version': '2.0',  # Version for future compatibility
                'timestamp': datetime.now().isoformat(),
                'tracked_positions': {},
                'archived_positions': []  # Last 100 closed positions for ML
            }

            # Convert tracked positions to JSON-serializable format
            for ticket, position in self.tracked_positions.items():
                pos_data = self._convert_position_datetimes(position)
                state['tracked_positions'][str(ticket)] = pos_data

            # Add archived positions (already converted during archival in reconcile_with_mt5)
            state['archived_positions'] = self.archived_positions

            # Convert numpy types to native Python types
            state = convert_numpy_types(state)

            # Write to temp file first (atomic write for Windows)
            temp_path = state_path.with_suffix('.json.tmp')
            try:
                with open(temp_path, 'w', encoding='utf-8', newline='') as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)

                # Atomic rename (Windows-safe)
                if state_path.exists():
                    state_path.unlink()
                temp_path.rename(state_path)
            except Exception as write_error:
                # Cleanup temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()
                raise write_error

            return True

        except Exception as e:
            print(f"[WARN]  Failed to save recovery state: {e}")
            return False

    def load_state(self, state_file: str = "data/recovery_state.json") -> bool:
        """
        Load tracking state from JSON file after crash/restart.

        Args:
            state_file: Path to state file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            state_path = Path(state_file)

            if not state_path.exists():
                print(f"ℹ️  No saved state found at {state_file}")
                return False

            # Load JSON with Windows-compatible encoding and error handling
            try:
                with open(state_path, 'r', encoding='utf-8', newline='') as f:
                    state = json.load(f)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Corrupted recovery state JSON at line {e.lineno}, col {e.colno}")
                print(f"        {e.msg}")
                # Backup corrupted file
                backup_path = state_path.parent / f"{state_path.stem}.corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                state_path.rename(backup_path)
                print(f"[OK] Corrupted file moved to: {backup_path}")
                print(f"[OK] Will reconstruct positions from MT5")
                return False
            except UnicodeDecodeError as e:
                print(f"[ERROR] Encoding error reading state file: {e}")
                backup_path = state_path.parent / f"{state_path.stem}.bad_encoding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                state_path.rename(backup_path)
                print(f"[OK] Bad file moved to: {backup_path}")
                return False

            # Check state age
            state_time = datetime.fromisoformat(state['timestamp'])
            age = datetime.now() - state_time
            age_hours = age.total_seconds() / 3600

            print(f"\n📂 Loading recovery state from {state['timestamp']}")
            print(f"   State age: {age_hours:.1f} hours")

            if age_hours > 24:
                print(f"[WARN]  State is over 24 hours old - may be stale")
                print(f"   Recommend manual verification")

            # Restore tracked positions
            restored_count = 0
            for ticket_str, pos_data in state['tracked_positions'].items():
                ticket = int(ticket_str)

                # Convert ISO strings back to datetime objects
                if 'open_time' in pos_data and pos_data['open_time']:
                    try:
                        pos_data['open_time'] = datetime.fromisoformat(pos_data['open_time'])
                    except:
                        pos_data['open_time'] = get_current_time()

                # Convert cooldown timer fields (added for race condition fix)
                if 'last_hedge_time' in pos_data and pos_data['last_hedge_time']:
                    try:
                        pos_data['last_hedge_time'] = datetime.fromisoformat(pos_data['last_hedge_time'])
                    except:
                        pos_data['last_hedge_time'] = None

                if 'last_grid_time' in pos_data and pos_data['last_grid_time']:
                    try:
                        pos_data['last_grid_time'] = datetime.fromisoformat(pos_data['last_grid_time'])
                    except:
                        pos_data['last_grid_time'] = None

                # Convert time fields in recovery orders
                for grid in pos_data.get('grid_levels', []):
                    if 'time' in grid and grid['time']:
                        try:
                            grid['time'] = datetime.fromisoformat(grid['time'])
                        except:
                            grid['time'] = None

                for hedge in pos_data.get('hedge_tickets', []):
                    if 'time' in hedge and hedge['time']:
                        try:
                            hedge['time'] = datetime.fromisoformat(hedge['time'])
                        except:
                            hedge['time'] = None

                for dca in pos_data.get('dca_levels', []):
                    if 'time' in dca and dca['time']:
                        try:
                            dca['time'] = datetime.fromisoformat(dca['time'])
                        except:
                            dca['time'] = None

                self.tracked_positions[ticket] = pos_data
                restored_count += 1

            # Load archived positions (v2.0+, backward compatible)
            archived_count = 0
            if 'archived_positions' in state:
                self.archived_positions = state['archived_positions']
                archived_count = len(self.archived_positions)

            print(f"[OK] Restored {restored_count} tracked positions from state file")
            if archived_count > 0:
                print(f"[OK] Loaded {archived_count} archived closed positions")
            print()

            return True

        except Exception as e:
            print(f"[WARN]  Failed to load recovery state: {e}")
            import traceback
            traceback.print_exc()
            return False

    def reconcile_with_mt5(self, mt5_manager) -> tuple:
        """
        Reconcile tracked positions with MT5 reality after loading state.
        Critical for crash recovery - ensures state matches actual open positions.

        Args:
            mt5_manager: MT5Manager instance for querying positions

        Returns:
            tuple: (positions_added, positions_removed, positions_validated)
        """
        # Get all open positions from MT5 with our magic number
        mt5_positions = mt5_manager.get_positions()
        mt5_tickets = {pos['ticket'] for pos in mt5_positions}
        tracked_tickets = set(self.tracked_positions.keys())

        # Find discrepancies
        closed_tickets = tracked_tickets - mt5_tickets  # Tracked but closed in MT5
        new_tickets = mt5_tickets - tracked_tickets      # Open in MT5 but not tracked

        # Archive closed positions (don't delete - preserve ML data)
        for ticket in closed_tickets:
            pos = self.tracked_positions[ticket]
            orphan_str = f" (orphaned {pos.get('orphan_source', 'unknown')})" if pos.get('is_orphaned') else ""

            # Add closure metadata
            pos['closed_time'] = get_current_time()  # datetime object (will be converted)
            pos['status'] = 'closed'

            # Convert all datetime objects to ISO strings for JSON serialization
            pos_serializable = self._convert_position_datetimes(pos)

            # Archive for ML analysis (now safe for JSON)
            self.archived_positions.append(pos_serializable)

            # Remove from active tracking
            del self.tracked_positions[ticket]

            print(f"   📦 Archived closed: {ticket} ({pos['symbol']} {pos['type']}){orphan_str}")

        # Prune archive to last 100 positions (keep state file manageable)
        if len(self.archived_positions) > 100:
            removed_count = len(self.archived_positions) - 100
            self.archived_positions = self.archived_positions[-100:]
            print(f"   🧹 Pruned {removed_count} old archived positions (keeping last 100)")

        # Add new MT5 positions to tracking
        for pos in mt5_positions:
            if pos['ticket'] in new_tickets:
                self.tracked_positions[pos['ticket']] = {
                    'ticket': pos['ticket'],
                    'symbol': pos['symbol'],
                    'entry_price': pos['price_open'],
                    'type': pos['type'],
                    'initial_volume': pos['volume'],
                    'grid_levels': [],
                    'hedge_tickets': [],
                    'dca_levels': [],
                    'total_volume': pos['volume'],
                    'max_underwater_pips': 0,
                    'recovery_active': False,
                    'open_time': pos['time'],
                    'is_orphaned': False,
                    'last_hedge_time': None,
                    'last_grid_time': None,
                }
                print(f"   ➕ Added new: {pos['ticket']} ({pos['symbol']} {pos['type']} @ {pos['price_open']})")

        # Clean up orphaned positions - convert to standalone
        orphaned = [
            (ticket, pos) for ticket, pos in self.tracked_positions.items()
            if pos.get('is_orphaned', False)
        ]

        if orphaned:
            print(f"   ⚠️  Found {len(orphaned)} orphaned recovery positions")
            for ticket, pos in orphaned:
                # Convert orphans to standalone positions (parent likely closed)
                pos['is_orphaned'] = False
                pos['orphan_source'] = None
                # Keep recovery data but mark as inactive
                pos['recovery_active'] = False

        validated = len(tracked_tickets & mt5_tickets)

        return len(new_tickets), len(closed_tickets), validated

    def check_grid_trigger(
        self,
        ticket: int,
        current_price: float,
        pip_value: float = 0.0001
    ) -> Optional[Dict]:
        """
        Check if we should add a grid level (pyramid strategy)

        RACE CONDITION FIXES APPLIED:
        - Thread lock for atomic operations
        - Pending flag to track in-flight grids
        - Cooldown timer (2 minutes between grids)
        - Enhanced logging for debugging

        Grid creates OVERLAPPING positions in the same direction to lock in profit
        while maintaining exposure. When parent position is in profit, opens new
        positions at current market price.

        Example: BUY @ 1.1000 moves to 1.1010 (+10 pips)
                 -> Opens Grid 1: BUY @ 1.1010 (new independent position)
                 Parent continues, reaches 1.1020 (+20 pips)
                 -> Opens Grid 2: BUY @ 1.1020 (another new position)
                 Result: 3 overlapping BUY positions (max_grid_levels = 2)

        Args:
            ticket: Position ticket
            current_price: Current market price
            pip_value: Pip value for symbol (0.0001 for most pairs)

        Returns:
            Dict with grid order details or None
        """
        if not GRID_ENABLED:
            logger.debug(f"Grid disabled globally")
            return None

        if ticket not in self.tracked_positions:
            logger.debug(f"Ticket {ticket} not tracked")
            return None

        # CRITICAL FIX #1: Thread lock for atomic grid operations
        # Create lock for this position if it doesn't exist
        if ticket not in self.hedge_locks:  # Reuse hedge_locks for all recovery operations
            self.hedge_locks[ticket] = threading.Lock()

        # Acquire lock to ensure only one grid check runs at a time for this position
        with self.hedge_locks[ticket]:
            position = self.tracked_positions[ticket]
            symbol = position['symbol']

            # ENHANCED LOGGING: Log current grid state for debugging
            logger.debug(f"[GRID CHECK] Position {ticket} ({symbol}):")
            logger.debug(f"  Current grid count: {len(position['grid_levels'])}")
            logger.debug(f"  Max grids allowed: {self._get_recovery_settings(symbol)['max_grid_levels']}")
            logger.debug(f"  Grids with tickets: {len([g for g in position['grid_levels'] if 'ticket' in g and g.get('ticket') is not None])}")
            logger.debug(f"  Pending grids: {len([g for g in position['grid_levels'] if g.get('pending', False)])}")
            for i, g in enumerate(position['grid_levels']):
                ticket_str = g.get('ticket', 'NONE')
                pending_str = 'PENDING' if g.get('pending', False) else 'PLACED'
                logger.debug(f"    Grid {i+1}: ticket={ticket_str}, price={g.get('price')}, status={pending_str}")

            # SAFETY: Prevent grid-on-grid for orphaned positions
            if position.get('is_orphaned', False):
                logger.debug(f"Grid: Position {ticket} is orphaned - NO grid-on-grid allowed (prevent cascade)")
                return None

            # SAFETY: Prevent grid spawning from grid children (stops cascade)
            # Grid daughters can have DCA + Hedge + Partial Close, but NO more grids
            if position.get('is_grid_child', False):
                logger.debug(f"Grid: Position {ticket} is a grid child - NO more grids allowed (prevent cascade)")
                return None

            # Get instrument-specific recovery settings
            recovery_settings = self._get_recovery_settings(symbol)
            grid_spacing = recovery_settings['grid_spacing_pips']
            max_grid_levels = recovery_settings['max_grid_levels']

            # Check if maxed out grid levels
            if len(position['grid_levels']) >= max_grid_levels:
                logger.debug(f"Grid: Max levels reached for {ticket} ({len(position['grid_levels'])}/{max_grid_levels})")
                return None

            # CRITICAL FIX #2: Enhanced pending grid check with explicit 'pending' flag
            # This prevents race condition where multiple grid actions are triggered
            # before any orders are actually placed
            pending_grids = [g for g in position['grid_levels']
                           if g.get('pending', False) or 'ticket' not in g or g.get('ticket') is None]
            if pending_grids:
                logger.info(f"[GRID BLOCKED] {ticket} already has {len(pending_grids)} pending grid order(s), skipping to prevent duplicate")
                return None

            # CRITICAL FIX #3: Cooldown timer (2 minutes between grids)
            # Shorter than hedge cooldown since grids are triggered by profit (safer)
            last_grid_time = position.get('last_grid_time')
            if last_grid_time:
                time_since_last_grid = (get_current_time() - last_grid_time).total_seconds()
                if time_since_last_grid < 120:  # 2 minute cooldown
                    logger.debug(f"[GRID BLOCKED] {ticket} cooldown active: {time_since_last_grid:.0f}s since last grid (need 120s)")
                    return None
                else:
                    logger.debug(f"[GRID CHECK] {ticket} cooldown expired ({time_since_last_grid:.0f}s), grid allowed")

            entry_price = position['entry_price']
            position_type = position['type']

            # Calculate pips moved
            if position_type == 'buy':
                pips_moved = (entry_price - current_price) / pip_value
            else:
                pips_moved = (current_price - entry_price) / pip_value

            # KILL SWITCH: Check if negative grid is disabled
            if DISABLE_NEGATIVE_GRID and pips_moved > 0:
                # Position is underwater (negative) and negative grid is disabled
                # pips_moved > 0 means losing money
                logger.debug(f"Grid: Ticket {ticket} underwater ({pips_moved:.1f} pips) - negative grid disabled")
                return None

            # Grid should only trigger when IN PROFIT (positive grid)
            # pips_moved <= 0 means position is in profit (price moved favorably)
            # pips_moved > 0 means position is underwater (price moved unfavorably)
            if pips_moved > 0:
                # Position is underwater - no grid on losing trades
                logger.debug(f"Grid: Ticket {ticket} underwater ({pips_moved:.1f} pips) - no grid on losing trades")
                return None

            # Position is in profit - calculate grid spacing in the profit direction
            pips_in_profit = abs(pips_moved)  # Convert to positive value

            # Calculate expected grid levels based on profit
            expected_levels = int(pips_in_profit / grid_spacing) + 1

            # Need to add grid level?
            if expected_levels > len(position['grid_levels']) + 1:  # +1 for original position
                logger.info(f"[GRID TRIGGER] Position {ticket} in profit {pips_in_profit:.1f} pips, adding Grid L{len(position['grid_levels']) + 1}")

                # Calculate grid price in the PROFIT direction
                # Positive grid = partial profit taking as price moves favorably
                levels_added = len(position['grid_levels']) + 1
                grid_distance = grid_spacing * levels_added * pip_value

                # Calculate grid entry price (current market price)
                # Pyramid strategy: Open new position in SAME direction at current price
                # For BUY: Price went UP (profit) -> Open new BUY at current higher price
                # For SELL: Price went DOWN (profit) -> Open new SELL at current lower price
                if position_type == 'buy':
                    grid_price = entry_price + grid_distance  # Current price (higher)
                else:
                    grid_price = entry_price - grid_distance  # Current price (lower)

                # Round grid volume to broker step size
                grid_volume = round_volume_to_step(GRID_LOT_SIZE)

                # Pyramid grid = SAME direction (overlapping positions)
                # BUY position in profit -> Open new BUY to pyramid
                # SELL position in profit -> Open new SELL to pyramid
                grid_type = position_type  # SAME direction as parent

                # CRITICAL FIX #4: Add to tracked levels with PENDING flag
                position['grid_levels'].append({
                    'price': grid_price,
                    'volume': grid_volume,
                    'type': grid_type,
                    'time': get_current_time(),
                    'pending': True,  # CRITICAL: Mark as pending until ticket is assigned
                })

                # Update last grid time for cooldown
                position['last_grid_time'] = get_current_time()

                position['total_volume'] += grid_volume
                # Note: Grid is PYRAMID strategy - overlapping positions in same direction
                # Locks in profit while maintaining exposure to continuation

                logger.info(f"[GRID] Grid L{len(position['grid_levels'])} activated for {ticket} (marked as PENDING)")
                logger.info(f"   Parent: {position_type.upper()} at {entry_price:.5f}")
                logger.info(f"   Grid: {grid_type.upper()} at {grid_price:.5f}")
                logger.info(f"   Distance: {grid_spacing * levels_added:.1f} pips IN PROFIT")
                logger.info(f"   Status: PENDING (waiting for ticket assignment)")
                logger.info(f"   Cooldown: 2 minutes from now")

                # Use last 5 digits of ticket for cleaner comment (MT5 31 char limit)
                short_ticket = str(ticket)[-5:]
                return {
                    'action': 'grid',
                    'original_ticket': ticket,  # Track which position this belongs to
                    'symbol': position['symbol'],
                    'type': grid_type,  # SAME direction for pyramid
                    'price': grid_price,
                    'volume': grid_volume,
                    'comment': f'Grid L{len(position["grid_levels"])} - {short_ticket}'
                }

            # Not enough profit for grid yet
            logger.debug(f"Grid: {ticket} profit not enough ({pips_in_profit:.1f}/{grid_spacing * (len(position['grid_levels']) + 1):.1f} pips)")
            return None

    def check_hedge_trigger(
        self,
        ticket: int,
        current_price: float,
        pip_value: float = 0.0001
    ) -> Optional[Dict]:
        """
        Check if we should activate a hedge

        RACE CONDITION FIXES APPLIED:
        - Thread lock for atomic operations
        - Pending flag to track in-flight hedges
        - Cooldown timer (5 minutes between hedges)
        - Enhanced logging for debugging

        Args:
            ticket: Position ticket
            current_price: Current market price
            pip_value: Pip value for symbol

        Returns:
            Dict with hedge order details or None
        """
        if not HEDGE_ENABLED or ticket not in self.tracked_positions:
            return None

        # CRITICAL FIX #1: Thread lock for atomic hedge operations
        # Create lock for this position if it doesn't exist
        if ticket not in self.hedge_locks:
            self.hedge_locks[ticket] = threading.Lock()

        # Acquire lock to ensure only one hedge check runs at a time for this position
        with self.hedge_locks[ticket]:
            position = self.tracked_positions[ticket]
            symbol = position['symbol']

            # ENHANCED LOGGING: Log current hedge state for debugging
            logger.debug(f"[HEDGE CHECK] Position {ticket} ({symbol}):")
            logger.debug(f"  Current hedge count: {len(position['hedge_tickets'])}")
            logger.debug(f"  Max hedges allowed: {MAX_HEDGES_PER_POSITION}")
            logger.debug(f"  Hedges with tickets: {len([h for h in position['hedge_tickets'] if 'ticket' in h and h.get('ticket') is not None])}")
            logger.debug(f"  Pending hedges: {len([h for h in position['hedge_tickets'] if h.get('pending', False)])}")
            for i, h in enumerate(position['hedge_tickets']):
                ticket_str = h.get('ticket', 'NONE')
                pending_str = 'PENDING' if h.get('pending', False) else 'PLACED'
                logger.debug(f"    Hedge {i+1}: ticket={ticket_str}, time={h.get('time')}, status={pending_str}")

            # CRITICAL: Prevent hedge-on-hedge and DCA-on-hedge cascades
            # Orphaned positions (hedge/DCA that lost parent) should NOT trigger their own recovery
            # This prevents the 7-hedge cascade bug where each hedge triggers another hedge
            if position.get('is_orphaned', False):
                orphan_source = position.get('orphan_source', 'unknown')
                logger.debug(f"Hedge: Position {ticket} is orphaned ({orphan_source}) - NO hedge-on-hedge allowed (prevent cascade)")
                return None

            # Get instrument-specific recovery settings
            recovery_settings = self._get_recovery_settings(symbol)
            hedge_trigger = recovery_settings['hedge_trigger_pips']

            # Check if already hedged (LIMIT: prevents cascade)
            # Count both placed hedges AND pending hedges (without ticket number)
            hedge_count = len(position['hedge_tickets'])
            if hedge_count >= MAX_HEDGES_PER_POSITION:
                logger.debug(f"Hedge: Max hedges reached for {ticket} ({hedge_count}/{MAX_HEDGES_PER_POSITION})")
                return None

            # CRITICAL FIX #2: Enhanced pending hedge check with explicit 'pending' flag
            # This prevents race condition where multiple hedge actions are triggered
            # before any orders are actually placed
            pending_hedges = [h for h in position['hedge_tickets']
                            if h.get('pending', False) or 'ticket' not in h or h.get('ticket') is None]
            if pending_hedges:
                logger.info(f"[HEDGE BLOCKED] {ticket} already has {len(pending_hedges)} pending hedge order(s), skipping to prevent duplicate")
                return None

            # CRITICAL FIX #3: Cooldown timer (5 minutes between hedges)
            # Prevents rapid-fire hedge creation even if other checks fail
            last_hedge_time = position.get('last_hedge_time')
            if last_hedge_time:
                time_since_last_hedge = (get_current_time() - last_hedge_time).total_seconds()
                if time_since_last_hedge < 300:  # 5 minute cooldown
                    logger.debug(f"[HEDGE BLOCKED] {ticket} cooldown active: {time_since_last_hedge:.0f}s since last hedge (need 300s)")
                    return None
                else:
                    logger.debug(f"[HEDGE CHECK] {ticket} cooldown expired ({time_since_last_hedge:.0f}s), hedge allowed")

            entry_price = position['entry_price']
            position_type = position['type']

            # Calculate pips underwater
            if position_type == 'buy':
                pips_underwater = (entry_price - current_price) / pip_value
            else:
                pips_underwater = (current_price - entry_price) / pip_value

            # SAFETY CHECK: Hedge is ONLY for underwater (losing) positions
            # If in profit, no hedge needed
            if pips_underwater <= 0:
                # Position is in profit - no hedge
                logger.debug(f"Hedge: {ticket} in profit ({pips_underwater:.1f} pips), no hedge needed")
                return None

            # Update max underwater
            if pips_underwater > position['max_underwater_pips']:
                position['max_underwater_pips'] = pips_underwater

            # Check if trigger reached
            if pips_underwater >= hedge_trigger:
                logger.info(f"[HEDGE TRIGGER] Position {ticket} underwater {pips_underwater:.1f} pips (trigger: {hedge_trigger} pips)")

                # Calculate hedge volume (overhedge) - based on INITIAL volume, not total
                # Original EA hedges the initial position size, not accumulated grid/DCA
                hedge_volume = position['initial_volume'] * HEDGE_RATIO

                # Apply maximum hedge volume cap for safety
                if hedge_volume > MAX_HEDGE_VOLUME:
                    logger.warning(f"[HEDGE] Volume capped: {hedge_volume:.2f} -> {MAX_HEDGE_VOLUME:.2f} (MAX_HEDGE_VOLUME)")
                    print(f"   [WARN] Hedge volume capped from {hedge_volume:.2f} to {MAX_HEDGE_VOLUME:.2f}")
                    hedge_volume = MAX_HEDGE_VOLUME

                # Round to broker step size (0.01)
                hedge_volume = round_volume_to_step(hedge_volume)

                # Opposite direction
                hedge_type = 'sell' if position_type == 'buy' else 'buy'

                # CRITICAL FIX #4: Mark hedge as PENDING with explicit flag
                # This ensures the next check will see this hedge as pending
                position['hedge_tickets'].append({
                    'type': hedge_type,
                    'volume': hedge_volume,
                    'trigger_pips': pips_underwater,
                    'time': get_current_time(),
                    'pending': True,  # CRITICAL: Mark as pending until ticket is assigned
                })

                # Update last hedge time for cooldown
                position['last_hedge_time'] = get_current_time()

                position['recovery_active'] = True

                logger.info(f"[HEDGE] Hedge #{len(position['hedge_tickets'])} activated for {ticket} (marked as PENDING)")
                print(f"[HEDGE] Hedge activated for {ticket}")
                print(f"   Original: {position_type.upper()} {position['initial_volume']:.2f} (total exposure: {position['total_volume']:.2f})")
                print(f"   Hedge: {hedge_type.upper()} {hedge_volume:.2f} (ratio: {HEDGE_RATIO}x on initial)")
                print(f"   Triggered at: {pips_underwater:.1f} pips underwater")
                print(f"   Status: PENDING (waiting for ticket assignment)")
                print(f"   Cooldown: 5 minutes from now")

                # Use last 5 digits of ticket for cleaner comment (MT5 31 char limit)
                short_ticket = str(ticket)[-5:]
                return {
                    'action': 'hedge',
                    'original_ticket': ticket,  # Track which position this belongs to
                    'symbol': position['symbol'],
                    'type': hedge_type,
                    'volume': hedge_volume,
                    'comment': f'Hedge - {short_ticket}'
                }

            # Not underwater enough for hedge
            logger.debug(f"Hedge: {ticket} not deep enough ({pips_underwater:.1f}/{hedge_trigger} pips)")
            return None

    def check_dca_trigger(
        self,
        ticket: int,
        current_price: float,
        pip_value: float = 0.0001
    ) -> Optional[Dict]:
        """
        Check if we should add DCA/Martingale level

        Args:
            ticket: Position ticket
            current_price: Current market price
            pip_value: Pip value for symbol

        Returns:
            Dict with DCA order details or None
        """
        if not DCA_ENABLED or ticket not in self.tracked_positions:
            return None

        position = self.tracked_positions[ticket]
        symbol = position['symbol']

        # CRITICAL: Prevent DCA-on-hedge and DCA-on-DCA cascades
        # Orphaned positions (hedge/DCA that lost parent) should NOT trigger their own recovery
        # This prevents cascade bugs where recovery positions spawn more recovery
        if position.get('is_orphaned', False):
            orphan_source = position.get('orphan_source', 'unknown')
            logger.debug(f"DCA: Position {ticket} is orphaned ({orphan_source}) - NO DCA-on-orphan allowed (prevent cascade)")
            return None

        # Get instrument-specific recovery settings
        recovery_settings = self._get_recovery_settings(symbol)
        dca_trigger = recovery_settings['dca_trigger_pips']
        dca_multiplier = recovery_settings['dca_multiplier']
        max_dca_levels = recovery_settings['max_dca_levels']

        # Check if maxed out DCA levels (LIMIT: prevents cascade)
        if max_dca_levels and len(position['dca_levels']) >= max_dca_levels:
            logger.debug(f"DCA: Max levels reached for {ticket} ({len(position['dca_levels'])}/{max_dca_levels})")
            return None

        # CRITICAL: Check if there's already a pending DCA (no ticket yet)
        # This prevents race condition where multiple DCA actions are triggered
        # before any orders are actually placed
        pending_dcas = [d for d in position['dca_levels'] if 'ticket' not in d or d.get('ticket') is None]
        if pending_dcas:
            logger.debug(f"DCA: {ticket} already has pending DCA order, skipping")
            return None

        entry_price = position['entry_price']
        position_type = position['type']

        # Calculate pips moved
        if position_type == 'buy':
            pips_moved = (entry_price - current_price) / pip_value
        else:
            pips_moved = (current_price - entry_price) / pip_value

        # SAFETY CHECK: DCA is ONLY for underwater (losing) positions
        # If in profit, no DCA needed
        if pips_moved <= 0:
            # Position is in profit - no DCA
            logger.debug(f"DCA: {ticket} in profit ({pips_moved:.1f} pips), no DCA needed")
            return None

        # Check if underwater enough for DCA trigger
        if pips_moved < dca_trigger:
            logger.debug(f"DCA: {ticket} not underwater enough ({pips_moved:.1f}/{dca_trigger} pips)")
            return None

        # Calculate expected DCA levels
        expected_levels = int(pips_moved / dca_trigger)

        # Need to add DCA level?
        if expected_levels > len(position['dca_levels']):
            # Calculate DCA volume (increase by multiplier)
            if len(position['dca_levels']) == 0:
                dca_volume = position['initial_volume'] * dca_multiplier
            else:
                last_dca = position['dca_levels'][-1]
                dca_volume = last_dca['volume'] * dca_multiplier

            # Round to broker step size (0.01)
            dca_volume = round_volume_to_step(dca_volume)

            # Add to tracked levels
            position['dca_levels'].append({
                'price': current_price,
                'volume': dca_volume,
                'level': len(position['dca_levels']) + 1,
                'time': get_current_time()
            })

            position['total_volume'] += dca_volume
            position['recovery_active'] = True

            print(f" DCA Level {len(position['dca_levels'])} triggered for {ticket}")
            print(f"   Price: {current_price:.5f}")
            print(f"   Volume: {dca_volume:.2f} (multiplier: {dca_multiplier}x)")
            print(f"   Total volume now: {position['total_volume']:.2f}")

            # Use last 5 digits of ticket for cleaner comment (MT5 31 char limit)
            short_ticket = str(ticket)[-5:]
            return {
                'action': 'dca',
                'original_ticket': ticket,  # Track which position this belongs to
                'symbol': position['symbol'],
                'type': position_type,  # Same direction
                'volume': dca_volume,
                'comment': f'DCA L{len(position["dca_levels"])} - {short_ticket}'
            }

        return None

    def check_all_recovery_triggers(
        self,
        ticket: int,
        current_price: float,
        pip_value: float = 0.0001
    ) -> List[Dict]:
        """
        Check all recovery mechanisms at once

        Args:
            ticket: Position ticket
            current_price: Current price
            pip_value: Pip value for symbol

        Returns:
            List of recovery actions to take
        """
        actions = []

        # Check grid
        grid_action = self.check_grid_trigger(ticket, current_price, pip_value)
        if grid_action:
            actions.append(grid_action)

        # Check hedge
        hedge_action = self.check_hedge_trigger(ticket, current_price, pip_value)
        if hedge_action:
            actions.append(hedge_action)

        # Check DCA
        dca_action = self.check_dca_trigger(ticket, current_price, pip_value)
        if dca_action:
            actions.append(dca_action)

        return actions

    def get_position_status(self, ticket: int) -> Optional[Dict]:
        """
        Get recovery status for a position

        Args:
            ticket: Position ticket

        Returns:
            Dict with position recovery status
        """
        if ticket not in self.tracked_positions:
            return None

        position = self.tracked_positions[ticket]

        return {
            'ticket': ticket,
            'symbol': position['symbol'],
            'entry_price': position['entry_price'],
            'type': position['type'],
            'initial_volume': position['initial_volume'],
            'current_volume': position['total_volume'],
            'grid_levels': len(position['grid_levels']),
            'hedges_active': len(position['hedge_tickets']),
            'dca_levels': len(position['dca_levels']),
            'max_underwater_pips': position['max_underwater_pips'],
            'recovery_active': position['recovery_active'],
        }

    def get_all_positions_status(self) -> List[Dict]:
        """Get status for all tracked positions"""
        return [self.get_position_status(ticket) for ticket in self.tracked_positions.keys()]

    def calculate_breakeven_price(self, ticket: int) -> Optional[float]:
        """
        Calculate breakeven price considering all grid/DCA levels and hedges

        NOTE: This calculates the breakeven for SAME-DIRECTION positions only.
        Hedges (opposite direction) are not included in breakeven calculation
        as they need to be closed separately.

        Args:
            ticket: Position ticket

        Returns:
            float: Breakeven price or None
        """
        if ticket not in self.tracked_positions:
            return None

        position = self.tracked_positions[ticket]

        total_volume = position['initial_volume']
        weighted_price = position['entry_price'] * position['initial_volume']

        # Add grid levels (same direction as original)
        for grid_level in position['grid_levels']:
            total_volume += grid_level['volume']
            weighted_price += grid_level['price'] * grid_level['volume']

        # Add DCA levels (same direction as original)
        for dca_level in position['dca_levels']:
            total_volume += dca_level['volume']
            weighted_price += dca_level['price'] * dca_level['volume']

        # NOTE: Hedges are opposite direction and should be tracked separately
        # They don't factor into the same-direction breakeven calculation
        # The net P&L calculation in calculate_net_profit() handles the full picture

        if total_volume == 0:
            return None

        breakeven = weighted_price / total_volume
        return breakeven

    def get_all_stack_tickets(self, ticket: int) -> List[int]:
        """
        Get all ticket numbers in a recovery stack (original + grid + hedge + DCA)

        Args:
            ticket: Original position ticket

        Returns:
            List[int]: All ticket numbers in the stack
        """
        if ticket not in self.tracked_positions:
            return [ticket]  # Just the original

        position = self.tracked_positions[ticket]
        tickets = [ticket]  # Start with original

        # Add grid tickets
        for grid_level in position['grid_levels']:
            if 'ticket' in grid_level:
                tickets.append(grid_level['ticket'])

        # Add hedge tickets
        for hedge_info in position['hedge_tickets']:
            if 'ticket' in hedge_info:
                tickets.append(hedge_info['ticket'])

        # Add DCA tickets
        for dca_level in position['dca_levels']:
            if 'ticket' in dca_level:
                tickets.append(dca_level['ticket'])

        return tickets

    def calculate_net_profit(self, ticket: int, mt5_positions: List[Dict]) -> Optional[float]:
        """
        Calculate net profit/loss for entire recovery stack

        Args:
            ticket: Original position ticket
            mt5_positions: List of all current MT5 positions

        Returns:
            float: Net profit in account currency, or None if error
        """
        if ticket not in self.tracked_positions:
            return None

        # Get all tickets in this stack
        stack_tickets = self.get_all_stack_tickets(ticket)

        # Calculate total P&L across all positions in stack
        total_profit = 0.0

        for mt5_pos in mt5_positions:
            if mt5_pos['ticket'] in stack_tickets:
                total_profit += mt5_pos.get('profit', 0.0)

        return total_profit

    def check_profit_target(
        self,
        ticket: int,
        mt5_positions: List[Dict],
        account_balance: float,
        profit_percent: float = 1.0
    ) -> bool:
        """
        Check if position stack reached profit target

        Args:
            ticket: Original position ticket
            mt5_positions: List of all current MT5 positions
            account_balance: Account balance
            profit_percent: Profit target as % of balance (default 1.0%)

        Returns:
            bool: True if profit target reached
        """
        net_profit = self.calculate_net_profit(ticket, mt5_positions)

        if net_profit is None:
            return False

        # Calculate target profit in dollars
        target_profit = account_balance * (profit_percent / 100.0)

        if net_profit >= target_profit:
            print(f"[OK] Profit target reached for {ticket}")
            print(f"   Net profit: ${net_profit:.2f}")
            print(f"   Target: ${target_profit:.2f} ({profit_percent}% of ${account_balance:.2f})")
            return True

        return False

    def check_time_limit(self, ticket: int, hours_limit: int = 4) -> bool:
        """
        Check if position has been open too long

        Args:
            ticket: Original position ticket
            hours_limit: Maximum hours before auto-close (default 4)

        Returns:
            bool: True if time limit exceeded
        """
        if ticket not in self.tracked_positions:
            return False

        position = self.tracked_positions[ticket]
        open_time = position.get('open_time')

        if open_time is None:
            return False

        # Calculate hours open
        time_open = get_current_time() - open_time
        hours_open = time_open.total_seconds() / 3600

        if hours_open >= hours_limit:
            print(f" Time limit reached for {ticket}")
            print(f"   Open for: {hours_open:.1f} hours")
            print(f"   Limit: {hours_limit} hours")
            print(f"   Auto-closing stuck position...")
            return True

        return False

    def check_stack_drawdown(
        self,
        ticket: int,
        mt5_positions: List[Dict],
        pip_value: float = 0.0001
    ) -> bool:
        """
        Check if recovery stack has exceeded drawdown threshold.
        Kills entire stack (original + grid + hedge + DCA) if net loss exceeds
        STACK_DRAWDOWN_MULTIPLIER × expected profit from original trade.

        Args:
            ticket: Original position ticket
            mt5_positions: List of all current MT5 positions
            pip_value: Pip value for symbol (0.0001 for most pairs, 0.01 for JPY)

        Returns:
            bool: True if stack should be closed due to excessive drawdown
        """
        if ticket not in self.tracked_positions:
            return False

        position = self.tracked_positions[ticket]
        symbol = position['symbol']
        initial_volume = position['initial_volume']

        # Calculate expected profit from original trade
        try:
            tp_settings = get_take_profit_settings(symbol)
            tp_pips = tp_settings['take_profit_pips']
        except (KeyError, Exception):
            # If no TP settings, can't calculate - skip check
            return False

        # Calculate expected profit in dollars
        # Formula: pips × pip_value × lot_size × 100,000 (standard lot)
        expected_profit = tp_pips * pip_value * initial_volume * 100000

        # Calculate drawdown threshold
        drawdown_threshold = -1 * (expected_profit * STACK_DRAWDOWN_MULTIPLIER)

        # Calculate current net P&L for entire stack
        net_profit = self.calculate_net_profit(ticket, mt5_positions)

        if net_profit is None:
            return False

        # Check if we've exceeded drawdown threshold (net profit is negative and below threshold)
        if net_profit <= drawdown_threshold:
            print(f"🛑 STACK DRAWDOWN EXCEEDED for {ticket}")
            print(f"   Symbol: {symbol}")
            print(f"   Expected profit: ${expected_profit:.2f}")
            print(f"   Drawdown threshold: ${drawdown_threshold:.2f} ({STACK_DRAWDOWN_MULTIPLIER}x)")
            print(f"   Current stack P&L: ${net_profit:.2f}")
            print(f"   [WARN]  Killing entire recovery stack to limit losses")
            return True

        return False

    def check_stack_stop_loss(
        self,
        ticket: int,
        mt5_positions: List[Dict],
        current_adx: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Check if recovery stack has exceeded per-stack stop loss limit.
        Monitors dollar-based loss limits to prevent catastrophic drawdowns.

        Uses different limits based on whether hedge is active:
        - DCA-only: DCA_ONLY_MAX_LOSS (e.g., -$25)
        - DCA+Hedge: DCA_HEDGE_MAX_LOSS (e.g., -$50)

        Args:
            ticket: Original position ticket
            mt5_positions: List of all current MT5 positions
            current_adx: Current ADX value for cascade detection (optional)

        Returns:
            Dict with stop loss info if triggered, None otherwise
        """
        # Check if stack stops are enabled
        if not ENABLE_STACK_STOPS:
            return None

        if ticket not in self.tracked_positions:
            return None

        position = self.tracked_positions[ticket]
        symbol = position['symbol']

        # Calculate current net P&L for entire stack
        net_profit = self.calculate_net_profit(ticket, mt5_positions)

        if net_profit is None:
            return None

        # Determine if hedge is active
        has_hedge = len(position.get('hedge_tickets', [])) > 0
        has_dca = len(position.get('dca_levels', [])) > 0

        # Determine appropriate stop loss limit
        if has_hedge:
            # Hedge is active - use DCA+Hedge limit
            stop_loss_limit = DCA_HEDGE_MAX_LOSS
            stack_type = "DCA+Hedge"
        elif has_dca:
            # Only DCA active, no hedge - use DCA-only limit
            stop_loss_limit = DCA_ONLY_MAX_LOSS
            stack_type = "DCA-only"
        else:
            # No recovery active - use DCA-only limit (most conservative)
            stop_loss_limit = DCA_ONLY_MAX_LOSS
            stack_type = "Initial"

        # Check if we've exceeded stop loss limit
        if net_profit <= stop_loss_limit:
            print(f"[STOP] PER-STACK STOP LOSS HIT for position #{ticket}")
            print(f"   Symbol: {symbol}")
            print(f"   Stack type: {stack_type}")
            print(f"   Current P&L: ${net_profit:.2f}")
            print(f"   Stop loss limit: ${stop_loss_limit:.2f}")
            print(f"   Loss exceeded by: ${abs(net_profit - stop_loss_limit):.2f}")
            print(f"   DCA levels: {len(position.get('dca_levels', []))}")
            print(f"   Hedges: {len(position.get('hedge_tickets', []))}")
            print(f"   [ACTION] Closing entire recovery stack to limit losses")

            # Record stop-out event for cascade detection
            if self.stop_out_tracker:
                self.stop_out_tracker.add_stop_out(
                    ticket=ticket,
                    symbol=symbol,
                    loss=abs(net_profit),
                    adx_value=current_adx,
                    stack_type=stack_type
                )

            return {
                'action': 'close_stack',
                'reason': 'per_stack_stop_loss',
                'ticket': ticket,
                'symbol': symbol,
                'stack_type': stack_type,
                'current_pnl': net_profit,
                'stop_loss_limit': stop_loss_limit,
                'loss_amount': abs(net_profit),
                'has_hedge': has_hedge,
                'has_dca': has_dca
            }

        return None

    def get_underwater_stacks(self, mt5_positions: List[Dict]) -> List[int]:
        """
        Get all tracked positions that are currently underwater (negative P&L)

        Used for cascade close - when 2nd stop-out triggers, close ALL underwater stacks

        Args:
            mt5_positions: List of all current MT5 positions

        Returns:
            List of ticket numbers for underwater stacks
        """
        underwater = []

        for ticket in self.tracked_positions.keys():
            net_profit = self.calculate_net_profit(ticket, mt5_positions)

            if net_profit is not None and net_profit < 0:
                underwater.append(ticket)

        return underwater

    def calculate_partial_close_volume(
        self,
        current_volume: float,
        close_percentage: float = 0.5,
        min_lot: float = 0.01,
        lot_step: float = 0.01
    ) -> float:
        """
        Calculate volume for partial close based on new lot size (0.04).

        Args:
            current_volume: Current position volume
            close_percentage: Percentage to close (0.0 to 1.0)
            min_lot: Minimum lot size allowed
            lot_step: Lot size step

        Returns:
            Volume to close (rounded to lot step)
        """
        close_volume = current_volume * close_percentage

        # Round to lot step
        close_volume = round_volume_to_step(
            close_volume,
            step=lot_step,
            min_lot=min_lot
        )

        # Ensure we're closing at least minimum lot
        if close_volume < min_lot:
            close_volume = min_lot

        # Ensure we don't close more than current volume
        if close_volume > current_volume:
            close_volume = current_volume

        return close_volume

    def get_recommended_partial_close(
        self,
        ticket: int,
        current_price: float,
        pip_value: float = 0.0001
    ) -> Optional[Dict]:
        """
        Get recommendation for partial close based on instrument-specific TP settings.

        Uses instrument-specific take profit levels instead of fixed dollar amounts.

        Args:
            ticket: Position ticket
            current_price: Current market price
            pip_value: Pip value for symbol (0.0001 for most pairs, 0.01 for JPY)

        Returns:
            Dictionary with partial close recommendation or None
        """
        if ticket not in self.tracked_positions:
            return None

        position = self.tracked_positions[ticket]
        symbol = position['symbol']
        entry_price = position['entry_price']
        position_type = position['type']
        total_volume = position['total_volume']

        # Get instrument-specific take profit settings
        try:
            tp_settings = get_take_profit_settings(symbol)
        except (KeyError, Exception):
            # Fallback to basic logic if no TP settings
            return None

        # Calculate pips in profit
        if position_type == 'buy':
            pips_profit = (current_price - entry_price) / pip_value
        else:
            pips_profit = (entry_price - current_price) / pip_value

        # Check if in profit
        if pips_profit <= 0:
            return None

        # Check TP levels and recommend partial closes
        partial_2_pips = tp_settings['partial_2_pips']
        partial_2_percent = tp_settings['partial_2_percent']
        partial_1_pips = tp_settings['partial_1_pips']
        partial_1_percent = tp_settings['partial_1_percent']
        full_tp_pips = tp_settings['full_tp_pips']

        # Determine recommended close based on instrument-specific TP levels
        if pips_profit >= full_tp_pips:
            # Full TP reached - close entire position
            recommended_volume = total_volume
            close_percent = 1.0
            reason = f"Full TP {full_tp_pips} pips reached - closing 100%"
        elif pips_profit >= partial_2_pips:
            # Second partial TP - close percentage from settings
            recommended_volume = self.calculate_partial_close_volume(total_volume, partial_2_percent)
            close_percent = partial_2_percent
            reason = f"Partial TP {partial_2_pips} pips reached - closing {int(partial_2_percent*100)}%"
        elif pips_profit >= partial_1_pips:
            # First partial TP - close percentage from settings
            recommended_volume = self.calculate_partial_close_volume(total_volume, partial_1_percent)
            close_percent = partial_1_percent
            reason = f"Partial TP {partial_1_pips} pips reached - closing {int(partial_1_percent*100)}%"
        else:
            # Not at any TP level yet
            return None

        return {
            'ticket': ticket,
            'symbol': symbol,
            'current_volume': total_volume,
            'close_volume': recommended_volume,
            'remaining_volume': total_volume - recommended_volume,
            'close_percentage': close_percent * 100,
            'pips_profit': pips_profit,
            'tp_level_hit': f"{pips_profit:.1f} pips",
            'reason': reason
        }

    def should_partial_close(
        self,
        ticket: int,
        current_profit: float,
        min_profit_for_partial: float = 5.0
    ) -> bool:
        """
        Determine if position should be partially closed.

        Args:
            ticket: Position ticket
            current_profit: Current profit in dollars
            min_profit_for_partial: Minimum profit to trigger partial close

        Returns:
            True if should partially close
        """
        if ticket not in self.tracked_positions:
            return False

        position = self.tracked_positions[ticket]

        # Only partial close if:
        # 1. Position is profitable above threshold
        # 2. Position has recovery levels active (want to reduce risk)
        # 3. Total volume is > 0.04 (enough to partially close)

        has_recovery = (
            len(position['grid_levels']) > 0 or
            len(position['hedge_tickets']) > 0 or
            len(position['dca_levels']) > 0
        )

        return (
            current_profit >= min_profit_for_partial and
            has_recovery and
            position['total_volume'] > 0.04
        )

    def calculate_atr_trailing_distance(self, symbol: str, tp_settings: Dict) -> float:
        """
        Calculate trailing stop distance using ATR with min/max bounds.

        Args:
            symbol: Trading symbol
            tp_settings: Take profit settings from instruments_config

        Returns:
            Trailing distance in pips
        """
        try:
            # Get ATR multiplier and bounds from config
            atr_multiplier = tp_settings.get('trailing_stop_atr_multiplier', 2.0)
            min_pips = tp_settings.get('trailing_stop_min_pips', 25)
            max_pips = tp_settings.get('trailing_stop_max_pips', 50)

            # Get current ATR (14-period on H1)
            bars = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_H1, 0, 50)
            if bars is None or len(bars) < 14:
                logger.warning(f"[TRAIL] Insufficient bars for ATR calculation, using minimum: {min_pips} pips")
                return min_pips

            # Calculate ATR(14)
            import pandas as pd
            import numpy as np
            df = pd.DataFrame(bars)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_14 = true_range.rolling(window=14).mean().iloc[-1]

            # Convert ATR to pips (multiply by 10000 for most pairs, 100 for JPY pairs)
            point = self.mt5.symbol_info(symbol).point
            pip_multiplier = 10000 if 'JPY' not in symbol else 100
            atr_pips = (atr_14 / point) / (pip_multiplier / 10)

            # Calculate trailing distance: ATR × multiplier, bounded by min/max
            trailing_pips = atr_pips * atr_multiplier
            trailing_pips = max(min_pips, min(trailing_pips, max_pips))

            logger.info(f"[TRAIL] {symbol} ATR(14)={atr_pips:.1f} pips, Trail={trailing_pips:.1f} pips ({atr_multiplier}×ATR, bounds: {min_pips}-{max_pips})")
            return trailing_pips

        except Exception as e:
            logger.error(f"[TRAIL] Error calculating ATR trailing distance: {e}")
            return tp_settings.get('trailing_stop_min_pips', 25)

    def activate_trailing_stop(self, ticket: int, current_price: float, tp_settings: Dict):
        """
        Activate trailing stop after PC2 closes.

        Args:
            ticket: Position ticket
            current_price: Current market price
            tp_settings: Take profit settings from instruments_config
        """
        if ticket not in self.tracked_positions:
            return

        position = self.tracked_positions[ticket]
        symbol = position['symbol']

        # Calculate trailing distance using ATR
        trailing_pips = self.calculate_atr_trailing_distance(symbol, tp_settings)

        # Get point value for symbol
        point = self.mt5.symbol_info(symbol).point
        trailing_distance = trailing_pips * point * 10

        # Set trailing stop
        if position['type'] == 'buy':
            stop_price = current_price - trailing_distance
        else:  # sell
            stop_price = current_price + trailing_distance

        position['trailing_stop_active'] = True
        position['trailing_stop_distance_pips'] = trailing_pips
        position['trailing_stop_price'] = stop_price
        position['highest_profit_price'] = current_price

        logger.info(f"[TRAIL] Activated for #{ticket} ({symbol}): Trail={trailing_pips:.0f} pips, Stop@{stop_price:.5f}")

    def update_trailing_stop(self, ticket: int, current_price: float):
        """
        Update trailing stop price if profit increases.

        Args:
            ticket: Position ticket
            current_price: Current market price
        """
        if ticket not in self.tracked_positions:
            return

        position = self.tracked_positions[ticket]

        if not position['trailing_stop_active']:
            return

        # Check if price moved in profitable direction
        symbol = position['symbol']
        position_type = position['type']

        if position_type == 'buy':
            if current_price > position['highest_profit_price']:
                # New high - update trailing stop
                position['highest_profit_price'] = current_price
                point = self.mt5.symbol_info(symbol).point
                trailing_distance = position['trailing_stop_distance_pips'] * point * 10
                new_stop = current_price - trailing_distance

                if new_stop > position['trailing_stop_price']:
                    old_stop = position['trailing_stop_price']
                    position['trailing_stop_price'] = new_stop
                    pips_moved = (new_stop - old_stop) / (point * 10)
                    logger.debug(f"[TRAIL] #{ticket} stop moved UP {pips_moved:.1f} pips to {new_stop:.5f}")
        else:  # sell
            if current_price < position['highest_profit_price']:
                # New low - update trailing stop
                position['highest_profit_price'] = current_price
                point = self.mt5.symbol_info(symbol).point
                trailing_distance = position['trailing_stop_distance_pips'] * point * 10
                new_stop = current_price + trailing_distance

                if new_stop < position['trailing_stop_price']:
                    old_stop = position['trailing_stop_price']
                    position['trailing_stop_price'] = new_stop
                    pips_moved = (old_stop - new_stop) / (point * 10)
                    logger.debug(f"[TRAIL] #{ticket} stop moved DOWN {pips_moved:.1f} pips to {new_stop:.5f}")

    def check_trailing_stop(self, ticket: int, current_price: float) -> bool:
        """
        Check if trailing stop has been hit.

        Args:
            ticket: Position ticket
            current_price: Current market price

        Returns:
            True if stop hit, False otherwise
        """
        if ticket not in self.tracked_positions:
            return False

        position = self.tracked_positions[ticket]

        if not position['trailing_stop_active']:
            return False

        position_type = position['type']
        stop_price = position['trailing_stop_price']

        if position_type == 'buy':
            if current_price <= stop_price:
                logger.info(f"[TRAIL] ✂️  #{ticket} BUY stop HIT: {current_price:.5f} <= {stop_price:.5f}")
                return True
        else:  # sell
            if current_price >= stop_price:
                logger.info(f"[TRAIL] ✂️  #{ticket} SELL stop HIT: {current_price:.5f} >= {stop_price:.5f}")
                return True

        return False

    def check_hedge_drawdown(
        self,
        ticket: int,
        mt5_positions: List[Dict],
        max_hedge_drawdown_pct: float = 15.0
    ) -> Optional[Dict]:
        """
        Monitor hedge positions for excessive drawdown and close if needed.
        This prevents hedges from accumulating losses if they move against us.

        Args:
            ticket: Original position ticket
            mt5_positions: List of all current MT5 positions
            max_hedge_drawdown_pct: Maximum drawdown % for hedge before closing (default 15%)

        Returns:
            Dict with hedge close recommendation or None
        """
        if ticket not in self.tracked_positions:
            return None

        position = self.tracked_positions[ticket]
        hedge_tickets = position.get('hedge_tickets', [])

        if len(hedge_tickets) == 0:
            return None

        # Debug: Log hedge tracking
        logger.debug(f"Checking hedge drawdown for {ticket}: {len(hedge_tickets)} hedges tracked")

        # Check each hedge position for drawdown
        hedges_to_close = []
        total_hedge_drawdown = 0
        missing_hedges = []  # Track hedges that are no longer in MT5

        for hedge_info in hedge_tickets:
            # hedge_info is a dict with 'type', 'volume', 'trigger_pips', 'time'
            # We need to find the actual MT5 position ticket
            # Hedges are tracked when opened, but we need to get their current profit

            # Find hedge position in MT5 positions
            # Hedges have comment like "Hedge - {last 5 digits of original_ticket}"
            short_ticket = str(ticket)[-5:]
            hedge_comment = f"Hedge - {short_ticket}"
            hedge_found = False

            for mt5_pos in mt5_positions:
                if hedge_comment in mt5_pos.get('comment', ''):  # Changed to 'in' to handle comment variations
                    hedge_found = True
                    # Found the hedge position
                    hedge_ticket = mt5_pos.get('ticket')
                    hedge_profit = mt5_pos.get('profit', 0)
                    hedge_volume = mt5_pos.get('volume', 0)

                    # Calculate drawdown
                    if hedge_profit < 0:
                        # Hedge is underwater
                        # Calculate drawdown as percentage of expected hedge value
                        # Expected hedge value would be positive if market reversed
                        # For now, use absolute loss amount as indicator
                        hedge_loss = abs(hedge_profit)
                        total_hedge_drawdown += hedge_loss

                        # Check if this hedge should be closed
                        # Close if loss exceeds fixed threshold of $75
                        max_loss_threshold = 75.0  # Fixed $75 threshold for ALL hedges

                        if hedge_loss >= max_loss_threshold:
                            hedges_to_close.append({
                                'ticket': hedge_ticket,
                                'original_ticket': ticket,
                                'profit': hedge_profit,
                                'volume': hedge_volume,
                                'loss': hedge_loss,
                                'threshold': max_loss_threshold,
                                'symbol': position['symbol']
                            })

            # Track missing hedges for cleanup
            if not hedge_found:
                missing_hedges.append(hedge_info)
                logger.info(f"Hedge closed/missing for {ticket} - removing from tracking (comment: '{hedge_comment}')")

        # CLEANUP: Remove hedges that are no longer in MT5 (closed manually or by SL/TP)
        if missing_hedges:
            for missing_hedge in missing_hedges:
                if missing_hedge in position['hedge_tickets']:
                    position['hedge_tickets'].remove(missing_hedge)
            logger.info(f"Cleaned up {len(missing_hedges)} closed hedge(s) from tracking for position {ticket}")

        # Return close recommendation if any hedges need closing
        if len(hedges_to_close) > 0:
            print(f"[WARN] Hedge drawdown detected for position {ticket}")
            print(f"   {len(hedges_to_close)} hedge(s) need closing")
            print(f"   Total hedge drawdown: ${total_hedge_drawdown:.2f}")

            return {
                'action': 'close_hedges',
                'original_ticket': ticket,
                'hedges_to_close': hedges_to_close,
                'total_drawdown': total_hedge_drawdown
            }

        return None

    def get_hedge_tickets_for_position(self, ticket: int) -> List[int]:
        """
        Get all hedge ticket numbers for a given position.

        Args:
            ticket: Original position ticket

        Returns:
            List of hedge ticket numbers
        """
        if ticket not in self.tracked_positions:
            return []

        position = self.tracked_positions[ticket]

        # Extract ticket numbers from hedge_tickets list
        # hedge_tickets is a list of dicts with 'ticket' key
        hedge_tickets = [
            h.get('ticket')
            for h in position.get('hedge_tickets', [])
            if h.get('ticket') is not None
        ]

        return hedge_tickets

    def remove_closed_hedge(self, original_ticket: int, hedge_ticket: int):
        """
        Remove a closed hedge from tracking.

        Args:
            original_ticket: Original position ticket
            hedge_ticket: Hedge ticket that was closed
        """
        if original_ticket not in self.tracked_positions:
            return

        position = self.tracked_positions[original_ticket]

        # Remove from hedge_tickets list
        # hedge_tickets is a list of dicts with 'ticket' key
        original_count = len(position['hedge_tickets'])

        position['hedge_tickets'] = [
            h for h in position['hedge_tickets']
            if h.get('ticket') != hedge_ticket
        ]

        removed_count = original_count - len(position['hedge_tickets'])

        if removed_count > 0:
            print(f"📝 Hedge {hedge_ticket} removed from tracking for position {original_ticket}")
        else:
            print(f"[WARN]  Hedge {hedge_ticket} not found in tracking for position {original_ticket}")

    def check_market_state_for_hedge_close(
        self,
        symbol: str,
        current_data: pd.DataFrame,
        adx_threshold: float = 25
    ) -> Dict:
        """
        Check market state when closing a hedge to determine if we should prevent new trades.
        Uses ADX and candle analysis to detect trending vs ranging markets.

        Args:
            symbol: Trading symbol
            current_data: Recent H1 candle data
            adx_threshold: ADX value above which market is considered trending

        Returns:
            Dict with market state info
        """
        try:
            from indicators.adx import calculate_adx, interpret_adx, analyze_candle_direction

            # Calculate ADX
            data_with_adx = calculate_adx(current_data.copy(), period=14)
            if data_with_adx is None or data_with_adx.empty:
                return {
                    'state': 'unknown',
                    'adx': None,
                    'is_trending': False,
                    'should_block_new_trades': False,
                    'reason': 'Unable to calculate ADX'
                }

            latest_adx = data_with_adx.iloc[-1]
            adx_value = latest_adx['adx']
            plus_di = latest_adx['plus_di']
            minus_di = latest_adx['minus_di']

            # Interpret ADX
            adx_info = interpret_adx(adx_value, plus_di, minus_di)

            # Analyze candle direction
            candle_info = analyze_candle_direction(current_data, lookback=5)

            # Determine if market is trending
            is_trending = adx_value >= adx_threshold
            is_strong_trend = adx_value >= 40

            # Determine if we should block new trades
            # Block ONLY if: EXTREME trend (ADX > 40)
            # Allow trades during normal trending markets (ADX 25-40) to enable breakout strategies
            should_block = False
            block_reason = ""

            if is_strong_trend:
                should_block = True
                block_reason = f"EXTREME trend detected (ADX: {adx_value:.1f}) - New trades blocked for safety"
            else:
                if is_trending:
                    block_reason = f"Trending market (ADX: {adx_value:.1f}) - Trades allowed (breakout strategies active)"
                else:
                    block_reason = f"Market suitable for trading (ADX: {adx_value:.1f})"

            market_state = {
                'state': adx_info['market_type'],
                'adx': adx_value,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'direction': adx_info['direction'],
                'is_trending': is_trending,
                'is_strong_trend': is_strong_trend,
                'candle_alignment': candle_info.get('alignment', 'unknown'),
                'candles_aligned': candle_info.get('aligned', False),
                'should_block_new_trades': should_block,
                'reason': block_reason
            }

            print(f" Market State Analysis for {symbol}:")
            print(f"   ADX: {adx_value:.1f} ({adx_info['market_type']})")
            print(f"   Direction: {adx_info['direction']}")
            print(f"   Candles: {candle_info.get('alignment', 'unknown')}")
            print(f"   Block new trades: {should_block}")
            print(f"   Reason: {block_reason}")

            return market_state

        except Exception as e:
            print(f"[WARN] Error analyzing market state: {e}")
            return {
                'state': 'error',
                'adx': None,
                'is_trending': False,
                'should_block_new_trades': False,
                'reason': f'Error: {str(e)}'
            }

