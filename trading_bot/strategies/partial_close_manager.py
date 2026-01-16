"""
Partial Close Manager
Implements scale-out functionality for progressive profit taking
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from config import strategy_config as cfg


class PartialCloseManager:
    """
    Manages partial close (scale out) operations
    Closes portions of position as it moves toward profit target
    """

    def __init__(self):
        """Initialize partial close manager"""
        self.partial_closes = {}  # Track partial closes per position
        self.trailing_stops = {}  # Track trailing stops after partial close

    def track_position(self, ticket: int, entry_price: float, volume: float, position_type: str, tp_price: float):
        """
        Start tracking a position for partial close

        Args:
            ticket: Position ticket
            entry_price: Entry price
            volume: Initial lot size
            position_type: 'buy' or 'sell'
            tp_price: Take profit target price
        """
        self.partial_closes[ticket] = {
            'ticket': ticket,
            'entry_price': entry_price,
            'initial_volume': volume,
            'remaining_volume': volume,
            'type': position_type,
            'tp_price': tp_price,
            'partial_levels_hit': [],
            'trail_stop_active': False,
            'trail_stop_price': None
        }

    def untrack_position(self, ticket: int):
        """Remove position from tracking"""
        if ticket in self.partial_closes:
            del self.partial_closes[ticket]
        if ticket in self.trailing_stops:
            del self.trailing_stops[ticket]

    def calculate_percent_to_tp(
        self,
        ticket: int,
        current_price: float
    ) -> Optional[float]:
        """
        Calculate what percentage of the way to TP the price has moved

        Args:
            ticket: Position ticket
            current_price: Current market price

        Returns:
            Percentage (0-100) or None if position not tracked
        """
        if ticket not in self.partial_closes:
            return None

        pos = self.partial_closes[ticket]
        entry = pos['entry_price']
        tp = pos['tp_price']
        pos_type = pos['type']

        # Calculate distance to TP
        total_distance = abs(tp - entry)
        if total_distance == 0:
            return 0

        if pos_type == 'buy':
            current_distance = current_price - entry
        else:  # sell
            current_distance = entry - current_price

        # Calculate percentage
        percent_to_tp = (current_distance / total_distance) * 100

        # Clamp to 0-100 range
        return max(0, min(100, percent_to_tp))

    def check_partial_close_levels(
        self,
        ticket: int,
        current_price: float,
        current_profit_pips: float
    ) -> Optional[Dict]:
        """
        Check if any partial close level has been reached

        Args:
            ticket: Position ticket
            current_price: Current market price
            current_profit_pips: Current profit in pips

        Returns:
            Dict with partial close action if level hit, None otherwise
        """
        if not cfg.PARTIAL_CLOSE_ENABLED:
            return None

        if ticket not in self.partial_closes:
            return None

        # Check minimum profit requirement
        if current_profit_pips < cfg.PARTIAL_CLOSE_MIN_PROFIT_PIPS:
            return None

        pos = self.partial_closes[ticket]
        percent_to_tp = self.calculate_percent_to_tp(ticket, current_price)

        if percent_to_tp is None:
            return None

        # Check each partial close level
        for level in cfg.PARTIAL_CLOSE_LEVELS:
            level_percent = level['percent_to_tp']
            close_percent = level['close_percent']

            # Has this level been hit already?
            if level_percent in pos['partial_levels_hit']:
                continue

            # Has price reached this level?
            if percent_to_tp >= level_percent:
                # Calculate volume to close
                volume_to_close = pos['remaining_volume'] * (close_percent / 100.0)
                volume_to_close = round(volume_to_close, 2)

                # Ensure we don't close more than remaining
                if volume_to_close > pos['remaining_volume']:
                    volume_to_close = pos['remaining_volume']

                # Mark this level as hit
                pos['partial_levels_hit'].append(level_percent)

                # Update remaining volume
                pos['remaining_volume'] -= volume_to_close

                # Activate trailing stop after first partial close
                if cfg.TRAIL_STOP_AFTER_PARTIAL and not pos['trail_stop_active']:
                    pos['trail_stop_active'] = True
                    pos['trail_stop_price'] = self._calculate_initial_trail_stop(
                        current_price,
                        pos['type']
                    )

                return {
                    'ticket': ticket,
                    'action': 'partial_close',
                    'close_volume': volume_to_close,
                    'remaining_volume': pos['remaining_volume'],
                    'level_percent': level_percent,
                    'close_percent': close_percent,
                    'percent_to_tp': percent_to_tp,
                    'activate_trail_stop': pos['trail_stop_active'],
                    'trail_stop_price': pos['trail_stop_price']
                }

        return None

    def _calculate_initial_trail_stop(self, current_price: float, position_type: str) -> float:
        """
        Calculate initial trailing stop price

        Args:
            current_price: Current market price
            position_type: 'buy' or 'sell'

        Returns:
            Initial trail stop price
        """
        pips_to_price = 0.0001  # Standard for forex pairs
        distance = cfg.TRAIL_STOP_DISTANCE_PIPS * pips_to_price

        if position_type == 'buy':
            return current_price - distance
        else:  # sell
            return current_price + distance

    def update_trailing_stop(
        self,
        ticket: int,
        current_price: float
    ) -> Optional[Dict]:
        """
        Update trailing stop if price has moved favorably

        Args:
            ticket: Position ticket
            current_price: Current market price

        Returns:
            Dict with updated trail stop info or None
        """
        if ticket not in self.partial_closes:
            return None

        pos = self.partial_closes[ticket]

        if not pos['trail_stop_active']:
            return None

        pips_to_price = 0.0001
        distance = cfg.TRAIL_STOP_DISTANCE_PIPS * pips_to_price
        old_stop = pos['trail_stop_price']
        new_stop = None

        if pos['type'] == 'buy':
            # For buy, trail stop moves up with price
            potential_stop = current_price - distance
            if potential_stop > old_stop:
                new_stop = potential_stop
        else:  # sell
            # For sell, trail stop moves down with price
            potential_stop = current_price + distance
            if potential_stop < old_stop:
                new_stop = potential_stop

        if new_stop:
            pos['trail_stop_price'] = new_stop
            return {
                'ticket': ticket,
                'action': 'update_trail_stop',
                'old_stop': old_stop,
                'new_stop': new_stop,
                'current_price': current_price
            }

        return None

    def check_trail_stop_hit(
        self,
        ticket: int,
        current_price: float
    ) -> bool:
        """
        Check if trailing stop has been hit

        Args:
            ticket: Position ticket
            current_price: Current market price

        Returns:
            True if trail stop hit (should close position)
        """
        if ticket not in self.partial_closes:
            return False

        pos = self.partial_closes[ticket]

        if not pos['trail_stop_active'] or pos['trail_stop_price'] is None:
            return False

        if pos['type'] == 'buy':
            # For buy, stop hit if price falls below stop
            return current_price <= pos['trail_stop_price']
        else:  # sell
            # For sell, stop hit if price rises above stop
            return current_price >= pos['trail_stop_price']

    def get_position_status(self, ticket: int) -> Optional[Dict]:
        """
        Get current status of position partial close tracking

        Args:
            ticket: Position ticket

        Returns:
            Status dict or None
        """
        if ticket not in self.partial_closes:
            return None

        pos = self.partial_closes[ticket]

        return {
            'ticket': ticket,
            'initial_volume': pos['initial_volume'],
            'remaining_volume': pos['remaining_volume'],
            'percent_closed': ((pos['initial_volume'] - pos['remaining_volume']) / pos['initial_volume']) * 100,
            'levels_hit': pos['partial_levels_hit'],
            'trail_stop_active': pos['trail_stop_active'],
            'trail_stop_price': pos['trail_stop_price']
        }


def calculate_partial_close_volume(
    initial_volume: float,
    close_percent: float,
    broker_lot_step: float = 0.01,
    min_lot: float = 0.01
) -> float:
    """
    Calculate volume for partial close with broker constraints

    Args:
        initial_volume: Starting position volume
        close_percent: Percentage to close (0-100)
        broker_lot_step: Broker's lot step (default 0.01)
        min_lot: Minimum lot size (default 0.01)

    Returns:
        Volume to close (rounded to broker step)
    """
    close_volume = initial_volume * (close_percent / 100.0)

    # Round to broker step
    close_volume = round(close_volume / broker_lot_step) * broker_lot_step

    # Ensure minimum
    if close_volume < min_lot:
        return min_lot

    # Ensure we don't close more than we have
    if close_volume > initial_volume:
        return initial_volume

    return round(close_volume, 2)
