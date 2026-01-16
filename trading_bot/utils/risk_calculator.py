"""
Risk Management and Position Sizing
Based on discovered parameters:
- Risk: 1% per trade
- Max total exposure: 5.04 lots
- Max drawdown: 10%
"""

from typing import Dict, Optional, List
from datetime import datetime

from config.strategy_config import (
    BASE_LOT_SIZE,
    RISK_PERCENT,
    USE_FIXED_LOT_SIZE,
    MAX_TOTAL_LOTS,
    MAX_DRAWDOWN_PERCENT,
    STOP_LOSS_PIPS,
    TAKE_PROFIT_PIPS,
)


class RiskCalculator:
    """Calculate position sizes and manage risk"""

    def __init__(self):
        """Initialize risk calculator"""
        self.initial_balance = None
        self.peak_balance = None

    def set_initial_balance(self, balance: float):
        """Set initial account balance for drawdown tracking"""
        self.initial_balance = balance
        self.peak_balance = balance

    def calculate_position_size(
        self,
        account_balance: float,
        symbol_info: Dict,
        stop_loss_pips: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on risk percentage or use fixed lot size

        Args:
            account_balance: Current account balance
            symbol_info: Symbol information from MT5
            stop_loss_pips: Stop loss in pips (optional)

        Returns:
            float: Lot size
        """
        # Get broker limits
        min_lot = symbol_info.get('volume_min', 0.01)
        max_lot = symbol_info.get('volume_max', 100.0)
        step = symbol_info.get('volume_step', 0.01)

        # Use fixed lot size if enabled
        if USE_FIXED_LOT_SIZE:
            lots = BASE_LOT_SIZE
        else:
            # Calculate based on risk percentage
            risk_amount = account_balance * (RISK_PERCENT / 100)

            # If no stop loss, use default position sizing
            if stop_loss_pips is None or STOP_LOSS_PIPS is None:
                # Simple position sizing: risk amount divided by typical risk per lot
                # For forex, 1 pip movement = ~$10 per standard lot on most pairs
                pip_value_per_lot = 10.0  # USD per pip for standard lot
                assumed_stop_pips = 50  # Conservative default

                lots = risk_amount / (assumed_stop_pips * pip_value_per_lot)
            else:
                # Calculate based on actual stop loss
                pip_value_per_lot = 10.0
                lots = risk_amount / (stop_loss_pips * pip_value_per_lot)

        # Round to step
        lots = round(lots / step) * step

        # Clamp to limits
        lots = max(min_lot, min(lots, max_lot))

        return lots

    def check_total_exposure(
        self,
        current_positions: List[Dict],
        new_position_volume: float
    ) -> bool:
        """
        Check if adding new position would exceed max exposure

        Args:
            current_positions: List of open positions
            new_position_volume: Volume of new position to add

        Returns:
            bool: True if within limits
        """
        # Calculate total current volume
        total_volume = sum(pos.get('volume', 0) for pos in current_positions)

        # Add new position
        total_volume += new_position_volume

        # Check against max
        if total_volume > MAX_TOTAL_LOTS:
            print(f"[WARN] Max exposure limit reached!")
            print(f"   Current: {total_volume:.2f} lots")
            print(f"   Limit: {MAX_TOTAL_LOTS:.2f} lots")
            return False

        return True

    def calculate_drawdown(self, current_balance: float, peak_balance: float) -> float:
        """
        Calculate current drawdown percentage

        Args:
            current_balance: Current account balance
            peak_balance: Peak account balance

        Returns:
            float: Drawdown percentage
        """
        if peak_balance == 0:
            return 0

        drawdown = ((peak_balance - current_balance) / peak_balance) * 100
        return max(0, drawdown)

    def check_drawdown_limit(
        self,
        current_equity: float,
        update_peak: bool = True
    ) -> bool:
        """
        Check if current drawdown exceeds limit (based on EQUITY)

        Args:
            current_equity: Current account equity (includes unrealized P&L)
            update_peak: Whether to update peak equity

        Returns:
            bool: True if within limits, False if should stop trading
        """
        if self.peak_balance is None:
            self.peak_balance = current_equity
            return True

        # Update peak if new high
        if update_peak and current_equity > self.peak_balance:
            self.peak_balance = current_equity

        # Calculate drawdown from peak equity
        drawdown = self.calculate_drawdown(current_equity, self.peak_balance)

        if drawdown >= MAX_DRAWDOWN_PERCENT:
            print(f"🛑 MAX DRAWDOWN REACHED!")
            print(f"   Current: {drawdown:.2f}%")
            print(f"   Limit: {MAX_DRAWDOWN_PERCENT:.2f}%")
            print(f"   Peak equity: ${self.peak_balance:.2f}")
            print(f"   Current equity: ${current_equity:.2f}")
            print(f"   [WARN]  STOPPING ALL TRADING TO PROTECT ACCOUNT")
            return False

        return True

    def calculate_stop_loss_price(
        self,
        entry_price: float,
        position_type: str,
        stop_pips: float,
        pip_value: float = 0.0001
    ) -> float:
        """
        Calculate stop loss price

        Args:
            entry_price: Entry price
            position_type: 'buy' or 'sell'
            stop_pips: Stop loss in pips
            pip_value: Pip value for symbol

        Returns:
            float: Stop loss price
        """
        stop_distance = stop_pips * pip_value

        if position_type == 'buy':
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit_price(
        self,
        entry_price: float,
        position_type: str,
        tp_pips: float,
        pip_value: float = 0.0001
    ) -> float:
        """
        Calculate take profit price

        Args:
            entry_price: Entry price
            position_type: 'buy' or 'sell'
            tp_pips: Take profit in pips
            pip_value: Pip value for symbol

        Returns:
            float: Take profit price
        """
        tp_distance = tp_pips * pip_value

        if position_type == 'buy':
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance

    def validate_trade(
        self,
        account_info: Dict,
        symbol_info: Dict,
        volume: float,
        current_positions: List[Dict]
    ) -> tuple[bool, str]:
        """
        Validate if trade can be placed

        Args:
            account_info: Account information
            symbol_info: Symbol information
            volume: Proposed lot size
            current_positions: Current open positions

        Returns:
            Tuple of (can_trade, reason)
        """
        # Check free margin
        free_margin = account_info.get('free_margin', 0)
        if free_margin < 100:  # Minimum $100 free margin
            return False, "Insufficient free margin"

        # Check total exposure
        if not self.check_total_exposure(current_positions, volume):
            return False, "Max total exposure exceeded"

        # Check drawdown (use EQUITY not balance - includes unrealized P&L)
        equity = account_info.get('equity', 0)
        if not self.check_drawdown_limit(equity):
            return False, "Max drawdown exceeded - TRADING STOPPED"

        # Check volume limits
        min_volume = symbol_info.get('volume_min', 0.01)
        max_volume = symbol_info.get('volume_max', 100)

        if volume < min_volume:
            return False, f"Volume below minimum ({min_volume})"

        if volume > max_volume:
            return False, f"Volume above maximum ({max_volume})"

        return True, "OK"

    def get_risk_metrics(
        self,
        account_info: Dict,
        positions: List[Dict]
    ) -> Dict:
        """
        Get current risk metrics

        Args:
            account_info: Account information
            positions: Open positions

        Returns:
            Dict with risk metrics
        """
        balance = account_info.get('balance', 0)
        equity = account_info.get('equity', 0)
        margin = account_info.get('margin', 0)
        free_margin = account_info.get('free_margin', 0)

        # Calculate total exposure
        total_volume = sum(pos.get('volume', 0) for pos in positions)

        # Calculate total profit/loss
        total_pl = sum(pos.get('profit', 0) for pos in positions)

        # Calculate drawdown from peak EQUITY (includes unrealized P&L)
        drawdown = self.calculate_drawdown(equity, self.peak_balance) if self.peak_balance else 0

        # Margin level
        margin_level = account_info.get('margin_level', 0)

        # Exposure percentage
        exposure_pct = (total_volume / MAX_TOTAL_LOTS) * 100 if MAX_TOTAL_LOTS > 0 else 0

        return {
            'balance': balance,
            'equity': equity,
            'margin': margin,
            'free_margin': free_margin,
            'margin_level': margin_level,
            'total_volume': total_volume,
            'total_pl': total_pl,
            'drawdown_pct': drawdown,
            'exposure_pct': exposure_pct,
            'max_exposure': MAX_TOTAL_LOTS,
            'positions_count': len(positions),
            'within_limits': drawdown < MAX_DRAWDOWN_PERCENT and total_volume < MAX_TOTAL_LOTS
        }
