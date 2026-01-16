"""
Trading Bot Health Check Module
Comprehensive system validation and monitoring for trading bot operations
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


class HealthCheck:
    """
    Performs comprehensive health checks on the trading bot system
    """

    def __init__(self, mt5_manager, recovery_manager=None, risk_calculator=None):
        """
        Initialize health check with required managers

        Args:
            mt5_manager: MT5Manager instance for broker connection
            recovery_manager: RecoveryManager instance for position tracking
            risk_calculator: RiskCalculator instance for risk validation
        """
        self.mt5_manager = mt5_manager
        self.recovery_manager = recovery_manager
        self.risk_calculator = risk_calculator
        self.health_status = {}
        self.warnings = []
        self.errors = []

    def run_full_check(self) -> Dict[str, Any]:
        """
        Execute all health checks and return comprehensive status

        Returns:
            Dict containing health check results
        """
        logger.info("=" * 60)
        logger.info("TRADING BOT HEALTH CHECK - STARTING")
        logger.info("=" * 60)

        self.health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'HEALTHY',
            'checks': {}
        }
        self.warnings = []
        self.errors = []

        # Run all health checks
        self._check_mt5_connection()
        self._check_account_health()
        self._check_market_data_freshness()
        self._check_active_positions()
        self._check_hedge_positions()
        self._check_risk_limits()
        self._check_recovery_mechanisms()
        self._check_margin_adequacy()

        # Determine overall status
        if len(self.errors) > 0:
            self.health_status['overall_status'] = 'CRITICAL'
        elif len(self.warnings) > 3:
            self.health_status['overall_status'] = 'WARNING'

        self.health_status['warnings'] = self.warnings
        self.health_status['errors'] = self.errors
        self.health_status['warning_count'] = len(self.warnings)
        self.health_status['error_count'] = len(self.errors)

        return self.health_status

    def _check_mt5_connection(self):
        """Check MT5 connection status"""
        check_name = 'MT5 Connection'
        logger.info(f"\n[CHECK] {check_name}")

        try:
            terminal_info = mt5.terminal_info()

            if terminal_info is None:
                self.errors.append(f"{check_name}: Terminal not connected")
                self.health_status['checks']['mt5_connection'] = {
                    'status': 'FAILED',
                    'connected': False
                }
                logger.error("  [ERROR] MT5 Terminal NOT connected")
                return

            # Check terminal details
            connected = terminal_info.connected
            trade_allowed = terminal_info.trade_allowed

            self.health_status['checks']['mt5_connection'] = {
                'status': 'PASS' if connected and trade_allowed else 'WARNING',
                'connected': connected,
                'trade_allowed': trade_allowed,
                'company': terminal_info.company,
                'name': terminal_info.name,
                'path': terminal_info.path
            }

            if not connected:
                self.errors.append(f"{check_name}: Not connected to broker")
                logger.error("  [ERROR] NOT connected to broker")
            elif not trade_allowed:
                self.warnings.append(f"{check_name}: Trading not allowed")
                logger.warning("  ! Trading NOT allowed")
            else:
                logger.info(f"  [OK] Connected to {terminal_info.company}")
                logger.info(f"  [OK] Trading allowed")

        except Exception as e:
            self.errors.append(f"{check_name}: Exception - {str(e)}")
            self.health_status['checks']['mt5_connection'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"  [ERROR] Exception: {e}")

    def _check_account_health(self):
        """Check account financial metrics"""
        check_name = 'Account Health'
        logger.info(f"\n[CHECK] {check_name}")

        try:
            account_info = self.mt5_manager.get_account_info()

            if not account_info:
                self.errors.append(f"{check_name}: Cannot retrieve account info")
                self.health_status['checks']['account_health'] = {
                    'status': 'FAILED'
                }
                logger.error("  [ERROR] Cannot retrieve account info")
                return

            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', 0)
            margin = account_info.get('margin', 0)
            free_margin = account_info.get('free_margin', 0)
            margin_level = account_info.get('margin_level', 0)
            profit = account_info.get('profit', 0)

            # Calculate drawdown
            drawdown_pct = ((balance - equity) / balance * 100) if balance > 0 else 0

            self.health_status['checks']['account_health'] = {
                'status': 'PASS',
                'balance': balance,
                'equity': equity,
                'margin': margin,
                'free_margin': free_margin,
                'margin_level': margin_level,
                'profit': profit,
                'drawdown_pct': round(drawdown_pct, 2)
            }

            logger.info(f"  Balance: ${balance:.2f}")
            logger.info(f"  Equity: ${equity:.2f}")
            logger.info(f"  Profit: ${profit:.2f}")
            logger.info(f"  Drawdown: {drawdown_pct:.2f}%")

            # Check warning conditions
            if margin_level > 0 and margin_level < 200:
                self.warnings.append(f"{check_name}: Low margin level ({margin_level:.0f}%)")
                logger.warning(f"  ! Low margin level: {margin_level:.0f}%")
                self.health_status['checks']['account_health']['status'] = 'WARNING'
            elif margin_level > 0:
                logger.info(f"  [OK] Margin level: {margin_level:.0f}%")

            if free_margin < 100:
                self.warnings.append(f"{check_name}: Low free margin (${free_margin:.2f})")
                logger.warning(f"  ! Low free margin: ${free_margin:.2f}")
                self.health_status['checks']['account_health']['status'] = 'WARNING'
            else:
                logger.info(f"  [OK] Free margin: ${free_margin:.2f}")

            if drawdown_pct > 20:
                self.errors.append(f"{check_name}: High drawdown ({drawdown_pct:.2f}%)")
                logger.error(f"  [ERROR] High drawdown: {drawdown_pct:.2f}%")
                self.health_status['checks']['account_health']['status'] = 'FAILED'
            elif drawdown_pct > 10:
                self.warnings.append(f"{check_name}: Moderate drawdown ({drawdown_pct:.2f}%)")
                logger.warning(f"  ! Moderate drawdown: {drawdown_pct:.2f}%")
                self.health_status['checks']['account_health']['status'] = 'WARNING'

        except Exception as e:
            self.errors.append(f"{check_name}: Exception - {str(e)}")
            self.health_status['checks']['account_health'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"  [ERROR] Exception: {e}")

    def _check_market_data_freshness(self):
        """Check if market data is fresh and updating"""
        check_name = 'Market Data Freshness'
        logger.info(f"\n[CHECK] {check_name}")

        try:
            # Check common symbols
            test_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
            stale_symbols = []
            fresh_symbols = []

            for symbol in test_symbols:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    # Check if tick is recent (within last 5 minutes)
                    tick_time = datetime.fromtimestamp(tick.time)
                    age = datetime.now() - tick_time

                    if age > timedelta(minutes=5):
                        stale_symbols.append({
                            'symbol': symbol,
                            'age_minutes': age.total_seconds() / 60
                        })
                    else:
                        fresh_symbols.append(symbol)

            self.health_status['checks']['market_data'] = {
                'status': 'PASS' if len(stale_symbols) == 0 else 'WARNING',
                'fresh_symbols': fresh_symbols,
                'stale_symbols': stale_symbols
            }

            if len(stale_symbols) > 0:
                self.warnings.append(f"{check_name}: {len(stale_symbols)} symbols have stale data")
                logger.warning(f"  ! Stale data for: {[s['symbol'] for s in stale_symbols]}")
            else:
                logger.info(f"  [OK] Market data fresh for {len(fresh_symbols)} symbols")

        except Exception as e:
            self.warnings.append(f"{check_name}: Exception - {str(e)}")
            self.health_status['checks']['market_data'] = {
                'status': 'WARNING',
                'error': str(e)
            }
            logger.warning(f"  ! Exception: {e}")

    def _check_active_positions(self):
        """Check status of active positions"""
        check_name = 'Active Positions'
        logger.info(f"\n[CHECK] {check_name}")

        try:
            positions = mt5.positions_get()

            if positions is None:
                positions = []

            position_count = len(positions)
            total_volume = sum(p.volume for p in positions)
            total_profit = sum(p.profit for p in positions)

            # Analyze position types
            buy_positions = [p for p in positions if p.type == 0]
            sell_positions = [p for p in positions if p.type == 1]

            # Check for old positions
            old_positions = []
            now = datetime.now()

            for pos in positions:
                open_time = datetime.fromtimestamp(pos.time)
                age_hours = (now - open_time).total_seconds() / 3600

                if age_hours > 4:  # MAX_POSITION_HOURS default
                    old_positions.append({
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'age_hours': round(age_hours, 1),
                        'profit': pos.profit
                    })

            self.health_status['checks']['active_positions'] = {
                'status': 'PASS',
                'count': position_count,
                'total_volume': round(total_volume, 2),
                'total_profit': round(total_profit, 2),
                'buy_count': len(buy_positions),
                'sell_count': len(sell_positions),
                'old_positions': old_positions
            }

            logger.info(f"  Total positions: {position_count}")
            logger.info(f"  Total volume: {total_volume:.2f} lots")
            logger.info(f"  Total profit: ${total_profit:.2f}")
            logger.info(f"  Buy/Sell: {len(buy_positions)}/{len(sell_positions)}")

            if len(old_positions) > 0:
                self.warnings.append(f"{check_name}: {len(old_positions)} positions older than 4 hours")
                logger.warning(f"  ! {len(old_positions)} positions older than 4 hours")
                for old_pos in old_positions:
                    logger.warning(f"    - Ticket {old_pos['ticket']}: {old_pos['age_hours']}h old, ${old_pos['profit']:.2f}")

            # Check for excessive volume
            if self.risk_calculator and total_volume > 15.0:  # MAX_TOTAL_LOTS
                self.warnings.append(f"{check_name}: High total volume ({total_volume:.2f} lots)")
                logger.warning(f"  ! High total volume: {total_volume:.2f} lots")
                self.health_status['checks']['active_positions']['status'] = 'WARNING'

        except Exception as e:
            self.errors.append(f"{check_name}: Exception - {str(e)}")
            self.health_status['checks']['active_positions'] = {
                'status': 'FAILED',
                'error': str(e)
            }
            logger.error(f"  [ERROR] Exception: {e}")

    def _check_hedge_positions(self):
        """Check status and health of hedge positions"""
        check_name = 'Hedge Positions'
        logger.info(f"\n[CHECK] {check_name}")

        try:
            if not self.recovery_manager:
                logger.info("  - Recovery manager not available, skipping")
                return

            tracked_positions = self.recovery_manager.tracked_positions

            # Find all positions with hedges
            positions_with_hedges = {}
            total_hedge_count = 0
            total_hedge_volume = 0
            hedge_drawdown_total = 0

            for ticket, position in tracked_positions.items():
                hedge_tickets = position.get('hedge_tickets', [])

                if len(hedge_tickets) > 0:
                    # Get current hedge profit/loss
                    hedge_pl = 0
                    hedge_volume = 0

                    for hedge_ticket in hedge_tickets:
                        hedge_pos = mt5.positions_get(ticket=hedge_ticket)
                        if hedge_pos and len(hedge_pos) > 0:
                            hedge_pl += hedge_pos[0].profit
                            hedge_volume += hedge_pos[0].volume

                    # Get original position
                    orig_pos = mt5.positions_get(ticket=ticket)
                    orig_pl = orig_pos[0].profit if orig_pos and len(orig_pos) > 0 else 0

                    net_pl = orig_pl + hedge_pl

                    positions_with_hedges[ticket] = {
                        'hedge_count': len(hedge_tickets),
                        'hedge_volume': hedge_volume,
                        'original_profit': orig_pl,
                        'hedge_profit': hedge_pl,
                        'net_profit': net_pl,
                        'symbol': position.get('symbol', 'UNKNOWN')
                    }

                    total_hedge_count += len(hedge_tickets)
                    total_hedge_volume += hedge_volume

                    if net_pl < 0:
                        hedge_drawdown_total += abs(net_pl)

            self.health_status['checks']['hedge_positions'] = {
                'status': 'PASS',
                'positions_with_hedges': len(positions_with_hedges),
                'total_hedge_count': total_hedge_count,
                'total_hedge_volume': round(total_hedge_volume, 2),
                'total_hedge_drawdown': round(hedge_drawdown_total, 2),
                'details': positions_with_hedges
            }

            logger.info(f"  Positions with hedges: {len(positions_with_hedges)}")
            logger.info(f"  Total hedge trades: {total_hedge_count}")
            logger.info(f"  Total hedge volume: {total_hedge_volume:.2f} lots")
            logger.info(f"  Total hedge drawdown: ${hedge_drawdown_total:.2f}")

            # Check for underwater hedges
            underwater_hedges = [
                ticket for ticket, data in positions_with_hedges.items()
                if data['net_profit'] < -50  # Significant loss threshold
            ]

            if len(underwater_hedges) > 0:
                self.warnings.append(f"{check_name}: {len(underwater_hedges)} hedge positions underwater > $50")
                logger.warning(f"  ! {len(underwater_hedges)} hedge positions significantly underwater")
                for ticket in underwater_hedges:
                    data = positions_with_hedges[ticket]
                    logger.warning(f"    - Ticket {ticket} ({data['symbol']}): Net ${data['net_profit']:.2f}")
                self.health_status['checks']['hedge_positions']['status'] = 'WARNING'

            # Check for excessive hedge volume
            if total_hedge_volume > 5.0:
                self.warnings.append(f"{check_name}: High hedge volume ({total_hedge_volume:.2f} lots)")
                logger.warning(f"  ! High total hedge volume: {total_hedge_volume:.2f} lots")
                self.health_status['checks']['hedge_positions']['status'] = 'WARNING'

        except Exception as e:
            self.warnings.append(f"{check_name}: Exception - {str(e)}")
            self.health_status['checks']['hedge_positions'] = {
                'status': 'WARNING',
                'error': str(e)
            }
            logger.warning(f"  ! Exception: {e}")

    def _check_risk_limits(self):
        """Check if risk limits are being respected"""
        check_name = 'Risk Limits'
        logger.info(f"\n[CHECK] {check_name}")

        try:
            if not self.risk_calculator:
                logger.info("  - Risk calculator not available, skipping")
                return

            account_info = self.mt5_manager.get_account_info()
            positions = mt5.positions_get() or []

            # Check drawdown limit
            balance = account_info.get('balance', 0)
            equity = account_info.get('equity', 0)

            # Get peak balance from risk calculator
            peak_balance = self.risk_calculator.peak_balance

            # Calculate current drawdown
            current_drawdown = ((peak_balance - equity) / peak_balance * 100) if peak_balance > 0 else 0
            max_drawdown = self.risk_calculator.max_drawdown_percent

            # Check total exposure
            total_volume = sum(p.volume for p in positions)
            max_total_lots = getattr(self.risk_calculator, 'max_total_lots', 15.0)

            self.health_status['checks']['risk_limits'] = {
                'status': 'PASS',
                'current_drawdown_pct': round(current_drawdown, 2),
                'max_drawdown_pct': max_drawdown,
                'total_volume': round(total_volume, 2),
                'max_total_lots': max_total_lots,
                'peak_balance': peak_balance
            }

            logger.info(f"  Current drawdown: {current_drawdown:.2f}% (max: {max_drawdown}%)")
            logger.info(f"  Total volume: {total_volume:.2f} (max: {max_total_lots})")
            logger.info(f"  Peak balance: ${peak_balance:.2f}")

            # Check limit violations
            if current_drawdown >= max_drawdown:
                self.errors.append(f"{check_name}: Drawdown limit exceeded ({current_drawdown:.2f}% >= {max_drawdown}%)")
                logger.error(f"  [ERROR] Drawdown limit EXCEEDED!")
                self.health_status['checks']['risk_limits']['status'] = 'FAILED'
            elif current_drawdown >= max_drawdown * 0.8:
                self.warnings.append(f"{check_name}: Approaching drawdown limit ({current_drawdown:.2f}%)")
                logger.warning(f"  ! Approaching drawdown limit")
                self.health_status['checks']['risk_limits']['status'] = 'WARNING'
            else:
                logger.info(f"  [OK] Drawdown within limits")

            if total_volume >= max_total_lots:
                self.errors.append(f"{check_name}: Volume limit exceeded ({total_volume:.2f} >= {max_total_lots})")
                logger.error(f"  [ERROR] Volume limit EXCEEDED!")
                self.health_status['checks']['risk_limits']['status'] = 'FAILED'
            elif total_volume >= max_total_lots * 0.8:
                self.warnings.append(f"{check_name}: Approaching volume limit ({total_volume:.2f})")
                logger.warning(f"  ! Approaching volume limit")
                self.health_status['checks']['risk_limits']['status'] = 'WARNING'
            else:
                logger.info(f"  [OK] Volume within limits")

        except Exception as e:
            self.warnings.append(f"{check_name}: Exception - {str(e)}")
            self.health_status['checks']['risk_limits'] = {
                'status': 'WARNING',
                'error': str(e)
            }
            logger.warning(f"  ! Exception: {e}")

    def _check_recovery_mechanisms(self):
        """Check health of recovery mechanisms (grid, hedge, DCA)"""
        check_name = 'Recovery Mechanisms'
        logger.info(f"\n[CHECK] {check_name}")

        try:
            if not self.recovery_manager:
                logger.info("  - Recovery manager not available, skipping")
                return

            tracked_positions = self.recovery_manager.tracked_positions

            # Count recovery mechanism usage
            grid_count = 0
            hedge_count = 0
            dca_count = 0

            positions_with_recovery = 0
            max_grid_levels = 0
            max_dca_levels = 0

            for ticket, position in tracked_positions.items():
                has_recovery = False

                grid_levels = len(position.get('grid_levels', []))
                if grid_levels > 0:
                    grid_count += grid_levels
                    max_grid_levels = max(max_grid_levels, grid_levels)
                    has_recovery = True

                hedge_tickets = len(position.get('hedge_tickets', []))
                if hedge_tickets > 0:
                    hedge_count += hedge_tickets
                    has_recovery = True

                dca_levels = len(position.get('dca_levels', []))
                if dca_levels > 0:
                    dca_count += dca_levels
                    max_dca_levels = max(max_dca_levels, dca_levels)
                    has_recovery = True

                if has_recovery:
                    positions_with_recovery += 1

            self.health_status['checks']['recovery_mechanisms'] = {
                'status': 'PASS',
                'positions_with_recovery': positions_with_recovery,
                'total_grid_levels': grid_count,
                'total_hedges': hedge_count,
                'total_dca_levels': dca_count,
                'max_grid_levels': max_grid_levels,
                'max_dca_levels': max_dca_levels
            }

            logger.info(f"  Positions with recovery: {positions_with_recovery}")
            logger.info(f"  Total grid levels: {grid_count}")
            logger.info(f"  Total hedges: {hedge_count}")
            logger.info(f"  Total DCA levels: {dca_count}")

            # Check for excessive recovery usage
            if max_grid_levels >= 4:
                self.warnings.append(f"{check_name}: Position at max grid levels ({max_grid_levels})")
                logger.warning(f"  ! Position at max grid levels: {max_grid_levels}")
                self.health_status['checks']['recovery_mechanisms']['status'] = 'WARNING'

            if max_dca_levels >= 4:
                self.warnings.append(f"{check_name}: Position at max DCA levels ({max_dca_levels})")
                logger.warning(f"  ! Position at max DCA levels: {max_dca_levels}")
                self.health_status['checks']['recovery_mechanisms']['status'] = 'WARNING'

        except Exception as e:
            self.warnings.append(f"{check_name}: Exception - {str(e)}")
            self.health_status['checks']['recovery_mechanisms'] = {
                'status': 'WARNING',
                'error': str(e)
            }
            logger.warning(f"  ! Exception: {e}")

    def _check_margin_adequacy(self):
        """Check if margin is adequate for current positions"""
        check_name = 'Margin Adequacy'
        logger.info(f"\n[CHECK] {check_name}")

        try:
            account_info = self.mt5_manager.get_account_info()

            free_margin = account_info.get('free_margin', 0)
            margin_level = account_info.get('margin_level', 0)
            margin = account_info.get('margin', 0)

            # Calculate margin utilization
            equity = account_info.get('equity', 0)
            margin_utilization = (margin / equity * 100) if equity > 0 else 0

            self.health_status['checks']['margin_adequacy'] = {
                'status': 'PASS',
                'free_margin': free_margin,
                'margin_level': margin_level,
                'margin_utilization_pct': round(margin_utilization, 2)
            }

            logger.info(f"  Free margin: ${free_margin:.2f}")
            logger.info(f"  Margin level: {margin_level:.0f}%")
            logger.info(f"  Margin utilization: {margin_utilization:.2f}%")

            # Check adequacy
            if margin_level > 0 and margin_level < 150:
                self.errors.append(f"{check_name}: Critical margin level ({margin_level:.0f}%)")
                logger.error(f"  [ERROR] CRITICAL margin level: {margin_level:.0f}%")
                self.health_status['checks']['margin_adequacy']['status'] = 'FAILED'
            elif margin_level > 0 and margin_level < 200:
                self.warnings.append(f"{check_name}: Low margin level ({margin_level:.0f}%)")
                logger.warning(f"  ! Low margin level: {margin_level:.0f}%")
                self.health_status['checks']['margin_adequacy']['status'] = 'WARNING'
            else:
                logger.info(f"  [OK] Margin level adequate")

            if margin_utilization > 80:
                self.warnings.append(f"{check_name}: High margin utilization ({margin_utilization:.2f}%)")
                logger.warning(f"  ! High margin utilization")
                self.health_status['checks']['margin_adequacy']['status'] = 'WARNING'
            else:
                logger.info(f"  [OK] Margin utilization healthy")

        except Exception as e:
            self.warnings.append(f"{check_name}: Exception - {str(e)}")
            self.health_status['checks']['margin_adequacy'] = {
                'status': 'WARNING',
                'error': str(e)
            }
            logger.warning(f"  ! Exception: {e}")

    def generate_report(self) -> str:
        """
        Generate a human-readable health check report

        Returns:
            Formatted report string
        """
        if not self.health_status:
            return "Health check has not been run yet."

        report_lines = []
        report_lines.append("\n" + "=" * 70)
        report_lines.append("TRADING BOT HEALTH CHECK REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Timestamp: {self.health_status['timestamp']}")
        report_lines.append(f"Overall Status: {self.health_status['overall_status']}")
        report_lines.append(f"Warnings: {self.health_status['warning_count']}")
        report_lines.append(f"Errors: {self.health_status['error_count']}")
        report_lines.append("=" * 70)

        # Individual check results
        report_lines.append("\nCHECK RESULTS:")
        report_lines.append("-" * 70)

        for check_name, check_data in self.health_status['checks'].items():
            status = check_data.get('status', 'UNKNOWN')
            status_symbol = {
                'PASS': '[OK]',
                'WARNING': '!',
                'FAILED': '[ERROR]',
                'UNKNOWN': '?'
            }.get(status, '?')

            report_lines.append(f"\n[{status_symbol}] {check_name.upper().replace('_', ' ')}: {status}")

            for key, value in check_data.items():
                if key != 'status' and key != 'details':
                    report_lines.append(f"    {key}: {value}")

        # Warnings section
        if self.warnings:
            report_lines.append("\n" + "-" * 70)
            report_lines.append("WARNINGS:")
            report_lines.append("-" * 70)
            for warning in self.warnings:
                report_lines.append(f"  ! {warning}")

        # Errors section
        if self.errors:
            report_lines.append("\n" + "-" * 70)
            report_lines.append("ERRORS:")
            report_lines.append("-" * 70)
            for error in self.errors:
                report_lines.append(f"  [ERROR] {error}")

        report_lines.append("\n" + "=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70 + "\n")

        return "\n".join(report_lines)


def run_health_check_standalone(mt5_manager, recovery_manager=None, risk_calculator=None):
    """
    Standalone function to run health check and print report

    Args:
        mt5_manager: MT5Manager instance
        recovery_manager: RecoveryManager instance (optional)
        risk_calculator: RiskCalculator instance (optional)

    Returns:
        Dict containing health status
    """
    health_check = HealthCheck(mt5_manager, recovery_manager, risk_calculator)
    status = health_check.run_full_check()
    print(health_check.generate_report())
    return status
