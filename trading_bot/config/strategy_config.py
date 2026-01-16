"""
Strategy Configuration - Discovered from EA Analysis
All values extracted from 428 trades analyzed
"""

# =============================================================================
# TRADING PARAMETERS
# =============================================================================

# Symbols to trade (EURUSD and GBPUSD based on analysis)
SYMBOLS = ['EURUSD', 'GBPUSD']

# Primary timeframe for trading
TIMEFRAME = 'H1'

# Higher timeframes for institutional levels
HTF_TIMEFRAMES = ['D1', 'W1']

# =============================================================================
# CONFLUENCE PARAMETERS (Discovered from Analysis)
# =============================================================================

# Minimum confluence score required to enter trade
MIN_CONFLUENCE_SCORE = 8

# Optimal confluence score (83.3% win rate)
OPTIMAL_CONFLUENCE_SCORE = 8

# Confluence factor weights (higher = more important)
CONFLUENCE_WEIGHTS = {
    # Primary signals from analysis
    'vwap_band_1': 1,        # Used in 28.0% of trades
    'vwap_band_2': 1,        # Used in 39.5% of trades
    'poc': 1,                # Used in 38.1% of trades
    'swing_low': 1,          # Used in 17.1% of trades
    'swing_high': 1,         # Used in 17.1% of trades
    'above_vah': 1,          # Used in 14.0% of trades
    'below_val': 1,          # Used in 18.7% of trades
    'lvn': 1,                # Used in 16.1% of trades

    # HTF factors (higher weight)
    'prev_day_vah': 2,       # 364 occurrences
    'prev_day_val': 2,       # High importance
    'prev_day_poc': 2,       # 325 occurrences
    'daily_hvn': 2,          # 310 occurrences
    'daily_poc': 2,          # Institutional level
    'weekly_hvn': 3,         # 328 occurrences (highest weight)
    'weekly_poc': 3,         # Strong institutional level
    'prev_week_swing_low': 2,  # 325 occurrences
    'prev_week_swing_high': 2, # Strong resistance
    'prev_week_vwap': 2,     # Weekly pivot

    # NOTE: Fair Value Gaps (FVGs) are NOT included in production yet
    # FVGs are tracked by ML system for data collection & validation
    # Weights will be added here once ML proves effectiveness (15+ trades)
    # Recommended weights (when enabled):
    #   'daily_bullish_fvg': 2
    #   'daily_bearish_fvg': 2
    #   'weekly_bullish_fvg': 3
    #   'weekly_bearish_fvg': 3
}

# Price tolerance for level detection (0.3% = 30 pips on most pairs)
LEVEL_TOLERANCE_PCT = 0.003

# =============================================================================
# VWAP PARAMETERS
# =============================================================================

# VWAP calculation period (bars)
VWAP_PERIOD = 200  # Approximately 8 days on H1

# Standard deviation multipliers for bands
VWAP_BAND_MULTIPLIERS = [1, 2, 3]  # ±1σ, ±2σ, ±3σ

# =============================================================================
# VOLUME PROFILE PARAMETERS
# =============================================================================

# Number of bins for volume profile
VP_BINS = 100

# Number of HVN/LVN levels to track
HVN_LEVELS = 5
LVN_LEVELS = 5

# Swing high/low detection
SWING_LOOKBACK = 10  # Bars to look back for swing points

# =============================================================================
# TREND FILTER PARAMETERS (ADX + Candle Direction)
# =============================================================================
# NOTE: With ENABLE_TIME_FILTERS = False, this is the PRIMARY entry filter
# ADX & candle lookback will determine when it's safe to trade (no time restrictions)

# Enable trend filtering (prevents trading in strong trends)
TREND_FILTER_ENABLED = True

# ADX parameters
ADX_PERIOD = 14  # Standard ADX period
ADX_THRESHOLD = 25  # Above this = trending market
ADX_STRONG_THRESHOLD = 40  # Above this = strong trend (never trade)

# Candle direction lookback
CANDLE_LOOKBACK = 5  # Number of recent candles to analyze
CANDLE_ALIGNMENT_PCT = 70  # % of candles in same direction = "aligned"

# Trading rules
ALLOW_WEAK_TRENDS = True  # Trade when ADX 20-25 (weak trend)
SKIP_STRONG_TRENDS = True  # Never trade when ADX > 40

# =============================================================================
# GRID TRADING PARAMETERS (AGGRESSIVE RECOVERY SETTINGS)
# =============================================================================

# [WARN] NOTE: AGGRESSIVE RECOVERY MODE - Higher lot sizes and more levels
# The fix prevents recovery orders from spawning more recovery (300 trade bug fixed)

# =============================================================================
# GRID PARAMETERS (PYRAMID OVERLAPPING STRATEGY)
# =============================================================================
# Grid creates OVERLAPPING positions in same direction when in profit
# Example: BUY @ 1.1000 -> +10 pips -> Open Grid 1: BUY @ 1.1010
#                       -> +20 pips -> Open Grid 2: BUY @ 1.1020
# Result: 3 independent BUY positions running (parent + 2 grids)
#
# NOTE: Per-instrument grid settings (grid_spacing_pips, max_grid_levels) are
#       defined in instruments_config.py and OVERRIDE these globals.
#       Current: All instruments use 10 pips spacing, max 2 grid levels.

GRID_ENABLED = True  # System-wide enable/disable

# KILL SWITCH: Disable negative grid (grid on losing trades)
DISABLE_NEGATIVE_GRID = True  # Only allow grid on profitable positions

# Grid lot size (applies to all instruments)
GRID_LOT_SIZE = 0.04

# FALLBACK DEFAULTS (only used if instrument not in instruments_config.py)
GRID_SPACING_PIPS = 10  # Matches current instrument settings
MAX_GRID_LEVELS = 2     # Matches current instrument settings

# =============================================================================
# HEDGING PARAMETERS (AGGRESSIVE RECOVERY SETTINGS)
# =============================================================================
# NOTE: Per-instrument hedge_trigger_pips defined in instruments_config.py
#       (EURUSD: 45 pips, GBPUSD: 55 pips, USDJPY: 50 pips)

HEDGE_ENABLED = True  # Re-enabled with orphan cascade protection (fc39955)
HEDGE_RATIO = 1.5  # 1.5x hedge ratio (conservative)
MAX_HEDGES_PER_POSITION = 1  # Strict limit: ONE hedge per position
MAX_HEDGE_VOLUME = 0.20  # Safety cap: max 0.20 lots per hedge

# FALLBACK DEFAULT (only used if instrument not in instruments_config.py)
HEDGE_TRIGGER_PIPS = 50  # Matches USDJPY current setting

# Recovery Stack Drawdown Protection
# Close entire stack (original + grid + hedge + DCA) if net loss exceeds this multiplier
# Example: Original trade expects $5 profit -> kill stack at $20 loss (4x)
STACK_DRAWDOWN_MULTIPLIER = 4.0  # Cut recovery stack at 4x expected profit loss

# =============================================================================
# DCA/MARTINGALE PARAMETERS (AGGRESSIVE RECOVERY SETTINGS)
# =============================================================================
# NOTE: Per-instrument DCA settings (dca_trigger_pips, max_dca_levels) are
#       defined in instruments_config.py and OVERRIDE these globals.
#       Current: EURUSD=30 pips, GBPUSD=40 pips, USDJPY=35 pips trigger
#                All instruments: max 4 DCA levels (ML-recommended increase)

DCA_ENABLED = True  # System-wide enable/disable
DCA_MULTIPLIER = 2.0  # ML-recommended: Increased from 1.5 to 2.0 (maxed trades losing avg $-3.76)

# FALLBACK DEFAULTS (only used if instrument not in instruments_config.py)
DCA_TRIGGER_PIPS = 35  # Matches USDJPY current setting
DCA_MAX_LEVELS = 4     # ML-recommended: Increased from 3 to 4

# =============================================================================
# PER-STACK STOP LOSS MANAGEMENT
# =============================================================================
# Limits maximum loss per recovery stack (original + DCA + hedge as one unit)
# Prevents catastrophic drawdown situations where both DCA and hedges are underwater

# Enable per-stack stop loss checks
ENABLE_STACK_STOPS = True

# Maximum loss for DCA-only stacks (no hedge deployed)
# Range: -$5 to -$25 depending on risk tolerance
# Test value: -$25 (more room for DCA to work)
DCA_ONLY_MAX_LOSS = -25.0

# Maximum loss for DCA+Hedge stacks (hedge deployed)
# Range: -$15 to -$50 depending on risk tolerance
# Test value: -$50 (allows both DCA and hedge room to recover)
DCA_HEDGE_MAX_LOSS = -50.0

# =============================================================================
# CASCADE STOP PROTECTION (TREND DETECTION VIA STOP-OUTS)
# =============================================================================
# Detects when multiple stops trigger in short time (range→trend transition)
# Closes all underwater stacks to prevent cascade of losses

# Enable cascade protection
ENABLE_CASCADE_PROTECTION = True

# Time window to detect multiple stop-outs (minutes)
# If 2+ stops occur within this window, cascade is triggered
STOP_OUT_WINDOW_MINUTES = 30

# Number of stop-outs in window that triggers cascade
# 2 = after 2nd stop, close ALL underwater stacks
CASCADE_THRESHOLD = 2

# How long to block new trades after cascade (minutes)
# Gives trend time to develop before re-entering
TREND_BLOCK_MINUTES = 60

# Minimum ADX to confirm trend during cascade
# If avg ADX < this at cascade, don't block trades (might be noise)
CASCADE_ADX_THRESHOLD = 25

# =============================================================================
# RISK MANAGEMENT (AGGRESSIVE SETTINGS)
# =============================================================================

# Base lot size for initial positions
BASE_LOT_SIZE = 0.04  # Updated to 0.04 with partial close strategy

# Number of initial trades to open per signal
# Opens multiple separate positions instead of one large position
# Example: INITIAL_TRADE_COUNT = 2 -> Opens 2 separate trades with BASE_LOT_SIZE each
INITIAL_TRADE_COUNT = 1  # DEFAULT: 1 (single trade)

# Risk per trade (if using dynamic position sizing)
RISK_PERCENT = 1.0

# Use fixed lot size (True) or calculate based on risk % (False)
USE_FIXED_LOT_SIZE = True

# Maximum total exposure across all positions
MAX_TOTAL_LOTS = 15.0  # AGGRESSIVE: Increased from 5.04 to accommodate larger recovery stacks

# Maximum drawdown before stopping
MAX_DRAWDOWN_PERCENT = 10.0

# Stop loss (if used)
STOP_LOSS_PIPS = None  # EA appears to not use hard stops

# Take profit (if used)
TAKE_PROFIT_PIPS = None  # EA uses VWAP reversion

# =============================================================================
# TIMING PARAMETERS
# =============================================================================

# Trading sessions (discovered from session analysis)
TRADE_SESSIONS = {
    'tokyo': {'start': '00:00', 'end': '09:00', 'enabled': True},
    'london': {'start': '08:00', 'end': '17:00', 'enabled': True},
    'new_york': {'start': '13:00', 'end': '22:00', 'enabled': True},
    'sydney': {'start': '22:00', 'end': '07:00', 'enabled': True},
}

# Days to trade
TRADE_DAYS = [0, 1, 2, 3, 4]  # Monday-Friday

# =============================================================================
# TIME FILTERS - STRATEGY-SPECIFIC TRADING WINDOWS
# =============================================================================

# ============================================================================
# STRATEGY ON/OFF SWITCHES - Quick enable/disable for each strategy
# ============================================================================

# Enable Mean Reversion strategy
# Set to False to disable ALL mean reversion trading (maintains BO only)
MEAN_REVERSION_ENABLED = True

# Enable Breakout strategy
# Set to False to disable ALL breakout trading (maintains MR only)
BREAKOUT_ENABLED = True

# Enable time filtering (False = trade all hours for enabled strategies)
# If False, enabled strategies will trade 24/7 regardless of configured windows
# TESTING: Disabled to rely solely on ADX & candle lookback filters
ENABLE_TIME_FILTERS = False

# ============================================================================

# =============================================================================
# BROKER TIMEZONE CONFIGURATION
# =============================================================================

# MT5 brokers use different server timezones. Set your broker's GMT offset here.
# This is CRITICAL for time filters to work correctly!
#
# Common broker timezones:
#   0  = GMT/UTC (rare, used by some brokers)
#   +2 = GMT+2 (EET - most European brokers in winter)
#   +3 = GMT+3 (EET summer / some brokers use this year-round)
#   -4 = GMT-4 (EDT - some US brokers in summer)
#   -5 = GMT-5 (EST - some US brokers in winter)
#
# HOW TO FIND YOUR BROKER'S OFFSET:
# 1. Check current GMT time: https://time.is/GMT
# 2. Check your MT5 terminal time (bottom right corner)
# 3. Calculate: MT5 time - GMT time = your offset
#    Example: MT5 shows 14:00, GMT is 12:00 -> offset is +2
#
# IMPORTANT: All trading hours in this config are in GMT/UTC.
# The bot will automatically convert broker time to GMT using this offset.
BROKER_GMT_OFFSET = 0  # SET THIS TO YOUR BROKER'S OFFSET!

# =============================================================================

# MEAN REVERSION TRADING HOURS (GMT/UTC)
# Based on analysis: Best win rates (79.3% at Value Area, 73.5% at VWAP ±2σ)
# Hours with highest success: 05:00 (100%), 12:00 (100%), 07:00 (93%), 06:00 (86%), 09:00 (80%)
# EXPANDED: Added 10, 11, 13 based on 3-week diagnostic showing high-quality signals (scores 10-13)
MEAN_REVERSION_HOURS = [5, 6, 7, 9, 10, 11, 12, 13]

# MEAN REVERSION TRADING DAYS (0=Monday, 6=Sunday)
# Best days: Tuesday (73%), Wednesday (70%), Thursday (69%)
# Monday included per user request
MEAN_REVERSION_DAYS = [0, 1, 2, 3]  # Mon, Tue, Wed, Thu

# MEAN REVERSION SESSIONS
# Tokyo: 74% win rate, London early: 68% win rate
# Avoid New York: 53% win rate
MEAN_REVERSION_SESSIONS = ['tokyo', 'london']

# BREAKOUT TRADING HOURS (UTC)
# Based on analysis: High volatility periods
# Hours: 03:00 (70% win, high ATR), 14:00 (London/NY overlap)
# EXPANDED: Added 18-23 based on 3-week diagnostic showing HIGHEST quality signals (scores 12-13!)
# OPTIMIZED: Added 04:00 (21 trades, $230.44) and 08:00 (10 trades, $101.88) from historical analysis
BREAKOUT_HOURS = [3, 4, 8, 14, 15, 16, 18, 19, 20, 21, 22, 23]  # Full optimization based on 3-week data

# BREAKOUT TRADING DAYS
# Best days: Tuesday (62% win, high volatility), Friday (trend exhaustion)
# Monday (week open breakouts)
# OPTIMIZED: Added Wed (2) and Thu (3) - captures 21 trades, $235.77 additional profit
BREAKOUT_DAYS = [0, 1, 2, 3, 4]  # Mon-Fri (full week coverage)

# BREAKOUT SESSIONS
# London/NY overlap = highest volatility for breakouts
# Tokyo included for 03:00 breakout hour (70% win rate in analysis)
BREAKOUT_SESSIONS = ['tokyo', 'london', 'new_york']

# =============================================================================
# BREAKOUT STRATEGY PARAMETERS
# =============================================================================

# NOTE: To enable/disable breakout strategy, see BREAKOUT_ENABLED at top of file
#       (STRATEGY ON/OFF SWITCHES section, line ~190)

# Breakout detection parameters (relaxed for testing to generate more signals)
BREAKOUT_LOOKBACK = 20  # Bars to identify range high/low
BREAKOUT_VOLUME_MULTIPLIER = 1.2  # Volume must be 1.2x average (was 1.5)
BREAKOUT_ATR_MULTIPLIER = 0.8  # ATR must be 0.8x median (was 1.2 - now allows lower volatility)

# Breakout entry conditions
BREAKOUT_MIN_RANGE_PIPS = 15  # Minimum range size to consider for breakout (was 20)
BREAKOUT_CLOSE_BEYOND_LEVEL = True  # Candle must close beyond level (not just wick)

# Breakout momentum filters (relaxed thresholds)
BREAKOUT_RSI_BUY_THRESHOLD = 55  # RSI > 55 for bullish breakouts (was 60)
BREAKOUT_RSI_SELL_THRESHOLD = 45  # RSI < 45 for bearish breakouts (was 40)

# Breakout position sizing (more conservative due to lower win rate)
BREAKOUT_LOT_SIZE_MULTIPLIER = 0.5  # Use 50% of normal lot size

# Breakout profit targets
BREAKOUT_TARGET_METHOD = 'range_projection'  # 'range_projection', 'atr_multiple', 'lvn'
BREAKOUT_TARGET_MULTIPLIER = 1.0  # 1x range for 'range_projection'
BREAKOUT_ATR_TARGET_MULTIPLE = 2.0  # 2x ATR for 'atr_multiple'

# Breakout stop loss (tight stops - breakouts should not reverse)
BREAKOUT_STOP_PERCENT = 0.2  # 20% of range back from breakout level

# =============================================================================
# POSITION MANAGEMENT
# =============================================================================

# Maximum open positions
MAX_OPEN_POSITIONS = 3  # Reduced from 10 for safety

# Maximum positions per symbol
MAX_POSITIONS_PER_SYMBOL = 1  # Only 1 position per symbol at a time

# =============================================================================
# EXIT MANAGEMENT (Net Profit Target + Time Limit + Partial Close)
# =============================================================================

# Net profit target for recovery stacks
# Close entire stack (original + grid + hedge + DCA) when combined P&L reaches this
PROFIT_TARGET_PERCENT = 0.5  # AGGRESSIVE: 0.5% target (easier to hit with larger lots)

# Time-based exit for stuck positions
# Auto-close recovery stack after this many hours if still open
MAX_POSITION_HOURS = 12  # AGGRESSIVE: 12 hours max (was 4) - gives recovery time to work

# =============================================================================
# PARTIAL CLOSE (SCALE OUT) SETTINGS
# =============================================================================

# Enable partial close functionality
PARTIAL_CLOSE_ENABLED = True

# Partial close levels (percentage of position to close at each milestone)
# Closes portions of the position as it moves toward TP
PARTIAL_CLOSE_LEVELS = [
    {'percent_to_tp': 50, 'close_percent': 50},  # Close 50% at halfway to TP
    {'percent_to_tp': 75, 'close_percent': 50},  # Close 50% of remaining (25% total) at 75% to TP
    # Final 25% closes at 100% TP or VWAP reversion
]

# Minimum profit required to enable partial close (in pips)
# Prevents partial close on small moves
PARTIAL_CLOSE_MIN_PROFIT_PIPS = 10

# Apply partial close to recovery stacks (grid/hedge/DCA)
PARTIAL_CLOSE_RECOVERY = False  # Only apply to original positions

# Trail stop on remaining position after first partial close
TRAIL_STOP_AFTER_PARTIAL = True
TRAIL_STOP_DISTANCE_PIPS = 15  # Trail stop 15 pips behind price

# =============================================================================
# DATA MANAGEMENT
# =============================================================================

# Historical data bars to load
HISTORY_BARS = {
    'H1': 10000,  # ~416 days
    'D1': 500,    # ~2 years
    'W1': 104,    # ~2 years
}

# Data cache refresh interval (minutes)
DATA_REFRESH_INTERVAL = 60

# =============================================================================
# LOGGING
# =============================================================================

LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_bot.log'
LOG_TRADES = True
LOG_SIGNALS = True

# =============================================================================
# BACKTESTING
# =============================================================================

BACKTEST_MODE = False
BACKTEST_START_DATE = '2024-01-01'
BACKTEST_END_DATE = '2024-12-31'
BACKTEST_INITIAL_BALANCE = 10000

# =============================================================================
# MT5 CONNECTION
# =============================================================================

MT5_TIMEOUT = 60000  # 60 seconds
MT5_MAGIC_NUMBER = 987654  # Unique identifier for bot trades
