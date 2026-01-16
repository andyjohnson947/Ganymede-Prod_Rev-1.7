# Trailing Stop Integration Guide

## Status: Implemented (Needs Integration into Main Loop)

The trailing stop system has been fully implemented in `recovery_manager.py` and configured in `instruments_config.py`.

## What's Complete

✅ Configuration added to all instruments (EURUSD, GBPUSD, USDJPY)
✅ Trailing stop state tracking in tracked_positions
✅ ATR calculation method (2×ATR with 25-50 pip bounds)
✅ Trailing stop activation method (after PC2)
✅ Trailing stop update method (moves with price)
✅ Trailing stop check method (detects when hit)
✅ Partial close tracking (PC1, PC2)

## Integration Required

The following needs to be added to the main position monitoring loop (likely in `confluence_strategy.py` or similar):

### 1. On Every Position Check

```python
# Get current positions from MT5
positions = mt5.get_positions()

for position in positions:
    ticket = position.ticket
    symbol = position.symbol
    current_price = position.price_current
    current_profit_pips = calculate_profit_pips(position)

    # Get instrument TP settings
    instrument_config = get_instrument_config(symbol)
    tp_settings = instrument_config['take_profit']

    # =================================================================
    # STEP 1: Update trailing stop (if active)
    # =================================================================
    recovery_manager.update_trailing_stop(ticket, current_price)

    # =================================================================
    # STEP 2: Check if trailing stop hit
    # =================================================================
    if recovery_manager.check_trailing_stop(ticket, current_price):
        # Close entire position
        logger.info(f"[TRAIL] Closing #{ticket} - trailing stop hit")
        mt5.close_position(ticket)
        continue  # Move to next position

    # =================================================================
    # STEP 3: Check for Partial Close 1
    # =================================================================
    if not position_state['partial_1_closed']:
        pc1_pips = tp_settings.get('partial_1_pips', 10)
        if current_profit_pips >= pc1_pips:
            # Close 25%
            pc1_percent = tp_settings.get('partial_1_percent', 0.25)
            close_volume = position.volume * pc1_percent

            logger.info(f"[PC1] Closing 25% of #{ticket} at {pc1_pips} pips")
            mt5.close_partial(ticket, close_volume)

            # Mark PC1 as closed
            position_state['partial_1_closed'] = True

            # DISABLE VWAP exits after PC1
            logger.info(f"[PC1] VWAP exits disabled for #{ticket}")

    # =================================================================
    # STEP 4: Check for Partial Close 2 + Activate Trailing
    # =================================================================
    if position_state['partial_1_closed'] and not position_state['partial_2_closed']:
        pc2_pips = tp_settings.get('partial_2_pips', 20)
        if current_profit_pips >= pc2_pips:
            # Close 50% of original (66.7% of remaining)
            pc2_percent = tp_settings.get('partial_2_percent', 0.50)
            # Note: We're 50% of ORIGINAL volume, which is 66.7% of current
            close_volume = position.volume * (pc2_percent / 0.75)  # Adjust for already-closed 25%

            logger.info(f"[PC2] Closing 50% of #{ticket} at {pc2_pips} pips")
            mt5.close_partial(ticket, close_volume)

            # Mark PC2 as closed
            position_state['partial_2_closed'] = True

            # ACTIVATE TRAILING STOP for remaining 25%
            if tp_settings.get('trailing_stop_enabled', False):
                logger.info(f"[PC2] Activating trailing stop for #{ticket}")
                recovery_manager.activate_trailing_stop(ticket, current_price, tp_settings)

    # =================================================================
    # STEP 5: VWAP Exit (ONLY if profit < vwap_exit_max_pips)
    # =================================================================
    vwap_enabled = tp_settings.get('vwap_exit_enabled', False)
    vwap_max_pips = tp_settings.get('vwap_exit_max_pips', 10)

    # Only use VWAP if:
    # 1. VWAP is enabled
    # 2. Profit is below threshold (quick scalp)
    # 3. PC1 has NOT triggered yet
    if vwap_enabled and current_profit_pips < vwap_max_pips and not position_state['partial_1_closed']:
        # Check VWAP mean reversion exit conditions
        vwap_should_exit = check_vwap_exit_conditions(position)

        if vwap_should_exit:
            logger.info(f"[VWAP] Closing #{ticket} at {current_profit_pips:.1f} pips (quick scalp)")
            mt5.close_position(ticket)
            continue
```

### 2. Position State Persistence

The position state (partial_1_closed, partial_2_closed, trailing_stop_active, etc.) is already tracked in `recovery_manager.tracked_positions`. No additional persistence needed.

### 3. Integration Points

**Main files to modify:**
- `trading_bot/strategies/confluence_strategy.py` - Main trading strategy
- Look for the position monitoring loop
- Add the integration code above

**Helper functions needed:**
- `calculate_profit_pips(position)` - Calculate current profit in pips
- `check_vwap_exit_conditions(position)` - Check if VWAP mean reversion complete
- `mt5.close_partial(ticket, volume)` - Close partial volume
- `mt5.close_position(ticket)` - Close entire position

## Configuration

All instruments are configured in `trading_bot/portfolio/instruments_config.py`:

```python
'take_profit': {
    'partial_1_pips': 10,           # First partial
    'partial_1_percent': 0.25,
    'partial_2_pips': 20,           # Second partial
    'partial_2_percent': 0.50,
    'trailing_stop_enabled': True,  # Enable trailing
    'trailing_stop_atr_multiplier': 2.0,  # 2×ATR
    'trailing_stop_min_pips': 25,   # Minimum trail
    'trailing_stop_max_pips': 50,   # Maximum trail
    'vwap_exit_enabled': True,      # Enable VWAP
    'vwap_exit_max_pips': 10,       # VWAP only < 10 pips
}
```

## Testing

Once integrated, test with paper trading:

1. **Small Move (1-9 pips)**: Should close at VWAP (before PC1)
2. **Medium Move (10-19 pips)**: Should hit PC1, then VWAP disabled
3. **Good Move (20-35 pips)**: Should hit PC1 + PC2, trailing activates
4. **Large Move (35-50+ pips)**: Trailing stop should capture extended move
5. **Reversal**: Trailing stop should close at 2×ATR from peak

## Expected Results

- Average profit per trade: **$2.28 → $8-12** (3.5-5x improvement)
- Captures 50-100 pip moves (currently exiting at 1-9 pips)
- Protects against reversals with dynamic trailing
- Keeps quick scalp exits via VWAP (<10 pips)

## Rollback

If issues arise, set in instruments_config.py:
```python
'trailing_stop_enabled': False,
'vwap_exit_enabled': True,
'vwap_exit_max_pips': 999,  # No limit
```

This reverts to old behavior (VWAP exits at any profit level).
