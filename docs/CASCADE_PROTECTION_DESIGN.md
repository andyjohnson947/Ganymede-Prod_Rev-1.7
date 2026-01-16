# Cascade Stop Protection - Implementation Design

## Problem Statement
Bot prefers ranging markets but ranges often transform into trends, causing multiple positions to go underwater simultaneously. Stop losses trigger one-by-one (cascade), causing multiple -$25/-$50 losses before recognizing the trend.

## Solution: Stop-Out Cascade Detection

### Concept
**First stop-out:** Close that stack only (might be isolated)
**Second stop-out within 30min:** CASCADE DETECTED → Close ALL underwater stacks immediately

### Implementation Components

---

## 1. StopOutTracker Class ✅ IMPLEMENTED

**Location:** `trading_bot/strategies/recovery_manager.py` (lines 91-196)

**Functionality:**
- Tracks stop-out events in a 30-minute rolling window
- Records: timestamp, ticket, symbol, loss, ADX, stack_type
- Logs to `logs/stop_out_events.log` for analysis
- Checks if cascade condition met (2+ stops in window)
- Returns cascade info including average ADX

**Key Methods:**
- `add_stop_out()` - Record a stop event
- `check_cascade()` - Detect if cascade triggered
- `_log_to_file()` - Write to stop-out log

---

## 2. RecoveryManager Integration (TODO)

**Modifications Needed:**

### A. Initialize StopOutTracker in `__init__`
```python
def __init__(self):
    """Initialize recovery manager"""
    self.tracked_positions = {}
    self.stop_out_tracker = StopOutTracker() if ENABLE_CASCADE_PROTECTION else None
```

### B. Modify `check_stack_stop_loss()` to Record Stops

**Current (line 1283):**
```python
def check_stack_stop_loss(self, ticket: int, mt5_positions: List[Dict]) -> Optional[Dict]:
    # ... existing logic ...

    if net_profit <= stop_loss_limit:
        print(f"[STOP] PER-STACK STOP LOSS HIT for position #{ticket}")
        # ... logging ...

        return {
            'action': 'close_stack',
            'reason': 'per_stack_stop_loss',
            'ticket': ticket,
            # ... other fields ...
        }
```

**Modified (add ADX retrieval and stop-out recording):**
```python
def check_stack_stop_loss(
    self,
    ticket: int,
    mt5_positions: List[Dict],
    current_adx: Optional[float] = None  # NEW parameter
) -> Optional[Dict]:
    # ... existing logic ...

    if net_profit <= stop_loss_limit:
        print(f"[STOP] PER-STACK STOP LOSS HIT for position #{ticket}")
        # ... logging ...

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
```

### C. Add Method to Get All Underwater Stacks

```python
def get_underwater_stacks(self, mt5_positions: List[Dict]) -> List[int]:
    """
    Get all tracked positions that are currently underwater

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
```

---

## 3. Confluence Strategy Integration (TODO)

**Location:** `trading_bot/strategies/confluence_strategy.py`

### A. Add Cascade Block Tracking

**In `__init__`:**
```python
def __init__(self, mt5_manager, test_mode=False):
    # ... existing code ...

    # Cascade protection - blocks new trades after cascade close
    self.cascade_blocks = {}  # {symbol: block_until_time}
```

### B. Modify Per-Stack Stop Loss Check (line ~405-416)

**Current:**
```python
# 0.25. Check per-stack stop loss (dollar-based limit per stack)
stack_stop = self.recovery_manager.check_stack_stop_loss(
    ticket=ticket,
    mt5_positions=all_positions
)
if stack_stop:
    print(f"\n[STOP] Per-stack stop loss triggered")
    print(f"   Stack type: {stack_stop['stack_type']}")
    print(f"   Loss: ${stack_stop['loss_amount']:.2f}")
    print(f"   Limit: ${stack_stop['stop_loss_limit']:.2f}")
    self._close_recovery_stack(ticket)
    continue
```

**Modified (add ADX and cascade detection):**
```python
# 0.25. Check per-stack stop loss (dollar-based limit per stack)

# Get current ADX for this symbol
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
    if ENABLE_CASCADE_PROTECTION:
        cascade_info = self.recovery_manager.stop_out_tracker.check_cascade()

        if cascade_info:
            # CASCADE DETECTED - close all underwater stacks
            print(f"\n{'='*70}")
            print(f"[CASCADE] MULTIPLE STOP-OUTS DETECTED - CLOSING ALL UNDERWATER STACKS")
            print(f"{'='*70}")
            print(f"   Stops in 30min: {cascade_info['stop_count']}")
            print(f"   Avg ADX: {cascade_info['avg_adx']:.1f}" if cascade_info['avg_adx'] else "   Avg ADX: N/A")
            print(f"   Trend confirmed: {cascade_info['trend_confirmed']}")
            print(f"   Symbols affected: {', '.join(cascade_info['symbols'])}")

            # Get all underwater stacks
            underwater_tickets = self.recovery_manager.get_underwater_stacks(all_positions)

            if underwater_tickets:
                print(f"\n   Closing {len(underwater_tickets)} underwater stack(s):")
                for uw_ticket in underwater_tickets:
                    uw_profit = self.recovery_manager.calculate_net_profit(uw_ticket, all_positions)
                    uw_symbol = self.recovery_manager.tracked_positions[uw_ticket]['symbol']
                    print(f"     #{uw_ticket} ({uw_symbol}): ${uw_profit:.2f}")
                    self._close_recovery_stack(uw_ticket)

                # Block new trades for affected symbols
                if cascade_info['trend_confirmed']:
                    block_until = get_current_time() + timedelta(minutes=TREND_BLOCK_MINUTES)
                    for affected_symbol in cascade_info['symbols']:
                        self.cascade_blocks[affected_symbol] = block_until
                        print(f"\n   [BLOCK] {affected_symbol} trades blocked for {TREND_BLOCK_MINUTES} minutes")
                        print(f"          Market likely trending (ADX: {cascade_info['avg_adx']:.1f})")
            else:
                print(f"   No other underwater stacks found")

            print(f"{'='*70}\n")

    continue
```

### C. Check Cascade Block in Signal Detection

**In `_process_signals()` method (before opening trades):**
```python
# Check if symbol is blocked due to cascade
if symbol in self.cascade_blocks:
    block_until = self.cascade_blocks[symbol]
    if get_current_time() < block_until:
        time_remaining = (block_until - get_current_time()).total_seconds() / 60
        logger.debug(f"Symbol {symbol} blocked due to cascade (trend detected) - {time_remaining:.0f}min remaining")
        continue
    else:
        # Block expired, remove it
        del self.cascade_blocks[symbol]
```

---

## 4. Configuration Parameters ✅ IMPLEMENTED

**Location:** `trading_bot/config/strategy_config.py` (lines 180-203)

```python
# CASCADE STOP PROTECTION (TREND DETECTION VIA STOP-OUTS)
ENABLE_CASCADE_PROTECTION = True
STOP_OUT_WINDOW_MINUTES = 30      # Time window for multiple stops
CASCADE_THRESHOLD = 2              # 2 stops = cascade
TREND_BLOCK_MINUTES = 60           # Block trades for 60min after cascade
CASCADE_ADX_THRESHOLD = 25         # Confirm trend if avg ADX >= 25
```

---

## 5. Stop-Out Analysis Log

**Location:** `logs/stop_out_events.log`

**Format:**
```
2026-01-12T10:15:23 | #12345 | EURUSD | $25.50 | ADX:32.5 | DCA-only
2026-01-12T10:22:41 | #12346 | EURUSD | $48.20 | ADX:35.2 | DCA+Hedge
2026-01-12T10:25:15 | #12347 | GBPUSD | $22.00 | ADX:28.1 | DCA-only
```

**Analysis Script (TODO):**
```bash
# Count stops by ADX range
grep "ADX:" logs/stop_out_events.log | awk -F'ADX:' '{print $2}' | cut -d' ' -f1 | sort -n

# Most common symbols for stops
grep "#" logs/stop_out_events.log | awk -F'|' '{print $3}' | sort | uniq -c | sort -rn

# Average loss by stack type
grep "DCA-only" logs/stop_out_events.log | awk -F'$' '{print $2}' | cut -d' ' -f1 | awk '{sum+=$1; n++} END {print "Avg:", sum/n}'
```

---

## Behavior Examples

### Example 1: Single Stop-Out (No Cascade)
```
[STOP] PER-STACK STOP LOSS HIT for position #12345
   Symbol: EURUSD
   Stack type: DCA-only
   Current P&L: $-25.50
   Stop loss limit: $-25.00
   [ACTION] Closing entire recovery stack to limit losses

[STOP-OUT] Event #12345 recorded:
   Symbol: EURUSD
   Loss: $25.50
   ADX: 22.5
   Stack: DCA-only
   Recent stops (30min): 1

→ Stack closed, no cascade (only 1 stop in window)
```

### Example 2: Cascade Triggered
```
[STOP] PER-STACK STOP LOSS HIT for position #12346
   Symbol: EURUSD
   Stack type: DCA+Hedge
   Current P&L: $-48.20
   Stop loss limit: $-50.00

[STOP-OUT] Event #12346 recorded:
   Symbol: EURUSD
   Loss: $48.20
   ADX: 35.2
   Stack: DCA+Hedge
   Recent stops (30min): 2

======================================================================
[CASCADE] MULTIPLE STOP-OUTS DETECTED - CLOSING ALL UNDERWATER STACKS
======================================================================
   Stops in 30min: 2
   Avg ADX: 33.9
   Trend confirmed: True
   Symbols affected: EURUSD

   Closing 3 underwater stack(s):
     #12346 (EURUSD): $-48.20
     #12350 (EURUSD): $-12.50
     #12352 (EURUSD): $-8.75

   [BLOCK] EURUSD trades blocked for 60 minutes
          Market likely trending (ADX: 33.9)
======================================================================

→ All EURUSD stacks closed, new trades blocked for 1 hour
```

### Example 3: Cascade Without Trend Confirmation (Low ADX)
```
[CASCADE] MULTIPLE STOP-OUTS DETECTED - CLOSING ALL UNDERWATER STACKS
   Stops in 30min: 2
   Avg ADX: 18.5
   Trend confirmed: False
   Symbols affected: GBPUSD

   Closing 2 underwater stack(s):
     #12355 (GBPUSD): $-23.00
     #12357 (GBPUSD): $-15.50

   No trade block applied (ADX < 25 - likely noise, not trend)
======================================================================

→ Stacks closed but trades NOT blocked (ADX too low for trend confirmation)
```

---

## Benefits

1. **Faster Response to Trends**
   - 1st stop: Individual problem
   - 2nd stop: Systemic problem → close all

2. **Prevents Cascade Losses**
   - Old: Take 5x -$25 losses before recognizing trend
   - New: Take 2x stops, close rest immediately

3. **Data-Driven Trend Detection**
   - Logs ADX at each stop
   - Can analyze: "Do stops happen during trends?"
   - Validates assumption that stops = trend changes

4. **Smart Trade Blocking**
   - Only blocks if ADX confirms trend
   - Per-symbol blocking (EURUSD trending ≠ GBPUSD trending)
   - Auto-expires after 60 minutes

---

## Tuning Parameters

**If too sensitive (closes too often):**
- Increase `CASCADE_THRESHOLD` from 2 to 3
- Increase `STOP_OUT_WINDOW_MINUTES` from 30 to 45
- Increase `CASCADE_ADX_THRESHOLD` from 25 to 30

**If not sensitive enough (still taking cascade losses):**
- Keep `CASCADE_THRESHOLD` at 2
- Decrease `STOP_OUT_WINDOW_MINUTES` from 30 to 20
- Decrease `CASCADE_ADX_THRESHOLD` from 25 to 20

**To disable:**
```python
ENABLE_CASCADE_PROTECTION = False
```

---

## Implementation Status

✅ **Completed:**
- StopOutTracker class
- Configuration parameters
- Stop-out logging to file

⏳ **Remaining:**
- Initialize StopOutTracker in RecoveryManager
- Modify check_stack_stop_loss() to pass ADX and record stops
- Add get_underwater_stacks() method
- Integrate cascade detection in confluence_strategy.py
- Add cascade block checking before trade entry

---

## Testing Recommendations

1. **Monitor logs/stop_out_events.log** after deployment
2. **Check if stops correlate with high ADX** (validates assumption)
3. **Analyze cascade frequency** - should be rare (2-3 times per week max)
4. **Tune thresholds** based on actual market behavior

This protects against your exact scenario: ranges breaking into trends causing batch drawdowns.
