# Recovery Strategy Performance Expectations

## Overview
This document explains what to expect from DCA and hedge deployments with the new conservative settings, and how the per-stack stop losses protect your account.

---

## Current Settings (Conservative)

### DCA Parameters
- **Trigger:** 20 pips underwater
- **Max Levels:** 3 (was 6)
- **Multiplier:** 1.5x (was 2.0x)
- **Volume Progression:** 0.04 → 0.06 → 0.09 lots

### Hedge Parameters
- **Trigger:** 50 pips underwater (was 8)
- **Ratio:** 1.5x (was 5.0x)
- **Max Volume:** 0.20 lots (was 0.50)

### Stop Loss Limits
- **DCA-only stacks:** -$25 max loss
- **DCA+Hedge stacks:** -$50 max loss

---

## Expected Performance by Recovery Type

### 1. Initial-Only Trades (No Recovery)
**Historical Performance:**
- Win Rate: 100%
- Avg Profit: $0.52
- Avg Hold Time: 18 minutes

**Expected with New Settings:**
- Same performance (no changes to entry)
- Quick exits, minimal risk
- Baseline profitability

**Stop Loss:** Uses DCA-only limit (-$25) as safety net

---

### 2. DCA-Only Trades

**When DCA Triggers:**
- Position is 20 pips underwater
- For 0.04 lot on EURUSD: approximately -$8 floating loss
- DCA adds volume to average down entry price

**Volume Progression (New Conservative):**
```
Initial:  0.04 lots at entry price
DCA L1:   0.06 lots at -20 pips  → Total: 0.10 lots
DCA L2:   0.09 lots at -40 pips  → Total: 0.19 lots
DCA L3:   0.14 lots at -60 pips  → Total: 0.33 lots (MAXED OUT)
```

**Expected Outcomes:**
- **Best Case:** Price reverses after DCA L1
  - Final profit: $2-5 (pays for DCA cost + profit)
  - Hold time: 1-3 hours

- **Typical Case:** Price reverses after DCA L2
  - Final profit: $0-3 (breakeven to small profit)
  - Hold time: 3-6 hours

- **Worst Case:** All 3 DCA levels deploy, price continues down
  - Floating loss at -60 pips with 0.33 lots ≈ -$20 to -$25
  - **Stop loss triggers at -$25** → Cuts loss before catastrophic drawdown
  - Alternative: Price reverses, recovers to small profit/loss

**Stop Loss Protection:**
- Old settings: Could reach -$100+ with 6 DCA levels (0.04→2.56 lots!)
- New settings: Capped at -$25, prevents runaway losses

---

### 3. Hedge-Only Trades (Rare)

**When Hedge Triggers WITHOUT DCA:**
- Position is 50 pips underwater (very deep)
- No DCA was deployed (unusual)
- Hedge ratio: 1.5x of initial volume

**Example:**
- Initial: 0.04 lot SELL at 1.1000
- Price moves to 1.1050 (50 pips up) = -$20 loss
- Hedge: 0.06 lot BUY at 1.1050

**Expected Outcomes:**
- Price reverses down: Original recovers, hedge closes at loss
- Price continues up: Hedge profits offset original loss
- Both underwater: Unlikely with 50 pip separation

**Note:** With hedge trigger at 50 pips, this scenario is rare. DCA should handle most drawdowns before hedge triggers.

---

### 4. DCA+Hedge Trades (Full Recovery)

**When Both Deploy:**
1. Price moves 20 pips down → DCA L1 activates (0.06 lots)
2. Price moves 40 pips down → DCA L2 activates (0.09 lots)
3. Price moves 50 pips down → **Hedge activates** (1.5x initial = 0.06 lots opposite)
4. Price moves 60 pips down → DCA L3 activates (0.14 lots)

**Position Structure at Max:**
```
ORIGINAL: 0.04 lots (direction: original)
DCA L1:   0.06 lots (same direction)
DCA L2:   0.09 lots (same direction)
DCA L3:   0.14 lots (same direction)
HEDGE:    0.06 lots (OPPOSITE direction)

Same Direction Total: 0.33 lots
Hedge: 0.06 lots opposite
Net Exposure: 0.27 lots in original direction
```

**Expected Outcomes:**
- **Best Case:** Price reverses after hedge, all positions profit
  - Final profit: $5-15 (large recovery profit)
  - Hold time: 4-8 hours

- **Typical Case:** Mixed results, net small profit/loss
  - Same-direction positions: Recover to breakeven
  - Hedge: Small loss or profit depending on timing
  - Final P&L: -$10 to +$10
  - Hold time: 6-12 hours (max time limit)

- **Worst Case:** Both directions underwater (double-sided loss)
  - Same-direction: Still underwater at -40 pips
  - Hedge: Moved against us by -20 pips
  - Total loss: -$35 to -$50
  - **Stop loss triggers at -$50** → Prevents worse outcomes

**Stop Loss Protection:**
- Old settings: Could reach -$600+ (example from user's -$689 situation)
  - 6 DCA levels = 2.56 lots
  - 5.0x hedge ratio = 1.28 lots opposite
  - Both underwater = catastrophic
- New settings: Capped at -$50, closes entire stack

---

## Stop Loss Validation Strategy

### How to Track Performance

Run this command weekly:
```bash
python3 ml_system/scripts/analyze_recovery_outcomes.py
```

This will show:
1. Average profit for each recovery type
2. Worst loss for each recovery type
3. Whether stop losses are triggering appropriately

### Tuning Guidelines

**DCA-Only Stop Loss (Currently -$25):**

- **If stop triggers on >20% of DCA trades:**
  - Losses hitting stop too often
  - **Action:** Increase to -$30 or -$35
  - Gives DCA more room to work

- **If worst DCA loss is only -$10 to -$15:**
  - Stop is too loose, not protecting enough
  - **Action:** Decrease to -$20 or -$15
  - Tightens risk control

- **Target:** Stop should catch worst 5-10% of trades

**DCA+Hedge Stop Loss (Currently -$50):**

- **If stop triggers on >20% of hedged trades:**
  - **Action:** Increase to -$60 or -$75
  - Allows recovery more time

- **If worst loss is only -$25 to -$35:**
  - **Action:** Decrease to -$40 or -$45
  - Tightens protection

- **Critical:** Should NEVER allow -$600+ situations again

---

## Real-World Examples

### Example 1: Successful DCA Recovery
```
Entry:    0.04 lot SELL @ 1.1000
          Price moves up to 1.1020 (20 pips against)
          Floating: -$8

DCA L1:   0.06 lot SELL @ 1.1020
          Total: 0.10 lots, Avg: 1.1014
          Price moves up to 1.1030
          Floating: -$16

DCA L2:   0.09 lot SELL @ 1.1040
          Total: 0.19 lots, Avg: 1.1030
          Price reverses to 1.1020
          Floating: -$19

EXIT:     Price hits 1.0995 (below avg entry)
          Close all: +$6.65 profit
          Hold time: 4.5 hours
```
**Result:** DCA worked, recovered from -$19 to +$6.65

### Example 2: Stop Loss Prevents Disaster
```
Entry:    0.04 lot BUY @ 1.1000
          Price moves down to 1.0980 (20 pips against)
          Floating: -$8

DCA L1:   0.06 lot BUY @ 1.0980
          Price moves down to 1.0960 (40 pips against)
          Floating: -$18

DCA L2:   0.09 lot BUY @ 1.0960
          Price continues down to 1.0940 (60 pips against)
          Floating: -$30

DCA L3:   0.14 lot BUY @ 1.0940
          Price moves down to 1.0935
          Floating: -$26

EXIT:     Per-stack stop loss triggers at -$26
          Close entire stack
          Loss: -$26
```
**Result:** Stop prevented further losses. Without stop, could continue to -$50, -$100+

### Example 3: DCA+Hedge Mixed Outcome
```
Entry:    0.04 lot SELL @ 1.1000
          Price moves up to 1.1020 (20 pips)

DCA L1:   0.06 lot SELL @ 1.1020
          Price moves up to 1.1040 (40 pips)

DCA L2:   0.09 lot SELL @ 1.1040
          Price moves up to 1.1050 (50 pips)
          Floating: -$28

HEDGE:    0.06 lot BUY @ 1.1050 (1.5x initial)
          Price moves up to 1.1060 (60 pips)

DCA L3:   0.14 lot SELL @ 1.1060

Current State:
  SELL: 0.33 lots avg 1.1040 | Price: 1.1055 | -$49.50
  BUY:  0.06 lots @ 1.1050   | Price: 1.1055 | +$3.00
  NET: -$46.50

EXIT:     Per-stack stop loss triggers at -$46.50
          Close all positions
```
**Result:** Stop prevented catastrophic -$600+ situation

---

## Key Takeaways

1. **DCA is designed for mean reversion** - works when price reverses within 60 pips
2. **Hedge is last resort** - only triggers at 50 pips when DCA can't handle it
3. **Stop losses are safety nets** - prevent situations from spiraling out of control
4. **Expected success rates:**
   - Initial-only: 100% (historical)
   - DCA-only: 70-80% (allows losses but limits them)
   - DCA+Hedge: 50-70% (deep drawdown situations, harder to recover)

5. **The -$25/-$50 limits are TEST VALUES:**
   - Monitor actual outcomes
   - Adjust based on real performance
   - Balance between "room to recover" vs "cut losses quickly"

---

## Monitoring Checklist

**Weekly Review:**
- [ ] Run `analyze_recovery_outcomes.py`
- [ ] Check average profit for each recovery type
- [ ] Verify stop losses aren't triggering too often (<20%)
- [ ] Confirm worst losses are within acceptable range

**Monthly Review:**
- [ ] Calculate overall recovery success rate
- [ ] Tune stop loss limits if needed
- [ ] Compare DCA-only vs DCA+Hedge performance
- [ ] Validate if hedge trigger (50 pips) is appropriate

**Emergency Actions:**
- If stop losses trigger >30%: Increase limits OR reduce DCA levels
- If losses exceed stop by >$20: Bug in code, investigate immediately
- If hedge deployments are common: Consider increasing hedge trigger to 60-75 pips
