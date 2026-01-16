# ML System Feature Specification

**Purpose:** Define all input features and target variable for the ML trading model
**Date:** 2026-01-09
**Status:** Day 3 - Approved for Implementation

---

## Overview

This specification defines **50 base features** (expanding to ~60 after encoding) across 7 categories to train a machine learning model that predicts trade outcomes. The feature set is designed to:

1. Capture all confluence factors with granular detail
2. Identify HTF level alignments for high-probability setups
3. Detect breakout conditions (vwap_band_3)
4. Understand market behavior and regime
5. Exploit temporal patterns
6. Analyze recovery mechanism effectiveness (DCA, hedge, grid, partials)

**Key Goal:** Enable feature importance analysis to inform strategy weighting decisions.

---

## Input Features (46 total)

### 1. VWAP Confluence Features (6)
These features capture price relationship to VWAP bands:

1. **vwap_distance_pct** (float)
   - Distance from VWAP in percentage [-5.0, 5.0]
   - Source: `vwap.distance_pct`
   - Example: 0.17 = 0.17% above VWAP

2. **vwap_band_1** (bool)
   - Price within ±1σ VWAP band
   - Source: `vwap.in_band_1`
   - Use: Mean reversion setups

3. **vwap_band_2** (bool)
   - Price within ±2σ VWAP band
   - Source: `vwap.in_band_2`
   - Use: Extended moves

4. **vwap_band_3** (bool)
   - Price beyond ±2σ (in 3rd band)
   - Source: `NOT in_band_2 AND NOT in_band_1`
   - Use: **Breakout/extended move detection**

5. **vwap_direction** (categorical → one-hot)
   - Price above/below/at VWAP
   - Source: `vwap.direction`
   - Encoded: vwap_above, vwap_below, vwap_at

6. **vwap_band_score** (int)
   - Combined band scores [0-3]
   - Source: `vwap.band_1_score + vwap.band_2_score`

---

### 2. Volume Profile Confluence Features (6)
These features identify key volume-based price levels:

7. **at_poc** (bool)
   - Price at Point of Control (highest volume node)
   - Source: `volume_profile.at_poc`
   - Use: High-probability reversal/support-resistance

8. **at_lvn** (bool)
   - Price at Low Volume Node
   - Source: `volume_profile.at_lvn`
   - Use: Breakout potential (price tends to move through LVNs)

9. **above_vah** (bool)
   - Price above Value Area High
   - Source: `volume_profile.above_vah`

10. **below_val** (bool)
    - Price below Value Area Low
    - Source: `volume_profile.below_val`

11. **at_swing_high** (bool)
    - Price at identified swing high
    - Source: `volume_profile.at_swing_high`

12. **at_swing_low** (bool)
    - Price at identified swing low
    - Source: `volume_profile.at_swing_low`

---

### 3. HTF (Higher Time Frame) Confluence Features (10)
These features detect alignment with higher timeframe levels:

13. **htf_total_score** (int)
    - Total HTF confluence score [0-20]
    - Source: `htf_levels.total_score`

14. **at_prev_day_vah** (bool)
    - At previous day's Value Area High
    - Source: `'Prev Day VAH' in htf_levels.factors_matched`

15. **at_prev_day_val** (bool)
    - At previous day's Value Area Low
    - Source: `'Prev Day VAL' in htf_levels.factors_matched`

16. **at_prev_day_poc** (bool)
    - At previous day's Point of Control
    - Source: `'Prev Day POC' in htf_levels.factors_matched`

17. **at_prev_day_high** (bool)
    - At previous day's high
    - Source: `'Prev Day High' in htf_levels.factors_matched`

18. **at_prev_day_low** (bool)
    - At previous day's low
    - Source: `'Prev Day Low' in htf_levels.factors_matched`

19. **at_weekly_hvn** (bool)
    - At weekly High Volume Node
    - Source: `'Weekly HVN' in htf_levels.factors_matched`

20. **at_daily_hvn** (bool)
    - At daily High Volume Node
    - Source: `'Daily HVN' in htf_levels.factors_matched`

21. **at_prev_week_high** (bool)
    - At previous week's high
    - Source: `'Prev Week High' in htf_levels.factors_matched`

22. **at_prev_week_low** (bool)
    - At previous week's low
    - Source: `'Prev Week Low' in htf_levels.factors_matched`

23. **weekly_hvn_count** (int)
    - Number of weekly HVNs nearby [0-10]
    - Source: `htf_levels.weekly_hvn_count`

---

### 4. Market Behavior & Trend Features (7)
These features characterize market regime and trend strength:

24. **adx** (float)
    - Average Directional Index [0-100]
    - Source: `trend_filter.adx`
    - Use: Trend strength (>25 = strong trend)

25. **plus_di** (float)
    - Positive Directional Indicator [0-100]
    - Source: `trend_filter.plus_di`

26. **minus_di** (float)
    - Negative Directional Indicator [0-100]
    - Source: `trend_filter.minus_di`

27. **trend_strength** (float)
    - Derived: `abs(plus_di - minus_di)`
    - Use: Directional conviction

28. **trend_direction** (categorical → one-hot)
    - Bullish if plus_di > minus_di, else Bearish
    - Encoded: trend_bullish, trend_bearish

29. **adx_regime** (categorical → one-hot)
    - Ranging (ADX < 20), Trending (20-25), Strong (>25)
    - Encoded: regime_ranging, regime_trending, regime_strong

30. **di_spread** (float)
    - `plus_di - minus_di` (can be negative)
    - Use: Directional bias strength

---

### 5. Temporal Features (8)
These features capture time-based patterns:

31. **hour** (int)
    - Hour of day [0-23]
    - Source: `market_context.hour`

32. **day_of_week** (int)
    - Day of week [0-6, Monday=0]
    - Source: `market_context.day_of_week` (parsed from entry_time)

33. **day_of_month** (int)
    - Day of month [1-31]
    - Source: Parsed from `entry_time`
    - Use: Monthly patterns (e.g., month-end flows)

34. **session** (categorical → one-hot)
    - Trading session: Tokyo, London, NY, Sydney
    - Source: `market_context.session`
    - Encoded: session_tokyo, session_london, session_ny, session_sydney

35. **is_session_open** (bool)
    - First hour of trading session
    - Derived: hour in [0-1, 7-8, 13-14]

36. **is_session_close** (bool)
    - Last hour of trading session
    - Derived: hour in [6, 12, 21]

37. **is_overlap** (bool)
    - During session overlap (high liquidity)
    - London/NY overlap: hour in [13-16]

38. **week_of_year** (int)
    - Week number [1-52]
    - Source: Parsed from `entry_time`
    - Use: Seasonal patterns

---

### 6. Recovery Mechanism Features (11)
These features capture trade management and recovery tactics:

39. **had_dca** (bool)
    - Whether DCA (Dollar Cost Averaging) was used
    - Source: `outcome.recovery.dca_count > 0`
    - Use: Identify if adding to positions helps/hurts

40. **dca_count** (int)
    - Number of DCA entries [0-10]
    - Source: `outcome.recovery.dca_count`
    - Use: Optimal DCA frequency

41. **had_hedge** (bool)
    - Whether hedging was used
    - Source: `outcome.recovery.hedge_count > 0`
    - Use: Does hedging improve outcomes?

42. **hedge_count** (int)
    - Number of hedge positions [0-5]
    - Source: `outcome.recovery.hedge_count`
    - Use: Optimal hedge frequency

43. **had_grid** (bool)
    - Whether grid trading was used
    - Source: `outcome.recovery.grid_count > 0`
    - Use: Grid effectiveness

44. **grid_count** (int)
    - Number of grid levels [0-10]
    - Source: `outcome.recovery.grid_count`
    - Use: Optimal grid size

45. **had_partial_close** (bool)
    - Whether partial closes were taken
    - Source: `outcome.partial_closes.count > 0`
    - Use: Do partials improve profitability?

46. **partial_close_count** (int)
    - Number of partial closes [0-5]
    - Source: `outcome.partial_closes.count`
    - Use: Optimal partial frequency

47. **total_recovery_volume** (float)
    - Total volume added via recovery mechanisms
    - Source: `outcome.recovery.total_recovery_volume`
    - Use: Volume exposure during recovery

48. **recovery_cost** (float)
    - Cost of recovery mechanisms (spreads, fees)
    - Source: `outcome.recovery.recovery_cost`
    - Use: Economic efficiency of recovery

49. **partial_profit_contribution** (float)
    - Profit from partial closes
    - Source: `outcome.partial_closes.total_profit_from_partials`
    - Use: Partial close effectiveness

**Note:** These features are only available for closed trades with outcomes. For open trades or predictions, these will be set to 0/False.

---

### 7. Overall Confluence Score (1)
This is the master confluence score:

50. **confluence_score** (int)
    - Total confluence score from all factors [4-30]
    - Source: `confluence_score`
    - Use: Overall setup quality

---

## Target Variable

**win** (binary classification)
- 1 if `outcome.profit > 0`
- 0 if `outcome.profit <= 0`
- Only defined for closed trades (`outcome.status == 'closed'`)

**Alternative (for future):**
- `profit` (regression) - Predict actual profit amount

---

## Feature Engineering Notes

### Derived Features
Some features are derived from raw data:
- `vwap_band_3` = NOT in_band_1 AND NOT in_band_2
- `trend_strength` = abs(plus_di - minus_di)
- `di_spread` = plus_di - minus_di
- `trend_direction` = 'bullish' if plus_di > minus_di else 'bearish'
- `adx_regime` based on ADX thresholds
- Temporal flags (is_session_open, is_session_close, is_overlap)

### One-Hot Encoding
Categorical features will be one-hot encoded:
- `vwap_direction` → 3 features
- `trend_direction` → 2 features
- `adx_regime` → 3 features
- `session` → 4 features

**Total after encoding: ~60 features** (50 base + one-hot encoded categoricals)

### Missing Value Handling
- HTF boolean features: Default to `False` if factor not in list
- Numeric features: Impute with median or 0 (domain-specific)
- Categorical: Add 'unknown' category if needed

---

## Feature Importance Analysis

**Critical for strategy development:**

After training, we will analyze:

1. **Individual Feature Importance**
   - Which single features are most predictive?
   - Example: Is `at_poc` more important than `at_prev_day_vah`?

2. **Feature Category Importance**
   - VWAP features vs HTF features vs Trend features
   - Which category dominates predictions?

3. **Confluence Factor Rankings**
   - Rank all 10 HTF factors by importance
   - Rank all volume profile factors
   - **Use this to weight confluence scoring in strategy**

4. **Interaction Analysis (SHAP values)**
   - Which feature combinations work best?
   - Example: `at_poc AND at_prev_day_vah` → high win rate?
   - Identify synergistic confluence patterns

5. **Worst Performing Features**
   - Which features hurt performance?
   - Remove or reduce weight in strategy

**Deliverable:** Feature importance report with actionable strategy adjustments

---

## Implementation Plan

### Day 4: Feature Extractor
- Implement `FeatureExtractor` class
- Extract all 39 base features
- Handle one-hot encoding
- Handle missing values
- Test on sample trades

### Day 5: Dataset Creation
- Apply feature extraction to all trades
- Create training dataset CSV
- Split train/validation/test
- Verify feature distributions

### Days 6-10: Model Training & Analysis
- Train models
- Generate feature importance plots
- Analyze confluence factor rankings
- **Provide strategy weighting recommendations**

---

## Success Criteria

✓ All 48 features (after encoding) extracted without errors
✓ No missing values in final dataset
✓ Features have expected ranges and distributions
✓ Feature importance analysis reveals actionable insights
✓ Clear recommendations for confluence weighting

---

**Next Step:** Implement feature extractor (Day 4)
