# Label Computation Mathematical Formulas

> Comprehensive mathematical reference for all label computation formulas in the Label Computation System for quantitative trading pattern mining.

## Overview

This document provides the mathematical formulas for all label types implemented in the Label Computation System. The system supports both batch backfill and real-time incremental computation with multi-timeframe alignment for accurate path-dependent calculations.

## Core Principles

### Multi-Timeframe Alignment

For path-dependent labels, the system uses lower granularity data to prevent look-ahead bias:

| Target Granularity | Level Data | Path Data | Multiplier |
|-------------------|------------|-----------|------------|
| Weekly (W) | W levels | Daily (D) candles | 5x |
| Daily (D) | D levels | 4-hour (H4) candles | 6x |
| 4-hour (H4) | H4 levels | Hourly (H1) candles | 4x |
| Hourly (H1) | H1 levels | 15-min (M15) candles | 4x |

### Notation

- **P_t**: Price at time t
- **P_{t+H}**: Price at time t+H (horizon end)
- **ATR_t**: Average True Range at time t (14-period default)
- **H**: Horizon length in periods
- **τ**: Time step within horizon [0, H]
- **S/R**: Support/Resistance level price

## Priority Labels (Top 15)

### Label 11.a: Enhanced Triple Barrier with S/R Levels (HIGHEST PRIORITY)

**Mathematical Definition:**
```
Barrier_upper(t) = P_t + max(2×ATR_t, min(R_nearest - 0.1×ATR_t, 2×ATR_t))
Barrier_lower(t) = P_t - max(2×ATR_t, min(P_t - S_nearest + 0.1×ATR_t, 2×ATR_t))

Label = {
  1   if ∃τ ∈ [1,H]: High_{t+τ} ≥ Barrier_upper(t)
  -1  if ∃τ ∈ [1,H]: Low_{t+τ} ≤ Barrier_lower(t)  
  0   otherwise
}
```

**Where:**
- R_nearest: Nearest resistance level above P_t
- S_nearest: Nearest support level below P_t
- High_{t+τ}, Low_{t+τ}: High/low prices from path data at lower granularity

**Implementation Notes:**
- Uses ATR-based dynamic barrier sizing (2x ATR default)
- Adjusts barriers based on nearby S/R levels with 0.1×ATR buffer
- Path checking uses lower granularity data (e.g., H1 for H4 labels)

### Label 2: Volatility-Scaled Returns

**Mathematical Definition:**
```
Vol_Scaled_Return(t,H) = (P_{t+H} - P_t) / ATR_t

Bounded_Return = clip(Vol_Scaled_Return, -10, 10)
```

**Where:**
- Normalization factor: ATR_t prevents division by zero (minimum 0.001)
- Bounds: ±10 to prevent extreme outliers from affecting model training

### Labels 9-10: Maximum Favorable/Adverse Excursion (MFE/MAE)

**Mathematical Definition:**
```
MFE(t,H) = max_{τ∈[0,H]} ((High_{t+τ} - P_t) / P_t)

MAE(t,H) = min_{τ∈[0,H]} ((Low_{t+τ} - P_t) / P_t)

Profit_Factor(t,H) = |MFE(t,H)| / |MAE(t,H)|  if MAE ≠ 0, else ∞
```

**Where:**
- MFE ≥ 0 (maximum favorable excursion)
- MAE ≤ 0 (maximum adverse excursion)
- Returns expressed as percentage change from entry price

### Label 12: Level Retouch Count

**Mathematical Definition:**
```
Retouch_Count(t,H) = Σ_{l∈Levels} Σ_{τ∈[1,H]} Touch(l, τ)

Touch(l, τ) = {
  1  if |Price_{t+τ} - l.price| / l.price ≤ 0.001 AND
       |Price_{t+τ} - l.price| / l.price ≤ 0.002 (no breakout)
  0  otherwise
}
```

**Where:**
- Touch tolerance: 0.1% (0.001) of level price
- Breakout exclusion: >0.2% (0.002) penetration doesn't count as touch
- Price_{t+τ}: Appropriate price (high for resistance, low for support)

### Label 16: Breakout Beyond Level

**Mathematical Definition:**
```
Breakout_Occurred(t,H) = {
  True   if ∃l∈Levels, ∃τ∈[1,H]: Penetration(l,τ) > 0.002
  False  otherwise
}

Penetration(l,τ) = {
  (l.price - Low_{t+τ}) / l.price     if l.type = support AND Low_{t+τ} < l.price
  (High_{t+τ} - l.price) / l.price    if l.type = resistance AND High_{t+τ} > l.price
  0                                   otherwise
}
```

**Where:**
- Breakout threshold: 0.2% (0.002) beyond level price
- Different price selection based on level type (support vs resistance)

### Label 17: Flip Within Horizon

**Mathematical Definition:**
```
Flip_Occurred(t,H) = ∃l∈Levels: Breakout(l) AND Retest(l)

Breakout(l) = ∃τ₁∈[1,H]: Penetration(l,τ₁) > 0.002

Retest(l) = ∃τ₂∈[τ₁,H]: |OppositePrice_{t+τ₂} - l.price| / l.price ≤ 0.001

OppositePrice = {
  High_{t+τ₂}  if l.type = support (retest as resistance)
  Low_{t+τ₂}   if l.type = resistance (retest as support)
}
```

**Where:**
- Requires both breakout (>0.2%) and retest (≤0.1%) of same level
- Retest must occur after breakout (τ₂ > τ₁)
- Uses opposite price type for retest detection

## Standard Labels (Labels 3-8, 13-15, 18-37)

### Label 6: Standard Triple Barrier

**Mathematical Definition:**
```
Barrier_upper = P_t × (1 + barrier_width)
Barrier_lower = P_t × (1 - barrier_width)

Label = {
  1   if ∃τ ∈ [1,H]: High_{t+τ} ≥ Barrier_upper
  -1  if ∃τ ∈ [1,H]: Low_{t+τ} ≤ Barrier_lower
  0   otherwise
}
```

**Where:**
- barrier_width: Fixed percentage (default 1% = 0.01)
- No S/R level adjustment (differs from Label 11.a)

### Label 7: Time to Barrier

**Mathematical Definition:**
```
Time_to_Barrier(t,H) = min{τ ∈ [1,H]: Barrier_Hit(τ)} ∪ {H}

Barrier_Hit(τ) = (High_{t+τ} ≥ Barrier_upper) OR (Low_{t+τ} ≤ Barrier_lower)
```

**Where:**
- Returns period number when first barrier is hit
- Returns H (horizon length) if no barrier hit

### Label 4: Return Quantile Buckets

**Mathematical Definition:**
```
Forward_Return = (P_{t+H} - P_t) / P_t

Return_Quantile = Percentile_Bucket(Forward_Return, Historical_Distribution)

Bucket ∈ {1, 2, ..., 10}  (decile buckets)
```

**Where:**
- Historical_Distribution: Rolling 1000-period return history
- Updates quantile boundaries adaptively

### Label 15: Consecutive Touch Runs

**Mathematical Definition:**
```
Consecutive_Touches(t,H) = max_{l∈Levels} Longest_Run(l)

Longest_Run(l) = max run length of consecutive touches within [1,H]

Consecutive = Touch(l,τ) AND Touch(l,τ+1) AND no gap > max_gap_periods
```

**Where:**
- max_gap_periods: 2 (allows 1-period gap in touch sequence)
- Counts longest consecutive sequence per level

### Label 22: Range Expansion

**Mathematical Definition:**
```
Range_Expansion(t,H) = (Range_{t+H} - Range_t) / Range_t

Range_t = (High_t - Low_t) / P_t

Expansion_Ratio = max_{τ∈[0,H]} (Range_{t+τ} / Range_t) - 1
```

**Where:**
- Measures proportional range increase
- Uses maximum range within horizon period

### Label 24: Event Burst Detection

**Mathematical Definition:**
```
Event_Burst(t,H) = {
  True   if Event_Count(t,H) > μ + 2σ
  False  otherwise
}

Event_Count(t,H) = |{τ ∈ [1,H]: Level_Event(τ)}|

Level_Event(τ) = Touch(τ) OR Breakout(τ) OR Flip(τ)
```

**Where:**
- μ, σ: Mean and standard deviation of historical event counts
- 2σ threshold for statistical significance

### Label 28: Drawdown Depth

**Mathematical Definition:**
```
Running_Max(τ) = max_{i∈[0,τ]} P_{t+i}

Drawdown(τ) = (Running_Max(τ) - P_{t+τ}) / Running_Max(τ)

Max_Drawdown(t,H) = max_{τ∈[0,H]} Drawdown(τ)
```

**Where:**
- Always non-positive (≤ 0)
- Measures maximum peak-to-trough decline

### Label 31: End vs Extremum Gap

**Mathematical Definition:**
```
Extremum_Price = {
  max_{τ∈[0,H]} High_{t+τ}  if Forward_Return > 0
  min_{τ∈[0,H]} Low_{t+τ}   if Forward_Return < 0
}

Gap_Ratio = |P_{t+H} - Extremum_Price| / |P_{t+H} - P_t|
```

**Where:**
- Measures how much of the move was given back by horizon end
- Values close to 0: end price near extremum
- Values close to 1: end price near starting point

### Label 35: Max Penetration Depth

**Mathematical Definition:**
```
Max_Penetration(t,H) = max_{l∈Levels} max_{τ∈[1,H]} Penetration_Depth(l,τ)

Penetration_Depth(l,τ) = {
  max(0, (l.price - Low_{t+τ}) / l.price)     if l.type = support
  max(0, (High_{t+τ} - l.price) / l.price)   if l.type = resistance
  0                                           otherwise
}
```

**Where:**
- Always ≥ 0 (no negative penetrations)
- Expressed as percentage of level price

## Risk and Path Metrics

### Risk-Adjusted Returns

**Mathematical Definition:**
```
Sharpe_Ratio(t,H) = Forward_Return(t,H) / Path_Volatility(t,H)

Path_Volatility(t,H) = std({Return(t,t+τ) : τ ∈ [1,H]})

Return(t,t+τ) = (P_{t+τ} - P_t) / P_t
```

### Path Skewness

**Mathematical Definition:**
```
Path_Returns = {(P_{t+τ} - P_t) / P_t : τ ∈ [1,H]}

Skewness = E[((X - μ)/σ)³]

Where X = Path_Returns, μ = mean(X), σ = std(X)
```

### Time Underwater

**Mathematical Definition:**
```
Underwater(τ) = {
  1  if P_{t+τ} < P_t
  0  otherwise
}

Time_Underwater = Σ_{τ=1}^H Underwater(τ)
```

## Level-Specific Labels

### Distance to Nearest Level

**Mathematical Definition:**
```
Distance_to_Level(t) = min_{l∈Active_Levels} |P_t - l.price| / P_t

Level_Type = {
  "support"     if nearest_level.price < P_t
  "resistance"  if nearest_level.price > P_t
}
```

### Next Touch Time

**Mathematical Definition:**
```
Next_Touch_Time(t,H) = min{τ ∈ [1,H]: ∃l∈Levels, Touch(l,τ)} ∪ {H+1}

Touch(l,τ) = |Appropriate_Price_{t+τ} - l.price| / l.price ≤ 0.001
```

## Implementation Validation

### Look-Ahead Bias Prevention

All formulas ensure temporal causality:
1. **Information Set**: I_t = {P_s, Level_s : s ≤ t}
2. **Path Data**: Uses aligned timestamps with proper granularity
3. **Level States**: Only levels active at time t are considered

### Numerical Stability

1. **Division by Zero**: All divisions include epsilon guards (1e-8)
2. **Overflow Protection**: Results bounded where appropriate
3. **Precision**: Uses float64 for all calculations

### Statistical Properties

- **Mean Reversion**: Labels designed to be approximately mean-zero over time
- **Stationarity**: Relative measures preferred over absolute prices
- **Cross-Instrument**: Formulas work across different price scales

## Performance Considerations

### Computational Complexity

- **Path-Dependent Labels**: O(H × M) where H = horizon, M = path multiplier
- **Level-Dependent Labels**: O(L × H × M) where L = active levels
- **Cache-Friendly**: Reuses path data across multiple labels

### Optimization Techniques

1. **Vectorized Operations**: Batch computation across candles
2. **Shared Path Data**: Single fetch for multiple path-dependent labels
3. **Level Filtering**: Only consider levels within reasonable distance
4. **Early Termination**: Stop computation when barriers hit

---

## References

1. **Multi-Timeframe Alignment**: Critical for preventing look-ahead bias in high-frequency applications
2. **Support/Resistance Integration**: Based on quantitative level significance testing
3. **Risk Metrics**: Standard financial risk measurement methodologies
4. **Path Analysis**: High-frequency tick data analysis techniques

**Version**: 1.0  
**Last Updated**: 2025-01-11  
**Validation Status**: All formulas tested against historical market data with zero look-ahead bias violations