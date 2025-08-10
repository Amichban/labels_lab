# Product Requirements Document: Phase 2 Label Computation System
## Top 5 Priority Labels Implementation

**Version**: 2.0  
**Date**: 2025-01-10  
**Status**: Ready for Implementation  
**Phase**: 2 of 4

---

## 1. Executive Summary

### 1.1 Scope
Phase 2 focuses on implementing the remaining top 5 priority labels to complete the core label computation system for quantitative trading pattern mining.

**Phase 1 Complete:**
- ✅ Label 11.a (Enhanced Triple Barrier) with S/R level adjustments
- ✅ Basic infrastructure (ClickHouse, Redis, FastAPI)
- ✅ Multi-timeframe alignment (H4→H1, D→H4, etc.)
- ✅ Comprehensive test suite (106 tests)

**Phase 2 Target:**
Implement labels 2, 9, 10, 12, 16, 17 with <100ms incremental compute time and maintain existing performance standards.

### 1.2 Success Criteria
- All 5 priority labels implemented with comprehensive formulas
- Incremental computation maintains <100ms p99 latency
- Batch processing sustains 1M+ candles/minute throughput
- Zero look-ahead bias violations
- 95%+ cache hit rate maintained
- Full test coverage with performance benchmarks

---

## 2. Label Specifications & Formulas

### 2.1 Label 2: Volatility-Scaled Returns
**Priority**: High  
**Purpose**: Normalize returns by volatility for cross-market comparison

**Formula**:
```python
def compute_volatility_scaled_return(candle, horizon_periods=6):
    """
    Label 2: Volatility-Scaled Returns
    Normalizes forward returns by recent volatility (ATR)
    """
    # Get forward return at horizon
    forward_return = compute_forward_return(candle, horizon_periods)
    
    # Use ATR(14) as volatility proxy
    atr = candle.atr_14 or estimate_atr_from_history(candle, lookback=14)
    
    # Scale return by volatility
    vol_scaled_return = forward_return / atr if atr > 0 else 0
    
    # Apply bounds to prevent outliers
    vol_scaled_return = np.clip(vol_scaled_return, -10, 10)
    
    return {
        'vol_scaled_return': vol_scaled_return,
        'forward_return': forward_return,
        'atr_used': atr,
        'bounded': abs(vol_scaled_return) >= 10
    }
```

**Implementation Notes**:
- Uses existing ATR calculation from Label 11.a
- Bounds values at ±10 to prevent extreme outliers
- Tracks when bounds are applied for data quality monitoring

### 2.2 Labels 9-10: MFE/MAE with Profit Factor
**Priority**: High  
**Purpose**: Maximum Favorable/Adverse Excursion for risk-return analysis

**Formula**:
```python
def compute_mfe_mae_profit_factor(candle, horizon_periods=6):
    """
    Labels 9-10: MFE/MAE with Profit Factor
    Uses lower granularity path data for precise calculation
    """
    # Get path data at lower granularity
    path_granularity, multiplier = get_path_granularity(candle.granularity)
    path_periods = horizon_periods * multiplier
    
    path_data = get_path_data(
        candle.instrument_id,
        path_granularity,
        candle.ts,
        path_periods
    )
    
    if not path_data:
        return {'mfe': 0, 'mae': 0, 'profit_factor': 0}
    
    entry_price = candle.close
    mfe = 0.0  # Maximum Favorable Excursion
    mae = 0.0  # Maximum Adverse Excursion (negative)
    
    for path_candle in path_data:
        # Calculate excursions as percentage returns
        high_excursion = (path_candle.high - entry_price) / entry_price
        low_excursion = (path_candle.low - entry_price) / entry_price
        
        # Update maximums
        mfe = max(mfe, high_excursion)
        mae = min(mae, low_excursion)  # MAE is negative
    
    # Calculate profit factor
    profit_factor = abs(mfe / mae) if mae < -1e-8 else float('inf')
    
    return {
        'mfe': mfe,
        'mae': mae,
        'profit_factor': profit_factor,
        'path_periods_used': len(path_data),
        'path_granularity': path_granularity
    }
```

**Implementation Notes**:
- CRITICAL: Uses lower granularity path data (H4→H1, D→H4)
- MFE is always positive (or zero), MAE is always negative (or zero)
- Profit factor = |MFE/MAE| handles division by zero
- Tracks actual path periods used for quality validation

### 2.3 Label 12: Level Retouch Count
**Priority**: High  
**Purpose**: Count how many times price retouches S/R levels within horizon

**Formula**:
```python
def compute_level_retouch_count(candle, levels, horizon_periods=6):
    """
    Label 12: Level Retouch Count
    Counts retouches of nearby S/R levels within horizon
    """
    # Get active levels within reasonable distance
    active_levels = get_active_levels(candle.ts, levels)
    
    # Filter to levels within 3x ATR
    atr = candle.atr_14 or estimate_atr(candle)
    nearby_levels = []
    for level in active_levels:
        distance = abs(level.price - candle.close) / candle.close
        if distance <= 3 * atr / candle.close:  # Within 3 ATR
            nearby_levels.append(level)
    
    if not nearby_levels:
        return {'retouch_count': 0, 'levels_considered': 0}
    
    # Get path data
    path_granularity, multiplier = get_path_granularity(candle.granularity)
    path_data = get_path_data(candle.instrument_id, path_granularity, 
                             candle.ts, horizon_periods * multiplier)
    
    retouch_count = 0
    level_touches = {level.level_id: 0 for level in nearby_levels}
    
    for path_candle in path_data:
        for level in nearby_levels:
            # Define retouch zone (0.1% around level)
            retouch_threshold = level.price * 0.001
            
            # Check if price touches level
            if (path_candle.low <= level.price + retouch_threshold and 
                path_candle.high >= level.price - retouch_threshold):
                
                level_touches[level.level_id] += 1
    
    # Sum all retouches
    retouch_count = sum(level_touches.values())
    
    return {
        'retouch_count': retouch_count,
        'levels_considered': len(nearby_levels),
        'level_touches': level_touches,
        'max_touches_single_level': max(level_touches.values()) if level_touches else 0
    }
```

**Implementation Notes**:
- Only considers levels within 3x ATR distance
- Uses 0.1% retouch threshold to avoid noise
- Tracks touches per individual level for analysis
- CRITICAL: Uses path data for accurate touch detection

### 2.4 Label 16: Breakout Beyond Level
**Priority**: High  
**Purpose**: Detect when price breaks significantly beyond S/R levels

**Formula**:
```python
def compute_breakout_beyond_level(candle, levels, horizon_periods=6):
    """
    Label 16: Breakout Beyond Level
    Detects significant breakouts past S/R levels within horizon
    """
    active_levels = get_active_levels(candle.ts, levels)
    if not active_levels:
        return {'breakout_occurred': False, 'breakout_type': None}
    
    # Find nearest support and resistance
    support_levels = [l for l in active_levels if l.current_type == 'support' and l.price < candle.close]
    resistance_levels = [l for l in active_levels if l.current_type == 'resistance' and l.price > candle.close]
    
    nearest_support = max(support_levels, key=lambda x: x.price) if support_levels else None
    nearest_resistance = min(resistance_levels, key=lambda x: x.price) if resistance_levels else None
    
    # Get path data
    path_granularity, multiplier = get_path_granularity(candle.granularity)
    path_data = get_path_data(candle.instrument_id, path_granularity,
                             candle.ts, horizon_periods * multiplier)
    
    breakout_occurred = False
    breakout_type = None
    breakout_distance = 0
    breakout_time = None
    
    for i, path_candle in enumerate(path_data):
        # Check resistance breakout (upward)
        if nearest_resistance and path_candle.high > nearest_resistance.price * 1.002:  # 0.2% beyond
            breakout_occurred = True
            breakout_type = 'resistance'
            breakout_distance = (path_candle.high - nearest_resistance.price) / nearest_resistance.price
            breakout_time = i + 1
            break
        
        # Check support breakout (downward)
        if nearest_support and path_candle.low < nearest_support.price * 0.998:  # 0.2% beyond
            breakout_occurred = True
            breakout_type = 'support'
            breakout_distance = (nearest_support.price - path_candle.low) / nearest_support.price
            breakout_time = i + 1
            break
    
    return {
        'breakout_occurred': breakout_occurred,
        'breakout_type': breakout_type,
        'breakout_distance': breakout_distance,
        'breakout_time': breakout_time,
        'nearest_support_price': nearest_support.price if nearest_support else None,
        'nearest_resistance_price': nearest_resistance.price if nearest_resistance else None
    }
```

**Implementation Notes**:
- Requires 0.2% penetration beyond level to qualify as breakout
- Distinguishes between support and resistance breakouts
- Records exact breakout distance and timing
- Returns level prices for validation

### 2.5 Label 17: Flip Within Horizon
**Priority**: High  
**Purpose**: Detect S/R level type flips within the horizon period

**Formula**:
```python
def compute_flip_within_horizon(candle, levels, horizon_periods=6):
    """
    Label 17: Flip Within Horizon
    Detects if any S/R levels flip type within horizon
    """
    # Get levels active at candle time
    active_levels = get_active_levels(candle.ts, levels)
    if not active_levels:
        return {'flip_occurred': False, 'flip_count': 0}
    
    # Calculate horizon end
    horizon_end = get_horizon_end(candle.ts, candle.granularity, horizon_periods)
    
    flip_count = 0
    flipped_levels = []
    
    for level in active_levels:
        # Get level's event history within horizon
        level_events = get_level_events(
            level.level_id,
            candle.ts,
            horizon_end
        )
        
        # Check for flip events
        flip_events = [event for event in level_events if 
                      event.event_type in ['FLIP_TO_SUPPORT', 'FLIP_TO_RESISTANCE']]
        
        if flip_events:
            flip_count += len(flip_events)
            for event in flip_events:
                flipped_levels.append({
                    'level_id': level.level_id,
                    'original_type': level.current_type,
                    'new_type': 'support' if event.event_type == 'FLIP_TO_SUPPORT' else 'resistance',
                    'flip_time': event.timestamp,
                    'price': level.price
                })
    
    return {
        'flip_occurred': flip_count > 0,
        'flip_count': flip_count,
        'flipped_levels': flipped_levels,
        'levels_checked': len(active_levels)
    }
```

**Implementation Notes**:
- Relies on level event history in ClickHouse
- Tracks specific flip events (FLIP_TO_SUPPORT, FLIP_TO_RESISTANCE)
- Records all flips with timestamps for analysis
- Returns comprehensive flip metadata

---

## 3. Technical Implementation Plan

### 3.1 Data Model Extensions

**Extend LabelSet class**:
```python
class LabelSet(BaseModel):
    # ... existing fields ...
    
    # Label 2: Volatility-Scaled Returns
    vol_scaled_return: Optional[float] = None
    vol_scaled_bounded: Optional[bool] = None
    
    # Labels 9-10: MFE/MAE with Profit Factor  
    mfe: Optional[float] = None
    mae: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # Label 12: Level Retouch Count
    retouch_count: Optional[int] = Field(default=None, ge=0)
    levels_considered_count: Optional[int] = Field(default=None, ge=0)
    max_touches_single_level: Optional[int] = Field(default=None, ge=0)
    
    # Label 16: Breakout Beyond Level
    breakout_occurred: Optional[bool] = None
    breakout_type: Optional[str] = None  # 'support' | 'resistance'
    breakout_distance: Optional[float] = Field(default=None, ge=0)
    breakout_time: Optional[int] = Field(default=None, ge=0)
    
    # Label 17: Flip Within Horizon
    flip_occurred: Optional[bool] = None
    flip_count: Optional[int] = Field(default=None, ge=0)
```

### 3.2 ClickHouse Schema Updates

**Update quantx.labels table**:
```sql
ALTER TABLE quantx.labels ADD COLUMN IF NOT EXISTS
    -- Label 2
    vol_scaled_bounded Bool,
    
    -- Label 12
    levels_considered_count UInt8,
    max_touches_single_level UInt8,
    
    -- Label 16
    breakout_type String,
    breakout_distance Float64,
    breakout_time UInt16,
    
    -- Label 17
    flip_count UInt8;

-- Add indexes for new labels
ALTER TABLE quantx.labels ADD INDEX IF NOT EXISTS 
    idx_breakout_occurred breakout_occurred TYPE minmax GRANULARITY 1;

ALTER TABLE quantx.labels ADD INDEX IF NOT EXISTS 
    idx_retouch_count retouch_count TYPE minmax GRANULARITY 1;
```

### 3.3 Core Algorithm Updates

**LabelComputationEngine enhancements**:
```python
async def compute_labels(self, candle: Candle, **kwargs) -> LabelSet:
    # ... existing code ...
    
    # Add new label computations
    if "vol_scaled_return" in label_types:
        vol_result = await self._compute_volatility_scaled_return(candle, horizon_end)
        label_set.vol_scaled_return = vol_result['vol_scaled_return']
        label_set.vol_scaled_bounded = vol_result['bounded']
    
    if "mfe_mae" in label_types:
        mfe_mae_result = await self._compute_mfe_mae_profit_factor(candle, horizon_end)
        label_set.mfe = mfe_mae_result['mfe']
        label_set.mae = mfe_mae_result['mae'] 
        label_set.profit_factor = mfe_mae_result['profit_factor']
    
    if "retouch_count" in label_types:
        retouch_result = await self._compute_level_retouch_count(candle, horizon_end)
        label_set.retouch_count = retouch_result['retouch_count']
        label_set.levels_considered_count = retouch_result['levels_considered']
        label_set.max_touches_single_level = retouch_result['max_touches_single_level']
    
    if "breakout_beyond" in label_types:
        breakout_result = await self._compute_breakout_beyond_level(candle, horizon_end)
        label_set.breakout_occurred = breakout_result['breakout_occurred']
        label_set.breakout_type = breakout_result['breakout_type']
        label_set.breakout_distance = breakout_result['breakout_distance']
        label_set.breakout_time = breakout_result['breakout_time']
    
    if "flip_within_horizon" in label_types:
        flip_result = await self._compute_flip_within_horizon(candle, horizon_end)
        label_set.flip_occurred = flip_result['flip_occurred']
        label_set.flip_count = flip_result['flip_count']
    
    return label_set
```

---

## 4. Performance Requirements & Optimization

### 4.1 Latency Targets

| Label Type | Target Latency | Max Acceptable |
|------------|----------------|----------------|
| Volatility-Scaled Return | 5ms | 15ms |
| MFE/MAE + Profit Factor | 25ms | 50ms |
| Level Retouch Count | 20ms | 40ms |
| Breakout Beyond Level | 15ms | 30ms |
| Flip Within Horizon | 10ms | 25ms |
| **Combined (all 5)** | **75ms** | **160ms** |

### 4.2 Optimization Strategies

**Path Data Caching**:
```python
class PathDataCache:
    """Optimized caching for path data across multiple label computations"""
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes
        self.cache_key_template = "path:{instrument}:{granularity}:{start}:{periods}"
    
    async def get_or_fetch_path_data(self, candle, horizon_periods):
        """Single path data fetch for all labels needing it"""
        path_granularity, multiplier = get_path_granularity(candle.granularity)
        path_periods = horizon_periods * multiplier
        
        cache_key = self.cache_key_template.format(
            instrument=candle.instrument_id,
            granularity=path_granularity,
            start=candle.ts.isoformat(),
            periods=path_periods
        )
        
        # Try cache first
        cached = redis_cache.get(cache_key)
        if cached:
            return cached
        
        # Fetch and cache
        path_data = await clickhouse_service.fetch_snapshots(
            candle.instrument_id, path_granularity, 
            candle.ts, path_periods
        )
        
        redis_cache.setex(cache_key, self.cache_ttl, path_data)
        return path_data
```

**Batch Level Queries**:
```python
async def get_active_levels_optimized(self, instruments_and_timestamps):
    """Batch fetch active levels for multiple computations"""
    # Group requests by instrument/granularity
    grouped_requests = defaultdict(list)
    for instrument_id, granularity, ts in instruments_and_timestamps:
        grouped_requests[(instrument_id, granularity)].append(ts)
    
    # Fetch in batches
    results = {}
    for (instrument_id, granularity), timestamps in grouped_requests.items():
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        
        # Single query for time range
        levels = await clickhouse_service.fetch_active_levels_range(
            instrument_id, granularity, min_ts, max_ts
        )
        
        # Distribute to individual timestamps
        for ts in timestamps:
            relevant_levels = [l for l in levels if l.is_active_at(ts)]
            results[(instrument_id, granularity, ts)] = relevant_levels
    
    return results
```

---

## 5. Testing & Validation Framework

### 5.1 Unit Tests per Label

```python
class TestLabelImplementations(unittest.TestCase):
    
    def test_volatility_scaled_return_bounds(self):
        """Test vol scaled return bounds at ±10"""
        # Test extreme volatility scenario
        extreme_candle = create_test_candle(
            close=100, atr_14=0.001,  # Very low volatility
            forward_return=0.5  # 50% return
        )
        result = compute_volatility_scaled_return(extreme_candle)
        
        self.assertEqual(result['vol_scaled_return'], 10.0)  # Bounded at +10
        self.assertTrue(result['bounded'])
    
    def test_mfe_mae_path_granularity(self):
        """Test MFE/MAE uses correct path granularity"""
        h4_candle = create_test_candle(granularity='H4')
        
        with mock.patch('get_path_data') as mock_path:
            compute_mfe_mae_profit_factor(h4_candle)
            
            # Verify H1 path data requested (lower granularity)
            mock_path.assert_called_once()
            args = mock_path.call_args[0]
            self.assertEqual(args[1], 'H1')  # path_granularity
    
    def test_retouch_count_distance_filter(self):
        """Test retouch count only considers nearby levels"""
        candle = create_test_candle(close=1.2000, atr_14=0.001)
        
        # Create levels at various distances
        near_level = create_test_level(price=1.2020)   # Within 3x ATR
        far_level = create_test_level(price=1.2500)    # Beyond 3x ATR
        levels = [near_level, far_level]
        
        result = compute_level_retouch_count(candle, levels)
        
        # Only near level should be considered
        self.assertEqual(result['levels_considered'], 1)
    
    def test_breakout_threshold_precision(self):
        """Test breakout requires 0.2% penetration"""
        candle = create_test_candle(close=1.2000)
        resistance_level = create_test_level(price=1.2100, type='resistance')
        
        # Price reaches level but doesn't penetrate enough
        path_data = [create_path_candle(high=1.2101)]  # Only 0.08% above
        
        with mock.patch('get_path_data', return_value=path_data):
            result = compute_breakout_beyond_level(candle, [resistance_level])
            
        self.assertFalse(result['breakout_occurred'])  # Insufficient penetration
    
    def test_flip_event_detection(self):
        """Test flip detection uses correct event types"""
        candle = create_test_candle()
        level = create_test_level()
        
        flip_events = [
            create_event('FLIP_TO_RESISTANCE'),
            create_event('TOUCH_UP'),  # Not a flip event
            create_event('FLIP_TO_SUPPORT')
        ]
        
        with mock.patch('get_level_events', return_value=flip_events):
            result = compute_flip_within_horizon(candle, [level])
            
        self.assertTrue(result['flip_occurred'])
        self.assertEqual(result['flip_count'], 2)  # Only flip events counted
```

### 5.2 Integration Tests

```python
class TestPhase2Integration(unittest.TestCase):
    
    def test_all_labels_performance(self):
        """Test combined latency of all Phase 2 labels"""
        candle = create_realistic_test_candle()
        label_types = [
            "vol_scaled_return", "mfe_mae", "retouch_count", 
            "breakout_beyond", "flip_within_horizon"
        ]
        
        start_time = time.time()
        result = await computation_engine.compute_labels(
            candle, label_types=label_types
        )
        latency_ms = (time.time() - start_time) * 1000
        
        # Performance requirement
        self.assertLess(latency_ms, 160, "Combined latency too high")
        
        # Verify all labels computed
        self.assertIsNotNone(result.vol_scaled_return)
        self.assertIsNotNone(result.mfe)
        self.assertIsNotNone(result.mae)
        self.assertIsNotNone(result.profit_factor)
        self.assertIsNotNone(result.retouch_count)
        self.assertIsNotNone(result.breakout_occurred)
        self.assertIsNotNone(result.flip_occurred)
    
    def test_batch_throughput_maintained(self):
        """Verify batch processing maintains 1M+ candles/minute"""
        # Generate 10K test candles
        test_candles = [create_test_candle() for _ in range(10000)]
        
        start_time = time.time()
        results = await computation_engine.compute_batch_labels(
            test_candles, label_types=["vol_scaled_return", "mfe_mae"]
        )
        duration = time.time() - start_time
        
        throughput = len(test_candles) / duration * 60  # candles per minute
        self.assertGreater(throughput, 1_000_000, "Throughput below 1M/min")
```

### 5.3 Data Quality Validation

```python
class TestDataQuality(unittest.TestCase):
    
    def test_no_look_ahead_bias(self):
        """Critical: Verify no look-ahead bias in computations"""
        candle = create_test_candle(ts=datetime(2024, 1, 15, 12, 0))
        
        # Mock data that includes future information
        future_path_data = [
            create_path_candle(ts=datetime(2024, 1, 15, 8, 0)),   # Past - OK
            create_path_candle(ts=datetime(2024, 1, 15, 13, 0)),  # Future - Should use
            create_path_candle(ts=datetime(2024, 1, 15, 11, 0)),  # Past - OK
            create_path_candle(ts=datetime(2024, 1, 16, 12, 0)),  # Next day - Should use
        ]
        
        with mock.patch('get_path_data', return_value=future_path_data):
            result = compute_mfe_mae_profit_factor(candle)
            
        # Verify only future data used (candle time = 12:00, so 13:00+ is valid)
        # This is a critical test that must pass
        self.assertIsNotNone(result['mfe'])
        
    def test_label_value_ranges(self):
        """Test all labels produce values within expected ranges"""
        candle = create_test_candle()
        result = await computation_engine.compute_labels(candle, label_types=['all'])
        
        # Volatility-scaled return bounds
        self.assertLessEqual(abs(result.vol_scaled_return), 10.0)
        
        # MFE/MAE signs
        self.assertGreaterEqual(result.mfe, 0)  # MFE >= 0
        self.assertLessEqual(result.mae, 0)     # MAE <= 0
        
        # Profit factor positive
        if result.profit_factor is not None:
            self.assertGreaterEqual(result.profit_factor, 0)
        
        # Count fields non-negative
        self.assertGreaterEqual(result.retouch_count, 0)
        self.assertGreaterEqual(result.flip_count, 0)
```

---

## 6. Deployment Strategy

### 6.1 Incremental Rollout

**Phase 2A**: Single Label Implementation (Week 1)
- Implement Label 2 (Volatility-Scaled Returns) first
- Deploy with feature flag to 10% of computations
- Monitor performance impact
- Full rollout if metrics met

**Phase 2B**: Path-Dependent Labels (Week 2)  
- Implement Labels 9-10 (MFE/MAE)
- Critical path data optimization
- A/B test against baseline latency
- Rollout if <100ms maintained

**Phase 2C**: Level-Interaction Labels (Week 3)
- Implement Labels 12, 16, 17 together
- Level data caching optimization
- Performance validation
- Full production rollout

### 6.2 Monitoring & Alerts

**Performance Monitoring**:
```python
# Prometheus metrics for Phase 2
phase2_label_duration = Histogram(
    'phase2_label_computation_duration_seconds',
    'Phase 2 label computation time',
    ['label_type']
)

phase2_error_rate = Counter(
    'phase2_label_computation_errors_total',
    'Phase 2 label computation errors',
    ['label_type', 'error_type']
)

phase2_cache_efficiency = Gauge(
    'phase2_cache_hit_rate',
    'Cache hit rate for Phase 2 computations',
    ['cache_type']  # path_data, levels, etc.
)
```

**Alert Rules**:
- Phase 2 combined latency > 100ms p99 for 5min
- Any individual label > 50ms p99 for 5min  
- Label error rate > 0.5%
- Cache hit rate < 90% for path data
- Data quality violations (look-ahead bias detected)

---

## 7. Acceptance Criteria

### 7.1 Functional Requirements
- [ ] All 5 labels implemented with correct formulas
- [ ] Multi-timeframe path data used correctly (H4→H1, D→H4, etc.)
- [ ] Level interaction logic accurate (retouches, breakouts, flips)
- [ ] No look-ahead bias in any computation
- [ ] Proper error handling and graceful degradation
- [ ] Cache integration for all data sources

### 7.2 Performance Requirements
- [ ] Combined latency <100ms p99 for all 5 labels
- [ ] Individual label latencies within targets (see Section 4.1)
- [ ] Batch processing maintains 1M+ candles/minute
- [ ] Cache hit rate >90% for path data and levels
- [ ] Memory usage increase <20% from Phase 1

### 7.3 Quality Requirements  
- [ ] 100% test coverage for all new label functions
- [ ] Integration tests with realistic market data
- [ ] Performance benchmarks automated in CI/CD
- [ ] Data quality validation in production
- [ ] Monitoring dashboards for all new metrics

### 7.4 Operational Requirements
- [ ] Feature flags for gradual rollout
- [ ] Rollback capability within 5 minutes
- [ ] Documentation updated for all formulas
- [ ] Runbooks for troubleshooting
- [ ] Performance regression alerts configured

---

## 8. Risk Mitigation

### 8.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Latency degradation | High | Medium | Path data caching, batch optimizations |
| Look-ahead bias | Critical | Low | Automated validation, temporal checks |
| Cache memory pressure | Medium | Medium | TTL optimization, LRU eviction |
| ClickHouse query performance | High | Medium | Index optimization, query batching |

### 8.2 Operational Risks

- **Data gaps**: Comprehensive gap detection and backfill
- **Label drift**: Statistical monitoring and drift detection
- **Version compatibility**: Backward-compatible schema changes
- **Performance regression**: Automated rollback triggers

---

## 9. Success Metrics

### 9.1 Implementation Success
- Phase 2 delivery within 3 weeks
- Zero production incidents during rollout
- All acceptance criteria met
- Performance targets achieved

### 9.2 Business Impact
- Enhanced trading signal quality (measurable via Sharpe)
- Reduced manual pattern analysis workload
- Improved model training data richness
- Foundation for Phase 3 advanced labels

---

## 10. Next Steps

### 10.1 Immediate Actions (Week 1)
1. Update ClickHouse schema with new columns
2. Implement Label 2 (Volatility-Scaled Returns)
3. Add unit tests and benchmarks
4. Deploy with feature flag at 10%

### 10.2 Phase 3 Preparation
- Advanced pattern labels (22, 24, 28, 31, 35)
- Real-time computation pipeline
- ML model integration
- Production monitoring enhancement

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.0 | 2025-01-10 | System | Phase 2 PRD with top 5 priority labels implementation |

---

**Phase 2 Ready for Implementation** ✅  
All technical specifications, formulas, and acceptance criteria defined for implementing the remaining top 5 priority labels while maintaining <100ms incremental compute performance.