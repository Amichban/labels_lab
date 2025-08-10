# Product Requirements Document: Label Computation System
## For Support/Resistance Pattern Mining

Version: 1.0  
Date: 2025-01-10  
Status: Draft

---

## 1. Executive Summary

### 1.1 Purpose
Build a high-performance label computation system for quantitative trading that processes support/resistance level events and computes forward-looking labels for pattern mining and ML models.

### 1.2 Key Requirements
- **Dual-mode processing**: Batch backfill (minutes) and real-time incremental (<1 second)
- **Multi-timeframe alignment**: Use lower granularity data for path-dependent calculations
- **Storage**: ClickHouse for persistence, Redis for hot cache
- **UI**: React dashboard for data visualization and quality monitoring
- **Scale**: Support billions of labels across 29 FX pairs and indices

### 1.3 Critical Success Metrics
- Incremental compute: <100ms p99 latency
- Batch throughput: 1M+ candles/minute
- Cache hit rate: >95% for recent 24 hours
- Zero look-ahead bias violations
- Expected Sharpe improvement: 0.3-0.5

---

## 2. System Architecture

### 2.1 Data Flow Architecture

```
┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  ClickHouse  │────>│   Compute   │────>│  ClickHouse  │
│  (Raw Data)  │     │   Engine    │     │   (Labels)   │
└──────────────┘     └──────┬──────┘     └──────────────┘
                            │
                     ┌──────▼──────┐
                     │    Redis     │
                     │   (Cache)    │
                     └──────┬──────┘
                            │
                     ┌──────▼──────┐     ┌──────────────┐
                     │   FastAPI   │────>│   React UI   │
                     │     API      │     │  Dashboard   │
                     └─────────────┘     └──────────────┘
```

### 2.2 Multi-Timeframe Data Alignment (CRITICAL)

**REQUIREMENT**: For path-dependent labels, ALWAYS use granularity one level below for intra-horizon checks:

| Target Granularity | Level Data | Path Data | Alignment Rule |
|-------------------|------------|-----------|----------------|
| Weekly (W) | W levels | Daily (D) candles | Align to week start |
| Daily (D) | D levels | 4-hour (H4) candles | Align to day start |
| 4-hour (H4) | H4 levels | Hourly (H1) candles | Align to H4 boundary (1,5,9,13,17,21 UTC) |
| Hourly (H1) | H1 levels | 15-min (M15) candles | Align to hour start |

**Implementation**:
```python
def get_path_data(instrument_id, granularity, ts_start, horizon_periods):
    """
    CRITICAL: Use lower granularity for path-dependent calculations
    """
    # Map to lower granularity
    path_granularity = {
        'W': 'D',
        'D': 'H4', 
        'H4': 'H1',
        'H1': 'M15'
    }[granularity]
    
    # Convert horizon to lower granularity periods
    multiplier = {'W': 5, 'D': 6, 'H4': 4, 'H1': 4}[granularity]
    path_periods = horizon_periods * multiplier
    
    # CRITICAL: Align timestamps
    aligned_start = align_timestamp(ts_start, granularity)
    
    return fetch_candles(instrument_id, path_granularity, 
                        aligned_start, path_periods)
```

---

## 3. Label Specifications

### 3.1 Priority Labels (Top 15)

#### Label 11.a: Enhanced Triple Barrier with S/R Levels (HIGHEST PRIORITY)
**Definition**: Triple barrier that adjusts barriers based on nearby support/resistance levels

**Computation**:
```python
def compute_enhanced_triple_barrier(candle, levels, horizon_h4_periods=6):
    """
    Label 11.a: Enhanced triple barrier using S/R levels
    Uses H1 data for path checking when computing H4 labels
    """
    # Get active S/R levels at candle time
    active_levels = get_active_levels(candle.ts, levels)
    
    # Base barriers from ATR
    base_upper = candle.close + 2 * candle.atr_14
    base_lower = candle.close - 2 * candle.atr_14
    
    # Adjust for nearby resistance (upper barrier)
    nearest_resistance = find_nearest_level(active_levels, 
                                           candle.close, 'resistance')
    if nearest_resistance and abs(nearest_resistance - candle.close) < 3 * candle.atr_14:
        upper_barrier = min(base_upper, nearest_resistance - 0.1 * candle.atr_14)
    else:
        upper_barrier = base_upper
    
    # Adjust for nearby support (lower barrier)
    nearest_support = find_nearest_level(active_levels, 
                                        candle.close, 'support')
    if nearest_support and abs(nearest_support - candle.close) < 3 * candle.atr_14:
        lower_barrier = max(base_lower, nearest_support + 0.1 * candle.atr_14)
    else:
        lower_barrier = base_lower
    
    # CRITICAL: Use H1 data for H4 path checking
    if candle.granularity == 'H4':
        path_data = get_h1_candles(candle.instrument_id, 
                                   candle.ts, 
                                   horizon_h4_periods * 4)  # 4 H1 per H4
    else:
        path_data = get_path_data_for_granularity(candle)
    
    # Check which barrier is hit first
    for i, path_candle in enumerate(path_data):
        if path_candle.high >= upper_barrier:
            return {
                'label': 1,
                'barrier_hit': 'upper',
                'time_to_barrier': i + 1,
                'barrier_price': upper_barrier,
                'level_adjusted': nearest_resistance is not None
            }
        if path_candle.low <= lower_barrier:
            return {
                'label': -1,
                'barrier_hit': 'lower',
                'time_to_barrier': i + 1,
                'barrier_price': lower_barrier,
                'level_adjusted': nearest_support is not None
            }
    
    return {
        'label': 0,
        'barrier_hit': 'none',
        'time_to_barrier': len(path_data),
        'barrier_price': None,
        'level_adjusted': False
    }
```

#### Other Priority Labels

2. **Volatility-Scaled Returns** (Label 2)
3. **MFE/MAE with Profit Factor** (Labels 9-10)
4. **Level Retouch Count** (Label 12)
5. **Breakout Beyond Level** (Label 16)
6. **Flip Within Horizon** (Label 17)
7. **Time to Barrier** (Label 7)
8. **Triple Barrier Standard** (Label 6)
9. **Return Quantile Buckets** (Label 4)
10. **Consecutive Touch Runs** (Label 15)
11. **Max Penetration Depth** (Label 35)
12. **Range Expansion** (Label 22)
13. **Event Burst Detection** (Label 24)
14. **Drawdown Depth** (Label 28)
15. **End vs Extremum Gap** (Label 31)

---

## 4. Data Pipeline Design

### 4.1 Batch Backfill Pipeline

```python
class BatchLabelComputer:
    def __init__(self, clickhouse_client, redis_client):
        self.ch = clickhouse_client
        self.redis = redis_client
        self.chunk_size = 10000  # Process 10k candles at a time
        
    def backfill_labels(self, instrument_id, granularity, start_date, end_date):
        """
        Batch compute labels with proper multi-timeframe alignment
        """
        # Fetch data in chunks
        for chunk_start in date_range(start_date, end_date, self.chunk_size):
            # Get candles at target granularity
            candles = self.ch.query(f"""
                SELECT * FROM quantx.snapshots
                WHERE instrument_id = '{instrument_id}'
                  AND granularity = '{granularity}'
                  AND ts >= '{chunk_start}'
                  AND ts < '{chunk_start + chunk_size}'
                ORDER BY ts
            """)
            
            # Get path data at lower granularity for path-dependent labels
            path_granularity = self.get_path_granularity(granularity)
            path_data = self.ch.query(f"""
                SELECT * FROM quantx.snapshots
                WHERE instrument_id = '{instrument_id}'
                  AND granularity = '{path_granularity}'
                  AND ts >= '{chunk_start}'
                  AND ts < '{chunk_end + horizon_buffer}'
                ORDER BY ts
            """)
            
            # Get levels
            levels = self.ch.query(f"""
                SELECT * FROM quantx.levels
                WHERE instrument_id = '{instrument_id}'
                  AND granularity = '{granularity}'
                  AND created_at <= '{chunk_end}'
                  AND (deactivated_at IS NULL OR deactivated_at > '{chunk_start}')
            """)
            
            # Compute labels in parallel
            with ProcessPoolExecutor(max_workers=8) as executor:
                label_futures = []
                for candle in candles:
                    # Extract relevant path window
                    candle_path = self.extract_path_window(
                        path_data, candle.ts, granularity
                    )
                    relevant_levels = self.filter_active_levels(
                        levels, candle.ts
                    )
                    
                    future = executor.submit(
                        compute_all_labels,
                        candle, candle_path, relevant_levels
                    )
                    label_futures.append((candle.ts, future))
                
                # Collect results
                labels_batch = []
                for ts, future in label_futures:
                    labels = future.result()
                    labels['ts'] = ts
                    labels['instrument_id'] = instrument_id
                    labels['granularity'] = granularity
                    labels_batch.append(labels)
            
            # Bulk insert to ClickHouse
            self.ch.insert('quantx.labels', labels_batch)
            
            # Update progress
            self.redis.set(
                f'backfill_progress:{instrument_id}:{granularity}',
                chunk_end.isoformat()
            )
```

### 4.2 Incremental Real-time Pipeline

```python
class IncrementalLabelComputer:
    def __init__(self, clickhouse_client, redis_client):
        self.ch = clickhouse_client
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour cache
        
    async def process_new_candle(self, candle):
        """
        Process single candle in real-time with <100ms target
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = f"labels:{candle.instrument_id}:{candle.granularity}:{candle.ts}"
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Get required lookback data from cache or DB
        lookback_key = f"lookback:{candle.instrument_id}:{candle.granularity}"
        lookback_data = self.redis.get(lookback_key)
        if not lookback_data:
            lookback_data = await self.fetch_lookback_data(candle)
            self.redis.setex(lookback_key, 300, lookback_data)  # 5 min cache
        
        # Get active levels from cache
        levels_key = f"levels:{candle.instrument_id}:{candle.granularity}:active"
        active_levels = self.redis.smembers(levels_key)
        if not active_levels:
            active_levels = await self.fetch_active_levels(candle)
            self.redis.sadd(levels_key, *active_levels)
            self.redis.expire(levels_key, 300)
        
        # CRITICAL: For path-dependent labels, fetch lower granularity forward data
        if candle.granularity == 'H4':
            # Need H1 data for next 24 hours (6 H4 periods * 4 H1 per H4)
            forward_path = await self.fetch_h1_forward_data(
                candle.instrument_id, 
                candle.ts, 
                periods=24
            )
        else:
            forward_path = await self.fetch_forward_path(candle)
        
        # Compute labels
        labels = {}
        
        # Label 11.a: Enhanced Triple Barrier
        labels['enhanced_triple_barrier'] = compute_enhanced_triple_barrier(
            candle, active_levels, forward_path
        )
        
        # Other labels computed in parallel
        labels.update(await self.compute_other_labels(
            candle, lookback_data, active_levels, forward_path
        ))
        
        # Store in cache and DB
        self.redis.setex(cache_key, self.cache_ttl, json.dumps(labels))
        
        # Async write to ClickHouse
        asyncio.create_task(self.write_to_clickhouse(candle, labels))
        
        # Monitor latency
        latency = (time.time() - start_time) * 1000
        if latency > 100:
            logger.warning(f"Slow label computation: {latency:.1f}ms for {candle.ts}")
        
        return labels
```

---

## 5. Storage Schema

### 5.1 Primary Labels Table

```sql
CREATE TABLE quantx.labels (
    instrument_id String,
    granularity String,
    ts DateTime64(3),
    
    -- Label 11.a: Enhanced Triple Barrier
    enhanced_triple_barrier_label Int8,  -- -1, 0, 1
    enhanced_triple_barrier_time UInt16,  -- periods to barrier
    enhanced_triple_barrier_price Float64,
    enhanced_triple_barrier_adjusted Bool,  -- was S/R adjusted
    
    -- Core labels
    forward_return Float64,
    vol_scaled_return Float64,
    return_sign Int8,
    return_quantile UInt8,
    
    -- Path metrics
    mfe Float64,
    mae Float64,
    profit_factor Float64,
    max_penetration Float64,
    
    -- Level-specific
    retouch_count UInt8,
    next_touch_time UInt16,
    breakout_occurred Bool,
    flip_occurred Bool,
    
    -- Risk metrics
    drawdown_depth Float64,
    time_underwater UInt16,
    path_skewness Float64,
    
    -- Metadata
    label_version String,
    computed_at DateTime64(3) DEFAULT now64(3),
    
    INDEX idx_enhanced_barrier enhanced_triple_barrier_label TYPE minmax GRANULARITY 1,
    INDEX idx_return_quantile return_quantile TYPE minmax GRANULARITY 1
) ENGINE = MergeTree()
ORDER BY (instrument_id, granularity, ts)
PARTITION BY toYYYYMM(ts)
SETTINGS index_granularity = 8192;
```

### 5.2 Level-Specific Labels Table

```sql
CREATE TABLE quantx.level_labels (
    level_id String,
    instrument_id String,
    granularity String,
    ts DateTime64(3),
    
    -- Level interaction labels
    distance_to_level Float64,
    penetration_depth Float64,
    time_at_level UInt16,
    consecutive_touches UInt8,
    
    -- Forward looking
    next_event_type String,
    time_to_next_event UInt16,
    level_holds Bool,
    
    computed_at DateTime64(3) DEFAULT now64(3)
) ENGINE = MergeTree()
ORDER BY (level_id, ts)
PARTITION BY toYYYYMM(ts);
```

---

## 6. React UI Dashboard

### 6.1 Core Components

```typescript
// Main Dashboard Layout
interface DashboardProps {
  instruments: string[];
  granularities: string[];
  dateRange: DateRange;
}

const LabelDashboard: React.FC = () => {
  return (
    <Grid container spacing={2}>
      {/* Real-time Monitor */}
      <Grid item xs={12} md={6}>
        <RealTimeMonitor />
      </Grid>
      
      {/* Label Quality Metrics */}
      <Grid item xs={12} md={6}>
        <LabelQualityPanel />
      </Grid>
      
      {/* Enhanced Triple Barrier Visualization */}
      <Grid item xs={12}>
        <EnhancedTripleBarrierChart />
      </Grid>
      
      {/* Label Distribution Analysis */}
      <Grid item xs={12} md={8}>
        <LabelDistributionChart />
      </Grid>
      
      {/* Performance Metrics */}
      <Grid item xs={12} md={4}>
        <PerformanceMetrics />
      </Grid>
      
      {/* Backfill Progress */}
      <Grid item xs={12}>
        <BackfillProgressTable />
      </Grid>
    </Grid>
  );
};
```

### 6.2 Key Visualizations

1. **Enhanced Triple Barrier Chart**
   - Candlestick chart with barrier levels overlaid
   - S/R levels highlighted
   - Color-coded by label outcome
   - Interactive zoom and pan

2. **Label Quality Monitor**
   - Real-time computation latency
   - Missing label alerts
   - Distribution drift detection
   - Information coefficient tracking

3. **Backfill Progress Table**
   - Instrument/granularity grid
   - Progress bars
   - ETA calculations
   - Error/retry status

---

## 7. Performance Requirements

### 7.1 Latency SLAs

| Operation | P50 | P95 | P99 | Max |
|-----------|-----|-----|-----|-----|
| Incremental label compute | 20ms | 50ms | 100ms | 500ms |
| Cache hit read | 1ms | 5ms | 10ms | 50ms |
| Batch compute (per 1k candles) | 100ms | 500ms | 1s | 5s |
| UI dashboard load | 200ms | 500ms | 1s | 3s |

### 7.2 Throughput Requirements

- **Batch backfill**: 1M+ candles/minute per instrument
- **Incremental**: 1000 candles/second across all instruments
- **Concurrent operations**: 100+ simultaneous label computations
- **Cache capacity**: 24 hours of labels for all active instruments

---

## 8. Monitoring & Observability

### 8.1 Key Metrics

```python
# Prometheus metrics
label_computation_duration = Histogram(
    'label_computation_duration_seconds',
    'Time to compute labels',
    ['instrument', 'granularity', 'label_type']
)

label_computation_errors = Counter(
    'label_computation_errors_total',
    'Total label computation errors',
    ['instrument', 'granularity', 'error_type']
)

cache_hit_rate = Gauge(
    'label_cache_hit_rate',
    'Cache hit rate for label queries',
    ['cache_level']
)

label_quality_score = Gauge(
    'label_quality_score',
    'Statistical quality score of computed labels',
    ['instrument', 'granularity', 'label_type']
)
```

### 8.2 Alerting Rules

1. **Critical Alerts**
   - Incremental latency > 200ms for 5 minutes
   - Cache hit rate < 80%
   - Label computation errors > 1% of requests
   - Missing labels for active trading hours

2. **Warning Alerts**
   - Batch backfill behind schedule > 1 hour
   - Label distribution drift > 2 standard deviations
   - Redis memory usage > 80%
   - ClickHouse query time > 1 second

---

## 9. Implementation Timeline

### Phase 1: Foundation (Week 1-2)
- [ ] Set up ClickHouse schema
- [ ] Implement Label 11.a (Enhanced Triple Barrier)
- [ ] Build multi-timeframe alignment logic
- [ ] Create Redis cache layer
- [ ] Deploy basic monitoring

### Phase 2: Core Labels (Week 3-4)
- [ ] Implement top 5 priority labels
- [ ] Build batch backfill pipeline
- [ ] Add data validation checks
- [ ] Create test harness
- [ ] Document computation logic

### Phase 3: Real-time Pipeline (Week 5-6)
- [ ] Connect to Firestore listener
- [ ] Build incremental computation
- [ ] Optimize cache warming
- [ ] Add circuit breakers
- [ ] Performance testing

### Phase 4: UI & Monitoring (Week 7-8)
- [ ] Deploy React dashboard
- [ ] Implement visualizations
- [ ] Add alerting rules
- [ ] Create operational runbooks
- [ ] Production deployment

---

## 10. Risk Mitigation

### 10.1 Technical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Look-ahead bias | Critical | Strict temporal checks, automated validation |
| Timeframe misalignment | High | Enforced alignment logic, unit tests |
| Cache inconsistency | Medium | TTL management, cache invalidation |
| ClickHouse overload | High | Query optimization, materialized views |
| Label computation errors | High | Graceful degradation, fallback values |

### 10.2 Operational Risks

- **Data gaps**: Implement gap detection and backfill automation
- **Version mismatch**: Label versioning system with migration tools
- **Performance degradation**: Auto-scaling, load balancing
- **Monitoring blind spots**: Comprehensive observability stack

---

## Appendix A: Timestamp Alignment Functions

```python
def align_timestamp(ts, granularity):
    """
    CRITICAL: Align timestamps to granularity boundaries
    """
    if granularity == 'H4':
        # H4 candles at 1,5,9,13,17,21 UTC
        hour = ts.hour
        aligned_hour = ((hour - 1) // 4) * 4 + 1
        if aligned_hour < 1:
            aligned_hour = 21
            ts = ts - timedelta(days=1)
        return ts.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
    
    elif granularity == 'D':
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    
    elif granularity == 'W':
        # Align to Monday
        days_to_monday = ts.weekday()
        return (ts - timedelta(days=days_to_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    
    elif granularity == 'H1':
        return ts.replace(minute=0, second=0, microsecond=0)
    
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")
```

---

## Appendix B: Label Computation Formulas

[Include detailed mathematical formulas for all 37 labels from labels.spec.md]

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-10 | System | Initial PRD with Label 11.a and multi-timeframe alignment |