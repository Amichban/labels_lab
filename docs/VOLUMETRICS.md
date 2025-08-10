# Data Volumetrics & Performance Requirements

## Current State

### Tables & Row Counts
| Table | Current Rows | Growth Rate | 1 Year Projection |
|-------|-------------|-------------|-------------------|
| users | 10,000 | +500/month | 16,000 |
| posts | 100,000 | +10,000/month | 220,000 |
| comments | 500,000 | +50,000/month | 1,100,000 |
| activities | 2,000,000 | +200,000/month | 4,400,000 |
| notifications | 5,000,000 | +500,000/month | 11,000,000 |
| analytics_events | 10,000,000 | +2,000,000/month | 34,000,000 |

### Access Patterns
| Pattern | Frequency | Latency Target | Current p95 |
|---------|-----------|----------------|-------------|
| User login | 1,000/min | <100ms | 85ms |
| Load feed | 5,000/min | <200ms | 180ms |
| Create post | 100/min | <500ms | 420ms |
| Search posts | 500/min | <1s | 950ms |
| Analytics query | 10/min | <5s | 4.2s |
| Bulk export | 1/hour | <30min | 25min |

### Hot Spots
- **Peak hours**: 8-10am, 12-1pm, 6-8pm (3x normal traffic)
- **Geographic distribution**: 60% US, 25% EU, 15% APAC
- **Mobile vs Desktop**: 70% mobile, 30% desktop

## Performance Requirements

### Query Performance
| Query Type | p50 Target | p95 Target | p99 Target |
|------------|------------|------------|------------|
| Point lookup (by ID) | <10ms | <50ms | <100ms |
| Range scan (pagination) | <50ms | <200ms | <500ms |
| Full-text search | <200ms | <1s | <2s |
| Aggregation | <500ms | <2s | <5s |
| Complex join (3+ tables) | <100ms | <500ms | <1s |

### Write Performance
| Operation | p50 Target | p95 Target | p99 Target |
|-----------|------------|------------|------------|
| Single insert | <20ms | <100ms | <200ms |
| Bulk insert (100 rows) | <200ms | <1s | <2s |
| Update with index | <30ms | <150ms | <300ms |
| Delete cascade | <50ms | <250ms | <500ms |

### Throughput Requirements
- **Reads**: 10,000 queries/second sustained
- **Writes**: 1,000 writes/second sustained
- **Peak multiplier**: 3x during peak hours
- **Batch jobs**: Process 1M records in <10 minutes

## Data Retention & Archival

### Retention Policies
| Data Type | Hot Storage | Warm Storage | Cold Storage | Delete After |
|-----------|-------------|--------------|--------------|--------------|
| User data | Forever | - | - | Never (GDPR export) |
| Posts | 2 years | 3 years | Forever | Never |
| Activities | 90 days | 1 year | 2 years | 3 years |
| Analytics | 30 days | 90 days | 1 year | 2 years |
| Logs | 7 days | 30 days | 90 days | 1 year |

### Archival Strategy
- **Hot → Warm**: After retention period, move to compressed tables
- **Warm → Cold**: After warm period, export to S3/GCS
- **Partitioning**: Monthly partitions for time-series data
- **Compression**: ZSTD for warm storage, Parquet for cold

## Indexing Strategy

### Critical Indexes
```sql
-- Users table
CREATE INDEX idx_users_email ON users(email);  -- Login
CREATE INDEX idx_users_created ON users(created_at DESC);  -- Recent users

-- Posts table  
CREATE INDEX idx_posts_user_created ON posts(user_id, created_at DESC);  -- User timeline
CREATE INDEX idx_posts_status_created ON posts(status, created_at DESC) WHERE status = 'published';  -- Feed

-- Activities table (partitioned by month)
CREATE INDEX idx_activities_user_date ON activities(user_id, created_at DESC);  -- User activity
CREATE INDEX idx_activities_type_date ON activities(type, created_at DESC);  -- Activity by type

-- Full-text search
CREATE INDEX idx_posts_search ON posts USING gin(to_tsvector('english', title || ' ' || content));
```

### Index Performance Targets
- **Index size**: <20% of table size
- **Index scan**: <100ms for 10k rows
- **Index rebuild**: <5 minutes for 1M rows
- **Cardinality**: >0.1 for effective indexes

## Partitioning Strategy

### Tables to Partition
| Table | Partition Key | Partition Type | Retention |
|-------|--------------|----------------|-----------|
| analytics_events | created_at | Range (monthly) | 24 months |
| activities | created_at | Range (monthly) | 36 months |
| notifications | created_at | Range (weekly) | 12 weeks |
| audit_logs | created_at | Range (daily) | 90 days |

### Partition Management
```sql
-- Example: Monthly partitions for analytics_events
CREATE TABLE analytics_events (
    id BIGSERIAL,
    user_id BIGINT,
    event_type VARCHAR(50),
    properties JSONB,
    created_at TIMESTAMP NOT NULL
) PARTITION BY RANGE (created_at);

-- Create partitions for next 3 months
CREATE TABLE analytics_events_2024_01 PARTITION OF analytics_events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

## Scaling Triggers

### Vertical Scaling
- **CPU**: Scale up when sustained >80% for 5 minutes
- **Memory**: Scale up when <10% free for 5 minutes
- **IOPS**: Scale up when queue depth >10 for 5 minutes
- **Storage**: Scale up when <20% free space

### Horizontal Scaling
- **Read replicas**: Add when read QPS >5000
- **Sharding trigger**: When single table >100GB
- **Connection pooling**: When connections >80% of max

## Migration Performance Requirements

### Online Migration Constraints
- **Maximum downtime**: 30 seconds (for critical tables)
- **Batch size**: 1000 rows per transaction
- **Throttling**: 50% of write capacity
- **Validation**: Sample 1% of migrated data
- **Rollback time**: <5 minutes

### Backfill Strategy
```python
# Chunked backfill with monitoring
CHUNK_SIZE = 1000
SLEEP_MS = 100  # Between chunks

def backfill_column():
    total = SELECT COUNT(*) FROM table
    for offset in range(0, total, CHUNK_SIZE):
        UPDATE table 
        SET new_column = calculate_value(old_column)
        WHERE id IN (
            SELECT id FROM table 
            ORDER BY id 
            LIMIT CHUNK_SIZE OFFSET offset
        )
        
        # Monitor impact
        check_replication_lag()
        check_cpu_usage()
        time.sleep(SLEEP_MS / 1000)
```

## Testing Requirements

### Load Testing Scenarios
1. **Normal load**: 1000 concurrent users, 5000 QPS
2. **Peak load**: 3000 concurrent users, 15000 QPS
3. **Spike test**: 0 → 5000 users in 1 minute
4. **Soak test**: 1000 users for 24 hours
5. **Batch processing**: 1M records in 10 minutes

### Performance Benchmarks
```python
# K6 test example
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp up
    { duration: '5m', target: 100 },  // Stay at 100
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<200'],  // 95% of requests under 200ms
    http_req_failed: ['rate<0.01'],    // Error rate under 1%
  },
};
```

## Cost Optimization

### Storage Costs
- **Hot storage**: $0.10/GB/month (SSD)
- **Warm storage**: $0.03/GB/month (HDD)
- **Cold storage**: $0.01/GB/month (S3)
- **Target**: <$1000/month for 1TB total

### Compute Costs
- **Database**: $500/month (current)
- **Read replicas**: $200/month each
- **Target**: <$1500/month total

### Optimization Opportunities
1. Archive old analytics data (save $200/month)
2. Compress large JSON columns (save 40% storage)
3. Drop unused indexes (save 10% storage, improve writes)
4. Implement caching layer (reduce DB load 30%)

## Monitoring & Alerts

### Key Metrics
| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Query latency p95 | >500ms | >1s | Scale up/optimize |
| Replication lag | >1s | >5s | Reduce write load |
| Connection pool | >70% | >90% | Increase pool size |
| Disk usage | >70% | >85% | Add storage |
| Lock waits | >100/min | >500/min | Review queries |
| Cache hit rate | <80% | <60% | Tune cache |

This volumetrics data drives all schema design and migration decisions!