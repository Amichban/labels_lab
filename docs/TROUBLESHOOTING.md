# Label Computation System Troubleshooting Guide

> Comprehensive guide for diagnosing and resolving common issues in the Label Computation System.

## Table of Contents

1. [Common Issues & Solutions](#common-issues--solutions)
2. [Performance Problems](#performance-problems)
3. [Database Issues](#database-issues)
4. [Cache Problems](#cache-problems)
5. [API Errors](#api-errors)
6. [Batch Processing Issues](#batch-processing-issues)
7. [Validation Failures](#validation-failures)
8. [Deployment Problems](#deployment-problems)
9. [Monitoring & Debugging](#monitoring--debugging)
10. [Emergency Procedures](#emergency-procedures)

## Common Issues & Solutions

### 1. Service Won't Start

#### Symptoms
- Container exits immediately
- "Connection refused" errors
- Health checks failing

#### Diagnosis
```bash
# Check container logs
docker logs labels-api

# For Kubernetes
kubectl logs -n label-computation deploy/labels-api

# Check service status
docker-compose ps
kubectl get pods -n label-computation
```

#### Common Causes & Solutions

**Missing Environment Variables**
```bash
# Error: Missing required environment variables
Error: Missing required environment variables: CLICKHOUSE_PASSWORD, REDIS_PASSWORD

# Solution: Check .env file or Kubernetes secrets
cat .env | grep -E "(CLICKHOUSE|REDIS)_PASSWORD"
kubectl get secret labels-secrets -n label-computation -o yaml
```

**Database Connection Failure**
```bash
# Error: ClickHouse connection failed
ERROR: ClickHouse connection error: Connection refused

# Solution: Verify ClickHouse is running and accessible
# For Docker Compose
docker-compose ps clickhouse
docker-compose logs clickhouse

# For Kubernetes
kubectl get pods -l app=clickhouse -n label-computation
kubectl logs -l app=clickhouse -n label-computation

# Test connection manually
clickhouse-client --host localhost --port 9000 --query "SELECT 1"
```

**Redis Connection Issues**
```bash
# Error: Redis connection failed
ERROR: Redis connection error: Connection refused

# Solution: Check Redis service
redis-cli -h localhost -p 6379 ping
# Should return: PONG

# Check password authentication
redis-cli -h localhost -p 6379 -a your_password ping
```

### 2. High Memory Usage

#### Symptoms
- Out of Memory (OOM) kills
- Swap usage increasing
- Slow response times

#### Diagnosis
```bash
# Check memory usage
docker stats labels-api

# For Kubernetes
kubectl top pods -n label-computation

# Check memory limits
kubectl describe pod <pod-name> -n label-computation
```

#### Solutions

**Adjust Memory Limits**
```yaml
# Kubernetes deployment
resources:
  requests:
    memory: "2Gi"
  limits:
    memory: "4Gi"
```

**Optimize Batch Size**
```python
# Reduce batch chunk size in configuration
BATCH_CHUNK_SIZE = 5000  # Default is 10000
PARALLEL_WORKERS = 4     # Reduce if memory constrained
```

**Enable Memory Profiling**
```python
# Add to environment variables
ENABLE_PROFILING=true
MEMORY_PROFILER_ENABLED=true

# View memory usage
curl http://localhost:8000/v1/debug/memory
```

### 3. Database Query Timeouts

#### Symptoms
- "Query timeout" errors
- Slow API responses
- ClickHouse connection errors

#### Diagnosis
```sql
-- Check running queries
SELECT query, elapsed FROM system.processes WHERE elapsed > 10;

-- Check query performance
SELECT query, query_duration_ms 
FROM system.query_log 
WHERE event_date = today() 
ORDER BY query_duration_ms DESC 
LIMIT 10;
```

#### Solutions

**Optimize Query Performance**
```sql
-- Add indexes for commonly queried columns
ALTER TABLE quantx.labels 
ADD INDEX idx_instrument_granularity (instrument_id, granularity) 
TYPE bloom_filter GRANULARITY 1;

-- Optimize table engine settings
ALTER TABLE quantx.labels 
MODIFY SETTING index_granularity = 8192;
```

**Increase Timeout Values**
```python
# Environment configuration
CLICKHOUSE_QUERY_TIMEOUT = 60  # seconds
CLICKHOUSE_MAX_EXECUTION_TIME = 300
```

## Performance Problems

### 1. Slow Label Computation

#### Symptoms
- API latency > 100ms
- Batch processing below 1M candles/minute
- High CPU usage

#### Diagnosis
```bash
# Check computation metrics
curl http://localhost:8000/v1/metrics | grep computation_duration

# Profile specific labels
curl -X POST http://localhost:8000/v1/labels/compute \
  -H "X-Profile: true" \
  -d @test_request.json
```

#### Solutions

**Cache Optimization**
```python
# Warm cache for active instruments
curl -X POST http://localhost:8000/v1/cache/warm \
  -d '{"instrument_id": "EURUSD", "granularity": "H4", "hours": 24}'

# Check cache hit rate
redis-cli --latency-history -i 1

# Increase cache TTL for stable data
CACHE_TTL_SECONDS = 7200  # 2 hours
```

**Database Connection Pooling**
```python
# Increase connection pool size
CLICKHOUSE_MAX_CONNECTIONS = 100
CLICKHOUSE_POOL_SIZE = 50

# Enable connection pooling
DATABASE_POOL_ENABLED = true
```

**Parallel Processing**
```python
# Optimize worker configuration
PARALLEL_WORKERS = min(16, cpu_count() * 2)
BATCH_CHUNK_SIZE = 25000  # Larger chunks for better throughput
```

### 2. Cache Miss Rate High

#### Symptoms
- Cache hit rate < 90%
- Increased database load
- Slower response times

#### Diagnosis
```bash
# Check cache statistics
redis-cli info stats | grep -E "(hits|misses|hit_rate)"

# Monitor cache keys
redis-cli monitor

# Check memory usage
redis-cli info memory
```

#### Solutions

**Cache Warming Strategy**
```python
# Implement proactive cache warming
def warm_cache_for_active_instruments():
    instruments = ["EURUSD", "GBPUSD", "USDJPY"]
    for instrument in instruments:
        warm_instrument_cache(instrument, hours=24)

# Schedule cache warming
asyncio.create_task(warm_cache_for_active_instruments())
```

**Optimize Cache Keys**
```python
# Use consistent key patterns
cache_key = f"labels:{instrument_id}:{granularity}:{timestamp_aligned}"

# Implement cache versioning
cache_key = f"labels:v1:{instrument_id}:{granularity}:{timestamp}"
```

## Database Issues

### 1. ClickHouse Storage Full

#### Symptoms
- "No space left on device" errors
- Insert operations failing
- Read-only mode activation

#### Diagnosis
```bash
# Check disk usage
df -h /var/lib/clickhouse

# Check table sizes
SELECT 
    table,
    formatReadableSize(sum(bytes)) as size
FROM system.parts 
GROUP BY table 
ORDER BY sum(bytes) DESC;
```

#### Solutions

**Data Cleanup**
```sql
-- Drop old partitions (older than 6 months)
ALTER TABLE quantx.labels DROP PARTITION '202306';

-- Optimize tables to reclaim space
OPTIMIZE TABLE quantx.labels FINAL;

-- Enable TTL for automatic cleanup
ALTER TABLE quantx.labels 
MODIFY TTL ts + INTERVAL 6 MONTH;
```

**Storage Expansion**
```bash
# For Docker volumes
docker volume inspect <volume_name>

# Expand Kubernetes PVC
kubectl patch pvc clickhouse-data-0 -n label-computation \
  -p '{"spec":{"resources":{"requests":{"storage":"1Ti"}}}}'
```

### 2. Data Corruption

#### Symptoms
- Invalid label values
- Inconsistent results
- Validation errors

#### Diagnosis
```sql
-- Check for data anomalies
SELECT 
    instrument_id,
    granularity,
    count(*) as records,
    countIf(isNull(enhanced_triple_barrier_label)) as null_labels
FROM quantx.labels
WHERE ts >= today() - 7
GROUP BY instrument_id, granularity
HAVING null_labels > 0;
```

#### Solutions

**Data Validation**
```sql
-- Identify corrupted records
SELECT * FROM quantx.labels 
WHERE enhanced_triple_barrier_label NOT IN (-1, 0, 1)
   OR computation_time_ms < 0
   OR isNull(ts);

-- Clean up invalid data
ALTER TABLE quantx.labels 
DELETE WHERE enhanced_triple_barrier_label NOT IN (-1, 0, 1);
```

**Recompute Corrupted Labels**
```python
# Trigger recomputation for specific period
curl -X POST http://localhost:8000/v1/batch/backfill \
  -d '{
    "instrument_id": "EURUSD",
    "granularity": "H4",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-02T00:00:00Z",
    "options": {"force_recompute": true}
  }'
```

## Cache Problems

### 1. Redis Memory Exhaustion

#### Symptoms
- Cache evictions increasing
- "OOM" errors in Redis logs
- Decreased cache hit rate

#### Diagnosis
```bash
# Check memory usage
redis-cli info memory | grep -E "(used_memory|maxmemory)"

# Monitor evictions
redis-cli info stats | grep evicted_keys

# Check key distribution
redis-cli --scan | head -100
```

#### Solutions

**Memory Optimization**
```bash
# Increase Redis memory limit
redis-server --maxmemory 8gb --maxmemory-policy allkeys-lru

# Optimize data structures
redis-cli config set hash-max-ziplist-entries 512
redis-cli config set hash-max-ziplist-value 64
```

**Cache Policy Tuning**
```python
# Adjust TTL values based on data access patterns
CACHE_TTL_LABELS = 3600      # 1 hour for computed labels
CACHE_TTL_LEVELS = 300       # 5 minutes for active levels
CACHE_TTL_PATH_DATA = 600    # 10 minutes for path data
```

### 2. Cache Consistency Issues

#### Symptoms
- Stale data returned
- Inconsistent results
- Cache/database mismatch

#### Solutions

**Cache Invalidation**
```python
# Implement proper cache invalidation
async def invalidate_related_cache(instrument_id, granularity, timestamp):
    patterns = [
        f"labels:{instrument_id}:{granularity}:*",
        f"levels:{instrument_id}:{granularity}:*",
        f"path:{instrument_id}:{granularity}:*"
    ]
    for pattern in patterns:
        await redis_cache.delete_pattern(pattern)
```

**Cache Versioning**
```python
# Add version tags to cache keys
cache_key = f"labels:v2:{instrument_id}:{granularity}:{timestamp}"

# Implement cache warming with version check
def warm_cache_with_version_check():
    current_version = get_label_computation_version()
    cached_version = redis_cache.get("system:version")
    if current_version != cached_version:
        flush_all_cache()
        redis_cache.set("system:version", current_version)
```

## API Errors

### 1. Authentication Failures

#### Symptoms
- 401 Unauthorized responses
- "Invalid token" errors
- Authentication timeouts

#### Diagnosis
```bash
# Test API key authentication
curl -H "X-API-Key: your_api_key" http://localhost:8000/v1/health

# Test JWT authentication
curl -H "Authorization: Bearer your_jwt_token" \
     http://localhost:8000/v1/labels/compute

# Check logs for auth errors
kubectl logs -n label-computation deploy/labels-api | grep -i auth
```

#### Solutions

**JWT Token Issues**
```python
# Verify JWT configuration
JWT_SECRET = "your-256-bit-secret"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Debug token validation
import jwt
try:
    decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    print("Token valid:", decoded)
except jwt.ExpiredSignatureError:
    print("Token expired")
except jwt.InvalidTokenError:
    print("Token invalid")
```

### 2. Rate Limiting

#### Symptoms
- 429 Too Many Requests
- Request queuing
- Client timeouts

#### Diagnosis
```bash
# Check rate limit headers
curl -I http://localhost:8000/v1/labels/compute

# Monitor request rates
curl http://localhost:8000/v1/metrics | grep requests_per_second
```

#### Solutions

**Adjust Rate Limits**
```python
# Increase rate limits for production
RATE_LIMIT_PER_MINUTE = 10000
BURST_LIMIT = 100

# Implement user-specific limits
RATE_LIMITS = {
    "premium": 5000,
    "standard": 1000,
    "basic": 100
}
```

## Batch Processing Issues

### 1. Batch Jobs Hanging

#### Symptoms
- Jobs stuck in "running" state
- No progress updates
- Worker processes consuming resources

#### Diagnosis
```bash
# Check job status
curl http://localhost:8000/v1/batch/jobs

# Monitor worker processes
ps aux | grep python | grep batch

# Check task queue
redis-cli llen batch:tasks
```

#### Solutions

**Job Recovery**
```python
# Implement job timeout handling
JOB_TIMEOUT_HOURS = 6
WORKER_HEALTH_CHECK_INTERVAL = 300  # 5 minutes

# Cancel stuck jobs
curl -X DELETE http://localhost:8000/v1/batch/jobs/{job_id}

# Restart worker pool
docker-compose restart labels-api
```

**Dead Letter Queue**
```python
# Check failed tasks
redis-cli llen batch:failed_tasks

# Retry failed tasks
curl -X POST http://localhost:8000/v1/batch/retry-failed
```

### 2. Memory Leaks in Workers

#### Symptoms
- Worker memory increasing over time
- OOM kills in worker processes
- Degrading performance

#### Solutions

**Worker Recycling**
```python
# Enable worker recycling
WORKER_MAX_TASKS = 1000
WORKER_RECYCLE_AFTER_HOURS = 4

# Monitor worker memory
def monitor_worker_memory():
    import psutil
    process = psutil.Process()
    if process.memory_info().rss > MAX_WORKER_MEMORY:
        restart_worker()
```

## Validation Failures

### 1. Look-ahead Bias Detection

#### Symptoms
- Validation warnings in logs
- Inconsistent backtest results
- Future data usage alerts

#### Diagnosis
```python
# Check validation logs
grep "look.ahead" /app/logs/labels-api.log

# Run validation tests
python -m pytest tests/unit/test_label_validator.py -v
```

#### Solutions

**Timestamp Alignment**
```python
# Ensure proper timestamp alignment
def align_timestamp(ts, granularity):
    if granularity == "H4":
        # H4 candles at 1,5,9,13,17,21 UTC
        hour = ts.hour
        aligned_hour = ((hour - 1) // 4) * 4 + 1
        return ts.replace(hour=aligned_hour, minute=0, second=0)
    # ... other granularities
```

**Path Data Validation**
```python
# Validate path data doesn't contain future information
def validate_path_data(candle_ts, path_data):
    for data_point in path_data:
        if data_point["ts"] <= candle_ts:
            raise ValidationError(f"Path data contains future information")
```

### 2. Statistical Anomalies

#### Symptoms
- Label distributions shifted
- Unusual correlation patterns
- Model performance degradation

#### Solutions

**Distribution Monitoring**
```python
# Check label distribution
def check_label_distribution():
    query = """
    SELECT 
        enhanced_triple_barrier_label,
        count(*) as count,
        count(*) / sum(count(*)) OVER () as percentage
    FROM quantx.labels 
    WHERE ts >= today() - 7
    GROUP BY enhanced_triple_barrier_label
    """
    # Expected: roughly 33% for each value (-1, 0, 1)
```

## Monitoring & Debugging

### 1. Health Check Endpoints

```bash
# Basic health check
curl http://localhost:8000/v1/health

# Detailed health with dependencies
curl http://localhost:8000/v1/health | jq .

# Kubernetes probes
curl http://localhost:8000/v1/health/ready
curl http://localhost:8000/v1/health/live
```

### 2. Debug Endpoints

```bash
# Memory usage
curl http://localhost:8000/v1/debug/memory

# Configuration dump
curl http://localhost:8000/v1/debug/config

# Active connections
curl http://localhost:8000/v1/debug/connections

# Performance metrics
curl http://localhost:8000/v1/metrics
```

### 3. Log Analysis

```bash
# Error patterns
grep -E "(ERROR|CRITICAL)" /app/logs/labels-api.log | tail -50

# Performance issues
grep -E "slow|timeout|latency" /app/logs/labels-api.log

# Validation failures
grep -E "validation.*fail" /app/logs/labels-api.log

# For structured logs (JSON)
jq 'select(.level == "ERROR")' /app/logs/labels-api.log
```

## Emergency Procedures

### 1. Service Recovery

**Immediate Actions**
1. Check service health: `curl http://localhost:8000/v1/health`
2. Review recent logs: `tail -100 /app/logs/labels-api.log`
3. Restart services: `docker-compose restart` or `kubectl rollout restart`
4. Verify dependencies: Check ClickHouse and Redis connectivity

**Escalation Steps**
1. Scale up replicas: `kubectl scale deploy/labels-api --replicas=5`
2. Check resource constraints: `kubectl describe node`
3. Review metrics: Check Grafana dashboards
4. Contact on-call engineer if issues persist

### 2. Data Recovery

**Database Recovery**
```bash
# Check recent backups
ls -la /backups/clickhouse/

# Restore from backup (if needed)
clickhouse-client --query="RESTORE DATABASE quantx FROM Disk('backups', '2024-01-10')"

# Verify data integrity
SELECT count(*) FROM quantx.labels WHERE ts >= today() - 1;
```

**Cache Recovery**
```bash
# Clear corrupted cache
redis-cli FLUSHALL

# Warm cache for critical instruments
curl -X POST http://localhost:8000/v1/cache/warm \
  -d '{"instrument_id": "EURUSD", "granularity": "H4"}'
```

### 3. Performance Emergency

**Quick Fixes**
```bash
# Reduce batch size
export BATCH_CHUNK_SIZE=5000

# Disable validation temporarily
export ENABLE_VALIDATION=false

# Scale down workers
export PARALLEL_WORKERS=4

# Restart with new settings
docker-compose restart labels-api
```

**Resource Scaling**
```bash
# Kubernetes horizontal scaling
kubectl scale deploy/labels-api --replicas=10

# Vertical scaling (requires restart)
kubectl patch deploy/labels-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"labels-api","resources":{"limits":{"memory":"4Gi","cpu":"2"}}}]}}}}'
```

### 4. Communication Templates

**Incident Alert**
```
🚨 INCIDENT: Label Computation System
Severity: HIGH
Start Time: 2024-01-10 14:30 UTC
Impact: API latency > 500ms, affecting 1000+ req/min
Actions: Investigating database performance issues
ETA: 30 minutes
Updates: Every 15 minutes
```

**Resolution Update**
```
✅ RESOLVED: Label Computation System
Duration: 45 minutes
Root Cause: ClickHouse query timeout due to missing index
Fix: Added index on (instrument_id, granularity, ts)
Prevention: Enhanced monitoring for query performance
Post-mortem: Scheduled for 2024-01-11 10:00 UTC
```

---

This troubleshooting guide covers the most common issues encountered in production environments. For issues not covered here, check the system logs, metrics dashboards, and contact the development team with specific error messages and reproduction steps.