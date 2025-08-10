
# Label Computation System API

A high-performance label computation system for quantitative trading that processes support/resistance level events and computes forward-looking labels for pattern mining and ML models.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Authentication](#authentication)
4. [API Endpoints](#api-endpoints)
5. [Rate Limiting](#rate-limiting)
6. [Error Handling](#error-handling)
7. [SDKs & Clients](#sdks--clients)
8. [Examples](#examples)
9. [Performance & SLAs](#performance--slas)
10. [Support](#support)

## Overview

### Key Features

- **Dual-mode processing**: Batch backfill (minutes) and real-time incremental (<1 second)
- **Multi-timeframe alignment**: Uses lower granularity data for path-dependent calculations
- **High performance**: 1M+ candles/minute batch throughput, <100ms p99 incremental latency
- **Scale**: Support for billions of labels across 29 FX pairs and indices
- **Reliability**: >95% cache hit rate for recent 24 hours, zero look-ahead bias

### Architecture

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

## Getting Started

### Base URLs

- **Production**: `https://api.labelcompute.com/v1`
- **Staging**: `https://staging-api.labelcompute.com/v1`
- **Development**: `http://localhost:8000/v1`

### Quick Start

1. **Get an API key** from your account dashboard
2. **Make your first request**:

```bash
curl -X POST "https://api.labelcompute.com/v1/labels/compute" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "instrument_id": "EURUSD",
    "granularity": "H4",
    "candle": {
      "ts": "2024-01-10T13:00:00Z",
      "open": 1.0950,
      "high": 1.0970,
      "low": 1.0940,
      "close": 1.0965,
      "volume": 1250000,
      "atr_14": 0.0025
    },
    "label_types": ["enhanced_triple_barrier"]
  }'
```

## Authentication

The API supports two authentication methods:

### Bearer Token (Recommended)
```http
Authorization: Bearer YOUR_JWT_TOKEN
```

### API Key
```http
X-API-Key: YOUR_API_KEY
```

## API Endpoints

### Label Computation

#### Compute Labels (Incremental)
```http
POST /v1/labels/compute
```

Compute labels for a single candle in real-time. **Target latency: <100ms p99**.

**Request Body**:
```json
{
  "instrument_id": "EURUSD",
  "granularity": "H4",
  "candle": {
    "ts": "2024-01-10T13:00:00Z",
    "open": 1.0950,
    "high": 1.0970,
    "low": 1.0940,
    "close": 1.0965,
    "volume": 1250000,
    "atr_14": 0.0025
  },
  "label_types": ["enhanced_triple_barrier", "vol_scaled_return"],
  "options": {
    "horizon_periods": 6,
    "use_cache": true
  }
}
```

**Response**:
```json
{
  "instrument_id": "EURUSD",
  "granularity": "H4",
  "ts": "2024-01-10T13:00:00Z",
  "labels": {
    "enhanced_triple_barrier": {
      "label": 1,
      "barrier_hit": "upper",
      "time_to_barrier": 3,
      "barrier_price": 1.0990,
      "level_adjusted": true
    },
    "vol_scaled_return": {
      "value": 2.35,
      "quantile": 0.85
    }
  },
  "computation_time_ms": 45,
  "cache_hit": false,
  "version": "1.0.0"
}
```

#### Query Labels
```http
GET /v1/labels
```

Retrieve computed labels with flexible filtering and pagination.

**Query Parameters**:
- `instrument_id` (required): Instrument identifier (e.g., "EURUSD")
- `granularity` (required): Time granularity ("M15", "H1", "H4", "D", "W")
- `start_date` (required): Start date (ISO 8601)
- `end_date` (required): End date (ISO 8601)
- `label_types` (optional): Comma-separated label types
- `page` (optional): Page number (default: 1)
- `per_page` (optional): Items per page (default: 100, max: 1000)
- `sort` (optional): Sort order ("ts_asc", "ts_desc", "forward_return_asc", "forward_return_desc")

**Example**:
```http
GET /v1/labels?instrument_id=EURUSD&granularity=H4&start_date=2024-01-01T00:00:00Z&end_date=2024-01-02T00:00:00Z&page=1&per_page=100
```

#### Get Labels by Timestamp
```http
GET /v1/labels/{instrument_id}/{granularity}/{timestamp}
```

Get all labels for a specific candle timestamp.

**Example**:
```http
GET /v1/labels/EURUSD/H4/2024-01-10T13:00:00Z
```

### Batch Operations

#### Start Batch Backfill
```http
POST /v1/batch/backfill
```

Initiate batch computation of labels for a date range. **Target throughput: 1M+ candles/minute**.

**Request Body**:
```json
{
  "instrument_id": "EURUSD",
  "granularity": "H4",
  "start_date": "2024-01-01T00:00:00Z",
  "end_date": "2024-01-31T23:59:59Z",
  "label_types": ["enhanced_triple_barrier", "vol_scaled_return"],
  "options": {
    "chunk_size": 10000,
    "parallel_workers": 8,
    "force_recompute": false
  }
}
```

**Response**:
```json
{
  "job_id": "bf_20240110_eurusd_h4_abc123",
  "status": "started",
  "estimated_duration_minutes": 45,
  "estimated_candles": 186000,
  "_links": {
    "self": "/batch/jobs/bf_20240110_eurusd_h4_abc123",
    "status": "/batch/jobs/bf_20240110_eurusd_h4_abc123/status"
  }
}
```

#### Get Batch Job Status
```http
GET /v1/batch/jobs/{job_id}
```

Monitor the progress of a batch backfill operation.

**Response**:
```json
{
  "job_id": "bf_20240110_eurusd_h4_abc123",
  "status": "running",
  "progress": {
    "completed_candles": 125000,
    "total_candles": 186000,
    "percentage": 67.2,
    "current_date": "2024-01-21T00:00:00Z"
  },
  "performance": {
    "candles_per_minute": 2750,
    "avg_compute_time_ms": 22,
    "cache_hit_rate": 0.94
  },
  "estimated_completion": "2024-01-10T15:30:00Z"
}
```

#### List Batch Jobs
```http
GET /v1/batch/jobs
```

Get paginated list of batch jobs with filtering.

**Query Parameters**:
- `status` (optional): Filter by status ("pending", "running", "completed", "failed", "cancelled")
- `instrument_id` (optional): Filter by instrument
- `granularity` (optional): Filter by granularity
- `page`, `per_page`, `sort`: Standard pagination parameters

### Health & Monitoring

#### System Health Check
```http
GET /v1/health
```

Comprehensive health check including all dependencies.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-10T15:30:00Z",
  "checks": {
    "clickhouse": "ok",
    "redis": "ok",
    "cache_hit_rate": "ok"
  },
  "metrics": {
    "cache_hit_rate": 0.96,
    "avg_computation_ms": 42,
    "active_batch_jobs": 2
  }
}
```

#### System Metrics
```http
GET /v1/metrics
```

Detailed system metrics for monitoring and observability.

**Query Parameters**:
- `format` (optional): Response format ("json", "prometheus")
- `window` (optional): Time window ("5m", "15m", "1h", "6h", "24h")

### Cache Management

#### Cache Statistics
```http
GET /v1/cache/stats
```

Get detailed cache performance statistics.

#### Warm Cache
```http
POST /v1/cache/warm
```

Pre-load cache with recent data for an instrument.

**Request Body**:
```json
{
  "instrument_id": "EURUSD",
  "granularity": "H4",
  "hours": 24
}
```

#### Invalidate Cache
```http
DELETE /v1/cache/invalidate
```

Clear cache for specific instrument or patterns.

**Query Parameters**:
- `instrument_id` (optional): Instrument to invalidate
- `granularity` (optional): Granularity to invalidate
- `pattern` (optional): Redis key pattern to match

## Rate Limiting

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Label compute | 1000 req/hour | Rolling |
| Queries | 5000 req/hour | Rolling |
| Batch operations | 100 req/hour | Rolling |
| Health checks | Unlimited | - |

**Rate Limit Headers**:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Unix timestamp when limit resets
- `Retry-After`: Seconds until requests can be made again (on 429)

## Error Handling

All errors follow a consistent format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data",
    "details": [
      {
        "field": "candle.close",
        "message": "Must be greater than 0",
        "code": "MIN_VALUE"
      }
    ],
    "trace_id": "req_abc123def456"
  }
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 202 | Accepted (for async operations) |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 409 | Conflict (e.g., backfill already running) |
| 422 | Validation Error |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Common Error Codes

- `BAD_REQUEST`: Invalid request parameters
- `UNAUTHORIZED`: Authentication required or invalid
- `VALIDATION_ERROR`: Input data validation failed
- `NOT_FOUND`: Resource not found
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `BACKFILL_IN_PROGRESS`: Backfill already running for this range
- `INTERNAL_ERROR`: Server error

## SDKs & Clients

### TypeScript/JavaScript

```typescript
import { createProductionClient } from './api/client';

const client = createProductionClient({
  bearerToken: 'your-jwt-token'
});

// Compute labels
const result = await client.computeLabels({
  instrument_id: 'EURUSD',
  granularity: 'H4',
  candle: {
    ts: '2024-01-10T13:00:00Z',
    open: 1.0950,
    high: 1.0970,
    low: 1.0940,
    close: 1.0965,
    volume: 1250000,
    atr_14: 0.0025
  }
});

console.log(result.data.labels);
```

### Python

```python
from src.api.schemas import CandleLabelRequest, CandleData
import httpx

client = httpx.Client(
    base_url="https://api.labelcompute.com/v1",
    headers={"Authorization": "Bearer your-jwt-token"}
)

request = CandleLabelRequest(
    instrument_id="EURUSD",
    granularity="H4",
    candle=CandleData(
        ts="2024-01-10T13:00:00Z",
        open=1.0950,
        high=1.0970,
        low=1.0940,
        close=1.0965,
        volume=1250000,
        atr_14=0.0025
    )
)

response = client.post("/labels/compute", json=request.dict())
result = response.json()
print(result['labels'])
```

## Examples

### Real-time Label Computation

```javascript
// Compute enhanced triple barrier label
const response = await client.computeLabels({
  instrument_id: 'EURUSD',
  granularity: 'H4',
  candle: {
    ts: '2024-01-10T13:00:00Z',
    open: 1.0950,
    high: 1.0970,
    low: 1.0940,
    close: 1.0965,
    volume: 1250000,
    atr_14: 0.0025
  },
  label_types: ['enhanced_triple_barrier'],
  options: {
    horizon_periods: 6,
    use_cache: true
  }
});

console.log('Label:', response.data.labels.enhanced_triple_barrier.label);
console.log('Barrier hit:', response.data.labels.enhanced_triple_barrier.barrier_hit);
console.log('Computation time:', response.data.computation_time_ms, 'ms');
```

### Batch Processing with Progress Monitoring

```javascript
// Start batch backfill
const job = await client.startBatchBackfill({
  instrument_id: 'EURUSD',
  granularity: 'H4',
  start_date: '2024-01-01T00:00:00Z',
  end_date: '2024-01-31T23:59:59Z',
  label_types: ['enhanced_triple_barrier', 'vol_scaled_return']
});

console.log('Job started:', job.data.job_id);

// Poll for completion
const finalStatus = await client.pollBatchJob(job.data.job_id, {
  intervalMs: 5000,
  onProgress: (status) => {
    console.log(`Progress: ${status.progress.percentage.toFixed(1)}%`);
    console.log(`Speed: ${status.performance?.candles_per_minute} candles/min`);
  }
});

console.log('Final status:', finalStatus.status);
```

### Querying Historical Labels

```javascript
// Query labels with filtering
const labels = await client.getLabels({
  instrument_id: 'EURUSD',
  granularity: 'H4',
  start_date: '2024-01-01T00:00:00Z',
  end_date: '2024-01-02T00:00:00Z',
  enhanced_triple_barrier_label: 1, // Only upper barrier hits
  return_quantile_min: 80, // Only high return quantiles
  page: 1,
  per_page: 100,
  sort: 'ts_asc'
});

console.log(`Found ${labels.data.data.length} labels`);
labels.data.data.forEach(label => {
  console.log(`${label.ts}: ${label.labels.forward_return}`);
});
```

### Health Monitoring

```javascript
// Check system health
const health = await client.getHealth();
console.log('System status:', health.data.status);
console.log('Cache hit rate:', health.data.metrics?.cache_hit_rate);

// Wait for system to become healthy
await client.waitForHealth({ timeoutMs: 60000 });
console.log('System is healthy!');

// Get detailed metrics
const metrics = await client.getMetrics({ window: '1h' });
console.log('P99 latency:', metrics.data.performance?.p99_computation_time_ms, 'ms');
```

## Performance & SLAs

### Latency Targets

| Operation | P50 | P95 | P99 | Max |
|-----------|-----|-----|-----|-----|
| Incremental compute | 20ms | 50ms | 100ms | 500ms |
| Cache hit read | 1ms | 5ms | 10ms | 50ms |
| Batch (per 1k candles) | 100ms | 500ms | 1s | 5s |
| UI dashboard load | 200ms | 500ms | 1s | 3s |

### Throughput Targets

- **Batch backfill**: 1M+ candles/minute per instrument
- **Incremental**: 1000 candles/second across all instruments
- **Concurrent operations**: 100+ simultaneous label computations
- **Cache capacity**: 24 hours of labels for all active instruments

### Availability

- **Uptime SLA**: 99.9%
- **Cache hit rate**: >95% for recent 24 hours
- **Data freshness**: <1 second for incremental updates
- **Recovery time**: <5 minutes for system restarts

## Support

### Documentation

- **OpenAPI Spec**: [openapi.yaml](./openapi.yaml)
- **Postman Collection**: [postman-collection.json](./postman-collection.json)
- **API Changelog**: [api-changelog.md](./api-changelog.md)
- **Example Requests**: [examples/requests/](./examples/requests/)
- **Example Responses**: [examples/responses/](./examples/responses/)

### Getting Help

- **Email**: support@labelcompute.com
- **Slack**: #label-compute-api
- **Status Page**: https://status.labelcompute.com
- **GitHub Issues**: For SDK bugs and feature requests

### Monitoring & Alerts

- **Grafana Dashboard**: https://monitoring.labelcompute.com
- **Prometheus Metrics**: `GET /v1/metrics?format=prometheus`
- **Health Check**: `GET /v1/health` (for load balancers)
- **Alerting**: Configurable via webhook endpoints

---

## License

This API and documentation are proprietary to Label Compute System.

© 2024 Label Compute System. All rights reserved.