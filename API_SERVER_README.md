# Label Computation System - FastAPI Server

A production-ready FastAPI implementation for high-performance label computation in quantitative trading. This system processes support/resistance level events and computes forward-looking labels for pattern mining and ML models.

## 🚀 Key Features

### Performance Targets (SLAs)
- **Incremental compute**: <100ms p99 latency
- **Batch throughput**: 1M+ candles/minute  
- **Cache hit rate**: >95% for recent 24 hours

### Core Capabilities
- **Dual-mode processing**: Real-time incremental (<1s) and batch backfill (minutes)
- **Multi-timeframe alignment**: Uses lower granularity data for path-dependent calculations
- **Enhanced Triple Barrier (Label 11.a)**: Primary label with S/R level adjustments
- **Production monitoring**: Health checks, metrics, and observability

## 📁 Project Structure

```
/Users/aminechbani/labels_lab/my-project/
├── main.py                          # FastAPI application entry point
├── start_server.py                  # Production server startup script
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container configuration
├── docker-compose.yml              # Multi-service deployment
├── .env.example                     # Environment configuration template
├── config/
│   └── settings.py                  # Application settings with Pydantic
├── src/
│   ├── api/
│   │   ├── middleware.py            # Auth, rate limiting, tracing middleware
│   │   ├── schemas.py               # Pydantic request/response models
│   │   └── routers/
│   │       ├── labels.py            # Label computation endpoints
│   │       ├── batch.py             # Batch processing endpoints
│   │       ├── health.py            # Health monitoring endpoints
│   │       ├── metrics.py           # System metrics endpoints
│   │       └── cache.py             # Cache management endpoints
│   ├── core/
│   │   └── label_computation.py     # Enhanced Triple Barrier computation
│   ├── models/
│   │   └── data_models.py           # Existing Pydantic models
│   ├── services/
│   │   ├── clickhouse_service.py    # Existing ClickHouse integration
│   │   └── redis_cache.py           # Existing Redis cache service
│   └── utils/
│       └── timestamp_aligner.py     # Existing timestamp utilities
└── docs/
    └── api/
        └── openapi.yaml             # OpenAPI specification
```

## 🔧 Installation & Setup

### 1. Environment Setup

Copy the environment template and configure:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```bash
# ClickHouse
CLICKHOUSE_HOST=your-clickhouse-host
CLICKHOUSE_PASSWORD=your-password
CLICKHOUSE_DATABASE=quantx

# Redis
REDIS_HOST=your-redis-host
REDIS_PASSWORD=your-password

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Server

#### Development Mode
```bash
python main.py
```

#### Production Mode
```bash
python start_server.py
```

#### Docker Deployment
```bash
# Single container
docker build -t labels-api .
docker run -p 8000:8000 labels-api

# Full stack with dependencies
docker-compose up -d
```

## 📊 API Endpoints

### Label Computation
- `POST /v1/labels/compute` - Real-time label computation (<100ms)
- `GET /v1/labels` - Query computed labels with filtering
- `GET /v1/labels/{instrument}/{granularity}/{timestamp}` - Get specific labels

### Batch Processing  
- `POST /v1/batch/backfill` - Start batch backfill operation
- `GET /v1/batch/jobs/{job_id}` - Monitor batch job progress
- `GET /v1/batch/jobs` - List batch jobs with filtering

### Health & Monitoring
- `GET /v1/health` - Comprehensive system health check
- `GET /v1/health/ready` - Kubernetes readiness probe
- `GET /v1/health/live` - Kubernetes liveness probe
- `GET /v1/metrics` - System metrics (JSON/Prometheus)
- `GET /metrics` - Prometheus metrics endpoint

### Cache Management
- `GET /v1/cache/stats` - Cache performance statistics
- `POST /v1/cache/warm` - Pre-warm cache with recent data
- `DELETE /v1/cache/invalidate` - Clear cache entries

## 🔐 Authentication

The API supports two authentication methods:

### API Key Authentication
```bash
curl -H "X-API-Key: label_your_api_key" http://localhost:8000/v1/labels/compute
```

### JWT Token Authentication  
```bash
curl -H "Authorization: Bearer your_jwt_token" http://localhost:8000/v1/labels/compute
```

## 💡 Enhanced Triple Barrier (Label 11.a)

The core label computation implements Label 11.a with advanced features:

### Features
- **Dynamic barrier sizing**: 2x ATR-based volatility scaling
- **S/R level integration**: Adjusts barriers based on active levels
- **Multi-timeframe paths**: Uses lower granularity for accurate checking
- **Path-dependent calculation**: Prevents look-ahead bias

### Example Request
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
  "label_types": ["enhanced_triple_barrier"],
  "options": {
    "horizon_periods": 6,
    "use_cache": true
  }
}
```

### Example Response
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
      "level_adjusted": true,
      "nearest_support": 1.0920,
      "nearest_resistance": 1.0990
    }
  },
  "computation_time_ms": 45,
  "cache_hit": false,
  "version": "1.0.0"
}
```

## 📈 Performance Monitoring

### Key Metrics
- **Computation latency**: p50, p95, p99 percentiles
- **Cache performance**: Hit rates, memory usage, evictions  
- **Throughput**: Requests/second, labels computed/hour
- **System health**: Database connections, error rates

### Prometheus Integration
```bash
# Metrics endpoint
curl http://localhost:8000/metrics

# Sample metrics
label_computation_duration_seconds 0.045
cache_hit_rate 0.96
http_requests_per_second 125.5
```

### Grafana Dashboards
Access Grafana at `http://localhost:3000` (admin/grafana_pass) when using docker-compose.

## 🏭 Production Deployment

### Docker Compose (Recommended)
```bash
# Full production stack
docker-compose up -d

# Services included:
# - labels-api (port 8000)
# - clickhouse (ports 8123, 9000) 
# - redis (port 6379)
# - prometheus (port 9090)
# - grafana (port 3000)
```

### Kubernetes Deployment
The application includes:
- Health check endpoints for readiness/liveness probes
- Graceful shutdown handling
- Resource limits and requests
- ConfigMaps for environment variables

### Load Balancing
- Health endpoint: `GET /v1/health` returns 503 if unhealthy
- Ready endpoint: `GET /v1/health/ready` for traffic routing
- Supports horizontal scaling with shared Redis cache

## 🚦 Rate Limiting

Default rate limits per endpoint:
- `/v1/labels/compute`: 1000 requests/hour
- `/v1/batch/backfill`: 10 requests/hour  
- `/v1/labels`: 10,000 requests/hour
- Default: 5,000 requests/hour

Customize in `src/api/middleware.py` `RateLimitingMiddleware.RATE_LIMITS`.

## 🔍 Logging & Tracing

### Request Tracing
Every request gets a trace ID:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input data", 
    "trace_id": "req_abc123def456"
  }
}
```

### Log Format
```
2024-01-10T15:30:00Z - labels.compute - INFO - Label computation completed: EURUSD H4 in 45ms [trace_id=req_abc123]
```

## 🧪 Testing

### Manual Testing
```bash
# Health check
curl http://localhost:8000/v1/health

# Compute labels
curl -X POST http://localhost:8000/v1/labels/compute \
  -H "Content-Type: application/json" \
  -H "X-API-Key: label_demo_key" \
  -d @docs/api/examples/requests/compute-labels.json
```

### Integration Tests
```bash
pytest tests/integration/
```

### Load Testing
```bash
# K6 performance tests
k6 run tests/k6/baseline.js
```

## 🚨 Error Handling

### Standard Error Format
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
- `200` - Success
- `202` - Accepted (async operations)
- `400` - Bad Request
- `401` - Unauthorized  
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Error
- `503` - Service Unavailable

## 🔧 Configuration

### Environment Variables
See `.env.example` for all available configuration options.

### Key Settings
- `MAX_INCREMENTAL_LATENCY_MS=100` - Target latency SLA
- `TARGET_CACHE_HIT_RATE=0.95` - Target cache hit rate
- `BATCH_CHUNK_SIZE=10000` - Batch processing chunk size
- `PARALLEL_WORKERS=8` - Parallel processing workers

## 📝 Development

### Adding New Endpoints
1. Define schemas in `src/api/schemas.py`
2. Implement router in `src/api/routers/`
3. Add to `main.py` with `app.include_router()`
4. Update OpenAPI spec in `docs/api/openapi.yaml`

### Adding New Label Types
1. Extend `LabelSet` model in `src/models/data_models.py`
2. Implement computation in `src/core/label_computation.py`
3. Add response schema in `src/api/schemas.py`
4. Update API documentation

## 📚 API Documentation

### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs` (development only)
- ReDoc: `http://localhost:8000/redoc` (development only)
- OpenAPI spec: `/openapi.json`

### Examples
See `docs/api/examples/` for:
- Request payloads
- Response formats  
- Postman collection
- cURL commands

## 🤝 Support

For issues and questions:
- Check health endpoint: `GET /v1/health`
- Review logs in `logs/app.log`
- Monitor metrics at `GET /v1/metrics`
- Use trace IDs for debugging

---

**Built with FastAPI** - High performance, easy to use, fast to code, ready for production.