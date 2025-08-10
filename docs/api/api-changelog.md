# API Changelog

## Version 1.0.0 - Initial Release (2024-01-10)

### 🆕 New Endpoints

#### Label Computation
- `POST /v1/labels/compute` - Compute labels for single candle (incremental mode)
- `GET /v1/labels` - Query computed labels with pagination and filtering
- `GET /v1/labels/{instrument_id}/{granularity}/{timestamp}` - Get labels for specific candle

#### Batch Operations
- `POST /v1/batch/backfill` - Start batch backfill operation
- `GET /v1/batch/jobs/{job_id}` - Get batch job status
- `GET /v1/batch/jobs` - List batch jobs with filtering

#### Health & Monitoring
- `GET /v1/health` - Comprehensive health check
- `GET /v1/health/ready` - Kubernetes readiness probe
- `GET /v1/health/live` - Kubernetes liveness probe
- `GET /v1/metrics` - System metrics (JSON/Prometheus formats)

#### Cache Management
- `GET /v1/cache/stats` - Cache performance statistics
- `POST /v1/cache/warm` - Pre-warm cache for instrument
- `DELETE /v1/cache/invalidate` - Invalidate cache entries

### 📝 Features

#### Authentication & Authorization
- Bearer token authentication (JWT)
- API key authentication for service-to-service
- Rate limiting with proper headers

#### Label Types Supported
- Enhanced Triple Barrier with S/R level adjustment (Label 11.a)
- Volatility-scaled returns
- Maximum Favorable/Adverse Excursion (MFE/MAE)
- Return quantile buckets
- Forward returns

#### Multi-timeframe Alignment
- Automatic alignment to granularity boundaries
- H4 → H1 path data for horizon calculations
- D → H4 path data for horizon calculations
- W → D path data for horizon calculations

#### Performance Optimizations
- Redis caching with configurable TTL
- Bulk operations for batch processing
- Parallel computation support
- Query optimization with proper indexing

#### Error Handling
- Consistent error response format
- Detailed validation errors
- Request tracing support
- Proper HTTP status codes

#### Pagination & Filtering
- Cursor-based pagination for large datasets
- Multi-field filtering support
- Sorting options
- HATEOAS navigation links

### 📊 Performance SLAs

| Operation | P50 | P95 | P99 | Max |
|-----------|-----|-----|-----|-----|
| Incremental compute | 20ms | 50ms | 100ms | 500ms |
| Cache hit read | 1ms | 5ms | 10ms | 50ms |
| Batch (per 1k candles) | 100ms | 500ms | 1s | 5s |

### 🔧 Rate Limits

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Label compute | 1000 req/hour | Rolling |
| Queries | 5000 req/hour | Rolling |
| Batch operations | 100 req/hour | Rolling |
| Health checks | Unlimited | - |

### 📈 Monitoring

#### Metrics Exposed
- Computation latency percentiles
- Cache hit rates by level
- Error rates by endpoint
- Active batch job counts
- Throughput metrics

#### Health Checks
- ClickHouse connectivity
- Redis connectivity and memory usage
- Cache performance thresholds
- Average latency monitoring

### 🚀 Deployment Notes

- OpenAPI 3.0 specification available
- Pydantic models for Python validation
- TypeScript types generation ready
- Postman collection included
- Contract tests included
- Docker health check support

---

## Migration Guide

This is the initial release - no migration required.

## Breaking Changes

None - initial release.

## Security Updates

- JWT token validation
- API key authentication
- Rate limiting protection
- Input validation and sanitization

## Known Issues

None at release time.

## Next Release Preview (v1.1.0)

Planned features:
- WebSocket support for real-time label streaming
- GraphQL endpoint for flexible queries
- Additional label types (Labels 2, 6, 7, 9, 10)
- Enhanced caching strategies
- Performance optimizations

---

## Support & Documentation

- OpenAPI Specification: `/docs/api/openapi.yaml`
- Postman Collection: `/docs/api/postman-collection.json`
- Example Requests: `/docs/api/examples/requests/`
- Example Responses: `/docs/api/examples/responses/`

For issues or questions, contact: support@labelcompute.com