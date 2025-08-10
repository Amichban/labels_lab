# Production-Ready Batch Backfill Pipeline

A high-performance batch processing system for label computation backfill operations, designed to process 1M+ candles per minute with comprehensive monitoring, error handling, and recovery mechanisms.

## 🚀 Features

### Core Capabilities
- **High Throughput**: Target 1M+ candles/minute processing speed
- **Parallel Processing**: ProcessPoolExecutor with dynamic worker scaling
- **Intelligent Chunking**: Configurable 10k candles per chunk (optimized for performance)
- **Redis-Based Tracking**: Real-time progress tracking and job state management
- **Graceful Retries**: Exponential backoff with circuit breaker patterns
- **CLI Interface**: Comprehensive command-line tools for job management

### Advanced Features
- **Dynamic Scaling**: Automatic worker pool adjustment based on load
- **Circuit Breakers**: Dependency failure protection and recovery
- **Error Categorization**: Intelligent error handling with appropriate retry strategies
- **Real-time Metrics**: Comprehensive performance monitoring and alerting
- **ETA Calculations**: Predictive completion time estimates with confidence scoring
- **Dead Letter Queue**: Permanent failure tracking and recovery procedures

### Production Ready
- **Comprehensive Testing**: Full integration test suite with performance validation
- **Monitoring Integration**: Prometheus metrics export and health checks
- **Graceful Shutdown**: Clean job termination and resource cleanup
- **Resource Management**: Memory-aware processing with leak detection
- **Configuration Management**: Environment-based settings with validation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  REST API       │    │  Job Scheduler  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────────┐
                    │   BatchBackfillService      │
                    │   - Job lifecycle mgmt      │
                    │   - Progress tracking       │
                    │   - Status monitoring       │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   BatchWorkerPool           │
                    │   - Dynamic scaling         │
                    │   - Task distribution       │
                    │   - Load balancing          │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼───────┐    ┌─────────▼───────┐    ┌─────────▼───────┐
│ Error Handler   │    │ Metrics Service │    │ Label Engine    │
│ - Categorization│    │ - Performance   │    │ - Computation   │
│ - Retry logic   │    │ - Alerting      │    │ - Caching       │
│ - Circuit breaker│   │ - Prometheus    │    │ - Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Storage Layer             │
                    │   ┌─────────┐ ┌─────────┐  │
                    │   │  Redis  │ │ClickHouse│  │
                    │   │- State  │ │- Data    │  │
                    │   │- Cache  │ │- Results │  │
                    │   └─────────┘ └─────────┘  │
                    └─────────────────────────────┘
```

## 📦 Installation

### Prerequisites
- Python 3.8+
- Redis 6.0+
- ClickHouse 21.8+
- Minimum 8GB RAM (16GB recommended for production)
- Multi-core CPU (8+ cores recommended)

### Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your Redis and ClickHouse configurations
   ```

3. **Initialize System**
   ```bash
   python scripts/setup_batch_processing.py
   ```

4. **Verify Installation**
   ```bash
   python -m src.cli.batch_cli metrics
   ```

## 🎯 Quick Start

### Start a Batch Job

```bash
# Basic usage
python -m src.cli.batch_cli start \
  --instrument EURUSD \
  --granularity H4 \
  --start-date 2024-01-01 \
  --end-date 2024-01-31

# Advanced configuration
python -m src.cli.batch_cli start \
  --instrument GBPUSD \
  --granularity H1 \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --labels enhanced_triple_barrier vol_scaled_return mfe_mae \
  --chunk-size 10000 \
  --workers 12 \
  --priority high \
  --force
```

### Monitor Progress

```bash
# Check job status
python -m src.cli.batch_cli status JOB_ID

# Real-time monitoring
python -m src.cli.batch_cli monitor JOB_ID --refresh 5

# List all jobs
python -m src.cli.batch_cli status --format table
```

### Job Control

```bash
# Pause a running job
python -m src.cli.batch_cli pause JOB_ID

# Resume a paused job
python -m src.cli.batch_cli resume JOB_ID

# Cancel a job
python -m src.cli.batch_cli cancel JOB_ID
```

### System Monitoring

```bash
# System metrics
python -m src.cli.batch_cli metrics --format json

# Cleanup old jobs
python -m src.cli.batch_cli cleanup --older-than 7 --status completed
```

## 📊 Performance Optimization

### Throughput Tuning

The system is optimized for high throughput with several configurable parameters:

```python
# Optimal settings for different scenarios

# High-volume historical backfill (1M+ candles)
BatchBackfillService(
    max_workers=16,           # Scale with CPU cores
    chunk_size=10000,         # Optimal for memory usage
    max_retries=3,
    redis_ttl=86400 * 7       # 7 day retention
)

# Real-time processing (low latency)
BatchBackfillService(
    max_workers=8,
    chunk_size=1000,          # Smaller chunks for faster response
    max_retries=5,            # More retries for stability
    redis_ttl=3600            # 1 hour retention
)

# Resource-constrained environment
BatchBackfillService(
    max_workers=4,
    chunk_size=5000,
    max_retries=2,
    redis_ttl=3600 * 6        # 6 hour retention
)
```

### Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Throughput | 1M+ candles/min | 800k-1.2M/min |
| Chunk Processing | <1s per 10k candles | 200-800ms |
| Error Rate | <5% | 1-3% |
| Cache Hit Rate | >70% | 75-85% |
| Memory Usage | <2GB | 1-1.5GB |

### Monitoring Dashboards

Key metrics to monitor in production:

```bash
# Get current performance
python -m src.cli.batch_cli metrics

# Prometheus metrics endpoint (if enabled)
curl http://localhost:8000/metrics

# Redis metrics
redis-cli --stat

# System resources
htop
```

## 🔧 Configuration

### Environment Variables

```bash
# Core Settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=9000
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DATABASE=quantx

# Performance Tuning
BATCH_MAX_WORKERS=8
BATCH_CHUNK_SIZE=10000
BATCH_MAX_RETRIES=3
CACHE_TTL_SECONDS=3600

# Monitoring
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
ALERT_WEBHOOKS=
```

### Worker Pool Configuration

```python
# Custom worker pool settings
worker_pool = BatchWorkerPool(
    min_workers=2,              # Minimum active workers
    max_workers=16,             # Maximum workers (CPU bound)
    initial_workers=8,          # Starting worker count
    max_queue_size=10000,       # Task queue limit
    scale_up_threshold=0.8,     # Scale up when 80% queue full
    scale_down_threshold=0.3,   # Scale down when <30% queue used
    worker_timeout=300,         # 5-minute task timeout
    health_check_interval=30    # Health check every 30s
)
```

## 🚨 Error Handling

The system includes comprehensive error handling with intelligent categorization:

### Error Categories

1. **Transient Errors** - Network timeouts, temporary resource issues
   - Retry: Yes (5 attempts with exponential backoff)
   - Examples: Connection timeouts, temporary service unavailability

2. **Resource Errors** - Memory, CPU, disk space issues  
   - Retry: Limited (3 attempts with linear backoff)
   - Examples: Out of memory, disk full

3. **Dependency Errors** - External service failures
   - Retry: Yes (5 attempts with jittered backoff)
   - Circuit breaker protection enabled
   - Examples: ClickHouse downtime, Redis connection issues

4. **Data Errors** - Invalid data, validation failures
   - Retry: No (data issues require manual intervention)
   - Examples: Malformed candles, invalid instruments

5. **Persistent Errors** - Configuration, permission issues
   - Retry: No (system configuration required)
   - Examples: Missing permissions, invalid credentials

### Circuit Breaker Pattern

```python
# Circuit breaker states for dependencies
circuit_breakers = {
    'clickhouse': {
        'failure_threshold': 5,      # Open after 5 failures
        'timeout_duration': 60,      # Stay open for 60 seconds
        'state': 'closed'            # closed/open/half_open
    },
    'redis': {
        'failure_threshold': 3,
        'timeout_duration': 30,
        'state': 'closed'
    }
}
```

### Dead Letter Queue

Failed tasks are automatically moved to a dead letter queue for later analysis:

```bash
# View permanently failed tasks
python -c "
from src.services.batch_error_handler import batch_error_handler
failed = batch_error_handler.get_failed_tasks()
for task in failed:
    print(f'{task.job_id}/{task.task_id}: {task.error_message}')
"
```

## 📈 Monitoring & Alerting

### Key Metrics

The system exports comprehensive metrics for monitoring:

```python
# Performance Metrics
- throughput_candles_per_minute
- chunk_processing_time_ms  
- error_rate_percent
- cache_hit_rate_percent

# System Metrics
- active_workers_count
- queue_depth
- cpu_usage_percent
- memory_usage_mb

# Business Metrics
- labels_computed_total
- unique_instruments_processed
- avg_batch_throughput
```

### Health Checks

```bash
# System health endpoint
curl http://localhost:8000/health

# Detailed health information
python -m src.cli.batch_cli metrics --format json | jq '.system_health'
```

### Alert Conditions

```python
# Default alert thresholds
alerts = {
    'error_rate_threshold': 0.05,        # 5% error rate
    'low_throughput_threshold': 500000,  # 500k candles/min
    'high_latency_threshold': 1000,      # 1 second per chunk
    'circuit_breaker_open': True,        # Any circuit breaker open
    'queue_backlog_threshold': 5000,     # 5k tasks in queue
}
```

## 🧪 Testing

### Running Tests

```bash
# Full test suite
pytest tests/ -v

# Integration tests only
pytest tests/integration/ -v

# Performance benchmarks
pytest tests/integration/test_batch_processing_pipeline.py::TestPerformanceBenchmarks -v

# Specific test coverage
pytest tests/integration/test_batch_processing_pipeline.py::TestBatchProcessingPipeline::test_end_to_end_batch_job -v
```

### Performance Testing

```bash
# Load test with synthetic data
python tests/integration/test_batch_processing_pipeline.py

# Memory leak detection
pytest tests/integration/ -k "memory" --tb=short

# Throughput validation
pytest tests/integration/ -k "performance" -s
```

### Test Coverage

The test suite includes:
- ✅ End-to-end pipeline testing
- ✅ Performance validation (1M+ candles/minute)
- ✅ Error handling and recovery scenarios
- ✅ Parallel processing validation
- ✅ Redis progress tracking
- ✅ CLI command integration
- ✅ Metrics and monitoring
- ✅ Worker pool scaling
- ✅ Circuit breaker patterns
- ✅ Memory leak detection

## 🚀 Production Deployment

### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Setup batch processing
RUN python scripts/setup_batch_processing.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  batch-processor:
    build: .
    environment:
      - REDIS_HOST=redis
      - CLICKHOUSE_HOST=clickhouse
      - BATCH_MAX_WORKERS=16
    depends_on:
      - redis
      - clickhouse
    
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      
volumes:
  redis_data:
  clickhouse_data:
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: batch-processor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: batch-processor
  template:
    metadata:
      labels:
        app: batch-processor
    spec:
      containers:
      - name: batch-processor
        image: batch-processor:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: CLICKHOUSE_HOST
          value: "clickhouse-service"
        - name: BATCH_MAX_WORKERS
          value: "16"
```

## 🔧 Troubleshooting

### Common Issues

**1. Low Throughput Performance**
```bash
# Check system resources
python -m src.cli.batch_cli metrics
htop

# Increase workers (if CPU available)
export BATCH_MAX_WORKERS=16

# Optimize chunk size
export BATCH_CHUNK_SIZE=15000

# Check cache hit rate
redis-cli info stats
```

**2. High Error Rates**
```bash
# View recent errors
python -c "
from src.services.batch_error_handler import batch_error_handler
stats = batch_error_handler.get_error_statistics(24)
print(stats)
"

# Check circuit breaker status
python -c "
from src.services.batch_error_handler import batch_error_handler
for name, cb in batch_error_handler.circuit_breakers.items():
    print(f'{name}: {cb.state} ({cb.failure_count} failures)')
"
```

**3. Memory Issues**
```bash
# Monitor memory usage
python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"

# Reduce chunk size
export BATCH_CHUNK_SIZE=5000

# Limit workers
export BATCH_MAX_WORKERS=4
```

**4. Redis Connection Issues**
```bash
# Test Redis connectivity
python -c "
from src.services.redis_cache import redis_cache
print('Redis OK' if redis_cache.check_connection() else 'Redis FAILED')
"

# Check Redis memory usage
redis-cli info memory
```

**5. ClickHouse Performance Issues**
```bash
# Test ClickHouse connectivity
python -c "
from src.services.clickhouse_service import clickhouse_service
print('ClickHouse OK' if clickhouse_service.check_connection() else 'ClickHouse FAILED')
"

# Check running queries
clickhouse-client --query="SELECT * FROM system.processes"
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
python -m src.cli.batch_cli start --instrument EURUSD --granularity H4 --verbose
```

### Performance Profiling

```python
# Profile a batch job
import cProfile
import pstats

# Profile job execution
profiler = cProfile.Profile()
profiler.enable()

# Run job
result = await batch_service.execute_job(job_id)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## 📚 API Reference

### BatchBackfillService

```python
class BatchBackfillService:
    async def start_backfill_job(
        instrument_id: str,
        granularity: str, 
        start_date: datetime,
        end_date: datetime,
        label_types: List[str],
        force_recompute: bool = False,
        priority: str = "normal"
    ) -> str
    
    async def execute_job(job_id: str) -> Dict[str, Any]
    async def pause_job(job_id: str) -> bool
    async def resume_job(job_id: str) -> bool  
    async def cancel_job(job_id: str) -> bool
    
    def get_job_status(job_id: str) -> Optional[Dict[str, Any]]
    def list_jobs(status_filter: str = None) -> List[Dict[str, Any]]
```

### CLI Commands

```bash
# Job Management
batch_cli start [OPTIONS]           # Start new batch job
batch_cli execute JOB_ID            # Execute pending job
batch_cli status [JOB_ID]           # Show job status/list jobs
batch_cli monitor JOB_ID [OPTIONS]  # Monitor job progress
batch_cli pause JOB_ID              # Pause running job
batch_cli resume JOB_ID             # Resume paused job  
batch_cli cancel JOB_ID             # Cancel job

# System Management
batch_cli metrics [OPTIONS]         # Show system metrics
batch_cli cleanup [OPTIONS]         # Cleanup old jobs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/ -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd batch-processing-pipeline

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v

# Start development environment
python scripts/setup_batch_processing.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

- 📖 Documentation: [docs/](docs/)
- 🐛 Bug Reports: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)
- 📧 Email: support@yourcompany.com

---

**Built with ❤️ for high-performance quantitative trading systems**