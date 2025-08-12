# Firestore Real-time Streaming System

This guide covers the real-time Firestore listener implementation for Issue #11, providing comprehensive market data streaming capabilities with advanced error handling, rate limiting, and monitoring.

## Architecture Overview

The streaming system consists of several key components:

### Core Components

1. **FirestoreListener** (`src/services/firestore_listener.py`)
   - Real-time document listener for Firestore collections
   - Filters for complete candles only (`complete=true`)
   - Automatic reconnection with exponential backoff
   - Dead letter queue for failed processing
   - Comprehensive metrics and monitoring

2. **StreamManager** (`src/services/stream_manager.py`)
   - Coordinates multiple concurrent streams
   - Priority-based stream management
   - Rate limiting and backpressure handling
   - Health monitoring and automatic recovery
   - Performance optimization

3. **DeadLetterQueue** (embedded in FirestoreListener)
   - Handles failed candle processing
   - Redis persistence for failed items
   - Automatic retry with configurable limits
   - Monitoring and alerting

## Data Structure

### Firestore Collection Structure
```
Collection: candles/{instrument}/{granularity}/data
Document ID: {timestamp_seconds}
Document Fields:
{
    "o": open_price,      // Open price
    "h": high_price,      // High price  
    "l": low_price,       // Low price
    "c": close_price,     // Close price
    "v": volume,          // Volume
    "ts": timestamp,      // Timestamp (DateTime or seconds)
    "bid": bid_price,     // Bid price
    "ask": ask_price,     // Ask price
    "complete": true/false // Completeness flag
}
```

### Supported Instruments & Granularities

**Default Instruments:**
- Major FX: EUR_USD, GBP_USD, USD_JPY, USD_CHF, AUD_USD
- Additional pairs can be configured via settings

**Granularities:**
- H1: Hourly candles
- H4: 4-hour candles
- D: Daily candles
- W: Weekly candles

## Configuration

### Environment Variables

#### Firestore Connection
```bash
# Required
GCP_PROJECT_ID=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json

# Optional - for emulator
FIRESTORE_EMULATOR_HOST=localhost:8080
```

#### Stream Configuration
```bash
# Instruments to stream (comma-separated)
STREAM_INSTRUMENTS=EUR_USD,GBP_USD,USD_JPY,USD_CHF,AUD_USD

# Granularities to stream (comma-separated)  
STREAM_GRANULARITIES=H1,H4

# Maximum concurrent streams
MAX_CONCURRENT_STREAMS=8

# Enable real-time streaming
ENABLE_REALTIME_STREAMING=true

# Processing rate limit (per minute)
STREAM_PROCESSING_RATE_LIMIT=1000

# Backpressure threshold
STREAM_BACKPRESSURE_THRESHOLD=10000
```

#### Retry Policies
```bash
# Firestore retry configuration
FIRESTORE_MAX_RETRY_ATTEMPTS=5
FIRESTORE_BASE_RETRY_DELAY=1.0
FIRESTORE_MAX_RETRY_DELAY=60.0

# Dead letter queue
DEAD_LETTER_QUEUE_MAX_SIZE=1000
DEAD_LETTER_QUEUE_RETRY_INTERVAL=300
```

#### Health Monitoring
```bash
# Health check interval (seconds)
STREAM_HEALTH_CHECK_INTERVAL=30

# Stream silence threshold (seconds)
STREAM_SILENCE_THRESHOLD=600

# Enable automatic recovery
ENABLE_STREAM_AUTO_RECOVERY=true

# Performance optimization
ENABLE_PERFORMANCE_OPTIMIZATION=true
PERFORMANCE_OPTIMIZATION_INTERVAL=300
```

### Python Configuration

```python
from config.settings import settings

# Get stream configuration
stream_config = settings.get_stream_configuration()
print(f"Instruments: {stream_config['instruments']}")
print(f"Granularities: {stream_config['granularities']}")

# Get Firestore configuration
firestore_config = settings.get_firestore_config()
print(f"Project ID: {firestore_config['project_id']}")
```

## Usage Examples

### Basic Usage

```python
import asyncio
from src.services.firestore_listener import firestore_listener
from src.services.stream_manager import stream_manager, StreamPriority

async def process_candle(candle):
    """Custom candle processing function"""
    print(f"Received: {candle.instrument_id} {candle.granularity} @ {candle.ts}")
    # Process candle for label computation, storage, etc.

async def main():
    # Add streams to manager
    await stream_manager.add_stream("EUR_USD", "H1", StreamPriority.HIGH)
    await stream_manager.add_stream("GBP_USD", "H1", StreamPriority.MEDIUM)
    
    # Add streams to listener with callbacks
    firestore_listener.add_stream("EUR_USD", "H1", callback=process_candle)
    firestore_listener.add_stream("GBP_USD", "H1", callback=process_candle)
    
    # Start all streams
    await stream_manager.start_all_streams()
    await firestore_listener.start_all_streams()
    
    # Keep running
    await asyncio.sleep(3600)  # Run for 1 hour
    
    # Graceful shutdown
    await stream_manager.shutdown()
    await firestore_listener.shutdown()

asyncio.run(main())
```

### Advanced Usage with Error Handling

```python
import asyncio
import logging
from src.services.firestore_listener import firestore_listener
from src.services.stream_manager import stream_manager, StreamPriority
from src.core.label_computation import computation_engine

logger = logging.getLogger(__name__)

class AdvancedCandleProcessor:
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
    
    async def process_candle(self, candle):
        """Advanced candle processing with error handling"""
        try:
            # Compute labels for the candle
            label_set = await computation_engine.compute_labels(
                candle=candle,
                horizon_periods=6,
                label_types=["enhanced_triple_barrier", "vol_scaled_return"],
                use_cache=True
            )
            
            # Store labels (implement your storage logic)
            # await self.store_labels(label_set)
            
            self.processed_count += 1
            
            if self.processed_count % 100 == 0:
                logger.info(f"Processed {self.processed_count} candles")
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error processing candle {candle.instrument_id} {candle.ts}: {e}")

async def main():
    processor = AdvancedCandleProcessor()
    
    try:
        # Setup streams for multiple instruments
        instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]
        granularities = ["H1", "H4"]
        
        for instrument in instruments:
            for granularity in granularities:
                # Add to stream manager with appropriate priority
                priority = StreamPriority.HIGH if instrument == "EUR_USD" else StreamPriority.MEDIUM
                await stream_manager.add_stream(instrument, granularity, priority)
                
                # Add to listener with processor callback
                firestore_listener.add_stream(
                    instrument, 
                    granularity, 
                    callback=processor.process_candle
                )
        
        # Start streaming
        logger.info("Starting streaming system...")
        await stream_manager.start_all_streams()
        await firestore_listener.start_all_streams()
        
        # Monitor and run
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            # Health check
            health = await firestore_listener.health_check()
            if health["overall_status"] not in ["healthy", "partial"]:
                logger.warning(f"System health: {health['overall_status']}")
            
            # Performance optimization
            await stream_manager.optimize_performance()
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await stream_manager.shutdown()
        await firestore_listener.shutdown()

asyncio.run(main())
```

## Stream Management

### Stream Priorities

The system supports four priority levels:

```python
from src.services.stream_manager import StreamPriority

# Priority levels (with rate limits)
StreamPriority.CRITICAL  # 500 docs/minute
StreamPriority.HIGH      # 200 docs/minute  
StreamPriority.MEDIUM    # 100 docs/minute
StreamPriority.LOW       # 50 docs/minute
```

### Rate Limiting

Rate limiting operates at multiple levels:

1. **Per-Stream Limits**: Based on priority level
2. **Global Limits**: System-wide processing capacity
3. **Backpressure**: Automatic throttling under high load

```python
# Check current rate limits
stream_status = stream_manager.get_stream_status("EUR_USD_H1")
print(f"Max rate: {stream_status['max_processing_rate']} docs/minute")
print(f"Current rate: {stream_status['current_processing_rate']} docs/minute")
```

### Health Monitoring

Comprehensive health monitoring includes:

- Stream connectivity status
- Processing rate monitoring
- Error rate tracking
- Silence detection (no data received)
- Automatic recovery triggers

```python
# Get detailed health information
health = await firestore_listener.health_check()
print(f"Overall Status: {health['overall_status']}")
print(f"Active Streams: {health['active_streams']}/{health['total_streams']}")

# Stream-specific health
for stream_id, stream_health in health['stream_health'].items():
    print(f"{stream_id}: {stream_health}")
```

## Error Handling & Recovery

### Dead Letter Queue

Failed candle processing is handled through a dead letter queue:

```python
# Check dead letter queue status
dlq_stats = firestore_listener.dead_letter_queue.get_stats()
print(f"Failed items in queue: {dlq_stats['current_size']}")
print(f"Total failures: {dlq_stats['total_failed']}")

# Retry failed items
retry_count = await firestore_listener.dead_letter_queue.retry_failed_items()
print(f"Retried {retry_count} items")
```

### Automatic Recovery

The system includes automatic recovery for:

- Connection failures (exponential backoff)
- Processing errors (dead letter queue)
- Stream silence (restart triggers)
- High error rates (automatic restart)

Recovery behavior can be configured:

```python
# Configure recovery settings
firestore_listener.max_reconnect_attempts = 10
firestore_listener.base_retry_delay = 1.0
firestore_listener.max_retry_delay = 60.0

stream_manager.auto_recovery_enabled = True
```

## Performance Optimization

### Automatic Optimization

The system includes automatic performance optimization:

```python
# Run optimization manually
result = await stream_manager.optimize_performance()
print(f"Optimizations applied: {result['optimizations_applied']}")

# Enable automatic optimization
settings.enable_performance_optimization = True
settings.performance_optimization_interval = 300  # 5 minutes
```

### Monitoring Metrics

Key metrics to monitor:

1. **Processing Rate**: Documents processed per second
2. **Success Rate**: Percentage of successful processing
3. **Error Rate**: Percentage of failed processing
4. **Latency**: Time from document change to processing
5. **Backpressure Events**: Number of throttling events

```python
# Get comprehensive metrics
metrics = firestore_listener.get_metrics()
print(f"Processing rate: {metrics['processing_rate_per_second']:.2f} docs/sec")
print(f"Success rate: {metrics['success_rate']:.1%}")

# Stream manager metrics
status = stream_manager.get_stream_status()
global_metrics = status["global_metrics"]
print(f"Total processed: {global_metrics['total_documents_processed']}")
print(f"Total errors: {global_metrics['total_errors']}")
```

## Testing

### Unit Tests

```bash
# Run Firestore listener tests
python -m pytest tests/unit/test_firestore_listener.py -v

# Run all streaming tests
python -m pytest tests/unit/test_firestore_listener.py tests/unit/test_stream_manager.py -v
```

### Integration Tests

```bash
# Run integration tests
python -m pytest tests/integration/test_stream_integration.py -v

# Run with coverage
python -m pytest tests/integration/test_stream_integration.py --cov=src/services --cov-report=html
```

### Demo Application

Run the comprehensive demo:

```bash
# Basic demo
python examples/firestore_streaming_demo.py

# Custom configuration
python examples/firestore_streaming_demo.py \
    --instruments EUR_USD,GBP_USD,USD_JPY \
    --granularities H1,H4 \
    --log-level DEBUG
```

## Deployment Considerations

### Production Settings

For production deployment:

1. **Resource Limits**: Configure appropriate CPU/memory limits
2. **Monitoring**: Set up Prometheus metrics collection  
3. **Alerting**: Configure alerts for system health
4. **Scaling**: Use horizontal scaling for high throughput
5. **Security**: Secure Firestore access with minimal permissions

### Cloud Run Deployment

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: firestore-streaming
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "1000m"
    spec:
      containers:
      - image: gcr.io/PROJECT-ID/firestore-streaming
        env:
        - name: GCP_PROJECT_ID
          value: "your-project-id"
        - name: ENABLE_REALTIME_STREAMING
          value: "true"
        - name: MAX_CONCURRENT_STREAMS
          value: "8"
        resources:
          limits:
            memory: "2Gi" 
            cpu: "1000m"
```

### Docker Configuration

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import asyncio; from src.services.firestore_listener import firestore_listener; print('healthy' if asyncio.run(firestore_listener.health_check())['overall_status'] in ['healthy', 'partial'] else exit(1))"

CMD ["python", "examples/firestore_streaming_demo.py"]
```

## Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   gcloud auth application-default login
   ```

2. **Connection Timeouts**
   - Check network connectivity to Firestore
   - Verify firewall rules allow Firestore traffic
   - Increase timeout settings

3. **High Error Rates**
   - Check dead letter queue for failed items
   - Review error logs for patterns
   - Verify document structure matches expected format

4. **Performance Issues**  
   - Monitor processing rates and adjust limits
   - Enable automatic optimization
   - Scale horizontally if needed

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.getLogger("src.services.firestore_listener").setLevel(logging.DEBUG)
logging.getLogger("src.services.stream_manager").setLevel(logging.DEBUG)
```

### Health Check Endpoints

For production monitoring, implement health check endpoints:

```python
from fastapi import FastAPI
from src.services.firestore_listener import firestore_listener

app = FastAPI()

@app.get("/health")
async def health_check():
    health = await firestore_listener.health_check()
    status_code = 200 if health["overall_status"] in ["healthy", "partial"] else 503
    return Response(content=health, status_code=status_code)
```

## Best Practices

1. **Graceful Shutdown**: Always use proper shutdown procedures
2. **Resource Management**: Monitor memory and CPU usage
3. **Error Handling**: Implement comprehensive error handling
4. **Monitoring**: Set up proper metrics and alerting
5. **Testing**: Test thoroughly with various failure scenarios
6. **Security**: Use least-privilege Firestore permissions
7. **Performance**: Monitor and optimize processing rates
8. **Documentation**: Keep configuration and processes documented

## Support

For issues or questions:

1. Check logs for error messages
2. Review configuration settings  
3. Run health checks to identify issues
4. Consult the troubleshooting guide
5. Check dead letter queue for failed processing

The system is designed for high reliability and performance in production trading environments.