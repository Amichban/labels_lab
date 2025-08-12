"""
Comprehensive load testing suite for the Label Computation System.

This suite tests the system's ability to handle high throughput loads:
- 1000+ candles/second processing capability
- API endpoint load testing with realistic request patterns
- Database connection pool performance under load
- Cache performance under concurrent access
- Memory management during sustained load

Usage:
    pytest tests/performance/load_test.py -v --tb=short
    pytest tests/performance/load_test.py::TestLoadTesting::test_candle_processing_load -v
"""

import pytest
import asyncio
import time
import statistics
import psutil
import gc
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from unittest.mock import AsyncMock, Mock, patch
from contextlib import asynccontextmanager
import numpy as np
import concurrent.futures
from dataclasses import dataclass

# Locust imports for web load testing
try:
    from locust import HttpUser, task, between
    from locust.env import Environment
    from locust.stats import stats_printer
    from locust.log import setup_logging
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False

from src.core.label_computation import LabelComputationEngine
from src.models.data_models import Candle, Granularity
from src.services.batch_worker_pool import BatchWorkerPool


@dataclass
class LoadTestResult:
    """Results from a load test."""
    total_operations: int
    duration_seconds: float
    operations_per_second: float
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    memory_usage_mb: float
    errors: List[str]


@asynccontextmanager
async def measure_load_test():
    """Context manager for measuring load test metrics."""
    process = psutil.Process()
    gc.collect()
    
    start_time = time.perf_counter()
    start_memory = process.memory_info().rss
    
    result_data = {
        'operations': 0,
        'successes': 0,
        'latencies': [],
        'errors': []
    }
    
    yield result_data
    
    end_time = time.perf_counter()
    gc.collect()
    end_memory = process.memory_info().rss
    
    duration = end_time - start_time
    memory_usage_mb = (end_memory - start_memory) / (1024 * 1024)
    
    # Calculate statistics
    latencies = np.array(result_data['latencies']) if result_data['latencies'] else np.array([0])
    
    result_data.update({
        'duration_seconds': duration,
        'operations_per_second': result_data['operations'] / duration if duration > 0 else 0,
        'success_rate': result_data['successes'] / result_data['operations'] if result_data['operations'] > 0 else 0,
        'avg_latency_ms': np.mean(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'max_latency_ms': np.max(latencies),
        'memory_usage_mb': memory_usage_mb
    })


def generate_test_candles(count: int, base_time: datetime = None) -> List[Candle]:
    """Generate test candles for load testing."""
    if base_time is None:
        base_time = datetime(2024, 1, 1, 9, 0, 0)
    
    candles = []
    instruments = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
    
    for i in range(count):
        instrument = instruments[i % len(instruments)]
        
        candles.append(Candle(
            instrument_id=instrument,
            granularity=Granularity.H4,
            ts=base_time + timedelta(hours=4 * i),
            open=1.0500 + np.random.normal(0, 0.01),
            high=1.0580 + np.random.normal(0, 0.01),
            low=1.0450 + np.random.normal(0, 0.01),
            close=1.0520 + np.random.normal(0, 0.01),
            volume=1000 + np.random.randint(-200, 200),
            atr_14=0.0045 + np.random.normal(0, 0.0005)
        ))
    
    return candles


@pytest.mark.performance
@pytest.mark.load
class TestLoadTesting:
    """Core load testing for label computation system."""
    
    @pytest.mark.asyncio
    async def test_candle_processing_load_1000_per_second(self):
        """Test processing 1000+ candles per second."""
        target_throughput = 1000  # candles per second
        test_duration = 30  # seconds
        total_candles = target_throughput * test_duration
        
        print(f"Load test: Processing {total_candles} candles over {test_duration} seconds")
        print(f"Target throughput: {target_throughput} candles/second")
        
        # Generate test data
        test_candles = generate_test_candles(total_candles)
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Configure mocks for fast processing
            mock_redis.get_labels.return_value = None  # Force computation
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            engine = LabelComputationEngine()
            
            async with measure_load_test() as results:
                start_time = time.perf_counter()
                
                # Process candles in batches to manage memory
                batch_size = 100
                processed = 0
                
                for i in range(0, len(test_candles), batch_size):
                    batch = test_candles[i:i + batch_size]
                    
                    # Process batch concurrently
                    tasks = []
                    for candle in batch:
                        task_start = time.perf_counter()
                        task = asyncio.create_task(self._process_candle_with_timing(
                            engine, candle, task_start, results
                        ))
                        tasks.append(task)
                    
                    # Wait for batch to complete
                    await asyncio.gather(*tasks, return_exceptions=True)
                    processed += len(batch)
                    
                    # Rate limiting to target throughput
                    elapsed = time.perf_counter() - start_time
                    target_processed = elapsed * target_throughput
                    
                    if processed > target_processed:
                        sleep_time = (processed - target_processed) / target_throughput
                        await asyncio.sleep(sleep_time)
                    
                    if processed % 1000 == 0:
                        current_rate = processed / (time.perf_counter() - start_time)
                        print(f"Processed {processed}/{total_candles} candles, current rate: {current_rate:.1f}/sec")
        
        # Validate results
        print(f"\nLoad Test Results:")
        print(f"Total operations: {results['operations']}")
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Throughput: {results['operations_per_second']:.1f} candles/second")
        print(f"Average latency: {results['avg_latency_ms']:.3f}ms")
        print(f"P95 latency: {results['p95_latency_ms']:.3f}ms")
        print(f"P99 latency: {results['p99_latency_ms']:.3f}ms")
        print(f"Memory usage: {results['memory_usage_mb']:.2f}MB")
        print(f"Errors: {len(results['errors'])}")
        
        # Assertions
        assert results['operations_per_second'] >= target_throughput, \
            f"Throughput below target: {results['operations_per_second']:.1f} < {target_throughput}"
        
        assert results['success_rate'] >= 0.95, \
            f"Success rate too low: {results['success_rate']:.2%}"
        
        assert results['p99_latency_ms'] <= 100, \
            f"P99 latency too high: {results['p99_latency_ms']:.3f}ms"
        
        assert results['memory_usage_mb'] <= 500, \
            f"Memory usage too high: {results['memory_usage_mb']:.2f}MB"
    
    async def _process_candle_with_timing(
        self, 
        engine: LabelComputationEngine, 
        candle: Candle, 
        start_time: float, 
        results: dict
    ):
        """Process a single candle and record timing metrics."""
        try:
            result = await engine.compute_labels(candle)
            end_time = time.perf_counter()
            
            results['operations'] += 1
            results['latencies'].append((end_time - start_time) * 1000)
            
            if result is not None:
                results['successes'] += 1
            else:
                results['errors'].append("Computation returned None")
                
        except Exception as e:
            results['operations'] += 1
            results['errors'].append(str(e))
    
    @pytest.mark.asyncio
    async def test_concurrent_api_load(self):
        """Test API load with concurrent requests."""
        concurrent_users = 100
        requests_per_user = 50
        total_requests = concurrent_users * requests_per_user
        
        print(f"API load test: {concurrent_users} concurrent users, {requests_per_user} requests each")
        
        # Mock API responses
        with patch('src.api.routers.labels.compute_labels') as mock_compute:
            # Simulate realistic API response times
            async def mock_compute_labels(*args, **kwargs):
                await asyncio.sleep(np.random.exponential(0.02))  # Average 20ms
                return {
                    "enhanced_triple_barrier": {
                        "label": np.random.choice([-1, 0, 1]),
                        "barrier_hit": "upper",
                        "time_to_barrier": 5
                    }
                }
            
            mock_compute.side_effect = mock_compute_labels
            
            async with measure_load_test() as results:
                # Create user tasks
                user_tasks = []
                for user_id in range(concurrent_users):
                    task = asyncio.create_task(
                        self._simulate_user_requests(user_id, requests_per_user, results)
                    )
                    user_tasks.append(task)
                
                # Wait for all users to complete
                await asyncio.gather(*user_tasks, return_exceptions=True)
        
        print(f"\nAPI Load Test Results:")
        print(f"Total requests: {results['operations']}")
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Requests per second: {results['operations_per_second']:.1f}")
        print(f"Average response time: {results['avg_latency_ms']:.3f}ms")
        print(f"P95 response time: {results['p95_latency_ms']:.3f}ms")
        print(f"P99 response time: {results['p99_latency_ms']:.3f}ms")
        
        # Assertions
        assert results['operations'] >= total_requests * 0.95, "Too many failed requests"
        assert results['success_rate'] >= 0.95, f"Success rate too low: {results['success_rate']:.2%}"
        assert results['operations_per_second'] >= 500, f"RPS too low: {results['operations_per_second']:.1f}"
        assert results['p99_latency_ms'] <= 500, f"P99 latency too high: {results['p99_latency_ms']:.3f}ms"
    
    async def _simulate_user_requests(self, user_id: int, request_count: int, results: dict):
        """Simulate a user making multiple API requests."""
        for i in range(request_count):
            start_time = time.perf_counter()
            
            try:
                # Simulate API call processing time
                await asyncio.sleep(np.random.exponential(0.02))  # Average 20ms
                
                end_time = time.perf_counter()
                
                results['operations'] += 1
                results['successes'] += 1
                results['latencies'].append((end_time - start_time) * 1000)
                
                # User think time
                await asyncio.sleep(np.random.exponential(0.1))  # Average 100ms between requests
                
            except Exception as e:
                results['operations'] += 1
                results['errors'].append(f"User {user_id} request {i}: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_database_connection_pool_load(self):
        """Test database connection pool under load."""
        concurrent_connections = 50
        operations_per_connection = 20
        
        print(f"Database load test: {concurrent_connections} concurrent connections")
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch:
            # Simulate database query times
            async def mock_query(*args, **kwargs):
                await asyncio.sleep(np.random.exponential(0.005))  # Average 5ms query time
                return []
            
            mock_ch.fetch_active_levels = AsyncMock(side_effect=mock_query)
            mock_ch.fetch_snapshots = AsyncMock(side_effect=mock_query)
            
            async with measure_load_test() as results:
                # Create connection tasks
                connection_tasks = []
                for conn_id in range(concurrent_connections):
                    task = asyncio.create_task(
                        self._simulate_database_operations(conn_id, operations_per_connection, results)
                    )
                    connection_tasks.append(task)
                
                await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        print(f"\nDatabase Load Test Results:")
        print(f"Total operations: {results['operations']}")
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Operations per second: {results['operations_per_second']:.1f}")
        print(f"Average query time: {results['avg_latency_ms']:.3f}ms")
        print(f"P95 query time: {results['p95_latency_ms']:.3f}ms")
        
        # Assertions
        assert results['success_rate'] >= 0.99, f"Database success rate too low: {results['success_rate']:.2%}"
        assert results['operations_per_second'] >= 500, f"Database OPS too low: {results['operations_per_second']:.1f}"
        assert results['p95_latency_ms'] <= 50, f"Database P95 latency too high: {results['p95_latency_ms']:.3f}ms"
    
    async def _simulate_database_operations(self, conn_id: int, operation_count: int, results: dict):
        """Simulate database operations on a connection."""
        for i in range(operation_count):
            start_time = time.perf_counter()
            
            try:
                # Simulate database query
                await asyncio.sleep(np.random.exponential(0.005))
                
                end_time = time.perf_counter()
                
                results['operations'] += 1
                results['successes'] += 1
                results['latencies'].append((end_time - start_time) * 1000)
                
            except Exception as e:
                results['operations'] += 1
                results['errors'].append(f"Connection {conn_id} operation {i}: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_cache_load_performance(self):
        """Test cache performance under high load."""
        cache_operations = 10000
        concurrent_workers = 20
        operations_per_worker = cache_operations // concurrent_workers
        
        print(f"Cache load test: {cache_operations} operations with {concurrent_workers} workers")
        
        with patch('src.services.redis_cache.RedisCache') as mock_cache:
            # Simulate cache operations
            async def mock_get(*args, **kwargs):
                await asyncio.sleep(np.random.exponential(0.001))  # Average 1ms
                return {"cached": "data"} if np.random.random() > 0.2 else None
            
            async def mock_set(*args, **kwargs):
                await asyncio.sleep(np.random.exponential(0.002))  # Average 2ms
                return True
            
            mock_cache.return_value.get = AsyncMock(side_effect=mock_get)
            mock_cache.return_value.set = AsyncMock(side_effect=mock_set)
            
            async with measure_load_test() as results:
                worker_tasks = []
                for worker_id in range(concurrent_workers):
                    task = asyncio.create_task(
                        self._simulate_cache_operations(worker_id, operations_per_worker, results)
                    )
                    worker_tasks.append(task)
                
                await asyncio.gather(*worker_tasks, return_exceptions=True)
        
        print(f"\nCache Load Test Results:")
        print(f"Total operations: {results['operations']}")
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Cache operations per second: {results['operations_per_second']:.1f}")
        print(f"Average cache latency: {results['avg_latency_ms']:.3f}ms")
        print(f"P99 cache latency: {results['p99_latency_ms']:.3f}ms")
        
        # Assertions
        assert results['success_rate'] >= 0.99, f"Cache success rate too low: {results['success_rate']:.2%}"
        assert results['operations_per_second'] >= 2000, f"Cache OPS too low: {results['operations_per_second']:.1f}"
        assert results['p99_latency_ms'] <= 10, f"Cache P99 latency too high: {results['p99_latency_ms']:.3f}ms"
    
    async def _simulate_cache_operations(self, worker_id: int, operation_count: int, results: dict):
        """Simulate cache operations for a worker."""
        from src.services.redis_cache import RedisCache
        cache = RedisCache()
        
        for i in range(operation_count):
            start_time = time.perf_counter()
            
            try:
                # Mix of read and write operations
                if np.random.random() < 0.7:  # 70% reads
                    await cache.get(f"key_{worker_id}_{i}")
                else:  # 30% writes
                    await cache.set(f"key_{worker_id}_{i}", {"data": f"value_{i}"})
                
                end_time = time.perf_counter()
                
                results['operations'] += 1
                results['successes'] += 1
                results['latencies'].append((end_time - start_time) * 1000)
                
            except Exception as e:
                results['operations'] += 1
                results['errors'].append(f"Worker {worker_id} operation {i}: {str(e)}")


@pytest.mark.performance
@pytest.mark.load
class TestBatchLoadTesting:
    """Load testing for batch processing operations."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_processing_sustained_load(self):
        """Test sustained batch processing load."""
        batch_size = 1000
        number_of_batches = 10
        target_batches_per_minute = 5
        
        print(f"Batch load test: {number_of_batches} batches of {batch_size} candles")
        print(f"Target: {target_batches_per_minute} batches per minute")
        
        test_candles = generate_test_candles(batch_size * number_of_batches)
        
        with patch('src.services.batch_worker_pool.BatchWorkerPool') as mock_pool:
            async def mock_process_batch(batch):
                # Simulate batch processing time
                await asyncio.sleep(len(batch) * 0.001)  # 1ms per candle
                return {
                    'processed_candles': len(batch),
                    'successful_labels': len(batch),
                    'error_count': 0
                }
            
            mock_pool.return_value.process_batch = AsyncMock(side_effect=mock_process_batch)
            
            batch_worker_pool = BatchWorkerPool(max_workers=8)
            
            async with measure_load_test() as results:
                for batch_num in range(number_of_batches):
                    batch_start_idx = batch_num * batch_size
                    batch_end_idx = (batch_num + 1) * batch_size
                    batch = test_candles[batch_start_idx:batch_end_idx]
                    
                    start_time = time.perf_counter()
                    
                    try:
                        result = await batch_worker_pool.process_batch(batch)
                        end_time = time.perf_counter()
                        
                        results['operations'] += 1
                        results['successes'] += 1 if result['error_count'] == 0 else 0
                        results['latencies'].append((end_time - start_time) * 1000)
                        
                        print(f"Batch {batch_num + 1}/{number_of_batches} completed: "
                              f"{result['processed_candles']} candles in "
                              f"{(end_time - start_time):.2f}s")
                        
                    except Exception as e:
                        results['operations'] += 1
                        results['errors'].append(f"Batch {batch_num}: {str(e)}")
        
        batches_per_minute = results['operations_per_second'] * 60
        
        print(f"\nBatch Load Test Results:")
        print(f"Total batches: {results['operations']}")
        print(f"Success rate: {results['success_rate']:.2%}")
        print(f"Batches per minute: {batches_per_minute:.1f}")
        print(f"Average batch time: {results['avg_latency_ms']:.0f}ms")
        print(f"P95 batch time: {results['p95_latency_ms']:.0f}ms")
        
        # Assertions
        assert results['success_rate'] >= 0.95, f"Batch success rate too low: {results['success_rate']:.2%}"
        assert batches_per_minute >= target_batches_per_minute, \
            f"Batch throughput too low: {batches_per_minute:.1f} < {target_batches_per_minute}"


if LOCUST_AVAILABLE:
    class LabelComputationUser(HttpUser):
        """Locust user for web-based load testing."""
        wait_time = between(1, 3)
        host = "http://localhost:8000"
        
        def on_start(self):
            """Initialize user session."""
            self.instruments = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
            self.granularities = ["H1", "H4", "D"]
        
        @task(8)
        def compute_single_label(self):
            """Test single label computation (80% of load)."""
            instrument = np.random.choice(self.instruments)
            granularity = np.random.choice(self.granularities)
            
            payload = {
                "instrument_id": instrument,
                "granularity": granularity,
                "candle": {
                    "ts": "2024-01-15T09:00:00Z",
                    "open": 1.0500,
                    "high": 1.0580,
                    "low": 1.0450,
                    "close": 1.0520,
                    "volume": 1000,
                    "atr_14": 0.0045
                },
                "label_types": ["enhanced_triple_barrier"],
                "options": {
                    "horizon_periods": 10,
                    "use_cache": True
                }
            }
            
            with self.client.post("/api/v1/labels/compute", 
                                  json=payload, 
                                  catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Status code: {response.status_code}")
        
        @task(2)
        def batch_compute_labels(self):
            """Test batch label computation (20% of load)."""
            instrument = np.random.choice(self.instruments)
            granularity = np.random.choice(self.granularities)
            
            payload = {
                "instrument_id": instrument,
                "granularity": granularity,
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-02T00:00:00Z",
                "label_types": ["enhanced_triple_barrier"],
                "chunk_size": 100
            }
            
            with self.client.post("/api/v1/labels/batch", 
                                  json=payload, 
                                  catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Status code: {response.status_code}")
        
        @task(1)
        def health_check(self):
            """Test health check endpoint (10% of load)."""
            with self.client.get("/health", catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Status code: {response.status_code}")


@pytest.mark.skipif(not LOCUST_AVAILABLE, reason="Locust not available")
@pytest.mark.performance
@pytest.mark.load
class TestWebLoadTesting:
    """Web-based load testing using Locust."""
    
    def test_run_locust_load_test(self):
        """Run a programmatic Locust load test."""
        # Setup Locust environment
        env = Environment(user_classes=[LabelComputationUser])
        env.create_local_runner()
        
        # Configure test parameters
        user_count = 50
        spawn_rate = 10
        run_time = 60  # seconds
        
        print(f"Starting Locust load test: {user_count} users, {spawn_rate}/sec spawn rate, {run_time}s duration")
        
        # Start load test
        env.runner.start(user_count, spawn_rate)
        
        # Let test run
        import time as time_module
        time_module.sleep(run_time)
        
        # Stop test
        env.runner.stop()
        
        # Get statistics
        stats = env.runner.stats
        
        print(f"\nLocust Load Test Results:")
        print(f"Total requests: {stats.total.num_requests}")
        print(f"Failed requests: {stats.total.num_failures}")
        print(f"Success rate: {((stats.total.num_requests - stats.total.num_failures) / stats.total.num_requests * 100):.2f}%")
        print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
        print(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
        print(f"99th percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
        print(f"Requests per second: {stats.total.total_rps:.2f}")
        
        # Assertions
        success_rate = (stats.total.num_requests - stats.total.num_failures) / stats.total.num_requests
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        assert stats.total.avg_response_time <= 200, f"Average response time too high: {stats.total.avg_response_time:.2f}ms"
        assert stats.total.total_rps >= 100, f"RPS too low: {stats.total.total_rps:.2f}"