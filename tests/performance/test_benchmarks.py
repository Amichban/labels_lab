"""
Performance benchmark tests for Enhanced Triple Barrier Label 11.a implementation.

These tests focus on:
- Cache hit/miss performance scenarios
- Throughput benchmarks for batch processing
- Memory usage analysis
- Latency measurements for real-time scenarios  
- Scalability testing with large datasets
- Database query optimization verification
"""

import pytest
import time
import asyncio
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from unittest.mock import AsyncMock, Mock, patch
from contextlib import contextmanager
import gc

from src.core.label_computation import LabelComputationEngine
from src.models.data_models import Candle, Granularity


@contextmanager
def measure_time():
    """Context manager to measure execution time."""
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start
    

@contextmanager
def measure_memory():
    """Context manager to measure memory usage."""
    process = psutil.Process()
    gc.collect()  # Clean up before measurement
    start_memory = process.memory_info().rss
    
    def get_memory_delta():
        gc.collect()
        current_memory = process.memory_info().rss
        return current_memory - start_memory
    
    yield get_memory_delta


@pytest.mark.performance
class TestCachePerformanceBenchmarks:
    """Performance benchmarks for caching scenarios."""
    
    @pytest.fixture
    def benchmark_candle(self) -> Candle:
        """Standard candle for benchmarking."""
        return Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=1.0500, high=1.0580, low=1.0450, close=1.0520,
            volume=1000, atr_14=0.0045
        )
    
    @pytest.fixture
    def mock_cached_result(self) -> Dict[str, Any]:
        """Mock cached label result."""
        return {
            "enhanced_triple_barrier": {
                "label": 1,
                "barrier_hit": "upper",
                "time_to_barrier": 5,
                "barrier_price": 1.0580,
                "level_adjusted": False,
                "upper_barrier": 1.0580,
                "lower_barrier": 1.0460,
                "path_granularity": None
            },
            "instrument_id": "EUR/USD",
            "granularity": "H4",
            "ts": datetime(2024, 1, 15, 9, 0, 0)
        }
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, benchmark_candle, mock_cached_result):
        """Benchmark cache hit performance."""
        with patch('src.core.label_computation.redis_cache') as mock_redis:
            # Configure cache hit
            mock_redis.get_labels.return_value = mock_cached_result
            
            engine = LabelComputationEngine()
            
            # Warmup
            for _ in range(10):
                await engine.compute_labels(benchmark_candle, use_cache=True)
            
            # Benchmark cache hits
            iterations = 1000
            
            with measure_time() as get_time:
                for _ in range(iterations):
                    result = await engine.compute_labels(benchmark_candle, use_cache=True)
                    assert result is not None
            
            total_time = get_time()
            avg_time_ms = (total_time / iterations) * 1000
            
            # Cache hits should be very fast (< 1ms per operation)
            assert avg_time_ms < 1.0, f"Cache hit too slow: {avg_time_ms:.3f}ms average"
            
            # Calculate throughput
            ops_per_second = iterations / total_time
            assert ops_per_second > 1000, f"Insufficient cache hit throughput: {ops_per_second:.0f} ops/sec"
            
            print(f"Cache hit performance: {avg_time_ms:.3f}ms avg, {ops_per_second:.0f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_cache_miss_performance(self, benchmark_candle):
        """Benchmark cache miss performance."""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Configure cache miss
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            engine = LabelComputationEngine()
            
            # Warmup
            for _ in range(5):
                await engine.compute_labels(benchmark_candle, use_cache=True)
            
            # Benchmark cache misses
            iterations = 100  # Fewer iterations for cache misses
            
            with measure_time() as get_time:
                for _ in range(iterations):
                    result = await engine.compute_labels(benchmark_candle, use_cache=True)
                    assert result is not None
            
            total_time = get_time()
            avg_time_ms = (total_time / iterations) * 1000
            
            # Cache misses should still be reasonable (< 50ms per operation)
            assert avg_time_ms < 50.0, f"Cache miss too slow: {avg_time_ms:.3f}ms average"
            
            # Calculate throughput
            ops_per_second = iterations / total_time
            assert ops_per_second > 20, f"Insufficient cache miss throughput: {ops_per_second:.0f} ops/sec"
            
            print(f"Cache miss performance: {avg_time_ms:.3f}ms avg, {ops_per_second:.0f} ops/sec")
    
    @pytest.mark.asyncio
    async def test_cache_hit_ratio_impact(self, benchmark_candle, mock_cached_result):
        """Test performance impact of different cache hit ratios."""
        hit_ratios = [0.0, 0.5, 0.8, 0.95, 1.0]
        performance_results = []
        
        for hit_ratio in hit_ratios:
            with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
                 patch('src.core.label_computation.redis_cache') as mock_redis:
                
                # Configure services
                mock_ch.fetch_active_levels.return_value = []
                mock_ch.fetch_snapshots.return_value = []
                
                # Set up cache hit ratio
                call_count = 0
                def mock_get_labels(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if np.random.random() < hit_ratio:
                        return mock_cached_result
                    return None
                
                mock_redis.get_labels.side_effect = mock_get_labels
                
                engine = LabelComputationEngine()
                
                # Benchmark mixed hit/miss scenario
                iterations = 200
                
                with measure_time() as get_time:
                    for _ in range(iterations):
                        result = await engine.compute_labels(benchmark_candle, use_cache=True)
                        assert result is not None
                
                total_time = get_time()
                avg_time_ms = (total_time / iterations) * 1000
                ops_per_second = iterations / total_time
                
                performance_results.append({
                    'hit_ratio': hit_ratio,
                    'avg_time_ms': avg_time_ms,
                    'ops_per_second': ops_per_second
                })
                
                print(f"Hit ratio {hit_ratio:.0%}: {avg_time_ms:.3f}ms avg, {ops_per_second:.0f} ops/sec")
        
        # Verify that higher hit ratios improve performance
        for i in range(1, len(performance_results)):
            current = performance_results[i]
            previous = performance_results[i-1]
            
            # Higher hit ratio should generally have better performance
            if current['hit_ratio'] > previous['hit_ratio']:
                assert current['avg_time_ms'] <= previous['avg_time_ms'] * 1.2, \
                    "Higher cache hit ratio should improve or maintain performance"


@pytest.mark.performance
class TestThroughputBenchmarks:
    """Throughput benchmarks for batch processing."""
    
    @pytest.fixture
    def large_candle_dataset(self, large_dataset) -> List[Candle]:
        """Large dataset for throughput testing."""
        return large_dataset[:1000]  # Limit to 1000 for performance tests
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_processing_throughput(self, large_candle_dataset):
        """Benchmark batch processing throughput."""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Configure mocks for fast processing
            mock_redis.get_labels.return_value = None  # Force computation
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            # Prepare batch snapshots
            batch_snapshots = []
            for candle in large_candle_dataset:
                batch_snapshots.append({
                    "ts": candle.ts,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "atr_14": candle.atr_14
                })
            
            mock_ch.fetch_snapshots.return_value = batch_snapshots
            
            engine = LabelComputationEngine()
            
            # Benchmark batch processing
            start_date = large_candle_dataset[0].ts
            end_date = large_candle_dataset[-1].ts + timedelta(hours=4)
            
            with measure_time() as get_time:
                result = await engine.compute_batch_labels(
                    instrument_id="EUR/USD",
                    granularity="H4",
                    start_date=start_date,
                    end_date=end_date,
                    label_types=["enhanced_triple_barrier"],
                    chunk_size=100
                )
            
            total_time = get_time()
            processed_count = result["processed_candles"]
            
            # Calculate metrics
            candles_per_second = processed_count / total_time
            avg_time_per_candle_ms = (total_time / processed_count) * 1000
            
            # Performance requirements
            assert candles_per_second > 10, f"Batch throughput too low: {candles_per_second:.1f} candles/sec"
            assert avg_time_per_candle_ms < 100, f"Per-candle time too high: {avg_time_per_candle_ms:.3f}ms"
            
            print(f"Batch processing: {candles_per_second:.1f} candles/sec, {avg_time_per_candle_ms:.3f}ms per candle")
            print(f"Success rate: {(result['successful_labels'] / processed_count) * 100:.1f}%")
    
    @pytest.mark.asyncio 
    async def test_concurrent_processing_throughput(self, large_candle_dataset):
        """Benchmark concurrent label computation."""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            engine = LabelComputationEngine()
            
            # Test different concurrency levels
            concurrency_levels = [1, 2, 4, 8, 16]
            test_candles = large_candle_dataset[:100]  # Smaller set for concurrency test
            
            for concurrency in concurrency_levels:
                # Create batches for concurrent processing
                batch_size = len(test_candles) // concurrency
                batches = [
                    test_candles[i:i + batch_size] 
                    for i in range(0, len(test_candles), batch_size)
                ]
                
                async def process_batch(candles):
                    tasks = []
                    for candle in candles:
                        task = engine.compute_labels(candle)
                        tasks.append(task)
                    return await asyncio.gather(*tasks)
                
                with measure_time() as get_time:
                    # Process batches concurrently
                    concurrent_tasks = [process_batch(batch) for batch in batches]
                    results = await asyncio.gather(*concurrent_tasks)
                
                total_time = get_time()
                total_processed = sum(len(batch_result) for batch_result in results)
                throughput = total_processed / total_time
                
                print(f"Concurrency {concurrency}: {throughput:.1f} labels/sec")
                
                # Verify concurrency helps (up to a point)
                if concurrency == 1:
                    baseline_throughput = throughput
                elif concurrency <= 4:
                    # Should see improvement with modest concurrency
                    assert throughput >= baseline_throughput * 0.8, \
                        f"Concurrency {concurrency} underperforming baseline"


@pytest.mark.performance
class TestMemoryBenchmarks:
    """Memory usage benchmarks."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_single_computation(self, sample_candle):
        """Test memory usage for single label computation."""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            engine = LabelComputationEngine()
            
            with measure_memory() as get_memory_delta:
                # Compute multiple labels to see memory pattern
                for _ in range(100):
                    result = await engine.compute_labels(sample_candle)
                    assert result is not None
            
            memory_used_mb = get_memory_delta() / (1024 * 1024)
            memory_per_computation_kb = (memory_used_mb * 1024) / 100
            
            # Should not use excessive memory
            assert memory_used_mb < 50, f"Memory usage too high: {memory_used_mb:.2f}MB"
            assert memory_per_computation_kb < 100, f"Per-computation memory too high: {memory_per_computation_kb:.2f}KB"
            
            print(f"Memory usage: {memory_used_mb:.2f}MB total, {memory_per_computation_kb:.2f}KB per computation")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_usage_batch_processing(self, large_candle_dataset):
        """Test memory usage for batch processing."""
        test_candles = large_candle_dataset[:500]  # Moderate size for memory test
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Prepare batch data
            batch_snapshots = []
            for candle in test_candles:
                batch_snapshots.append({
                    "ts": candle.ts,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "atr_14": candle.atr_14
                })
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_snapshots.return_value = batch_snapshots
            mock_ch.fetch_active_levels.return_value = []
            
            engine = LabelComputationEngine()
            
            with measure_memory() as get_memory_delta:
                result = await engine.compute_batch_labels(
                    instrument_id="EUR/USD",
                    granularity="H4",
                    start_date=test_candles[0].ts,
                    end_date=test_candles[-1].ts + timedelta(hours=4),
                    label_types=["enhanced_triple_barrier"],
                    chunk_size=50
                )
            
            memory_used_mb = get_memory_delta() / (1024 * 1024)
            memory_per_candle_kb = (memory_used_mb * 1024) / result["processed_candles"]
            
            # Memory usage should scale reasonably
            assert memory_used_mb < 200, f"Batch memory usage too high: {memory_used_mb:.2f}MB"
            assert memory_per_candle_kb < 50, f"Per-candle memory too high: {memory_per_candle_kb:.2f}KB"
            
            print(f"Batch memory usage: {memory_used_mb:.2f}MB total, {memory_per_candle_kb:.2f}KB per candle")
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, sample_candle):
        """Test for memory leaks in repeated computations."""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            engine = LabelComputationEngine()
            
            # Baseline measurement
            process = psutil.Process()
            gc.collect()
            baseline_memory = process.memory_info().rss
            
            # Run many computations
            iterations = 1000
            for i in range(iterations):
                result = await engine.compute_labels(sample_candle)
                assert result is not None
                
                # Force garbage collection periodically
                if i % 100 == 0:
                    gc.collect()
            
            # Final measurement
            gc.collect()
            final_memory = process.memory_info().rss
            memory_growth_mb = (final_memory - baseline_memory) / (1024 * 1024)
            
            # Memory growth should be minimal
            acceptable_growth_mb = 20  # 20MB acceptable growth
            assert memory_growth_mb < acceptable_growth_mb, \
                f"Potential memory leak detected: {memory_growth_mb:.2f}MB growth after {iterations} iterations"
            
            print(f"Memory stability: {memory_growth_mb:.2f}MB growth over {iterations} iterations")


@pytest.mark.performance
class TestLatencyBenchmarks:
    """Latency benchmarks for real-time scenarios."""
    
    @pytest.mark.asyncio
    async def test_single_computation_latency_distribution(self, sample_candle):
        """Test latency distribution for single computations."""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            engine = LabelComputationEngine()
            
            # Collect latency measurements
            latencies = []
            iterations = 200
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                result = await engine.compute_labels(sample_candle)
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                assert result is not None
            
            # Calculate statistics
            latencies = np.array(latencies)
            mean_latency = np.mean(latencies)
            median_latency = np.median(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            max_latency = np.max(latencies)
            
            # Performance requirements
            assert mean_latency < 30, f"Mean latency too high: {mean_latency:.3f}ms"
            assert p95_latency < 50, f"P95 latency too high: {p95_latency:.3f}ms"
            assert p99_latency < 100, f"P99 latency too high: {p99_latency:.3f}ms"
            
            print(f"Latency distribution:")
            print(f"  Mean: {mean_latency:.3f}ms")
            print(f"  Median: {median_latency:.3f}ms")
            print(f"  P95: {p95_latency:.3f}ms")
            print(f"  P99: {p99_latency:.3f}ms")
            print(f"  Max: {max_latency:.3f}ms")
    
    @pytest.mark.asyncio
    async def test_cold_start_vs_warm_performance(self, sample_candle):
        """Test cold start vs warm performance."""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            # Cold start measurement
            engine = LabelComputationEngine()
            
            cold_start_time = time.perf_counter()
            cold_result = await engine.compute_labels(sample_candle)
            cold_end_time = time.perf_counter()
            cold_latency_ms = (cold_end_time - cold_start_time) * 1000
            
            assert cold_result is not None
            
            # Warm measurements
            warm_latencies = []
            for _ in range(10):
                warm_start_time = time.perf_counter()
                warm_result = await engine.compute_labels(sample_candle)
                warm_end_time = time.perf_counter()
                warm_latency_ms = (warm_end_time - warm_start_time) * 1000
                warm_latencies.append(warm_latency_ms)
                
                assert warm_result is not None
            
            avg_warm_latency = np.mean(warm_latencies)
            
            print(f"Cold start latency: {cold_latency_ms:.3f}ms")
            print(f"Warm average latency: {avg_warm_latency:.3f}ms")
            print(f"Cold start overhead: {cold_latency_ms - avg_warm_latency:.3f}ms")
            
            # Cold start should not be dramatically slower
            assert cold_latency_ms < avg_warm_latency * 3, \
                f"Cold start too slow: {cold_latency_ms:.3f}ms vs {avg_warm_latency:.3f}ms warm"


@pytest.mark.performance
class TestScalabilityBenchmarks:
    """Scalability benchmarks for large datasets."""
    
    @pytest.mark.parametrize("dataset_size", [100, 500, 1000, 2000])
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_processing_time_scalability(self, dataset_size):
        """Test how processing time scales with dataset size."""
        # Generate dataset of specified size
        candles = []
        base_time = datetime(2024, 1, 1, 1, 0, 0)
        
        for i in range(dataset_size):
            candles.append(Candle(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=base_time + timedelta(hours=4 * i),
                open=1.0500, high=1.0580, low=1.0450, close=1.0520,
                volume=1000, atr_14=0.0045
            ))
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Prepare batch data
            batch_snapshots = []
            for candle in candles:
                batch_snapshots.append({
                    "ts": candle.ts,
                    "open": candle.open,
                    "high": candle.high, 
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "atr_14": candle.atr_14
                })
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_snapshots.return_value = batch_snapshots
            mock_ch.fetch_active_levels.return_value = []
            
            engine = LabelComputationEngine()
            
            with measure_time() as get_time:
                result = await engine.compute_batch_labels(
                    instrument_id="EUR/USD",
                    granularity="H4",
                    start_date=candles[0].ts,
                    end_date=candles[-1].ts + timedelta(hours=4),
                    label_types=["enhanced_triple_barrier"],
                    chunk_size=100
                )
            
            total_time = get_time()
            throughput = result["processed_candles"] / total_time
            
            print(f"Dataset size {dataset_size}: {throughput:.1f} candles/sec, {total_time:.2f}s total")
            
            # Throughput should not degrade significantly with size
            expected_min_throughput = 5  # Minimum acceptable throughput
            assert throughput > expected_min_throughput, \
                f"Throughput too low for size {dataset_size}: {throughput:.1f} candles/sec"
    
    @pytest.mark.asyncio
    async def test_database_query_optimization(self, sample_candle):
        """Test that database queries are optimized for performance."""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            
            # Track database calls
            levels_call_count = 0
            snapshots_call_count = 0
            
            async def track_levels_calls(*args, **kwargs):
                nonlocal levels_call_count
                levels_call_count += 1
                return []
            
            async def track_snapshots_calls(*args, **kwargs):
                nonlocal snapshots_call_count
                snapshots_call_count += 1
                return []
            
            mock_ch.fetch_active_levels = AsyncMock(side_effect=track_levels_calls)
            mock_ch.fetch_snapshots = AsyncMock(side_effect=track_snapshots_calls)
            
            engine = LabelComputationEngine()
            
            # Compute multiple labels
            iterations = 10
            for _ in range(iterations):
                result = await engine.compute_labels(sample_candle)
                assert result is not None
            
            # Verify reasonable number of database calls
            print(f"Database calls for {iterations} computations:")
            print(f"  Levels calls: {levels_call_count}")
            print(f"  Snapshots calls: {snapshots_call_count}")
            
            # Each computation should make exactly one call to each service
            # (assuming no caching optimization)
            assert levels_call_count == iterations, \
                f"Unexpected levels call count: {levels_call_count} vs {iterations}"
            assert snapshots_call_count == iterations, \
                f"Unexpected snapshots call count: {snapshots_call_count} vs {iterations}"


@pytest.mark.performance
class TestStressTests:
    """Stress tests for extreme conditions."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_high_load_stress(self, sample_candle):
        """Test system behavior under high load."""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            engine = LabelComputationEngine()
            
            # High load test
            high_load_iterations = 5000
            start_time = time.perf_counter()
            successful_computations = 0
            errors = 0
            
            for i in range(high_load_iterations):
                try:
                    result = await engine.compute_labels(sample_candle)
                    if result is not None:
                        successful_computations += 1
                    
                    # Brief pause every 100 iterations to prevent overwhelming
                    if i % 100 == 0:
                        await asyncio.sleep(0.001)  # 1ms pause
                        
                except Exception as e:
                    errors += 1
                    if errors > high_load_iterations * 0.05:  # > 5% error rate
                        pytest.fail(f"Too many errors under high load: {errors}/{i+1}")
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            success_rate = successful_computations / high_load_iterations
            throughput = successful_computations / total_time
            
            print(f"High load stress test:")
            print(f"  Iterations: {high_load_iterations}")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Throughput: {throughput:.1f} computations/sec")
            print(f"  Errors: {errors}")
            
            # Requirements under stress
            assert success_rate > 0.95, f"Success rate too low under stress: {success_rate:.1%}"
            assert throughput > 50, f"Throughput too low under stress: {throughput:.1f} computations/sec"
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, sample_candle):
        """Test behavior under memory pressure."""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            
            # Create large path data to simulate memory pressure
            large_path_data = []
            for i in range(10000):  # Large dataset
                large_path_data.append({
                    "ts": sample_candle.ts + timedelta(minutes=i),
                    "high": 1.0520 + np.random.normal(0, 0.01),
                    "low": 1.0520 - abs(np.random.normal(0, 0.01)),
                    "close": 1.0520 + np.random.normal(0, 0.005),
                    "volume": 100
                })
            
            mock_ch.fetch_snapshots.return_value = large_path_data
            
            engine = LabelComputationEngine()
            
            # Monitor memory during computation
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # This should handle large datasets gracefully
            result = await engine.compute_labels(sample_candle)
            
            final_memory = process.memory_info().rss
            memory_growth_mb = (final_memory - initial_memory) / (1024 * 1024)
            
            assert result is not None
            assert memory_growth_mb < 500, f"Excessive memory usage: {memory_growth_mb:.2f}MB"
            
            print(f"Memory pressure test: {memory_growth_mb:.2f}MB growth with large dataset")