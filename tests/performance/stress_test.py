"""
Comprehensive stress testing suite for the Label Computation System.

This suite identifies system breaking points and failure modes:
- Progressive load increases until system failure
- Resource exhaustion scenarios (memory, CPU, connections)
- Error cascade and recovery testing
- System stability under extreme conditions
- Breaking point identification and documentation

Usage:
    pytest tests/performance/stress_test.py -v --tb=short
    pytest tests/performance/stress_test.py::TestStressTesting::test_progressive_load_until_failure -v
"""

import pytest
import asyncio
import time
import psutil
import gc
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
import concurrent.futures
from collections import deque
import logging
import traceback

from src.core.label_computation import LabelComputationEngine
from src.models.data_models import Candle, Granularity
from src.services.batch_worker_pool import BatchWorkerPool


@dataclass
class StressTestResult:
    """Results from a stress test."""
    test_name: str
    breaking_point: Optional[int]
    max_successful_load: int
    failure_mode: str
    error_rate_at_breaking_point: float
    system_metrics_at_breaking_point: Dict[str, Any]
    recovery_time_seconds: Optional[float]
    recommendations: List[str]


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    open_files: int
    network_connections: int
    thread_count: int
    
    @classmethod
    def capture_current(cls) -> 'SystemMetrics':
        """Capture current system metrics."""
        process = psutil.Process()
        
        return cls(
            timestamp=time.time(),
            cpu_percent=process.cpu_percent(),
            memory_percent=process.memory_percent(),
            memory_mb=process.memory_info().rss / (1024 * 1024),
            open_files=len(process.open_files()) if hasattr(process, 'open_files') else 0,
            network_connections=len(process.connections()) if hasattr(process, 'connections') else 0,
            thread_count=process.num_threads()
        )


class StressTestMonitor:
    """Monitors system resources during stress testing."""
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=1000)
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring system metrics."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        return list(self.metrics_history)
    
    def _monitor_loop(self, interval: float):
        """Monitor loop running in separate thread."""
        while self.monitoring:
            try:
                metrics = SystemMetrics.capture_current()
                self.metrics_history.append(metrics)
                time.sleep(interval)
            except Exception as e:
                logging.warning(f"Error capturing metrics: {e}")
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def detect_resource_exhaustion(self) -> List[str]:
        """Detect potential resource exhaustion conditions."""
        if not self.metrics_history:
            return []
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 measurements
        warnings = []
        
        # CPU exhaustion
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        if avg_cpu > 95:
            warnings.append(f"High CPU usage: {avg_cpu:.1f}%")
        
        # Memory exhaustion
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        if avg_memory > 90:
            warnings.append(f"High memory usage: {avg_memory:.1f}%")
        
        # Memory growth
        if len(recent_metrics) >= 5:
            memory_trend = np.polyfit(
                range(len(recent_metrics)), 
                [m.memory_mb for m in recent_metrics], 
                1
            )[0]  # Slope
            if memory_trend > 10:  # Growing by >10MB per measurement
                warnings.append(f"Memory leak detected: {memory_trend:.1f}MB/sec growth")
        
        # Thread exhaustion
        max_threads = max([m.thread_count for m in recent_metrics])
        if max_threads > 500:
            warnings.append(f"High thread count: {max_threads}")
        
        return warnings


def generate_stress_test_candles(count: int, complexity_factor: float = 1.0) -> List[Candle]:
    """Generate candles for stress testing with configurable complexity."""
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    candles = []
    
    instruments = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "EURGBP", "EURJPY"]
    
    for i in range(count):
        # Add complexity through price volatility
        volatility = 0.01 * complexity_factor
        
        instrument = instruments[i % len(instruments)]
        
        candles.append(Candle(
            instrument_id=instrument,
            granularity=Granularity.H4,
            ts=base_time + timedelta(hours=4 * i),
            open=1.0500 + np.random.normal(0, volatility),
            high=1.0580 + np.random.normal(0, volatility),
            low=1.0450 + np.random.normal(0, volatility),
            close=1.0520 + np.random.normal(0, volatility),
            volume=int(1000 * (1 + np.random.normal(0, complexity_factor))),
            atr_14=0.0045 * (1 + np.random.normal(0, complexity_factor * 0.5))
        ))
    
    return candles


@pytest.mark.performance
@pytest.mark.stress
class TestStressTesting:
    """Core stress testing for system breaking points."""
    
    def setup_method(self):
        """Set up monitoring for each test."""
        self.monitor = StressTestMonitor()
        self.stress_results = []
    
    def teardown_method(self):
        """Clean up after each test."""
        if self.monitor:
            self.monitor.stop_monitoring()
        
        # Force garbage collection
        gc.collect()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_progressive_load_until_failure(self):
        """Progressively increase load until system breaks."""
        print("Starting progressive load stress test...")
        
        self.monitor.start_monitoring(interval=2.0)
        
        # Test configuration
        load_levels = [100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7500, 10000]
        failure_threshold = 0.1  # 10% error rate indicates failure
        breaking_point = None
        max_successful_load = 0
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Configure mocks for realistic load
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            engine = LabelComputationEngine()
            
            for load_level in load_levels:
                print(f"\nTesting load level: {load_level} candles/minute")
                
                # Generate test candles for this load level
                test_candles = generate_stress_test_candles(min(load_level, 1000))
                
                # Perform load test
                start_time = time.perf_counter()
                successful_operations = 0
                failed_operations = 0
                errors = []
                
                # Process candles with target rate
                target_ops_per_second = load_level / 60
                batch_size = min(50, max(1, int(target_ops_per_second / 10)))
                
                try:
                    for i in range(0, len(test_candles), batch_size):
                        batch = test_candles[i:i + batch_size]
                        batch_start = time.perf_counter()
                        
                        # Process batch
                        tasks = []
                        for candle in batch:
                            task = asyncio.create_task(
                                self._process_candle_with_error_tracking(engine, candle)
                            )
                            tasks.append(task)
                        
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Count successes and failures
                        for result in results:
                            if isinstance(result, Exception):
                                failed_operations += 1
                                errors.append(str(result))
                            elif result is not None:
                                successful_operations += 1
                            else:
                                failed_operations += 1
                                errors.append("Computation returned None")
                        
                        # Rate limiting
                        batch_time = time.perf_counter() - batch_start
                        target_batch_time = len(batch) / target_ops_per_second
                        if batch_time < target_batch_time:
                            await asyncio.sleep(target_batch_time - batch_time)
                        
                        # Check for resource exhaustion
                        resource_warnings = self.monitor.detect_resource_exhaustion()
                        if resource_warnings:
                            print(f"Resource warnings: {resource_warnings}")
                            if any("exhaustion" in w.lower() or "High" in w for w in resource_warnings):
                                print(f"Resource exhaustion detected at load level {load_level}")
                                breaking_point = load_level
                                break
                
                except Exception as e:
                    print(f"Critical error at load level {load_level}: {e}")
                    breaking_point = load_level
                    errors.append(f"Critical error: {str(e)}")
                
                # Calculate error rate
                total_operations = successful_operations + failed_operations
                error_rate = failed_operations / total_operations if total_operations > 0 else 1.0
                
                elapsed_time = time.perf_counter() - start_time
                actual_ops_per_second = total_operations / elapsed_time if elapsed_time > 0 else 0
                
                print(f"Load level {load_level} results:")
                print(f"  Operations: {total_operations} ({successful_operations} success, {failed_operations} failed)")
                print(f"  Error rate: {error_rate:.2%}")
                print(f"  Actual OPS: {actual_ops_per_second:.1f}/sec")
                print(f"  Duration: {elapsed_time:.2f}s")
                
                # Check if this load level failed
                if error_rate >= failure_threshold or breaking_point == load_level:
                    print(f"BREAKING POINT IDENTIFIED: {load_level} candles/minute")
                    breaking_point = load_level
                    
                    # Capture system metrics at breaking point
                    metrics_at_breaking_point = self.monitor.get_current_metrics()
                    
                    break
                else:
                    max_successful_load = load_level
                
                # Brief recovery period between load levels
                await asyncio.sleep(5)
        
        # Stop monitoring and get metrics history
        metrics_history = self.monitor.stop_monitoring()
        
        # Analyze results
        result = StressTestResult(
            test_name="Progressive Load Until Failure",
            breaking_point=breaking_point,
            max_successful_load=max_successful_load,
            failure_mode="Error rate exceeded threshold" if breaking_point else "Test completed without failure",
            error_rate_at_breaking_point=error_rate if breaking_point else 0.0,
            system_metrics_at_breaking_point=asdict(metrics_at_breaking_point) if metrics_at_breaking_point else {},
            recovery_time_seconds=None,
            recommendations=self._generate_load_test_recommendations(breaking_point, max_successful_load, metrics_history)
        )
        
        self.stress_results.append(result)
        
        print(f"\nProgressive Load Stress Test Results:")
        print(f"Breaking point: {breaking_point or 'Not reached'} candles/minute")
        print(f"Max successful load: {max_successful_load} candles/minute")
        print(f"Recommendations: {result.recommendations}")
        
        # Assertions
        assert max_successful_load >= 1000, f"System cannot handle minimum required load: {max_successful_load}"
        if breaking_point:
            assert breaking_point > max_successful_load, "Breaking point should be higher than max successful load"
    
    async def _process_candle_with_error_tracking(self, engine, candle):
        """Process a candle with comprehensive error tracking."""
        try:
            return await engine.compute_labels(candle)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Log the full traceback for analysis
            logging.error(f"Error processing candle {candle.instrument_id} at {candle.ts}: {e}")
            logging.error(traceback.format_exc())
            raise
    
    @pytest.mark.asyncio
    async def test_memory_exhaustion_stress(self):
        """Test system behavior under memory pressure."""
        print("Starting memory exhaustion stress test...")
        
        self.monitor.start_monitoring(interval=1.0)
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Configure mocks to return large datasets
            def create_large_dataset(size_mb: int):
                # Create a dataset that uses approximately size_mb of memory
                rows = size_mb * 1000  # Rough estimate
                return [
                    {
                        "ts": datetime.now() + timedelta(minutes=i),
                        "high": 1.0520 + np.random.normal(0, 0.01),
                        "low": 1.0520 - abs(np.random.normal(0, 0.01)),
                        "close": 1.0520 + np.random.normal(0, 0.005),
                        "volume": 100
                    } for i in range(rows)
                ]
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            
            engine = LabelComputationEngine()
            breaking_point_mb = None
            max_successful_memory_mb = 0
            
            # Progressively increase memory usage
            memory_levels_mb = [10, 25, 50, 100, 200, 500, 1000, 1500, 2000]
            
            for memory_mb in memory_levels_mb:
                print(f"\nTesting with {memory_mb}MB dataset...")
                
                try:
                    # Create large dataset
                    large_dataset = create_large_dataset(memory_mb)
                    mock_ch.fetch_snapshots.return_value = large_dataset
                    
                    # Monitor memory before operation
                    initial_metrics = SystemMetrics.capture_current()
                    print(f"Initial memory: {initial_metrics.memory_mb:.1f}MB ({initial_metrics.memory_percent:.1f}%)")
                    
                    # Process candle with large dataset
                    test_candle = generate_stress_test_candles(1)[0]
                    
                    start_time = time.perf_counter()
                    result = await engine.compute_labels(test_candle)
                    end_time = time.perf_counter()
                    
                    # Monitor memory after operation
                    final_metrics = SystemMetrics.capture_current()
                    memory_growth = final_metrics.memory_mb - initial_metrics.memory_mb
                    
                    print(f"Memory after: {final_metrics.memory_mb:.1f}MB ({final_metrics.memory_percent:.1f}%)")
                    print(f"Memory growth: {memory_growth:.1f}MB")
                    print(f"Processing time: {(end_time - start_time):.2f}s")
                    
                    if result is not None:
                        max_successful_memory_mb = memory_mb
                        print(f"✓ Successfully processed {memory_mb}MB dataset")
                    else:
                        print(f"✗ Failed to process {memory_mb}MB dataset")
                        breaking_point_mb = memory_mb
                        break
                    
                    # Check for memory exhaustion signals
                    if final_metrics.memory_percent > 95:
                        print(f"Memory exhaustion detected at {memory_mb}MB")
                        breaking_point_mb = memory_mb
                        break
                    
                    # Clean up
                    del large_dataset
                    gc.collect()
                    
                    # Brief pause for recovery
                    await asyncio.sleep(2)
                    
                except MemoryError:
                    print(f"MemoryError at {memory_mb}MB dataset")
                    breaking_point_mb = memory_mb
                    break
                except Exception as e:
                    print(f"Error at {memory_mb}MB dataset: {e}")
                    breaking_point_mb = memory_mb
                    break
        
        metrics_history = self.monitor.stop_monitoring()
        
        result = StressTestResult(
            test_name="Memory Exhaustion Stress",
            breaking_point=breaking_point_mb,
            max_successful_load=max_successful_memory_mb,
            failure_mode="Memory exhaustion" if breaking_point_mb else "Test completed without failure",
            error_rate_at_breaking_point=1.0 if breaking_point_mb else 0.0,
            system_metrics_at_breaking_point=asdict(self.monitor.get_current_metrics() or SystemMetrics.capture_current()),
            recovery_time_seconds=None,
            recommendations=self._generate_memory_test_recommendations(breaking_point_mb, max_successful_memory_mb)
        )
        
        self.stress_results.append(result)
        
        print(f"\nMemory Exhaustion Stress Test Results:")
        print(f"Breaking point: {breaking_point_mb or 'Not reached'}MB dataset")
        print(f"Max successful: {max_successful_memory_mb}MB dataset")
        print(f"Recommendations: {result.recommendations}")
        
        # Assertions
        assert max_successful_memory_mb >= 50, f"System cannot handle reasonable memory loads: {max_successful_memory_mb}MB"
    
    @pytest.mark.asyncio
    async def test_connection_exhaustion_stress(self):
        """Test system behavior when connection pools are exhausted."""
        print("Starting connection exhaustion stress test...")
        
        self.monitor.start_monitoring(interval=1.0)
        
        max_concurrent_connections = 200
        connection_increment = 25
        breaking_point_connections = None
        max_successful_connections = 0
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Track connection attempts
            active_connections = 0
            connection_errors = 0
            
            async def mock_db_operation_with_delay(*args, **kwargs):
                nonlocal active_connections, connection_errors
                
                if active_connections >= max_concurrent_connections:
                    connection_errors += 1
                    raise Exception("Connection pool exhausted")
                
                active_connections += 1
                try:
                    # Simulate database operation with realistic delay
                    await asyncio.sleep(np.random.exponential(0.1))  # Average 100ms
                    return []
                finally:
                    active_connections -= 1
            
            mock_ch.fetch_active_levels = AsyncMock(side_effect=mock_db_operation_with_delay)
            mock_ch.fetch_snapshots = AsyncMock(side_effect=mock_db_operation_with_delay)
            mock_redis.get_labels.return_value = None
            
            engine = LabelComputationEngine()
            
            for concurrent_ops in range(connection_increment, max_concurrent_connections + connection_increment, connection_increment):
                print(f"\nTesting {concurrent_ops} concurrent operations...")
                
                # Reset counters
                connection_errors = 0
                successful_operations = 0
                
                # Create test candles
                test_candles = generate_stress_test_candles(concurrent_ops)
                
                try:
                    # Launch all operations concurrently
                    tasks = []
                    for candle in test_candles:
                        task = asyncio.create_task(engine.compute_labels(candle))
                        tasks.append(task)
                    
                    # Wait for all operations to complete
                    start_time = time.perf_counter()
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    end_time = time.perf_counter()
                    
                    # Count results
                    for result in results:
                        if isinstance(result, Exception):
                            if "Connection pool exhausted" in str(result):
                                connection_errors += 1
                        else:
                            successful_operations += 1
                    
                    error_rate = connection_errors / len(results)
                    
                    print(f"Concurrent operations: {concurrent_ops}")
                    print(f"Successful: {successful_operations}")
                    print(f"Connection errors: {connection_errors}")
                    print(f"Error rate: {error_rate:.2%}")
                    print(f"Duration: {(end_time - start_time):.2f}s")
                    
                    if error_rate > 0.1:  # 10% error threshold
                        print(f"Connection breaking point: {concurrent_ops}")
                        breaking_point_connections = concurrent_ops
                        break
                    else:
                        max_successful_connections = concurrent_ops
                
                except Exception as e:
                    print(f"Critical error at {concurrent_ops} connections: {e}")
                    breaking_point_connections = concurrent_ops
                    break
                
                # Brief pause between tests
                await asyncio.sleep(2)
        
        metrics_history = self.monitor.stop_monitoring()
        
        result = StressTestResult(
            test_name="Connection Exhaustion Stress",
            breaking_point=breaking_point_connections,
            max_successful_load=max_successful_connections,
            failure_mode="Connection pool exhaustion" if breaking_point_connections else "Test completed without failure",
            error_rate_at_breaking_point=error_rate if breaking_point_connections else 0.0,
            system_metrics_at_breaking_point=asdict(self.monitor.get_current_metrics() or SystemMetrics.capture_current()),
            recovery_time_seconds=None,
            recommendations=self._generate_connection_test_recommendations(breaking_point_connections, max_successful_connections)
        )
        
        self.stress_results.append(result)
        
        print(f"\nConnection Exhaustion Stress Test Results:")
        print(f"Breaking point: {breaking_point_connections or 'Not reached'} concurrent connections")
        print(f"Max successful: {max_successful_connections} concurrent connections")
        print(f"Recommendations: {result.recommendations}")
        
        # Assertions
        assert max_successful_connections >= 50, f"System cannot handle reasonable connection load: {max_successful_connections}"
    
    @pytest.mark.asyncio
    async def test_error_cascade_and_recovery(self):
        """Test system recovery after error cascades."""
        print("Starting error cascade and recovery test...")
        
        self.monitor.start_monitoring(interval=0.5)
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Configure system to fail after certain number of operations
            operation_count = 0
            failure_start = 50
            failure_duration = 20  # Operations during which system fails
            
            async def failing_service(*args, **kwargs):
                nonlocal operation_count
                operation_count += 1
                
                if failure_start <= operation_count < failure_start + failure_duration:
                    raise Exception("Simulated cascade failure")
                
                # Normal operation
                await asyncio.sleep(0.01)  # 10ms normal operation
                return []
            
            mock_ch.fetch_active_levels = AsyncMock(side_effect=failing_service)
            mock_ch.fetch_snapshots = AsyncMock(side_effect=failing_service)
            mock_redis.get_labels.return_value = None
            
            engine = LabelComputationEngine()
            
            # Test scenario: process candles before, during, and after failure
            total_candles = 100
            test_candles = generate_stress_test_candles(total_candles)
            
            results = []
            error_start_time = None
            error_end_time = None
            
            print(f"Processing {total_candles} candles with simulated cascade failure...")
            
            for i, candle in enumerate(test_candles):
                try:
                    start_time = time.perf_counter()
                    result = await engine.compute_labels(candle)
                    end_time = time.perf_counter()
                    
                    results.append({
                        'candle_index': i,
                        'success': result is not None,
                        'latency_ms': (end_time - start_time) * 1000,
                        'error': None
                    })
                    
                    if error_start_time and not error_end_time:
                        error_end_time = time.perf_counter()
                        recovery_time = error_end_time - error_start_time
                        print(f"System recovered after {recovery_time:.2f} seconds")
                
                except Exception as e:
                    end_time = time.perf_counter()
                    
                    if not error_start_time:
                        error_start_time = time.perf_counter()
                        print(f"Error cascade started at candle {i}")
                    
                    results.append({
                        'candle_index': i,
                        'success': False,
                        'latency_ms': (end_time - start_time) * 1000,
                        'error': str(e)
                    })
                
                # Brief delay between operations
                await asyncio.sleep(0.05)
        
        metrics_history = self.monitor.stop_monitoring()
        
        # Analyze results
        successful_operations = sum(1 for r in results if r['success'])
        failed_operations = len(results) - successful_operations
        error_rate = failed_operations / len(results)
        
        # Calculate recovery time
        recovery_time = None
        if error_start_time and error_end_time:
            recovery_time = error_end_time - error_start_time
        
        # Check if system recovered
        last_10_results = results[-10:]
        recovery_success_rate = sum(1 for r in last_10_results if r['success']) / len(last_10_results)
        
        result = StressTestResult(
            test_name="Error Cascade and Recovery",
            breaking_point=None,
            max_successful_load=successful_operations,
            failure_mode="Simulated cascade failure",
            error_rate_at_breaking_point=error_rate,
            system_metrics_at_breaking_point=asdict(self.monitor.get_current_metrics() or SystemMetrics.capture_current()),
            recovery_time_seconds=recovery_time,
            recommendations=self._generate_recovery_test_recommendations(recovery_time, recovery_success_rate)
        )
        
        self.stress_results.append(result)
        
        print(f"\nError Cascade and Recovery Test Results:")
        print(f"Total operations: {len(results)}")
        print(f"Successful: {successful_operations}")
        print(f"Failed: {failed_operations}")
        print(f"Error rate: {error_rate:.2%}")
        print(f"Recovery time: {recovery_time:.2f}s" if recovery_time else "No recovery detected")
        print(f"Post-recovery success rate: {recovery_success_rate:.2%}")
        print(f"Recommendations: {result.recommendations}")
        
        # Assertions
        assert recovery_success_rate >= 0.8, f"Poor recovery: {recovery_success_rate:.2%} success rate after cascade"
        assert recovery_time and recovery_time <= 30, f"Recovery too slow: {recovery_time}s"
    
    def _generate_load_test_recommendations(self, breaking_point: Optional[int], max_successful: int, metrics_history: List[SystemMetrics]) -> List[str]:
        """Generate recommendations based on load test results."""
        recommendations = []
        
        if breaking_point:
            recommendations.append(f"System breaking point identified at {breaking_point} candles/minute")
            recommendations.append(f"Recommend operating at {max_successful * 0.8:.0f} candles/minute (80% of max)")
        
        if metrics_history:
            peak_memory = max([m.memory_percent for m in metrics_history])
            peak_cpu = max([m.cpu_percent for m in metrics_history])
            
            if peak_memory > 80:
                recommendations.append(f"High memory usage detected ({peak_memory:.1f}%) - consider memory optimization")
            
            if peak_cpu > 85:
                recommendations.append(f"High CPU usage detected ({peak_cpu:.1f}%) - consider horizontal scaling")
        
        recommendations.append("Implement circuit breakers to prevent cascade failures")
        recommendations.append("Add auto-scaling based on queue depth and response times")
        
        return recommendations
    
    def _generate_memory_test_recommendations(self, breaking_point_mb: Optional[int], max_successful_mb: int) -> List[str]:
        """Generate recommendations based on memory test results."""
        recommendations = []
        
        if breaking_point_mb:
            recommendations.append(f"Memory breaking point: {breaking_point_mb}MB dataset size")
            recommendations.append("Implement streaming processing for large datasets")
            recommendations.append("Add memory monitoring and alerts")
        
        recommendations.append(f"Safe operating limit: {max_successful_mb * 0.7:.0f}MB dataset size")
        recommendations.append("Implement data pagination for large queries")
        recommendations.append("Consider memory pooling for frequently allocated objects")
        
        return recommendations
    
    def _generate_connection_test_recommendations(self, breaking_point: Optional[int], max_successful: int) -> List[str]:
        """Generate recommendations based on connection test results."""
        recommendations = []
        
        if breaking_point:
            recommendations.append(f"Connection pool exhaustion at {breaking_point} concurrent operations")
            recommendations.append("Increase connection pool sizes")
            recommendations.append("Implement connection queueing and timeout handling")
        
        recommendations.append(f"Safe concurrent operation limit: {max_successful * 0.8:.0f}")
        recommendations.append("Monitor connection pool utilization")
        recommendations.append("Implement graceful degradation when pools are full")
        
        return recommendations
    
    def _generate_recovery_test_recommendations(self, recovery_time: Optional[float], recovery_success_rate: float) -> List[str]:
        """Generate recommendations based on recovery test results."""
        recommendations = []
        
        if recovery_time:
            if recovery_time > 10:
                recommendations.append(f"Slow recovery time: {recovery_time:.1f}s - implement faster error detection")
            else:
                recommendations.append(f"Good recovery time: {recovery_time:.1f}s")
        
        if recovery_success_rate < 0.9:
            recommendations.append(f"Poor recovery success rate: {recovery_success_rate:.2%} - improve error handling")
        
        recommendations.append("Implement health checks and automatic recovery")
        recommendations.append("Add circuit breakers with exponential backoff")
        recommendations.append("Monitor and alert on error rates")
        
        return recommendations
    
    def test_generate_stress_test_report(self):
        """Generate a comprehensive stress test report."""
        if not self.stress_results:
            pytest.skip("No stress test results to report")
        
        report_lines = [
            "# Stress Testing Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Executive Summary",
            ""
        ]
        
        for result in self.stress_results:
            report_lines.extend([
                f"### {result.test_name}",
                f"- Breaking Point: {result.breaking_point or 'Not reached'}",
                f"- Max Successful Load: {result.max_successful_load}",
                f"- Failure Mode: {result.failure_mode}",
                f"- Recovery Time: {result.recovery_time_seconds or 'N/A'}s",
                "",
                "**Recommendations:**"
            ])
            
            for rec in result.recommendations:
                report_lines.append(f"- {rec}")
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Write report to file
        report_path = "/Users/aminechbani/labels_lab/my-project/stress_test_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"Stress test report written to: {report_path}")
        print("\nStress Test Summary:")
        for result in self.stress_results:
            print(f"- {result.test_name}: Breaking point {result.breaking_point or 'Not reached'}")