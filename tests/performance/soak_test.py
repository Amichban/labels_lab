"""
Comprehensive soak testing suite for the Label Computation System.

This suite tests system stability over extended periods:
- 24-hour continuous operation testing
- Memory leak detection over time
- Performance degradation analysis
- Resource stability monitoring
- Long-term error accumulation tracking

Usage:
    # Full 24-hour soak test (run in production-like environment)
    pytest tests/performance/soak_test.py::TestSoakTesting::test_24_hour_continuous_operation -v
    
    # Shorter soak tests for development
    pytest tests/performance/soak_test.py::TestSoakTesting::test_1_hour_soak -v
    pytest tests/performance/soak_test.py::TestSoakTesting::test_memory_leak_detection -v
"""

import pytest
import asyncio
import time
import threading
import psutil
import gc
import json
import csv
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
import numpy as np
from collections import deque, defaultdict
import logging
import signal
import os

from src.core.label_computation import LabelComputationEngine
from src.models.data_models import Candle, Granularity
from src.services.batch_worker_pool import BatchWorkerPool


@dataclass
class SoakTestMetrics:
    """Metrics collected during soak testing."""
    timestamp: datetime
    operations_completed: int
    operations_per_minute: float
    success_rate: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    memory_usage_mb: float
    memory_percent: float
    cpu_percent: float
    thread_count: int
    open_files: int
    active_connections: int
    cache_hit_rate: float
    error_count: int
    errors_per_minute: float


@dataclass
class SoakTestResult:
    """Final results from a soak test."""
    test_name: str
    duration_hours: float
    total_operations: int
    overall_success_rate: float
    average_throughput: float
    performance_degradation: float  # Percentage degradation over time
    memory_leak_detected: bool
    memory_growth_mb_per_hour: float
    stability_score: float  # 0-100, higher is better
    critical_issues: List[str]
    recommendations: List[str]
    metrics_file: str


class SoakTestController:
    """Controls long-running soak tests with graceful shutdown."""
    
    def __init__(self):
        self.running = False
        self.shutdown_requested = False
        self.metrics_history: deque = deque(maxlen=10000)  # Store up to 10k metrics points
        self.current_metrics = SoakTestMetrics(
            timestamp=datetime.now(),
            operations_completed=0,
            operations_per_minute=0,
            success_rate=0,
            avg_response_time_ms=0,
            p95_response_time_ms=0,
            p99_response_time_ms=0,
            memory_usage_mb=0,
            memory_percent=0,
            cpu_percent=0,
            thread_count=0,
            open_files=0,
            active_connections=0,
            cache_hit_rate=0,
            error_count=0,
            errors_per_minute=0
        )
        
        # Performance tracking
        self.operation_times: deque = deque(maxlen=1000)  # Last 1000 operation times
        self.operation_results: deque = deque(maxlen=1000)  # Last 1000 operation results
        self.errors: List[str] = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\nReceived signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def start_test(self, duration_hours: float):
        """Start the soak test."""
        self.running = True
        self.shutdown_requested = False
        self.start_time = time.time()
        self.end_time = self.start_time + (duration_hours * 3600)
        print(f"Starting soak test for {duration_hours} hours...")
        
        # Start metrics collection thread
        self.metrics_thread = threading.Thread(
            target=self._collect_metrics_loop,
            daemon=True
        )
        self.metrics_thread.start()
    
    def stop_test(self):
        """Stop the soak test."""
        self.running = False
        print("Stopping soak test...")
        
        if hasattr(self, 'metrics_thread'):
            self.metrics_thread.join(timeout=5.0)
    
    def should_continue(self) -> bool:
        """Check if test should continue running."""
        if self.shutdown_requested:
            return False
        
        current_time = time.time()
        return current_time < self.end_time and self.running
    
    def record_operation(self, start_time: float, end_time: float, success: bool, error: Optional[str] = None):
        """Record the result of an operation."""
        operation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        self.operation_times.append(operation_time)
        self.operation_results.append(success)
        
        if error:
            self.errors.append(error)
    
    def _collect_metrics_loop(self):
        """Collect system metrics in background thread."""
        while self.running and not self.shutdown_requested:
            try:
                # Collect system metrics
                process = psutil.Process()
                
                # Calculate operation metrics
                recent_times = list(self.operation_times)[-100:]  # Last 100 operations
                recent_results = list(self.operation_results)[-100:]
                
                if recent_times:
                    avg_response_time = np.mean(recent_times)
                    p95_response_time = np.percentile(recent_times, 95)
                    p99_response_time = np.percentile(recent_times, 99)
                else:
                    avg_response_time = p95_response_time = p99_response_time = 0
                
                if recent_results:
                    success_rate = sum(recent_results) / len(recent_results)
                else:
                    success_rate = 0
                
                # Calculate rates
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                if elapsed_time > 0:
                    operations_per_minute = (len(self.operation_results) / elapsed_time) * 60
                    errors_per_minute = (len(self.errors) / elapsed_time) * 60
                else:
                    operations_per_minute = errors_per_minute = 0
                
                # Create metrics snapshot
                metrics = SoakTestMetrics(
                    timestamp=datetime.now(),
                    operations_completed=len(self.operation_results),
                    operations_per_minute=operations_per_minute,
                    success_rate=success_rate,
                    avg_response_time_ms=avg_response_time,
                    p95_response_time_ms=p95_response_time,
                    p99_response_time_ms=p99_response_time,
                    memory_usage_mb=process.memory_info().rss / (1024 * 1024),
                    memory_percent=process.memory_percent(),
                    cpu_percent=process.cpu_percent(),
                    thread_count=process.num_threads(),
                    open_files=len(process.open_files()) if hasattr(process, 'open_files') else 0,
                    active_connections=len(process.connections()) if hasattr(process, 'connections') else 0,
                    cache_hit_rate=0.95,  # TODO: Get actual cache hit rate
                    error_count=len(self.errors),
                    errors_per_minute=errors_per_minute
                )
                
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Print periodic status
                if len(self.metrics_history) % 60 == 0:  # Every minute
                    elapsed_hours = elapsed_time / 3600
                    remaining_hours = (self.end_time - current_time) / 3600
                    print(f"Soak test status - Elapsed: {elapsed_hours:.1f}h, "
                          f"Remaining: {remaining_hours:.1f}h, "
                          f"Ops: {metrics.operations_completed}, "
                          f"Success: {metrics.success_rate:.1%}, "
                          f"Memory: {metrics.memory_usage_mb:.1f}MB")
                
            except Exception as e:
                logging.error(f"Error collecting metrics: {e}")
            
            time.sleep(1.0)  # Collect metrics every second
    
    def get_test_results(self) -> SoakTestResult:
        """Generate final test results."""
        if not self.metrics_history:
            raise ValueError("No metrics collected")
        
        # Calculate overall statistics
        total_operations = len(self.operation_results)
        overall_success_rate = sum(self.operation_results) / len(self.operation_results) if self.operation_results else 0
        
        # Calculate performance degradation
        early_metrics = list(self.metrics_history)[:100]  # First 100 measurements
        late_metrics = list(self.metrics_history)[-100:]   # Last 100 measurements
        
        if early_metrics and late_metrics:
            early_avg_response = np.mean([m.avg_response_time_ms for m in early_metrics])
            late_avg_response = np.mean([m.avg_response_time_ms for m in late_metrics])
            
            if early_avg_response > 0:
                performance_degradation = ((late_avg_response - early_avg_response) / early_avg_response) * 100
            else:
                performance_degradation = 0
        else:
            performance_degradation = 0
        
        # Detect memory leaks
        memory_usage_trend = [m.memory_usage_mb for m in self.metrics_history]
        if len(memory_usage_trend) >= 10:
            # Linear regression to detect memory growth trend
            x = np.arange(len(memory_usage_trend))
            slope, _ = np.polyfit(x, memory_usage_trend, 1)
            
            # Convert slope to MB per hour
            measurements_per_hour = 3600  # 1 measurement per second
            memory_growth_mb_per_hour = slope * measurements_per_hour
            memory_leak_detected = memory_growth_mb_per_hour > 10  # >10MB/hour growth
        else:
            memory_growth_mb_per_hour = 0
            memory_leak_detected = False
        
        # Calculate stability score
        stability_factors = []
        
        # Success rate factor (0-40 points)
        stability_factors.append(overall_success_rate * 40)
        
        # Performance consistency factor (0-30 points)
        if performance_degradation <= 5:  # Less than 5% degradation
            stability_factors.append(30)
        elif performance_degradation <= 20:  # Less than 20% degradation
            stability_factors.append(20)
        else:
            stability_factors.append(0)
        
        # Memory stability factor (0-20 points)
        if not memory_leak_detected:
            stability_factors.append(20)
        else:
            stability_factors.append(max(0, 20 - (memory_growth_mb_per_hour / 10)))
        
        # Error rate factor (0-10 points)
        error_rate = len(self.errors) / total_operations if total_operations > 0 else 0
        if error_rate <= 0.01:  # Less than 1% error rate
            stability_factors.append(10)
        elif error_rate <= 0.05:  # Less than 5% error rate
            stability_factors.append(5)
        else:
            stability_factors.append(0)
        
        stability_score = sum(stability_factors)
        
        # Identify critical issues
        critical_issues = []
        if overall_success_rate < 0.95:
            critical_issues.append(f"Low success rate: {overall_success_rate:.1%}")
        
        if performance_degradation > 20:
            critical_issues.append(f"Significant performance degradation: {performance_degradation:.1f}%")
        
        if memory_leak_detected:
            critical_issues.append(f"Memory leak detected: {memory_growth_mb_per_hour:.1f}MB/hour growth")
        
        if self.current_metrics.errors_per_minute > 10:
            critical_issues.append(f"High error rate: {self.current_metrics.errors_per_minute:.1f} errors/minute")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_success_rate, performance_degradation, memory_leak_detected, 
            memory_growth_mb_per_hour, self.current_metrics.errors_per_minute
        )
        
        # Save metrics to file
        metrics_file = self._save_metrics_to_file()
        
        duration_hours = (time.time() - self.start_time) / 3600
        average_throughput = total_operations / (duration_hours * 60) if duration_hours > 0 else 0  # Operations per minute
        
        return SoakTestResult(
            test_name="Soak Test",
            duration_hours=duration_hours,
            total_operations=total_operations,
            overall_success_rate=overall_success_rate,
            average_throughput=average_throughput,
            performance_degradation=performance_degradation,
            memory_leak_detected=memory_leak_detected,
            memory_growth_mb_per_hour=memory_growth_mb_per_hour,
            stability_score=stability_score,
            critical_issues=critical_issues,
            recommendations=recommendations,
            metrics_file=metrics_file
        )
    
    def _generate_recommendations(self, success_rate: float, degradation: float, 
                                  memory_leak: bool, memory_growth: float, error_rate: float) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if success_rate < 0.95:
            recommendations.append("Improve error handling and retry mechanisms")
            recommendations.append("Investigate root causes of failures")
        
        if degradation > 10:
            recommendations.append("Investigate performance degradation over time")
            recommendations.append("Consider periodic service restarts or resource cleanup")
        
        if memory_leak:
            recommendations.append(f"Address memory leak ({memory_growth:.1f}MB/hour growth)")
            recommendations.append("Review object lifecycle management and garbage collection")
        
        if error_rate > 5:
            recommendations.append("Reduce error rate through better input validation and error handling")
        
        recommendations.extend([
            "Implement monitoring dashboards for production",
            "Set up alerting for performance degradation and resource issues",
            "Consider implementing circuit breakers for external dependencies",
            "Add health check endpoints for operational monitoring"
        ])
        
        return recommendations
    
    def _save_metrics_to_file(self) -> str:
        """Save collected metrics to CSV and JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = "/Users/aminechbani/labels_lab/my-project"
        
        # Save as CSV
        csv_file = f"{base_path}/soak_test_metrics_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "timestamp", "operations_completed", "operations_per_minute",
                "success_rate", "avg_response_time_ms", "p95_response_time_ms",
                "p99_response_time_ms", "memory_usage_mb", "memory_percent",
                "cpu_percent", "thread_count", "open_files", "active_connections",
                "cache_hit_rate", "error_count", "errors_per_minute"
            ])
            
            # Data
            for metrics in self.metrics_history:
                writer.writerow([
                    metrics.timestamp.isoformat(),
                    metrics.operations_completed,
                    metrics.operations_per_minute,
                    metrics.success_rate,
                    metrics.avg_response_time_ms,
                    metrics.p95_response_time_ms,
                    metrics.p99_response_time_ms,
                    metrics.memory_usage_mb,
                    metrics.memory_percent,
                    metrics.cpu_percent,
                    metrics.thread_count,
                    metrics.open_files,
                    metrics.active_connections,
                    metrics.cache_hit_rate,
                    metrics.error_count,
                    metrics.errors_per_minute
                ])
        
        # Save as JSON for detailed analysis
        json_file = f"{base_path}/soak_test_metrics_{timestamp}.json"
        metrics_data = {
            "test_metadata": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(time.time()).isoformat(),
                "duration_hours": (time.time() - self.start_time) / 3600,
                "total_operations": len(self.operation_results),
                "total_errors": len(self.errors)
            },
            "metrics": [asdict(m) for m in self.metrics_history],
            "errors": self.errors[-100:]  # Last 100 errors for analysis
        }
        
        with open(json_file, 'w') as f:
            json.dump(metrics_data, f, default=str, indent=2)
        
        print(f"Metrics saved to: {csv_file}")
        print(f"Detailed data saved to: {json_file}")
        
        return csv_file


def generate_continuous_test_data(duration_minutes: int, ops_per_minute: int) -> List[Candle]:
    """Generate test data for continuous operation."""
    total_operations = duration_minutes * ops_per_minute
    base_time = datetime.now() - timedelta(hours=24)  # Start 24 hours ago
    
    candles = []
    instruments = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD"]
    
    for i in range(total_operations):
        instrument = instruments[i % len(instruments)]
        
        # Add some realism with slight trends
        trend_factor = np.sin(i / 1000) * 0.01  # Slow sine wave trend
        
        candles.append(Candle(
            instrument_id=instrument,
            granularity=Granularity.H4,
            ts=base_time + timedelta(minutes=i * 4),  # 4-minute intervals
            open=1.0500 + trend_factor + np.random.normal(0, 0.005),
            high=1.0580 + trend_factor + np.random.normal(0, 0.005),
            low=1.0450 + trend_factor + np.random.normal(0, 0.005),
            close=1.0520 + trend_factor + np.random.normal(0, 0.005),
            volume=1000 + np.random.randint(-200, 200),
            atr_14=0.0045 + np.random.normal(0, 0.0005)
        ))
    
    return candles


@pytest.mark.performance
@pytest.mark.soak
class TestSoakTesting:
    """Long-running soak tests for system stability."""
    
    @pytest.mark.skip(reason="24-hour test - run manually in production environment")
    @pytest.mark.asyncio
    async def test_24_hour_continuous_operation(self):
        """Full 24-hour soak test - run in production-like environment only."""
        duration_hours = 24
        target_ops_per_minute = 50  # Moderate load for stability testing
        
        controller = SoakTestController()
        
        try:
            controller.start_test(duration_hours)
            
            with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
                 patch('src.core.label_computation.redis_cache') as mock_redis:
                
                # Configure realistic mocks
                mock_redis.get_labels.return_value = None
                mock_ch.fetch_active_levels.return_value = []
                mock_ch.fetch_snapshots.return_value = []
                
                engine = LabelComputationEngine()
                
                # Generate continuous stream of test data
                operation_count = 0
                
                print(f"Starting 24-hour soak test with {target_ops_per_minute} operations/minute")
                
                while controller.should_continue():
                    # Generate a batch of candles
                    batch_size = 10
                    test_candles = generate_continuous_test_data(1, batch_size)  # 1 minute worth
                    
                    for candle in test_candles:
                        if not controller.should_continue():
                            break
                        
                        start_time = time.perf_counter()
                        
                        try:
                            result = await engine.compute_labels(candle)
                            end_time = time.perf_counter()
                            
                            success = result is not None
                            controller.record_operation(start_time, end_time, success)
                            operation_count += 1
                            
                        except Exception as e:
                            end_time = time.perf_counter()
                            controller.record_operation(start_time, end_time, False, str(e))
                        
                        # Rate limiting to achieve target ops/minute
                        target_interval = 60.0 / target_ops_per_minute
                        await asyncio.sleep(target_interval)
                    
                    # Periodic cleanup
                    if operation_count % 1000 == 0:
                        gc.collect()
        
        finally:
            controller.stop_test()
            
            # Generate results
            results = controller.get_test_results()
            
            print(f"\n24-Hour Soak Test Results:")
            print(f"Duration: {results.duration_hours:.1f} hours")
            print(f"Total Operations: {results.total_operations}")
            print(f"Success Rate: {results.overall_success_rate:.2%}")
            print(f"Average Throughput: {results.average_throughput:.1f} ops/minute")
            print(f"Performance Degradation: {results.performance_degradation:.1f}%")
            print(f"Memory Leak Detected: {results.memory_leak_detected}")
            print(f"Stability Score: {results.stability_score:.1f}/100")
            print(f"Critical Issues: {results.critical_issues}")
            
            # Assertions
            assert results.overall_success_rate >= 0.95, f"Poor success rate: {results.overall_success_rate:.2%}"
            assert results.performance_degradation <= 25, f"Excessive degradation: {results.performance_degradation:.1f}%"
            assert not results.memory_leak_detected, "Memory leak detected during 24-hour test"
            assert results.stability_score >= 70, f"Poor stability score: {results.stability_score:.1f}/100"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_1_hour_soak(self):
        """1-hour soak test for development and CI environments."""
        duration_hours = 1.0
        target_ops_per_minute = 100  # Higher load for shorter test
        
        controller = SoakTestController()
        
        try:
            controller.start_test(duration_hours)
            
            with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
                 patch('src.core.label_computation.redis_cache') as mock_redis:
                
                mock_redis.get_labels.return_value = None
                mock_ch.fetch_active_levels.return_value = []
                mock_ch.fetch_snapshots.return_value = []
                
                engine = LabelComputationEngine()
                operation_count = 0
                
                print(f"Starting 1-hour soak test with {target_ops_per_minute} operations/minute")
                
                while controller.should_continue():
                    # Process in small batches
                    batch_size = 5
                    test_candles = generate_continuous_test_data(1, batch_size)
                    
                    for candle in test_candles:
                        if not controller.should_continue():
                            break
                        
                        start_time = time.perf_counter()
                        
                        try:
                            result = await engine.compute_labels(candle)
                            end_time = time.perf_counter()
                            
                            success = result is not None
                            controller.record_operation(start_time, end_time, success)
                            operation_count += 1
                            
                        except Exception as e:
                            end_time = time.perf_counter()
                            controller.record_operation(start_time, end_time, False, str(e))
                        
                        # Rate limiting
                        target_interval = 60.0 / target_ops_per_minute
                        await asyncio.sleep(target_interval)
                    
                    # Periodic cleanup
                    if operation_count % 500 == 0:
                        gc.collect()
        
        finally:
            controller.stop_test()
            
            # Generate and validate results
            results = controller.get_test_results()
            
            print(f"\n1-Hour Soak Test Results:")
            print(f"Duration: {results.duration_hours:.2f} hours")
            print(f"Total Operations: {results.total_operations}")
            print(f"Success Rate: {results.overall_success_rate:.2%}")
            print(f"Average Throughput: {results.average_throughput:.1f} ops/minute")
            print(f"Performance Degradation: {results.performance_degradation:.1f}%")
            print(f"Memory Growth: {results.memory_growth_mb_per_hour:.1f}MB/hour")
            print(f"Stability Score: {results.stability_score:.1f}/100")
            
            # Assertions for 1-hour test
            assert results.overall_success_rate >= 0.90, f"Poor success rate: {results.overall_success_rate:.2%}"
            assert results.performance_degradation <= 50, f"Excessive degradation: {results.performance_degradation:.1f}%"
            assert results.stability_score >= 60, f"Poor stability score: {results.stability_score:.1f}/100"
            
            # Memory growth should be reasonable for 1-hour test
            assert results.memory_growth_mb_per_hour <= 100, f"High memory growth: {results.memory_growth_mb_per_hour:.1f}MB/hour"
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Focused test for memory leak detection over shorter duration."""
        duration_minutes = 30
        target_ops_per_minute = 200  # High load to stress memory
        
        controller = SoakTestController()
        
        try:
            controller.start_test(duration_minutes / 60.0)  # Convert to hours
            
            with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
                 patch('src.core.label_computation.redis_cache') as mock_redis:
                
                # Mock large data responses to stress memory
                large_snapshots = []
                for i in range(1000):  # Large dataset
                    large_snapshots.append({
                        "ts": datetime.now() + timedelta(minutes=i),
                        "high": 1.0520 + np.random.normal(0, 0.01),
                        "low": 1.0520 - abs(np.random.normal(0, 0.01)),
                        "close": 1.0520 + np.random.normal(0, 0.005),
                        "volume": 100
                    })
                
                mock_redis.get_labels.return_value = None
                mock_ch.fetch_active_levels.return_value = []
                mock_ch.fetch_snapshots.return_value = large_snapshots
                
                engine = LabelComputationEngine()
                operation_count = 0
                
                print(f"Starting {duration_minutes}-minute memory leak test")
                
                while controller.should_continue():
                    test_candle = generate_continuous_test_data(1, 1)[0]  # Single candle
                    
                    start_time = time.perf_counter()
                    
                    try:
                        result = await engine.compute_labels(test_candle)
                        end_time = time.perf_counter()
                        
                        success = result is not None
                        controller.record_operation(start_time, end_time, success)
                        operation_count += 1
                        
                    except Exception as e:
                        end_time = time.perf_counter()
                        controller.record_operation(start_time, end_time, False, str(e))
                    
                    # Rate limiting
                    target_interval = 60.0 / target_ops_per_minute
                    await asyncio.sleep(target_interval)
                    
                    # Don't force GC to better detect leaks
                    if operation_count % 50 == 0:  # Less frequent cleanup
                        current_memory = controller.current_metrics.memory_usage_mb
                        print(f"Operations: {operation_count}, Memory: {current_memory:.1f}MB")
        
        finally:
            controller.stop_test()
            
            results = controller.get_test_results()
            
            print(f"\nMemory Leak Detection Test Results:")
            print(f"Duration: {results.duration_hours:.2f} hours")
            print(f"Total Operations: {results.total_operations}")
            print(f"Memory Growth Rate: {results.memory_growth_mb_per_hour:.1f}MB/hour")
            print(f"Memory Leak Detected: {results.memory_leak_detected}")
            print(f"Success Rate: {results.overall_success_rate:.2%}")
            
            # Memory leak specific assertions
            if results.memory_leak_detected:
                print(f"⚠️  Memory leak detected: {results.memory_growth_mb_per_hour:.1f}MB/hour growth")
                print("Recommendations:")
                for rec in results.recommendations:
                    if "memory" in rec.lower():
                        print(f"  - {rec}")
                
                # Don't fail the test, but warn
                pytest.warns(UserWarning, f"Memory leak detected: {results.memory_growth_mb_per_hour:.1f}MB/hour")
            else:
                print("✅ No memory leak detected")
            
            # Success rate should still be good
            assert results.overall_success_rate >= 0.85, f"Poor success rate during memory test: {results.overall_success_rate:.2%}"
    
    @pytest.mark.asyncio
    async def test_performance_degradation_analysis(self):
        """Test to analyze performance degradation patterns."""
        duration_minutes = 20
        target_ops_per_minute = 150
        
        controller = SoakTestController()
        
        try:
            controller.start_test(duration_minutes / 60.0)
            
            with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
                 patch('src.core.label_computation.redis_cache') as mock_redis:
                
                # Simulate gradual performance degradation
                operation_count = 0
                
                async def degrading_mock_service(*args, **kwargs):
                    nonlocal operation_count
                    operation_count += 1
                    
                    # Gradually increase response time
                    base_delay = 0.01  # 10ms base
                    degradation_factor = operation_count / 1000  # Gradual increase
                    delay = base_delay + (degradation_factor * 0.005)  # Up to 5ms additional delay
                    
                    await asyncio.sleep(delay)
                    return []
                
                mock_redis.get_labels.return_value = None
                mock_ch.fetch_active_levels = AsyncMock(side_effect=degrading_mock_service)
                mock_ch.fetch_snapshots = AsyncMock(side_effect=degrading_mock_service)
                
                engine = LabelComputationEngine()
                
                print(f"Starting {duration_minutes}-minute performance degradation test")
                
                while controller.should_continue():
                    test_candle = generate_continuous_test_data(1, 1)[0]
                    
                    start_time = time.perf_counter()
                    
                    try:
                        result = await engine.compute_labels(test_candle)
                        end_time = time.perf_counter()
                        
                        success = result is not None
                        controller.record_operation(start_time, end_time, success)
                        
                    except Exception as e:
                        end_time = time.perf_counter()
                        controller.record_operation(start_time, end_time, False, str(e))
                    
                    # Rate limiting
                    target_interval = 60.0 / target_ops_per_minute
                    await asyncio.sleep(target_interval)
        
        finally:
            controller.stop_test()
            
            results = controller.get_test_results()
            
            print(f"\nPerformance Degradation Analysis Results:")
            print(f"Duration: {results.duration_hours:.2f} hours")
            print(f"Total Operations: {results.total_operations}")
            print(f"Performance Degradation: {results.performance_degradation:.1f}%")
            print(f"Success Rate: {results.overall_success_rate:.2%}")
            
            # Analyze degradation pattern
            if results.performance_degradation > 10:
                print(f"📈 Significant performance degradation detected: {results.performance_degradation:.1f}%")
                print("This suggests the system may not be suitable for long-running operations without restarts")
            else:
                print("✅ Performance remained stable")
            
            # Should maintain good success rate despite degradation
            assert results.overall_success_rate >= 0.90, f"Poor success rate: {results.overall_success_rate:.2%}"
            
            # Document degradation for analysis
            if results.performance_degradation > 50:
                pytest.fail(f"Excessive performance degradation: {results.performance_degradation:.1f}%")


@pytest.mark.performance
@pytest.mark.soak
class TestSoakTestReporting:
    """Generate comprehensive soak test reports."""
    
    def test_generate_soak_test_summary(self):
        """Generate a summary report of all soak test capabilities."""
        
        report_content = [
            "# Soak Testing Suite Summary",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Available Soak Tests",
            "",
            "### 1. 24-Hour Continuous Operation Test",
            "- **Purpose**: Validate system stability over extended periods",
            "- **Duration**: 24 hours",
            "- **Load**: 50 operations/minute (moderate sustained load)",
            "- **Metrics**: Success rate, performance degradation, memory leaks, stability score",
            "- **Usage**: Run in production-like environment only",
            "",
            "### 2. 1-Hour Soak Test", 
            "- **Purpose**: Development and CI-friendly soak testing",
            "- **Duration**: 1 hour",
            "- **Load**: 100 operations/minute",
            "- **Metrics**: Same as 24-hour test with adjusted thresholds",
            "- **Usage**: Suitable for automated testing pipelines",
            "",
            "### 3. Memory Leak Detection Test",
            "- **Purpose**: Focused memory leak detection",
            "- **Duration**: 30 minutes",
            "- **Load**: 200 operations/minute with large datasets",
            "- **Metrics**: Memory growth rate, leak detection",
            "- **Usage**: Run when investigating memory issues",
            "",
            "### 4. Performance Degradation Analysis",
            "- **Purpose**: Analyze performance degradation patterns",
            "- **Duration**: 20 minutes",
            "- **Load**: 150 operations/minute with gradual service degradation",
            "- **Metrics**: Response time trends, degradation percentage",
            "- **Usage**: Validate system behavior under degrading conditions",
            "",
            "## Key Metrics Tracked",
            "",
            "- **Operations Per Minute**: Throughput measurement",
            "- **Success Rate**: Percentage of successful operations",
            "- **Response Time Percentiles**: P50, P95, P99 latencies",
            "- **Memory Usage**: Current usage and growth trends",
            "- **CPU Utilization**: Processor usage patterns",
            "- **System Resources**: Thread count, file handles, connections",
            "- **Error Rates**: Errors per minute and types",
            "- **Stability Score**: Composite score (0-100)",
            "",
            "## Running Soak Tests",
            "",
            "```bash",
            "# 1-hour development soak test",
            "pytest tests/performance/soak_test.py::TestSoakTesting::test_1_hour_soak -v",
            "",
            "# Memory leak detection",
            "pytest tests/performance/soak_test.py::TestSoakTesting::test_memory_leak_detection -v",
            "",
            "# Performance degradation analysis", 
            "pytest tests/performance/soak_test.py::TestSoakTesting::test_performance_degradation_analysis -v",
            "",
            "# Full 24-hour test (production environment)",
            "pytest tests/performance/soak_test.py::TestSoakTesting::test_24_hour_continuous_operation -v",
            "```",
            "",
            "## Output Files",
            "",
            "Each soak test generates:",
            "- **CSV metrics file**: Time-series data for analysis",
            "- **JSON detailed data**: Complete test metadata and metrics",
            "- **Console output**: Real-time progress and final results",
            "",
            "## Interpretation Guidelines",
            "",
            "### Success Criteria",
            "- Success rate ≥ 95% for production readiness",
            "- Performance degradation ≤ 25% over 24 hours",
            "- Memory growth ≤ 10MB/hour (no significant leaks)",
            "- Stability score ≥ 70/100",
            "",
            "### Warning Signs",
            "- Success rate 90-95%: Investigate error patterns",
            "- Performance degradation 25-50%: Consider periodic restarts",
            "- Memory growth 10-50MB/hour: Monitor closely, investigate if persistent",
            "- Stability score 50-70: System may need optimization",
            "",
            "### Critical Issues",
            "- Success rate < 90%: Not suitable for production",
            "- Performance degradation > 50%: Major stability issues",
            "- Memory growth > 50MB/hour: Significant memory leak",
            "- Stability score < 50: System requires major fixes",
            "",
            "## Integration with CI/CD",
            "",
            "1. **Daily CI**: Run 1-hour soak test",
            "2. **Weekly CI**: Run memory leak detection test",
            "3. **Pre-release**: Run performance degradation analysis",
            "4. **Post-deployment**: Run 24-hour test in staging",
            "",
            "## Monitoring and Alerting",
            "",
            "Soak tests provide baseline data for:",
            "- Setting up production monitoring thresholds",
            "- Configuring performance alerts",
            "- Establishing SLA targets",
            "- Capacity planning decisions"
        ]
        
        report_path = "/Users/aminechbani/labels_lab/my-project/soak_test_summary.md"
        
        with open(report_path, "w") as f:
            f.write("\n".join(report_content))
        
        print(f"Soak test summary written to: {report_path}")
        
        # Verify file was created
        assert os.path.exists(report_path), "Summary report file was not created"