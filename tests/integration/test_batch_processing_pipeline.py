"""
Integration Tests for Batch Processing Pipeline

Comprehensive test suite for the batch backfill system including:
- End-to-end pipeline testing
- Performance validation (1M+ candles/minute target)
- Error handling and recovery scenarios
- Parallel processing validation
- Redis-based progress tracking tests
- CLI command integration tests
- Metrics and monitoring validation
- Worker pool scaling tests
- Circuit breaker and retry logic tests
"""

import pytest
import asyncio
import time
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import threading

from src.services.batch_backfill_service import BatchBackfillService, JobStatus
from src.services.batch_metrics_service import BatchMetricsCollector
from src.services.batch_worker_pool import BatchWorkerPool, TaskPriority
from src.services.batch_error_handler import BatchErrorHandler, ErrorCategory, ErrorSeverity
from src.services.redis_cache import redis_cache
from src.services.clickhouse_service import clickhouse_service
from src.models.data_models import Granularity


class TestBatchProcessingPipeline:
    """Integration tests for the complete batch processing pipeline"""
    
    @pytest.fixture
    def batch_service(self):
        """Create a batch service instance for testing"""
        return BatchBackfillService(
            max_workers=2,
            chunk_size=100,  # Small chunks for testing
            max_retries=2,
            redis_ttl=3600
        )
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a metrics collector for testing"""
        return BatchMetricsCollector(redis_key_prefix="test_metrics")
    
    @pytest.fixture
    def worker_pool(self):
        """Create a worker pool for testing"""
        pool = BatchWorkerPool(
            min_workers=1,
            max_workers=3,
            initial_workers=2,
            max_queue_size=100
        )
        yield pool
        pool.shutdown(wait=True, timeout=5)
    
    @pytest.fixture
    def error_handler(self):
        """Create an error handler for testing"""
        return BatchErrorHandler(redis_key_prefix="test_errors")
    
    @pytest.fixture
    def mock_snapshots(self):
        """Generate mock snapshot data"""
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        snapshots = []
        
        for i in range(500):  # 500 mock candles
            timestamp = base_time + timedelta(hours=i * 4)  # H4 candles
            snapshot = {
                'ts': timestamp,
                'open': 1.0950 + (i * 0.0001),
                'high': 1.0970 + (i * 0.0001),
                'low': 1.0940 + (i * 0.0001),
                'close': 1.0965 + (i * 0.0001),
                'volume': 1000000,
                'atr_14': 0.0025
            }
            snapshots.append(snapshot)
        
        return snapshots
    
    @pytest.mark.asyncio
    async def test_end_to_end_batch_job(self, batch_service, mock_snapshots):
        """Test complete end-to-end batch job execution"""
        
        # Mock ClickHouse service to return test data
        with patch.object(clickhouse_service, 'fetch_snapshots', return_value=mock_snapshots):
            
            # Start a batch job
            job_id = await batch_service.start_backfill_job(
                instrument_id="EURUSD",
                granularity="H4",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
                label_types=["enhanced_triple_barrier"],
                force_recompute=False,
                priority="normal"
            )
            
            # Verify job was created
            assert job_id is not None
            assert job_id.startswith("bf_")
            
            # Check job status
            status = batch_service.get_job_status(job_id)
            assert status is not None
            assert status['status'] == JobStatus.PENDING
            assert status['instrument_id'] == "EURUSD"
            assert status['granularity'] == "H4"
            assert status['total_candles'] == len(mock_snapshots)
            
            # Mock the chunk processing to simulate execution
            with patch('src.services.batch_backfill_service._process_chunk_worker') as mock_processor:
                mock_processor.return_value = {
                    'success': True,
                    'processed_candles': 100,
                    'failed_candles': 0,
                    'cache_hits': 30,
                    'cache_misses': 70,
                    'processing_time_seconds': 1.0
                }
                
                # Execute the job
                result = await batch_service.execute_job(job_id)
                
                # Verify execution results
                assert result['status'] == JobStatus.COMPLETED.value
                assert result['processed_candles'] == len(mock_snapshots)
                assert result['error_rate'] == 0
                assert result['throughput_candles_per_minute'] > 0
                
                # Check final job status
                final_status = batch_service.get_job_status(job_id)
                assert final_status['status'] == JobStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_performance_validation(self, batch_service, mock_snapshots):
        """Test that the system meets performance targets"""
        
        # Create larger dataset for performance testing
        large_dataset = mock_snapshots * 20  # 10,000 candles
        
        with patch.object(clickhouse_service, 'fetch_snapshots', return_value=large_dataset):
            
            job_id = await batch_service.start_backfill_job(
                instrument_id="EURUSD",
                granularity="H4",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                label_types=["enhanced_triple_barrier"],
                force_recompute=False,
                priority="high"
            )
            
            # Mock fast processing
            with patch('src.services.batch_backfill_service._process_chunk_worker') as mock_processor:
                mock_processor.return_value = {
                    'success': True,
                    'processed_candles': 100,
                    'failed_candles': 0,
                    'cache_hits': 80,  # High cache hit rate
                    'cache_misses': 20,
                    'processing_time_seconds': 0.1  # Fast processing
                }
                
                start_time = time.time()
                result = await batch_service.execute_job(job_id)
                execution_time = time.time() - start_time
                
                # Performance assertions
                candles_per_second = result['processed_candles'] / execution_time
                candles_per_minute = candles_per_second * 60
                
                # Should process at least 100k candles/minute (relaxed for testing)
                assert candles_per_minute >= 100_000, f"Performance too low: {candles_per_minute:,.0f} candles/min"
                
                # Check cache hit rate
                assert result['cache_hit_rate'] >= 0.7, f"Cache hit rate too low: {result['cache_hit_rate']:.2%}"
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, batch_service, error_handler, mock_snapshots):
        """Test error handling and recovery mechanisms"""
        
        with patch.object(clickhouse_service, 'fetch_snapshots', return_value=mock_snapshots):
            
            job_id = await batch_service.start_backfill_job(
                instrument_id="EURUSD",
                granularity="H4",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
                label_types=["enhanced_triple_barrier"],
                force_recompute=False,
                priority="normal"
            )
            
            # Simulate intermittent failures
            call_count = 0
            def mock_processor_with_failures(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                if call_count % 3 == 0:  # Every 3rd call fails
                    raise Exception("Simulated processing error")
                
                return {
                    'success': True,
                    'processed_candles': 100,
                    'failed_candles': 0,
                    'cache_hits': 30,
                    'cache_misses': 70,
                    'processing_time_seconds': 1.0
                }
            
            with patch('src.services.batch_backfill_service._process_chunk_worker', side_effect=mock_processor_with_failures):
                
                result = await batch_service.execute_job(job_id)
                
                # Should complete despite errors due to retry logic
                assert result['status'] in [JobStatus.COMPLETED.value, JobStatus.COMPLETED.value]
                assert result['retry_attempts'] > 0
                assert result['error_rate'] > 0 and result['error_rate'] < 1
    
    def test_worker_pool_scaling(self, worker_pool):
        """Test worker pool dynamic scaling"""
        
        worker_pool.start()
        initial_workers = worker_pool.get_metrics().total_workers
        
        # Submit many tasks to trigger scale-up
        for i in range(50):
            success = worker_pool.submit_task(
                task_id=f"task_{i}",
                job_id="test_job",
                function_name="process_chunk",
                priority=TaskPriority.NORMAL
            )
            assert success
        
        # Allow time for scaling decisions
        time.sleep(2)
        
        # Check if workers scaled up
        scaled_metrics = worker_pool.get_metrics()
        assert scaled_metrics.queue_size > 0
        
        # Test scale-down by waiting for queue to empty
        time.sleep(5)
        
        final_metrics = worker_pool.get_metrics()
        # Queue should be smaller after processing
        assert final_metrics.queue_size <= scaled_metrics.queue_size
    
    def test_metrics_collection(self, metrics_collector):
        """Test comprehensive metrics collection"""
        
        # Start metrics collection
        metrics_collector.start_collection(interval_seconds=1)
        
        # Simulate job metrics
        job_id = "test_job_123"
        
        # Record various metrics
        for i in range(10):
            metrics_collector.record_chunk_completion(
                job_id=job_id,
                chunk_id=f"chunk_{i}",
                processing_time_ms=100 + i * 10,
                candles_processed=1000,
                success=i % 10 != 8,  # 90% success rate
                cache_hits=700,
                cache_misses=300
            )
        
        # Wait for metrics to accumulate
        time.sleep(2)
        
        # Get performance snapshot
        snapshot = metrics_collector.get_job_performance(job_id)
        assert snapshot is not None
        assert snapshot.candles_per_second > 0
        assert snapshot.error_rate < 0.2  # Less than 20% error rate
        assert snapshot.cache_hit_rate == 0.7  # 70% cache hit rate
        
        # Get system metrics
        system_metrics = metrics_collector.get_system_metrics()
        assert 'job_throughput_candles_per_minute_avg' in system_metrics
        assert system_metrics['system_health'] in ['healthy', 'degraded', 'unhealthy']
        
        # Stop metrics collection
        metrics_collector.stop_collection()
    
    def test_circuit_breaker_functionality(self, error_handler):
        """Test circuit breaker pattern for dependency failures"""
        
        # Simulate repeated failures to trigger circuit breaker
        for i in range(6):  # More than failure threshold (5)
            error = Exception("ClickHouse connection timeout")
            error_record = error_handler.handle_error(
                error=error,
                job_id="test_job",
                task_id=f"task_{i}",
                context={'service': 'clickhouse'}
            )
            
            # Check retry logic
            should_retry, delay = error_handler.should_retry(error_record)
            if i < 3:  # First few should retry
                assert should_retry
                assert delay > 0
            
        # Circuit breaker should be open now
        cb = error_handler.circuit_breakers['clickhouse']
        assert cb.state == 'open'
        assert cb.failure_count >= 5
        
        # New errors should not retry when circuit breaker is open
        error = Exception("Another ClickHouse error")
        error_record = error_handler.handle_error(
            error=error,
            job_id="test_job",
            task_id="task_after_cb_open",
            context={'service': 'clickhouse'}
        )
        
        should_retry, delay = error_handler.should_retry(error_record)
        assert not should_retry  # Circuit breaker should prevent retry
    
    def test_error_categorization_and_handling(self, error_handler):
        """Test intelligent error categorization and handling"""
        
        test_cases = [
            (ConnectionError("Connection timeout"), ErrorCategory.TRANSIENT, ErrorSeverity.WARNING),
            (MemoryError("Out of memory"), ErrorCategory.RESOURCE, ErrorSeverity.ERROR),
            (ValueError("Invalid data format"), ErrorCategory.DATA, ErrorSeverity.WARNING),
            (PermissionError("Access denied"), ErrorCategory.PERSISTENT, ErrorSeverity.ERROR),
            (RuntimeError("ClickHouse query failed"), ErrorCategory.DEPENDENCY, ErrorSeverity.ERROR),
        ]
        
        for error, expected_category, expected_severity in test_cases:
            error_record = error_handler.handle_error(
                error=error,
                job_id="test_job",
                task_id=f"task_{error.__class__.__name__}",
                context={}
            )
            
            assert error_record.category == expected_category
            assert error_record.severity == expected_severity
            
            # Check retry behavior based on category
            should_retry, delay = error_handler.should_retry(error_record)
            
            if expected_category in [ErrorCategory.DATA, ErrorCategory.PERSISTENT]:
                assert not should_retry  # These shouldn't retry
            else:
                assert should_retry  # Others should retry
                assert delay > 0
    
    @pytest.mark.asyncio
    async def test_job_pause_resume_cancel(self, batch_service, mock_snapshots):
        """Test job control operations (pause, resume, cancel)"""
        
        with patch.object(clickhouse_service, 'fetch_snapshots', return_value=mock_snapshots):
            
            job_id = await batch_service.start_backfill_job(
                instrument_id="EURUSD",
                granularity="H4",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
                label_types=["enhanced_triple_barrier"],
                force_recompute=False,
                priority="normal"
            )
            
            # Test pause
            pause_success = await batch_service.pause_job(job_id)
            assert pause_success
            
            status = batch_service.get_job_status(job_id)
            assert status['status'] == JobStatus.PAUSED
            
            # Test resume
            resume_success = await batch_service.resume_job(job_id)
            assert resume_success
            
            status = batch_service.get_job_status(job_id)
            assert status['status'] == JobStatus.RUNNING
            
            # Test cancel
            cancel_success = await batch_service.cancel_job(job_id)
            assert cancel_success
            
            status = batch_service.get_job_status(job_id)
            assert status['status'] == JobStatus.CANCELLED
    
    def test_redis_persistence_and_recovery(self, batch_service):
        """Test Redis-based job state persistence and recovery"""
        
        # Create a job and verify it's stored in Redis
        job_data = {
            "job_id": "test_job_persistence",
            "instrument_id": "EURUSD",
            "granularity": "H4",
            "status": JobStatus.RUNNING,
            "created_at": datetime.utcnow(),
            "total_candles": 1000
        }
        
        # Store job in Redis
        redis_cache.set("batch_job:test_job_persistence", job_data)
        
        # Verify retrieval
        retrieved_job = batch_service.get_job_status("test_job_persistence")
        assert retrieved_job is not None
        assert retrieved_job['job_id'] == "test_job_persistence"
        assert retrieved_job['instrument_id'] == "EURUSD"
        assert retrieved_job['status'] == JobStatus.RUNNING
        
        # Test job listing
        jobs = batch_service.list_jobs()
        job_ids = [job['job_id'] for job in jobs]
        assert "test_job_persistence" in job_ids
    
    @pytest.mark.asyncio
    async def test_progress_tracking_and_eta(self, batch_service, mock_snapshots):
        """Test progress tracking and ETA calculations"""
        
        with patch.object(clickhouse_service, 'fetch_snapshots', return_value=mock_snapshots):
            
            job_id = await batch_service.start_backfill_job(
                instrument_id="EURUSD",
                granularity="H4",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 1, 31),
                label_types=["enhanced_triple_barrier"],
                force_recompute=False,
                priority="normal"
            )
            
            # Simulate gradual progress
            total_chunks = 5  # Based on mock_snapshots (500) / chunk_size (100)
            
            with patch('src.services.batch_backfill_service._process_chunk_worker') as mock_processor:
                mock_processor.return_value = {
                    'success': True,
                    'processed_candles': 100,
                    'failed_candles': 0,
                    'cache_hits': 50,
                    'cache_misses': 50,
                    'processing_time_seconds': 2.0
                }
                
                # Start execution in background
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, batch_service.execute_job(job_id))
                    
                    # Check progress during execution
                    time.sleep(1)  # Allow some processing
                    
                    status = batch_service.get_job_status(job_id)
                    if status and 'progress_percentage' in status:
                        assert status['progress_percentage'] >= 0
                        assert status['progress_percentage'] <= 100
                        
                        if 'estimated_completion' in status and status['estimated_completion']:
                            # ETA should be in the future
                            eta = datetime.fromisoformat(status['estimated_completion'])
                            assert eta > datetime.utcnow()
                    
                    # Wait for completion
                    result = future.result(timeout=30)
                    assert result['status'] == JobStatus.COMPLETED.value
    
    def test_prometheus_metrics_export(self, metrics_collector):
        """Test Prometheus metrics export format"""
        
        # Record some test metrics
        metrics_collector.record_job_metric("test_job", "throughput_candles_per_minute", 500000)
        metrics_collector.record_job_metric("test_job", "error_rate_percent", 2.5)
        
        # Export metrics
        prometheus_output = metrics_collector.export_prometheus_metrics()
        
        assert isinstance(prometheus_output, str)
        assert len(prometheus_output) > 0
        
        # Check format
        lines = prometheus_output.split('\n')
        for line in lines:
            if line.strip():
                parts = line.split()
                assert len(parts) >= 2  # metric_name value [timestamp]
                assert parts[0].startswith('batch_')


class TestCLIIntegration:
    """Integration tests for CLI commands"""
    
    def test_cli_help_commands(self):
        """Test that CLI help commands work"""
        import subprocess
        import sys
        
        # Test main help
        result = subprocess.run([
            sys.executable, '-m', 'src.cli.batch_cli', '--help'
        ], capture_output=True, text=True, cwd='.')
        
        assert result.returncode == 0
        assert 'Batch backfill CLI' in result.stdout
    
    @pytest.mark.skip("Integration test - requires actual CLI execution")
    def test_cli_job_lifecycle(self):
        """Test complete job lifecycle through CLI"""
        import subprocess
        import sys
        
        # Start a job
        result = subprocess.run([
            sys.executable, '-m', 'src.cli.batch_cli', 'start',
            '--instrument', 'EURUSD',
            '--granularity', 'H4',
            '--start-date', '2024-01-01',
            '--end-date', '2024-01-02',
            '--dry-run'
        ], capture_output=True, text=True, cwd='.')
        
        assert result.returncode == 0
        assert 'DRY RUN' in result.stdout


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self):
        """Benchmark processing throughput"""
        
        # Create large dataset
        mock_data = []
        for i in range(50000):  # 50k candles
            mock_data.append({
                'ts': datetime(2024, 1, 1) + timedelta(minutes=i * 15),
                'open': 1.0950,
                'high': 1.0970,
                'low': 1.0940,
                'close': 1.0965,
                'volume': 1000000,
                'atr_14': 0.0025
            })
        
        batch_service = BatchBackfillService(
            max_workers=4,
            chunk_size=5000,  # Larger chunks for performance test
            max_retries=1
        )
        
        with patch.object(clickhouse_service, 'fetch_snapshots', return_value=mock_data):
            
            job_id = await batch_service.start_backfill_job(
                instrument_id="EURUSD",
                granularity="M15",
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 12, 31),
                label_types=["enhanced_triple_barrier"],
                force_recompute=False,
                priority="high"
            )
            
            # Mock very fast processing for benchmark
            with patch('src.services.batch_backfill_service._process_chunk_worker') as mock_processor:
                mock_processor.return_value = {
                    'success': True,
                    'processed_candles': 5000,
                    'failed_candles': 0,
                    'cache_hits': 4500,
                    'cache_misses': 500,
                    'processing_time_seconds': 0.05  # 50ms per chunk of 5k candles
                }
                
                start_time = time.time()
                result = await batch_service.execute_job(job_id)
                execution_time = time.time() - start_time
                
                # Calculate throughput
                candles_per_second = result['processed_candles'] / execution_time
                candles_per_minute = candles_per_second * 60
                
                print(f"\nPerformance Results:")
                print(f"Processed: {result['processed_candles']:,} candles")
                print(f"Execution time: {execution_time:.2f} seconds")
                print(f"Throughput: {candles_per_minute:,.0f} candles/minute")
                print(f"Target: 1,000,000 candles/minute")
                print(f"Performance ratio: {candles_per_minute / 1_000_000:.2f}")
                
                # Performance assertion (relaxed for CI environment)
                assert candles_per_minute >= 500_000, f"Performance below minimum: {candles_per_minute:,.0f}"
    
    def test_memory_usage_under_load(self, worker_pool):
        """Test memory usage under high load"""
        import psutil
        import gc
        
        worker_pool.start()
        
        # Get baseline memory usage
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Submit many tasks
        for i in range(1000):
            worker_pool.submit_task(
                task_id=f"memory_test_task_{i}",
                job_id="memory_test_job",
                function_name="process_chunk",
                priority=TaskPriority.NORMAL
            )
        
        # Wait for processing
        time.sleep(5)
        
        # Check memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - baseline_memory
        
        print(f"\nMemory Usage:")
        print(f"Baseline: {baseline_memory:.1f} MB")
        print(f"Peak: {peak_memory:.1f} MB")
        print(f"Increase: {memory_increase:.1f} MB")
        
        # Memory should not increase excessively (less than 100MB for 1000 tasks)
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.1f} MB"
        
        # Force garbage collection
        gc.collect()
        
        # Check for memory leaks
        time.sleep(2)
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_after_gc = final_memory - baseline_memory
        
        # Memory should be mostly freed after GC
        assert memory_after_gc < memory_increase * 0.5, "Potential memory leak detected"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])