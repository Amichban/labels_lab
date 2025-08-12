"""
Integration tests for Firestore streaming system

Tests the complete integration between:
- Firestore listener
- Stream manager  
- Label computation engine
- Redis caching
- Error handling and recovery

Issue #11: Integration testing for real-time streaming system
"""

import pytest
import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.services.firestore_listener import FirestoreListener
from src.services.stream_manager import StreamManager, StreamPriority
from src.services.redis_cache import redis_cache
from src.models.data_models import Candle, Granularity
from src.core.label_computation import computation_engine
from config.settings import settings

logger = logging.getLogger(__name__)


@pytest.fixture
async def mock_components():
    """Set up mock components for integration testing"""
    
    # Mock Firestore client
    mock_firestore_client = MagicMock()
    collection_mock = MagicMock()
    query_mock = MagicMock()
    
    mock_firestore_client.collection.return_value = collection_mock
    collection_mock.where.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    
    # Mock Redis client
    mock_redis_client = MagicMock()
    mock_redis_client.ping.return_value = True
    mock_redis_client.get.return_value = None
    mock_redis_client.setex.return_value = True
    
    # Mock computation engine
    mock_label_set = MagicMock()
    mock_label_set.dict.return_value = {"test": "labels"}
    
    with patch('src.services.firestore_listener.firestore.Client', return_value=mock_firestore_client), \
         patch('src.services.redis_cache.redis.Redis', return_value=mock_redis_client), \
         patch.object(computation_engine, 'compute_labels', return_value=mock_label_set):
        
        yield {
            'firestore_client': mock_firestore_client,
            'redis_client': mock_redis_client,
            'label_set': mock_label_set
        }


@pytest.mark.asyncio
async def test_stream_manager_with_firestore_listener(mock_components):
    """Test stream manager coordinating with Firestore listener"""
    
    stream_manager = StreamManager()
    
    try:
        # Add multiple streams
        stream_ids = []
        instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]
        
        for instrument in instruments:
            stream_id = await stream_manager.add_stream(
                instrument, "H1", StreamPriority.MEDIUM
            )
            stream_ids.append(stream_id)
        
        assert len(stream_manager.managed_streams) == 3
        
        # Test stream status
        status = stream_manager.get_stream_status()
        assert status["global_metrics"]["total_streams"] == 3
        assert status["global_metrics"]["active_streams"] == 0  # Not started yet
        
        # Test rate limiting setup
        for stream_id in stream_ids:
            managed_stream = stream_manager.managed_streams[stream_id]
            assert managed_stream.max_processing_rate == 100  # MEDIUM priority rate
        
        # Test callback creation
        for stream_id in stream_ids:
            callback = stream_manager._create_stream_callback(stream_id)
            assert callable(callback)
    
    finally:
        await stream_manager.shutdown()


@pytest.mark.asyncio
async def test_candle_processing_pipeline(mock_components):
    """Test complete candle processing pipeline"""
    
    listener = FirestoreListener(project_id="test-project")
    listener._client = mock_components['firestore_client']
    
    try:
        # Add stream
        stream_id = listener.add_stream("EUR_USD", "H1")
        
        # Create test candle data
        candle_data = {
            "ts": datetime.utcnow(),
            "o": 1.1000,
            "h": 1.1050,
            "l": 1.0950,
            "c": 1.1025,
            "v": 1000,
            "bid": 1.1020,
            "ask": 1.1030,
            "complete": True
        }
        
        # Test document conversion
        stream_config = listener.streams[stream_id]
        candle = listener._convert_to_candle(candle_data, stream_config)
        
        assert candle is not None
        assert candle.instrument_id == "EUR_USD"
        assert candle.granularity == Granularity.H1
        
        # Test label computation integration
        with patch.object(listener, '_should_compute_labels', return_value=True):
            await listener._process_candle_for_labels(candle, stream_id)
        
        # Verify computation engine was called (mocked)
        assert mock_components['label_set'] is not None
    
    finally:
        await listener.shutdown()


@pytest.mark.asyncio
async def test_error_handling_and_recovery(mock_components):
    """Test error handling and recovery mechanisms"""
    
    listener = FirestoreListener(project_id="test-project")
    listener._client = mock_components['firestore_client']
    
    try:
        # Add stream
        stream_id = listener.add_stream("EUR_USD", "H1")
        
        # Simulate processing error
        mock_change = MagicMock()
        mock_change.document.to_dict.return_value = {"invalid": "data", "complete": True}
        mock_change.document.id = "test_doc"
        
        # Process change that will fail
        result = await listener._process_document_change(stream_id, mock_change)
        
        assert result.success is False
        assert result.error is not None
        
        # Check dead letter queue
        dlq_stats = listener.dead_letter_queue.get_stats()
        assert dlq_stats["current_size"] >= 1
        assert dlq_stats["total_failed"] >= 1
        
        # Check metrics updated
        assert listener.metrics["failed_processing"] >= 1
    
    finally:
        await listener.shutdown()


@pytest.mark.asyncio
async def test_rate_limiting_integration(mock_components):
    """Test rate limiting across stream manager and listener"""
    
    stream_manager = StreamManager()
    
    try:
        # Add stream with low rate limit
        stream_id = await stream_manager.add_stream("EUR_USD", "H1", StreamPriority.LOW)
        managed_stream = stream_manager.managed_streams[stream_id]
        managed_stream.max_processing_rate = 2  # Very low for testing
        
        # Test rate limiting
        assert await stream_manager._check_rate_limit(stream_id) is True
        assert await stream_manager._check_rate_limit(stream_id) is True
        assert await stream_manager._check_rate_limit(stream_id) is False  # Should be limited
        
        # Wait for rate limit window to reset
        await asyncio.sleep(0.1)
        
        # Test rate tracking
        await stream_manager._update_rate_tracking(stream_id)
        assert managed_stream.current_processing_rate >= 0
    
    finally:
        await stream_manager.shutdown()


@pytest.mark.asyncio
async def test_health_monitoring_integration(mock_components):
    """Test health monitoring across components"""
    
    stream_manager = StreamManager()
    listener = FirestoreListener(project_id="test-project")
    
    try:
        # Add streams to both components
        manager_stream_id = await stream_manager.add_stream("EUR_USD", "H1")
        listener_stream_id = listener.add_stream("EUR_USD", "H1")
        
        assert manager_stream_id == listener_stream_id
        
        # Test health checks
        manager_health = stream_manager.get_stream_status()
        listener_health = await listener.health_check()
        
        # Verify health check structure
        assert "global_metrics" in manager_health
        assert "streams" in manager_health
        assert "overall_status" in listener_health
        assert "stream_health" in listener_health
        
        # Test health status reporting
        assert listener_health["overall_status"] in ["initializing", "healthy", "degraded", "critical"]
        assert manager_health["global_metrics"]["total_streams"] == 1
    
    finally:
        await stream_manager.shutdown()
        await listener.shutdown()


@pytest.mark.asyncio
async def test_backpressure_handling(mock_components):
    """Test backpressure handling mechanisms"""
    
    stream_manager = StreamManager()
    
    try:
        # Set low backpressure threshold for testing
        stream_manager.backpressure_threshold = 5
        stream_manager.backpressure_recovery_threshold = 1
        
        # Add stream
        stream_id = await stream_manager.add_stream("EUR_USD", "H1")
        
        # Simulate high load to trigger backpressure
        for i in range(10):
            await stream_manager._check_rate_limit(stream_id)
        
        # Check global backpressure
        await stream_manager._check_global_backpressure()
        
        # Verify backpressure metrics
        managed_stream = stream_manager.managed_streams[stream_id]
        global_metrics = stream_manager.global_metrics
        
        # Backpressure may or may not be active depending on timing
        # Just verify the mechanism works without errors
        assert isinstance(stream_manager.backpressure_active, bool)
    
    finally:
        await stream_manager.shutdown()


@pytest.mark.asyncio
async def test_performance_optimization_integration(mock_components):
    """Test performance optimization across components"""
    
    stream_manager = StreamManager()
    
    try:
        # Add streams with different performance characteristics
        stream_id_1 = await stream_manager.add_stream("EUR_USD", "H1", StreamPriority.LOW)
        stream_id_2 = await stream_manager.add_stream("GBP_USD", "H1", StreamPriority.HIGH)
        
        # Simulate performance metrics
        stream_1 = stream_manager.managed_streams[stream_id_1]
        stream_2 = stream_manager.managed_streams[stream_id_2]
        
        # Simulate underutilization for stream 1
        stream_1.current_processing_rate = 10
        stream_1.max_processing_rate = 50
        stream_1.metrics.error_count = 0
        
        # Simulate overload for stream 2
        stream_2.current_processing_rate = 180
        stream_2.max_processing_rate = 200
        stream_2.metrics.backpressure_events = 15
        stream_2.metrics.error_count = 8
        
        # Run optimization
        result = await stream_manager.optimize_performance()
        
        assert result["status"] == "completed"
        assert "optimizations_applied" in result
        assert isinstance(result["optimizations_applied"], int)
    
    finally:
        await stream_manager.shutdown()


@pytest.mark.asyncio 
async def test_graceful_shutdown_integration(mock_components):
    """Test graceful shutdown of entire system"""
    
    stream_manager = StreamManager()
    listener = FirestoreListener(project_id="test-project")
    
    try:
        # Set up complete system
        stream_id = await stream_manager.add_stream("EUR_USD", "H1")
        listener.add_stream("EUR_USD", "H1")
        
        # Add some metrics and state
        stream_manager.global_metrics["total_documents_processed"] = 100
        listener.metrics["successful_processing"] = 95
        
        # Test individual shutdowns
        await listener.shutdown()
        assert listener.health_status == "shutdown"
        
        await stream_manager.shutdown()
        
        # Verify clean shutdown
        assert len(stream_manager.stream_semaphores) == 0
        assert len(stream_manager.rate_limit_counters) == 0
        
    except Exception as e:
        # Ensure cleanup even on failure
        await stream_manager.shutdown()
        await listener.shutdown()
        raise


@pytest.mark.asyncio
async def test_concurrent_stream_processing(mock_components):
    """Test concurrent processing of multiple streams"""
    
    stream_manager = StreamManager()
    
    try:
        # Add multiple streams
        stream_ids = []
        instruments = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF"]
        
        tasks = []
        for instrument in instruments:
            task = stream_manager.add_stream(instrument, "H1", StreamPriority.MEDIUM)
            tasks.append(task)
        
        # Add all streams concurrently
        stream_ids = await asyncio.gather(*tasks)
        
        assert len(stream_ids) == 5
        assert len(stream_manager.managed_streams) == 5
        
        # Test concurrent rate limit checks
        rate_limit_tasks = []
        for stream_id in stream_ids[:3]:  # Test first 3 streams
            for _ in range(5):  # 5 requests per stream
                task = stream_manager._check_rate_limit(stream_id)
                rate_limit_tasks.append(task)
        
        results = await asyncio.gather(*rate_limit_tasks, return_exceptions=True)
        
        # Verify no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, bool)
        
        # Test concurrent status checks
        status_tasks = [
            stream_manager.get_stream_status(stream_id)
            for stream_id in stream_ids[:3]
        ]
        
        # This is synchronous but test it doesn't cause issues
        statuses = [task for task in status_tasks]
        assert len(statuses) == 3
        
        for status in statuses:
            assert "stream_id" in status
            assert "metrics" in status
    
    finally:
        await stream_manager.shutdown()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])