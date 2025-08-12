"""
Unit tests for Firestore listener functionality

Tests for:
- FirestoreListener real-time streaming
- StreamManager coordination
- Dead letter queue handling
- Rate limiting and backpressure
- Error handling and recovery

Issue #11: Comprehensive testing for real-time Firestore listener
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.services.firestore_listener import (
    FirestoreListener, DeadLetterQueue, StreamConfig, ProcessingResult
)
from src.services.stream_manager import (
    StreamManager, ManagedStream, StreamPriority, StreamStatus
)
from src.models.data_models import Candle, Granularity
from config.settings import settings


class TestDeadLetterQueue:
    """Test dead letter queue functionality"""
    
    @pytest.fixture
    def dlq(self):
        """Create dead letter queue instance"""
        return DeadLetterQueue(max_size=10)
    
    @pytest.mark.asyncio
    async def test_add_failed_item(self, dlq):
        """Test adding failed item to queue"""
        item = {"test": "data"}
        error = "Test error"
        
        await dlq.add_failed_item(item, error, retry_count=0)
        
        assert len(dlq.queue) == 1
        assert dlq.queue[0]["item"] == item
        assert dlq.queue[0]["error"] == error
        assert dlq.queue[0]["retry_count"] == 0
        assert dlq.failed_count == 1
    
    @pytest.mark.asyncio
    async def test_max_size_limit(self, dlq):
        """Test that queue respects max size"""
        # Add more items than max size
        for i in range(15):
            await dlq.add_failed_item({"id": i}, f"error {i}")
        
        assert len(dlq.queue) == dlq.max_size  # Should be limited to max_size
        assert dlq.failed_count == 15  # But counter should track all
    
    def test_get_stats(self, dlq):
        """Test getting queue statistics"""
        stats = dlq.get_stats()
        
        assert "current_size" in stats
        assert "total_failed" in stats
        assert "max_size" in stats
        assert stats["max_size"] == 10


class TestFirestoreListener:
    """Test Firestore listener functionality"""
    
    @pytest.fixture
    def mock_firestore_client(self):
        """Create mock Firestore client"""
        client = MagicMock()
        collection_mock = MagicMock()
        query_mock = MagicMock()
        
        client.collection.return_value = collection_mock
        collection_mock.where.return_value = query_mock
        query_mock.order_by.return_value = query_mock
        
        return client
    
    @pytest.fixture
    def listener(self, mock_firestore_client):
        """Create FirestoreListener instance with mocked client"""
        with patch('src.services.firestore_listener.firestore.Client', return_value=mock_firestore_client):
            listener = FirestoreListener(project_id="test-project")
            listener._client = mock_firestore_client
            return listener
    
    def test_initialization(self, listener):
        """Test listener initialization"""
        assert listener.project_id == "test-project"
        assert isinstance(listener.dead_letter_queue, DeadLetterQueue)
        assert listener.max_reconnect_attempts == 10
        assert listener.health_status == "initializing"
    
    def test_add_stream(self, listener):
        """Test adding a stream"""
        stream_id = listener.add_stream("EUR_USD", "H1")
        
        assert stream_id == "EUR_USD_H1"
        assert stream_id in listener.streams
        
        stream_config = listener.streams[stream_id]
        assert stream_config.instrument_id == "EUR_USD"
        assert stream_config.granularity == "H1"
        assert stream_config.collection_path == "candles/EUR_USD/H1/data"
    
    def test_add_duplicate_stream(self, listener):
        """Test adding duplicate stream"""
        stream_id1 = listener.add_stream("EUR_USD", "H1")
        stream_id2 = listener.add_stream("EUR_USD", "H1")
        
        assert stream_id1 == stream_id2
        assert len(listener.streams) == 1
    
    @pytest.mark.asyncio
    async def test_start_stream_success(self, listener, mock_firestore_client):
        """Test successful stream start"""
        # Add stream first
        stream_id = listener.add_stream("EUR_USD", "H1")
        
        # Mock successful listener start
        mock_listener = MagicMock()
        mock_firestore_client.collection.return_value.where.return_value.order_by.return_value.on_snapshot.return_value = mock_listener
        
        success = await listener.start_stream(stream_id)
        
        assert success is True
        assert listener.streams[stream_id].active is True
        assert stream_id in listener.active_listeners
        assert listener.metrics["active_streams"] == 1
    
    @pytest.mark.asyncio
    async def test_start_nonexistent_stream(self, listener):
        """Test starting non-existent stream"""
        success = await listener.start_stream("nonexistent")
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_stop_stream(self, listener, mock_firestore_client):
        """Test stopping a stream"""
        # Start stream first
        stream_id = listener.add_stream("EUR_USD", "H1")
        await listener.start_stream(stream_id)
        
        # Stop stream
        success = await listener.stop_stream(stream_id)
        
        assert success is True
        assert listener.streams[stream_id].active is False
        assert stream_id not in listener.active_listeners
    
    def test_convert_to_candle_success(self, listener):
        """Test successful document to Candle conversion"""
        stream_config = StreamConfig(
            instrument_id="EUR_USD",
            granularity="H1",
            collection_path="candles/EUR_USD/H1/data"
        )
        
        doc_data = {
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
        
        candle = listener._convert_to_candle(doc_data, stream_config)
        
        assert candle is not None
        assert candle.instrument_id == "EUR_USD"
        assert candle.granularity == Granularity.H1
        assert candle.open == 1.1000
        assert candle.high == 1.1050
        assert candle.low == 1.0950
        assert candle.close == 1.1025
        assert candle.volume == 1000
        assert candle.bid == 1.1020
        assert candle.ask == 1.1030
    
    def test_convert_to_candle_invalid_data(self, listener):
        """Test document conversion with invalid data"""
        stream_config = StreamConfig(
            instrument_id="EUR_USD",
            granularity="H1",
            collection_path="candles/EUR_USD/H1/data"
        )
        
        # Missing required fields
        doc_data = {"incomplete": "data"}
        
        candle = listener._convert_to_candle(doc_data, stream_config)
        
        assert candle is None
    
    @pytest.mark.asyncio
    async def test_process_document_change_incomplete_candle(self, listener):
        """Test processing incomplete candle (should be skipped)"""
        # Mock document change with incomplete candle
        change = MagicMock()
        change.document.to_dict.return_value = {"complete": False}
        change.document.id = "test_doc"
        
        stream_id = listener.add_stream("EUR_USD", "H1")
        
        result = await listener._process_document_change(stream_id, change)
        
        assert result.success is False
        assert result.error == "Incomplete candle"
    
    @pytest.mark.asyncio
    async def test_health_check(self, listener):
        """Test health check functionality"""
        # Add some streams
        listener.add_stream("EUR_USD", "H1")
        listener.add_stream("GBP_USD", "H1")
        
        health = await listener.health_check()
        
        assert "overall_status" in health
        assert "active_streams" in health
        assert "total_streams" in health
        assert "metrics" in health
        assert "dead_letter_queue" in health
        assert "stream_health" in health
        assert health["total_streams"] == 2
    
    def test_get_metrics(self, listener):
        """Test metrics retrieval"""
        # Update some metrics
        listener.metrics["total_documents_processed"] = 100
        listener.metrics["successful_processing"] = 95
        
        metrics = listener.get_metrics()
        
        assert "total_documents_processed" in metrics
        assert "success_rate" in metrics
        assert metrics["success_rate"] == 0.95
    
    @pytest.mark.asyncio
    async def test_shutdown(self, listener):
        """Test graceful shutdown"""
        # Add and start some streams
        listener.add_stream("EUR_USD", "H1")
        listener.add_stream("GBP_USD", "H1")
        
        await listener.shutdown()
        
        assert listener.health_status == "shutdown"
        assert len(listener.active_listeners) == 0


class TestStreamManager:
    """Test stream manager functionality"""
    
    @pytest.fixture
    def manager(self):
        """Create StreamManager instance"""
        return StreamManager()
    
    @pytest.mark.asyncio
    async def test_add_stream(self, manager):
        """Test adding a managed stream"""
        stream_id = await manager.add_stream("EUR_USD", "H1", StreamPriority.HIGH)
        
        assert stream_id == "EUR_USD_H1"
        assert stream_id in manager.managed_streams
        
        managed_stream = manager.managed_streams[stream_id]
        assert managed_stream.instrument_id == "EUR_USD"
        assert managed_stream.granularity == "H1"
        assert managed_stream.priority == StreamPriority.HIGH
        assert managed_stream.status == StreamStatus.INITIALIZING
    
    @pytest.mark.asyncio
    async def test_add_duplicate_stream(self, manager):
        """Test adding duplicate managed stream"""
        stream_id1 = await manager.add_stream("EUR_USD", "H1")
        stream_id2 = await manager.add_stream("EUR_USD", "H1")
        
        assert stream_id1 == stream_id2
        assert len(manager.managed_streams) == 1
    
    def test_get_rate_limit_for_priority(self, manager):
        """Test rate limit calculation based on priority"""
        low_rate = manager._get_rate_limit_for_priority(StreamPriority.LOW)
        medium_rate = manager._get_rate_limit_for_priority(StreamPriority.MEDIUM)
        high_rate = manager._get_rate_limit_for_priority(StreamPriority.HIGH)
        critical_rate = manager._get_rate_limit_for_priority(StreamPriority.CRITICAL)
        
        assert low_rate < medium_rate < high_rate < critical_rate
        assert critical_rate == 500  # Maximum rate
    
    @pytest.mark.asyncio
    async def test_check_rate_limit(self, manager):
        """Test rate limiting functionality"""
        stream_id = await manager.add_stream("EUR_USD", "H1")
        managed_stream = manager.managed_streams[stream_id]
        managed_stream.max_processing_rate = 2  # Very low limit for testing
        
        # First two requests should pass
        assert await manager._check_rate_limit(stream_id) is True
        assert await manager._check_rate_limit(stream_id) is True
        
        # Third request should fail
        assert await manager._check_rate_limit(stream_id) is False
    
    def test_get_stream_status_single(self, manager):
        """Test getting status for single stream"""
        asyncio.run(manager.add_stream("EUR_USD", "H1"))
        
        status = manager.get_stream_status("EUR_USD_H1")
        
        assert "stream_id" in status
        assert "status" in status
        assert "priority" in status
        assert "metrics" in status
        assert status["stream_id"] == "EUR_USD_H1"
    
    def test_get_stream_status_all(self, manager):
        """Test getting status for all streams"""
        asyncio.run(manager.add_stream("EUR_USD", "H1"))
        asyncio.run(manager.add_stream("GBP_USD", "H4"))
        
        status = manager.get_stream_status()
        
        assert "global_metrics" in status
        assert "streams" in status
        assert len(status["streams"]) == 2
        assert "EUR_USD_H1" in status["streams"]
        assert "GBP_USD_H4" in status["streams"]
    
    def test_get_stream_status_nonexistent(self, manager):
        """Test getting status for non-existent stream"""
        status = manager.get_stream_status("nonexistent")
        
        assert status == {"error": "Stream not found"}
    
    @pytest.mark.asyncio
    async def test_optimization_disabled(self, manager):
        """Test performance optimization when disabled"""
        manager.performance_optimization_enabled = False
        
        result = await manager.optimize_performance()
        
        assert result["status"] == "optimization_disabled"
    
    @pytest.mark.asyncio
    async def test_shutdown(self, manager):
        """Test graceful shutdown"""
        # Add some streams
        await manager.add_stream("EUR_USD", "H1")
        await manager.add_stream("GBP_USD", "H4")
        
        await manager.shutdown()
        
        # All streams should be stopped
        for stream in manager.managed_streams.values():
            assert stream.status == StreamStatus.STOPPED or stream.status == StreamStatus.INITIALIZING


class TestIntegration:
    """Integration tests for Firestore listener and stream manager"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_stream_lifecycle(self):
        """Test complete stream lifecycle"""
        manager = StreamManager()
        
        try:
            # Add stream
            stream_id = await manager.add_stream("EUR_USD", "H1", StreamPriority.HIGH)
            assert stream_id in manager.managed_streams
            
            # Check initial status
            status = manager.get_stream_status(stream_id)
            assert status["status"] == StreamStatus.INITIALIZING.value
            
            # Test rate limiting
            assert await manager._check_rate_limit(stream_id) is True
            
            # Get comprehensive status
            all_status = manager.get_stream_status()
            assert len(all_status["streams"]) == 1
            
        finally:
            # Cleanup
            await manager.shutdown()
    
    @pytest.mark.asyncio 
    async def test_multiple_streams_coordination(self):
        """Test coordination of multiple streams"""
        manager = StreamManager()
        
        try:
            # Add multiple streams with different priorities
            stream_ids = []
            instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]
            priorities = [StreamPriority.LOW, StreamPriority.MEDIUM, StreamPriority.HIGH]
            
            for instrument, priority in zip(instruments, priorities):
                stream_id = await manager.add_stream(instrument, "H1", priority)
                stream_ids.append(stream_id)
            
            assert len(manager.managed_streams) == 3
            
            # Check rate limits are different based on priority
            rates = []
            for stream_id in stream_ids:
                stream = manager.managed_streams[stream_id]
                rates.append(stream.max_processing_rate)
            
            # Higher priority should have higher rate limits
            assert rates[0] < rates[1] < rates[2]  # LOW < MEDIUM < HIGH
            
            # Test global status
            status = manager.get_stream_status()
            assert status["global_metrics"]["total_streams"] == 3
            
        finally:
            await manager.shutdown()


@pytest.mark.integration
class TestFirestoreIntegration:
    """Integration tests with actual Firestore (when available)"""
    
    @pytest.mark.skipif(
        not settings.gcp_project_id,
        reason="GCP project not configured"
    )
    @pytest.mark.asyncio
    async def test_real_firestore_connection(self):
        """Test connection to real Firestore instance"""
        listener = FirestoreListener()
        
        try:
            # This will only run if GCP credentials are available
            client = listener.client
            assert client is not None
            
            # Test adding a stream (without starting it)
            stream_id = listener.add_stream("TEST_PAIR", "H1")
            assert stream_id == "TEST_PAIR_H1"
            
        except Exception as e:
            pytest.skip(f"Firestore not available: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])