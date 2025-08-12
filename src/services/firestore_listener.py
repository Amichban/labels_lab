"""
Real-time Firestore listener for market data streaming with circuit breaker protection

Implements a high-performance listener for Firestore candle data with:
- Multi-instrument/granularity streaming
- Complete candle filtering
- Circuit breaker protection and automatic fallbacks
- Automatic reconnection and error handling
- Dead letter queue for failed processing
- Integration with incremental label computation

Issue #11: Real-time Firestore listener implementation
Issue #14: Circuit breakers and failover mechanisms
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass
from google.cloud import firestore
from google.cloud.exceptions import GoogleCloudError
from google.api_core import retry

from src.models.data_models import Candle, Granularity
from .circuit_breaker import CircuitBreakerConfig, with_retry
from .resilience_manager import get_resilience_manager
from .fallback_handlers import get_fallback_handler

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for a single Firestore stream"""
    instrument_id: str
    granularity: str
    collection_path: str
    listener_callback: Optional[Callable] = None
    active: bool = False
    reconnect_count: int = 0
    last_reconnect: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[Exception] = None


@dataclass
class ProcessingResult:
    """Result of processing a candle document"""
    success: bool
    candle: Optional[Candle] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    timestamp: datetime = None


class DeadLetterQueue:
    """Dead letter queue for failed candle processing"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: List[Dict[str, Any]] = []
        self.failed_count = 0
        
    async def add_failed_item(self, 
                             item: Dict[str, Any], 
                             error: str, 
                             retry_count: int = 0):
        """Add failed item to dead letter queue"""
        if len(self.queue) >= self.max_size:
            # Remove oldest item
            self.queue.pop(0)
        
        failed_item = {
            "item": item,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "retry_count": retry_count
        }
        
        self.queue.append(failed_item)
        self.failed_count += 1
        
        # Store in Redis for persistence
        try:
            from src.services.redis_cache import redis_cache
            await redis_cache.add_to_dead_letter_queue(failed_item)
        except Exception as e:
            logger.error(f"Failed to persist dead letter item: {e}")
    
    async def retry_failed_items(self, max_retry_count: int = 3) -> int:
        """Retry processing of failed items"""
        retry_count = 0
        items_to_retry = [
            item for item in self.queue 
            if item["retry_count"] < max_retry_count
        ]
        
        for item in items_to_retry:
            try:
                # Attempt to reprocess the item
                # This would call the original processing logic
                success = await self._retry_process_item(item)
                if success:
                    self.queue.remove(item)
                    retry_count += 1
                else:
                    item["retry_count"] += 1
            except Exception as e:
                logger.error(f"Failed to retry dead letter item: {e}")
                item["retry_count"] += 1
        
        return retry_count
    
    async def _retry_process_item(self, item: Dict[str, Any]) -> bool:
        """Retry processing a failed item"""
        # This would be implemented by the calling service
        # For now, return False to indicate retry failed
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dead letter queue statistics"""
        return {
            "current_size": len(self.queue),
            "total_failed": self.failed_count,
            "max_size": self.max_size,
            "oldest_item": self.queue[0]["timestamp"] if self.queue else None,
            "newest_item": self.queue[-1]["timestamp"] if self.queue else None
        }


class FirestoreListener:
    """
    High-performance real-time Firestore listener for market data.
    
    Features:
    - Multiple instrument/granularity streams
    - Complete candle filtering (complete=true)
    - Automatic reconnection with exponential backoff
    - Dead letter queue for failed processing
    - Rate limiting and backpressure handling
    - Comprehensive metrics and monitoring
    """
    
    def __init__(self, 
                 project_id: Optional[str] = None,
                 credentials_path: Optional[str] = None):
        """Initialize Firestore listener"""
        
        # Firestore client configuration
        self.project_id = project_id
        self.credentials_path = credentials_path
        
        # Load settings when needed
        if not self.project_id or not self.credentials_path:
            try:
                from config.settings import settings
                self.project_id = self.project_id or settings.gcp_project_id
                self.credentials_path = self.credentials_path or settings.google_application_credentials
            except Exception as e:
                logger.warning(f"Could not load settings: {e}. Using provided parameters only.")
        self._client: Optional[firestore.Client] = None
        
        # Stream management
        self.streams: Dict[str, StreamConfig] = {}
        self.active_listeners: Dict[str, Any] = {}  # Firestore watch objects
        self.processing_callbacks: Dict[str, Callable] = {}
        
        # Error handling and resilience
        self.dead_letter_queue = DeadLetterQueue()
        self.max_reconnect_attempts = 10
        self.base_retry_delay = 1.0  # seconds
        self.max_retry_delay = 60.0  # seconds
        
        # Rate limiting
        self.max_processing_rate = 1000  # candles per second
        self.processing_semaphore = asyncio.Semaphore(100)  # concurrent processing limit
        
        # Metrics
        self.metrics = {
            "total_documents_processed": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "reconnections": 0,
            "active_streams": 0,
            "processing_rate_per_second": 0.0,
            "last_reset_time": datetime.utcnow()
        }
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.last_health_check = datetime.utcnow()
        self.health_status = "initializing"
        
        # Circuit breaker integration
        self.resilience_manager = get_resilience_manager()
        self.fallback_handler = get_fallback_handler('firestore')
        
        # Register service if not already registered
        try:
            self.circuit_breaker = self.resilience_manager.get_circuit_breaker('firestore')
            if not self.circuit_breaker:
                # Register with resilience manager
                circuit_config = CircuitBreakerConfig(
                    failure_threshold=8,
                    recovery_timeout=60.0,
                    timeout=10.0,
                    success_threshold=3,
                    expected_exception=(GoogleCloudError, Exception)
                )
                self.circuit_breaker = self.resilience_manager.register_service(
                    'firestore',
                    'stream',
                    'important',
                    circuit_config,
                    self._fallback_stream_handler
                )
        except Exception as e:
            logger.warning(f"Could not register Firestore with resilience manager: {e}")
            self.circuit_breaker = None
        
        logger.info("FirestoreListener initialized with circuit breaker protection")
    
    @property
    def client(self) -> firestore.Client:
        """Get or create Firestore client with retry logic"""
        if self._client is None:
            try:
                if self.credentials_path:
                    import os
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
                
                self._client = firestore.Client(project=self.project_id)
                logger.info(f"Connected to Firestore project: {self.project_id}")
            except Exception as e:
                logger.error(f"Failed to initialize Firestore client: {e}")
                raise
        
        return self._client
    
    def _fallback_stream_handler(self, instrument_id: str, granularity: str) -> Optional[Dict[str, Any]]:
        """Fallback handler for Firestore stream failures"""
        if self.fallback_handler:
            try:
                # Convert sync call to async for fallback handler
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create task for async fallback
                    asyncio.create_task(
                        self.fallback_handler.handle_stream_failure(instrument_id, granularity)
                    )
                    logger.warning(f"Firestore fallback triggered for stream {instrument_id}:{granularity}")
                    return {
                        'status': 'fallback_active',
                        'instrument_id': instrument_id,
                        'granularity': granularity,
                        'message': 'Stream using fallback mode'
                    }
                else:
                    return loop.run_until_complete(
                        self.fallback_handler.handle_stream_failure(instrument_id, granularity)
                    )
            except Exception as e:
                logger.error(f"Firestore fallback handler failed: {e}")
        
        return {
            'status': 'offline',
            'instrument_id': instrument_id,
            'granularity': granularity,
            'message': 'Stream unavailable'
        }
    
    def add_stream(self,
                   instrument_id: str,
                   granularity: str,
                   callback: Optional[Callable[[Candle], None]] = None) -> str:
        """
        Add a new stream for instrument/granularity pair.
        
        Args:
            instrument_id: Instrument to stream (e.g., 'EUR_USD')
            granularity: Time granularity (e.g., 'H1', 'H4')
            callback: Optional callback for processed candles
            
        Returns:
            Stream ID for management
        """
        stream_id = f"{instrument_id}_{granularity}"
        collection_path = f"candles/{instrument_id}/{granularity}/data"
        
        if stream_id in self.streams:
            logger.warning(f"Stream {stream_id} already exists")
            return stream_id
        
        stream_config = StreamConfig(
            instrument_id=instrument_id,
            granularity=granularity,
            collection_path=collection_path,
            listener_callback=callback
        )
        
        self.streams[stream_id] = stream_config
        
        if callback:
            self.processing_callbacks[stream_id] = callback
        
        logger.info(f"Added stream: {stream_id} -> {collection_path}")
        return stream_id
    
    async def start_stream(self, stream_id: str) -> bool:
        """
        Start listening to a specific stream.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            True if stream started successfully
        """
        if stream_id not in self.streams:
            logger.error(f"Stream {stream_id} not found")
            return False
        
        stream_config = self.streams[stream_id]
        
        if stream_config.active:
            logger.warning(f"Stream {stream_id} is already active")
            return True
        
        try:
            # Create Firestore query for complete candles only
            collection = self.client.collection(stream_config.collection_path)
            query = collection.where('complete', '==', True).order_by('ts', direction=firestore.Query.DESCENDING)
            
            # Create callback wrapper
            def on_snapshot(doc_snapshot, changes, read_time):
                asyncio.create_task(
                    self._handle_snapshot(stream_id, doc_snapshot, changes, read_time)
                )
            
            # Start listening
            listener = query.on_snapshot(on_snapshot)
            self.active_listeners[stream_id] = listener
            
            # Update stream status
            stream_config.active = True
            self.streams[stream_id] = stream_config
            self.metrics["active_streams"] += 1
            
            logger.info(f"Started listening to stream: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start stream {stream_id}: {e}")
            await self._handle_stream_error(stream_id, e)
            return False
    
    async def stop_stream(self, stream_id: str) -> bool:
        """
        Stop listening to a specific stream.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            True if stream stopped successfully
        """
        if stream_id not in self.streams:
            logger.error(f"Stream {stream_id} not found")
            return False
        
        try:
            # Stop Firestore listener
            if stream_id in self.active_listeners:
                listener = self.active_listeners[stream_id]
                listener.unsubscribe()
                del self.active_listeners[stream_id]
            
            # Update stream status
            stream_config = self.streams[stream_id]
            if stream_config.active:
                stream_config.active = False
                self.metrics["active_streams"] -= 1
            
            logger.info(f"Stopped listening to stream: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop stream {stream_id}: {e}")
            return False
    
    async def start_all_streams(self) -> Dict[str, bool]:
        """Start all configured streams"""
        results = {}
        
        for stream_id in self.streams.keys():
            results[stream_id] = await self.start_stream(stream_id)
        
        active_count = sum(1 for success in results.values() if success)
        logger.info(f"Started {active_count}/{len(results)} streams")
        
        return results
    
    async def stop_all_streams(self) -> Dict[str, bool]:
        """Stop all active streams"""
        results = {}
        
        for stream_id in list(self.active_listeners.keys()):
            results[stream_id] = await self.stop_stream(stream_id)
        
        logger.info("All streams stopped")
        return results
    
    async def _handle_snapshot(self,
                              stream_id: str,
                              doc_snapshot,
                              changes,
                              read_time) -> None:
        """
        Handle Firestore snapshot changes with rate limiting and error handling.
        
        Args:
            stream_id: Stream identifier
            doc_snapshot: Firestore document snapshot
            changes: Document changes
            read_time: Read timestamp
        """
        try:
            async with self.processing_semaphore:
                for change in changes:
                    await self._process_document_change(stream_id, change)
        except Exception as e:
            logger.error(f"Error handling snapshot for {stream_id}: {e}")
            await self._handle_stream_error(stream_id, e)
    
    async def _process_document_change(self, stream_id: str, change) -> ProcessingResult:
        """
        Process a single document change from Firestore.
        
        Args:
            stream_id: Stream identifier
            change: Firestore document change
            
        Returns:
            Processing result
        """
        start_time = datetime.utcnow()
        self.metrics["total_documents_processed"] += 1
        
        try:
            # Extract document data
            doc = change.document
            data = doc.to_dict()
            
            # Validate that candle is complete
            if not data.get('complete', False):
                logger.debug(f"Skipping incomplete candle: {doc.id}")
                return ProcessingResult(
                    success=False,
                    error="Incomplete candle",
                    processing_time_ms=0,
                    timestamp=start_time
                )
            
            # Parse stream configuration
            stream_config = self.streams[stream_id]
            
            # Convert to Candle object
            candle = self._convert_to_candle(data, stream_config)
            if not candle:
                error_msg = "Failed to convert document to Candle"
                await self.dead_letter_queue.add_failed_item(
                    {"stream_id": stream_id, "doc_id": doc.id, "data": data},
                    error_msg
                )
                return ProcessingResult(
                    success=False,
                    error=error_msg,
                    processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                    timestamp=start_time
                )
            
            # Process candle for label computation
            await self._process_candle_for_labels(candle, stream_id)
            
            # Call user callback if provided
            if stream_id in self.processing_callbacks:
                callback = self.processing_callbacks[stream_id]
                if callback:
                    try:
                        await callback(candle) if asyncio.iscoroutinefunction(callback) else callback(candle)
                    except Exception as e:
                        logger.warning(f"User callback failed for {stream_id}: {e}")
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics["successful_processing"] += 1
            
            return ProcessingResult(
                success=True,
                candle=candle,
                processing_time_ms=processing_time,
                timestamp=start_time
            )
            
        except Exception as e:
            error_msg = f"Failed to process document change: {str(e)}"
            logger.error(error_msg)
            
            self.metrics["failed_processing"] += 1
            
            # Add to dead letter queue
            await self.dead_letter_queue.add_failed_item(
                {"stream_id": stream_id, "change": str(change)},
                error_msg
            )
            
            return ProcessingResult(
                success=False,
                error=error_msg,
                processing_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                timestamp=start_time
            )
    
    def _convert_to_candle(self, data: Dict[str, Any], stream_config: StreamConfig) -> Optional[Candle]:
        """
        Convert Firestore document to Candle object.
        
        Args:
            data: Firestore document data
            stream_config: Stream configuration
            
        Returns:
            Candle object or None if conversion fails
        """
        try:
            # Map Firestore fields to Candle fields
            candle_data = {
                "instrument_id": stream_config.instrument_id,
                "granularity": Granularity(stream_config.granularity),
                "ts": data.get("ts"),  # Assumes timestamp is already datetime
                "open": float(data.get("o", 0)),
                "high": float(data.get("h", 0)),
                "low": float(data.get("l", 0)),
                "close": float(data.get("c", 0)),
                "volume": float(data.get("v", 0)),
                "bid": float(data.get("bid", 0)),
                "ask": float(data.get("ask", 0)),
            }
            
            # Handle timestamp conversion if needed
            if isinstance(candle_data["ts"], (int, float)):
                candle_data["ts"] = datetime.fromtimestamp(candle_data["ts"])
            
            return Candle(**candle_data)
            
        except Exception as e:
            logger.error(f"Failed to convert document to Candle: {e}")
            return None
    
    async def _process_candle_for_labels(self, candle: Candle, stream_id: str) -> None:
        """
        Process candle for incremental label computation.
        
        Args:
            candle: Candle to process
            stream_id: Stream identifier
        """
        try:
            # Check if we should compute labels for this candle
            # This could be based on configuration or business logic
            should_compute = await self._should_compute_labels(candle)
            
            if not should_compute:
                logger.debug(f"Skipping label computation for {candle.instrument_id} {candle.ts}")
                return
            
            # Compute labels asynchronously
            from src.core.label_computation import computation_engine
            label_set = await computation_engine.compute_labels(
                candle=candle,
                horizon_periods=6,  # Default horizon
                label_types=["enhanced_triple_barrier", "vol_scaled_return"],  # Priority labels
                use_cache=True,
                force_recompute=False
            )
            
            # Store or update labels as needed
            # This could trigger additional processing workflows
            logger.debug(f"Computed labels for {candle.instrument_id} {candle.ts}")
            
        except Exception as e:
            logger.error(f"Failed to process candle for labels: {e}")
            # This error should not stop the stream, just log it
    
    async def _should_compute_labels(self, candle: Candle) -> bool:
        """
        Determine if labels should be computed for this candle.
        
        Could be based on:
        - Time of day (market hours)
        - Volatility conditions
        - Configuration settings
        - Resource availability
        
        Args:
            candle: Candle to evaluate
            
        Returns:
            True if labels should be computed
        """
        # For now, compute labels for all complete candles
        # This could be made more sophisticated based on requirements
        return True
    
    async def _handle_stream_error(self, stream_id: str, error: Exception) -> None:
        """
        Handle stream errors with automatic reconnection.
        
        Args:
            stream_id: Stream that encountered error
            error: Error that occurred
        """
        logger.error(f"Stream error for {stream_id}: {error}")
        
        stream_config = self.streams.get(stream_id)
        if not stream_config:
            return
        
        stream_config.error_count += 1
        stream_config.last_error = error
        
        # Stop the current stream
        await self.stop_stream(stream_id)
        
        # Attempt reconnection with exponential backoff
        if stream_config.reconnect_count < self.max_reconnect_attempts:
            await self._attempt_reconnection(stream_id)
        else:
            logger.critical(f"Max reconnection attempts exceeded for {stream_id}")
            self.health_status = f"critical_error_{stream_id}"
    
    async def _attempt_reconnection(self, stream_id: str) -> None:
        """
        Attempt to reconnect a failed stream with exponential backoff.
        
        Args:
            stream_id: Stream to reconnect
        """
        stream_config = self.streams.get(stream_id)
        if not stream_config:
            return
        
        stream_config.reconnect_count += 1
        stream_config.last_reconnect = datetime.utcnow()
        self.metrics["reconnections"] += 1
        
        # Calculate backoff delay
        delay = min(
            self.base_retry_delay * (2 ** (stream_config.reconnect_count - 1)),
            self.max_retry_delay
        )
        
        logger.info(f"Attempting reconnection for {stream_id} in {delay:.1f} seconds (attempt {stream_config.reconnect_count})")
        
        await asyncio.sleep(delay)
        
        # Try to restart the stream
        success = await self.start_stream(stream_id)
        if success:
            logger.info(f"Successfully reconnected stream {stream_id}")
            stream_config.reconnect_count = 0  # Reset on success
        else:
            logger.warning(f"Failed to reconnect stream {stream_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all streams and overall system.
        
        Returns:
            Health status information
        """
        self.last_health_check = datetime.utcnow()
        
        stream_health = {}
        for stream_id, stream_config in self.streams.items():
            stream_health[stream_id] = {
                "active": stream_config.active,
                "error_count": stream_config.error_count,
                "reconnect_count": stream_config.reconnect_count,
                "last_error": str(stream_config.last_error) if stream_config.last_error else None,
                "last_reconnect": stream_config.last_reconnect.isoformat() if stream_config.last_reconnect else None
            }
        
        # Overall system health
        total_errors = sum(config.error_count for config in self.streams.values())
        active_streams = sum(1 for config in self.streams.values() if config.active)
        
        if active_streams == 0 and len(self.streams) > 0:
            self.health_status = "critical"
        elif total_errors > 10:
            self.health_status = "degraded"
        elif active_streams == len(self.streams):
            self.health_status = "healthy"
        else:
            self.health_status = "partial"
        
        return {
            "overall_status": self.health_status,
            "active_streams": active_streams,
            "total_streams": len(self.streams),
            "total_errors": total_errors,
            "metrics": self.get_metrics(),
            "dead_letter_queue": self.dead_letter_queue.get_stats(),
            "stream_health": stream_health,
            "last_check": self.last_health_check.isoformat()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        current_time = datetime.utcnow()
        time_since_reset = (current_time - self.metrics["last_reset_time"]).total_seconds()
        
        if time_since_reset > 0:
            processing_rate = self.metrics["total_documents_processed"] / time_since_reset
            self.metrics["processing_rate_per_second"] = processing_rate
        
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_processing"] / 
                max(self.metrics["total_documents_processed"], 1)
            ),
            "current_time": current_time.isoformat()
        }
    
    def reset_metrics(self) -> None:
        """Reset metrics counters"""
        self.metrics = {
            "total_documents_processed": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "reconnections": 0,
            "active_streams": self.metrics["active_streams"],  # Keep current count
            "processing_rate_per_second": 0.0,
            "last_reset_time": datetime.utcnow()
        }
        
        logger.info("Metrics reset")
    
    async def shutdown(self) -> None:
        """Graceful shutdown of all streams and resources"""
        logger.info("Starting graceful shutdown...")
        
        # Stop all streams
        await self.stop_all_streams()
        
        # Process remaining items in dead letter queue if needed
        try:
            retry_count = await self.dead_letter_queue.retry_failed_items()
            if retry_count > 0:
                logger.info(f"Retried {retry_count} items from dead letter queue during shutdown")
        except Exception as e:
            logger.error(f"Error processing dead letter queue during shutdown: {e}")
        
        # Close Firestore client
        if self._client:
            # Firestore client doesn't have explicit close method
            self._client = None
        
        self.health_status = "shutdown"
        logger.info("Firestore listener shutdown complete")


# Global listener instance
firestore_listener = FirestoreListener()