"""
Stream Manager for coordinating multiple Firestore listeners

Manages multiple concurrent streams with:
- Rate limiting and backpressure handling
- Stream health monitoring and automatic recovery
- Graceful startup and shutdown
- Resource management and optimization
- Comprehensive monitoring and alerting

Issue #11: Stream management for real-time Firestore listeners
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from src.services.firestore_listener import FirestoreListener, StreamConfig

logger = logging.getLogger(__name__)


class StreamPriority(Enum):
    """Stream priority levels for resource allocation"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class StreamStatus(Enum):
    """Stream status states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class StreamMetrics:
    """Metrics for a single stream"""
    documents_processed: int = 0
    processing_rate_per_minute: float = 0.0
    error_count: int = 0
    last_document_time: Optional[datetime] = None
    avg_processing_time_ms: float = 0.0
    backpressure_events: int = 0
    reconnections: int = 0


@dataclass
class ManagedStream:
    """Managed stream with configuration and monitoring"""
    stream_id: str
    instrument_id: str
    granularity: str
    priority: StreamPriority = StreamPriority.MEDIUM
    status: StreamStatus = StreamStatus.INITIALIZING
    listener: Optional[FirestoreListener] = None
    metrics: StreamMetrics = field(default_factory=StreamMetrics)
    created_at: datetime = field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Rate limiting
    max_processing_rate: int = 100  # documents per minute
    current_processing_rate: float = 0.0
    
    # Backpressure handling
    backpressure_threshold: int = 1000  # pending documents
    backpressure_active: bool = False
    
    # Health monitoring
    health_check_interval: int = 30  # seconds
    last_health_check: Optional[datetime] = None
    consecutive_errors: int = 0
    max_consecutive_errors: int = 5


class StreamManager:
    """
    Manages multiple concurrent Firestore streams with advanced features:
    
    - Prioritized stream processing
    - Rate limiting and backpressure handling
    - Health monitoring and automatic recovery
    - Resource optimization and load balancing
    - Graceful shutdown and startup
    """
    
    def __init__(self):
        """Initialize Stream Manager"""
        
        # Core components
        self.managed_streams: Dict[str, ManagedStream] = {}
        self.stream_listeners: Dict[str, FirestoreListener] = {}
        
        # Resource management
        try:
            from config.settings import settings
            self.max_concurrent_streams = settings.parallel_workers or 8
        except Exception:
            self.max_concurrent_streams = 8
        self.global_processing_semaphore = asyncio.Semaphore(1000)
        self.stream_semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Rate limiting
        self.global_rate_limiter = asyncio.Semaphore(5000)  # Global rate limit
        self.rate_limit_window = 60  # seconds
        self.rate_limit_counters: Dict[str, List[datetime]] = {}
        
        # Health monitoring
        self.health_monitor_task: Optional[asyncio.Task] = None
        self.health_check_interval = 30  # seconds
        self.monitoring_active = False
        
        # Backpressure handling
        self.backpressure_threshold = 10000  # global pending documents
        self.backpressure_recovery_threshold = 1000
        self.backpressure_active = False
        
        # Metrics and monitoring
        self.global_metrics = {
            "total_streams": 0,
            "active_streams": 0,
            "total_documents_processed": 0,
            "total_errors": 0,
            "avg_processing_time_ms": 0.0,
            "backpressure_events": 0,
            "reconnections": 0,
            "uptime_start": datetime.utcnow()
        }
        
        # Configuration
        self.auto_recovery_enabled = True
        self.performance_optimization_enabled = True
        
        logger.info("StreamManager initialized")
    
    async def add_stream(self,
                        instrument_id: str,
                        granularity: str,
                        priority: StreamPriority = StreamPriority.MEDIUM,
                        config: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new managed stream.
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            priority: Stream priority level
            config: Optional stream configuration
            
        Returns:
            Stream ID
        """
        stream_id = f"{instrument_id}_{granularity}"
        
        if stream_id in self.managed_streams:
            logger.warning(f"Stream {stream_id} already exists")
            return stream_id
        
        # Create managed stream
        managed_stream = ManagedStream(
            stream_id=stream_id,
            instrument_id=instrument_id,
            granularity=granularity,
            priority=priority,
            config=config or {}
        )
        
        # Set rate limits based on priority
        managed_stream.max_processing_rate = self._get_rate_limit_for_priority(priority)
        
        # Create semaphore for this stream
        self.stream_semaphores[stream_id] = asyncio.Semaphore(100)
        
        # Initialize rate limit counter
        self.rate_limit_counters[stream_id] = []
        
        self.managed_streams[stream_id] = managed_stream
        self.global_metrics["total_streams"] += 1
        
        logger.info(f"Added managed stream: {stream_id} (priority: {priority.name})")
        return stream_id
    
    async def start_stream(self, stream_id: str) -> bool:
        """
        Start a managed stream with full monitoring and recovery.
        
        Args:
            stream_id: Stream to start
            
        Returns:
            True if started successfully
        """
        if stream_id not in self.managed_streams:
            logger.error(f"Stream {stream_id} not found")
            return False
        
        managed_stream = self.managed_streams[stream_id]
        
        if managed_stream.status == StreamStatus.RUNNING:
            logger.warning(f"Stream {stream_id} is already running")
            return True
        
        try:
            # Check if we can start another stream (resource limits)
            if len([s for s in self.managed_streams.values() if s.status == StreamStatus.RUNNING]) >= self.max_concurrent_streams:
                logger.warning(f"Max concurrent streams ({self.max_concurrent_streams}) reached, cannot start {stream_id}")
                return False
            
            # Create Firestore listener
            listener = FirestoreListener()
            listener.add_stream(
                managed_stream.instrument_id,
                managed_stream.granularity,
                callback=self._create_stream_callback(stream_id)
            )
            
            # Start the listener
            success = await listener.start_stream(f"{managed_stream.instrument_id}_{managed_stream.granularity}")
            
            if success:
                managed_stream.listener = listener
                managed_stream.status = StreamStatus.RUNNING
                managed_stream.last_health_check = datetime.utcnow()
                
                self.stream_listeners[stream_id] = listener
                self.global_metrics["active_streams"] += 1
                
                logger.info(f"Started managed stream: {stream_id}")
                
                # Start health monitoring if not already running
                if not self.monitoring_active:
                    await self._start_health_monitoring()
                
                return True
            else:
                managed_stream.status = StreamStatus.ERROR
                logger.error(f"Failed to start stream {stream_id}")
                return False
                
        except Exception as e:
            logger.error(f"Exception starting stream {stream_id}: {e}")
            managed_stream.status = StreamStatus.ERROR
            managed_stream.consecutive_errors += 1
            return False
    
    async def stop_stream(self, stream_id: str, graceful: bool = True) -> bool:
        """
        Stop a managed stream.
        
        Args:
            stream_id: Stream to stop
            graceful: Whether to wait for pending operations
            
        Returns:
            True if stopped successfully
        """
        if stream_id not in self.managed_streams:
            logger.error(f"Stream {stream_id} not found")
            return False
        
        managed_stream = self.managed_streams[stream_id]
        
        try:
            if managed_stream.listener:
                if graceful:
                    # Allow pending operations to complete
                    await asyncio.sleep(1)
                
                # Stop the Firestore listener
                await managed_stream.listener.stop_stream(f"{managed_stream.instrument_id}_{managed_stream.granularity}")
                
                if stream_id in self.stream_listeners:
                    del self.stream_listeners[stream_id]
            
            managed_stream.status = StreamStatus.STOPPED
            managed_stream.listener = None
            
            if self.global_metrics["active_streams"] > 0:
                self.global_metrics["active_streams"] -= 1
            
            logger.info(f"Stopped managed stream: {stream_id}")
            return True
            
        except Exception as e:
            logger.error(f"Exception stopping stream {stream_id}: {e}")
            return False
    
    async def start_all_streams(self) -> Dict[str, bool]:
        """Start all managed streams based on priority"""
        results = {}
        
        # Sort streams by priority (highest first)
        sorted_streams = sorted(
            self.managed_streams.items(),
            key=lambda x: x[1].priority.value,
            reverse=True
        )
        
        for stream_id, managed_stream in sorted_streams:
            if managed_stream.status != StreamStatus.STOPPED:
                continue
            
            success = await self.start_stream(stream_id)
            results[stream_id] = success
            
            # Small delay between starts to avoid overwhelming Firestore
            await asyncio.sleep(0.1)
        
        active_count = sum(1 for success in results.values() if success)
        logger.info(f"Started {active_count}/{len(results)} managed streams")
        
        return results
    
    async def stop_all_streams(self, graceful: bool = True) -> Dict[str, bool]:
        """Stop all managed streams"""
        results = {}
        
        # Stop health monitoring first
        await self._stop_health_monitoring()
        
        for stream_id in list(self.managed_streams.keys()):
            results[stream_id] = await self.stop_stream(stream_id, graceful)
        
        logger.info("All managed streams stopped")
        return results
    
    def _create_stream_callback(self, stream_id: str) -> Callable:
        """Create callback function for stream processing"""
        
        async def callback(candle):
            """Process candle with rate limiting and backpressure handling"""
            managed_stream = self.managed_streams.get(stream_id)
            if not managed_stream:
                return
            
            # Apply rate limiting
            if not await self._check_rate_limit(stream_id):
                logger.warning(f"Rate limit exceeded for {stream_id}, dropping candle")
                managed_stream.metrics.backpressure_events += 1
                return
            
            # Apply backpressure if needed
            if self.backpressure_active:
                logger.debug(f"Backpressure active, queuing candle for {stream_id}")
                await self._handle_backpressure(stream_id, candle)
                return
            
            # Process candle
            async with self.stream_semaphores[stream_id]:
                async with self.global_processing_semaphore:
                    await self._process_candle(stream_id, candle)
        
        return callback
    
    async def _process_candle(self, stream_id: str, candle) -> None:
        """Process a single candle with metrics tracking"""
        start_time = datetime.utcnow()
        managed_stream = self.managed_streams.get(stream_id)
        
        if not managed_stream:
            return
        
        try:
            # Update rate tracking
            await self._update_rate_tracking(stream_id)
            
            # Process the candle (this would integrate with label computation)
            # For now, we'll just track metrics
            managed_stream.metrics.documents_processed += 1
            managed_stream.metrics.last_document_time = datetime.utcnow()
            
            # Update global metrics
            self.global_metrics["total_documents_processed"] += 1
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update average processing time
            if managed_stream.metrics.avg_processing_time_ms == 0:
                managed_stream.metrics.avg_processing_time_ms = processing_time
            else:
                managed_stream.metrics.avg_processing_time_ms = (
                    managed_stream.metrics.avg_processing_time_ms * 0.9 + 
                    processing_time * 0.1
                )
            
            # Reset consecutive errors on success
            managed_stream.consecutive_errors = 0
            
            logger.debug(f"Processed candle for {stream_id} in {processing_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Error processing candle for {stream_id}: {e}")
            managed_stream.metrics.error_count += 1
            managed_stream.consecutive_errors += 1
            self.global_metrics["total_errors"] += 1
            
            # Trigger recovery if too many consecutive errors
            if managed_stream.consecutive_errors >= managed_stream.max_consecutive_errors:
                await self._trigger_stream_recovery(stream_id)
    
    async def _check_rate_limit(self, stream_id: str) -> bool:
        """Check if stream is within rate limits"""
        managed_stream = self.managed_streams.get(stream_id)
        if not managed_stream:
            return False
        
        now = datetime.utcnow()
        
        # Clean old entries
        cutoff_time = now - timedelta(seconds=self.rate_limit_window)
        self.rate_limit_counters[stream_id] = [
            t for t in self.rate_limit_counters[stream_id] 
            if t > cutoff_time
        ]
        
        # Check rate limit
        current_count = len(self.rate_limit_counters[stream_id])
        max_count = managed_stream.max_processing_rate
        
        if current_count >= max_count:
            return False
        
        # Add current request
        self.rate_limit_counters[stream_id].append(now)
        return True
    
    async def _update_rate_tracking(self, stream_id: str) -> None:
        """Update processing rate metrics"""
        managed_stream = self.managed_streams.get(stream_id)
        if not managed_stream:
            return
        
        now = datetime.utcnow()
        
        # Calculate rate over last minute
        minute_ago = now - timedelta(minutes=1)
        recent_requests = [
            t for t in self.rate_limit_counters[stream_id] 
            if t > minute_ago
        ]
        
        managed_stream.current_processing_rate = len(recent_requests)
        managed_stream.metrics.processing_rate_per_minute = len(recent_requests)
    
    async def _handle_backpressure(self, stream_id: str, candle) -> None:
        """Handle backpressure by queueing or dropping candles"""
        # For now, just drop the candle and log
        # In a full implementation, this might queue to Redis or similar
        logger.warning(f"Backpressure: dropping candle for {stream_id}")
        
        managed_stream = self.managed_streams.get(stream_id)
        if managed_stream:
            managed_stream.metrics.backpressure_events += 1
            self.global_metrics["backpressure_events"] += 1
    
    async def _trigger_stream_recovery(self, stream_id: str) -> None:
        """Trigger automatic recovery for a failing stream"""
        if not self.auto_recovery_enabled:
            return
        
        logger.warning(f"Triggering recovery for stream {stream_id}")
        
        managed_stream = self.managed_streams.get(stream_id)
        if not managed_stream:
            return
        
        try:
            # Stop the stream
            await self.stop_stream(stream_id, graceful=False)
            
            # Wait a bit before restarting
            await asyncio.sleep(5)
            
            # Restart the stream
            success = await self.start_stream(stream_id)
            
            if success:
                logger.info(f"Successfully recovered stream {stream_id}")
                managed_stream.metrics.reconnections += 1
                self.global_metrics["reconnections"] += 1
            else:
                logger.error(f"Failed to recover stream {stream_id}")
                managed_stream.status = StreamStatus.ERROR
                
        except Exception as e:
            logger.error(f"Exception during stream recovery for {stream_id}: {e}")
            managed_stream.status = StreamStatus.ERROR
    
    async def _start_health_monitoring(self) -> None:
        """Start background health monitoring task"""
        if self.health_monitor_task:
            return
        
        self.monitoring_active = True
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Health monitoring started")
    
    async def _stop_health_monitoring(self) -> None:
        """Stop background health monitoring task"""
        self.monitoring_active = False
        
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
            self.health_monitor_task = None
        
        logger.info("Health monitoring stopped")
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while self.monitoring_active:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all streams"""
        now = datetime.utcnow()
        
        for stream_id, managed_stream in self.managed_streams.items():
            try:
                # Check if stream is healthy
                await self._check_stream_health(stream_id, now)
                
                # Update last health check time
                managed_stream.last_health_check = now
                
            except Exception as e:
                logger.error(f"Health check failed for {stream_id}: {e}")
        
        # Check global backpressure
        await self._check_global_backpressure()
    
    async def _check_stream_health(self, stream_id: str, now: datetime) -> None:
        """Check health of a single stream"""
        managed_stream = self.managed_streams.get(stream_id)
        if not managed_stream:
            return
        
        # Check if stream has been silent too long
        if managed_stream.metrics.last_document_time:
            silence_duration = now - managed_stream.metrics.last_document_time
            if silence_duration > timedelta(minutes=10):  # Configurable threshold
                logger.warning(f"Stream {stream_id} has been silent for {silence_duration}")
                
                # Trigger recovery if needed
                if managed_stream.status == StreamStatus.RUNNING:
                    await self._trigger_stream_recovery(stream_id)
        
        # Check error rate
        if managed_stream.consecutive_errors >= managed_stream.max_consecutive_errors:
            logger.warning(f"Stream {stream_id} has too many consecutive errors: {managed_stream.consecutive_errors}")
            
            if managed_stream.status == StreamStatus.RUNNING:
                await self._trigger_stream_recovery(stream_id)
    
    async def _check_global_backpressure(self) -> None:
        """Check and manage global backpressure"""
        # This would check system resources, queue sizes, etc.
        # For now, we'll use a simple metric
        
        total_pending = sum(
            len(self.rate_limit_counters.get(stream_id, []))
            for stream_id in self.managed_streams.keys()
        )
        
        if not self.backpressure_active and total_pending > self.backpressure_threshold:
            logger.warning(f"Activating global backpressure: {total_pending} pending items")
            self.backpressure_active = True
            
        elif self.backpressure_active and total_pending < self.backpressure_recovery_threshold:
            logger.info(f"Deactivating global backpressure: {total_pending} pending items")
            self.backpressure_active = False
    
    def _get_rate_limit_for_priority(self, priority: StreamPriority) -> int:
        """Get rate limit based on stream priority"""
        rate_limits = {
            StreamPriority.LOW: 50,      # 50 docs/minute
            StreamPriority.MEDIUM: 100,  # 100 docs/minute
            StreamPriority.HIGH: 200,    # 200 docs/minute
            StreamPriority.CRITICAL: 500 # 500 docs/minute
        }
        return rate_limits.get(priority, 100)
    
    def get_stream_status(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status information for streams"""
        if stream_id:
            if stream_id not in self.managed_streams:
                return {"error": "Stream not found"}
            
            managed_stream = self.managed_streams[stream_id]
            return {
                "stream_id": stream_id,
                "status": managed_stream.status.value,
                "priority": managed_stream.priority.name,
                "metrics": {
                    "documents_processed": managed_stream.metrics.documents_processed,
                    "processing_rate_per_minute": managed_stream.metrics.processing_rate_per_minute,
                    "error_count": managed_stream.metrics.error_count,
                    "avg_processing_time_ms": managed_stream.metrics.avg_processing_time_ms,
                    "backpressure_events": managed_stream.metrics.backpressure_events,
                    "reconnections": managed_stream.metrics.reconnections,
                    "consecutive_errors": managed_stream.consecutive_errors
                },
                "last_document_time": managed_stream.metrics.last_document_time.isoformat() if managed_stream.metrics.last_document_time else None,
                "last_health_check": managed_stream.last_health_check.isoformat() if managed_stream.last_health_check else None,
                "created_at": managed_stream.created_at.isoformat()
            }
        
        # Return status for all streams
        return {
            "global_metrics": {
                **self.global_metrics,
                "uptime_seconds": (datetime.utcnow() - self.global_metrics["uptime_start"]).total_seconds(),
                "backpressure_active": self.backpressure_active,
                "monitoring_active": self.monitoring_active
            },
            "streams": {
                sid: self.get_stream_status(sid)
                for sid in self.managed_streams.keys()
            }
        }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimize stream performance based on current metrics"""
        if not self.performance_optimization_enabled:
            return {"status": "optimization_disabled"}
        
        optimizations = []
        
        # Analyze stream performance
        for stream_id, managed_stream in self.managed_streams.items():
            if managed_stream.status != StreamStatus.RUNNING:
                continue
            
            # Check if stream is underutilizing resources
            if (managed_stream.current_processing_rate < managed_stream.max_processing_rate * 0.5 and
                managed_stream.metrics.error_count == 0):
                # Could increase rate limit
                new_rate = min(managed_stream.max_processing_rate * 1.2, 1000)
                managed_stream.max_processing_rate = int(new_rate)
                optimizations.append(f"Increased rate limit for {stream_id} to {new_rate}")
            
            # Check if stream is overloaded
            if (managed_stream.metrics.backpressure_events > 10 and
                managed_stream.metrics.error_count > 5):
                # Decrease rate limit
                new_rate = max(managed_stream.max_processing_rate * 0.8, 10)
                managed_stream.max_processing_rate = int(new_rate)
                optimizations.append(f"Decreased rate limit for {stream_id} to {new_rate}")
        
        # Global optimizations
        if self.global_metrics["total_errors"] > self.global_metrics["total_documents_processed"] * 0.1:
            logger.warning("High global error rate detected, may need manual intervention")
        
        return {
            "status": "completed",
            "optimizations_applied": len(optimizations),
            "details": optimizations,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the stream manager"""
        logger.info("Starting StreamManager shutdown...")
        
        # Stop health monitoring
        await self._stop_health_monitoring()
        
        # Stop all streams
        await self.stop_all_streams(graceful=True)
        
        # Clear resources
        self.stream_semaphores.clear()
        self.rate_limit_counters.clear()
        
        logger.info("StreamManager shutdown complete")


# Global stream manager instance
stream_manager = StreamManager()