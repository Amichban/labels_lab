"""
Redis cache service for high-performance label caching with circuit breaker protection

Enhanced with resilience patterns including:
- Circuit breaker protection
- Fallback to in-memory cache and persistent storage
- Automatic retry with exponential backoff
- Health monitoring and recovery

Issue #14: Circuit breakers and failover mechanisms
"""
from typing import Any, Optional, Dict, List, Set
from datetime import datetime, timedelta
import logging
import msgpack
import redis
from redis import Redis
from config.settings import settings
from .circuit_breaker import CircuitBreakerConfig, with_retry
from .resilience_manager import get_resilience_manager
from .fallback_handlers import get_fallback_handler

logger = logging.getLogger(__name__)


class RedisCacheService:
    """Redis cache service with msgpack serialization and resilience features"""
    
    def __init__(self):
        """Initialize Redis connection with circuit breaker"""
        self.redis_params = {
            'host': settings.redis_host,
            'port': settings.redis_port,
            'password': settings.redis_password,
            'db': settings.redis_db,
            'decode_responses': False,  # Use msgpack for serialization
            'socket_keepalive': True,
            'socket_connect_timeout': 5,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
        self._client: Optional[Redis] = None
        self.default_ttl = settings.cache_ttl_seconds
        
        # Get circuit breaker from resilience manager
        self.resilience_manager = get_resilience_manager()
        self.fallback_handler = get_fallback_handler('redis')
        
        # Register service if not already registered
        try:
            self.circuit_breaker = self.resilience_manager.get_circuit_breaker('redis')
            if not self.circuit_breaker:
                # Register with resilience manager
                circuit_config = CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=30.0,
                    timeout=5.0,
                    success_threshold=3,
                    expected_exception=(redis.RedisError, redis.ConnectionError, redis.TimeoutError, Exception)
                )
                self.circuit_breaker = self.resilience_manager.register_service(
                    'redis',
                    'cache',
                    'important',
                    circuit_config,
                    self._fallback_operation_handler
                )
        except Exception as e:
            logger.warning(f"Could not register Redis with resilience manager: {e}")
            self.circuit_breaker = None
            
        logger.info("Redis cache service initialized with circuit breaker protection")
    
    @property
    def client(self) -> Redis:
        """Get or create Redis client"""
        if self._client is None:
            self._client = redis.Redis(**self.redis_params)
            logger.info(f"Connected to Redis at {settings.redis_host}:{settings.redis_port}")
        return self._client
    
    def _serialize(self, data: Any) -> bytes:
        """Serialize data using msgpack"""
        return msgpack.packb(data, use_bin_type=True)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data using msgpack"""
        if data is None:
            return None
        return msgpack.unpackb(data, raw=False)
    
    def _fallback_operation_handler(self, operation: str, *args, **kwargs) -> Any:
        """Fallback handler for Redis operation failures"""
        if self.fallback_handler:
            import asyncio
            # Handle different Redis operations
            if operation == 'get':
                key = args[0] if args else kwargs.get('key')
                if key:
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # Create a task for the async fallback
                            future = asyncio.create_task(self.fallback_handler.handle_get_failure(key))
                            # Since we can't await in sync context, return None and log
                            logger.warning(f"Redis fallback triggered for GET {key}")
                            return None
                        else:
                            return loop.run_until_complete(self.fallback_handler.handle_get_failure(key))
                    except Exception as e:
                        logger.error(f"Fallback handler failed: {e}")
                        return None
            elif operation == 'set':
                key = args[0] if len(args) > 0 else kwargs.get('key')
                value = args[1] if len(args) > 1 else kwargs.get('value')
                ttl = args[2] if len(args) > 2 else kwargs.get('ttl')
                if key is not None and value is not None:
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            logger.warning(f"Redis fallback triggered for SET {key}")
                            return False
                        else:
                            return loop.run_until_complete(
                                self.fallback_handler.handle_set_failure(key, value, ttl)
                            )
                    except Exception as e:
                        logger.error(f"Fallback handler failed: {e}")
                        return False
        
        # Return appropriate default values
        if operation == 'get':
            return None
        elif operation in ['set', 'delete', 'exists']:
            return False
        elif operation in ['mget', 'get_many']:
            return {}
        else:
            return None
    
    @with_retry(max_attempts=2, base_delay=0.5, max_delay=5.0,
                exceptions=(redis.RedisError, redis.ConnectionError, redis.TimeoutError))
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache with circuit breaker protection
        
        Args:
            key: Cache key
            
        Returns:
            Deserialized value or None if not found
        """
        # Use circuit breaker if available
        if self.circuit_breaker:
            return self.circuit_breaker.call(self._get_operation, key)
        else:
            return self._get_operation(key)
    
    def _get_operation(self, key: str) -> Optional[Any]:
        """Internal get operation"""
        try:
            data = self.client.get(key)
            result = self._deserialize(data) if data else None
            
            # Signal success to resilience manager
            if result is not None and hasattr(self.resilience_manager, 'fallback_orchestrator'):
                self._signal_service_recovery()
            
            return result
        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            self._signal_service_degradation()
            raise
    
    def _signal_service_recovery(self) -> None:
        """Signal service recovery to resilience manager"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    self.resilience_manager.fallback_orchestrator.service_recovered('redis')
                )
        except Exception:
            pass
    
    def _signal_service_degradation(self) -> None:
        """Signal service degradation to resilience manager"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    self.resilience_manager.fallback_orchestrator.service_degraded('redis')
                )
        except Exception:
            pass
    
    @with_retry(max_attempts=2, base_delay=0.5, max_delay=5.0,
                exceptions=(redis.RedisError, redis.ConnectionError, redis.TimeoutError))
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with optional TTL and circuit breaker protection
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: settings.cache_ttl_seconds)
            
        Returns:
            True if successful
        """
        # Use circuit breaker if available
        if self.circuit_breaker:
            return self.circuit_breaker.call(self._set_operation, key, value, ttl)
        else:
            return self._set_operation(key, value, ttl)
    
    def _set_operation(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Internal set operation"""
        try:
            serialized = self._serialize(value)
            ttl = ttl or self.default_ttl
            result = self.client.setex(key, ttl, serialized)
            
            # Signal success to resilience manager
            if result and hasattr(self.resilience_manager, 'fallback_orchestrator'):
                self._signal_service_recovery()
            
            return bool(result)
        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            self._signal_service_degradation()
            raise
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Redis exists check failed for key {key}: {e}")
            return False
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
        """
        if not keys:
            return {}
        
        try:
            values = self.client.mget(keys)
            result = {}
            for key, value in zip(keys, values):
                if value:
                    result[key] = self._deserialize(value)
            return result
        except Exception as e:
            logger.error(f"Redis mget failed: {e}")
            return {}
    
    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values in cache
        
        Args:
            mapping: Dictionary of key-value pairs
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        if not mapping:
            return True
        
        try:
            pipe = self.client.pipeline()
            ttl = ttl or self.default_ttl
            
            for key, value in mapping.items():
                serialized = self._serialize(value)
                pipe.setex(key, ttl, serialized)
            
            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Redis mset failed: {e}")
            return False
    
    # Specialized cache methods for label computation
    
    def cache_labels(self,
                    instrument_id: str,
                    granularity: str,
                    ts: datetime,
                    labels: Dict[str, Any],
                    ttl: Optional[int] = None) -> bool:
        """
        Cache computed labels
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            ts: Timestamp
            labels: Label dictionary
            ttl: Cache TTL (default: 1 hour)
            
        Returns:
            True if successful
        """
        key = f"labels:{instrument_id}:{granularity}:{ts.isoformat()}"
        ttl = ttl or 3600  # 1 hour default for labels
        return self.set(key, labels, ttl)
    
    def get_labels(self,
                  instrument_id: str,
                  granularity: str,
                  ts: datetime) -> Optional[Dict[str, Any]]:
        """
        Get cached labels
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            ts: Timestamp
            
        Returns:
            Label dictionary or None if not cached
        """
        key = f"labels:{instrument_id}:{granularity}:{ts.isoformat()}"
        return self.get(key)
    
    def cache_active_levels(self,
                           instrument_id: str,
                           granularity: str,
                           levels: List[Dict[str, Any]],
                           ttl: int = 300) -> bool:
        """
        Cache active support/resistance levels
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            levels: List of level dictionaries
            ttl: Cache TTL (default: 5 minutes)
            
        Returns:
            True if successful
        """
        key = f"levels:{instrument_id}:{granularity}:active"
        return self.set(key, levels, ttl)
    
    def get_active_levels(self,
                         instrument_id: str,
                         granularity: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached active levels
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            
        Returns:
            List of level dictionaries or None
        """
        key = f"levels:{instrument_id}:{granularity}:active"
        return self.get(key)
    
    def cache_path_data(self,
                       instrument_id: str,
                       granularity: str,
                       start_ts: datetime,
                       end_ts: datetime,
                       data: List[Dict[str, Any]],
                       ttl: int = 900) -> bool:
        """
        Cache path data for multi-timeframe calculations
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            start_ts: Start timestamp
            end_ts: End timestamp
            data: Path data
            ttl: Cache TTL (default: 15 minutes)
            
        Returns:
            True if successful
        """
        key = f"path:{instrument_id}:{granularity}:{start_ts.isoformat()}:{end_ts.isoformat()}"
        return self.set(key, data, ttl)
    
    def get_path_data(self,
                     instrument_id: str,
                     granularity: str,
                     start_ts: datetime,
                     end_ts: datetime) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached path data
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            start_ts: Start timestamp
            end_ts: End timestamp
            
        Returns:
            Path data or None
        """
        key = f"path:{instrument_id}:{granularity}:{start_ts.isoformat()}:{end_ts.isoformat()}"
        return self.get(key)
    
    def set_backfill_progress(self,
                             instrument_id: str,
                             granularity: str,
                             last_processed: datetime) -> bool:
        """
        Track backfill progress
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            last_processed: Last processed timestamp
            
        Returns:
            True if successful
        """
        key = f"backfill_progress:{instrument_id}:{granularity}"
        return self.set(key, last_processed.isoformat(), ttl=None)  # No expiry
    
    def get_backfill_progress(self,
                              instrument_id: str,
                              granularity: str) -> Optional[datetime]:
        """
        Get backfill progress
        
        Args:
            instrument_id: Instrument identifier
            granularity: Time granularity
            
        Returns:
            Last processed timestamp or None
        """
        key = f"backfill_progress:{instrument_id}:{granularity}"
        value = self.get(key)
        if value:
            return datetime.fromisoformat(value)
        return None
    
    def increment_metric(self, metric_name: str, amount: int = 1) -> int:
        """
        Increment a metric counter
        
        Args:
            metric_name: Name of the metric
            amount: Amount to increment
            
        Returns:
            New value
        """
        try:
            return self.client.incrby(f"metric:{metric_name}", amount)
        except Exception as e:
            logger.error(f"Failed to increment metric {metric_name}: {e}")
            return 0
    
    def get_metric(self, metric_name: str) -> int:
        """Get metric value"""
        try:
            value = self.client.get(f"metric:{metric_name}")
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Failed to get metric {metric_name}: {e}")
            return 0
    
    def flush_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern
        
        Args:
            pattern: Key pattern (e.g., 'labels:EUR_USD:*')
            
        Returns:
            Number of keys deleted
        """
        try:
            keys = list(self.client.scan_iter(match=pattern, count=1000))
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to flush pattern {pattern}: {e}")
            return 0
    
    def check_connection(self) -> bool:
        """Check if Redis connection is working"""
        try:
            # Use direct ping without circuit breaker to avoid recursive calls
            is_healthy = self.client.ping()
            
            # Signal recovery if healthy
            if is_healthy and hasattr(self.resilience_manager, 'fallback_orchestrator'):
                self._signal_service_recovery()
            
            return is_healthy
        except Exception as e:
            logger.error(f"Redis connection check failed: {e}")
            self._signal_service_degradation()
            return False
    
    def add_to_dead_letter_queue(self, item: Dict[str, Any]) -> bool:
        """
        Add item to dead letter queue (DLQ) for failed processing
        
        Args:
            item: Failed item with error information
            
        Returns:
            True if successful
        """
        try:
            # Add to list with timestamp
            dlq_key = "dead_letter_queue"
            serialized_item = self._serialize(item)
            
            # Use LPUSH to add to beginning of list
            self.client.lpush(dlq_key, serialized_item)
            
            # Trim list to max size (keep most recent items)
            max_size = getattr(settings, 'dead_letter_queue_max_size', 1000)
            self.client.ltrim(dlq_key, 0, max_size - 1)
            
            return True
        except Exception as e:
            logger.error(f"Failed to add item to dead letter queue: {e}")
            return False
    
    def get_dead_letter_queue_items(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get items from dead letter queue
        
        Args:
            limit: Maximum number of items to retrieve
            
        Returns:
            List of dead letter queue items
        """
        try:
            dlq_key = "dead_letter_queue"
            items = self.client.lrange(dlq_key, 0, limit - 1)
            
            return [self._deserialize(item) for item in items if item]
        except Exception as e:
            logger.error(f"Failed to get dead letter queue items: {e}")
            return []
    
    def remove_from_dead_letter_queue(self, count: int = 1) -> int:
        """
        Remove items from dead letter queue (from the right/oldest)
        
        Args:
            count: Number of items to remove
            
        Returns:
            Number of items actually removed
        """
        try:
            dlq_key = "dead_letter_queue"
            removed = 0
            
            for _ in range(count):
                result = self.client.rpop(dlq_key)
                if result:
                    removed += 1
                else:
                    break
            
            return removed
        except Exception as e:
            logger.error(f"Failed to remove items from dead letter queue: {e}")
            return 0
    
    def get_dead_letter_queue_size(self) -> int:
        """Get current size of dead letter queue"""
        try:
            dlq_key = "dead_letter_queue"
            return self.client.llen(dlq_key)
        except Exception as e:
            logger.error(f"Failed to get dead letter queue size: {e}")
            return 0

    def close(self):
        """Close Redis connection"""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Redis connection closed")


# Global cache service instance
redis_cache = RedisCacheService()