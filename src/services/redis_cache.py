"""
Redis cache service for high-performance label caching
"""
from typing import Any, Optional, Dict, List, Set
from datetime import datetime, timedelta
import logging
import msgpack
import redis
from redis import Redis
from config.settings import settings

logger = logging.getLogger(__name__)


class RedisCacheService:
    """Redis cache service with msgpack serialization"""
    
    def __init__(self):
        """Initialize Redis connection"""
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
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Deserialized value or None if not found
        """
        try:
            data = self.client.get(key)
            return self._deserialize(data) if data else None
        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache with optional TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: settings.cache_ttl_seconds)
            
        Returns:
            True if successful
        """
        try:
            serialized = self._serialize(value)
            ttl = ttl or self.default_ttl
            return self.client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            return False
    
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
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis connection check failed: {e}")
            return False
    
    def close(self):
        """Close Redis connection"""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Redis connection closed")


# Global cache service instance
redis_cache = RedisCacheService()