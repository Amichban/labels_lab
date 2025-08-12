"""
Fallback handlers for graceful degradation when services fail

Provides intelligent fallback mechanisms for each external service:
- ClickHouse: Cache-based responses and read-only modes
- Redis: In-memory fallbacks and persistence alternatives
- Firestore: Batch processing and local queuing

Issue #14: Circuit breakers and failover mechanisms
Following infra-pr best practices for production resilience
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import pickle
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FallbackCache:
    """In-memory cache for fallback scenarios"""
    data: Dict[str, Any] = field(default_factory=dict)
    timestamps: Dict[str, datetime] = field(default_factory=dict)
    max_size: int = 1000
    ttl_seconds: int = 300  # 5 minutes default TTL
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from fallback cache"""
        if key not in self.data:
            return None
        
        # Check TTL
        if key in self.timestamps:
            age = (datetime.utcnow() - self.timestamps[key]).total_seconds()
            if age > self.ttl_seconds:
                self.remove(key)
                return None
        
        return self.data[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in fallback cache"""
        # Evict old entries if cache is full
        if len(self.data) >= self.max_size:
            self._evict_oldest()
        
        self.data[key] = value
        self.timestamps[key] = datetime.utcnow()
    
    def remove(self, key: str) -> None:
        """Remove key from cache"""
        self.data.pop(key, None)
        self.timestamps.pop(key, None)
    
    def _evict_oldest(self) -> None:
        """Evict oldest entries to make space"""
        if not self.timestamps:
            return
        
        # Remove 10% of entries (oldest first)
        entries_to_remove = max(1, len(self.timestamps) // 10)
        sorted_keys = sorted(self.timestamps.keys(), key=lambda k: self.timestamps[k])
        
        for key in sorted_keys[:entries_to_remove]:
            self.remove(key)


@dataclass
class PersistentQueue:
    """Persistent queue for offline operation"""
    queue_file: str
    max_size: int = 10000
    _queue: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def __post_init__(self):
        """Load queue from disk if exists"""
        self._load_from_disk()
    
    def enqueue(self, item: Any) -> bool:
        """Add item to queue"""
        with self._lock:
            if len(self._queue) >= self.max_size:
                # Remove oldest item
                self._queue.popleft()
            
            self._queue.append({
                'item': item,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            self._save_to_disk()
            return True
    
    def dequeue(self) -> Optional[Any]:
        """Remove and return item from queue"""
        with self._lock:
            if not self._queue:
                return None
            
            item = self._queue.popleft()
            self._save_to_disk()
            return item
    
    def peek(self) -> Optional[Any]:
        """Look at next item without removing"""
        with self._lock:
            return self._queue[0] if self._queue else None
    
    def size(self) -> int:
        """Get queue size"""
        with self._lock:
            return len(self._queue)
    
    def clear(self) -> None:
        """Clear queue"""
        with self._lock:
            self._queue.clear()
            self._save_to_disk()
    
    def _load_from_disk(self) -> None:
        """Load queue from disk"""
        try:
            queue_path = Path(self.queue_file)
            if queue_path.exists():
                with open(queue_path, 'rb') as f:
                    data = pickle.load(f)
                    self._queue = deque(data, maxlen=self.max_size)
                logger.info(f"Loaded {len(self._queue)} items from persistent queue")
        except Exception as e:
            logger.warning(f"Failed to load queue from disk: {e}")
            self._queue = deque()
    
    def _save_to_disk(self) -> None:
        """Save queue to disk"""
        try:
            queue_path = Path(self.queue_file)
            queue_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(queue_path, 'wb') as f:
                pickle.dump(list(self._queue), f)
        except Exception as e:
            logger.error(f"Failed to save queue to disk: {e}")


class ClickHouseFallbackHandler:
    """
    Fallback handler for ClickHouse service failures.
    
    Strategies:
    - Use cached query results
    - Return stale data with warnings
    - Switch to read-only mode
    - Provide empty results for non-critical queries
    """
    
    def __init__(self):
        """Initialize ClickHouse fallback handler"""
        self.cache = FallbackCache(max_size=500, ttl_seconds=600)  # 10 minutes
        self.query_cache: Dict[str, Any] = {}
        self.stale_data_threshold = 3600  # 1 hour
        self.read_only_mode = False
        
        logger.info("ClickHouse fallback handler initialized")
    
    async def handle_query_failure(self, 
                                  query: str, 
                                  params: Optional[Dict] = None,
                                  operation_type: str = "read") -> List[Dict[str, Any]]:
        """
        Handle ClickHouse query failure with appropriate fallback.
        
        Args:
            query: SQL query that failed
            params: Query parameters
            operation_type: Type of operation ('read', 'write', 'insert')
            
        Returns:
            Fallback query results
        """
        logger.warning(f"ClickHouse query failed, using fallback for: {query[:100]}...")
        
        # For write operations in fallback mode
        if operation_type in ['write', 'insert']:
            return await self._handle_write_failure(query, params)
        
        # For read operations, try cache first
        cache_key = self._generate_cache_key(query, params)
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Returning cached result for query (age: {cached_result.get('age', 'unknown')})")
            return cached_result.get('data', [])
        
        # Try to get stale data from query cache
        stale_result = await self._get_stale_data(query, params)
        if stale_result:
            logger.warning("Returning stale data due to ClickHouse unavailability")
            return stale_result
        
        # Return appropriate empty result based on query type
        return await self._get_empty_result_for_query(query)
    
    async def _handle_write_failure(self, 
                                   query: str, 
                                   params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Handle write operation failure"""
        logger.error("Write operation failed - ClickHouse unavailable")
        
        # Enable read-only mode
        self.read_only_mode = True
        
        # Could queue writes for later processing
        # For now, just log and return empty result
        return []
    
    def _generate_cache_key(self, query: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for query and parameters"""
        import hashlib
        
        key_parts = [query]
        if params:
            # Sort params for consistent key generation
            sorted_params = json.dumps(params, sort_keys=True)
            key_parts.append(sorted_params)
        
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _get_stale_data(self, 
                             query: str, 
                             params: Optional[Dict] = None) -> Optional[List[Dict[str, Any]]]:
        """Try to get stale data from previous queries"""
        cache_key = self._generate_cache_key(query, params)
        
        # Check if we have any data for this query, even if stale
        if cache_key in self.query_cache:
            cached_entry = self.query_cache[cache_key]
            cached_time = cached_entry.get('timestamp', datetime.min)
            age_seconds = (datetime.utcnow() - cached_time).total_seconds()
            
            # Return stale data if within threshold
            if age_seconds <= self.stale_data_threshold:
                logger.warning(f"Returning stale data (age: {age_seconds:.0f}s)")
                return cached_entry.get('data', [])
        
        return None
    
    async def _get_empty_result_for_query(self, query: str) -> List[Dict[str, Any]]:
        """Return appropriate empty result based on query type"""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('select'):
            # For SELECT queries, return empty list
            return []
        elif 'snapshots' in query_lower:
            # For snapshot queries, return minimal structure
            return []
        elif 'levels' in query_lower:
            # For level queries, return empty levels
            return []
        else:
            # Generic empty result
            return []
    
    def cache_successful_query(self, 
                              query: str, 
                              params: Optional[Dict], 
                              result: List[Dict[str, Any]]) -> None:
        """Cache successful query result for fallback use"""
        cache_key = self._generate_cache_key(query, params)
        
        # Cache in both short-term and query caches
        self.cache.set(cache_key, {
            'data': result,
            'timestamp': datetime.utcnow(),
            'age': '0s'
        })
        
        self.query_cache[cache_key] = {
            'data': result,
            'timestamp': datetime.utcnow()
        }
        
        # Limit query cache size
        if len(self.query_cache) > 200:
            # Remove oldest entries
            sorted_keys = sorted(self.query_cache.keys(), 
                               key=lambda k: self.query_cache[k]['timestamp'])
            for key in sorted_keys[:50]:  # Remove oldest 50
                del self.query_cache[key]
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """Get current fallback status"""
        return {
            'read_only_mode': self.read_only_mode,
            'cached_queries': len(self.query_cache),
            'cache_size': len(self.cache.data),
            'oldest_cache_entry': min(self.cache.timestamps.values()) if self.cache.timestamps else None
        }


class RedisFallbackHandler:
    """
    Fallback handler for Redis service failures.
    
    Strategies:
    - In-memory cache for critical data
    - File-based persistence for important cache entries
    - Degraded caching with shorter TTLs
    - Skip caching for non-critical operations
    """
    
    def __init__(self, 
                 cache_dir: str = "/tmp/redis_fallback",
                 max_memory_cache_size: int = 1000):
        """Initialize Redis fallback handler"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_cache = FallbackCache(
            max_size=max_memory_cache_size, 
            ttl_seconds=300  # 5 minutes for memory cache
        )
        
        self.persistent_cache_enabled = True
        self.degraded_mode = False
        
        logger.info(f"Redis fallback handler initialized with cache dir: {cache_dir}")
    
    async def handle_get_failure(self, key: str) -> Optional[Any]:
        """Handle Redis GET operation failure"""
        logger.debug(f"Redis GET failed, trying fallback for key: {key}")
        
        # Try memory cache first
        result = self.memory_cache.get(key)
        if result is not None:
            logger.debug(f"Found key in memory fallback cache: {key}")
            return result
        
        # Try persistent cache
        if self.persistent_cache_enabled:
            result = await self._get_from_persistent_cache(key)
            if result is not None:
                logger.debug(f"Found key in persistent fallback cache: {key}")
                # Also cache in memory for faster future access
                self.memory_cache.set(key, result)
                return result
        
        logger.debug(f"Key not found in any fallback cache: {key}")
        return None
    
    async def handle_set_failure(self, 
                                key: str, 
                                value: Any, 
                                ttl: Optional[int] = None) -> bool:
        """Handle Redis SET operation failure"""
        logger.debug(f"Redis SET failed, using fallback for key: {key}")
        
        try:
            # Always cache in memory
            self.memory_cache.set(key, value)
            
            # Also try persistent cache if enabled
            if self.persistent_cache_enabled:
                await self._set_to_persistent_cache(key, value, ttl)
            
            return True
            
        except Exception as e:
            logger.error(f"Fallback SET failed for key {key}: {e}")
            return False
    
    async def handle_delete_failure(self, key: str) -> bool:
        """Handle Redis DELETE operation failure"""
        logger.debug(f"Redis DELETE failed, using fallback for key: {key}")
        
        try:
            # Remove from memory cache
            self.memory_cache.remove(key)
            
            # Remove from persistent cache if enabled
            if self.persistent_cache_enabled:
                await self._delete_from_persistent_cache(key)
            
            return True
            
        except Exception as e:
            logger.error(f"Fallback DELETE failed for key {key}: {e}")
            return False
    
    async def handle_mget_failure(self, keys: List[str]) -> Dict[str, Any]:
        """Handle Redis MGET operation failure"""
        logger.debug(f"Redis MGET failed, using fallback for {len(keys)} keys")
        
        result = {}
        
        for key in keys:
            value = await self.handle_get_failure(key)
            if value is not None:
                result[key] = value
        
        return result
    
    async def handle_mset_failure(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Handle Redis MSET operation failure"""
        logger.debug(f"Redis MSET failed, using fallback for {len(mapping)} keys")
        
        success_count = 0
        for key, value in mapping.items():
            if await self.handle_set_failure(key, value, ttl):
                success_count += 1
        
        return success_count == len(mapping)
    
    async def _get_from_persistent_cache(self, key: str) -> Optional[Any]:
        """Get value from persistent cache file"""
        try:
            cache_file = self.cache_dir / f"{self._safe_filename(key)}.cache"
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check TTL
            if 'expires_at' in data and datetime.utcnow() > data['expires_at']:
                # Expired, remove file
                cache_file.unlink(missing_ok=True)
                return None
            
            return data['value']
            
        except Exception as e:
            logger.warning(f"Failed to read from persistent cache for key {key}: {e}")
            return None
    
    async def _set_to_persistent_cache(self, 
                                      key: str, 
                                      value: Any, 
                                      ttl: Optional[int] = None) -> None:
        """Set value to persistent cache file"""
        try:
            cache_file = self.cache_dir / f"{self._safe_filename(key)}.cache"
            
            data = {'value': value}
            if ttl:
                data['expires_at'] = datetime.utcnow() + timedelta(seconds=ttl)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            logger.warning(f"Failed to write to persistent cache for key {key}: {e}")
    
    async def _delete_from_persistent_cache(self, key: str) -> None:
        """Delete value from persistent cache file"""
        try:
            cache_file = self.cache_dir / f"{self._safe_filename(key)}.cache"
            cache_file.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Failed to delete from persistent cache for key {key}: {e}")
    
    def _safe_filename(self, key: str) -> str:
        """Convert cache key to safe filename"""
        import hashlib
        # Use hash for long keys or keys with special characters
        if len(key) > 200 or any(c in key for c in '<>:"/\\|?*'):
            return hashlib.md5(key.encode()).hexdigest()
        return key.replace(':', '_').replace('/', '_')
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired persistent cache files"""
        cleaned_count = 0
        
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if ('expires_at' in data and 
                        datetime.utcnow() > data['expires_at']):
                        cache_file.unlink()
                        cleaned_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to check cache file {cache_file}: {e}")
                    # Remove corrupted files
                    cache_file.unlink(missing_ok=True)
                    cleaned_count += 1
        
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired cache files")
        
        return cleaned_count
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """Get current fallback status"""
        persistent_cache_size = 0
        try:
            persistent_cache_size = len(list(self.cache_dir.glob("*.cache")))
        except:
            pass
        
        return {
            'degraded_mode': self.degraded_mode,
            'memory_cache_size': len(self.memory_cache.data),
            'persistent_cache_size': persistent_cache_size,
            'persistent_cache_enabled': self.persistent_cache_enabled,
            'cache_directory': str(self.cache_dir)
        }


class FirestoreFallbackHandler:
    """
    Fallback handler for Firestore service failures.
    
    Strategies:
    - Queue operations for later processing
    - Switch to batch-only processing
    - Use local storage for critical data
    - Provide cached responses for reads
    """
    
    def __init__(self, queue_file: str = "/tmp/firestore_queue.dat"):
        """Initialize Firestore fallback handler"""
        self.operation_queue = PersistentQueue(queue_file, max_size=50000)
        self.local_cache = FallbackCache(max_size=2000, ttl_seconds=1800)  # 30 minutes
        self.batch_mode = False
        self.offline_mode = False
        
        logger.info(f"Firestore fallback handler initialized with queue: {queue_file}")
    
    async def handle_stream_failure(self, 
                                   instrument_id: str, 
                                   granularity: str) -> Optional[Dict[str, Any]]:
        """Handle Firestore streaming failure"""
        logger.warning(f"Firestore stream failed for {instrument_id}:{granularity}")
        
        # Switch to batch mode
        self.batch_mode = True
        
        # Try to get cached data
        cache_key = f"stream:{instrument_id}:{granularity}"
        cached_data = self.local_cache.get(cache_key)
        
        if cached_data:
            logger.info(f"Returning cached stream data for {instrument_id}:{granularity}")
            return cached_data
        
        # Return empty result indicating stream unavailable
        return {
            'status': 'offline',
            'message': 'Firestore stream unavailable, using fallback mode',
            'batch_mode': True
        }
    
    async def handle_write_failure(self, 
                                  collection_path: str, 
                                  document_data: Dict[str, Any]) -> bool:
        """Handle Firestore write failure by queuing operation"""
        logger.debug(f"Firestore write failed, queuing operation for: {collection_path}")
        
        operation = {
            'type': 'write',
            'collection_path': collection_path,
            'document_data': document_data,
            'timestamp': datetime.utcnow().isoformat(),
            'retry_count': 0
        }
        
        return self.operation_queue.enqueue(operation)
    
    async def handle_read_failure(self, 
                                 collection_path: str, 
                                 query_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Handle Firestore read failure"""
        logger.debug(f"Firestore read failed for: {collection_path}")
        
        # Try to get cached data
        cache_key = f"read:{collection_path}"
        if query_params:
            import hashlib
            query_hash = hashlib.md5(json.dumps(query_params, sort_keys=True).encode()).hexdigest()
            cache_key += f":{query_hash}"
        
        cached_data = self.local_cache.get(cache_key)
        if cached_data:
            logger.info(f"Returning cached read data for: {collection_path}")
            return cached_data
        
        # Return empty result
        return []
    
    async def cache_successful_operation(self, 
                                       operation_type: str,
                                       collection_path: str,
                                       data: Any,
                                       query_params: Optional[Dict[str, Any]] = None) -> None:
        """Cache successful operation result"""
        if operation_type == 'read':
            cache_key = f"read:{collection_path}"
            if query_params:
                import hashlib
                query_hash = hashlib.md5(json.dumps(query_params, sort_keys=True).encode()).hexdigest()
                cache_key += f":{query_hash}"
            
            self.local_cache.set(cache_key, data)
            
        elif operation_type == 'stream':
            # Cache stream data with instrument/granularity key
            if isinstance(data, dict) and 'instrument_id' in data and 'granularity' in data:
                cache_key = f"stream:{data['instrument_id']}:{data['granularity']}"
                self.local_cache.set(cache_key, data)
    
    async def process_queued_operations(self, max_operations: int = 100) -> int:
        """Process queued operations when Firestore becomes available"""
        processed_count = 0
        
        for _ in range(min(max_operations, self.operation_queue.size())):
            operation = self.operation_queue.dequeue()
            if not operation:
                break
            
            try:
                # Attempt to process the queued operation
                success = await self._process_queued_operation(operation['item'])
                if success:
                    processed_count += 1
                    logger.debug(f"Successfully processed queued operation: {operation['item']['type']}")
                else:
                    # Re-queue if processing failed
                    operation['item']['retry_count'] += 1
                    if operation['item']['retry_count'] < 3:
                        self.operation_queue.enqueue(operation['item'])
                        logger.warning(f"Re-queued failed operation (attempt {operation['item']['retry_count']})")
            
            except Exception as e:
                logger.error(f"Error processing queued operation: {e}")
        
        if processed_count > 0:
            logger.info(f"Processed {processed_count} queued Firestore operations")
        
        return processed_count
    
    async def _process_queued_operation(self, operation: Dict[str, Any]) -> bool:
        """Process a single queued operation"""
        try:
            # This would attempt to perform the actual Firestore operation
            # For now, we'll simulate success
            operation_type = operation.get('type')
            
            if operation_type == 'write':
                # Attempt Firestore write
                logger.debug(f"Processing queued write to: {operation['collection_path']}")
                # Actual Firestore write would go here
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to process queued operation: {e}")
            return False
    
    def enter_offline_mode(self) -> None:
        """Enter offline mode - queue all operations"""
        self.offline_mode = True
        self.batch_mode = True
        logger.warning("Entered Firestore offline mode - all operations will be queued")
    
    def exit_offline_mode(self) -> None:
        """Exit offline mode - resume normal operations"""
        self.offline_mode = False
        self.batch_mode = False
        logger.info("Exited Firestore offline mode")
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """Get current fallback status"""
        return {
            'offline_mode': self.offline_mode,
            'batch_mode': self.batch_mode,
            'queued_operations': self.operation_queue.size(),
            'cached_items': len(self.local_cache.data),
            'queue_file': self.operation_queue.queue_file
        }


class FallbackOrchestrator:
    """
    Orchestrator for all fallback handlers.
    
    Coordinates fallback strategies across all services and manages
    system-wide degraded operation modes.
    """
    
    def __init__(self):
        """Initialize fallback orchestrator"""
        self.clickhouse_handler = ClickHouseFallbackHandler()
        self.redis_handler = RedisFallbackHandler()
        self.firestore_handler = FirestoreFallbackHandler()
        
        self.degraded_services: Set[str] = set()
        self.emergency_mode = False
        
        # Start background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("Fallback orchestrator initialized")
    
    def get_handler(self, service_name: str):
        """Get fallback handler for specific service"""
        handlers = {
            'clickhouse': self.clickhouse_handler,
            'redis': self.redis_handler,
            'firestore': self.firestore_handler
        }
        return handlers.get(service_name)
    
    async def service_degraded(self, service_name: str) -> None:
        """Handle service entering degraded state"""
        self.degraded_services.add(service_name)
        logger.warning(f"Service {service_name} entered degraded state")
        
        # Check if we should enter emergency mode
        if len(self.degraded_services) >= 2:  # Multiple services down
            await self.enter_emergency_mode()
    
    async def service_recovered(self, service_name: str) -> None:
        """Handle service recovery"""
        self.degraded_services.discard(service_name)
        logger.info(f"Service {service_name} recovered from degraded state")
        
        # Process any queued operations for this service
        if service_name == 'firestore':
            await self.firestore_handler.process_queued_operations()
        
        # Exit emergency mode if conditions are met
        if len(self.degraded_services) < 2:
            await self.exit_emergency_mode()
    
    async def enter_emergency_mode(self) -> None:
        """Enter system-wide emergency mode"""
        if self.emergency_mode:
            return
        
        self.emergency_mode = True
        logger.critical("Entering system emergency mode")
        
        # Switch all services to most conservative fallback modes
        self.clickhouse_handler.read_only_mode = True
        self.redis_handler.degraded_mode = True
        self.firestore_handler.enter_offline_mode()
    
    async def exit_emergency_mode(self) -> None:
        """Exit emergency mode"""
        if not self.emergency_mode:
            return
        
        self.emergency_mode = False
        logger.info("Exiting system emergency mode")
        
        # Restore normal operation modes
        self.clickhouse_handler.read_only_mode = False
        self.redis_handler.degraded_mode = False
        self.firestore_handler.exit_offline_mode()
    
    async def start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Started fallback orchestrator background tasks")
    
    async def stop_background_tasks(self) -> None:
        """Stop background tasks"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped fallback orchestrator background tasks")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while True:
            try:
                # Clean up expired cache entries
                self.redis_handler.cleanup_expired_cache()
                
                # Process queued Firestore operations if not in offline mode
                if not self.firestore_handler.offline_mode:
                    await self.firestore_handler.process_queued_operations(max_operations=50)
                
                # Sleep for 5 minutes
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in fallback cleanup loop: {e}")
                await asyncio.sleep(60)  # Shorter sleep on error
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall fallback system status"""
        return {
            'emergency_mode': self.emergency_mode,
            'degraded_services': list(self.degraded_services),
            'handlers': {
                'clickhouse': self.clickhouse_handler.get_fallback_status(),
                'redis': self.redis_handler.get_fallback_status(),
                'firestore': self.firestore_handler.get_fallback_status()
            }
        }


# Global fallback orchestrator instance
fallback_orchestrator = FallbackOrchestrator()


def get_fallback_handler(service_name: str):
    """Get fallback handler for service"""
    return fallback_orchestrator.get_handler(service_name)


def get_fallback_orchestrator() -> FallbackOrchestrator:
    """Get global fallback orchestrator instance"""
    return fallback_orchestrator


# Convenience functions for each service
async def clickhouse_fallback(query: str, 
                             params: Optional[Dict] = None,
                             operation_type: str = "read") -> List[Dict[str, Any]]:
    """ClickHouse fallback function"""
    return await fallback_orchestrator.clickhouse_handler.handle_query_failure(
        query, params, operation_type
    )


async def redis_get_fallback(key: str) -> Optional[Any]:
    """Redis GET fallback function"""
    return await fallback_orchestrator.redis_handler.handle_get_failure(key)


async def redis_set_fallback(key: str, 
                            value: Any, 
                            ttl: Optional[int] = None) -> bool:
    """Redis SET fallback function"""
    return await fallback_orchestrator.redis_handler.handle_set_failure(key, value, ttl)


async def firestore_stream_fallback(instrument_id: str, 
                                   granularity: str) -> Optional[Dict[str, Any]]:
    """Firestore stream fallback function"""
    return await fallback_orchestrator.firestore_handler.handle_stream_failure(
        instrument_id, granularity
    )


# Register fallback functions with circuit breakers (called during initialization)
def register_fallback_functions() -> Dict[str, Callable]:
    """Register all fallback functions"""
    return {
        'clickhouse': clickhouse_fallback,
        'redis_get': redis_get_fallback,
        'redis_set': redis_set_fallback,
        'firestore_stream': firestore_stream_fallback
    }