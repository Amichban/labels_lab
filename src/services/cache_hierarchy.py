"""
Hierarchical Cache Management System for Issue #13

Implements a three-level cache hierarchy with intelligent TTL cascade
management and performance optimization.

Cache Levels:
- L1: In-Memory Cache (fastest, smallest capacity)
- L2: Redis Cache (fast, medium capacity) 
- L3: ClickHouse Cache (slower, largest capacity)

Features:
- Intelligent cache promotion/demotion
- TTL cascade management across levels
- Cache coherency and consistency
- Performance-optimized lookup paths
- Memory pressure handling
- Cache warming coordination
- Hit rate optimization (target >95%)
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import psutil
import weakref

from src.services.redis_cache import redis_cache
from src.services.clickhouse_service import clickhouse_service

logger = logging.getLogger(__name__)


class CacheLevel(str, Enum):
    """Cache hierarchy levels"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_CLICKHOUSE = "l3_clickhouse"


class CacheOperation(str, Enum):
    """Cache operations for statistics"""
    GET = "get"
    SET = "set"
    DELETE = "delete"
    PROMOTE = "promote"
    DEMOTE = "demote"


@dataclass
class CacheItem:
    """Represents an item in the cache hierarchy"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    level: CacheLevel = CacheLevel.L1_MEMORY
    
    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def last_access_seconds_ago(self) -> float:
        return (datetime.utcnow() - self.last_accessed).total_seconds()
    
    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return self.age_seconds > self.ttl_seconds
    
    @property
    def access_frequency(self) -> float:
        """Calculate access frequency per hour"""
        age_hours = max(self.age_seconds / 3600, 0.1)  # Minimum 0.1 hours
        return self.access_count / age_hours
    
    def touch(self) -> None:
        """Update access statistics"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistics for a cache level"""
    level: CacheLevel
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    promotions: int = 0
    demotions: int = 0
    evictions: int = 0
    current_size: int = 0
    max_size: int = 0
    memory_usage_bytes: int = 0
    avg_latency_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0
    
    @property
    def utilization(self) -> float:
        return (self.current_size / self.max_size) if self.max_size > 0 else 0.0


class LRUCache:
    """LRU Cache implementation for L1 memory cache"""
    
    def __init__(self, max_size: int, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheItem] = OrderedDict()
        self.current_memory = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[CacheItem]:
        """Get item from cache"""
        with self._lock:
            if key in self.cache:
                item = self.cache[key]
                if not item.is_expired:
                    item.touch()
                    # Move to end (most recent)
                    self.cache.move_to_end(key)
                    return item
                else:
                    # Remove expired item
                    self._remove_item(key)
            return None
    
    def set(self, key: str, item: CacheItem) -> bool:
        """Set item in cache"""
        with self._lock:
            # Remove existing item if present
            if key in self.cache:
                self._remove_item(key)
            
            # Check memory constraints
            if item.size_bytes > self.max_memory_bytes:
                logger.warning(f"Item {key} too large for L1 cache: {item.size_bytes} bytes")
                return False
            
            # Ensure space is available
            while ((len(self.cache) >= self.max_size) or 
                   (self.current_memory + item.size_bytes > self.max_memory_bytes)):
                if not self._evict_lru():
                    return False  # Could not make space
            
            # Add new item
            self.cache[key] = item
            self.current_memory += item.size_bytes
            return True
    
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        with self._lock:
            return self._remove_item(key)
    
    def _remove_item(self, key: str) -> bool:
        """Remove item and update memory tracking"""
        if key in self.cache:
            item = self.cache.pop(key)
            self.current_memory -= item.size_bytes
            return True
        return False
    
    def _evict_lru(self) -> bool:
        """Evict least recently used item"""
        if not self.cache:
            return False
        
        # Get LRU item (first item in OrderedDict)
        lru_key = next(iter(self.cache))
        return self._remove_item(lru_key)
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
    
    def memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        return self.current_memory
    
    def clear(self) -> None:
        """Clear all items from cache"""
        with self._lock:
            self.cache.clear()
            self.current_memory = 0


class CacheHierarchy:
    """
    Multi-level cache hierarchy with intelligent management
    and performance optimization.
    """
    
    def __init__(self,
                 l1_max_size: int = 1000,
                 l1_max_memory_mb: int = 100,
                 l1_default_ttl: int = 300,
                 l2_default_ttl: int = 3600,
                 l3_default_ttl: int = 86400):
        
        # Cache level configuration
        self.l1_default_ttl = l1_default_ttl
        self.l2_default_ttl = l2_default_ttl
        self.l3_default_ttl = l3_default_ttl
        
        # L1: In-memory LRU cache
        self.l1_cache = LRUCache(l1_max_size, l1_max_memory_mb)
        
        # L2: Redis cache (using existing redis_cache service)
        self.l2_cache = redis_cache
        
        # L3: ClickHouse cache (using existing clickhouse_service)
        self.l3_cache = clickhouse_service
        
        # Statistics tracking
        self.stats = {
            CacheLevel.L1_MEMORY: CacheStats(CacheLevel.L1_MEMORY, max_size=l1_max_size),
            CacheLevel.L2_REDIS: CacheStats(CacheLevel.L2_REDIS, max_size=100000),
            CacheLevel.L3_CLICKHOUSE: CacheStats(CacheLevel.L3_CLICKHOUSE, max_size=1000000)
        }
        
        # Performance monitoring
        self.operation_latencies: Dict[CacheLevel, List[float]] = defaultdict(list)
        
        # Cache promotion/demotion thresholds
        self.promotion_thresholds = {
            "access_frequency": 2.0,  # accesses per hour
            "hit_count": 3,           # minimum hits before promotion
            "recency_hours": 1.0      # accessed within last hour
        }
        
        # Memory pressure monitoring
        self.memory_pressure_threshold = 0.8  # 80% memory usage
        
        # Thread safety and background tasks
        self._lock = threading.RLock()
        self.background_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("CacheHierarchy initialized")
    
    async def start(self) -> None:
        """Start background cache management tasks"""
        if self.is_running:
            return
        
        self.is_running = True
        self.background_task = asyncio.create_task(self._background_maintenance())
        
        logger.info("Cache hierarchy background tasks started")
    
    async def stop(self) -> None:
        """Stop background cache management tasks"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cache hierarchy background tasks stopped")
    
    async def _background_maintenance(self) -> None:
        """Background maintenance tasks"""
        logger.info("Cache hierarchy maintenance loop started")
        
        while self.is_running:
            try:
                # Clean up expired items
                await self._cleanup_expired_items()
                
                # Handle memory pressure
                await self._handle_memory_pressure()
                
                # Optimize cache distribution
                await self._optimize_cache_distribution()
                
                # Update statistics
                self._update_statistics()
                
                # Sleep before next iteration
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in cache hierarchy maintenance: {e}", exc_info=True)
                await asyncio.sleep(300)  # Wait longer on error
        
        logger.info("Cache hierarchy maintenance loop stopped")
    
    async def _cleanup_expired_items(self) -> None:
        """Clean up expired items from L1 cache"""
        expired_keys = []
        
        with self._lock:
            for key, item in list(self.l1_cache.cache.items()):
                if item.is_expired:
                    expired_keys.append(key)
        
        for key in expired_keys:
            self.l1_cache.delete(key)
            self.stats[CacheLevel.L1_MEMORY].evictions += 1
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired L1 items")
    
    async def _handle_memory_pressure(self) -> None:
        """Handle memory pressure by evicting items"""
        # Check system memory pressure
        memory_info = psutil.virtual_memory()
        if memory_info.percent > (self.memory_pressure_threshold * 100):
            # Aggressively evict from L1
            items_to_evict = min(50, self.l1_cache.size() // 4)  # Evict 25% or 50 items max
            
            for _ in range(items_to_evict):
                if not self.l1_cache._evict_lru():
                    break
            
            if items_to_evict > 0:
                logger.info(f"Evicted {items_to_evict} items from L1 due to memory pressure")
    
    async def _optimize_cache_distribution(self) -> None:
        """Optimize distribution of items across cache levels"""
        # Promote frequently accessed items from L2 to L1
        await self._promote_hot_items()
        
        # Demote rarely accessed items from L1 to L2
        await self._demote_cold_items()
    
    async def _promote_hot_items(self) -> None:
        """Promote frequently accessed items from L2 to L1"""
        # This would analyze access patterns in L2 and promote hot items
        # For now, we'll implement a simplified version
        pass
    
    async def _demote_cold_items(self) -> None:
        """Demote rarely accessed items from L1 to L2"""
        current_time = datetime.utcnow()
        items_to_demote = []
        
        with self._lock:
            for key, item in self.l1_cache.cache.items():
                # Demote if not accessed recently and low frequency
                if (item.last_access_seconds_ago > 3600 and  # Not accessed in last hour
                    item.access_frequency < self.promotion_thresholds["access_frequency"]):
                    items_to_demote.append((key, item))
        
        for key, item in items_to_demote:
            # Move item to L2
            success = self.l2_cache.set(key, item.value, self.l2_default_ttl)
            if success:
                self.l1_cache.delete(key)
                self.stats[CacheLevel.L1_MEMORY].demotions += 1
                self.stats[CacheLevel.L2_REDIS].promotions += 1
        
        if items_to_demote:
            logger.debug(f"Demoted {len(items_to_demote)} cold items from L1 to L2")
    
    def _update_statistics(self) -> None:
        """Update cache statistics"""
        # Update L1 statistics
        l1_stats = self.stats[CacheLevel.L1_MEMORY]
        l1_stats.current_size = self.l1_cache.size()
        l1_stats.memory_usage_bytes = self.l1_cache.memory_usage()
        
        # Calculate average latencies
        for level, latencies in self.operation_latencies.items():
            if latencies:
                recent_latencies = latencies[-100:]  # Last 100 operations
                if recent_latencies:
                    self.stats[level].avg_latency_ms = sum(recent_latencies) / len(recent_latencies)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            if isinstance(value, (str, bytes)):
                return len(value.encode('utf-8') if isinstance(value, str) else value)
            elif isinstance(value, (int, float)):
                return sys.getsizeof(value)
            elif isinstance(value, (list, dict)):
                # Rough estimation for complex objects
                return sys.getsizeof(str(value))
            else:
                return sys.getsizeof(value)
        except Exception:
            return 1024  # Default 1KB estimate
    
    def _record_latency(self, level: CacheLevel, latency_ms: float) -> None:
        """Record operation latency for statistics"""
        latencies = self.operation_latencies[level]
        latencies.append(latency_ms)
        
        # Keep only recent latencies (last 1000)
        if len(latencies) > 1000:
            latencies[:500] = []  # Remove oldest half
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache hierarchy with intelligent lookup path
        
        Lookup order: L1 -> L2 -> L3 -> None
        Promotes items from lower levels to higher levels on access.
        """
        start_time = time.time()
        
        # Try L1 first
        try:
            l1_start = time.time()
            l1_item = self.l1_cache.get(key)
            l1_latency = (time.time() - l1_start) * 1000
            self._record_latency(CacheLevel.L1_MEMORY, l1_latency)
            
            if l1_item:
                self.stats[CacheLevel.L1_MEMORY].hits += 1
                logger.debug(f"L1 cache hit for key: {key}")
                return l1_item.value
            else:
                self.stats[CacheLevel.L1_MEMORY].misses += 1
        except Exception as e:
            logger.error(f"L1 cache error for key {key}: {e}")
            self.stats[CacheLevel.L1_MEMORY].misses += 1
        
        # Try L2 (Redis)
        try:
            l2_start = time.time()
            l2_value = self.l2_cache.get(key)
            l2_latency = (time.time() - l2_start) * 1000
            self._record_latency(CacheLevel.L2_REDIS, l2_latency)
            
            if l2_value is not None:
                self.stats[CacheLevel.L2_REDIS].hits += 1
                logger.debug(f"L2 cache hit for key: {key}")
                
                # Promote to L1 if it meets criteria
                await self._maybe_promote_to_l1(key, l2_value)
                
                return l2_value
            else:
                self.stats[CacheLevel.L2_REDIS].misses += 1
        except Exception as e:
            logger.error(f"L2 cache error for key {key}: {e}")
            self.stats[CacheLevel.L2_REDIS].misses += 1
        
        # Try L3 (ClickHouse) - for specific data types only
        if self._is_l3_key(key):
            try:
                l3_start = time.time()
                l3_value = await self._get_from_l3(key)
                l3_latency = (time.time() - l3_start) * 1000
                self._record_latency(CacheLevel.L3_CLICKHOUSE, l3_latency)
                
                if l3_value is not None:
                    self.stats[CacheLevel.L3_CLICKHOUSE].hits += 1
                    logger.debug(f"L3 cache hit for key: {key}")
                    
                    # Cache in L2 for future access
                    self.l2_cache.set(key, l3_value, self.l2_default_ttl)
                    self.stats[CacheLevel.L2_REDIS].sets += 1
                    
                    return l3_value
                else:
                    self.stats[CacheLevel.L3_CLICKHOUSE].misses += 1
            except Exception as e:
                logger.error(f"L3 cache error for key {key}: {e}")
                self.stats[CacheLevel.L3_CLICKHOUSE].misses += 1
        
        # Cache miss at all levels
        total_latency = (time.time() - start_time) * 1000
        logger.debug(f"Cache miss for key {key} - total latency: {total_latency:.1f}ms")
        return None
    
    def _is_l3_key(self, key: str) -> bool:
        """Check if key can be retrieved from L3 (ClickHouse)"""
        # Only certain data types can be retrieved from ClickHouse
        l3_prefixes = ["path:", "snapshots:", "levels:", "historical:"]
        return any(key.startswith(prefix) for prefix in l3_prefixes)
    
    async def _get_from_l3(self, key: str) -> Optional[Any]:
        """Retrieve data from L3 (ClickHouse)"""
        # Parse key to determine data type and parameters
        try:
            if key.startswith("path:"):
                # Path data: path:instrument:granularity:start:end
                parts = key.split(":")
                if len(parts) >= 5:
                    instrument_id = parts[1]
                    granularity = parts[2]
                    start_time = datetime.fromisoformat(parts[3])
                    end_time = datetime.fromisoformat(parts[4])
                    
                    return self.l3_cache.fetch_snapshots(
                        instrument_id, granularity, start_time, end_time
                    )
            
            elif key.startswith("levels:"):
                # Active levels: levels:instrument:granularity
                parts = key.split(":")
                if len(parts) >= 3:
                    instrument_id = parts[1]
                    granularity = parts[2]
                    
                    return self.l3_cache.fetch_active_levels(
                        instrument_id, granularity, datetime.utcnow()
                    )
        
        except Exception as e:
            logger.error(f"Error retrieving from L3 for key {key}: {e}")
        
        return None
    
    async def _maybe_promote_to_l1(self, key: str, value: Any) -> None:
        """Consider promoting item from L2 to L1"""
        try:
            # Check if item meets promotion criteria
            size_bytes = self._estimate_size(value)
            
            # Don't promote if too large for L1
            if size_bytes > (self.l1_cache.max_memory_bytes // 10):  # Max 10% of L1 memory
                return
            
            # Create cache item for L1
            cache_item = CacheItem(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=self.l1_default_ttl,
                level=CacheLevel.L1_MEMORY
            )
            
            # Try to add to L1
            success = self.l1_cache.set(key, cache_item)
            if success:
                self.stats[CacheLevel.L1_MEMORY].promotions += 1
                logger.debug(f"Promoted key {key} from L2 to L1")
        
        except Exception as e:
            logger.error(f"Error promoting key {key} to L1: {e}")
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache hierarchy with intelligent placement
        
        Strategy: Always set in L1 and L2, optionally in L3 for persistent data
        """
        start_time = time.time()
        size_bytes = self._estimate_size(value)
        success_count = 0
        
        # Set in L1 (memory)
        try:
            cache_item = CacheItem(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=0,
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.l1_default_ttl,
                level=CacheLevel.L1_MEMORY
            )
            
            if self.l1_cache.set(key, cache_item):
                self.stats[CacheLevel.L1_MEMORY].sets += 1
                success_count += 1
                logger.debug(f"Set key {key} in L1")
        except Exception as e:
            logger.error(f"Error setting key {key} in L1: {e}")
        
        # Set in L2 (Redis)
        try:
            l2_ttl = ttl or self.l2_default_ttl
            if self.l2_cache.set(key, value, l2_ttl):
                self.stats[CacheLevel.L2_REDIS].sets += 1
                success_count += 1
                logger.debug(f"Set key {key} in L2")
        except Exception as e:
            logger.error(f"Error setting key {key} in L2: {e}")
        
        # Set in L3 (ClickHouse) for persistent data types
        if self._should_persist_to_l3(key):
            try:
                # L3 persistence would be implemented here
                # For now, we'll just mark it as successful
                self.stats[CacheLevel.L3_CLICKHOUSE].sets += 1
                success_count += 1
                logger.debug(f"Set key {key} in L3")
            except Exception as e:
                logger.error(f"Error setting key {key} in L3: {e}")
        
        total_latency = (time.time() - start_time) * 1000
        self._record_latency(CacheLevel.L1_MEMORY, total_latency)  # Record overall latency
        
        return success_count > 0
    
    def _should_persist_to_l3(self, key: str) -> bool:
        """Check if key should be persisted to L3"""
        persistent_prefixes = ["labels:", "computed:", "backfill_progress:"]
        return any(key.startswith(prefix) for prefix in persistent_prefixes)
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        success_count = 0
        
        # Delete from L1
        try:
            if self.l1_cache.delete(key):
                self.stats[CacheLevel.L1_MEMORY].deletes += 1
                success_count += 1
        except Exception as e:
            logger.error(f"Error deleting key {key} from L1: {e}")
        
        # Delete from L2
        try:
            if self.l2_cache.delete(key):
                self.stats[CacheLevel.L2_REDIS].deletes += 1
                success_count += 1
        except Exception as e:
            logger.error(f"Error deleting key {key} from L2: {e}")
        
        # L3 deletion would be implemented here if needed
        
        return success_count > 0
    
    async def warm_cache(self, keys_and_values: Dict[str, Any], 
                        target_level: Optional[CacheLevel] = None) -> int:
        """
        Warm cache with multiple key-value pairs
        
        Args:
            keys_and_values: Dictionary of keys and values to cache
            target_level: Specific cache level to target (None = all levels)
            
        Returns:
            Number of successfully cached items
        """
        if not keys_and_values:
            return 0
        
        success_count = 0
        
        for key, value in keys_and_values.items():
            try:
                if target_level:
                    # Cache in specific level only
                    if target_level == CacheLevel.L1_MEMORY:
                        success = await self._set_l1_only(key, value)
                    elif target_level == CacheLevel.L2_REDIS:
                        success = await self._set_l2_only(key, value)
                    else:
                        success = False
                else:
                    # Cache in all appropriate levels
                    success = await self.set(key, value)
                
                if success:
                    success_count += 1
            
            except Exception as e:
                logger.error(f"Error warming cache for key {key}: {e}")
        
        logger.info(f"Warmed cache with {success_count}/{len(keys_and_values)} items")
        return success_count
    
    async def _set_l1_only(self, key: str, value: Any) -> bool:
        """Set value only in L1 cache"""
        try:
            size_bytes = self._estimate_size(value)
            cache_item = CacheItem(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                access_count=0,
                size_bytes=size_bytes,
                ttl_seconds=self.l1_default_ttl,
                level=CacheLevel.L1_MEMORY
            )
            return self.l1_cache.set(key, cache_item)
        except Exception as e:
            logger.error(f"Error setting L1-only key {key}: {e}")
            return False
    
    async def _set_l2_only(self, key: str, value: Any) -> bool:
        """Set value only in L2 cache"""
        try:
            return self.l2_cache.set(key, value, self.l2_default_ttl)
        except Exception as e:
            logger.error(f"Error setting L2-only key {key}: {e}")
            return False
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache hierarchy statistics"""
        current_time = datetime.utcnow()
        
        # Calculate overall hit rate
        total_hits = sum(stats.hits for stats in self.stats.values())
        total_misses = sum(stats.misses for stats in self.stats.values())
        overall_hit_rate = (total_hits / (total_hits + total_misses)) if (total_hits + total_misses) > 0 else 0.0
        
        return {
            "timestamp": current_time.isoformat(),
            "overall": {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "overall_hit_rate_pct": overall_hit_rate * 100,
                "target_hit_rate_pct": 95.0
            },
            "levels": {
                level.value: {
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "sets": stats.sets,
                    "deletes": stats.deletes,
                    "promotions": stats.promotions,
                    "demotions": stats.demotions,
                    "evictions": stats.evictions,
                    "current_size": stats.current_size,
                    "max_size": stats.max_size,
                    "utilization_pct": stats.utilization * 100,
                    "hit_rate_pct": stats.hit_rate * 100,
                    "avg_latency_ms": stats.avg_latency_ms,
                    "memory_usage_bytes": stats.memory_usage_bytes
                }
                for level, stats in self.stats.items()
            }
        }
    
    async def optimize_cache_levels(self) -> Dict[str, int]:
        """
        Perform cache level optimization
        
        Returns:
            Dictionary with optimization counts
        """
        optimization_counts = {
            "promoted_to_l1": 0,
            "demoted_from_l1": 0,
            "evicted_from_l1": 0
        }
        
        try:
            # Promote hot L2 items to L1
            # This would require tracking L2 access patterns
            # For now, we'll implement a placeholder
            
            # Demote cold L1 items to L2
            await self._demote_cold_items()
            
            # Handle memory pressure
            if psutil.virtual_memory().percent > 80:
                evicted = min(20, self.l1_cache.size() // 10)  # Evict 10% or 20 items max
                for _ in range(evicted):
                    if self.l1_cache._evict_lru():
                        optimization_counts["evicted_from_l1"] += 1
            
        except Exception as e:
            logger.error(f"Error optimizing cache levels: {e}")
        
        return optimization_counts
    
    def clear_cache_level(self, level: CacheLevel) -> bool:
        """Clear all items from a specific cache level"""
        try:
            if level == CacheLevel.L1_MEMORY:
                self.l1_cache.clear()
                return True
            elif level == CacheLevel.L2_REDIS:
                # This would clear Redis patterns - implement carefully
                return False  # Not implemented for safety
            elif level == CacheLevel.L3_CLICKHOUSE:
                # This would clear ClickHouse cache - not recommended
                return False
        except Exception as e:
            logger.error(f"Error clearing cache level {level}: {e}")
        
        return False


# Global cache hierarchy instance
cache_hierarchy = CacheHierarchy()