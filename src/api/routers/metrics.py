"""
System metrics endpoint

Provides detailed system metrics for monitoring and observability:
- /metrics - System metrics in JSON or Prometheus format
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request, Query, status
from fastapi.responses import Response, JSONResponse

from src.api.schemas import MetricsResponse, PerformanceMetrics, CacheMetrics, BusinessMetrics
from src.services.redis_cache import redis_cache
from src.services.clickhouse_service import clickhouse_service
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class MetricsCollector:
    """Collects and aggregates system metrics"""
    
    def __init__(self):
        self.metrics_cache = {}
        self.cache_ttl = 60  # Cache metrics for 60 seconds
    
    async def get_performance_metrics(self, window: str = "1h") -> PerformanceMetrics:
        """Get performance metrics for the specified time window"""
        try:
            # In production, these would come from actual metrics collection
            # For demo purposes, return realistic sample values
            
            window_minutes = self._parse_window_minutes(window)
            
            # Get request count from Redis metrics
            total_requests = redis_cache.get_metric("http_requests_total") or 1000
            requests_per_second = total_requests / (window_minutes * 60)
            
            # Sample performance metrics - would be calculated from actual data
            performance_metrics = PerformanceMetrics(
                avg_computation_time_ms=42.5,
                p50_computation_time_ms=35.0,
                p95_computation_time_ms=85.0,
                p99_computation_time_ms=150.0,
                requests_per_second=min(requests_per_second, 50.0),  # Cap for demo
                error_rate=0.001  # 0.1% error rate
            )
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics()
    
    async def get_cache_metrics(self) -> CacheMetrics:
        """Get cache performance metrics"""
        try:
            # Get Redis info
            redis_info = redis_cache.client.info()
            
            # Calculate hit rate
            cache_hits = redis_cache.get_metric("cache_hits_labels") or 0
            cache_misses = redis_cache.get_metric("cache_misses_labels") or 0
            total_cache_requests = cache_hits + cache_misses
            hit_rate = (cache_hits / total_cache_requests) if total_cache_requests > 0 else 0.0
            
            # Memory usage from Redis info
            memory_usage_bytes = redis_info.get("used_memory", 0)
            memory_usage_mb = memory_usage_bytes / (1024 * 1024)
            
            # Evictions (from Redis info)
            evictions = redis_info.get("evicted_keys", 0)
            
            # Total keys
            total_keys = redis_cache.client.dbsize()
            
            cache_metrics = CacheMetrics(
                hit_rate=hit_rate,
                memory_usage_mb=memory_usage_mb,
                evictions_per_minute=evictions / 60,  # Rough approximation
                keys_total=total_keys
            )
            
            return cache_metrics
            
        except Exception as e:
            logger.error(f"Failed to get cache metrics: {e}")
            return CacheMetrics()
    
    async def get_business_metrics(self, window: str = "1h") -> BusinessMetrics:
        """Get business-specific metrics"""
        try:
            # Get label computation metrics
            labels_computed = redis_cache.get_metric("labels_computed_total") or 0
            
            # Count unique instruments (would query ClickHouse in production)
            unique_instruments = 29  # Sample value for demo
            
            # Active batch jobs (would query job manager)
            active_batch_jobs = 0  # Placeholder
            
            # Batch throughput
            window_minutes = self._parse_window_minutes(window)
            throughput = labels_computed / max(window_minutes, 1)
            
            business_metrics = BusinessMetrics(
                labels_computed_total=labels_computed,
                unique_instruments=unique_instruments,
                active_batch_jobs=active_batch_jobs,
                avg_batch_throughput_candles_per_min=throughput
            )
            
            return business_metrics
            
        except Exception as e:
            logger.error(f"Failed to get business metrics: {e}")
            return BusinessMetrics()
    
    def _parse_window_minutes(self, window: str) -> int:
        """Parse time window string to minutes"""
        window_map = {
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "6h": 360,
            "24h": 1440
        }
        return window_map.get(window, 60)
    
    async def get_prometheus_metrics(self) -> str:
        """Generate Prometheus format metrics"""
        try:
            # Get all metrics
            performance = await self.get_performance_metrics()
            cache = await self.get_cache_metrics()
            business = await self.get_business_metrics()
            
            # Generate Prometheus format
            metrics_lines = []
            
            # Performance metrics
            if performance.avg_computation_time_ms:
                metrics_lines.append(
                    f"# HELP label_computation_duration_seconds Average label computation time\n"
                    f"# TYPE label_computation_duration_seconds gauge\n"
                    f"label_computation_duration_seconds {performance.avg_computation_time_ms / 1000:.6f}"
                )
            
            if performance.requests_per_second:
                metrics_lines.append(
                    f"# HELP http_requests_per_second HTTP requests per second\n"
                    f"# TYPE http_requests_per_second gauge\n"
                    f"http_requests_per_second {performance.requests_per_second:.2f}"
                )
            
            if performance.error_rate:
                metrics_lines.append(
                    f"# HELP error_rate Error rate (0-1)\n"
                    f"# TYPE error_rate gauge\n"
                    f"error_rate {performance.error_rate:.6f}"
                )
            
            # Cache metrics
            if cache.hit_rate:
                metrics_lines.append(
                    f"# HELP cache_hit_rate Cache hit rate (0-1)\n"
                    f"# TYPE cache_hit_rate gauge\n"
                    f"cache_hit_rate {cache.hit_rate:.6f}"
                )
            
            if cache.memory_usage_mb:
                metrics_lines.append(
                    f"# HELP cache_memory_usage_megabytes Cache memory usage in MB\n"
                    f"# TYPE cache_memory_usage_megabytes gauge\n"
                    f"cache_memory_usage_megabytes {cache.memory_usage_mb:.2f}"
                )
            
            if cache.keys_total:
                metrics_lines.append(
                    f"# HELP cache_keys_total Total number of cache keys\n"
                    f"# TYPE cache_keys_total gauge\n"
                    f"cache_keys_total {cache.keys_total}"
                )
            
            # Business metrics
            if business.labels_computed_total:
                metrics_lines.append(
                    f"# HELP labels_computed_total Total labels computed\n"
                    f"# TYPE labels_computed_total counter\n"
                    f"labels_computed_total {business.labels_computed_total}"
                )
            
            if business.unique_instruments:
                metrics_lines.append(
                    f"# HELP unique_instruments_total Number of unique instruments\n"
                    f"# TYPE unique_instruments_total gauge\n"
                    f"unique_instruments_total {business.unique_instruments}"
                )
            
            if business.active_batch_jobs is not None:
                metrics_lines.append(
                    f"# HELP active_batch_jobs Number of active batch jobs\n"
                    f"# TYPE active_batch_jobs gauge\n"
                    f"active_batch_jobs {business.active_batch_jobs}"
                )
            
            return "\n".join(metrics_lines) + "\n"
            
        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {e}")
            return "# Error generating metrics\n"


# Global metrics collector
metrics_collector = MetricsCollector()


@router.get(
    "/metrics",
    summary="System metrics",
    description="""
    Detailed system metrics for monitoring and observability.
    Includes performance, cache, and business metrics.
    
    Supports both JSON and Prometheus formats.
    """,
    responses={
        200: {"description": "Metrics retrieved successfully"}
    }
)
async def get_metrics(
    request: Request,
    format: str = Query(
        "json",
        description="Response format",
        regex="^(json|prometheus)$"
    ),
    window: str = Query(
        "1h",
        description="Time window for metrics",
        regex="^(5m|15m|1h|6h|24h)$"
    )
):
    """
    Get detailed system metrics for monitoring and observability.
    
    Supports both JSON and Prometheus formats for integration with
    different monitoring systems.
    """
    start_time = time.time()
    trace_id = getattr(request.state, "trace_id", "unknown")
    
    try:
        if format == "prometheus":
            # Return Prometheus format
            if not settings.prometheus_enabled:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Prometheus metrics disabled"
                )
            
            prometheus_data = await metrics_collector.get_prometheus_metrics()
            
            return Response(
                content=prometheus_data,
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
        
        else:
            # Return JSON format
            performance = await metrics_collector.get_performance_metrics(window)
            cache = await metrics_collector.get_cache_metrics()
            business = await metrics_collector.get_business_metrics(window)
            
            response = MetricsResponse(
                timestamp=datetime.utcnow(),
                window=window,
                performance=performance,
                cache=cache,
                business=business
            )
            
            # Add response timing
            response_time_ms = int((time.time() - start_time) * 1000)
            
            return JSONResponse(
                content=response.dict(),
                headers={
                    "X-Response-Time-Ms": str(response_time_ms),
                    "X-Request-ID": trace_id
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get metrics: {str(e)}",
            extra={"trace_id": trace_id, "format": format, "window": window},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "METRICS_FAILED",
                    "message": "Failed to retrieve metrics",
                    "trace_id": trace_id
                }
            }
        )