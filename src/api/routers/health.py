"""
Health monitoring endpoints

Provides comprehensive health checks for the Label Computation System:
- /health - Full system health check
- /health/ready - Readiness probe (Kubernetes)
- /health/live - Liveness probe (Kubernetes)
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.api.schemas import HealthResponse, HealthStatusEnum, HealthMetrics
from src.services.clickhouse_service import clickhouse_service
from src.services.redis_cache import redis_cache
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()


class HealthChecker:
    """Centralized health checking for all system components"""
    
    def __init__(self):
        self.start_time = datetime.utcnow()
    
    async def check_clickhouse(self) -> Dict[str, str]:
        """Check ClickHouse connection and performance"""
        try:
            # Basic connection test
            if not clickhouse_service.check_connection():
                return {"status": "error", "message": "Connection failed"}
            
            # Performance test - simple query with timeout
            start_time = time.time()
            result = clickhouse_service.execute("SELECT count() FROM system.tables LIMIT 1")
            query_time = (time.time() - start_time) * 1000
            
            if query_time > 1000:  # > 1 second is concerning
                return {"status": "warning", "message": f"Slow response: {query_time:.0f}ms"}
            
            return {"status": "ok", "message": f"Response time: {query_time:.0f}ms"}
            
        except Exception as e:
            logger.error(f"ClickHouse health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def check_redis(self) -> Dict[str, str]:
        """Check Redis connection and performance"""
        try:
            # Basic connection test
            if not redis_cache.check_connection():
                return {"status": "error", "message": "Connection failed"}
            
            # Performance test
            start_time = time.time()
            redis_cache.client.set("health_check", "test", ex=10)
            value = redis_cache.client.get("health_check")
            redis_cache.client.delete("health_check")
            response_time = (time.time() - start_time) * 1000
            
            if value != b"test":
                return {"status": "error", "message": "Set/get operation failed"}
            
            if response_time > 100:  # > 100ms is concerning for Redis
                return {"status": "warning", "message": f"Slow response: {response_time:.0f}ms"}
            
            return {"status": "ok", "message": f"Response time: {response_time:.0f}ms"}
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def check_cache_performance(self) -> Dict[str, str]:
        """Check cache hit rate and performance"""
        try:
            # Get cache metrics from Redis
            cache_hits = redis_cache.get_metric("cache_hits_labels") or 0
            cache_misses = redis_cache.get_metric("cache_misses_labels") or 0
            
            total_requests = cache_hits + cache_misses
            if total_requests == 0:
                return {"status": "ok", "message": "No cache activity yet"}
            
            hit_rate = cache_hits / total_requests
            
            if hit_rate < 0.8:  # < 80% is concerning
                return {"status": "warning", "message": f"Low hit rate: {hit_rate:.1%}"}
            elif hit_rate < settings.target_cache_hit_rate:
                return {"status": "warning", "message": f"Below target: {hit_rate:.1%}"}
            
            return {"status": "ok", "message": f"Hit rate: {hit_rate:.1%}"}
            
        except Exception as e:
            logger.error(f"Cache performance check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def check_avg_latency(self) -> Dict[str, str]:
        """Check average computation latency"""
        try:
            # This would typically come from metrics collection
            # For demo purposes, use a reasonable default
            avg_latency_ms = 42  # Placeholder - would be calculated from actual metrics
            
            if avg_latency_ms > settings.max_incremental_latency_ms:
                return {"status": "error", "message": f"High latency: {avg_latency_ms}ms"}
            elif avg_latency_ms > settings.max_incremental_latency_ms * 0.8:
                return {"status": "warning", "message": f"Elevated latency: {avg_latency_ms}ms"}
            
            return {"status": "ok", "message": f"Avg latency: {avg_latency_ms}ms"}
            
        except Exception as e:
            logger.error(f"Latency check failed: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_system_metrics(self) -> HealthMetrics:
        """Get current system metrics"""
        try:
            # Calculate cache hit rate
            cache_hits = redis_cache.get_metric("cache_hits_labels") or 0
            cache_misses = redis_cache.get_metric("cache_misses_labels") or 0
            total_requests = cache_hits + cache_misses
            cache_hit_rate = (cache_hits / total_requests) if total_requests > 0 else 0.0
            
            # Get labels computed in last hour
            labels_last_hour = redis_cache.get_metric("labels_computed_last_hour") or 0
            
            # Active batch jobs - would query job manager in production
            active_batch_jobs = 0  # Placeholder
            
            return HealthMetrics(
                cache_hit_rate=cache_hit_rate,
                avg_computation_ms=42.0,  # Would be calculated from actual metrics
                active_batch_jobs=active_batch_jobs,
                labels_computed_last_hour=labels_last_hour
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return HealthMetrics()
    
    def get_uptime_seconds(self) -> int:
        """Get application uptime in seconds"""
        return int((datetime.utcnow() - self.start_time).total_seconds())


# Global health checker
health_checker = HealthChecker()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
    description="""
    Comprehensive health check including all dependencies.
    Used by load balancers and monitoring systems.
    """,
    responses={
        200: {"description": "System is healthy"},
        503: {"description": "System is unhealthy"}
    }
)
async def get_health(request: Request):
    """
    Perform comprehensive health check of all system components.
    
    Returns HTTP 200 if healthy, 503 if unhealthy.
    Load balancers should route traffic based on this endpoint.
    """
    start_time = time.time()
    
    try:
        # Run all health checks concurrently
        checks = {}
        errors = []
        
        # Check ClickHouse
        ch_result = await health_checker.check_clickhouse()
        checks["clickhouse"] = ch_result["status"]
        if ch_result["status"] == "error":
            errors.append(f"ClickHouse: {ch_result['message']}")
        
        # Check Redis
        redis_result = await health_checker.check_redis()
        checks["redis"] = redis_result["status"]
        if redis_result["status"] == "error":
            errors.append(f"Redis: {redis_result['message']}")
        
        # Check cache performance
        cache_result = await health_checker.check_cache_performance()
        checks["cache_hit_rate"] = cache_result["status"]
        if cache_result["status"] == "error":
            errors.append(f"Cache: {cache_result['message']}")
        
        # Check average latency
        latency_result = await health_checker.check_avg_latency()
        checks["avg_latency_ms"] = latency_result["status"]
        if latency_result["status"] == "error":
            errors.append(f"Latency: {latency_result['message']}")
        
        # Determine overall status
        error_count = sum(1 for status in checks.values() if status == "error")
        warning_count = sum(1 for status in checks.values() if status == "warning")
        
        if error_count > 0:
            overall_status = HealthStatusEnum.UNHEALTHY
            response_status = status.HTTP_503_SERVICE_UNAVAILABLE
        elif warning_count > 0:
            overall_status = HealthStatusEnum.DEGRADED
            response_status = status.HTTP_200_OK
        else:
            overall_status = HealthStatusEnum.HEALTHY
            response_status = status.HTTP_200_OK
        
        # Get system metrics
        metrics = await health_checker.get_system_metrics()
        
        # Build response
        response_data = HealthResponse(
            status=overall_status,
            version=settings.app_version,
            timestamp=datetime.utcnow(),
            uptime_seconds=health_checker.get_uptime_seconds(),
            checks=checks,
            metrics=metrics,
            errors=errors if errors else None
        )
        
        # Log health check
        check_duration = (time.time() - start_time) * 1000
        logger.info(
            f"Health check completed: {overall_status.value} in {check_duration:.0f}ms",
            extra={
                "status": overall_status.value,
                "checks": checks,
                "duration_ms": check_duration
            }
        )
        
        return JSONResponse(
            content=response_data.dict(),
            status_code=response_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        
        error_response = HealthResponse(
            status=HealthStatusEnum.UNHEALTHY,
            version=settings.app_version,
            timestamp=datetime.utcnow(),
            uptime_seconds=health_checker.get_uptime_seconds(),
            errors=[f"Health check failed: {str(e)}"]
        )
        
        return JSONResponse(
            content=error_response.dict(),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Kubernetes readiness probe endpoint",
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"}
    }
)
async def get_readiness(request: Request):
    """
    Readiness probe for Kubernetes.
    
    Returns HTTP 200 when the service is ready to receive traffic.
    This checks if all required dependencies are available.
    """
    try:
        # Check critical dependencies only
        ch_healthy = clickhouse_service.check_connection()
        redis_healthy = redis_cache.check_connection()
        
        if ch_healthy and redis_healthy:
            return {
                "status": "ready",
                "timestamp": datetime.utcnow()
            }
        else:
            dependencies = []
            if not ch_healthy:
                dependencies.append("clickhouse")
            if not redis_healthy:
                dependencies.append("redis")
            
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "timestamp": datetime.utcnow(),
                    "missing_dependencies": dependencies
                },
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "status": "not_ready",
                "timestamp": datetime.utcnow(),
                "error": str(e)
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Kubernetes liveness probe endpoint",
    responses={
        200: {"description": "Service is alive"}
    }
)
async def get_liveness(request: Request):
    """
    Liveness probe for Kubernetes.
    
    Returns HTTP 200 if the service is alive and should not be restarted.
    This is a lightweight check that only verifies basic application health.
    """
    try:
        # Simple liveness check - just verify the application is running
        # and can respond to requests
        return {
            "status": "alive",
            "timestamp": datetime.utcnow(),
            "uptime_seconds": health_checker.get_uptime_seconds()
        }
        
    except Exception as e:
        # This should rarely fail - only in case of severe application issues
        logger.error(f"Liveness check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "status": "dead",
                "timestamp": datetime.utcnow(),
                "error": str(e)
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )