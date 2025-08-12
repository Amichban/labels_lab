"""
FastAPI application for Label Computation System

High-performance label computation system for quantitative trading that processes
support/resistance level events and computes forward-looking labels for pattern
mining and ML models.

Key Features:
- Dual-mode processing: Batch backfill (minutes) and real-time incremental (<1 second)
- Multi-timeframe alignment using lower granularity data for path-dependent calculations
- Support for 29 FX pairs and indices with billions of labels

Critical SLAs:
- Incremental compute: <100ms p99 latency
- Batch throughput: 1M+ candles/minute
- Cache hit rate: >95% for recent 24 hours
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.exceptions import HTTPException as StarletteHTTPException

from config.settings import settings
from src.api.routers import labels, batch, health, cache, metrics
from src.api.middleware import (
    AuthenticationMiddleware,
    RequestTracingMiddleware,
    RateLimitingMiddleware,
    ErrorHandlingMiddleware
)
from src.services.clickhouse_service import clickhouse_service
from src.services.redis_cache import redis_cache
from src.services.resilience_init import (
    initialize_resilience,
    shutdown_resilience,
    health_check_endpoint
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

LABEL_COMPUTATION_COUNT = Counter(
    'label_computations_total',
    'Total label computations',
    ['instrument_id', 'granularity', 'label_type']
)

LABEL_COMPUTATION_DURATION = Histogram(
    'label_computation_duration_seconds',
    'Label computation duration in seconds',
    ['label_type']
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with resilience system integration"""
    # Startup
    logger.info("Starting Label Computation System...")
    
    # Initialize resilience system first
    try:
        logger.info("Initializing resilience system...")
        resilience_status = await initialize_resilience()
        logger.info(
            f"Resilience system initialized in {resilience_status.get('duration_ms', 0):.0f}ms - "
            f"{resilience_status.get('services_registered', 0)} services registered"
        )
        
        # Store resilience status in app state
        app.state.resilience_status = resilience_status
        
    except Exception as e:
        logger.error(f"Failed to initialize resilience system: {e}")
        # Continue with basic connection checks as fallback
        logger.warning("Falling back to basic connection checks...")
    
    # Check database connections (with circuit breaker protection if available)
    clickhouse_healthy = clickhouse_service.check_connection()
    redis_healthy = redis_cache.check_connection()
    
    if not clickhouse_healthy:
        logger.warning("ClickHouse connection failed - circuit breaker protection active")
    
    if not redis_healthy:
        logger.warning("Redis connection failed - circuit breaker protection active")
    
    # With resilience system, we can continue even if some services are down
    services_healthy = sum([clickhouse_healthy, redis_healthy])
    logger.info(f"Service health check: {services_healthy}/2 services healthy")
    
    if services_healthy == 0:
        logger.error("All critical services failed - cannot start application")
        raise RuntimeError("All services failed to connect")
    
    # Initialize application state
    app.state.start_time = datetime.utcnow()
    app.state.request_count = 0
    app.state.services_healthy = services_healthy
    
    logger.info("Label Computation System startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Label Computation System...")
    
    try:
        # Shutdown resilience system gracefully
        await shutdown_resilience()
        logger.info("Resilience system shutdown complete")
    except Exception as e:
        logger.error(f"Error during resilience system shutdown: {e}")
    
    # Close service connections
    clickhouse_service.close()
    redis_cache.close()
    
    logger.info("Label Computation System shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Label Computation System API",
    description="""
    High-performance label computation system for quantitative trading that processes support/resistance level events 
    and computes forward-looking labels for pattern mining and ML models.
    
    Key features:
    - Dual-mode processing: Batch backfill (minutes) and real-time incremental (<1 second)
    - Multi-timeframe alignment using lower granularity data for path-dependent calculations
    - Support for 29 FX pairs and indices with billions of labels
    
    Critical SLAs:
    - Incremental compute: <100ms p99 latency
    - Batch throughput: 1M+ candles/minute
    - Cache hit rate: >95% for recent 24 hours
    """,
    version=settings.app_version,
    contact={
        "name": "Label Computation System Team",
        "email": "support@labelcompute.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestTracingMiddleware)
app.add_middleware(RateLimitingMiddleware)
app.add_middleware(AuthenticationMiddleware)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware for collecting Prometheus metrics"""
    start_time = time.time()
    
    # Generate trace ID if not present
    if "x-request-id" not in request.headers:
        request.state.trace_id = str(uuid.uuid4())
    else:
        request.state.trace_id = request.headers["x-request-id"]
    
    # Increment request count
    app.state.request_count += 1
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    endpoint = request.url.path
    method = request.method
    status_code = str(response.status_code)
    
    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status_code=status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)
    
    # Add performance headers
    response.headers["X-Response-Time-Ms"] = str(int(duration * 1000))
    response.headers["X-Request-ID"] = request.state.trace_id
    
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with consistent error format"""
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    
    error_response = {
        "error": {
            "code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "trace_id": trace_id
        }
    }
    
    logger.warning(
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={"trace_id": trace_id, "path": request.url.path}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response,
        headers={"X-Request-ID": trace_id}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    
    error_response = {
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An internal error occurred",
            "trace_id": trace_id
        }
    }
    
    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={"trace_id": trace_id, "path": request.url.path},
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response,
        headers={"X-Request-ID": trace_id}
    )


# Include API routers
app.include_router(
    labels.router,
    prefix="/v1",
    tags=["Labels"]
)

app.include_router(
    batch.router,
    prefix="/v1",
    tags=["Batch"]
)

app.include_router(
    health.router,
    prefix="/v1",
    tags=["Health"]
)

app.include_router(
    cache.router,
    prefix="/v1",
    tags=["Cache"]
)

app.include_router(
    metrics.router,
    prefix="/v1",
    tags=["Metrics"]
)


@app.get("/metrics", response_class=Response)
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    if not settings.prometheus_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics endpoint disabled"
        )
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "operational",
        "endpoints": {
            "docs": "/docs" if settings.debug else "disabled",
            "health": "/v1/health",
            "metrics": "/metrics" if settings.prometheus_enabled else "disabled"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=1,  # Use 1 worker for development
        log_level="info",
        reload=settings.debug,
        access_log=True
    )