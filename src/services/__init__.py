"""
Services package with integrated resilience features

This package provides all external service integrations with built-in:
- Circuit breaker protection
- Automatic retry with exponential backoff
- Fallback handlers for graceful degradation
- Health monitoring and recovery
- System-wide resilience coordination

Issue #14: Circuit breakers and failover mechanisms
"""

# Import all services with resilience features
from .clickhouse_service import ClickHouseService, clickhouse_service
from .redis_cache import RedisCacheService, redis_cache
from .firestore_listener import FirestoreListener, firestore_listener

# Import resilience components
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerError,
    ExponentialBackoff,
    circuit_breaker,
    with_retry,
    get_circuit_breaker,
    get_all_circuit_breakers,
    get_circuit_breaker_summary
)

from .resilience_manager import (
    ResilienceManager,
    SystemHealthState,
    get_resilience_manager,
    initialize_resilience_system
)

from .fallback_handlers import (
    ClickHouseFallbackHandler,
    RedisFallbackHandler, 
    FirestoreFallbackHandler,
    FallbackOrchestrator,
    get_fallback_handler,
    get_fallback_orchestrator
)

from .resilience_init import (
    ResilienceInitializer,
    initialize_resilience,
    shutdown_resilience,
    get_resilience_status,
    get_resilience_status_async,
    force_service_recovery,
    get_degraded_mode_config,
    health_check_endpoint
)

__all__ = [
    # Core services
    'ClickHouseService',
    'clickhouse_service',
    'RedisCacheService', 
    'redis_cache',
    'FirestoreListener',
    'firestore_listener',
    
    # Circuit breaker components
    'CircuitBreaker',
    'CircuitBreakerConfig',
    'CircuitBreakerState',
    'CircuitBreakerError',
    'ExponentialBackoff',
    'circuit_breaker',
    'with_retry',
    'get_circuit_breaker',
    'get_all_circuit_breakers',
    'get_circuit_breaker_summary',
    
    # Resilience management
    'ResilienceManager',
    'SystemHealthState',
    'get_resilience_manager',
    'initialize_resilience_system',
    
    # Fallback handlers
    'ClickHouseFallbackHandler',
    'RedisFallbackHandler',
    'FirestoreFallbackHandler', 
    'FallbackOrchestrator',
    'get_fallback_handler',
    'get_fallback_orchestrator',
    
    # Resilience initialization
    'ResilienceInitializer',
    'initialize_resilience',
    'shutdown_resilience',
    'get_resilience_status',
    'get_resilience_status_async',
    'force_service_recovery',
    'get_degraded_mode_config',
    'health_check_endpoint'
]