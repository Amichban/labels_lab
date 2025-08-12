"""
Resilience manager for coordinating circuit breakers and system health

Provides centralized management of all resilience mechanisms including:
- Circuit breaker coordination
- Health monitoring
- Automatic recovery
- Degraded mode operations
- System-wide fallback strategies

Issue #14: Circuit breakers and failover mechanisms
Following infra-pr best practices for production resilience
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict

from .circuit_breaker import (
    CircuitBreaker, 
    CircuitBreakerConfig, 
    CircuitBreakerState,
    get_all_circuit_breakers,
    register_circuit_breaker
)

logger = logging.getLogger(__name__)


class SystemHealthState(Enum):
    """Overall system health states"""
    HEALTHY = "healthy"           # All services operational
    DEGRADED = "degraded"         # Some services down, fallbacks active
    CRITICAL = "critical"         # Major services down
    EMERGENCY = "emergency"       # System in emergency mode


@dataclass
class ServiceDependency:
    """Service dependency configuration"""
    name: str
    service_type: str  # 'database', 'cache', 'stream', 'external'
    criticality: str   # 'critical', 'important', 'optional'
    fallback_available: bool = False
    max_downtime_minutes: int = 5
    recovery_check_interval: int = 30  # seconds


@dataclass
class ResilienceMetrics:
    """System-wide resilience metrics"""
    total_services: int = 0
    healthy_services: int = 0
    degraded_services: int = 0
    failed_services: int = 0
    fallbacks_active: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    last_health_check: Optional[datetime] = None
    uptime_percentage: float = 100.0
    
    def calculate_health_percentage(self) -> float:
        """Calculate overall system health percentage"""
        if self.total_services == 0:
            return 100.0
        return (self.healthy_services / self.total_services) * 100.0


class ResilienceManager:
    """
    Central coordination point for all resilience mechanisms.
    
    Responsibilities:
    - Register and manage circuit breakers for all external services
    - Monitor system health and coordinate fallback strategies
    - Implement automatic recovery procedures
    - Provide degraded mode operations
    - Maintain system-wide resilience metrics
    """
    
    def __init__(self):
        """Initialize resilience manager"""
        self.services: Dict[str, ServiceDependency] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        
        # Health monitoring
        self.system_health = SystemHealthState.HEALTHY
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        self.last_health_check = datetime.utcnow()
        
        # Recovery management
        self.recovery_tasks: Dict[str, asyncio.Task] = {}
        self.recovery_in_progress: Set[str] = set()
        
        # Metrics
        self.metrics = ResilienceMetrics()
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.emergency_mode_threshold = 0.5  # Trigger emergency if <50% services healthy
        self.degraded_mode_threshold = 0.8   # Trigger degraded if <80% services healthy
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info("ResilienceManager initialized")
    
    def register_service(self, 
                        name: str,
                        service_type: str,
                        criticality: str = "important",
                        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                        fallback_handler: Optional[Callable] = None) -> CircuitBreaker:
        """
        Register a service with resilience management.
        
        Args:
            name: Service name (e.g., 'clickhouse', 'redis', 'firestore')
            service_type: Type of service ('database', 'cache', 'stream', 'external')
            criticality: Service criticality ('critical', 'important', 'optional')
            circuit_breaker_config: Circuit breaker configuration
            fallback_handler: Fallback function when service fails
            
        Returns:
            Circuit breaker instance for the service
        """
        with self.lock:
            # Register service dependency
            service_dep = ServiceDependency(
                name=name,
                service_type=service_type,
                criticality=criticality,
                fallback_available=fallback_handler is not None
            )
            self.services[name] = service_dep
            
            # Create circuit breaker with service-specific config
            if circuit_breaker_config is None:
                circuit_breaker_config = self._get_default_config(service_type, criticality)
            
            circuit_breaker = CircuitBreaker(
                name=f"{name}_circuit_breaker",
                config=circuit_breaker_config,
                fallback_function=fallback_handler
            )
            
            self.circuit_breakers[name] = circuit_breaker
            register_circuit_breaker(circuit_breaker)
            
            # Register fallback handler
            if fallback_handler:
                self.fallback_handlers[name] = fallback_handler
            
            self.metrics.total_services += 1
            
            logger.info(f"Registered service '{name}' with {criticality} criticality")
            return circuit_breaker
    
    def _get_default_config(self, service_type: str, criticality: str) -> CircuitBreakerConfig:
        """Get default circuit breaker configuration based on service characteristics"""
        
        base_config = CircuitBreakerConfig()
        
        # Adjust based on service type
        if service_type == "database":
            # Databases are slower but more reliable
            base_config.timeout = 30.0
            base_config.failure_threshold = 3
            base_config.recovery_timeout = 120.0
        elif service_type == "cache":
            # Caches should fail fast
            base_config.timeout = 5.0
            base_config.failure_threshold = 5
            base_config.recovery_timeout = 30.0
        elif service_type == "stream":
            # Streaming services need more tolerance
            base_config.timeout = 10.0
            base_config.failure_threshold = 8
            base_config.recovery_timeout = 60.0
        elif service_type == "external":
            # External services are unpredictable
            base_config.timeout = 15.0
            base_config.failure_threshold = 4
            base_config.recovery_timeout = 90.0
        
        # Adjust based on criticality
        if criticality == "critical":
            # Critical services get more tolerance
            base_config.failure_threshold += 2
            base_config.recovery_timeout *= 1.5
        elif criticality == "optional":
            # Optional services fail faster
            base_config.failure_threshold = max(2, base_config.failure_threshold - 2)
            base_config.recovery_timeout *= 0.5
        
        return base_config
    
    def get_circuit_breaker(self, service_name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for a service"""
        return self.circuit_breakers.get(service_name)
    
    async def start_health_monitoring(self) -> None:
        """Start continuous health monitoring"""
        if self.health_check_task and not self.health_check_task.done():
            logger.warning("Health monitoring already running")
            return
        
        self.health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started health monitoring")
    
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
            self.health_check_task = None
        
        logger.info("Stopped health monitoring")
    
    async def _health_monitor_loop(self) -> None:
        """Continuous health monitoring loop"""
        while True:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retrying
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check.
        
        Returns:
            Health status for all services and overall system
        """
        with self.lock:
            self.last_health_check = datetime.utcnow()
            
            # Reset metrics
            self.metrics.healthy_services = 0
            self.metrics.degraded_services = 0
            self.metrics.failed_services = 0
            self.metrics.fallbacks_active = 0
            
            service_health = {}
            
            # Check each registered service
            for service_name, service_dep in self.services.items():
                health_info = await self._check_service_health(service_name, service_dep)
                service_health[service_name] = health_info
                
                # Update metrics based on service state
                if health_info['status'] == 'healthy':
                    self.metrics.healthy_services += 1
                elif health_info['status'] == 'degraded':
                    self.metrics.degraded_services += 1
                    if health_info.get('fallback_active', False):
                        self.metrics.fallbacks_active += 1
                else:
                    self.metrics.failed_services += 1
            
            # Determine overall system health
            old_health = self.system_health
            self.system_health = self._calculate_system_health()
            
            # Log health state changes
            if old_health != self.system_health:
                logger.warning(f"System health changed: {old_health.value} -> {self.system_health.value}")
                await self._handle_health_state_change(old_health, self.system_health)
            
            # Update metrics
            self.metrics.last_health_check = self.last_health_check
            self.metrics.uptime_percentage = self.metrics.calculate_health_percentage()
            
            # Store metrics history
            self._store_metrics_snapshot()
            
            health_report = {
                'system_health': self.system_health.value,
                'timestamp': self.last_health_check.isoformat(),
                'services': service_health,
                'metrics': {
                    'total_services': self.metrics.total_services,
                    'healthy_services': self.metrics.healthy_services,
                    'degraded_services': self.metrics.degraded_services,
                    'failed_services': self.metrics.failed_services,
                    'fallbacks_active': self.metrics.fallbacks_active,
                    'health_percentage': self.metrics.uptime_percentage
                },
                'circuit_breakers': self._get_circuit_breaker_summary()
            }
            
            return health_report
    
    async def _check_service_health(self, 
                                   service_name: str, 
                                   service_dep: ServiceDependency) -> Dict[str, Any]:
        """Check health of individual service"""
        health_info = {
            'service_name': service_name,
            'service_type': service_dep.service_type,
            'criticality': service_dep.criticality,
            'status': 'unknown',
            'circuit_breaker_state': 'unknown',
            'fallback_available': service_dep.fallback_available,
            'fallback_active': False,
            'last_error': None,
            'response_time_ms': None
        }
        
        # Check circuit breaker state
        circuit_breaker = self.circuit_breakers.get(service_name)
        if circuit_breaker:
            cb_state = circuit_breaker.get_state()
            health_info['circuit_breaker_state'] = cb_state.value
            
            if cb_state == CircuitBreakerState.CLOSED:
                health_info['status'] = 'healthy'
            elif cb_state == CircuitBreakerState.HALF_OPEN:
                health_info['status'] = 'degraded'
            else:  # OPEN
                health_info['status'] = 'failed'
                health_info['fallback_active'] = service_dep.fallback_available
                
                # Get last error from circuit breaker metrics
                cb_metrics = circuit_breaker.get_metrics()
                if cb_metrics.get('metrics', {}).get('last_failure_time'):
                    health_info['last_error'] = 'Circuit breaker open due to failures'
        
        # Perform actual health check if circuit is not open
        if health_info['status'] != 'failed':
            try:
                health_result = await self._perform_service_health_check(service_name)
                health_info.update(health_result)
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
                health_info['status'] = 'failed'
                health_info['last_error'] = str(e)
        
        return health_info
    
    async def _perform_service_health_check(self, service_name: str) -> Dict[str, Any]:
        """Perform actual health check for service"""
        start_time = datetime.utcnow()
        
        try:
            if service_name == 'clickhouse':
                from .clickhouse_service import clickhouse_service
                is_healthy = clickhouse_service.check_connection()
                
            elif service_name == 'redis':
                from .redis_cache import redis_cache
                is_healthy = redis_cache.check_connection()
                
            elif service_name == 'firestore':
                from .firestore_listener import firestore_listener
                health_check = await firestore_listener.health_check()
                is_healthy = health_check.get('overall_status') in ['healthy', 'partial']
                
            else:
                # Generic health check
                is_healthy = True
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                'status': 'healthy' if is_healthy else 'failed',
                'response_time_ms': response_time,
                'checked_at': start_time.isoformat()
            }
            
        except Exception as e:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return {
                'status': 'failed',
                'response_time_ms': response_time,
                'last_error': str(e),
                'checked_at': start_time.isoformat()
            }
    
    def _calculate_system_health(self) -> SystemHealthState:
        """Calculate overall system health state"""
        if self.metrics.total_services == 0:
            return SystemHealthState.HEALTHY
        
        health_percentage = self.metrics.calculate_health_percentage()
        
        # Check critical services specifically
        critical_services_down = 0
        for service_name, service_dep in self.services.items():
            if service_dep.criticality == 'critical':
                circuit_breaker = self.circuit_breakers.get(service_name)
                if circuit_breaker and circuit_breaker.get_state() == CircuitBreakerState.OPEN:
                    critical_services_down += 1
        
        # Determine health state
        if critical_services_down > 0:
            return SystemHealthState.CRITICAL
        elif health_percentage < self.emergency_mode_threshold * 100:
            return SystemHealthState.EMERGENCY
        elif health_percentage < self.degraded_mode_threshold * 100:
            return SystemHealthState.DEGRADED
        else:
            return SystemHealthState.HEALTHY
    
    async def _handle_health_state_change(self, 
                                         old_state: SystemHealthState, 
                                         new_state: SystemHealthState) -> None:
        """Handle system health state transitions"""
        logger.warning(f"System health transition: {old_state.value} -> {new_state.value}")
        
        if new_state == SystemHealthState.CRITICAL:
            await self._enter_critical_mode()
        elif new_state == SystemHealthState.EMERGENCY:
            await self._enter_emergency_mode()
        elif new_state == SystemHealthState.DEGRADED:
            await self._enter_degraded_mode()
        elif new_state == SystemHealthState.HEALTHY and old_state != SystemHealthState.HEALTHY:
            await self._exit_degraded_mode()
    
    async def _enter_critical_mode(self) -> None:
        """Enter critical mode - attempt immediate recovery"""
        logger.critical("Entering CRITICAL mode - attempting immediate recovery")
        
        # Attempt recovery of all failed critical services
        for service_name, service_dep in self.services.items():
            if service_dep.criticality == 'critical':
                circuit_breaker = self.circuit_breakers.get(service_name)
                if circuit_breaker and circuit_breaker.get_state() == CircuitBreakerState.OPEN:
                    await self._attempt_service_recovery(service_name)
    
    async def _enter_emergency_mode(self) -> None:
        """Enter emergency mode - activate all fallbacks"""
        logger.critical("Entering EMERGENCY mode - activating all fallbacks")
        
        # Activate fallbacks for all services
        for service_name in self.services.keys():
            if service_name in self.fallback_handlers:
                logger.warning(f"Activating emergency fallback for {service_name}")
    
    async def _enter_degraded_mode(self) -> None:
        """Enter degraded mode - selective fallbacks"""
        logger.warning("Entering DEGRADED mode - activating selective fallbacks")
        
        # Activate fallbacks for failed non-critical services
        for service_name, service_dep in self.services.items():
            circuit_breaker = self.circuit_breakers.get(service_name)
            if (circuit_breaker and 
                circuit_breaker.get_state() == CircuitBreakerState.OPEN and
                service_dep.criticality != 'critical' and
                service_name in self.fallback_handlers):
                logger.info(f"Activating fallback for {service_name}")
    
    async def _exit_degraded_mode(self) -> None:
        """Exit degraded mode - return to normal operations"""
        logger.info("Exiting degraded mode - returning to normal operations")
        
        # This would deactivate fallbacks and return to normal service calls
        # Implementation depends on how fallbacks are structured
    
    async def _attempt_service_recovery(self, service_name: str) -> bool:
        """
        Attempt to recover a failed service.
        
        Args:
            service_name: Name of service to recover
            
        Returns:
            True if recovery successful
        """
        if service_name in self.recovery_in_progress:
            logger.debug(f"Recovery already in progress for {service_name}")
            return False
        
        self.recovery_in_progress.add(service_name)
        self.metrics.recovery_attempts += 1
        
        try:
            logger.info(f"Attempting recovery for service: {service_name}")
            
            # Get circuit breaker
            circuit_breaker = self.circuit_breakers.get(service_name)
            if not circuit_breaker:
                logger.error(f"No circuit breaker found for {service_name}")
                return False
            
            # Force circuit to half-open for testing
            circuit_breaker.force_closed()
            
            # Perform health check
            health_result = await self._perform_service_health_check(service_name)
            
            if health_result['status'] == 'healthy':
                logger.info(f"Service {service_name} recovery successful")
                self.metrics.successful_recoveries += 1
                return True
            else:
                logger.warning(f"Service {service_name} recovery failed: {health_result.get('last_error')}")
                circuit_breaker.force_open()
                return False
                
        except Exception as e:
            logger.error(f"Error during recovery attempt for {service_name}: {e}")
            return False
        finally:
            self.recovery_in_progress.discard(service_name)
    
    def _get_circuit_breaker_summary(self) -> Dict[str, Any]:
        """Get summary of all circuit breaker states"""
        summary = {
            'total': len(self.circuit_breakers),
            'by_state': defaultdict(int),
            'details': {}
        }
        
        for service_name, circuit_breaker in self.circuit_breakers.items():
            state = circuit_breaker.get_state()
            summary['by_state'][state.value] += 1
            summary['details'][service_name] = circuit_breaker.get_metrics()
        
        return summary
    
    def _store_metrics_snapshot(self) -> None:
        """Store current metrics for historical tracking"""
        snapshot = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_health': self.system_health.value,
            'healthy_services': self.metrics.healthy_services,
            'degraded_services': self.metrics.degraded_services,
            'failed_services': self.metrics.failed_services,
            'fallbacks_active': self.metrics.fallbacks_active,
            'health_percentage': self.metrics.calculate_health_percentage()
        }
        
        self.metrics_history.append(snapshot)
        
        # Keep only last 100 snapshots
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_health': self.system_health.value,
            'last_health_check': self.last_health_check.isoformat(),
            'metrics': {
                'total_services': self.metrics.total_services,
                'healthy_services': self.metrics.healthy_services,
                'degraded_services': self.metrics.degraded_services,
                'failed_services': self.metrics.failed_services,
                'fallbacks_active': self.metrics.fallbacks_active,
                'recovery_attempts': self.metrics.recovery_attempts,
                'successful_recoveries': self.metrics.successful_recoveries,
                'health_percentage': self.metrics.calculate_health_percentage()
            },
            'services': {
                name: {
                    'type': dep.service_type,
                    'criticality': dep.criticality,
                    'fallback_available': dep.fallback_available,
                    'circuit_breaker_state': self.circuit_breakers[name].get_state().value if name in self.circuit_breakers else 'none'
                }
                for name, dep in self.services.items()
            },
            'circuit_breakers': self._get_circuit_breaker_summary(),
            'recent_metrics': self.metrics_history[-10:] if self.metrics_history else []
        }
    
    async def emergency_shutdown(self) -> None:
        """Emergency shutdown of all services"""
        logger.critical("Initiating emergency shutdown")
        
        # Stop health monitoring
        await self.stop_health_monitoring()
        
        # Open all circuit breakers
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.force_open()
        
        # Cancel all recovery tasks
        for task in self.recovery_tasks.values():
            if not task.done():
                task.cancel()
        
        self.system_health = SystemHealthState.EMERGENCY
        logger.critical("Emergency shutdown complete")
    
    async def force_recovery_all(self) -> Dict[str, bool]:
        """Force recovery attempt for all failed services"""
        logger.info("Forcing recovery for all failed services")
        
        recovery_results = {}
        for service_name in self.services.keys():
            circuit_breaker = self.circuit_breakers.get(service_name)
            if circuit_breaker and circuit_breaker.get_state() == CircuitBreakerState.OPEN:
                recovery_results[service_name] = await self._attempt_service_recovery(service_name)
        
        successful_recoveries = sum(1 for success in recovery_results.values() if success)
        logger.info(f"Recovery attempt complete: {successful_recoveries}/{len(recovery_results)} services recovered")
        
        return recovery_results
    
    def get_degraded_mode_config(self) -> Dict[str, Any]:
        """Get configuration for degraded mode operations"""
        return {
            'cache_only_mode': self.system_health in [SystemHealthState.CRITICAL, SystemHealthState.EMERGENCY],
            'reduced_functionality': self.system_health == SystemHealthState.DEGRADED,
            'fallback_responses': self.metrics.fallbacks_active > 0,
            'read_only_mode': self.system_health == SystemHealthState.EMERGENCY,
            'active_fallbacks': list(self.fallback_handlers.keys()) if self.system_health != SystemHealthState.HEALTHY else []
        }
    
    async def shutdown(self) -> None:
        """Graceful shutdown of resilience manager"""
        logger.info("Shutting down resilience manager")
        
        # Stop health monitoring
        await self.stop_health_monitoring()
        
        # Cancel all recovery tasks
        for task in self.recovery_tasks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        logger.info("Resilience manager shutdown complete")


# Global resilience manager instance
resilience_manager = ResilienceManager()


def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager instance"""
    return resilience_manager


async def initialize_resilience_system() -> None:
    """Initialize the complete resilience system"""
    logger.info("Initializing resilience system...")
    
    # This will be called during application startup to register all services
    # with their circuit breakers and fallback handlers
    
    try:
        # Register core services with appropriate configurations
        await _register_core_services()
        
        # Start health monitoring
        await resilience_manager.start_health_monitoring()
        
        logger.info("Resilience system initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize resilience system: {e}")
        raise


async def _register_core_services() -> None:
    """Register core services with resilience manager"""
    
    # ClickHouse - Critical database service
    clickhouse_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=120.0,
        timeout=30.0,
        success_threshold=2
    )
    
    # Redis - Important cache service  
    redis_config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=30.0,
        timeout=5.0,
        success_threshold=3
    )
    
    # Firestore - Important streaming service
    firestore_config = CircuitBreakerConfig(
        failure_threshold=8,
        recovery_timeout=60.0,
        timeout=10.0,
        success_threshold=3
    )
    
    # Register services (fallback handlers will be added later)
    resilience_manager.register_service(
        'clickhouse', 
        'database', 
        'critical', 
        clickhouse_config
    )
    
    resilience_manager.register_service(
        'redis', 
        'cache', 
        'important', 
        redis_config
    )
    
    resilience_manager.register_service(
        'firestore', 
        'stream', 
        'important', 
        firestore_config
    )
    
    logger.info("Core services registered with resilience manager")