"""
Resilience system initialization module

Provides initialization and setup functions for the complete resilience system,
including circuit breakers, fallback handlers, health monitoring, and recovery mechanisms.

Issue #14: Circuit breakers and failover mechanisms
Following infra-pr best practices for production resilience
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .circuit_breaker import CircuitBreakerConfig
from .resilience_manager import get_resilience_manager, initialize_resilience_system
from .fallback_handlers import (
    get_fallback_orchestrator, 
    register_fallback_functions
)

logger = logging.getLogger(__name__)


class ResilienceInitializer:
    """
    Centralizes initialization of all resilience components.
    
    Responsibilities:
    - Initialize circuit breakers for all services
    - Set up fallback handlers and orchestration
    - Start health monitoring and recovery systems
    - Configure retry policies and timeouts
    """
    
    def __init__(self):
        """Initialize resilience initializer"""
        self.resilience_manager = get_resilience_manager()
        self.fallback_orchestrator = get_fallback_orchestrator()
        self.initialized = False
        self.initialization_time: Optional[datetime] = None
        
        logger.info("ResilienceInitializer created")
    
    async def initialize_complete_system(self) -> Dict[str, Any]:
        """
        Initialize the complete resilience system.
        
        Returns:
            Initialization status and metrics
        """
        if self.initialized:
            logger.warning("Resilience system already initialized")
            return await self.get_initialization_status()
        
        logger.info("Initializing complete resilience system...")
        
        initialization_results = {
            'started_at': datetime.utcnow(),
            'services_registered': 0,
            'circuit_breakers_created': 0,
            'fallback_handlers_registered': 0,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Register core services with circuit breakers
            service_results = await self._register_core_services()
            initialization_results.update(service_results)
            
            # Step 2: Set up fallback handlers
            fallback_results = await self._setup_fallback_handlers()
            initialization_results.update(fallback_results)
            
            # Step 3: Start health monitoring
            health_results = await self._start_health_monitoring()
            initialization_results.update(health_results)
            
            # Step 4: Start background maintenance tasks
            background_results = await self._start_background_tasks()
            initialization_results.update(background_results)
            
            # Mark as initialized
            self.initialized = True
            self.initialization_time = datetime.utcnow()
            
            initialization_results['completed_at'] = self.initialization_time
            initialization_results['duration_ms'] = (
                initialization_results['completed_at'] - initialization_results['started_at']
            ).total_seconds() * 1000
            
            logger.info(
                f"Resilience system initialization complete in "
                f"{initialization_results['duration_ms']:.0f}ms"
            )
            
            return initialization_results
            
        except Exception as e:
            logger.error(f"Failed to initialize resilience system: {e}")
            initialization_results['errors'].append(str(e))
            initialization_results['failed_at'] = datetime.utcnow()
            raise
    
    async def _register_core_services(self) -> Dict[str, Any]:
        """Register core services with circuit breakers"""
        logger.info("Registering core services with circuit breakers...")
        
        results = {
            'services_registered': 0,
            'circuit_breakers_created': 0,
            'service_registration_errors': []
        }
        
        # Define service configurations
        service_configs = [
            {
                'name': 'clickhouse',
                'type': 'database',
                'criticality': 'critical',
                'config': CircuitBreakerConfig(
                    failure_threshold=3,
                    recovery_timeout=120.0,
                    timeout=30.0,
                    success_threshold=2,
                    expected_exception=(Exception,)
                )
            },
            {
                'name': 'redis',
                'type': 'cache',
                'criticality': 'important',
                'config': CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=30.0,
                    timeout=5.0,
                    success_threshold=3,
                    expected_exception=(Exception,)
                )
            },
            {
                'name': 'firestore',
                'type': 'stream',
                'criticality': 'important',
                'config': CircuitBreakerConfig(
                    failure_threshold=8,
                    recovery_timeout=60.0,
                    timeout=10.0,
                    success_threshold=3,
                    expected_exception=(Exception,)
                )
            }
        ]
        
        # Register each service
        for service_config in service_configs:
            try:
                circuit_breaker = self.resilience_manager.register_service(
                    name=service_config['name'],
                    service_type=service_config['type'],
                    criticality=service_config['criticality'],
                    circuit_breaker_config=service_config['config']
                )
                
                if circuit_breaker:
                    results['services_registered'] += 1
                    results['circuit_breakers_created'] += 1
                    logger.info(f"Registered service: {service_config['name']}")
                else:
                    logger.warning(f"Failed to register service: {service_config['name']}")
                    
            except Exception as e:
                error_msg = f"Error registering {service_config['name']}: {e}"
                logger.error(error_msg)
                results['service_registration_errors'].append(error_msg)
        
        logger.info(f"Service registration complete: {results['services_registered']} services registered")
        return results
    
    async def _setup_fallback_handlers(self) -> Dict[str, Any]:
        """Set up fallback handlers for all services"""
        logger.info("Setting up fallback handlers...")
        
        results = {
            'fallback_handlers_registered': 0,
            'fallback_setup_errors': []
        }
        
        try:
            # Register fallback functions
            fallback_functions = register_fallback_functions()
            results['fallback_handlers_registered'] = len(fallback_functions)
            
            # Start background tasks for fallback orchestrator
            await self.fallback_orchestrator.start_background_tasks()
            
            logger.info(f"Fallback handlers setup complete: {results['fallback_handlers_registered']} handlers")
            
        except Exception as e:
            error_msg = f"Error setting up fallback handlers: {e}"
            logger.error(error_msg)
            results['fallback_setup_errors'].append(error_msg)
        
        return results
    
    async def _start_health_monitoring(self) -> Dict[str, Any]:
        """Start health monitoring and recovery systems"""
        logger.info("Starting health monitoring...")
        
        results = {
            'health_monitoring_started': False,
            'health_monitoring_errors': []
        }
        
        try:
            # Start resilience manager health monitoring
            await self.resilience_manager.start_health_monitoring()
            results['health_monitoring_started'] = True
            
            logger.info("Health monitoring started successfully")
            
        except Exception as e:
            error_msg = f"Error starting health monitoring: {e}"
            logger.error(error_msg)
            results['health_monitoring_errors'].append(error_msg)
        
        return results
    
    async def _start_background_tasks(self) -> Dict[str, Any]:
        """Start background maintenance tasks"""
        logger.info("Starting background maintenance tasks...")
        
        results = {
            'background_tasks_started': 0,
            'background_task_errors': []
        }
        
        try:
            # Start periodic health checks
            asyncio.create_task(self._periodic_system_health_check())
            results['background_tasks_started'] += 1
            
            # Start circuit breaker metrics collection
            asyncio.create_task(self._collect_circuit_breaker_metrics())
            results['background_tasks_started'] += 1
            
            logger.info(f"Background tasks started: {results['background_tasks_started']} tasks")
            
        except Exception as e:
            error_msg = f"Error starting background tasks: {e}"
            logger.error(error_msg)
            results['background_task_errors'].append(error_msg)
        
        return results
    
    async def _periodic_system_health_check(self) -> None:
        """Periodic system health check task"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                
                # Perform system health check
                health_status = await self.resilience_manager.get_system_status()
                
                # Log health summary
                logger.info(
                    f"System health check: {health_status['system_health']} "
                    f"({health_status['metrics']['health_percentage']:.1f}% healthy services)"
                )
                
                # Check for critical issues
                if health_status['system_health'] in ['critical', 'emergency']:
                    logger.critical(
                        f"System in {health_status['system_health']} state - "
                        f"Failed services: {health_status['metrics']['failed_services']}"
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic health check: {e}")
                await asyncio.sleep(60)  # Shorter sleep on error
    
    async def _collect_circuit_breaker_metrics(self) -> None:
        """Collect and log circuit breaker metrics"""
        while True:
            try:
                await asyncio.sleep(600)  # 10 minutes
                
                # Get circuit breaker summary
                from .circuit_breaker import get_circuit_breaker_summary
                cb_summary = get_circuit_breaker_summary()
                
                if cb_summary['total_breakers'] > 0:
                    logger.info(
                        f"Circuit breaker status: {cb_summary['total_breakers']} breakers, "
                        f"States: {cb_summary['states']}"
                    )
                    
                    # Log any open breakers
                    for name, metrics in cb_summary['breakers'].items():
                        if metrics['state'] == 'open':
                            logger.warning(
                                f"Circuit breaker '{name}' is OPEN - "
                                f"failures: {metrics['failure_count']}, "
                                f"failure_rate: {metrics['failure_rate']:.2f}"
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting circuit breaker metrics: {e}")
                await asyncio.sleep(300)  # Shorter sleep on error
    
    async def get_initialization_status(self) -> Dict[str, Any]:
        """Get current initialization status"""
        return {
            'initialized': self.initialized,
            'initialization_time': self.initialization_time.isoformat() if self.initialization_time else None,
            'resilience_manager_status': await self.resilience_manager.get_system_status(),
            'fallback_orchestrator_status': self.fallback_orchestrator.get_system_status(),
            'uptime_seconds': (
                (datetime.utcnow() - self.initialization_time).total_seconds()
                if self.initialization_time else 0
            )
        }
    
    async def shutdown(self) -> None:
        """Graceful shutdown of resilience system"""
        logger.info("Shutting down resilience system...")
        
        try:
            # Stop health monitoring
            await self.resilience_manager.stop_health_monitoring()
            
            # Stop fallback orchestrator background tasks
            await self.fallback_orchestrator.stop_background_tasks()
            
            # Shutdown resilience manager
            await self.resilience_manager.shutdown()
            
            self.initialized = False
            logger.info("Resilience system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during resilience system shutdown: {e}")
    
    async def force_recovery_all_services(self) -> Dict[str, Any]:
        """Force recovery attempt for all services"""
        logger.info("Forcing recovery for all services...")
        
        recovery_results = await self.resilience_manager.force_recovery_all()
        
        logger.info(
            f"Forced recovery complete: {sum(recovery_results.values())} of "
            f"{len(recovery_results)} services recovered"
        )
        
        return {
            'recovery_results': recovery_results,
            'services_recovered': sum(recovery_results.values()),
            'total_services': len(recovery_results),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_degraded_mode_config(self) -> Dict[str, Any]:
        """Get configuration for degraded mode operations"""
        return self.resilience_manager.get_degraded_mode_config()


# Global resilience initializer instance
resilience_initializer = ResilienceInitializer()


async def initialize_resilience() -> Dict[str, Any]:
    """Initialize the complete resilience system"""
    return await resilience_initializer.initialize_complete_system()


async def shutdown_resilience() -> None:
    """Shutdown the resilience system"""
    await resilience_initializer.shutdown()


def get_resilience_status() -> Dict[str, Any]:
    """Get current resilience system status (sync version)"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If called from async context, return basic status
            return {
                'initialized': resilience_initializer.initialized,
                'initialization_time': (
                    resilience_initializer.initialization_time.isoformat() 
                    if resilience_initializer.initialization_time else None
                ),
                'note': 'Full status requires async call to get_resilience_status_async()'
            }
        else:
            return loop.run_until_complete(resilience_initializer.get_initialization_status())
    except Exception as e:
        logger.error(f"Error getting resilience status: {e}")
        return {
            'error': str(e),
            'initialized': False
        }


async def get_resilience_status_async() -> Dict[str, Any]:
    """Get current resilience system status (async version)"""
    return await resilience_initializer.get_initialization_status()


async def force_service_recovery() -> Dict[str, Any]:
    """Force recovery for all failed services"""
    return await resilience_initializer.force_recovery_all_services()


def get_degraded_mode_config() -> Dict[str, Any]:
    """Get current degraded mode configuration"""
    return resilience_initializer.get_degraded_mode_config()


# Health check endpoint function
async def health_check_endpoint() -> Dict[str, Any]:
    """
    Health check endpoint for external monitoring systems.
    
    Returns comprehensive health status suitable for load balancers,
    monitoring systems, and operational dashboards.
    """
    try:
        status = await get_resilience_status_async()
        
        # Determine overall health
        if not status.get('initialized', False):
            health_status = 'unhealthy'
            http_status = 503  # Service Unavailable
        else:
            system_health = status.get('resilience_manager_status', {}).get('system_health', 'unknown')
            if system_health == 'healthy':
                health_status = 'healthy'
                http_status = 200
            elif system_health == 'degraded':
                health_status = 'degraded'
                http_status = 200  # Still serving traffic
            else:
                health_status = 'unhealthy'
                http_status = 503
        
        return {
            'status': health_status,
            'http_status': http_status,
            'timestamp': datetime.utcnow().isoformat(),
            'details': status,
            'service': 'label-computation-system',
            'version': '1.0.0'
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'http_status': 503,
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e),
            'service': 'label-computation-system',
            'version': '1.0.0'
        }