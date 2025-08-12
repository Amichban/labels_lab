"""
System manager for the incremental computation engine ecosystem.

Integrates and orchestrates:
- Incremental computation engine (Issue #12)
- Firestore listener (Issue #11) 
- Performance optimizer with monitoring and alerting
- Comprehensive health checking and alerting

This is the main entry point for managing the entire real-time label computation system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import signal
import sys

from src.services.incremental_engine import incremental_engine, ComputationResult
from src.services.performance_optimizer import performance_optimizer, PerformanceAlert, AlertSeverity
from src.services.firestore_listener import firestore_listener

logger = logging.getLogger(__name__)


@dataclass
class SystemHealth:
    """Overall system health status"""
    healthy: bool
    status: str  # healthy, degraded, critical
    components: Dict[str, Dict[str, Any]]
    overall_score: float  # 0-100
    last_check: datetime
    uptime_seconds: float
    alerts_summary: Dict[str, int]


class IncrementalSystemManager:
    """
    Orchestrates the complete incremental computation system.
    
    Features:
    - Unified lifecycle management
    - Cross-component health monitoring
    - Integrated alerting and notifications
    - Graceful shutdown handling
    - Performance dashboard integration
    - System-wide configuration management
    """
    
    def __init__(self):
        # System state
        self.is_running = False
        self.started_at: Optional[datetime] = None
        self.shutdown_requested = False
        
        # Component references
        self.engine = incremental_engine
        self.optimizer = performance_optimizer
        self.firestore_listener = firestore_listener
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        self.last_health_check: Optional[datetime] = None
        self.health_history: List[SystemHealth] = []
        
        # Alerting
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self.critical_alert_count = 0
        self.total_alert_count = 0
        
        # Performance tracking
        self.startup_time_ms: Optional[float] = None
        self.total_computations = 0
        self.total_alerts_generated = 0
        
        logger.info("IncrementalSystemManager initialized")
    
    async def start_system(self,
                          engine_workers: int = 8,
                          enable_performance_monitoring: bool = True,
                          enable_firestore_integration: bool = True) -> None:
        """
        Start the complete incremental computation system.
        
        Args:
            engine_workers: Number of computation workers
            enable_performance_monitoring: Enable performance monitoring
            enable_firestore_integration: Enable Firestore listener integration
        """
        if self.is_running:
            logger.warning("System already running")
            return
        
        startup_start = datetime.utcnow()
        logger.info("Starting incremental computation system...")
        
        try:
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # 1. Start performance monitoring first
            if enable_performance_monitoring:
                logger.info("Starting performance monitoring...")
                await self.optimizer.start_monitoring(self.engine)
                
                # Set up alert callbacks
                self.optimizer.add_alert_callback(self._handle_performance_alert)
                
                # Set performance baselines based on targets
                self._set_performance_baselines()
            
            # 2. Start incremental computation engine
            logger.info("Starting incremental computation engine...")
            await self.engine.start()
            
            # Add result callback for monitoring
            self.engine.add_result_callback(self._handle_computation_result)
            
            # 3. Start Firestore integration if enabled
            if enable_firestore_integration:
                logger.info("Starting Firestore listener integration...")
                # The engine already sets up Firestore subscriptions in its start method
                
            # 4. Start health monitoring
            await self._start_health_monitoring()
            
            # System is now running
            self.is_running = True
            self.started_at = datetime.utcnow()
            self.startup_time_ms = (self.started_at - startup_start).total_seconds() * 1000
            
            logger.info(f"Incremental computation system started successfully in {self.startup_time_ms:.1f}ms")
            
            # Log initial system status
            await self._log_system_status()
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}", exc_info=True)
            await self._emergency_shutdown()
            raise
    
    async def stop_system(self, timeout_seconds: int = 30) -> None:
        """
        Gracefully stop the complete system.
        
        Args:
            timeout_seconds: Maximum time to wait for graceful shutdown
        """
        if not self.is_running:
            logger.warning("System not running")
            return
        
        logger.info("Stopping incremental computation system...")
        self.shutdown_requested = True
        
        try:
            # Stop health monitoring first
            await self._stop_health_monitoring()
            
            # Stop performance monitoring
            if self.optimizer.is_monitoring:
                logger.info("Stopping performance monitoring...")
                await self.optimizer.stop_monitoring()
            
            # Stop incremental computation engine
            if self.engine.is_running:
                logger.info("Stopping incremental computation engine...")
                await self.engine.stop()
            
            # Firestore listener is stopped by the engine
            
            self.is_running = False
            uptime = (datetime.utcnow() - self.started_at).total_seconds() if self.started_at else 0
            
            logger.info(f"Incremental computation system stopped. Uptime: {uptime:.1f} seconds")
            
            # Log final statistics
            await self._log_final_statistics()
            
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}", exc_info=True)
    
    async def _start_health_monitoring(self) -> None:
        """Start system-wide health monitoring"""
        self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("Health monitoring started")
    
    async def _stop_health_monitoring(self) -> None:
        """Stop health monitoring"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _health_monitoring_loop(self) -> None:
        """Main health monitoring loop"""
        while self.is_running and not self.shutdown_requested:
            try:
                # Perform comprehensive health check
                health_status = await self._perform_health_check()
                
                # Store in history
                self.health_history.append(health_status)
                
                # Keep only last 100 health checks
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]
                
                # Log health status if degraded
                if not health_status.healthy:
                    logger.warning(f"System health degraded: {health_status.status}")
                    
                    # Log component issues
                    for component_name, component_health in health_status.components.items():
                        if not component_health.get("healthy", True):
                            issues = component_health.get("issues", [])
                            logger.warning(f"{component_name} issues: {', '.join(issues)}")
                
                # Critical health check - generate alert if score too low
                if health_status.overall_score < 50:
                    await self._generate_critical_health_alert(health_status)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}", exc_info=True)
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_check(self) -> SystemHealth:
        """Perform comprehensive system health check"""
        check_time = datetime.utcnow()
        self.last_health_check = check_time
        
        components = {}
        overall_healthy = True
        health_scores = []
        
        # Check incremental engine
        try:
            engine_health = self.engine.get_health_status()
            components["incremental_engine"] = engine_health
            
            if not engine_health.get("healthy", False):
                overall_healthy = False
            
            # Extract numeric score for overall calculation
            key_metrics = engine_health.get("key_metrics", {})
            engine_score = self._calculate_component_score("engine", key_metrics)
            health_scores.append(("engine", engine_score))
            
        except Exception as e:
            logger.error(f"Engine health check failed: {e}")
            components["incremental_engine"] = {"healthy": False, "error": str(e)}
            overall_healthy = False
            health_scores.append(("engine", 0))
        
        # Check performance optimizer
        try:
            optimizer_summary = self.optimizer.get_performance_summary()
            optimizer_healthy = optimizer_summary.get("health_score", 0) > 70
            
            components["performance_optimizer"] = {
                "healthy": optimizer_healthy,
                "health_score": optimizer_summary.get("health_score", 0),
                "active_alerts": optimizer_summary.get("active_alerts_count", 0),
                "critical_alerts": optimizer_summary.get("critical_alerts_count", 0),
                "bottlenecks": optimizer_summary.get("bottlenecks", [])
            }
            
            if not optimizer_healthy:
                overall_healthy = False
            
            health_scores.append(("optimizer", optimizer_summary.get("health_score", 0)))
            
        except Exception as e:
            logger.error(f"Optimizer health check failed: {e}")
            components["performance_optimizer"] = {"healthy": False, "error": str(e)}
            overall_healthy = False
            health_scores.append(("optimizer", 0))
        
        # Check Firestore listener
        try:
            firestore_health = await self.firestore_listener.health_check()
            firestore_healthy = firestore_health.get("overall_status") in ["healthy", "partial"]
            
            components["firestore_listener"] = {
                "healthy": firestore_healthy,
                "status": firestore_health.get("overall_status"),
                "active_streams": firestore_health.get("active_streams", 0),
                "total_streams": firestore_health.get("total_streams", 0),
                "total_errors": firestore_health.get("total_errors", 0)
            }
            
            if not firestore_healthy:
                overall_healthy = False
            
            # Calculate Firestore score based on stream health
            active_streams = firestore_health.get("active_streams", 0)
            total_streams = firestore_health.get("total_streams", 1)
            firestore_score = (active_streams / total_streams) * 100
            health_scores.append(("firestore", firestore_score))
            
        except Exception as e:
            logger.error(f"Firestore health check failed: {e}")
            components["firestore_listener"] = {"healthy": False, "error": str(e)}
            overall_healthy = False
            health_scores.append(("firestore", 0))
        
        # Calculate overall health score
        if health_scores:
            total_score = sum(score for _, score in health_scores)
            overall_score = total_score / len(health_scores)
        else:
            overall_score = 0
        
        # Determine status
        if overall_score >= 90 and overall_healthy:
            status = "healthy"
        elif overall_score >= 70:
            status = "degraded"  
        else:
            status = "critical"
        
        # Get alerts summary
        active_alerts = self.optimizer.get_active_alerts()
        alerts_summary = {
            "total": len(active_alerts),
            "critical": len([a for a in active_alerts if a["severity"] == "critical"]),
            "warning": len([a for a in active_alerts if a["severity"] == "warning"]),
            "info": len([a for a in active_alerts if a["severity"] == "info"])
        }
        
        uptime = (check_time - self.started_at).total_seconds() if self.started_at else 0
        
        return SystemHealth(
            healthy=overall_healthy and overall_score >= 70,
            status=status,
            components=components,
            overall_score=overall_score,
            last_check=check_time,
            uptime_seconds=uptime,
            alerts_summary=alerts_summary
        )
    
    def _calculate_component_score(self, component_type: str, metrics: Dict[str, Any]) -> float:
        """Calculate health score for a component based on its metrics"""
        if component_type == "engine":
            # Engine score based on key performance indicators
            p99_latency = metrics.get("p99_latency_ms", 0)
            success_rate = metrics.get("success_rate_pct", 100)
            cache_hit_rate = metrics.get("cache_hit_rate_pct", 100)
            throughput = metrics.get("throughput_rps", 0)
            
            # Score components (0-100 each)
            latency_score = max(0, 100 - p99_latency)  # Perfect score at 0ms, 0 points at 100ms+
            success_score = success_rate  # Use percentage directly
            cache_score = cache_hit_rate  # Use percentage directly  
            throughput_score = min(100, throughput / 10)  # 100 points at 1000+ rps
            
            # Weighted average
            return (latency_score * 0.4 + success_score * 0.3 + cache_score * 0.2 + throughput_score * 0.1)
        
        return 50  # Default neutral score
    
    async def _generate_critical_health_alert(self, health_status: SystemHealth) -> None:
        """Generate critical system health alert"""
        try:
            # Create a synthetic alert for critical system health
            alert_id = f"system_health_critical_{int(health_status.last_check.timestamp())}"
            
            # Find the most problematic components
            problem_components = [
                name for name, component in health_status.components.items()
                if not component.get("healthy", True)
            ]
            
            description = (f"System health score is critically low ({health_status.overall_score:.1f}/100). "
                         f"Problematic components: {', '.join(problem_components) if problem_components else 'Unknown'}")
            
            recommendations = [
                "Check individual component health status",
                "Review recent performance alerts",
                "Consider scaling infrastructure resources",
                "Review system logs for error patterns"
            ]
            
            # This would integrate with your alerting system
            logger.critical(f"CRITICAL SYSTEM HEALTH: {description}")
            
        except Exception as e:
            logger.error(f"Failed to generate critical health alert: {e}")
    
    async def _handle_performance_alert(self, alert: PerformanceAlert) -> None:
        """Handle performance alerts from optimizer"""
        self.total_alerts_generated += 1
        
        if alert.severity == AlertSeverity.CRITICAL:
            self.critical_alert_count += 1
            
        # Log the alert
        log_level = logging.CRITICAL if alert.severity == AlertSeverity.CRITICAL else logging.WARNING
        logger.log(log_level, f"Performance Alert [{alert.severity.value.upper()}]: {alert.title}")
        
        # Forward to registered callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _handle_computation_result(self, result: ComputationResult) -> None:
        """Handle computation results for monitoring"""
        self.total_computations += 1
        
        # This could be extended to track computation patterns,
        # detect anomalies, etc.
    
    def _set_performance_baselines(self) -> None:
        """Set performance baselines based on target requirements"""
        # Based on performance-analyzer targets and Issue #12 requirements
        baselines = {
            "p99_latency_ms": 100.0,      # Target: <100ms P99
            "average_latency_ms": 50.0,    # Target: <50ms average
            "throughput_rps": 1000.0,      # Target: >1000 computations/second
            "cache_hit_rate_pct": 95.0,    # Target: >95% cache hit rate
            "success_rate_pct": 99.9,      # Target: >99.9% success rate
            "memory_usage_pct": 80.0,      # Alert at 80% memory usage
            "cpu_usage_pct": 70.0          # Alert at 70% CPU usage
        }
        
        for metric_name, baseline_value in baselines.items():
            self.optimizer.set_performance_baseline(metric_name, baseline_value, acceptable_deviation_pct=15.0)
        
        logger.info(f"Set {len(baselines)} performance baselines")
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, initiating graceful shutdown...")
            self.shutdown_requested = True
            
            # Create shutdown task
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.stop_system())
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _emergency_shutdown(self) -> None:
        """Emergency shutdown in case of startup failure"""
        logger.critical("Performing emergency shutdown due to startup failure")
        
        try:
            if self.optimizer.is_monitoring:
                await self.optimizer.stop_monitoring()
        except Exception as e:
            logger.error(f"Error stopping optimizer during emergency shutdown: {e}")
        
        try:
            if self.engine.is_running:
                await self.engine.stop()
        except Exception as e:
            logger.error(f"Error stopping engine during emergency shutdown: {e}")
    
    async def _log_system_status(self) -> None:
        """Log current system status"""
        try:
            health_status = await self._perform_health_check()
            
            logger.info(f"System Status: {health_status.status.upper()} (Score: {health_status.overall_score:.1f}/100)")
            
            for component_name, component_health in health_status.components.items():
                status = "HEALTHY" if component_health.get("healthy", False) else "UNHEALTHY"
                logger.info(f"  {component_name}: {status}")
                
        except Exception as e:
            logger.error(f"Failed to log system status: {e}")
    
    async def _log_final_statistics(self) -> None:
        """Log final system statistics before shutdown"""
        if not self.started_at:
            return
        
        uptime = (datetime.utcnow() - self.started_at).total_seconds()
        
        logger.info("=== FINAL SYSTEM STATISTICS ===")
        logger.info(f"Uptime: {uptime:.1f} seconds ({uptime/3600:.1f} hours)")
        logger.info(f"Startup time: {self.startup_time_ms:.1f}ms")
        logger.info(f"Total computations: {self.total_computations}")
        logger.info(f"Total alerts generated: {self.total_alerts_generated}")
        logger.info(f"Critical alerts: {self.critical_alert_count}")
        
        if uptime > 0:
            logger.info(f"Average computations per second: {self.total_computations / uptime:.1f}")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add system-wide alert callback"""
        self.alert_callbacks.append(callback)
        logger.debug(f"Added system alert callback, total: {len(self.alert_callbacks)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_health = self.health_history[-1] if self.health_history else None
        
        return {
            "is_running": self.is_running,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime_seconds": (datetime.utcnow() - self.started_at).total_seconds() if self.started_at else 0,
            "startup_time_ms": self.startup_time_ms,
            "current_health": {
                "healthy": current_health.healthy if current_health else False,
                "status": current_health.status if current_health else "unknown",
                "overall_score": current_health.overall_score if current_health else 0,
                "last_check": current_health.last_check.isoformat() if current_health else None
            } if current_health else None,
            "statistics": {
                "total_computations": self.total_computations,
                "total_alerts_generated": self.total_alerts_generated,
                "critical_alerts": self.critical_alert_count
            },
            "components": {
                "incremental_engine": {
                    "running": self.engine.is_running,
                    "workers": len(self.engine.workers) if hasattr(self.engine, 'workers') else 0
                },
                "performance_optimizer": {
                    "monitoring": self.optimizer.is_monitoring,
                    "active_alerts": len(self.optimizer.get_active_alerts())
                },
                "firestore_listener": {
                    "health_status": self.firestore_listener.health_status
                }
            }
        }
    
    async def run_until_stopped(self) -> None:
        """Run the system until stop is requested"""
        if not self.is_running:
            raise RuntimeError("System not started")
        
        logger.info("System running. Press Ctrl+C to stop.")
        
        try:
            while self.is_running and not self.shutdown_requested:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            if self.is_running:
                await self.stop_system()


# Global system manager instance
system_manager = IncrementalSystemManager()


# Convenience function for starting the complete system
async def start_incremental_computation_system(
    engine_workers: int = 8,
    enable_performance_monitoring: bool = True,
    enable_firestore_integration: bool = True
) -> IncrementalSystemManager:
    """
    Start the complete incremental computation system.
    
    This is the main entry point for applications wanting to use
    the real-time label computation system.
    
    Args:
        engine_workers: Number of computation engine workers
        enable_performance_monitoring: Enable performance monitoring and optimization
        enable_firestore_integration: Enable Firestore listener integration
        
    Returns:
        System manager instance for further control
    """
    await system_manager.start_system(
        engine_workers=engine_workers,
        enable_performance_monitoring=enable_performance_monitoring,
        enable_firestore_integration=enable_firestore_integration
    )
    
    return system_manager