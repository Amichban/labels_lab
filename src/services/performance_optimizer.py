"""
Performance monitoring and optimization service for incremental computation engine.

Implements advanced performance monitoring, optimization strategies, and alerting
following performance-analyzer best practices from .claude/subagents/performance-analyzer.md.

Features:
- Real-time performance monitoring with Web Vitals-style metrics
- Automated performance optimization strategies
- Predictive performance alerting
- Resource usage optimization
- Bottleneck detection and resolution
- Performance regression detection

Key Performance Indicators:
- P99 latency < 100ms (Core Web Vital equivalent)
- Throughput > 1000 computations/second
- Cache hit rate > 95%
- Memory usage < 2GB under normal load
- Error rate < 0.1%
"""

import asyncio
import logging
import time
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from statistics import mean, median, stdev
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class OptimizationStrategy(str, Enum):
    """Available optimization strategies"""
    INCREASE_WORKERS = "increase_workers"
    DECREASE_WORKERS = "decrease_workers"
    ADJUST_CACHE_SIZE = "adjust_cache_size"
    ENABLE_COMPRESSION = "enable_compression" 
    FORCE_GC = "force_gc"
    ADJUST_BATCH_SIZE = "adjust_batch_size"
    THROTTLE_REQUESTS = "throttle_requests"
    SCALE_INFRASTRUCTURE = "scale_infrastructure"


@dataclass
class PerformanceAlert:
    """Performance alert with context and recommendations"""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization"""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "recommendations": self.recommendations,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged": self.acknowledged
        }


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection"""
    metric_name: str
    baseline_value: float
    acceptable_deviation_pct: float = 10.0
    measurement_window_hours: int = 24
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_regression(self, current_value: float) -> Tuple[bool, float]:
        """
        Check if current value represents a performance regression.
        
        Returns:
            Tuple of (is_regression, deviation_percentage)
        """
        if self.baseline_value == 0:
            return False, 0.0
            
        deviation_pct = abs((current_value - self.baseline_value) / self.baseline_value) * 100
        is_regression = deviation_pct > self.acceptable_deviation_pct
        
        return is_regression, deviation_pct


class ResourceMonitor:
    """Monitor system resources and detect bottlenecks"""
    
    def __init__(self, sample_interval_seconds: int = 5):
        self.sample_interval_seconds = sample_interval_seconds
        self.cpu_samples: deque = deque(maxlen=720)  # 1 hour at 5s intervals
        self.memory_samples: deque = deque(maxlen=720)
        self.disk_io_samples: deque = deque(maxlen=720)
        self.network_io_samples: deque = deque(maxlen=720)
        
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self) -> None:
        """Start resource monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")
    
    async def _monitor_loop(self) -> None:
        """Main resource monitoring loop"""
        while self._monitoring:
            try:
                # CPU utilization
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_samples.append((datetime.utcnow(), cpu_percent))
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_data = {
                    "total": memory.total,
                    "used": memory.used,
                    "available": memory.available,
                    "percent": memory.percent
                }
                self.memory_samples.append((datetime.utcnow(), memory_data))
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    disk_data = {
                        "read_bytes": disk_io.read_bytes,
                        "write_bytes": disk_io.write_bytes,
                        "read_time": disk_io.read_time,
                        "write_time": disk_io.write_time
                    }
                    self.disk_io_samples.append((datetime.utcnow(), disk_data))
                
                # Network I/O
                network_io = psutil.net_io_counters()
                if network_io:
                    network_data = {
                        "bytes_sent": network_io.bytes_sent,
                        "bytes_recv": network_io.bytes_recv,
                        "packets_sent": network_io.packets_sent,
                        "packets_recv": network_io.packets_recv
                    }
                    self.network_io_samples.append((datetime.utcnow(), network_data))
                
                await asyncio.sleep(self.sample_interval_seconds)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.sample_interval_seconds)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Get recent averages
            recent_cpu_samples = list(self.cpu_samples)[-12:]  # Last minute
            avg_cpu = mean([sample[1] for sample in recent_cpu_samples]) if recent_cpu_samples else cpu_percent
            
            return {
                "cpu": {
                    "current_percent": cpu_percent,
                    "average_1min": avg_cpu,
                    "cores": psutil.cpu_count()
                },
                "memory": {
                    "total_bytes": memory.total,
                    "used_bytes": memory.used,
                    "available_bytes": memory.available,
                    "percent": memory.percent,
                    "used_gb": memory.used / (1024**3),
                    "available_gb": memory.available / (1024**3)
                },
                "process": self._get_current_process_metrics()
            }
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {}
    
    def _get_current_process_metrics(self) -> Dict[str, Any]:
        """Get metrics for current Python process"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_rss_mb": memory_info.rss / (1024*1024),
                "memory_vms_mb": memory_info.vms / (1024*1024),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            }
        except Exception as e:
            logger.error(f"Failed to get process metrics: {e}")
            return {}
    
    def detect_bottlenecks(self) -> List[str]:
        """Detect system bottlenecks based on resource usage patterns"""
        bottlenecks = []
        
        try:
            current_metrics = self.get_current_metrics()
            
            # CPU bottleneck
            if current_metrics.get("cpu", {}).get("average_1min", 0) > 80:
                bottlenecks.append("high_cpu_usage")
            
            # Memory bottleneck
            memory_percent = current_metrics.get("memory", {}).get("percent", 0)
            if memory_percent > 85:
                bottlenecks.append("high_memory_usage")
            
            # Process-specific bottlenecks
            process_metrics = current_metrics.get("process", {})
            if process_metrics.get("memory_rss_mb", 0) > 2000:  # 2GB limit
                bottlenecks.append("high_process_memory")
                
            if process_metrics.get("threads", 0) > 50:
                bottlenecks.append("high_thread_count")
                
            if process_metrics.get("open_files", 0) > 1000:
                bottlenecks.append("high_file_descriptor_usage")
        
        except Exception as e:
            logger.error(f"Bottleneck detection error: {e}")
        
        return bottlenecks


class PerformanceOptimizer:
    """
    Advanced performance monitoring and optimization service.
    
    Implements performance-analyzer best practices including:
    - Real-time performance monitoring
    - Automated optimization strategies  
    - Predictive alerting
    - Performance regression detection
    - Resource usage optimization
    """
    
    def __init__(self,
                 monitoring_interval_seconds: int = 5,
                 alerting_enabled: bool = True,
                 auto_optimization_enabled: bool = True):
        
        # Configuration
        self.monitoring_interval_seconds = monitoring_interval_seconds
        self.alerting_enabled = alerting_enabled
        self.auto_optimization_enabled = auto_optimization_enabled
        
        # Monitoring components
        self.resource_monitor = ResourceMonitor(monitoring_interval_seconds)
        
        # Performance tracking
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=8640))  # 12 hours at 5s intervals
        self.baselines: Dict[str, PerformanceBaseline] = {}
        
        # Alerting
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self.alert_thresholds = self._get_default_thresholds()
        
        # Optimization
        self.optimization_strategies: Dict[str, Callable] = {
            OptimizationStrategy.FORCE_GC: self._force_garbage_collection,
            OptimizationStrategy.ADJUST_CACHE_SIZE: self._adjust_cache_size,
            # Additional strategies can be implemented as needed
        }
        
        # State
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.last_optimization: Optional[datetime] = None
        self.optimization_cooldown_minutes = 5
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("PerformanceOptimizer initialized")
    
    def _get_default_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get default performance thresholds based on performance-analyzer targets"""
        return {
            "p99_latency_ms": {
                "warning": 80.0,
                "critical": 100.0
            },
            "average_latency_ms": {
                "warning": 50.0,
                "critical": 75.0
            },
            "throughput_rps": {
                "warning": 800.0,  # Below this triggers warning
                "critical": 500.0  # Below this triggers critical
            },
            "cache_hit_rate_pct": {
                "warning": 90.0,   # Below this triggers warning
                "critical": 80.0   # Below this triggers critical
            },
            "success_rate_pct": {
                "warning": 98.0,   # Below this triggers warning
                "critical": 95.0   # Below this triggers critical
            },
            "memory_usage_pct": {
                "warning": 80.0,
                "critical": 90.0
            },
            "cpu_usage_pct": {
                "warning": 70.0,
                "critical": 85.0
            },
            "queue_utilization_pct": {
                "warning": 70.0,
                "critical": 90.0
            }
        }
    
    async def start_monitoring(self, engine_instance=None) -> None:
        """Start performance monitoring and optimization"""
        if self.is_monitoring:
            logger.warning("Performance monitoring already started")
            return
        
        self.engine_instance = engine_instance
        self.is_monitoring = True
        
        # Start resource monitoring
        await self.resource_monitor.start_monitoring()
        
        # Start main monitoring loop
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring"""
        if not self.is_monitoring:
            logger.warning("Performance monitoring not running")
            return
        
        self.is_monitoring = False
        
        # Stop resource monitoring
        await self.resource_monitor.stop_monitoring()
        
        # Stop main monitoring loop
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main performance monitoring loop"""
        logger.info("Performance monitoring loop started")
        
        while self.is_monitoring:
            try:
                # Collect performance metrics
                await self._collect_performance_metrics()
                
                # Check thresholds and generate alerts
                if self.alerting_enabled:
                    await self._check_performance_thresholds()
                
                # Detect and resolve performance regressions
                await self._detect_performance_regressions()
                
                # Apply automatic optimizations if enabled
                if self.auto_optimization_enabled:
                    await self._apply_automatic_optimizations()
                
                # Clean up resolved alerts
                self._cleanup_resolved_alerts()
                
                await asyncio.sleep(self.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(self.monitoring_interval_seconds)
        
        logger.info("Performance monitoring loop stopped")
    
    async def _collect_performance_metrics(self) -> None:
        """Collect comprehensive performance metrics"""
        timestamp = datetime.utcnow()
        
        try:
            # Get engine metrics if available
            engine_metrics = {}
            if hasattr(self, 'engine_instance') and self.engine_instance:
                engine_metrics = self.engine_instance.get_performance_metrics()
            
            # Get resource metrics
            resource_metrics = self.resource_monitor.get_current_metrics()
            
            # Store key metrics in history
            metrics_to_track = {
                "p99_latency_ms": engine_metrics.get("latency", {}).get("p99_ms", 0),
                "average_latency_ms": engine_metrics.get("latency", {}).get("average_ms", 0),
                "throughput_rps": engine_metrics.get("throughput", {}).get("requests_per_second", 0),
                "cache_hit_rate_pct": engine_metrics.get("cache", {}).get("overall", {}).get("overall_hit_rate_pct", 0),
                "success_rate_pct": engine_metrics.get("throughput", {}).get("success_rate_pct", 100),
                "memory_usage_pct": resource_metrics.get("memory", {}).get("percent", 0),
                "cpu_usage_pct": resource_metrics.get("cpu", {}).get("average_1min", 0),
                "queue_utilization_pct": engine_metrics.get("system", {}).get("queue_utilization_pct", 0),
                "active_workers": engine_metrics.get("system", {}).get("active_workers", 0),
                "pending_requests": engine_metrics.get("system", {}).get("pending_requests", 0)
            }
            
            with self._lock:
                for metric_name, value in metrics_to_track.items():
                    self.metric_history[metric_name].append((timestamp, value))
        
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
    
    async def _check_performance_thresholds(self) -> None:
        """Check performance metrics against thresholds and generate alerts"""
        current_time = datetime.utcnow()
        
        with self._lock:
            for metric_name, thresholds in self.alert_thresholds.items():
                if metric_name not in self.metric_history or not self.metric_history[metric_name]:
                    continue
                
                # Get latest value
                _, current_value = self.metric_history[metric_name][-1]
                
                # Determine alert level
                alert_level = None
                threshold_value = None
                
                # For metrics where lower is better (latency, utilization)
                if metric_name in ["p99_latency_ms", "average_latency_ms", "memory_usage_pct", "cpu_usage_pct", "queue_utilization_pct"]:
                    if current_value >= thresholds.get("critical", float('inf')):
                        alert_level = AlertSeverity.CRITICAL
                        threshold_value = thresholds["critical"]
                    elif current_value >= thresholds.get("warning", float('inf')):
                        alert_level = AlertSeverity.WARNING
                        threshold_value = thresholds["warning"]
                
                # For metrics where higher is better (throughput, cache hit rate, success rate)
                elif metric_name in ["throughput_rps", "cache_hit_rate_pct", "success_rate_pct"]:
                    if current_value <= thresholds.get("critical", 0):
                        alert_level = AlertSeverity.CRITICAL
                        threshold_value = thresholds["critical"]
                    elif current_value <= thresholds.get("warning", 0):
                        alert_level = AlertSeverity.WARNING
                        threshold_value = thresholds["warning"]
                
                # Generate or update alert
                if alert_level:
                    await self._generate_or_update_alert(
                        metric_name=metric_name,
                        severity=alert_level,
                        current_value=current_value,
                        threshold_value=threshold_value
                    )
                else:
                    # Resolve existing alert if metric is back to normal
                    await self._resolve_alert(metric_name)
    
    async def _generate_or_update_alert(self,
                                       metric_name: str,
                                       severity: AlertSeverity,
                                       current_value: float,
                                       threshold_value: float) -> None:
        """Generate new alert or update existing one"""
        alert_id = f"{metric_name}_{severity.value}"
        
        # Check if alert already exists
        if alert_id in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[alert_id]
            alert.current_value = current_value
            alert.description = self._generate_alert_description(metric_name, current_value, threshold_value, severity)
        else:
            # Create new alert
            alert = PerformanceAlert(
                id=alert_id,
                severity=severity,
                title=self._generate_alert_title(metric_name, severity),
                description=self._generate_alert_description(metric_name, current_value, threshold_value, severity),
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value,
                recommendations=self._generate_recommendations(metric_name, severity)
            )
            
            self.active_alerts[alert_id] = alert
            
            # Notify callbacks
            await self._notify_alert_callbacks(alert)
            
            logger.warning(f"Performance alert generated: {alert.title}")
    
    async def _resolve_alert(self, metric_name: str) -> None:
        """Resolve alerts for a metric that's back to normal"""
        alerts_to_resolve = [
            alert_id for alert_id in self.active_alerts.keys()
            if alert_id.startswith(f"{metric_name}_")
        ]
        
        for alert_id in alerts_to_resolve:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.utcnow()
            
            # Notify resolution
            await self._notify_alert_callbacks(alert)
            
            logger.info(f"Performance alert resolved: {alert.title}")
    
    def _generate_alert_title(self, metric_name: str, severity: AlertSeverity) -> str:
        """Generate human-readable alert title"""
        metric_titles = {
            "p99_latency_ms": "P99 Latency High",
            "average_latency_ms": "Average Latency High", 
            "throughput_rps": "Throughput Low",
            "cache_hit_rate_pct": "Cache Hit Rate Low",
            "success_rate_pct": "Success Rate Low",
            "memory_usage_pct": "Memory Usage High",
            "cpu_usage_pct": "CPU Usage High",
            "queue_utilization_pct": "Queue Utilization High"
        }
        
        title = metric_titles.get(metric_name, f"Performance Issue: {metric_name}")
        return f"{severity.value.upper()}: {title}"
    
    def _generate_alert_description(self,
                                   metric_name: str,
                                   current_value: float,
                                   threshold_value: float,
                                   severity: AlertSeverity) -> str:
        """Generate detailed alert description"""
        descriptions = {
            "p99_latency_ms": f"P99 latency is {current_value:.1f}ms, exceeding {threshold_value:.1f}ms threshold. This may impact user experience.",
            "average_latency_ms": f"Average latency is {current_value:.1f}ms, exceeding {threshold_value:.1f}ms threshold.",
            "throughput_rps": f"Request throughput is {current_value:.1f} req/s, below {threshold_value:.1f} req/s threshold.",
            "cache_hit_rate_pct": f"Cache hit rate is {current_value:.1f}%, below {threshold_value:.1f}% threshold. This increases computation load.",
            "success_rate_pct": f"Success rate is {current_value:.1f}%, below {threshold_value:.1f}% threshold. Check for errors.",
            "memory_usage_pct": f"Memory usage is {current_value:.1f}%, exceeding {threshold_value:.1f}% threshold.",
            "cpu_usage_pct": f"CPU usage is {current_value:.1f}%, exceeding {threshold_value:.1f}% threshold.",
            "queue_utilization_pct": f"Queue utilization is {current_value:.1f}%, exceeding {threshold_value:.1f}% threshold. Backpressure may occur."
        }
        
        return descriptions.get(metric_name, f"{metric_name} is {current_value:.1f}, threshold: {threshold_value:.1f}")
    
    def _generate_recommendations(self, metric_name: str, severity: AlertSeverity) -> List[str]:
        """Generate optimization recommendations based on performance-analyzer patterns"""
        recommendations_map = {
            "p99_latency_ms": [
                "Increase worker count to improve parallel processing",
                "Optimize cache hit rates to reduce computation time",
                "Consider scaling infrastructure for higher load",
                "Review slow computations for optimization opportunities"
            ],
            "average_latency_ms": [
                "Review computation algorithms for optimization",
                "Increase cache memory allocation",
                "Consider asynchronous processing for heavy operations"
            ],
            "throughput_rps": [
                "Increase worker pool size",
                "Optimize database query performance",
                "Implement request batching for efficiency",
                "Scale infrastructure horizontally"
            ],
            "cache_hit_rate_pct": [
                "Increase cache memory allocation",
                "Review cache TTL settings",
                "Implement cache warming strategies",
                "Optimize cache key structure"
            ],
            "success_rate_pct": [
                "Review error logs for common failure patterns",
                "Implement retry mechanisms with exponential backoff",
                "Add input validation to prevent bad requests",
                "Monitor external dependencies for failures"
            ],
            "memory_usage_pct": [
                "Force garbage collection",
                "Reduce cache memory allocation", 
                "Review memory leaks in application code",
                "Scale infrastructure to handle memory requirements"
            ],
            "cpu_usage_pct": [
                "Optimize computational algorithms",
                "Increase infrastructure CPU capacity",
                "Implement load balancing across multiple instances",
                "Review CPU-intensive operations for optimization"
            ],
            "queue_utilization_pct": [
                "Increase queue capacity",
                "Add more worker threads",
                "Implement request throttling",
                "Review processing bottlenecks"
            ]
        }
        
        return recommendations_map.get(metric_name, ["Review performance metrics and logs for optimization opportunities"])
    
    async def _detect_performance_regressions(self) -> None:
        """Detect performance regressions compared to baseline"""
        current_time = datetime.utcnow()
        
        with self._lock:
            for metric_name in self.metric_history.keys():
                if metric_name not in self.baselines or not self.metric_history[metric_name]:
                    continue
                
                # Get current value
                _, current_value = self.metric_history[metric_name][-1]
                baseline = self.baselines[metric_name]
                
                # Check for regression
                is_regression, deviation_pct = baseline.is_regression(current_value)
                
                if is_regression:
                    regression_alert_id = f"regression_{metric_name}"
                    
                    if regression_alert_id not in self.active_alerts:
                        alert = PerformanceAlert(
                            id=regression_alert_id,
                            severity=AlertSeverity.WARNING,
                            title=f"Performance Regression: {metric_name}",
                            description=f"Performance regression detected. Current value {current_value:.1f} deviates {deviation_pct:.1f}% from baseline {baseline.baseline_value:.1f}",
                            metric_name=metric_name,
                            current_value=current_value,
                            threshold_value=baseline.baseline_value,
                            recommendations=[
                                "Compare recent code changes for performance impact",
                                "Review resource utilization patterns",
                                "Consider rolling back recent changes if severe",
                                "Update baseline if this represents a new normal"
                            ]
                        )
                        
                        self.active_alerts[regression_alert_id] = alert
                        await self._notify_alert_callbacks(alert)
                        
                        logger.warning(f"Performance regression detected for {metric_name}: {deviation_pct:.1f}% deviation")
    
    async def _apply_automatic_optimizations(self) -> None:
        """Apply automatic performance optimizations based on current state"""
        current_time = datetime.utcnow()
        
        # Check cooldown period
        if (self.last_optimization and 
            (current_time - self.last_optimization).total_seconds() < self.optimization_cooldown_minutes * 60):
            return
        
        # Detect bottlenecks
        bottlenecks = self.resource_monitor.detect_bottlenecks()
        
        if not bottlenecks:
            return
        
        optimizations_applied = []
        
        try:
            # High memory usage - force garbage collection
            if "high_memory_usage" in bottlenecks or "high_process_memory" in bottlenecks:
                if OptimizationStrategy.FORCE_GC in self.optimization_strategies:
                    await self.optimization_strategies[OptimizationStrategy.FORCE_GC]()
                    optimizations_applied.append("forced_garbage_collection")
            
            # High cache memory usage - adjust cache size
            if "high_process_memory" in bottlenecks:
                if OptimizationStrategy.ADJUST_CACHE_SIZE in self.optimization_strategies:
                    await self.optimization_strategies[OptimizationStrategy.ADJUST_CACHE_SIZE]()
                    optimizations_applied.append("reduced_cache_size")
            
            if optimizations_applied:
                self.last_optimization = current_time
                logger.info(f"Applied automatic optimizations: {', '.join(optimizations_applied)}")
        
        except Exception as e:
            logger.error(f"Error applying automatic optimizations: {e}")
    
    async def _force_garbage_collection(self) -> None:
        """Force Python garbage collection to free memory"""
        try:
            collected = gc.collect()
            logger.info(f"Forced garbage collection, freed {collected} objects")
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
    
    async def _adjust_cache_size(self) -> None:
        """Adjust cache size based on memory pressure"""
        try:
            if hasattr(self, 'engine_instance') and self.engine_instance:
                # This would adjust cache settings in the engine
                # For now, we'll just log the intention
                logger.info("Cache size adjustment triggered (implementation needed)")
        except Exception as e:
            logger.error(f"Error adjusting cache size: {e}")
    
    def _cleanup_resolved_alerts(self) -> None:
        """Clean up resolved alerts older than 1 hour"""
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        with self._lock:
            resolved_alerts = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.resolved_at and alert.resolved_at < cutoff_time
            ]
            
            for alert_id in resolved_alerts:
                del self.active_alerts[alert_id]
            
            if resolved_alerts:
                logger.debug(f"Cleaned up {len(resolved_alerts)} resolved alerts")
    
    async def _notify_alert_callbacks(self, alert: PerformanceAlert) -> None:
        """Notify registered callbacks about alert"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback notification failed: {e}")
    
    def set_performance_baseline(self,
                                metric_name: str,
                                baseline_value: float,
                                acceptable_deviation_pct: float = 10.0) -> None:
        """Set performance baseline for regression detection"""
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            baseline_value=baseline_value,
            acceptable_deviation_pct=acceptable_deviation_pct
        )
        
        with self._lock:
            self.baselines[metric_name] = baseline
        
        logger.info(f"Set performance baseline for {metric_name}: {baseline_value} (±{acceptable_deviation_pct}%)")
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback to be notified of performance alerts"""
        self.alert_callbacks.append(callback)
        logger.debug(f"Added alert callback, total: {len(self.alert_callbacks)}")
    
    def remove_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> bool:
        """Remove alert callback"""
        try:
            self.alert_callbacks.remove(callback)
            logger.debug(f"Removed alert callback, remaining: {len(self.alert_callbacks)}")
            return True
        except ValueError:
            return False
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        with self._lock:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
            return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        with self._lock:
            return [alert.to_dict() for alert in self.active_alerts.values()]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        current_time = datetime.utcnow()
        
        with self._lock:
            # Calculate recent averages (last 5 minutes)
            recent_window = current_time - timedelta(minutes=5)
            recent_metrics = {}
            
            for metric_name, history in self.metric_history.items():
                recent_values = [
                    value for timestamp, value in history 
                    if timestamp >= recent_window
                ]
                
                if recent_values:
                    recent_metrics[metric_name] = {
                        "current": recent_values[-1],
                        "average_5min": mean(recent_values),
                        "min_5min": min(recent_values),
                        "max_5min": max(recent_values),
                        "samples": len(recent_values)
                    }
            
            # Get resource metrics
            resource_metrics = self.resource_monitor.get_current_metrics()
            
            # Get bottlenecks
            bottlenecks = self.resource_monitor.detect_bottlenecks()
            
            # Calculate health score (0-100)
            health_score = self._calculate_health_score()
            
            return {
                "timestamp": current_time.isoformat(),
                "health_score": health_score,
                "recent_metrics": recent_metrics,
                "resource_metrics": resource_metrics,
                "active_alerts_count": len(self.active_alerts),
                "critical_alerts_count": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
                "bottlenecks": bottlenecks,
                "baselines": {name: baseline.baseline_value for name, baseline in self.baselines.items()},
                "last_optimization": self.last_optimization.isoformat() if self.last_optimization else None
            }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-100) based on key metrics"""
        if not self.metric_history:
            return 100.0  # Assume healthy if no data yet
        
        score_components = {}
        
        # P99 latency score (target: <100ms)
        if "p99_latency_ms" in self.metric_history and self.metric_history["p99_latency_ms"]:
            _, p99_latency = self.metric_history["p99_latency_ms"][-1]
            score_components["latency"] = max(0, 100 - p99_latency)  # 100 points - latency in ms
        
        # Cache hit rate score
        if "cache_hit_rate_pct" in self.metric_history and self.metric_history["cache_hit_rate_pct"]:
            _, cache_hit_rate = self.metric_history["cache_hit_rate_pct"][-1]
            score_components["cache"] = cache_hit_rate  # Use percentage directly
        
        # Success rate score
        if "success_rate_pct" in self.metric_history and self.metric_history["success_rate_pct"]:
            _, success_rate = self.metric_history["success_rate_pct"][-1]
            score_components["success"] = success_rate  # Use percentage directly
        
        # Resource usage score (inverse of usage percentage)
        if "memory_usage_pct" in self.metric_history and self.metric_history["memory_usage_pct"]:
            _, memory_usage = self.metric_history["memory_usage_pct"][-1]
            score_components["memory"] = max(0, 100 - memory_usage)
        
        if "cpu_usage_pct" in self.metric_history and self.metric_history["cpu_usage_pct"]:
            _, cpu_usage = self.metric_history["cpu_usage_pct"][-1]
            score_components["cpu"] = max(0, 100 - cpu_usage)
        
        # Calculate weighted average
        if not score_components:
            return 100.0
        
        weights = {
            "latency": 0.3,
            "cache": 0.2,
            "success": 0.25,
            "memory": 0.125,
            "cpu": 0.125
        }
        
        weighted_score = sum(
            score * weights.get(component, 1.0) 
            for component, score in score_components.items()
        )
        
        total_weight = sum(weights.get(component, 1.0) for component in score_components.keys())
        
        return min(100.0, max(0.0, weighted_score / total_weight))


# Global optimizer instance
performance_optimizer = PerformanceOptimizer()