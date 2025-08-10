"""
Batch Processing Metrics and Monitoring Service

Comprehensive monitoring system for batch backfill operations:
- Real-time performance metrics collection
- Throughput and latency tracking
- Error rate monitoring and alerting
- ETA calculations with confidence intervals
- System resource utilization tracking
- Business metrics and KPIs
- Prometheus-compatible metrics export
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor

from src.services.redis_cache import redis_cache

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric measurement"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class TimeSeriesMetric:
    """Time series metric with windowing"""
    name: str
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    window_size: int = 300  # 5 minutes default
    
    def add_point(self, value: float, labels: Dict[str, str] = None):
        """Add a new metric point"""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels or {}
        )
        self.values.append(point)
    
    def get_recent_values(self, seconds: int = None) -> List[float]:
        """Get values from the last N seconds"""
        if not seconds:
            seconds = self.window_size
        
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)
        return [p.value for p in self.values if p.timestamp >= cutoff]
    
    def get_avg(self, seconds: int = None) -> float:
        """Get average value over time window"""
        values = self.get_recent_values(seconds)
        return statistics.mean(values) if values else 0.0
    
    def get_percentile(self, percentile: float, seconds: int = None) -> float:
        """Get percentile value over time window"""
        values = self.get_recent_values(seconds)
        if not values:
            return 0.0
        
        return statistics.quantiles(sorted(values), n=100)[int(percentile) - 1]
    
    def get_rate(self, seconds: int = None) -> float:
        """Get rate (values per second) over time window"""
        values = self.get_recent_values(seconds)
        return len(values) / (seconds or self.window_size) if values else 0.0


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics"""
    timestamp: datetime
    job_id: str
    
    # Throughput metrics
    candles_per_second: float
    chunks_per_second: float
    
    # Latency metrics
    avg_chunk_time_ms: float
    p50_chunk_time_ms: float
    p95_chunk_time_ms: float
    p99_chunk_time_ms: float
    
    # Error metrics
    error_rate: float
    retry_rate: float
    
    # Cache metrics
    cache_hit_rate: float
    
    # Resource metrics
    cpu_usage_percent: float
    memory_usage_mb: float
    
    # Progress metrics
    progress_percent: float
    eta_minutes: Optional[float] = None
    eta_confidence: Optional[float] = None  # 0-1 confidence score


class BatchMetricsCollector:
    """
    Collects and aggregates metrics from batch processing operations.
    
    Features:
    - Real-time metric collection and aggregation
    - Time-windowed statistics calculation
    - Automatic ETA calculation with confidence scoring
    - Resource utilization monitoring
    - Prometheus metric export
    - Configurable alerting thresholds
    """
    
    def __init__(self, redis_key_prefix: str = "batch_metrics"):
        self.redis_key_prefix = redis_key_prefix
        self.metrics: Dict[str, TimeSeriesMetric] = {}
        self.job_metrics: Dict[str, Dict[str, TimeSeriesMetric]] = defaultdict(
            lambda: defaultdict(lambda: TimeSeriesMetric(""))
        )
        
        # Performance targets and thresholds
        self.target_throughput = 1_000_000  # candles per minute
        self.max_error_rate = 0.05  # 5%
        self.max_latency_ms = 1000  # 1 second per chunk
        
        # Metric collection thread
        self._collector_thread = None
        self._stop_collection = threading.Event()
        
        # Initialize core metrics
        self._init_metrics()
        
        logger.info("BatchMetricsCollector initialized")
    
    def _init_metrics(self):
        """Initialize core metric collectors"""
        core_metrics = [
            'throughput_candles_per_minute',
            'throughput_chunks_per_second',
            'latency_chunk_processing_ms',
            'error_rate_percent',
            'retry_rate_percent',
            'cache_hit_rate_percent',
            'active_jobs_count',
            'queue_depth',
            'cpu_usage_percent',
            'memory_usage_mb',
            'redis_memory_mb',
            'clickhouse_connections'
        ]
        
        for metric_name in core_metrics:
            self.metrics[metric_name] = TimeSeriesMetric(metric_name)
    
    def start_collection(self, interval_seconds: int = 10):
        """Start background metric collection"""
        if self._collector_thread and self._collector_thread.is_alive():
            logger.warning("Metric collection already running")
            return
        
        self._stop_collection.clear()
        self._collector_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._collector_thread.start()
        
        logger.info(f"Started metric collection (interval: {interval_seconds}s)")
    
    def stop_collection(self):
        """Stop background metric collection"""
        self._stop_collection.set()
        if self._collector_thread:
            self._collector_thread.join(timeout=5)
        logger.info("Stopped metric collection")
    
    def record_job_metric(self,
                         job_id: str,
                         metric_name: str,
                         value: float,
                         labels: Dict[str, str] = None):
        """Record a metric for a specific job"""
        self.job_metrics[job_id][metric_name].add_point(value, labels)
        
        # Also record to global metrics
        global_metric_name = f"job_{metric_name}"
        if global_metric_name not in self.metrics:
            self.metrics[global_metric_name] = TimeSeriesMetric(global_metric_name)
        
        job_labels = {'job_id': job_id}
        if labels:
            job_labels.update(labels)
        
        self.metrics[global_metric_name].add_point(value, job_labels)
    
    def record_chunk_completion(self,
                               job_id: str,
                               chunk_id: str,
                               processing_time_ms: float,
                               candles_processed: int,
                               success: bool,
                               cache_hits: int = 0,
                               cache_misses: int = 0):
        """Record metrics for a completed chunk"""
        
        # Throughput metrics
        candles_per_ms = candles_processed / max(processing_time_ms, 1)
        candles_per_minute = candles_per_ms * 60 * 1000
        
        self.record_job_metric(job_id, 'throughput_candles_per_minute', candles_per_minute)
        self.record_job_metric(job_id, 'chunk_processing_time_ms', processing_time_ms)
        self.record_job_metric(job_id, 'candles_per_chunk', candles_processed)
        
        # Success/error tracking
        self.record_job_metric(job_id, 'chunk_success', 1.0 if success else 0.0)
        
        # Cache metrics
        total_cache_requests = cache_hits + cache_misses
        if total_cache_requests > 0:
            cache_hit_rate = cache_hits / total_cache_requests
            self.record_job_metric(job_id, 'cache_hit_rate', cache_hit_rate)
        
        # Store in Redis for persistence
        chunk_metrics = {
            'job_id': job_id,
            'chunk_id': chunk_id,
            'timestamp': datetime.utcnow().isoformat(),
            'processing_time_ms': processing_time_ms,
            'candles_processed': candles_processed,
            'throughput_candles_per_minute': candles_per_minute,
            'success': success,
            'cache_hit_rate': cache_hit_rate if total_cache_requests > 0 else 0
        }
        
        redis_cache.set(
            f"{self.redis_key_prefix}:chunk:{job_id}:{chunk_id}",
            chunk_metrics,
            ttl=86400  # 24 hours
        )
    
    def get_job_performance(self, job_id: str, window_seconds: int = 300) -> Optional[PerformanceSnapshot]:
        """Get current performance snapshot for a job"""
        
        if job_id not in self.job_metrics:
            return None
        
        job_metrics = self.job_metrics[job_id]
        
        # Throughput metrics
        throughput_values = job_metrics['throughput_candles_per_minute'].get_recent_values(window_seconds)
        avg_throughput = statistics.mean(throughput_values) if throughput_values else 0
        candles_per_second = avg_throughput / 60
        
        # Chunk processing rate
        chunk_times = job_metrics['chunk_processing_time_ms'].get_recent_values(window_seconds)
        chunks_completed = len(chunk_times)
        chunks_per_second = chunks_completed / window_seconds if chunks_completed > 0 else 0
        
        # Latency percentiles
        avg_chunk_time = statistics.mean(chunk_times) if chunk_times else 0
        p50_time = statistics.median(chunk_times) if chunk_times else 0
        p95_time = job_metrics['chunk_processing_time_ms'].get_percentile(95, window_seconds)
        p99_time = job_metrics['chunk_processing_time_ms'].get_percentile(99, window_seconds)
        
        # Error rates
        success_values = job_metrics['chunk_success'].get_recent_values(window_seconds)
        error_rate = 1 - statistics.mean(success_values) if success_values else 0
        
        # Cache hit rate
        cache_values = job_metrics['cache_hit_rate'].get_recent_values(window_seconds)
        cache_hit_rate = statistics.mean(cache_values) if cache_values else 0
        
        # Get current job progress from Redis
        job_data = redis_cache.get(f"batch_job:{job_id}")
        progress_percent = 0
        eta_minutes = None
        eta_confidence = None
        
        if job_data:
            # Calculate progress and ETA
            total_chunks = job_data.get('total_chunks', 1)
            completed_chunks = len(job_metrics['chunk_success'].values)
            progress_percent = (completed_chunks / total_chunks) * 100
            
            # ETA calculation
            if chunks_per_second > 0 and completed_chunks > 0:
                remaining_chunks = total_chunks - completed_chunks
                eta_seconds = remaining_chunks / chunks_per_second
                eta_minutes = eta_seconds / 60
                
                # Confidence based on data points and variance
                eta_confidence = min(1.0, completed_chunks / 10)  # More confident with more data
                if len(chunk_times) > 1:
                    cv = statistics.stdev(chunk_times) / max(avg_chunk_time, 1)  # Coefficient of variation
                    eta_confidence *= max(0.1, 1 - cv)  # Lower confidence with high variance
        
        # System resource metrics (would be collected from system monitoring)
        cpu_usage = 0  # Would get from psutil or system monitoring
        memory_usage = 0  # Would get from system monitoring
        
        return PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            job_id=job_id,
            candles_per_second=candles_per_second,
            chunks_per_second=chunks_per_second,
            avg_chunk_time_ms=avg_chunk_time,
            p50_chunk_time_ms=p50_time,
            p95_chunk_time_ms=p95_time,
            p99_chunk_time_ms=p99_time,
            error_rate=error_rate,
            retry_rate=0,  # Would calculate from retry metrics
            cache_hit_rate=cache_hit_rate,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage,
            progress_percent=progress_percent,
            eta_minutes=eta_minutes,
            eta_confidence=eta_confidence
        )
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        
        metrics = {}
        
        # Aggregate metrics across all jobs
        for metric_name, metric in self.metrics.items():
            recent_values = metric.get_recent_values()
            if recent_values:
                metrics[f"{metric_name}_avg"] = statistics.mean(recent_values)
                metrics[f"{metric_name}_p95"] = metric.get_percentile(95)
                metrics[f"{metric_name}_current"] = recent_values[-1]
        
        # Job counts by status
        all_jobs = redis_cache.client.keys("batch_job:*")
        job_status_counts = defaultdict(int)
        
        for job_key in all_jobs:
            job_data = redis_cache.get(job_key.decode())
            if job_data and isinstance(job_data, dict):
                status = job_data.get('status', 'unknown')
                job_status_counts[status] += 1
        
        metrics['jobs_by_status'] = dict(job_status_counts)
        metrics['total_active_jobs'] = job_status_counts.get('running', 0) + job_status_counts.get('pending', 0)
        
        # System health indicators
        metrics['system_health'] = self._calculate_system_health(metrics)
        
        return metrics
    
    def get_alerts(self, severity_threshold: str = 'warning') -> List[Dict[str, Any]]:
        """Get active performance alerts"""
        alerts = []
        
        # Check global metrics against thresholds
        system_metrics = self.get_system_metrics()
        
        # Throughput alerts
        avg_throughput = system_metrics.get('job_throughput_candles_per_minute_avg', 0)
        if avg_throughput < self.target_throughput * 0.5:  # 50% of target
            alerts.append({
                'severity': 'critical' if avg_throughput < self.target_throughput * 0.25 else 'warning',
                'metric': 'throughput',
                'message': f"Low throughput: {avg_throughput:,.0f} candles/min (target: {self.target_throughput:,})",
                'value': avg_throughput,
                'threshold': self.target_throughput * 0.5
            })
        
        # Error rate alerts
        avg_error_rate = system_metrics.get('job_chunk_success_avg', 1.0)
        error_rate = 1 - avg_error_rate
        if error_rate > self.max_error_rate:
            alerts.append({
                'severity': 'critical' if error_rate > self.max_error_rate * 2 else 'warning',
                'metric': 'error_rate',
                'message': f"High error rate: {error_rate:.1%} (max: {self.max_error_rate:.1%})",
                'value': error_rate,
                'threshold': self.max_error_rate
            })
        
        # Latency alerts
        avg_latency = system_metrics.get('job_chunk_processing_time_ms_avg', 0)
        if avg_latency > self.max_latency_ms:
            alerts.append({
                'severity': 'warning',
                'metric': 'latency',
                'message': f"High latency: {avg_latency:.0f}ms (max: {self.max_latency_ms}ms)",
                'value': avg_latency,
                'threshold': self.max_latency_ms
            })
        
        # Filter by severity
        severity_levels = ['info', 'warning', 'critical']
        min_level = severity_levels.index(severity_threshold)
        
        return [alert for alert in alerts 
                if severity_levels.index(alert['severity']) >= min_level]
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        output = []
        timestamp = int(time.time() * 1000)
        
        # Export all time series metrics
        for metric_name, metric in self.metrics.items():
            # Current value
            recent_values = metric.get_recent_values(60)  # Last minute
            if recent_values:
                current_value = recent_values[-1]
                output.append(f"batch_{metric_name} {current_value} {timestamp}")
                
                # Average over last 5 minutes
                avg_value = statistics.mean(recent_values)
                output.append(f"batch_{metric_name}_avg {avg_value} {timestamp}")
        
        # Export job-specific metrics
        for job_id, job_metrics in self.job_metrics.items():
            for metric_name, metric in job_metrics.items():
                recent_values = metric.get_recent_values(60)
                if recent_values:
                    current_value = recent_values[-1]
                    output.append(f'batch_job_{metric_name}{{job_id="{job_id}"}} {current_value} {timestamp}')
        
        return '\n'.join(output)
    
    def _collection_loop(self, interval_seconds: int):
        """Background metric collection loop"""
        while not self._stop_collection.wait(interval_seconds):
            try:
                self._collect_system_metrics()
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        
        # Redis metrics
        try:
            redis_info = redis_cache.client.info()
            self.metrics['redis_memory_mb'].add_point(
                redis_info.get('used_memory', 0) / 1024 / 1024
            )
            
            # Key count
            db_info = redis_info.get('db0', {})
            if isinstance(db_info, dict):
                key_count = db_info.get('keys', 0)
                self.metrics['redis_keys_total'].add_point(key_count)
                
        except Exception as e:
            logger.warning(f"Failed to collect Redis metrics: {e}")
        
        # Job count metrics
        try:
            all_jobs = redis_cache.client.keys("batch_job:*")
            active_count = 0
            
            for job_key in all_jobs[:100]:  # Limit to avoid performance issues
                job_data = redis_cache.get(job_key.decode())
                if job_data and isinstance(job_data, dict):
                    status = job_data.get('status')
                    if status in ['running', 'pending']:
                        active_count += 1
            
            self.metrics['active_jobs_count'].add_point(active_count)
            
        except Exception as e:
            logger.warning(f"Failed to collect job metrics: {e}")
    
    def _calculate_system_health(self, metrics: Dict[str, Any]) -> str:
        """Calculate overall system health score"""
        
        health_score = 100
        
        # Throughput health (30% weight)
        avg_throughput = metrics.get('job_throughput_candles_per_minute_avg', 0)
        throughput_ratio = avg_throughput / self.target_throughput
        if throughput_ratio < 0.5:
            health_score -= 30 * (1 - throughput_ratio * 2)
        
        # Error rate health (40% weight)
        avg_error_rate = 1 - metrics.get('job_chunk_success_avg', 1.0)
        if avg_error_rate > 0:
            health_score -= 40 * min(1.0, avg_error_rate / self.max_error_rate)
        
        # Latency health (20% weight)
        avg_latency = metrics.get('job_chunk_processing_time_ms_avg', 0)
        if avg_latency > self.max_latency_ms:
            health_score -= 20 * min(1.0, avg_latency / self.max_latency_ms - 1)
        
        # System resource health (10% weight)
        redis_memory = metrics.get('redis_memory_mb_current', 0)
        if redis_memory > 1000:  # 1GB threshold
            health_score -= 10 * min(1.0, redis_memory / 2000)  # Max penalty at 2GB
        
        health_score = max(0, health_score)
        
        if health_score >= 80:
            return 'healthy'
        elif health_score >= 60:
            return 'degraded'
        else:
            return 'unhealthy'


# Global metrics collector instance
metrics_collector = BatchMetricsCollector()