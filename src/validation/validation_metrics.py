"""
Validation metrics and alerting system for Issue #8

This module provides comprehensive metrics collection and alerting
for the validation framework, following test-runner best practices
for monitoring and observability.

Features:
- Real-time validation metrics collection
- Threshold-based alerting
- Statistical analysis of validation patterns
- Performance monitoring
- Integration with monitoring systems
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from threading import Lock

from src.validation.label_validator import ValidationResult, ValidationSeverity, ValidationCategory

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Validation metrics container"""
    timestamp: datetime
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    critical_issues: int = 0
    error_issues: int = 0
    warning_issues: int = 0
    info_issues: int = 0
    avg_validation_time_ms: float = 0.0
    max_validation_time_ms: float = 0.0
    validation_time_p95_ms: float = 0.0
    category_counts: Dict[str, int] = field(default_factory=dict)
    

@dataclass
class AlertRule:
    """Validation alert rule configuration"""
    name: str
    description: str
    condition: str  # Python expression evaluated against metrics
    severity: str   # "low", "medium", "high", "critical"
    threshold: float
    window_minutes: int = 5
    cooldown_minutes: int = 15
    enabled: bool = True


class ValidationMetricsCollector:
    """
    Collects and analyzes validation metrics for monitoring and alerting.
    
    Provides real-time metrics collection, threshold-based alerting,
    and statistical analysis of validation patterns.
    """
    
    def __init__(self, max_history_size: int = 10000):
        """
        Initialize metrics collector
        
        Args:
            max_history_size: Maximum number of validation results to keep in memory
        """
        self.max_history_size = max_history_size
        self.validation_history = deque(maxlen=max_history_size)
        self.validation_times = deque(maxlen=1000)  # Keep last 1000 timing measurements
        self.metrics_lock = Lock()
        
        # Current metrics
        self.current_metrics = ValidationMetrics(timestamp=datetime.utcnow())
        
        # Category and severity counters
        self.category_counters = defaultdict(int)
        self.severity_counters = defaultdict(int)
        
        # Alert management
        self.alert_rules = self._initialize_default_alert_rules()
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Performance tracking
        self.hourly_metrics = deque(maxlen=24 * 7)  # 7 days of hourly metrics
        self.last_hourly_update = datetime.utcnow()
    
    def record_validation_result(self, result: ValidationResult):
        """
        Record a validation result and update metrics
        
        Args:
            result: Validation result to record
        """
        with self.metrics_lock:
            # Add to history
            self.validation_history.append(result)
            
            # Update current metrics
            self.current_metrics.total_validations += 1
            
            if result.is_valid:
                self.current_metrics.successful_validations += 1
            else:
                self.current_metrics.failed_validations += 1
            
            # Count issues by severity
            for issue in result.issues:
                self.severity_counters[issue.severity.value] += 1
                self.category_counters[issue.category.value] += 1
                
                if issue.severity == ValidationSeverity.CRITICAL:
                    self.current_metrics.critical_issues += 1
                elif issue.severity == ValidationSeverity.ERROR:
                    self.current_metrics.error_issues += 1
                elif issue.severity == ValidationSeverity.WARNING:
                    self.current_metrics.warning_issues += 1
                elif issue.severity == ValidationSeverity.INFO:
                    self.current_metrics.info_issues += 1
            
            # Update validation time metrics
            if result.validation_time_ms is not None:
                self.validation_times.append(result.validation_time_ms)
                self._update_timing_metrics()
            
            # Update category counts
            self.current_metrics.category_counts = dict(self.category_counters)
            self.current_metrics.timestamp = datetime.utcnow()
            
            # Check for alerts
            self._check_alert_rules()
            
            # Update hourly metrics if needed
            self._update_hourly_metrics()
    
    def get_current_metrics(self) -> ValidationMetrics:
        """Get current validation metrics"""
        with self.metrics_lock:
            return self.current_metrics
    
    def get_metrics_summary(self, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Get validation metrics summary
        
        Args:
            window_minutes: Time window for metrics (None for all-time)
            
        Returns:
            Metrics summary dictionary
        """
        with self.metrics_lock:
            if window_minutes is None:
                # All-time metrics
                relevant_results = list(self.validation_history)
            else:
                # Window-based metrics
                cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
                relevant_results = [
                    r for r in self.validation_history
                    if hasattr(r, 'timestamp') and r.timestamp and r.timestamp >= cutoff_time
                ]
            
            if not relevant_results:
                return {
                    "total_validations": 0,
                    "success_rate": 0.0,
                    "failure_rate": 0.0,
                    "window_minutes": window_minutes,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Calculate metrics
            total = len(relevant_results)
            successful = sum(1 for r in relevant_results if r.is_valid)
            failed = total - successful
            
            # Issue counts by severity
            severity_counts = defaultdict(int)
            category_counts = defaultdict(int)
            
            for result in relevant_results:
                for issue in result.issues:
                    severity_counts[issue.severity.value] += 1
                    category_counts[issue.category.value] += 1
            
            # Timing statistics
            validation_times = [
                r.validation_time_ms for r in relevant_results
                if r.validation_time_ms is not None
            ]
            
            timing_stats = {}
            if validation_times:
                import numpy as np
                timing_stats = {
                    "avg_ms": float(np.mean(validation_times)),
                    "median_ms": float(np.median(validation_times)),
                    "p95_ms": float(np.percentile(validation_times, 95)),
                    "p99_ms": float(np.percentile(validation_times, 99)),
                    "max_ms": float(np.max(validation_times)),
                    "min_ms": float(np.min(validation_times))
                }
            
            return {
                "total_validations": total,
                "successful_validations": successful,
                "failed_validations": failed,
                "success_rate": successful / total if total > 0 else 0.0,
                "failure_rate": failed / total if total > 0 else 0.0,
                "severity_counts": dict(severity_counts),
                "category_counts": dict(category_counts),
                "timing_stats": timing_stats,
                "window_minutes": window_minutes,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def get_failure_analysis(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Analyze validation failures to identify patterns
        
        Args:
            top_n: Number of top issues to include
            
        Returns:
            Failure analysis summary
        """
        with self.metrics_lock:
            failed_results = [r for r in self.validation_history if not r.is_valid]
            
            if not failed_results:
                return {
                    "total_failures": 0,
                    "analysis": "No validation failures recorded"
                }
            
            # Analyze issue patterns
            issue_patterns = defaultdict(int)
            category_severity_matrix = defaultdict(lambda: defaultdict(int))
            
            for result in failed_results:
                for issue in result.issues:
                    # Count issue types
                    issue_key = f"{issue.category.value}:{issue.severity.value}"
                    issue_patterns[issue_key] += 1
                    
                    # Build category-severity matrix
                    category_severity_matrix[issue.category.value][issue.severity.value] += 1
            
            # Get top issue patterns
            top_patterns = sorted(
                issue_patterns.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
            return {
                "total_failures": len(failed_results),
                "failure_rate": len(failed_results) / len(self.validation_history),
                "top_issue_patterns": top_patterns,
                "category_severity_matrix": {
                    cat: dict(severities) 
                    for cat, severities in category_severity_matrix.items()
                },
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    def get_health_score(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate validation health score (0-100)
        
        Returns:
            Tuple of (health_score, detailed_breakdown)
        """
        with self.metrics_lock:
            if not self.validation_history:
                return 100.0, {"status": "no_data"}
            
            # Recent validation results (last hour)
            recent_cutoff = datetime.utcnow() - timedelta(hours=1)
            recent_results = [
                r for r in self.validation_history
                if hasattr(r, 'timestamp') and r.timestamp and r.timestamp >= recent_cutoff
            ]
            
            if not recent_results:
                recent_results = list(self.validation_history)[-10:]  # Last 10 if no recent data
            
            # Calculate health components
            success_rate = sum(1 for r in recent_results if r.is_valid) / len(recent_results)
            
            # Critical issue penalty
            critical_issues = sum(
                len([i for i in r.issues if i.severity == ValidationSeverity.CRITICAL])
                for r in recent_results
            )
            critical_penalty = min(critical_issues * 10, 50)  # Up to 50 points penalty
            
            # Error issue penalty
            error_issues = sum(
                len([i for i in r.issues if i.severity == ValidationSeverity.ERROR])
                for r in recent_results
            )
            error_penalty = min(error_issues * 5, 30)  # Up to 30 points penalty
            
            # Performance penalty (if avg time > 1 second)
            validation_times = [
                r.validation_time_ms for r in recent_results
                if r.validation_time_ms is not None
            ]
            performance_penalty = 0
            if validation_times:
                avg_time = sum(validation_times) / len(validation_times)
                if avg_time > 1000:  # > 1 second
                    performance_penalty = min((avg_time - 1000) / 1000 * 10, 20)  # Up to 20 points
            
            # Calculate health score
            base_score = success_rate * 100
            health_score = max(0, base_score - critical_penalty - error_penalty - performance_penalty)
            
            breakdown = {
                "base_score": base_score,
                "success_rate": success_rate,
                "critical_penalty": critical_penalty,
                "error_penalty": error_penalty,
                "performance_penalty": performance_penalty,
                "final_score": health_score,
                "samples_analyzed": len(recent_results),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return health_score, breakdown
    
    def _update_timing_metrics(self):
        """Update timing-related metrics"""
        if not self.validation_times:
            return
            
        import numpy as np
        times = list(self.validation_times)
        
        self.current_metrics.avg_validation_time_ms = float(np.mean(times))
        self.current_metrics.max_validation_time_ms = float(np.max(times))
        self.current_metrics.validation_time_p95_ms = float(np.percentile(times, 95))
    
    def _initialize_default_alert_rules(self) -> List[AlertRule]:
        """Initialize default alert rules"""
        return [
            AlertRule(
                name="high_failure_rate",
                description="Validation failure rate exceeds 10%",
                condition="failure_rate > 0.10",
                severity="high",
                threshold=0.10,
                window_minutes=5
            ),
            AlertRule(
                name="critical_issues_detected",
                description="Critical validation issues detected",
                condition="critical_issues > 0",
                severity="critical",
                threshold=1,
                window_minutes=1
            ),
            AlertRule(
                name="slow_validation_performance",
                description="Average validation time exceeds 1 second",
                condition="avg_validation_time_ms > 1000",
                severity="medium",
                threshold=1000,
                window_minutes=10
            ),
            AlertRule(
                name="high_error_rate",
                description="Error issue rate exceeds 5%",
                condition="(error_issues / total_validations) > 0.05",
                severity="high",
                threshold=0.05,
                window_minutes=5
            ),
            AlertRule(
                name="lookahead_bias_detected",
                description="Look-ahead bias violations detected",
                condition="category_counts.get('lookahead_bias', 0) > 0",
                severity="critical",
                threshold=1,
                window_minutes=1
            )
        ]
    
    def _check_alert_rules(self):
        """Check all alert rules against current metrics"""
        current_time = datetime.utcnow()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
                
            # Check if alert is in cooldown
            alert_key = rule.name
            if alert_key in self.active_alerts:
                last_alert_time = self.active_alerts[alert_key]["timestamp"]
                if (current_time - last_alert_time).total_seconds() < rule.cooldown_minutes * 60:
                    continue
            
            # Get metrics for the rule's time window
            window_metrics = self.get_metrics_summary(rule.window_minutes)
            
            # Evaluate condition
            try:
                # Create evaluation context
                eval_context = {
                    **window_metrics,
                    "failure_rate": window_metrics["failure_rate"],
                    "critical_issues": window_metrics["severity_counts"].get("critical", 0),
                    "error_issues": window_metrics["severity_counts"].get("error", 0),
                    "warning_issues": window_metrics["severity_counts"].get("warning", 0),
                    "avg_validation_time_ms": window_metrics["timing_stats"].get("avg_ms", 0),
                    "category_counts": window_metrics["category_counts"]
                }
                
                # Safely evaluate condition
                if eval(rule.condition, {"__builtins__": {}}, eval_context):
                    # Alert condition met
                    alert = {
                        "rule_name": rule.name,
                        "description": rule.description,
                        "severity": rule.severity,
                        "condition": rule.condition,
                        "threshold": rule.threshold,
                        "current_value": eval_context,
                        "timestamp": current_time,
                        "window_minutes": rule.window_minutes
                    }
                    
                    # Record alert
                    self.active_alerts[alert_key] = alert
                    self.alert_history.append(alert)
                    
                    # Log alert
                    log_level = {
                        "low": logging.INFO,
                        "medium": logging.WARNING,
                        "high": logging.ERROR,
                        "critical": logging.CRITICAL
                    }.get(rule.severity, logging.WARNING)
                    
                    logger.log(
                        log_level,
                        f"Validation alert: {rule.name} - {rule.description} "
                        f"(severity: {rule.severity})"
                    )
            
            except Exception as e:
                logger.warning(f"Error evaluating alert rule {rule.name}: {e}")
    
    def _update_hourly_metrics(self):
        """Update hourly metrics aggregation"""
        current_time = datetime.utcnow()
        
        # Check if we need to create a new hourly bucket
        if (current_time - self.last_hourly_update).total_seconds() >= 3600:  # 1 hour
            # Aggregate last hour's metrics
            hourly_cutoff = current_time - timedelta(hours=1)
            hourly_results = [
                r for r in self.validation_history
                if hasattr(r, 'timestamp') and r.timestamp and r.timestamp >= hourly_cutoff
            ]
            
            if hourly_results:
                hourly_summary = {
                    "timestamp": current_time.replace(minute=0, second=0, microsecond=0),
                    "total_validations": len(hourly_results),
                    "successful_validations": sum(1 for r in hourly_results if r.is_valid),
                    "failed_validations": sum(1 for r in hourly_results if not r.is_valid),
                    "critical_issues": sum(
                        len([i for i in r.issues if i.severity == ValidationSeverity.CRITICAL])
                        for r in hourly_results
                    ),
                    "error_issues": sum(
                        len([i for i in r.issues if i.severity == ValidationSeverity.ERROR])
                        for r in hourly_results
                    )
                }
                
                self.hourly_metrics.append(hourly_summary)
            
            self.last_hourly_update = current_time
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        current_time = datetime.utcnow()
        active = []
        
        for alert_key, alert in self.active_alerts.items():
            # Check if alert should still be active (within cooldown period)
            rule = next((r for r in self.alert_rules if r.name == alert_key), None)
            if rule and (current_time - alert["timestamp"]).total_seconds() < rule.cooldown_minutes * 60:
                active.append(alert)
        
        return active
    
    def get_historical_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get historical trends for the specified time period
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Historical trends analysis
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        relevant_metrics = [
            m for m in self.hourly_metrics
            if m["timestamp"] >= cutoff_time
        ]
        
        if not relevant_metrics:
            return {"status": "insufficient_data", "hours_requested": hours}
        
        # Calculate trends
        failure_rates = [
            m["failed_validations"] / m["total_validations"] if m["total_validations"] > 0 else 0
            for m in relevant_metrics
        ]
        
        critical_issue_counts = [m["critical_issues"] for m in relevant_metrics]
        error_issue_counts = [m["error_issues"] for m in relevant_metrics]
        
        return {
            "hours_analyzed": hours,
            "data_points": len(relevant_metrics),
            "failure_rate_trend": {
                "current": failure_rates[-1] if failure_rates else 0,
                "average": sum(failure_rates) / len(failure_rates) if failure_rates else 0,
                "max": max(failure_rates) if failure_rates else 0,
                "min": min(failure_rates) if failure_rates else 0,
                "trend": "increasing" if len(failure_rates) >= 2 and failure_rates[-1] > failure_rates[0] else "stable"
            },
            "critical_issues_trend": {
                "current": critical_issue_counts[-1] if critical_issue_counts else 0,
                "total": sum(critical_issue_counts),
                "peak": max(critical_issue_counts) if critical_issue_counts else 0
            },
            "error_issues_trend": {
                "current": error_issue_counts[-1] if error_issue_counts else 0,
                "total": sum(error_issue_counts),
                "peak": max(error_issue_counts) if error_issue_counts else 0
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """
        Export metrics in specified format
        
        Args:
            format: Export format ("json", "prometheus")
            
        Returns:
            Formatted metrics string
        """
        if format == "json":
            return self._export_json()
        elif format == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON"""
        export_data = {
            "current_metrics": {
                "timestamp": self.current_metrics.timestamp.isoformat(),
                "total_validations": self.current_metrics.total_validations,
                "successful_validations": self.current_metrics.successful_validations,
                "failed_validations": self.current_metrics.failed_validations,
                "critical_issues": self.current_metrics.critical_issues,
                "error_issues": self.current_metrics.error_issues,
                "warning_issues": self.current_metrics.warning_issues,
                "info_issues": self.current_metrics.info_issues,
                "avg_validation_time_ms": self.current_metrics.avg_validation_time_ms,
                "max_validation_time_ms": self.current_metrics.max_validation_time_ms,
                "validation_time_p95_ms": self.current_metrics.validation_time_p95_ms,
                "category_counts": self.current_metrics.category_counts
            },
            "summary": self.get_metrics_summary(),
            "health_score": self.get_health_score()[0],
            "active_alerts": self.get_active_alerts(),
            "export_timestamp": datetime.utcnow().isoformat()
        }
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        
        # Basic counters
        lines.append(f"validation_total_count {self.current_metrics.total_validations}")
        lines.append(f"validation_successful_count {self.current_metrics.successful_validations}")
        lines.append(f"validation_failed_count {self.current_metrics.failed_validations}")
        
        # Issue counters by severity
        lines.append(f"validation_critical_issues {self.current_metrics.critical_issues}")
        lines.append(f"validation_error_issues {self.current_metrics.error_issues}")
        lines.append(f"validation_warning_issues {self.current_metrics.warning_issues}")
        lines.append(f"validation_info_issues {self.current_metrics.info_issues}")
        
        # Timing metrics
        lines.append(f"validation_time_avg_ms {self.current_metrics.avg_validation_time_ms}")
        lines.append(f"validation_time_max_ms {self.current_metrics.max_validation_time_ms}")
        lines.append(f"validation_time_p95_ms {self.current_metrics.validation_time_p95_ms}")
        
        # Category counts
        for category, count in self.current_metrics.category_counts.items():
            lines.append(f'validation_category_count{{category="{category}"}} {count}')
        
        # Health score
        health_score, _ = self.get_health_score()
        lines.append(f"validation_health_score {health_score}")
        
        return "\n".join(lines)


# Global metrics collector instance
validation_metrics_collector = ValidationMetricsCollector()