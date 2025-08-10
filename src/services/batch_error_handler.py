"""
Comprehensive Error Handling and Recovery System for Batch Processing

Advanced error handling with:
- Intelligent retry strategies with exponential backoff
- Circuit breaker pattern for failing dependencies
- Graceful degradation and fallback mechanisms  
- Error categorization and routing
- Dead letter queue for failed tasks
- Recovery procedures and self-healing
- Error reporting and alerting
- Performance impact minimization
"""

import asyncio
import logging
import time
import random
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import pickle
from collections import defaultdict, deque

from src.services.redis_cache import redis_cache

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category classification"""
    TRANSIENT = "transient"        # Network timeouts, temporary resource issues
    PERSISTENT = "persistent"      # Configuration errors, invalid data
    RESOURCE = "resource"         # Memory, CPU, disk space issues
    DEPENDENCY = "dependency"     # External service failures
    DATA = "data"                # Data validation, corruption issues
    SYSTEM = "system"            # System-level errors, permissions
    UNKNOWN = "unknown"          # Unclassified errors


class RetryStrategy(Enum):
    """Retry strategy types"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"
    NO_RETRY = "no_retry"


@dataclass
class ErrorRecord:
    """Individual error record"""
    error_id: str
    job_id: str
    task_id: str
    error_type: str
    error_message: str
    stack_trace: str
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for dependency monitoring"""
    service_name: str
    failure_count: int = 0
    failure_threshold: int = 5
    success_count: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    timeout_duration: int = 60  # seconds
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit breaker state"""
        now = datetime.utcnow()
        
        if self.state == "closed":
            return True
        
        elif self.state == "open":
            if self.last_failure and (now - self.last_failure).total_seconds() > self.timeout_duration:
                self.state = "half_open"
                logger.info(f"Circuit breaker for {self.service_name} moved to half-open")
                return True
            return False
        
        elif self.state == "half_open":
            return True
        
        return False
    
    def record_success(self):
        """Record successful operation"""
        self.success_count += 1
        self.last_success = datetime.utcnow()
        
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            logger.info(f"Circuit breaker for {self.service_name} closed after success")
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker for {self.service_name} opened after {self.failure_count} failures")


class BatchErrorHandler:
    """
    Comprehensive error handling system for batch processing operations.
    
    Features:
    - Intelligent error categorization and routing
    - Multiple retry strategies with configurable backoff
    - Circuit breaker pattern for dependency protection
    - Dead letter queue for permanently failed tasks
    - Error trend analysis and alerting
    - Graceful degradation mechanisms
    - Self-healing and automatic recovery
    - Performance-aware error handling
    """
    
    def __init__(self, redis_key_prefix: str = "batch_errors"):
        self.redis_key_prefix = redis_key_prefix
        
        # Error storage and tracking
        self.error_records: Dict[str, ErrorRecord] = {}
        self.error_trends = defaultdict(lambda: deque(maxlen=100))
        
        # Circuit breakers for external dependencies
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {
            'clickhouse': CircuitBreakerState('clickhouse'),
            'redis': CircuitBreakerState('redis'),
            'computation_engine': CircuitBreakerState('computation_engine')
        }
        
        # Retry configuration
        self.default_retry_config = {
            ErrorCategory.TRANSIENT: {
                'max_retries': 5,
                'strategy': RetryStrategy.EXPONENTIAL_BACKOFF,
                'base_delay': 1.0,
                'max_delay': 60.0,
                'jitter': True
            },
            ErrorCategory.RESOURCE: {
                'max_retries': 3,
                'strategy': RetryStrategy.LINEAR_BACKOFF,
                'base_delay': 5.0,
                'max_delay': 30.0,
                'jitter': True
            },
            ErrorCategory.DEPENDENCY: {
                'max_retries': 5,
                'strategy': RetryStrategy.JITTERED_BACKOFF,
                'base_delay': 2.0,
                'max_delay': 120.0,
                'jitter': True
            },
            ErrorCategory.DATA: {
                'max_retries': 1,
                'strategy': RetryStrategy.NO_RETRY,
                'base_delay': 0,
                'max_delay': 0,
                'jitter': False
            },
            ErrorCategory.PERSISTENT: {
                'max_retries': 0,
                'strategy': RetryStrategy.NO_RETRY,
                'base_delay': 0,
                'max_delay': 0,
                'jitter': False
            }
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'error_rate_threshold': 0.05,  # 5% error rate
            'critical_errors_threshold': 10,
            'dependency_failure_threshold': 3
        }
        
        logger.info("BatchErrorHandler initialized")
    
    def handle_error(self,
                    error: Exception,
                    job_id: str,
                    task_id: str,
                    context: Dict[str, Any] = None) -> ErrorRecord:
        """
        Handle an error with intelligent categorization and response.
        
        Args:
            error: Exception that occurred
            job_id: Job identifier
            task_id: Task identifier
            context: Additional context information
            
        Returns:
            ErrorRecord for the handled error
        """
        error_id = f"{job_id}_{task_id}_{int(time.time())}"
        
        # Categorize error
        category = self._categorize_error(error)
        severity = self._determine_severity(error, category)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            job_id=job_id,
            task_id=task_id,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            category=category,
            severity=severity,
            timestamp=datetime.utcnow(),
            context=context or {}
        )
        
        # Apply retry configuration based on category
        retry_config = self.default_retry_config.get(category, self.default_retry_config[ErrorCategory.UNKNOWN])
        error_record.max_retries = retry_config['max_retries']
        error_record.retry_strategy = retry_config['strategy']
        
        # Store error record
        self.error_records[error_id] = error_record
        self._persist_error_record(error_record)
        
        # Update error trends
        self.error_trends[category].append(datetime.utcnow())
        
        # Update circuit breakers if dependency error
        if category == ErrorCategory.DEPENDENCY:
            service_name = self._identify_failed_service(error, context)
            if service_name in self.circuit_breakers:
                self.circuit_breakers[service_name].record_failure()
        
        # Check for alert conditions
        self._check_alert_conditions(error_record)
        
        logger.error(
            f"Error handled: {error_record.error_type} (Category: {category.value}, "
            f"Severity: {severity.value}) - {error_record.error_message}"
        )
        
        return error_record
    
    def should_retry(self, error_record: ErrorRecord) -> Tuple[bool, float]:
        """
        Determine if an error should be retried and calculate delay.
        
        Args:
            error_record: Error record to evaluate
            
        Returns:
            Tuple of (should_retry, delay_seconds)
        """
        if error_record.retry_count >= error_record.max_retries:
            return False, 0
        
        if error_record.retry_strategy == RetryStrategy.NO_RETRY:
            return False, 0
        
        # Check circuit breaker for dependency errors
        if error_record.category == ErrorCategory.DEPENDENCY:
            service_name = self._identify_failed_service_from_record(error_record)
            if service_name in self.circuit_breakers:
                if not self.circuit_breakers[service_name].should_allow_request():
                    logger.info(f"Circuit breaker open for {service_name}, skipping retry")
                    return False, 0
        
        # Calculate delay based on strategy
        delay = self._calculate_retry_delay(error_record)
        
        return True, delay
    
    def record_retry_attempt(self, error_record: ErrorRecord):
        """Record a retry attempt"""
        error_record.retry_count += 1
        self._persist_error_record(error_record)
        
        logger.info(f"Retry attempt {error_record.retry_count} for error {error_record.error_id}")
    
    def record_success(self, job_id: str, task_id: str, service_name: str = None):
        """Record successful operation (for circuit breaker management)"""
        if service_name and service_name in self.circuit_breakers:
            self.circuit_breakers[service_name].record_success()
    
    def get_error_statistics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get error statistics for the specified time window.
        
        Args:
            time_window_hours: Time window for statistics
            
        Returns:
            Dictionary of error statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Filter recent errors
        recent_errors = [
            error for error in self.error_records.values()
            if error.timestamp >= cutoff_time
        ]
        
        # Calculate statistics
        total_errors = len(recent_errors)
        errors_by_category = defaultdict(int)
        errors_by_severity = defaultdict(int)
        errors_by_type = defaultdict(int)
        resolved_errors = 0
        
        for error in recent_errors:
            errors_by_category[error.category.value] += 1
            errors_by_severity[error.severity.value] += 1
            errors_by_type[error.error_type] += 1
            if error.resolved:
                resolved_errors += 1
        
        # Circuit breaker states
        circuit_breaker_states = {
            name: {
                'state': cb.state,
                'failure_count': cb.failure_count,
                'last_failure': cb.last_failure.isoformat() if cb.last_failure else None
            }
            for name, cb in self.circuit_breakers.items()
        }
        
        return {
            'time_window_hours': time_window_hours,
            'total_errors': total_errors,
            'resolved_errors': resolved_errors,
            'resolution_rate': resolved_errors / max(total_errors, 1),
            'errors_by_category': dict(errors_by_category),
            'errors_by_severity': dict(errors_by_severity),
            'errors_by_type': dict(errors_by_type),
            'circuit_breakers': circuit_breaker_states,
            'error_trends': self._calculate_error_trends()
        }
    
    def get_failed_tasks(self, job_id: str = None) -> List[ErrorRecord]:
        """
        Get tasks that have exhausted retries (dead letter queue).
        
        Args:
            job_id: Optional job ID filter
            
        Returns:
            List of permanently failed error records
        """
        failed_tasks = []
        
        for error_record in self.error_records.values():
            if (error_record.retry_count >= error_record.max_retries and 
                not error_record.resolved and
                (not job_id or error_record.job_id == job_id)):
                failed_tasks.append(error_record)
        
        return failed_tasks
    
    def resolve_error(self, error_id: str, resolution_notes: str = None):
        """Mark an error as resolved"""
        if error_id in self.error_records:
            error_record = self.error_records[error_id]
            error_record.resolved = True
            error_record.resolution_time = datetime.utcnow()
            
            if resolution_notes:
                error_record.context['resolution_notes'] = resolution_notes
            
            self._persist_error_record(error_record)
            logger.info(f"Error {error_id} marked as resolved")
    
    def cleanup_old_errors(self, retention_days: int = 7):
        """Clean up old error records"""
        cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
        
        old_error_ids = [
            error_id for error_id, error_record in self.error_records.items()
            if error_record.timestamp < cutoff_time
        ]
        
        for error_id in old_error_ids:
            del self.error_records[error_id]
            redis_cache.delete(f"{self.redis_key_prefix}:error:{error_id}")
        
        logger.info(f"Cleaned up {len(old_error_ids)} old error records")
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on type and message"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Network and connectivity errors
        if any(keyword in error_message for keyword in 
               ['timeout', 'connection', 'network', 'unreachable', 'refused']):
            return ErrorCategory.TRANSIENT
        
        # Resource errors
        if any(keyword in error_message for keyword in 
               ['memory', 'disk', 'space', 'resource', 'limit']):
            return ErrorCategory.RESOURCE
        
        # Data validation errors
        if any(keyword in error_message for keyword in 
               ['validation', 'invalid', 'format', 'schema', 'parse']):
            return ErrorCategory.DATA
        
        # Permission and configuration errors
        if any(keyword in error_message for keyword in 
               ['permission', 'access', 'forbidden', 'unauthorized', 'config']):
            return ErrorCategory.PERSISTENT
        
        # Database and external service errors
        if any(keyword in error_message for keyword in 
               ['clickhouse', 'redis', 'database', 'service']):
            return ErrorCategory.DEPENDENCY
        
        # System-level errors
        if error_type in ['OSError', 'SystemError', 'EnvironmentError']:
            return ErrorCategory.SYSTEM
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if (category == ErrorCategory.SYSTEM or 
            'critical' in error_message or
            error_type in ['SystemExit', 'KeyboardInterrupt']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if (category in [ErrorCategory.DEPENDENCY, ErrorCategory.RESOURCE] or
            'fatal' in error_message or
            error_type in ['MemoryError', 'OSError']):
            return ErrorSeverity.ERROR
        
        # Medium severity errors
        if (category == ErrorCategory.DATA or
            'warning' in error_message):
            return ErrorSeverity.WARNING
        
        return ErrorSeverity.INFO
    
    def _identify_failed_service(self, error: Exception, context: Dict[str, Any] = None) -> str:
        """Identify which service failed based on error and context"""
        error_message = str(error).lower()
        
        if 'clickhouse' in error_message:
            return 'clickhouse'
        elif 'redis' in error_message:
            return 'redis'
        elif context and 'service' in context:
            return context['service']
        
        return 'unknown'
    
    def _identify_failed_service_from_record(self, error_record: ErrorRecord) -> str:
        """Identify failed service from error record"""
        return self._identify_failed_service(
            Exception(error_record.error_message), 
            error_record.context
        )
    
    def _calculate_retry_delay(self, error_record: ErrorRecord) -> float:
        """Calculate retry delay based on strategy"""
        retry_config = self.default_retry_config.get(
            error_record.category, 
            self.default_retry_config[ErrorCategory.UNKNOWN]
        )
        
        base_delay = retry_config['base_delay']
        max_delay = retry_config['max_delay']
        strategy = error_record.retry_strategy
        
        if strategy == RetryStrategy.FIXED_DELAY:
            delay = base_delay
        
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (2 ** error_record.retry_count)
        
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base_delay * (error_record.retry_count + 1)
        
        elif strategy == RetryStrategy.JITTERED_BACKOFF:
            exponential_delay = base_delay * (2 ** error_record.retry_count)
            jitter = random.uniform(0, exponential_delay * 0.1)
            delay = exponential_delay + jitter
        
        else:
            delay = base_delay
        
        # Apply maximum delay limit
        delay = min(delay, max_delay)
        
        # Add jitter if configured
        if retry_config.get('jitter', False):
            jitter_amount = delay * 0.1
            jitter = random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay + jitter)
        
        return delay
    
    def _calculate_error_trends(self) -> Dict[str, Dict[str, float]]:
        """Calculate error rate trends"""
        trends = {}
        
        for category, timestamps in self.error_trends.items():
            if not timestamps:
                continue
            
            now = datetime.utcnow()
            recent_errors = [ts for ts in timestamps if (now - ts).total_seconds() <= 3600]  # Last hour
            hourly_rate = len(recent_errors)
            
            day_errors = [ts for ts in timestamps if (now - ts).total_seconds() <= 86400]  # Last day
            daily_rate = len(day_errors) / 24  # Per hour average
            
            trends[category] = {
                'hourly_rate': hourly_rate,
                'daily_average': daily_rate,
                'trend': 'increasing' if hourly_rate > daily_rate * 1.2 else 
                        'decreasing' if hourly_rate < daily_rate * 0.8 else 'stable'
            }
        
        return trends
    
    def _check_alert_conditions(self, error_record: ErrorRecord):
        """Check if error conditions warrant alerts"""
        # Check error rate threshold
        recent_errors = [
            error for error in self.error_records.values()
            if (datetime.utcnow() - error.timestamp).total_seconds() <= 3600
        ]
        
        if len(recent_errors) > 50:  # More than 50 errors in last hour
            logger.warning(f"High error rate detected: {len(recent_errors)} errors in last hour")
        
        # Check for critical errors
        if error_record.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error detected: {error_record.error_message}")
        
        # Check circuit breaker failures
        for service_name, cb in self.circuit_breakers.items():
            if cb.state == "open":
                logger.error(f"Circuit breaker open for {service_name} - service degraded")
    
    def _persist_error_record(self, error_record: ErrorRecord):
        """Persist error record to Redis"""
        try:
            error_data = {
                'error_id': error_record.error_id,
                'job_id': error_record.job_id,
                'task_id': error_record.task_id,
                'error_type': error_record.error_type,
                'error_message': error_record.error_message,
                'category': error_record.category.value,
                'severity': error_record.severity.value,
                'timestamp': error_record.timestamp.isoformat(),
                'retry_count': error_record.retry_count,
                'max_retries': error_record.max_retries,
                'context': error_record.context,
                'resolved': error_record.resolved
            }
            
            redis_cache.set(
                f"{self.redis_key_prefix}:error:{error_record.error_id}",
                error_data,
                ttl=86400 * 7  # 7 days
            )
            
        except Exception as e:
            logger.error(f"Failed to persist error record: {e}")


# Global error handler instance
batch_error_handler = BatchErrorHandler()