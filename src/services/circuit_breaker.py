"""
Circuit breaker implementation for external service resilience

Implements the circuit breaker pattern to prevent cascade failures and provide
graceful degradation when external services (ClickHouse, Redis, Firestore) fail.

Issue #14: Circuit breakers and failover mechanisms
Following infra-pr best practices for production resilience
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Generic, Union
from dataclasses import dataclass, field
from functools import wraps
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Service failed, requests blocked
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5           # Failures before opening
    recovery_timeout: float = 60.0       # Seconds before trying half-open
    success_threshold: int = 3           # Successes in half-open to close
    timeout: float = 30.0                # Request timeout in seconds
    expected_exception: tuple = (Exception,)  # Exceptions that count as failures
    
    # Advanced configuration
    sliding_window_size: int = 10        # Window for failure tracking
    half_open_max_calls: int = 5         # Max calls in half-open state
    failure_rate_threshold: float = 0.5  # Failure rate to open (0.0-1.0)
    
    # Monitoring
    enable_metrics: bool = True
    reset_timeout_on_success: bool = True


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_open_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: list = field(default_factory=list)
    
    def add_state_change(self, from_state: CircuitBreakerState, to_state: CircuitBreakerState):
        """Record state transition"""
        self.state_changes.append({
            'from': from_state.value,
            'to': to_state.value,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Keep only last 20 state changes
        if len(self.state_changes) > 20:
            self.state_changes = self.state_changes[-20:]
    
    def get_failure_rate(self) -> float:
        """Calculate current failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def get_success_rate(self) -> float:
        """Calculate current success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    
    def __init__(self, message: str, state: CircuitBreakerState, last_failure: Optional[Exception] = None):
        super().__init__(message)
        self.state = state
        self.last_failure = last_failure


class CircuitBreaker(Generic[T]):
    """
    Generic circuit breaker implementation following industry best practices.
    
    Features:
    - Three states: CLOSED, OPEN, HALF_OPEN
    - Configurable thresholds and timeouts
    - Sliding window failure tracking
    - Comprehensive metrics and monitoring
    - Thread-safe operation
    - Support for both sync and async functions
    """
    
    def __init__(self, 
                 name: str,
                 config: CircuitBreakerConfig,
                 fallback_function: Optional[Callable] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            config: Circuit breaker configuration
            fallback_function: Optional fallback when circuit is open
        """
        self.name = name
        self.config = config
        self.fallback_function = fallback_function
        
        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.next_attempt_time: float = 0
        self.half_open_calls = 0
        
        # Sliding window for failure tracking
        self.recent_calls: list = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        
        logger.info(f"Initialized circuit breaker '{name}' with config: {config}")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap function with circuit breaker"""
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func)
        else:
            return self._sync_wrapper(func)
    
    def _sync_wrapper(self, func: Callable[..., T]) -> Callable[..., T]:
        """Synchronous function wrapper"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def _async_wrapper(self, func: Callable[..., T]) -> Callable[..., T]:
        """Asynchronous function wrapper"""
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.call_async(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection (synchronous).
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: When circuit is open
        """
        with self.lock:
            self.metrics.total_requests += 1
            
            # Check if circuit should remain open
            if self.state == CircuitBreakerState.OPEN:
                if time.time() < self.next_attempt_time:
                    # Circuit is open and recovery timeout not reached
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN",
                        self.state,
                        getattr(self, 'last_exception', None)
                    )
                else:
                    # Try to transition to half-open
                    self._transition_to_half_open()
            
            # Check half-open state limits
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN with max calls reached",
                        self.state
                    )
                self.half_open_calls += 1
        
        # Execute the function
        start_time = time.time()
        try:
            # Apply timeout
            if hasattr(func, '__timeout__'):
                # Function has built-in timeout
                result = func(*args, **kwargs)
            else:
                # Apply circuit breaker timeout
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Function timeout after {self.config.timeout}s")
                
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.config.timeout))
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except self.config.expected_exception as e:
            # Record failure
            execution_time = time.time() - start_time
            self._record_failure(e, execution_time)
            
            # If we have a fallback, try it
            if self.fallback_function and self.state == CircuitBreakerState.OPEN:
                try:
                    logger.warning(f"Circuit breaker '{self.name}' using fallback for: {e}")
                    return self.fallback_function(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback failed for '{self.name}': {fallback_error}")
            
            raise
    
    async def call_async(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute async function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: When circuit is open
        """
        with self.lock:
            self.metrics.total_requests += 1
            
            # Check if circuit should remain open
            if self.state == CircuitBreakerState.OPEN:
                if time.time() < self.next_attempt_time:
                    # Circuit is open and recovery timeout not reached
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN",
                        self.state,
                        getattr(self, 'last_exception', None)
                    )
                else:
                    # Try to transition to half-open
                    self._transition_to_half_open()
            
            # Check half-open state limits
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is HALF_OPEN with max calls reached",
                        self.state
                    )
                self.half_open_calls += 1
        
        # Execute the function with timeout
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(execution_time)
            
            return result
            
        except (asyncio.TimeoutError, *self.config.expected_exception) as e:
            # Record failure
            execution_time = time.time() - start_time
            if isinstance(e, asyncio.TimeoutError):
                self.metrics.timeouts += 1
            self._record_failure(e, execution_time)
            
            # If we have a fallback, try it
            if self.fallback_function and self.state == CircuitBreakerState.OPEN:
                try:
                    logger.warning(f"Circuit breaker '{self.name}' using fallback for: {e}")
                    if asyncio.iscoroutinefunction(self.fallback_function):
                        return await self.fallback_function(*args, **kwargs)
                    else:
                        return self.fallback_function(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"Fallback failed for '{self.name}': {fallback_error}")
            
            raise
    
    def _record_success(self, execution_time: float) -> None:
        """Record successful execution"""
        with self.lock:
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.utcnow()
            
            # Update sliding window
            self._update_sliding_window(True, execution_time)
            
            # Handle state transitions
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success if configured
                if self.config.reset_timeout_on_success:
                    self.failure_count = 0
    
    def _record_failure(self, exception: Exception, execution_time: float) -> None:
        """Record failed execution"""
        with self.lock:
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.utcnow()
            self.last_exception = exception
            self.last_failure_time = time.time()
            
            # Update sliding window
            self._update_sliding_window(False, execution_time)
            
            # Handle state transitions
            if self.state == CircuitBreakerState.CLOSED:
                self.failure_count += 1
                
                # Check if we should open the circuit
                if (self.failure_count >= self.config.failure_threshold or
                    self._get_failure_rate() >= self.config.failure_rate_threshold):
                    self._transition_to_open()
                    
            elif self.state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to_open()
    
    def _update_sliding_window(self, success: bool, execution_time: float) -> None:
        """Update sliding window with recent call result"""
        current_time = time.time()
        
        # Add current call
        self.recent_calls.append({
            'timestamp': current_time,
            'success': success,
            'execution_time': execution_time
        })
        
        # Remove old calls outside the sliding window
        window_start = current_time - 60  # 1 minute window
        self.recent_calls = [
            call for call in self.recent_calls 
            if call['timestamp'] > window_start
        ]
        
        # Keep only the most recent calls within size limit
        if len(self.recent_calls) > self.config.sliding_window_size:
            self.recent_calls = self.recent_calls[-self.config.sliding_window_size:]
    
    def _get_failure_rate(self) -> float:
        """Calculate current failure rate from sliding window"""
        if not self.recent_calls:
            return 0.0
        
        total_calls = len(self.recent_calls)
        failed_calls = sum(1 for call in self.recent_calls if not call['success'])
        
        return failed_calls / total_calls
    
    def _transition_to_open(self) -> None:
        """Transition circuit breaker to OPEN state"""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.next_attempt_time = time.time() + self.config.recovery_timeout
        self.half_open_calls = 0
        self.success_count = 0
        self.metrics.circuit_open_count += 1
        
        self.metrics.add_state_change(old_state, self.state)
        logger.warning(
            f"Circuit breaker '{self.name}' opened due to {self.failure_count} failures. "
            f"Next attempt in {self.config.recovery_timeout}s"
        )
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit breaker to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        
        self.metrics.add_state_change(old_state, self.state)
        logger.info(f"Circuit breaker '{self.name}' transitioning to HALF_OPEN for testing")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit breaker to CLOSED state"""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        
        self.metrics.add_state_change(old_state, self.state)
        logger.info(f"Circuit breaker '{self.name}' closed - service recovered")
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        with self.lock:
            return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'failure_rate': self._get_failure_rate(),
                'next_attempt_time': self.next_attempt_time if self.state == CircuitBreakerState.OPEN else None,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'timeout': self.config.timeout
                },
                'metrics': {
                    'total_requests': self.metrics.total_requests,
                    'successful_requests': self.metrics.successful_requests,
                    'failed_requests': self.metrics.failed_requests,
                    'timeouts': self.metrics.timeouts,
                    'circuit_open_count': self.metrics.circuit_open_count,
                    'success_rate': self.metrics.get_success_rate(),
                    'failure_rate': self.metrics.get_failure_rate(),
                    'last_failure_time': self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                    'last_success_time': self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
                    'recent_state_changes': self.metrics.state_changes[-5:]  # Last 5 changes
                }
            }
    
    def force_open(self) -> None:
        """Manually force circuit breaker to open state"""
        with self.lock:
            old_state = self.state
            self._transition_to_open()
            logger.warning(f"Circuit breaker '{self.name}' manually forced to OPEN from {old_state.value}")
    
    def force_closed(self) -> None:
        """Manually force circuit breaker to closed state"""
        with self.lock:
            old_state = self.state
            self._transition_to_closed()
            logger.info(f"Circuit breaker '{self.name}' manually forced to CLOSED from {old_state.value}")
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state"""
        with self.lock:
            old_state = self.state
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.recent_calls.clear()
            self.last_failure_time = None
            self.next_attempt_time = 0
            
            self.metrics.add_state_change(old_state, self.state)
            logger.info(f"Circuit breaker '{self.name}' reset to CLOSED state")


class ExponentialBackoff:
    """
    Exponential backoff implementation for retry logic.
    
    Features:
    - Configurable base delay and max delay
    - Jitter to prevent thundering herd
    - Maximum retry attempts
    """
    
    def __init__(self,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 max_attempts: int = 5,
                 jitter: bool = True):
        """
        Initialize exponential backoff.
        
        Args:
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            max_attempts: Maximum retry attempts
            jitter: Add random jitter to prevent thundering herd
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts
        self.jitter = jitter
        self.attempt_count = 0
    
    def get_delay(self) -> float:
        """Get delay for current attempt"""
        if self.attempt_count >= self.max_attempts:
            raise Exception(f"Maximum retry attempts ({self.max_attempts}) exceeded")
        
        # Calculate exponential delay
        delay = min(self.base_delay * (2 ** self.attempt_count), self.max_delay)
        
        # Add jitter if enabled
        if self.jitter:
            import random
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        self.attempt_count += 1
        return max(0, delay)
    
    def reset(self) -> None:
        """Reset attempt counter"""
        self.attempt_count = 0
    
    def should_retry(self) -> bool:
        """Check if should retry based on attempt count"""
        return self.attempt_count < self.max_attempts


def circuit_breaker(name: str, 
                   config: Optional[CircuitBreakerConfig] = None,
                   fallback: Optional[Callable] = None):
    """
    Decorator factory for circuit breaker.
    
    Args:
        name: Circuit breaker name
        config: Optional configuration (uses defaults if not provided)
        fallback: Optional fallback function
    
    Returns:
        Circuit breaker decorator
    """
    if config is None:
        config = CircuitBreakerConfig()
    
    breaker = CircuitBreaker(name, config, fallback)
    return breaker


def with_retry(max_attempts: int = 3,
               base_delay: float = 1.0,
               max_delay: float = 60.0,
               exceptions: tuple = (Exception,)):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exceptions to retry on
    
    Returns:
        Retry decorator
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                backoff = ExponentialBackoff(base_delay, max_delay, max_attempts)
                last_exception = None
                
                while backoff.should_retry():
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if backoff.should_retry():
                            delay = backoff.get_delay()
                            logger.warning(f"Retrying {func.__name__} in {delay:.2f}s after: {e}")
                            await asyncio.sleep(delay)
                        else:
                            break
                
                # All retries exhausted
                logger.error(f"All retry attempts exhausted for {func.__name__}")
                raise last_exception
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                backoff = ExponentialBackoff(base_delay, max_delay, max_attempts)
                last_exception = None
                
                while backoff.should_retry():
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if backoff.should_retry():
                            delay = backoff.get_delay()
                            logger.warning(f"Retrying {func.__name__} in {delay:.2f}s after: {e}")
                            time.sleep(delay)
                        else:
                            break
                
                # All retries exhausted
                logger.error(f"All retry attempts exhausted for {func.__name__}")
                raise last_exception
            
            return sync_wrapper
    
    return decorator


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get circuit breaker by name"""
    return _circuit_breakers.get(name)


def register_circuit_breaker(breaker: CircuitBreaker) -> None:
    """Register circuit breaker in global registry"""
    _circuit_breakers[breaker.name] = breaker
    logger.info(f"Registered circuit breaker: {breaker.name}")


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers"""
    return _circuit_breakers.copy()


def get_circuit_breaker_summary() -> Dict[str, Any]:
    """Get summary of all circuit breakers"""
    summary = {
        'total_breakers': len(_circuit_breakers),
        'states': {},
        'breakers': {}
    }
    
    for name, breaker in _circuit_breakers.items():
        state = breaker.get_state()
        if state.value not in summary['states']:
            summary['states'][state.value] = 0
        summary['states'][state.value] += 1
        
        summary['breakers'][name] = breaker.get_metrics()
    
    return summary