"""
Middleware for FastAPI application

Provides authentication, rate limiting, request tracing, and error handling.
"""

import time
import uuid
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import jwt

from config.settings import settings
from src.services.redis_cache import redis_cache

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware using JWT tokens"""
    
    # Endpoints that don't require authentication
    PUBLIC_ENDPOINTS = {
        "/", "/docs", "/redoc", "/openapi.json",
        "/v1/health", "/v1/health/ready", "/v1/health/live",
        "/metrics"
    }
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public endpoints
        if request.url.path in self.PUBLIC_ENDPOINTS:
            return await call_next(request)
        
        # Check for API key in headers
        api_key = request.headers.get("X-API-Key")
        if api_key:
            # Validate API key (implement your logic here)
            if self._validate_api_key(api_key):
                request.state.authenticated = True
                request.state.auth_type = "api_key"
                return await call_next(request)
        
        # Check for JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            user_data = self._validate_jwt_token(token)
            if user_data:
                request.state.authenticated = True
                request.state.auth_type = "jwt"
                request.state.user_data = user_data
                return await call_next(request)
        
        # Return unauthorized response
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "code": "UNAUTHORIZED",
                    "message": "Authentication required",
                    "trace_id": getattr(request.state, "trace_id", str(uuid.uuid4()))
                }
            },
            headers={"WWW-Authenticate": 'Bearer realm="api"'}
        )
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key (implement your validation logic)"""
        # For demo purposes, accept any key starting with 'label_'
        return api_key.startswith("label_")
    
    def _validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user data"""
        try:
            # For demo purposes, skip JWT validation
            # In production, use proper JWT secret and validation
            payload = jwt.decode(
                token, 
                "your-secret-key",  # Use proper secret from settings
                algorithms=["HS256"],
                options={"verify_signature": False}  # Remove in production
            )
            return payload
        except jwt.InvalidTokenError:
            return None


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using Redis"""
    
    # Rate limits per endpoint pattern
    RATE_LIMITS = {
        "/v1/labels/compute": {"requests": 1000, "window": 3600},  # 1000/hour
        "/v1/batch/backfill": {"requests": 10, "window": 3600},    # 10/hour
        "/v1/labels": {"requests": 10000, "window": 3600},         # 10000/hour
        "default": {"requests": 5000, "window": 3600}              # 5000/hour default
    }
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path.startswith("/v1/health"):
            return await call_next(request)
        
        # Get client identifier (IP or authenticated user)
        client_id = self._get_client_id(request)
        
        # Get rate limit for endpoint
        endpoint_pattern = self._get_endpoint_pattern(request.url.path)
        rate_limit = self.RATE_LIMITS.get(endpoint_pattern, self.RATE_LIMITS["default"])
        
        # Check rate limit
        if not await self._check_rate_limit(client_id, endpoint_pattern, rate_limit):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests",
                        "details": [{"message": f"Rate limit of {rate_limit['requests']} requests per hour exceeded"}],
                        "trace_id": getattr(request.state, "trace_id", str(uuid.uuid4()))
                    }
                },
                headers={
                    "X-RateLimit-Limit": str(rate_limit["requests"]),
                    "X-RateLimit-Window": str(rate_limit["window"]),
                    "Retry-After": "3600"
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers to response
        remaining = await self._get_remaining_requests(client_id, endpoint_pattern, rate_limit)
        response.headers["X-RateLimit-Limit"] = str(rate_limit["requests"])
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + rate_limit["window"])
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Use authenticated user ID if available
        if hasattr(request.state, "user_data") and "user_id" in request.state.user_data:
            return f"user:{request.state.user_data['user_id']}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"
    
    def _get_endpoint_pattern(self, path: str) -> str:
        """Get rate limit pattern for endpoint"""
        # Simple pattern matching - could be enhanced with regex
        for pattern in self.RATE_LIMITS:
            if pattern != "default" and path.startswith(pattern):
                return pattern
        return "default"
    
    async def _check_rate_limit(self, client_id: str, endpoint: str, rate_limit: Dict) -> bool:
        """Check if request is within rate limit"""
        key = f"rate_limit:{client_id}:{endpoint}"
        current_count = redis_cache.get_metric(key)
        
        if current_count >= rate_limit["requests"]:
            return False
        
        # Increment counter with TTL
        redis_cache.client.incr(key)
        redis_cache.client.expire(key, rate_limit["window"])
        return True
    
    async def _get_remaining_requests(self, client_id: str, endpoint: str, rate_limit: Dict) -> int:
        """Get remaining requests for client"""
        key = f"rate_limit:{client_id}:{endpoint}"
        current_count = redis_cache.get_metric(key)
        return max(0, rate_limit["requests"] - current_count)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """Request tracing middleware for logging and monitoring"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate or extract trace ID
        trace_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.trace_id = trace_id
        
        # Log request start
        start_time = time.time()
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_host": request.client.host if request.client else None
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log successful request
            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Request completed: {request.method} {request.url.path} - "
                f"{response.status_code} - {duration_ms}ms",
                extra={
                    "trace_id": trace_id,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms
                }
            )
            
            return response
            
        except Exception as e:
            # Log failed request
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)} - {duration_ms}ms",
                extra={
                    "trace_id": trace_id,
                    "error": str(e),
                    "duration_ms": duration_ms
                },
                exc_info=True
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException:
            # Re-raise HTTP exceptions to be handled by FastAPI
            raise
        except Exception as e:
            # Handle unexpected errors
            trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
            
            logger.error(
                f"Unexpected error in {request.method} {request.url.path}: {str(e)}",
                extra={"trace_id": trace_id},
                exc_info=True
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "An internal error occurred",
                        "trace_id": trace_id
                    }
                },
                headers={"X-Request-ID": trace_id}
            )