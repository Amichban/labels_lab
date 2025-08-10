"""
Pydantic models for Label Computation System API
Generated from OpenAPI 3.0 specification

These models provide automatic validation, serialization, and documentation
for all API request and response schemas.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum

from pydantic import BaseModel, Field, validator, EmailStr
from pydantic.types import conint, confloat, constr


# ===========================================================================
# ENUMS
# ===========================================================================

class GranularityEnum(str, Enum):
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D = "D"
    W = "W"


class JobStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class HealthStatusEnum(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class BarrierHitEnum(str, Enum):
    UPPER = "upper"
    LOWER = "lower"
    NONE = "none"


class CacheStrategyEnum(str, Enum):
    PREFER_CACHE = "prefer_cache"
    BYPASS_CACHE = "bypass_cache"
    REFRESH_CACHE = "refresh_cache"


class PriorityEnum(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


# ===========================================================================
# REQUEST MODELS
# ===========================================================================

class CandleData(BaseModel):
    """OHLCV candle data with technical indicators"""
    
    ts: datetime = Field(
        ..., 
        description="Candle timestamp (aligned to granularity boundary)",
        example="2024-01-10T13:00:00Z"
    )
    open: confloat(gt=0) = Field(
        ..., 
        description="Opening price"
    )
    high: confloat(gt=0) = Field(
        ..., 
        description="Highest price"
    )
    low: confloat(gt=0) = Field(
        ..., 
        description="Lowest price"
    )
    close: confloat(gt=0) = Field(
        ..., 
        description="Closing price"
    )
    volume: conint(ge=0) = Field(
        ..., 
        description="Volume"
    )
    atr_14: Optional[confloat(ge=0)] = Field(
        None,
        description="14-period Average True Range (required for some labels)"
    )

    @validator('high')
    def high_gte_low(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('high must be >= low')
        return v

    @validator('high')
    def high_gte_open_close(cls, v, values):
        if 'open' in values and v < values['open']:
            raise ValueError('high must be >= open')
        if 'close' in values and v < values['close']:
            raise ValueError('high must be >= close')
        return v

    @validator('low')
    def low_lte_open_close(cls, v, values):
        if 'open' in values and v > values['open']:
            raise ValueError('low must be <= open')
        if 'close' in values and v > values['close']:
            raise ValueError('low must be <= close')
        return v

    class Config:
        schema_extra = {
            "example": {
                "ts": "2024-01-10T13:00:00Z",
                "open": 1.0950,
                "high": 1.0970,
                "low": 1.0940,
                "close": 1.0965,
                "volume": 1250000,
                "atr_14": 0.0025
            }
        }


class LabelComputeOptions(BaseModel):
    """Options for label computation"""
    
    horizon_periods: conint(ge=1, le=100) = Field(
        6,
        description="Forward-looking horizon in periods"
    )
    use_cache: bool = Field(
        True,
        description="Whether to use cached results"
    )
    force_recompute: bool = Field(
        False,
        description="Force recomputation even if cached"
    )


class CandleLabelRequest(BaseModel):
    """Request to compute labels for a single candle"""
    
    instrument_id: constr(regex=r'^[A-Z]{6}$|^[A-Z0-9]+$') = Field(
        ...,
        description="Instrument identifier (e.g., EURUSD, GBPJPY)",
        example="EURUSD"
    )
    granularity: GranularityEnum = Field(
        ...,
        description="Time granularity for labels"
    )
    candle: CandleData = Field(
        ...,
        description="Candle data for label computation"
    )
    label_types: Optional[List[str]] = Field(
        None,
        description="Specific label types to compute (optional, computes all if empty)",
        example=["enhanced_triple_barrier", "vol_scaled_return"]
    )
    options: Optional[LabelComputeOptions] = Field(
        LabelComputeOptions(),
        description="Computation options"
    )

    class Config:
        schema_extra = {
            "example": {
                "instrument_id": "EURUSD",
                "granularity": "H4",
                "candle": {
                    "ts": "2024-01-10T13:00:00Z",
                    "open": 1.0950,
                    "high": 1.0970,
                    "low": 1.0940,
                    "close": 1.0965,
                    "volume": 1250000,
                    "atr_14": 0.0025
                },
                "label_types": ["enhanced_triple_barrier", "vol_scaled_return", "mfe_mae"],
                "options": {
                    "horizon_periods": 6,
                    "use_cache": True,
                    "force_recompute": False
                }
            }
        }


class BatchBackfillOptions(BaseModel):
    """Options for batch backfill operation"""
    
    chunk_size: conint(ge=1000, le=50000) = Field(
        10000,
        description="Candles to process per chunk"
    )
    parallel_workers: conint(ge=1, le=16) = Field(
        8,
        description="Number of parallel workers"
    )
    force_recompute: bool = Field(
        False,
        description="Recompute existing labels"
    )
    priority: PriorityEnum = Field(
        PriorityEnum.NORMAL,
        description="Job priority"
    )


class BatchBackfillRequest(BaseModel):
    """Request to start a batch backfill operation"""
    
    instrument_id: constr(regex=r'^[A-Z]{6}$|^[A-Z0-9]+$') = Field(
        ...,
        example="EURUSD"
    )
    granularity: GranularityEnum = Field(...)
    start_date: datetime = Field(
        ...,
        description="Start date (inclusive)"
    )
    end_date: datetime = Field(
        ...,
        description="End date (exclusive)"
    )
    label_types: Optional[List[str]] = Field(
        None,
        description="Label types to compute (all if empty)"
    )
    options: Optional[BatchBackfillOptions] = Field(
        BatchBackfillOptions(),
        description="Backfill options"
    )

    @validator('end_date')
    def end_after_start(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

    class Config:
        schema_extra = {
            "example": {
                "instrument_id": "EURUSD",
                "granularity": "H4",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-31T23:59:59Z",
                "label_types": ["enhanced_triple_barrier", "vol_scaled_return", "return_quantile"],
                "options": {
                    "chunk_size": 10000,
                    "parallel_workers": 8,
                    "force_recompute": False,
                    "priority": "normal"
                }
            }
        }


# ===========================================================================
# RESPONSE MODELS
# ===========================================================================

class EnhancedTripleBarrierLabel(BaseModel):
    """Enhanced triple barrier label with S/R level adjustment"""
    
    label: conint(ge=-1, le=1) = Field(
        ...,
        description="-1=lower barrier hit, 0=no barrier hit, 1=upper barrier hit"
    )
    barrier_hit: BarrierHitEnum = Field(
        ...,
        description="Which barrier was hit first"
    )
    time_to_barrier: conint(ge=1) = Field(
        ...,
        description="Periods until barrier hit (or horizon if none)"
    )
    barrier_price: Optional[float] = Field(
        None,
        description="Price of barrier that was hit"
    )
    level_adjusted: bool = Field(
        ...,
        description="Whether barriers were adjusted based on S/R levels"
    )
    nearest_support: Optional[float] = Field(
        None,
        description="Nearest support level price"
    )
    nearest_resistance: Optional[float] = Field(
        None,
        description="Nearest resistance level price"
    )


class VolScaledReturnLabel(BaseModel):
    """Volatility-scaled return label"""
    
    value: float = Field(
        ...,
        description="Volatility-scaled return value"
    )
    quantile: confloat(ge=0, le=1) = Field(
        ...,
        description="Quantile of the scaled return (0-1)"
    )
    raw_return: Optional[float] = Field(
        None,
        description="Raw forward return before scaling"
    )
    volatility_factor: Optional[confloat(ge=0)] = Field(
        None,
        description="ATR-based volatility scaling factor"
    )


class MfeMaeLabel(BaseModel):
    """Maximum Favorable/Adverse Excursion label"""
    
    mfe: float = Field(
        ...,
        description="Maximum Favorable Excursion"
    )
    mae: float = Field(
        ...,
        description="Maximum Adverse Excursion"
    )
    profit_factor: confloat(ge=0) = Field(
        ...,
        description="Ratio of MFE to MAE"
    )
    mfe_time: Optional[conint(ge=1)] = Field(
        None,
        description="Periods to reach MFE"
    )
    mae_time: Optional[conint(ge=1)] = Field(
        None,
        description="Periods to reach MAE"
    )


class LabelValues(BaseModel):
    """Container for all computed label values"""
    
    enhanced_triple_barrier: Optional[EnhancedTripleBarrierLabel] = None
    vol_scaled_return: Optional[VolScaledReturnLabel] = None
    mfe_mae: Optional[MfeMaeLabel] = None
    return_quantile: Optional[conint(ge=0, le=100)] = Field(
        None,
        description="Return quantile bucket (0-100)"
    )
    forward_return: Optional[float] = Field(
        None,
        description="Forward return over horizon"
    )

    class Config:
        extra = "allow"  # Allow additional label types


class ComputedLabels(BaseModel):
    """Response containing computed labels for a candle"""
    
    instrument_id: str
    granularity: GranularityEnum
    ts: datetime
    labels: LabelValues = Field(
        ...,
        description="Computed label values"
    )
    computation_time_ms: conint(ge=0) = Field(
        ...,
        description="Time taken to compute labels"
    )
    cache_hit: bool = Field(
        ...,
        description="Whether result came from cache"
    )
    version: str = Field(
        ...,
        description="Label computation version",
        example="1.0.0"
    )

    class Config:
        schema_extra = {
            "example": {
                "instrument_id": "EURUSD",
                "granularity": "H4",
                "ts": "2024-01-10T13:00:00Z",
                "labels": {
                    "enhanced_triple_barrier": {
                        "label": 1,
                        "barrier_hit": "upper",
                        "time_to_barrier": 3,
                        "barrier_price": 1.0990,
                        "level_adjusted": True,
                        "nearest_support": 1.0920,
                        "nearest_resistance": 1.0990
                    },
                    "vol_scaled_return": {
                        "value": 2.35,
                        "quantile": 0.85,
                        "raw_return": 0.0025,
                        "volatility_factor": 0.0025
                    },
                    "mfe_mae": {
                        "mfe": 0.0045,
                        "mae": -0.0012,
                        "profit_factor": 3.75,
                        "mfe_time": 2,
                        "mae_time": 1
                    },
                    "return_quantile": 85,
                    "forward_return": 0.0025
                },
                "computation_time_ms": 45,
                "cache_hit": False,
                "version": "1.0.0"
            }
        }


class HateoasLinks(BaseModel):
    """HATEOAS navigation links"""
    
    self: Optional[str] = Field(None, format="uri")
    status: Optional[str] = Field(None, format="uri")
    cancel: Optional[str] = Field(None, format="uri")


class BatchJobResponse(BaseModel):
    """Response for starting a batch job"""
    
    job_id: constr(regex=r'^bf_[0-9]{8}_[a-z0-9]+_[a-z0-9]+_[a-z0-9]{6}$') = Field(
        ...,
        description="Unique job identifier"
    )
    status: str = Field(
        ...,
        description="Initial job status"
    )
    estimated_duration_minutes: conint(ge=1) = Field(
        ...,
        description="Estimated completion time in minutes"
    )
    estimated_candles: conint(ge=1) = Field(
        ...,
        description="Total candles to process"
    )
    priority: PriorityEnum = Field(
        ...,
        description="Job priority"
    )
    _links: Optional[HateoasLinks] = Field(
        None,
        description="HATEOAS links for job operations",
        alias="links"
    )

    class Config:
        allow_population_by_field_name = True


class JobProgress(BaseModel):
    """Job progress information"""
    
    completed_candles: conint(ge=0)
    total_candles: conint(ge=1)
    percentage: confloat(ge=0, le=100)
    current_date: Optional[datetime] = Field(
        None,
        description="Current processing date"
    )
    chunks_completed: Optional[conint(ge=0)] = None
    chunks_total: Optional[conint(ge=1)] = None


class JobPerformance(BaseModel):
    """Job performance metrics"""
    
    candles_per_minute: Optional[confloat(ge=0)] = None
    avg_compute_time_ms: Optional[confloat(ge=0)] = None
    cache_hit_rate: Optional[confloat(ge=0, le=1)] = None
    error_rate: Optional[confloat(ge=0, le=1)] = None


class BatchJobStatus(BaseModel):
    """Detailed batch job status"""
    
    job_id: constr(regex=r'^bf_[0-9]{8}_[a-z0-9]+_[a-z0-9]+_[a-z0-9]{6}$')
    status: JobStatusEnum
    progress: JobProgress = Field(
        ...,
        description="Job progress information"
    )
    performance: Optional[JobPerformance] = Field(
        None,
        description="Performance metrics"
    )
    estimated_completion: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = Field(
        None,
        description="Error message if status is failed"
    )


class PaginationInfo(BaseModel):
    """Pagination information for list responses"""
    
    page: conint(ge=1) = Field(
        ...,
        description="Current page number"
    )
    per_page: conint(ge=1) = Field(
        ...,
        description="Items per page"
    )
    total: conint(ge=0) = Field(
        ...,
        description="Total items count"
    )
    total_pages: conint(ge=0) = Field(
        ...,
        description="Total pages count"
    )
    has_next: bool = Field(
        ...,
        description="Whether next page exists"
    )
    has_prev: bool = Field(
        ...,
        description="Whether previous page exists"
    )
    next_page: Optional[int] = Field(
        None,
        description="Next page number"
    )
    prev_page: Optional[int] = Field(
        None,
        description="Previous page number"
    )


class BatchJobsList(BaseModel):
    """Paginated list of batch jobs"""
    
    data: List[BatchJobStatus]
    pagination: PaginationInfo


class LabelsList(BaseModel):
    """Paginated list of computed labels"""
    
    data: List[ComputedLabels]
    pagination: PaginationInfo


# ===========================================================================
# HEALTH & MONITORING MODELS
# ===========================================================================

class HealthMetrics(BaseModel):
    """Health metrics"""
    
    cache_hit_rate: Optional[confloat(ge=0, le=1)] = None
    avg_computation_ms: Optional[confloat(ge=0)] = None
    active_batch_jobs: Optional[conint(ge=0)] = None
    labels_computed_last_hour: Optional[conint(ge=0)] = None


class HealthResponse(BaseModel):
    """System health check response"""
    
    status: HealthStatusEnum
    version: str = Field(
        ...,
        description="Application version"
    )
    timestamp: datetime
    uptime_seconds: Optional[conint(ge=0)] = None
    checks: Optional[Dict[str, str]] = Field(
        None,
        description="Health check results for dependencies"
    )
    metrics: Optional[HealthMetrics] = None
    errors: Optional[List[str]] = Field(
        None,
        description="Error messages for failed checks"
    )


class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    
    avg_computation_time_ms: Optional[float] = None
    p50_computation_time_ms: Optional[float] = None
    p95_computation_time_ms: Optional[float] = None
    p99_computation_time_ms: Optional[float] = None
    requests_per_second: Optional[float] = None
    error_rate: Optional[confloat(ge=0, le=1)] = None


class CacheMetrics(BaseModel):
    """Cache metrics"""
    
    hit_rate: Optional[confloat(ge=0, le=1)] = None
    memory_usage_mb: Optional[float] = None
    evictions_per_minute: Optional[float] = None
    keys_total: Optional[int] = None


class BusinessMetrics(BaseModel):
    """Business metrics"""
    
    labels_computed_total: Optional[int] = None
    unique_instruments: Optional[int] = None
    active_batch_jobs: Optional[int] = None
    avg_batch_throughput_candles_per_min: Optional[float] = None


class MetricsResponse(BaseModel):
    """System metrics response"""
    
    timestamp: datetime
    window: str = Field(
        ...,
        description="Time window for metrics"
    )
    performance: Optional[PerformanceMetrics] = None
    cache: Optional[CacheMetrics] = None
    business: Optional[BusinessMetrics] = None


# ===========================================================================
# ERROR MODELS
# ===========================================================================

class ErrorDetail(BaseModel):
    """Detailed error information"""
    
    field: Optional[str] = Field(
        None,
        description="Field name with error"
    )
    message: str = Field(
        ...,
        description="Field-specific error message"
    )
    code: Optional[str] = Field(
        None,
        description="Field-specific error code"
    )


class ErrorInfo(BaseModel):
    """Error information"""
    
    code: str = Field(
        ...,
        description="Machine-readable error code",
        example="VALIDATION_ERROR"
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
        example="Invalid input data"
    )
    details: Optional[List[ErrorDetail]] = Field(
        None,
        description="Detailed validation errors"
    )
    trace_id: Optional[str] = Field(
        None,
        description="Request trace ID for debugging",
        example="req_abc123def456"
    )


class ErrorResponse(BaseModel):
    """Standard error response format"""
    
    error: ErrorInfo

    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid input data",
                    "details": [
                        {
                            "field": "candle.close",
                            "message": "Must be greater than 0",
                            "code": "MIN_VALUE"
                        },
                        {
                            "field": "granularity",
                            "message": "Must be one of: M15, H1, H4, D, W",
                            "code": "INVALID_ENUM"
                        }
                    ],
                    "trace_id": "req_abc123def456"
                }
            }
        }