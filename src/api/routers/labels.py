"""
Label computation and query endpoints

Implements all endpoints related to label computation and retrieval:
- /labels/compute - Real-time label computation
- /labels - Query computed labels
- /labels/{instrument_id}/{granularity}/{timestamp} - Get specific labels
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

from fastapi import APIRouter, HTTPException, Request, Query, Path, Header, status, Depends
from fastapi.responses import JSONResponse

from src.api.schemas import (
    CandleLabelRequest, ComputedLabels, LabelsList, PaginationInfo,
    CacheStrategyEnum, GranularityEnum, ErrorResponse
)
from src.core.label_computation import computation_engine
from src.services.clickhouse_service import clickhouse_service
from src.services.redis_cache import redis_cache
from src.models.data_models import Candle, Granularity
from src.utils.timestamp_aligner import TimestampAligner

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/labels/compute",
    response_model=ComputedLabels,
    status_code=200,
    summary="Compute labels for a single candle",
    description="""
    Compute all configured labels for a single candle in real-time.
    Target latency: <100ms p99.
    
    For path-dependent labels, uses lower granularity data:
    - H4 → H1 data for horizon checks
    - D → H4 data for horizon checks  
    - W → D data for horizon checks
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request - invalid parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        422: {"model": ErrorResponse, "description": "Validation error - invalid input data"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def compute_labels(
    request: CandleLabelRequest,
    req: Request,
    x_request_id: Optional[str] = Header(None, description="Unique request identifier for tracing"),
    x_cache_strategy: CacheStrategyEnum = Header(
        CacheStrategyEnum.PREFER_CACHE,
        description="Cache strategy hint"
    )
):
    """
    Compute labels for a single candle with configurable options.
    
    This is the primary endpoint for real-time label computation with
    sub-100ms target latency for incremental processing.
    """
    start_time = time.time()
    trace_id = x_request_id or getattr(req.state, "trace_id", str(uuid.uuid4()))
    
    try:
        # Validate timestamp alignment
        if not TimestampAligner.validate_alignment(
            request.candle.ts, request.granularity.value
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "INVALID_TIMESTAMP_ALIGNMENT",
                        "message": f"Timestamp {request.candle.ts} is not aligned to {request.granularity.value} granularity",
                        "trace_id": trace_id
                    }
                }
            )
        
        # Create Candle object
        candle = Candle(
            instrument_id=request.instrument_id,
            granularity=request.granularity,
            ts=request.candle.ts,
            open=request.candle.open,
            high=request.candle.high,
            low=request.candle.low,
            close=request.candle.close,
            volume=request.candle.volume,
            atr_14=request.candle.atr_14
        )
        
        # Determine cache behavior
        use_cache = x_cache_strategy != CacheStrategyEnum.BYPASS_CACHE
        force_recompute = x_cache_strategy == CacheStrategyEnum.REFRESH_CACHE
        
        # Override with request options if provided
        if request.options:
            use_cache = request.options.use_cache if not force_recompute else False
            force_recompute = request.options.force_recompute or force_recompute
        
        # Compute labels
        label_set = await computation_engine.compute_labels(
            candle=candle,
            horizon_periods=request.options.horizon_periods if request.options else 6,
            label_types=request.label_types,
            use_cache=use_cache,
            force_recompute=force_recompute
        )
        
        # Calculate response metrics
        computation_time_ms = int((time.time() - start_time) * 1000)
        cache_hit = use_cache and not force_recompute and label_set.computation_time_ms < 10
        
        # Convert to response model
        response_data = ComputedLabels(
            instrument_id=label_set.instrument_id,
            granularity=label_set.granularity,
            ts=label_set.ts,
            labels={
                "enhanced_triple_barrier": label_set.enhanced_triple_barrier.dict() if label_set.enhanced_triple_barrier else None,
                "vol_scaled_return": {
                    "value": label_set.vol_scaled_return,
                    "quantile": 0.5,  # Would be calculated properly in production
                } if label_set.vol_scaled_return is not None else None,
                "mfe_mae": {
                    "mfe": label_set.mfe,
                    "mae": label_set.mae,
                    "profit_factor": label_set.profit_factor
                } if label_set.mfe is not None and label_set.mae is not None else None,
                "forward_return": label_set.forward_return
            },
            computation_time_ms=computation_time_ms,
            cache_hit=cache_hit,
            version=label_set.label_version
        )
        
        # Update metrics
        redis_cache.increment_metric(f"labels_computed:{request.instrument_id}:{request.granularity.value}")
        if cache_hit:
            redis_cache.increment_metric("cache_hits_labels")
        else:
            redis_cache.increment_metric("cache_misses_labels")
        
        # Add response headers
        headers = {
            "X-Compute-Time-Ms": str(computation_time_ms),
            "X-Cache-Hit": str(cache_hit).lower(),
            "X-Request-ID": trace_id
        }
        
        return JSONResponse(
            content=response_data.dict(),
            headers=headers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Label computation failed: {str(e)}",
            extra={"trace_id": trace_id, "instrument": request.instrument_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "COMPUTATION_FAILED",
                    "message": "Label computation failed",
                    "trace_id": trace_id
                }
            }
        )


@router.get(
    "/labels",
    response_model=LabelsList,
    summary="Query computed labels",
    description="""
    Retrieve computed labels with flexible filtering and pagination.
    Optimized for time-series queries with proper indexing.
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request - invalid parameters"},
        422: {"model": ErrorResponse, "description": "Validation error"}
    }
)
async def get_labels(
    request: Request,
    instrument_id: str = Query(..., description="Instrument identifier", example="EURUSD"),
    granularity: GranularityEnum = Query(..., description="Time granularity", example="H4"),
    start_date: datetime = Query(..., description="Start date (inclusive)"),
    end_date: datetime = Query(..., description="End date (exclusive)"),
    label_types: Optional[str] = Query(
        None,
        description="Comma-separated list of label types to include",
        example="enhanced_triple_barrier,vol_scaled_return"
    ),
    enhanced_triple_barrier_label: Optional[int] = Query(
        None,
        description="Filter by enhanced triple barrier outcome",
        ge=-1, le=1
    ),
    return_quantile_min: Optional[int] = Query(
        None,
        description="Minimum return quantile (0-100)",
        ge=0, le=100
    ),
    return_quantile_max: Optional[int] = Query(
        None,
        description="Maximum return quantile (0-100)",
        ge=0, le=100
    ),
    page: int = Query(1, description="Page number (1-based)", ge=1),
    per_page: int = Query(100, description="Items per page", ge=1, le=1000),
    sort: str = Query(
        "ts_asc",
        description="Sort field and direction",
        regex="^(ts_asc|ts_desc|forward_return_asc|forward_return_desc)$"
    )
):
    """
    Query computed labels with filtering and pagination.
    
    Supports complex filtering by label values and time ranges
    with efficient ClickHouse queries.
    """
    start_time = time.time()
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    
    try:
        # Validate date range
        if end_date <= start_date:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "INVALID_DATE_RANGE",
                        "message": "end_date must be after start_date",
                        "trace_id": trace_id
                    }
                }
            )
        
        # Parse label types
        requested_label_types = []
        if label_types:
            requested_label_types = [t.strip() for t in label_types.split(",")]
        
        # Build ClickHouse query
        query_parts = [
            "SELECT instrument_id, granularity, ts,",
            "  enhanced_triple_barrier_label,",
            "  enhanced_triple_barrier_barrier_hit,",
            "  enhanced_triple_barrier_time_to_barrier,",
            "  vol_scaled_return,",
            "  forward_return,",
            "  mfe, mae, profit_factor",
            "FROM quantx.labels",
            "WHERE instrument_id = %(instrument_id)s",
            "  AND granularity = %(granularity)s",
            "  AND ts >= %(start_date)s",
            "  AND ts < %(end_date)s"
        ]
        
        query_params = {
            "instrument_id": instrument_id,
            "granularity": granularity.value,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        
        # Add filters
        if enhanced_triple_barrier_label is not None:
            query_parts.append("  AND enhanced_triple_barrier_label = %(etb_label)s")
            query_params["etb_label"] = enhanced_triple_barrier_label
        
        if return_quantile_min is not None:
            query_parts.append("  AND return_quantile >= %(rq_min)s")
            query_params["rq_min"] = return_quantile_min
        
        if return_quantile_max is not None:
            query_parts.append("  AND return_quantile <= %(rq_max)s")
            query_params["rq_max"] = return_quantile_max
        
        # Add sorting
        sort_mapping = {
            "ts_asc": "ts ASC",
            "ts_desc": "ts DESC",
            "forward_return_asc": "forward_return ASC",
            "forward_return_desc": "forward_return DESC"
        }
        query_parts.append(f"ORDER BY {sort_mapping[sort]}")
        
        # Add pagination
        offset = (page - 1) * per_page
        query_parts.extend([
            f"LIMIT {per_page}",
            f"OFFSET {offset}"
        ])
        
        query = " ".join(query_parts)
        
        # Execute query
        results = clickhouse_service.execute(query, query_params)
        
        # Get total count for pagination
        count_query = """
        SELECT count() as total
        FROM quantx.labels
        WHERE instrument_id = %(instrument_id)s
          AND granularity = %(granularity)s
          AND ts >= %(start_date)s
          AND ts < %(end_date)s
        """
        
        count_result = clickhouse_service.execute(count_query, query_params)
        total_count = count_result[0]["total"] if count_result else 0
        
        # Convert results to response format
        labels_data = []
        for row in results:
            # Build label values
            label_values = {}
            
            if row.get("enhanced_triple_barrier_label") is not None:
                label_values["enhanced_triple_barrier"] = {
                    "label": row["enhanced_triple_barrier_label"],
                    "barrier_hit": row.get("enhanced_triple_barrier_barrier_hit", "none"),
                    "time_to_barrier": row.get("enhanced_triple_barrier_time_to_barrier", 0),
                    "level_adjusted": False  # Would come from DB in production
                }
            
            if row.get("vol_scaled_return") is not None:
                label_values["vol_scaled_return"] = {
                    "value": row["vol_scaled_return"],
                    "quantile": 0.5  # Would be calculated properly
                }
            
            if row.get("mfe") is not None and row.get("mae") is not None:
                label_values["mfe_mae"] = {
                    "mfe": row["mfe"],
                    "mae": row["mae"],
                    "profit_factor": row.get("profit_factor", 0)
                }
            
            if row.get("forward_return") is not None:
                label_values["forward_return"] = row["forward_return"]
            
            labels_data.append(ComputedLabels(
                instrument_id=row["instrument_id"],
                granularity=GranularityEnum(row["granularity"]),
                ts=row["ts"],
                labels=label_values,
                computation_time_ms=0,  # Historical data
                cache_hit=False,
                version="1.0.0"
            ))
        
        # Build pagination info
        total_pages = (total_count + per_page - 1) // per_page
        pagination = PaginationInfo(
            page=page,
            per_page=per_page,
            total=total_count,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
            next_page=page + 1 if page < total_pages else None,
            prev_page=page - 1 if page > 1 else None
        )
        
        # Calculate query time
        query_time_ms = int((time.time() - start_time) * 1000)
        
        response = LabelsList(
            data=labels_data,
            pagination=pagination
        )
        
        return JSONResponse(
            content=response.dict(),
            headers={
                "X-Query-Time-Ms": str(query_time_ms),
                "X-Request-ID": trace_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Label query failed: {str(e)}",
            extra={"trace_id": trace_id, "instrument": instrument_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "QUERY_FAILED",
                    "message": "Label query failed",
                    "trace_id": trace_id
                }
            }
        )


@router.get(
    "/labels/{instrument_id}/{granularity}/{timestamp}",
    response_model=ComputedLabels,
    summary="Get labels for specific candle",
    description="Retrieve all labels for a specific candle timestamp",
    responses={
        404: {"model": ErrorResponse, "description": "Labels not found"}
    }
)
async def get_labels_by_timestamp(
    request: Request,
    instrument_id: str = Path(..., description="Instrument identifier", example="EURUSD"),
    granularity: GranularityEnum = Path(..., description="Time granularity", example="H4"),
    timestamp: datetime = Path(..., description="Candle timestamp (ISO 8601)"),
    label_types: Optional[str] = Query(
        None,
        description="Comma-separated list of label types to include",
        example="enhanced_triple_barrier,vol_scaled_return"
    )
):
    """
    Get labels for a specific candle timestamp.
    
    Returns cached labels if available, otherwise queries ClickHouse.
    """
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    
    try:
        # Check cache first
        cached_labels = redis_cache.get_labels(instrument_id, granularity.value, timestamp)
        if cached_labels:
            # Filter requested label types if specified
            if label_types:
                requested_types = [t.strip() for t in label_types.split(",")]
                filtered_labels = {
                    k: v for k, v in cached_labels.get("labels", {}).items()
                    if k in requested_types
                }
                cached_labels["labels"] = filtered_labels
            
            return ComputedLabels(**cached_labels)
        
        # Query ClickHouse
        query = """
        SELECT instrument_id, granularity, ts,
          enhanced_triple_barrier_label,
          enhanced_triple_barrier_barrier_hit,
          vol_scaled_return,
          forward_return,
          mfe, mae, profit_factor
        FROM quantx.labels
        WHERE instrument_id = %(instrument_id)s
          AND granularity = %(granularity)s
          AND ts = %(timestamp)s
        LIMIT 1
        """
        
        params = {
            "instrument_id": instrument_id,
            "granularity": granularity.value,
            "timestamp": timestamp.isoformat()
        }
        
        results = clickhouse_service.execute(query, params)
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "Labels not found for specified parameters",
                        "trace_id": trace_id
                    }
                }
            )
        
        row = results[0]
        
        # Build response
        label_values = {}
        
        if row.get("enhanced_triple_barrier_label") is not None:
            label_values["enhanced_triple_barrier"] = {
                "label": row["enhanced_triple_barrier_label"],
                "barrier_hit": row.get("enhanced_triple_barrier_barrier_hit", "none"),
                "time_to_barrier": 0,  # Would come from DB
                "level_adjusted": False
            }
        
        if row.get("vol_scaled_return") is not None:
            label_values["vol_scaled_return"] = {
                "value": row["vol_scaled_return"],
                "quantile": 0.5
            }
        
        if row.get("mfe") is not None and row.get("mae") is not None:
            label_values["mfe_mae"] = {
                "mfe": row["mfe"],
                "mae": row["mae"],
                "profit_factor": row.get("profit_factor", 0)
            }
        
        if row.get("forward_return") is not None:
            label_values["forward_return"] = row["forward_return"]
        
        # Filter by requested label types
        if label_types:
            requested_types = [t.strip() for t in label_types.split(",")]
            label_values = {k: v for k, v in label_values.items() if k in requested_types}
        
        return ComputedLabels(
            instrument_id=row["instrument_id"],
            granularity=GranularityEnum(row["granularity"]),
            ts=row["ts"],
            labels=label_values,
            computation_time_ms=0,
            cache_hit=False,
            version="1.0.0"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Get labels by timestamp failed: {str(e)}",
            extra={"trace_id": trace_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "QUERY_FAILED",
                    "message": "Failed to retrieve labels",
                    "trace_id": trace_id
                }
            }
        )