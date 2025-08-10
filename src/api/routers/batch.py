"""
Batch processing endpoints

Handles large-scale backfill operations for label computation:
- /batch/backfill - Start batch backfill operation
- /batch/jobs/{job_id} - Get batch job status
- /batch/jobs - List batch jobs
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from fastapi import APIRouter, HTTPException, Request, Query, Path, status, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchBackfillRequest, BatchJobResponse, BatchJobStatus, BatchJobsList,
    PaginationInfo, JobStatusEnum, GranularityEnum, PriorityEnum, ErrorResponse
)
from src.core.label_computation import computation_engine
from src.services.redis_cache import redis_cache
from src.services.clickhouse_service import clickhouse_service

logger = logging.getLogger(__name__)

router = APIRouter()


class BatchJobManager:
    """Manages batch job lifecycle and status tracking"""
    
    def __init__(self):
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
    
    def generate_job_id(self, instrument_id: str, granularity: str, start_date: datetime) -> str:
        """Generate unique job ID"""
        date_str = start_date.strftime("%Y%m%d")
        instrument_lower = instrument_id.lower()
        granularity_lower = granularity.lower()
        random_suffix = uuid.uuid4().hex[:6]
        return f"bf_{date_str}_{instrument_lower}_{granularity_lower}_{random_suffix}"
    
    def create_job(
        self,
        job_id: str,
        request: BatchBackfillRequest
    ) -> Dict[str, Any]:
        """Create new batch job"""
        # Estimate candles and duration
        time_delta = request.end_date - request.start_date
        
        # Rough estimation based on granularity
        granularity_minutes = {
            "M15": 15,
            "H1": 60,
            "H4": 240,
            "D": 1440,
            "W": 10080
        }
        
        minutes_in_period = granularity_minutes.get(request.granularity.value, 60)
        total_minutes = time_delta.total_seconds() / 60
        estimated_candles = int(total_minutes / minutes_in_period)
        
        # Estimate duration (1M candles per minute target)
        estimated_duration_minutes = max(1, estimated_candles // 1_000_000)
        
        job_data = {
            "job_id": job_id,
            "instrument_id": request.instrument_id,
            "granularity": request.granularity.value,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "label_types": request.label_types or ["enhanced_triple_barrier"],
            "options": request.options.dict() if request.options else {},
            "status": JobStatusEnum.PENDING,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "estimated_candles": estimated_candles,
            "estimated_duration_minutes": estimated_duration_minutes,
            "progress": {
                "completed_candles": 0,
                "total_candles": estimated_candles,
                "percentage": 0.0,
                "current_date": None,
                "chunks_completed": 0,
                "chunks_total": 0
            },
            "performance": {
                "candles_per_minute": 0.0,
                "avg_compute_time_ms": 0.0,
                "cache_hit_rate": 0.0,
                "error_rate": 0.0
            },
            "error_message": None
        }
        
        self.active_jobs[job_id] = job_data
        
        # Store in Redis for persistence
        redis_cache.set(f"batch_job:{job_id}", job_data, ttl=86400 * 7)  # 7 days
        
        return job_data
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Try Redis
        job_data = redis_cache.get(f"batch_job:{job_id}")
        if job_data:
            self.active_jobs[job_id] = job_data
        
        return job_data
    
    def update_job(self, job_id: str, updates: Dict[str, Any]):
        """Update job data"""
        if job_id in self.active_jobs:
            self.active_jobs[job_id].update(updates)
            self.active_jobs[job_id]["updated_at"] = datetime.utcnow()
            
            # Update Redis
            redis_cache.set(f"batch_job:{job_id}", self.active_jobs[job_id], ttl=86400 * 7)
    
    def list_jobs(
        self,
        status_filter: Optional[str] = None,
        instrument_filter: Optional[str] = None,
        granularity_filter: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List jobs with filtering"""
        # In production, this would query ClickHouse or dedicated job storage
        # For now, use in-memory storage
        
        all_jobs = list(self.active_jobs.values())
        
        # Apply filters
        if status_filter:
            all_jobs = [j for j in all_jobs if j["status"] == status_filter]
        
        if instrument_filter:
            all_jobs = [j for j in all_jobs if j["instrument_id"] == instrument_filter]
        
        if granularity_filter:
            all_jobs = [j for j in all_jobs if j["granularity"] == granularity_filter]
        
        # Sort by created_at desc
        all_jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        total_count = len(all_jobs)
        paginated_jobs = all_jobs[offset:offset + limit]
        
        return paginated_jobs, total_count
    
    async def execute_batch_job(self, job_id: str):
        """Execute batch job in background"""
        job_data = self.get_job(job_id)
        if not job_data:
            logger.error(f"Job {job_id} not found for execution")
            return
        
        try:
            # Update status to running
            self.update_job(job_id, {"status": JobStatusEnum.RUNNING})
            
            # Execute batch computation
            result = await computation_engine.compute_batch_labels(
                instrument_id=job_data["instrument_id"],
                granularity=job_data["granularity"],
                start_date=job_data["start_date"],
                end_date=job_data["end_date"],
                label_types=job_data["label_types"],
                chunk_size=job_data["options"].get("chunk_size", 10000),
                force_recompute=job_data["options"].get("force_recompute", False)
            )
            
            # Update job with results
            updates = {
                "status": JobStatusEnum.COMPLETED,
                "progress": {
                    **job_data["progress"],
                    "completed_candles": result["processed_candles"],
                    "percentage": 100.0
                },
                "performance": {
                    "candles_per_minute": result["processed_candles"] / max(1, 
                        (datetime.utcnow() - job_data["created_at"]).total_seconds() / 60
                    ),
                    "error_rate": result["error_rate"],
                    "cache_hit_rate": 0.5,  # Would be calculated from actual metrics
                    "avg_compute_time_ms": 25.0  # Would be calculated from actual metrics
                },
                "estimated_completion": datetime.utcnow()
            }
            
            self.update_job(job_id, updates)
            
            logger.info(f"Batch job {job_id} completed successfully")
            
        except Exception as e:
            # Update job with error
            error_updates = {
                "status": JobStatusEnum.FAILED,
                "error_message": str(e)
            }
            self.update_job(job_id, error_updates)
            
            logger.error(f"Batch job {job_id} failed: {str(e)}", exc_info=True)


# Global job manager
job_manager = BatchJobManager()


@router.post(
    "/batch/backfill",
    response_model=BatchJobResponse,
    status_code=202,
    summary="Start batch backfill operation",
    description="""
    Initiate batch computation of labels for a date range.
    Target throughput: 1M+ candles/minute.
    
    Returns immediately with job ID for status tracking.
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request - invalid parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        409: {"description": "Backfill already in progress for this range"}
    }
)
async def start_batch_backfill(
    request: BatchBackfillRequest,
    background_tasks: BackgroundTasks,
    req: Request
):
    """
    Start a batch backfill operation for computing labels over a date range.
    
    Returns immediately with a job ID that can be used to track progress.
    """
    trace_id = getattr(req.state, "trace_id", str(uuid.uuid4()))
    
    try:
        # Validate date range
        if request.end_date <= request.start_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "INVALID_DATE_RANGE",
                        "message": "end_date must be after start_date",
                        "trace_id": trace_id
                    }
                }
            )
        
        # Check for existing jobs for same parameters
        existing_jobs, _ = job_manager.list_jobs(
            instrument_filter=request.instrument_id,
            granularity_filter=request.granularity.value
        )
        
        # Check if there's a running job with overlapping date range
        for job in existing_jobs:
            if job["status"] in [JobStatusEnum.PENDING, JobStatusEnum.RUNNING]:
                job_start = job["start_date"]
                job_end = job["end_date"]
                
                # Check for overlap
                if (request.start_date < job_end and request.end_date > job_start):
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "error": {
                                "code": "BACKFILL_IN_PROGRESS",
                                "message": f"Backfill already running for {request.instrument_id} {request.granularity.value}",
                                "details": {
                                    "existing_job_id": job["job_id"]
                                },
                                "trace_id": trace_id
                            }
                        }
                    )
        
        # Generate job ID
        job_id = job_manager.generate_job_id(
            request.instrument_id,
            request.granularity.value,
            request.start_date
        )
        
        # Create job
        job_data = job_manager.create_job(job_id, request)
        
        # Start background task
        background_tasks.add_task(job_manager.execute_batch_job, job_id)
        
        # Build response
        response = BatchJobResponse(
            job_id=job_id,
            status="started",
            estimated_duration_minutes=job_data["estimated_duration_minutes"],
            estimated_candles=job_data["estimated_candles"],
            priority=request.options.priority if request.options else PriorityEnum.NORMAL,
            _links={
                "self": f"/v1/batch/jobs/{job_id}",
                "status": f"/v1/batch/jobs/{job_id}",
                "cancel": f"/v1/batch/jobs/{job_id}/cancel"  # Would be implemented
            }
        )
        
        logger.info(
            f"Started batch backfill job {job_id} for {request.instrument_id} "
            f"{request.granularity.value} from {request.start_date} to {request.end_date}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to start batch backfill: {str(e)}",
            extra={"trace_id": trace_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "JOB_CREATION_FAILED",
                    "message": "Failed to create batch job",
                    "trace_id": trace_id
                }
            }
        )


@router.get(
    "/batch/jobs/{job_id}",
    response_model=BatchJobStatus,
    summary="Get batch job status",
    description="Monitor the progress of a batch backfill operation",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"}
    }
)
async def get_batch_job_status(
    request: Request,
    job_id: str = Path(
        ...,
        description="Batch job identifier",
        regex=r'^bf_[0-9]{8}_[a-z0-9]+_[a-z0-9]+_[a-z0-9]{6}$'
    )
):
    """Get detailed status and progress information for a batch job."""
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    
    try:
        job_data = job_manager.get_job(job_id)
        if not job_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "JOB_NOT_FOUND",
                        "message": f"Job {job_id} not found",
                        "trace_id": trace_id
                    }
                }
            )
        
        # Build response
        response = BatchJobStatus(
            job_id=job_data["job_id"],
            status=JobStatusEnum(job_data["status"]),
            progress=job_data["progress"],
            performance=job_data.get("performance"),
            estimated_completion=job_data.get("estimated_completion"),
            created_at=job_data["created_at"],
            updated_at=job_data["updated_at"],
            error_message=job_data.get("error_message")
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get job status for {job_id}: {str(e)}",
            extra={"trace_id": trace_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "STATUS_QUERY_FAILED",
                    "message": "Failed to retrieve job status",
                    "trace_id": trace_id
                }
            }
        )


@router.get(
    "/batch/jobs",
    response_model=BatchJobsList,
    summary="List batch jobs",
    description="Get paginated list of batch jobs with filtering"
)
async def list_batch_jobs(
    request: Request,
    status: Optional[JobStatusEnum] = Query(None, description="Filter by job status"),
    instrument_id: Optional[str] = Query(None, description="Filter by instrument"),
    granularity: Optional[GranularityEnum] = Query(None, description="Filter by granularity"),
    page: int = Query(1, description="Page number (1-based)", ge=1),
    per_page: int = Query(20, description="Items per page", ge=1, le=100),
    sort: str = Query(
        "created_at_desc",
        description="Sort order",
        regex="^(created_at_desc|created_at_asc|updated_at_desc)$"
    )
):
    """List batch jobs with filtering and pagination."""
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    
    try:
        offset = (page - 1) * per_page
        
        jobs_data, total_count = job_manager.list_jobs(
            status_filter=status.value if status else None,
            instrument_filter=instrument_id,
            granularity_filter=granularity.value if granularity else None,
            limit=per_page,
            offset=offset
        )
        
        # Convert to response models
        job_statuses = []
        for job_data in jobs_data:
            job_status = BatchJobStatus(
                job_id=job_data["job_id"],
                status=JobStatusEnum(job_data["status"]),
                progress=job_data["progress"],
                performance=job_data.get("performance"),
                estimated_completion=job_data.get("estimated_completion"),
                created_at=job_data["created_at"],
                updated_at=job_data["updated_at"],
                error_message=job_data.get("error_message")
            )
            job_statuses.append(job_status)
        
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
        
        response = BatchJobsList(
            data=job_statuses,
            pagination=pagination
        )
        
        return response
        
    except Exception as e:
        logger.error(
            f"Failed to list batch jobs: {str(e)}",
            extra={"trace_id": trace_id},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "LIST_JOBS_FAILED",
                    "message": "Failed to list batch jobs",
                    "trace_id": trace_id
                }
            }
        )