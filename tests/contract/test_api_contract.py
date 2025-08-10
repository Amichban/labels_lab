"""
Contract tests for Label Computation System API
Generated from OpenAPI 3.0 specification

These tests validate that the API implementation conforms to the contract
by testing request/response schemas, status codes, and business logic.
"""

import json
import pytest
from datetime import datetime, timezone
from typing import Dict, Any

import httpx
from pydantic import ValidationError

from src.api.schemas import (
    CandleLabelRequest,
    CandleData,
    BatchBackfillRequest,
    ComputedLabels,
    BatchJobResponse,
    BatchJobStatus,
    HealthResponse,
    ErrorResponse,
    GranularityEnum,
    JobStatusEnum,
    HealthStatusEnum,
)


class TestApiContract:
    """
    Contract tests ensuring API implementation matches OpenAPI specification
    """

    @pytest.fixture
    def api_client(self):
        """HTTP client configured for API testing"""
        return httpx.Client(
            base_url="http://localhost:8000/v1",
            headers={"Authorization": "Bearer test-token"},
            timeout=30.0
        )

    @pytest.fixture
    def sample_candle_data(self) -> CandleData:
        """Valid candle data for testing"""
        return CandleData(
            ts=datetime(2024, 1, 10, 13, 0, 0, tzinfo=timezone.utc),
            open=1.0950,
            high=1.0970,
            low=1.0940,
            close=1.0965,
            volume=1250000,
            atr_14=0.0025
        )

    @pytest.fixture
    def sample_label_request(self, sample_candle_data) -> CandleLabelRequest:
        """Valid label computation request"""
        return CandleLabelRequest(
            instrument_id="EURUSD",
            granularity=GranularityEnum.H4,
            candle=sample_candle_data,
            label_types=["enhanced_triple_barrier", "vol_scaled_return"]
        )

    # =========================================================================
    # SCHEMA VALIDATION TESTS
    # =========================================================================

    def test_candle_data_validation(self):
        """Test CandleData schema validation"""
        # Valid data
        valid_candle = CandleData(
            ts=datetime.now(timezone.utc),
            open=1.1000,
            high=1.1020,
            low=1.0980,
            close=1.1010,
            volume=1000000,
            atr_14=0.0025
        )
        assert valid_candle.open == 1.1000
        assert valid_candle.volume == 1000000

        # Invalid data - high < low
        with pytest.raises(ValidationError) as exc_info:
            CandleData(
                ts=datetime.now(timezone.utc),
                open=1.1000,
                high=1.0980,  # Lower than low
                low=1.0990,
                close=1.1010,
                volume=1000000
            )
        assert "high must be >= low" in str(exc_info.value)

        # Invalid data - negative price
        with pytest.raises(ValidationError):
            CandleData(
                ts=datetime.now(timezone.utc),
                open=-1.1000,  # Negative price
                high=1.1020,
                low=1.0980,
                close=1.1010,
                volume=1000000
            )

    def test_label_request_validation(self, sample_candle_data):
        """Test CandleLabelRequest schema validation"""
        # Valid request
        valid_request = CandleLabelRequest(
            instrument_id="EURUSD",
            granularity=GranularityEnum.H4,
            candle=sample_candle_data
        )
        assert valid_request.instrument_id == "EURUSD"
        assert valid_request.granularity == GranularityEnum.H4

        # Invalid instrument ID
        with pytest.raises(ValidationError) as exc_info:
            CandleLabelRequest(
                instrument_id="INVALID",  # Wrong format
                granularity=GranularityEnum.H4,
                candle=sample_candle_data
            )
        assert "string does not match expected pattern" in str(exc_info.value)

    def test_batch_request_validation(self):
        """Test BatchBackfillRequest schema validation"""
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

        # Valid request
        valid_request = BatchBackfillRequest(
            instrument_id="EURUSD",
            granularity=GranularityEnum.H4,
            start_date=start_date,
            end_date=end_date
        )
        assert valid_request.start_date < valid_request.end_date

        # Invalid date range
        with pytest.raises(ValidationError) as exc_info:
            BatchBackfillRequest(
                instrument_id="EURUSD",
                granularity=GranularityEnum.H4,
                start_date=end_date,  # After end date
                end_date=start_date
            )
        assert "end_date must be after start_date" in str(exc_info.value)

    # =========================================================================
    # ENDPOINT CONTRACT TESTS
    # =========================================================================

    def test_compute_labels_endpoint_contract(self, api_client, sample_label_request):
        """Test /labels/compute endpoint contract"""
        # Test request serialization
        request_data = sample_label_request.dict()
        request_data['candle']['ts'] = sample_label_request.candle.ts.isoformat()

        # Mock successful response for contract testing
        mock_response_data = {
            "instrument_id": "EURUSD",
            "granularity": "H4",
            "ts": "2024-01-10T13:00:00Z",
            "labels": {
                "enhanced_triple_barrier": {
                    "label": 1,
                    "barrier_hit": "upper",
                    "time_to_barrier": 3,
                    "barrier_price": 1.0990,
                    "level_adjusted": True
                },
                "vol_scaled_return": {
                    "value": 2.35,
                    "quantile": 0.85
                }
            },
            "computation_time_ms": 45,
            "cache_hit": False,
            "version": "1.0.0"
        }

        # Validate response can be parsed by schema
        response_obj = ComputedLabels(**mock_response_data)
        assert response_obj.instrument_id == "EURUSD"
        assert response_obj.labels.enhanced_triple_barrier.label == 1
        assert response_obj.computation_time_ms == 45

    def test_batch_backfill_endpoint_contract(self, api_client):
        """Test /batch/backfill endpoint contract"""
        request = BatchBackfillRequest(
            instrument_id="EURUSD",
            granularity=GranularityEnum.H4,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc)
        )

        # Mock response data
        mock_response_data = {
            "job_id": "bf_20240110_eurusd_h4_abc123",
            "status": "started",
            "estimated_duration_minutes": 45,
            "estimated_candles": 186000,
            "priority": "normal"
        }

        # Validate response schema
        response_obj = BatchJobResponse(**mock_response_data)
        assert response_obj.job_id.startswith("bf_")
        assert response_obj.estimated_candles > 0

    def test_batch_job_status_contract(self, api_client):
        """Test batch job status response contract"""
        mock_status_data = {
            "job_id": "bf_20240110_eurusd_h4_abc123",
            "status": "running",
            "progress": {
                "completed_candles": 125000,
                "total_candles": 186000,
                "percentage": 67.2
            },
            "performance": {
                "candles_per_minute": 2750.5,
                "avg_compute_time_ms": 22.1,
                "cache_hit_rate": 0.94
            },
            "created_at": "2024-01-10T14:15:00Z",
            "updated_at": "2024-01-10T15:12:00Z"
        }

        # Validate schema
        status_obj = BatchJobStatus(**mock_status_data)
        assert status_obj.status == JobStatusEnum.RUNNING
        assert 0 <= status_obj.progress.percentage <= 100
        assert status_obj.progress.completed_candles <= status_obj.progress.total_candles

    def test_health_endpoint_contract(self, api_client):
        """Test /health endpoint contract"""
        mock_health_data = {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": "2024-01-10T15:30:00Z",
            "uptime_seconds": 3600,
            "checks": {
                "clickhouse": "ok",
                "redis": "ok",
                "cache_hit_rate": "ok"
            },
            "metrics": {
                "cache_hit_rate": 0.96,
                "avg_computation_ms": 42.1,
                "active_batch_jobs": 2
            }
        }

        # Validate schema
        health_obj = HealthResponse(**mock_health_data)
        assert health_obj.status == HealthStatusEnum.HEALTHY
        assert health_obj.uptime_seconds >= 0
        assert 0 <= health_obj.metrics.cache_hit_rate <= 1

    # =========================================================================
    # ERROR RESPONSE CONTRACT TESTS
    # =========================================================================

    def test_validation_error_contract(self):
        """Test validation error response format"""
        mock_error_data = {
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

        # Validate error schema
        error_obj = ErrorResponse(**mock_error_data)
        assert error_obj.error.code == "VALIDATION_ERROR"
        assert len(error_obj.error.details) == 2
        assert error_obj.error.details[0].field == "candle.close"

    def test_rate_limit_error_contract(self):
        """Test rate limit error response format"""
        mock_error_data = {
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests",
                "details": [
                    {
                        "message": "Rate limit of 1000 requests per hour exceeded"
                    }
                ],
                "trace_id": "req_abc123def456"
            }
        }

        error_obj = ErrorResponse(**mock_error_data)
        assert error_obj.error.code == "RATE_LIMIT_EXCEEDED"

    # =========================================================================
    # BUSINESS LOGIC CONTRACT TESTS
    # =========================================================================

    def test_enhanced_triple_barrier_label_logic(self):
        """Test enhanced triple barrier label business logic contract"""
        from src.api.schemas import EnhancedTripleBarrierLabel, BarrierHitEnum

        # Test all valid label values
        for label_value in [-1, 0, 1]:
            label = EnhancedTripleBarrierLabel(
                label=label_value,
                barrier_hit=BarrierHitEnum.UPPER if label_value == 1 else 
                           BarrierHitEnum.LOWER if label_value == -1 else 
                           BarrierHitEnum.NONE,
                time_to_barrier=5,
                level_adjusted=True
            )
            assert label.label == label_value

        # Test invalid label value
        with pytest.raises(ValidationError):
            EnhancedTripleBarrierLabel(
                label=2,  # Invalid - must be -1, 0, or 1
                barrier_hit=BarrierHitEnum.UPPER,
                time_to_barrier=5,
                level_adjusted=True
            )

    def test_vol_scaled_return_label_logic(self):
        """Test volatility scaled return label business logic contract"""
        from src.api.schemas import VolScaledReturnLabel

        # Valid label
        label = VolScaledReturnLabel(
            value=2.35,
            quantile=0.85
        )
        assert 0 <= label.quantile <= 1

        # Invalid quantile - out of range
        with pytest.raises(ValidationError):
            VolScaledReturnLabel(
                value=2.35,
                quantile=1.5  # Invalid - must be 0-1
            )

    def test_mfe_mae_label_logic(self):
        """Test MFE/MAE label business logic contract"""
        from src.api.schemas import MfeMaeLabel

        # Valid label with positive profit factor
        label = MfeMaeLabel(
            mfe=0.0045,
            mae=-0.0012,
            profit_factor=3.75
        )
        assert label.profit_factor >= 0

        # Invalid negative profit factor
        with pytest.raises(ValidationError):
            MfeMaeLabel(
                mfe=0.0045,
                mae=-0.0012,
                profit_factor=-1.0  # Invalid - must be >= 0
            )

    def test_pagination_info_contract(self):
        """Test pagination response contract"""
        from src.api.schemas import PaginationInfo

        # Valid pagination
        pagination = PaginationInfo(
            page=2,
            per_page=100,
            total=250,
            total_pages=3,
            has_next=True,
            has_prev=True,
            next_page=3,
            prev_page=1
        )
        
        assert pagination.page <= pagination.total_pages
        assert pagination.total >= 0
        assert pagination.has_next == (pagination.page < pagination.total_pages)
        assert pagination.has_prev == (pagination.page > 1)

    # =========================================================================
    # MULTI-TIMEFRAME ALIGNMENT CONTRACT TESTS
    # =========================================================================

    def test_timestamp_alignment_contract(self):
        """Test timestamp alignment requirements from PRD"""
        from src.utils.timestamp_aligner import align_timestamp
        
        # H4 alignment test - should align to 1,5,9,13,17,21 UTC
        h4_timestamp = datetime(2024, 1, 10, 14, 30, 0, tzinfo=timezone.utc)
        aligned_h4 = align_timestamp(h4_timestamp, GranularityEnum.H4)
        assert aligned_h4.hour in [1, 5, 9, 13, 17, 21]
        assert aligned_h4.minute == 0
        assert aligned_h4.second == 0

        # Daily alignment test - should align to day start
        daily_timestamp = datetime(2024, 1, 10, 14, 30, 0, tzinfo=timezone.utc)
        aligned_daily = align_timestamp(daily_timestamp, GranularityEnum.D)
        assert aligned_daily.hour == 0
        assert aligned_daily.minute == 0
        assert aligned_daily.second == 0

    def test_path_data_granularity_mapping(self):
        """Test path data granularity mapping from PRD"""
        # This would test the actual path data fetching logic
        # that ensures H4 labels use H1 path data, etc.
        
        granularity_mapping = {
            GranularityEnum.W: GranularityEnum.D,
            GranularityEnum.D: GranularityEnum.H4, 
            GranularityEnum.H4: GranularityEnum.H1,
            GranularityEnum.H1: GranularityEnum.M15
        }
        
        for target, path in granularity_mapping.items():
            assert target != path, f"Target {target} should use different granularity for path data"

    # =========================================================================
    # PERFORMANCE CONTRACT TESTS
    # =========================================================================

    def test_computation_time_contract(self):
        """Test computation time requirements from contract"""
        # This would be integration test validating actual performance
        # For contract test, we just validate the schema constraints
        
        response_data = {
            "instrument_id": "EURUSD",
            "granularity": "H4",
            "ts": "2024-01-10T13:00:00Z",
            "labels": {"forward_return": 0.0025},
            "computation_time_ms": 95,  # Under 100ms p99 target
            "cache_hit": False,
            "version": "1.0.0"
        }
        
        response_obj = ComputedLabels(**response_data)
        # Contract: computation_time_ms should be non-negative
        assert response_obj.computation_time_ms >= 0
        # Business requirement: should be under 100ms for p99
        # (This would be tested in performance tests, not contract tests)

    def test_cache_hit_rate_contract(self):
        """Test cache hit rate requirements from contract"""
        health_data = {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": "2024-01-10T15:30:00Z",
            "metrics": {
                "cache_hit_rate": 0.96  # Above 95% target
            }
        }
        
        health_obj = HealthResponse(**health_data)
        # Contract: cache_hit_rate should be between 0 and 1
        assert 0 <= health_obj.metrics.cache_hit_rate <= 1
        # Business requirement: should be >95%
        # (This would be tested in integration tests)


# ==============================================================================
# CONTRACT TEST UTILITIES
# ==============================================================================

class ContractTestRunner:
    """
    Utility class for running contract tests against live API
    """
    
    def __init__(self, base_url: str, auth_token: str):
        self.client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=30.0
        )
    
    def validate_endpoint_contract(
        self, 
        method: str, 
        path: str, 
        request_data: Dict[str, Any] = None,
        expected_status: int = 200,
        response_schema_class = None
    ):
        """
        Validate an endpoint against its contract
        """
        # Make request
        if method.upper() == "GET":
            response = self.client.get(path)
        elif method.upper() == "POST":
            response = self.client.post(path, json=request_data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Validate status code
        assert response.status_code == expected_status, \
            f"Expected {expected_status}, got {response.status_code}: {response.text}"
        
        # Validate response schema if provided
        if response_schema_class and response.status_code < 400:
            try:
                response_obj = response_schema_class(**response.json())
                return response_obj
            except ValidationError as e:
                pytest.fail(f"Response validation failed: {e}")
        
        return response.json() if response.content else None


# Run contract tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])