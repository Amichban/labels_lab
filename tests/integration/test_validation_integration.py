"""
Integration tests for comprehensive validation framework (Issue #8)

These tests verify the complete validation system integration:
- Label computation with validation enabled
- Pre and post-computation validation flows
- Validation metrics collection and alerting
- End-to-end validation workflows
- Performance under validation load

Following test-runner guidance for integration test strategy:
- Tests run after unit tests pass
- Verify system components work together
- Test realistic data scenarios
- Validate performance requirements
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

from src.core.label_computation import LabelComputationEngine
from src.validation.label_validator import (
    LabelValidator, ValidationResult, ValidationSeverity, ValidationCategory
)
from src.validation.validation_metrics import ValidationMetricsCollector
from src.models.data_models import (
    Candle, LabelSet, EnhancedTripleBarrierLabel,
    Granularity, BarrierHit
)


@pytest.mark.integration
class TestValidationIntegration:
    """Integration tests for validation framework"""
    
    @pytest.fixture
    def computation_engine_with_validation(self):
        """Create computation engine with validation enabled"""
        return LabelComputationEngine(enable_validation=True)
    
    @pytest.fixture
    def computation_engine_no_validation(self):
        """Create computation engine with validation disabled"""
        return LabelComputationEngine(enable_validation=False)
    
    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return LabelValidator(strict_mode=False)
    
    @pytest.fixture
    def metrics_collector(self):
        """Create fresh metrics collector"""
        return ValidationMetricsCollector()
    
    @pytest.fixture
    def valid_h4_candle(self):
        """Valid H4 candle for testing"""
        return Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),  # Valid H4 timestamp
            open=1.0500,
            high=1.0580,
            low=1.0450,
            close=1.0520,
            volume=1000.0,
            bid=1.0519,
            ask=1.0521,
            atr_14=0.0045
        )
    
    @pytest.fixture
    def invalid_h4_candle(self):
        """Invalid H4 candle for testing validation failures"""
        return Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 8, 0, 0),  # Invalid H4 timestamp (should be 9:00)
            open=1.0500,
            high=1.0450,  # High < Open - INVALID
            low=1.0550,   # Low > Open - INVALID
            close=1.0520,
            volume=1000.0,
            atr_14=0.0045
        )
    
    @pytest.fixture
    def mock_services(self):
        """Mock external services"""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_ch.fetch_snapshots = AsyncMock(return_value=[])
            mock_ch.fetch_active_levels = AsyncMock(return_value=[])
            
            mock_redis.get_labels = Mock(return_value=None)
            mock_redis.cache_labels = Mock()
            mock_redis.get_active_levels = Mock(return_value=None)
            mock_redis.cache_active_levels = Mock()
            mock_redis.get_path_data = Mock(return_value=None)
            mock_redis.cache_path_data = Mock()
            
            yield mock_ch, mock_redis
    
    # Core integration tests
    
    @pytest.mark.asyncio
    async def test_label_computation_with_validation_success(
        self, computation_engine_with_validation, valid_h4_candle, mock_services
    ):
        """Test successful label computation with validation enabled"""
        
        # Mock the internal methods to return valid data
        computation_engine_with_validation._get_active_levels = AsyncMock(return_value=[])
        computation_engine_with_validation._get_path_data = AsyncMock(return_value=[
            {"ts": valid_h4_candle.ts, "high": 1.0530, "low": 1.0510, "close": 1.0520},
            {"ts": valid_h4_candle.ts + timedelta(hours=1), "high": 1.0540, "low": 1.0520, "close": 1.0535}
        ])
        
        # Perform computation
        result = await computation_engine_with_validation.compute_labels(valid_h4_candle)
        
        # Verify successful computation
        assert isinstance(result, LabelSet)
        assert result.instrument_id == valid_h4_candle.instrument_id
        assert result.granularity == valid_h4_candle.granularity
        assert result.ts == valid_h4_candle.ts
        assert result.computation_time_ms is not None
        assert result.computation_time_ms > 0
        
        # Verify validation stats were updated
        stats = computation_engine_with_validation.get_validation_stats()
        assert stats["total_computations"] == 1
        # Should have no failures for valid input
        assert stats["pre_validation_failures"] == 0
        assert stats["post_validation_failures"] == 0
    
    @pytest.mark.asyncio
    async def test_label_computation_with_validation_pre_failure(
        self, computation_engine_with_validation, invalid_h4_candle, mock_services
    ):
        """Test label computation with pre-validation failure"""
        
        # Mock the internal methods
        computation_engine_with_validation._get_active_levels = AsyncMock(return_value=[])
        computation_engine_with_validation._get_path_data = AsyncMock(return_value=[])
        
        # Perform computation (should log validation issues but still complete)
        result = await computation_engine_with_validation.compute_labels(invalid_h4_candle)
        
        # Computation should still complete despite validation issues
        assert isinstance(result, LabelSet)
        
        # Verify validation stats show failures
        stats = computation_engine_with_validation.get_validation_stats()
        assert stats["total_computations"] == 1
        assert stats["pre_validation_failures"] == 1  # Should detect invalid OHLC and timestamp
    
    @pytest.mark.asyncio
    async def test_label_computation_without_validation(
        self, computation_engine_no_validation, invalid_h4_candle, mock_services
    ):
        """Test label computation with validation disabled"""
        
        # Mock the internal methods
        computation_engine_no_validation._get_active_levels = AsyncMock(return_value=[])
        computation_engine_no_validation._get_path_data = AsyncMock(return_value=[])
        
        # Perform computation
        result = await computation_engine_no_validation.compute_labels(invalid_h4_candle)
        
        # Should complete without validation
        assert isinstance(result, LabelSet)
        
        # Validation stats should show no activity
        stats = computation_engine_no_validation.get_validation_stats()
        assert stats["total_computations"] == 1
        assert stats["pre_validation_failures"] == 0
        assert stats["post_validation_failures"] == 0
    
    @pytest.mark.asyncio
    async def test_cached_label_validation(
        self, computation_engine_with_validation, valid_h4_candle, mock_services
    ):
        """Test that cached labels are also validated"""
        
        mock_ch, mock_redis = mock_services
        
        # Setup cache to return pre-computed labels
        cached_labels = {
            "instrument_id": valid_h4_candle.instrument_id,
            "granularity": valid_h4_candle.granularity.value,
            "ts": valid_h4_candle.ts,
            "enhanced_triple_barrier": {
                "label": 1,
                "barrier_hit": "upper",
                "time_to_barrier": 5,
                "barrier_price": 1.0580,
                "level_adjusted": False,
                "upper_barrier": 1.0580,
                "lower_barrier": 1.0460
            },
            "forward_return": 0.0057,
            "computation_time_ms": 150.0
        }
        
        mock_redis.get_labels.return_value = cached_labels
        
        # Perform computation (should use cached data)
        result = await computation_engine_with_validation.compute_labels(
            valid_h4_candle, use_cache=True
        )
        
        # Should use cached data
        assert isinstance(result, LabelSet)
        assert result.enhanced_triple_barrier is not None
        assert result.enhanced_triple_barrier.label == 1
        
        # Cache should have been checked
        mock_redis.get_labels.assert_called_once()
    
    # Validation metrics integration tests
    
    def test_validation_metrics_integration(self, validator, metrics_collector):
        """Test validation metrics collection integration"""
        
        # Create test candles with various validation outcomes
        test_cases = [
            # Valid candle
            Candle(
                instrument_id="EUR/USD", granularity=Granularity.H4,
                ts=datetime(2024, 1, 15, 9, 0, 0),
                open=1.0500, high=1.0580, low=1.0450, close=1.0520,
                volume=1000.0
            ),
            # Invalid timestamp
            Candle(
                instrument_id="EUR/USD", granularity=Granularity.H4,
                ts=datetime(2024, 1, 15, 8, 0, 0),  # Invalid H4 hour
                open=1.0500, high=1.0580, low=1.0450, close=1.0520,
                volume=1000.0
            ),
            # Invalid OHLC
            Candle(
                instrument_id="EUR/USD", granularity=Granularity.H4,
                ts=datetime(2024, 1, 15, 13, 0, 0),
                open=1.0500, high=1.0450, low=1.0550, close=1.0520,  # High < Low
                volume=1000.0
            )
        ]
        
        # Run validations and collect metrics
        for candle in test_cases:
            result = validator.validate_pre_computation(candle, 6)
            metrics_collector.record_validation_result(result)
        
        # Check metrics
        current_metrics = metrics_collector.get_current_metrics()
        assert current_metrics.total_validations == 3
        assert current_metrics.successful_validations == 1  # Only first candle is valid
        assert current_metrics.failed_validations == 2
        
        # Check metrics summary
        summary = metrics_collector.get_metrics_summary()
        assert summary["total_validations"] == 3
        assert summary["failure_rate"] == 2/3
        
        # Check that category counts are tracked
        assert "timestamp_alignment" in summary["category_counts"]
        assert "data_consistency" in summary["category_counts"]
    
    def test_validation_alerting_integration(self, validator, metrics_collector):
        """Test validation alerting system integration"""
        
        # Create candles that will trigger critical issues (look-ahead bias)
        future_timestamp = datetime(2024, 1, 15, 9, 0, 0)
        past_timestamp = future_timestamp - timedelta(hours=1)
        
        # Path data with look-ahead bias
        lookahead_path_data = [
            {"ts": past_timestamp, "high": 1.0540, "low": 1.0510, "close": 1.0535}  # FUTURE DATA
        ]
        
        candle = Candle(
            instrument_id="EUR/USD", granularity=Granularity.H4,
            ts=future_timestamp,
            open=1.0500, high=1.0580, low=1.0450, close=1.0520,
            volume=1000.0
        )
        
        # Run validation with look-ahead bias data
        result = validator.validate_pre_computation(candle, 6, path_data=lookahead_path_data)
        metrics_collector.record_validation_result(result)
        
        # Should trigger critical alert
        active_alerts = metrics_collector.get_active_alerts()
        assert len(active_alerts) > 0
        
        # Check for look-ahead bias alert
        lookahead_alerts = [
            alert for alert in active_alerts
            if "lookahead_bias" in alert["rule_name"] or "critical" in alert["severity"]
        ]
        assert len(lookahead_alerts) > 0
    
    # Performance integration tests
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_validation_performance_impact(
        self, computation_engine_with_validation, computation_engine_no_validation,
        valid_h4_candle, mock_services, performance_timer
    ):
        """Test validation performance impact on label computation"""
        
        # Mock internal methods for both engines
        for engine in [computation_engine_with_validation, computation_engine_no_validation]:
            engine._get_active_levels = AsyncMock(return_value=[])
            engine._get_path_data = AsyncMock(return_value=[
                {"ts": valid_h4_candle.ts, "high": 1.0530, "low": 1.0510, "close": 1.0520}
            ])
        
        # Measure computation time without validation
        performance_timer.start()
        for _ in range(10):
            await computation_engine_no_validation.compute_labels(valid_h4_candle)
        time_without_validation = performance_timer.stop()
        
        # Reset timer and measure with validation
        performance_timer.start()
        for _ in range(10):
            await computation_engine_with_validation.compute_labels(valid_h4_candle)
        time_with_validation = performance_timer.stop()
        
        # Validation should add reasonable overhead (< 50% increase)
        overhead_ratio = time_with_validation / time_without_validation
        assert overhead_ratio < 1.5, f"Validation overhead too high: {overhead_ratio:.2f}x"
        
        # Both should complete within reasonable time
        assert time_without_validation < 1000, f"Base computation too slow: {time_without_validation:.2f}ms"
        assert time_with_validation < 1500, f"Validation-enabled computation too slow: {time_with_validation:.2f}ms"
    
    # Batch validation integration tests
    
    @pytest.mark.asyncio 
    async def test_batch_validation_integration(
        self, computation_engine_with_validation, mock_services
    ):
        """Test batch validation integration with label computation"""
        
        # Mock batch processing
        mock_ch, mock_redis = mock_services
        
        # Create mock snapshot data
        base_time = datetime(2024, 1, 15, 1, 0, 0)
        mock_snapshots = []
        
        for i in range(10):
            mock_snapshots.append({
                "ts": base_time + timedelta(hours=4 * i),
                "open": 1.0500 + np.random.normal(0, 0.001),
                "high": 1.0580 + np.random.normal(0, 0.001),
                "low": 1.0450 + np.random.normal(0, 0.001),
                "close": 1.0520 + np.random.normal(0, 0.001),
                "volume": 1000 + np.random.normal(0, 100),
                "atr_14": 0.0045
            })
        
        mock_ch.fetch_snapshots.return_value = mock_snapshots
        
        # Mock internal methods
        computation_engine_with_validation._get_active_levels = AsyncMock(return_value=[])
        computation_engine_with_validation._get_path_data = AsyncMock(return_value=[])
        
        # Run batch computation
        result = await computation_engine_with_validation.compute_batch_labels(
            instrument_id="EUR/USD",
            granularity="H4",
            start_date=base_time,
            end_date=base_time + timedelta(days=2),
            label_types=["enhanced_triple_barrier"],
            chunk_size=5
        )
        
        # Verify batch results include validation stats
        assert "validation_stats" in result
        assert "batch_validation" in result
        assert result["total_candles"] == 10
        assert result["successful_labels"] == 10  # All should succeed with valid mock data
        assert result["error_rate"] == 0.0
        
        # Check validation stats
        validation_stats = result["validation_stats"]
        assert validation_stats["total_computations"] == 10
    
    # Error handling integration tests
    
    @pytest.mark.asyncio
    async def test_validation_error_handling_integration(
        self, computation_engine_with_validation, valid_h4_candle, mock_services
    ):
        """Test validation error handling in integration context"""
        
        # Mock services to raise exceptions
        mock_ch, mock_redis = mock_services
        mock_ch.fetch_active_levels.side_effect = Exception("ClickHouse connection error")
        mock_redis.get_labels.side_effect = Exception("Redis connection error")
        
        # Mock internal methods to also raise exceptions during validation data fetch
        computation_engine_with_validation._get_active_levels = AsyncMock(
            side_effect=Exception("Service error during validation")
        )
        computation_engine_with_validation._get_path_data = AsyncMock(return_value=[])
        
        # Computation should handle validation errors gracefully
        result = await computation_engine_with_validation.compute_labels(valid_h4_candle)
        
        # Should still return valid result despite validation data fetch errors
        assert isinstance(result, LabelSet)
        assert result.instrument_id == valid_h4_candle.instrument_id
        
        # Validation stats should show attempts
        stats = computation_engine_with_validation.get_validation_stats()
        assert stats["total_computations"] == 1
    
    # Statistical validation integration tests
    
    def test_statistical_validation_integration(self, validator):
        """Test statistical validation with realistic data patterns"""
        
        # Create batch with known statistical properties
        base_time = datetime(2024, 1, 15, 1, 0, 0)
        label_sets = []
        
        # Generate data with specific statistical properties
        np.random.seed(42)  # For reproducible test
        
        # Normal distribution for forward returns
        normal_returns = np.random.normal(0, 0.01, 100)
        
        # Skewed distribution for volatility-scaled returns
        skewed_vol_returns = np.random.exponential(1, 100) - 1  # Exponential - 1 to center around 0
        
        for i in range(100):
            # Create Enhanced Triple Barrier labels with realistic distribution
            etb_label = 1 if normal_returns[i] > 0.005 else (-1 if normal_returns[i] < -0.005 else 0)
            
            label_sets.append(LabelSet(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=base_time + timedelta(hours=4 * i),
                enhanced_triple_barrier=EnhancedTripleBarrierLabel(
                    label=etb_label,
                    barrier_hit=BarrierHit.UPPER if etb_label == 1 else (
                        BarrierHit.LOWER if etb_label == -1 else BarrierHit.NONE
                    ),
                    time_to_barrier=5 if etb_label != 0 else 24,
                    barrier_price=1.0580 if etb_label == 1 else (1.0460 if etb_label == -1 else None),
                    level_adjusted=False,
                    upper_barrier=1.0580,
                    lower_barrier=1.0460
                ),
                forward_return=float(normal_returns[i]),
                vol_scaled_return=float(skewed_vol_returns[i]),
                mfe=float(abs(normal_returns[i]) if normal_returns[i] > 0 else 0.001),
                mae=float(-abs(normal_returns[i]) if normal_returns[i] < 0 else -0.001),
                profit_factor=float(abs(normal_returns[i]) / 0.001) if normal_returns[i] != 0 else 1.0,
                computation_time_ms=100 + np.random.normal(0, 20)
            ))
        
        # Run batch validation with statistical tests
        result = validator.validate_batch_labels(label_sets, statistical_tests=True)
        
        # Should detect statistical patterns
        assert "forward_returns_stats" in result.metrics
        assert "vol_scaled_returns_stats" in result.metrics
        
        forward_stats = result.metrics["forward_returns_stats"]
        vol_stats = result.metrics["vol_scaled_returns_stats"]
        
        # Normal returns should have relatively low skewness
        assert abs(forward_stats["skewness"]) < 1.0  # Normal distribution has low skew
        
        # Exponential-based vol returns should have high skewness
        assert vol_stats["skewness"] > 1.0  # Exponential has high positive skew
        
        # Should detect statistical issues for the skewed distribution
        stat_issues = result.get_issues_by_category(ValidationCategory.STATISTICAL_DISTRIBUTION)
        assert len(stat_issues) > 0
    
    # End-to-end workflow tests
    
    @pytest.mark.asyncio
    async def test_end_to_end_validation_workflow(
        self, computation_engine_with_validation, validator, metrics_collector, mock_services
    ):
        """Test complete end-to-end validation workflow"""
        
        # Create mixed batch: some valid, some invalid candles
        candles = [
            # Valid H4 candle
            Candle(
                instrument_id="EUR/USD", granularity=Granularity.H4,
                ts=datetime(2024, 1, 15, 9, 0, 0),
                open=1.0500, high=1.0580, low=1.0450, close=1.0520, volume=1000.0
            ),
            # Invalid timestamp
            Candle(
                instrument_id="EUR/USD", granularity=Granularity.H4,
                ts=datetime(2024, 1, 15, 10, 0, 0),  # Invalid H4 hour
                open=1.0500, high=1.0580, low=1.0450, close=1.0520, volume=1000.0
            ),
            # Valid H4 candle
            Candle(
                instrument_id="EUR/USD", granularity=Granularity.H4,
                ts=datetime(2024, 1, 15, 13, 0, 0),
                open=1.0520, high=1.0590, low=1.0470, close=1.0540, volume=1100.0
            )
        ]
        
        # Mock internal methods
        computation_engine_with_validation._get_active_levels = AsyncMock(return_value=[])
        computation_engine_with_validation._get_path_data = AsyncMock(return_value=[])
        
        # Process each candle and collect metrics
        results = []
        for candle in candles:
            # Compute labels (includes validation)
            label_result = await computation_engine_with_validation.compute_labels(candle)
            results.append(label_result)
            
            # Also run standalone validation for metrics collection
            validation_result = validator.validate_pre_computation(candle, 6)
            metrics_collector.record_validation_result(validation_result)
        
        # Check computation results
        assert len(results) == 3
        for result in results:
            assert isinstance(result, LabelSet)
        
        # Check validation metrics
        summary = metrics_collector.get_metrics_summary()
        assert summary["total_validations"] == 3
        assert summary["failure_rate"] > 0  # Should detect invalid timestamp
        
        # Check validation stats from computation engine
        engine_stats = computation_engine_with_validation.get_validation_stats()
        assert engine_stats["total_computations"] == 3
        assert engine_stats["pre_validation_failures"] > 0  # Should detect issues
        
        # Check health score
        health_score, health_breakdown = metrics_collector.get_health_score()
        assert 0 <= health_score <= 100
        assert "success_rate" in health_breakdown
        
        # Check for alerts
        active_alerts = metrics_collector.get_active_alerts()
        # May or may not have alerts depending on thresholds and timing
        assert isinstance(active_alerts, list)
    
    # Configuration and deployment tests
    
    def test_validation_configuration_integration(self):
        """Test validation system configuration options"""
        
        # Test different validation configurations
        strict_engine = LabelComputationEngine(enable_validation=True)
        lenient_engine = LabelComputationEngine(enable_validation=False)
        
        # Both should be configured correctly
        assert strict_engine.enable_validation == True
        assert lenient_engine.enable_validation == False
        
        # Validation stats should be initialized
        strict_stats = strict_engine.get_validation_stats()
        lenient_stats = lenient_engine.get_validation_stats()
        
        assert "total_computations" in strict_stats
        assert "total_computations" in lenient_stats
        assert strict_stats["total_computations"] == 0
        assert lenient_stats["total_computations"] == 0
    
    @pytest.mark.slow
    def test_validation_under_load(self, validator, metrics_collector):
        """Test validation system performance under load"""
        
        # Create large number of validation requests
        base_time = datetime(2024, 1, 15, 1, 0, 0)
        
        # Mix of valid and invalid candles
        test_candles = []
        for i in range(1000):
            # 80% valid, 20% invalid
            if i % 5 == 0:
                # Invalid timestamp
                test_candles.append(Candle(
                    instrument_id="EUR/USD", granularity=Granularity.H4,
                    ts=base_time + timedelta(hours=4 * i, minutes=30),  # Invalid H4 alignment
                    open=1.0500, high=1.0580, low=1.0450, close=1.0520, volume=1000.0
                ))
            else:
                # Valid candle
                test_candles.append(Candle(
                    instrument_id="EUR/USD", granularity=Granularity.H4,
                    ts=base_time + timedelta(hours=4 * i),
                    open=1.0500, high=1.0580, low=1.0450, close=1.0520, volume=1000.0
                ))
        
        # Process all candles
        start_time = datetime.utcnow()
        
        for candle in test_candles:
            result = validator.validate_pre_computation(candle, 6)
            metrics_collector.record_validation_result(result)
        
        end_time = datetime.utcnow()
        total_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Check performance
        avg_time_per_validation = total_time_ms / len(test_candles)
        assert avg_time_per_validation < 10, f"Average validation time too slow: {avg_time_per_validation:.2f}ms"
        
        # Check metrics accuracy
        summary = metrics_collector.get_metrics_summary()
        assert summary["total_validations"] == 1000
        expected_failure_rate = 0.2  # 20% invalid
        assert abs(summary["failure_rate"] - expected_failure_rate) < 0.05  # Within 5% tolerance