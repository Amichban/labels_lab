"""
Unit tests for LabelValidator - Comprehensive validation framework (Issue #8)

Tests follow the test-runner guidance for systematic validation:
- Unit tests run first and fail fast
- Comprehensive coverage of validation logic
- Edge cases and error conditions
- Performance validation
- Statistical distribution testing

These tests ensure the validation framework properly detects:
1. Look-ahead bias violations
2. Data consistency issues
3. Timestamp alignment problems
4. Statistical distribution anomalies
5. Path granularity mapping errors
"""

# Import test config first to mock dependencies
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import test_config

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.validation.label_validator import (
    LabelValidator, ValidationResult, ValidationIssue,
    ValidationSeverity, ValidationCategory
)
from src.models.data_models import (
    Candle, LabelSet, EnhancedTripleBarrierLabel,
    Granularity, BarrierHit
)


@pytest.mark.unit
class TestLabelValidator:
    """Test suite for LabelValidator class"""
    
    @pytest.fixture
    def validator(self):
        """Create LabelValidator instance"""
        return LabelValidator(strict_mode=False)
    
    @pytest.fixture
    def strict_validator(self):
        """Create strict LabelValidator instance"""
        return LabelValidator(strict_mode=True)
    
    @pytest.fixture
    def sample_candle(self):
        """Sample candle for testing"""
        return Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),  # H4 aligned
            open=1.0500,
            high=1.0580,
            low=1.0450,
            close=1.0520,
            volume=1000.0,
            atr_14=0.0045
        )
    
    @pytest.fixture
    def valid_label_set(self, sample_candle):
        """Valid label set for testing"""
        return LabelSet(
            instrument_id=sample_candle.instrument_id,
            granularity=sample_candle.granularity,
            ts=sample_candle.ts,
            enhanced_triple_barrier=EnhancedTripleBarrierLabel(
                label=1,
                barrier_hit=BarrierHit.UPPER,
                time_to_barrier=5,
                barrier_price=1.0580,
                level_adjusted=False,
                upper_barrier=1.0580,
                lower_barrier=1.0460
            ),
            forward_return=0.0057,
            vol_scaled_return=1.267,
            mfe=0.0060,
            mae=-0.0030,
            profit_factor=2.0,
            retouch_count=2,
            breakout_occurred=True,
            flip_occurred=False,
            computation_time_ms=150.5
        )
    
    @pytest.fixture
    def valid_path_data(self, sample_candle):
        """Valid path data for testing"""
        return [
            {
                "ts": sample_candle.ts,
                "open": 1.0520, "high": 1.0540, "low": 1.0510, "close": 1.0535
            },
            {
                "ts": sample_candle.ts + timedelta(hours=1),
                "open": 1.0535, "high": 1.0560, "low": 1.0520, "close": 1.0550
            },
            {
                "ts": sample_candle.ts + timedelta(hours=2),
                "open": 1.0550, "high": 1.0585, "low": 1.0545, "close": 1.0580
            }
        ]
    
    # Pre-computation validation tests
    
    def test_pre_computation_validation_valid_data(self, validator, sample_candle, valid_path_data):
        """Test pre-computation validation with valid data"""
        
        result = validator.validate_pre_computation(
            candle=sample_candle,
            horizon_periods=6,
            path_data=valid_path_data,
            levels=[]
        )
        
        assert result.is_valid
        assert len(result.issues) == 0
        assert result.validation_time_ms is not None
        assert result.validation_time_ms >= 0
    
    def test_pre_computation_candle_integrity_invalid_ohlc(self, validator):
        """Test candle integrity validation with invalid OHLC"""
        
        invalid_candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=1.0500,
            high=1.0450,  # High < Open - INVALID
            low=1.0550,   # Low > Open - INVALID
            close=1.0520,
            volume=1000.0
        )
        
        result = validator.validate_pre_computation(invalid_candle, 6)
        
        assert not result.is_valid
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert len(error_issues) > 0
        assert any("Invalid OHLC relationships" in issue.message for issue in error_issues)
    
    def test_pre_computation_negative_prices(self, validator):
        """Test validation with negative prices"""
        
        invalid_candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=-1.0500,  # Negative price - INVALID
            high=1.0580,
            low=1.0450,
            close=1.0520,
            volume=1000.0
        )
        
        result = validator.validate_pre_computation(invalid_candle, 6)
        
        assert not result.is_valid
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert any("OHLC values must be positive" in issue.message for issue in error_issues)
    
    def test_pre_computation_timestamp_alignment_h4(self, validator):
        """Test H4 timestamp alignment validation"""
        
        # Invalid H4 timestamp (not at 1,5,9,13,17,21 UTC)
        invalid_candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 8, 0, 0),  # 8:00 UTC - INVALID for H4
            open=1.0500,
            high=1.0580,
            low=1.0450,
            close=1.0520,
            volume=1000.0
        )
        
        result = validator.validate_pre_computation(invalid_candle, 6)
        
        assert not result.is_valid
        alignment_issues = result.get_issues_by_category(ValidationCategory.TIMESTAMP_ALIGNMENT)
        assert len(alignment_issues) > 0
        assert any("H4 timestamp not at expected hours" in issue.message for issue in alignment_issues)
    
    def test_pre_computation_horizon_validation(self, validator, sample_candle):
        """Test horizon parameters validation"""
        
        # Test negative horizon
        result = validator.validate_pre_computation(sample_candle, -1)
        assert not result.is_valid
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert any("Horizon periods must be positive" in issue.message for issue in error_issues)
        
        # Test very large horizon (should warn)
        result = validator.validate_pre_computation(sample_candle, 200)
        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert any("Very large horizon periods" in issue.message for issue in warning_issues)
    
    def test_pre_computation_lookahead_bias_path_data(self, validator, sample_candle):
        """Test look-ahead bias detection in path data"""
        
        # Path data with future timestamps (before candle timestamp)
        lookahead_path_data = [
            {
                "ts": sample_candle.ts - timedelta(hours=1),  # FUTURE DATA - INVALID
                "open": 1.0520, "high": 1.0540, "low": 1.0510, "close": 1.0535
            },
            {
                "ts": sample_candle.ts,
                "open": 1.0535, "high": 1.0560, "low": 1.0520, "close": 1.0550
            }
        ]
        
        result = validator.validate_pre_computation(
            sample_candle, 6, path_data=lookahead_path_data
        )
        
        assert not result.is_valid
        lookahead_issues = result.get_issues_by_category(ValidationCategory.LOOKAHEAD_BIAS)
        assert len(lookahead_issues) > 0
        assert any("Path data contains timestamps before candle timestamp" in issue.message 
                  for issue in lookahead_issues)
    
    def test_pre_computation_levels_lookahead_bias(self, validator, sample_candle):
        """Test look-ahead bias detection in levels data"""
        
        # Level created after candle timestamp
        lookahead_levels = [
            {
                "price": 1.0580,
                "current_type": "resistance",
                "created_at": sample_candle.ts + timedelta(hours=1)  # FUTURE DATA - INVALID
            }
        ]
        
        result = validator.validate_pre_computation(
            sample_candle, 6, levels=lookahead_levels
        )
        
        assert not result.is_valid
        lookahead_issues = result.get_issues_by_category(ValidationCategory.LOOKAHEAD_BIAS)
        assert any("Level at index 0 created after candle timestamp" in issue.message 
                  for issue in lookahead_issues)
    
    def test_pre_computation_bid_ask_validation(self, validator):
        """Test bid/ask spread validation"""
        
        # Invalid spread (bid >= ask)
        invalid_candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=1.0500,
            high=1.0580,
            low=1.0450,
            close=1.0520,
            volume=1000.0,
            bid=1.0522,  # Bid > Ask - INVALID
            ask=1.0521
        )
        
        result = validator.validate_pre_computation(invalid_candle, 6)
        
        assert not result.is_valid
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert any("Bid must be less than Ask" in issue.message for issue in error_issues)
    
    # Post-computation validation tests
    
    def test_post_computation_validation_valid_data(self, validator, sample_candle, valid_label_set):
        """Test post-computation validation with valid data"""
        
        result = validator.validate_post_computation(
            candle=sample_candle,
            label_set=valid_label_set
        )
        
        assert result.is_valid
        # May have info-level issues but should be valid overall
        critical_errors = (
            result.get_issues_by_severity(ValidationSeverity.CRITICAL) +
            result.get_issues_by_severity(ValidationSeverity.ERROR)
        )
        assert len(critical_errors) == 0
    
    def test_post_computation_no_lookahead_bias_computed_at(self, validator, sample_candle):
        """Test look-ahead bias detection in computed_at timestamp"""
        
        invalid_label_set = LabelSet(
            instrument_id=sample_candle.instrument_id,
            granularity=sample_candle.granularity,
            ts=sample_candle.ts,
            computed_at=sample_candle.ts - timedelta(seconds=1)  # BEFORE candle - INVALID
        )
        
        result = validator.validate_post_computation(sample_candle, invalid_label_set)
        
        assert not result.is_valid
        critical_issues = result.get_issues_by_severity(ValidationSeverity.CRITICAL)
        assert any("Label computed before candle timestamp" in issue.message for issue in critical_issues)
    
    def test_post_computation_barrier_logic_validation(self, validator, sample_candle):
        """Test Enhanced Triple Barrier logic validation"""
        
        # Invalid barrier ordering
        invalid_etb = EnhancedTripleBarrierLabel(
            label=1,
            barrier_hit=BarrierHit.UPPER,
            time_to_barrier=5,
            barrier_price=1.0580,
            level_adjusted=False,
            upper_barrier=1.0460,  # Upper < Lower - INVALID
            lower_barrier=1.0580
        )
        
        invalid_label_set = LabelSet(
            instrument_id=sample_candle.instrument_id,
            granularity=sample_candle.granularity,
            ts=sample_candle.ts,
            enhanced_triple_barrier=invalid_etb
        )
        
        result = validator.validate_post_computation(sample_candle, invalid_label_set)
        
        assert not result.is_valid
        barrier_issues = result.get_issues_by_category(ValidationCategory.BARRIER_LOGIC)
        assert any("Upper barrier must be greater than lower barrier" in issue.message 
                  for issue in barrier_issues)
    
    def test_post_computation_barrier_label_consistency(self, validator, sample_candle):
        """Test barrier hit and label consistency"""
        
        # Upper barrier hit but label != 1
        inconsistent_etb = EnhancedTripleBarrierLabel(
            label=-1,  # Should be 1 for upper hit - INCONSISTENT
            barrier_hit=BarrierHit.UPPER,
            time_to_barrier=5,
            barrier_price=1.0580,
            level_adjusted=False,
            upper_barrier=1.0580,
            lower_barrier=1.0460
        )
        
        invalid_label_set = LabelSet(
            instrument_id=sample_candle.instrument_id,
            granularity=sample_candle.granularity,
            ts=sample_candle.ts,
            enhanced_triple_barrier=inconsistent_etb
        )
        
        result = validator.validate_post_computation(sample_candle, invalid_label_set)
        
        assert not result.is_valid
        barrier_issues = result.get_issues_by_category(ValidationCategory.BARRIER_LOGIC)
        assert any("Upper barrier hit should result in label = 1" in issue.message 
                  for issue in barrier_issues)
    
    def test_post_computation_mfe_mae_consistency_critical(self, validator, sample_candle):
        """Test MFE/MAE consistency - CRITICAL constraint: MFE >= -MAE"""
        
        # Violate fundamental constraint: MFE < -MAE
        invalid_label_set = LabelSet(
            instrument_id=sample_candle.instrument_id,
            granularity=sample_candle.granularity,
            ts=sample_candle.ts,
            mfe=0.0020,   # MFE = 0.002
            mae=-0.0030,  # MAE = -0.003, so -MAE = 0.003
                         # MFE (0.002) < -MAE (0.003) - CRITICAL VIOLATION
            profit_factor=2.0
        )
        
        result = validator.validate_post_computation(sample_candle, invalid_label_set)
        
        assert not result.is_valid
        critical_issues = result.get_issues_by_severity(ValidationSeverity.CRITICAL)
        assert any("MFE must be >= -MAE" in issue.message for issue in critical_issues)
    
    def test_post_computation_mfe_mae_sign_validation(self, validator, sample_candle):
        """Test MFE/MAE sign validation"""
        
        # MFE should be >= 0, MAE should be <= 0
        invalid_label_set = LabelSet(
            instrument_id=sample_candle.instrument_id,
            granularity=sample_candle.granularity,
            ts=sample_candle.ts,
            mfe=-0.0020,  # MFE negative - INVALID
            mae=0.0030    # MAE positive - INVALID
        )
        
        result = validator.validate_post_computation(sample_candle, invalid_label_set)
        
        assert not result.is_valid
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        
        mfe_error = any("MFE (Maximum Favorable Excursion) should be non-negative" in issue.message 
                       for issue in error_issues)
        mae_error = any("MAE (Maximum Adverse Excursion) should be non-positive" in issue.message 
                       for issue in error_issues)
        
        assert mfe_error and mae_error
    
    def test_post_computation_profit_factor_consistency(self, validator, sample_candle):
        """Test profit factor calculation consistency"""
        
        mfe = 0.0060
        mae = -0.0030
        incorrect_pf = 3.0  # Should be 0.006 / 0.003 = 2.0
        
        invalid_label_set = LabelSet(
            instrument_id=sample_candle.instrument_id,
            granularity=sample_candle.granularity,
            ts=sample_candle.ts,
            mfe=mfe,
            mae=mae,
            profit_factor=incorrect_pf
        )
        
        result = validator.validate_post_computation(sample_candle, invalid_label_set)
        
        assert not result.is_valid
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert any("Profit factor inconsistent with MFE/MAE" in issue.message for issue in error_issues)
    
    def test_post_computation_negative_counts_validation(self, validator, sample_candle):
        """Test validation of count fields (must be non-negative)"""
        
        invalid_label_set = LabelSet(
            instrument_id=sample_candle.instrument_id,
            granularity=sample_candle.granularity,
            ts=sample_candle.ts,
            retouch_count=-5,     # Negative - INVALID
            time_underwater=-10   # Negative - INVALID
        )
        
        result = validator.validate_post_computation(sample_candle, invalid_label_set)
        
        assert not result.is_valid
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        
        retouch_error = any("retouch_count cannot be negative" in issue.message for issue in error_issues)
        underwater_error = any("time_underwater cannot be negative" in issue.message for issue in error_issues)
        
        assert retouch_error and underwater_error
    
    def test_post_computation_performance_validation(self, validator, sample_candle):
        """Test computation performance validation"""
        
        # Very slow computation time
        slow_label_set = LabelSet(
            instrument_id=sample_candle.instrument_id,
            granularity=sample_candle.granularity,
            ts=sample_candle.ts,
            computation_time_ms=15000.0  # 15 seconds - SLOW
        )
        
        result = validator.validate_post_computation(sample_candle, slow_label_set)
        
        # Should be valid but with performance warning
        assert result.is_valid
        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert any("Computation time exceeds 10 seconds" in issue.message for issue in warning_issues)
        
        # Negative computation time
        invalid_label_set = LabelSet(
            instrument_id=sample_candle.instrument_id,
            granularity=sample_candle.granularity,
            ts=sample_candle.ts,
            computation_time_ms=-100.0  # Negative - INVALID
        )
        
        result = validator.validate_post_computation(sample_candle, invalid_label_set)
        
        assert not result.is_valid
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert any("Negative computation time" in issue.message for issue in error_issues)
    
    # Batch validation tests
    
    def test_batch_validation_valid_data(self, validator):
        """Test batch validation with valid data"""
        
        # Create batch of label sets
        label_sets = []
        base_time = datetime(2024, 1, 15, 1, 0, 0)
        
        for i in range(10):
            label_sets.append(LabelSet(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=base_time + timedelta(hours=4 * i),
                forward_return=np.random.normal(0, 0.01),
                vol_scaled_return=np.random.normal(0, 1),
                computation_time_ms=100 + np.random.normal(0, 20)
            ))
        
        result = validator.validate_batch_labels(label_sets, statistical_tests=True)
        
        assert result.is_valid
        assert result.metrics["batch_size"] == 10
        assert result.validation_time_ms is not None
    
    def test_batch_validation_empty_batch(self, validator):
        """Test batch validation with empty batch"""
        
        result = validator.validate_batch_labels([])
        
        assert not result.is_valid
        error_issues = result.get_issues_by_severity(ValidationSeverity.ERROR)
        assert any("Empty batch provided" in issue.message for issue in error_issues)
    
    def test_batch_validation_temporal_ordering(self, validator):
        """Test temporal ordering validation in batch"""
        
        # Create batch with wrong temporal order
        base_time = datetime(2024, 1, 15, 1, 0, 0)
        label_sets = [
            LabelSet(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=base_time + timedelta(hours=8),  # Later timestamp first
                forward_return=0.01
            ),
            LabelSet(
                instrument_id="EUR/USD", 
                granularity=Granularity.H4,
                ts=base_time,  # Earlier timestamp second - WRONG ORDER
                forward_return=0.02
            )
        ]
        
        result = validator.validate_batch_labels(label_sets)
        
        # Should be valid but with ordering warning
        assert result.is_valid
        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        assert any("Batch is not temporally ordered" in issue.message for issue in warning_issues)
    
    def test_batch_validation_statistical_distributions(self, validator):
        """Test statistical distribution validation (Jarque-Bera test)"""
        
        # Create batch with known non-normal distribution (heavily skewed)
        label_sets = []
        base_time = datetime(2024, 1, 15, 1, 0, 0)
        
        # Generate heavily skewed data
        skewed_returns = np.random.exponential(0.01, 50)  # Exponential is skewed
        
        for i in range(50):
            label_sets.append(LabelSet(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=base_time + timedelta(hours=4 * i),
                forward_return=float(skewed_returns[i]),
                vol_scaled_return=float(skewed_returns[i] / 0.005)  # Scale by ATR
            ))
        
        result = validator.validate_batch_labels(label_sets, statistical_tests=True)
        
        # Should detect statistical issues
        stat_issues = result.get_issues_by_category(ValidationCategory.STATISTICAL_DISTRIBUTION)
        assert len(stat_issues) > 0
        
        # Check that statistics are computed
        assert "forward_returns_stats" in result.metrics
        assert "vol_scaled_returns_stats" in result.metrics
        
        forward_stats = result.metrics["forward_returns_stats"]
        assert "jarque_bera_stat" in forward_stats
        assert "jarque_bera_p_value" in forward_stats
        assert "skewness" in forward_stats
        assert "kurtosis" in forward_stats
    
    def test_batch_validation_cross_label_consistency(self, validator):
        """Test cross-label consistency validation"""
        
        # Create batch with systematic inconsistencies
        label_sets = []
        base_time = datetime(2024, 1, 15, 1, 0, 0)
        
        for i in range(20):
            # Create systematic inconsistency: positive ETB labels with negative returns
            etb_label = 1 if i % 2 == 0 else -1
            forward_return = -0.01 if i % 2 == 0 else 0.01  # OPPOSITE of ETB label
            
            label_sets.append(LabelSet(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=base_time + timedelta(hours=4 * i),
                enhanced_triple_barrier=EnhancedTripleBarrierLabel(
                    label=etb_label,
                    barrier_hit=BarrierHit.UPPER if etb_label == 1 else BarrierHit.LOWER,
                    time_to_barrier=5,
                    barrier_price=1.0580 if etb_label == 1 else 1.0460,
                    level_adjusted=False,
                    upper_barrier=1.0580,
                    lower_barrier=1.0460
                ),
                forward_return=forward_return
            ))
        
        result = validator.validate_batch_labels(label_sets)
        
        # Should detect high inconsistency
        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        inconsistency_warning = any(
            "High inconsistency between Enhanced Triple Barrier and forward returns" in issue.message
            for issue in warning_issues
        )
        assert inconsistency_warning
        
        # Check consistency metrics
        assert "cross_label_consistency" in result.metrics
        consistency_metrics = result.metrics["cross_label_consistency"]
        assert consistency_metrics["inconsistency_rate"] > 0.5  # Should be ~100%
    
    def test_batch_validation_performance_metrics(self, validator):
        """Test batch performance metrics validation"""
        
        # Create batch with performance issues
        label_sets = []
        base_time = datetime(2024, 1, 15, 1, 0, 0)
        
        computation_times = [100, 200, 500, 12000, 150, 300, 800, 15000, 250, 400]  # Some very slow
        
        for i, comp_time in enumerate(computation_times):
            label_sets.append(LabelSet(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=base_time + timedelta(hours=4 * i),
                computation_time_ms=comp_time
            ))
        
        result = validator.validate_batch_labels(label_sets)
        
        # Should detect performance issues
        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        slow_computation_warning = any(
            "Very slow computation detected in batch" in issue.message
            for issue in warning_issues
        )
        assert slow_computation_warning
        
        # Check performance metrics
        assert "batch_performance" in result.metrics
        perf_metrics = result.metrics["batch_performance"]
        assert perf_metrics["max_computation_time_ms"] == 15000
        assert perf_metrics["total_samples"] == 10
    
    # Path granularity validation tests
    
    def test_path_granularity_mapping_validation(self, validator, sample_candle):
        """Test path data granularity mapping validation"""
        
        # Create path data with wrong granularity (D instead of H1 for H4 candle)
        wrong_granularity_path = [
            {
                "ts": sample_candle.ts,
                "open": 1.0520, "high": 1.0540, "low": 1.0510, "close": 1.0535
            },
            {
                "ts": sample_candle.ts + timedelta(days=1),  # Daily increment instead of hourly
                "open": 1.0535, "high": 1.0560, "low": 1.0520, "close": 1.0550
            }
        ]
        
        result = validator.validate_post_computation(
            candle=sample_candle,
            label_set=LabelSet(
                instrument_id=sample_candle.instrument_id,
                granularity=sample_candle.granularity,
                ts=sample_candle.ts
            ),
            path_data=wrong_granularity_path
        )
        
        # Should warn about granularity mismatch
        warning_issues = result.get_issues_by_severity(ValidationSeverity.WARNING)
        granularity_warning = any(
            "Path data granularity may not match expected" in issue.message
            for issue in warning_issues
        )
        assert granularity_warning
    
    # Edge cases and error handling
    
    def test_validation_exception_handling(self, validator, sample_candle):
        """Test validation exception handling"""
        
        # Create label set that might cause calculation errors
        problematic_label_set = LabelSet(
            instrument_id=sample_candle.instrument_id,
            granularity=sample_candle.granularity,
            ts=sample_candle.ts,
            mfe=float('inf'),     # Infinite value
            mae=float('-inf'),    # Infinite value
            profit_factor=float('nan')  # NaN value
        )
        
        # Should handle gracefully without crashing
        result = validator.validate_post_computation(sample_candle, problematic_label_set)
        
        # May not be valid due to infinite/NaN values, but shouldn't crash
        assert isinstance(result, ValidationResult)
    
    def test_strict_mode_validator(self, strict_validator, sample_candle):
        """Test strict mode validator behavior"""
        
        # In strict mode, warnings should become errors
        candle_with_issues = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=1.0500,
            high=1.0580,
            low=1.0450,
            close=1.0520,
            volume=1000.0,
            bid=1.0519,
            ask=1.0521
        )
        
        # Test with very large horizon (normally warning)
        result = strict_validator.validate_pre_computation(candle_with_issues, 150)
        
        # In strict mode, may have more severe validation
        assert isinstance(result, ValidationResult)
    
    # Statistical test edge cases
    
    def test_jarque_bera_with_insufficient_data(self, validator):
        """Test Jarque-Bera test with insufficient data"""
        
        # Small sample size (< 20)
        small_batch = []
        base_time = datetime(2024, 1, 15, 1, 0, 0)
        
        for i in range(5):  # Only 5 samples
            small_batch.append(LabelSet(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=base_time + timedelta(hours=4 * i),
                forward_return=0.01 * i
            ))
        
        result = validator.validate_batch_labels(small_batch, statistical_tests=True)
        
        # Should handle gracefully, no statistical tests performed
        assert result.is_valid
        # No statistical distribution metrics for small samples
        assert "forward_returns_stats" not in result.metrics
    
    def test_validator_stats_tracking(self, validator, sample_candle):
        """Test validator statistics tracking"""
        
        # Perform several validations
        for i in range(5):
            result = validator.validate_pre_computation(sample_candle, 6)
            
        stats = validator.get_validation_stats()
        
        assert stats["total_validations"] == 5
        assert stats["success_rate"] >= 0.0
        assert "avg_validation_time_ms" in stats
    
    def test_alerting_summary_creation(self, validator, sample_candle):
        """Test alerting summary creation"""
        
        # Create validation result with issues
        invalid_candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 8, 0, 0),  # Invalid H4 timestamp
            open=-1.0500,  # Negative price
            high=1.0580,
            low=1.0450,
            close=1.0520,
            volume=1000.0
        )
        
        result = validator.validate_pre_computation(invalid_candle, 6)
        alert_summary = validator.create_alerting_summary(result)
        
        assert alert_summary["alert_level"] in ["red", "yellow", "orange", "green"]
        assert alert_summary["is_valid"] == result.is_valid
        assert alert_summary["total_issues"] == len(result.issues)
        assert "top_issues" in alert_summary
        assert "timestamp" in alert_summary
        
        # Should be red alert due to critical/error issues
        if not result.is_valid:
            assert alert_summary["alert_level"] in ["red", "yellow"]
    
    # Performance tests
    
    @pytest.mark.performance
    def test_validation_performance_single(self, validator, performance_timer, sample_candle, valid_label_set):
        """Test single validation performance"""
        
        performance_timer.start()
        
        for _ in range(100):  # 100 validations
            validator.validate_post_computation(sample_candle, valid_label_set)
        
        elapsed_ms = performance_timer.stop()
        
        # Should complete 100 validations quickly (< 1 second)
        assert elapsed_ms < 1000, f"100 validations took {elapsed_ms:.2f}ms (too slow)"
        
        # Average time per validation
        avg_time_per_validation = elapsed_ms / 100
        assert avg_time_per_validation < 10, f"Average validation time {avg_time_per_validation:.2f}ms (too slow)"
    
    @pytest.mark.performance  
    def test_batch_validation_performance(self, validator, performance_timer):
        """Test batch validation performance"""
        
        # Create large batch
        label_sets = []
        base_time = datetime(2024, 1, 15, 1, 0, 0)
        
        for i in range(1000):  # 1000 label sets
            label_sets.append(LabelSet(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=base_time + timedelta(hours=4 * i),
                forward_return=np.random.normal(0, 0.01),
                vol_scaled_return=np.random.normal(0, 1),
                computation_time_ms=100 + np.random.normal(0, 20)
            ))
        
        performance_timer.start()
        
        result = validator.validate_batch_labels(label_sets, statistical_tests=True)
        
        elapsed_ms = performance_timer.stop()
        
        # Batch validation should complete within reasonable time
        assert elapsed_ms < 5000, f"Batch validation of 1000 items took {elapsed_ms:.2f}ms (too slow)"
        assert result.validation_time_ms < 5000
    
    # Integration test with real computation engine
    
    @pytest.mark.integration
    def test_integration_with_label_computation(self, validator):
        """Test integration with actual label computation"""
        
        # This test would integrate with the actual LabelComputationEngine
        # For unit tests, we'll mock this interaction
        
        with patch('src.core.label_computation.LabelComputationEngine') as mock_engine:
            mock_engine.compute_labels.return_value = LabelSet(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=datetime(2024, 1, 15, 9, 0, 0)
            )
            
            # Simulate integration validation
            candle = Candle(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=datetime(2024, 1, 15, 9, 0, 0),
                open=1.0500, high=1.0580, low=1.0450, close=1.0520,
                volume=1000.0
            )
            
            pre_result = validator.validate_pre_computation(candle, 6)
            assert pre_result.is_valid
            
            # Would call label computation here in real integration
            label_set = mock_engine.compute_labels()
            
            post_result = validator.validate_post_computation(candle, label_set)
            assert isinstance(post_result, ValidationResult)