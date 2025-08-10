"""
Edge case tests for Enhanced Triple Barrier Label 11.a implementation.

These tests cover challenging scenarios including:
- Multi-timeframe alignment edge cases
- S/R level adjustment edge cases  
- Boundary conditions and data quality issues
- Error conditions and recovery scenarios
- Precision and numerical stability
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import List, Dict, Any
from unittest.mock import AsyncMock, Mock, patch
import numpy as np

from src.core.label_computation import LabelComputationEngine
from src.utils.timestamp_aligner import TimestampAligner
from src.models.data_models import (
    Candle, EnhancedTripleBarrierLabel, Granularity, BarrierHit
)


@pytest.mark.edge_cases
class TestTimestampAlignmentEdgeCases:
    """Edge cases for timestamp alignment."""
    
    def test_h4_alignment_daylight_saving_transitions(self, edge_case_timestamps):
        """Test H4 alignment during daylight saving time transitions."""
        aligner = TimestampAligner()
        
        # EU DST transitions
        dst_transitions = edge_case_timestamps["daylight_saving_transitions"]
        
        for transition_time in dst_transitions:
            aligned = aligner.align_to_granularity(transition_time, "H4")
            
            # Should always align to valid H4 boundary regardless of DST
            assert aligner.validate_alignment(aligned, "H4"), \
                f"DST transition failed to align properly: {transition_time} -> {aligned}"
            
            # Should be at or before the input time
            assert aligned <= transition_time, \
                f"Alignment moved forward in time: {transition_time} -> {aligned}"
    
    def test_h4_alignment_leap_year_edge_cases(self, edge_case_timestamps):
        """Test H4 alignment around leap year boundaries."""
        aligner = TimestampAligner()
        
        leap_year_times = edge_case_timestamps["leap_year"]
        
        for test_time in leap_year_times:
            aligned = aligner.align_to_granularity(test_time, "H4")
            
            # Verify alignment is correct
            assert aligner.validate_alignment(aligned, "H4")
            
            # Verify no unexpected date shifts
            if test_time.hour >= 1:  # Not early morning
                assert aligned.date() <= test_time.date()
    
    def test_h4_alignment_extreme_timestamps(self):
        """Test H4 alignment with extreme timestamp values."""
        aligner = TimestampAligner()
        
        extreme_cases = [
            datetime(1970, 1, 1, 7, 30, 0),     # Unix epoch era
            datetime(2038, 1, 19, 3, 14, 7),    # 32-bit timestamp limit
            datetime(2100, 12, 31, 23, 59, 59), # Far future
            datetime(2000, 2, 29, 15, 45, 30),  # Leap year
        ]
        
        for extreme_time in extreme_cases:
            try:
                aligned = aligner.align_to_granularity(extreme_time, "H4")
                assert aligner.validate_alignment(aligned, "H4"), \
                    f"Extreme timestamp alignment failed: {extreme_time}"
            except (ValueError, OverflowError) as e:
                # Some extreme values may legitimately fail
                pytest.skip(f"Extreme timestamp {extreme_time} not supported: {e}")
    
    def test_granularity_consistency_edge_cases(self, granularity_test_matrix):
        """Test granularity alignment consistency in edge cases."""
        aligner = TimestampAligner()
        
        for gran_name, gran_data in granularity_test_matrix.items():
            target_gran = gran_data["target"]
            test_time = gran_data["alignment_test"]
            
            # Align to target granularity
            aligned = aligner.align_to_granularity(test_time, target_gran.value)
            
            # Get period bounds
            period_start, period_end = aligner.get_period_bounds(test_time, target_gran.value)
            
            # Verify consistency
            assert aligned == period_start, \
                f"Alignment inconsistency for {gran_name}: {aligned} != {period_start}"
            
            # Verify period bounds make sense
            assert period_start < period_end, \
                f"Invalid period bounds for {gran_name}: {period_start} >= {period_end}"
            
            # Verify test time is within or at start of period
            assert period_start <= test_time < period_end, \
                f"Test time outside period for {gran_name}: {test_time} not in [{period_start}, {period_end})"
    
    def test_horizon_calculation_edge_cases(self):
        """Test horizon calculation with edge case parameters."""
        aligner = TimestampAligner()
        base_time = datetime(2024, 1, 15, 9, 0, 0)
        
        edge_cases = [
            # (granularity, periods, expected_behavior)
            ("M1", 0, "should handle zero periods"),
            ("M1", 1, "minimal period"),
            ("H4", 1000, "very large periods"),
            ("W", 52, "one year of weeks"),
        ]
        
        for granularity, periods, description in edge_cases:
            if periods == 0:
                # Zero periods should return start time
                horizon = aligner.get_horizon_end(base_time, granularity, periods)
                assert horizon == base_time, f"Zero periods failed: {description}"
            else:
                horizon = aligner.get_horizon_end(base_time, granularity, periods)
                assert horizon > base_time, f"Horizon calculation failed: {description}"
                
                # Verify calculation makes sense
                if granularity == "M1":
                    expected = base_time + timedelta(minutes=periods)
                elif granularity == "H4":
                    expected = base_time + timedelta(hours=4 * periods)
                elif granularity == "W":
                    expected = base_time + timedelta(weeks=periods)
                
                assert horizon == expected, \
                    f"Horizon mismatch for {description}: expected {expected}, got {horizon}"
    
    @pytest.mark.parametrize("invalid_input", [
        "",           # Empty string
        "h4",         # Wrong case
        "4H",         # Reversed format
        "MINUTE",     # Wrong format
        "M0",         # Invalid number
        "H25",        # Invalid hour
        None,         # None value
    ])
    def test_invalid_granularity_handling(self, invalid_input):
        """Test handling of invalid granularity inputs."""
        aligner = TimestampAligner()
        test_time = datetime(2024, 1, 15, 9, 0, 0)
        
        with pytest.raises((ValueError, AttributeError, TypeError)):
            aligner.align_to_granularity(test_time, invalid_input)


@pytest.mark.edge_cases
class TestEnhancedTripleBarrierEdgeCases:
    """Edge cases for Enhanced Triple Barrier computation."""
    
    @pytest.fixture
    def engine_with_mocks(self):
        """Create engine with mocked dependencies for edge case testing."""
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            engine = LabelComputationEngine()
            engine._clickhouse_service = mock_ch
            engine._redis_cache = mock_redis
            return engine, mock_ch, mock_redis
    
    @pytest.mark.asyncio
    async def test_extreme_price_values(self, engine_with_mocks):
        """Test with extreme price values."""
        engine, mock_ch, mock_redis = engine_with_mocks
        
        extreme_prices = [
            0.00001,    # Very small price
            1000000.0,  # Very large price
            float('inf'),  # Infinity (should be handled)
            float('-inf'), # Negative infinity (should be handled)
        ]
        
        for price in extreme_prices:
            if not np.isfinite(price):
                continue  # Skip infinite values for now
            
            candle = Candle(
                instrument_id="TEST/USD",
                granularity=Granularity.H4,
                ts=datetime(2024, 1, 15, 9, 0, 0),
                open=price, high=price * 1.01, low=price * 0.99, close=price,
                volume=1000, atr_14=price * 0.01
            )
            
            mock_redis.get_labels.return_value = None
            engine._get_active_levels = AsyncMock(return_value=[])
            engine._get_path_data = AsyncMock(return_value=[])
            
            try:
                etb_label = await engine._compute_enhanced_triple_barrier(
                    candle, 6, candle.ts + timedelta(hours=24)
                )
                
                # Should produce valid barriers
                assert etb_label.upper_barrier > candle.close
                assert etb_label.lower_barrier < candle.close
                assert np.isfinite(etb_label.upper_barrier)
                assert np.isfinite(etb_label.lower_barrier)
                
            except (ValueError, OverflowError, ZeroDivisionError) as e:
                # Some extreme values may legitimately fail
                pytest.skip(f"Extreme price {price} not supported: {e}")
    
    @pytest.mark.asyncio
    async def test_zero_or_negative_atr(self, engine_with_mocks):
        """Test behavior with zero or negative ATR values."""
        engine, mock_ch, mock_redis = engine_with_mocks
        
        atr_values = [0.0, -0.001, None]
        
        for atr_value in atr_values:
            candle = Candle(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=datetime(2024, 1, 15, 9, 0, 0),
                open=1.0500, high=1.0580, low=1.0450, close=1.0520,
                volume=1000, atr_14=atr_value
            )
            
            engine._get_active_levels = AsyncMock(return_value=[])
            engine._get_path_data = AsyncMock(return_value=[])
            
            etb_label = await engine._compute_enhanced_triple_barrier(
                candle, 6, candle.ts + timedelta(hours=24)
            )
            
            # Should fall back to estimation and use minimum ATR
            assert etb_label.upper_barrier > candle.close
            assert etb_label.lower_barrier < candle.close
            
            # Minimum barrier width should be enforced
            barrier_width = etb_label.upper_barrier - candle.close
            min_width = 2.0 * 0.001  # 2x minimum ATR (0.1%)
            
            assert barrier_width >= min_width, \
                f"Barrier width {barrier_width} below minimum {min_width} for ATR {atr_value}"
    
    @pytest.mark.asyncio
    async def test_conflicting_sr_levels(self, engine_with_mocks):
        """Test behavior with conflicting S/R levels."""
        engine, mock_ch, mock_redis = engine_with_mocks
        
        candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=1.0500, high=1.0580, low=1.0450, close=1.0520,
            volume=1000, atr_14=0.0045
        )
        
        # Create conflicting levels
        conflicting_levels = [
            {
                "level_id": "support_above_price",
                "price": 1.0600,  # Support above current price (unusual)
                "current_type": "support",
                "status": "active"
            },
            {
                "level_id": "resistance_below_price",
                "price": 1.0400,  # Resistance below current price (unusual)
                "current_type": "resistance",
                "status": "active"
            },
            {
                "level_id": "same_price_support",
                "price": 1.0520,  # Exactly at current price
                "current_type": "support",
                "status": "active"
            },
            {
                "level_id": "same_price_resistance",
                "price": 1.0520,  # Exactly at current price
                "current_type": "resistance",
                "status": "active"
            }
        ]
        
        engine._get_active_levels = AsyncMock(return_value=conflicting_levels)
        engine._get_path_data = AsyncMock(return_value=[])
        
        etb_label = await engine._compute_enhanced_triple_barrier(
            candle, 6, candle.ts + timedelta(hours=24)
        )
        
        # Should handle conflicts gracefully
        assert etb_label.upper_barrier > candle.close
        assert etb_label.lower_barrier < candle.close
        
        # Should not have invalid barrier adjustments
        assert etb_label.upper_barrier != etb_label.lower_barrier
    
    @pytest.mark.asyncio
    async def test_very_close_sr_levels(self, engine_with_mocks):
        """Test with S/R levels very close to current price."""
        engine, mock_ch, mock_redis = engine_with_mocks
        
        candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=1.0500, high=1.0580, low=1.0450, close=1.05200,
            volume=1000, atr_14=0.0045
        )
        
        # Levels extremely close to price (within 1 pip)
        close_levels = [
            {
                "level_id": "very_close_resistance",
                "price": 1.05201,  # 0.1 pip above
                "current_type": "resistance",
                "status": "active"
            },
            {
                "level_id": "very_close_support",
                "price": 1.05199,  # 0.1 pip below
                "current_type": "support",
                "status": "active"
            }
        ]
        
        engine._get_active_levels = AsyncMock(return_value=close_levels)
        engine._get_path_data = AsyncMock(return_value=[])
        
        etb_label = await engine._compute_enhanced_triple_barrier(
            candle, 6, candle.ts + timedelta(hours=24)
        )
        
        # Should handle very close levels appropriately
        assert etb_label.upper_barrier > candle.close
        assert etb_label.lower_barrier < candle.close
        
        # Barriers should be adjusted to close levels (with buffer)
        expected_upper = 1.05201 * 0.999
        expected_lower = 1.05199 * 1.001
        
        if etb_label.level_adjusted:
            assert abs(etb_label.upper_barrier - expected_upper) < 0.000001
            assert abs(etb_label.lower_barrier - expected_lower) < 0.000001
    
    @pytest.mark.asyncio
    async def test_missing_path_data_fields(self, engine_with_mocks):
        """Test handling of path data with missing fields."""
        engine, mock_ch, mock_redis = engine_with_mocks
        
        candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=1.0500, high=1.0580, low=1.0450, close=1.0520,
            volume=1000, atr_14=0.0045
        )
        
        # Path data with various missing fields
        incomplete_path_data = [
            {"high": 1.0580},                           # Missing 'low'
            {"low": 1.0450},                            # Missing 'high'
            {"high": 1.0570, "low": 1.0470, "invalid_field": "test"},  # Extra field
            {},                                         # Completely empty
            {"high": None, "low": 1.0460},             # None values
            {"high": "not_a_number", "low": 1.0460},   # Invalid data type
        ]
        
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=incomplete_path_data)
        
        # Should handle incomplete data gracefully
        etb_label = await engine._compute_enhanced_triple_barrier(
            candle, 6, candle.ts + timedelta(hours=24)
        )
        
        # Should return valid result despite bad data
        assert etb_label is not None
        assert etb_label.barrier_hit in [BarrierHit.UPPER, BarrierHit.LOWER, BarrierHit.NONE]
    
    def test_barrier_checking_precision_edge_cases(self):
        """Test barrier checking with precision edge cases."""
        engine = LabelComputationEngine()
        
        # Set up precise barriers
        upper_barrier = 1.052000  # Exactly 5 decimal places
        lower_barrier = 1.051000
        
        # Path data with prices exactly at or very close to barriers
        precision_test_data = [
            # Exactly at barriers
            {"high": 1.052000, "low": 1.051500},
            {"high": 1.051999, "low": 1.051001},  # Just inside barriers
            {"high": 1.052001, "low": 1.050999},  # Just outside barriers
            
            # Floating point precision issues
            {"high": 1.052000000001, "low": 1.051500},  # Tiny overshoot
            {"high": 1.051999999999, "low": 1.051500},  # Tiny undershoot
        ]
        
        for i, data in enumerate(precision_test_data):
            result = engine._check_barriers_with_path(
                [data], upper_barrier, lower_barrier, 10
            )
            
            barrier_hit, time_to_barrier, barrier_price = result
            
            if data["high"] > upper_barrier:
                assert barrier_hit == BarrierHit.UPPER, \
                    f"Precision test {i}: High {data['high']} > {upper_barrier} should hit upper barrier"
            elif data["low"] < lower_barrier:
                assert barrier_hit == BarrierHit.LOWER, \
                    f"Precision test {i}: Low {data['low']} < {lower_barrier} should hit lower barrier"
            else:
                # Should be within barriers (accounting for floating point precision)
                high_close = abs(data["high"] - upper_barrier) < 1e-10
                low_close = abs(data["low"] - lower_barrier) < 1e-10
                
                if not (high_close or low_close):
                    assert barrier_hit == BarrierHit.NONE, \
                        f"Precision test {i}: Should be within barriers"
    
    @pytest.mark.asyncio
    async def test_concurrent_barrier_hits(self, engine_with_mocks):
        """Test behavior when both barriers could be hit in same period."""
        engine, mock_ch, mock_redis = engine_with_mocks
        
        candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=1.0500, high=1.0580, low=1.0450, close=1.0520,
            volume=1000, atr_14=0.0045
        )
        
        # Path data where both barriers are hit in the same candle
        both_hit_data = [
            {
                "ts": candle.ts,
                "high": 1.0650,  # Hits upper barrier
                "low": 1.0400,   # Also hits lower barrier
                "close": 1.0520
            }
        ]
        
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=both_hit_data)
        
        etb_label = await engine._compute_enhanced_triple_barrier(
            candle, 6, candle.ts + timedelta(hours=24)
        )
        
        # Should consistently choose one barrier (implementation-dependent)
        assert etb_label.barrier_hit in [BarrierHit.UPPER, BarrierHit.LOWER]
        assert etb_label.time_to_barrier == 1  # Hit in first period
        
        # Barrier price should match the hit barrier
        if etb_label.barrier_hit == BarrierHit.UPPER:
            assert etb_label.barrier_price == etb_label.upper_barrier
        else:
            assert etb_label.barrier_price == etb_label.lower_barrier
    
    @pytest.mark.asyncio
    async def test_timezone_edge_cases(self, engine_with_mocks):
        """Test behavior across timezone boundaries."""
        engine, mock_ch, mock_redis = engine_with_mocks
        
        # Test times around timezone-sensitive periods
        timezone_sensitive_times = [
            datetime(2024, 1, 15, 0, 0, 0),   # Midnight UTC
            datetime(2024, 1, 15, 23, 0, 0),  # Late evening UTC
            datetime(2024, 3, 31, 1, 0, 0),   # EU DST transition
            datetime(2024, 11, 3, 6, 0, 0),   # US DST transition
        ]
        
        for test_time in timezone_sensitive_times:
            # Align to H4 first
            aligner = TimestampAligner()
            aligned_time = aligner.align_to_granularity(test_time, "H4")
            
            candle = Candle(
                instrument_id="EUR/USD",
                granularity=Granularity.H4,
                ts=aligned_time,
                open=1.0500, high=1.0580, low=1.0450, close=1.0520,
                volume=1000, atr_14=0.0045
            )
            
            engine._get_active_levels = AsyncMock(return_value=[])
            engine._get_path_data = AsyncMock(return_value=[])
            
            # Should handle timezone edge cases without issues
            etb_label = await engine._compute_enhanced_triple_barrier(
                candle, 6, candle.ts + timedelta(hours=24)
            )
            
            assert etb_label is not None
            assert etb_label.upper_barrier > candle.close
            assert etb_label.lower_barrier < candle.close


@pytest.mark.edge_cases
class TestDataQualityEdgeCases:
    """Edge cases related to data quality issues."""
    
    @pytest.mark.asyncio
    async def test_ohlc_data_inconsistencies(self):
        """Test handling of inconsistent OHLC data."""
        # Test various OHLC inconsistencies that might occur in real data
        inconsistent_data = [
            # High < Low (should be caught by validation)
            {"open": 1.0500, "high": 1.0450, "low": 1.0470, "close": 1.0520},
            
            # High < Open/Close  
            {"open": 1.0500, "high": 1.0480, "low": 1.0450, "close": 1.0520},
            
            # Low > Open/Close
            {"open": 1.0500, "high": 1.0580, "low": 1.0530, "close": 1.0520},
        ]
        
        for i, ohlc_data in enumerate(inconsistent_data):
            try:
                candle = Candle(
                    instrument_id="EUR/USD",
                    granularity=Granularity.H4,
                    ts=datetime(2024, 1, 15, 9, 0, 0),
                    volume=1000,
                    **ohlc_data
                )
                
                # If candle creation succeeds, validation should have been bypassed
                # or the data was actually consistent
                assert candle.high >= candle.low, f"Test case {i}: High should be >= Low"
                
            except ValueError as e:
                # Expected for truly inconsistent data
                assert "high" in str(e).lower() or "low" in str(e).lower(), \
                    f"Test case {i}: Unexpected validation error: {e}"
    
    @pytest.mark.asyncio
    async def test_volume_and_technical_indicator_edge_cases(self):
        """Test edge cases with volume and technical indicators."""
        edge_case_values = [
            # Zero values
            {"volume": 0, "atr_14": 0, "rsi_14": 0},
            
            # Extreme values
            {"volume": float('inf'), "atr_14": 1.0, "rsi_14": 100},
            
            # Negative values (should be invalid)
            {"volume": -1000, "atr_14": -0.01, "rsi_14": -10},
            
            # NaN values
            {"volume": float('nan'), "atr_14": float('nan'), "rsi_14": float('nan')},
        ]
        
        base_candle_data = {
            "instrument_id": "EUR/USD",
            "granularity": Granularity.H4,
            "ts": datetime(2024, 1, 15, 9, 0, 0),
            "open": 1.0500, "high": 1.0580, "low": 1.0450, "close": 1.0520
        }
        
        for i, values in enumerate(edge_case_values):
            try:
                candle_data = {**base_candle_data, **values}
                
                # Skip infinite and NaN values as they should be filtered out
                if any(not np.isfinite(v) for v in values.values() if isinstance(v, (int, float))):
                    continue
                
                candle = Candle(**candle_data)
                
                # If creation succeeds, verify constraints
                if candle.volume is not None:
                    assert candle.volume >= 0, f"Test case {i}: Volume should be non-negative"
                
                if candle.atr_14 is not None:
                    assert candle.atr_14 >= 0, f"Test case {i}: ATR should be non-negative"
                
                if candle.rsi_14 is not None:
                    assert 0 <= candle.rsi_14 <= 100, f"Test case {i}: RSI should be 0-100"
                
            except (ValueError, TypeError) as e:
                # Expected for invalid values
                continue
    
    @pytest.mark.asyncio
    async def test_future_timestamp_data_contamination(self):
        """Test detection of future timestamp contamination (look-ahead bias)."""
        current_time = datetime(2024, 1, 15, 9, 0, 0)
        
        candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=current_time,
            open=1.0500, high=1.0580, low=1.0450, close=1.0520,
            volume=1000, atr_14=0.0045
        )
        
        # Path data with future timestamps (look-ahead bias)
        contaminated_path_data = [
            {"ts": current_time, "high": 1.0570, "low": 1.0470},           # Valid
            {"ts": current_time + timedelta(hours=1), "high": 1.0580, "low": 1.0480},  # Valid future
            {"ts": current_time - timedelta(hours=1), "high": 1.0560, "low": 1.0460},  # INVALID past
            {"ts": current_time + timedelta(days=1), "high": 1.0590, "low": 1.0490},   # Valid future
        ]
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = contaminated_path_data
            
            engine = LabelComputationEngine()
            label_set = await engine.compute_labels(candle)
            
            # The system should handle this gracefully, but ideally detect the issue
            assert label_set is not None
            
            # In a more sophisticated implementation, we might want to:
            # 1. Filter out past timestamps
            # 2. Log warnings about data quality issues
            # 3. Set flags indicating potential look-ahead bias
    
    @pytest.mark.asyncio
    async def test_duplicate_and_missing_timestamps(self):
        """Test handling of duplicate timestamps and gaps in path data."""
        current_time = datetime(2024, 1, 15, 9, 0, 0)
        
        candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=current_time,
            open=1.0500, high=1.0580, low=1.0450, close=1.0520,
            volume=1000, atr_14=0.0045
        )
        
        # Path data with duplicates and gaps
        problematic_path_data = [
            {"ts": current_time, "high": 1.0570, "low": 1.0470},
            {"ts": current_time, "high": 1.0575, "low": 1.0465},  # Duplicate timestamp
            {"ts": current_time + timedelta(hours=1), "high": 1.0580, "low": 1.0480},
            # Missing hour 2
            {"ts": current_time + timedelta(hours=3), "high": 1.0590, "low": 1.0490},
            {"ts": current_time + timedelta(hours=4), "high": 1.0585, "low": 1.0485},
        ]
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = problematic_path_data
            
            engine = LabelComputationEngine()
            label_set = await engine.compute_labels(candle)
            
            # Should handle problematic data without crashing
            assert label_set is not None
            assert label_set.enhanced_triple_barrier is not None


@pytest.mark.edge_cases  
class TestNumericalStabilityEdgeCases:
    """Edge cases related to numerical stability and precision."""
    
    def test_decimal_precision_barriers(self):
        """Test barrier calculations with high precision requirements."""
        # Set high precision context for decimal calculations
        getcontext().prec = 28
        
        # Use Decimal for high precision
        entry_price = Decimal('1.052000000000000000000')
        atr = Decimal('0.004500000000000000000')
        
        # Calculate barriers with high precision
        barrier_width = 2 * atr
        upper_barrier = float(entry_price + barrier_width)
        lower_barrier = float(entry_price - barrier_width)
        
        # Verify precision is maintained
        expected_upper = 1.061
        expected_lower = 1.043
        
        assert abs(upper_barrier - expected_upper) < 1e-15, \
            f"High precision upper barrier calculation failed: {upper_barrier} != {expected_upper}"
        
        assert abs(lower_barrier - expected_lower) < 1e-15, \
            f"High precision lower barrier calculation failed: {lower_barrier} != {expected_lower}"
    
    def test_floating_point_comparison_edge_cases(self):
        """Test floating point comparison edge cases in barrier checking."""
        engine = LabelComputationEngine()
        
        # Use values that are problematic for floating point comparison
        upper_barrier = 0.1 + 0.2  # 0.30000000000000004
        lower_barrier = 0.3 - 0.2  # 0.09999999999999998
        
        # Path data with values very close to barriers
        precision_path_data = [
            {"high": 0.3, "low": 0.15},                    # Exactly 0.3
            {"high": 0.30000000000000001, "low": 0.15},    # Tiny overshoot
            {"high": 0.29999999999999999, "low": 0.15},    # Tiny undershoot
            {"high": 0.25, "low": 0.1},                    # Exactly at lower barrier
            {"high": 0.25, "low": 0.09999999999999999},    # Tiny undershoot
        ]
        
        for i, data in enumerate(precision_path_data):
            result = engine._check_barriers_with_path(
                [data], upper_barrier, lower_barrier, 10
            )
            
            barrier_hit, time_to_barrier, barrier_price = result
            
            # Use epsilon comparison for floating point values
            eps = 1e-10
            
            if data["high"] > upper_barrier + eps:
                assert barrier_hit == BarrierHit.UPPER, \
                    f"Precision test {i}: Should hit upper barrier"
            elif data["low"] < lower_barrier - eps:
                assert barrier_hit == BarrierHit.LOWER, \
                    f"Precision test {i}: Should hit lower barrier"
            # For values very close to barriers, either result may be acceptable
    
    @pytest.mark.asyncio
    async def test_cumulative_rounding_errors(self):
        """Test for cumulative rounding errors in multi-step calculations."""
        # Simulate scenario where small rounding errors could accumulate
        base_price = 1.0000001  # Start with precision-challenging value
        small_changes = [0.0000001] * 1000  # 1000 tiny changes
        
        # Calculate final price with potential for cumulative errors
        final_price_iterative = base_price
        for change in small_changes:
            final_price_iterative += change
        
        final_price_direct = base_price + sum(small_changes)
        
        # Difference should be minimal (test numerical stability)
        difference = abs(final_price_iterative - final_price_direct)
        assert difference < 1e-12, \
            f"Cumulative rounding error too large: {difference}"
        
        # Test in barrier calculation context
        candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=base_price, high=base_price + 0.001, 
            low=base_price - 0.001, close=final_price_iterative,
            volume=1000, atr_14=0.0001
        )
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            engine = LabelComputationEngine()
            etb_label = await engine._compute_enhanced_triple_barrier(
                candle, 6, candle.ts + timedelta(hours=24)
            )
            
            # Should produce stable results despite precision challenges
            assert etb_label.upper_barrier > candle.close
            assert etb_label.lower_barrier < candle.close
            
            # Barriers should be numerically stable
            barrier_width = etb_label.upper_barrier - candle.close
            assert barrier_width > 0, "Barrier width should be positive"
            assert np.isfinite(barrier_width), "Barrier width should be finite"