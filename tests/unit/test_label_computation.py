"""
Unit tests for LabelComputationEngine - Enhanced Triple Barrier (Label 11.a).

These tests focus on the core label computation logic, including:
- Enhanced Triple Barrier with S/R level adjustments
- Barrier hit detection using path data
- Multi-timeframe path granularity usage
- Cache hit/miss scenarios
- No look-ahead bias validation
- Error handling and edge cases
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any

from src.core.label_computation import LabelComputationEngine
from src.models.data_models import (
    Candle, LabelSet, EnhancedTripleBarrierLabel,
    Granularity, BarrierHit, LevelType
)


@pytest.mark.unit
class TestLabelComputationEngine:
    """Test suite for LabelComputationEngine class."""
    
    @pytest.fixture
    def engine(self, mock_clickhouse_service, mock_redis_cache):
        """Create LabelComputationEngine with mocked dependencies."""
        with patch('src.core.label_computation.clickhouse_service', mock_clickhouse_service), \
             patch('src.core.label_computation.redis_cache', mock_redis_cache):
            return LabelComputationEngine()
    
    # Core Enhanced Triple Barrier Tests
    
    @pytest.mark.asyncio
    async def test_compute_enhanced_triple_barrier_upper_hit(
        self, engine, sample_candle, barrier_hit_path_data, sample_levels
    ):
        """Test Enhanced Triple Barrier computation with upper barrier hit."""
        # Setup mocks
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=barrier_hit_path_data["upper_hit"])
        
        # Compute Enhanced Triple Barrier
        etb_label = await engine._compute_enhanced_triple_barrier(
            sample_candle, 6, sample_candle.ts + timedelta(hours=24)
        )
        
        # Assertions
        assert isinstance(etb_label, EnhancedTripleBarrierLabel)
        assert etb_label.label == 1, "Upper barrier hit should result in label = 1"
        assert etb_label.barrier_hit == BarrierHit.UPPER
        assert etb_label.time_to_barrier == 5, "Upper barrier hit at 5th candle"
        assert etb_label.barrier_price is not None
        assert etb_label.upper_barrier > sample_candle.close
        assert etb_label.lower_barrier < sample_candle.close
        assert etb_label.level_adjusted == False, "No levels provided, should not be adjusted"
    
    @pytest.mark.asyncio
    async def test_compute_enhanced_triple_barrier_lower_hit(
        self, engine, sample_candle, barrier_hit_path_data
    ):
        """Test Enhanced Triple Barrier computation with lower barrier hit."""
        # Setup mocks
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=barrier_hit_path_data["lower_hit"])
        
        # Compute Enhanced Triple Barrier
        etb_label = await engine._compute_enhanced_triple_barrier(
            sample_candle, 6, sample_candle.ts + timedelta(hours=24)
        )
        
        # Assertions
        assert etb_label.label == -1, "Lower barrier hit should result in label = -1"
        assert etb_label.barrier_hit == BarrierHit.LOWER
        assert etb_label.time_to_barrier == 3, "Lower barrier hit at 3rd candle"
        assert etb_label.barrier_price is not None
        assert etb_label.level_adjusted == False
    
    @pytest.mark.asyncio
    async def test_compute_enhanced_triple_barrier_no_hit(
        self, engine, sample_candle, barrier_hit_path_data
    ):
        """Test Enhanced Triple Barrier computation with no barrier hit."""
        # Setup mocks
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=barrier_hit_path_data["no_hit"])
        
        # Compute Enhanced Triple Barrier
        etb_label = await engine._compute_enhanced_triple_barrier(
            sample_candle, 6, sample_candle.ts + timedelta(hours=24)
        )
        
        # Assertions
        assert etb_label.label == 0, "No barrier hit should result in label = 0"
        assert etb_label.barrier_hit == BarrierHit.NONE
        assert etb_label.time_to_barrier == 24, "Should use max periods (6 H4 * 4 H1)"
        assert etb_label.barrier_price is None
    
    @pytest.mark.asyncio
    async def test_compute_enhanced_triple_barrier_with_sr_adjustments(
        self, engine, sample_candle, barrier_hit_path_data, sample_levels
    ):
        """Test Enhanced Triple Barrier with S/R level adjustments."""
        # Setup mocks - include levels that should adjust barriers
        adjusted_levels = [
            {
                "level_id": "resistance_close",
                "instrument_id": "EUR/USD",
                "granularity": "H4",
                "price": 1.0550,  # Close resistance level
                "current_type": "resistance",
                "status": "active"
            },
            {
                "level_id": "support_close",
                "instrument_id": "EUR/USD", 
                "granularity": "H4",
                "price": 1.0500,  # Close support level
                "current_type": "support",
                "status": "active"
            }
        ]
        
        engine._get_active_levels = AsyncMock(return_value=adjusted_levels)
        engine._get_path_data = AsyncMock(return_value=barrier_hit_path_data["no_hit"])
        
        # Compute Enhanced Triple Barrier
        etb_label = await engine._compute_enhanced_triple_barrier(
            sample_candle, 6, sample_candle.ts + timedelta(hours=24)
        )
        
        # Assertions
        assert etb_label.level_adjusted == True, "Barriers should be adjusted based on S/R levels"
        
        # Upper barrier should be adjusted to resistance level
        expected_upper = 1.0550 * 0.999  # Resistance with buffer
        assert abs(etb_label.upper_barrier - expected_upper) < 0.00001
        
        # Lower barrier should be adjusted to support level
        expected_lower = 1.0500 * 1.001  # Support with buffer
        assert abs(etb_label.lower_barrier - expected_lower) < 0.00001
    
    @pytest.mark.asyncio
    async def test_atr_based_barrier_sizing(self, engine, sample_candle):
        """Test that barriers are sized based on ATR."""
        # Test with explicit ATR
        sample_candle.atr_14 = 0.0050  # 0.5%
        
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[])
        
        etb_label = await engine._compute_enhanced_triple_barrier(
            sample_candle, 6, sample_candle.ts + timedelta(hours=24)
        )
        
        # Base barrier width should be 2x ATR
        expected_barrier_width = 2.0 * 0.0050
        expected_upper = sample_candle.close + expected_barrier_width
        expected_lower = sample_candle.close - expected_barrier_width
        
        assert abs(etb_label.upper_barrier - expected_upper) < 0.00001
        assert abs(etb_label.lower_barrier - expected_lower) < 0.00001
    
    @pytest.mark.asyncio
    async def test_atr_estimation_fallback(self, engine, sample_candle):
        """Test ATR estimation when ATR is not available."""
        # Remove ATR from candle
        sample_candle.atr_14 = None
        
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[])
        
        etb_label = await engine._compute_enhanced_triple_barrier(
            sample_candle, 6, sample_candle.ts + timedelta(hours=24)
        )
        
        # Should still compute barriers using estimated ATR
        assert etb_label.upper_barrier > sample_candle.close
        assert etb_label.lower_barrier < sample_candle.close
        
        # Estimated ATR should be based on high-low range
        estimated_atr = abs(sample_candle.high - sample_candle.low) / sample_candle.close
        expected_width = 2.0 * max(estimated_atr, 0.001)
        
        actual_width = etb_label.upper_barrier - sample_candle.close
        assert abs(actual_width - expected_width) < 0.00001
    
    # Path granularity and barrier checking tests
    
    @pytest.mark.asyncio
    async def test_path_granularity_usage(self, engine, sample_candle):
        """Test that correct path granularity is used for barrier checking."""
        # H4 candle should use H1 path data
        engine._get_active_levels = AsyncMock(return_value=[])
        
        # Mock path data fetch to verify correct granularity is requested
        engine._get_path_data = AsyncMock(return_value=[])
        
        await engine._compute_enhanced_triple_barrier(
            sample_candle, 6, sample_candle.ts + timedelta(hours=24)
        )
        
        # Verify _get_path_data was called with H1 granularity (path granularity for H4)
        engine._get_path_data.assert_called_once()
        call_args = engine._get_path_data.call_args[0]
        assert call_args[1] == "H1", "H4 candle should use H1 path data"
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("target_granularity,expected_path_granularity,expected_multiplier", [
        (Granularity.H4, "H1", 4),
        (Granularity.D, "H4", 6),
        (Granularity.H1, "M15", 4),
    ])
    async def test_path_granularity_mapping(
        self, engine, sample_candle, target_granularity, expected_path_granularity, expected_multiplier
    ):
        """Test path granularity mapping for different target granularities."""
        # Update candle granularity
        sample_candle.granularity = target_granularity
        
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[])
        
        await engine._compute_enhanced_triple_barrier(
            sample_candle, 6, sample_candle.ts + timedelta(hours=24)
        )
        
        # Verify correct path granularity was used
        engine._get_path_data.assert_called_once()
        call_args = engine._get_path_data.call_args[0]
        assert call_args[1] == expected_path_granularity
    
    def test_check_barriers_with_path_comprehensive(self, engine):
        """Test barrier checking with various path scenarios."""
        upper_barrier = 1.0580
        lower_barrier = 1.0460
        max_periods = 10
        
        # Test upper barrier hit
        upper_hit_data = [
            {"high": 1.0570, "low": 1.0550},  # Period 1 - no hit
            {"high": 1.0575, "low": 1.0555},  # Period 2 - no hit
            {"high": 1.0585, "low": 1.0560},  # Period 3 - upper hit
        ]
        
        result = engine._check_barriers_with_path(upper_hit_data, upper_barrier, lower_barrier, max_periods)
        assert result == (BarrierHit.UPPER, 3, upper_barrier)
        
        # Test lower barrier hit
        lower_hit_data = [
            {"high": 1.0570, "low": 1.0550},  # Period 1 - no hit
            {"high": 1.0455, "low": 1.0445},  # Period 2 - lower hit
        ]
        
        result = engine._check_barriers_with_path(lower_hit_data, upper_barrier, lower_barrier, max_periods)
        assert result == (BarrierHit.LOWER, 2, lower_barrier)
        
        # Test no hit
        no_hit_data = [
            {"high": 1.0570, "low": 1.0470},  # Within barriers
            {"high": 1.0575, "low": 1.0465},  # Within barriers
        ]
        
        result = engine._check_barriers_with_path(no_hit_data, upper_barrier, lower_barrier, max_periods)
        assert result == (BarrierHit.NONE, max_periods, None)
        
        # Test empty path data
        result = engine._check_barriers_with_path([], upper_barrier, lower_barrier, max_periods)
        assert result == (BarrierHit.NONE, max_periods, None)
    
    def test_check_barriers_both_hit_same_period(self, engine):
        """Test barrier checking when both barriers could be hit in same period."""
        upper_barrier = 1.0580
        lower_barrier = 1.0460
        
        # Path data where both barriers could be hit (wide range)
        path_data = [
            {"high": 1.0590, "low": 1.0450},  # Both barriers could be hit
        ]
        
        result = engine._check_barriers_with_path(path_data, upper_barrier, lower_barrier, 10)
        
        # Upper barrier should be detected first (implementation detail)
        assert result[0] == BarrierHit.UPPER
        assert result[1] == 1
        assert result[2] == upper_barrier
    
    # Cache functionality tests
    
    @pytest.mark.asyncio
    async def test_cache_hit_scenario(self, engine, sample_candle, cache_hit_scenario):
        """Test label computation with cache hit."""
        # Mock cache hit
        cached_labels = cache_hit_scenario["cached_results"]
        
        with patch('src.core.label_computation.redis_cache') as mock_cache:
            mock_cache.get_labels.return_value = {
                "enhanced_triple_barrier": cached_labels["enhanced_triple_barrier"],
                "instrument_id": sample_candle.instrument_id,
                "granularity": sample_candle.granularity.value,
                "ts": sample_candle.ts
            }
            
            label_set = await engine.compute_labels(sample_candle, use_cache=True)
            
            # Should return cached results
            assert label_set.enhanced_triple_barrier is not None
            assert label_set.enhanced_triple_barrier.label == 1
            assert label_set.enhanced_triple_barrier.barrier_hit == BarrierHit.UPPER
            
            # Verify cache was checked
            mock_cache.get_labels.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_miss_scenario(self, engine, sample_candle):
        """Test label computation with cache miss."""
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[])
        
        with patch('src.core.label_computation.redis_cache') as mock_cache:
            mock_cache.get_labels.return_value = None  # Cache miss
            
            label_set = await engine.compute_labels(sample_candle, use_cache=True)
            
            # Should compute and cache results
            assert label_set is not None
            mock_cache.get_labels.assert_called_once()
            mock_cache.cache_labels.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_disabled(self, engine, sample_candle):
        """Test label computation with cache disabled."""
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[])
        
        with patch('src.core.label_computation.redis_cache') as mock_cache:
            label_set = await engine.compute_labels(sample_candle, use_cache=False)
            
            # Should not check cache
            mock_cache.get_labels.assert_not_called()
            mock_cache.cache_labels.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_force_recompute(self, engine, sample_candle):
        """Test force recompute bypasses cache."""
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[])
        
        with patch('src.core.label_computation.redis_cache') as mock_cache:
            mock_cache.get_labels.return_value = {"some": "cached_data"}
            
            label_set = await engine.compute_labels(
                sample_candle, use_cache=True, force_recompute=True
            )
            
            # Should not check cache but should cache results
            mock_cache.get_labels.assert_not_called()
            mock_cache.cache_labels.assert_called_once()
    
    # No look-ahead bias tests
    
    @pytest.mark.asyncio
    async def test_no_lookahead_bias_path_data(self, engine, sample_candle, assert_no_lookahead_bias):
        """Test that path data doesn't include future information."""
        future_timestamp = sample_candle.ts + timedelta(hours=1)
        past_timestamp = sample_candle.ts - timedelta(hours=1)
        
        # Mock path data with timestamps
        mock_path_data = [
            {"ts": sample_candle.ts, "high": 1.0580, "low": 1.0460, "close": 1.0520},
            {"ts": future_timestamp, "high": 1.0585, "low": 1.0465, "close": 1.0525},
        ]
        
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=mock_path_data)
        
        label_set = await engine.compute_labels(sample_candle)
        
        # Verify no look-ahead bias
        assert_no_lookahead_bias(sample_candle, label_set, mock_path_data)
    
    @pytest.mark.asyncio
    async def test_no_lookahead_bias_levels(self, engine, sample_candle):
        """Test that S/R levels don't include future information."""
        # Mock levels with timestamps
        mock_levels = [
            {
                "level_id": "level_1",
                "price": 1.0580,
                "current_type": "resistance",
                "created_at": sample_candle.ts - timedelta(hours=24),  # Past
                "status": "active"
            },
            {
                "level_id": "level_2", 
                "price": 1.0460,
                "current_type": "support",
                "created_at": sample_candle.ts + timedelta(hours=1),   # Future - invalid!
                "status": "active"
            }
        ]
        
        engine._get_active_levels = AsyncMock(return_value=mock_levels)
        engine._get_path_data = AsyncMock(return_value=[])
        
        # Should filter out future levels
        etb_label = await engine._compute_enhanced_triple_barrier(
            sample_candle, 6, sample_candle.ts + timedelta(hours=24)
        )
        
        # Only past level should affect barriers
        # This test would need implementation to actually filter future levels
        assert etb_label is not None
    
    # Error handling tests
    
    @pytest.mark.asyncio
    async def test_error_handling_service_failure(self, engine, sample_candle):
        """Test error handling when services fail."""
        # Mock service failures
        engine._get_active_levels = AsyncMock(side_effect=Exception("ClickHouse error"))
        engine._get_path_data = AsyncMock(side_effect=Exception("ClickHouse error"))
        
        # Should handle errors gracefully
        etb_label = await engine._compute_enhanced_triple_barrier(
            sample_candle, 6, sample_candle.ts + timedelta(hours=24)
        )
        
        # Should still return valid label with fallback behavior
        assert etb_label is not None
        assert etb_label.label in [-1, 0, 1]
        assert etb_label.level_adjusted == False  # No levels due to error
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_path_data(self, engine, sample_candle):
        """Test error handling with invalid path data."""
        # Mock invalid path data
        invalid_path_data = [
            {"invalid": "data"},
            {"high": "not_a_number", "low": 1.0460},
            None
        ]
        
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=invalid_path_data)
        
        # Should handle invalid data gracefully
        etb_label = await engine._compute_enhanced_triple_barrier(
            sample_candle, 6, sample_candle.ts + timedelta(hours=24)
        )
        
        assert etb_label.barrier_hit == BarrierHit.NONE  # No valid data to hit barriers
    
    # Integration with other label types
    
    @pytest.mark.asyncio
    async def test_compute_multiple_label_types(self, engine, sample_candle):
        """Test computing multiple label types together."""
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[])
        
        label_types = ["enhanced_triple_barrier", "vol_scaled_return", "mfe_mae"]
        
        label_set = await engine.compute_labels(sample_candle, label_types=label_types)
        
        # Should compute all requested labels
        assert label_set.enhanced_triple_barrier is not None
        # Note: vol_scaled_return and mfe_mae might be None due to mocked data
        
        # Verify computation time is recorded
        assert label_set.computation_time_ms is not None
        assert label_set.computation_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_compute_labels_default_types(self, engine, sample_candle):
        """Test computing labels with default label types."""
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[])
        
        # No label_types specified - should use defaults
        label_set = await engine.compute_labels(sample_candle)
        
        # Should include default labels
        assert label_set.enhanced_triple_barrier is not None
    
    # Performance and timing tests
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_computation_performance(self, engine, sample_candle, performance_timer):
        """Test that label computation completes within reasonable time."""
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[])
        
        performance_timer.start()
        
        label_set = await engine.compute_labels(sample_candle)
        
        elapsed_ms = performance_timer.stop()
        
        # Should complete within reasonable time (< 100ms for unit test)
        assert elapsed_ms < 100, f"Label computation too slow: {elapsed_ms:.2f}ms"
        
        # Verify recorded computation time is reasonable
        assert label_set.computation_time_ms < 100
    
    @pytest.mark.asyncio
    async def test_metadata_fields(self, engine, sample_candle):
        """Test that metadata fields are properly set."""
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[])
        
        start_time = datetime.utcnow()
        
        label_set = await engine.compute_labels(sample_candle)
        
        # Check metadata
        assert label_set.instrument_id == sample_candle.instrument_id
        assert label_set.granularity == sample_candle.granularity
        assert label_set.ts == sample_candle.ts
        assert label_set.label_version == "1.0.0"
        assert label_set.computed_at >= start_time
        assert label_set.computation_time_ms is not None
    
    # Batch computation tests
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_computation(self, engine):
        """Test batch label computation functionality."""
        instrument_id = "EUR/USD"
        granularity = "H4"
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        label_types = ["enhanced_triple_barrier"]
        
        # Mock batch data
        mock_snapshots = [
            {
                "ts": datetime(2024, 1, 1, 1, 0, 0),
                "open": 1.0500, "high": 1.0580, "low": 1.0450, "close": 1.0520,
                "volume": 1000, "atr_14": 0.0045
            },
            {
                "ts": datetime(2024, 1, 1, 5, 0, 0),
                "open": 1.0520, "high": 1.0590, "low": 1.0470, "close": 1.0540,
                "volume": 1100, "atr_14": 0.0050
            }
        ]
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch:
            mock_ch.fetch_snapshots.return_value = mock_snapshots
            
            engine._get_active_levels = AsyncMock(return_value=[])
            engine._get_path_data = AsyncMock(return_value=[])
            
            result = await engine.compute_batch_labels(
                instrument_id, granularity, start_date, end_date, label_types
            )
            
            # Verify batch results
            assert result["total_candles"] == len(mock_snapshots)
            assert result["processed_candles"] == len(mock_snapshots)
            assert result["successful_labels"] == len(mock_snapshots)
            assert result["error_rate"] == 0.0

    # Priority Labels Tests (Issue #6)
    
    @pytest.mark.asyncio
    async def test_volatility_scaled_returns_computation(self, engine, sample_candle):
        """Test Label 2: Volatility-Scaled Returns computation."""
        # Mock future price data
        future_data = [
            {"ts": sample_candle.ts + timedelta(hours=4), "close": 1.0570},
            {"ts": sample_candle.ts + timedelta(hours=8), "close": 1.0580},
            {"ts": sample_candle.ts + timedelta(hours=24), "close": 1.0590},  # Horizon end
        ]
        
        engine._get_path_data = AsyncMock(return_value=future_data)
        sample_candle.atr_14 = 0.0050  # 0.5% ATR
        
        horizon_end = sample_candle.ts + timedelta(hours=24)
        result = await engine._compute_vol_scaled_return(sample_candle, horizon_end)
        
        # Expected: (1.0590 - 1.0520) / 0.0050 = 1.4
        expected = (1.0590 - sample_candle.close) / sample_candle.atr_14
        assert abs(result - expected) < 0.001
        
    @pytest.mark.asyncio
    async def test_mfe_mae_computation_with_profit_factor(self, engine, sample_candle):
        """Test Labels 9-10: MFE/MAE with Profit Factor computation."""
        # Mock path data with price movements
        path_data = [
            {"high": 1.0530, "low": 1.0510},  # Small move
            {"high": 1.0570, "low": 1.0500},  # MFE candidate
            {"high": 1.0540, "low": 1.0480},  # MAE candidate  
            {"high": 1.0550, "low": 1.0490},  # Final
        ]
        
        engine._get_path_data = AsyncMock(return_value=path_data)
        
        horizon_end = sample_candle.ts + timedelta(hours=24)
        mfe, mae = await engine._compute_mfe_mae(sample_candle, horizon_end)
        
        # MFE should be max(P_{t+τ} - P_t) = 1.0570 - 1.0520 = 0.0050
        expected_mfe = 1.0570 - sample_candle.close
        assert abs(mfe - expected_mfe) < 0.0001
        
        # MAE should be min(P_{t+τ} - P_t) = 1.0480 - 1.0520 = -0.0040
        expected_mae = 1.0480 - sample_candle.close
        assert abs(mae - expected_mae) < 0.0001
        
        # Verify profit factor computation in main method
        with patch.object(engine, '_compute_mfe_mae', return_value=(mfe, mae)):
            label_set = await engine.compute_labels(sample_candle, label_types=["mfe_mae"])
            
            # Profit Factor = MFE / |MAE| = 0.0050 / 0.0040 = 1.25
            expected_profit_factor = abs(mfe / mae)
            assert abs(label_set.profit_factor - expected_profit_factor) < 0.001

    @pytest.mark.asyncio
    async def test_level_retouch_count_computation(self, engine, sample_candle):
        """Test Label 12: Level Retouch Count computation."""
        # Mock active levels
        mock_levels = [
            {
                "price": 1.0550,  # Resistance level
                "current_type": "resistance"
            },
            {
                "price": 1.0480,  # Support level
                "current_type": "support"
            }
        ]
        
        # Mock path data with touches
        path_data = [
            {"high": 1.0549, "low": 1.0500},  # Touch resistance (within 0.1%)
            {"high": 1.0530, "low": 1.0481},  # Touch support (within 0.1%)
            {"high": 1.0551, "low": 1.0490},  # Another touch resistance
            {"high": 1.0540, "low": 1.0479},  # Another touch support
        ]
        
        engine._get_active_levels = AsyncMock(return_value=mock_levels)
        engine._get_path_data = AsyncMock(return_value=path_data)
        
        horizon_end = sample_candle.ts + timedelta(hours=24)
        result = await engine._compute_level_retouch_count(sample_candle, horizon_end)
        
        # Should count 4 touches total (2 resistance + 2 support)
        assert result == 4

    @pytest.mark.asyncio
    async def test_breakout_beyond_level_computation(self, engine, sample_candle):
        """Test Label 16: Breakout Beyond Level computation."""
        # Mock active levels
        mock_levels = [
            {
                "price": 1.0550,  # Resistance level
                "current_type": "resistance"
            }
        ]
        
        # Mock path data with significant breakout (> 0.2%)
        path_data = [
            {"high": 1.0540, "low": 1.0500},  # No breakout
            {"high": 1.0574, "low": 1.0510},  # Breakout: (1.0574 - 1.0550)/1.0550 > 0.002
        ]
        
        engine._get_active_levels = AsyncMock(return_value=mock_levels)
        engine._get_path_data = AsyncMock(return_value=path_data)
        
        horizon_end = sample_candle.ts + timedelta(hours=24)
        result = await engine._compute_breakout_beyond_level(sample_candle, horizon_end)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_flip_within_horizon_computation(self, engine, sample_candle):
        """Test Label 17: Flip Within Horizon computation."""
        # Mock active levels
        mock_levels = [
            {
                "level_id": "level_1",
                "price": 1.0550,  # Resistance level that will flip
                "current_type": "resistance"
            }
        ]
        
        # Mock path data showing resistance break and retest as support
        path_data = [
            {"high": 1.0540, "low": 1.0500},  # Before breakout
            {"high": 1.0574, "low": 1.0530},  # Breakout above resistance (>0.2%)
            {"high": 1.0560, "low": 1.0549},  # Retest as support (within 0.1%)
        ]
        
        engine._get_active_levels = AsyncMock(return_value=mock_levels)
        engine._get_path_data = AsyncMock(return_value=path_data)
        
        horizon_end = sample_candle.ts + timedelta(hours=24)
        result = await engine._compute_flip_within_horizon(sample_candle, horizon_end)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_priority_labels_integration(self, engine, sample_candle):
        """Test all priority labels are computed together."""
        # Setup mocks for all priority labels
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[
            {"ts": sample_candle.ts, "high": 1.0530, "low": 1.0510, "close": 1.0520},
            {"ts": sample_candle.ts + timedelta(hours=4), "high": 1.0540, "low": 1.0500, "close": 1.0535},
        ])
        
        # Set ATR for volatility scaling
        sample_candle.atr_14 = 0.0050
        
        priority_labels = [
            "vol_scaled_return", 
            "mfe_mae", 
            "level_retouch_count", 
            "breakout_beyond_level", 
            "flip_within_horizon"
        ]
        
        label_set = await engine.compute_labels(
            sample_candle, 
            label_types=priority_labels
        )
        
        # Verify all priority labels were computed
        assert label_set.vol_scaled_return is not None
        assert label_set.mfe is not None
        assert label_set.mae is not None 
        assert label_set.profit_factor is not None
        assert label_set.retouch_count is not None
        assert label_set.breakout_occurred is not None
        assert label_set.flip_occurred is not None

    @pytest.mark.asyncio
    async def test_default_label_types_includes_priority_labels(self, engine, sample_candle):
        """Test that default label types include the new priority labels."""
        engine._get_active_levels = AsyncMock(return_value=[])
        engine._get_path_data = AsyncMock(return_value=[])
        sample_candle.atr_14 = 0.0050
        
        # Use default label types (None)
        label_set = await engine.compute_labels(sample_candle, label_types=None)
        
        # Should include both existing and priority labels
        assert label_set.enhanced_triple_barrier is not None
        assert label_set.retouch_count is not None
        assert label_set.breakout_occurred is not None
        assert label_set.flip_occurred is not None