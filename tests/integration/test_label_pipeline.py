"""
Integration tests for the complete Label 11.a Enhanced Triple Barrier pipeline.

These tests verify the end-to-end functionality including:
- Integration between TimestampAligner and LabelComputationEngine
- ClickHouse and Redis service integration
- Real data flow scenarios
- Multi-timeframe coordination
- Error propagation and recovery
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, Mock, patch
import numpy as np

from src.core.label_computation import LabelComputationEngine, computation_engine
from src.utils.timestamp_aligner import TimestampAligner
from src.models.data_models import (
    Candle, LabelSet, EnhancedTripleBarrierLabel,
    Granularity, BarrierHit
)


@pytest.mark.integration
class TestLabelComputationPipeline:
    """Integration tests for the complete label computation pipeline."""
    
    @pytest.fixture
    def realistic_candle_sequence(self) -> List[Candle]:
        """Generate realistic sequence of H4 candles."""
        candles = []
        base_time = datetime(2024, 1, 15, 1, 0, 0)  # Start with H4 aligned time
        base_price = 1.0500
        
        h4_hours = [1, 5, 9, 13, 17, 21]
        
        for day in range(3):  # 3 days of data
            for hour in h4_hours:
                ts = base_time + timedelta(days=day, hours=hour-1)
                
                # Simulate realistic price movement
                price_change = np.random.normal(0, 0.005)  # 0.5% volatility
                new_price = base_price + price_change
                
                # Create realistic OHLC
                volatility = 0.002 * (1 + 0.5 * np.random.random())
                high = new_price + volatility
                low = new_price - volatility
                open_price = base_price
                close_price = new_price
                
                candle = Candle(
                    instrument_id="EUR/USD",
                    granularity=Granularity.H4,
                    ts=ts,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close_price,
                    volume=1000 + np.random.normal(0, 200),
                    atr_14=0.0045 + np.random.normal(0, 0.001),
                    ema_20=new_price * (1 + np.random.normal(0, 0.01)),
                    ema_50=new_price * (1 + np.random.normal(0, 0.02)),
                    rsi_14=50 + np.random.normal(0, 15)
                )
                
                candles.append(candle)
                base_price = close_price
        
        return candles
    
    @pytest.fixture
    def realistic_h1_path_data(self) -> List[Dict[str, Any]]:
        """Generate realistic H1 path data for H4 candle period."""
        path_data = []
        base_time = datetime(2024, 1, 15, 9, 0, 0)  # H4 candle start
        base_price = 1.0520
        
        # Generate 24 H1 candles (6 H4 periods worth)
        for i in range(24):
            # Simulate intraperiod movement
            price_change = np.random.normal(0, 0.001)
            current_price = base_price + price_change
            
            volatility = 0.0005 * (1 + np.random.random())
            high = current_price + volatility
            low = current_price - volatility
            
            path_data.append({
                "ts": base_time + timedelta(hours=i),
                "open": base_price,
                "high": high,
                "low": low,
                "close": current_price,
                "volume": 250 + np.random.normal(0, 50)
            })
            
            base_price = current_price
        
        return path_data
    
    @pytest.fixture
    def realistic_sr_levels(self) -> List[Dict[str, Any]]:
        """Generate realistic support/resistance levels."""
        return [
            {
                "level_id": "weekly_support",
                "instrument_id": "EUR/USD",
                "granularity": "H4",
                "price": 1.0450,
                "created_at": datetime(2024, 1, 8, 9, 0, 0),
                "current_type": "support",
                "status": "active",
                "last_event_type": "NEW_SUPPORT",
                "last_event_at": datetime(2024, 1, 8, 9, 0, 0),
                "deactivated_at": None
            },
            {
                "level_id": "daily_resistance",
                "instrument_id": "EUR/USD",
                "granularity": "H4", 
                "price": 1.0580,
                "created_at": datetime(2024, 1, 12, 13, 0, 0),
                "current_type": "resistance",
                "status": "active",
                "last_event_type": "FLIP_TO_RESISTANCE",
                "last_event_at": datetime(2024, 1, 12, 13, 0, 0),
                "deactivated_at": None
            },
            {
                "level_id": "short_term_support",
                "instrument_id": "EUR/USD",
                "granularity": "H4",
                "price": 1.0490,
                "created_at": datetime(2024, 1, 14, 17, 0, 0),
                "current_type": "support", 
                "status": "active",
                "last_event_type": "TOUCH_DOWN",
                "last_event_at": datetime(2024, 1, 14, 21, 0, 0),
                "deactivated_at": None
            }
        ]
    
    # End-to-end pipeline tests
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_h4_candle(
        self, realistic_candle_sequence, realistic_h1_path_data, realistic_sr_levels
    ):
        """Test complete pipeline for H4 candle processing."""
        candle = realistic_candle_sequence[0]  # First candle
        
        # Mock services with realistic data
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Configure mocks
            mock_redis.get_labels.return_value = None  # Cache miss
            mock_ch.fetch_active_levels.return_value = realistic_sr_levels
            mock_ch.fetch_snapshots.return_value = realistic_h1_path_data
            
            # Process through pipeline
            engine = LabelComputationEngine()
            label_set = await engine.compute_labels(candle, horizon_periods=6)
            
            # Verify complete pipeline execution
            assert isinstance(label_set, LabelSet)
            assert label_set.enhanced_triple_barrier is not None
            assert label_set.instrument_id == candle.instrument_id
            assert label_set.granularity == candle.granularity
            assert label_set.ts == candle.ts
            
            # Verify Enhanced Triple Barrier computation
            etb = label_set.enhanced_triple_barrier
            assert isinstance(etb, EnhancedTripleBarrierLabel)
            assert etb.label in [-1, 0, 1]
            assert etb.barrier_hit in [BarrierHit.UPPER, BarrierHit.LOWER, BarrierHit.NONE]
            assert etb.upper_barrier > candle.close
            assert etb.lower_barrier < candle.close
            
            # Verify path granularity was used correctly
            assert etb.path_granularity == Granularity.H1 or etb.path_granularity is None
            
            # Verify service interactions
            mock_ch.fetch_active_levels.assert_called_once()
            mock_ch.fetch_snapshots.assert_called_once()
            mock_redis.cache_labels.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pipeline_with_level_adjustments(
        self, realistic_candle_sequence, realistic_h1_path_data, realistic_sr_levels
    ):
        """Test pipeline when S/R levels cause barrier adjustments."""
        candle = realistic_candle_sequence[0]
        
        # Create levels that should adjust barriers
        close_levels = [
            {
                "level_id": "close_resistance",
                "instrument_id": "EUR/USD",
                "granularity": "H4",
                "price": candle.close + 0.0030,  # Close resistance
                "created_at": candle.ts - timedelta(hours=4),
                "current_type": "resistance",
                "status": "active"
            },
            {
                "level_id": "close_support", 
                "instrument_id": "EUR/USD",
                "granularity": "H4",
                "price": candle.close - 0.0025,  # Close support
                "created_at": candle.ts - timedelta(hours=8),
                "current_type": "support",
                "status": "active"
            }
        ]
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = close_levels
            mock_ch.fetch_snapshots.return_value = realistic_h1_path_data
            
            engine = LabelComputationEngine()
            label_set = await engine.compute_labels(candle)
            
            etb = label_set.enhanced_triple_barrier
            
            # Verify levels caused adjustments
            assert etb.level_adjusted == True, "Close S/R levels should adjust barriers"
            
            # Verify barriers were adjusted to level prices (with buffer)
            resistance_price = candle.close + 0.0030
            support_price = candle.close - 0.0025
            
            assert abs(etb.upper_barrier - resistance_price * 0.999) < 0.00001
            assert abs(etb.lower_barrier - support_price * 1.001) < 0.00001
    
    @pytest.mark.asyncio
    async def test_pipeline_timestamp_alignment_integration(self):
        """Test integration between TimestampAligner and LabelComputationEngine."""
        # Test with non-aligned timestamp
        misaligned_time = datetime(2024, 1, 15, 7, 30, 45)  # Between H4 boundaries
        
        candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=misaligned_time,
            open=1.0500, high=1.0580, low=1.0450, close=1.0520,
            volume=1000, atr_14=0.0045
        )
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            engine = LabelComputationEngine()
            
            # Verify TimestampAligner is used correctly in horizon calculation
            with patch.object(TimestampAligner, 'get_horizon_end') as mock_horizon:
                mock_horizon.return_value = misaligned_time + timedelta(hours=24)
                
                label_set = await engine.compute_labels(candle, horizon_periods=6)
                
                # Verify TimestampAligner was called
                mock_horizon.assert_called_once_with(
                    candle.ts, candle.granularity.value, 6
                )
                
                assert label_set is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_multi_granularity_coordination(self):
        """Test pipeline coordination across multiple granularities."""
        # Test D (Daily) candle that should use H4 path data
        daily_candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.D,
            ts=datetime(2024, 1, 15, 0, 0, 0),  # Daily aligned
            open=1.0500, high=1.0600, low=1.0400, close=1.0550,
            volume=5000, atr_14=0.0080
        )
        
        # Generate H4 path data (6 H4 periods for daily horizon)
        h4_path_data = []
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        h4_hours = [1, 5, 9, 13, 17, 21]
        
        for i, hour in enumerate(h4_hours):
            h4_path_data.append({
                "ts": base_time + timedelta(hours=hour),
                "open": 1.0500 + i * 0.0010,
                "high": 1.0500 + i * 0.0010 + 0.0020,
                "low": 1.0500 + i * 0.0010 - 0.0020,
                "close": 1.0500 + i * 0.0010 + 0.0005,
                "volume": 800
            })
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = h4_path_data
            
            engine = LabelComputationEngine()
            label_set = await engine.compute_labels(daily_candle, horizon_periods=1)
            
            etb = label_set.enhanced_triple_barrier
            
            # Verify D candle used H4 path granularity
            assert etb.path_granularity == Granularity.H4 or etb.path_granularity is None
            
            # Verify ClickHouse was called with H4 granularity
            mock_ch.fetch_snapshots.assert_called_once()
            call_args = mock_ch.fetch_snapshots.call_args[0]
            assert call_args[1] == "H4", "Daily candle should use H4 path data"
    
    # Error handling and recovery tests
    
    @pytest.mark.asyncio
    async def test_pipeline_service_failure_recovery(self, realistic_candle_sequence):
        """Test pipeline recovery from service failures."""
        candle = realistic_candle_sequence[0]
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Simulate ClickHouse failure
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.side_effect = Exception("Database connection failed")
            mock_ch.fetch_snapshots.side_effect = Exception("Database connection failed")
            
            engine = LabelComputationEngine()
            
            # Should handle errors gracefully and return partial results
            label_set = await engine.compute_labels(candle)
            
            assert label_set is not None
            assert label_set.enhanced_triple_barrier is not None
            
            # Should use fallback behavior (no level adjustments, basic barriers)
            etb = label_set.enhanced_triple_barrier
            assert etb.level_adjusted == False
            assert etb.barrier_hit == BarrierHit.NONE  # No path data available
    
    @pytest.mark.asyncio
    async def test_pipeline_partial_data_scenarios(self, realistic_candle_sequence):
        """Test pipeline behavior with partial or missing data."""
        candle = realistic_candle_sequence[0]
        
        scenarios = [
            # Scenario 1: No S/R levels
            {"levels": [], "path_data": [], "description": "no_levels_no_path"},
            
            # Scenario 2: Levels but no path data
            {"levels": [{"price": 1.0580, "current_type": "resistance"}], 
             "path_data": [], "description": "levels_no_path"},
            
            # Scenario 3: Path data but no levels
            {"levels": [], "path_data": [{"high": 1.0560, "low": 1.0480}], 
             "description": "path_no_levels"},
            
            # Scenario 4: Incomplete path data
            {"levels": [], "path_data": [{"high": 1.0560}], # Missing 'low'
             "description": "incomplete_path"},
        ]
        
        for scenario in scenarios:
            with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
                 patch('src.core.label_computation.redis_cache') as mock_redis:
                
                mock_redis.get_labels.return_value = None
                mock_ch.fetch_active_levels.return_value = scenario["levels"]
                mock_ch.fetch_snapshots.return_value = scenario["path_data"]
                
                engine = LabelComputationEngine()
                label_set = await engine.compute_labels(candle)
                
                # Should always return valid label set
                assert label_set is not None, f"Failed for scenario: {scenario['description']}"
                assert label_set.enhanced_triple_barrier is not None
                
                etb = label_set.enhanced_triple_barrier
                assert etb.label in [-1, 0, 1]
                assert etb.upper_barrier > candle.close
                assert etb.lower_barrier < candle.close
    
    # Performance and caching integration tests
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_pipeline_cache_performance(self, realistic_candle_sequence):
        """Test pipeline performance with various cache scenarios."""
        candle = realistic_candle_sequence[0]
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Scenario 1: Cold cache (cache miss)
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = []
            
            engine = LabelComputationEngine()
            
            import time
            start_time = time.perf_counter()
            label_set1 = await engine.compute_labels(candle, use_cache=True)
            cold_cache_time = time.perf_counter() - start_time
            
            # Scenario 2: Warm cache (cache hit)
            cached_labels = {
                "enhanced_triple_barrier": {
                    "label": 1,
                    "barrier_hit": "upper",
                    "time_to_barrier": 5,
                    "barrier_price": 1.0580,
                    "level_adjusted": False,
                    "upper_barrier": 1.0580,
                    "lower_barrier": 1.0460,
                    "path_granularity": None
                },
                "instrument_id": candle.instrument_id,
                "granularity": candle.granularity.value,
                "ts": candle.ts
            }
            
            mock_redis.get_labels.return_value = cached_labels
            
            start_time = time.perf_counter()
            label_set2 = await engine.compute_labels(candle, use_cache=True)
            warm_cache_time = time.perf_counter() - start_time
            
            # Warm cache should be significantly faster
            assert warm_cache_time < cold_cache_time * 0.1, \
                f"Cache hit not fast enough: {warm_cache_time:.3f}s vs {cold_cache_time:.3f}s"
            
            # Results should be valid in both cases
            assert label_set1.enhanced_triple_barrier is not None
            assert label_set2.enhanced_triple_barrier is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_batch_processing(self, realistic_candle_sequence):
        """Test pipeline batch processing capabilities."""
        candles = realistic_candle_sequence[:5]  # Process 5 candles
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Mock batch data fetch
            batch_snapshots = []
            for candle in candles:
                batch_snapshots.append({
                    "ts": candle.ts,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "atr_14": candle.atr_14
                })
            
            mock_ch.fetch_snapshots.return_value = batch_snapshots
            mock_redis.get_labels.return_value = None  # Force computation
            
            engine = LabelComputationEngine()
            
            # Test batch computation
            result = await engine.compute_batch_labels(
                instrument_id="EUR/USD",
                granularity="H4",
                start_date=candles[0].ts,
                end_date=candles[-1].ts + timedelta(hours=4),
                label_types=["enhanced_triple_barrier"],
                chunk_size=3  # Small chunks for testing
            )
            
            # Verify batch results
            assert result["total_candles"] == len(candles)
            assert result["processed_candles"] == len(candles)
            assert result["successful_labels"] <= len(candles)  # May have errors
            assert result["error_rate"] <= 0.2  # Allow some errors in test
    
    # Real-world scenario tests
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_pipeline_weekend_gap_handling(self):
        """Test pipeline handling of weekend gaps in data."""
        # Friday 21:00 UTC (last H4 of week)
        friday_candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 12, 21, 0, 0),  # Friday
            open=1.0500, high=1.0580, low=1.0450, close=1.0520,
            volume=1000, atr_14=0.0045
        )
        
        # Monday 01:00 UTC (first H4 of week)
        monday_candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 1, 0, 0),  # Monday
            open=1.0530, high=1.0590, low=1.0480, close=1.0550,
            volume=1200, atr_14=0.0050
        )
        
        # Test both candles
        for candle, description in [(friday_candle, "Friday"), (monday_candle, "Monday")]:
            with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
                 patch('src.core.label_computation.redis_cache') as mock_redis:
                
                mock_redis.get_labels.return_value = None
                mock_ch.fetch_active_levels.return_value = []
                mock_ch.fetch_snapshots.return_value = []
                
                engine = LabelComputationEngine()
                label_set = await engine.compute_labels(candle)
                
                # Should handle weekend boundaries correctly
                assert label_set is not None, f"{description} candle processing failed"
                assert label_set.enhanced_triple_barrier is not None
                
                # Verify timestamp alignment is correct
                aligner = TimestampAligner()
                assert aligner.validate_alignment(candle.ts, "H4"), \
                    f"{description} candle timestamp not properly H4 aligned"
    
    @pytest.mark.asyncio
    async def test_pipeline_high_volatility_scenario(self):
        """Test pipeline during high volatility periods."""
        # Create high volatility candle
        high_vol_candle = Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=1.0500, high=1.0700, low=1.0300, close=1.0520,  # 4% range
            volume=5000, atr_14=0.0200  # Very high ATR
        )
        
        # Create path data showing extreme movement
        volatile_path_data = []
        base_time = high_vol_candle.ts
        prices = [1.0500, 1.0650, 1.0350, 1.0600, 1.0400, 1.0520]  # Volatile path
        
        for i, price in enumerate(prices):
            volatile_path_data.append({
                "ts": base_time + timedelta(hours=i),
                "open": prices[i-1] if i > 0 else 1.0500,
                "high": price + 0.0050,
                "low": price - 0.0050,
                "close": price,
                "volume": 1000
            })
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels.return_value = []
            mock_ch.fetch_snapshots.return_value = volatile_path_data
            
            engine = LabelComputationEngine()
            label_set = await engine.compute_labels(high_vol_candle)
            
            etb = label_set.enhanced_triple_barrier
            
            # Should handle high volatility appropriately
            assert etb is not None
            
            # Barriers should be wider due to high ATR
            barrier_width = etb.upper_barrier - high_vol_candle.close
            expected_width = 2.0 * high_vol_candle.atr_14  # 2x ATR
            
            assert abs(barrier_width - expected_width) < 0.001, \
                "High volatility should result in wider barriers"
            
            # May hit barriers due to extreme movement
            if etb.barrier_hit != BarrierHit.NONE:
                assert etb.time_to_barrier <= 6, "Should hit barriers within horizon"