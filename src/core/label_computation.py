"""
Core label computation engine

Implements all label types including Enhanced Triple Barrier (Label 11.a)
with support/resistance level adjustments for high-frequency trading models.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.models.data_models import (
    Candle, Level, LabelSet, EnhancedTripleBarrierLabel,
    BarrierHit, Granularity, LevelType
)
from src.services.clickhouse_service import clickhouse_service
from src.services.redis_cache import redis_cache
from src.utils.timestamp_aligner import TimestampAligner

logger = logging.getLogger(__name__)


class LabelComputationEngine:
    """
    Core engine for computing various label types for quantitative trading.
    
    Implements Label 11.a (Enhanced Triple Barrier) as the primary label type
    with support/resistance level integration for improved precision.
    """
    
    def __init__(self):
        self.default_horizon_periods = 6
        self.default_barrier_width = 0.01  # 1% default barrier width
    
    async def compute_labels(
        self,
        candle: Candle,
        horizon_periods: int = 6,
        label_types: Optional[List[str]] = None,
        use_cache: bool = True,
        force_recompute: bool = False
    ) -> LabelSet:
        """
        Compute all requested labels for a single candle.
        
        Args:
            candle: Market candle data
            horizon_periods: Forward-looking horizon in periods
            label_types: Specific label types to compute (all if None)
            use_cache: Whether to use cached results
            force_recompute: Force recomputation even if cached
            
        Returns:
            Complete label set for the candle
        """
        start_time = datetime.utcnow()
        
        # Check cache first
        if use_cache and not force_recompute:
            cached_labels = redis_cache.get_labels(
                candle.instrument_id, 
                candle.granularity.value, 
                candle.ts
            )
            if cached_labels:
                logger.debug(f"Cache hit for {candle.instrument_id} {candle.ts}")
                return LabelSet(**cached_labels)
        
        # Initialize label set
        label_set = LabelSet(
            instrument_id=candle.instrument_id,
            granularity=candle.granularity,
            ts=candle.ts
        )
        
        # Determine which labels to compute
        if label_types is None:
            label_types = ["enhanced_triple_barrier", "vol_scaled_return", "mfe_mae"]
        
        # Get horizon end timestamp
        horizon_end = TimestampAligner.get_horizon_end(
            candle.ts, candle.granularity.value, horizon_periods
        )
        
        # Compute Enhanced Triple Barrier (Label 11.a)
        if "enhanced_triple_barrier" in label_types:
            etb_label = await self._compute_enhanced_triple_barrier(
                candle, horizon_periods, horizon_end
            )
            label_set.enhanced_triple_barrier = etb_label
        
        # Compute other labels
        if "vol_scaled_return" in label_types:
            label_set.vol_scaled_return = await self._compute_vol_scaled_return(
                candle, horizon_end
            )
        
        if "mfe_mae" in label_types:
            mfe, mae = await self._compute_mfe_mae(candle, horizon_end)
            label_set.mfe = mfe
            label_set.mae = mae
            if mae != 0:
                label_set.profit_factor = abs(mfe / mae) if mae < 0 else mfe / abs(mae)
        
        # Compute basic metrics
        label_set.forward_return = await self._compute_forward_return(candle, horizon_end)
        
        # Set metadata
        computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        label_set.computation_time_ms = computation_time
        
        # Cache results
        if use_cache:
            redis_cache.cache_labels(
                candle.instrument_id,
                candle.granularity.value,
                candle.ts,
                label_set.dict()
            )
        
        return label_set
    
    async def _compute_enhanced_triple_barrier(
        self,
        candle: Candle,
        horizon_periods: int,
        horizon_end: datetime
    ) -> EnhancedTripleBarrierLabel:
        """
        Compute Enhanced Triple Barrier label (Label 11.a) with S/R level adjustments.
        
        This is the primary label for pattern mining and ML models.
        Uses dynamic barrier placement based on:
        1. ATR-based volatility scaling
        2. Support/Resistance level proximity
        3. Multi-timeframe path analysis
        
        Args:
            candle: Input candle
            horizon_periods: Forward horizon in periods
            horizon_end: End timestamp for horizon
            
        Returns:
            Enhanced triple barrier label with all metadata
        """
        entry_price = candle.close
        
        # Get ATR for dynamic barrier sizing
        atr = candle.atr_14 if candle.atr_14 else self._estimate_atr(candle)
        
        # Base barrier width (2x ATR as default)
        base_barrier_width = 2.0 * atr
        
        # Get active support/resistance levels
        active_levels = await self._get_active_levels(
            candle.instrument_id, candle.granularity, candle.ts
        )
        
        # Calculate initial barriers
        upper_barrier = entry_price + base_barrier_width
        lower_barrier = entry_price - base_barrier_width
        level_adjusted = False
        nearest_support = None
        nearest_resistance = None
        
        # Adjust barriers based on S/R levels
        if active_levels:
            support_levels = [l for l in active_levels if l["current_type"] == "support"]
            resistance_levels = [l for l in active_levels if l["current_type"] == "resistance"]
            
            # Find nearest levels
            if support_levels:
                support_prices = [l["price"] for l in support_levels if l["price"] < entry_price]
                if support_prices:
                    nearest_support = max(support_prices)
                    
            if resistance_levels:
                resistance_prices = [l["price"] for l in resistance_levels if l["price"] > entry_price]
                if resistance_prices:
                    nearest_resistance = min(resistance_prices)
            
            # Adjust barriers if levels are within barrier range
            if nearest_resistance and nearest_resistance < upper_barrier:
                # Resistance level is closer than upper barrier
                upper_barrier = nearest_resistance * 0.999  # Slight buffer to avoid false triggers
                level_adjusted = True
                
            if nearest_support and nearest_support > lower_barrier:
                # Support level is closer than lower barrier
                lower_barrier = nearest_support * 1.001  # Slight buffer to avoid false triggers
                level_adjusted = True
        
        # Get path granularity for accurate barrier checking
        path_granularity, multiplier = TimestampAligner.get_path_granularity(
            candle.granularity.value
        )
        
        # Fetch path data for barrier checking
        path_data = await self._get_path_data(
            candle.instrument_id,
            path_granularity,
            candle.ts,
            horizon_end
        )
        
        # Check barriers using path data
        barrier_hit, time_to_barrier, barrier_price = self._check_barriers_with_path(
            path_data, upper_barrier, lower_barrier, horizon_periods * multiplier
        )
        
        # Determine label value
        if barrier_hit == BarrierHit.UPPER:
            label = 1
        elif barrier_hit == BarrierHit.LOWER:
            label = -1
        else:
            label = 0
        
        return EnhancedTripleBarrierLabel(
            label=label,
            barrier_hit=barrier_hit,
            time_to_barrier=time_to_barrier,
            barrier_price=barrier_price,
            level_adjusted=level_adjusted,
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
            path_granularity=Granularity(path_granularity) if path_granularity != candle.granularity.value else None
        )
    
    async def _get_active_levels(
        self,
        instrument_id: str,
        granularity: Granularity,
        timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """Get active support/resistance levels at timestamp"""
        try:
            # Check cache first
            cached_levels = redis_cache.get_active_levels(instrument_id, granularity.value)
            if cached_levels:
                return cached_levels
            
            # Fetch from ClickHouse
            levels = clickhouse_service.fetch_active_levels(
                instrument_id, granularity.value, timestamp
            )
            
            # Cache for future use
            redis_cache.cache_active_levels(instrument_id, granularity.value, levels)
            
            return levels
        except Exception as e:
            logger.error(f"Failed to get active levels: {e}")
            return []
    
    async def _get_path_data(
        self,
        instrument_id: str,
        granularity: str,
        start_ts: datetime,
        end_ts: datetime
    ) -> List[Dict[str, Any]]:
        """Get path data for barrier checking"""
        try:
            # Check cache first
            cached_data = redis_cache.get_path_data(instrument_id, granularity, start_ts, end_ts)
            if cached_data:
                return cached_data
            
            # Fetch from ClickHouse
            path_data = clickhouse_service.fetch_snapshots(
                instrument_id, granularity, start_ts, end_ts
            )
            
            # Cache the data
            redis_cache.cache_path_data(instrument_id, granularity, start_ts, end_ts, path_data)
            
            return path_data
        except Exception as e:
            logger.error(f"Failed to get path data: {e}")
            return []
    
    def _check_barriers_with_path(
        self,
        path_data: List[Dict[str, Any]],
        upper_barrier: float,
        lower_barrier: float,
        max_periods: int
    ) -> Tuple[BarrierHit, int, Optional[float]]:
        """
        Check which barrier is hit first using path data.
        
        Returns:
            Tuple of (barrier_hit, time_to_barrier, barrier_price)
        """
        if not path_data:
            return BarrierHit.NONE, max_periods, None
        
        for i, candle_data in enumerate(path_data[:max_periods]):
            high = candle_data.get("high", 0)
            low = candle_data.get("low", 0)
            
            # Check upper barrier
            if high >= upper_barrier:
                return BarrierHit.UPPER, i + 1, upper_barrier
            
            # Check lower barrier
            if low <= lower_barrier:
                return BarrierHit.LOWER, i + 1, lower_barrier
        
        return BarrierHit.NONE, max_periods, None
    
    async def _compute_vol_scaled_return(
        self,
        candle: Candle,
        horizon_end: datetime
    ) -> Optional[float]:
        """Compute volatility-scaled return"""
        try:
            forward_return = await self._compute_forward_return(candle, horizon_end)
            if forward_return is None:
                return None
            
            # Use ATR for volatility scaling
            atr = candle.atr_14 if candle.atr_14 else self._estimate_atr(candle)
            if atr <= 0:
                return None
            
            # Scale return by volatility
            vol_scaled_return = forward_return / atr
            return vol_scaled_return
        except Exception as e:
            logger.error(f"Failed to compute vol scaled return: {e}")
            return None
    
    async def _compute_forward_return(
        self,
        candle: Candle,
        horizon_end: datetime
    ) -> Optional[float]:
        """Compute forward return over horizon"""
        try:
            # Get future candle data
            future_data = await self._get_path_data(
                candle.instrument_id,
                candle.granularity.value,
                candle.ts,
                horizon_end + timedelta(hours=4)  # Add buffer
            )
            
            if not future_data:
                return None
            
            # Find candle at horizon end (or closest)
            target_candle = None
            for data in future_data:
                if data["ts"] >= horizon_end:
                    target_candle = data
                    break
            
            if not target_candle:
                # Use last available candle
                target_candle = future_data[-1]
            
            # Calculate return
            entry_price = candle.close
            exit_price = target_candle["close"]
            
            forward_return = (exit_price - entry_price) / entry_price
            return forward_return
        except Exception as e:
            logger.error(f"Failed to compute forward return: {e}")
            return None
    
    async def _compute_mfe_mae(
        self,
        candle: Candle,
        horizon_end: datetime
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
        """
        try:
            path_data = await self._get_path_data(
                candle.instrument_id,
                candle.granularity.value,
                candle.ts,
                horizon_end
            )
            
            if not path_data:
                return None, None
            
            entry_price = candle.close
            mfe = 0.0  # Maximum favorable move
            mae = 0.0  # Maximum adverse move
            
            for data in path_data:
                high = data.get("high", entry_price)
                low = data.get("low", entry_price)
                
                # Calculate excursions
                favorable_excursion = (high - entry_price) / entry_price
                adverse_excursion = (low - entry_price) / entry_price
                
                # Update maximums
                if favorable_excursion > mfe:
                    mfe = favorable_excursion
                if adverse_excursion < mae:
                    mae = adverse_excursion
            
            return mfe, mae
        except Exception as e:
            logger.error(f"Failed to compute MFE/MAE: {e}")
            return None, None
    
    def _estimate_atr(self, candle: Candle) -> float:
        """Estimate ATR from candle data if not available"""
        # Simple estimation using high-low range
        range_estimate = abs(candle.high - candle.low) / candle.close
        return max(range_estimate, 0.001)  # Minimum 0.1% ATR
    
    async def compute_batch_labels(
        self,
        instrument_id: str,
        granularity: str,
        start_date: datetime,
        end_date: datetime,
        label_types: List[str],
        chunk_size: int = 10000,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Compute labels for a batch of candles.
        
        Used for backfill operations with high throughput requirements.
        
        Args:
            instrument_id: Instrument to process
            granularity: Time granularity
            start_date: Start date (inclusive)
            end_date: End date (exclusive)
            label_types: Label types to compute
            chunk_size: Candles per processing chunk
            force_recompute: Whether to recompute existing labels
            
        Returns:
            Batch processing results with statistics
        """
        logger.info(
            f"Starting batch label computation: {instrument_id} {granularity} "
            f"from {start_date} to {end_date}"
        )
        
        # Get all candles in date range
        snapshots = clickhouse_service.fetch_snapshots(
            instrument_id, granularity, start_date, end_date
        )
        
        total_candles = len(snapshots)
        processed_candles = 0
        successful_labels = 0
        errors = []
        
        # Process in chunks
        for i in range(0, total_candles, chunk_size):
            chunk = snapshots[i:i + chunk_size]
            
            for snapshot_data in chunk:
                try:
                    # Convert to Candle object
                    candle = Candle(
                        instrument_id=instrument_id,
                        granularity=Granularity(granularity),
                        **snapshot_data
                    )
                    
                    # Compute labels
                    label_set = await self.compute_labels(
                        candle,
                        label_types=label_types,
                        force_recompute=force_recompute
                    )
                    
                    # Store results in ClickHouse (implement as needed)
                    # await self._store_label_set(label_set)
                    
                    successful_labels += 1
                    
                except Exception as e:
                    error_msg = f"Failed to compute labels for {candle.ts}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                
                processed_candles += 1
            
            # Log progress
            progress = (processed_candles / total_candles) * 100
            logger.info(f"Batch progress: {progress:.1f}% ({processed_candles}/{total_candles})")
        
        return {
            "total_candles": total_candles,
            "processed_candles": processed_candles,
            "successful_labels": successful_labels,
            "errors": errors,
            "error_rate": len(errors) / max(processed_candles, 1)
        }


# Global computation engine instance
computation_engine = LabelComputationEngine()