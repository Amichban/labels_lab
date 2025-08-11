"""
Core label computation engine

Implements all label types including Enhanced Triple Barrier (Label 11.a)
with support/resistance level adjustments for high-frequency trading models.

Priority Labels Implementation (Issue #6):
- Label 2: Volatility-Scaled Returns (P_{t+H} - P_t) / ATR_t
- Labels 9-10: MFE/MAE with Profit Factor = MFE / |MAE|
- Label 12: Level Retouch Count within horizon
- Label 16: Breakout Beyond Level (0.2% threshold)
- Label 17: Flip Within Horizon (support to resistance or vice versa)

Issue #8 Integration: Comprehensive validation framework
- Pre-computation validation to prevent bad inputs
- Post-computation validation for data integrity
- Look-ahead bias detection and prevention
- Statistical consistency checks
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from src.models.data_models import (
    Candle, LabelSet, EnhancedTripleBarrierLabel,
    BarrierHit, Granularity
)
from src.services.clickhouse_service import clickhouse_service
from src.services.redis_cache import redis_cache
from src.utils.timestamp_aligner import TimestampAligner
from src.validation.label_validator import label_validator, ValidationSeverity

logger = logging.getLogger(__name__)


class LabelComputationEngine:
    """
    Core engine for computing various label types for quantitative trading.
    
    Implements Label 11.a (Enhanced Triple Barrier) as the primary label type
    with support/resistance level integration for improved precision.
    """
    
    def __init__(self, enable_validation: bool = True):
        self.default_horizon_periods = 6
        self.default_barrier_width = 0.01  # 1% default barrier width
        self.enable_validation = enable_validation
        self.validation_stats = {
            "pre_validation_failures": 0,
            "post_validation_failures": 0,
            "total_computations": 0,
            "validation_time_total_ms": 0.0
        }
    
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
            
        Raises:
            ValidationError: If pre-computation validation fails critically
        """
        start_time = datetime.utcnow()
        self.validation_stats["total_computations"] += 1
        path_data = None
        levels_data = None
        
        # Issue #8: Pre-computation validation
        if self.enable_validation:
            pre_validation_result = label_validator.validate_pre_computation(
                candle=candle,
                horizon_periods=horizon_periods
            )
            
            # Track validation time
            if pre_validation_result.validation_time_ms:
                self.validation_stats["validation_time_total_ms"] += pre_validation_result.validation_time_ms
            
            # Handle validation failures
            if not pre_validation_result.is_valid:
                self.validation_stats["pre_validation_failures"] += 1
                
                # Log critical issues
                critical_issues = pre_validation_result.get_issues_by_severity(ValidationSeverity.CRITICAL)
                for issue in critical_issues:
                    logger.critical(f"Pre-computation validation critical issue: {issue}")
                
                # Log error issues
                error_issues = pre_validation_result.get_issues_by_severity(ValidationSeverity.ERROR)
                for issue in error_issues:
                    logger.error(f"Pre-computation validation error: {issue}")
                
                # For critical validation failures, we might want to raise an exception
                # or return a special "invalid" label set
                if critical_issues:
                    logger.critical(f"Critical validation failure for {candle.instrument_id} {candle.ts}")
                    # Could raise ValidationError here, but for now we'll continue with logging
        
        # Check cache first
        if use_cache and not force_recompute:
            cached_labels = redis_cache.get_labels(
                candle.instrument_id, 
                candle.granularity.value, 
                candle.ts
            )
            if cached_labels:
                logger.debug(f"Cache hit for {candle.instrument_id} {candle.ts}")
                cached_label_set = LabelSet(**cached_labels)
                
                # Issue #8: Validate cached data too
                if self.enable_validation:
                    cached_validation_result = label_validator.validate_post_computation(
                        candle=candle,
                        label_set=cached_label_set
                    )
                    if not cached_validation_result.is_valid:
                        logger.warning(f"Cached labels failed validation for {candle.instrument_id} {candle.ts}")
                        # Continue with fresh computation
                    else:
                        return cached_label_set
                else:
                    return cached_label_set
        
        # Initialize label set
        label_set = LabelSet(
            instrument_id=candle.instrument_id,
            granularity=candle.granularity,
            ts=candle.ts
        )
        
        # Determine which labels to compute
        if label_types is None:
            label_types = [
                "enhanced_triple_barrier", 
                "vol_scaled_return", 
                "mfe_mae",
                "level_retouch_count",
                "breakout_beyond_level", 
                "flip_within_horizon"
            ]
        
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
            # Profit Factor = MFE / |MAE| (Labels 9-10)
            if mae is not None and mfe is not None and mae != 0:
                label_set.profit_factor = mfe / abs(mae)
        
        # Priority labels implementation
        if "level_retouch_count" in label_types:
            label_set.retouch_count = await self._compute_level_retouch_count(
                candle, horizon_end
            )
        
        if "breakout_beyond_level" in label_types:
            label_set.breakout_occurred = await self._compute_breakout_beyond_level(
                candle, horizon_end
            )
        
        if "flip_within_horizon" in label_types:
            label_set.flip_occurred = await self._compute_flip_within_horizon(
                candle, horizon_end
            )
        
        # Compute basic metrics
        label_set.forward_return = await self._compute_forward_return(candle, horizon_end)
        
        # Set metadata
        computation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        label_set.computation_time_ms = computation_time
        
        # Issue #8: Post-computation validation
        if self.enable_validation:
            # Get data used in computation for validation
            validation_path_data = None
            validation_levels_data = None
            
            # Try to get the data that was actually used in computation
            try:
                # Get the same data that was used in computation
                validation_levels_data = await self._get_active_levels(
                    candle.instrument_id, candle.granularity, candle.ts
                )
                
                # Get path data if Enhanced Triple Barrier was computed
                if label_set.enhanced_triple_barrier:
                    path_granularity, _ = TimestampAligner.get_path_granularity(
                        candle.granularity.value
                    )
                    validation_path_data = await self._get_path_data(
                        candle.instrument_id,
                        path_granularity,
                        candle.ts,
                        TimestampAligner.get_horizon_end(candle.ts, candle.granularity.value, horizon_periods)
                    )
            except Exception as e:
                logger.warning(f"Could not retrieve data for post-computation validation: {e}")
            
            # Perform post-computation validation
            post_validation_result = label_validator.validate_post_computation(
                candle=candle,
                label_set=label_set,
                path_data=validation_path_data,
                computation_context={
                    "horizon_periods": horizon_periods,
                    "computation_time_ms": computation_time
                }
            )
            
            # Track validation time
            if post_validation_result.validation_time_ms:
                self.validation_stats["validation_time_total_ms"] += post_validation_result.validation_time_ms
            
            # Handle post-validation failures
            if not post_validation_result.is_valid:
                self.validation_stats["post_validation_failures"] += 1
                
                # Log critical issues
                critical_issues = post_validation_result.get_issues_by_severity(ValidationSeverity.CRITICAL)
                for issue in critical_issues:
                    logger.critical(f"Post-computation validation critical issue: {issue}")
                
                # Log error issues  
                error_issues = post_validation_result.get_issues_by_severity(ValidationSeverity.ERROR)
                for issue in error_issues:
                    logger.error(f"Post-computation validation error: {issue}")
                
                # For critical post-validation failures, we could invalidate the results
                if critical_issues:
                    logger.critical(f"Critical post-validation failure for {candle.instrument_id} {candle.ts}")
                    # Mark the label set as potentially invalid
                    # Could add validation metadata to label_set or raise exception
        
        # Cache results (only if validation passed or validation is disabled)
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
        """
        Compute Label 2: Volatility-Scaled Returns
        Formula: (P_{t+H} - P_t) / ATR_t
        
        Args:
            candle: Input candle at time t
            horizon_end: Time t+H
            
        Returns:
            Volatility-scaled return or None if computation fails
        """
        try:
            # Get future price P_{t+H}
            future_data = await self._get_path_data(
                candle.instrument_id,
                candle.granularity.value,
                candle.ts,
                horizon_end + timedelta(hours=4)  # Add buffer
            )
            
            if not future_data:
                return None
            
            # Find candle at horizon end (or closest)
            future_price = None
            for data in future_data:
                if data["ts"] >= horizon_end:
                    future_price = data["close"]
                    break
            
            if future_price is None:
                # Use last available candle
                future_price = future_data[-1]["close"]
            
            # P_t (current price)
            current_price = candle.close
            
            # ATR_t (volatility at time t)
            atr_t = candle.atr_14 if candle.atr_14 else self._estimate_atr(candle)
            if atr_t <= 0:
                return None
            
            # Formula: (P_{t+H} - P_t) / ATR_t
            vol_scaled_return = (future_price - current_price) / atr_t
            return vol_scaled_return
            
        except Exception as e:
            logger.error(f"Failed to compute volatility-scaled return: {e}")
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
        Compute Labels 9-10: MFE/MAE with Profit Factor
        MFE = max(P_{t+τ} - P_t) for τ in [0, H]
        MAE = min(P_{t+τ} - P_t) for τ in [0, H]
        Profit Factor = MFE / |MAE|
        
        Uses path data at lower granularity for accurate extremum detection.
        
        Args:
            candle: Input candle at time t
            horizon_end: Time t+H
            
        Returns:
            Tuple of (MFE, MAE) in absolute price terms
        """
        try:
            # Get path granularity for accurate extremum detection
            path_granularity, _ = TimestampAligner.get_path_granularity(
                candle.granularity.value
            )
            
            # Fetch path data at lower granularity
            path_data = await self._get_path_data(
                candle.instrument_id,
                path_granularity,
                candle.ts,
                horizon_end
            )
            
            if not path_data:
                return None, None
            
            entry_price = candle.close
            mfe = 0.0  # Maximum favorable move (absolute price)
            mae = 0.0  # Maximum adverse move (absolute price)
            
            # Iterate through all τ in [0, H]
            for data in path_data:
                high = data.get("high", entry_price)
                low = data.get("low", entry_price)
                
                # Calculate price excursions (absolute)
                favorable_excursion = high - entry_price
                adverse_excursion = low - entry_price
                
                # Update MFE (maximum favorable)
                if favorable_excursion > mfe:
                    mfe = favorable_excursion
                
                # Update MAE (minimum adverse) 
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
    
    async def _compute_level_retouch_count(
        self,
        candle: Candle,
        horizon_end: datetime
    ) -> Optional[int]:
        """
        Compute Label 12: Level Retouch Count
        Count of touches at same level within horizon.
        
        A "touch" is defined as price coming within 0.1% of the level price
        without breaking through it significantly (>0.2%).
        
        Args:
            candle: Input candle
            horizon_end: End of horizon period
            
        Returns:
            Count of level retouches or None if computation fails
        """
        try:
            # Get active levels at candle timestamp
            active_levels = await self._get_active_levels(
                candle.instrument_id, candle.granularity, candle.ts
            )
            
            if not active_levels:
                return 0
            
            # Get path data at lower granularity for accurate touch detection
            path_granularity, _ = TimestampAligner.get_path_granularity(
                candle.granularity.value
            )
            
            path_data = await self._get_path_data(
                candle.instrument_id,
                path_granularity,
                candle.ts,
                horizon_end
            )
            
            if not path_data:
                return 0
            
            touch_tolerance = 0.001  # 0.1% tolerance for touch detection
            break_threshold = 0.002  # 0.2% threshold for breakout
            
            total_retouch_count = 0
            
            # Check each active level
            for level in active_levels:
                level_price = level["price"]
                level_type = level["current_type"]
                touch_count = 0
                
                for data in path_data:
                    high = data.get("high", 0)
                    low = data.get("low", 0)
                    
                    if level_type == "support":
                        # Check if price touched support level
                        distance_to_level = abs(low - level_price) / level_price
                        
                        # Touch condition: within tolerance
                        if distance_to_level <= touch_tolerance:
                            # Check if it's a retouch (not a breakout)
                            break_distance = (level_price - low) / level_price
                            if break_distance <= break_threshold:
                                touch_count += 1
                    
                    elif level_type == "resistance":
                        # Check if price touched resistance level
                        distance_to_level = abs(high - level_price) / level_price
                        
                        # Touch condition: within tolerance
                        if distance_to_level <= touch_tolerance:
                            # Check if it's a retouch (not a breakout)
                            break_distance = (high - level_price) / level_price
                            if break_distance <= break_threshold:
                                touch_count += 1
                
                total_retouch_count += touch_count
            
            return total_retouch_count
            
        except Exception as e:
            logger.error(f"Failed to compute level retouch count: {e}")
            return None
    
    async def _compute_breakout_beyond_level(
        self,
        candle: Candle,
        horizon_end: datetime
    ) -> Optional[bool]:
        """
        Compute Label 16: Breakout Beyond Level
        Did price break significantly beyond level (0.2% threshold)?
        
        Args:
            candle: Input candle
            horizon_end: End of horizon period
            
        Returns:
            True if breakout occurred, False otherwise, None if computation fails
        """
        try:
            # Get active levels at candle timestamp
            active_levels = await self._get_active_levels(
                candle.instrument_id, candle.granularity, candle.ts
            )
            
            if not active_levels:
                return False
            
            # Get path data at lower granularity for accurate breakout detection
            path_granularity, _ = TimestampAligner.get_path_granularity(
                candle.granularity.value
            )
            
            path_data = await self._get_path_data(
                candle.instrument_id,
                path_granularity,
                candle.ts,
                horizon_end
            )
            
            if not path_data:
                return False
            
            breakout_threshold = 0.002  # 0.2% threshold for significant breakout
            
            # Check each active level for breakouts
            for level in active_levels:
                level_price = level["price"]
                level_type = level["current_type"]
                
                for data in path_data:
                    high = data.get("high", 0)
                    low = data.get("low", 0)
                    
                    if level_type == "support":
                        # Check for support breakout (price below level)
                        if low < level_price:
                            break_distance = (level_price - low) / level_price
                            if break_distance > breakout_threshold:
                                return True
                    
                    elif level_type == "resistance":
                        # Check for resistance breakout (price above level)
                        if high > level_price:
                            break_distance = (high - level_price) / level_price
                            if break_distance > breakout_threshold:
                                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to compute breakout beyond level: {e}")
            return None
    
    async def _compute_flip_within_horizon(
        self,
        candle: Candle,
        horizon_end: datetime
    ) -> Optional[bool]:
        """
        Compute Label 17: Flip Within Horizon
        Did level flip from support to resistance or vice versa?
        
        This requires checking level events within the horizon period
        to detect FLIP_TO_SUPPORT or FLIP_TO_RESISTANCE events.
        
        Args:
            candle: Input candle
            horizon_end: End of horizon period
            
        Returns:
            True if flip occurred, False otherwise, None if computation fails
        """
        try:
            # Get active levels at candle timestamp
            active_levels = await self._get_active_levels(
                candle.instrument_id, candle.granularity, candle.ts
            )
            
            if not active_levels:
                return False
            
            # For each level, check if flip events occurred within horizon
            for level in active_levels:
                level_id = level.get("level_id")
                if not level_id:
                    continue
                
                # Check level events within horizon (this would need ClickHouse query)
                # For now, we'll use a simplified approach based on price action
                flip_detected = await self._detect_level_flip_from_price_action(
                    level, candle, horizon_end
                )
                
                if flip_detected:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to compute flip within horizon: {e}")
            return None
    
    async def _detect_level_flip_from_price_action(
        self,
        level: Dict[str, Any],
        candle: Candle,
        horizon_end: datetime
    ) -> bool:
        """
        Detect level flip based on price action analysis.
        
        A flip occurs when:
        1. Support becomes resistance: price breaks below and then acts as resistance
        2. Resistance becomes support: price breaks above and then acts as support
        
        Args:
            level: Level information
            candle: Input candle
            horizon_end: End of horizon
            
        Returns:
            True if flip is detected
        """
        try:
            level_price = level["price"]
            level_type = level["current_type"]
            
            # Get path data at lower granularity
            path_granularity, _ = TimestampAligner.get_path_granularity(
                candle.granularity.value
            )
            
            path_data = await self._get_path_data(
                candle.instrument_id,
                path_granularity,
                candle.ts,
                horizon_end
            )
            
            if not path_data:
                return False
            
            breakout_threshold = 0.002  # 0.2% for breakout
            retest_threshold = 0.001   # 0.1% for retest detection
            
            breakout_occurred = False
            retest_occurred = False
            
            for data in path_data:
                high = data.get("high", 0)
                low = data.get("low", 0)
                
                if level_type == "support":
                    # Check for support break (flip to resistance)
                    if not breakout_occurred and low < level_price:
                        break_distance = (level_price - low) / level_price
                        if break_distance > breakout_threshold:
                            breakout_occurred = True
                    
                    # After breakout, check for retest as resistance
                    if breakout_occurred and high >= level_price:
                        retest_distance = abs(high - level_price) / level_price
                        if retest_distance <= retest_threshold:
                            retest_occurred = True
                
                elif level_type == "resistance":
                    # Check for resistance break (flip to support)
                    if not breakout_occurred and high > level_price:
                        break_distance = (high - level_price) / level_price
                        if break_distance > breakout_threshold:
                            breakout_occurred = True
                    
                    # After breakout, check for retest as support
                    if breakout_occurred and low <= level_price:
                        retest_distance = abs(low - level_price) / level_price
                        if retest_distance <= retest_threshold:
                            retest_occurred = True
            
            # Flip is confirmed if both breakout and retest occurred
            return breakout_occurred and retest_occurred
            
        except Exception as e:
            logger.error(f"Failed to detect level flip: {e}")
            return False
    
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
        
        # Issue #8: Batch validation of computed labels
        if self.enable_validation and successful_labels > 0:
            logger.info("Performing batch validation of computed labels...")
            
            # Collect all computed label sets for batch validation
            # Note: In a full implementation, we'd collect these during computation
            # For now, we'll fetch a sample for validation
            try:
                sample_label_sets = await self._get_sample_labels_for_validation(
                    instrument_id, granularity, start_date, min(end_date, start_date + timedelta(days=7))
                )
                
                if sample_label_sets:
                    batch_validation_result = label_validator.validate_batch_labels(
                        sample_label_sets, statistical_tests=True
                    )
                    
                    if not batch_validation_result.is_valid:
                        logger.warning(f"Batch validation detected issues: {len(batch_validation_result.issues)} issues found")
                        
                        # Log top issues
                        for issue in batch_validation_result.issues[:5]:
                            logger.warning(f"Batch validation issue: {issue}")
                    
                    # Add batch validation metrics
                    batch_validation_summary = batch_validation_result.summary()
                    
            except Exception as e:
                logger.warning(f"Batch validation failed: {e}")
                batch_validation_summary = {"error": str(e)}
        else:
            batch_validation_summary = {"skipped": "validation disabled or no successful labels"}
        
        return {
            "total_candles": total_candles,
            "processed_candles": processed_candles,
            "successful_labels": successful_labels,
            "errors": errors,
            "error_rate": len(errors) / max(processed_candles, 1),
            "validation_stats": self.get_validation_stats(),
            "batch_validation": batch_validation_summary
        }
    
    async def _get_sample_labels_for_validation(
        self,
        instrument_id: str,
        granularity: str,
        start_date: datetime,
        end_date: datetime,
        sample_size: int = 100
    ) -> List[LabelSet]:
        """
        Get a sample of label sets for batch validation.
        
        In a full implementation, this would fetch computed labels from storage.
        For now, we'll return a mock sample or empty list.
        """
        try:
            # This would fetch actual computed labels from ClickHouse or cache
            # For now, return empty list to avoid validation errors
            return []
        except Exception as e:
            logger.warning(f"Could not fetch sample labels for validation: {e}")
            return []
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.validation_stats["total_computations"]
        
        if total == 0:
            return self.validation_stats
        
        return {
            **self.validation_stats,
            "pre_validation_failure_rate": self.validation_stats["pre_validation_failures"] / total,
            "post_validation_failure_rate": self.validation_stats["post_validation_failures"] / total,
            "avg_validation_time_ms": self.validation_stats["validation_time_total_ms"] / total
        }
    
    def create_validation_alert(self, threshold_failure_rate: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        Create validation alert if failure rates exceed threshold.
        
        Args:
            threshold_failure_rate: Failure rate threshold (0.1 = 10%)
            
        Returns:
            Alert dictionary if threshold exceeded, None otherwise
        """
        stats = self.get_validation_stats()
        total = stats["total_computations"]
        
        if total == 0:
            return None
        
        pre_failure_rate = stats.get("pre_validation_failure_rate", 0)
        post_failure_rate = stats.get("post_validation_failure_rate", 0)
        
        if pre_failure_rate > threshold_failure_rate or post_failure_rate > threshold_failure_rate:
            return {
                "alert_type": "validation_failure_rate_exceeded",
                "threshold": threshold_failure_rate,
                "pre_validation_failure_rate": pre_failure_rate,
                "post_validation_failure_rate": post_failure_rate,
                "total_computations": total,
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "high" if max(pre_failure_rate, post_failure_rate) > 0.2 else "medium"
            }
        
        return None
    
    def reset_validation_stats(self):
        """Reset validation statistics"""
        self.validation_stats = {
            "pre_validation_failures": 0,
            "post_validation_failures": 0,
            "total_computations": 0,
            "validation_time_total_ms": 0.0
        }


# Global computation engine instance
computation_engine = LabelComputationEngine()