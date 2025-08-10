"""
Timestamp alignment utilities for multi-timeframe data processing
CRITICAL: Ensures proper alignment across different granularities
"""
from datetime import datetime, timedelta
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TimestampAligner:
    """
    Handles multi-timeframe timestamp alignment.
    Critical for ensuring no look-ahead bias in label computation.
    """
    
    # Granularity mappings for path-dependent calculations
    GRANULARITY_MAPPING: Dict[str, Dict[str, any]] = {
        'W': {'lower': 'D', 'multiplier': 5},    # Weekly uses Daily
        'D': {'lower': 'H4', 'multiplier': 6},   # Daily uses 4-hour
        'H4': {'lower': 'H1', 'multiplier': 4},  # 4-hour uses Hourly
        'H1': {'lower': 'M15', 'multiplier': 4}, # Hourly uses 15-min
        'M15': {'lower': 'M5', 'multiplier': 3}, # 15-min uses 5-min
        'M5': {'lower': 'M1', 'multiplier': 5},  # 5-min uses 1-min
    }
    
    @staticmethod
    def align_to_granularity(ts: datetime, granularity: str) -> datetime:
        """
        Align timestamp to granularity boundary.
        
        CRITICAL: H4 candles align to 1,5,9,13,17,21 UTC (not 0,4,8,12,16,20)
        
        Args:
            ts: Timestamp to align
            granularity: Target granularity (W, D, H4, H1, M15, M5)
            
        Returns:
            Aligned timestamp
        """
        if granularity == 'H4':
            # CRITICAL: H4 candles at 1,5,9,13,17,21 UTC
            hour = ts.hour
            # Find the closest H4 boundary
            h4_hours = [1, 5, 9, 13, 17, 21]
            
            # Find the floor H4 hour
            aligned_hour = 21  # Default to previous day's last H4
            for h in h4_hours:
                if hour >= h:
                    aligned_hour = h
                else:
                    break
            
            # If we're before 1:00, go to previous day's 21:00
            if hour < 1:
                aligned_hour = 21
                ts = ts - timedelta(days=1)
                
            return ts.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
        
        elif granularity == 'H1':
            return ts.replace(minute=0, second=0, microsecond=0)
        
        elif granularity == 'D':
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)
        
        elif granularity == 'W':
            # Align to Monday 00:00
            days_to_monday = ts.weekday()
            aligned = ts - timedelta(days=days_to_monday)
            return aligned.replace(hour=0, minute=0, second=0, microsecond=0)
        
        elif granularity == 'M15':
            minute = (ts.minute // 15) * 15
            return ts.replace(minute=minute, second=0, microsecond=0)
        
        elif granularity == 'M5':
            minute = (ts.minute // 5) * 5
            return ts.replace(minute=minute, second=0, microsecond=0)
        
        elif granularity == 'M1':
            return ts.replace(second=0, microsecond=0)
        
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")
    
    @staticmethod
    def get_horizon_end(ts: datetime, granularity: str, periods: int) -> datetime:
        """
        Calculate horizon end timestamp for given periods.
        
        Args:
            ts: Start timestamp (should be aligned)
            granularity: Time granularity
            periods: Number of periods
            
        Returns:
            End timestamp
        """
        if granularity == 'M1':
            return ts + timedelta(minutes=periods)
        elif granularity == 'M5':
            return ts + timedelta(minutes=5 * periods)
        elif granularity == 'M15':
            return ts + timedelta(minutes=15 * periods)
        elif granularity == 'H1':
            return ts + timedelta(hours=periods)
        elif granularity == 'H4':
            return ts + timedelta(hours=4 * periods)
        elif granularity == 'D':
            return ts + timedelta(days=periods)
        elif granularity == 'W':
            return ts + timedelta(weeks=periods)
        else:
            raise ValueError(f"Unsupported granularity: {granularity}")
    
    @classmethod
    def get_path_granularity(cls, target_granularity: str) -> Tuple[str, int]:
        """
        Get the appropriate lower granularity for path-dependent calculations.
        
        CRITICAL: Always use lower granularity for accurate path checking
        
        Args:
            target_granularity: The granularity of the label being computed
            
        Returns:
            Tuple of (path_granularity, multiplier)
        """
        if target_granularity not in cls.GRANULARITY_MAPPING:
            raise ValueError(
                f"No path granularity mapping for {target_granularity}. "
                f"Available: {list(cls.GRANULARITY_MAPPING.keys())}"
            )
        
        mapping = cls.GRANULARITY_MAPPING[target_granularity]
        return mapping['lower'], mapping['multiplier']
    
    @staticmethod
    def get_h4_candle_times(date: datetime) -> list[datetime]:
        """
        Get all H4 candle times for a given date.
        
        Returns:
            List of H4 timestamps (1,5,9,13,17,21 UTC)
        """
        base_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        h4_hours = [1, 5, 9, 13, 17, 21]
        
        times = []
        # Previous day's 21:00
        prev_day = base_date - timedelta(days=1)
        times.append(prev_day.replace(hour=21))
        
        # Current day's H4 times
        for hour in h4_hours[:-1]:  # Exclude 21 as it belongs to next period
            times.append(base_date.replace(hour=hour))
        
        return times
    
    @staticmethod
    def validate_alignment(ts: datetime, granularity: str) -> bool:
        """
        Validate if a timestamp is properly aligned to granularity.
        
        Args:
            ts: Timestamp to validate
            granularity: Expected granularity
            
        Returns:
            True if properly aligned
        """
        aligned = TimestampAligner.align_to_granularity(ts, granularity)
        return ts == aligned
    
    @staticmethod
    def get_period_bounds(ts: datetime, granularity: str) -> Tuple[datetime, datetime]:
        """
        Get the start and end timestamps for the period containing ts.
        
        Args:
            ts: Timestamp within the period
            granularity: Period granularity
            
        Returns:
            Tuple of (period_start, period_end)
        """
        period_start = TimestampAligner.align_to_granularity(ts, granularity)
        period_end = TimestampAligner.get_horizon_end(period_start, granularity, 1)
        return period_start, period_end