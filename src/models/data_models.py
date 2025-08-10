"""
Data models for label computation system
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class Granularity(str, Enum):
    """Supported time granularities"""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D = "D"
    W = "W"


class LevelType(str, Enum):
    """Support/Resistance level types"""
    SUPPORT = "support"
    RESISTANCE = "resistance"


class EventType(str, Enum):
    """Level event types"""
    NEW_SUPPORT = "NEW_SUPPORT"
    NEW_RESISTANCE = "NEW_RESISTANCE"
    FLIP_TO_SUPPORT = "FLIP_TO_SUPPORT"
    FLIP_TO_RESISTANCE = "FLIP_TO_RESISTANCE"
    TOUCH_UP = "TOUCH_UP"
    TOUCH_DOWN = "TOUCH_DOWN"
    BREAK_SUPPORT = "BREAK_SUPPORT"
    BREAK_RESISTANCE = "BREAK_RESISTANCE"
    DEACTIVATE_UP = "DEACTIVATE_UP"
    DEACTIVATE_DOWN = "DEACTIVATE_DOWN"


class BarrierHit(str, Enum):
    """Triple barrier hit types"""
    UPPER = "upper"
    LOWER = "lower"
    NONE = "none"
    NO_DATA = "no_data"


class Candle(BaseModel):
    """Market candle data"""
    instrument_id: str
    granularity: Granularity
    ts: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    bid: Optional[float] = Field(default=None, gt=0)
    ask: Optional[float] = Field(default=None, gt=0)
    
    # Technical indicators (optional)
    atr_14: Optional[float] = Field(default=None, ge=0)
    ema_20: Optional[float] = Field(default=None, gt=0)
    ema_50: Optional[float] = Field(default=None, gt=0)
    ema_200: Optional[float] = Field(default=None, gt=0)
    rsi_14: Optional[float] = Field(default=None, ge=0, le=100)
    volume_sma_20: Optional[float] = Field(default=None, ge=0)
    volatility_20: Optional[float] = Field(default=None, ge=0)
    
    @validator('high')
    def high_must_be_highest(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('high must be >= low')
        if 'open' in values and v < values['open']:
            raise ValueError('high must be >= open')
        if 'close' in values and v < values['close']:
            raise ValueError('high must be >= close')
        return v
    
    @validator('low')
    def low_must_be_lowest(cls, v, values):
        if 'open' in values and v > values['open']:
            raise ValueError('low must be <= open')
        if 'close' in values and v > values['close']:
            raise ValueError('low must be <= close')
        return v
    
    class Config:
        use_enum_values = True


class Level(BaseModel):
    """Support/Resistance level"""
    level_id: str
    instrument_id: str
    granularity: Granularity
    price: float = Field(gt=0)
    created_at: datetime
    current_type: LevelType
    status: str = Field(default="active")
    last_event_type: Optional[EventType] = None
    last_event_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None
    
    @validator('deactivated_at')
    def deactivated_must_be_after_created(cls, v, values):
        if v and 'created_at' in values and v < values['created_at']:
            raise ValueError('deactivated_at must be after created_at')
        return v
    
    def is_active_at(self, timestamp: datetime) -> bool:
        """Check if level is active at given timestamp"""
        if self.created_at > timestamp:
            return False
        if self.status == "inactive" and self.deactivated_at and self.deactivated_at <= timestamp:
            return False
        return True
    
    class Config:
        use_enum_values = True


class EnhancedTripleBarrierLabel(BaseModel):
    """Label 11.a: Enhanced Triple Barrier with S/R adjustments"""
    label: int = Field(ge=-1, le=1)  # -1, 0, 1
    barrier_hit: BarrierHit
    time_to_barrier: int = Field(ge=0)
    barrier_price: Optional[float] = Field(default=None, gt=0)
    level_adjusted: bool = Field(default=False)
    upper_barrier: float = Field(gt=0)
    lower_barrier: float = Field(gt=0)
    path_granularity: Optional[Granularity] = None
    
    class Config:
        use_enum_values = True


class LabelSet(BaseModel):
    """Complete set of computed labels for a candle"""
    instrument_id: str
    granularity: Granularity
    ts: datetime
    
    # Label 11.a: Enhanced Triple Barrier
    enhanced_triple_barrier: Optional[EnhancedTripleBarrierLabel] = None
    
    # Core labels
    forward_return: Optional[float] = None
    vol_scaled_return: Optional[float] = None
    return_sign: Optional[int] = Field(default=None, ge=-1, le=1)
    return_quantile: Optional[int] = Field(default=None, ge=1, le=10)
    
    # Path metrics
    mfe: Optional[float] = None  # Maximum Favorable Excursion
    mae: Optional[float] = None  # Maximum Adverse Excursion
    profit_factor: Optional[float] = Field(default=None, ge=0)
    max_penetration: Optional[float] = Field(default=None, ge=0)
    
    # Level-specific
    retouch_count: Optional[int] = Field(default=None, ge=0)
    next_touch_time: Optional[int] = Field(default=None, ge=0)
    breakout_occurred: Optional[bool] = None
    flip_occurred: Optional[bool] = None
    nearest_level_distance: Optional[float] = None
    
    # Risk metrics
    drawdown_depth: Optional[float] = Field(default=None, le=0)
    time_underwater: Optional[int] = Field(default=None, ge=0)
    path_skewness: Optional[float] = None
    
    # Metadata
    label_version: str = Field(default="1.0.0")
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    computation_time_ms: Optional[float] = Field(default=None, ge=0)
    
    class Config:
        use_enum_values = True


class ComputeRequest(BaseModel):
    """Request to compute labels"""
    instrument_id: str
    granularity: Granularity
    start_date: datetime
    end_date: datetime
    horizon_periods: int = Field(default=6, ge=1, le=100)
    labels_to_compute: List[str] = Field(default=["enhanced_triple_barrier"])
    force_recompute: bool = Field(default=False)
    
    @validator('end_date')
    def end_must_be_after_start(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v
    
    class Config:
        use_enum_values = True


class BackfillStatus(BaseModel):
    """Backfill job status"""
    job_id: str
    instrument_id: str
    granularity: Granularity
    start_date: datetime
    end_date: datetime
    status: str = Field(default="pending")  # pending, running, completed, failed
    progress_percent: float = Field(default=0, ge=0, le=100)
    last_processed_ts: Optional[datetime] = None
    total_candles: int = Field(default=0, ge=0)
    processed_candles: int = Field(default=0, ge=0)
    errors: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    class Config:
        use_enum_values = True