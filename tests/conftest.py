"""
Pytest configuration and shared fixtures for Label 11.a Enhanced Triple Barrier tests.

This module provides comprehensive fixtures for testing the Enhanced Triple Barrier
implementation, including mock data generation, service mocks, and test utilities.
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from faker import Faker

from src.models.data_models import (
    Candle, Level, LabelSet, EnhancedTripleBarrierLabel,
    Granularity, LevelType, EventType, BarrierHit
)
from src.core.label_computation import LabelComputationEngine
from src.utils.timestamp_aligner import TimestampAligner
from src.services.clickhouse_service import ClickHouseService
from src.services.redis_cache import RedisCacheService

fake = Faker()
Faker.seed(42)  # For reproducible test data


@pytest.fixture
def sample_candle() -> Candle:
    """Generate a sample candle for testing."""
    return Candle(
        instrument_id="EUR/USD",
        granularity=Granularity.H4,
        ts=datetime(2024, 1, 15, 9, 0, 0),  # H4 aligned timestamp
        open=1.0500,
        high=1.0580,
        low=1.0450,
        close=1.0520,
        volume=1000.0,
        bid=1.0519,
        ask=1.0521,
        atr_14=0.0045,
        ema_20=1.0510,
        ema_50=1.0480,
        ema_200=1.0400,
        rsi_14=55.5,
        volume_sma_20=950.0,
        volatility_20=0.0050
    )


@pytest.fixture
def h4_aligned_timestamps() -> List[datetime]:
    """Generate list of H4 aligned timestamps for testing."""
    base_date = datetime(2024, 1, 15)
    timestamps = []
    
    # H4 hours: 1, 5, 9, 13, 17, 21 UTC
    h4_hours = [1, 5, 9, 13, 17, 21]
    
    for day in range(5):  # 5 days of H4 timestamps
        current_date = base_date + timedelta(days=day)
        for hour in h4_hours:
            timestamps.append(current_date.replace(hour=hour, minute=0, second=0, microsecond=0))
    
    return timestamps


@pytest.fixture
def sample_levels() -> List[Dict[str, Any]]:
    """Generate sample support/resistance levels."""
    return [
        {
            "level_id": "support_1",
            "instrument_id": "EUR/USD",
            "granularity": "H4",
            "price": 1.0400,
            "created_at": datetime(2024, 1, 10, 5, 0, 0),
            "current_type": "support",
            "status": "active",
            "last_event_type": "NEW_SUPPORT",
            "last_event_at": datetime(2024, 1, 10, 5, 0, 0),
            "deactivated_at": None
        },
        {
            "level_id": "resistance_1",
            "instrument_id": "EUR/USD",
            "granularity": "H4",
            "price": 1.0650,
            "created_at": datetime(2024, 1, 10, 9, 0, 0),
            "current_type": "resistance",
            "status": "active",
            "last_event_type": "NEW_RESISTANCE",
            "last_event_at": datetime(2024, 1, 10, 9, 0, 0),
            "deactivated_at": None
        },
        {
            "level_id": "support_2",
            "instrument_id": "EUR/USD", 
            "granularity": "H4",
            "price": 1.0480,
            "created_at": datetime(2024, 1, 12, 13, 0, 0),
            "current_type": "support",
            "status": "active",
            "last_event_type": "FLIP_TO_SUPPORT",
            "last_event_at": datetime(2024, 1, 12, 13, 0, 0),
            "deactivated_at": None
        }
    ]


@pytest.fixture
def sample_path_data() -> List[Dict[str, Any]]:
    """Generate sample path data for barrier checking."""
    base_price = 1.0520
    path_data = []
    
    # Generate 24 H1 candles (6 H4 periods * 4 H1 per H4)
    for i in range(24):
        # Simulate price movement with some volatility
        price_change = np.random.normal(0, 0.002)  # 0.2% std deviation
        current_price = base_price + price_change
        
        # Create realistic OHLC
        volatility = 0.001 * (1 + 0.5 * np.random.random())
        high = current_price + volatility
        low = current_price - volatility
        
        path_data.append({
            "ts": datetime(2024, 1, 15, 9, 0, 0) + timedelta(hours=i),
            "open": base_price,
            "high": high,
            "low": low,
            "close": current_price,
            "volume": 100 + 50 * np.random.random()
        })
        
        base_price = current_price  # Next candle starts where this one ends
    
    return path_data


@pytest.fixture
def barrier_hit_path_data() -> Dict[str, List[Dict[str, Any]]]:
    """Generate path data that hits barriers for testing."""
    base_price = 1.0520
    upper_barrier = 1.0580
    lower_barrier = 1.0460
    
    # Upper barrier hit scenario
    upper_hit_data = []
    price = base_price
    for i in range(10):
        if i == 5:  # Hit upper barrier at 5th candle
            high = upper_barrier + 0.0005
            low = price - 0.0010
            close = upper_barrier - 0.0002
        else:
            high = price + 0.0010
            low = price - 0.0010
            close = price + np.random.normal(0, 0.0005)
        
        upper_hit_data.append({
            "ts": datetime(2024, 1, 15, 9, 0, 0) + timedelta(hours=i),
            "open": price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100
        })
        price = close
    
    # Lower barrier hit scenario
    lower_hit_data = []
    price = base_price
    for i in range(10):
        if i == 3:  # Hit lower barrier at 3rd candle
            high = price + 0.0010
            low = lower_barrier - 0.0005
            close = lower_barrier + 0.0002
        else:
            high = price + 0.0010
            low = price - 0.0010
            close = price + np.random.normal(0, 0.0005)
        
        lower_hit_data.append({
            "ts": datetime(2024, 1, 15, 9, 0, 0) + timedelta(hours=i),
            "open": price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100
        })
        price = close
    
    # No barrier hit scenario
    no_hit_data = []
    price = base_price
    for i in range(10):
        # Keep price within barriers
        high = min(price + 0.0020, upper_barrier - 0.0005)
        low = max(price - 0.0020, lower_barrier + 0.0005)
        close = price + np.random.normal(0, 0.0010)
        close = max(min(close, upper_barrier - 0.0005), lower_barrier + 0.0005)
        
        no_hit_data.append({
            "ts": datetime(2024, 1, 15, 9, 0, 0) + timedelta(hours=i),
            "open": price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100
        })
        price = close
    
    return {
        "upper_hit": upper_hit_data,
        "lower_hit": lower_hit_data,
        "no_hit": no_hit_data
    }


@pytest.fixture
def mock_clickhouse_service():
    """Mock ClickHouse service for testing."""
    mock_service = Mock(spec=ClickHouseService)
    
    # Mock methods
    mock_service.fetch_snapshots = AsyncMock(return_value=[])
    mock_service.fetch_active_levels = AsyncMock(return_value=[])
    
    return mock_service


@pytest.fixture
def mock_redis_cache():
    """Mock Redis cache service for testing."""
    mock_cache = Mock(spec=RedisCacheService)
    
    # Mock methods
    mock_cache.get_labels = Mock(return_value=None)
    mock_cache.cache_labels = Mock()
    mock_cache.get_active_levels = Mock(return_value=None)
    mock_cache.cache_active_levels = Mock()
    mock_cache.get_path_data = Mock(return_value=None)
    mock_cache.cache_path_data = Mock()
    
    return mock_cache


@pytest.fixture
def label_computation_engine(mock_clickhouse_service, mock_redis_cache):
    """Create LabelComputationEngine with mocked dependencies."""
    with patch('src.core.label_computation.clickhouse_service', mock_clickhouse_service), \
         patch('src.core.label_computation.redis_cache', mock_redis_cache):
        engine = LabelComputationEngine()
        engine._clickhouse_service = mock_clickhouse_service
        engine._redis_cache = mock_redis_cache
        return engine


@pytest.fixture
def timestamp_aligner():
    """Create TimestampAligner instance."""
    return TimestampAligner()


# Performance testing fixtures

@pytest.fixture
def large_dataset() -> List[Candle]:
    """Generate large dataset for performance testing."""
    candles = []
    base_time = datetime(2024, 1, 1, 1, 0, 0)  # Start with H4 aligned timestamp
    
    for i in range(10000):  # 10k candles for performance testing
        candles.append(Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=base_time + timedelta(hours=4 * i),
            open=1.0500 + np.random.normal(0, 0.01),
            high=1.0500 + abs(np.random.normal(0.01, 0.01)),
            low=1.0500 - abs(np.random.normal(0.01, 0.01)),
            close=1.0500 + np.random.normal(0, 0.01),
            volume=1000 + np.random.normal(0, 100),
            atr_14=0.0045
        ))
    
    return candles


@pytest.fixture(scope="session")
def cache_hit_scenario():
    """Fixture for testing cache hit scenarios."""
    return {
        "cache_enabled": True,
        "cached_results": {
            "enhanced_triple_barrier": {
                "label": 1,
                "barrier_hit": "upper",
                "time_to_barrier": 5,
                "barrier_price": 1.0580,
                "level_adjusted": True,
                "upper_barrier": 1.0580,
                "lower_barrier": 1.0460,
                "path_granularity": "H1"
            }
        }
    }


@pytest.fixture(scope="session")
def cache_miss_scenario():
    """Fixture for testing cache miss scenarios."""
    return {
        "cache_enabled": True,
        "cached_results": None
    }


# Edge case fixtures

@pytest.fixture
def edge_case_timestamps() -> Dict[str, List[datetime]]:
    """Generate edge case timestamps for thorough testing."""
    return {
        "weekend_transitions": [
            datetime(2024, 1, 12, 21, 0, 0),  # Friday 21:00 UTC
            datetime(2024, 1, 15, 1, 0, 0),   # Monday 01:00 UTC
        ],
        "daylight_saving_transitions": [
            datetime(2024, 3, 31, 1, 0, 0),   # EU DST start
            datetime(2024, 10, 27, 1, 0, 0),  # EU DST end
        ],
        "month_boundaries": [
            datetime(2024, 1, 31, 21, 0, 0),  # End of January
            datetime(2024, 2, 1, 1, 0, 0),    # Start of February
        ],
        "year_boundaries": [
            datetime(2023, 12, 31, 21, 0, 0), # End of 2023
            datetime(2024, 1, 1, 1, 0, 0),    # Start of 2024
        ],
        "leap_year": [
            datetime(2024, 2, 28, 21, 0, 0),  # Feb 28 (leap year)
            datetime(2024, 2, 29, 1, 0, 0),   # Feb 29 (leap day)
        ]
    }


@pytest.fixture
def granularity_test_matrix() -> Dict[str, Dict[str, Any]]:
    """Test matrix for different granularity combinations."""
    return {
        "W": {
            "target": Granularity.W,
            "path": Granularity.D,
            "multiplier": 5,
            "test_periods": [1, 2, 4],
            "alignment_test": datetime(2024, 1, 17, 14, 30, 0)  # Wednesday
        },
        "D": {
            "target": Granularity.D,
            "path": Granularity.H4,
            "multiplier": 6,
            "test_periods": [1, 3, 7],
            "alignment_test": datetime(2024, 1, 15, 14, 30, 0)  # Mid-day
        },
        "H4": {
            "target": Granularity.H4,
            "path": Granularity.H1,
            "multiplier": 4,
            "test_periods": [1, 6, 12],
            "alignment_test": datetime(2024, 1, 15, 7, 30, 0)  # Between H4 boundaries
        },
        "H1": {
            "target": Granularity.H1,
            "path": Granularity.M15,
            "multiplier": 4,
            "test_periods": [1, 4, 24],
            "alignment_test": datetime(2024, 1, 15, 9, 30, 0)  # Mid-hour
        },
        "M15": {
            "target": Granularity.M15,
            "path": Granularity.M5,
            "multiplier": 3,
            "test_periods": [1, 4, 16],
            "alignment_test": datetime(2024, 1, 15, 9, 7, 0)  # Between 15min boundaries
        },
        "M5": {
            "target": Granularity.M5,
            "path": Granularity.M1,
            "multiplier": 5,
            "test_periods": [1, 12, 60],
            "alignment_test": datetime(2024, 1, 15, 9, 2, 0)  # Between 5min boundaries
        }
    }


# Test utilities

@pytest.fixture
def assert_no_lookahead_bias():
    """Utility fixture to assert no look-ahead bias in computations."""
    def _assert_no_lookahead_bias(candle: Candle, label_set: LabelSet, path_data: List[Dict]):
        """
        Assert that no future information is used in label computation.
        
        Args:
            candle: The input candle
            label_set: The computed label set
            path_data: The path data used for computation
        """
        # Ensure all path data timestamps are >= candle timestamp
        for data in path_data:
            assert data["ts"] >= candle.ts, f"Look-ahead bias detected: path data {data['ts']} < candle timestamp {candle.ts}"
        
        # Ensure computation timestamp is realistic
        if label_set.computed_at:
            assert label_set.computed_at >= candle.ts, "Computation timestamp cannot be before candle timestamp"
    
    return _assert_no_lookahead_bias


@pytest.fixture
def performance_timer():
    """Utility fixture for measuring performance."""
    import time
    
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed_ms
        
        @property
        def elapsed_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return None
    
    return PerformanceTimer()


# Markers for test categories

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "edge_cases: Edge case tests")
    config.addinivalue_line("markers", "slow: Slow running tests")