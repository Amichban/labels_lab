"""
Unit tests for TimestampAligner class.

These tests focus on critical timestamp alignment functionality, especially
the H4 alignment at 1,5,9,13,17,21 UTC which is essential for preventing
look-ahead bias in Label 11.a Enhanced Triple Barrier computation.
"""

import pytest
from datetime import datetime, timedelta
from typing import List, Tuple

from src.utils.timestamp_aligner import TimestampAligner


class TestTimestampAligner:
    """Test suite for TimestampAligner class."""
    
    # H4 Alignment Tests (CRITICAL for Label 11.a)
    
    @pytest.mark.unit
    def test_h4_alignment_exact_boundaries(self):
        """Test H4 alignment at exact boundary times."""
        aligner = TimestampAligner()
        h4_hours = [1, 5, 9, 13, 17, 21]
        base_date = datetime(2024, 1, 15)
        
        for hour in h4_hours:
            test_time = base_date.replace(hour=hour, minute=0, second=0, microsecond=0)
            aligned = aligner.align_to_granularity(test_time, 'H4')
            
            assert aligned == test_time, f"H4 boundary {hour}:00 should align to itself"
            assert aligned.minute == 0 and aligned.second == 0 and aligned.microsecond == 0
    
    @pytest.mark.unit
    def test_h4_alignment_between_boundaries(self):
        """Test H4 alignment for timestamps between boundaries."""
        aligner = TimestampAligner()
        test_cases = [
            # (input_hour, input_minute, expected_aligned_hour)
            (2, 30, 1),    # Between 1:00 and 5:00 -> align to 1:00
            (4, 45, 1),    # Between 1:00 and 5:00 -> align to 1:00
            (6, 15, 5),    # Between 5:00 and 9:00 -> align to 5:00
            (8, 59, 5),    # Between 5:00 and 9:00 -> align to 5:00
            (11, 30, 9),   # Between 9:00 and 13:00 -> align to 9:00
            (15, 0, 13),   # Between 13:00 and 17:00 -> align to 13:00
            (19, 30, 17),  # Between 17:00 and 21:00 -> align to 17:00
            (23, 0, 21),   # Between 21:00 and 1:00 (next day) -> align to 21:00
        ]
        
        base_date = datetime(2024, 1, 15)
        
        for input_hour, input_minute, expected_hour in test_cases:
            test_time = base_date.replace(hour=input_hour, minute=input_minute, second=0, microsecond=0)
            aligned = aligner.align_to_granularity(test_time, 'H4')
            
            assert aligned.hour == expected_hour, \
                f"H4 alignment failed: {input_hour}:{input_minute:02d} should align to {expected_hour}:00, got {aligned.hour}:00"
            assert aligned.minute == 0 and aligned.second == 0 and aligned.microsecond == 0
    
    @pytest.mark.unit
    def test_h4_alignment_early_morning_edge_case(self):
        """Test H4 alignment for times before 1:00 UTC (should go to previous day's 21:00)."""
        aligner = TimestampAligner()
        
        # Test midnight and early morning hours
        test_cases = [
            (0, 0),   # Midnight
            (0, 30),  # 00:30
            (0, 59),  # 00:59
        ]
        
        base_date = datetime(2024, 1, 15)  # Monday
        
        for hour, minute in test_cases:
            test_time = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
            aligned = aligner.align_to_granularity(test_time, 'H4')
            
            # Should align to previous day's 21:00
            expected_date = base_date - timedelta(days=1)
            expected_time = expected_date.replace(hour=21, minute=0, second=0, microsecond=0)
            
            assert aligned == expected_time, \
                f"Early morning {hour}:{minute:02d} should align to previous day 21:00, got {aligned}"
    
    @pytest.mark.unit
    def test_h4_alignment_cross_day_boundary(self):
        """Test H4 alignment across day boundaries."""
        aligner = TimestampAligner()
        
        # Test around midnight on different days
        test_dates = [
            datetime(2024, 1, 15, 23, 30),  # Monday night
            datetime(2024, 1, 31, 0, 15),   # Month boundary
            datetime(2024, 2, 29, 0, 45),   # Leap year boundary
            datetime(2024, 12, 31, 23, 59), # Year boundary
        ]
        
        for test_time in test_dates:
            aligned = aligner.align_to_granularity(test_time, 'H4')
            
            if test_time.hour >= 21:
                # Should align to same day's 21:00
                expected = test_time.replace(hour=21, minute=0, second=0, microsecond=0)
            else:
                # Should align to previous day's 21:00
                prev_day = test_time - timedelta(days=1)
                expected = prev_day.replace(hour=21, minute=0, second=0, microsecond=0)
            
            assert aligned == expected, f"Cross-day boundary alignment failed for {test_time}"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("granularity,test_time,expected_aligned", [
        # H1 tests
        ("H1", datetime(2024, 1, 15, 9, 30, 45), datetime(2024, 1, 15, 9, 0, 0)),
        ("H1", datetime(2024, 1, 15, 23, 59, 59), datetime(2024, 1, 15, 23, 0, 0)),
        
        # Daily tests
        ("D", datetime(2024, 1, 15, 14, 30, 45), datetime(2024, 1, 15, 0, 0, 0)),
        ("D", datetime(2024, 1, 31, 23, 59, 59), datetime(2024, 1, 31, 0, 0, 0)),
        
        # Weekly tests (align to Monday)
        ("W", datetime(2024, 1, 17, 14, 30, 45), datetime(2024, 1, 15, 0, 0, 0)),  # Wed -> Mon
        ("W", datetime(2024, 1, 21, 23, 59, 59), datetime(2024, 1, 15, 0, 0, 0)),  # Sun -> Mon
        
        # M15 tests
        ("M15", datetime(2024, 1, 15, 9, 7, 30), datetime(2024, 1, 15, 9, 0, 0)),
        ("M15", datetime(2024, 1, 15, 9, 22, 59), datetime(2024, 1, 15, 9, 15, 0)),
        ("M15", datetime(2024, 1, 15, 9, 37, 15), datetime(2024, 1, 15, 9, 30, 0)),
        ("M15", datetime(2024, 1, 15, 9, 52, 45), datetime(2024, 1, 15, 9, 45, 0)),
        
        # M5 tests
        ("M5", datetime(2024, 1, 15, 9, 2, 30), datetime(2024, 1, 15, 9, 0, 0)),
        ("M5", datetime(2024, 1, 15, 9, 7, 59), datetime(2024, 1, 15, 9, 5, 0)),
        ("M5", datetime(2024, 1, 15, 9, 13, 15), datetime(2024, 1, 15, 9, 10, 0)),
        
        # M1 tests
        ("M1", datetime(2024, 1, 15, 9, 5, 30), datetime(2024, 1, 15, 9, 5, 0)),
        ("M1", datetime(2024, 1, 15, 9, 5, 59), datetime(2024, 1, 15, 9, 5, 0)),
    ])
    def test_granularity_alignment_comprehensive(self, granularity, test_time, expected_aligned):
        """Comprehensive test for all granularity alignments."""
        aligner = TimestampAligner()
        aligned = aligner.align_to_granularity(test_time, granularity)
        
        assert aligned == expected_aligned, \
            f"{granularity} alignment failed: {test_time} -> expected {expected_aligned}, got {aligned}"
    
    @pytest.mark.unit
    def test_invalid_granularity_raises_error(self):
        """Test that invalid granularity raises ValueError."""
        aligner = TimestampAligner()
        test_time = datetime(2024, 1, 15, 9, 0, 0)
        
        with pytest.raises(ValueError, match="Unsupported granularity"):
            aligner.align_to_granularity(test_time, "INVALID")
    
    # Horizon calculation tests
    
    @pytest.mark.unit
    @pytest.mark.parametrize("granularity,periods,expected_delta", [
        ("M1", 5, timedelta(minutes=5)),
        ("M5", 3, timedelta(minutes=15)),
        ("M15", 4, timedelta(hours=1)),
        ("H1", 6, timedelta(hours=6)),
        ("H4", 6, timedelta(hours=24)),  # 6 H4 periods = 24 hours
        ("D", 7, timedelta(days=7)),
        ("W", 4, timedelta(weeks=4)),
    ])
    def test_get_horizon_end(self, granularity, periods, expected_delta):
        """Test horizon end calculation for different granularities."""
        aligner = TimestampAligner()
        start_time = datetime(2024, 1, 15, 9, 0, 0)
        
        horizon_end = aligner.get_horizon_end(start_time, granularity, periods)
        expected_end = start_time + expected_delta
        
        assert horizon_end == expected_end, \
            f"Horizon calculation failed for {granularity}: expected {expected_end}, got {horizon_end}"
    
    @pytest.mark.unit
    def test_get_horizon_end_invalid_granularity(self):
        """Test horizon calculation with invalid granularity."""
        aligner = TimestampAligner()
        start_time = datetime(2024, 1, 15, 9, 0, 0)
        
        with pytest.raises(ValueError, match="Unsupported granularity"):
            aligner.get_horizon_end(start_time, "INVALID", 6)
    
    # Path granularity mapping tests (CRITICAL for Label 11.a)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("target,expected_path,expected_multiplier", [
        ("W", "D", 5),
        ("D", "H4", 6),
        ("H4", "H1", 4),
        ("H1", "M15", 4),
        ("M15", "M5", 3),
        ("M5", "M1", 5),
    ])
    def test_get_path_granularity(self, target, expected_path, expected_multiplier):
        """Test path granularity mapping for accurate barrier checking."""
        aligner = TimestampAligner()
        
        path_gran, multiplier = aligner.get_path_granularity(target)
        
        assert path_gran == expected_path, \
            f"Path granularity mapping failed: {target} -> expected {expected_path}, got {path_gran}"
        assert multiplier == expected_multiplier, \
            f"Multiplier mapping failed: {target} -> expected {expected_multiplier}, got {multiplier}"
    
    @pytest.mark.unit
    def test_get_path_granularity_invalid_target(self):
        """Test path granularity mapping with invalid target."""
        aligner = TimestampAligner()
        
        with pytest.raises(ValueError, match="No path granularity mapping"):
            aligner.get_path_granularity("INVALID")
    
    # H4 candle times tests
    
    @pytest.mark.unit
    def test_get_h4_candle_times(self):
        """Test H4 candle times generation for a given date."""
        aligner = TimestampAligner()
        test_date = datetime(2024, 1, 15)  # Monday
        
        h4_times = aligner.get_h4_candle_times(test_date)
        
        # Should include previous day's 21:00 and current day's 1,5,9,13,17 (not 21 as it belongs to next period)
        expected_times = [
            datetime(2024, 1, 14, 21, 0, 0),  # Previous day 21:00
            datetime(2024, 1, 15, 1, 0, 0),   # 01:00
            datetime(2024, 1, 15, 5, 0, 0),   # 05:00
            datetime(2024, 1, 15, 9, 0, 0),   # 09:00
            datetime(2024, 1, 15, 13, 0, 0),  # 13:00
            datetime(2024, 1, 15, 17, 0, 0),  # 17:00
        ]
        
        assert len(h4_times) == len(expected_times), \
            f"Expected {len(expected_times)} H4 times, got {len(h4_times)}"
        
        for i, expected_time in enumerate(expected_times):
            assert h4_times[i] == expected_time, \
                f"H4 time mismatch at index {i}: expected {expected_time}, got {h4_times[i]}"
    
    # Validation tests
    
    @pytest.mark.unit
    @pytest.mark.parametrize("granularity,aligned_time,should_be_valid", [
        ("H4", datetime(2024, 1, 15, 1, 0, 0), True),   # Valid H4
        ("H4", datetime(2024, 1, 15, 5, 0, 0), True),   # Valid H4
        ("H4", datetime(2024, 1, 15, 2, 0, 0), False),  # Invalid H4
        ("H1", datetime(2024, 1, 15, 9, 0, 0), True),   # Valid H1
        ("H1", datetime(2024, 1, 15, 9, 30, 0), False), # Invalid H1
        ("D", datetime(2024, 1, 15, 0, 0, 0), True),    # Valid Daily
        ("D", datetime(2024, 1, 15, 12, 0, 0), False),  # Invalid Daily
        ("M15", datetime(2024, 1, 15, 9, 15, 0), True), # Valid M15
        ("M15", datetime(2024, 1, 15, 9, 7, 0), False), # Invalid M15
    ])
    def test_validate_alignment(self, granularity, aligned_time, should_be_valid):
        """Test timestamp alignment validation."""
        aligner = TimestampAligner()
        
        is_valid = aligner.validate_alignment(aligned_time, granularity)
        
        assert is_valid == should_be_valid, \
            f"Alignment validation failed for {granularity} at {aligned_time}: expected {should_be_valid}, got {is_valid}"
    
    # Period bounds tests
    
    @pytest.mark.unit
    def test_get_period_bounds(self):
        """Test period bounds calculation."""
        aligner = TimestampAligner()
        
        test_cases = [
            # (granularity, test_time, expected_start, expected_end)
            ("H4", datetime(2024, 1, 15, 7, 30, 0), 
             datetime(2024, 1, 15, 5, 0, 0), datetime(2024, 1, 15, 9, 0, 0)),
            ("H1", datetime(2024, 1, 15, 9, 30, 0), 
             datetime(2024, 1, 15, 9, 0, 0), datetime(2024, 1, 15, 10, 0, 0)),
            ("D", datetime(2024, 1, 15, 14, 30, 0), 
             datetime(2024, 1, 15, 0, 0, 0), datetime(2024, 1, 16, 0, 0, 0)),
        ]
        
        for granularity, test_time, expected_start, expected_end in test_cases:
            period_start, period_end = aligner.get_period_bounds(test_time, granularity)
            
            assert period_start == expected_start, \
                f"Period start mismatch for {granularity}: expected {expected_start}, got {period_start}"
            assert period_end == expected_end, \
                f"Period end mismatch for {granularity}: expected {expected_end}, got {period_end}"
    
    # Edge cases and robustness tests
    
    @pytest.mark.unit
    @pytest.mark.edge_cases
    def test_edge_cases_weekend_transitions(self, edge_case_timestamps):
        """Test alignment during weekend transitions."""
        aligner = TimestampAligner()
        
        weekend_times = edge_case_timestamps["weekend_transitions"]
        
        for test_time in weekend_times:
            aligned = aligner.align_to_granularity(test_time, "H4")
            
            # Validate that alignment is still correct during weekend
            assert aligner.validate_alignment(aligned, "H4"), \
                f"Weekend transition alignment failed for {test_time}"
    
    @pytest.mark.unit
    @pytest.mark.edge_cases
    def test_edge_cases_month_year_boundaries(self, edge_case_timestamps):
        """Test alignment across month and year boundaries."""
        aligner = TimestampAligner()
        
        boundary_times = (edge_case_timestamps["month_boundaries"] + 
                         edge_case_timestamps["year_boundaries"] +
                         edge_case_timestamps["leap_year"])
        
        for test_time in boundary_times:
            for granularity in ["H4", "H1", "D", "M15"]:
                aligned = aligner.align_to_granularity(test_time, granularity)
                
                # Validate alignment correctness
                assert aligner.validate_alignment(aligned, granularity), \
                    f"Boundary alignment failed for {test_time} with {granularity}"
                
                # Ensure no unexpected date shifts
                if granularity in ["H4", "H1", "M15"] and test_time.hour > 0:
                    assert aligned.date() <= test_time.date(), \
                        f"Unexpected date shift for {granularity} alignment: {test_time} -> {aligned}"
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_alignment_performance(self):
        """Test that timestamp alignment is performant for large datasets."""
        aligner = TimestampAligner()
        
        # Generate 10,000 random timestamps
        import random
        base_time = datetime(2024, 1, 1)
        test_times = [
            base_time + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59)
            )
            for _ in range(10000)
        ]
        
        # Time the alignment operations
        import time
        start_time = time.perf_counter()
        
        for test_time in test_times:
            aligner.align_to_granularity(test_time, "H4")
        
        elapsed_time = time.perf_counter() - start_time
        
        # Should complete in reasonable time (< 1 second for 10k alignments)
        assert elapsed_time < 1.0, f"Alignment performance too slow: {elapsed_time:.3f}s for 10k operations"
        
        # Calculate operations per second
        ops_per_second = len(test_times) / elapsed_time
        assert ops_per_second > 10000, f"Insufficient throughput: {ops_per_second:.0f} ops/sec"
    
    @pytest.mark.unit
    def test_alignment_consistency_across_granularities(self):
        """Test that alignment is consistent across different granularities."""
        aligner = TimestampAligner()
        
        # Test various timestamps
        test_times = [
            datetime(2024, 1, 15, 9, 37, 23),
            datetime(2024, 2, 29, 14, 18, 45),  # Leap year
            datetime(2024, 12, 31, 23, 59, 59), # Year end
        ]
        
        for test_time in test_times:
            # H4 alignment should be contained within Daily alignment
            h4_aligned = aligner.align_to_granularity(test_time, "H4")
            daily_aligned = aligner.align_to_granularity(test_time, "D")
            
            assert h4_aligned >= daily_aligned, \
                f"H4 alignment {h4_aligned} should be >= Daily alignment {daily_aligned}"
            
            assert h4_aligned.date() <= test_time.date(), \
                f"H4 alignment should not shift to future date: {test_time} -> {h4_aligned}"
            
            # H1 alignment should be contained within H4 alignment when both are on same day
            h1_aligned = aligner.align_to_granularity(test_time, "H1")
            
            if h1_aligned.date() == h4_aligned.date():
                assert h1_aligned >= h4_aligned, \
                    f"H1 alignment {h1_aligned} should be >= H4 alignment {h4_aligned} on same day"