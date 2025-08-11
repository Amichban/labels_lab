#!/usr/bin/env python3
"""
Final comprehensive validation test script
Tests the complete validation framework implementation for Issue #8
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Mock dependencies first
class MockSettings:
    clickhouse_host = "localhost"
    clickhouse_port = 8123
    clickhouse_user = "default"
    clickhouse_password = "password"
    clickhouse_database = "test_db"
    redis_url = "redis://localhost:6379"

# Mock modules
mock_settings = MockSettings()
sys.modules['config.settings'] = Mock()
sys.modules['config.settings'].settings = mock_settings
sys.modules['src.services.clickhouse_service'] = Mock()
sys.modules['src.services.clickhouse_service'].clickhouse_service = Mock()
sys.modules['src.services.redis_cache'] = Mock()
sys.modules['src.services.redis_cache'].redis_cache = Mock()

# Import our modules
from src.validation.label_validator import LabelValidator, ValidationSeverity, ValidationCategory
from src.validation.validation_metrics import ValidationMetricsCollector
from src.models.data_models import Candle, LabelSet, Granularity, EnhancedTripleBarrierLabel, BarrierHit


def test_1_no_lookahead_bias():
    """Test #1: No look-ahead bias detection"""
    print("🧪 Test #1: No look-ahead bias detection")
    
    validator = LabelValidator()
    
    # Valid candle
    candle_ts = datetime(2024, 1, 15, 9, 0, 0)
    candle = Candle(
        instrument_id="EUR/USD",
        granularity=Granularity.H4,
        ts=candle_ts,
        open=1.0500,
        high=1.0580,
        low=1.0450,
        close=1.0520,
        volume=1000.0
    )
    
    # Path data with look-ahead bias (timestamp before candle timestamp)
    lookahead_path_data = [
        {
            "ts": candle_ts - timedelta(hours=1),  # FUTURE DATA - INVALID
            "open": 1.0520, "high": 1.0540, "low": 1.0510, "close": 1.0535
        }
    ]
    
    result = validator.validate_pre_computation(candle, 6, path_data=lookahead_path_data)
    
    lookahead_issues = [i for i in result.issues if i.category == ValidationCategory.LOOKAHEAD_BIAS]
    success = not result.is_valid and len(lookahead_issues) > 0
    
    print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"   Look-ahead bias issues detected: {len(lookahead_issues)}")
    
    return success


def test_2_data_consistency():
    """Test #2: Data consistency validation (MFE >= -MAE)"""
    print("\n🧪 Test #2: Data consistency validation (MFE >= -MAE)")
    
    validator = LabelValidator()
    
    candle = Candle(
        instrument_id="EUR/USD",
        granularity=Granularity.H4,
        ts=datetime(2024, 1, 15, 9, 0, 0),
        open=1.0500,
        high=1.0580,
        low=1.0450,
        close=1.0520,
        volume=1000.0
    )
    
    # Label set that violates MFE >= -MAE constraint
    invalid_label_set = LabelSet(
        instrument_id=candle.instrument_id,
        granularity=candle.granularity,
        ts=candle.ts,
        mfe=0.0020,   # MFE = 0.002
        mae=-0.0030,  # MAE = -0.003, so -MAE = 0.003
                     # MFE (0.002) < -MAE (0.003) - CRITICAL VIOLATION
        profit_factor=2.0
    )
    
    result = validator.validate_post_computation(candle, invalid_label_set)
    
    critical_issues = [i for i in result.issues if i.severity == ValidationSeverity.CRITICAL]
    mfe_mae_issues = [i for i in critical_issues if "MFE must be >= -MAE" in i.message]
    success = not result.is_valid and len(mfe_mae_issues) > 0
    
    print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"   Critical MFE/MAE constraint violations: {len(mfe_mae_issues)}")
    
    return success


def test_3_timestamp_alignment():
    """Test #3: Timestamp alignment (H4 at 1,5,9,13,17,21 UTC)"""
    print("\n🧪 Test #3: Timestamp alignment (H4 at 1,5,9,13,17,21 UTC)")
    
    validator = LabelValidator()
    
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
    
    alignment_issues = [i for i in result.issues if i.category == ValidationCategory.TIMESTAMP_ALIGNMENT]
    h4_alignment_issues = [i for i in alignment_issues if "H4 timestamp not at expected hours" in i.message]
    success = not result.is_valid and len(h4_alignment_issues) > 0
    
    print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"   H4 alignment issues detected: {len(h4_alignment_issues)}")
    
    return success


def test_4_barrier_logic():
    """Test #4: Barrier logic validation (barriers properly ordered)"""
    print("\n🧪 Test #4: Barrier logic validation")
    
    validator = LabelValidator()
    
    candle = Candle(
        instrument_id="EUR/USD",
        granularity=Granularity.H4,
        ts=datetime(2024, 1, 15, 9, 0, 0),
        open=1.0500,
        high=1.0580,
        low=1.0450,
        close=1.0520,
        volume=1000.0
    )
    
    # Invalid Enhanced Triple Barrier (upper < lower)
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
        instrument_id=candle.instrument_id,
        granularity=candle.granularity,
        ts=candle.ts,
        enhanced_triple_barrier=invalid_etb
    )
    
    result = validator.validate_post_computation(candle, invalid_label_set)
    
    barrier_issues = [i for i in result.issues if "Upper barrier must be greater than lower barrier" in i.message]
    success = not result.is_valid and len(barrier_issues) > 0
    
    print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"   Barrier logic issues detected: {len(barrier_issues)}")
    
    return success


def test_5_validation_metrics():
    """Test #5: Validation metrics collection"""
    print("\n🧪 Test #5: Validation metrics collection")
    
    validator = LabelValidator()
    metrics_collector = ValidationMetricsCollector()
    
    # Create test candles: some valid, some invalid
    candles = [
        # Valid candle
        Candle(
            instrument_id="EUR/USD", granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 9, 0, 0),
            open=1.0500, high=1.0580, low=1.0450, close=1.0520, volume=1000.0
        ),
        # Invalid timestamp
        Candle(
            instrument_id="EUR/USD", granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 8, 0, 0),  # Invalid H4 hour
            open=1.0500, high=1.0580, low=1.0450, close=1.0520, volume=1000.0
        ),
        # Valid candle
        Candle(
            instrument_id="EUR/USD", granularity=Granularity.H4,
            ts=datetime(2024, 1, 15, 13, 0, 0),
            open=1.0520, high=1.0590, low=1.0470, close=1.0540, volume=1100.0
        )
    ]
    
    # Run validations and collect metrics
    for candle in candles:
        result = validator.validate_pre_computation(candle, 6)
        metrics_collector.record_validation_result(result)
    
    # Check metrics
    current_metrics = metrics_collector.get_current_metrics()
    summary = metrics_collector.get_metrics_summary()
    
    success = (
        current_metrics.total_validations == 3 and
        current_metrics.successful_validations == 2 and  # 2 valid candles
        current_metrics.failed_validations == 1 and     # 1 invalid timestamp
        summary["failure_rate"] > 0
    )
    
    print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"   Total validations: {current_metrics.total_validations}")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Failure rate: {summary['failure_rate']:.1%}")
    
    return success


def test_6_performance():
    """Test #6: Validation performance"""
    print("\n🧪 Test #6: Validation performance")
    
    validator = LabelValidator()
    
    # Create 1000 valid candles for performance testing
    candles = []
    base_time = datetime(2024, 1, 15, 1, 0, 0)
    
    for i in range(1000):
        candles.append(Candle(
            instrument_id="EUR/USD",
            granularity=Granularity.H4,
            ts=base_time + timedelta(hours=4 * i),
            open=1.0500,
            high=1.0580,
            low=1.0450,
            close=1.0520,
            volume=1000.0
        ))
    
    # Time the validation
    start_time = datetime.now()
    
    for candle in candles:
        result = validator.validate_pre_computation(candle, 6)
    
    end_time = datetime.now()
    total_time_ms = (end_time - start_time).total_seconds() * 1000
    avg_time_per_validation = total_time_ms / len(candles)
    
    # Performance should be reasonable (< 5ms per validation)
    success = avg_time_per_validation < 5.0
    
    print(f"   Result: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"   Validated {len(candles)} candles in {total_time_ms:.2f}ms")
    print(f"   Average time per validation: {avg_time_per_validation:.2f}ms")
    
    return success


def main():
    """Run comprehensive validation framework tests"""
    print("🚀 Comprehensive Validation Framework Tests (Issue #8)")
    print("=" * 70)
    
    # Test suite
    tests = [
        ("No look-ahead bias detection", test_1_no_lookahead_bias),
        ("Data consistency (MFE >= -MAE)", test_2_data_consistency),
        ("Timestamp alignment (H4)", test_3_timestamp_alignment),
        ("Barrier logic validation", test_4_barrier_logic),
        ("Validation metrics collection", test_5_validation_metrics),
        ("Performance validation", test_6_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests PASSED! Comprehensive validation framework is working correctly.")
        print("\n🎯 Issue #8 Implementation Complete:")
        print("   ✅ No look-ahead bias validation")
        print("   ✅ Data consistency checks (MFE >= -MAE)")
        print("   ✅ Timestamp alignment validation")
        print("   ✅ Barrier logic validation")
        print("   ✅ Validation metrics and alerting")
        print("   ✅ Performance monitoring")
        return 0
    else:
        print(f"❌ {total - passed} tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())