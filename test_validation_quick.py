#!/usr/bin/env python3
"""
Quick validation framework test - essential functionality only
"""

import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Mock dependencies
class MockSettings:
    clickhouse_host = "localhost"
    clickhouse_port = 8123
    clickhouse_user = "default"
    clickhouse_password = "password"
    clickhouse_database = "test_db"
    redis_url = "redis://localhost:6379"

mock_settings = MockSettings()
sys.modules['config.settings'] = Mock()
sys.modules['config.settings'].settings = mock_settings
sys.modules['src.services.clickhouse_service'] = Mock()
sys.modules['src.services.clickhouse_service'].clickhouse_service = Mock()
sys.modules['src.services.redis_cache'] = Mock()
sys.modules['src.services.redis_cache'].redis_cache = Mock()

from src.validation.label_validator import LabelValidator, ValidationSeverity, ValidationCategory
from src.models.data_models import Candle, LabelSet, Granularity, EnhancedTripleBarrierLabel, BarrierHit


def test_core_validations():
    """Test core validation functionality"""
    print("🧪 Testing core validation functionality...")
    
    validator = LabelValidator()
    tests_passed = 0
    
    # Test 1: Look-ahead bias detection
    print("\n1️⃣ Look-ahead bias detection")
    candle_ts = datetime(2024, 1, 15, 9, 0, 0)
    candle = Candle(
        instrument_id="EUR/USD", granularity=Granularity.H4, ts=candle_ts,
        open=1.0500, high=1.0580, low=1.0450, close=1.0520, volume=1000.0
    )
    
    lookahead_path_data = [{"ts": candle_ts - timedelta(hours=1), "high": 1.0540, "low": 1.0510}]
    result = validator.validate_pre_computation(candle, 6, path_data=lookahead_path_data)
    
    lookahead_issues = [i for i in result.issues if i.category == ValidationCategory.LOOKAHEAD_BIAS]
    if not result.is_valid and len(lookahead_issues) > 0:
        print("   ✅ PASS - Detected look-ahead bias")
        tests_passed += 1
    else:
        print("   ❌ FAIL - Should have detected look-ahead bias")
    
    # Test 2: MFE/MAE constraint
    print("\n2️⃣ MFE/MAE constraint (MFE >= -MAE)")
    invalid_label_set = LabelSet(
        instrument_id=candle.instrument_id, granularity=candle.granularity, ts=candle.ts,
        mfe=0.0020, mae=-0.0030  # MFE < -MAE violation
    )
    
    result = validator.validate_post_computation(candle, invalid_label_set)
    critical_issues = [i for i in result.issues if i.severity == ValidationSeverity.CRITICAL]
    
    if not result.is_valid and len(critical_issues) > 0:
        print("   ✅ PASS - Detected MFE/MAE constraint violation")
        tests_passed += 1
    else:
        print("   ❌ FAIL - Should have detected MFE/MAE violation")
    
    # Test 3: H4 timestamp alignment
    print("\n3️⃣ H4 timestamp alignment")
    invalid_h4_candle = Candle(
        instrument_id="EUR/USD", granularity=Granularity.H4,
        ts=datetime(2024, 1, 15, 8, 0, 0),  # 8:00 UTC - invalid for H4
        open=1.0500, high=1.0580, low=1.0450, close=1.0520, volume=1000.0
    )
    
    result = validator.validate_pre_computation(invalid_h4_candle, 6)
    alignment_issues = [i for i in result.issues if i.category == ValidationCategory.TIMESTAMP_ALIGNMENT]
    
    if not result.is_valid and len(alignment_issues) > 0:
        print("   ✅ PASS - Detected H4 timestamp misalignment")
        tests_passed += 1
    else:
        print("   ❌ FAIL - Should have detected H4 timestamp issue")
    
    # Test 4: Valid candle passes
    print("\n4️⃣ Valid candle validation")
    valid_candle = Candle(
        instrument_id="EUR/USD", granularity=Granularity.H4,
        ts=datetime(2024, 1, 15, 9, 0, 0),  # Valid H4 timestamp
        open=1.0500, high=1.0580, low=1.0450, close=1.0520, volume=1000.0
    )
    
    result = validator.validate_pre_computation(valid_candle, 6)
    
    if result.is_valid:
        print("   ✅ PASS - Valid candle passes validation")
        tests_passed += 1
    else:
        print("   ❌ FAIL - Valid candle should pass validation")
        for issue in result.issues:
            print(f"     Issue: {issue.message}")
    
    return tests_passed, 4


def main():
    """Run quick validation tests"""
    print("🚀 Quick Validation Framework Tests (Issue #8)")
    print("=" * 60)
    
    try:
        passed, total = test_core_validations()
        
        print("\n" + "=" * 60)
        print(f"🏁 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ All core validation tests PASSED!")
            print("\n🎯 Issue #8 Implementation Verified:")
            print("   ✅ Look-ahead bias detection")
            print("   ✅ Data consistency validation (MFE >= -MAE)")
            print("   ✅ Timestamp alignment (H4 at 1,5,9,13,17,21 UTC)")
            print("   ✅ Valid data passes validation")
            return 0
        else:
            print(f"❌ {total - passed} tests FAILED!")
            return 1
            
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())