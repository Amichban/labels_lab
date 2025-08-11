#!/usr/bin/env python3
"""
Debug validation test to find the enum issue
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

# Now import our validation modules
try:
    from src.validation.label_validator import LabelValidator, ValidationSeverity, ValidationCategory
    print("✅ Successfully imported LabelValidator")
except Exception as e:
    print(f"❌ Failed to import LabelValidator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from src.models.data_models import Candle, LabelSet, Granularity, EnhancedTripleBarrierLabel, BarrierHit
    print("✅ Successfully imported data models")
except Exception as e:
    print(f"❌ Failed to import data models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def test_enums():
    """Test enum values"""
    print("\n🔍 Testing enum values...")
    
    # Test ValidationSeverity
    severity = ValidationSeverity.ERROR
    print(f"ValidationSeverity.ERROR = {severity}")
    print(f"Has .value attribute: {hasattr(severity, 'value')}")
    if hasattr(severity, 'value'):
        print(f"Value: {severity.value}")
    else:
        print(f"String representation: {str(severity)}")
    
    # Test ValidationCategory
    category = ValidationCategory.DATA_CONSISTENCY
    print(f"ValidationCategory.DATA_CONSISTENCY = {category}")
    print(f"Has .value attribute: {hasattr(category, 'value')}")
    if hasattr(category, 'value'):
        print(f"Value: {category.value}")
    else:
        print(f"String representation: {str(category)}")
    
    # Test Granularity
    granularity = Granularity.H4
    print(f"Granularity.H4 = {granularity}")
    print(f"Has .value attribute: {hasattr(granularity, 'value')}")
    if hasattr(granularity, 'value'):
        print(f"Value: {granularity.value}")
    else:
        print(f"String representation: {str(granularity)}")

def test_basic_validation():
    """Test basic validation step by step"""
    print("\n🧪 Testing validation step by step...")
    
    try:
        # Create validator
        validator = LabelValidator()
        print("✅ Created validator")
        
        # Create candle
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
        print("✅ Created candle")
        
        # Test validation
        print("🔍 Running validation...")
        result = validator.validate_pre_computation(candle, 6)
        print("✅ Validation completed")
        
        print(f"Result is_valid: {result.is_valid}")
        print(f"Number of issues: {len(result.issues)}")
        
        for i, issue in enumerate(result.issues):
            print(f"Issue {i+1}:")
            print(f"  severity: {issue.severity} (type: {type(issue.severity)})")
            print(f"  category: {issue.category} (type: {type(issue.category)})")
            print(f"  message: {issue.message}")
            
            # Try to access value attribute
            try:
                severity_val = issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity)
                category_val = issue.category.value if hasattr(issue.category, 'value') else str(issue.category)
                print(f"  severity.value: {severity_val}")
                print(f"  category.value: {category_val}")
            except Exception as e:
                print(f"  Error accessing value: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run debug tests"""
    print("🔍 Starting validation debug tests...")
    print("=" * 50)
    
    test_enums()
    test_basic_validation()
    
    print("\n" + "=" * 50)
    print("🏁 Debug tests completed")

if __name__ == "__main__":
    main()