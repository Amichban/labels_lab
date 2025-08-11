"""Test configuration to avoid dependency issues during testing"""

import os
import sys
from unittest.mock import Mock

# Mock settings for testing to avoid ClickHouse connection requirements
class MockSettings:
    clickhouse_host = "localhost"
    clickhouse_port = 8123
    clickhouse_user = "default"
    clickhouse_password = "password"
    clickhouse_database = "test_db"
    redis_url = "redis://localhost:6379"

# Mock the settings module
mock_settings = MockSettings()
sys.modules['config.settings'] = Mock()
sys.modules['config.settings'].settings = mock_settings

# Mock ClickHouse service
sys.modules['src.services.clickhouse_service'] = Mock()
sys.modules['src.services.clickhouse_service'].clickhouse_service = Mock()

# Mock Redis cache
sys.modules['src.services.redis_cache'] = Mock()
sys.modules['src.services.redis_cache'].redis_cache = Mock()