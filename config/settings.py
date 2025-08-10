"""
Application configuration using Pydantic Settings
"""
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    app_name: str = "Label Computation System"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # ClickHouse
    clickhouse_host: str = Field(env="CLICKHOUSE_HOST")
    clickhouse_port: int = Field(default=9440, env="CLICKHOUSE_PORT")
    clickhouse_user: str = Field(default="default", env="CLICKHOUSE_USER")
    clickhouse_password: str = Field(env="CLICKHOUSE_PASSWORD")
    clickhouse_database: str = Field(default="quantx", env="CLICKHOUSE_DATABASE")
    clickhouse_secure: bool = Field(default=True, env="CLICKHOUSE_SECURE")
    
    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_decode_responses: bool = False  # Use msgpack for serialization
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Label Computation
    batch_chunk_size: int = Field(default=10000, env="BATCH_CHUNK_SIZE")
    parallel_workers: int = Field(default=8, env="PARALLEL_WORKERS")
    cache_ttl_seconds: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    
    # Performance Targets
    max_incremental_latency_ms: int = 100
    target_cache_hit_rate: float = 0.95
    batch_throughput_target: int = 1_000_000  # candles per minute
    
    # Monitoring
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    
    # Firestore (optional)
    gcp_project_id: Optional[str] = Field(default=None, env="GCP_PROJECT_ID")
    google_application_credentials: Optional[str] = Field(
        default=None, env="GOOGLE_APPLICATION_CREDENTIALS"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()