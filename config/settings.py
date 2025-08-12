"""
Application configuration using Pydantic Settings
"""
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator


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
    
    # Firestore Connection
    gcp_project_id: Optional[str] = Field(default=None, env="GCP_PROJECT_ID")
    google_application_credentials: Optional[str] = Field(
        default=None, env="GOOGLE_APPLICATION_CREDENTIALS"
    )
    firestore_emulator_host: Optional[str] = Field(default=None, env="FIRESTORE_EMULATOR_HOST")
    
    # Stream Configuration
    stream_instruments: List[str] = Field(
        default=["EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD"], 
        env="STREAM_INSTRUMENTS"
    )
    stream_granularities: List[str] = Field(
        default=["H1", "H4"], 
        env="STREAM_GRANULARITIES"
    )
    max_concurrent_streams: int = Field(default=8, env="MAX_CONCURRENT_STREAMS")
    
    # Stream Processing
    enable_realtime_streaming: bool = Field(default=False, env="ENABLE_REALTIME_STREAMING")
    stream_processing_rate_limit: int = Field(default=1000, env="STREAM_PROCESSING_RATE_LIMIT")  # per minute
    stream_backpressure_threshold: int = Field(default=10000, env="STREAM_BACKPRESSURE_THRESHOLD")
    
    # Retry Policies
    firestore_max_retry_attempts: int = Field(default=5, env="FIRESTORE_MAX_RETRY_ATTEMPTS")
    firestore_base_retry_delay: float = Field(default=1.0, env="FIRESTORE_BASE_RETRY_DELAY")  # seconds
    firestore_max_retry_delay: float = Field(default=60.0, env="FIRESTORE_MAX_RETRY_DELAY")  # seconds
    
    # Dead Letter Queue
    dead_letter_queue_max_size: int = Field(default=1000, env="DEAD_LETTER_QUEUE_MAX_SIZE")
    dead_letter_queue_retry_interval: int = Field(default=300, env="DEAD_LETTER_QUEUE_RETRY_INTERVAL")  # seconds
    
    # Health Monitoring
    stream_health_check_interval: int = Field(default=30, env="STREAM_HEALTH_CHECK_INTERVAL")  # seconds
    stream_silence_threshold: int = Field(default=600, env="STREAM_SILENCE_THRESHOLD")  # seconds
    enable_stream_auto_recovery: bool = Field(default=True, env="ENABLE_STREAM_AUTO_RECOVERY")
    
    # Performance Optimization
    enable_performance_optimization: bool = Field(default=True, env="ENABLE_PERFORMANCE_OPTIMIZATION")
    performance_optimization_interval: int = Field(default=300, env="PERFORMANCE_OPTIMIZATION_INTERVAL")  # seconds
    
    @validator('stream_instruments', pre=True)
    def parse_stream_instruments(cls, v):
        if isinstance(v, str):
            return [instrument.strip() for instrument in v.split(',') if instrument.strip()]
        return v
    
    @validator('stream_granularities', pre=True)
    def parse_stream_granularities(cls, v):
        if isinstance(v, str):
            return [granularity.strip() for granularity in v.split(',') if granularity.strip()]
        return v
    
    def get_stream_configuration(self) -> Dict[str, Any]:
        """Get complete stream configuration"""
        return {
            "instruments": self.stream_instruments,
            "granularities": self.stream_granularities,
            "max_concurrent_streams": self.max_concurrent_streams,
            "processing_rate_limit": self.stream_processing_rate_limit,
            "backpressure_threshold": self.stream_backpressure_threshold,
            "enable_realtime_streaming": self.enable_realtime_streaming,
            "retry_policy": {
                "max_attempts": self.firestore_max_retry_attempts,
                "base_delay": self.firestore_base_retry_delay,
                "max_delay": self.firestore_max_retry_delay
            },
            "dead_letter_queue": {
                "max_size": self.dead_letter_queue_max_size,
                "retry_interval": self.dead_letter_queue_retry_interval
            },
            "health_monitoring": {
                "check_interval": self.stream_health_check_interval,
                "silence_threshold": self.stream_silence_threshold,
                "auto_recovery": self.enable_stream_auto_recovery
            },
            "performance": {
                "optimization_enabled": self.enable_performance_optimization,
                "optimization_interval": self.performance_optimization_interval
            }
        }
    
    def get_firestore_config(self) -> Dict[str, Any]:
        """Get Firestore connection configuration"""
        config = {
            "project_id": self.gcp_project_id,
            "credentials_path": self.google_application_credentials
        }
        
        if self.firestore_emulator_host:
            config["emulator_host"] = self.firestore_emulator_host
        
        return config
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()