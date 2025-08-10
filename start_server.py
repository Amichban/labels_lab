#!/usr/bin/env python3
"""
Production server startup script for Label Computation System

This script provides production-ready server startup with:
- Proper logging configuration
- Environment validation
- Health checks before serving
- Graceful shutdown handling
"""

import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

import uvicorn

from config.settings import settings


def setup_logging():
    """Configure logging for production"""
    log_level = logging.DEBUG if settings.debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/app.log') if Path('logs').exists() else logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from some libraries
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('clickhouse_driver').setLevel(logging.WARNING)


def validate_environment():
    """Validate critical environment variables"""
    required_settings = [
        'clickhouse_host',
        'clickhouse_password',
        'redis_host'
    ]
    
    missing = []
    for setting in required_settings:
        if not getattr(settings, setting, None):
            missing.append(setting.upper())
    
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("Please check your .env file or environment configuration.")
        sys.exit(1)


async def check_dependencies():
    """Check that all dependencies are available before starting"""
    from src.services.clickhouse_service import clickhouse_service
    from src.services.redis_cache import redis_cache
    
    logger = logging.getLogger(__name__)
    
    # Check ClickHouse
    try:
        if not clickhouse_service.check_connection():
            logger.error("ClickHouse connection failed")
            return False
        logger.info("ClickHouse connection verified")
    except Exception as e:
        logger.error(f"ClickHouse connection error: {e}")
        return False
    
    # Check Redis
    try:
        if not redis_cache.check_connection():
            logger.error("Redis connection failed")
            return False
        logger.info("Redis connection verified")
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        return False
    
    return True


def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'temp']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def signal_handler(signum, frame):
    """Handle graceful shutdown"""
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


def main():
    """Main entry point"""
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Environment: {settings.environment}")
    
    # Setup
    setup_logging()
    create_directories()
    validate_environment()
    
    logger = logging.getLogger(__name__)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check dependencies
    logger.info("Checking dependencies...")
    if not asyncio.run(check_dependencies()):
        logger.error("Dependency check failed, exiting")
        sys.exit(1)
    
    logger.info("All dependencies verified, starting server...")
    
    # Configure uvicorn
    uvicorn_config = {
        "app": "main:app",
        "host": settings.api_host,
        "port": settings.api_port,
        "log_level": "info",
        "access_log": True,
        "loop": "auto",
        "http": "auto",
    }
    
    # Production settings
    if not settings.debug:
        uvicorn_config.update({
            "workers": settings.api_workers,
            "log_level": "warning",  # Reduce noise in production
        })
    else:
        uvicorn_config.update({
            "reload": True,
            "reload_dirs": ["src", "config"],
        })
    
    logger.info(f"Server configuration: {uvicorn_config}")
    
    try:
        uvicorn.run(**uvicorn_config)
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()