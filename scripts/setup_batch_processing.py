#!/usr/bin/env python3
"""
Setup Script for Batch Processing System

Initializes and configures the complete batch processing pipeline:
- Validates system requirements and dependencies
- Sets up Redis keys and data structures
- Initializes ClickHouse schemas if needed
- Configures worker pool and monitoring
- Runs system health checks
- Generates example configurations
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.services.batch_backfill_service import batch_backfill_service
from src.services.batch_metrics_service import metrics_collector
from src.services.batch_worker_pool import batch_worker_pool
from src.services.batch_error_handler import batch_error_handler
from src.services.redis_cache import redis_cache
from src.services.clickhouse_service import clickhouse_service
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_system_requirements():
    """Check system requirements and dependencies"""
    
    logger.info("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required")
        return False
    
    # Check required packages
    required_packages = [
        'redis', 'clickhouse_driver', 'pydantic', 'fastapi',
        'msgpack', 'psutil', 'tabulate', 'rich', 'click'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✓ System requirements check passed")
    return True


def check_service_connections():
    """Check connections to external services"""
    
    logger.info("Checking service connections...")
    
    # Check Redis connection
    try:
        if redis_cache.check_connection():
            logger.info("✓ Redis connection successful")
        else:
            logger.error("✗ Redis connection failed")
            return False
    except Exception as e:
        logger.error(f"✗ Redis connection error: {e}")
        return False
    
    # Check ClickHouse connection
    try:
        if clickhouse_service.check_connection():
            logger.info("✓ ClickHouse connection successful")
        else:
            logger.error("✗ ClickHouse connection failed")
            return False
    except Exception as e:
        logger.error(f"✗ ClickHouse connection error: {e}")
        return False
    
    return True


def initialize_redis_structures():
    """Initialize Redis data structures for batch processing"""
    
    logger.info("Initializing Redis structures...")
    
    try:
        # Clear any existing test data
        patterns_to_clear = [
            "batch_job:*",
            "batch_chunk:*",
            "batch_metrics:*",
            "batch_errors:*"
        ]
        
        for pattern in patterns_to_clear:
            keys = list(redis_cache.client.scan_iter(match=pattern))
            if keys:
                redis_cache.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} keys matching {pattern}")
        
        # Initialize system metrics
        metrics_collector.metrics['system_initialized'].add_point(1.0)
        redis_cache.set("batch_system:initialized", {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        })
        
        logger.info("✓ Redis structures initialized")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize Redis structures: {e}")
        return False


def setup_worker_pool():
    """Setup and test worker pool"""
    
    logger.info("Setting up worker pool...")
    
    try:
        # Start worker pool
        batch_worker_pool.start()
        
        # Submit a test task
        success = batch_worker_pool.submit_task(
            task_id="setup_test_task",
            job_id="setup_test_job",
            function_name="test_task",
            test_data="hello"
        )
        
        if success:
            logger.info("✓ Worker pool test task submitted")
        else:
            logger.warning("Worker pool task submission failed")
        
        # Check metrics
        metrics = batch_worker_pool.get_metrics()
        logger.info(f"Worker pool: {metrics.total_workers} workers, {metrics.queue_size} queued tasks")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Worker pool setup failed: {e}")
        return False


def start_metrics_collection():
    """Start metrics collection system"""
    
    logger.info("Starting metrics collection...")
    
    try:
        # Start metrics collection with 10-second interval
        metrics_collector.start_collection(interval_seconds=10)
        
        # Record initial metrics
        metrics_collector.record_job_metric("system", "startup_time", time.time())
        
        logger.info("✓ Metrics collection started")
        return True
        
    except Exception as e:
        logger.error(f"✗ Metrics collection startup failed: {e}")
        return False


def create_example_configurations():
    """Create example configuration files"""
    
    logger.info("Creating example configurations...")
    
    try:
        # Create examples directory
        examples_dir = project_root / "examples"
        examples_dir.mkdir(exist_ok=True)
        
        # Example batch job configuration
        example_job_config = {
            "job_name": "Monthly EUR/USD H4 Backfill",
            "instrument_id": "EURUSD",
            "granularity": "H4",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-31T23:59:59Z",
            "label_types": [
                "enhanced_triple_barrier",
                "vol_scaled_return",
                "mfe_mae",
                "level_retouch_count",
                "breakout_beyond_level",
                "flip_within_horizon"
            ],
            "options": {
                "chunk_size": 10000,
                "parallel_workers": 8,
                "force_recompute": False,
                "priority": "normal"
            },
            "monitoring": {
                "alert_on_error_rate": 0.05,
                "alert_on_low_throughput": 500000,
                "progress_notification_interval": 300
            }
        }
        
        with open(examples_dir / "batch_job_config.json", "w") as f:
            json.dump(example_job_config, f, indent=2)
        
        # Example CLI commands script
        cli_examples = """#!/bin/bash
# Batch Processing CLI Examples

# Start a batch job
python -m src.cli.batch_cli start \\
  --instrument EURUSD \\
  --granularity H4 \\
  --start-date 2024-01-01 \\
  --end-date 2024-01-31 \\
  --labels enhanced_triple_barrier vol_scaled_return \\
  --chunk-size 10000 \\
  --workers 8 \\
  --priority normal

# Monitor a running job
python -m src.cli.batch_cli monitor JOB_ID --refresh 5

# List all jobs
python -m src.cli.batch_cli status --format table

# Get system metrics
python -m src.cli.batch_cli metrics --format json

# Cleanup old jobs
python -m src.cli.batch_cli cleanup --older-than 7 --status completed

# Dry run to estimate scope
python -m src.cli.batch_cli start \\
  --instrument GBPUSD \\
  --granularity H1 \\
  --start-date 2024-01-01 \\
  --end-date 2024-12-31 \\
  --dry-run
"""
        
        with open(examples_dir / "cli_examples.sh", "w") as f:
            f.write(cli_examples)
        
        # Make script executable
        (examples_dir / "cli_examples.sh").chmod(0o755)
        
        logger.info(f"✓ Example configurations created in {examples_dir}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create example configurations: {e}")
        return False


def run_health_checks():
    """Run comprehensive system health checks"""
    
    logger.info("Running system health checks...")
    
    health_results = []
    
    # Check Redis performance
    try:
        start_time = time.time()
        redis_cache.set("health_check", "test_value")
        value = redis_cache.get("health_check")
        redis_time = (time.time() - start_time) * 1000
        
        if value == "test_value" and redis_time < 10:  # Less than 10ms
            health_results.append(("Redis Performance", "✓", f"{redis_time:.1f}ms"))
        else:
            health_results.append(("Redis Performance", "⚠", f"{redis_time:.1f}ms (slow)"))
            
    except Exception as e:
        health_results.append(("Redis Performance", "✗", str(e)))
    
    # Check ClickHouse performance
    try:
        start_time = time.time()
        result = clickhouse_service.execute("SELECT 1 as test")
        ch_time = (time.time() - start_time) * 1000
        
        if result and result[0]['test'] == 1 and ch_time < 100:  # Less than 100ms
            health_results.append(("ClickHouse Performance", "✓", f"{ch_time:.1f}ms"))
        else:
            health_results.append(("ClickHouse Performance", "⚠", f"{ch_time:.1f}ms (slow)"))
            
    except Exception as e:
        health_results.append(("ClickHouse Performance", "✗", str(e)))
    
    # Check system resources
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_results.append(("CPU Usage", "✓" if cpu_percent < 80 else "⚠", f"{cpu_percent:.1f}%"))
        health_results.append(("Memory Usage", "✓" if memory.percent < 80 else "⚠", f"{memory.percent:.1f}%"))
        health_results.append(("Disk Usage", "✓" if disk.percent < 80 else "⚠", f"{disk.percent:.1f}%"))
        
    except Exception as e:
        health_results.append(("System Resources", "✗", str(e)))
    
    # Check worker pool health
    try:
        metrics = batch_worker_pool.get_metrics()
        if metrics.total_workers > 0:
            health_results.append(("Worker Pool", "✓", f"{metrics.total_workers} workers"))
        else:
            health_results.append(("Worker Pool", "✗", "No active workers"))
            
    except Exception as e:
        health_results.append(("Worker Pool", "✗", str(e)))
    
    # Display results
    logger.info("\nHealth Check Results:")
    logger.info("=" * 60)
    
    all_healthy = True
    for check_name, status, details in health_results:
        logger.info(f"{check_name:<25} {status} {details}")
        if status == "✗":
            all_healthy = False
    
    logger.info("=" * 60)
    
    if all_healthy:
        logger.info("🎉 All health checks passed! System is ready for production use.")
    else:
        logger.warning("⚠️  Some health checks failed. Review the results above.")
    
    return all_healthy


def display_system_info():
    """Display system information and next steps"""
    
    logger.info("\n" + "=" * 80)
    logger.info("BATCH PROCESSING SYSTEM SETUP COMPLETE")
    logger.info("=" * 80)
    
    logger.info(f"""
System Configuration:
- Redis Host: {settings.redis_host}:{settings.redis_port}
- ClickHouse Host: {settings.clickhouse_host}:{settings.clickhouse_port}
- Worker Pool: {batch_worker_pool.max_workers} max workers
- Chunk Size: {batch_backfill_service.chunk_size:,} candles per chunk
- Target Throughput: 1,000,000+ candles/minute

Next Steps:
1. Start a batch job:
   python -m src.cli.batch_cli start --instrument EURUSD --granularity H4 \\
     --start-date 2024-01-01 --end-date 2024-01-31

2. Monitor progress:
   python -m src.cli.batch_cli status JOB_ID

3. View system metrics:
   python -m src.cli.batch_cli metrics

4. Check example configurations:
   ls {project_root}/examples/

For help: python -m src.cli.batch_cli --help
""")
    
    logger.info("=" * 80)


async def main():
    """Main setup function"""
    
    logger.info("🚀 Starting Batch Processing System Setup...")
    logger.info(f"Project Root: {project_root}")
    
    setup_steps = [
        ("System Requirements", check_system_requirements),
        ("Service Connections", check_service_connections),
        ("Redis Structures", initialize_redis_structures),
        ("Worker Pool", setup_worker_pool),
        ("Metrics Collection", start_metrics_collection),
        ("Example Configurations", create_example_configurations),
    ]
    
    failed_steps = []
    
    for step_name, step_function in setup_steps:
        logger.info(f"\n📋 {step_name}...")
        
        try:
            if asyncio.iscoroutinefunction(step_function):
                success = await step_function()
            else:
                success = step_function()
            
            if success:
                logger.info(f"✅ {step_name} completed successfully")
            else:
                logger.error(f"❌ {step_name} failed")
                failed_steps.append(step_name)
                
        except Exception as e:
            logger.error(f"❌ {step_name} failed with exception: {e}")
            failed_steps.append(step_name)
    
    # Run health checks
    logger.info("\n🏥 Running Health Checks...")
    health_passed = run_health_checks()
    
    # Final results
    if not failed_steps and health_passed:
        logger.info("\n🎉 Setup completed successfully!")
        display_system_info()
        return True
    else:
        logger.error(f"\n❌ Setup completed with errors:")
        if failed_steps:
            logger.error(f"Failed steps: {', '.join(failed_steps)}")
        if not health_passed:
            logger.error("Health checks failed")
        
        logger.info("\nPlease resolve the issues above and run setup again.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\nSetup interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Setup failed with unexpected error: {e}")
        sys.exit(1)
        
    finally:
        # Cleanup
        try:
            batch_worker_pool.shutdown(wait=True, timeout=5)
            metrics_collector.stop_collection()
        except:
            pass