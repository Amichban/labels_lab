#!/usr/bin/env python3
"""
Firestore Real-time Streaming Demo

Demonstrates the complete Firestore listener system for Issue #11:
- Real-time candle streaming from Firestore
- Multi-instrument/granularity coordination
- Rate limiting and backpressure handling
- Error handling and recovery
- Performance monitoring

Usage:
    python examples/firestore_streaming_demo.py --instruments EUR_USD,GBP_USD --granularities H1,H4
"""

import asyncio
import logging
import argparse
import signal
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our services
from src.services.firestore_listener import firestore_listener
from src.services.stream_manager import stream_manager, StreamPriority
from src.models.data_models import Candle
from config.settings import settings


class FirestoreStreamingDemo:
    """
    Complete demonstration of Firestore real-time streaming system
    """
    
    def __init__(self, instruments: List[str], granularities: List[str]):
        """
        Initialize demo with specified instruments and granularities
        
        Args:
            instruments: List of instrument IDs (e.g., ['EUR_USD', 'GBP_USD'])
            granularities: List of granularities (e.g., ['H1', 'H4'])
        """
        self.instruments = instruments
        self.granularities = granularities
        self.running = False
        self.candle_count = 0
        self.start_time = None
        self.stats_task = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def setup_streams(self):
        """Setup all streams for instruments and granularities"""
        logger.info("Setting up Firestore streams...")
        
        # Configure priority based on common trading importance
        priority_map = {
            'EUR_USD': StreamPriority.CRITICAL,
            'GBP_USD': StreamPriority.HIGH,
            'USD_JPY': StreamPriority.HIGH,
            'USD_CHF': StreamPriority.MEDIUM,
            'AUD_USD': StreamPriority.MEDIUM,
            'NZD_USD': StreamPriority.LOW,
        }
        
        stream_ids = []
        
        for instrument in self.instruments:
            for granularity in self.granularities:
                # Get priority for this instrument
                priority = priority_map.get(instrument, StreamPriority.MEDIUM)
                
                # Add to stream manager
                stream_id = await stream_manager.add_stream(
                    instrument, 
                    granularity, 
                    priority
                )
                stream_ids.append(stream_id)
                
                # Add to Firestore listener with callback
                firestore_listener.add_stream(
                    instrument,
                    granularity,
                    callback=self._create_candle_callback(instrument, granularity)
                )
                
                logger.info(f"Added stream: {stream_id} (priority: {priority.name})")
        
        logger.info(f"Setup complete: {len(stream_ids)} streams configured")
        return stream_ids
    
    def _create_candle_callback(self, instrument: str, granularity: str):
        """Create callback function for processing candles"""
        
        async def process_candle(candle: Candle):
            """Process received candle"""
            self.candle_count += 1
            
            logger.info(
                f"Received candle: {instrument} {granularity} "
                f"@ {candle.ts} - O:{candle.open:.5f} H:{candle.high:.5f} "
                f"L:{candle.low:.5f} C:{candle.close:.5f} V:{candle.volume:.0f}"
            )
            
            # In a real application, this is where you'd:
            # 1. Validate the candle
            # 2. Store it in your database
            # 3. Trigger label computation
            # 4. Update ML models
            # 5. Generate trading signals
            
            # For demo, we'll just log some statistics periodically
            if self.candle_count % 10 == 0:
                await self._log_performance_stats()
        
        return process_candle
    
    async def start_streaming(self):
        """Start all streams and begin processing"""
        logger.info("Starting Firestore streaming system...")
        self.start_time = datetime.utcnow()
        self.running = True
        
        try:
            # Start stream manager streams
            stream_results = await stream_manager.start_all_streams()
            active_count = sum(1 for success in stream_results.values() if success)
            logger.info(f"Stream Manager: Started {active_count}/{len(stream_results)} streams")
            
            # Start Firestore listener streams  
            listener_results = await firestore_listener.start_all_streams()
            active_listener_count = sum(1 for success in listener_results.values() if success)
            logger.info(f"Firestore Listener: Started {active_listener_count}/{len(listener_results)} streams")
            
            if active_count == 0 and active_listener_count == 0:
                logger.error("No streams started successfully!")
                return False
            
            # Start background statistics reporting
            self.stats_task = asyncio.create_task(self._stats_reporter())
            
            logger.info("🚀 Firestore streaming system is now active!")
            logger.info("Press Ctrl+C to stop gracefully...")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    async def _stats_reporter(self):
        """Background task to report statistics"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Report every 30 seconds
                await self._log_comprehensive_stats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats reporter: {e}")
    
    async def _log_performance_stats(self):
        """Log basic performance statistics"""
        if not self.start_time:
            return
        
        uptime = datetime.utcnow() - self.start_time
        rate = self.candle_count / max(uptime.total_seconds(), 1)
        
        logger.info(
            f"📊 Performance: {self.candle_count} candles processed "
            f"in {uptime.total_seconds():.1f}s ({rate:.2f} candles/sec)"
        )
    
    async def _log_comprehensive_stats(self):
        """Log comprehensive system statistics"""
        logger.info("=" * 60)
        logger.info("📈 COMPREHENSIVE SYSTEM STATISTICS")
        logger.info("=" * 60)
        
        # Stream Manager stats
        manager_status = stream_manager.get_stream_status()
        global_metrics = manager_status["global_metrics"]
        
        logger.info(f"Stream Manager:")
        logger.info(f"  Active Streams: {global_metrics['active_streams']}/{global_metrics['total_streams']}")
        logger.info(f"  Documents Processed: {global_metrics['total_documents_processed']}")
        logger.info(f"  Total Errors: {global_metrics['total_errors']}")
        logger.info(f"  Backpressure Active: {global_metrics.get('backpressure_active', False)}")
        
        # Firestore Listener stats
        listener_health = await firestore_listener.health_check()
        
        logger.info(f"Firestore Listener:")
        logger.info(f"  Overall Status: {listener_health['overall_status']}")
        logger.info(f"  Active Streams: {listener_health['active_streams']}/{listener_health['total_streams']}")
        logger.info(f"  Processing Rate: {listener_health['metrics']['processing_rate_per_second']:.2f} docs/sec")
        logger.info(f"  Success Rate: {listener_health['metrics']['success_rate']:.1%}")
        
        # Dead Letter Queue stats
        dlq_stats = listener_health['dead_letter_queue']
        if dlq_stats['current_size'] > 0:
            logger.warning(f"  Dead Letter Queue: {dlq_stats['current_size']} items ({dlq_stats['total_failed']} total failed)")
        
        # Stream-specific stats
        logger.info(f"Stream Details:")
        for stream_id, stream_status in manager_status["streams"].items():
            if isinstance(stream_status, dict):
                metrics = stream_status.get("metrics", {})
                logger.info(
                    f"  {stream_id}: {stream_status.get('status', 'unknown')} - "
                    f"{metrics.get('documents_processed', 0)} docs, "
                    f"{metrics.get('error_count', 0)} errors"
                )
        
        # Performance stats
        if self.start_time:
            uptime = datetime.utcnow() - self.start_time
            rate = self.candle_count / max(uptime.total_seconds(), 1)
            logger.info(f"Demo Performance:")
            logger.info(f"  Uptime: {uptime}")
            logger.info(f"  Total Candles: {self.candle_count}")
            logger.info(f"  Processing Rate: {rate:.2f} candles/sec")
        
        logger.info("=" * 60)
    
    async def run_health_checks(self):
        """Run periodic health checks"""
        logger.info("🏥 Running system health checks...")
        
        # Check Stream Manager
        manager_status = stream_manager.get_stream_status()
        manager_healthy = (
            manager_status["global_metrics"]["active_streams"] > 0 and
            manager_status["global_metrics"]["total_errors"] < 100
        )
        
        # Check Firestore Listener
        listener_health = await firestore_listener.health_check()
        listener_healthy = listener_health["overall_status"] in ["healthy", "partial"]
        
        if manager_healthy and listener_healthy:
            logger.info("✅ System health check: HEALTHY")
        else:
            logger.warning("⚠️  System health check: DEGRADED")
            
            if not manager_healthy:
                logger.warning("   - Stream Manager issues detected")
            if not listener_healthy:
                logger.warning(f"   - Firestore Listener status: {listener_health['overall_status']}")
        
        return manager_healthy and listener_healthy
    
    async def run_performance_optimization(self):
        """Run performance optimization"""
        logger.info("🔧 Running performance optimization...")
        
        result = await stream_manager.optimize_performance()
        
        if result["status"] == "completed":
            if result["optimizations_applied"] > 0:
                logger.info(f"✅ Applied {result['optimizations_applied']} optimizations:")
                for detail in result.get("details", []):
                    logger.info(f"   - {detail}")
            else:
                logger.info("✅ System already optimized")
        else:
            logger.warning(f"⚠️  Optimization {result['status']}")
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("🛑 Initiating graceful shutdown...")
        
        # Cancel background tasks
        if self.stats_task:
            self.stats_task.cancel()
            try:
                await self.stats_task
            except asyncio.CancelledError:
                pass
        
        # Stop all streams
        logger.info("Stopping Stream Manager...")
        await stream_manager.stop_all_streams()
        
        logger.info("Stopping Firestore Listener...")
        await firestore_listener.stop_all_streams()
        
        # Final statistics
        await self._log_comprehensive_stats()
        
        logger.info("✅ Graceful shutdown complete")
    
    async def run(self):
        """Main execution loop"""
        logger.info("🔥 Starting Firestore Real-time Streaming Demo")
        logger.info(f"Instruments: {', '.join(self.instruments)}")
        logger.info(f"Granularities: {', '.join(self.granularities)}")
        
        try:
            # Setup streams
            stream_ids = await self.setup_streams()
            
            # Start streaming
            if not await self.start_streaming():
                logger.error("Failed to start streaming system")
                return
            
            # Main loop - run until interrupted
            health_check_interval = 300  # 5 minutes
            optimization_interval = 600  # 10 minutes
            last_health_check = datetime.utcnow()
            last_optimization = datetime.utcnow()
            
            while self.running:
                try:
                    # Sleep for a short interval
                    await asyncio.sleep(10)
                    
                    # Periodic health checks
                    now = datetime.utcnow()
                    if (now - last_health_check).total_seconds() > health_check_interval:
                        await self.run_health_checks()
                        last_health_check = now
                    
                    # Periodic optimization
                    if (now - last_optimization).total_seconds() > optimization_interval:
                        await self.run_performance_optimization()
                        last_optimization = now
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    # Continue running unless it's a critical error
        
        finally:
            await self.shutdown()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Firestore Real-time Streaming Demo')
    parser.add_argument(
        '--instruments',
        type=str,
        default='EUR_USD,GBP_USD',
        help='Comma-separated list of instruments (default: EUR_USD,GBP_USD)'
    )
    parser.add_argument(
        '--granularities',
        type=str,
        default='H1',
        help='Comma-separated list of granularities (default: H1)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Parse instruments and granularities
    instruments = [i.strip() for i in args.instruments.split(',') if i.strip()]
    granularities = [g.strip() for g in args.granularities.split(',') if g.strip()]
    
    # Validate configuration
    if not settings.gcp_project_id:
        logger.error("GCP_PROJECT_ID not configured. Please set your Google Cloud project ID.")
        sys.exit(1)
    
    if not settings.google_application_credentials:
        logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Using default credentials.")
    
    # Create and run demo
    demo = FirestoreStreamingDemo(instruments, granularities)
    await demo.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)