"""
Demo script for the incremental computation engine system.

Demonstrates:
- Starting the complete system
- Processing candles in real-time
- Performance monitoring and alerting
- Health checking and metrics
- Graceful shutdown

Run this script to see the incremental computation engine in action.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List

from src.models.data_models import Candle, Granularity
from src.services.incremental_system_manager import start_incremental_computation_system, system_manager
from src.services.performance_optimizer import PerformanceAlert
from src.services.incremental_engine import ComputationResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_candle(instrument: str = "EUR_USD", 
                      granularity: str = "H1", 
                      offset_hours: int = 0) -> Candle:
    """Create a demo candle for testing"""
    base_time = datetime.utcnow() - timedelta(hours=offset_hours)
    base_price = 1.0500  # EUR/USD base price
    
    # Add some realistic price variation
    import random
    price_variation = random.uniform(-0.002, 0.002)  # ±0.2%
    
    open_price = base_price + price_variation
    close_price = open_price + random.uniform(-0.001, 0.001)
    high_price = max(open_price, close_price) + random.uniform(0, 0.001)
    low_price = min(open_price, close_price) - random.uniform(0, 0.001)
    
    return Candle(
        instrument_id=instrument,
        granularity=Granularity(granularity),
        ts=base_time,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=random.uniform(1000, 10000),
        bid=close_price - 0.0002,
        ask=close_price + 0.0002,
        atr_14=0.001  # 0.1% ATR
    )


async def demo_alert_handler(alert: PerformanceAlert) -> None:
    """Demo alert handler"""
    print(f"\n🚨 PERFORMANCE ALERT [{alert.severity.value.upper()}]")
    print(f"   Title: {alert.title}")
    print(f"   Description: {alert.description}")
    print(f"   Current Value: {alert.current_value:.2f}")
    print(f"   Threshold: {alert.threshold_value:.2f}")
    
    if alert.recommendations:
        print("   Recommendations:")
        for i, rec in enumerate(alert.recommendations[:3], 1):  # Show top 3
            print(f"     {i}. {rec}")
    print()


async def demo_computation_result_handler(result: ComputationResult) -> None:
    """Demo computation result handler"""
    status = "✅ SUCCESS" if result.success else "❌ FAILED"
    cache_status = "💾 CACHE HIT" if result.cache_hit else "⚡ COMPUTED"
    
    print(f"{status} [{cache_status}] {result.request_id} - {result.processing_time_ms:.1f}ms")
    
    if not result.success and result.error:
        print(f"   Error: {result.error}")


async def submit_demo_candles(batch_size: int = 50, delay_ms: int = 100) -> None:
    """Submit demo candles for processing"""
    print(f"\n📊 Submitting {batch_size} demo candles...")
    
    instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]
    granularities = ["H1", "H4"]
    
    submitted = 0
    accepted = 0
    
    for i in range(batch_size):
        # Create random candle
        instrument = instruments[i % len(instruments)]
        granularity = granularities[i % len(granularities)]
        candle = create_demo_candle(instrument, granularity, offset_hours=i)
        
        # Submit for processing
        success, reason = await system_manager.engine.submit_computation_request(
            candle=candle,
            priority=1,
            horizon_periods=6,
            label_types=["enhanced_triple_barrier", "vol_scaled_return"]
        )
        
        submitted += 1
        if success:
            accepted += 1
        else:
            print(f"   Rejected: {reason}")
        
        # Small delay between submissions
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000)
    
    print(f"   Submitted: {submitted}, Accepted: {accepted}, Rejected: {submitted - accepted}")


async def display_performance_dashboard():
    """Display real-time performance dashboard"""
    print("\n" + "="*80)
    print("📈 REAL-TIME PERFORMANCE DASHBOARD")
    print("="*80)
    
    # Get system status
    system_status = system_manager.get_system_status()
    
    # System overview
    uptime_minutes = system_status.get("uptime_seconds", 0) / 60
    print(f"🖥️  System Status: {'🟢 RUNNING' if system_status['is_running'] else '🔴 STOPPED'}")
    print(f"⏱️  Uptime: {uptime_minutes:.1f} minutes")
    
    if system_status.get("current_health"):
        health = system_status["current_health"]
        health_emoji = "🟢" if health["healthy"] else "🟡" if health["status"] == "degraded" else "🔴"
        print(f"❤️  Health: {health_emoji} {health['status'].upper()} (Score: {health['overall_score']:.1f}/100)")
    
    # Performance metrics
    if system_manager.engine.is_running:
        metrics = system_manager.engine.get_performance_metrics()
        
        print(f"\n⚡ PERFORMANCE METRICS")
        print(f"   Throughput: {metrics['throughput']['requests_per_second']:.1f} req/s")
        print(f"   P99 Latency: {metrics['latency']['p99_ms']:.1f}ms")
        print(f"   Avg Latency: {metrics['latency']['average_ms']:.1f}ms")
        print(f"   Success Rate: {metrics['throughput']['success_rate_pct']:.1f}%")
        print(f"   Cache Hit Rate: {metrics['cache']['overall']['overall_hit_rate_pct']:.1f}%")
        print(f"   Active Workers: {metrics['system']['active_workers']}")
        print(f"   Pending Requests: {metrics['system']['pending_requests']}")
    
    # Alerts summary
    active_alerts = system_manager.optimizer.get_active_alerts()
    critical_alerts = len([a for a in active_alerts if a["severity"] == "critical"])
    warning_alerts = len([a for a in active_alerts if a["severity"] == "warning"])
    
    if active_alerts:
        print(f"\n🚨 ACTIVE ALERTS: {len(active_alerts)} total")
        if critical_alerts > 0:
            print(f"   🔴 Critical: {critical_alerts}")
        if warning_alerts > 0:
            print(f"   🟡 Warning: {warning_alerts}")
    else:
        print(f"\n✅ NO ACTIVE ALERTS")
    
    print("="*80)


async def main():
    """Main demo function"""
    print("🚀 INCREMENTAL COMPUTATION ENGINE DEMO")
    print("="*50)
    
    try:
        # Setup alert handling
        system_manager.add_alert_callback(demo_alert_handler)
        
        # Setup computation result monitoring (for demo only)
        system_manager.engine.add_result_callback(demo_computation_result_handler)
        
        # Start the complete system
        print("⏳ Starting incremental computation system...")
        await start_incremental_computation_system(
            engine_workers=4,  # Smaller number for demo
            enable_performance_monitoring=True,
            enable_firestore_integration=False  # Disable for demo to avoid Firestore dependency
        )
        
        print("✅ System started successfully!")
        
        # Display initial dashboard
        await display_performance_dashboard()
        
        # Wait a moment for system to stabilize
        await asyncio.sleep(2)
        
        # Submit demo candles in batches
        print("\n🧪 TESTING WITH DEMO DATA")
        
        # Batch 1: Normal load
        await submit_demo_candles(batch_size=20, delay_ms=50)
        await asyncio.sleep(1)
        
        # Batch 2: Higher load to trigger performance monitoring
        await submit_demo_candles(batch_size=100, delay_ms=10)
        await asyncio.sleep(2)
        
        # Display updated metrics
        await display_performance_dashboard()
        
        # Let the system run for a bit to see monitoring in action
        print("\n⌛ Running system for 30 seconds to demonstrate monitoring...")
        print("   (You should see periodic performance metrics in the logs)")
        
        for i in range(6):  # 6 * 5 seconds = 30 seconds
            await asyncio.sleep(5)
            
            # Submit occasional candles to keep system active
            if i % 2 == 0:
                candle = create_demo_candle()
                await system_manager.engine.submit_computation_request(candle)
        
        # Final dashboard update
        await display_performance_dashboard()
        
        print("\n✨ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        logger.error("Demo error", exc_info=True)
    finally:
        # Graceful shutdown
        print("\n🛑 Shutting down system...")
        if system_manager.is_running:
            await system_manager.stop_system()
        print("👋 Goodbye!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())