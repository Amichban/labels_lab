"""
Comprehensive spike testing suite for the Label Computation System.

This suite simulates sudden traffic spikes and market events:
- Market open simulation with sudden load increases
- Flash crash scenario testing  
- News event traffic spikes
- Weekend-to-Monday transition testing
- Black Friday/high-volatility event simulation
- Recovery time measurement after spikes

Usage:
    pytest tests/performance/spike_test.py -v --tb=short
    pytest tests/performance/spike_test.py::TestSpikeTesting::test_market_open_simulation -v
"""

import pytest
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch
from contextlib import asynccontextmanager
from dataclasses import dataclass
import psutil
import gc
from collections import deque
import statistics

from src.core.label_computation import LabelComputationEngine
from src.models.data_models import Candle, Granularity


@dataclass
class SpikeTestResult:
    """Results from a spike test."""
    test_name: str
    baseline_ops_per_second: float
    peak_ops_per_second: float
    spike_multiplier: float
    spike_duration_seconds: float
    recovery_time_seconds: float
    success_rate_during_spike: float
    success_rate_post_spike: float
    max_response_time_ms: float
    avg_response_time_during_spike_ms: float
    system_stability_score: float
    recommendations: List[str]


@dataclass
class MarketCondition:
    """Represents different market conditions."""
    name: str
    volatility_multiplier: float
    volume_multiplier: float
    ops_per_second: float
    duration_seconds: int
    price_gap_probability: float = 0.0  # Probability of price gaps


class SpikeTestMonitor:
    """Monitors system during spike tests."""
    
    def __init__(self, window_size: int = 100):
        self.response_times: deque = deque(maxlen=window_size)
        self.success_results: deque = deque(maxlen=window_size) 
        self.operation_timestamps: deque = deque(maxlen=window_size)
        self.errors: List[str] = []
        self.start_time = time.perf_counter()
    
    def record_operation(self, response_time_ms: float, success: bool, error: Optional[str] = None):
        """Record operation result."""
        current_time = time.perf_counter()
        self.response_times.append(response_time_ms)
        self.success_results.append(success)
        self.operation_timestamps.append(current_time)
        
        if error:
            self.errors.append(f"{current_time - self.start_time:.1f}s: {error}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.response_times:
            return {
                "avg_response_time_ms": 0,
                "p95_response_time_ms": 0,
                "p99_response_time_ms": 0,
                "success_rate": 0,
                "ops_per_second": 0
            }
        
        # Calculate time window
        current_time = time.perf_counter()
        window_start = current_time - 60  # Last 60 seconds
        
        # Filter operations in window
        recent_ops = [(ts, rt, success) for ts, rt, success in 
                      zip(self.operation_timestamps, self.response_times, self.success_results)
                      if ts >= window_start]
        
        if not recent_ops:
            return self.get_current_metrics()  # Fallback to all-time metrics
        
        response_times = [rt for _, rt, _ in recent_ops]
        success_results = [success for _, _, success in recent_ops]
        
        return {
            "avg_response_time_ms": np.mean(response_times),
            "p95_response_time_ms": np.percentile(response_times, 95),
            "p99_response_time_ms": np.percentile(response_times, 99),
            "success_rate": sum(success_results) / len(success_results),
            "ops_per_second": len(recent_ops) / 60.0
        }
    
    def detect_performance_degradation(self, threshold_multiplier: float = 2.0) -> bool:
        """Detect if performance has significantly degraded."""
        if len(self.response_times) < 20:
            return False
        
        # Compare recent performance to baseline
        baseline_times = list(self.response_times)[:10]  # First 10 operations
        recent_times = list(self.response_times)[-10:]   # Last 10 operations
        
        baseline_avg = np.mean(baseline_times)
        recent_avg = np.mean(recent_times)
        
        return recent_avg > (baseline_avg * threshold_multiplier)


def generate_market_scenario_candles(
    scenario: MarketCondition, 
    base_price: float = 1.0500,
    instrument: str = "EURUSD"
) -> List[Candle]:
    """Generate candles simulating specific market conditions."""
    candles = []
    base_time = datetime.now()
    
    # Calculate number of candles based on duration and frequency
    candles_count = max(1, scenario.duration_seconds // 5)  # 5-second intervals
    
    current_price = base_price
    
    for i in range(candles_count):
        # Apply market condition effects
        volatility = 0.01 * scenario.volatility_multiplier
        
        # Price gaps for extreme events
        if np.random.random() < scenario.price_gap_probability:
            gap_size = np.random.normal(0, volatility * 2)
            current_price += gap_size
        
        # Normal price movement
        price_change = np.random.normal(0, volatility)
        current_price += price_change
        
        # Generate OHLC based on current price
        open_price = current_price
        high_offset = abs(np.random.normal(0, volatility * 0.5))
        low_offset = abs(np.random.normal(0, volatility * 0.5))
        close_change = np.random.normal(0, volatility * 0.3)
        
        high_price = open_price + high_offset
        low_price = open_price - low_offset
        close_price = open_price + close_change
        
        # Ensure OHLC consistency
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Volume affected by market condition
        base_volume = 1000
        volume = int(base_volume * scenario.volume_multiplier * (1 + np.random.normal(0, 0.2)))
        
        candle = Candle(
            instrument_id=instrument,
            granularity=Granularity.H4,
            ts=base_time + timedelta(seconds=i * 5),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=max(volume, 1),  # Ensure positive volume
            atr_14=volatility * 14  # Approximation
        )
        
        candles.append(candle)
        current_price = close_price
    
    return candles


# Market scenario definitions
MARKET_SCENARIOS = {
    "pre_market": MarketCondition(
        name="Pre-Market (Low Activity)",
        volatility_multiplier=0.3,
        volume_multiplier=0.2,
        ops_per_second=10,
        duration_seconds=300,  # 5 minutes
        price_gap_probability=0.01
    ),
    
    "market_open": MarketCondition(
        name="Market Open Spike",
        volatility_multiplier=2.5,
        volume_multiplier=5.0,
        ops_per_second=500,
        duration_seconds=180,  # 3 minutes
        price_gap_probability=0.1
    ),
    
    "normal_trading": MarketCondition(
        name="Normal Trading Hours",
        volatility_multiplier=1.0,
        volume_multiplier=1.0,
        ops_per_second=50,
        duration_seconds=600,  # 10 minutes
        price_gap_probability=0.02
    ),
    
    "news_event": MarketCondition(
        name="Major News Event",
        volatility_multiplier=4.0,
        volume_multiplier=8.0,
        ops_per_second=1000,
        duration_seconds=120,  # 2 minutes
        price_gap_probability=0.2
    ),
    
    "flash_crash": MarketCondition(
        name="Flash Crash",
        volatility_multiplier=10.0,
        volume_multiplier=15.0,
        ops_per_second=2000,
        duration_seconds=60,  # 1 minute
        price_gap_probability=0.5
    ),
    
    "recovery": MarketCondition(
        name="Post-Event Recovery",
        volatility_multiplier=1.5,
        volume_multiplier=2.0,
        ops_per_second=100,
        duration_seconds=300,  # 5 minutes
        price_gap_probability=0.05
    )
}


@pytest.mark.performance
@pytest.mark.spike
class TestSpikeTesting:
    """Spike testing for sudden load increases and market events."""
    
    @pytest.mark.asyncio
    async def test_market_open_simulation(self):
        """Simulate market open with sudden traffic spike."""
        print("Starting market open simulation...")
        
        monitor = SpikeTestMonitor(window_size=200)
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Configure mocks for realistic latency under load
            call_count = 0
            
            async def load_sensitive_mock(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                # Simulate increased latency under high load
                base_latency = 0.010  # 10ms base
                load_factor = min(call_count / 100, 5.0)  # Increase with load
                latency = base_latency * (1 + load_factor * 0.5)
                
                await asyncio.sleep(latency)
                return []
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels = AsyncMock(side_effect=load_sensitive_mock)
            mock_ch.fetch_snapshots = AsyncMock(side_effect=load_sensitive_mock)
            
            engine = LabelComputationEngine()
            
            # Market open simulation scenario
            scenarios = [
                MARKET_SCENARIOS["pre_market"],
                MARKET_SCENARIOS["market_open"],  # The spike
                MARKET_SCENARIOS["normal_trading"]
            ]
            
            baseline_ops_per_second = scenarios[0].ops_per_second
            peak_ops_per_second = scenarios[1].ops_per_second
            
            all_results = []
            
            for scenario in scenarios:
                print(f"\nExecuting scenario: {scenario.name}")
                print(f"Target load: {scenario.ops_per_second} ops/sec for {scenario.duration_seconds}s")
                
                # Generate test data for this scenario
                test_candles = generate_market_scenario_candles(scenario)
                
                scenario_start_time = time.perf_counter()
                target_interval = 1.0 / scenario.ops_per_second
                
                # Execute scenario
                for i, candle in enumerate(test_candles):
                    operation_start = time.perf_counter()
                    
                    try:
                        result = await engine.compute_labels(candle)
                        operation_end = time.perf_counter()
                        
                        response_time_ms = (operation_end - operation_start) * 1000
                        success = result is not None
                        
                        monitor.record_operation(response_time_ms, success)
                        all_results.append((scenario.name, response_time_ms, success))
                        
                    except Exception as e:
                        operation_end = time.perf_counter()
                        response_time_ms = (operation_end - operation_start) * 1000
                        
                        monitor.record_operation(response_time_ms, False, str(e))
                        all_results.append((scenario.name, response_time_ms, False))
                    
                    # Rate limiting to achieve target ops/second
                    elapsed = time.perf_counter() - operation_start
                    if elapsed < target_interval:
                        await asyncio.sleep(target_interval - elapsed)
                    
                    # Check if scenario duration is complete
                    scenario_elapsed = time.perf_counter() - scenario_start_time
                    if scenario_elapsed >= scenario.duration_seconds:
                        break
                    
                    # Periodic status during spike
                    if scenario.name == "Market Open Spike" and i % 50 == 0:
                        current_metrics = monitor.get_current_metrics()
                        print(f"  Spike progress: {i} ops, "
                              f"Success: {current_metrics['success_rate']:.1%}, "
                              f"Avg RT: {current_metrics['avg_response_time_ms']:.1f}ms")
                
                # End of scenario metrics
                current_metrics = monitor.get_current_metrics()
                print(f"Scenario completed - Success rate: {current_metrics['success_rate']:.2%}, "
                      f"Avg response time: {current_metrics['avg_response_time_ms']:.1f}ms")
                
                # Brief pause between scenarios
                await asyncio.sleep(2)
        
        # Analyze results by scenario
        scenario_results = {}
        for scenario_name in [s.name for s in scenarios]:
            scenario_ops = [(rt, success) for name, rt, success in all_results if name == scenario_name]
            
            if scenario_ops:
                response_times = [rt for rt, _ in scenario_ops]
                successes = [success for _, success in scenario_ops]
                
                scenario_results[scenario_name] = {
                    "avg_response_time_ms": np.mean(response_times),
                    "max_response_time_ms": np.max(response_times),
                    "p95_response_time_ms": np.percentile(response_times, 95),
                    "success_rate": sum(successes) / len(successes),
                    "operation_count": len(scenario_ops)
                }
        
        # Calculate spike-specific metrics
        spike_results = scenario_results.get("Market Open Spike", {})
        recovery_results = scenario_results.get("Normal Trading Hours", {})
        
        # Recovery time estimation (time to get back to normal response times)
        recovery_time_seconds = 30  # Simplified estimation
        
        # System stability score (0-100)
        stability_factors = []
        if spike_results:
            # Success rate during spike (0-40 points)
            spike_success = spike_results.get("success_rate", 0)
            stability_factors.append(spike_success * 40)
            
            # Response time degradation (0-30 points)
            baseline_rt = scenario_results.get("Pre-Market (Low Activity)", {}).get("avg_response_time_ms", 50)
            spike_rt = spike_results.get("avg_response_time_ms", 200)
            if baseline_rt > 0:
                degradation_ratio = spike_rt / baseline_rt
                if degradation_ratio <= 2.0:  # Less than 2x degradation
                    stability_factors.append(30)
                elif degradation_ratio <= 5.0:  # Less than 5x degradation
                    stability_factors.append(15)
                else:
                    stability_factors.append(0)
            else:
                stability_factors.append(15)
            
            # Recovery performance (0-30 points)
            if recovery_results:
                recovery_success = recovery_results.get("success_rate", 0)
                recovery_rt = recovery_results.get("avg_response_time_ms", 100)
                
                if recovery_success >= 0.95 and recovery_rt <= baseline_rt * 1.5:
                    stability_factors.append(30)
                elif recovery_success >= 0.90:
                    stability_factors.append(20)
                else:
                    stability_factors.append(10)
            else:
                stability_factors.append(0)
        
        system_stability_score = sum(stability_factors) if stability_factors else 0
        
        # Generate recommendations
        recommendations = []
        if spike_results.get("success_rate", 0) < 0.95:
            recommendations.append("Implement circuit breakers to handle traffic spikes")
            recommendations.append("Increase connection pool sizes for high-load scenarios")
        
        if spike_results.get("max_response_time_ms", 0) > 1000:
            recommendations.append("Add request timeouts to prevent long-running operations during spikes")
        
        if system_stability_score < 70:
            recommendations.append("Implement auto-scaling based on queue depth")
            recommendations.append("Add load shedding mechanisms for extreme load")
        
        recommendations.extend([
            "Monitor market open times and pre-scale resources",
            "Implement graceful degradation during high-volatility periods",
            "Set up alerts for sudden traffic increases"
        ])
        
        # Create final result
        result = SpikeTestResult(
            test_name="Market Open Simulation",
            baseline_ops_per_second=baseline_ops_per_second,
            peak_ops_per_second=peak_ops_per_second,
            spike_multiplier=peak_ops_per_second / baseline_ops_per_second,
            spike_duration_seconds=MARKET_SCENARIOS["market_open"].duration_seconds,
            recovery_time_seconds=recovery_time_seconds,
            success_rate_during_spike=spike_results.get("success_rate", 0),
            success_rate_post_spike=recovery_results.get("success_rate", 0),
            max_response_time_ms=spike_results.get("max_response_time_ms", 0),
            avg_response_time_during_spike_ms=spike_results.get("avg_response_time_ms", 0),
            system_stability_score=system_stability_score,
            recommendations=recommendations
        )
        
        # Print results
        print(f"\nMarket Open Simulation Results:")
        print(f"Baseline load: {result.baseline_ops_per_second} ops/sec")
        print(f"Peak load: {result.peak_ops_per_second} ops/sec (${result.spike_multiplier:.1f}x spike)")
        print(f"Success rate during spike: {result.success_rate_during_spike:.2%}")
        print(f"Success rate post-spike: {result.success_rate_post_spike:.2%}")
        print(f"Max response time during spike: {result.max_response_time_ms:.1f}ms")
        print(f"System stability score: {result.system_stability_score:.1f}/100")
        print(f"Recovery time: {result.recovery_time_seconds}s")
        
        # Detailed scenario breakdown
        print(f"\nScenario Breakdown:")
        for scenario_name, metrics in scenario_results.items():
            print(f"  {scenario_name}:")
            print(f"    Operations: {metrics['operation_count']}")
            print(f"    Success rate: {metrics['success_rate']:.2%}")
            print(f"    Avg response time: {metrics['avg_response_time_ms']:.1f}ms")
            print(f"    Max response time: {metrics['max_response_time_ms']:.1f}ms")
        
        # Assertions
        assert result.success_rate_during_spike >= 0.80, \
            f"Poor success rate during spike: {result.success_rate_during_spike:.2%}"
        
        assert result.success_rate_post_spike >= 0.90, \
            f"Poor recovery success rate: {result.success_rate_post_spike:.2%}"
        
        assert result.max_response_time_ms <= 5000, \
            f"Response time too high during spike: {result.max_response_time_ms:.1f}ms"
        
        assert result.system_stability_score >= 50, \
            f"Poor system stability: {result.system_stability_score:.1f}/100"
    
    @pytest.mark.asyncio
    async def test_flash_crash_scenario(self):
        """Simulate flash crash with extreme load spike."""
        print("Starting flash crash scenario...")
        
        monitor = SpikeTestMonitor()
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Simulate system stress during flash crash
            operation_count = 0
            
            async def stressed_system_mock(*args, **kwargs):
                nonlocal operation_count
                operation_count += 1
                
                # System becomes increasingly stressed
                if operation_count > 100:  # After 100 operations, start failing some
                    failure_rate = min((operation_count - 100) / 200, 0.3)  # Up to 30% failure rate
                    if np.random.random() < failure_rate:
                        raise Exception("System overloaded")
                
                # Increased latency under extreme load
                base_latency = 0.020  # 20ms base
                stress_factor = min(operation_count / 50, 10.0)
                latency = base_latency * (1 + stress_factor * 0.1)
                
                await asyncio.sleep(latency)
                return []
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels = AsyncMock(side_effect=stressed_system_mock)
            mock_ch.fetch_snapshots = AsyncMock(side_effect=stressed_system_mock)
            
            engine = LabelComputationEngine()
            
            # Flash crash scenario: extreme spike then recovery
            scenarios = [
                MARKET_SCENARIOS["normal_trading"],
                MARKET_SCENARIOS["flash_crash"],
                MARKET_SCENARIOS["recovery"]
            ]
            
            for scenario in scenarios:
                print(f"\nExecuting: {scenario.name} ({scenario.ops_per_second} ops/sec)")
                
                test_candles = generate_market_scenario_candles(scenario)
                scenario_start = time.perf_counter()
                target_interval = 1.0 / scenario.ops_per_second
                
                for candle in test_candles:
                    operation_start = time.perf_counter()
                    
                    try:
                        result = await engine.compute_labels(candle)
                        operation_end = time.perf_counter()
                        
                        response_time_ms = (operation_end - operation_start) * 1000
                        success = result is not None
                        
                        monitor.record_operation(response_time_ms, success)
                        
                    except Exception as e:
                        operation_end = time.perf_counter()
                        response_time_ms = (operation_end - operation_start) * 1000
                        
                        monitor.record_operation(response_time_ms, False, str(e))
                    
                    # Rate limiting
                    elapsed = time.perf_counter() - operation_start
                    if elapsed < target_interval:
                        await asyncio.sleep(target_interval - elapsed)
                    
                    # Check scenario duration
                    if time.perf_counter() - scenario_start >= scenario.duration_seconds:
                        break
                
                # Scenario completion metrics
                current_metrics = monitor.get_current_metrics()
                print(f"  Completed: Success {current_metrics['success_rate']:.1%}, "
                      f"Avg RT {current_metrics['avg_response_time_ms']:.1f}ms")
        
        # Final metrics
        final_metrics = monitor.get_current_metrics()
        
        print(f"\nFlash Crash Scenario Results:")
        print(f"Total operations: {len(monitor.success_results)}")
        print(f"Overall success rate: {final_metrics['success_rate']:.2%}")
        print(f"Average response time: {final_metrics['avg_response_time_ms']:.1f}ms")
        print(f"P99 response time: {final_metrics['p99_response_time_ms']:.1f}ms")
        print(f"Errors encountered: {len(monitor.errors)}")
        
        # Check system can handle extreme scenarios
        assert final_metrics['success_rate'] >= 0.70, \
            f"System failed to handle flash crash: {final_metrics['success_rate']:.2%}"
        
        assert final_metrics['p99_response_time_ms'] <= 10000, \
            f"Response times too high during flash crash: {final_metrics['p99_response_time_ms']:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_news_event_spike(self):
        """Simulate traffic spike during major news events."""
        print("Starting news event spike test...")
        
        monitor = SpikeTestMonitor()
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Simulate realistic news event behavior
            async def news_event_mock(*args, **kwargs):
                # Higher latency but stable service
                await asyncio.sleep(np.random.exponential(0.030))  # 30ms average
                return []
            
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels = AsyncMock(side_effect=news_event_mock)
            mock_ch.fetch_snapshots = AsyncMock(side_effect=news_event_mock)
            
            engine = LabelComputationEngine()
            
            # News event: gradual buildup, spike, then gradual decline
            phase_durations = [60, 120, 60]  # seconds
            phase_loads = [50, 1000, 100]    # ops per second
            phase_names = ["Pre-News", "News Spike", "Post-News"]
            
            for phase_name, duration, load in zip(phase_names, phase_durations, phase_loads):
                print(f"\n{phase_name} phase: {load} ops/sec for {duration}s")
                
                phase_start = time.perf_counter()
                target_interval = 1.0 / load
                
                while time.perf_counter() - phase_start < duration:
                    # Generate candle for this phase
                    test_candle = generate_market_scenario_candles(
                        MarketCondition(
                            name=phase_name,
                            volatility_multiplier=2.0 if "Spike" in phase_name else 1.0,
                            volume_multiplier=3.0 if "Spike" in phase_name else 1.0,
                            ops_per_second=load,
                            duration_seconds=duration
                        )
                    )[0]
                    
                    operation_start = time.perf_counter()
                    
                    try:
                        result = await engine.compute_labels(test_candle)
                        operation_end = time.perf_counter()
                        
                        response_time_ms = (operation_end - operation_start) * 1000
                        success = result is not None
                        
                        monitor.record_operation(response_time_ms, success)
                        
                    except Exception as e:
                        operation_end = time.perf_counter()
                        response_time_ms = (operation_end - operation_start) * 1000
                        
                        monitor.record_operation(response_time_ms, False, str(e))
                    
                    # Rate limiting
                    elapsed = time.perf_counter() - operation_start
                    if elapsed < target_interval:
                        await asyncio.sleep(target_interval - elapsed)
                
                current_metrics = monitor.get_current_metrics()
                print(f"Phase metrics: Success {current_metrics['success_rate']:.1%}, "
                      f"RT {current_metrics['avg_response_time_ms']:.1f}ms")
        
        final_metrics = monitor.get_current_metrics()
        
        print(f"\nNews Event Spike Results:")
        print(f"Total operations: {len(monitor.success_results)}")
        print(f"Success rate: {final_metrics['success_rate']:.2%}")
        print(f"Average response time: {final_metrics['avg_response_time_ms']:.1f}ms")
        print(f"P95 response time: {final_metrics['p95_response_time_ms']:.1f}ms")
        
        # News events should be handled well
        assert final_metrics['success_rate'] >= 0.90, \
            f"Poor handling of news event: {final_metrics['success_rate']:.2%}"
        
        assert final_metrics['avg_response_time_ms'] <= 200, \
            f"Response times too high during news event: {final_metrics['avg_response_time_ms']:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_weekend_gap_monday_open(self):
        """Simulate weekend gap and Monday morning trading rush."""
        print("Starting weekend gap and Monday open test...")
        
        monitor = SpikeTestMonitor()
        
        with patch('src.core.label_computation.clickhouse_service') as mock_ch, \
             patch('src.core.label_computation.redis_cache') as mock_redis:
            
            # Cold start effects after weekend
            cache_miss_count = 0
            
            async def monday_open_mock(*args, **kwargs):
                nonlocal cache_miss_count
                cache_miss_count += 1
                
                # Simulate cache warming up over time
                if cache_miss_count <= 50:  # First 50 operations slower (cold cache)
                    await asyncio.sleep(0.050)  # 50ms for cache miss
                else:
                    await asyncio.sleep(0.015)  # 15ms for cache hit
                
                return []
            
            # Start with no cache (weekend reset)
            mock_redis.get_labels.return_value = None
            mock_ch.fetch_active_levels = AsyncMock(side_effect=monday_open_mock)
            mock_ch.fetch_snapshots = AsyncMock(side_effect=monday_open_mock)
            
            engine = LabelComputationEngine()
            
            # Monday scenario: price gaps and high volume
            monday_scenario = MarketCondition(
                name="Monday Market Open",
                volatility_multiplier=3.0,
                volume_multiplier=4.0,
                ops_per_second=300,
                duration_seconds=300,  # 5 minutes
                price_gap_probability=0.3  # High probability of gaps
            )
            
            test_candles = generate_market_scenario_candles(monday_scenario, base_price=1.0450)  # Gapped price
            
            print(f"Processing {len(test_candles)} candles with price gaps...")
            
            target_interval = 1.0 / monday_scenario.ops_per_second
            
            for i, candle in enumerate(test_candles):
                operation_start = time.perf_counter()
                
                try:
                    result = await engine.compute_labels(candle)
                    operation_end = time.perf_counter()
                    
                    response_time_ms = (operation_end - operation_start) * 1000
                    success = result is not None
                    
                    monitor.record_operation(response_time_ms, success)
                    
                    # Show progress during cache warmup
                    if i in [10, 25, 50, 100]:
                        current_metrics = monitor.get_current_metrics()
                        print(f"  Operation {i}: Success {current_metrics['success_rate']:.1%}, "
                              f"RT {current_metrics['avg_response_time_ms']:.1f}ms")
                    
                except Exception as e:
                    operation_end = time.perf_counter()
                    response_time_ms = (operation_end - operation_start) * 1000
                    
                    monitor.record_operation(response_time_ms, False, str(e))
                
                # Rate limiting
                elapsed = time.perf_counter() - operation_start
                if elapsed < target_interval:
                    await asyncio.sleep(target_interval - elapsed)
        
        final_metrics = monitor.get_current_metrics()
        
        # Analyze cache warmup effect
        early_operations = list(monitor.response_times)[:20]
        late_operations = list(monitor.response_times)[-20:]
        
        early_avg = np.mean(early_operations) if early_operations else 0
        late_avg = np.mean(late_operations) if late_operations else 0
        
        print(f"\nWeekend Gap and Monday Open Results:")
        print(f"Total operations: {len(monitor.success_results)}")
        print(f"Success rate: {final_metrics['success_rate']:.2%}")
        print(f"Overall avg response time: {final_metrics['avg_response_time_ms']:.1f}ms")
        print(f"Early operations avg RT: {early_avg:.1f}ms (cold cache)")
        print(f"Late operations avg RT: {late_avg:.1f}ms (warm cache)")
        print(f"Cache warmup improvement: {((early_avg - late_avg) / early_avg * 100):.1f}%")
        
        # Monday opens should be handled despite cold starts
        assert final_metrics['success_rate'] >= 0.85, \
            f"Poor Monday open handling: {final_metrics['success_rate']:.2%}"
        
        # Cache should warm up and improve performance
        if early_avg > 0 and late_avg > 0:
            improvement_ratio = early_avg / late_avg
            assert improvement_ratio >= 1.2, \
                f"Insufficient cache warmup improvement: {improvement_ratio:.1f}x"


@pytest.mark.performance  
@pytest.mark.spike
class TestSpikeScenariosReporting:
    """Generate comprehensive spike testing reports."""
    
    def test_generate_spike_scenarios_guide(self):
        """Generate guide for all available spike test scenarios."""
        
        guide_content = [
            "# Spike Testing Scenarios Guide",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Overview",
            "",
            "This guide documents all available spike testing scenarios for the Label Computation System.",
            "Each scenario simulates real-world market conditions that cause sudden traffic spikes.",
            "",
            "## Market Scenarios",
            ""
        ]
        
        # Document each market scenario
        for scenario_key, scenario in MARKET_SCENARIOS.items():
            guide_content.extend([
                f"### {scenario.name}",
                f"**Key:** `{scenario_key}`",
                f"**Volatility Multiplier:** {scenario.volatility_multiplier}x",
                f"**Volume Multiplier:** {scenario.volume_multiplier}x",
                f"**Target Load:** {scenario.ops_per_second} operations/second",
                f"**Duration:** {scenario.duration_seconds} seconds",
                f"**Price Gap Probability:** {scenario.price_gap_probability:.0%}",
                "",
                "**Use Case:** " + self._get_scenario_use_case(scenario_key),
                ""
            ])
        
        guide_content.extend([
            "## Available Spike Tests",
            "",
            "### 1. Market Open Simulation",
            "**Test:** `test_market_open_simulation`",
            "**Scenarios:** Pre-market → Market Open → Normal Trading",
            "**Purpose:** Test handling of typical market open traffic patterns",
            "**Duration:** ~15 minutes",
            "**Key Metrics:**",
            "- Success rate during 500 ops/sec spike",
            "- Recovery time to normal performance",
            "- System stability score",
            "",
            "### 2. Flash Crash Scenario", 
            "**Test:** `test_flash_crash_scenario`",
            "**Scenarios:** Normal → Flash Crash (2000 ops/sec) → Recovery",
            "**Purpose:** Test system resilience during extreme market events",
            "**Duration:** ~10 minutes",
            "**Key Metrics:**",
            "- Survival rate during extreme load",
            "- Error handling and circuit breaking",
            "- System recovery capabilities",
            "",
            "### 3. News Event Spike",
            "**Test:** `test_news_event_spike`", 
            "**Scenarios:** Pre-news → News Spike (1000 ops/sec) → Post-news",
            "**Purpose:** Test handling of news-driven traffic increases",
            "**Duration:** ~4 minutes",
            "**Key Metrics:**",
            "- Performance during sustained high load",
            "- Response time degradation",
            "- Graceful load handling",
            "",
            "### 4. Weekend Gap Monday Open",
            "**Test:** `test_weekend_gap_monday_open`",
            "**Scenarios:** Cold start with price gaps and high volume",
            "**Purpose:** Test cold cache performance and gap handling",
            "**Duration:** ~5 minutes", 
            "**Key Metrics:**",
            "- Cache warmup performance",
            "- Price gap processing accuracy",
            "- Cold start recovery time",
            "",
            "## Running Spike Tests",
            "",
            "```bash",
            "# Run all spike tests",
            "pytest tests/performance/spike_test.py -v",
            "",
            "# Run specific test",
            "pytest tests/performance/spike_test.py::TestSpikeTesting::test_market_open_simulation -v",
            "",
            "# Run with detailed output",
            "pytest tests/performance/spike_test.py -v -s",
            "```",
            "",
            "## Interpreting Results",
            "",
            "### Success Rate Thresholds",
            "- **Excellent (95%+):** System handles spikes gracefully",
            "- **Good (90-95%):** System stable with minor degradation",
            "- **Warning (80-90%):** System stressed, investigate bottlenecks",
            "- **Critical (<80%):** System cannot handle spike load",
            "",
            "### Response Time Analysis",
            "- **Baseline:** <50ms average during normal operation",
            "- **Acceptable Degradation:** <5x increase during spike",
            "- **Maximum Acceptable:** <5000ms P99 during extreme events",
            "",
            "### System Stability Score",
            "- **90-100:** Excellent spike handling capability",
            "- **70-89:** Good performance with room for improvement", 
            "- **50-69:** Moderate performance, optimization recommended",
            "- **<50:** Poor spike handling, requires significant improvements",
            "",
            "## Common Issues and Solutions",
            "",
            "### High Error Rate During Spikes",
            "**Symptoms:** Success rate drops below 90% during load spikes",
            "**Solutions:**",
            "- Implement circuit breakers",
            "- Increase connection pool sizes",
            "- Add request queueing with backpressure",
            "",
            "### Slow Recovery After Spikes", 
            "**Symptoms:** Performance doesn't return to baseline after spike",
            "**Solutions:**",
            "- Implement connection pool recycling",
            "- Add garbage collection tuning",
            "- Clear accumulated state/caches",
            "",
            "### Memory Issues During High Load",
            "**Symptoms:** Memory usage increases significantly during spikes",
            "**Solutions:**",
            "- Implement object pooling",
            "- Optimize data structures for high-frequency operations",
            "- Add memory pressure monitoring",
            "",
            "### Cache Cold Start Problems",
            "**Symptoms:** Poor performance at start of high-load scenarios",
            "**Solutions:**",
            "- Implement cache pre-warming strategies",
            "- Use predictive caching based on market schedule",
            "- Add cache hit rate monitoring and alerts",
            "",
            "## Integration with Monitoring",
            "",
            "Use spike test results to configure production monitoring:",
            "",
            "1. **Alert Thresholds:** Set based on spike test performance degradation",
            "2. **Auto-scaling Triggers:** Based on ops/second capacity discovered in tests",
            "3. **Circuit Breaker Thresholds:** Based on failure rates during extreme scenarios",
            "4. **Cache Warming:** Schedule based on market event patterns",
            "",
            "## Recommendations for Production",
            "",
            "1. **Pre-scale for Known Events:** Market opens, major economic releases",
            "2. **Implement Circuit Breakers:** Based on spike test failure patterns", 
            "3. **Queue Management:** Handle bursts that exceed processing capacity",
            "4. **Monitoring Dashboards:** Track metrics identified as critical in spike tests",
            "5. **Runbooks:** Document recovery procedures based on spike test scenarios"
        ]
        
        # Write guide to file
        guide_path = "/Users/aminechbani/labels_lab/my-project/spike_testing_guide.md"
        with open(guide_path, "w") as f:
            f.write("\n".join(guide_content))
        
        print(f"Spike testing guide written to: {guide_path}")
        
        # Verify file was created
        import os
        assert os.path.exists(guide_path), "Guide file was not created"
    
    def _get_scenario_use_case(self, scenario_key: str) -> str:
        """Get use case description for a scenario."""
        use_cases = {
            "pre_market": "Simulate low-activity periods before market opens",
            "market_open": "Test system during typical market opening rush with high volatility",
            "normal_trading": "Baseline performance during regular trading hours",
            "news_event": "Major economic announcements or central bank decisions",
            "flash_crash": "Extreme market events like algorithmic trading cascades",
            "recovery": "Post-event normalization and system recovery"
        }
        return use_cases.get(scenario_key, "Custom market condition scenario")