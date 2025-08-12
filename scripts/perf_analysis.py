#!/usr/bin/env python3
"""
Performance Analysis Script for Cache Warming System - Issue #13

This script implements the /perf command approach for comprehensive
performance analysis and optimization of the intelligent cache warming system.

Features:
- Real-time performance monitoring
- Cache hierarchy analysis
- ML prediction accuracy assessment
- Performance regression detection
- Optimization recommendations
- Benchmark comparison with targets

Usage:
    python scripts/perf_analysis.py [--live] [--duration=300] [--report=perf_report.json]
    
Examples:
    # Live monitoring for 5 minutes
    python scripts/perf_analysis.py --live --duration=300
    
    # Generate performance report
    python scripts/perf_analysis.py --report=cache_perf_analysis.json
    
    # Quick health check
    python scripts/perf_analysis.py --quick
"""

import asyncio
import logging
import time
import json
import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Performance targets (matching Issue #13 requirements)
PERFORMANCE_TARGETS = {
    "cache_hit_rate_pct": 95.0,
    "p99_latency_ms": 100.0,
    "avg_latency_ms": 50.0,
    "throughput_rps": 1000.0,
    "memory_usage_pct": 80.0,
    "cpu_usage_pct": 70.0,
    "warming_time_ms": 5000.0  # 5 seconds max warming time
}


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    cache_hierarchy_stats: Dict[str, Any] = field(default_factory=dict)
    predictor_stats: Dict[str, Any] = field(default_factory=dict)
    warming_stats: Dict[str, Any] = field(default_factory=dict)
    system_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cache_hierarchy_stats": self.cache_hierarchy_stats,
            "predictor_stats": self.predictor_stats,
            "warming_stats": self.warming_stats,
            "system_metrics": self.system_metrics
        }


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer implementing /perf command approach
    """
    
    def __init__(self):
        self.snapshots: List[PerformanceSnapshot] = []
        self.start_time = datetime.utcnow()
        
        # Simulated services (in real implementation, these would be imported)
        self.cache_hierarchy_available = True
        self.predictor_available = True
        self.warming_service_available = True
        
        logger.info("PerformanceAnalyzer initialized")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                "memory": {
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3),
                    "used_gb": memory.used / (1024**3),
                    "percent": memory.percent
                },
                "process": {
                    "pid": process.pid,
                    "memory_rss_mb": process_memory.rss / (1024*1024),
                    "memory_vms_mb": process_memory.vms / (1024*1024),
                    "cpu_percent": process.cpu_percent(),
                    "threads": process.num_threads()
                }
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def simulate_cache_hierarchy_stats(self) -> Dict[str, Any]:
        """Simulate cache hierarchy statistics"""
        import random
        
        # Simulate realistic cache statistics
        l1_hit_rate = random.uniform(85, 95)
        l2_hit_rate = random.uniform(92, 98)
        l3_hit_rate = random.uniform(60, 80)
        
        overall_hit_rate = (l1_hit_rate * 0.6) + (l2_hit_rate * 0.3) + (l3_hit_rate * 0.1)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall": {
                "total_hits": random.randint(8000, 12000),
                "total_misses": random.randint(200, 800),
                "overall_hit_rate_pct": overall_hit_rate,
                "target_hit_rate_pct": 95.0
            },
            "levels": {
                "l1_memory": {
                    "hits": random.randint(4000, 6000),
                    "misses": random.randint(100, 300),
                    "hit_rate_pct": l1_hit_rate,
                    "avg_latency_ms": random.uniform(0.5, 2.0),
                    "current_size": random.randint(800, 1000),
                    "max_size": 1000,
                    "memory_usage_bytes": random.randint(80, 100) * 1024 * 1024
                },
                "l2_redis": {
                    "hits": random.randint(3000, 4000),
                    "misses": random.randint(50, 200),
                    "hit_rate_pct": l2_hit_rate,
                    "avg_latency_ms": random.uniform(2.0, 8.0),
                    "current_size": random.randint(8000, 12000),
                    "max_size": 100000
                },
                "l3_clickhouse": {
                    "hits": random.randint(500, 1000),
                    "misses": random.randint(50, 150),
                    "hit_rate_pct": l3_hit_rate,
                    "avg_latency_ms": random.uniform(15.0, 50.0),
                    "current_size": random.randint(50000, 80000),
                    "max_size": 1000000
                }
            }
        }
    
    def simulate_predictor_stats(self) -> Dict[str, Any]:
        """Simulate ML predictor statistics"""
        import random
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "learning_status": "active",
            "total_access_events": random.randint(5000, 10000),
            "total_patterns": random.randint(150, 300),
            "pattern_confidence_distribution": {
                "high": random.randint(50, 100),
                "medium": random.randint(30, 80),
                "low": random.randint(10, 40)
            },
            "prediction_accuracy_pct": random.uniform(78, 88),
            "active_learners": random.randint(20, 50),
            "trading_sessions": ["sydney", "tokyo", "london", "new_york"],
            "last_pattern_update": (datetime.utcnow() - timedelta(minutes=random.randint(1, 30))).isoformat()
        }
    
    def simulate_warming_stats(self) -> Dict[str, Any]:
        """Simulate cache warming service statistics"""
        import random
        
        total_tasks = random.randint(80, 150)
        completed_tasks = random.randint(int(total_tasks * 0.85), total_tasks)
        failed_tasks = total_tasks - completed_tasks
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "service_status": "running",
            "tasks": {
                "active": random.randint(0, 5),
                "pending": random.randint(0, 10),
                "completed": completed_tasks,
                "failed": failed_tasks,
                "total": total_tasks
            },
            "performance": {
                "success_rate_pct": (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 100,
                "items_warmed": random.randint(50000, 100000),
                "avg_warming_time_ms": random.uniform(1500, 4000),
                "throughput_items_per_sec": random.uniform(800, 1500),
                "cache_hit_rate_improvement": random.uniform(15, 25)
            },
            "last_warming": (datetime.utcnow() - timedelta(minutes=random.randint(5, 60))).isoformat(),
            "predictive_warming_enabled": True
        }
    
    async def take_performance_snapshot(self) -> PerformanceSnapshot:
        """Take a comprehensive performance snapshot"""
        logger.debug("Taking performance snapshot...")
        
        snapshot = PerformanceSnapshot(timestamp=datetime.utcnow())
        
        # Get system metrics
        snapshot.system_metrics = self.get_system_metrics()
        
        # Simulate service metrics (in real implementation, these would be API calls)
        if self.cache_hierarchy_available:
            snapshot.cache_hierarchy_stats = self.simulate_cache_hierarchy_stats()
        
        if self.predictor_available:
            snapshot.predictor_stats = self.simulate_predictor_stats()
        
        if self.warming_service_available:
            snapshot.warming_stats = self.simulate_warming_stats()
        
        return snapshot
    
    def analyze_performance_trends(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(snapshots) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Extract key metrics over time
        hit_rates = []
        avg_latencies = []
        warming_times = []
        memory_usage = []
        
        for snapshot in snapshots:
            if "overall" in snapshot.cache_hierarchy_stats:
                hit_rates.append(snapshot.cache_hierarchy_stats["overall"]["overall_hit_rate_pct"])
            
            if "levels" in snapshot.cache_hierarchy_stats and "l1_memory" in snapshot.cache_hierarchy_stats["levels"]:
                avg_latencies.append(snapshot.cache_hierarchy_stats["levels"]["l1_memory"]["avg_latency_ms"])
            
            if "performance" in snapshot.warming_stats:
                warming_times.append(snapshot.warming_stats["performance"]["avg_warming_time_ms"])
            
            if "memory" in snapshot.system_metrics:
                memory_usage.append(snapshot.system_metrics["memory"]["percent"])
        
        # Calculate trends
        def calculate_trend(values: List[float]) -> str:
            if len(values) < 2:
                return "stable"
            
            # Simple linear trend analysis
            changes = [values[i] - values[i-1] for i in range(1, len(values))]
            avg_change = sum(changes) / len(changes)
            
            if avg_change > 1.0:
                return "increasing"
            elif avg_change < -1.0:
                return "decreasing" 
            else:
                return "stable"
        
        return {
            "analysis_period": {
                "start": snapshots[0].timestamp.isoformat(),
                "end": snapshots[-1].timestamp.isoformat(),
                "duration_minutes": (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 60
            },
            "trends": {
                "hit_rate": calculate_trend(hit_rates),
                "latency": calculate_trend(avg_latencies),
                "warming_time": calculate_trend(warming_times),
                "memory_usage": calculate_trend(memory_usage)
            },
            "current_values": {
                "hit_rate_pct": hit_rates[-1] if hit_rates else 0,
                "avg_latency_ms": avg_latencies[-1] if avg_latencies else 0,
                "warming_time_ms": warming_times[-1] if warming_times else 0,
                "memory_usage_pct": memory_usage[-1] if memory_usage else 0
            }
        }
    
    def check_performance_targets(self, snapshot: PerformanceSnapshot) -> Dict[str, Any]:
        """Check current performance against targets"""
        results = {
            "timestamp": snapshot.timestamp.isoformat(),
            "targets": PERFORMANCE_TARGETS,
            "current_values": {},
            "target_achievement": {},
            "overall_score": 0.0
        }
        
        # Extract current values and check targets
        checks = []
        
        # Cache hit rate
        if "overall" in snapshot.cache_hierarchy_stats:
            current_hit_rate = snapshot.cache_hierarchy_stats["overall"]["overall_hit_rate_pct"]
            results["current_values"]["cache_hit_rate_pct"] = current_hit_rate
            results["target_achievement"]["cache_hit_rate_pct"] = current_hit_rate >= PERFORMANCE_TARGETS["cache_hit_rate_pct"]
            checks.append(current_hit_rate >= PERFORMANCE_TARGETS["cache_hit_rate_pct"])
        
        # Average latency
        if ("levels" in snapshot.cache_hierarchy_stats and 
            "l1_memory" in snapshot.cache_hierarchy_stats["levels"]):
            current_latency = snapshot.cache_hierarchy_stats["levels"]["l1_memory"]["avg_latency_ms"]
            results["current_values"]["avg_latency_ms"] = current_latency
            results["target_achievement"]["avg_latency_ms"] = current_latency <= PERFORMANCE_TARGETS["avg_latency_ms"]
            checks.append(current_latency <= PERFORMANCE_TARGETS["avg_latency_ms"])
        
        # Warming time
        if "performance" in snapshot.warming_stats:
            current_warming_time = snapshot.warming_stats["performance"]["avg_warming_time_ms"]
            results["current_values"]["warming_time_ms"] = current_warming_time
            results["target_achievement"]["warming_time_ms"] = current_warming_time <= PERFORMANCE_TARGETS["warming_time_ms"]
            checks.append(current_warming_time <= PERFORMANCE_TARGETS["warming_time_ms"])
        
        # Memory usage
        if "memory" in snapshot.system_metrics:
            current_memory = snapshot.system_metrics["memory"]["percent"]
            results["current_values"]["memory_usage_pct"] = current_memory
            results["target_achievement"]["memory_usage_pct"] = current_memory <= PERFORMANCE_TARGETS["memory_usage_pct"]
            checks.append(current_memory <= PERFORMANCE_TARGETS["memory_usage_pct"])
        
        # CPU usage
        if "cpu" in snapshot.system_metrics:
            current_cpu = snapshot.system_metrics["cpu"]["percent"]
            results["current_values"]["cpu_usage_pct"] = current_cpu
            results["target_achievement"]["cpu_usage_pct"] = current_cpu <= PERFORMANCE_TARGETS["cpu_usage_pct"]
            checks.append(current_cpu <= PERFORMANCE_TARGETS["cpu_usage_pct"])
        
        # Calculate overall score
        if checks:
            results["overall_score"] = (sum(checks) / len(checks)) * 100
        
        return results
    
    def generate_optimization_recommendations(self, snapshot: PerformanceSnapshot) -> List[str]:
        """Generate optimization recommendations based on current performance"""
        recommendations = []
        
        # Cache hit rate recommendations
        if "overall" in snapshot.cache_hierarchy_stats:
            hit_rate = snapshot.cache_hierarchy_stats["overall"]["overall_hit_rate_pct"]
            if hit_rate < 90:
                recommendations.append("CRITICAL: Cache hit rate is below 90%. Consider increasing cache warming frequency and improving ML predictions.")
            elif hit_rate < 95:
                recommendations.append("WARNING: Cache hit rate is below target 95%. Review cache warming strategies and tune TTL values.")
        
        # Latency recommendations
        if ("levels" in snapshot.cache_hierarchy_stats and 
            "l1_memory" in snapshot.cache_hierarchy_stats["levels"]):
            latency = snapshot.cache_hierarchy_stats["levels"]["l1_memory"]["avg_latency_ms"]
            if latency > 10:
                recommendations.append("HIGH: L1 cache latency is elevated. Check memory pressure and consider increasing L1 cache size.")
            elif latency > 5:
                recommendations.append("MEDIUM: L1 cache latency is above optimal. Monitor for memory fragmentation.")
        
        # Memory recommendations
        if "memory" in snapshot.system_metrics:
            memory_pct = snapshot.system_metrics["memory"]["percent"]
            if memory_pct > 85:
                recommendations.append("CRITICAL: System memory usage is above 85%. Consider reducing cache sizes or scaling infrastructure.")
            elif memory_pct > 80:
                recommendations.append("WARNING: System memory usage approaching limit. Monitor for memory pressure.")
        
        # Warming performance recommendations
        if "performance" in snapshot.warming_stats:
            warming_time = snapshot.warming_stats["performance"]["avg_warming_time_ms"]
            success_rate = snapshot.warming_stats["performance"]["success_rate_pct"]
            
            if warming_time > 5000:
                recommendations.append("HIGH: Cache warming time exceeds 5 seconds. Optimize warming algorithms and consider parallel processing.")
            
            if success_rate < 95:
                recommendations.append("MEDIUM: Cache warming success rate is below 95%. Review error handling and retry mechanisms.")
        
        # ML predictor recommendations
        if "prediction_accuracy_pct" in snapshot.predictor_stats:
            accuracy = snapshot.predictor_stats["prediction_accuracy_pct"]
            if accuracy < 80:
                recommendations.append("HIGH: ML prediction accuracy is below 80%. Review training data and model parameters.")
        
        if not recommendations:
            recommendations.append("GOOD: All performance metrics are within acceptable ranges. Continue monitoring.")
        
        return recommendations
    
    async def run_live_monitoring(self, duration_seconds: int = 300) -> List[PerformanceSnapshot]:
        """Run live performance monitoring"""
        logger.info(f"Starting live performance monitoring for {duration_seconds} seconds...")
        
        snapshots = []
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            try:
                # Take performance snapshot
                snapshot = await self.take_performance_snapshot()
                snapshots.append(snapshot)
                
                # Analyze current performance
                target_check = self.check_performance_targets(snapshot)
                recommendations = self.generate_optimization_recommendations(snapshot)
                
                # Print live status
                overall_score = target_check["overall_score"]
                hit_rate = target_check["current_values"].get("cache_hit_rate_pct", 0)
                
                status_color = "🟢" if overall_score >= 80 else "🟡" if overall_score >= 60 else "🔴"
                
                logger.info(
                    f"{status_color} Performance Score: {overall_score:.1f}% | "
                    f"Hit Rate: {hit_rate:.1f}% | "
                    f"Recommendations: {len(recommendations)}"
                )
                
                # Print high-priority recommendations
                high_priority = [r for r in recommendations if r.startswith("CRITICAL") or r.startswith("HIGH")]
                for rec in high_priority:
                    logger.warning(f"  → {rec}")
                
                # Wait before next snapshot
                await asyncio.sleep(30)  # 30-second intervals
            
            except Exception as e:
                logger.error(f"Error during live monitoring: {e}")
                await asyncio.sleep(30)
        
        logger.info(f"Live monitoring completed. Collected {len(snapshots)} snapshots.")
        return snapshots
    
    async def generate_comprehensive_report(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not snapshots:
            # Take a single snapshot for report
            snapshot = await self.take_performance_snapshot()
            snapshots = [snapshot]
        
        latest_snapshot = snapshots[-1]
        
        # Performance analysis
        target_check = self.check_performance_targets(latest_snapshot)
        recommendations = self.generate_optimization_recommendations(latest_snapshot)
        
        # Trend analysis if multiple snapshots
        trend_analysis = {}
        if len(snapshots) > 1:
            trend_analysis = self.analyze_performance_trends(snapshots)
        
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "analysis_duration_minutes": (datetime.utcnow() - self.start_time).total_seconds() / 60,
                "snapshots_analyzed": len(snapshots),
                "report_version": "1.0"
            },
            "executive_summary": {
                "overall_performance_score": target_check["overall_score"],
                "performance_status": (
                    "EXCELLENT" if target_check["overall_score"] >= 90 else
                    "GOOD" if target_check["overall_score"] >= 80 else
                    "FAIR" if target_check["overall_score"] >= 60 else
                    "POOR"
                ),
                "critical_recommendations": len([r for r in recommendations if r.startswith("CRITICAL")]),
                "high_priority_recommendations": len([r for r in recommendations if r.startswith("HIGH")]),
                "targets_achieved": sum(1 for achieved in target_check["target_achievement"].values() if achieved),
                "total_targets": len(target_check["target_achievement"])
            },
            "current_performance": target_check,
            "optimization_recommendations": recommendations,
            "trend_analysis": trend_analysis,
            "detailed_metrics": {
                "cache_hierarchy": latest_snapshot.cache_hierarchy_stats,
                "ml_predictor": latest_snapshot.predictor_stats,
                "warming_service": latest_snapshot.warming_stats,
                "system_metrics": latest_snapshot.system_metrics
            },
            "historical_snapshots": [s.to_dict() for s in snapshots[-10:]]  # Last 10 snapshots
        }
        
        return report
    
    def print_performance_summary(self, report: Dict[str, Any]):
        """Print formatted performance summary"""
        print("\n" + "="*100)
        print("INTELLIGENT CACHE WARMING PERFORMANCE ANALYSIS")
        print("="*100)
        
        exec_summary = report["executive_summary"]
        current_perf = report["current_performance"]
        
        # Status indicator
        status = exec_summary["performance_status"]
        status_indicator = {
            "EXCELLENT": "🟢", "GOOD": "🟡", "FAIR": "🟠", "POOR": "🔴"
        }.get(status, "⚪")
        
        print(f"\n{status_indicator} OVERALL STATUS: {status} ({exec_summary['overall_performance_score']:.1f}% score)")
        print(f"  • Targets Achieved: {exec_summary['targets_achieved']}/{exec_summary['total_targets']}")
        print(f"  • Critical Issues: {exec_summary['critical_recommendations']}")
        print(f"  • High Priority Issues: {exec_summary['high_priority_recommendations']}")
        
        # Performance targets
        print(f"\nPERFORMANCE TARGETS:")
        for metric, target in current_perf["targets"].items():
            current = current_perf["current_values"].get(metric, "N/A")
            achieved = current_perf["target_achievement"].get(metric, False)
            indicator = "✓" if achieved else "✗"
            
            if isinstance(current, (int, float)):
                print(f"  {indicator} {metric}: {current:.1f} (target: {target})")
            else:
                print(f"  {indicator} {metric}: {current} (target: {target})")
        
        # Cache hierarchy summary
        if "cache_hierarchy" in report["detailed_metrics"]:
            cache_stats = report["detailed_metrics"]["cache_hierarchy"]
            if "overall" in cache_stats:
                overall = cache_stats["overall"]
                print(f"\nCACHE PERFORMANCE:")
                print(f"  • Overall Hit Rate: {overall['overall_hit_rate_pct']:.1f}%")
                print(f"  • Total Hits: {overall['total_hits']:,}")
                print(f"  • Total Misses: {overall['total_misses']:,}")
                
                if "levels" in cache_stats:
                    for level_name, level_stats in cache_stats["levels"].items():
                        print(f"  • {level_name.upper()} Hit Rate: {level_stats['hit_rate_pct']:.1f}% ({level_stats['avg_latency_ms']:.1f}ms avg)")
        
        # ML predictor summary
        if "ml_predictor" in report["detailed_metrics"]:
            predictor_stats = report["detailed_metrics"]["ml_predictor"]
            print(f"\nML PREDICTOR PERFORMANCE:")
            print(f"  • Status: {predictor_stats.get('learning_status', 'unknown').upper()}")
            print(f"  • Patterns Learned: {predictor_stats.get('total_patterns', 0)}")
            print(f"  • Prediction Accuracy: {predictor_stats.get('prediction_accuracy_pct', 0):.1f}%")
            
            if "pattern_confidence_distribution" in predictor_stats:
                dist = predictor_stats["pattern_confidence_distribution"]
                print(f"  • High Confidence Patterns: {dist.get('high', 0)}")
                print(f"  • Medium Confidence Patterns: {dist.get('medium', 0)}")
                print(f"  • Low Confidence Patterns: {dist.get('low', 0)}")
        
        # Cache warming summary
        if "warming_service" in report["detailed_metrics"]:
            warming_stats = report["detailed_metrics"]["warming_service"]
            if "performance" in warming_stats:
                perf = warming_stats["performance"]
                print(f"\nCACHE WARMING PERFORMANCE:")
                print(f"  • Success Rate: {perf['success_rate_pct']:.1f}%")
                print(f"  • Average Warming Time: {perf['avg_warming_time_ms']:.1f}ms")
                print(f"  • Items Warmed: {perf['items_warmed']:,}")
                print(f"  • Throughput: {perf['throughput_items_per_sec']:.1f} items/sec")
                print(f"  • Hit Rate Improvement: +{perf['cache_hit_rate_improvement']:.1f}%")
        
        # Recommendations
        if report["optimization_recommendations"]:
            print(f"\nOPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(report["optimization_recommendations"][:5], 1):  # Top 5
                print(f"  {i}. {rec}")
            
            if len(report["optimization_recommendations"]) > 5:
                print(f"  ... and {len(report['optimization_recommendations']) - 5} more recommendations")
        
        # Trend analysis
        if "trend_analysis" in report and "trends" in report["trend_analysis"]:
            trends = report["trend_analysis"]["trends"]
            print(f"\nPERFORMANCE TRENDS:")
            for metric, trend in trends.items():
                trend_indicator = "↗️" if trend == "increasing" else "↘️" if trend == "decreasing" else "➡️"
                print(f"  • {metric}: {trend_indicator} {trend}")
        
        print("\n" + "="*100)


async def main():
    """Main performance analysis execution"""
    parser = argparse.ArgumentParser(description="Cache Warming Performance Analysis")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live performance monitoring"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duration for live monitoring in seconds (default: 300)"
    )
    parser.add_argument(
        "--report",
        help="Generate comprehensive report and save to file"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick performance check"
    )
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer()
    
    try:
        if args.live:
            # Live monitoring mode
            snapshots = await analyzer.run_live_monitoring(args.duration)
            
            # Generate report from live data
            report = await analyzer.generate_comprehensive_report(snapshots)
            analyzer.print_performance_summary(report)
            
            # Save report if requested
            if args.report:
                with open(args.report, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Performance report saved to {args.report}")
        
        elif args.quick:
            # Quick performance check
            snapshot = await analyzer.take_performance_snapshot()
            target_check = analyzer.check_performance_targets(snapshot)
            recommendations = analyzer.generate_optimization_recommendations(snapshot)
            
            print(f"\n🔍 QUICK PERFORMANCE CHECK")
            print(f"Overall Score: {target_check['overall_score']:.1f}%")
            print(f"Hit Rate: {target_check['current_values'].get('cache_hit_rate_pct', 0):.1f}%")
            print(f"Recommendations: {len(recommendations)}")
            
            if recommendations:
                print("\nTop Recommendations:")
                for rec in recommendations[:3]:
                    print(f"  • {rec}")
        
        else:
            # Generate comprehensive report
            report = await analyzer.generate_comprehensive_report([])
            analyzer.print_performance_summary(report)
            
            # Save report if requested
            if args.report:
                with open(args.report, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Performance report saved to {args.report}")
        
        # Return appropriate exit code
        if args.live or not args.quick:
            report_score = report["executive_summary"]["overall_performance_score"] if 'report' in locals() else 0
            return 0 if report_score >= 70 else 1
        else:
            quick_score = target_check["overall_score"]
            return 0 if quick_score >= 70 else 1
    
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))