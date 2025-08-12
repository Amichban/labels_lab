#!/usr/bin/env python3
"""
Comprehensive Performance Report Generator for Label Computation System.

This script orchestrates all performance tests and generates detailed reports:
- Executes load, stress, soak, and spike tests
- Collects system metrics and performance data
- Generates HTML and markdown reports with visualizations
- Provides performance recommendations and optimization guidance
- Integrates with CI/CD for automated performance reporting

Usage:
    # Full performance test suite
    python scripts/performance_report.py --full
    
    # Quick performance check
    python scripts/performance_report.py --quick
    
    # Specific test categories
    python scripts/performance_report.py --load --stress
    
    # Generate report only (from existing data)
    python scripts/performance_report.py --report-only
"""

import asyncio
import argparse
import json
import sys
import time
import subprocess
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import tempfile
import logging
import traceback
from dataclasses import dataclass, asdict
from collections import defaultdict

# Third party imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, visual reports will be limited")

try:
    import psutil
    SYSTEM_MONITORING_AVAILABLE = True
except ImportError:
    SYSTEM_MONITORING_AVAILABLE = False
    print("Warning: psutil not available, system monitoring limited")


@dataclass
class PerformanceTestSuite:
    """Configuration for performance test execution."""
    name: str
    test_command: List[str]
    timeout_minutes: int
    required_metrics: List[str]
    success_criteria: Dict[str, float]
    category: str


@dataclass
class TestResult:
    """Results from a single performance test."""
    suite_name: str
    success: bool
    duration_seconds: float
    metrics: Dict[str, Any]
    errors: List[str]
    output: str
    recommendations: List[str]


@dataclass
class SystemSnapshot:
    """System resource snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io: Dict[str, int]
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]


class PerformanceReportGenerator:
    """Generates comprehensive performance reports."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or "/Users/aminechbani/labels_lab/my-project/performance_reports")
        self.output_dir.mkdir(exist_ok=True)
        
        self.test_results: List[TestResult] = []
        self.system_snapshots: List[SystemSnapshot] = []
        self.start_time = datetime.now()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / "performance_test.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def define_test_suites(self) -> Dict[str, PerformanceTestSuite]:
        """Define all available performance test suites."""
        base_path = Path("/Users/aminechbani/labels_lab/my-project")
        
        return {
            "load": PerformanceTestSuite(
                name="Load Testing",
                test_command=["python", "-m", "pytest", "tests/performance/load_test.py", "-v", "--tb=short"],
                timeout_minutes=30,
                required_metrics=["throughput", "latency", "success_rate"],
                success_criteria={
                    "throughput_ops_per_second": 1000,
                    "success_rate": 0.95,
                    "p99_latency_ms": 100
                },
                category="load"
            ),
            
            "stress": PerformanceTestSuite(
                name="Stress Testing",
                test_command=["python", "-m", "pytest", "tests/performance/stress_test.py", "-v", "--tb=short"],
                timeout_minutes=60,
                required_metrics=["breaking_point", "max_load", "stability"],
                success_criteria={
                    "max_successful_load": 1000,
                    "stability_score": 70
                },
                category="stress"
            ),
            
            "soak_short": PerformanceTestSuite(
                name="Soak Testing (1 hour)",
                test_command=["python", "-m", "pytest", "tests/performance/soak_test.py::TestSoakTesting::test_1_hour_soak", "-v", "--tb=short"],
                timeout_minutes=90,
                required_metrics=["stability", "memory_growth", "degradation"],
                success_criteria={
                    "overall_success_rate": 0.90,
                    "memory_growth_mb_per_hour": 100,
                    "performance_degradation": 50
                },
                category="soak"
            ),
            
            "memory_leak": PerformanceTestSuite(
                name="Memory Leak Detection",
                test_command=["python", "-m", "pytest", "tests/performance/soak_test.py::TestSoakTesting::test_memory_leak_detection", "-v", "--tb=short"],
                timeout_minutes=45,
                required_metrics=["memory_growth", "leak_detected"],
                success_criteria={
                    "memory_growth_mb_per_hour": 50,
                    "overall_success_rate": 0.85
                },
                category="soak"
            ),
            
            "spike": PerformanceTestSuite(
                name="Spike Testing",
                test_command=["python", "-m", "pytest", "tests/performance/spike_test.py", "-v", "--tb=short"],
                timeout_minutes=45,
                required_metrics=["spike_handling", "recovery", "stability"],
                success_criteria={
                    "success_rate_during_spike": 0.80,
                    "success_rate_post_spike": 0.90,
                    "system_stability_score": 50
                },
                category="spike"
            ),
            
            "baseline": PerformanceTestSuite(
                name="Baseline Benchmarks",
                test_command=["python", "-m", "pytest", "tests/performance/test_benchmarks.py", "-v", "--tb=short"],
                timeout_minutes=20,
                required_metrics=["cache_performance", "latency_distribution"],
                success_criteria={
                    "cache_hit_latency_ms": 1.0,
                    "p99_latency_ms": 100
                },
                category="baseline"
            )
        }
    
    async def run_performance_tests(self, test_categories: List[str] = None) -> bool:
        """Run performance tests and collect results."""
        test_suites = self.define_test_suites()
        
        if test_categories:
            # Filter test suites by category
            test_suites = {k: v for k, v in test_suites.items() if v.category in test_categories}
        
        self.logger.info(f"Starting performance tests: {list(test_suites.keys())}")
        
        # Start system monitoring
        monitoring_task = None
        if SYSTEM_MONITORING_AVAILABLE:
            monitoring_task = asyncio.create_task(self._monitor_system_resources())
        
        success = True
        
        for suite_name, suite in test_suites.items():
            self.logger.info(f"Running {suite.name}...")
            
            try:
                result = await self._run_test_suite(suite)
                self.test_results.append(result)
                
                if not result.success:
                    success = False
                    self.logger.error(f"{suite.name} failed")
                else:
                    self.logger.info(f"{suite.name} completed successfully")
                    
            except Exception as e:
                self.logger.error(f"Error running {suite.name}: {e}")
                self.logger.error(traceback.format_exc())
                success = False
                
                # Record failed test
                self.test_results.append(TestResult(
                    suite_name=suite.name,
                    success=False,
                    duration_seconds=0,
                    metrics={},
                    errors=[str(e)],
                    output="",
                    recommendations=[f"Fix error: {str(e)}"]
                ))
        
        # Stop system monitoring
        if monitoring_task:
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        
        return success
    
    async def _run_test_suite(self, suite: PerformanceTestSuite) -> TestResult:
        """Run a single test suite."""
        start_time = time.time()
        
        try:
            # Change to project directory
            project_dir = Path("/Users/aminechbani/labels_lab/my-project")
            
            # Run test command
            process = await asyncio.create_subprocess_exec(
                *suite.test_command,
                cwd=project_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=suite.timeout_minutes * 60
                )
            except asyncio.TimeoutError:
                process.terminate()
                raise Exception(f"Test suite timed out after {suite.timeout_minutes} minutes")
            
            duration_seconds = time.time() - start_time
            
            # Decode output
            stdout_text = stdout.decode('utf-8') if stdout else ""
            stderr_text = stderr.decode('utf-8') if stderr else ""
            
            # Parse results from output
            metrics = self._parse_test_output(stdout_text, stderr_text, suite)
            
            # Determine success
            success = process.returncode == 0 and self._check_success_criteria(metrics, suite)
            
            # Extract errors
            errors = []
            if process.returncode != 0:
                errors.append(f"Test command failed with return code {process.returncode}")
            
            if stderr_text:
                errors.extend(stderr_text.split('\n')[-10:])  # Last 10 lines of errors
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, suite, success)
            
            return TestResult(
                suite_name=suite.name,
                success=success,
                duration_seconds=duration_seconds,
                metrics=metrics,
                errors=errors,
                output=stdout_text,
                recommendations=recommendations
            )
            
        except Exception as e:
            duration_seconds = time.time() - start_time
            return TestResult(
                suite_name=suite.name,
                success=False,
                duration_seconds=duration_seconds,
                metrics={},
                errors=[str(e)],
                output="",
                recommendations=[f"Fix error: {str(e)}"]
            )
    
    def _parse_test_output(self, stdout: str, stderr: str, suite: PerformanceTestSuite) -> Dict[str, Any]:
        """Parse metrics from test output."""
        metrics = {}
        
        # Look for common performance metrics patterns in output
        lines = stdout.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Load test metrics
            if "candles/second" in line:
                try:
                    value = float(line.split("candles/second")[0].strip().split()[-1])
                    metrics["throughput_ops_per_second"] = value
                except (ValueError, IndexError):
                    pass
            
            if "Success rate:" in line:
                try:
                    rate_str = line.split("Success rate:")[1].strip().split()[0].rstrip('%')
                    metrics["success_rate"] = float(rate_str) / 100
                except (ValueError, IndexError):
                    pass
            
            if "P99 latency:" in line:
                try:
                    value = float(line.split("P99 latency:")[1].strip().split()[0].rstrip('ms'))
                    metrics["p99_latency_ms"] = value
                except (ValueError, IndexError):
                    pass
            
            if "Average latency:" in line:
                try:
                    value = float(line.split("Average latency:")[1].strip().split()[0].rstrip('ms'))
                    metrics["avg_latency_ms"] = value
                except (ValueError, IndexError):
                    pass
            
            # Stress test metrics
            if "Breaking point:" in line:
                try:
                    value = line.split("Breaking point:")[1].strip().split()[0]
                    if value != "Not reached":
                        metrics["breaking_point"] = int(value)
                    else:
                        metrics["breaking_point"] = None
                except (ValueError, IndexError):
                    pass
            
            if "Max successful load:" in line:
                try:
                    value = int(line.split("Max successful load:")[1].strip().split()[0])
                    metrics["max_successful_load"] = value
                except (ValueError, IndexError):
                    pass
            
            if "Stability score:" in line:
                try:
                    value = float(line.split("Stability score:")[1].strip().split('/')[0])
                    metrics["stability_score"] = value
                except (ValueError, IndexError):
                    pass
            
            # Soak test metrics
            if "Memory growth:" in line and "MB/hour" in line:
                try:
                    value = float(line.split("Memory growth:")[1].strip().split()[0])
                    metrics["memory_growth_mb_per_hour"] = value
                except (ValueError, IndexError):
                    pass
            
            if "Performance degradation:" in line:
                try:
                    value = float(line.split("Performance degradation:")[1].strip().split('%')[0])
                    metrics["performance_degradation"] = value
                except (ValueError, IndexError):
                    pass
            
            if "Overall success rate:" in line:
                try:
                    rate_str = line.split("Overall success rate:")[1].strip().split()[0].rstrip('%')
                    metrics["overall_success_rate"] = float(rate_str) / 100
                except (ValueError, IndexError):
                    pass
            
            # Spike test metrics
            if "Success rate during spike:" in line:
                try:
                    rate_str = line.split("Success rate during spike:")[1].strip().split()[0].rstrip('%')
                    metrics["success_rate_during_spike"] = float(rate_str) / 100
                except (ValueError, IndexError):
                    pass
            
            if "Success rate post-spike:" in line:
                try:
                    rate_str = line.split("Success rate post-spike:")[1].strip().split()[0].rstrip('%')
                    metrics["success_rate_post_spike"] = float(rate_str) / 100
                except (ValueError, IndexError):
                    pass
            
            if "System stability score:" in line:
                try:
                    value = float(line.split("System stability score:")[1].strip().split('/')[0])
                    metrics["system_stability_score"] = value
                except (ValueError, IndexError):
                    pass
        
        return metrics
    
    def _check_success_criteria(self, metrics: Dict[str, Any], suite: PerformanceTestSuite) -> bool:
        """Check if test results meet success criteria."""
        for criterion, threshold in suite.success_criteria.items():
            if criterion not in metrics:
                continue
            
            value = metrics[criterion]
            if value is None:
                continue
            
            # For rates and percentages, higher is better
            if criterion.endswith("_rate") or criterion.endswith("success_rate"):
                if value < threshold:
                    return False
            
            # For latency and memory growth, lower is better
            elif "latency" in criterion or "growth" in criterion or "degradation" in criterion:
                if value > threshold:
                    return False
            
            # For load and throughput, higher is better
            elif "load" in criterion or "throughput" in criterion or "score" in criterion:
                if value < threshold:
                    return False
        
        return True
    
    def _generate_recommendations(self, metrics: Dict[str, Any], suite: PerformanceTestSuite, success: bool) -> List[str]:
        """Generate performance recommendations based on test results."""
        recommendations = []
        
        if not success:
            recommendations.append(f"❌ {suite.name} did not meet success criteria")
        
        # Load test recommendations
        if suite.category == "load":
            throughput = metrics.get("throughput_ops_per_second", 0)
            if throughput < 1000:
                recommendations.append(f"🔧 Low throughput ({throughput:.1f} ops/sec) - optimize database queries and connection pooling")
            
            p99_latency = metrics.get("p99_latency_ms", 0)
            if p99_latency > 100:
                recommendations.append(f"🔧 High P99 latency ({p99_latency:.1f}ms) - implement caching and optimize slow operations")
            
            success_rate = metrics.get("success_rate", 1.0)
            if success_rate < 0.95:
                recommendations.append(f"🔧 Low success rate ({success_rate:.1%}) - improve error handling and resilience")
        
        # Stress test recommendations
        elif suite.category == "stress":
            breaking_point = metrics.get("breaking_point")
            if breaking_point and breaking_point < 2000:
                recommendations.append(f"🔧 Low breaking point ({breaking_point}) - implement auto-scaling and load balancing")
            
            stability_score = metrics.get("stability_score", 0)
            if stability_score < 70:
                recommendations.append(f"🔧 Low stability score ({stability_score:.1f}) - improve error recovery and resource management")
        
        # Soak test recommendations  
        elif suite.category == "soak":
            memory_growth = metrics.get("memory_growth_mb_per_hour", 0)
            if memory_growth > 50:
                recommendations.append(f"🔧 High memory growth ({memory_growth:.1f}MB/hour) - investigate memory leaks")
            
            degradation = metrics.get("performance_degradation", 0)
            if degradation > 25:
                recommendations.append(f"🔧 Performance degradation ({degradation:.1f}%) - consider periodic restarts or cleanup")
        
        # Spike test recommendations
        elif suite.category == "spike":
            spike_success = metrics.get("success_rate_during_spike", 1.0)
            if spike_success < 0.80:
                recommendations.append(f"🔧 Poor spike handling ({spike_success:.1%}) - implement circuit breakers and backpressure")
            
            recovery_success = metrics.get("success_rate_post_spike", 1.0)
            if recovery_success < 0.90:
                recommendations.append(f"🔧 Poor recovery ({recovery_success:.1%}) - improve system resilience and monitoring")
        
        # General recommendations
        if success:
            recommendations.append("✅ Test passed - system performing within acceptable parameters")
        
        return recommendations
    
    async def _monitor_system_resources(self):
        """Monitor system resources during tests."""
        try:
            while True:
                snapshot = SystemSnapshot(
                    timestamp=datetime.now(),
                    cpu_percent=psutil.cpu_percent(interval=1),
                    memory_percent=psutil.virtual_memory().percent,
                    memory_mb=psutil.virtual_memory().used / (1024 * 1024),
                    disk_io=dict(psutil.disk_io_counters()._asdict()) if psutil.disk_io_counters() else {},
                    network_io=dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {},
                    process_count=len(psutil.pids()),
                    load_average=list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
                )
                
                self.system_snapshots.append(snapshot)
                await asyncio.sleep(5)  # Sample every 5 seconds
                
        except asyncio.CancelledError:
            pass
    
    def generate_reports(self) -> Dict[str, str]:
        """Generate comprehensive performance reports."""
        report_files = {}
        
        # Generate markdown report
        markdown_report = self._generate_markdown_report()
        markdown_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        report_files['markdown'] = str(markdown_file)
        
        # Generate JSON report for programmatic use
        json_report = self._generate_json_report()
        json_file = self.output_dir / f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        report_files['json'] = str(json_file)
        
        # Generate HTML report with visualizations
        if PLOTTING_AVAILABLE:
            html_report = self._generate_html_report()
            html_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(html_file, 'w') as f:
                f.write(html_report)
            report_files['html'] = str(html_file)
        
        # Generate CSV data for analysis
        csv_file = self._generate_csv_data()
        report_files['csv'] = csv_file
        
        return report_files
    
    def _generate_markdown_report(self) -> str:
        """Generate comprehensive markdown report."""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        lines = [
            "# Performance Test Report",
            f"Generated: {datetime.now().isoformat()}",
            f"Total Duration: {total_duration / 60:.1f} minutes",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Overall results
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        lines.extend([
            f"- **Total Tests:** {total_tests}",
            f"- **Passed:** {passed_tests}",
            f"- **Failed:** {total_tests - passed_tests}",
            f"- **Success Rate:** {success_rate:.1%}",
            ""
        ])
        
        # Performance scorecard
        lines.append("## Performance Scorecard")
        lines.append("")
        
        scorecard = self._calculate_performance_scores()
        
        for category, score in scorecard.items():
            status = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            lines.append(f"- **{category}:** {score:.0f}/100 {status}")
        
        lines.append("")
        
        # Test results by category
        categories = defaultdict(list)
        for result in self.test_results:
            category = self._get_test_category(result.suite_name)
            categories[category].append(result)
        
        for category, results in categories.items():
            lines.extend([
                f"## {category.title()} Testing Results",
                ""
            ])
            
            for result in results:
                status = "✅ PASSED" if result.success else "❌ FAILED"
                lines.extend([
                    f"### {result.suite_name} {status}",
                    f"**Duration:** {result.duration_seconds / 60:.1f} minutes",
                    ""
                ])
                
                # Key metrics
                if result.metrics:
                    lines.append("**Key Metrics:**")
                    for metric, value in result.metrics.items():
                        if isinstance(value, float):
                            lines.append(f"- {metric}: {value:.2f}")
                        else:
                            lines.append(f"- {metric}: {value}")
                    lines.append("")
                
                # Recommendations
                if result.recommendations:
                    lines.append("**Recommendations:**")
                    for rec in result.recommendations:
                        lines.append(f"- {rec}")
                    lines.append("")
                
                # Errors (if any)
                if result.errors:
                    lines.append("**Errors:**")
                    for error in result.errors[-5:]:  # Last 5 errors
                        lines.append(f"- {error}")
                    lines.append("")
        
        # System resource analysis
        if self.system_snapshots:
            lines.extend([
                "## System Resource Analysis",
                "",
                self._analyze_system_resources(),
                ""
            ])
        
        # Overall recommendations
        lines.extend([
            "## Overall Recommendations",
            ""
        ])
        
        overall_recommendations = self._generate_overall_recommendations()
        for rec in overall_recommendations:
            lines.append(f"- {rec}")
        
        lines.extend([
            "",
            "## Next Steps",
            "",
            "1. **Address Critical Issues:** Focus on failed tests and low scores first",
            "2. **Optimize Performance:** Implement recommendations for your lowest-scoring areas",
            "3. **Monitor Production:** Set up monitoring based on these test results",
            "4. **Schedule Regular Testing:** Run these tests regularly to track improvements",
            "",
            "---",
            f"*Report generated by Performance Report Generator v1.0*"
        ])
        
        return "\n".join(lines)
    
    def _generate_json_report(self) -> Dict[str, Any]:
        """Generate JSON report for programmatic use."""
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "generator_version": "1.0"
            },
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for r in self.test_results if r.success),
                "success_rate": sum(1 for r in self.test_results if r.success) / len(self.test_results) if self.test_results else 0,
                "performance_scores": self._calculate_performance_scores()
            },
            "test_results": [asdict(result) for result in self.test_results],
            "system_snapshots": [asdict(snapshot) for snapshot in self.system_snapshots],
            "recommendations": self._generate_overall_recommendations()
        }
    
    def _generate_html_report(self) -> str:
        """Generate HTML report with visualizations."""
        # Create visualizations
        chart_files = self._create_visualizations()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
        .metric {{ margin: 10px 0; }}
        .chart {{ margin: 20px 0; text-align: center; }}
        .recommendation {{ background: #e9ecef; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #dee2e6; padding: 8px; text-align: left; }}
        th {{ background-color: #e9ecef; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Test Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Duration: {(datetime.now() - self.start_time).total_seconds() / 60:.1f} minutes</p>
    </div>
    
    <h2>Executive Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Tests</td><td>{len(self.test_results)}</td></tr>
        <tr><td>Passed</td><td class="success">{sum(1 for r in self.test_results if r.success)}</td></tr>
        <tr><td>Failed</td><td class="danger">{len(self.test_results) - sum(1 for r in self.test_results if r.success)}</td></tr>
    </table>
    
    <h2>Performance Scores</h2>
    <table>
        <tr><th>Category</th><th>Score</th><th>Status</th></tr>
"""
        
        scorecard = self._calculate_performance_scores()
        for category, score in scorecard.items():
            status_class = "success" if score >= 80 else "warning" if score >= 60 else "danger"
            status_text = "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
            html_content += f'        <tr><td>{category}</td><td class="{status_class}">{score:.0f}/100</td><td class="{status_class}">{status_text}</td></tr>\n'
        
        html_content += """    </table>
    
    <h2>Visualizations</h2>
"""
        
        # Add charts
        for chart_name, chart_file in chart_files.items():
            html_content += f"""
    <div class="chart">
        <h3>{chart_name}</h3>
        <img src="{chart_file}" alt="{chart_name}" style="max-width: 800px;">
    </div>
"""
        
        # Test results
        html_content += """
    <h2>Test Results</h2>
"""
        
        for result in self.test_results:
            status_class = "success" if result.success else "danger"
            status_text = "PASSED" if result.success else "FAILED"
            
            html_content += f"""
    <h3>{result.suite_name} <span class="{status_class}">{status_text}</span></h3>
    <p><strong>Duration:</strong> {result.duration_seconds / 60:.1f} minutes</p>
"""
            
            if result.metrics:
                html_content += "    <h4>Key Metrics</h4>\n    <ul>\n"
                for metric, value in result.metrics.items():
                    if isinstance(value, float):
                        html_content += f"        <li>{metric}: {value:.2f}</li>\n"
                    else:
                        html_content += f"        <li>{metric}: {value}</li>\n"
                html_content += "    </ul>\n"
            
            if result.recommendations:
                html_content += "    <h4>Recommendations</h4>\n    <ul>\n"
                for rec in result.recommendations:
                    html_content += f"        <li class='recommendation'>{rec}</li>\n"
                html_content += "    </ul>\n"
        
        html_content += """
</body>
</html>
"""
        
        return html_content
    
    def _create_visualizations(self) -> Dict[str, str]:
        """Create performance visualization charts."""
        if not PLOTTING_AVAILABLE:
            return {}
        
        chart_files = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Performance scores chart
        if self.test_results:
            scores = self._calculate_performance_scores()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            categories = list(scores.keys())
            values = list(scores.values())
            
            colors = ['green' if v >= 80 else 'orange' if v >= 60 else 'red' for v in values]
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            
            ax.set_ylabel('Score (0-100)')
            ax.set_title('Performance Test Scores by Category')
            ax.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.0f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            chart_file = self.output_dir / "performance_scores.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            chart_files['Performance Scores'] = chart_file.name
        
        # System resources over time
        if self.system_snapshots:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            timestamps = [s.timestamp for s in self.system_snapshots]
            cpu_data = [s.cpu_percent for s in self.system_snapshots]
            memory_data = [s.memory_percent for s in self.system_snapshots]
            memory_mb_data = [s.memory_mb for s in self.system_snapshots]
            process_data = [s.process_count for s in self.system_snapshots]
            
            ax1.plot(timestamps, cpu_data, 'b-', linewidth=2)
            ax1.set_ylabel('CPU Usage (%)')
            ax1.set_title('CPU Usage Over Time')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(timestamps, memory_data, 'r-', linewidth=2)
            ax2.set_ylabel('Memory Usage (%)')
            ax2.set_title('Memory Usage Over Time')
            ax2.grid(True, alpha=0.3)
            
            ax3.plot(timestamps, memory_mb_data, 'g-', linewidth=2)
            ax3.set_ylabel('Memory Usage (MB)')
            ax3.set_title('Memory Usage (Absolute)')
            ax3.grid(True, alpha=0.3)
            
            ax4.plot(timestamps, process_data, 'm-', linewidth=2)
            ax4.set_ylabel('Process Count')
            ax4.set_title('Process Count Over Time')
            ax4.grid(True, alpha=0.3)
            
            for ax in [ax1, ax2, ax3, ax4]:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            chart_file = self.output_dir / "system_resources.png"
            plt.savefig(chart_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            chart_files['System Resources'] = chart_file.name
        
        return chart_files
    
    def _calculate_performance_scores(self) -> Dict[str, float]:
        """Calculate performance scores by category."""
        category_results = defaultdict(list)
        
        for result in self.test_results:
            category = self._get_test_category(result.suite_name)
            category_results[category].append(result)
        
        scores = {}
        
        for category, results in category_results.items():
            category_score = 0
            weight_sum = 0
            
            for result in results:
                # Base score from success
                base_score = 100 if result.success else 0
                
                # Adjust based on specific metrics
                if category == "Load" and result.metrics:
                    throughput = result.metrics.get("throughput_ops_per_second", 0)
                    if throughput >= 1000:
                        base_score = min(100, base_score + 20)
                    elif throughput >= 500:
                        base_score = min(100, base_score + 10)
                    
                    success_rate = result.metrics.get("success_rate", 0)
                    if success_rate >= 0.95:
                        base_score = min(100, base_score + 10)
                
                elif category == "Stress" and result.metrics:
                    stability_score = result.metrics.get("stability_score", 0)
                    base_score = min(100, stability_score)
                
                elif category == "Soak" and result.metrics:
                    memory_growth = result.metrics.get("memory_growth_mb_per_hour", 100)
                    if memory_growth <= 10:
                        base_score = min(100, base_score + 20)
                    elif memory_growth <= 50:
                        base_score = min(100, base_score + 10)
                
                category_score += base_score
                weight_sum += 1
            
            scores[category] = category_score / weight_sum if weight_sum > 0 else 0
        
        return scores
    
    def _get_test_category(self, suite_name: str) -> str:
        """Get category for a test suite."""
        suite_name_lower = suite_name.lower()
        
        if "load" in suite_name_lower:
            return "Load"
        elif "stress" in suite_name_lower:
            return "Stress"
        elif "soak" in suite_name_lower or "memory" in suite_name_lower:
            return "Soak"
        elif "spike" in suite_name_lower:
            return "Spike"
        else:
            return "Baseline"
    
    def _analyze_system_resources(self) -> str:
        """Analyze system resource usage during tests."""
        if not self.system_snapshots:
            return "No system resource data collected."
        
        cpu_data = [s.cpu_percent for s in self.system_snapshots]
        memory_data = [s.memory_percent for s in self.system_snapshots]
        memory_mb_data = [s.memory_mb for s in self.system_snapshots]
        
        analysis = []
        
        analysis.append("**CPU Usage:**")
        analysis.append(f"- Average: {np.mean(cpu_data):.1f}%")
        analysis.append(f"- Peak: {np.max(cpu_data):.1f}%")
        analysis.append(f"- 95th percentile: {np.percentile(cpu_data, 95):.1f}%")
        analysis.append("")
        
        analysis.append("**Memory Usage:**")
        analysis.append(f"- Average: {np.mean(memory_data):.1f}%")
        analysis.append(f"- Peak: {np.max(memory_data):.1f}%")
        analysis.append(f"- Growth: {np.max(memory_mb_data) - np.min(memory_mb_data):.1f}MB")
        analysis.append("")
        
        # Resource warnings
        warnings = []
        if np.max(cpu_data) > 90:
            warnings.append("🔴 High CPU usage detected - consider scaling")
        if np.max(memory_data) > 85:
            warnings.append("🔴 High memory usage detected - monitor for leaks")
        
        memory_growth = np.max(memory_mb_data) - np.min(memory_mb_data)
        if memory_growth > 500:  # >500MB growth
            warnings.append("🔴 Significant memory growth detected")
        
        if warnings:
            analysis.append("**Resource Warnings:**")
            for warning in warnings:
                analysis.append(f"- {warning}")
        else:
            analysis.append("✅ **Resource usage within acceptable limits**")
        
        return "\n".join(analysis)
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations across all tests."""
        recommendations = []
        
        # Collect all individual recommendations
        all_recs = []
        for result in self.test_results:
            all_recs.extend(result.recommendations)
        
        # Count recommendation themes
        theme_counts = defaultdict(int)
        for rec in all_recs:
            if "throughput" in rec.lower() or "ops/sec" in rec.lower():
                theme_counts["throughput"] += 1
            if "latency" in rec.lower() or "response time" in rec.lower():
                theme_counts["latency"] += 1
            if "memory" in rec.lower():
                theme_counts["memory"] += 1
            if "error" in rec.lower() or "failure" in rec.lower():
                theme_counts["reliability"] += 1
            if "cache" in rec.lower():
                theme_counts["caching"] += 1
            if "connection" in rec.lower() or "pool" in rec.lower():
                theme_counts["connections"] += 1
        
        # Generate prioritized recommendations
        if theme_counts.get("reliability", 0) > 2:
            recommendations.append("🔥 **Priority 1:** Improve system reliability - multiple tests show error handling issues")
        
        if theme_counts.get("throughput", 0) > 1:
            recommendations.append("⚡ **Priority 2:** Optimize throughput - implement connection pooling and async processing")
        
        if theme_counts.get("latency", 0) > 1:
            recommendations.append("🏃 **Priority 3:** Reduce latency - optimize database queries and implement caching")
        
        if theme_counts.get("memory", 0) > 1:
            recommendations.append("💾 **Priority 4:** Address memory issues - investigate leaks and optimize data structures")
        
        # System-wide recommendations
        recommendations.extend([
            "📊 **Monitoring:** Set up performance monitoring based on these test results",
            "🔄 **Automation:** Integrate these tests into CI/CD pipeline",
            "📈 **Capacity Planning:** Use load test results for production capacity planning",
            "🛠️ **Optimization:** Focus on areas with lowest performance scores first"
        ])
        
        return recommendations
    
    def _generate_csv_data(self) -> str:
        """Generate CSV file with test data for analysis."""
        csv_file = self.output_dir / f"performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_file, 'w') as f:
            # Write header
            f.write("suite_name,success,duration_seconds,")
            
            # Get all unique metric names
            all_metrics = set()
            for result in self.test_results:
                all_metrics.update(result.metrics.keys())
            
            for metric in sorted(all_metrics):
                f.write(f"{metric},")
            f.write("error_count\n")
            
            # Write data
            for result in self.test_results:
                f.write(f"{result.suite_name},{result.success},{result.duration_seconds},")
                
                for metric in sorted(all_metrics):
                    value = result.metrics.get(metric, "")
                    f.write(f"{value},")
                
                f.write(f"{len(result.errors)}\n")
        
        return str(csv_file)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Performance Test Report Generator")
    
    parser.add_argument("--full", action="store_true", 
                       help="Run full performance test suite (all categories)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick performance check (baseline only)")
    parser.add_argument("--load", action="store_true",
                       help="Run load testing")
    parser.add_argument("--stress", action="store_true", 
                       help="Run stress testing")
    parser.add_argument("--soak", action="store_true",
                       help="Run soak testing")
    parser.add_argument("--spike", action="store_true",
                       help="Run spike testing")
    parser.add_argument("--baseline", action="store_true",
                       help="Run baseline benchmarks")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report from existing data only")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Determine test categories
    test_categories = []
    
    if args.full:
        test_categories = ["load", "stress", "soak", "spike", "baseline"]
    elif args.quick:
        test_categories = ["baseline"]
    else:
        if args.load:
            test_categories.append("load")
        if args.stress:
            test_categories.append("stress")
        if args.soak:
            test_categories.append("soak")
        if args.spike:
            test_categories.append("spike")
        if args.baseline:
            test_categories.append("baseline")
    
    # Default to baseline if nothing specified
    if not test_categories and not args.report_only:
        test_categories = ["baseline"]
    
    # Create report generator
    generator = PerformanceReportGenerator(output_dir=args.output_dir)
    
    async def run_tests():
        success = True
        
        if not args.report_only:
            print(f"Starting performance tests: {test_categories}")
            success = await generator.run_performance_tests(test_categories)
        else:
            print("Generating report from existing data...")
        
        # Generate reports
        print("Generating performance reports...")
        report_files = generator.generate_reports()
        
        print("\n" + "="*60)
        print("PERFORMANCE TEST REPORT GENERATION COMPLETE")
        print("="*60)
        
        for report_type, file_path in report_files.items():
            print(f"{report_type.upper()} Report: {file_path}")
        
        if success:
            print("\n✅ All tests completed successfully!")
        else:
            print("\n❌ Some tests failed - check report for details")
            sys.exit(1)
    
    # Run the async function
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()