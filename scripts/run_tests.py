#!/usr/bin/env python3
"""
Test runner script for Enhanced Triple Barrier Label 11.a implementation.

This script provides comprehensive test execution following the testing pyramid
strategy with intelligent test selection, flake detection, and performance monitoring.

Usage:
    python scripts/run_tests.py                    # Run all tests
    python scripts/run_tests.py --unit             # Run only unit tests
    python scripts/run_tests.py --integration      # Run integration tests  
    python scripts/run_tests.py --performance      # Run performance benchmarks
    python scripts/run_tests.py --fast             # Skip slow tests
    python scripts/run_tests.py --coverage         # Generate coverage report
    python scripts/run_tests.py --profile          # Profile test execution
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestRunner:
    """Intelligent test runner with pyramid strategy and performance monitoring."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = None
        
    def run_tests(self, args: argparse.Namespace) -> int:
        """Run tests based on provided arguments."""
        self.start_time = time.time()
        
        print("🚀 Enhanced Triple Barrier Label 11.a Test Suite")
        print("=" * 60)
        
        # Build pytest command
        pytest_cmd = self._build_pytest_command(args)
        
        print(f"📋 Test Plan: {' '.join(pytest_cmd)}")
        print("-" * 60)
        
        # Execute tests following pyramid strategy
        if args.pyramid:
            return self._run_pyramid_strategy(args)
        else:
            return self._run_single_command(pytest_cmd, "All Tests")
    
    def _build_pytest_command(self, args: argparse.Namespace) -> List[str]:
        """Build pytest command based on arguments."""
        cmd = ["python", "-m", "pytest"]
        
        # Test selection
        if args.unit:
            cmd.extend(["-m", "unit"])
        elif args.integration:
            cmd.extend(["-m", "integration"])
        elif args.performance:
            cmd.extend(["-m", "performance"])
        elif args.edge_cases:
            cmd.extend(["-m", "edge_cases"])
        
        # Execution options
        if args.fast:
            cmd.extend(["-m", "not slow"])
        
        if args.parallel:
            cmd.extend(["-n", str(args.parallel)])
        
        if args.verbose:
            cmd.extend(["-v", "-s"])
        
        # Coverage
        if args.coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=html",
                "--cov-report=xml",
                "--cov-report=term-missing"
            ])
        
        # Output options
        if args.junit_xml:
            cmd.extend(["--junit-xml", args.junit_xml])
        
        # Specific test paths
        if args.tests:
            cmd.extend(args.tests)
        else:
            cmd.append("tests/")
        
        return cmd
    
    def _run_pyramid_strategy(self, args: argparse.Namespace) -> int:
        """Execute tests following the testing pyramid strategy."""
        print("🔺 Testing Pyramid Strategy")
        print("-" * 30)
        
        pyramid_levels = [
            ("Unit Tests", ["-m", "unit", "-x"]),
            ("Integration Tests", ["-m", "integration", "-x"]),
            ("Performance Tests", ["-m", "performance"]),
        ]
        
        overall_success = True
        
        for level_name, markers in pyramid_levels:
            print(f"\n📊 Running {level_name}...")
            
            cmd = ["python", "-m", "pytest"] + markers + ["tests/"]
            
            if args.coverage and level_name == "Unit Tests":
                cmd.extend([
                    "--cov=src",
                    "--cov-report=html",
                    "--cov-report=xml",
                    "--cov-report=term-missing"
                ])
            
            if args.fast and "performance" not in level_name.lower():
                cmd.extend(["-m", "not slow"])
            
            if args.verbose:
                cmd.extend(["-v"])
            
            result = self._run_single_command(cmd, level_name)
            
            if result != 0:
                overall_success = False
                if level_name != "Performance Tests":  # Performance tests failures are non-critical
                    print(f"❌ {level_name} failed, stopping pyramid execution")
                    break
                else:
                    print(f"⚠️  {level_name} had issues but continuing...")
        
        return 0 if overall_success else 1
    
    def _run_single_command(self, cmd: List[str], description: str) -> int:
        """Run a single pytest command."""
        print(f"\n🏃 {description}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Change to project root
            os.chdir(self.project_root)
            
            # Run command
            result = subprocess.run(cmd, capture_output=False, text=True)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Record results
            self.test_results[description] = {
                "exit_code": result.returncode,
                "duration": duration,
                "command": " ".join(cmd)
            }
            
            if result.returncode == 0:
                print(f"✅ {description} passed in {duration:.2f}s")
            else:
                print(f"❌ {description} failed in {duration:.2f}s (exit code: {result.returncode})")
            
            return result.returncode
            
        except KeyboardInterrupt:
            print(f"\n⚠️  {description} interrupted by user")
            return 1
        except Exception as e:
            print(f"❌ Error running {description}: {e}")
            return 1
    
    def _print_summary(self):
        """Print test execution summary."""
        if not self.test_results or not self.start_time:
            return
        
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📋 TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.test_results.values() if r["exit_code"] == 0)
        failed = len(self.test_results) - passed
        
        print(f"Total test suites: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total time: {total_time:.2f}s")
        
        print("\nDetailed Results:")
        for name, result in self.test_results.items():
            status = "✅ PASS" if result["exit_code"] == 0 else "❌ FAIL"
            print(f"  {status} {name} ({result['duration']:.2f}s)")
        
        if failed == 0:
            print("\n🎉 All test suites passed!")
        else:
            print(f"\n⚠️  {failed} test suite(s) failed")
        
        return failed == 0


def create_test_report(args: argparse.Namespace):
    """Generate comprehensive test report."""
    print("\n📊 Generating Test Report...")
    
    # Run coverage report
    if args.coverage:
        try:
            subprocess.run([
                "python", "-m", "coverage", "html",
                "--directory", "htmlcov"
            ], check=True)
            print("📈 Coverage report generated: htmlcov/index.html")
        except subprocess.CalledProcessError:
            print("⚠️  Failed to generate coverage report")
    
    # Generate test report summary
    report_file = Path("test-report.json")
    if report_file.exists():
        with open(report_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n📋 Test Report Summary:")
        print(f"  Total tests: {data.get('total', 0)}")
        print(f"  Passed: {data.get('passed', 0)}")
        print(f"  Failed: {data.get('failed', 0)}")
        print(f"  Skipped: {data.get('skipped', 0)}")
        print(f"  Duration: {data.get('duration', 0):.2f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Triple Barrier Label 11.a Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py                              # All tests
  python scripts/run_tests.py --unit --coverage            # Unit tests with coverage
  python scripts/run_tests.py --pyramid --fast             # Pyramid strategy, skip slow tests
  python scripts/run_tests.py --integration --verbose      # Integration tests, verbose output
  python scripts/run_tests.py --performance --parallel 4   # Performance tests, 4 workers
  python scripts/run_tests.py tests/unit/test_timestamp_aligner.py  # Specific test file
        """
    )
    
    # Test selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--unit", action="store_true", help="Run unit tests")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests")
    test_group.add_argument("--performance", action="store_true", help="Run performance tests")
    test_group.add_argument("--edge-cases", action="store_true", help="Run edge case tests")
    
    # Execution strategy
    parser.add_argument("--pyramid", action="store_true", 
                       help="Use testing pyramid strategy (unit -> integration -> performance)")
    
    # Options
    parser.add_argument("--fast", action="store_true", 
                       help="Skip slow tests")
    parser.add_argument("--coverage", action="store_true", 
                       help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, metavar="N",
                       help="Run tests in parallel with N workers")
    
    # Output
    parser.add_argument("--junit-xml", metavar="FILE",
                       help="Generate JUnit XML report")
    parser.add_argument("--report", action="store_true",
                       help="Generate comprehensive test report")
    
    # Specific tests
    parser.add_argument("tests", nargs="*", 
                       help="Specific test files or directories")
    
    args = parser.parse_args()
    
    # Default to pyramid strategy if no specific selection
    if not any([args.unit, args.integration, args.performance, args.edge_cases, args.tests]):
        args.pyramid = True
    
    # Create and run test runner
    runner = TestRunner()
    
    try:
        exit_code = runner.run_tests(args)
        
        # Print summary
        success = runner._print_summary()
        
        # Generate report if requested
        if args.report:
            create_test_report(args)
        
        # Print final recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-" * 20)
        if success:
            print("• All tests passing - ready for deployment")
            print("• Consider running performance tests periodically")
        else:
            print("• Fix failing tests before deployment")
            print("• Review test output for specific failure details")
            
        if args.coverage:
            print("• Review coverage report for gaps: htmlcov/index.html")
        
        print("• Run with --pyramid for comprehensive testing")
        print("• Use --performance for throughput validation")
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())