"""
Comprehensive Load Testing Suite for Floodingnaque API.

This module provides a complete load testing suite with multiple test types:
- Smoke Test: Quick sanity check (10 users, 1 min)
- Load Test: Normal expected load (100 users, 10 min)
- Stress Test: Find breaking point (ramp to 500+ users)
- Spike Test: Sudden traffic surge simulation
- Endurance Test: Memory leak detection (20 users, 1 hour)

Features:
- Realistic data patterns with geographically appropriate weather data
- Performance assertions (CI/CD integration)
- HTML report generation
- Both authenticated and public endpoint testing

Usage:
    # Run specific test type
    python -m tests.load.load_test_suite --test-type smoke
    python -m tests.load.load_test_suite --test-type load
    python -m tests.load.load_test_suite --test-type stress
    python -m tests.load.load_test_suite --test-type spike
    python -m tests.load.load_test_suite --test-type endurance
    
    # Run all tests
    python -m tests.load.load_test_suite --test-type all
    
    # With custom host
    python -m tests.load.load_test_suite --test-type load --host http://api.example.com
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TestConfig:
    """Configuration for a load test."""
    name: str
    description: str
    users: int
    spawn_rate: int
    duration: str
    shape_class: Optional[str] = None
    tags: Optional[List[str]] = None
    performance_thresholds: Optional[Dict[str, float]] = None


# =============================================================================
# TEST CONFIGURATIONS
# =============================================================================

TEST_CONFIGS: Dict[str, TestConfig] = {
    "smoke": TestConfig(
        name="Smoke Test",
        description="Quick sanity check with minimal load",
        users=10,
        spawn_rate=5,
        duration="1m",
        shape_class="SmokeTestShape",
        tags=["health", "lightweight"],
        performance_thresholds={
            "avg_response_time_ms": 500,
            "p99_response_time_ms": 2000,
            "failure_rate_percent": 1.0,
        }
    ),
    "load": TestConfig(
        name="Load Test",
        description="Normal expected load simulation",
        users=100,
        spawn_rate=10,
        duration="10m",
        shape_class="StepLoadShape",
        performance_thresholds={
            "avg_response_time_ms": 500,
            "p99_response_time_ms": 2000,
            "failure_rate_percent": 1.0,
            "requests_per_second_min": 50,
        }
    ),
    "stress": TestConfig(
        name="Stress Test",
        description="Find system breaking point",
        users=500,
        spawn_rate=25,
        duration="15m",
        performance_thresholds={
            "avg_response_time_ms": 2000,
            "p99_response_time_ms": 5000,
            "failure_rate_percent": 5.0,
        }
    ),
    "spike": TestConfig(
        name="Spike Test",
        description="Sudden traffic surge simulation",
        users=100,
        spawn_rate=50,
        duration="2m",
        shape_class="SpikeLoadShape",
        performance_thresholds={
            "avg_response_time_ms": 1000,
            "p99_response_time_ms": 3000,
            "failure_rate_percent": 2.0,
        }
    ),
    "endurance": TestConfig(
        name="Endurance Test",
        description="Long-running test for memory leak detection",
        users=20,
        spawn_rate=2,
        duration="1h",
        shape_class="EnduranceLoadShape",
        performance_thresholds={
            "avg_response_time_ms": 500,
            "failure_rate_percent": 0.5,
        }
    ),
}


# =============================================================================
# PERFORMANCE ASSERTIONS
# =============================================================================

@dataclass
class TestResult:
    """Result of a load test run."""
    test_name: str
    passed: bool
    total_requests: int
    failed_requests: int
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    failure_rate: float
    threshold_violations: List[str]
    duration_seconds: float
    timestamp: str


def parse_locust_stats(stats_file: Path) -> Optional[Dict]:
    """Parse Locust stats JSON output."""
    if not stats_file.exists():
        return None
    
    try:
        with open(stats_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def check_thresholds(
    stats: Dict,
    thresholds: Dict[str, float]
) -> Tuple[bool, List[str]]:
    """
    Check performance metrics against thresholds.
    
    Returns:
        Tuple of (all_passed, list_of_violations)
    """
    violations = []
    
    # Get aggregated stats
    total_stats = None
    for entry in stats.get("stats", []):
        if entry.get("name") == "Aggregated":
            total_stats = entry
            break
    
    if not total_stats:
        return False, ["Could not find aggregated stats"]
    
    # Check average response time
    if "avg_response_time_ms" in thresholds:
        avg_rt = total_stats.get("avg_response_time", 0)
        if avg_rt > thresholds["avg_response_time_ms"]:
            violations.append(
                f"Average response time {avg_rt:.0f}ms > "
                f"threshold {thresholds['avg_response_time_ms']}ms"
            )
    
    # Check p99 response time
    if "p99_response_time_ms" in thresholds:
        # Locust stores percentiles differently
        percentiles = total_stats.get("response_time_percentiles", {})
        p99 = percentiles.get("0.99", total_stats.get("max_response_time", 0))
        if p99 > thresholds["p99_response_time_ms"]:
            violations.append(
                f"P99 response time {p99:.0f}ms > "
                f"threshold {thresholds['p99_response_time_ms']}ms"
            )
    
    # Check failure rate
    if "failure_rate_percent" in thresholds:
        total_reqs = total_stats.get("num_requests", 1)
        failed_reqs = total_stats.get("num_failures", 0)
        failure_rate = (failed_reqs / total_reqs) * 100 if total_reqs > 0 else 0
        if failure_rate > thresholds["failure_rate_percent"]:
            violations.append(
                f"Failure rate {failure_rate:.2f}% > "
                f"threshold {thresholds['failure_rate_percent']}%"
            )
    
    # Check minimum RPS
    if "requests_per_second_min" in thresholds:
        rps = total_stats.get("current_rps", 0)
        if rps < thresholds["requests_per_second_min"]:
            violations.append(
                f"RPS {rps:.1f} < minimum threshold {thresholds['requests_per_second_min']}"
            )
    
    return len(violations) == 0, violations


# =============================================================================
# TEST RUNNER
# =============================================================================

def get_locustfile_path() -> Path:
    """Get the path to the locustfile."""
    return Path(__file__).parent / "locustfile.py"


def run_load_test(
    config: TestConfig,
    host: str,
    output_dir: Path,
    verbose: bool = False
) -> TestResult:
    """
    Run a single load test.
    
    Args:
        config: Test configuration
        host: Target host URL
        output_dir: Directory for output files
        verbose: Enable verbose output
        
    Returns:
        TestResult object
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_id = f"{config.name.lower().replace(' ', '_')}_{timestamp}"
    
    # Output files
    html_report = output_dir / f"report_{test_id}.html"
    stats_file = output_dir / f"stats_{test_id}.json"
    
    # Build locust command
    locustfile = get_locustfile_path()
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(locustfile),
        "--host", host,
        "--headless",
        "-u", str(config.users),
        "-r", str(config.spawn_rate),
        "-t", config.duration,
        "--html", str(html_report),
        "--json",
    ]
    
    # Add shape class if specified
    if config.shape_class:
        # Note: Shape classes override user/spawn_rate settings
        pass  # Locust will use the shape class tick() method
    
    # Add tags if specified
    if config.tags:
        for tag in config.tags:
            cmd.extend(["--tags", tag])
    
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"Description: {config.description}")
    print(f"Users: {config.users}, Duration: {config.duration}")
    print(f"Target: {host}")
    print(f"{'='*60}\n")
    
    # Run locust
    start_time = datetime.now()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            cwd=str(Path(__file__).parent.parent.parent)
        )
        
        # Capture JSON output
        if result.stdout and not verbose:
            try:
                # Locust outputs JSON to stdout with --json flag
                stats = json.loads(result.stdout.split('\n')[-2])  # Last line before empty
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
            except (json.JSONDecodeError, IndexError):
                pass
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Locust failed with exit code {e.returncode}")
        if e.stderr:
            print(e.stderr)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Parse results
    stats = parse_locust_stats(stats_file)
    
    if stats:
        passed, violations = check_thresholds(
            stats,
            config.performance_thresholds or {}
        )
        
        total_stats = None
        for entry in stats.get("stats", []):
            if entry.get("name") == "Aggregated":
                total_stats = entry
                break
        
        if total_stats:
            total_reqs = total_stats.get("num_requests", 0)
            failed_reqs = total_stats.get("num_failures", 0)
            
            return TestResult(
                test_name=config.name,
                passed=passed,
                total_requests=total_reqs,
                failed_requests=failed_reqs,
                avg_response_time=total_stats.get("avg_response_time", 0),
                p50_response_time=total_stats.get("median_response_time", 0),
                p95_response_time=total_stats.get("response_time_percentile_95", 0),
                p99_response_time=total_stats.get("response_time_percentile_99", 0),
                requests_per_second=total_stats.get("current_rps", 0),
                failure_rate=(failed_reqs / total_reqs * 100) if total_reqs > 0 else 0,
                threshold_violations=violations,
                duration_seconds=duration,
                timestamp=timestamp
            )
    
    # Return a failed result if we couldn't parse stats
    return TestResult(
        test_name=config.name,
        passed=False,
        total_requests=0,
        failed_requests=0,
        avg_response_time=0,
        p50_response_time=0,
        p95_response_time=0,
        p99_response_time=0,
        requests_per_second=0,
        failure_rate=0,
        threshold_violations=["Failed to parse test results"],
        duration_seconds=duration,
        timestamp=timestamp
    )


def print_result_summary(result: TestResult) -> None:
    """Print a formatted test result summary."""
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    
    print(f"\n{'='*60}")
    print(f"{result.test_name}: {status}")
    print(f"{'='*60}")
    print(f"Total Requests:     {result.total_requests:,}")
    print(f"Failed Requests:    {result.failed_requests:,}")
    print(f"Failure Rate:       {result.failure_rate:.2f}%")
    print(f"Avg Response Time:  {result.avg_response_time:.0f}ms")
    print(f"P50 Response Time:  {result.p50_response_time:.0f}ms")
    print(f"P95 Response Time:  {result.p95_response_time:.0f}ms")
    print(f"P99 Response Time:  {result.p99_response_time:.0f}ms")
    print(f"Requests/sec:       {result.requests_per_second:.1f}")
    print(f"Duration:           {result.duration_seconds:.0f}s")
    
    if result.threshold_violations:
        print(f"\nThreshold Violations:")
        for violation in result.threshold_violations:
            print(f"  ❌ {violation}")
    
    print(f"{'='*60}\n")


def generate_summary_report(
    results: List[TestResult],
    output_dir: Path
) -> None:
    """Generate a summary report for all test results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"summary_report_{timestamp}.json"
    
    summary = {
        "timestamp": timestamp,
        "total_tests": len(results),
        "passed_tests": sum(1 for r in results if r.passed),
        "failed_tests": sum(1 for r in results if not r.passed),
        "tests": [
            {
                "name": r.test_name,
                "passed": r.passed,
                "total_requests": r.total_requests,
                "failure_rate": r.failure_rate,
                "avg_response_time_ms": r.avg_response_time,
                "p99_response_time_ms": r.p99_response_time,
                "violations": r.threshold_violations,
            }
            for r in results
        ]
    }
    
    with open(report_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary report saved to: {report_file}")
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests:  {summary['total_tests']}")
    print(f"Passed:       {summary['passed_tests']}")
    print(f"Failed:       {summary['failed_tests']}")
    
    if summary['failed_tests'] > 0:
        print(f"\nFailed Tests:")
        for r in results:
            if not r.passed:
                print(f"  ❌ {r.test_name}")
                for v in r.threshold_violations:
                    print(f"      - {v}")
    
    print(f"{'='*60}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Load Testing Suite for Floodingnaque API"
    )
    parser.add_argument(
        "--test-type",
        choices=["smoke", "load", "stress", "spike", "endurance", "all"],
        default="smoke",
        help="Type of load test to run"
    )
    parser.add_argument(
        "--host",
        default="http://localhost:5000",
        help="Target host URL"
    )
    parser.add_argument(
        "--output-dir",
        default="reports/load_tests",
        help="Directory for output files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit with non-zero code if tests fail"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which tests to run
    if args.test_type == "all":
        test_names = ["smoke", "load", "stress", "spike"]  # Skip endurance for 'all'
    else:
        test_names = [args.test_type]
    
    # Run tests
    results = []
    for test_name in test_names:
        config = TEST_CONFIGS[test_name]
        result = run_load_test(config, args.host, output_dir, args.verbose)
        results.append(result)
        print_result_summary(result)
    
    # Generate summary report
    if len(results) > 1:
        generate_summary_report(results, output_dir)
    
    # Exit with appropriate code for CI
    if args.ci:
        failed_tests = sum(1 for r in results if not r.passed)
        if failed_tests > 0:
            print(f"\n❌ CI FAILURE: {failed_tests} test(s) failed performance thresholds")
            sys.exit(1)
        else:
            print(f"\n✅ CI SUCCESS: All tests passed performance thresholds")
            sys.exit(0)


if __name__ == "__main__":
    main()
