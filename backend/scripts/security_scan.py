#!/usr/bin/env python3
"""
Security Scanning Script for Floodingnaque.

Performs comprehensive security scans including:
- pip-audit for known CVE vulnerabilities
- bandit for Python security linting
- safety for dependency vulnerability checking

Usage:
    # Run all scans
    python scripts/security_scan.py

    # Run specific scan type
    python scripts/security_scan.py --scan-type pip-audit
    python scripts/security_scan.py --scan-type bandit
    python scripts/security_scan.py --scan-type safety

    # Generate HTML report
    python scripts/security_scan.py --output-format html --output-dir reports/security

    # CI mode (exit with non-zero on critical issues)
    python scripts/security_scan.py --ci --fail-on critical
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SeverityLevel(str, Enum):
    """Severity levels for vulnerabilities."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class Vulnerability:
    """Represents a security vulnerability."""
    id: str
    package: str
    installed_version: str
    fixed_version: Optional[str]
    description: str
    severity: SeverityLevel
    source: str  # pip-audit, bandit, safety


@dataclass
class ScanResult:
    """Result of a security scan."""
    scanner: str
    passed: bool
    vulnerabilities: List[Vulnerability]
    raw_output: str
    duration_seconds: float


@dataclass
class SecurityReport:
    """Aggregate security scan report."""
    timestamp: str
    total_vulnerabilities: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    scans: List[ScanResult]
    
    @property
    def has_critical(self) -> bool:
        return self.critical_count > 0
    
    @property
    def has_high(self) -> bool:
        return self.high_count > 0


def check_tool_installed(tool: str) -> bool:
    """Check if a security tool is installed."""
    try:
        subprocess.run(
            [sys.executable, "-m", tool, "--help"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_pip_audit(requirements_file: Path) -> ScanResult:
    """
    Run pip-audit to check for known CVEs in dependencies.
    
    pip-audit queries the Python Package Index (PyPI) and the 
    Open Source Vulnerability (OSV) database.
    """
    import time
    start = time.time()
    vulnerabilities = []
    raw_output = ""
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip_audit",
                "-r", str(requirements_file),
                "--format", "json",
                "--desc", "on",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        raw_output = result.stdout + result.stderr
        
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                for dep in data.get("dependencies", []):
                    for vuln in dep.get("vulns", []):
                        # Map severity (pip-audit doesn't always provide severity)
                        severity = SeverityLevel.UNKNOWN
                        aliases = vuln.get("aliases", [])
                        for alias in aliases:
                            if alias.startswith("CVE-"):
                                # Could query NVD for actual severity
                                severity = SeverityLevel.HIGH  # Default assumption
                                break
                        
                        vulnerabilities.append(Vulnerability(
                            id=vuln.get("id", "Unknown"),
                            package=dep.get("name", "Unknown"),
                            installed_version=dep.get("version", "Unknown"),
                            fixed_version=vuln.get("fix_versions", [None])[0] if vuln.get("fix_versions") else None,
                            description=vuln.get("description", "No description available"),
                            severity=severity,
                            source="pip-audit",
                        ))
            except json.JSONDecodeError:
                pass
        
        passed = result.returncode == 0
        
    except subprocess.TimeoutExpired:
        raw_output = "pip-audit timed out after 5 minutes"
        passed = False
    except Exception as e:
        raw_output = f"pip-audit error: {e}"
        passed = False
    
    duration = time.time() - start
    
    return ScanResult(
        scanner="pip-audit",
        passed=passed,
        vulnerabilities=vulnerabilities,
        raw_output=raw_output,
        duration_seconds=duration,
    )


def run_bandit(source_dir: Path) -> ScanResult:
    """
    Run bandit for Python security linting.
    
    Bandit checks for common security issues in Python code:
    - Hardcoded passwords
    - SQL injection risks
    - Shell injection risks
    - Insecure crypto usage
    """
    import time
    start = time.time()
    vulnerabilities = []
    raw_output = ""
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "bandit",
                "-r", str(source_dir),
                "-f", "json",
                "-ll",  # Only report medium and higher
                "--exclude", ".venv,venv,__pycache__,tests",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        raw_output = result.stdout + result.stderr
        
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                for issue in data.get("results", []):
                    # Map bandit severity
                    severity_map = {
                        "HIGH": SeverityLevel.HIGH,
                        "MEDIUM": SeverityLevel.MEDIUM,
                        "LOW": SeverityLevel.LOW,
                    }
                    severity = severity_map.get(
                        issue.get("issue_severity", "UNKNOWN"),
                        SeverityLevel.UNKNOWN
                    )
                    
                    vulnerabilities.append(Vulnerability(
                        id=issue.get("test_id", "Unknown"),
                        package=issue.get("filename", "Unknown"),
                        installed_version="N/A",
                        fixed_version=None,
                        description=f"{issue.get('issue_text', 'No description')} (line {issue.get('line_number', '?')})",
                        severity=severity,
                        source="bandit",
                    ))
            except json.JSONDecodeError:
                pass
        
        # Bandit returns non-zero if issues found, not just on error
        passed = len(vulnerabilities) == 0
        
    except subprocess.TimeoutExpired:
        raw_output = "bandit timed out after 5 minutes"
        passed = False
    except Exception as e:
        raw_output = f"bandit error: {e}"
        passed = False
    
    duration = time.time() - start
    
    return ScanResult(
        scanner="bandit",
        passed=passed,
        vulnerabilities=vulnerabilities,
        raw_output=raw_output,
        duration_seconds=duration,
    )


def run_safety(requirements_file: Path) -> ScanResult:
    """
    Run safety to check dependencies against known vulnerability database.
    """
    import time
    start = time.time()
    vulnerabilities = []
    raw_output = ""
    
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "safety",
                "check",
                "-r", str(requirements_file),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        raw_output = result.stdout + result.stderr
        
        if result.stdout:
            try:
                # Safety 3.x has different JSON format
                data = json.loads(result.stdout)
                
                # Handle different safety output formats
                vuln_list = []
                if isinstance(data, list):
                    vuln_list = data
                elif isinstance(data, dict):
                    vuln_list = data.get("vulnerabilities", [])
                
                for vuln in vuln_list:
                    if isinstance(vuln, dict):
                        vulnerabilities.append(Vulnerability(
                            id=str(vuln.get("vulnerability_id", vuln.get("id", "Unknown"))),
                            package=vuln.get("package_name", vuln.get("package", "Unknown")),
                            installed_version=vuln.get("analyzed_version", vuln.get("installed_version", "Unknown")),
                            fixed_version=vuln.get("fixed_versions", [None])[0] if vuln.get("fixed_versions") else None,
                            description=vuln.get("advisory", vuln.get("description", "No description")),
                            severity=SeverityLevel.HIGH,  # Safety doesn't provide severity
                            source="safety",
                        ))
            except json.JSONDecodeError:
                pass
        
        passed = result.returncode == 0
        
    except subprocess.TimeoutExpired:
        raw_output = "safety timed out after 5 minutes"
        passed = False
    except Exception as e:
        raw_output = f"safety error: {e}"
        passed = False
    
    duration = time.time() - start
    
    return ScanResult(
        scanner="safety",
        passed=passed,
        vulnerabilities=vulnerabilities,
        raw_output=raw_output,
        duration_seconds=duration,
    )


def generate_report(scans: List[ScanResult]) -> SecurityReport:
    """Generate aggregate security report from scan results."""
    all_vulns = []
    for scan in scans:
        all_vulns.extend(scan.vulnerabilities)
    
    return SecurityReport(
        timestamp=datetime.now().isoformat(),
        total_vulnerabilities=len(all_vulns),
        critical_count=sum(1 for v in all_vulns if v.severity == SeverityLevel.CRITICAL),
        high_count=sum(1 for v in all_vulns if v.severity == SeverityLevel.HIGH),
        medium_count=sum(1 for v in all_vulns if v.severity == SeverityLevel.MEDIUM),
        low_count=sum(1 for v in all_vulns if v.severity == SeverityLevel.LOW),
        scans=scans,
    )


def print_report(report: SecurityReport) -> None:
    """Print formatted security report to console."""
    print("\n" + "=" * 70)
    print("SECURITY SCAN REPORT")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print("-" * 70)
    
    for scan in report.scans:
        status = "✅ PASSED" if scan.passed else "❌ FAILED"
        print(f"\n{scan.scanner.upper()}: {status}")
        print(f"  Duration: {scan.duration_seconds:.1f}s")
        print(f"  Vulnerabilities: {len(scan.vulnerabilities)}")
        
        if scan.vulnerabilities:
            for vuln in scan.vulnerabilities[:10]:  # Limit output
                print(f"    - [{vuln.severity.value.upper()}] {vuln.package}: {vuln.id}")
                print(f"      {vuln.description[:80]}...")
            
            if len(scan.vulnerabilities) > 10:
                print(f"    ... and {len(scan.vulnerabilities) - 10} more")
    
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Total Vulnerabilities: {report.total_vulnerabilities}")
    print(f"  Critical: {report.critical_count}")
    print(f"  High:     {report.high_count}")
    print(f"  Medium:   {report.medium_count}")
    print(f"  Low:      {report.low_count}")
    print("=" * 70)
    
    if report.has_critical:
        print("\n⚠️  CRITICAL vulnerabilities found - immediate action required!")
    elif report.has_high:
        print("\n⚠️  HIGH severity vulnerabilities found - review and remediate.")
    elif report.total_vulnerabilities > 0:
        print("\nℹ️  Some vulnerabilities found - consider reviewing.")
    else:
        print("\n✅ No vulnerabilities detected!")


def save_json_report(report: SecurityReport, output_path: Path) -> None:
    """Save report as JSON file."""
    data = {
        "timestamp": report.timestamp,
        "summary": {
            "total": report.total_vulnerabilities,
            "critical": report.critical_count,
            "high": report.high_count,
            "medium": report.medium_count,
            "low": report.low_count,
        },
        "scans": [
            {
                "scanner": scan.scanner,
                "passed": scan.passed,
                "duration_seconds": scan.duration_seconds,
                "vulnerability_count": len(scan.vulnerabilities),
                "vulnerabilities": [
                    {
                        "id": v.id,
                        "package": v.package,
                        "installed_version": v.installed_version,
                        "fixed_version": v.fixed_version,
                        "description": v.description,
                        "severity": v.severity.value,
                    }
                    for v in scan.vulnerabilities
                ]
            }
            for scan in report.scans
        ]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nJSON report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Security scanning for Floodingnaque dependencies and code"
    )
    parser.add_argument(
        "--scan-type",
        choices=["all", "pip-audit", "bandit", "safety"],
        default="all",
        help="Type of security scan to run"
    )
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="Path to requirements file"
    )
    parser.add_argument(
        "--source-dir",
        default="app",
        help="Source directory for code scanning"
    )
    parser.add_argument(
        "--output-dir",
        default="reports/security",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit with non-zero code on failures"
    )
    parser.add_argument(
        "--fail-on",
        choices=["critical", "high", "medium", "low", "any"],
        default="critical",
        help="Severity level that causes CI failure"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    backend_dir = Path(__file__).parent.parent
    requirements_file = backend_dir / args.requirements
    source_dir = backend_dir / args.source_dir
    output_dir = backend_dir / args.output_dir
    
    # Validate paths
    if not requirements_file.exists():
        print(f"Error: Requirements file not found: {requirements_file}")
        sys.exit(1)
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    # Run scans
    scans = []
    scan_types = ["pip-audit", "bandit", "safety"] if args.scan_type == "all" else [args.scan_type]
    
    for scan_type in scan_types:
        print(f"\n🔍 Running {scan_type}...")
        
        if scan_type == "pip-audit":
            scans.append(run_pip_audit(requirements_file))
        elif scan_type == "bandit":
            scans.append(run_bandit(source_dir))
        elif scan_type == "safety":
            scans.append(run_safety(requirements_file))
    
    # Generate and print report
    report = generate_report(scans)
    print_report(report)
    
    # Save JSON report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json_report(report, output_dir / f"security_report_{timestamp}.json")
    
    # CI exit code
    if args.ci:
        fail_levels = {
            "any": [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM, SeverityLevel.LOW],
            "low": [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM, SeverityLevel.LOW],
            "medium": [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM],
            "high": [SeverityLevel.CRITICAL, SeverityLevel.HIGH],
            "critical": [SeverityLevel.CRITICAL],
        }
        
        levels_to_check = fail_levels[args.fail_on]
        
        for scan in scans:
            for vuln in scan.vulnerabilities:
                if vuln.severity in levels_to_check:
                    print(f"\n❌ CI FAILURE: Found {args.fail_on} or higher severity vulnerabilities")
                    sys.exit(1)
        
        print(f"\n✅ CI SUCCESS: No {args.fail_on} or higher severity vulnerabilities found")
        sys.exit(0)


if __name__ == "__main__":
    main()
