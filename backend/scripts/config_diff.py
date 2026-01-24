#!/usr/bin/env python
"""
Floodingnaque Configuration Diff Tool
=====================================

Compare configuration files between environments to identify differences.

Features:
- Side-by-side comparison of YAML configs
- Highlights added, removed, and changed values
- Supports nested comparison
- Output formats: text, JSON, HTML
- Ignores environment-specific expected differences

Usage:
    # Compare development and production
    python -m scripts.config_diff development production

    # Compare specific config files
    python -m scripts.config_diff --file1 config/dev.yaml --file2 config/prod.yaml

    # Output as JSON
    python -m scripts.config_diff development staging --format json

    # Ignore specific paths
    python -m scripts.config_diff dev prod --ignore mlflow.tracking_uri --ignore logging.file.path

Examples:
    python scripts/config_diff.py development production
    python scripts/config_diff.py staging production --format json
    python scripts/config_diff.py dev prod --show-values
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


class DiffType(str, Enum):
    """Type of configuration difference."""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    TYPE_CHANGED = "type_changed"


@dataclass
class ConfigDiff:
    """Represents a difference between two config values."""

    path: str
    diff_type: DiffType
    value1: Any = None
    value2: Any = None

    def __str__(self) -> str:
        if self.diff_type == DiffType.ADDED:
            return f"+ {self.path}: {self.value2}"
        elif self.diff_type == DiffType.REMOVED:
            return f"- {self.path}: {self.value1}"
        elif self.diff_type == DiffType.CHANGED:
            return f"~ {self.path}: {self.value1} -> {self.value2}"
        elif self.diff_type == DiffType.TYPE_CHANGED:
            type1 = type(self.value1).__name__
            type2 = type(self.value2).__name__
            return f"! {self.path}: type changed {type1} -> {type2}"
        return f"? {self.path}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "path": self.path,
            "type": self.diff_type.value,
        }
        if self.value1 is not None:
            result["old_value"] = self.value1
        if self.value2 is not None:
            result["new_value"] = self.value2
        return result


@dataclass
class DiffResult:
    """Result of comparing two configurations."""

    env1: str
    env2: str
    diffs: List[ConfigDiff] = field(default_factory=list)
    ignored_paths: Set[str] = field(default_factory=set)

    @property
    def has_differences(self) -> bool:
        return len(self.diffs) > 0

    @property
    def added_count(self) -> int:
        return sum(1 for d in self.diffs if d.diff_type == DiffType.ADDED)

    @property
    def removed_count(self) -> int:
        return sum(1 for d in self.diffs if d.diff_type == DiffType.REMOVED)

    @property
    def changed_count(self) -> int:
        return sum(1 for d in self.diffs if d.diff_type in (DiffType.CHANGED, DiffType.TYPE_CHANGED))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "env1": self.env1,
            "env2": self.env2,
            "summary": {
                "total_differences": len(self.diffs),
                "added": self.added_count,
                "removed": self.removed_count,
                "changed": self.changed_count,
            },
            "differences": [d.to_dict() for d in self.diffs],
            "ignored_paths": list(self.ignored_paths),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


class ConfigDiffer:
    """
    Configuration comparison tool.

    Compares two configuration dictionaries and identifies differences.
    """

    # Paths that are expected to differ between environments
    DEFAULT_IGNORE_PATHS = {
        "mlflow.tracking_uri",
        "mlflow.experiment_name",
        "logging.file.path",
        "registry.models_dir",
        "data.processed_dir",
        "data.raw_dir",
    }

    def __init__(
        self, ignore_paths: Optional[Set[str]] = None, include_defaults: bool = True, show_values: bool = True
    ):
        """
        Initialize config differ.

        Args:
            ignore_paths: Paths to ignore in comparison
            include_defaults: Include default ignore paths
            show_values: Include actual values in diff output
        """
        self.ignore_paths = ignore_paths or set()
        if include_defaults:
            self.ignore_paths.update(self.DEFAULT_IGNORE_PATHS)
        self.show_values = show_values

    def compare(
        self, config1: Dict[str, Any], config2: Dict[str, Any], env1: str = "config1", env2: str = "config2"
    ) -> DiffResult:
        """
        Compare two configuration dictionaries.

        Args:
            config1: First configuration
            config2: Second configuration
            env1: Name/label for first config
            env2: Name/label for second config

        Returns:
            DiffResult with all differences
        """
        result = DiffResult(env1=env1, env2=env2, ignored_paths=self.ignore_paths.copy())

        self._compare_dicts(config1, config2, "", result)

        # Sort diffs by path for consistent output
        result.diffs.sort(key=lambda d: d.path)

        return result

    def _compare_dicts(self, d1: Dict[str, Any], d2: Dict[str, Any], path_prefix: str, result: DiffResult) -> None:
        """Recursively compare dictionaries."""
        all_keys = set(d1.keys()) | set(d2.keys())

        for key in all_keys:
            path = f"{path_prefix}.{key}" if path_prefix else key

            # Check if path should be ignored
            if self._should_ignore(path):
                continue

            in_d1 = key in d1
            in_d2 = key in d2

            if in_d1 and not in_d2:
                # Key removed in config2
                value = d1[key] if self.show_values else "[hidden]"
                result.diffs.append(ConfigDiff(path=path, diff_type=DiffType.REMOVED, value1=value))
            elif in_d2 and not in_d1:
                # Key added in config2
                value = d2[key] if self.show_values else "[hidden]"
                result.diffs.append(ConfigDiff(path=path, diff_type=DiffType.ADDED, value2=value))
            else:
                # Key exists in both
                v1, v2 = d1[key], d2[key]
                self._compare_values(v1, v2, path, result)

    def _compare_values(self, v1: Any, v2: Any, path: str, result: DiffResult) -> None:
        """Compare two values at a given path."""
        # Type comparison
        if type(v1) is not type(v2):
            # Allow int/float comparison
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if v1 != v2:
                    result.diffs.append(
                        ConfigDiff(
                            path=path,
                            diff_type=DiffType.CHANGED,
                            value1=v1 if self.show_values else "[hidden]",
                            value2=v2 if self.show_values else "[hidden]",
                        )
                    )
            else:
                result.diffs.append(
                    ConfigDiff(
                        path=path,
                        diff_type=DiffType.TYPE_CHANGED,
                        value1=v1 if self.show_values else type(v1).__name__,
                        value2=v2 if self.show_values else type(v2).__name__,
                    )
                )
            return

        # Recursive dict comparison
        if isinstance(v1, dict):
            self._compare_dicts(v1, v2, path, result)
            return

        # List comparison
        if isinstance(v1, list):
            self._compare_lists(v1, v2, path, result)
            return

        # Value comparison
        if v1 != v2:
            result.diffs.append(
                ConfigDiff(
                    path=path,
                    diff_type=DiffType.CHANGED,
                    value1=v1 if self.show_values else "[hidden]",
                    value2=v2 if self.show_values else "[hidden]",
                )
            )

    def _compare_lists(self, l1: List[Any], l2: List[Any], path: str, result: DiffResult) -> None:
        """Compare two lists."""
        # For simple lists, compare as sets for order-independent comparison
        # For complex lists (dicts), compare element by element

        if not l1 and not l2:
            return

        # Check if lists contain simple types
        has_complex_items = any(isinstance(item, (dict, list)) for item in l1 + l2)

        if has_complex_items:
            # Element-by-element comparison
            max_len = max(len(l1), len(l2))
            for i in range(max_len):
                item_path = f"{path}[{i}]"

                if i >= len(l1):
                    result.diffs.append(
                        ConfigDiff(
                            path=item_path, diff_type=DiffType.ADDED, value2=l2[i] if self.show_values else "[hidden]"
                        )
                    )
                elif i >= len(l2):
                    result.diffs.append(
                        ConfigDiff(
                            path=item_path, diff_type=DiffType.REMOVED, value1=l1[i] if self.show_values else "[hidden]"
                        )
                    )
                else:
                    self._compare_values(l1[i], l2[i], item_path, result)
        else:
            # Set-based comparison for simple lists
            set1 = set(str(x) for x in l1)
            set2 = set(str(x) for x in l2)

            if set1 != set2:
                result.diffs.append(
                    ConfigDiff(
                        path=path,
                        diff_type=DiffType.CHANGED,
                        value1=l1 if self.show_values else f"[{len(l1)} items]",
                        value2=l2 if self.show_values else f"[{len(l2)} items]",
                    )
                )

    def _should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        for ignore in self.ignore_paths:
            if path == ignore or path.startswith(f"{ignore}."):
                return True
            # Support wildcards
            if "*" in ignore:
                import fnmatch

                if fnmatch.fnmatch(path, ignore):
                    return True
        return False


def load_config_for_env(environment: str) -> Tuple[Dict[str, Any], str]:
    """
    Load configuration for a specific environment.

    Args:
        environment: Environment name (development, staging, production)

    Returns:
        Tuple of (config_dict, environment_name)
    """
    config_dir = Path(__file__).parent.parent / "config"
    base_config = config_dir / "training_config.yaml"
    env_config = config_dir / f"{environment}.yaml"

    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")

    # Load base config
    with open(base_config) as f:
        config = yaml.safe_load(f)

    # Merge environment config if exists
    if env_config.exists():
        with open(env_config) as f:
            env_overrides = yaml.safe_load(f) or {}

        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        config = deep_merge(config, env_overrides)

    return config, environment


def format_text_output(result: DiffResult) -> str:
    """Format diff result as colored text output."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append(f"Configuration Diff: {result.env1} vs {result.env2}")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    lines.append(f"Total differences: {len(result.diffs)}")
    lines.append(f"  Added:   {result.added_count}")
    lines.append(f"  Removed: {result.removed_count}")
    lines.append(f"  Changed: {result.changed_count}")
    lines.append("")

    if result.ignored_paths:
        lines.append(f"Ignored paths: {', '.join(sorted(result.ignored_paths))}")
        lines.append("")

    if not result.has_differences:
        lines.append("✓ Configurations are identical (excluding ignored paths)")
    else:
        lines.append("-" * 70)
        lines.append("Differences:")
        lines.append("-" * 70)

        # Group by type
        for diff_type in [DiffType.REMOVED, DiffType.ADDED, DiffType.CHANGED, DiffType.TYPE_CHANGED]:
            type_diffs = [d for d in result.diffs if d.diff_type == diff_type]
            if type_diffs:
                lines.append("")
                lines.append(f"  [{diff_type.value.upper()}]")
                for diff in type_diffs:
                    lines.append(f"    {diff}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def format_html_output(result: DiffResult) -> str:
    """Format diff result as HTML table."""
    html = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<style>",
        "body { font-family: monospace; margin: 20px; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #4CAF50; color: white; }",
        ".added { background-color: #90EE90; }",
        ".removed { background-color: #FFB6C1; }",
        ".changed { background-color: #FFE4B5; }",
        ".type-changed { background-color: #DDA0DD; }",
        "h1 { color: #333; }",
        ".summary { margin: 20px 0; padding: 10px; background: #f5f5f5; }",
        "</style>",
        "</head><body>",
        f"<h1>Config Diff: {result.env1} vs {result.env2}</h1>",
        "<div class='summary'>",
        f"<p>Total: {len(result.diffs)} | Added: {result.added_count} | "
        f"Removed: {result.removed_count} | Changed: {result.changed_count}</p>",
        "</div>",
        "<table>",
        "<tr><th>Type</th><th>Path</th><th>Old Value</th><th>New Value</th></tr>",
    ]

    for diff in result.diffs:
        css_class = diff.diff_type.value.replace("_", "-")
        old_val = str(diff.value1) if diff.value1 is not None else "-"
        new_val = str(diff.value2) if diff.value2 is not None else "-"
        html.append(
            f"<tr class='{css_class}'>"
            f"<td>{diff.diff_type.value}</td>"
            f"<td>{diff.path}</td>"
            f"<td>{old_val}</td>"
            f"<td>{new_val}</td>"
            "</tr>"
        )

    html.extend(["</table>", "</body></html>"])
    return "\n".join(html)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Floodingnaque configuration between environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s development production
  %(prog)s staging production --format json
  %(prog)s dev prod --ignore mlflow.* --show-values
  %(prog)s --file1 old_config.yaml --file2 new_config.yaml
        """,
    )

    parser.add_argument("env1", nargs="?", help="First environment (development, staging, production)")
    parser.add_argument("env2", nargs="?", help="Second environment to compare against")
    parser.add_argument("--file1", type=Path, help="First YAML file to compare (instead of environment)")
    parser.add_argument("--file2", type=Path, help="Second YAML file to compare (instead of environment)")
    parser.add_argument(
        "--format", "-f", choices=["text", "json", "html"], default="text", help="Output format (default: text)"
    )
    parser.add_argument(
        "--ignore", "-i", action="append", default=[], help="Paths to ignore (can be specified multiple times)"
    )
    parser.add_argument("--no-defaults", action="store_true", help="Don't use default ignore paths")
    parser.add_argument(
        "--show-values", action="store_true", default=True, help="Show actual values in diff (default: True)"
    )
    parser.add_argument("--hide-values", action="store_true", help="Hide actual values in diff (for security)")
    parser.add_argument("--output", "-o", type=Path, help="Output file (default: stdout)")

    args = parser.parse_args()

    # Validate arguments
    if args.file1 and args.file2:
        # Compare two files directly
        with open(args.file1) as f:
            config1 = yaml.safe_load(f)
        with open(args.file2) as f:
            config2 = yaml.safe_load(f)
        env1 = args.file1.name
        env2 = args.file2.name
    elif args.env1 and args.env2:
        # Compare environments
        config1, env1 = load_config_for_env(args.env1)
        config2, env2 = load_config_for_env(args.env2)
    else:
        parser.error("Provide either two environments or --file1 and --file2")

    # Setup differ
    ignore_paths = set(args.ignore)
    show_values = args.show_values and not args.hide_values

    differ = ConfigDiffer(ignore_paths=ignore_paths, include_defaults=not args.no_defaults, show_values=show_values)

    # Compare
    result = differ.compare(config1, config2, env1, env2)

    # Format output
    if args.format == "json":
        output = result.to_json()
    elif args.format == "html":
        output = format_html_output(result)
    else:
        output = format_text_output(result)

    # Write output
    if args.output:
        args.output.write_text(output)
        print(f"Output written to {args.output}")
    else:
        print(output)

    # Return non-zero exit code if differences found
    sys.exit(1 if result.has_differences else 0)


if __name__ == "__main__":
    main()
