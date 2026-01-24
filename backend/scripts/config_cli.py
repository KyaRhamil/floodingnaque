#!/usr/bin/env python
"""
Floodingnaque Configuration Management CLI
==========================================

Command-line tools for managing configuration encryption, diffing, and validation.

Commands:
    encrypt     - Encrypt a sensitive value for use in config files
    decrypt     - Decrypt an ENC[] wrapped value
    genkey      - Generate a new encryption key
    diff        - Compare configurations between environments
    validate    - Validate feature references in configuration
    resources   - Display detected system resources

Usage:
    python -m scripts.config_cli encrypt "my-secret-value"
    python -m scripts.config_cli genkey
    python -m scripts.config_cli diff development production
    python -m scripts.config_cli validate
    python -m scripts.config_cli resources

Environment Variables:
    FLOODINGNAQUE_CONFIG_KEY - Base64-encoded encryption key
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_encrypt(args):
    """Encrypt a value."""
    from config.encryption import ConfigEncryption, KeyNotFoundError

    try:
        encryptor = ConfigEncryption()
        encrypted = encryptor.encrypt(args.value)
        print(f"\nEncrypted value:\n{encrypted}\n")
        print("Copy this value into your config file to store it securely.")
    except KeyNotFoundError:
        print("Error: No encryption key configured.", file=sys.stderr)
        print("Set FLOODINGNAQUE_CONFIG_KEY environment variable or use 'genkey' to create one.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_decrypt(args):
    """Decrypt a value."""
    from config.encryption import ConfigEncryption, DecryptionError, KeyNotFoundError

    try:
        encryptor = ConfigEncryption()
        decrypted = encryptor.decrypt(args.value)
        print(f"\nDecrypted value:\n{decrypted}\n")
    except KeyNotFoundError:
        print("Error: No encryption key configured.", file=sys.stderr)
        return 1
    except DecryptionError as e:
        print(f"Error: Decryption failed - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_genkey(args):
    """Generate a new encryption key."""
    from config.encryption import generate_key

    key = generate_key()
    print("\nGenerated encryption key (base64-encoded):")
    print(f"\n{key}\n")
    print("Store this securely! Options:")
    print("  1. Set as environment variable: FLOODINGNAQUE_CONFIG_KEY")
    print("  2. Save to a file and set FLOODINGNAQUE_CONFIG_KEY_FILE")
    print("\nExample (PowerShell):")
    print(f'  $env:FLOODINGNAQUE_CONFIG_KEY = "{key}"')
    print("\nExample (Bash):")
    print(f'  export FLOODINGNAQUE_CONFIG_KEY="{key}"')
    return 0


def cmd_diff(args):
    """Compare configurations between environments."""
    from scripts.config_diff import ConfigDiffer, format_text_output, load_config_for_env

    try:
        config1, env1 = load_config_for_env(args.env1)
        config2, env2 = load_config_for_env(args.env2)

        differ = ConfigDiffer(
            ignore_paths=set(args.ignore) if args.ignore else set(),
            include_defaults=not args.no_defaults,
            show_values=not args.hide_values,
        )

        result = differ.compare(config1, config2, env1, env2)

        if args.format == "json":
            print(result.to_json())
        else:
            print(format_text_output(result))

        return 1 if result.has_differences else 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_validate(args):
    """Validate feature references."""
    from config import get_config
    from config.feature_validation import FeatureValidator

    config = get_config()
    validator = FeatureValidator(strict_mode=args.strict)

    result = validator.validate_config(config._config)
    print(str(result))

    if result.valid:
        print("\n✓ All feature references are valid")
        return 0
    else:
        return 1


def cmd_resources(args):
    """Display detected system resources."""
    from config.resource_detection import print_system_info

    print_system_info()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Floodingnaque Configuration Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # encrypt command
    encrypt_parser = subparsers.add_parser("encrypt", help="Encrypt a sensitive value")
    encrypt_parser.add_argument("value", help="Value to encrypt")
    encrypt_parser.set_defaults(func=cmd_encrypt)

    # decrypt command
    decrypt_parser = subparsers.add_parser("decrypt", help="Decrypt an encrypted value")
    decrypt_parser.add_argument("value", help="ENC[...] value to decrypt")
    decrypt_parser.set_defaults(func=cmd_decrypt)

    # genkey command
    genkey_parser = subparsers.add_parser("genkey", help="Generate new encryption key")
    genkey_parser.set_defaults(func=cmd_genkey)

    # diff command
    diff_parser = subparsers.add_parser("diff", help="Compare configurations")
    diff_parser.add_argument("env1", help="First environment")
    diff_parser.add_argument("env2", help="Second environment")
    diff_parser.add_argument("--format", "-f", choices=["text", "json"], default="text")
    diff_parser.add_argument("--ignore", "-i", action="append", help="Paths to ignore")
    diff_parser.add_argument("--no-defaults", action="store_true", help="Don't use default ignores")
    diff_parser.add_argument("--hide-values", action="store_true", help="Hide actual values")
    diff_parser.set_defaults(func=cmd_diff)

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate feature references")
    validate_parser.add_argument("--strict", action="store_true", help="Fail on any issue")
    validate_parser.set_defaults(func=cmd_validate)

    # resources command
    resources_parser = subparsers.add_parser("resources", help="Show system resources")
    resources_parser.set_defaults(func=cmd_resources)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
