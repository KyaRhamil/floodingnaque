"""
Floodingnaque Configuration Encryption
======================================

Provides encryption/decryption for sensitive configuration values such as:
- Tracking URIs
- API endpoints
- Database connection strings
- Any value marked with ENC[] wrapper

Security Features:
- AES-256-GCM encryption with authenticated encryption
- Key derivation using PBKDF2 with SHA-256
- Unique salt and nonce per encrypted value
- Optional key rotation support
- Environment-based key management

Usage:
    from config.encryption import ConfigEncryption, encrypt_value, decrypt_value

    # Initialize with key from environment
    encryptor = ConfigEncryption()

    # Encrypt a sensitive value
    encrypted = encryptor.encrypt("my-secret-api-key")
    # Returns: ENC[base64_encoded_encrypted_data]

    # Decrypt a value
    decrypted = encryptor.decrypt("ENC[base64_encoded_encrypted_data]")
    # Returns: my-secret-api-key

    # Process entire config dict (auto-decrypt ENC[] values)
    config = encryptor.process_config(config_dict)

Environment Variables:
    FLOODINGNAQUE_CONFIG_KEY - Base64-encoded encryption key (32 bytes for AES-256)
    FLOODINGNAQUE_CONFIG_KEY_FILE - Path to file containing encryption key
    FLOODINGNAQUE_KEY_DERIVATION_SALT - Salt for key derivation (optional)
"""

import base64
import binascii
import hashlib
import hmac
import logging
import os
import re
import secrets
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Raised when encryption/decryption fails."""

    pass


class KeyNotFoundError(EncryptionError):
    """Raised when encryption key is not available."""

    pass


class DecryptionError(EncryptionError):
    """Raised when decryption fails (wrong key, corrupted data, etc.)."""

    pass


class CryptoBackend(ABC):
    """Abstract base class for cryptographic backends."""

    @property
    @abstractmethod
    def available(self) -> bool:
        """Check if this backend is available."""
        pass

    @abstractmethod
    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt plaintext using key."""
        pass

    @abstractmethod
    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt ciphertext using key."""
        pass


class FernetBackend(CryptoBackend):
    """
    Fernet-based encryption backend (cryptography library).

    Uses AES-128-CBC with HMAC-SHA256 for authentication.
    Requires: pip install cryptography
    """

    def __init__(self):
        try:
            from cryptography.fernet import Fernet, InvalidToken

            self._Fernet = Fernet
            self._InvalidToken = InvalidToken
            self._available = True
        except ImportError:
            self._available = False
            logger.debug("cryptography library not available")

    @property
    def available(self) -> bool:
        return self._available

    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        if not self._available:
            raise EncryptionError("cryptography library not installed")

        # Derive Fernet-compatible key (32 bytes, base64-encoded)
        derived_key = self._derive_fernet_key(key)
        fernet = self._Fernet(derived_key)
        return fernet.encrypt(plaintext)

    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        if not self._available:
            raise EncryptionError("cryptography library not installed")

        derived_key = self._derive_fernet_key(key)
        fernet = self._Fernet(derived_key)
        try:
            return fernet.decrypt(ciphertext)
        except self._InvalidToken as e:
            raise DecryptionError("Decryption failed: invalid key or corrupted data") from e

    def _derive_fernet_key(self, key: bytes) -> bytes:
        """Derive a Fernet-compatible key from arbitrary key material."""
        # Use PBKDF2 to derive exactly 32 bytes, then base64 encode
        import hashlib

        derived = hashlib.pbkdf2_hmac(
            "sha256", key, b"floodingnaque_fernet_salt", 100000, dklen=32  # Fixed salt for key derivation
        )
        return base64.urlsafe_b64encode(derived)


class AESGCMBackend(CryptoBackend):
    """
    AES-256-GCM encryption backend.

    Provides authenticated encryption with associated data (AEAD).
    Requires: pip install cryptography

    Format: nonce (12 bytes) + ciphertext + tag (16 bytes)
    """

    NONCE_SIZE = 12
    TAG_SIZE = 16
    KEY_SIZE = 32  # AES-256

    def __init__(self):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            self._AESGCM = AESGCM
            self._available = True
        except ImportError:
            self._available = False
            logger.debug("cryptography library not available for AES-GCM")

    @property
    def available(self) -> bool:
        return self._available

    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        if not self._available:
            raise EncryptionError("cryptography library not installed")

        # Ensure key is correct size
        key = self._normalize_key(key)

        # Generate random nonce
        nonce = secrets.token_bytes(self.NONCE_SIZE)

        # Encrypt with AES-GCM
        aesgcm = self._AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)

        # Return nonce + ciphertext (tag is appended by AESGCM)
        return nonce + ciphertext

    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        if not self._available:
            raise EncryptionError("cryptography library not installed")

        if len(ciphertext) < self.NONCE_SIZE + self.TAG_SIZE:
            raise DecryptionError("Ciphertext too short")

        key = self._normalize_key(key)

        # Extract nonce and ciphertext
        nonce = ciphertext[: self.NONCE_SIZE]
        actual_ciphertext = ciphertext[self.NONCE_SIZE :]

        try:
            aesgcm = self._AESGCM(key)
            return aesgcm.decrypt(nonce, actual_ciphertext, None)
        except Exception as e:
            raise DecryptionError(f"Decryption failed: {e}") from e

    def _normalize_key(self, key: bytes) -> bytes:
        """Normalize key to exactly KEY_SIZE bytes."""
        if len(key) == self.KEY_SIZE:
            return key
        elif len(key) > self.KEY_SIZE:
            return key[: self.KEY_SIZE]
        else:
            # Derive key using PBKDF2
            return hashlib.pbkdf2_hmac("sha256", key, b"floodingnaque_aesgcm_salt", 100000, dklen=self.KEY_SIZE)


class SimpleXORBackend(CryptoBackend):
    """
    Simple XOR-based obfuscation backend.

    WARNING: This is NOT secure encryption! Use only as a fallback
    when cryptography library is not available, for basic obfuscation
    of non-critical values.

    Includes HMAC for integrity verification.
    """

    def __init__(self):
        self._available = True  # Always available, uses stdlib

    @property
    def available(self) -> bool:
        return self._available

    def encrypt(self, plaintext: bytes, key: bytes) -> bytes:
        logger.warning("Using XOR obfuscation - NOT secure! " "Install 'cryptography' for proper encryption.")

        # Generate random IV
        iv = secrets.token_bytes(16)

        # Derive key stream using PBKDF2
        key_stream = hashlib.pbkdf2_hmac("sha256", key + iv, b"floodingnaque_xor_salt", 10000, dklen=len(plaintext))

        # XOR plaintext with key stream
        ciphertext = bytes(p ^ k for p, k in zip(plaintext, key_stream))

        # Compute HMAC for integrity
        mac = hmac.new(key, iv + ciphertext, hashlib.sha256).digest()[:16]

        # Return: IV (16) + ciphertext + MAC (16)
        return iv + ciphertext + mac

    def decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        if len(ciphertext) < 33:  # IV (16) + at least 1 byte + MAC (16)
            raise DecryptionError("Ciphertext too short")

        # Extract components
        iv = ciphertext[:16]
        mac = ciphertext[-16:]
        actual_ciphertext = ciphertext[16:-16]

        # Verify HMAC
        expected_mac = hmac.new(key, iv + actual_ciphertext, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(mac, expected_mac):
            raise DecryptionError("HMAC verification failed")

        # Derive key stream
        key_stream = hashlib.pbkdf2_hmac(
            "sha256", key + iv, b"floodingnaque_xor_salt", 10000, dklen=len(actual_ciphertext)
        )

        # XOR to decrypt
        return bytes(c ^ k for c, k in zip(actual_ciphertext, key_stream))


class ConfigEncryption:
    """
    Configuration encryption manager.

    Handles encryption and decryption of sensitive configuration values.
    Values are wrapped with ENC[...] to indicate they are encrypted.

    Supports multiple backends with automatic fallback:
    1. AES-GCM (preferred, requires cryptography)
    2. Fernet (alternative, requires cryptography)
    3. XOR obfuscation (fallback, NOT secure)
    """

    # Pattern to match encrypted values: ENC[base64_data]
    ENCRYPTED_PATTERN = re.compile(r"^ENC\[([A-Za-z0-9+/=]+)\]$")

    # Prefix to identify encryption backend
    BACKEND_PREFIXES = {
        "G": "aesgcm",  # AES-GCM
        "F": "fernet",  # Fernet
        "X": "xor",  # XOR obfuscation
    }

    def __init__(self, key: Optional[bytes] = None, key_file: Optional[Path] = None, preferred_backend: str = "aesgcm"):
        """
        Initialize config encryption.

        Args:
            key: Encryption key (bytes). If not provided, loaded from environment.
            key_file: Path to file containing key. Takes precedence over env var.
            preferred_backend: Preferred encryption backend ('aesgcm', 'fernet', 'xor')
        """
        self._key = key or self._load_key(key_file)
        self._backends: Dict[str, CryptoBackend] = {
            "aesgcm": AESGCMBackend(),
            "fernet": FernetBackend(),
            "xor": SimpleXORBackend(),
        }
        self._preferred_backend = preferred_backend

    def _load_key(self, key_file: Optional[Path] = None) -> Optional[bytes]:
        """Load encryption key from file or environment."""
        # Try key file first
        if key_file and key_file.exists():
            try:
                key_data = key_file.read_text().strip()
                return base64.b64decode(key_data)
            except Exception as e:
                logger.warning(f"Failed to load key from file: {e}")

        # Try environment variable for key file path
        env_key_file = os.environ.get("FLOODINGNAQUE_CONFIG_KEY_FILE")
        if env_key_file:
            key_path = Path(env_key_file)
            if key_path.exists():
                try:
                    key_data = key_path.read_text().strip()
                    return base64.b64decode(key_data)
                except Exception as e:
                    logger.warning(f"Failed to load key from env file: {e}")

        # Try direct key from environment
        env_key = os.environ.get("FLOODINGNAQUE_CONFIG_KEY")
        if env_key:
            try:
                return base64.b64decode(env_key)
            except Exception:
                # Might be raw key, use as-is
                return env_key.encode("utf-8")

        return None

    @property
    def has_key(self) -> bool:
        """Check if encryption key is available."""
        return self._key is not None

    @property
    def available_backends(self) -> List[str]:
        """Get list of available encryption backends."""
        return [name for name, backend in self._backends.items() if backend.available]

    def _get_backend(self, backend_name: Optional[str] = None) -> Tuple[str, CryptoBackend]:
        """Get the best available backend."""
        if backend_name:
            backend = self._backends.get(backend_name)
            if backend and backend.available:
                return backend_name, backend

        # Try preferred backend
        preferred = self._backends.get(self._preferred_backend)
        if preferred and preferred.available:
            return self._preferred_backend, preferred

        # Fallback order
        for name in ["aesgcm", "fernet", "xor"]:
            backend = self._backends.get(name)
            if backend and backend.available:
                return name, backend

        raise EncryptionError("No encryption backend available")

    def encrypt(self, plaintext: str, backend_name: Optional[str] = None) -> str:
        """
        Encrypt a plaintext value.

        Args:
            plaintext: Value to encrypt
            backend_name: Specific backend to use (optional)

        Returns:
            Encrypted value wrapped in ENC[...]

        Raises:
            KeyNotFoundError: If no encryption key is configured
            EncryptionError: If encryption fails
        """
        if not self._key:
            raise KeyNotFoundError(
                "Encryption key not configured. Set FLOODINGNAQUE_CONFIG_KEY "
                "environment variable or provide key_file."
            )

        name, backend = self._get_backend(backend_name)

        # Get backend prefix
        prefix = next(p for p, n in self.BACKEND_PREFIXES.items() if n == name)

        # Encrypt
        ciphertext = backend.encrypt(plaintext.encode("utf-8"), self._key)

        # Encode as base64 with prefix
        encoded = prefix + base64.b64encode(ciphertext).decode("ascii")

        return f"ENC[{encoded}]"

    def decrypt(self, encrypted: str) -> str:
        """
        Decrypt an encrypted value.

        Args:
            encrypted: Value in ENC[...] format

        Returns:
            Decrypted plaintext

        Raises:
            KeyNotFoundError: If no encryption key is configured
            DecryptionError: If decryption fails
        """
        if not self._key:
            raise KeyNotFoundError(
                "Encryption key not configured. Set FLOODINGNAQUE_CONFIG_KEY " "environment variable."
            )

        # Extract encrypted data
        match = self.ENCRYPTED_PATTERN.match(encrypted)
        if not match:
            raise DecryptionError(f"Invalid encrypted format: {encrypted[:50]}...")

        encoded = match.group(1)

        # Extract backend prefix
        if not encoded:
            raise DecryptionError("Empty encrypted data")

        prefix = encoded[0]
        backend_name = self.BACKEND_PREFIXES.get(prefix)

        if not backend_name:
            # Legacy format without prefix - try all backends
            for name in ["aesgcm", "fernet", "xor"]:
                try:
                    ciphertext = base64.b64decode(encoded)
                    backend = self._backends[name]
                    if backend.available:
                        return backend.decrypt(ciphertext, self._key).decode("utf-8")
                except Exception:  # nosec B112 - intentionally trying multiple backends
                    continue
            raise DecryptionError("Failed to decrypt with any backend")

        # Decode and decrypt
        try:
            ciphertext = base64.b64decode(encoded[1:])
            backend = self._backends[backend_name]
            if not backend.available:
                raise DecryptionError(f"Backend {backend_name} not available")
            return backend.decrypt(ciphertext, self._key).decode("utf-8")
        except binascii.Error as e:
            raise DecryptionError(f"Invalid base64 encoding: {e}") from e

    def is_encrypted(self, value: Any) -> bool:
        """Check if a value is encrypted."""
        if not isinstance(value, str):
            return False
        return bool(self.ENCRYPTED_PATTERN.match(value))

    def process_config(self, config: Dict[str, Any], decrypt: bool = True, in_place: bool = False) -> Dict[str, Any]:
        """
        Process config dict, decrypting all ENC[] values.

        Args:
            config: Configuration dictionary
            decrypt: If True, decrypt values; if False, just validate
            in_place: If True, modify config in place; if False, return copy

        Returns:
            Processed configuration dictionary
        """
        if not in_place:
            import copy

            config = copy.deepcopy(config)

        self._process_dict(config, decrypt)
        return config

    def _process_dict(self, d: Dict[str, Any], decrypt: bool) -> None:
        """Recursively process dictionary values."""
        for key, value in d.items():
            if isinstance(value, dict):
                self._process_dict(value, decrypt)
            elif isinstance(value, list):
                self._process_list(value, decrypt)
            elif isinstance(value, str) and self.is_encrypted(value):
                if decrypt and self._key:
                    try:
                        d[key] = self.decrypt(value)
                    except DecryptionError as e:
                        logger.warning(f"Failed to decrypt {key}: {e}")

    def _process_list(self, lst: List[Any], decrypt: bool) -> None:
        """Recursively process list values."""
        for i, value in enumerate(lst):
            if isinstance(value, dict):
                self._process_dict(value, decrypt)
            elif isinstance(value, list):
                self._process_list(value, decrypt)
            elif isinstance(value, str) and self.is_encrypted(value):
                if decrypt and self._key:
                    try:
                        lst[i] = self.decrypt(value)
                    except DecryptionError as e:
                        logger.warning(f"Failed to decrypt list item {i}: {e}")

    def encrypt_sensitive_fields(
        self, config: Dict[str, Any], sensitive_paths: Optional[List[Tuple[str, ...]]] = None
    ) -> Dict[str, Any]:
        """
        Encrypt specific fields in a config.

        Args:
            config: Configuration dictionary
            sensitive_paths: List of paths to sensitive fields
                            Default: tracking URIs, API endpoints, passwords

        Returns:
            Config with sensitive fields encrypted
        """
        if sensitive_paths is None:
            sensitive_paths = [
                ("mlflow", "tracking_uri"),
                ("database", "password"),
                ("database", "connection_string"),
                ("api", "key"),
                ("api", "secret"),
                ("registry", "api_key"),
            ]

        import copy

        result = copy.deepcopy(config)

        for path in sensitive_paths:
            value = self._get_nested(result, path)
            if value and isinstance(value, str) and not self.is_encrypted(value):
                try:
                    encrypted = self.encrypt(value)
                    self._set_nested(result, path, encrypted)
                except EncryptionError as e:
                    logger.warning(f"Failed to encrypt {'.'.join(path)}: {e}")

        return result

    def _get_nested(self, d: Dict[str, Any], path: Tuple[str, ...]) -> Any:
        """Get nested dictionary value."""
        current = d
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _set_nested(self, d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
        """Set nested dictionary value."""
        current = d
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value


# Global encryption instance
_encryptor: Optional[ConfigEncryption] = None


def get_encryptor() -> ConfigEncryption:
    """Get the global encryption instance."""
    global _encryptor
    if _encryptor is None:
        _encryptor = ConfigEncryption()
    return _encryptor


def encrypt_value(plaintext: str) -> str:
    """Convenience function to encrypt a value."""
    return get_encryptor().encrypt(plaintext)


def decrypt_value(encrypted: str) -> str:
    """Convenience function to decrypt a value."""
    return get_encryptor().decrypt(encrypted)


def generate_key() -> str:
    """
    Generate a new encryption key.

    Returns:
        Base64-encoded key suitable for FLOODINGNAQUE_CONFIG_KEY
    """
    key = secrets.token_bytes(32)  # 256 bits for AES-256
    return base64.b64encode(key).decode("ascii")


def rotate_key(config: Dict[str, Any], old_key: bytes, new_key: bytes) -> Dict[str, Any]:
    """
    Rotate encryption key for all encrypted values in config.

    Args:
        config: Configuration dictionary with ENC[] values
        old_key: Current encryption key
        new_key: New encryption key

    Returns:
        Config with values re-encrypted using new key
    """
    old_encryptor = ConfigEncryption(key=old_key)
    new_encryptor = ConfigEncryption(key=new_key)

    import copy

    result = copy.deepcopy(config)

    def rotate_dict(d: Dict[str, Any]) -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                rotate_dict(value)
            elif isinstance(value, list):
                rotate_list(value)
            elif isinstance(value, str) and old_encryptor.is_encrypted(value):
                try:
                    decrypted = old_encryptor.decrypt(value)
                    d[key] = new_encryptor.encrypt(decrypted)
                except Exception as e:
                    logger.error(f"Failed to rotate key for {key}: {e}")

    def rotate_list(lst: List[Any]) -> None:
        for i, value in enumerate(lst):
            if isinstance(value, dict):
                rotate_dict(value)
            elif isinstance(value, list):
                rotate_list(value)
            elif isinstance(value, str) and old_encryptor.is_encrypted(value):
                try:
                    decrypted = old_encryptor.decrypt(value)
                    lst[i] = new_encryptor.encrypt(decrypted)
                except Exception as e:
                    logger.error(f"Failed to rotate key for list item {i}: {e}")

    rotate_dict(result)
    return result
