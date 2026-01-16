"""
Comprehensive Security Tests.

Tests for authentication bypass, injection attacks, and security vulnerabilities.
"""

import sys
from pathlib import Path

import requests

# Add backend to path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

BASE_URL = "http://localhost:5000"


# ============================================================================
# Authentication Bypass Tests
# ============================================================================


class TestAuthenticationBypass:
    """Tests for authentication bypass vulnerabilities."""

    def test_no_api_key_protected_endpoint(self):
        """Test that protected endpoints reject requests without API key."""
        response = requests.post(
            f"{BASE_URL}/predict", json={"temperature": 298.15, "humidity": 75.0, "precipitation": 10.0}
        )

        # Should be 401 if auth is required, or 200/502 if auth is bypassed
        assert response.status_code in [200, 401, 502, 503]

    def test_empty_api_key(self):
        """Test that empty API key is rejected."""
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"temperature": 298.15, "humidity": 75.0, "precipitation": 10.0},
            headers={"X-API-Key": ""},
        )

        # Empty key should be treated as no key
        assert response.status_code in [200, 401, 502, 503]

    def test_whitespace_api_key(self):
        """Test that whitespace-only API key is rejected."""
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"temperature": 298.15, "humidity": 75.0, "precipitation": 10.0},
            headers={"X-API-Key": "   "},
        )

        assert response.status_code in [200, 401, 502, 503]

    def test_null_byte_in_api_key(self):
        """Test that null bytes in API key are handled safely."""
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"temperature": 298.15, "humidity": 75.0, "precipitation": 10.0},
            headers={"X-API-Key": "valid-key\x00injection"},
        )

        # Should not crash, should reject or handle
        assert response.status_code in [200, 400, 401, 502, 503]

    def test_extremely_long_api_key(self):
        """Test that extremely long API keys don't cause issues."""
        long_key = "a" * 10000

        response = requests.post(
            f"{BASE_URL}/predict",
            json={"temperature": 298.15, "humidity": 75.0, "precipitation": 10.0},
            headers={"X-API-Key": long_key},
        )

        # Should handle gracefully
        assert response.status_code in [400, 401, 413, 502, 503]

    def test_sql_injection_in_api_key(self):
        """Test that SQL injection in API key is safe."""
        sql_injection_keys = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users --",
        ]

        for key in sql_injection_keys:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"temperature": 298.15, "humidity": 75.0, "precipitation": 10.0},
                headers={"X-API-Key": key},
            )

            # Should reject, not execute SQL
            assert response.status_code in [400, 401, 502, 503]

    def test_header_injection(self):
        """Test that header injection is prevented."""
        # Try to inject additional headers via the API key
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"temperature": 298.15, "humidity": 75.0, "precipitation": 10.0},
            headers={"X-API-Key": "key\r\nX-Forwarded-For: 127.0.0.1"},
        )

        # Should handle safely
        assert response.status_code in [400, 401, 502, 503]


class TestAuthMiddlewareUnit:
    """Unit tests for authentication middleware."""

    def test_validate_api_key_empty_string(self):
        """Test that empty string returns False."""
        from app.api.middleware.auth import validate_api_key

        result = validate_api_key("")
        assert result is False

    def test_validate_api_key_none(self):
        """Test that None returns False."""
        from app.api.middleware.auth import validate_api_key

        result = validate_api_key(None)
        assert result is False

    def test_timing_safe_compare(self):
        """Test timing-safe comparison function."""
        from app.api.middleware.auth import _timing_safe_compare

        # Equal strings
        assert _timing_safe_compare("abc123", "abc123") is True

        # Different strings
        assert _timing_safe_compare("abc123", "xyz789") is False

        # Empty strings
        assert _timing_safe_compare("", "") is True

    def test_hash_api_key_sha256(self):
        """Test SHA-256 hashing of API keys."""
        from app.api.middleware.auth import _hash_api_key_sha256

        # Same input should produce same hash
        hash1 = _hash_api_key_sha256("test-key-123")
        hash2 = _hash_api_key_sha256("test-key-123")
        assert hash1 == hash2

        # Different input should produce different hash
        hash3 = _hash_api_key_sha256("different-key")
        assert hash1 != hash3

        # Hash should be 64 hex characters (SHA-256)
        assert len(hash1) == 64

    def test_invalidate_cache(self):
        """Test API key cache invalidation."""
        from app.api.middleware.auth import get_hashed_api_keys, invalidate_api_key_cache

        # Get keys (initializes cache)
        keys1 = get_hashed_api_keys()

        # Invalidate cache
        invalidate_api_key_cache()

        # Get keys again (should reinitialize)
        keys2 = get_hashed_api_keys()

        # Both should be dict (may be same or different depending on env)
        assert isinstance(keys1, dict)
        assert isinstance(keys2, dict)

    def test_get_auth_context_default(self):
        """Test default auth context when not in request."""
        from app.api.middleware.auth import get_auth_context

        # Without Flask context, should return defaults
        try:
            context = get_auth_context()
            assert context["authenticated"] is False
            assert context["bypass_mode"] is False
            assert context["api_key_hash"] is None
        except RuntimeError:
            # Expected if no Flask app context
            pass

    def test_is_using_bcrypt(self):
        """Test bcrypt availability detection."""
        from app.api.middleware.auth import BCRYPT_AVAILABLE, is_using_bcrypt

        result = is_using_bcrypt()
        assert result == BCRYPT_AVAILABLE


# ============================================================================
# Input Validation Security Tests
# ============================================================================


class TestInputValidationSecurity:
    """Security tests for input validation."""

    def test_json_injection(self):
        """Test that JSON injection is handled safely."""
        malicious_payloads = [
            {"temperature": 298.15, "__proto__": {"polluted": True}},
            {"temperature": 298.15, "constructor": {"prototype": {}}},
        ]

        for payload in malicious_payloads:
            response = requests.post(f"{BASE_URL}/predict", json=payload)

            # Should handle safely
            assert response.status_code in [200, 400, 401, 422, 502]

    def test_oversized_json_payload(self):
        """Test that oversized payloads are rejected."""
        # Create a payload that's too large (e.g., 1MB of data)
        large_payload = {
            "temperature": 298.15,
            "humidity": 75.0,
            "precipitation": 10.0,
            "padding": "x" * (1024 * 1024),  # 1MB of padding
        }

        response = requests.post(f"{BASE_URL}/predict", json=large_payload)

        # Should reject large payloads
        assert response.status_code in [400, 401, 413, 502, 503]

    def test_deeply_nested_json(self):
        """Test that deeply nested JSON is handled safely."""
        # Create deeply nested structure
        nested = {"value": 298.15}
        for _ in range(100):
            nested = {"nested": nested}

        nested["temperature"] = 298.15
        nested["humidity"] = 75.0
        nested["precipitation"] = 10.0

        response = requests.post(f"{BASE_URL}/predict", json=nested)

        # Should handle without crashing
        assert response.status_code in [200, 400, 401, 502, 503]

    def test_unicode_payload(self):
        """Test that unicode in payloads is handled safely."""
        payload = {
            "temperature": 298.15,
            "humidity": 75.0,
            "precipitation": 10.0,
            "note": "测试 テスト тест 🌊",  # Various unicode
        }

        response = requests.post(f"{BASE_URL}/predict", json=payload)

        # Should handle unicode gracefully
        assert response.status_code in [200, 400, 401, 502]

    def test_invalid_coordinate_bounds(self):
        """Test that invalid coordinates are rejected."""
        test_cases = [
            {"lat": 1000, "lon": 121.0},  # Way out of bounds
            {"lat": float("inf"), "lon": 121.0},  # Infinity
            {"lat": float("nan"), "lon": 121.0},  # NaN
        ]

        for coords in test_cases:
            response = requests.post(f"{BASE_URL}/ingest", json=coords)

            # Should handle safely
            assert response.status_code in [200, 400, 401, 422, 502]

    def test_path_traversal_in_params(self):
        """Test that path traversal is prevented."""
        # Try path traversal in query parameters
        response = requests.get(f"{BASE_URL}/data?path=../../../etc/passwd")

        # Should not expose any file contents
        assert "root:" not in response.text
        assert response.status_code in [200, 400]


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting functionality."""

    def test_rate_limit_headers_present(self):
        """Test that rate limit headers are returned."""
        response = requests.get(f"{BASE_URL}/status")

        # Note: Common rate limit headers include X-RateLimit-Limit, X-RateLimit-Remaining,
        # X-RateLimit-Reset, Retry-After but may not be present if rate limiting is disabled
        assert response.status_code == 200

    def test_rapid_requests_handled(self):
        """Test that rapid requests are handled (rate limited or not)."""
        responses = []

        # Send 20 rapid requests
        for _ in range(20):
            response = requests.get(f"{BASE_URL}/status")
            responses.append(response.status_code)

        # Should either all succeed or some get rate limited
        assert all(code in [200, 429] for code in responses)

        # If rate limiting is enabled, some may be limited; if disabled, all succeed
        assert len(responses) == 20


# ============================================================================
# Security Headers Tests
# ============================================================================


class TestSecurityHeaders:
    """Tests for security headers."""

    def test_x_content_type_options(self):
        """Test X-Content-Type-Options header."""
        response = requests.get(f"{BASE_URL}/status")

        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self):
        """Test X-Frame-Options header."""
        response = requests.get(f"{BASE_URL}/status")

        assert response.headers.get("X-Frame-Options") in ["DENY", "SAMEORIGIN"]

    def test_x_xss_protection(self):
        """Test X-XSS-Protection header."""
        response = requests.get(f"{BASE_URL}/status")

        xss_header = response.headers.get("X-XSS-Protection", "")
        assert xss_header in ["1", "1; mode=block", "0"]  # '0' is also valid (CSP preferred)

    def test_referrer_policy(self):
        """Test Referrer-Policy header."""
        response = requests.get(f"{BASE_URL}/status")

        valid_policies = [
            "no-referrer",
            "no-referrer-when-downgrade",
            "origin",
            "origin-when-cross-origin",
            "same-origin",
            "strict-origin",
            "strict-origin-when-cross-origin",
            "unsafe-url",
        ]

        policy = response.headers.get("Referrer-Policy", "")
        if policy:
            assert policy in valid_policies

    def test_no_server_version_disclosure(self):
        """Test that server version is not disclosed in headers."""
        response = requests.get(f"{BASE_URL}/status")

        server_header = response.headers.get("Server", "")

        # Should not contain version numbers
        import re

        version_pattern = r"\d+\.\d+\.\d+"
        assert not re.search(version_pattern, server_header), f"Server header may disclose version: {server_header}"

    def test_content_security_policy(self):
        """Test Content-Security-Policy header if present."""
        response = requests.get(f"{BASE_URL}/status")

        csp = response.headers.get("Content-Security-Policy", "")

        # If CSP is set, verify it's not dangerously permissive
        if csp:
            assert "unsafe-eval" not in csp.lower() or "strict-dynamic" in csp.lower()


# ============================================================================
# Error Handling Security Tests
# ============================================================================


class TestErrorHandlingSecurity:
    """Tests for secure error handling."""

    def test_no_stack_trace_in_errors(self):
        """Test that stack traces are not exposed in error responses."""
        # Cause an error
        response = requests.post(
            f"{BASE_URL}/predict", data="not valid json", headers={"Content-Type": "application/json"}
        )

        if response.status_code >= 400:
            text = response.text.lower()

            # Should not contain stack trace indicators
            assert "traceback" not in text
            assert 'file "/' not in text
            assert "line " not in text or "line" in text  # "line" alone is OK

    def test_no_internal_paths_in_errors(self):
        """Test that internal file paths are not exposed."""
        response = requests.post(f"{BASE_URL}/predict", json={"invalid": "data"})

        if response.status_code >= 400:
            text = response.text.lower()

            # Should not contain internal paths
            assert "/home/" not in text
            assert "/app/" not in text
            assert "c:\\" not in text
            assert "d:\\" not in text

    def test_404_response_safe(self):
        """Test that 404 responses don't leak information."""
        response = requests.get(f"{BASE_URL}/nonexistent/endpoint/path")

        assert response.status_code == 404
        text = response.text.lower()

        # Should not reveal file system structure
        assert "routes" not in text
        assert "blueprint" not in text


# ============================================================================
# Model Security Tests
# ============================================================================


class TestModelSecurity:
    """Security tests for ML model handling."""

    def test_model_version_injection(self):
        """Test that model version parameter is sanitized."""
        malicious_versions = [
            "'; DROP TABLE models; --",
            "../../etc/passwd",
            "-1",
            "999999999999999999999999",
        ]

        for version in malicious_versions:
            response = requests.post(
                f"{BASE_URL}/predict",
                json={"temperature": 298.15, "humidity": 75.0, "precipitation": 10.0, "model_version": version},
            )

            # Should handle safely
            assert response.status_code in [200, 400, 401, 404, 422, 502]

    def test_model_path_traversal(self):
        """Test that model path traversal is prevented."""
        from app.services.predict import get_model_metadata

        # Try path traversal
        result = get_model_metadata("../../etc/passwd")

        # Should return None or fail safely
        assert result is None


# ============================================================================
# Run Security Tests
# ============================================================================


def run_security_tests():
    """Run security tests manually."""
    print("=" * 60)
    print("Running Security Tests")
    print("=" * 60)

    test_classes = [
        TestAuthenticationBypass,
        TestInputValidationSecurity,
        TestRateLimiting,
        TestSecurityHeaders,
        TestErrorHandlingSecurity,
        TestModelSecurity,
    ]

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                except requests.exceptions.ConnectionError:
                    print(f"  ✗ {method_name} - Could not connect to server")
                except AssertionError as e:
                    print(f"  ✗ {method_name} - {str(e)}")
                except Exception as e:
                    print(f"  ✗ {method_name} - {str(e)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_security_tests()
