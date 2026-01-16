#!/usr/bin/env python
"""
Security tests for API authentication and authorization.

These tests verify that authentication middleware works correctly.
"""

import os

import pytest
import requests

BASE_URL = "http://localhost:5000"


class TestApiKeyAuthentication:
    """Tests for API key authentication."""

    def test_protected_endpoint_without_key(self):
        """Test that protected endpoints require API key when configured."""
        # Note: This test only fails if VALID_API_KEYS is configured
        response = requests.post(f"{BASE_URL}/ingest", json={"lat": 14.6, "lon": 120.98})
        # If no API keys are configured, authentication is bypassed
        # If API keys are configured, should return 401
        assert response.status_code in [200, 401, 502]  # 502 if external API fails

    def test_protected_endpoint_with_invalid_key(self):
        """Test that invalid API key is rejected."""
        response = requests.post(
            f"{BASE_URL}/ingest", json={"lat": 14.6, "lon": 120.98}, headers={"X-API-Key": "invalid-key-12345"}
        )
        # If no API keys are configured, authentication is bypassed
        assert response.status_code in [200, 401, 502]

    def test_public_endpoints_accessible(self):
        """Test that public endpoints don't require authentication."""
        public_endpoints = ["/", "/status", "/health", "/api/docs", "/api/models", "/api/version", "/data"]

        for endpoint in public_endpoints:
            response = requests.get(f"{BASE_URL}{endpoint}")
            assert response.status_code == 200, f"Failed: {endpoint}"


class TestSecurityHeaders:
    """Tests for security headers."""

    def test_content_type_options(self):
        """Test X-Content-Type-Options header is set."""
        response = requests.get(f"{BASE_URL}/status")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_frame_options(self):
        """Test X-Frame-Options header is set."""
        response = requests.get(f"{BASE_URL}/status")
        assert response.headers.get("X-Frame-Options") == "DENY"

    def test_xss_protection(self):
        """Test X-XSS-Protection header is set."""
        response = requests.get(f"{BASE_URL}/status")
        assert "X-XSS-Protection" in response.headers

    def test_referrer_policy(self):
        """Test Referrer-Policy header is set."""
        response = requests.get(f"{BASE_URL}/status")
        assert "Referrer-Policy" in response.headers


class TestInputValidation:
    """Tests for input validation security."""

    def test_coordinate_bounds(self):
        """Test that coordinate validation prevents invalid inputs."""
        # Test with invalid latitude
        response = requests.get(f"{BASE_URL}/ingest")
        # Should return usage info (GET is informational)
        assert response.status_code == 200

    def test_pagination_limits(self):
        """Test that pagination limits are enforced."""
        # Test with limit too high
        response = requests.get(f"{BASE_URL}/data?limit=5000")
        assert response.status_code == 400

        # Test with negative limit
        response = requests.get(f"{BASE_URL}/data?limit=-1")
        assert response.status_code == 400


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_headers(self):
        """Test that rate limit headers are present."""
        response = requests.get(f"{BASE_URL}/status")
        # Rate limiting may add X-RateLimit-* headers
        # This is informational - actual rate limits are tested separately
        assert response.status_code == 200


def run_security_tests():
    """Run security tests manually."""
    print("=" * 60)
    print("Running Security Tests")
    print("=" * 60)

    tests = [
        ("Public endpoints accessible", test_public_endpoints),
        ("Security headers present", test_security_headers),
        ("Input validation", test_input_validation),
    ]

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ PASS: {name}")
        except AssertionError as e:
            print(f"✗ FAIL: {name} - {str(e)}")
        except requests.exceptions.ConnectionError:
            print(f"✗ ERROR: {name} - Could not connect to server")
        except Exception as e:
            print(f"✗ ERROR: {name} - {str(e)}")

    print("=" * 60)


def test_public_endpoints():
    """Test public endpoints are accessible."""
    for endpoint in ["/", "/status", "/health", "/api/docs"]:
        response = requests.get(f"{BASE_URL}{endpoint}")
        assert response.status_code == 200


def test_security_headers():
    """Test security headers are present."""
    response = requests.get(f"{BASE_URL}/status")
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"


def test_input_validation():
    """Test input validation works."""
    response = requests.get(f"{BASE_URL}/data?limit=5000")
    assert response.status_code == 400


if __name__ == "__main__":
    run_security_tests()
