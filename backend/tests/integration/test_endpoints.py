#!/usr/bin/env python
"""
Integration tests for API endpoints.

These tests require a running Flask application.
"""

import json

import pytest
import requests

BASE_URL = "http://localhost:5000"


class TestHealthEndpoints:
    """Integration tests for health check endpoints."""

    def test_root_endpoint(self):
        """Test the root endpoint returns API information."""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_status_endpoint(self):
        """Test the /status endpoint."""
        response = requests.get(f"{BASE_URL}/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "running"

    def test_health_endpoint(self):
        """Test the /health endpoint."""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_available" in data


class TestModelsEndpoint:
    """Integration tests for model management endpoints."""

    def test_list_models(self):
        """Test the /api/models endpoint."""
        response = requests.get(f"{BASE_URL}/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total_versions" in data

    def test_api_version(self):
        """Test the /api/version endpoint."""
        response = requests.get(f"{BASE_URL}/api/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "name" in data


class TestDocsEndpoint:
    """Integration tests for documentation endpoint."""

    def test_api_docs(self):
        """Test the /api/docs endpoint."""
        response = requests.get(f"{BASE_URL}/api/docs")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        assert "version" in data


class TestDataEndpoint:
    """Integration tests for data endpoint."""

    def test_get_data_default(self):
        """Test the /data endpoint with default parameters."""
        response = requests.get(f"{BASE_URL}/data")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert "total" in data
        assert "limit" in data

    def test_get_data_with_limit(self):
        """Test the /data endpoint with custom limit."""
        response = requests.get(f"{BASE_URL}/data?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 10

    def test_get_data_invalid_limit(self):
        """Test the /data endpoint with invalid limit."""
        response = requests.get(f"{BASE_URL}/data?limit=9999")
        assert response.status_code == 400


def run_integration_tests():
    """Run integration tests manually."""
    print("=" * 60)
    print("Running Integration Tests")
    print("=" * 60)

    tests = [
        ("Root endpoint", lambda: requests.get(f"{BASE_URL}/")),
        ("Status endpoint", lambda: requests.get(f"{BASE_URL}/status")),
        ("Health endpoint", lambda: requests.get(f"{BASE_URL}/health")),
        ("Models endpoint", lambda: requests.get(f"{BASE_URL}/api/models")),
        ("API docs", lambda: requests.get(f"{BASE_URL}/api/docs")),
        ("Data endpoint", lambda: requests.get(f"{BASE_URL}/data")),
    ]

    for name, test_func in tests:
        try:
            response = test_func()
            status = "✓ PASS" if response.status_code == 200 else f"✗ FAIL ({response.status_code})"
            print(f"{status}: {name}")
        except requests.exceptions.ConnectionError:
            print(f"✗ ERROR: {name} - Could not connect to server")
        except Exception as e:
            print(f"✗ ERROR: {name} - {str(e)}")

    print("=" * 60)


if __name__ == "__main__":
    run_integration_tests()
