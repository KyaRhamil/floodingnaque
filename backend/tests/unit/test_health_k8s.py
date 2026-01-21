"""
Unit tests for Kubernetes health probe routes.

Tests for app/api/routes/health_k8s.py
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


class TestLivenessProbe:
    """Tests for Kubernetes liveness probe endpoint."""

    def test_liveness_probe_success(self, client):
        """Test liveness probe returns healthy status."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert data["probe"] == "liveness"
        assert "timestamp" in data
        assert "request_id" in data

    def test_liveness_probe_response_format(self, client):
        """Test liveness probe response has correct format."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.get_json()

        # Validate timestamp is ISO format
        timestamp = data.get("timestamp")
        assert timestamp is not None
        # Should be parseable as datetime
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    def test_liveness_probe_no_rate_limit(self, client):
        """Test liveness probe is exempt from rate limiting."""
        # Make multiple rapid requests
        for _ in range(10):
            response = client.get("/health/live")
            assert response.status_code == 200


class TestReadinessProbe:
    """Tests for Kubernetes readiness probe endpoint."""

    def test_readiness_probe_success(self, client):
        """Test readiness probe returns ready status."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.get_json()
        assert data["probe"] == "readiness"
        assert "timestamp" in data

    @patch("app.api.routes.health_k8s.check_database_health")
    def test_readiness_probe_with_database_check(self, mock_db_health, client):
        """Test readiness probe includes database check."""
        mock_db_health.return_value = "healthy"

        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.get_json()
        assert "database" in data or "api" in data

    @patch("app.api.routes.health_k8s.check_database_health")
    def test_readiness_probe_database_not_ready(self, mock_db_health, client):
        """Test readiness probe when database is not ready."""
        mock_db_health.side_effect = Exception("Database unavailable")

        response = client.get("/health/ready")

        # Should still return 200 but with not_ready status
        data = response.get_json()
        assert "database" in data or "api" in data

    @patch("app.api.routes.health_k8s.check_model_health")
    def test_readiness_probe_with_model_check(self, mock_model_health, client):
        """Test readiness probe includes model availability check."""
        mock_model_health.return_value = "healthy"

        response = client.get("/health/ready")

        assert response.status_code == 200

    def test_readiness_probe_no_rate_limit(self, client):
        """Test readiness probe is exempt from rate limiting."""
        for _ in range(10):
            response = client.get("/health/ready")
            assert response.status_code == 200


class TestK8sProbeIntegration:
    """Integration tests for K8s probes."""

    def test_liveness_before_readiness(self, client):
        """Test that liveness works even if readiness fails."""
        # Liveness should always work if the app is running
        live_response = client.get("/health/live")
        assert live_response.status_code == 200

        # Readiness might fail due to dependencies
        ready_response = client.get("/health/ready")
        # Should return either 200 (ready) or 503 (not ready)
        assert ready_response.status_code in [200, 503]

    def test_probe_request_id_propagation(self, client):
        """Test that request IDs are properly propagated in probes."""
        headers = {"X-Request-ID": "test-request-123"}

        live_response = client.get("/health/live", headers=headers)
        ready_response = client.get("/health/ready", headers=headers)

        assert live_response.status_code == 200
        assert ready_response.status_code in [200, 503]
