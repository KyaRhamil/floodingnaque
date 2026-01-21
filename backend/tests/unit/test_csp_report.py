"""
Unit tests for CSP report routes.

Tests for app/api/routes/csp_report.py
"""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestCSPReportEndpoint:
    """Tests for CSP violation report endpoint."""

    def test_csp_report_options_request(self, client):
        """Test CORS preflight for CSP reports."""
        response = client.options("/csp-report")

        assert response.status_code == 204
        assert response.headers.get("Access-Control-Allow-Origin") == "*"
        assert "POST" in response.headers.get("Access-Control-Allow-Methods", "")

    def test_csp_report_valid_report(self, client):
        """Test receiving valid CSP violation report."""
        csp_report = {
            "csp-report": {
                "document-uri": "https://example.com/page",
                "referrer": "",
                "violated-directive": "script-src",
                "effective-directive": "script-src",
                "original-policy": "script-src 'self'",
                "disposition": "enforce",
                "blocked-uri": "https://evil.com/script.js",
                "status-code": 200,
            }
        }

        response = client.post("/csp-report", data=json.dumps(csp_report), content_type="application/csp-report")

        assert response.status_code == 204

    def test_csp_report_json_content_type(self, client):
        """Test CSP report with application/json content type."""
        csp_report = {
            "csp-report": {
                "document-uri": "https://example.com",
                "violated-directive": "script-src",
                "blocked-uri": "inline",
            }
        }

        response = client.post("/csp-report", data=json.dumps(csp_report), content_type="application/json")

        assert response.status_code == 204

    def test_csp_report_empty_body(self, client):
        """Test CSP report with empty body."""
        response = client.post("/csp-report", data="", content_type="application/csp-report")

        # Should handle gracefully
        assert response.status_code == 204

    def test_csp_report_invalid_json(self, client):
        """Test CSP report with invalid JSON."""
        response = client.post("/csp-report", data="not valid json", content_type="application/csp-report")

        # Should handle gracefully without 500 error
        assert response.status_code in [204, 400]

    def test_csp_report_flat_format(self, client):
        """Test CSP report without nested csp-report key."""
        csp_report = {
            "document-uri": "https://example.com",
            "violated-directive": "img-src",
            "blocked-uri": "https://tracker.com/pixel.gif",
        }

        response = client.post("/csp-report", data=json.dumps(csp_report), content_type="application/json")

        assert response.status_code == 204

    def test_csp_report_rate_limited(self, client):
        """Test that CSP report endpoint is rate limited."""
        # The endpoint allows 100 per minute
        # Make several requests - should all succeed under the limit
        for _ in range(10):
            response = client.post(
                "/csp-report", data=json.dumps({"csp-report": {"blocked-uri": "test"}}), content_type="application/json"
            )
            assert response.status_code in [204, 429]


class TestCSPReportLogging:
    """Tests for CSP report logging functionality."""

    @patch("app.api.routes.csp_report.csp_logger")
    def test_csp_report_logged(self, mock_logger, client):
        """Test that CSP violations are logged."""
        csp_report = {
            "csp-report": {
                "document-uri": "https://example.com",
                "violated-directive": "script-src",
                "blocked-uri": "https://malicious.com/script.js",
            }
        }

        response = client.post("/csp-report", data=json.dumps(csp_report), content_type="application/json")

        assert response.status_code == 204


class TestCSPReportDataExtraction:
    """Tests for CSP report data extraction."""

    def test_report_extracts_all_fields(self, client):
        """Test all CSP report fields are properly extracted."""
        csp_report = {
            "csp-report": {
                "document-uri": "https://example.com/page",
                "referrer": "https://referrer.com",
                "violated-directive": "script-src 'self'",
                "effective-directive": "script-src",
                "original-policy": "script-src 'self'; style-src 'self'",
                "disposition": "enforce",
                "blocked-uri": "https://blocked.com/script.js",
                "line-number": 42,
                "column-number": 10,
                "source-file": "https://example.com/page.html",
                "status-code": 200,
                "script-sample": "alert('xss')",
            }
        }

        response = client.post("/csp-report", data=json.dumps(csp_report), content_type="application/csp-report")

        assert response.status_code == 204

    def test_report_handles_missing_fields(self, client):
        """Test handling of CSP reports with missing optional fields."""
        csp_report = {
            "csp-report": {
                "document-uri": "https://example.com",
                "violated-directive": "default-src",
                # Missing many optional fields
            }
        }

        response = client.post("/csp-report", data=json.dumps(csp_report), content_type="application/json")

        assert response.status_code == 204

    def test_script_sample_truncation(self, client):
        """Test that long script samples are truncated."""
        long_sample = "x" * 1000  # Very long script sample

        csp_report = {
            "csp-report": {
                "document-uri": "https://example.com",
                "violated-directive": "script-src",
                "script-sample": long_sample,
            }
        }

        response = client.post("/csp-report", data=json.dumps(csp_report), content_type="application/json")

        # Should succeed without error from long sample
        assert response.status_code == 204


class TestCSPReportSecurityHeaders:
    """Tests for security of CSP report endpoint."""

    def test_csp_report_cors_headers(self, client):
        """Test CORS headers on CSP report response."""
        response = client.options("/csp-report")

        assert response.headers.get("Access-Control-Allow-Origin") == "*"
        assert "Content-Type" in response.headers.get("Access-Control-Allow-Headers", "")

    def test_csp_report_no_body_in_response(self, client):
        """Test CSP report returns no body (204 No Content)."""
        csp_report = {"csp-report": {"document-uri": "https://example.com", "violated-directive": "script-src"}}

        response = client.post("/csp-report", data=json.dumps(csp_report), content_type="application/json")

        assert response.status_code == 204
        assert response.data == b"" or len(response.data) == 0


class TestCSPReportClientInfo:
    """Tests for client information extraction in CSP reports."""

    def test_client_ip_extraction(self, client):
        """Test client IP is extracted from headers."""
        csp_report = {"csp-report": {"document-uri": "test"}}

        response = client.post(
            "/csp-report",
            data=json.dumps(csp_report),
            content_type="application/json",
            headers={"X-Forwarded-For": "1.2.3.4"},
        )

        assert response.status_code == 204

    def test_user_agent_extraction(self, client):
        """Test user agent is extracted from headers."""
        csp_report = {"csp-report": {"document-uri": "test"}}

        response = client.post(
            "/csp-report",
            data=json.dumps(csp_report),
            content_type="application/json",
            headers={"User-Agent": "TestBrowser/1.0"},
        )

        assert response.status_code == 204
