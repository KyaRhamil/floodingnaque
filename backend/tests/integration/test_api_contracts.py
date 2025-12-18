"""
API Contract Tests for Integration Testing.

Tests API endpoints against expected response schemas and contracts.
Ensures API responses maintain consistent structure.
"""

import pytest
import requests
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

BASE_URL = "http://localhost:5000"


# ============================================================================
# Response Schema Definitions
# ============================================================================

@dataclass
class ResponseField:
    """Definition of an expected response field."""
    name: str
    field_type: type
    required: bool = True
    allowed_values: Optional[List] = None


# Schema definitions for each endpoint
ENDPOINT_SCHEMAS = {
    '/': {
        'fields': [
            ResponseField('name', str),
            ResponseField('version', str),
            ResponseField('endpoints', dict),
        ]
    },
    '/status': {
        'fields': [
            ResponseField('status', str, allowed_values=['running', 'error']),
            ResponseField('service', str),
        ]
    },
    '/health': {
        'fields': [
            ResponseField('status', str, allowed_values=['healthy', 'unhealthy', 'degraded']),
            ResponseField('model_available', bool),
        ]
    },
    '/api/models': {
        'fields': [
            ResponseField('models', list),
            ResponseField('total_versions', int),
        ]
    },
    '/api/version': {
        'fields': [
            ResponseField('version', str),
            ResponseField('name', str),
        ]
    },
    '/api/docs': {
        'fields': [
            ResponseField('endpoints', dict),
            ResponseField('version', str),
        ]
    },
    '/data': {
        'fields': [
            ResponseField('data', list),
            ResponseField('total', int),
            ResponseField('limit', int),
        ]
    },
}

PREDICT_RESPONSE_SCHEMA = {
    'fields': [
        ResponseField('prediction', int, allowed_values=[0, 1]),
        ResponseField('flood_risk', str, allowed_values=['high', 'low']),
        ResponseField('request_id', str),
    ],
    'optional_fields': [
        ResponseField('model_version', (str, type(None)), required=False),
        ResponseField('probability', dict, required=False),
        ResponseField('risk_level', int, required=False),
        ResponseField('risk_label', str, required=False),
        ResponseField('risk_color', str, required=False),
        ResponseField('risk_description', str, required=False),
        ResponseField('confidence', float, required=False),
    ]
}

ERROR_RESPONSE_SCHEMA = {
    'fields': [
        ResponseField('error', str),
        ResponseField('message', str),
    ],
    'optional_fields': [
        ResponseField('request_id', str, required=False),
    ]
}


# ============================================================================
# Schema Validation Utilities
# ============================================================================

def validate_response_schema(response_data: Dict, schema: Dict) -> List[str]:
    """
    Validate response data against a schema definition.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required fields
    for field in schema.get('fields', []):
        if field.name not in response_data:
            errors.append(f"Missing required field: {field.name}")
        else:
            value = response_data[field.name]
            
            # Check type
            if not isinstance(value, field.field_type):
                errors.append(
                    f"Field '{field.name}' has wrong type. "
                    f"Expected {field.field_type}, got {type(value)}"
                )
            
            # Check allowed values
            if field.allowed_values and value not in field.allowed_values:
                errors.append(
                    f"Field '{field.name}' has invalid value '{value}'. "
                    f"Allowed: {field.allowed_values}"
                )
    
    return errors


def validate_optional_fields(response_data: Dict, schema: Dict) -> List[str]:
    """Validate optional fields if present."""
    errors = []
    
    for field in schema.get('optional_fields', []):
        if field.name in response_data:
            value = response_data[field.name]
            if value is not None and not isinstance(value, field.field_type):
                errors.append(
                    f"Optional field '{field.name}' has wrong type. "
                    f"Expected {field.field_type}, got {type(value)}"
                )
    
    return errors


# ============================================================================
# Contract Tests - Public Endpoints
# ============================================================================

class TestRootEndpointContract:
    """Contract tests for the root endpoint."""
    
    def test_root_returns_json(self):
        """Test that root endpoint returns JSON."""
        response = requests.get(f"{BASE_URL}/")
        assert response.headers.get('Content-Type', '').startswith('application/json')
    
    def test_root_schema_compliance(self):
        """Test root endpoint response matches expected schema."""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200
        
        data = response.json()
        errors = validate_response_schema(data, ENDPOINT_SCHEMAS['/'])
        
        assert len(errors) == 0, f"Schema validation errors: {errors}"
    
    def test_root_endpoints_structure(self):
        """Test that endpoints field contains expected structure."""
        response = requests.get(f"{BASE_URL}/")
        data = response.json()
        
        endpoints = data.get('endpoints', {})
        assert isinstance(endpoints, dict)
        
        # Should list available endpoints
        assert len(endpoints) > 0


class TestStatusEndpointContract:
    """Contract tests for the status endpoint."""
    
    def test_status_schema_compliance(self):
        """Test status endpoint response matches expected schema."""
        response = requests.get(f"{BASE_URL}/status")
        assert response.status_code == 200
        
        data = response.json()
        errors = validate_response_schema(data, ENDPOINT_SCHEMAS['/status'])
        
        assert len(errors) == 0, f"Schema validation errors: {errors}"
    
    def test_status_running_value(self):
        """Test that status is 'running' when healthy."""
        response = requests.get(f"{BASE_URL}/status")
        data = response.json()
        
        assert data['status'] == 'running'


class TestHealthEndpointContract:
    """Contract tests for the health endpoint."""
    
    def test_health_schema_compliance(self):
        """Test health endpoint response matches expected schema."""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        errors = validate_response_schema(data, ENDPOINT_SCHEMAS['/health'])
        
        assert len(errors) == 0, f"Schema validation errors: {errors}"
    
    def test_health_model_available_boolean(self):
        """Test that model_available is a boolean."""
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        
        assert isinstance(data['model_available'], bool)


class TestModelsEndpointContract:
    """Contract tests for the models endpoint."""
    
    def test_models_schema_compliance(self):
        """Test models endpoint response matches expected schema."""
        response = requests.get(f"{BASE_URL}/api/models")
        assert response.status_code == 200
        
        data = response.json()
        errors = validate_response_schema(data, ENDPOINT_SCHEMAS['/api/models'])
        
        assert len(errors) == 0, f"Schema validation errors: {errors}"
    
    def test_models_list_structure(self):
        """Test that models list contains expected fields per model."""
        response = requests.get(f"{BASE_URL}/api/models")
        data = response.json()
        
        # If models exist, check their structure
        if data['models']:
            for model in data['models']:
                assert isinstance(model, dict)
                # Models should have at minimum a version or path


class TestDataEndpointContract:
    """Contract tests for the data endpoint."""
    
    def test_data_schema_compliance(self):
        """Test data endpoint response matches expected schema."""
        response = requests.get(f"{BASE_URL}/data")
        assert response.status_code == 200
        
        data = response.json()
        errors = validate_response_schema(data, ENDPOINT_SCHEMAS['/data'])
        
        assert len(errors) == 0, f"Schema validation errors: {errors}"
    
    def test_data_pagination_defaults(self):
        """Test data endpoint pagination defaults."""
        response = requests.get(f"{BASE_URL}/data")
        data = response.json()
        
        assert 'limit' in data
        assert data['limit'] >= 1
        assert data['limit'] <= 1000
    
    def test_data_pagination_custom_limit(self):
        """Test data endpoint respects custom limit."""
        response = requests.get(f"{BASE_URL}/data?limit=5")
        assert response.status_code == 200
        
        data = response.json()
        assert data['limit'] == 5
    
    def test_data_records_structure(self):
        """Test that data records have expected structure."""
        response = requests.get(f"{BASE_URL}/data?limit=1")
        data = response.json()
        
        if data['data']:
            record = data['data'][0]
            assert isinstance(record, dict)


# ============================================================================
# Contract Tests - Predict Endpoint
# ============================================================================

class TestPredictEndpointContract:
    """Contract tests for the predict endpoint."""
    
    def _get_auth_headers(self):
        """Get authentication headers if required."""
        return {'Content-Type': 'application/json'}
    
    def test_predict_schema_compliance(self):
        """Test predict endpoint response matches expected schema."""
        payload = {
            'temperature': 298.15,
            'humidity': 75.0,
            'precipitation': 10.0
        }
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=payload,
            headers=self._get_auth_headers()
        )
        
        # Skip if auth is required
        if response.status_code == 401:
            pytest.skip("API key required for predict endpoint")
        
        assert response.status_code == 200
        
        data = response.json()
        errors = validate_response_schema(data, PREDICT_RESPONSE_SCHEMA)
        errors.extend(validate_optional_fields(data, PREDICT_RESPONSE_SCHEMA))
        
        assert len(errors) == 0, f"Schema validation errors: {errors}"
    
    def test_predict_with_risk_level(self):
        """Test predict endpoint with risk level classification."""
        payload = {
            'temperature': 298.15,
            'humidity': 75.0,
            'precipitation': 10.0
        }
        
        response = requests.post(
            f"{BASE_URL}/predict?risk_level=true",
            json=payload,
            headers=self._get_auth_headers()
        )
        
        if response.status_code == 401:
            pytest.skip("API key required")
        
        assert response.status_code == 200
        data = response.json()
        
        # When risk_level=true, should include risk classification
        assert 'risk_level' in data
        assert 'risk_label' in data
        assert data['risk_level'] in [0, 1, 2]
        assert data['risk_label'] in ['Safe', 'Alert', 'Critical']
    
    def test_predict_probability_format(self):
        """Test that probability is returned in expected format."""
        payload = {
            'temperature': 298.15,
            'humidity': 75.0,
            'precipitation': 10.0
        }
        
        response = requests.post(
            f"{BASE_URL}/predict?return_proba=true",
            json=payload,
            headers=self._get_auth_headers()
        )
        
        if response.status_code == 401:
            pytest.skip("API key required")
        
        if response.status_code == 200:
            data = response.json()
            if 'probability' in data:
                prob = data['probability']
                assert 'flood' in prob or 'no_flood' in prob
                for key, value in prob.items():
                    assert 0.0 <= value <= 1.0


# ============================================================================
# Contract Tests - Error Responses
# ============================================================================

class TestErrorResponseContract:
    """Contract tests for error responses."""
    
    def test_invalid_limit_error_schema(self):
        """Test error response for invalid limit parameter."""
        response = requests.get(f"{BASE_URL}/data?limit=99999")
        assert response.status_code == 400
        
        data = response.json()
        errors = validate_response_schema(data, ERROR_RESPONSE_SCHEMA)
        
        assert len(errors) == 0, f"Error schema validation errors: {errors}"
    
    def test_negative_limit_error_schema(self):
        """Test error response for negative limit."""
        response = requests.get(f"{BASE_URL}/data?limit=-1")
        assert response.status_code == 400
        
        data = response.json()
        assert 'error' in data or 'message' in data
    
    def test_predict_missing_body_error_schema(self):
        """Test error response for missing request body."""
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 401:
            pytest.skip("API key required")
        
        assert response.status_code == 400
        data = response.json()
        assert 'error' in data


# ============================================================================
# Contract Tests - HTTP Headers
# ============================================================================

class TestResponseHeaders:
    """Contract tests for HTTP response headers."""
    
    def test_json_content_type(self):
        """Test that all API endpoints return JSON content type."""
        endpoints = ['/', '/status', '/health', '/api/models', '/data']
        
        for endpoint in endpoints:
            response = requests.get(f"{BASE_URL}{endpoint}")
            content_type = response.headers.get('Content-Type', '')
            
            assert 'application/json' in content_type, \
                f"Endpoint {endpoint} should return JSON, got {content_type}"
    
    def test_security_headers_present(self):
        """Test that security headers are present."""
        response = requests.get(f"{BASE_URL}/status")
        
        # Check for common security headers
        expected_headers = ['X-Content-Type-Options', 'X-Frame-Options']
        
        for header in expected_headers:
            assert header in response.headers, f"Missing security header: {header}"
    
    def test_cors_headers_on_options(self):
        """Test CORS headers for preflight requests."""
        response = requests.options(
            f"{BASE_URL}/status",
            headers={'Origin': 'http://localhost:3000'}
        )
        
        # May or may not be configured - just ensure no error
        assert response.status_code in [200, 204, 405]


# ============================================================================
# Contract Tests - Content Negotiation
# ============================================================================

class TestContentNegotiation:
    """Tests for content negotiation behavior."""
    
    def test_accepts_json_content_type(self):
        """Test that endpoints accept JSON content type."""
        response = requests.get(
            f"{BASE_URL}/status",
            headers={'Accept': 'application/json'}
        )
        
        assert response.status_code == 200
        assert 'application/json' in response.headers.get('Content-Type', '')
    
    def test_post_requires_json(self):
        """Test that POST endpoints require JSON body."""
        response = requests.post(
            f"{BASE_URL}/predict",
            data="not json",
            headers={'Content-Type': 'text/plain'}
        )
        
        # Should fail or be handled appropriately
        if response.status_code != 401:  # Skip auth check
            assert response.status_code in [400, 415]


# ============================================================================
# Run Tests
# ============================================================================

def run_contract_tests():
    """Run contract tests manually."""
    print("=" * 60)
    print("Running API Contract Tests")
    print("=" * 60)
    
    test_classes = [
        TestRootEndpointContract,
        TestStatusEndpointContract,
        TestHealthEndpointContract,
        TestModelsEndpointContract,
        TestDataEndpointContract,
        TestPredictEndpointContract,
        TestErrorResponseContract,
        TestResponseHeaders,
        TestContentNegotiation,
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
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


if __name__ == '__main__':
    run_contract_tests()
