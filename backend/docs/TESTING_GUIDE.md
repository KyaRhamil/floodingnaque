# Comprehensive Testing Guide

## Overview

The Floodingnaque API implements a multi-layered testing strategy with 85% code coverage requirement. This guide covers all testing approaches including unit tests, integration tests, property-based testing, contract testing, and snapshot testing.

## Table of Contents

1. [Testing Strategy](#testing-strategy)
2. [Test Organization](#test-organization)
3. [Property-Based Testing](#property-based-testing)
4. [Contract Testing](#contract-testing)
5. [Snapshot Testing](#snapshot-testing)
6. [Running Tests](#running-tests)
7. [Writing New Tests](#writing-new-tests)
8. [CI/CD Integration](#cicd-integration)

---

## Testing Strategy

### Coverage Requirements

- **Minimum Coverage**: 85% (enforced by pytest)
- **Target Coverage**: 90%+
- **Critical Paths**: 100% coverage (authentication, prediction logic, data validation)

### Test Categories

Our testing pyramid consists of:

```
     /\
    /  \  Integration Tests (15%)
   /----\
  /      \  Contract Tests (10%)
 /--------\
/  Unit   \  Unit Tests (75%)
\  Tests  /
 \--------/
```

Additional specialized tests:
- **Property-Based Tests**: Edge case exploration
- **Snapshot Tests**: Regression detection
- **Security Tests**: Vulnerability scanning
- **Load Tests**: Performance validation

---

## Test Organization

### Directory Structure

```
backend/tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── strategies.py            # Hypothesis strategies
│
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_*.py
│   ├── test_property_based_validation.py
│   └── test_property_based_prediction.py
│
├── integration/             # Integration tests (require services)
│   └── test_*.py
│
├── contracts/               # Contract tests (API compatibility)
│   ├── __init__.py
│   └── test_api_contracts.py
│
├── snapshots/               # Snapshot tests (regression)
│   ├── __init__.py
│   ├── test_model_snapshots.py
│   └── __snapshots__/       # Snapshot files (auto-generated)
│
├── security/                # Security tests
│   └── test_*.py
│
└── load/                    # Load/performance tests
    └── test_*.py
```

### Test Markers

Tests are categorized using pytest markers:

```python
@pytest.mark.unit           # Fast, no external dependencies
@pytest.mark.integration    # Requires running services
@pytest.mark.security       # Security-focused tests
@pytest.mark.load          # Load/performance tests
@pytest.mark.model         # ML model-specific tests
@pytest.mark.slow          # Tests that take longer
@pytest.mark.property      # Property-based tests
@pytest.mark.contract      # Contract tests
@pytest.mark.snapshot      # Snapshot tests
```

---

## Property-Based Testing

Property-based testing uses [Hypothesis](https://hypothesis.readthedocs.io/) to automatically generate test cases and find edge cases.

### What is Property-Based Testing?

Instead of writing individual test cases, you define **properties** that should always hold true:

```python
# Traditional test
def test_temperature_validation():
    assert validate_temperature(25.0) == 25.0
    assert validate_temperature(30.0) == 30.0
    # ... many more examples

# Property-based test
@given(temp=valid_temperature())
def test_temperature_validation_property(temp):
    """Property: All valid temperatures should pass validation"""
    result = validate_temperature(temp)
    assert result == temp
    assert -50 <= result <= 50
```

### Available Strategies

We provide pre-built strategies in `tests/strategies.py`:

#### Weather Data
```python
from tests.strategies import (
    valid_temperature,      # -50°C to 50°C
    valid_humidity,         # 0% to 100%
    valid_precipitation,    # 0mm to 500mm
    weather_data,          # Complete weather dict
    extreme_weather_data,   # Boundary conditions
)

@given(data=weather_data())
def test_weather_processing(data):
    result = process_weather(data)
    assert result is not None
```

#### Location Data
```python
from tests.strategies import (
    valid_latitude,         # -90° to 90°
    valid_longitude,        # -180° to 180°
    coordinates,           # Complete lat/lon dict
    paranaque_coordinates, # Parañaque-specific
)
```

#### Security Testing
```python
from tests.strategies import (
    sql_injection_string,   # SQL injection patterns
    xss_string,            # XSS attack patterns
    path_traversal_string, # Path traversal attempts
)

@given(injection=sql_injection_string())
def test_sql_injection_prevention(injection):
    """Property: SQL injections should be blocked"""
    with pytest.raises(ValidationError):
        sanitize_input(injection, check_patterns=['sql_injection'])
```

#### Model Outputs
```python
from tests.strategies import (
    model_prediction_output,  # Complete prediction structure
    flood_probability,       # 0.0 to 1.0
    probability_dict,        # Complementary probabilities
)
```

### Running Property-Based Tests

```bash
# Run all property-based tests
pytest -m property

# Run with verbose output
pytest -m property -v

# Run with more examples (default: 100)
pytest -m property --hypothesis-verbosity=verbose

# Generate and save failing examples
pytest -m property --hypothesis-show-statistics
```

### Writing Property-Based Tests

**Best Practices:**

1. **Focus on invariants** - What should always be true?
2. **Use assumptions** - Filter out invalid cases
3. **Test boundaries** - Use extreme values
4. **Keep tests fast** - Limit max_examples if needed

Example:

```python
from hypothesis import given, assume, settings
from tests.strategies import valid_humidity

@given(humidity=valid_humidity())
@settings(max_examples=200, deadline=None)
def test_humidity_invariants(humidity):
    """Property: Humidity validation invariants"""
    # Assume we're testing valid range
    assume(0 <= humidity <= 100)
    
    result = validate_humidity(humidity)
    
    # Test invariants
    assert result == humidity
    assert 0 <= result <= 100
    assert isinstance(result, float)
```

---

## Contract Testing

Contract testing ensures API backward compatibility and adherence to specifications.

### What is Contract Testing?

Contract tests verify that:
1. **Request schemas** remain stable
2. **Response schemas** remain consistent
3. **Error formats** are standardized
4. **Breaking changes** are detected

### Running Contract Tests

```bash
# Run all contract tests
pytest -m contract

# Run specific contract suite
pytest tests/contracts/test_api_contracts.py -v

# Generate contract documentation
pytest -m contract --json-report --json-report-file=contract_report.json
```

### Contract Test Structure

```python
@pytest.mark.contract
def test_prediction_response_schema(contract_client):
    """
    Contract: Prediction endpoint returns standard response format.
    
    Response schema:
    {
        "success": boolean,
        "prediction": integer (0 or 1),
        "risk_level": integer (0, 1, or 2),
        "risk_label": string,
        "confidence": float
    }
    """
    response = contract_client.post('/api/v1/predict', json=valid_data)
    data = response.get_json()
    
    # Verify contract
    assert 'success' in data
    assert isinstance(data['prediction'], int)
    assert data['prediction'] in (0, 1)
    assert data['risk_level'] in (0, 1, 2)
```

### Writing Contract Tests

**Key Principles:**

1. **Test structure, not values** - Focus on schema
2. **Verify required fields** - Ensure presence
3. **Check types strictly** - Enforce type contracts
4. **Test error responses** - Include failure cases
5. **Document contracts** - Use docstrings

Example:

```python
@pytest.mark.contract
def test_new_endpoint_contract(contract_client):
    """
    Contract: [Endpoint Name] API specification.
    
    Request: {...}
    Response: {...}
    Errors: {...}
    """
    # Test successful response
    response = contract_client.post('/api/endpoint', json=request_data)
    assert response.status_code == 200
    
    data = response.get_json()
    assert 'required_field' in data
    assert isinstance(data['required_field'], expected_type)
    
    # Test error response
    response = contract_client.post('/api/endpoint', json=invalid_data)
    assert response.status_code == 400
    assert 'error' in response.get_json()
```

---

## Snapshot Testing

Snapshot testing captures and compares output structures to detect unintended changes.

### What is Snapshot Testing?

Snapshot tests:
1. **Capture output** on first run
2. **Compare** subsequent runs to snapshot
3. **Flag changes** for review
4. **Update** snapshots when changes are intentional

### Running Snapshot Tests

```bash
# Run snapshot tests
pytest -m snapshot

# Update snapshots after intentional changes
pytest -m snapshot --snapshot-update

# View snapshot differences
pytest -m snapshot -vv
```

### Snapshot Test Examples

```python
from syrupy.assertion import SnapshotAssertion

@pytest.mark.snapshot
def test_prediction_output_snapshot(snapshot: SnapshotAssertion):
    """Snapshot: Safe weather prediction output"""
    result = make_prediction({
        'temperature': 25.0,
        'humidity': 60.0,
        'precipitation': 5.0
    })
    
    # This will create/compare snapshot
    assert result == snapshot
```

### When to Use Snapshots

**Good for:**
- ✅ Complex nested structures
- ✅ Model output formats
- ✅ API response structures
- ✅ Risk classification outputs

**Not good for:**
- ❌ Dynamic data (timestamps, IDs)
- ❌ Floating-point comparisons
- ❌ Non-deterministic outputs

### Managing Snapshots

```bash
# Review all snapshots
ls tests/snapshots/__snapshots__/

# Update specific test snapshots
pytest tests/snapshots/test_model_snapshots.py::test_name --snapshot-update

# Clear all snapshots (will regenerate)
rm -rf tests/snapshots/__snapshots__/
pytest -m snapshot --snapshot-update
```

---

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Exclude slow tests
```

### Test Commands Reference

```bash
# Coverage enforcement (85% minimum)
pytest --cov=app --cov-fail-under=85

# Parallel execution (faster)
pytest -n auto

# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Run tests matching pattern
pytest -k "prediction"

# Generate HTML report
pytest --html=report.html --self-contained-html
```

### Environment Configuration

```bash
# Set test environment
export APP_ENV=test

# Run with specific configuration
pytest --envfile=.env.test
```

---

## Writing New Tests

### Unit Test Template

```python
"""
Test module for [component name].

Tests cover [brief description].
"""

import pytest
from unittest.mock import Mock, patch
from app.module import function_to_test


class TestComponentName:
    """Tests for [component] functionality."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        return {'key': 'value'}
    
    def test_happy_path(self, setup_data):
        """Test normal operation."""
        result = function_to_test(setup_data)
        assert result is not None
    
    def test_edge_case(self):
        """Test boundary condition."""
        result = function_to_test(edge_case_input)
        assert result == expected_output
    
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ExpectedException):
            function_to_test(invalid_input)
```

### Property-Based Test Template

```python
from hypothesis import given, settings
from tests.strategies import relevant_strategy

class TestPropertyBased:
    """Property-based tests for [component]."""
    
    @given(data=relevant_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_holds(self, data):
        """Property: [description of invariant]."""
        result = function_to_test(data)
        
        # Assert property
        assert property_holds(result)
```

### Contract Test Template

```python
@pytest.mark.contract
def test_endpoint_contract(contract_client):
    """
    Contract: [Endpoint] API specification.
    
    Request schema: {...}
    Response schema: {...}
    """
    response = contract_client.post('/api/endpoint', json=data)
    
    # Verify contract
    assert response.status_code == 200
    data = response.get_json()
    assert 'required_field' in data
```

### Snapshot Test Template

```python
@pytest.mark.snapshot
def test_output_snapshot(snapshot: SnapshotAssertion):
    """Snapshot: [Component] output structure."""
    result = function_to_test(input_data)
    assert result == snapshot
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run unit tests
        run: pytest -m unit --cov=app --cov-report=xml
      
      - name: Run property-based tests
        run: pytest -m property
      
      - name: Run contract tests
        run: pytest -m contract
      
      - name: Run snapshot tests
        run: pytest -m snapshot
      
      - name: Upload coverage
        uses: codecov/codecov-action@v2
        with:
          file: ./coverage.xml
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest -m "unit and not slow" --cov=app --cov-fail-under=85
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

---

## Best Practices

### General Guidelines

1. **Test Naming**: Use descriptive names
   ```python
   # Good
   def test_prediction_with_extreme_precipitation_returns_critical_risk():
   
   # Bad
   def test_prediction_1():
   ```

2. **One Assertion per Test**: Keep tests focused
   ```python
   # Good
   def test_temperature_validation():
       assert validate_temperature(25.0) == 25.0
   
   def test_temperature_range():
       assert -50 <= validate_temperature(25.0) <= 50
   
   # Bad
   def test_temperature():
       assert validate_temperature(25.0) == 25.0
       assert -50 <= validate_temperature(25.0) <= 50
       assert isinstance(validate_temperature(25.0), float)
   ```

3. **Use Fixtures**: DRY principle
   ```python
   @pytest.fixture
   def weather_data():
       return {'temperature': 25.0, 'humidity': 60.0}
   
   def test_with_fixture(weather_data):
       result = process(weather_data)
   ```

4. **Mock External Dependencies**: Isolate tests
   ```python
   with patch('app.services.external_api.fetch_data') as mock_fetch:
       mock_fetch.return_value = test_data
       result = function_using_api()
   ```

5. **Test Error Paths**: Not just happy paths
   ```python
   def test_handles_missing_data():
       with pytest.raises(ValidationError):
           process_data(None)
   ```

### Property-Based Testing Tips

- Start with simple properties
- Use `assume()` to filter inputs
- Increase `max_examples` for critical code
- Save and investigate failing examples

### Contract Testing Tips

- Version your contracts
- Test both success and error cases
- Document breaking changes
- Use contract tests as API documentation

### Snapshot Testing Tips

- Exclude dynamic fields (timestamps, IDs)
- Round floating-point values
- Review snapshot changes carefully
- Update snapshots only when intentional

---

## Troubleshooting

### Common Issues

#### Coverage Not Met

```bash
# View missing coverage
pytest --cov=app --cov-report=term-missing

# Focus on specific module
pytest --cov=app.services.predict --cov-report=term-missing
```

#### Hypothesis Takes Too Long

```python
# Reduce examples or disable deadline
@settings(max_examples=50, deadline=None)
```

#### Snapshot Test Failing

```bash
# View diff
pytest tests/snapshots/test_name.py -vv

# Update if intentional
pytest tests/snapshots/test_name.py --snapshot-update
```

#### Flaky Tests

```bash
# Run multiple times
pytest --count=10 tests/flaky_test.py

# Run with random order
pytest --random-order
```

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Syrupy Documentation](https://github.com/tophat/syrupy)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

## Summary

Our comprehensive testing strategy ensures:

✅ **High Coverage**: 85%+ code coverage  
✅ **Edge Case Detection**: Property-based testing  
✅ **API Compatibility**: Contract testing  
✅ **Regression Prevention**: Snapshot testing  
✅ **Performance Validation**: Load testing  
✅ **Security Assurance**: Security testing  

For questions or issues, refer to the team documentation or create an issue in the repository.
