# Testing Quick Reference

Quick commands and examples for the Floodingnaque testing framework.

## Quick Commands

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests with coverage
pytest --cov=app --cov-report=html

# Run by category
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests
pytest -m property          # Property-based tests
pytest -m contract          # Contract tests
pytest -m snapshot          # Snapshot tests
pytest -m "not slow"        # Exclude slow tests

# Update snapshots
pytest -m snapshot --snapshot-update

# Run in parallel (faster)
pytest -n auto

# Coverage enforcement
pytest --cov=app --cov-fail-under=85

# Verbose output
pytest -v -s
```

## Test Markers

```python
@pytest.mark.unit           # Fast, no dependencies
@pytest.mark.integration    # Requires services
@pytest.mark.property       # Property-based
@pytest.mark.contract       # API contract
@pytest.mark.snapshot       # Snapshot testing
@pytest.mark.security       # Security tests
@pytest.mark.slow          # Long-running tests
```

## Property-Based Testing

```python
from hypothesis import given, settings
from tests.strategies import weather_data, valid_humidity

# Basic property test
@given(data=weather_data())
@settings(max_examples=100, deadline=None)
def test_weather_validation(data):
    """Property: Valid weather data should always pass"""
    result = validate(data)
    assert result is not None

# With custom strategy
@given(humidity=valid_humidity())
def test_humidity_range(humidity):
    """Property: Humidity should be 0-100"""
    assert 0 <= humidity <= 100
```

### Available Strategies

```python
# Weather
valid_temperature()         # -50°C to 50°C
valid_humidity()           # 0% to 100%
valid_precipitation()      # 0mm to 500mm
weather_data()            # Complete weather dict
extreme_weather_data()    # Boundary conditions

# Location
valid_latitude()          # -90° to 90°
valid_longitude()         # -180° to 180°
coordinates()            # Lat/lon dict

# Security
sql_injection_string()    # SQL injection patterns
xss_string()             # XSS patterns
path_traversal_string()  # Path traversal patterns

# Model
model_prediction_output() # Prediction structure
flood_probability()      # 0.0 to 1.0
probability_dict()       # Complementary probs
```

## Contract Testing

```python
@pytest.mark.contract
def test_api_contract(contract_client):
    """Contract: Endpoint returns expected schema"""
    response = contract_client.post('/api/endpoint', json=data)
    
    # Verify structure
    assert response.status_code == 200
    data = response.get_json()
    assert 'required_field' in data
    assert isinstance(data['required_field'], expected_type)
```

## Snapshot Testing

```python
from syrupy.assertion import SnapshotAssertion

@pytest.mark.snapshot
def test_output_snapshot(snapshot: SnapshotAssertion):
    """Snapshot: Capture output structure"""
    result = function_to_test(input_data)
    assert result == snapshot
```

## Common Patterns

### Mocking External Services

```python
from unittest.mock import Mock, patch

# Mock database
with patch('app.api.routes.endpoint.get_db_session') as mock_db:
    session = Mock()
    mock_db.return_value.__enter__ = Mock(return_value=session)
    # test code

# Mock ML model
with patch('app.services.predict.load_model') as mock_load:
    mock_model = Mock()
    mock_model.predict.return_value = [[0]]
    mock_model.predict_proba.return_value = [[0.8, 0.2]]
    mock_load.return_value = mock_model
    # test code
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (0, 'Safe'),
    (1, 'Alert'),
    (2, 'Critical'),
])
def test_risk_labels(input, expected):
    result = get_risk_label(input)
    assert result == expected
```

### Fixtures

```python
@pytest.fixture
def weather_data():
    return {
        'temperature': 25.0,
        'humidity': 60.0,
        'precipitation': 5.0
    }

def test_with_fixture(weather_data):
    result = process(weather_data)
    assert result is not None
```

## Debugging Tests

```bash
# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Run only failed tests
pytest --lf

# Show local variables on failure
pytest -l

# Run specific test
pytest tests/unit/test_file.py::TestClass::test_method

# Debug with pdb
pytest --pdb
```

## Coverage Analysis

```bash
# HTML report (detailed)
pytest --cov=app --cov-report=html
open htmlcov/index.html

# Terminal report with missing lines
pytest --cov=app --cov-report=term-missing

# XML report (CI/CD)
pytest --cov=app --cov-report=xml

# Focus on specific module
pytest --cov=app.services.predict --cov-report=term-missing
```

## CI/CD Integration

```bash
# Fast CI run (unit tests only)
pytest -m "unit and not slow" --cov=app --cov-fail-under=85

# Full test suite
pytest -m "not load" --cov=app --cov-report=xml

# Property-based tests
pytest -m property --hypothesis-verbosity=verbose

# Contract validation
pytest -m contract -v
```

## Troubleshooting

### Tests Too Slow
```bash
# Run in parallel
pytest -n auto

# Skip slow tests
pytest -m "not slow"

# Reduce property test examples
@settings(max_examples=50)
```

### Flaky Tests
```bash
# Run multiple times
pytest --count=10 tests/flaky_test.py

# Random order
pytest --random-order
```

### Coverage Not Met
```bash
# Find missing coverage
pytest --cov=app --cov-report=term-missing

# Focus on module
pytest tests/unit/test_module.py --cov=app.module
```

### Snapshot Failures
```bash
# View diff
pytest tests/snapshots/ -vv

# Update intentional changes
pytest -m snapshot --snapshot-update

# Update specific test
pytest tests/snapshots/test_file.py::test_name --snapshot-update
```

## Example Test Suite Run

```bash
# Complete validation before commit
pytest -m unit --cov=app --cov-fail-under=85 && \
pytest -m property && \
pytest -m contract && \
pytest -m snapshot
```

## Key Files

- `tests/conftest.py` - Shared fixtures
- `tests/strategies.py` - Hypothesis strategies
- `pytest.ini` - Test configuration
- `requirements-dev.txt` - Test dependencies

## Documentation

- Full Guide: `docs/TESTING_GUIDE.md`
- Pytest: https://docs.pytest.org/
- Hypothesis: https://hypothesis.readthedocs.io/
- Syrupy: https://github.com/tophat/syrupy
