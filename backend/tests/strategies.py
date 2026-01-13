"""
Hypothesis Strategies for Property-Based Testing.

Provides reusable hypothesis strategies for generating test data
across the floodingnaque API testing suite.
"""

from hypothesis import strategies as st
from datetime import datetime, timezone, timedelta
from typing import Dict, Any


# ============================================================================
# Weather Data Strategies
# ============================================================================

@st.composite
def valid_temperature(draw, min_val: float = -50, max_val: float = 50) -> float:
    """
    Generate valid temperature values in Celsius.
    
    Args:
        min_val: Minimum temperature (default: -50°C)
        max_val: Maximum temperature (default: 50°C)
    """
    return draw(st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False))


@st.composite
def valid_humidity(draw, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """
    Generate valid humidity percentage values.
    
    Args:
        min_val: Minimum humidity (default: 0%)
        max_val: Maximum humidity (default: 100%)
    """
    return draw(st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False))


@st.composite
def valid_precipitation(draw, min_val: float = 0.0, max_val: float = 500.0) -> float:
    """
    Generate valid precipitation values in mm.
    
    Args:
        min_val: Minimum precipitation (default: 0 mm)
        max_val: Maximum precipitation (default: 500 mm)
    """
    return draw(st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False))


@st.composite
def valid_wind_speed(draw, min_val: float = 0.0, max_val: float = 150.0) -> float:
    """
    Generate valid wind speed values in m/s.
    
    Args:
        min_val: Minimum wind speed (default: 0 m/s)
        max_val: Maximum wind speed (default: 150 m/s)
    """
    return draw(st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False))


@st.composite
def valid_pressure(draw, min_val: float = 870.0, max_val: float = 1085.0) -> float:
    """
    Generate valid atmospheric pressure values in hPa.
    
    Args:
        min_val: Minimum pressure (default: 870 hPa)
        max_val: Maximum pressure (default: 1085 hPa)
    """
    return draw(st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False))


@st.composite
def weather_data(draw) -> Dict[str, Any]:
    """
    Generate a complete valid weather data dictionary.
    
    Returns:
        Dictionary with temperature, humidity, and precipitation
    """
    return {
        'temperature': draw(valid_temperature()),
        'humidity': draw(valid_humidity()),
        'precipitation': draw(valid_precipitation())
    }


@st.composite
def extended_weather_data(draw) -> Dict[str, Any]:
    """
    Generate extended weather data with additional fields.
    
    Returns:
        Dictionary with all weather parameters
    """
    return {
        'temperature': draw(valid_temperature()),
        'humidity': draw(valid_humidity()),
        'precipitation': draw(valid_precipitation()),
        'wind_speed': draw(valid_wind_speed()),
        'pressure': draw(valid_pressure())
    }


# ============================================================================
# Location Strategies
# ============================================================================

@st.composite
def valid_latitude(draw, min_val: float = -90.0, max_val: float = 90.0) -> float:
    """
    Generate valid latitude values.
    
    Args:
        min_val: Minimum latitude (default: -90°)
        max_val: Maximum latitude (default: 90°)
    """
    return draw(st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False))


@st.composite
def valid_longitude(draw, min_val: float = -180.0, max_val: float = 180.0) -> float:
    """
    Generate valid longitude values.
    
    Args:
        min_val: Minimum longitude (default: -180°)
        max_val: Maximum longitude (default: 180°)
    """
    return draw(st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False))


@st.composite
def coordinates(draw) -> Dict[str, float]:
    """
    Generate valid geographic coordinates.
    
    Returns:
        Dictionary with lat and lon keys
    """
    return {
        'lat': draw(valid_latitude()),
        'lon': draw(valid_longitude())
    }


@st.composite
def paranaque_coordinates(draw) -> Dict[str, float]:
    """
    Generate coordinates within Parañaque City bounds.
    
    Returns:
        Dictionary with lat and lon for Parañaque area
    """
    return {
        'lat': draw(st.floats(min_value=14.45, max_value=14.52, allow_nan=False, allow_infinity=False)),
        'lon': draw(st.floats(min_value=120.98, max_value=121.05, allow_nan=False, allow_infinity=False))
    }


# ============================================================================
# Edge Case Strategies
# ============================================================================

@st.composite
def extreme_weather_data(draw) -> Dict[str, Any]:
    """
    Generate extreme but valid weather data for boundary testing.
    
    Returns:
        Dictionary with extreme weather values
    """
    scenarios = draw(st.sampled_from([
        'extreme_cold', 'extreme_hot', 'extreme_humid', 'extreme_dry',
        'extreme_precipitation', 'no_precipitation'
    ]))
    
    if scenarios == 'extreme_cold':
        return {
            'temperature': draw(st.floats(min_value=-50, max_value=-30, allow_nan=False, allow_infinity=False)),
            'humidity': draw(valid_humidity()),
            'precipitation': draw(valid_precipitation())
        }
    elif scenarios == 'extreme_hot':
        return {
            'temperature': draw(st.floats(min_value=40, max_value=50, allow_nan=False, allow_infinity=False)),
            'humidity': draw(valid_humidity()),
            'precipitation': draw(valid_precipitation())
        }
    elif scenarios == 'extreme_humid':
        return {
            'temperature': draw(valid_temperature()),
            'humidity': draw(st.floats(min_value=95, max_value=100, allow_nan=False, allow_infinity=False)),
            'precipitation': draw(valid_precipitation())
        }
    elif scenarios == 'extreme_dry':
        return {
            'temperature': draw(valid_temperature()),
            'humidity': draw(st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False)),
            'precipitation': 0.0
        }
    elif scenarios == 'extreme_precipitation':
        return {
            'temperature': draw(valid_temperature()),
            'humidity': draw(st.floats(min_value=80, max_value=100, allow_nan=False, allow_infinity=False)),
            'precipitation': draw(st.floats(min_value=300, max_value=500, allow_nan=False, allow_infinity=False))
        }
    else:  # no_precipitation
        return {
            'temperature': draw(valid_temperature()),
            'humidity': draw(valid_humidity()),
            'precipitation': 0.0
        }


@st.composite
def invalid_weather_data(draw) -> Dict[str, Any]:
    """
    Generate invalid weather data for negative testing.
    
    Returns:
        Dictionary with at least one invalid weather value
    """
    invalid_type = draw(st.sampled_from([
        'negative_humidity', 'over_100_humidity', 'negative_precipitation',
        'extreme_temperature', 'nan_value', 'infinity_value', 'string_value'
    ]))
    
    base_data = {
        'temperature': draw(valid_temperature()),
        'humidity': draw(valid_humidity()),
        'precipitation': draw(valid_precipitation())
    }
    
    if invalid_type == 'negative_humidity':
        base_data['humidity'] = draw(st.floats(min_value=-100, max_value=-0.1, allow_nan=False, allow_infinity=False))
    elif invalid_type == 'over_100_humidity':
        base_data['humidity'] = draw(st.floats(min_value=100.1, max_value=200, allow_nan=False, allow_infinity=False))
    elif invalid_type == 'negative_precipitation':
        base_data['precipitation'] = draw(st.floats(min_value=-100, max_value=-0.1, allow_nan=False, allow_infinity=False))
    elif invalid_type == 'extreme_temperature':
        base_data['temperature'] = draw(st.floats(min_value=100, max_value=200, allow_nan=False, allow_infinity=False))
    elif invalid_type == 'nan_value':
        field = draw(st.sampled_from(['temperature', 'humidity', 'precipitation']))
        base_data[field] = float('nan')
    elif invalid_type == 'infinity_value':
        field = draw(st.sampled_from(['temperature', 'humidity', 'precipitation']))
        base_data[field] = float('inf')
    elif invalid_type == 'string_value':
        field = draw(st.sampled_from(['temperature', 'humidity', 'precipitation']))
        base_data[field] = draw(st.text(min_size=1, max_size=20))
    
    return base_data


# ============================================================================
# Timestamp Strategies
# ============================================================================

@st.composite
def valid_timestamp(draw, start_date: datetime = None, end_date: datetime = None) -> datetime:
    """
    Generate valid timestamps.
    
    Args:
        start_date: Minimum date (default: 2020-01-01)
        end_date: Maximum date (default: now + 1 year)
    """
    if start_date is None:
        start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    if end_date is None:
        end_date = datetime.now(timezone.utc) + timedelta(days=365)
    
    return draw(st.datetimes(min_value=start_date, max_value=end_date, timezones=st.just(timezone.utc)))


@st.composite
def iso_timestamp(draw) -> str:
    """
    Generate ISO-format timestamp strings.
    
    Returns:
        ISO 8601 formatted timestamp string
    """
    dt = draw(valid_timestamp())
    return dt.isoformat()


# ============================================================================
# Risk Level Strategies
# ============================================================================

@st.composite
def risk_level(draw) -> int:
    """
    Generate valid risk level values (0=Safe, 1=Alert, 2=Critical).
    """
    return draw(st.integers(min_value=0, max_value=2))


@st.composite
def risk_label(draw) -> str:
    """
    Generate valid risk label strings.
    """
    return draw(st.sampled_from(['Safe', 'Alert', 'Critical']))


@st.composite
def flood_probability(draw, min_prob: float = 0.0, max_prob: float = 1.0) -> float:
    """
    Generate valid flood probability values.
    
    Args:
        min_prob: Minimum probability (default: 0.0)
        max_prob: Maximum probability (default: 1.0)
    """
    return draw(st.floats(min_value=min_prob, max_value=max_prob, allow_nan=False, allow_infinity=False))


@st.composite
def probability_dict(draw) -> Dict[str, float]:
    """
    Generate probability dictionary with no_flood and flood probabilities.
    
    Returns:
        Dictionary with complementary probabilities that sum to 1.0
    """
    flood_prob = draw(flood_probability())
    return {
        'no_flood': 1.0 - flood_prob,
        'flood': flood_prob
    }


# ============================================================================
# API Request Strategies
# ============================================================================

@st.composite
def api_key(draw) -> str:
    """
    Generate valid API key format strings.
    
    Returns:
        String in API key format (32 hexadecimal characters)
    """
    return draw(st.text(alphabet='0123456789abcdef', min_size=32, max_size=32))


@st.composite
def pagination_params(draw) -> Dict[str, int]:
    """
    Generate valid pagination parameters.
    
    Returns:
        Dictionary with limit and offset
    """
    return {
        'limit': draw(st.integers(min_value=1, max_value=1000)),
        'offset': draw(st.integers(min_value=0, max_value=10000))
    }


@st.composite
def invalid_pagination_params(draw) -> Dict[str, int]:
    """
    Generate invalid pagination parameters for negative testing.
    
    Returns:
        Dictionary with at least one invalid pagination value
    """
    invalid_type = draw(st.sampled_from(['negative_limit', 'zero_limit', 'excessive_limit', 'negative_offset']))
    
    if invalid_type == 'negative_limit':
        return {'limit': draw(st.integers(min_value=-100, max_value=-1)), 'offset': 0}
    elif invalid_type == 'zero_limit':
        return {'limit': 0, 'offset': 0}
    elif invalid_type == 'excessive_limit':
        return {'limit': draw(st.integers(min_value=1001, max_value=10000)), 'offset': 0}
    else:  # negative_offset
        return {'limit': 10, 'offset': draw(st.integers(min_value=-100, max_value=-1))}


# ============================================================================
# String Strategies for Security Testing
# ============================================================================

@st.composite
def sql_injection_string(draw) -> str:
    """
    Generate SQL injection attempt strings for security testing.
    
    Returns:
        String containing SQL injection patterns
    """
    patterns = [
        "' OR '1'='1",
        "'; DROP TABLE users; --",
        "' UNION SELECT * FROM users--",
        "admin'--",
        "1' OR '1' = '1",
        "'; EXEC sp_MSForEachTable 'DROP TABLE ?'; --"
    ]
    return draw(st.sampled_from(patterns))


@st.composite
def xss_string(draw) -> str:
    """
    Generate XSS attempt strings for security testing.
    
    Returns:
        String containing XSS patterns
    """
    patterns = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<svg onload=alert('XSS')>",
        "<iframe src='javascript:alert(\"XSS\")'></iframe>"
    ]
    return draw(st.sampled_from(patterns))


@st.composite
def path_traversal_string(draw) -> str:
    """
    Generate path traversal attempt strings for security testing.
    
    Returns:
        String containing path traversal patterns
    """
    patterns = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "....//....//....//etc/passwd"
    ]
    return draw(st.sampled_from(patterns))


# ============================================================================
# Model Output Strategies
# ============================================================================

@st.composite
def model_prediction_output(draw) -> Dict[str, Any]:
    """
    Generate valid model prediction output structure.
    
    Returns:
        Dictionary matching the expected prediction output format
    """
    prediction = draw(st.integers(min_value=0, max_value=1))
    flood_prob = draw(flood_probability())
    
    # Determine risk level based on probability
    if flood_prob < 0.3:
        risk_level_val = 0
        risk_label_val = 'Safe'
    elif flood_prob < 0.75:
        risk_level_val = 1
        risk_label_val = 'Alert'
    else:
        risk_level_val = 2
        risk_label_val = 'Critical'
    
    return {
        'prediction': prediction,
        'flood_risk': 'high' if prediction == 1 else 'low',
        'risk_level': risk_level_val,
        'risk_label': risk_label_val,
        'confidence': draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False)),
        'probability': {
            'no_flood': 1.0 - flood_prob,
            'flood': flood_prob
        },
        'model_version': draw(st.text(alphabet='0123456789.', min_size=3, max_size=10)),
        'model_name': 'flood_predictor'
    }
