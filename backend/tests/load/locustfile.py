"""
Locust Load Testing Configuration for Flood Prediction API.

Usage:
    # Install locust
    pip install locust

    # Run with web UI
    locust -f tests/load/locustfile.py --host=http://localhost:5000

    # Run headless (no web UI)
    locust -f tests/load/locustfile.py --host=http://localhost:5000 \
        --headless -u 100 -r 10 -t 60s

    # Run with HTML report
    locust -f tests/load/locustfile.py --host=http://localhost:5000 \
        --headless -u 100 -r 10 -t 60s --html=load_test_report.html

Configuration:
    -u, --users: Number of concurrent users
    -r, --spawn-rate: Users spawned per second
    -t, --run-time: Total run time (e.g., 60s, 5m, 1h)
    -H, --host: Target host (default in code is localhost:5000)
"""

from locust import HttpUser, task, between, tag, events
import json
import random
import time
import logging
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# REALISTIC DATA GENERATORS
# =============================================================================

class ParanaqueWeatherGenerator:
    """
    Generates realistic weather data for Paranaque City, Philippines.
    
    Paranaque climate characteristics:
    - Tropical monsoon climate (Köppen: Am)
    - Average temperature: 26-28°C (299-301 K)
    - Wet season: June-November (heavy rainfall)
    - Dry season: December-May
    - High humidity year-round (70-90%)
    """
    
    # Seasonal patterns for Philippines
    SEASONS = {
        'dry': {  # Dec-May
            'temp_range': (297.15, 307.15),  # 24-34°C
            'humidity_range': (60.0, 80.0),
            'precipitation_range': (0.0, 30.0),
        },
        'wet': {  # Jun-Nov
            'temp_range': (296.15, 304.15),  # 23-31°C
            'humidity_range': (75.0, 95.0),
            'precipitation_range': (0.0, 150.0),
        },
        'typhoon': {  # Jul-Oct peak
            'temp_range': (295.15, 302.15),  # 22-29°C
            'humidity_range': (85.0, 100.0),
            'precipitation_range': (50.0, 300.0),
        }
    }
    
    @classmethod
    def get_current_season(cls) -> str:
        """Determine current season based on month."""
        month = datetime.now().month
        if month in [7, 8, 9, 10]:
            # Higher chance of typhoon conditions during peak season
            return random.choice(['wet', 'wet', 'typhoon'])
        elif month in [6, 11]:
            return 'wet'
        else:
            return 'dry'
    
    @classmethod
    def generate_weather(cls, scenario: str = 'random') -> dict:
        """
        Generate realistic weather data.
        
        Args:
            scenario: 'normal', 'rainy', 'typhoon', 'extreme', or 'random'
        """
        if scenario == 'random':
            season = cls.get_current_season()
        elif scenario == 'typhoon':
            season = 'typhoon'
        elif scenario == 'rainy':
            season = 'wet'
        else:
            season = 'dry'
        
        params = cls.SEASONS[season]
        
        # Add some natural variance
        temp = random.uniform(*params['temp_range'])
        humidity = random.uniform(*params['humidity_range'])
        precipitation = random.uniform(*params['precipitation_range'])
        
        # Correlate precipitation with humidity (more rain = higher humidity)
        if precipitation > 50:
            humidity = min(100.0, humidity + random.uniform(5, 15))
        
        return {
            'temperature': round(temp, 2),
            'humidity': round(humidity, 2),
            'precipitation': round(precipitation, 2),
        }
    
    @classmethod
    def generate_historical_batch(cls, count: int = 10) -> list:
        """Generate batch of historical weather data with timestamps."""
        batch = []
        base_time = datetime.now() - timedelta(days=count)
        
        for i in range(count):
            weather = cls.generate_weather('random')
            weather['timestamp'] = (base_time + timedelta(days=i)).isoformat()
            batch.append(weather)
        
        return batch


class AuthenticatedUser:
    """
    Mixin for authenticated API access.
    Provides JWT token management for protected endpoints.
    """
    
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.token_expires = None
    
    def get_auth_headers(self) -> dict:
        """Get headers with authentication."""
        headers = {'Content-Type': 'application/json'}
        
        if self.access_token:
            headers['Authorization'] = f'Bearer {self.access_token}'
        
        # Fallback to API key
        api_key = os.getenv('LOAD_TEST_API_KEY', 'test-api-key')
        headers['X-API-Key'] = api_key
        
        return headers
    
    def ensure_authenticated(self, client) -> bool:
        """Ensure user has valid authentication."""
        # Check if we have a valid token
        if self.access_token and self.token_expires:
            if datetime.now() < self.token_expires:
                return True
        
        # Try to authenticate
        return self._authenticate(client)
    
    def _authenticate(self, client) -> bool:
        """Perform authentication."""
        # Try API key auth first (simpler for load testing)
        test_key = os.getenv('LOAD_TEST_API_KEY')
        if test_key:
            # API key auth doesn't require login
            return True
        
        # Otherwise try JWT login
        credentials = {
            'username': os.getenv('LOAD_TEST_USER', 'loadtest@example.com'),
            'password': os.getenv('LOAD_TEST_PASS', 'loadtest123'),
        }
        
        try:
            response = client.post('/auth/login', json=credentials)
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('access_token')
                self.refresh_token = data.get('refresh_token')
                # Default to 1 hour expiry if not specified
                self.token_expires = datetime.now() + timedelta(hours=1)
                return True
        except Exception:
            pass
        
        return False


class FloodPredictionAPIUser(HttpUser):
    """
    Simulates a typical user of the Flood Prediction API.
    
    Behavior patterns:
    - Primarily checks status and health endpoints (lightweight)
    - Occasionally makes prediction requests (heavier)
    - Infrequently queries data and model info
    
    Uses realistic Paranaque weather data for predictions.
    """
    
    # Wait time between tasks (simulates real user behavior)
    wait_time = between(1, 5)
    
    # Default host
    host = "http://localhost:5000"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auth = AuthenticatedUser()
        self.weather_gen = ParanaqueWeatherGenerator()
    
    def on_start(self):
        """Called when a simulated user starts."""
        self.headers = self.auth.get_auth_headers()
        # Pre-warm authentication
        self.auth.ensure_authenticated(self.client)
    
    # ========================================================================
    # Health and Status Endpoints (Most Common)
    # ========================================================================
    
    @task(10)  # Higher weight = more frequent
    @tag('health', 'lightweight')
    def check_status(self):
        """Check API status endpoint - most common operation."""
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get('status') != 'running':
                    response.failure(f"Status not running: {data}")
            else:
                response.failure(f"Status check failed: {response.status_code}")
    
    @task(8)
    @tag('health', 'lightweight')
    def check_health(self):
        """Check health endpoint with model availability."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'status' not in data:
                    response.failure("Missing status in health response")
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(5)
    @tag('info', 'lightweight')
    def get_root(self):
        """Get API root information."""
        self.client.get("/")
    
    # ========================================================================
    # Prediction Endpoints (Core Functionality)
    # ========================================================================
    
    @task(4)
    @tag('predict', 'heavyweight')
    def predict_normal_conditions(self):
        """Make prediction with realistic Paranaque weather conditions."""
        payload = self.weather_gen.generate_weather('normal')
        
        with self.client.post(
            "/predict?risk_level=true",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'prediction' not in data:
                    response.failure("Missing prediction in response")
            elif response.status_code == 401:
                response.success()  # Auth required is OK
            else:
                response.failure(f"Predict failed: {response.status_code}")
    
    @task(2)
    @tag('predict', 'heavyweight')
    def predict_rainy_conditions(self):
        """Make prediction with realistic rainy season conditions."""
        payload = self.weather_gen.generate_weather('rainy')
        
        with self.client.post(
            "/predict?risk_level=true&return_proba=true",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Check for valid risk labels
                risk_label = data.get('risk_label', '')
                if risk_label in ['Safe', 'Alert', 'Critical']:
                    response.success()
                else:
                    response.failure(f"Unexpected risk label: {risk_label}")
            elif response.status_code == 401:
                response.success()
            else:
                response.failure(f"Predict failed: {response.status_code}")
    
    @task(1)
    @tag('predict', 'heavyweight')
    def predict_typhoon_conditions(self):
        """Make prediction with typhoon-level conditions."""
        payload = self.weather_gen.generate_weather('typhoon')
        
        with self.client.post(
            "/predict?risk_level=true&return_proba=true",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 401, 429]:
                response.success()
            else:
                response.failure(f"Predict failed: {response.status_code}")
    
    # ========================================================================
    # Data and Model Endpoints
    # ========================================================================
    
    @task(3)
    @tag('data', 'moderate')
    def get_data_paginated(self):
        """Get weather data with pagination."""
        limit = random.choice([10, 25, 50, 100])
        offset = random.randint(0, 100)
        
        with self.client.get(
            f"/data?limit={limit}&offset={offset}",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if 'data' not in data or 'total' not in data:
                    response.failure("Missing data or total in response")
            else:
                response.failure(f"Data fetch failed: {response.status_code}")
    
    @task(2)
    @tag('models', 'moderate')
    def get_models(self):
        """Get available models list."""
        with self.client.get("/api/models", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'models' not in data:
                    response.failure("Missing models in response")
            else:
                response.failure(f"Models fetch failed: {response.status_code}")
    
    @task(1)
    @tag('docs', 'lightweight')
    def get_api_docs(self):
        """Get API documentation."""
        self.client.get("/api/docs")
    
    @task(1)
    @tag('version', 'lightweight')
    def get_version(self):
        """Get API version."""
        self.client.get("/api/version")


class PredictionHeavyUser(HttpUser):
    """
    Simulates a heavy user who primarily makes prediction requests.
    Uses realistic Paranaque weather data for stress testing the ML model endpoint.
    """
    
    wait_time = between(0.5, 2)
    host = "http://localhost:5000"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auth = AuthenticatedUser()
        self.weather_gen = ParanaqueWeatherGenerator()
    
    def on_start(self):
        self.headers = self.auth.get_auth_headers()
    
    @task
    @tag('predict', 'stress')
    def continuous_predictions(self):
        """Make continuous prediction requests with realistic data for stress testing."""
        # Mix of different weather scenarios
        scenarios = ['normal', 'rainy', 'rainy', 'typhoon', 'random', 'random']
        payload = self.weather_gen.generate_weather(random.choice(scenarios))
        
        with self.client.post(
            "/predict?risk_level=true",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 401, 429]:  # Include rate limited
                response.success()
            else:
                response.failure(f"Prediction failed: {response.status_code}")


class AuthenticatedAPIUser(HttpUser):
    """
    Simulates an authenticated user accessing protected endpoints.
    Tests JWT authentication flow and protected resources.
    """
    
    wait_time = between(1, 3)
    host = "http://localhost:5000"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auth = AuthenticatedUser()
        self.weather_gen = ParanaqueWeatherGenerator()
    
    def on_start(self):
        """Authenticate on start."""
        self.auth.ensure_authenticated(self.client)
        self.headers = self.auth.get_auth_headers()
    
    @task(3)
    @tag('auth', 'predict')
    def authenticated_prediction(self):
        """Make authenticated prediction request."""
        payload = self.weather_gen.generate_weather('random')
        
        with self.client.post(
            "/predict?risk_level=true&return_proba=true",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 401:
                # Token might have expired, try to re-auth
                if self.auth.ensure_authenticated(self.client):
                    self.headers = self.auth.get_auth_headers()
                response.success()  # Don't count auth issues as failures
            else:
                response.failure(f"Auth predict failed: {response.status_code}")
    
    @task(2)
    @tag('auth', 'data')
    def authenticated_data_access(self):
        """Access protected data endpoints."""
        with self.client.get(
            "/data?limit=50",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 401]:
                response.success()
            else:
                response.failure(f"Data access failed: {response.status_code}")
    
    @task(1)
    @tag('auth', 'batch')
    def batch_prediction(self):
        """Submit batch prediction request."""
        batch_data = self.weather_gen.generate_historical_batch(5)
        
        with self.client.post(
            "/batch/predict",
            json={'items': batch_data},
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 202, 401, 404]:  # 404 if batch endpoint not available
                response.success()
            else:
                response.failure(f"Batch predict failed: {response.status_code}")


class ReadOnlyUser(HttpUser):
    """
    Simulates a read-only user who only queries data.
    Good for baseline performance testing.
    """
    
    wait_time = between(1, 3)
    host = "http://localhost:5000"
    
    @task(5)
    @tag('readonly', 'baseline')
    def status_check(self):
        self.client.get("/status")
    
    @task(3)
    @tag('readonly', 'baseline')
    def health_check(self):
        self.client.get("/health")
    
    @task(2)
    @tag('readonly', 'data')
    def data_fetch(self):
        self.client.get("/data?limit=10")
    
    @task(1)
    @tag('readonly', 'baseline')
    def models_list(self):
        self.client.get("/api/models")


# ============================================================================
# Event Handlers for Reporting
# ============================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    logger.info("=" * 60)
    logger.info("LOAD TEST STARTING")
    logger.info(f"Target Host: {environment.host}")
    logger.info("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    logger.info("=" * 60)
    logger.info("LOAD TEST COMPLETED")
    
    # Log summary statistics
    stats = environment.stats
    logger.info(f"Total Requests: {stats.total.num_requests}")
    logger.info(f"Failed Requests: {stats.total.num_failures}")
    logger.info(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"Requests/sec: {stats.total.current_rps:.2f}")
    
    # Check for performance thresholds
    if stats.total.avg_response_time > 1000:
        logger.warning("WARNING: Average response time > 1 second!")
    if stats.total.fail_ratio > 0.01:
        logger.warning(f"WARNING: Failure rate {stats.total.fail_ratio*100:.2f}% > 1%!")
    
    logger.info("=" * 60)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, 
               response, context, exception, **kwargs):
    """Called on each request for custom logging/metrics."""
    if exception:
        logger.debug(f"Request failed: {name} - {exception}")
    elif response_time > 2000:  # Log slow requests
        logger.warning(f"Slow request: {name} took {response_time}ms")


# ============================================================================
# Custom Test Shapes (Optional)
# ============================================================================

from locust import LoadTestShape

class StepLoadShape(LoadTestShape):
    """
    Step load pattern: gradually increase load in steps.
    
    Stages:
    1. Ramp up to 10 users
    2. Step up to 25 users
    3. Step up to 50 users
    4. Step up to 100 users
    5. Ramp down
    """
    
    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 2},
        {"duration": 120, "users": 25, "spawn_rate": 5},
        {"duration": 180, "users": 50, "spawn_rate": 5},
        {"duration": 240, "users": 100, "spawn_rate": 10},
        {"duration": 300, "users": 50, "spawn_rate": 5},
        {"duration": 360, "users": 10, "spawn_rate": 2},
    ]
    
    def tick(self):
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        
        return None  # Stop test


class SpikeLoadShape(LoadTestShape):
    """
    Spike load pattern: sudden spike in traffic.
    Useful for testing how the system handles traffic spikes.
    """
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < 30:
            # Baseline load
            return (10, 5)
        elif run_time < 45:
            # Sudden spike
            return (100, 50)
        elif run_time < 60:
            # Spike maintained
            return (100, 10)
        elif run_time < 90:
            # Return to baseline
            return (10, 10)
        elif run_time < 120:
            # Cool down
            return (5, 1)
        else:
            return None


class EnduranceLoadShape(LoadTestShape):
    """
    Endurance test pattern: sustained low load for memory leak detection.
    Runs for 1 hour at low load to detect memory leaks and resource exhaustion.
    """
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < 60:
            # Initial ramp-up
            return (10, 2)
        elif run_time < 3540:  # 59 minutes
            # Sustained low load
            return (20, 1)
        elif run_time < 3600:  # Last minute
            # Ramp down
            return (5, 1)
        else:
            return None


class SmokeTestShape(LoadTestShape):
    """
    Quick smoke test: sanity check with minimal load.
    10 users for 1 minute - just verify the system works.
    """
    
    def tick(self):
        run_time = self.get_run_time()
        
        if run_time < 60:
            return (10, 5)
        else:
            return None


# To use a custom shape, run:
# locust -f locustfile.py --host=... --class-picker
# Then select the shape class
