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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FloodPredictionAPIUser(HttpUser):
    """
    Simulates a typical user of the Flood Prediction API.
    
    Behavior patterns:
    - Primarily checks status and health endpoints (lightweight)
    - Occasionally makes prediction requests (heavier)
    - Infrequently queries data and model info
    """
    
    # Wait time between tasks (simulates real user behavior)
    wait_time = between(1, 5)
    
    # Default host
    host = "http://localhost:5000"
    
    def on_start(self):
        """Called when a simulated user starts."""
        self.api_key = "test-api-key"  # Replace with valid key if auth enabled
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key
        }
    
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
        """Make prediction with normal weather conditions."""
        payload = {
            'temperature': random.uniform(293.15, 303.15),  # 20-30°C
            'humidity': random.uniform(40.0, 80.0),
            'precipitation': random.uniform(0.0, 20.0)
        }
        
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
    def predict_extreme_conditions(self):
        """Make prediction with extreme weather conditions."""
        payload = {
            'temperature': random.uniform(303.15, 315.0),  # Hot
            'humidity': random.uniform(85.0, 100.0),  # Very humid
            'precipitation': random.uniform(50.0, 150.0)  # Heavy rain
        }
        
        with self.client.post(
            "/predict?risk_level=true&return_proba=true",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Extreme conditions should likely trigger Alert or Critical
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
    def predict_edge_cases(self):
        """Make prediction with boundary value conditions."""
        edge_cases = [
            {'temperature': 200.0, 'humidity': 0.0, 'precipitation': 0.0},
            {'temperature': 330.0, 'humidity': 100.0, 'precipitation': 500.0},
            {'temperature': 273.15, 'humidity': 50.0, 'precipitation': 0.0},
        ]
        
        payload = random.choice(edge_cases)
        
        with self.client.post(
            "/predict",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 401]:
                response.success()
            else:
                response.failure(f"Edge case predict failed: {response.status_code}")
    
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
    Useful for stress testing the ML model endpoint.
    """
    
    wait_time = between(0.5, 2)
    host = "http://localhost:5000"
    
    def on_start(self):
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-Key': 'test-api-key'
        }
    
    @task
    @tag('predict', 'stress')
    def continuous_predictions(self):
        """Make continuous prediction requests for stress testing."""
        payload = {
            'temperature': random.uniform(273.15, 330.0),
            'humidity': random.uniform(0.0, 100.0),
            'precipitation': random.uniform(0.0, 200.0)
        }
        
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


# To use a custom shape, run:
# locust -f locustfile.py --host=... --class-picker
# Then select the shape class
