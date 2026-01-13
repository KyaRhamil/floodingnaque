# Observability Guide

## Overview

Floodingnaque API includes comprehensive observability infrastructure for monitoring, tracing, and debugging in production environments. This guide covers structured logging, distributed tracing, metrics collection, and dashboard usage.

---

## 🔍 Table of Contents

1. [Structured JSON Logging](#structured-json-logging)
2. [Correlation IDs & Distributed Tracing](#correlation-ids--distributed-tracing)
3. [Metrics & Prometheus](#metrics--prometheus)
4. [Grafana Dashboards](#grafana-dashboards)
5. [Log Sampling](#log-sampling)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Structured JSON Logging

### Configuration

The API supports multiple log formats configured via environment variables:

```bash
# Environment Variables
LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json                   # json, ecs, text
LOG_COLORS=true                   # Enable ANSI colors for text format
LOG_DIR=logs                      # Directory for log files
```

### Log Formats

#### 1. **JSON Format** (Default)
Standard JSON format for log aggregation:

```json
{
  "timestamp": "2026-01-13T12:34:56.789Z",
  "level": "INFO",
  "logger": "app.services.ingest",
  "message": "Weather data fetched successfully",
  "correlation_id": "18d4f2a3-8b7c9def",
  "request_id": "a1b2c3d4e5f6",
  "trace_id": "9876543210abcdef",
  "span_id": "1234567890abcdef",
  "service": {
    "name": "floodingnaque-api",
    "version": "2.0.0",
    "environment": "production"
  },
  "http": {
    "method": "GET",
    "path": "/api/v1/predict",
    "status_code": 200
  },
  "duration_ms": 145.2
}
```

#### 2. **ECS Format** (Elastic Common Schema)
Compatible with Elasticsearch/ELK Stack:

```json
{
  "@timestamp": "2026-01-13T12:34:56.789Z",
  "log": {
    "level": "info",
    "logger": "app.services.ingest"
  },
  "message": "Weather data fetched successfully",
  "trace": {
    "id": "9876543210abcdef"
  },
  "span": {
    "id": "1234567890abcdef"
  },
  "service": {
    "name": "floodingnaque-api",
    "version": "2.0.0",
    "environment": "production"
  }
}
```

#### 3. **Text Format** (Development)
Human-readable format with colors for local development:

```
2026-01-13 12:34:56.789 INFO     [a1b2c3d4|98765432] app.services.ingest: Weather data fetched successfully | duration_ms=145.2
```

### Using Structured Logging

#### Basic Logging

```python
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Simple log
logger.info("User authenticated")

# Log with extra context
logger.info(
    "Prediction completed",
    extra={
        'risk_level': 'Alert',
        'confidence': 0.85,
        'location': 'Parañaque',
        'duration_ms': 125.4
    }
)

# Error logging with exception
try:
    result = risky_operation()
except Exception as e:
    logger.error("Operation failed", exc_info=True, extra={'user_id': '12345'})
```

#### Context Manager for Temporary Context

```python
from app.utils.logging import LogContext, get_logger

logger = get_logger(__name__)

with LogContext(user_id="12345", operation="batch_prediction"):
    logger.info("Starting batch processing")
    # All logs within this block will include user_id and operation
    process_batch()
    logger.info("Batch processing complete")
```

---

## Correlation IDs & Distributed Tracing

### Architecture

The API uses W3C Trace Context standard for distributed tracing across microservices and external APIs.

**Key Components:**
- **Correlation ID**: Unique identifier for the entire request chain
- **Request ID**: Unique identifier for this specific request
- **Trace ID**: W3C trace context identifier (32 hex chars)
- **Span ID**: Identifier for current operation (16 hex chars)

### How It Works

1. **Incoming Request**: Middleware extracts or creates correlation IDs from headers
2. **Request Processing**: All logs automatically include correlation context
3. **Outbound Requests**: Correlation headers injected into external API calls
4. **Response**: Correlation headers returned to client for end-to-end tracing

### Headers

**Incoming Headers (Optional):**
```
X-Correlation-ID: 18d4f2a3-8b7c9def1234
X-Request-ID: a1b2c3d4e5f6
X-Trace-ID: 9876543210abcdef1234567890abcdef
X-Span-ID: 1234567890abcdef
traceparent: 00-9876543210abcdef1234567890abcdef-1234567890abcdef-01
```

**Outgoing Headers (Automatic):**
```
X-Correlation-ID: 18d4f2a3-8b7c9def1234
X-Request-ID: a1b2c3d4e5f6
X-Trace-ID: 9876543210abcdef1234567890abcdef
X-Span-ID: fedcba0987654321
X-Service-Name: floodingnaque-api
X-Service-Version: 2.0.0
traceparent: 00-9876543210abcdef1234567890abcdef-fedcba0987654321-01
```

### Using Correlation Context

#### In Your Code

```python
from app.utils.correlation import get_correlation_context, get_correlation_id

# Get current correlation context
ctx = get_correlation_context()
if ctx:
    print(f"Correlation ID: {ctx.correlation_id}")
    print(f"Trace ID: {ctx.trace_id}")
    print(f"Request ID: {ctx.request_id}")

# Quick access to correlation ID
correlation_id = get_correlation_id()
```

#### Manual Span Creation (Advanced)

```python
from app.utils.tracing import get_current_trace

trace_ctx = get_current_trace()
if trace_ctx:
    span = trace_ctx.start_span("database_query", tags={'query_type': 'select'})
    try:
        result = execute_query()
        span.set_tag('rows_returned', len(result))
    except Exception as e:
        span.set_error(e)
    finally:
        trace_ctx.finish_span(span)
```

#### Decorator for Automatic Tracing

```python
from app.utils.tracing import trace_operation

@trace_operation("fetch_weather_data")
def fetch_weather():
    # Automatically traced with span
    return call_external_api()
```

#### Injecting Headers to External Services

```python
import requests
from app.utils.correlation import inject_correlation_headers

# Automatic header injection
headers = inject_correlation_headers({
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {token}'
})

response = requests.get('https://api.example.com/data', headers=headers)
```

### Searching Logs by Correlation ID

**Elasticsearch/Kibana:**
```
correlation_id:"18d4f2a3-8b7c9def1234"
```

**grep/jq (JSON logs):**
```bash
# Find all logs for a correlation ID
grep "18d4f2a3-8b7c9def1234" logs/app.log | jq .

# Extract specific fields
grep "18d4f2a3-8b7c9def1234" logs/app.log | jq '{timestamp, level, message, duration_ms}'
```

**Log Aggregation Query (Splunk/Datadog):**
```
correlation_id="18d4f2a3-8b7c9def1234" | sort @timestamp
```

---

## Metrics & Prometheus

### Exposed Metrics

The API exposes Prometheus metrics at `/metrics` endpoint.

#### HTTP Metrics

```promql
# Request rate
floodingnaque_http_request_total

# Request duration
floodingnaque_http_request_duration_seconds

# Requests in flight
floodingnaque_http_requests_in_progress
```

#### Prediction Metrics

```promql
# Total predictions by risk level
floodingnaque_predictions_total{risk_level="Alert"}

# Prediction latency
floodingnaque_prediction_duration_seconds

# Prediction latency summary (percentiles)
floodingnaque_prediction_latency_seconds
```

#### External API Metrics

```promql
# External API call count
floodingnaque_external_api_calls_total{api="openweathermap", status="success"}

# External API latency
floodingnaque_external_api_duration_seconds{api="openweathermap"}

# Circuit breaker state (0=closed, 1=open, 2=half-open)
floodingnaque_circuit_breaker_state{api="openweathermap"}
```

#### Database Metrics

```promql
# Database query latency
floodingnaque_db_query_duration_seconds{query_type="select"}

# Connection pool status
floodingnaque_db_pool_connections{status="checked_out"}
```

#### Cache Metrics

```promql
# Cache operations
floodingnaque_cache_operations_total{operation="get", result="hit"}

# Cache hit rate (0-100)
floodingnaque_cache_hit_rate{cache_type="redis"}

# Cache entries
floodingnaque_cache_entries{cache_type="redis"}
```

### Example Queries

**Error Rate:**
```promql
sum(rate(floodingnaque_http_request_total{status=~"5.."}[5m])) 
/ 
sum(rate(floodingnaque_http_request_total[5m]))
```

**P95 Latency:**
```promql
histogram_quantile(0.95, 
  sum(rate(floodingnaque_http_request_duration_seconds_bucket[5m])) by (le)
)
```

**Request Rate by Endpoint:**
```promql
sum(rate(floodingnaque_http_request_total[1m])) by (endpoint)
```

**External API Success Rate:**
```promql
sum(rate(floodingnaque_external_api_calls_total{status="success"}[5m])) by (api)
/
sum(rate(floodingnaque_external_api_calls_total[5m])) by (api)
```

---

## Grafana Dashboards

### Available Dashboards

The API includes three pre-built Grafana dashboards located in `monitoring/grafana/dashboards/`:

#### 1. **API Overview** (`api-overview.json`)
- Service health status
- Request rate and throughput
- Error rates (4xx, 5xx)
- Response time percentiles
- Top endpoints by traffic

#### 2. **Error Tracking** (`error-tracking.json`)
- 5xx/4xx error counts
- Error rate trends
- Error distribution by status code
- Error distribution by endpoint
- Circuit breaker status
- External API error rates

#### 3. **Performance Analysis** (`performance-analysis.json`)
- Response time percentiles (P50, P95, P99)
- Throughput by status code
- Top 10 endpoints by traffic
- Prediction performance metrics
- Database query latency
- Database connection pool status
- External API latency

### Importing Dashboards

1. **In Grafana UI:**
   - Navigate to Dashboards → Import
   - Upload JSON file or paste content
   - Select Prometheus data source
   - Click Import

2. **Via Provisioning:**
   ```yaml
   # grafana/provisioning/dashboards/floodingnaque.yaml
   apiVersion: 1
   providers:
     - name: 'Floodingnaque'
       folder: 'Floodingnaque API'
       type: file
       options:
         path: /etc/grafana/dashboards/floodingnaque
   ```

### Creating Alerts

**Example: High Error Rate Alert**

```yaml
# prometheus/alerts/floodingnaque.yml
groups:
  - name: floodingnaque_api
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(floodingnaque_http_request_total{status=~"5.."}[5m]))
          /
          sum(rate(floodingnaque_http_request_total[5m]))
          > 0.05
        for: 5m
        labels:
          severity: critical
          service: floodingnaque-api
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
          
      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(floodingnaque_http_request_duration_seconds_bucket[5m])) by (le)
          ) > 1.0
        for: 5m
        labels:
          severity: warning
          service: floodingnaque-api
        annotations:
          summary: "High P95 latency detected"
          description: "P95 latency is {{ $value }}s (threshold: 1s)"
```

---

## Log Sampling

### Configuration

For high-traffic production environments, enable log sampling to reduce storage costs:

```bash
# Environment Variables
LOG_SAMPLING_ENABLED=true         # Enable log sampling
LOG_SAMPLING_RATE=0.1             # Sample 10% of logs (0.0-1.0)
LOG_SAMPLING_EXCLUDE_ERRORS=true  # Always keep ERROR and CRITICAL logs
```

### How It Works

- **Sampling Rate**: `0.1` means 10% of logs are kept, `1.0` means all logs (no sampling)
- **Error Exclusion**: ERROR and CRITICAL logs are always kept regardless of sample rate
- **Console vs File**: Sampling only applies to file handlers, console logs are not sampled
- **Random Sampling**: Uses probabilistic sampling with Python's `random` module

### Example Scenarios

| Scenario | Sample Rate | Result |
|----------|-------------|--------|
| Development | `1.0` (disabled) | All logs kept |
| Low traffic production | `0.5` (50%) | Half of INFO/DEBUG logs, all errors |
| High traffic production | `0.1` (10%) | 10% of INFO/DEBUG logs, all errors |
| Very high traffic | `0.01` (1%) | 1% of INFO/DEBUG logs, all errors |

### Cost Savings

For a system logging 1 million INFO messages per day:

| Sample Rate | Logs Kept | Storage Saved |
|-------------|-----------|---------------|
| 1.0 (no sampling) | 1,000,000 | 0% |
| 0.5 | 500,000 | 50% |
| 0.1 | 100,000 | 90% |
| 0.01 | 10,000 | 99% |

**Note**: All ERROR and CRITICAL logs are always kept for debugging.

---

## Best Practices

### 1. **Always Include Context**

❌ **Bad:**
```python
logger.info("Prediction completed")
```

✅ **Good:**
```python
logger.info(
    "Prediction completed successfully",
    extra={
        'risk_level': risk_level,
        'confidence': confidence,
        'model_version': model_version,
        'duration_ms': duration
    }
)
```

### 2. **Use Appropriate Log Levels**

- **DEBUG**: Detailed diagnostic information (disabled in production)
- **INFO**: General informational messages (business logic milestones)
- **WARNING**: Warning messages (degraded performance, fallback behavior)
- **ERROR**: Error messages (recoverable errors)
- **CRITICAL**: Critical errors (system-level failures)

### 3. **Log Structured Data, Not Messages**

❌ **Bad:**
```python
logger.info(f"User {user_id} made prediction with risk {risk_level}")
```

✅ **Good:**
```python
logger.info(
    "Prediction made",
    extra={
        'user_id': user_id,
        'risk_level': risk_level,
        'event': 'prediction.completed'
    }
)
```

### 4. **Include Correlation IDs in External Calls**

Always inject correlation headers when calling external services:

```python
from app.utils.correlation import inject_correlation_headers

headers = inject_correlation_headers({'Content-Type': 'application/json'})
response = requests.post(external_api_url, headers=headers, json=data)
```

### 5. **Use Tracing Decorators**

For expensive operations, use automatic tracing:

```python
from app.utils.tracing import trace_operation

@trace_operation("ml_model_inference")
def predict_flood_risk(data):
    return model.predict(data)
```

### 6. **Monitor External Dependencies**

Always record metrics for external service calls:

```python
from app.utils.metrics import record_external_api_call
import time

start = time.time()
try:
    response = call_external_api()
    duration = time.time() - start
    record_external_api_call('openweathermap', 'success', duration)
except Exception as e:
    duration = time.time() - start
    record_external_api_call('openweathermap', 'error', duration)
    raise
```

### 7. **Set Up Alerts**

Configure alerts for critical metrics:
- Error rate > 5%
- P95 latency > 1 second
- Circuit breaker opens
- Database connection pool exhaustion
- External API failure rate > 10%

---

## Troubleshooting

### Issue: Missing Correlation IDs in Logs

**Symptoms:** Logs don't include `correlation_id`, `trace_id`, or `span_id`

**Solution:**
1. Verify middleware is loaded in `app/api/app.py`
2. Check that `CorrelationContext` is set in `before_request`
3. Ensure `app.utils.logging` imports are correct

### Issue: Logs Not Showing in JSON Format

**Symptoms:** Logs appear as plain text instead of JSON

**Solution:**
```bash
# Check environment variable
echo $LOG_FORMAT  # Should be "json" or "ecs"

# Update .env file
LOG_FORMAT=json

# Restart application
```

### Issue: High Log Volume

**Symptoms:** Log files growing too large, high storage costs

**Solution:**
Enable log sampling:
```bash
LOG_SAMPLING_ENABLED=true
LOG_SAMPLING_RATE=0.1        # Adjust based on traffic
LOG_SAMPLING_EXCLUDE_ERRORS=true
```

### Issue: Metrics Not Appearing in Prometheus

**Symptoms:** `/metrics` endpoint returns empty or incomplete data

**Solution:**
1. Check that Prometheus exporter is initialized:
   ```python
   # In app/api/app.py
   init_prometheus_metrics(app)
   ```
2. Verify environment variable:
   ```bash
   PROMETHEUS_METRICS_ENABLED=true
   ```
3. Check Prometheus scrape configuration in `prometheus.yml`

### Issue: Correlation Headers Not Propagated

**Symptoms:** Downstream services don't receive correlation IDs

**Solution:**
Ensure `inject_correlation_headers()` is used:
```python
from app.utils.correlation import inject_correlation_headers

# Before making external request
headers = inject_correlation_headers({'Authorization': token})
response = requests.get(url, headers=headers)
```

### Issue: Dashboard Shows No Data

**Symptoms:** Grafana dashboard panels are empty

**Solution:**
1. Verify Prometheus data source is configured correctly
2. Check that metrics are being exported: `curl http://localhost:5000/metrics`
3. Verify Prometheus is scraping: check Targets page in Prometheus UI
4. Update dashboard queries to match your metric names

---

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [W3C Trace Context Specification](https://www.w3.org/TR/trace-context/)
- [Elastic Common Schema (ECS)](https://www.elastic.co/guide/en/ecs/current/)
- [Structured Logging Best Practices](https://www.loggly.com/ultimate-guide/python-logging-basics/)

---

## Summary

The Floodingnaque API provides enterprise-grade observability:

✅ **Structured JSON logging** with ECS compatibility  
✅ **Distributed tracing** with W3C Trace Context  
✅ **Correlation IDs** automatically propagated across all services  
✅ **Prometheus metrics** for all critical operations  
✅ **Pre-built Grafana dashboards** for monitoring and alerting  
✅ **Log sampling** for cost optimization  
✅ **Automatic header injection** for external API calls  

For questions or issues, refer to the main project documentation or contact the development team.
