# Observability Quick Reference

## 🚀 Quick Start

### Enable Structured Logging
```bash
# In .env
LOG_FORMAT=json                 # or 'ecs' for Elastic Common Schema
LOG_LEVEL=INFO
```

### Enable Prometheus Metrics
```bash
# In .env
PROMETHEUS_METRICS_ENABLED=true
```

### View Metrics
```bash
curl http://localhost:5000/metrics
```

---

## 🔍 Correlation IDs

### Automatic in All Logs
Every log entry automatically includes:
- `correlation_id` - End-to-end request identifier
- `request_id` - This specific request
- `trace_id` - W3C trace context (32 hex chars)
- `span_id` - Current operation (16 hex chars)

### Client-Side Usage
Send correlation headers to track requests:
```bash
curl -H "X-Correlation-ID: my-custom-id" \
     -H "X-Request-ID: req-123" \
     http://localhost:5000/api/v1/predict
```

### Response Headers
Every response includes:
```
X-Correlation-ID: 18d4f2a3-8b7c9def1234
X-Request-ID: a1b2c3d4e5f6
X-Trace-ID: 9876543210abcdef1234567890abcdef
X-Span-ID: 1234567890abcdef
```

---

## 📊 Common Log Queries

### Find All Logs for a Request
```bash
# Using grep + jq
grep "correlation_id\":\"18d4f2a3-8b7c9def1234\"" logs/app.log | jq .

# Elasticsearch/Kibana
correlation_id:"18d4f2a3-8b7c9def1234"

# Splunk/Datadog
correlation_id="18d4f2a3-8b7c9def1234" | sort @timestamp
```

### Find Error Logs
```bash
# JSON logs
jq 'select(.log.level == "error")' logs/app.log

# With correlation ID
grep "\"level\":\"error\"" logs/app.log | jq 'select(.correlation_id != null)'
```

### Extract Performance Metrics
```bash
# Get all requests > 1 second
jq 'select(.duration_ms > 1000) | {correlation_id, endpoint: .http.route, duration_ms}' logs/app.log
```

---

## 📈 Prometheus Metrics Cheat Sheet

### HTTP Metrics
```promql
# Request rate
rate(floodingnaque_http_request_total[5m])

# Error rate
sum(rate(floodingnaque_http_request_total{status=~"5.."}[5m])) 
/ 
sum(rate(floodingnaque_http_request_total[5m]))

# P95 latency
histogram_quantile(0.95, 
  rate(floodingnaque_http_request_duration_seconds_bucket[5m])
)

# Requests by endpoint
sum(rate(floodingnaque_http_request_total[1m])) by (endpoint)
```

### Prediction Metrics
```promql
# Predictions by risk level
sum(rate(floodingnaque_predictions_total[1m])) by (risk_level)

# Average prediction time
rate(floodingnaque_prediction_duration_seconds_sum[5m])
/
rate(floodingnaque_prediction_duration_seconds_count[5m])
```

### External API Metrics
```promql
# API call rate by status
sum(rate(floodingnaque_external_api_calls_total[1m])) by (api, status)

# Circuit breaker status (0=closed, 1=open, 2=half-open)
floodingnaque_circuit_breaker_state

# API latency
histogram_quantile(0.95,
  rate(floodingnaque_external_api_duration_seconds_bucket[5m])
) by (api)
```

### Database Metrics
```promql
# Query latency by type
histogram_quantile(0.95,
  rate(floodingnaque_db_query_duration_seconds_bucket[5m])
) by (query_type)

# Connection pool usage
floodingnaque_db_pool_connections{status="checked_out"}
```

---

## 🎯 Grafana Dashboards

### Available Dashboards
1. **API Overview** - Service health, request rates, errors
2. **Error Tracking** - Error analysis, circuit breakers, external API failures
3. **Performance Analysis** - Latency percentiles, throughput, database performance

### Import to Grafana
```bash
# Navigate to: Dashboards → Import
# Upload files from: monitoring/grafana/dashboards/
```

---

## ⚙️ Production Configuration

### Recommended .env Settings
```bash
# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json                          # or 'ecs' for Elasticsearch
LOG_SAMPLING_ENABLED=true                # Reduce costs
LOG_SAMPLING_RATE=0.1                    # Keep 10% of INFO/DEBUG logs
LOG_SAMPLING_EXCLUDE_ERRORS=true         # Always keep errors

# Observability
PROMETHEUS_METRICS_ENABLED=true
SERVICE_NAME=floodingnaque-api
APP_VERSION=2.0.0

# Security
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id  # Optional
```

---

## 🔧 Troubleshooting

### Missing Correlation IDs
**Problem:** Logs don't include correlation IDs

**Solution:**
```python
# Verify middleware is loaded
from app.utils.correlation import get_correlation_context
ctx = get_correlation_context()
print(f"Correlation ID: {ctx.correlation_id if ctx else 'Not set'}")
```

### Logs Not in JSON
**Problem:** Logs appear as plain text

**Solution:**
```bash
# Check environment
echo $LOG_FORMAT  # Should be 'json' or 'ecs'

# Update .env
LOG_FORMAT=json

# Restart
```

### High Log Volume
**Problem:** Too many logs, high storage costs

**Solution:**
```bash
# Enable log sampling
LOG_SAMPLING_ENABLED=true
LOG_SAMPLING_RATE=0.1      # Adjust based on traffic
```

### Metrics Not in Prometheus
**Problem:** `/metrics` endpoint empty

**Solution:**
```bash
# Verify metrics enabled
echo $PROMETHEUS_METRICS_ENABLED  # Should be 'true'

# Check endpoint
curl http://localhost:5000/metrics

# Verify Prometheus scrape config
cat monitoring/prometheus.yml
```

---

## 📚 Code Examples

### Log with Context
```python
from app.utils.logging import get_logger

logger = get_logger(__name__)

logger.info(
    "Prediction completed",
    extra={
        'risk_level': 'Alert',
        'confidence': 0.85,
        'model_version': '2.0',
        'duration_ms': 125.4
    }
)
```

### Inject Correlation Headers
```python
import requests
from app.utils.correlation import inject_correlation_headers

headers = inject_correlation_headers({
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {token}'
})

response = requests.post(url, headers=headers, json=data)
```

### Trace Operation
```python
from app.utils.tracing import trace_operation

@trace_operation("database_query")
def fetch_data():
    return db.query().all()
```

### Record Metrics
```python
from app.utils.metrics import record_prediction
import time

start = time.time()
result = model.predict(data)
duration = time.time() - start

record_prediction(
    risk_level=result['risk_level'],
    model_version='2.0',
    duration=duration
)
```

---

## 🎓 Best Practices

### DO ✅
- ✅ Use structured logging with `extra={}` for context
- ✅ Always inject correlation headers for external calls
- ✅ Set up alerts for error rate > 5%
- ✅ Monitor P95 latency, not just averages
- ✅ Enable log sampling in high-traffic production
- ✅ Use ECS format if shipping to Elasticsearch

### DON'T ❌
- ❌ Log sensitive data (passwords, API keys, PII)
- ❌ Use string interpolation in log messages
- ❌ Forget to include correlation IDs in external calls
- ❌ Ignore circuit breaker open states
- ❌ Skip error logging with `exc_info=True`

---

## 📖 Full Documentation

For comprehensive information, see:
- [Complete Observability Guide](./OBSERVABILITY.md)
- [Prometheus Metrics Documentation](./METRICS.md)
- [Logging Best Practices](./LOGGING_BEST_PRACTICES.md)

---

## 🆘 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review [OBSERVABILITY.md](./OBSERVABILITY.md)
3. Contact the development team

---

**Last Updated:** January 13, 2026  
**Version:** 2.0.0
