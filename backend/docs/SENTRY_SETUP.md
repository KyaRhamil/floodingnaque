# Sentry Error Tracking Setup Guide

This guide explains how to set up and use Sentry for error tracking and performance monitoring in the Floodingnaque backend.

## What is Sentry?

Sentry is a real-time error tracking and performance monitoring platform that helps you:
- **Catch errors before users report them**
- **Track performance bottlenecks**
- **Get detailed stack traces and context**
- **Monitor application health in production**
- **Set up alerts for critical issues**

## Quick Setup (5 minutes)

### 1. Create a Sentry Account

1. Go to [sentry.io](https://sentry.io) and sign up (free tier available)
2. Create a new project and select **Flask** as the platform
3. Copy your **DSN** (Data Source Name) - it looks like:
   ```
   https://abc123def456@o123456.ingest.sentry.io/7890123
   ```

### 2. Configure Environment Variables

Add your Sentry DSN to your `.env` file:

```bash
# Sentry Error Tracking
SENTRY_DSN=https://your-key@your-org.ingest.sentry.io/your-project-id
SENTRY_ENVIRONMENT=production
SENTRY_RELEASE=2.0.0

# Performance Monitoring (adjust based on traffic)
SENTRY_TRACES_SAMPLE_RATE=0.1  # 10% of transactions
SENTRY_PROFILES_SAMPLE_RATE=0.1  # 10% of profiles
```

### 3. Start Your Application

That's it! Sentry is now automatically capturing errors and performance data.

```bash
python main.py
```

## Configuration Options

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SENTRY_DSN` | Your Sentry project DSN (required) | None | `https://key@org.ingest.sentry.io/id` |
| `SENTRY_ENVIRONMENT` | Environment name | `development` | `production`, `staging` |
| `SENTRY_RELEASE` | Release version | `2.0.0` | `2.1.0`, `git-sha` |
| `SENTRY_TRACES_SAMPLE_RATE` | % of transactions to track | `0.1` | `0.0` to `1.0` |
| `SENTRY_PROFILES_SAMPLE_RATE` | % of profiles to capture | `0.1` | `0.0` to `1.0` |

### Sample Rates Explained

**Traces Sample Rate** controls performance monitoring:
- `1.0` = Track 100% of requests (high volume, expensive)
- `0.1` = Track 10% of requests (recommended for production)
- `0.01` = Track 1% of requests (high-traffic sites)
- `0.0` = Disable performance monitoring

**Profiles Sample Rate** controls profiling:
- Similar to traces, but for CPU/memory profiling
- Keep this low in production (0.1 or less)

## What Gets Tracked Automatically

Sentry automatically captures:

### 1. **Exceptions and Errors**
- All unhandled exceptions
- HTTP 500 errors
- Database errors
- External API failures

### 2. **Performance Data**
- Request/response times
- Database query performance
- External API call latency
- Redis operations

### 3. **Context Information**
- Request headers (sensitive data filtered)
- User information (if set)
- Breadcrumbs (recent actions)
- Environment details

## Manual Error Tracking

### Capture Exceptions

```python
from app.utils.sentry import capture_exception

try:
    risky_operation()
except Exception as e:
    # Capture with additional context
    capture_exception(e, context={
        'tags': {
            'operation': 'data_ingestion',
            'source': 'openweathermap'
        },
        'extra': {
            'lat': 14.4793,
            'lon': 121.0198
        }
    })
```

### Capture Messages

```python
from app.utils.sentry import capture_message

# Log important events
capture_message(
    "Model training completed successfully",
    level='info',
    context={
        'tags': {'model_version': '2.0'},
        'extra': {'accuracy': 0.95}
    }
)
```

### Add Breadcrumbs

```python
from app.utils.sentry import add_breadcrumb

# Track user actions for debugging context
add_breadcrumb(
    message="User requested flood prediction",
    category="prediction",
    level="info",
    data={'temperature': 298.15, 'humidity': 65}
)
```

### Set User Context

```python
from app.utils.sentry import set_user_context

# Associate errors with users
set_user_context(
    user_id="user_123",
    email="admin@example.com",
    username="admin"
)
```

### Performance Monitoring

```python
from app.utils.sentry import start_transaction

# Track custom operations
with start_transaction(name="model_prediction", op="ml.predict") as transaction:
    result = model.predict(data)
    transaction.set_tag("model_version", "2.0")
```

## Filtering Sensitive Data

Sentry automatically filters sensitive information:

### Automatically Filtered
- `Authorization` headers
- `X-API-Key` headers
- `Cookie` headers
- Password fields

### Custom Filtering

Edit `app/utils/sentry.py` in the `before_send_hook` function:

```python
def before_send_hook(event, hint):
    # Filter specific exceptions
    if 'exc_info' in hint:
        exc_type, exc_value, tb = hint['exc_info']
        if exc_type.__name__ == 'NotFound':
            return None  # Don't send 404 errors
    
    # Scrub custom sensitive data
    if 'request' in event:
        body = event['request'].get('data', {})
        if 'api_key' in body:
            body['api_key'] = '[Filtered]'
    
    return event
```

## Sentry Dashboard Features

### 1. **Issues**
- View all errors grouped by type
- See frequency and affected users
- Track resolution status

### 2. **Performance**
- Monitor transaction times
- Identify slow endpoints
- Track database query performance

### 3. **Releases**
- Track errors by version
- Compare performance across releases
- Set up deploy notifications

### 4. **Alerts**
- Email/Slack notifications
- Custom alert rules
- Threshold-based alerts

## Best Practices

### Development
```bash
# Don't send errors to Sentry in development
SENTRY_DSN=  # Leave empty
```

### Staging
```bash
SENTRY_DSN=your_staging_dsn
SENTRY_ENVIRONMENT=staging
SENTRY_TRACES_SAMPLE_RATE=0.5  # Higher sampling for testing
```

### Production
```bash
SENTRY_DSN=your_production_dsn
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1  # Lower sampling to reduce costs
```

## Troubleshooting

### Sentry Not Capturing Errors

1. **Check DSN is set**:
   ```bash
   echo $SENTRY_DSN
   ```

2. **Check initialization logs**:
   ```
   INFO: Sentry initialized successfully (env=production, release=2.0.0)
   ```

3. **Test manually**:
   ```python
   from app.utils.sentry import capture_message
   capture_message("Test message", level='info')
   ```

### Too Many Events

Reduce sample rates:
```bash
SENTRY_TRACES_SAMPLE_RATE=0.01  # 1% instead of 10%
```

### Missing Context

Add more breadcrumbs:
```python
add_breadcrumb(
    message="Processing weather data",
    category="data",
    data={'source': 'OWM', 'records': 100}
)
```

## Cost Management

Sentry pricing is based on events:
- **Free tier**: 5,000 errors + 10,000 transactions/month
- **Team tier**: $26/month for more events

**Tips to stay within limits:**
- Use appropriate sample rates
- Filter out noisy errors (404s, etc.)
- Use separate projects for dev/staging/prod
- Monitor your quota in Sentry dashboard

## Integration with CI/CD

### Set Release on Deploy

```bash
# In your deployment script
export SENTRY_RELEASE=$(git rev-parse --short HEAD)
python main.py
```

### GitHub Integration

1. Go to Sentry → Settings → Integrations
2. Connect your GitHub repository
3. Sentry will automatically link commits to errors

## Advanced Features

### Source Maps (for debugging)

Sentry can show exact code lines where errors occur:

1. Upload source maps on deploy:
   ```bash
   sentry-cli releases files $RELEASE upload-sourcemaps ./app
   ```

### Custom Tags

Add searchable tags to all events:
```python
from app.utils.sentry import set_tag

set_tag("customer_tier", "premium")
set_tag("region", "asia-pacific")
```

### Performance Insights

Track custom metrics:
```python
import sentry_sdk

with sentry_sdk.start_span(op="db.query", description="fetch_weather_data"):
    data = db.query(WeatherData).all()
```

## Support

- **Sentry Docs**: https://docs.sentry.io/platforms/python/guides/flask/
- **Sentry Support**: https://sentry.io/support/
- **Community**: https://discord.gg/sentry

## Summary

✅ **Automatic error tracking** - No code changes needed  
✅ **Performance monitoring** - Track slow requests  
✅ **Rich context** - Breadcrumbs, tags, user info  
✅ **Alerts** - Get notified of critical issues  
✅ **Free tier available** - 5K errors/month  

Sentry is now integrated and ready to help you catch and fix issues faster! 🚀
