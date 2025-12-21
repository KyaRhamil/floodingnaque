# Vercel Speed Insights Setup Guide for Floodingnaque

This document provides a complete guide on how Vercel Speed Insights has been integrated into the Floodingnaque flood prediction system.

## Overview

Vercel Speed Insights is now integrated into the Floodingnaque project to monitor real-world performance metrics of the flood prediction dashboard. This helps us understand how our application performs for end users and identify optimization opportunities.

## What is Vercel Speed Insights?

Vercel Speed Insights is a Web Vitals monitoring platform that:
- Tracks Core Web Vitals (LCP, FID, CLS)
- Monitors real user performance data
- Provides insights on performance trends
- Helps identify performance bottlenecks

### Core Web Vitals Explained

- **LCP (Largest Contentful Paint)**: Time for the largest element to render (target: < 2.5s)
- **FID (First Input Delay)**: Response time to first user interaction (target: < 100ms)
- **CLS (Cumulative Layout Shift)**: Visual stability during page load (target: < 0.1)

## Installation

The `@vercel/speed-insights` package is already included in the project:

```json
{
  "dependencies": {
    "@vercel/speed-insights": "^1.3.1"
  }
}
```

### Installing Dependencies

If dependencies need to be installed or reinstalled:

```bash
npm install
# or
pnpm install
# or
yarn install
# or
bun install
```

## Integration Points

### 1. HTML Integration (Frontend)

Speed Insights scripts have been added to all HTML pages:

#### `frontend/index.html` (Dashboard)
```html
<script>
  window.si = window.si || function () { (window.siq = window.siq || []).push(arguments); };
</script>
<script defer src="/_vercel/speed-insights/script.js"></script>
```

#### `frontend/alert.html` (Alerts Page)
```html
<script>
  window.si = window.si || function () { (window.siq = window.siq || []).push(arguments); };
</script>
<script defer src="/_vercel/speed-insights/script.js"></script>
```

### 2. Vercel Deployment Configuration

The `vercel.json` file has been configured for proper deployment:

```json
{
  "version": 2,
  "name": "floodingnaque",
  "public": false,
  "builds": [
    {
      "src": "backend/main.py",
      "use": "@vercel/python"
    },
    {
      "src": "frontend/**/*.{html,css,js}",
      "use": "@vercel/static"
    }
  ]
}
```

## Deployment Steps

### 1. Ensure Speed Insights is Enabled in Vercel Dashboard

1. Go to https://vercel.com/dashboard
2. Select your Floodingnaque project
3. Navigate to the **Speed Insights** tab
4. Click **Enable** if not already enabled

### 2. Deploy to Vercel

```bash
vercel deploy --prod
```

Or connect your GitHub repository to Vercel for automatic deployments.

### 3. Monitor Performance

After deployment:
1. Visit your deployed application
2. Wait 24-48 hours for initial metrics to be collected
3. Check the Speed Insights dashboard to see real user metrics

## Performance Optimization Tips

### 1. Frontend Optimization

**Image Optimization**
```javascript
// Use responsive images
<img 
  src="image.jpg" 
  srcset="image-small.jpg 480w, image-large.jpg 1200w"
  alt="Description"
/>
```

**Lazy Loading**
```html
<img loading="lazy" src="image.jpg" alt="Description" />
```

**Code Splitting**
```javascript
// Load heavy libraries only when needed
const leaflet = await import('leaflet');
const chartjs = await import('chart.js');
```

### 2. Backend Optimization

**Flask Caching**
```python
from flask import Flask
from flask_caching import Cache

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/api/predictions')
@cache.cached(timeout=300)
def get_predictions():
    return prediction_results
```

**Gzip Compression**
```python
from flask_compress import Compress

app = Flask(__name__)
Compress(app)
```

### 3. Model Performance

**Async Processing**
```python
from flask import Flask
from celery import Celery

app = Flask(__name__)
celery = Celery(app.name, broker='redis://localhost:6379')

@celery.task
def predict_flood_risk(data):
    # Heavy computation
    return results
```

## Monitoring Dashboard

### Key Metrics to Watch

1. **Page Load Time**: Total time to fully load the page
2. **Time to Interactive**: When users can interact with the page
3. **First Input Delay**: Response time to user interactions
4. **Layout Shifts**: Visual stability during loading

### Performance Targets

| Metric | Target |
|--------|--------|
| LCP | < 2.5s |
| FID | < 100ms |
| CLS | < 0.1 |
| TTFB | < 600ms |
| FCP | < 1.8s |

## Data Privacy and Compliance

### Data Collected

Vercel Speed Insights collects:
- Page load metrics
- User interaction timings
- Core Web Vitals
- Browser and device information
- Geographic location (country-level)

### Data NOT Collected

- Personal information
- Sensitive query parameters
- Form input data
- Page content

### Privacy Compliance

To remove sensitive information from URLs before sending to Vercel, add the `speedInsightsBeforeSend` function:

```html
<script>
  window.speedInsightsBeforeSend = function(data) {
    // Remove query parameters
    if (data.url && data.url.includes('?')) {
      data.url = data.url.split('?')[0];
    }
    return data;
  };
</script>
```

## Troubleshooting

### Issue: Speed Insights script not loading

**Solution:**
1. Verify deployment completed successfully
2. Check browser console for errors
3. Ensure `/_vercel/speed-insights/script.js` is accessible
4. Verify correct script tags in HTML

```html
<!-- Check both script tags are present -->
<script>
  window.si = window.si || function () { (window.siq = window.siq || []).push(arguments); };
</script>
<script defer src="/_vercel/speed-insights/script.js"></script>
```

### Issue: No metrics showing in dashboard

**Solution:**
1. Wait 24-48 hours for initial metrics
2. Ensure actual users are visiting the site
3. Check Speed Insights is enabled in Vercel dashboard
4. Verify page visits are from the deployed domain

### Issue: High metrics after deployment

**Suggested optimizations:**
1. Reduce bundle size of JavaScript
2. Optimize images and assets
3. Implement caching strategy
4. Reduce API response times
5. Use CDN for static files

## Environment-Specific Configuration

### Local Development

Speed Insights won't track metrics locally (expected). To test:

```bash
vercel dev
```

This simulates the production environment locally.

### Staging Environment

Test Speed Insights before production:

```bash
vercel deploy --prod=false
```

### Production Environment

Deploy to production:

```bash
vercel deploy --prod
```

## Integration with CI/CD

### GitHub Integration

1. Connect GitHub repository to Vercel
2. Enable automatic deployments on push/PR
3. Speed Insights will be available on all deployments

### Pre-deployment Checks

```bash
# Install dependencies
npm install

# Run linter
npm run lint

# Run tests (if available)
npm run test

# Build frontend
npm run build

# Deploy
vercel deploy --prod
```

## Future Enhancements

### Planned Improvements

1. **Real-time Monitoring Dashboard**: Custom dashboard showing live metrics
2. **Performance Budgets**: Set and monitor performance targets
3. **Alerting System**: Notifications when metrics degrade
4. **Custom Metrics**: Track flood prediction specific metrics
5. **A/B Testing Integration**: Measure performance impact of changes

### Custom Metrics Example

```javascript
// Track custom flood prediction metrics
window.si('mark', 'flood-prediction-start');

// ... perform prediction ...

window.si('mark', 'flood-prediction-end');
window.si('measure', 'flood-prediction', 'flood-prediction-start', 'flood-prediction-end');
```

## Resources

- [Vercel Speed Insights Documentation](https://vercel.com/docs/speed-insights)
- [Web Vitals Guide](https://web.dev/vitals/)
- [Core Web Vitals Best Practices](https://web.dev/vitals-measurement-getting-started/)
- [Performance Optimization](https://vercel.com/docs/concepts/performance)

## Support

For issues or questions about Speed Insights integration:

1. Check Vercel Documentation: https://vercel.com/docs/speed-insights
2. Review Web.dev guides: https://web.dev/
3. Contact Vercel Support: https://vercel.com/support

## Summary

Vercel Speed Insights is now integrated into the Floodingnaque project. After deployment to Vercel, you'll be able to monitor real-world performance metrics, identify optimization opportunities, and ensure excellent user experience for your flood prediction dashboard.

Key benefits:
- ✅ Monitor real user performance
- ✅ Identify performance bottlenecks
- ✅ Track Core Web Vitals
- ✅ Make data-driven optimization decisions
- ✅ Ensure responsive dashboard for critical flood alerts
