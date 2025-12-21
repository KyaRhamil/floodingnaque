# Vercel Speed Insights Integration Guide

This guide will help you get started with using Vercel Speed Insights on the Floodingnaque project, showing you how to enable it, add the package to your project, deploy your app to Vercel, and view your data in the dashboard.

## Prerequisites

- A Vercel account. If you don't have one, you can [sign up for free](https://vercel.com/signup).
- A Vercel project. If you don't have one, you can [create a new project](https://vercel.com/new).
- The Vercel CLI installed. If you don't have it, you can install it using the following command:

```bash
# Using pnpm
pnpm i vercel

# Using yarn
yarn i vercel

# Using npm
npm i vercel

# Using bun
bun i vercel
```

## Step 1: Enable Speed Insights in Vercel

1. Go to the [Vercel dashboard](/dashboard)
2. Select your Project
3. Click on the **Speed Insights** tab
4. Select **Enable** from the dialog

> **💡 Note:** Enabling Speed Insights will add new routes (scoped at `/_vercel/speed-insights/*`) after your next deployment.

## Step 2: Add `@vercel/speed-insights` to your project

The `@vercel/speed-insights` package is already included in the project's `package.json`:

```json
{
  "dependencies": {
    "@vercel/speed-insights": "^1.3.1"
  }
}
```

If it's not already installed, you can add it using:

```bash
# Using pnpm
pnpm i @vercel/speed-insights

# Using yarn
yarn i @vercel/speed-insights

# Using npm
npm i @vercel/speed-insights

# Using bun
bun i @vercel/speed-insights
```

## Step 3: Integrate Speed Insights into your application

Since this project uses static HTML with JavaScript, follow the HTML implementation approach.

### For Static HTML Files

Add the following scripts before the closing `</body>` tag in your HTML files:

#### In `frontend/index.html`:

```html
<script>
  window.si = window.si || function () { (window.siq = window.siq || []).push(arguments); };
</script>
<script defer src="/_vercel/speed-insights/script.js"></script>
</body>
</html>
```

#### In `frontend/alert.html`:

```html
<script>
  window.si = window.si || function () { (window.siq = window.siq || []).push(arguments); };
</script>
<script defer src="/_vercel/speed-insights/script.js"></script>
</body>
</html>
```

### Optional: Remove Sensitive Information from URLs

You can optionally add a `speedInsightsBeforeSend` function to your HTML to remove sensitive information before sending data to Vercel:

```html
<script>
  window.speedInsightsBeforeSend = function(data) {
    // Remove or modify sensitive query parameters
    if (data.url && data.url.includes('?')) {
      data.url = data.url.split('?')[0];
    }
    return data;
  };

  window.si = window.si || function () { (window.siq = window.siq || []).push(arguments); };
</script>
<script defer src="/_vercel/speed-insights/script.js"></script>
```

## Step 4: Deploy your app to Vercel

You can deploy your app to Vercel's global CDN by running:

```bash
vercel deploy
```

Alternatively, you can [connect your project's git repository](/docs/git#deploying-a-git-repository), which will enable Vercel to deploy your latest pushes and merges to main.

Once your app is deployed, it's ready to begin tracking performance metrics.

> **💡 Note:** If everything is set up correctly, you should be able to find the `/_vercel/speed-insights/script.js` script inside the body tag of your page.

## Step 5: View your data in the dashboard

Once your app is deployed and users have visited your site, you can view the data in the dashboard:

1. Go to your [Vercel dashboard](/dashboard)
2. Select your project
3. Click the **Speed Insights** tab
4. After a few days of visitors, you'll be able to start exploring your metrics

For more information on how to use Speed Insights, see [Using Speed Insights](/docs/speed-insights/using-speed-insights).

## Important Notes for Floodingnaque Project

### Backend Integration (Flask)

If you're serving your frontend through Flask, ensure that:

1. Static files are properly served by Flask
2. The `/_vercel/speed-insights/script.js` endpoint is accessible (Vercel handles this automatically when deployed)
3. CORS headers don't block the Speed Insights beacon requests

Example Flask configuration:

```python
from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/alerts')
def alerts():
    return render_template('alert.html')
```

### Performance Optimization Tips

1. **Minimize Bundle Size**: Keep your JavaScript bundles small
2. **Cache Strategy**: Leverage browser caching for static assets
3. **API Optimization**: Ensure your Flask API endpoints respond quickly
4. **Image Optimization**: Optimize all images in your dashboard
5. **Code Splitting**: Load only necessary JavaScript on each page

## Environment-Specific Deployment

### Local Development

When developing locally, Speed Insights won't track metrics. This is expected behavior. To test Speed Insights:

```bash
vercel deploy --prod
```

### Staging Environment

Create a staging deployment to test Speed Insights before production:

```bash
vercel deploy --prod=false
```

### Production Deployment

Deploy to production when ready:

```bash
vercel deploy --prod
```

## Troubleshooting

### Speed Insights script not loading

1. Check browser console for errors
2. Ensure Vercel deployment is complete
3. Verify `/_vercel/speed-insights/script.js` is accessible
4. Check that CORS headers are properly configured

### No data showing in dashboard

1. Wait 24-48 hours for initial data to appear
2. Ensure users are actually visiting the deployed site
3. Check that Speed Insights is enabled in your project
4. Verify the script tag is properly added to your HTML

### Performance issues

1. Use Speed Insights data to identify bottlenecks
2. Monitor Core Web Vitals (LCP, FID, CLS)
3. Optimize API response times
4. Review network waterfall in Speed Insights dashboard

## Next Steps

Now that you have Vercel Speed Insights set up, you can explore the following topics:

- [Learn how to use the `@vercel/speed-insights` package](/docs/speed-insights/package)
- [Learn about metrics](/docs/speed-insights/metrics)
- [Read about privacy and compliance](/docs/speed-insights/privacy-policy)
- [Explore pricing](/docs/speed-insights/limits-and-pricing)
- [Troubleshooting](/docs/speed-insights/troubleshooting)

## Additional Resources

- [Vercel Speed Insights Documentation](https://vercel.com/docs/speed-insights)
- [Core Web Vitals Guide](https://web.dev/vitals/)
- [Performance Best Practices](https://vercel.com/docs/concepts/performance)

Learn more about how Vercel supports [privacy and data compliance standards](/docs/speed-insights/privacy-policy) with Vercel Speed Insights.
