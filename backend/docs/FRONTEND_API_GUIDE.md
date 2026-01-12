# Frontend API Integration Guide

Complete guide for integrating frontend applications with the Floodingnaque API.

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Overview](#api-overview)
3. [Authentication](#authentication)
4. [Core Endpoints](#core-endpoints)
5. [Real-time Subscriptions](#real-time-subscriptions)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Best Practices](#best-practices)
9. [Code Examples](#code-examples)

---

## Quick Start

### Base URL

```
Development: http://localhost:5000
Production:  https://api.floodingnaque.com
```

### Required Headers

```javascript
const headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  // For authenticated requests:
  'Authorization': 'Bearer <access_token>',
  // Or API key authentication:
  'X-API-Key': '<your-api-key>'
};
```

### Minimal Example

```javascript
// Fetch flood prediction
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-api-key'
  },
  body: JSON.stringify({
    temperature: 303.15,    // Kelvin (30°C)
    humidity: 85,           // Percentage
    precipitation: 50,      // mm/hour
    wind_speed: 15,         // m/s
    pressure: 1005          // hPa
  })
});

const data = await response.json();
console.log(data.risk_label); // "Alert" or "Critical" or "Safe"
```

---

## API Overview

### Response Format

All responses follow a consistent structure:

**Success Response:**
```json
{
  "success": true,
  "data": { ... },
  "request_id": "abc123"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "ValidationError",
    "title": "Validation Failed",
    "status": 400,
    "detail": "Temperature is required"
  },
  "request_id": "abc123"
}
```

### Available Endpoints

| Category | Base Path | Description |
|----------|-----------|-------------|
| Health | `/` | API status and health checks |
| Prediction | `/predict` | Flood risk predictions |
| Data | `/data` | Historical weather data |
| Alerts | `/api/alerts` | Alert history and management |
| Users | `/api/users` | Authentication and user management |
| Webhooks | `/api/webhooks` | Webhook management |
| SSE | `/sse` | Real-time alert streaming |

---

## Authentication

The API supports two authentication methods:

### 1. JWT Authentication (Recommended for Users)

```javascript
// Login to get tokens
const loginResponse = await fetch('/api/users/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'user@example.com',
    password: 'SecurePassword123!'
  })
});

const { access_token, refresh_token } = await loginResponse.json();

// Use access token for API requests
const headers = {
  'Authorization': `Bearer ${access_token}`
};
```

### 2. API Key Authentication (Recommended for Services)

```javascript
const headers = {
  'X-API-Key': 'your-api-key'
};
```

### Token Refresh

See [AUTH_FLOW.md](AUTH_FLOW.md) for detailed token refresh implementation.

```javascript
// Refresh access token
const refreshResponse = await fetch('/api/users/refresh', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    refresh_token: storedRefreshToken
  })
});

const { access_token } = await refreshResponse.json();
```

---

## Core Endpoints

### Health & Status

#### Root Endpoint
```http
GET /
```

Returns API information and available endpoints.

```json
{
  "name": "Floodingnaque API",
  "version": "2.0.0",
  "endpoints": {
    "status": "/status",
    "health": "/health",
    "predict": "/predict",
    "data": "/data"
  }
}
```

#### Status Check
```http
GET /status
```

Quick health check with database and model status.

```json
{
  "status": "running",
  "database": "healthy",
  "model": "loaded",
  "model_version": "v2.0"
}
```

#### Detailed Health
```http
GET /health
```

Comprehensive health check including all dependencies.

---

### Flood Prediction

#### Make Prediction
```http
POST /predict
Authorization: Bearer <token> or X-API-Key: <key>
```

**Request Body:**
```json
{
  "temperature": 303.15,
  "humidity": 85,
  "precipitation": 50,
  "wind_speed": 15,
  "pressure": 1005
}
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `risk_level` | bool | true | Include 3-level risk classification |
| `return_proba` | bool | false | Include probability values |

**Response:**
```json
{
  "prediction": 1,
  "flood_risk": "high",
  "risk_level": 2,
  "risk_label": "Critical",
  "risk_color": "#d32f2f",
  "risk_description": "Severe flood risk. Immediate precautions recommended.",
  "probability": 0.87,
  "confidence": 0.92,
  "model_version": "v2.0",
  "cache_hit": false,
  "request_id": "abc123"
}
```

**Risk Levels:**
| Level | Label | Color | Threshold |
|-------|-------|-------|-----------|
| 0 | Safe | Green (#4caf50) | probability < 0.3 |
| 1 | Alert | Orange (#ff9800) | 0.3 ≤ probability < 0.7 |
| 2 | Critical | Red (#d32f2f) | probability ≥ 0.7 |

---

### Weather Data

#### Get Weather Data
```http
GET /data
```

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 100 | Max records (1-1000) |
| `offset` | int | 0 | Records to skip |
| `start_date` | string | - | Filter after date (ISO format) |
| `end_date` | string | - | Filter before date (ISO format) |
| `sort_by` | string | timestamp | Sort field |
| `order` | string | desc | Sort order (asc/desc) |
| `source` | string | - | Filter by source (OWM, Manual, Meteostat) |

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "temperature": 303.15,
      "humidity": 85,
      "precipitation": 50,
      "wind_speed": 15,
      "pressure": 1005,
      "source": "OWM",
      "timestamp": "2024-01-15T10:00:00Z"
    }
  ],
  "total": 1234,
  "limit": 100,
  "offset": 0,
  "count": 100
}
```

#### Get Hourly Weather
```http
GET /data/weather/hourly
```

**Query Parameters:**
| Parameter | Type | Default |
|-----------|------|---------|
| `lat` | float | Default location |
| `lon` | float | Default location |
| `days` | int | 7 |

---

### Alerts

#### Get Alerts
```http
GET /api/alerts
```

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | int | Max alerts (max 500) |
| `offset` | int | Pagination offset |
| `risk_level` | int | Filter by risk (0, 1, 2) |
| `status` | string | Filter by status (delivered/pending/failed) |
| `start_date` | string | Filter after date |
| `end_date` | string | Filter before date |

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "risk_level": 2,
      "risk_label": "Critical",
      "location": "Parañaque, NCR",
      "message": "Severe flooding detected",
      "delivery_status": "delivered",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 50,
  "count": 10
}
```

#### Get Alert by ID
```http
GET /api/alerts/{id}
```

#### Get Alert History
```http
GET /api/alerts/history?days=7
```

Returns alert history with summary statistics.

---

### Webhooks

#### Register Webhook
```http
POST /api/webhooks/register
X-API-Key: <key>
```

**Request:**
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["flood_detected", "critical_risk", "high_risk"],
  "secret": "optional-custom-secret"
}
```

**Valid Events:**
- `flood_detected` - Any flood detection
- `critical_risk` - Risk level 2
- `high_risk` - Risk level 1+
- `medium_risk` - Risk level 1
- `low_risk` - Risk level 0

**Response:**
```json
{
  "success": true,
  "webhook_id": 1,
  "url": "https://your-server.com/webhook",
  "events": ["flood_detected", "critical_risk"],
  "secret": "generated-or-custom-secret"
}
```

#### List Webhooks
```http
GET /api/webhooks/list
X-API-Key: <key>
```

#### Toggle Webhook
```http
POST /api/webhooks/{id}/toggle
X-API-Key: <key>
```

#### Delete Webhook
```http
DELETE /api/webhooks/{id}
X-API-Key: <key>
```

#### Webhook Payload Format

When events occur, webhooks receive:

```json
{
  "event_type": "critical_risk",
  "data": {
    "alert_id": 123,
    "risk_level": 2,
    "risk_label": "Critical",
    "location": "Parañaque, NCR",
    "message": "Severe flooding detected",
    "prediction_id": 456
  },
  "timestamp": "2024-01-15T10:30:00Z",
  "delivery_attempt": 1
}
```

**Webhook Headers:**
```
Content-Type: application/json
X-Webhook-Signature: sha256=<signature>
X-Webhook-Event: critical_risk
X-Webhook-Delivery-Attempt: 1
X-Webhook-Timestamp: 2024-01-15T10:30:00Z
```

**Signature Verification:**
```javascript
const crypto = require('crypto');

function verifyWebhookSignature(payload, signature, secret) {
  const expected = 'sha256=' + crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');
  
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expected)
  );
}
```

---

## Real-time Subscriptions

### SSE Alert Stream

Connect to real-time alert notifications using Server-Sent Events.

```http
GET /sse/alerts
Accept: text/event-stream
```

See [REALTIME_GUIDE.md](REALTIME_GUIDE.md) for complete implementation details.

**Quick Example:**
```javascript
const eventSource = new EventSource('/sse/alerts');

eventSource.addEventListener('alert', (event) => {
  const data = JSON.parse(event.data);
  console.log('Alert:', data.alert);
});
```

---

## Error Handling

All errors follow RFC 7807 Problem Details format. See [ERROR_CODES.md](ERROR_CODES.md) for complete reference.

### Common Errors

| Code | Status | Action |
|------|--------|--------|
| ValidationError | 400 | Fix input data |
| UnauthorizedError | 401 | Refresh token or login |
| ForbiddenError | 403 | Check permissions |
| NotFoundError | 404 | Verify resource exists |
| RateLimitExceededError | 429 | Wait and retry |
| InternalServerError | 500 | Retry with backoff |

### Error Handler Example

```javascript
async function handleApiError(error) {
  const { status, code, detail, retry_after } = error;
  
  switch (status) {
    case 401:
      return await handleTokenRefresh();
    case 429:
      await delay(retry_after * 1000);
      return 'retry';
    case 500:
    case 502:
    case 503:
      showErrorNotification('Server error. Please try again.');
      break;
    default:
      showErrorNotification(detail);
  }
}
```

---

## Rate Limiting

### Default Limits

| Endpoint | Limit |
|----------|-------|
| `/predict` | 60/hour |
| `/data` | 100/minute |
| `/api/alerts` | 60/minute |
| `/api/users/login` | 10/minute |
| `/api/users/register` | 5/hour |
| `/sse/alerts/test` | 5/minute |

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705312200
```

### Handling Rate Limits

```javascript
api.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 429) {
      const retryAfter = error.response.headers['x-ratelimit-reset'];
      const waitTime = retryAfter - Date.now() / 1000;
      
      showNotification(`Rate limited. Wait ${Math.ceil(waitTime)} seconds.`);
      
      return new Promise(resolve => {
        setTimeout(() => resolve(api(error.config)), waitTime * 1000);
      });
    }
    return Promise.reject(error);
  }
);
```

---

## Best Practices

### 1. Use Axios/Fetch Wrapper

```javascript
// api/client.js
import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor for auth
api.interceptors.request.use(config => {
  const token = getAccessToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor for errors
api.interceptors.response.use(
  response => response.data,
  error => handleApiError(error)
);

export default api;
```

### 2. Type Definitions (TypeScript)

```typescript
// types/api.ts
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: ApiError;
  request_id: string;
}

interface ApiError {
  code: string;
  title: string;
  status: number;
  detail: string;
  errors?: FieldError[];
}

interface PredictionRequest {
  temperature: number;  // Kelvin
  humidity: number;     // 0-100
  precipitation: number;
  wind_speed?: number;
  pressure?: number;
}

interface PredictionResponse {
  prediction: number;
  flood_risk: 'high' | 'low';
  risk_level?: number;
  risk_label?: string;
  risk_color?: string;
  probability?: number;
  confidence?: number;
}

interface Alert {
  id: number;
  risk_level: number;
  risk_label: string;
  location: string;
  message: string;
  created_at: string;
}
```

### 3. Caching Strategy

```javascript
// Use React Query or similar
import { useQuery } from 'react-query';

function useWeatherData(params) {
  return useQuery(
    ['weatherData', params],
    () => api.get('/data', { params }),
    {
      staleTime: 60000,      // 1 minute
      cacheTime: 300000,     // 5 minutes
      retry: 3,
      retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 30000)
    }
  );
}
```

### 4. Request Debouncing

```javascript
import { debounce } from 'lodash';

const debouncedPredict = debounce(async (data) => {
  return await api.post('/predict', data);
}, 300);
```

### 5. Optimistic Updates

```javascript
// Show immediate UI update, then sync with server
async function toggleWebhook(id) {
  // Optimistic update
  setWebhooks(prev => prev.map(w => 
    w.id === id ? { ...w, is_active: !w.is_active } : w
  ));
  
  try {
    await api.post(`/api/webhooks/${id}/toggle`);
  } catch (error) {
    // Revert on failure
    setWebhooks(prev => prev.map(w => 
      w.id === id ? { ...w, is_active: !w.is_active } : w
    ));
    showError('Failed to toggle webhook');
  }
}
```

---

## Code Examples

### Complete React Integration

```jsx
// App.jsx
import React from 'react';
import { QueryClient, QueryClientProvider } from 'react-query';
import { AuthProvider } from './hooks/useAuth';
import { AlertProvider } from './hooks/useAlerts';
import Dashboard from './components/Dashboard';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 60000,
      refetchOnWindowFocus: false
    }
  }
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <AlertProvider>
          <Dashboard />
        </AlertProvider>
      </AuthProvider>
    </QueryClientProvider>
  );
}
```

### Prediction Component

```jsx
// components/PredictionForm.jsx
import React, { useState } from 'react';
import { useMutation } from 'react-query';
import api from '../api/client';

function PredictionForm() {
  const [formData, setFormData] = useState({
    temperature: '',
    humidity: '',
    precipitation: '',
    wind_speed: '',
    pressure: ''
  });

  const mutation = useMutation(
    (data) => api.post('/predict?risk_level=true', data),
    {
      onSuccess: (result) => {
        setPrediction(result);
      },
      onError: (error) => {
        showError(error.detail || 'Prediction failed');
      }
    }
  );

  const handleSubmit = (e) => {
    e.preventDefault();
    
    // Convert temperature to Kelvin if needed
    const data = {
      ...formData,
      temperature: parseFloat(formData.temperature) + 273.15
    };
    
    mutation.mutate(data);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="number"
        placeholder="Temperature (°C)"
        value={formData.temperature}
        onChange={e => setFormData({...formData, temperature: e.target.value})}
        required
      />
      {/* ... other fields ... */}
      <button type="submit" disabled={mutation.isLoading}>
        {mutation.isLoading ? 'Predicting...' : 'Get Prediction'}
      </button>
    </form>
  );
}
```

### Data Table Component

```jsx
// components/WeatherDataTable.jsx
import React from 'react';
import { useQuery } from 'react-query';
import api from '../api/client';

function WeatherDataTable({ page = 0, pageSize = 20 }) {
  const { data, isLoading, error } = useQuery(
    ['weatherData', page, pageSize],
    () => api.get('/data', {
      params: {
        limit: pageSize,
        offset: page * pageSize,
        sort_by: 'timestamp',
        order: 'desc'
      }
    }),
    { keepPreviousData: true }
  );

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <table>
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Temperature</th>
          <th>Humidity</th>
          <th>Precipitation</th>
          <th>Source</th>
        </tr>
      </thead>
      <tbody>
        {data.data.map(row => (
          <tr key={row.id}>
            <td>{new Date(row.timestamp).toLocaleString()}</td>
            <td>{(row.temperature - 273.15).toFixed(1)}°C</td>
            <td>{row.humidity}%</td>
            <td>{row.precipitation} mm</td>
            <td>{row.source}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
```

---

## Environment Configuration

```env
# .env.local
REACT_APP_API_URL=http://localhost:5000
REACT_APP_SSE_URL=http://localhost:5000/sse
REACT_APP_API_KEY=your-api-key-for-dev
```

```javascript
// config.js
export const config = {
  apiUrl: process.env.REACT_APP_API_URL || 'http://localhost:5000',
  sseUrl: process.env.REACT_APP_SSE_URL || 'http://localhost:5000/sse',
  apiKey: process.env.REACT_APP_API_KEY
};
```

---

## See Also

- [AUTH_FLOW.md](AUTH_FLOW.md) - JWT authentication details
- [ERROR_CODES.md](ERROR_CODES.md) - Complete error reference
- [REALTIME_GUIDE.md](REALTIME_GUIDE.md) - SSE implementation guide
- Swagger UI: `{API_URL}/api/docs`
