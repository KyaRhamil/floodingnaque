# Frontend Integration Guide

## ✅ Backend Status: COMPLETE & READY

The backend is fully implemented and ready for frontend integration.

## API Base URL

```
http://localhost:5000
```

## Available Endpoints

### 1. Root Endpoint
- **GET** `/` - API information and endpoint list

### 2. Health Checks
- **GET** `/status` - Basic health check
- **GET** `/health` - Detailed health check

### 3. Weather Data
- **GET/POST** `/ingest` - Ingest weather data (GET shows usage info)
- **GET** `/data` - Retrieve historical weather data with pagination

### 4. Predictions
- **POST** `/predict` - Predict flood risk

### 5. Documentation
- **GET** `/api/docs` - Full API documentation
- **GET** `/api/version` - API version info

## Response Format

All endpoints return JSON with consistent structure:

### Success Response
```json
{
  "data": {...},
  "request_id": "uuid-string"
}
```

### Error Response
```json
{
  "error": "Error message",
  "request_id": "uuid-string"
}
```

## CORS Configuration

✅ CORS is enabled for all origins. Your frontend can make requests from any domain.

## Example Frontend Integration

### JavaScript/TypeScript

```javascript
const API_BASE_URL = 'http://localhost:5000';

// Ingest weather data
async function ingestWeatherData(lat, lon) {
  const response = await fetch(`${API_BASE_URL}/ingest`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ lat, lon })
  });
  return await response.json();
}

// Get historical data
async function getHistoricalData(limit = 100, offset = 0) {
  const response = await fetch(`${API_BASE_URL}/data?limit=${limit}&offset=${offset}`);
  return await response.json();
}

// Predict flood risk
async function predictFlood(temperature, humidity, precipitation) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ temperature, humidity, precipitation })
  });
  return await response.json();
}

// Health check
async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  return await response.json();
}
```

### React Example

```jsx
import { useState, useEffect } from 'react';

function WeatherDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:5000/data?limit=10')
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error:', err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div>Loading...</div>;
  if (!data) return <div>No data</div>;

  return (
    <div>
      <h1>Weather Data</h1>
      <p>Total records: {data.total}</p>
      {data.data.map(item => (
        <div key={item.id}>
          <p>Temp: {item.temperature}°K</p>
          <p>Humidity: {item.humidity}%</p>
          <p>Precipitation: {item.precipitation}mm</p>
        </div>
      ))}
    </div>
  );
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200` - Success
- `400` - Bad Request (validation errors)
- `404` - Not Found (model file, etc.)
- `500` - Internal Server Error

Always check the response status and handle errors:

```javascript
async function safeApiCall(url, options) {
  try {
    const response = await fetch(url, options);
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || 'API request failed');
    }
    
    return data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
}
```

## Request ID Tracking

Every response includes a `request_id` field. Use this for:
- Debugging
- Logging
- Error tracking
- Support requests

## Testing the API

### Using Browser
- Visit `http://localhost:5000/` for API info
- Visit `http://localhost:5000/api/docs` for full documentation
- Visit `http://localhost:5000/ingest` to see usage instructions

### Using curl
```bash
# Health check
curl http://localhost:5000/health

# Ingest data
curl -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{"lat": 14.6, "lon": 120.98}'

# Get data
curl http://localhost:5000/data?limit=10

# Predict
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 298.15, "humidity": 65.0, "precipitation": 5.0}'
```

## Data Models

### Weather Data Record
```typescript
interface WeatherData {
  id: number;
  temperature: number;      // Kelvin
  humidity: number;         // Percentage
  precipitation: number;     // mm
  timestamp: string;         // ISO datetime
}
```

### Prediction Request
```typescript
interface PredictionRequest {
  temperature: number;      // Required
  humidity: number;         // Required
  precipitation: number;     // Required
}
```

### Prediction Response
```typescript
interface PredictionResponse {
  prediction: 0 | 1;        // 0 = no flood, 1 = flood
  flood_risk: 'low' | 'high';
  request_id: string;
}
```

## Next Steps

1. ✅ Backend is complete and tested
2. ✅ CORS is enabled
3. ✅ All endpoints are documented
4. ✅ Error handling is consistent
5. 🚀 **Ready to build frontend!**

## Support

- API Documentation: `http://localhost:5000/api/docs`
- Backend README: `README.md`
- Complete Guide: `BACKEND_COMPLETE.md`

