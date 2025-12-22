# Frontend Integration Guide

## ✅ Backend Status: COMPLETE & READY

The backend is fully implemented and ready for frontend integration.

## ⚠️ Frontend Status: SCAFFOLDING ONLY (As of Dec 2024)

The frontend folder contains **empty directory structures** with no implementation files yet.

### Existing Folder Structure

```
frontend/
└── src/
    ├── public/              (empty)
    ├── scripts/             (empty)
    └── app/
        ├── admin/           (empty)
        ├── assets/fonts|icons|images/  (empty)
        ├── components/charts|feedback|map|tables|ui/  (empty)
        ├── config/          (empty)
        ├── features/
        │   ├── auth/        (empty - components/hooks/services)
        │   ├── flooding/    (empty - components/hooks/services/utils)
        │   ├── reports/     (empty - components/hooks/services)
        │   └── settings/    (empty - components/hooks/services)
        ├── hooks/           (empty)
        ├── lib/             (empty)
        ├── map/             (empty)
        ├── reports/         (empty)
        ├── state/stores/    (empty)
        ├── styles/          (empty)
        ├── tests/e2e|integration|unit/  (empty)
        └── types/api|domain/  (empty)
```

### Required Setup Files (Not Yet Created)

| File | Purpose | Priority |
|------|---------|----------|
| `frontend/package.json` | Dependencies & scripts | 🔴 Critical |
| `frontend/tsconfig.json` | TypeScript configuration | 🔴 Critical |
| `frontend/vite.config.ts` | Build configuration (if using Vite) | 🔴 Critical |
| `frontend/index.html` | HTML entry point | 🔴 Critical |
| `frontend/src/main.tsx` | App entry point | 🔴 Critical |
| `frontend/src/App.tsx` | Root component | 🔴 Critical |
| `frontend/.env.example` | Environment variables template | 🟡 High |
| `frontend/.eslintrc.js` | Linting rules | 🟡 High |
| `frontend/tailwind.config.js` | Styling (if using Tailwind) | 🟢 Medium |

### Recommended Tech Stack

- **Framework**: React 18+ with TypeScript or Next.js 14+
- **Build Tool**: Vite
- **State Management**: Zustand or TanStack Query (React Query)
- **Styling**: Tailwind CSS + shadcn/ui components
- **Maps**: Leaflet.js for flood visualization (Parañaque City)
- **Charts**: Recharts for weather/prediction data visualization
- **HTTP Client**: Axios or native fetch with React Query

### Implementation Priority

| Order | Feature | Target Folder |
|-------|---------|---------------|
| 1️⃣ | Project setup & configs | `frontend/` root |
| 2️⃣ | API types & client | `types/api/`, `lib/` |
| 3️⃣ | UI components | `components/ui/` |
| 4️⃣ | Authentication | `features/auth/` |
| 5️⃣ | Flood prediction form | `features/flooding/` |
| 6️⃣ | Map visualization | `components/map/` |
| 7️⃣ | Charts & dashboard | `components/charts/` |
| 8️⃣ | Historical data tables | `components/tables/` |
| 9️⃣ | Reports generation | `features/reports/` |

---

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
// Location: frontend/src/app/types/api/weather.ts
interface WeatherData {
  id: number;
  temperature: number;      // Kelvin
  humidity: number;         // Percentage
  precipitation: number;    // mm
  timestamp: string;        // ISO datetime
}
```

### Prediction Request
```typescript
// Location: frontend/src/app/types/api/prediction.ts
interface PredictionRequest {
  temperature: number;      // Required
  humidity: number;         // Required
  precipitation: number;    // Required
}
```

### Prediction Response
```typescript
// Location: frontend/src/app/types/api/prediction.ts
interface PredictionResponse {
  prediction: 0 | 1;        // 0 = no flood, 1 = flood
  flood_risk: 'low' | 'high';
  request_id: string;
}
```

### API Response Wrapper
```typescript
// Location: frontend/src/app/types/api/common.ts
interface ApiResponse<T> {
  data: T;
  request_id: string;
}

interface ApiError {
  error: string;
  request_id: string;
}

interface PaginatedResponse<T> {
  data: T[];
  total: number;
  limit: number;
  offset: number;
  request_id: string;
}
```

## Environment Variables

Create `frontend/.env.example`:
```env
VITE_API_BASE_URL=http://localhost:5000
VITE_APP_NAME=Floodingnaque
VITE_MAP_DEFAULT_LAT=14.4793
VITE_MAP_DEFAULT_LNG=121.0198
VITE_MAP_DEFAULT_ZOOM=13
```

## Next Steps

### Phase 1: Project Initialization
1. ⬜ Create `frontend/package.json` with dependencies
2. ⬜ Set up Vite + React + TypeScript
3. ⬜ Configure Tailwind CSS
4. ⬜ Create entry point files (`index.html`, `main.tsx`, `App.tsx`)

### Phase 2: Core Infrastructure
5. ⬜ Define TypeScript types in `types/api/` and `types/domain/`
6. ⬜ Create API client in `lib/api.ts`
7. ⬜ Set up React Query for data fetching
8. ⬜ Build reusable UI components in `components/ui/`

### Phase 3: Features
9. ⬜ Implement flood prediction form (`features/flooding/`)
10. ⬜ Add Leaflet map for Parañaque City (`components/map/`)
11. ⬜ Create weather data charts (`components/charts/`)
12. ⬜ Build historical data tables (`components/tables/`)

### Phase 4: Polish
13. ⬜ Add error boundaries and loading states
14. ⬜ Implement responsive design
15. ⬜ Write unit tests (`tests/unit/`)
16. ⬜ Add E2E tests (`tests/e2e/`)

### Backend Status
- ✅ Backend is complete and tested
- ✅ CORS is enabled for all origins
- ✅ All endpoints are documented
- ✅ Error handling is consistent

## Support

- API Documentation: `http://localhost:5000/api/docs`
- Backend README: `../README.md`
- Complete Guide: `BACKEND_COMPLETE.md`

