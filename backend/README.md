# Flooding Naque Backend API

A Flask-based REST API for flood prediction using machine learning and weather data ingestion.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the `backend/` directory:

```env
DATABASE_URL=sqlite:///floodingnaque.db
OWM_API_KEY=your_openweathermap_api_key_here
METEOSTAT_API_KEY=your_weatherstack_api_key_here
PORT=5000
HOST=0.0.0.0
FLASK_DEBUG=False
```

### 3. Train the Model (First Time Only)

```bash
python scripts/train.py
```

### 4. Start the Server

**Development:**
```bash
python main.py
```

**Production:**
```bash
gunicorn --bind 0.0.0.0:$PORT --workers 4 --threads 2 --timeout 120 main:app
```

## API Endpoints

### Base URL
```
http://localhost:5000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/status` | Basic health check |
| GET | `/health` | Detailed health check |
| GET/POST | `/ingest` | Ingest weather data (GET shows usage) |
| GET | `/data` | Retrieve historical weather data |
| POST | `/predict` | Predict flood risk |
| GET | `/api/docs` | API documentation |
| GET | `/api/version` | API version |

## Example Usage

### Ingest Weather Data

```bash
curl -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{"lat": 14.6, "lon": 120.98}'
```

### Get Historical Data

```bash
curl http://localhost:5000/data?limit=10
```

### Predict Flood Risk

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 298.15, "humidity": 65.0, "precipitation": 5.0}'
```

## Frontend Integration

The API is CORS-enabled and ready for frontend integration. All endpoints return JSON responses with consistent error handling.

### Response Format

**Success:**
```json
{
  "data": {...},
  "request_id": "uuid-string"
}
```

**Error:**
```json
{
  "error": "Error message",
  "request_id": "uuid-string"
}
```

## Documentation

- Full API documentation: `http://localhost:5000/api/docs`
- Database setup: See `docs/DATABASE_SETUP.md`
- Complete guide: See `docs/BACKEND_COMPLETE.md`

## Project Structure

```
backend/
├── main.py             # Application entry point
├── app/                # Main application code
│   ├── __init__.py
│   ├── api/            # API routes and schemas
│   ├── core/           # Core functionality (config, logging)
│   ├── services/       # Business logic (weather, prediction, alerts)
│   ├── models/         # Data models
│   └── utils/          # Utilities
├── scripts/            # Utility scripts (training, validation, etc.)
├── tests/              # Test files
├── docs/               # Documentation
├── data/               # Data files
├── models/             # ML models
├── logs/               # Log files
├── evaluation_results/  # Evaluation outputs
├── requirements.txt    # Python dependencies
├── Procfile           # Production deployment config
└── Dockerfile         # Docker configuration
```

## Features

- ✅ RESTful API with comprehensive endpoints
- ✅ Machine learning flood prediction
- ✅ Weather data ingestion (OpenWeatherMap + Weatherstack)
- ✅ Historical data retrieval with pagination
- ✅ Request ID tracking for debugging
- ✅ CORS support for frontend
- ✅ Comprehensive error handling
- ✅ Production-ready configuration

## License

MIT

