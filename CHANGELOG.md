# Changelog

All notable changes to the Floodingnaque project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Feature flags system for gradual rollout of new prediction models
- Emergency bypass flags for external API calls (OpenWeatherMap, Weatherstack, WorldTides)
- A/B testing support for alert threshold experiments
- Pre-commit hooks configuration (black, isort, flake8, mypy, bandit)
- Environment variable validation on startup
- Health check endpoint response time SLA monitoring
- Rate limit bypass for internal services
- Feature flags management API endpoints

### Changed
- Enhanced configuration validation at application startup

### Security
- Added admin-only access controls for feature flag management

## [2.0.0] - 2026-01-10

### Added
- Complete API overhaul with Flask blueprints architecture
- JWT authentication system with refresh tokens
- Role-based access control (RBAC)
- API versioning (v1 prefix)
- GraphQL endpoint for flexible queries
- Server-Sent Events (SSE) for real-time alerts
- Comprehensive rate limiting with tiered API key support
- IP reputation system for rate limiting
- Distributed tracing and correlation IDs
- Prometheus metrics integration
- Sentry error tracking integration
- Database connection pooling with PgBouncer support
- Time-series partitioning for historical data
- Read replica support for analytics queries
- Circuit breaker pattern for external APIs
- Webhook support for alert notifications
- Batch prediction endpoint
- Data export functionality (CSV, JSON)
- Performance monitoring dashboard
- Comprehensive health check endpoints (Kubernetes-ready)
- Security headers middleware (HSTS, CSP, CORS)
- Request/response logging with correlation
- Model versioning and A/B testing infrastructure

### Changed
- Migrated from SQLite to Supabase PostgreSQL for production
- Restructured project layout with modular blueprints
- Enhanced error handling with RFC 7807 Problem Details format
- Improved logging with JSON and ECS format support
- Updated all dependencies to latest stable versions

### Security
- Implemented API key authentication with entropy validation
- Added brute force protection for authentication endpoints
- Enforced HTTPS/TLS for production deployments
- Added Content Security Policy (CSP) headers
- Implemented security.txt endpoint (RFC 9116)
- Database SSL/TLS with certificate verification
- Encryption at rest for sensitive data

### Fixed
- Connection pool exhaustion under high load
- Memory leaks in long-running prediction jobs
- Race conditions in concurrent data ingestion
- Timezone handling in weather data processing

## [1.5.0] - 2025-09-15

### Added
- Satellite precipitation data integration (GPM, CHIRPS)
- Earth Engine integration for geospatial analysis
- WorldTides API integration for tidal data
- Enhanced prediction model with ensemble methods
- Historical flood records from Parañaque City
- PAGASA weather data preprocessing pipeline
- Model evaluation and comparison scripts
- Thesis report generation utilities

### Changed
- Improved model accuracy with feature engineering
- Enhanced data validation and preprocessing
- Updated weather data sources and fallback logic

### Fixed
- Meteostat station ID detection
- Weather data caching inconsistencies
- Model retraining scheduler reliability

## [1.0.0] - 2025-06-01

### Added
- Initial release of Floodingnaque flood prediction API
- Random Forest prediction model for flood risk classification
- OpenWeatherMap integration for real-time weather data
- Weatherstack integration as fallback weather source
- Meteostat integration for historical weather data
- Basic REST API for predictions and data ingestion
- SQLite database for development
- Scheduled data ingestion and model retraining
- Basic alert system with configurable thresholds
- Docker support for containerized deployment
- Comprehensive API documentation with Swagger

### Security
- Basic API key authentication
- Rate limiting for API endpoints
- Input validation and sanitization

---

## Version History Summary

| Version | Date | Description |
|---------|------|-------------|
| 2.0.0 | 2026-01-10 | Major overhaul with production-ready architecture |
| 1.5.0 | 2025-09-15 | Satellite data and enhanced predictions |
| 1.0.0 | 2025-06-01 | Initial release |

## Migration Guides

### Upgrading from 1.x to 2.0

1. **Database Migration**: Run Alembic migrations to update schema
   ```bash
   cd backend
   alembic upgrade head
   ```

2. **Environment Variables**: Review `.env.example` for new required variables
   - `JWT_SECRET_KEY` - Required for authentication
   - `SUPABASE_URL`, `SUPABASE_KEY` - Required for database
   - `SENTRY_DSN` - Recommended for error tracking

3. **API Endpoints**: All API endpoints now use `/api/v1/` prefix
   - Old: `/predict` → New: `/api/v1/predict/`
   - Old: `/data` → New: `/api/v1/data/`

4. **Authentication**: API key format and validation updated
   - Generate new keys with sufficient entropy
   - Update client applications with new keys

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Links

- [Project Repository](https://github.com/floodingnaque/floodingnaque)
- [Issue Tracker](https://github.com/floodingnaque/floodingnaque/issues)
- [Documentation](https://docs.floodingnaque.com)

[Unreleased]: https://github.com/floodingnaque/floodingnaque/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/floodingnaque/floodingnaque/compare/v1.5.0...v2.0.0
[1.5.0]: https://github.com/floodingnaque/floodingnaque/compare/v1.0.0...v1.5.0
[1.0.0]: https://github.com/floodingnaque/floodingnaque/releases/tag/v1.0.0
