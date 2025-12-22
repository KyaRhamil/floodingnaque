# Floodingnaque Production Runbook

## Overview

This runbook provides operational procedures for managing the Floodingnaque flood prediction system in production.

**System Components:**
- Backend API (Flask + Gunicorn)
- Celery Workers (async task processing)
- Redis Cloud (caching, rate limiting, task queue)
- Supabase PostgreSQL (database)
- Datadog Agent (monitoring)
- Prometheus (metrics)

---

## Quick Reference

| Action | Command |
|--------|---------|
| Start all services | `docker-compose -f docker-compose-production.yml up -d` |
| Stop all services | `docker-compose -f docker-compose-production.yml down` |
| View logs | `docker-compose -f docker-compose-production.yml logs -f` |
| Restart backend | `docker-compose -f docker-compose-production.yml restart backend` |
| Health check | `curl http://localhost:5000/health` |

---

## 1. Startup Procedures

### 1.1 Full System Startup

```bash
# Navigate to project root
cd /path/to/floodingnaque

# Verify .env.production exists and is configured
cat backend/.env.production | grep -v "^#" | grep -v "^$"

# Build and start all services
docker-compose -f docker-compose-production.yml up -d --build

# Verify services are running
docker-compose -f docker-compose-production.yml ps

# Check backend health
curl -s http://localhost:5000/health | jq .

# Check detailed health
curl -s http://localhost:5000/health/detailed | jq .
```

### 1.2 Verify Startup Checklist

- [ ] Backend container running: `docker ps | grep floodingnaque-api-prod`
- [ ] Celery worker running: `docker ps | grep floodingnaque-celery`
- [ ] Health endpoint returns 200: `curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health`
- [ ] Database connected: Check `/health/detailed` response
- [ ] Redis connected: Check `/health/detailed` response
- [ ] ML model loaded: Check `/health/detailed` response

### 1.3 Post-Startup Validation

```bash
# Test prediction endpoint
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"temperature": 28, "humidity": 85, "rainfall": 50}'

# Check metrics endpoint
curl http://localhost:5000/metrics
```

---

## 2. Shutdown Procedures

### 2.1 Graceful Shutdown

```bash
# Stop services gracefully (allows in-flight requests to complete)
docker-compose -f docker-compose-production.yml stop

# Or with timeout (default 10s, extend for long-running tasks)
docker-compose -f docker-compose-production.yml stop -t 30
```

### 2.2 Full Shutdown (with cleanup)

```bash
# Stop and remove containers
docker-compose -f docker-compose-production.yml down

# Stop, remove containers AND volumes (CAUTION: data loss)
docker-compose -f docker-compose-production.yml down -v
```

### 2.3 Emergency Shutdown

```bash
# Force kill all containers immediately
docker-compose -f docker-compose-production.yml kill

# Remove stopped containers
docker-compose -f docker-compose-production.yml rm -f
```

---

## 3. Rollback Procedures

### 3.1 Rollback to Previous Version

```bash
# List available images
docker images | grep floodingnaque

# Stop current deployment
docker-compose -f docker-compose-production.yml down

# Tag current image as backup
docker tag floodingnaque-api:production-v2.0.0 floodingnaque-api:rollback-backup

# Pull/restore previous version
docker tag floodingnaque-api:production-v1.9.0 floodingnaque-api:production-v2.0.0

# Restart services
docker-compose -f docker-compose-production.yml up -d
```

### 3.2 Database Rollback

```bash
# List available backups
./backend/scripts/backup_database.sh --list

# Restore from backup (CAUTION: data will be overwritten)
./backend/scripts/backup_database.sh --restore /app/backups/floodingnaque_backup_YYYYMMDD_HHMMSS_full.sql.gz
```

### 3.3 Configuration Rollback

```bash
# If .env.production was changed, restore from backup
cp backend/.env.production.backup backend/.env.production

# Restart to apply
docker-compose -f docker-compose-production.yml restart backend
```

---

## 4. Common Issues & Resolutions

### 4.1 Backend Won't Start

**Symptoms:** Container exits immediately, health check fails

**Diagnosis:**
```bash
# Check container logs
docker-compose -f docker-compose-production.yml logs backend

# Check for configuration errors
docker-compose -f docker-compose-production.yml run --rm backend python -c "from app.core.config import Config; Config.validate()"
```

**Common Causes & Fixes:**

| Issue | Error Message | Fix |
|-------|---------------|-----|
| Missing SECRET_KEY | `CRITICAL: SECRET_KEY must be set` | Set SECRET_KEY in .env.production |
| Missing DATABASE_URL | `CRITICAL: DATABASE_URL must be set` | Configure Supabase connection string |
| SQLite in production | `SQLite is not allowed in production` | Use PostgreSQL DATABASE_URL |
| Invalid Redis URL | Connection refused | Verify REDIS_URL in .env.production |

### 4.2 Database Connection Issues

**Symptoms:** 500 errors, health check shows database unhealthy

**Diagnosis:**
```bash
# Test database connectivity
docker-compose -f docker-compose-production.yml exec backend python -c "
from app.core.database import engine
from sqlalchemy import text
with engine.connect() as conn:
    result = conn.execute(text('SELECT 1'))
    print('Database OK:', result.scalar())
"
```

**Fixes:**
1. Check Supabase dashboard for connection limits
2. Verify DATABASE_URL format
3. Check if IP is whitelisted in Supabase
4. Increase DB_POOL_SIZE if connection exhausted

### 4.3 High Memory Usage

**Symptoms:** Container OOM killed, slow responses

**Diagnosis:**
```bash
# Check container stats
docker stats floodingnaque-api-prod

# Check memory inside container
docker exec floodingnaque-api-prod ps aux --sort=-%mem
```

**Fixes:**
1. Reduce GUNICORN_WORKERS (each worker uses ~200MB)
2. Enable GUNICORN_MAX_REQUESTS to recycle workers
3. Check for memory leaks in ML model loading
4. Increase container memory limit in docker-compose

### 4.4 Rate Limiting Issues

**Symptoms:** 429 Too Many Requests

**Diagnosis:**
```bash
# Check Redis rate limit keys
docker-compose -f docker-compose-production.yml exec backend python -c "
import redis
r = redis.from_url('${REDIS_URL}')
for key in r.scan_iter('LIMITER:*'):
    print(key, r.ttl(key))
"
```

**Fixes:**
1. Increase RATE_LIMIT_DEFAULT in .env.production
2. Configure per-endpoint limits for high-traffic routes
3. Implement API key tiers for different rate limits

### 4.5 ML Model Not Loading

**Symptoms:** Prediction endpoint returns 500, health shows model unavailable

**Diagnosis:**
```bash
# Check model files exist
docker exec floodingnaque-api-prod ls -la /app/models/

# Try loading model manually
docker exec floodingnaque-api-prod python -c "
import joblib
model = joblib.load('/app/models/flood_rf_model.joblib')
print('Model loaded:', type(model))
"
```

**Fixes:**
1. Ensure model files are in the volume
2. Check model file permissions
3. Verify MODEL_DIR and MODEL_NAME in config
4. If REQUIRE_MODEL_SIGNATURE=True, ensure model is signed

---

## 5. Monitoring & Alerts

### 5.1 Health Check Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/health` | Basic liveness | `{"status": "healthy"}` |
| `/health/detailed` | Full system status | Includes DB, Redis, Model status |
| `/health/ready` | Kubernetes readiness | 200 if ready, 503 if not |
| `/metrics` | Prometheus metrics | Prometheus format |

### 5.2 Key Metrics to Monitor

- **Request latency:** P95 < 500ms for predictions
- **Error rate:** < 1% 5xx errors
- **Database connections:** Pool utilization < 80%
- **Memory usage:** < 80% of limit
- **CPU usage:** < 70% average

### 5.3 Log Locations

```bash
# Backend logs
docker-compose -f docker-compose-production.yml logs backend

# Specific time range
docker-compose -f docker-compose-production.yml logs --since 1h backend

# Follow logs
docker-compose -f docker-compose-production.yml logs -f backend

# Log file inside container
docker exec floodingnaque-api-prod cat /app/logs/floodingnaque.log
```

### 5.4 Sentry Error Tracking

If SENTRY_DSN is configured, errors are automatically reported.

Dashboard: https://sentry.io (check your organization)

### 5.5 Datadog Dashboard

If DD_API_KEY is configured:
- APM: https://app.datadoghq.com/apm
- Infrastructure: https://app.datadoghq.com/infrastructure

---

## 6. Backup & Recovery

### 6.1 Scheduled Backups

Add to crontab:
```bash
# Daily full backup at 2 AM (Manila time)
0 2 * * * /path/to/floodingnaque/backend/scripts/backup_database.sh >> /var/log/floodingnaque-backup.log 2>&1

# Weekly schema-only backup on Sunday at 3 AM
0 3 * * 0 /path/to/floodingnaque/backend/scripts/backup_database.sh --schema-only >> /var/log/floodingnaque-backup.log 2>&1
```

### 6.2 Manual Backup

```bash
# Create immediate backup
./backend/scripts/backup_database.sh

# Verify backup
./backend/scripts/backup_database.sh --verify /app/backups/latest_full.sql.gz
```

### 6.3 Disaster Recovery

1. **Stop services** to prevent further data changes
2. **Create backup** of current state (even if corrupted)
3. **Restore** from last known good backup
4. **Run migrations** if needed: `alembic upgrade head`
5. **Verify data** integrity
6. **Restart services**

---

## 7. Scaling Procedures

### 7.1 Vertical Scaling (Single Node)

Edit docker-compose-production.yml:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'      # Increase from 2
      memory: 4G     # Increase from 2G
```

Also increase:
- GUNICORN_WORKERS (2-4 per CPU core)
- DB_POOL_SIZE

### 7.2 Horizontal Scaling (Multiple Nodes)

For multi-node deployment:
1. Deploy behind a load balancer (nginx/Traefik)
2. Ensure all nodes connect to same Redis and PostgreSQL
3. Use sticky sessions if needed for WebSocket
4. Scale Celery workers independently

---

## 8. Security Procedures

### 8.1 Rotate Secrets

```bash
# Generate new secrets
python -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_hex(32))"
python -c "import secrets; print('API_KEY=' + secrets.token_urlsafe(32))"

# Update .env.production with new values

# Restart services
docker-compose -f docker-compose-production.yml restart backend
```

**Note:** Rotating JWT_SECRET_KEY will invalidate all existing tokens.

### 8.2 Security Audit Checklist

- [ ] All secrets in environment variables (not in code)
- [ ] FLASK_DEBUG=False
- [ ] AUTH_BYPASS_ENABLED=False
- [ ] ENABLE_HTTPS=True
- [ ] Rate limiting enabled
- [ ] CORS restricted to known origins
- [ ] API_KEY is unique and >32 characters

---

## 9. Contact & Escalation

| Level | Contact | Criteria |
|-------|---------|----------|
| L1 | On-call engineer | Service degraded, non-critical issues |
| L2 | Backend team lead | Service down, data issues |
| L3 | System architect | Security incident, major outage |

---

## 10. Maintenance Windows

Preferred maintenance window: **Sunday 02:00-06:00 PHT (Asia/Manila)**

Notify stakeholders 48 hours in advance for planned maintenance.
