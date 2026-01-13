# =====================================================
# FLOODINGNAQUE - PgBouncer Connection Pooler
# =====================================================
# Lightweight connection pooler for PostgreSQL
# Optimized for production deployments
# =====================================================

FROM alpine:3.19

# Labels for container metadata
LABEL maintainer="Floodingnaque Team"
LABEL description="PgBouncer connection pooler for Floodingnaque"
LABEL version="1.21.0"

# Install PgBouncer and dependencies
RUN apk add --no-cache \
    pgbouncer \
    gettext \
    ca-certificates \
    && rm -rf /var/cache/apk/*

# Create directories
RUN mkdir -p /etc/pgbouncer /var/log/pgbouncer /var/run/pgbouncer

# Copy configuration files
COPY pgbouncer.ini /etc/pgbouncer/pgbouncer.ini.template
COPY entrypoint.sh /entrypoint.sh

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

# Create non-root user for security
RUN addgroup -S pgbouncer && adduser -S pgbouncer -G pgbouncer
RUN chown -R pgbouncer:pgbouncer /etc/pgbouncer /var/log/pgbouncer /var/run/pgbouncer

# Switch to non-root user
USER pgbouncer

# Expose PgBouncer port
EXPOSE 6432

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD echo "SELECT 1;" | psql -h localhost -p 6432 -U pgbouncer_stats pgbouncer || exit 1

# Run PgBouncer
ENTRYPOINT ["/entrypoint.sh"]
CMD ["pgbouncer", "/etc/pgbouncer/pgbouncer.ini"]
