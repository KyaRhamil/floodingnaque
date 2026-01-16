"""
Dashboard Analytics Service

Uses read replica for analytics queries to reduce load on primary database.
Implements efficient time-series queries using partitioning.

Usage:
    from app.services.dashboard_analytics import DashboardAnalytics

    analytics = DashboardAnalytics()
    stats = analytics.get_dashboard_stats()
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.utils.db_optimization import get_router, use_read_replica
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class DashboardAnalytics:
    """
    Analytics service for dashboard queries.

    All read queries are routed to the read replica (if available)
    to reduce load on the primary database.
    """

    def __init__(self):
        self.router = get_router()

    @use_read_replica
    def get_dashboard_stats(self, session: Session = None, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive dashboard statistics.

        Args:
            session: Database session (injected by decorator)
            days: Number of days to include in stats

        Returns:
            Dictionary with dashboard statistics
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        # Weather data stats
        weather_stats = session.execute(
            text(
                """
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT DATE(timestamp)) as days_with_data,
                AVG(temperature) as avg_temperature,
                AVG(humidity) as avg_humidity,
                AVG(precipitation) as avg_precipitation,
                MAX(precipitation) as max_precipitation,
                MIN(created_at) as earliest_record,
                MAX(created_at) as latest_record
            FROM weather_data
            WHERE created_at >= :cutoff
            AND is_deleted = false
        """
            ),
            {"cutoff": cutoff_date},
        ).fetchone()

        # Prediction stats
        prediction_stats = session.execute(
            text(
                """
            SELECT
                COUNT(*) as total_predictions,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as flood_predictions,
                SUM(CASE WHEN risk_level = 0 THEN 1 ELSE 0 END) as safe_count,
                SUM(CASE WHEN risk_level = 1 THEN 1 ELSE 0 END) as alert_count,
                SUM(CASE WHEN risk_level = 2 THEN 1 ELSE 0 END) as critical_count,
                AVG(confidence) as avg_confidence
            FROM predictions
            WHERE created_at >= :cutoff
            AND is_deleted = false
        """
            ),
            {"cutoff": cutoff_date},
        ).fetchone()

        # Alert stats
        alert_stats = session.execute(
            text(
                """
            SELECT
                COUNT(*) as total_alerts,
                SUM(CASE WHEN delivery_status = 'delivered' THEN 1 ELSE 0 END) as delivered,
                SUM(CASE WHEN delivery_status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN delivery_status = 'pending' THEN 1 ELSE 0 END) as pending
            FROM alert_history
            WHERE created_at >= :cutoff
            AND is_deleted = false
        """
            ),
            {"cutoff": cutoff_date},
        ).fetchone()

        return {
            "period_days": days,
            "weather": {
                "total_records": weather_stats[0] or 0,
                "days_with_data": weather_stats[1] or 0,
                "avg_temperature_k": round(weather_stats[2], 2) if weather_stats[2] else None,
                "avg_humidity_pct": round(weather_stats[3], 2) if weather_stats[3] else None,
                "avg_precipitation_mm": round(weather_stats[4], 2) if weather_stats[4] else None,
                "max_precipitation_mm": round(weather_stats[5], 2) if weather_stats[5] else None,
            },
            "predictions": {
                "total": prediction_stats[0] or 0,
                "flood_predictions": prediction_stats[1] or 0,
                "safe_count": prediction_stats[2] or 0,
                "alert_count": prediction_stats[3] or 0,
                "critical_count": prediction_stats[4] or 0,
                "avg_confidence": round(prediction_stats[5], 4) if prediction_stats[5] else None,
            },
            "alerts": {
                "total": alert_stats[0] or 0,
                "delivered": alert_stats[1] or 0,
                "failed": alert_stats[2] or 0,
                "pending": alert_stats[3] or 0,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @use_read_replica
    def get_time_series_data(
        self,
        session: Session = None,
        metric: str = "precipitation",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "day",
    ) -> List[Dict[str, Any]]:
        """
        Get time-series data for dashboard charts.

        Args:
            session: Database session (injected by decorator)
            metric: Metric to retrieve (precipitation, temperature, humidity, predictions)
            start_date: Start of date range
            end_date: End of date range
            interval: Aggregation interval (hour, day, week, month)

        Returns:
            List of time-series data points
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Build date truncation expression based on interval
        interval_map = {
            "hour": "DATE_TRUNC('hour', timestamp)",
            "day": "DATE_TRUNC('day', timestamp)",
            "week": "DATE_TRUNC('week', timestamp)",
            "month": "DATE_TRUNC('month', timestamp)",
        }
        date_trunc = interval_map.get(interval, interval_map["day"])

        # Build metric aggregation
        metric_map = {
            "precipitation": "AVG(precipitation)",
            "temperature": "AVG(temperature)",
            "humidity": "AVG(humidity)",
            "wind_speed": "AVG(wind_speed)",
            "pressure": "AVG(pressure)",
        }
        metric_agg = metric_map.get(metric, metric_map["precipitation"])

        query = f"""
            SELECT
                {date_trunc} as period,
                {metric_agg} as value,
                COUNT(*) as sample_count
            FROM weather_data
            WHERE timestamp >= :start_date
            AND timestamp <= :end_date
            AND is_deleted = false
            GROUP BY 1
            ORDER BY 1
        """  # nosec B608

        results = session.execute(text(query), {"start_date": start_date, "end_date": end_date}).fetchall()

        return [
            {
                "period": row[0].isoformat() if row[0] else None,
                "value": round(row[1], 4) if row[1] else None,
                "sample_count": row[2],
            }
            for row in results
        ]

    @use_read_replica
    def get_flood_risk_trend(self, session: Session = None, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get flood risk trend over time.

        Args:
            session: Database session (injected by decorator)
            days: Number of days to analyze

        Returns:
            List of daily risk distributions
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        results = session.execute(
            text(
                """
            SELECT
                DATE(created_at) as date,
                risk_level,
                COUNT(*) as count
            FROM predictions
            WHERE created_at >= :cutoff
            AND is_deleted = false
            GROUP BY DATE(created_at), risk_level
            ORDER BY 1, 2
        """
            ),
            {"cutoff": cutoff_date},
        ).fetchall()

        # Organize by date
        daily_data = {}
        for row in results:
            date_str = row[0].isoformat() if row[0] else None
            if date_str not in daily_data:
                daily_data[date_str] = {"date": date_str, "safe": 0, "alert": 0, "critical": 0}

            risk_level = row[1]
            count = row[2]

            if risk_level == 0:
                daily_data[date_str]["safe"] = count
            elif risk_level == 1:
                daily_data[date_str]["alert"] = count
            elif risk_level == 2:
                daily_data[date_str]["critical"] = count

        return list(daily_data.values())

    @use_read_replica
    def get_partition_stats(self, session: Session = None, table_name: str = "weather_data") -> List[Dict[str, Any]]:
        """
        Get partition statistics for a table.

        Args:
            session: Database session (injected by decorator)
            table_name: Name of partitioned table

        Returns:
            List of partition statistics
        """
        try:
            results = session.execute(
                text(
                    """
                SELECT * FROM get_partition_stats(:table_name)
            """
                ),
                {"table_name": table_name},
            ).fetchall()

            return [
                {
                    "partition_name": row[0],
                    "row_count": row[1],
                    "table_size": row[2],
                    "index_size": row[3],
                    "total_size": row[4],
                }
                for row in results
            ]
        except Exception as e:
            logger.warning(f"Failed to get partition stats: {e}")
            return []

    @use_read_replica
    def get_performance_metrics(self, session: Session = None, hours: int = 24) -> Dict[str, Any]:
        """
        Get API performance metrics.

        Args:
            session: Database session (injected by decorator)
            hours: Hours of data to include

        Returns:
            Performance metrics dictionary
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        result = session.execute(
            text(
                """
            SELECT
                COUNT(*) as total_requests,
                AVG(response_time_ms) as avg_response_time,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY response_time_ms) as p50_response_time,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) as p99_response_time,
                SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) as error_count,
                SUM(CASE WHEN status_code >= 200 AND status_code < 300 THEN 1 ELSE 0 END) as success_count
            FROM api_requests
            WHERE created_at >= :cutoff
            AND is_deleted = false
        """
            ),
            {"cutoff": cutoff},
        ).fetchone()

        total = result[0] or 1
        errors = result[5] or 0

        return {
            "period_hours": hours,
            "total_requests": result[0] or 0,
            "avg_response_time_ms": round(result[1], 2) if result[1] else None,
            "p50_response_time_ms": round(result[2], 2) if result[2] else None,
            "p95_response_time_ms": round(result[3], 2) if result[3] else None,
            "p99_response_time_ms": round(result[4], 2) if result[4] else None,
            "error_count": errors,
            "success_count": result[6] or 0,
            "error_rate": round((errors / total) * 100, 2),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


# Singleton instance
_dashboard_analytics: Optional[DashboardAnalytics] = None


def get_dashboard_analytics() -> DashboardAnalytics:
    """Get or create the dashboard analytics instance."""
    global _dashboard_analytics
    if _dashboard_analytics is None:
        _dashboard_analytics = DashboardAnalytics()
    return _dashboard_analytics
