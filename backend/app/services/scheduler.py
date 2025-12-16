"""
Background Scheduler for Floodingnaque.

Handles periodic tasks like weather data ingestion.
Designed to work correctly with Gunicorn workers.
"""

from apscheduler.schedulers.background import BackgroundScheduler
import logging
import os

logger = logging.getLogger(__name__)

# Scheduler instance - jobs are added when start() is called
scheduler = BackgroundScheduler(
    timezone=os.getenv('SCHEDULER_TIMEZONE', 'Asia/Manila'),
    job_defaults={
        'coalesce': True,  # Combine missed runs into one
        'max_instances': 1,  # Only one instance of each job
        'misfire_grace_time': 300  # 5 minute grace period
    }
)

# Track if scheduler has been initialized
_scheduler_initialized = False


def scheduled_ingest():
    """
    Scheduled task for weather data ingestion.
    
    Runs periodically to fetch fresh weather data from APIs.
    """
    # Import here to avoid circular imports and ensure app context
    from app.services.ingest import ingest_data
    
    try:
        # Get default coordinates from environment
        lat = float(os.getenv('DEFAULT_LATITUDE', '14.4793'))
        lon = float(os.getenv('DEFAULT_LONGITUDE', '121.0198'))
        
        ingest_data(lat=lat, lon=lon)
        logger.info("Scheduled data ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Error in scheduled ingestion: {str(e)}", exc_info=True)


def init_scheduler():
    """
    Initialize scheduler with jobs.
    
    Call this AFTER the Flask app is fully configured.
    Only initializes once to prevent duplicate jobs with Gunicorn workers.
    """
    global _scheduler_initialized
    
    if _scheduler_initialized:
        logger.debug("Scheduler already initialized, skipping.")
        return
    
    # Check if scheduler is enabled
    scheduler_enabled = os.getenv('SCHEDULER_ENABLED', 'True').lower() == 'true'
    if not scheduler_enabled:
        logger.info("Scheduler is disabled via SCHEDULER_ENABLED=False")
        return
    
    # Get ingest interval from environment (default: 1 hour)
    ingest_interval = int(os.getenv('DATA_INGEST_INTERVAL_HOURS', '1'))
    
    # Add the ingestion job
    scheduler.add_job(
        scheduled_ingest,
        'interval',
        hours=ingest_interval,
        id='weather_ingest',
        name='Weather Data Ingestion',
        replace_existing=True  # Replace if exists (handles restarts)
    )
    
    logger.info(f"Scheduler initialized with ingestion interval: {ingest_interval} hour(s)")
    _scheduler_initialized = True


def start():
    """
    Start the scheduler.
    
    Initializes jobs if not already done, then starts the scheduler.
    Safe to call multiple times - will not start if already running.
    """
    if scheduler.running:
        logger.debug("Scheduler already running.")
        return
    
    try:
        # Initialize jobs first
        init_scheduler()
        
        # Start the scheduler
        scheduler.start()
        logger.info("Background scheduler started successfully.")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {str(e)}", exc_info=True)


def shutdown():
    """Gracefully shutdown the scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=True)
        logger.info("Scheduler shut down gracefully.")
