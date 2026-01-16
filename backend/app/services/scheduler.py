"""
Background Scheduler for Floodingnaque.

Handles periodic tasks like weather data ingestion.
Designed to work correctly with Gunicorn workers using distributed locking.
"""

import logging
import os
import sys
import tempfile

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

# Scheduler instance - jobs are added when start() is called
scheduler = BackgroundScheduler(
    timezone=os.getenv("SCHEDULER_TIMEZONE", "Asia/Manila"),
    job_defaults={
        "coalesce": True,  # Combine missed runs into one
        "max_instances": 1,  # Only one instance of each job
        "misfire_grace_time": 300,  # 5 minute grace period
    },
)

# Track if scheduler has been initialized
_scheduler_initialized = False
_scheduler_lock_fd = None  # File descriptor for lock file


def _get_lock_file_path() -> str:
    """
    Get the path to the scheduler lock file.
    Uses temp directory for cross-platform compatibility.
    """
    if sys.platform == "win32":
        # Windows: Use temp directory
        return os.path.join(tempfile.gettempdir(), "floodingnaque_scheduler.lock")
    else:
        # Unix-like: Use /tmp for better compatibility with containers
        return "/tmp/floodingnaque_scheduler.lock"  # nosec B108


def should_run_scheduler() -> bool:
    """
    Check if this process should run the scheduler.

    Uses file locking to ensure only one Gunicorn worker runs the scheduler.
    The first worker to acquire the lock becomes the scheduler master.

    Returns:
        bool: True if this process should run the scheduler
    """
    global _scheduler_lock_fd

    lock_file = _get_lock_file_path()

    try:
        if sys.platform == "win32":
            # Windows: Use msvcrt for file locking
            import msvcrt

            _scheduler_lock_fd = open(lock_file, "w")
            msvcrt.locking(_scheduler_lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
            _scheduler_lock_fd.write(str(os.getpid()))
            _scheduler_lock_fd.flush()
            logger.info(f"Acquired scheduler lock (PID: {os.getpid()})")
            return True
        else:
            # Unix: Use fcntl for file locking
            import fcntl

            _scheduler_lock_fd = open(lock_file, "w")
            fcntl.flock(_scheduler_lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            _scheduler_lock_fd.write(str(os.getpid()))
            _scheduler_lock_fd.flush()
            logger.info(f"Acquired scheduler lock (PID: {os.getpid()})")
            return True
    except (IOError, OSError) as e:
        # Another process holds the lock
        logger.info("Scheduler lock held by another worker, skipping scheduler init")
        if _scheduler_lock_fd:
            _scheduler_lock_fd.close()
            _scheduler_lock_fd = None
        return False
    except ImportError:
        # fcntl/msvcrt not available, fall back to allowing scheduler
        logger.warning("File locking not available, scheduler will run on all workers")
        return True


def scheduled_ingest():
    """
    Scheduled task for weather data ingestion.

    Runs periodically to fetch fresh weather data from APIs.
    """
    # Import here to avoid circular imports and ensure app context
    from app.services.ingest import ingest_data

    try:
        # Get default coordinates from environment
        lat = float(os.getenv("DEFAULT_LATITUDE", "14.4793"))
        lon = float(os.getenv("DEFAULT_LONGITUDE", "121.0198"))

        ingest_data(lat=lat, lon=lon)
        logger.info("Scheduled data ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Error in scheduled ingestion: {str(e)}", exc_info=True)


def init_scheduler():
    """
    Initialize scheduler with jobs.

    Call this AFTER the Flask app is fully configured.
    Only initializes once to prevent duplicate jobs with Gunicorn workers.
    Uses file locking to ensure only one worker runs the scheduler.
    """
    global _scheduler_initialized

    if _scheduler_initialized:
        logger.debug("Scheduler already initialized, skipping.")
        return

    # Check if scheduler is enabled
    scheduler_enabled = os.getenv("SCHEDULER_ENABLED", "True").lower() == "true"
    if not scheduler_enabled:
        logger.info("Scheduler is disabled via SCHEDULER_ENABLED=False")
        return

    # Check if this worker should run the scheduler (distributed lock)
    if not should_run_scheduler():
        logger.info("Scheduler will run on a different worker")
        _scheduler_initialized = True  # Mark as initialized to prevent retry
        return

    # Get ingest interval from environment (default: 1 hour)
    ingest_interval = int(os.getenv("DATA_INGEST_INTERVAL_HOURS", "1"))

    # Add the ingestion job
    scheduler.add_job(
        scheduled_ingest,
        "interval",
        hours=ingest_interval,
        id="weather_ingest",
        name="Weather Data Ingestion",
        replace_existing=True,  # Replace if exists (handles restarts)
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
    """Gracefully shutdown the scheduler and release lock."""
    global _scheduler_lock_fd

    if scheduler.running:
        scheduler.shutdown(wait=True)
        logger.info("Scheduler shut down gracefully.")

    # Release the lock file
    if _scheduler_lock_fd:
        try:
            _scheduler_lock_fd.close()
            _scheduler_lock_fd = None
            logger.info("Released scheduler lock")
        except Exception as e:
            logger.warning(f"Error releasing scheduler lock: {str(e)}")
