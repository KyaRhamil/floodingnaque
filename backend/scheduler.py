from apscheduler.schedulers.background import BackgroundScheduler
from ingest import ingest_data
import logging

scheduler = BackgroundScheduler()

def scheduled_ingest():
    try:
        ingest_data()
        logging.info("Scheduled data ingestion completed.")
    except Exception as e:
        logging.error(f"Error in scheduled ingestion: {str(e)}")

scheduler.add_job(scheduled_ingest, 'interval', hours=1)  # Run every hour
