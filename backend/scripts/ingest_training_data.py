"""
Training Data Ingestion Utility
===============================

Ingests new training data from various sources into the processed data pipeline.

Usage:
    python scripts/ingest_training_data.py --file new_data.csv
    python scripts/ingest_training_data.py --dir data/raw/
    python scripts/ingest_training_data.py --fetch-google --days 365
    python scripts/ingest_training_data.py --fetch-meteostat --fetch-tides --days 180
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv


def load_env_files():
    """Load environment variables from .env files in priority order."""
    env_files = [
        BACKEND_DIR / ".env",
        BACKEND_DIR / ".env.local",
        BACKEND_DIR / ".env.production",
    ]

    loaded = False
    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file, override=False)
            print(f"Loaded environment from: {env_file.name}")
            loaded = True

    if not loaded:
        print("Warning: No .env file found. Using system environment variables.")


load_env_files()

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["temperature", "humidity", "precipitation", "flood"]


# Lazy imports for services - import inside functions to avoid circular import issues
def _get_google_service():
    from app.services.google_weather_service import GoogleWeatherService

    return GoogleWeatherService.get_instance()


def _get_meteostat_service():
    from app.services.meteostat_service import MeteostatService

    return MeteostatService.get_instance()


def _get_worldtides_service():
    from app.services.worldtides_service import WorldTidesService

    return WorldTidesService.get_instance()


def validate_data(df: pd.DataFrame) -> bool:
    """Validate ingested data has required columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False
    return True


def merge_weather_and_tide_data(weather_df: pd.DataFrame, tide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge weather data with tide data on timestamp.

    Args:
        weather_df: Weather data (temperature, humidity, precipitation)
        tide_df: Tide data (tide_height)

    Returns:
        Merged DataFrame with all features
    """
    if weather_df.empty:
        logger.warning("No weather data available for merging")
        return pd.DataFrame()

    weather_df = weather_df.copy()

    # Normalize timestamp column name (Meteostat uses 'date', others may use 'timestamp')
    if "date" in weather_df.columns and "timestamp" not in weather_df.columns:
        weather_df["timestamp"] = pd.to_datetime(weather_df["date"])
    elif "timestamp" in weather_df.columns:
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
    else:
        # No timestamp column, cannot merge with tide data
        logger.warning("Weather data has no date/timestamp column, cannot merge with tides")
        weather_df["tide_height"] = 0.0
        weather_df["has_tide_data"] = False
        return weather_df

    if tide_df is None or tide_df.empty:
        logger.warning("No tide data available, using weather data only")
        weather_df["tide_height"] = 0.0
        weather_df["has_tide_data"] = False
        return weather_df

    # Ensure tide timestamp is datetime
    tide_df = tide_df.copy()
    tide_df["timestamp"] = pd.to_datetime(tide_df["timestamp"])

    # Round timestamps to nearest hour for better matching
    weather_df["timestamp_hour"] = weather_df["timestamp"].dt.floor("H")
    tide_df["timestamp_hour"] = tide_df["timestamp"].dt.floor("H")

    # Aggregate multiple tide readings per hour (take mean)
    tide_hourly = tide_df.groupby("timestamp_hour").agg({"tide_height": "mean", "datum": "first"}).reset_index()

    # Merge on rounded timestamp
    merged = pd.merge(
        weather_df, tide_hourly[["timestamp_hour", "tide_height", "datum"]], on="timestamp_hour", how="left"
    )

    # Interpolate missing tide heights
    merged["tide_height"] = merged["tide_height"].interpolate(method="linear").fillna(0.0)
    merged["has_tide_data"] = merged["tide_height"] != 0.0

    # Drop helper column
    merged = merged.drop(columns=["timestamp_hour"], errors="ignore")

    logger.info(f"Merged {len(weather_df)} weather records with {len(tide_df)} tide records")
    logger.info(f"Result: {len(merged)} records, {merged['has_tide_data'].sum()} with actual tide data")

    return merged


def ingest_file(file_path: Path, output_name: Optional[str] = None) -> pd.DataFrame:
    """Ingest a single data file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded: {file_path.name} ({len(df)} records)")

    if not validate_data(df):
        raise ValueError(f"Invalid data format in {file_path.name}")

    output_name = output_name or f"ingested_{file_path.stem}.csv"
    output_path = PROCESSED_DIR / output_name
    df.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")

    return df


def ingest_directory(dir_path: Path) -> int:
    """Ingest all CSV files from a directory."""
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    files = list(dir_path.glob("*.csv"))
    logger.info(f"Found {len(files)} CSV files in {dir_path}")

    success_count = 0
    for file_path in files:
        try:
            ingest_file(file_path)
            success_count += 1
        except Exception as e:
            logger.warning(f"Failed to ingest {file_path.name}: {e}")

    return success_count


def fetch_google_data(days: int = 365) -> pd.DataFrame:
    """Fetch data from GoogleCloud (Google Earth Engine)."""
    logger.info(f"Fetching data from GoogleCloud (Google Earth Engine) for {days} days...")
    try:
        service = _get_google_service()
        historical_data = service.get_historical_for_training(days=days)

        if not historical_data:
            logger.warning("No data returned from GoogleCloud")
            return pd.DataFrame()

        df = pd.DataFrame(historical_data)

        if "flood" not in df.columns:
            df["flood"] = 0

        logger.info(f"Successfully fetched {len(df)} records from GoogleCloud")
        return df
    except Exception as e:
        logger.error(f"Error fetching Google data: {e}")
        return pd.DataFrame()


def fetch_meteostat_data(days: int = 365) -> pd.DataFrame:
    """Fetch data from Meteostat."""
    logger.info(f"Fetching data from Meteostat for {days} days...")
    try:
        service = _get_meteostat_service()
        df = service.get_historical_for_training(days=days)

        if df.empty:
            logger.warning("No data returned from Meteostat")
            return pd.DataFrame()

        if "flood" not in df.columns:
            df["flood"] = 0

        logger.info(f"Successfully fetched {len(df)} records from Meteostat")
        return df
    except Exception as e:
        logger.error(f"Error fetching Meteostat data: {e}")
        return pd.DataFrame()


def fetch_worldtides_data(days: int = 365) -> pd.DataFrame:
    """Fetch data from WorldTides."""
    logger.info(f"Fetching data from WorldTides for {days} days...")
    try:
        service = _get_worldtides_service()
        tide_heights = service.get_tide_heights(days=min(days, 7))

        if not tide_heights:
            logger.warning("No data returned from WorldTides")
            return pd.DataFrame()

        records = []
        for tide in tide_heights:
            records.append(
                {
                    "timestamp": tide.timestamp,
                    "tide_height": tide.height,
                    "datum": tide.datum,
                    "source": tide.source,
                }
            )

        df = pd.DataFrame(records)
        logger.info(f"Successfully fetched {len(df)} records from WorldTides")
        return df
    except Exception as e:
        logger.error(f"Error fetching Tides data: {e}")
        return pd.DataFrame()


def process_and_merge_fetched_data(fetched_data: Dict[str, pd.DataFrame], args) -> int:
    """
    Process and merge fetched data from multiple sources.

    Strategy:
    - Use weather data (Meteostat or Google) as base
    - Merge tide data onto weather data by timestamp
    - Save individual sources and merged result
    """
    success_count = 0

    # Identify data types
    weather_df = None
    tide_df = None

    # Prioritize Meteostat over Google for weather data
    if "Meteostat" in fetched_data and not fetched_data["Meteostat"].empty:
        weather_df = fetched_data["Meteostat"]
        weather_source = "Meteostat"
    elif "GoogleCloud" in fetched_data and not fetched_data["GoogleCloud"].empty:
        weather_df = fetched_data["GoogleCloud"]
        weather_source = "GoogleCloud"

    if "WorldTides" in fetched_data and not fetched_data["WorldTides"].empty:
        tide_df = fetched_data["WorldTides"]

    # Save individual source files
    for source_name, df in fetched_data.items():
        if df.empty:
            logger.warning(f"No data fetched from {source_name}. Skipping...")
            continue

        output_name = f"fetched_{source_name.lower().replace(' ', '_')}.csv"
        output_path = PROCESSED_DIR / output_name

        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} records from {source_name} to: {output_path}")
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to save data from {source_name}: {e}")

    # Create merged training dataset
    if weather_df is not None:
        logger.info(f"Creating merged training dataset from {weather_source} + WorldTides...")

        merged_df = merge_weather_and_tide_data(weather_df, tide_df)

        if not merged_df.empty:
            # Ensure flood column exists
            if "flood" not in merged_df.columns:
                merged_df["flood"] = 0
                logger.info("Added 'flood' column with default value 0 (requires manual labeling)")

            # Validate final dataset
            if validate_data(merged_df):
                output_path = PROCESSED_DIR / "training_data_merged.csv"
                merged_df.to_csv(output_path, index=False)
                logger.info(f"✓ Saved merged training dataset: {output_path} ({len(merged_df)} records)")
                success_count += 1
            else:
                logger.error("Merged dataset failed validation")
        else:
            logger.warning("No merged dataset created (no valid weather data)")
    else:
        logger.warning("No weather data available for creating training dataset")

    return success_count


def main():
    parser = argparse.ArgumentParser(description="Ingest training data from files or fetch from external sources")

    parser.add_argument("--file", type=str, help="Single file to ingest")
    parser.add_argument("--dir", type=str, help="Directory to ingest")
    parser.add_argument("--output", type=str, help="Output filename")

    parser.add_argument("--fetch-google", action="store_true", help="Fetch data from GoogleCloud (Google Earth Engine)")
    parser.add_argument("--fetch-meteostat", action="store_true", help="Fetch data from Meteostat")
    parser.add_argument("--fetch-tides", action="store_true", help="Fetch data from WorldTides")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data to fetch (default: 365)")

    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if args.file:
            logger.info("=== File Ingestion Mode ===")
            ingest_file(Path(args.file), args.output)
            logger.info("File ingestion completed successfully")

        elif args.dir:
            logger.info("=== Directory Ingestion Mode ===")
            count = ingest_directory(Path(args.dir))
            logger.info(f"Successfully ingested {count} files")

        elif args.fetch_google or args.fetch_meteostat or args.fetch_tides:
            logger.info("=== Data Fetching Mode ===")
            fetched_data = {}

            if args.fetch_google:
                df = fetch_google_data(args.days)
                fetched_data["GoogleCloud"] = df

            if args.fetch_meteostat:
                df = fetch_meteostat_data(args.days)
                fetched_data["Meteostat"] = df

            if args.fetch_tides:
                df = fetch_worldtides_data(args.days)
                fetched_data["WorldTides"] = df

            success_count = process_and_merge_fetched_data(fetched_data, args)

            if success_count > 0:
                logger.info(f"✓ Successfully processed and saved {success_count} dataset(s)")
            else:
                logger.warning("No data sources were successfully processed")
                sys.exit(1)

        else:
            logger.warning("No action specified. Use --help for usage information.")
            parser.print_help()
            sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
