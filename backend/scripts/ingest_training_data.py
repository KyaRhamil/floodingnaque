"""
Enhanced Training Data Ingestion Script
========================================

Pulls data from ALL production sources to create comprehensive training datasets:
1. Supabase weather_data table (real-time ingested data)
2. Google Earth Engine (GPM, CHIRPS, ERA5 satellite data)
3. Meteostat (historical weather station data)
4. WorldTides API (tidal data for coastal flooding)
5. Existing processed CSVs (baseline official flood records)

This replaces the outdated approach of training only on static CSVs.

Usage:
    python scripts/ingest_training_data.py --days 365 --output data/training/enhanced_dataset.csv
    python scripts/ingest_training_data.py --include-satellite --include-tides
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.db import (
    WeatherData, SatelliteWeatherCache, TideDataCache,
    get_db_session, engine
)
from app.services.meteostat_service import MeteostatService
from app.services.worldtides_service import WorldTidesService
from app.services.google_weather_service import GoogleWeatherService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingDataIngestion:
    """Ingests data from all production sources for ML training."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize data ingestion.
        
        Args:
            env_file: Path to .env file (default: .env.production)
        """
        # Load environment
        if env_file:
            load_dotenv(env_file)
        else:
            # Try production first, fallback to .env
            if Path('.env.production').exists():
                load_dotenv('.env.production')
                logger.info("Loaded .env.production")
            else:
                load_dotenv()
                logger.info("Loaded .env")
        
        # Initialize services
        self.meteostat = MeteostatService.get_instance()
        self.worldtides = WorldTidesService.get_instance()
        self.google_weather = GoogleWeatherService.get_instance()
        
        # Default location (Parañaque City)
        self.default_lat = float(os.getenv('DEFAULT_LATITUDE', '14.4793'))
        self.default_lon = float(os.getenv('DEFAULT_LONGITUDE', '121.0198'))
    
    def fetch_supabase_weather_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch weather data from Supabase production database.
        
        Args:
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: now)
            
        Returns:
            DataFrame with weather data
        """
        logger.info("Fetching weather data from Supabase...")
        
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=365))
        
        try:
            with get_db_session() as session:
                query = session.query(WeatherData).filter(
                    WeatherData.timestamp >= start_date,
                    WeatherData.timestamp <= end_date
                )
                
                results = query.all()
                
                if not results:
                    logger.warning("No weather data found in Supabase")
                    return pd.DataFrame()
                
                data = []
                for record in results:
                    data.append({
                        'timestamp': record.timestamp,
                        'temperature': record.temperature,
                        'humidity': record.humidity,
                        'precipitation': record.precipitation,
                        'weather_type': record.weather_description,
                        'latitude': record.latitude,
                        'longitude': record.longitude,
                        'source': 'supabase_db'
                    })
                
                df = pd.DataFrame(data)
                logger.info(f"Fetched {len(df)} records from Supabase")
                return df
                
        except Exception as e:
            logger.error(f"Error fetching Supabase data: {e}")
            return pd.DataFrame()
    
    def fetch_satellite_weather_cache(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch cached satellite weather data (GPM, CHIRPS, ERA5).
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with satellite data
        """
        logger.info("Fetching satellite weather cache...")
        
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=365))
        
        try:
            with get_db_session() as session:
                query = session.query(SatelliteWeatherCache).filter(
                    SatelliteWeatherCache.timestamp >= start_date,
                    SatelliteWeatherCache.timestamp <= end_date,
                    SatelliteWeatherCache.is_valid == True
                )
                
                results = query.all()
                
                if not results:
                    logger.warning("No satellite data found in cache")
                    return pd.DataFrame()
                
                data = []
                for record in results:
                    data.append({
                        'timestamp': record.timestamp,
                        'temperature': record.era5_temperature - 273.15 if record.era5_temperature else None,  # K to C
                        'humidity': record.era5_humidity,
                        'precipitation': record.precipitation_rate or record.era5_precipitation,
                        'precipitation_1h': record.precipitation_1h,
                        'precipitation_3h': record.precipitation_3h,
                        'precipitation_24h': record.precipitation_24h,
                        'dataset': record.dataset,
                        'latitude': record.latitude,
                        'longitude': record.longitude,
                        'source': 'satellite_cache'
                    })
                
                df = pd.DataFrame(data)
                logger.info(f"Fetched {len(df)} satellite records from cache")
                return df
                
        except Exception as e:
            logger.error(f"Error fetching satellite cache: {e}")
            return pd.DataFrame()
    
    def fetch_meteostat_historical(
        self,
        days: int = 365,
        lat: Optional[float] = None,
        lon: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Fetch historical weather from Meteostat stations.
        
        Args:
            days: Number of days of history
            lat: Latitude
            lon: Longitude
            
        Returns:
            DataFrame with Meteostat data
        """
        logger.info(f"Fetching {days} days of Meteostat data...")
        
        lat = lat or self.default_lat
        lon = lon or self.default_lon
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            observations = self.meteostat.get_hourly_data(
                lat=lat, lon=lon,
                start=start_date, end=end_date
            )
            
            if not observations:
                logger.warning("No Meteostat data available")
                return pd.DataFrame()
            
            data = []
            for obs in observations:
                data.append({
                    'timestamp': obs.timestamp,
                    'temperature': obs.temperature,
                    'humidity': obs.humidity,
                    'precipitation': obs.precipitation,
                    'wind_speed': obs.wind_speed,
                    'pressure': obs.pressure,
                    'latitude': lat,
                    'longitude': lon,
                    'source': 'meteostat'
                })
            
            df = pd.DataFrame(data)
            logger.info(f"Fetched {len(df)} Meteostat records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Meteostat data: {e}")
            return pd.DataFrame()
    
    def fetch_tide_data_cache(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch cached tidal data from WorldTides.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with tide data
        """
        logger.info("Fetching tide data cache...")
        
        end_date = end_date or datetime.now()
        start_date = start_date or (end_date - timedelta(days=365))
        
        try:
            with get_db_session() as session:
                query = session.query(TideDataCache).filter(
                    TideDataCache.timestamp >= start_date,
                    TideDataCache.timestamp <= end_date,
                    TideDataCache.is_valid == True
                )
                
                results = query.all()
                
                if not results:
                    logger.warning("No tide data found in cache")
                    return pd.DataFrame()
                
                data = []
                for record in results:
                    data.append({
                        'timestamp': record.timestamp,
                        'tide_height': record.height,
                        'tide_type': record.tide_type,
                        'latitude': record.latitude,
                        'longitude': record.longitude,
                        'source': 'worldtides_cache'
                    })
                
                df = pd.DataFrame(data)
                logger.info(f"Fetched {len(df)} tide records from cache")
                return df
                
        except Exception as e:
            logger.error(f"Error fetching tide cache: {e}")
            return pd.DataFrame()
    
    def load_processed_flood_records(self, data_dir: str = 'data/processed') -> pd.DataFrame:
        """
        Load existing processed flood records (2022-2025).
        
        Args:
            data_dir: Directory containing processed CSVs
            
        Returns:
            Combined DataFrame of all official flood records
        """
        logger.info("Loading processed flood records...")
        
        data_path = Path(data_dir)
        
        # Load cumulative dataset (includes all years)
        cumulative_file = data_path / 'cumulative_up_to_2025.csv'
        
        if cumulative_file.exists():
            df = pd.read_csv(cumulative_file)
            logger.info(f"Loaded {len(df)} official flood records")
            
            # Ensure timestamp column
            if 'date' in df.columns and 'timestamp' not in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
            
            # Add source
            df['source'] = 'official_records'
            
            return df
        else:
            logger.warning(f"Cumulative file not found: {cumulative_file}")
            return pd.DataFrame()
    
    def merge_datasets(
        self,
        datasets: List[pd.DataFrame],
        on_columns: List[str] = ['timestamp'],
        how: str = 'outer'
    ) -> pd.DataFrame:
        """
        Merge multiple datasets on common columns.
        
        Args:
            datasets: List of DataFrames to merge
            on_columns: Columns to merge on
            how: Merge strategy (outer, inner, left)
            
        Returns:
            Merged DataFrame
        """
        logger.info(f"Merging {len(datasets)} datasets...")
        
        # Filter out empty datasets
        valid_datasets = [df for df in datasets if not df.empty]
        
        if not valid_datasets:
            logger.error("No valid datasets to merge")
            return pd.DataFrame()
        
        if len(valid_datasets) == 1:
            return valid_datasets[0]
        
        # Start with first dataset
        merged = valid_datasets[0]
        
        # Merge remaining datasets
        for df in valid_datasets[1:]:
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            if 'timestamp' in merged.columns:
                merged['timestamp'] = pd.to_datetime(merged['timestamp'])
            
            # Merge
            merged = pd.concat([merged, df], axis=0, ignore_index=True, sort=False)
        
        # Remove duplicates based on timestamp
        if 'timestamp' in merged.columns:
            merged = merged.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
        
        logger.info(f"Merged dataset size: {len(merged)} records")
        return merged
    
    def enrich_with_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features for better model training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added time features
        """
        if 'timestamp' not in df.columns:
            return df
        
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Monsoon season (June-November in Philippines)
        df['is_monsoon_season'] = df['month'].isin([6, 7, 8, 9, 10, 11]).astype(int)
        
        # Season
        def get_season(month):
            if month in [12, 1, 2]:
                return 'dry_cool'
            elif month in [3, 4, 5]:
                return 'dry_hot'
            elif month in [6, 7, 8]:
                return 'wet_monsoon'
            else:  # 9, 10, 11
                return 'wet_typhoon'
        
        df['season'] = df['month'].apply(get_season)
        
        return df
    
    def clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the merged dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning and validating dataset...")
        
        initial_count = len(df)
        
        # Remove rows with missing critical features
        required_cols = ['temperature', 'humidity', 'precipitation']
        existing_cols = [col for col in required_cols if col in df.columns]
        
        if existing_cols:
            df = df.dropna(subset=existing_cols)
        
        # Handle temperature (should be in Celsius, 15-40°C for Philippines)
        if 'temperature' in df.columns:
            # Convert Kelvin to Celsius if needed
            df.loc[df['temperature'] > 100, 'temperature'] = df.loc[df['temperature'] > 100, 'temperature'] - 273.15
            
            # Filter outliers
            df = df[(df['temperature'] >= 15) & (df['temperature'] <= 45)]
        
        # Handle humidity (0-100%)
        if 'humidity' in df.columns:
            df = df[(df['humidity'] >= 0) & (df['humidity'] <= 100)]
        
        # Handle precipitation (should be non-negative)
        if 'precipitation' in df.columns:
            df = df[df['precipitation'] >= 0]
            df.loc[df['precipitation'].isna(), 'precipitation'] = 0
        
        final_count = len(df)
        removed = initial_count - final_count
        
        logger.info(f"Removed {removed} invalid records ({removed/initial_count*100:.1f}%)")
        logger.info(f"Final dataset: {final_count} records")
        
        return df
    
    def ingest_all_sources(
        self,
        days: int = 365,
        include_satellite: bool = True,
        include_tides: bool = True,
        include_meteostat: bool = True,
        include_official: bool = True
    ) -> pd.DataFrame:
        """
        Ingest data from all available sources.
        
        Args:
            days: Number of days of history to fetch
            include_satellite: Include Google Earth Engine data
            include_tides: Include WorldTides data
            include_meteostat: Include Meteostat weather station data
            include_official: Include official flood records
            
        Returns:
            Comprehensive training dataset
        """
        logger.info("="*80)
        logger.info("ENHANCED TRAINING DATA INGESTION")
        logger.info("="*80)
        logger.info(f"Fetching {days} days of data from multiple sources...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        datasets = []
        
        # 1. Supabase weather data (production database)
        supabase_df = self.fetch_supabase_weather_data(start_date, end_date)
        if not supabase_df.empty:
            datasets.append(supabase_df)
        
        # 2. Satellite data (GPM, CHIRPS, ERA5)
        if include_satellite:
            satellite_df = self.fetch_satellite_weather_cache(start_date, end_date)
            if not satellite_df.empty:
                datasets.append(satellite_df)
        
        # 3. Meteostat historical weather
        if include_meteostat:
            meteostat_df = self.fetch_meteostat_historical(days=days)
            if not meteostat_df.empty:
                datasets.append(meteostat_df)
        
        # 4. Tide data
        if include_tides:
            tide_df = self.fetch_tide_data_cache(start_date, end_date)
            if not tide_df.empty:
                datasets.append(tide_df)
        
        # 5. Official flood records (2022-2025 baseline)
        if include_official:
            official_df = self.load_processed_flood_records()
            if not official_df.empty:
                datasets.append(official_df)
        
        # Merge all datasets
        if not datasets:
            logger.error("No datasets fetched!")
            return pd.DataFrame()
        
        merged_df = self.merge_datasets(datasets)
        
        # Add time features
        merged_df = self.enrich_with_time_features(merged_df)
        
        # Clean and validate
        merged_df = self.clean_and_validate(merged_df)
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("INGESTION COMPLETE")
        logger.info("="*80)
        logger.info(f"Total records: {len(merged_df)}")
        logger.info(f"Date range: {merged_df['timestamp'].min()} to {merged_df['timestamp'].max()}")
        logger.info(f"Sources: {merged_df['source'].unique().tolist() if 'source' in merged_df.columns else 'N/A'}")
        
        if 'flood' in merged_df.columns:
            flood_dist = merged_df['flood'].value_counts()
            logger.info(f"Flood distribution: {flood_dist.to_dict()}")
        
        logger.info("="*80)
        
        return merged_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Ingest training data from all production sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Fetch 1 year of data from all sources:
    python scripts/ingest_training_data.py --days 365 --output data/training/enhanced_2025.csv
  
  Fetch only Supabase and official records:
    python scripts/ingest_training_data.py --no-satellite --no-tides --no-meteostat
  
  Use custom .env file:
    python scripts/ingest_training_data.py --env .env.staging
        """
    )
    
    parser.add_argument('--days', type=int, default=365,
                        help='Number of days of data to fetch (default: 365)')
    parser.add_argument('--output', type=str, default='data/training/enhanced_dataset.csv',
                        help='Output CSV file path')
    parser.add_argument('--env', type=str, default=None,
                        help='Path to .env file (default: .env.production or .env)')
    parser.add_argument('--no-satellite', action='store_true',
                        help='Exclude satellite data (GPM, CHIRPS, ERA5)')
    parser.add_argument('--no-tides', action='store_true',
                        help='Exclude tide data')
    parser.add_argument('--no-meteostat', action='store_true',
                        help='Exclude Meteostat weather station data')
    parser.add_argument('--no-official', action='store_true',
                        help='Exclude official flood records')
    
    args = parser.parse_args()
    
    # Initialize ingestion
    ingestion = TrainingDataIngestion(env_file=args.env)
    
    # Ingest data
    df = ingestion.ingest_all_sources(
        days=args.days,
        include_satellite=not args.no_satellite,
        include_tides=not args.no_tides,
        include_meteostat=not args.no_meteostat,
        include_official=not args.no_official
    )
    
    if df.empty:
        logger.error("No data ingested! Check your data sources and environment configuration.")
        sys.exit(1)
    
    # Save to output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Training dataset saved to: {output_path}")
    logger.info(f"  Size: {len(df)} records")
    logger.info(f"  Columns: {list(df.columns)}")


if __name__ == '__main__':
    main()
