"""
Preprocess PAGASA Weather Station Data for Flood Model Training
================================================================

Transforms raw PAGASA climate data into ML-ready format compatible
with the existing Floodingnaque training pipeline.

Data Source: DOST-PAGASA Climate Data Service
Stations: Port Area, NAIA, Science Garden
Period: 2020-2025 (daily observations)

Features Created:
- Rolling precipitation (3-day, 7-day sums)
- Temperature metrics (avg, range, heat index)
- Monsoon/seasonal indicators
- Rain streak (consecutive rain days)
- Flood risk classification

Usage:
    python scripts/preprocess_pagasa_data.py
    python scripts/preprocess_pagasa_data.py --station naia
    python scripts/preprocess_pagasa_data.py --merge-flood-records
    python scripts/preprocess_pagasa_data.py --create-training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Optional, List
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Station metadata from PAGASA ReadMe
STATIONS = {
    'port_area': {
        'file': 'Floodingnaque_CADS-S0126006_Port Area Daily Data.csv',
        'latitude': 14.58841,
        'longitude': 120.967866,
        'elevation': 15,
        'name': 'Port Area'
    },
    'naia': {
        'file': 'Floodingnaque_CADS-S0126006_NAIA Daily Data.csv',
        'latitude': 14.5047,
        'longitude': 121.004751,
        'elevation': 21,
        'name': 'NAIA'
    },
    'science_garden': {
        'file': 'Floodingnaque_CADS-S0126006_Science Garden Daily Data.csv',
        'latitude': 14.645072,
        'longitude': 121.044282,
        'elevation': 42,
        'name': 'Science Garden'
    }
}

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DATA_DIR = BACKEND_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

# Physical bounds for validation
VALID_RANGES = {
    'temperature': (15, 45),      # °C - Metro Manila range
    'humidity': (20, 100),        # %
    'precipitation': (0, 500),    # mm/day (historical max ~450mm typhoon)
    'wind_speed': (0, 50),        # m/s (extreme typhoon)
}

# Month to season mapping for Philippines
MONTH_TO_SEASON = {
    1: 'dry', 2: 'dry', 3: 'dry', 4: 'dry', 5: 'dry',
    6: 'wet', 7: 'wet', 8: 'wet', 9: 'wet', 10: 'wet', 11: 'wet',
    12: 'dry'
}

# PAGASA Rainfall Intensity Classification (mm/hr)
# Adapted for daily totals
RAINFALL_INTENSITY = {
    'none': (0, 0.1),
    'light': (0.1, 7.5),
    'moderate': (7.5, 30),
    'heavy': (30, 75),
    'intense': (75, 150),
    'torrential': (150, float('inf'))
}


def load_pagasa_data(station_key: str) -> pd.DataFrame:
    """Load and validate PAGASA data for a specific station."""
    if station_key not in STATIONS:
        raise ValueError(f"Unknown station: {station_key}. Available: {list(STATIONS.keys())}")
    
    station = STATIONS[station_key]
    file_path = DATA_DIR / station['file']
    
    if not file_path.exists():
        raise FileNotFoundError(f"Station data not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records from {station['name']} ({station['file']})")
    
    return df


def validate_pagasa_data(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and handle physically impossible values."""
    df = df.copy()
    validation_issues = {}
    
    for col, (min_val, max_val) in VALID_RANGES.items():
        if col in df.columns:
            invalid = (df[col] < min_val) | (df[col] > max_val)
            if invalid.any():
                count = invalid.sum()
                validation_issues[col] = count
                logger.warning(f"{col}: {count} values outside valid range [{min_val}, {max_val}]")
                df.loc[invalid, col] = np.nan
    
    if validation_issues:
        logger.info(f"Total validation issues: {sum(validation_issues.values())}")
    
    return df


def clean_pagasa_data(df: pd.DataFrame, station_key: str) -> pd.DataFrame:
    """Clean and transform PAGASA data."""
    df = df.copy()
    station = STATIONS[station_key]
    
    # Create date column
    df['date'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    
    # Handle PAGASA special values
    # -999 = Missing value
    # -1 = Trace rainfall (< 0.1mm)
    for col in ['RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION']:
        if col in df.columns:
            df[col] = df[col].replace([-999, -999.0], np.nan)
    
    # Handle trace rainfall (-1 → 0.05mm)
    if 'RAINFALL' in df.columns:
        df['RAINFALL'] = df['RAINFALL'].replace([-1, -1.0], 0.05)
    
    # Calculate average temperature from TMAX/TMIN
    if 'TMAX' in df.columns and 'TMIN' in df.columns:
        df['temperature'] = (df['TMAX'] + df['TMIN']) / 2
        df['temperature_kelvin'] = df['temperature'] + 273.15
        df['temp_range'] = df['TMAX'] - df['TMIN']
    
    # Rename columns to match training pipeline
    column_mapping = {
        'RAINFALL': 'precipitation',
        'RH': 'humidity',
        'WIND_SPEED': 'wind_speed',
        'WIND_DIRECTION': 'wind_direction',
        'YEAR': 'year',
        'MONTH': 'month',
        'DAY': 'day',
        'TMAX': 'temp_max',
        'TMIN': 'temp_min'
    }
    df = df.rename(columns=column_mapping)
    
    # Add station metadata
    df['latitude'] = station['latitude']
    df['longitude'] = station['longitude']
    df['station'] = station['name']
    df['elevation'] = station['elevation']
    
    # Add seasonal features
    df['season'] = df['month'].map(MONTH_TO_SEASON)
    
    # Monsoon season indicator (June-November)
    df['is_monsoon_season'] = df['month'].isin([6, 7, 8, 9, 10, 11]).astype(int)
    
    # Heat index calculation
    df['heat_index'] = calculate_heat_index(df['temperature'], df['humidity'])
    
    # Validate after transformation
    df = validate_pagasa_data(df)
    
    return df


def calculate_heat_index(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """
    Calculate heat index from temperature (°C) and relative humidity (%).
    Uses the simplified Steadman formula for most cases.
    """
    # Convert to Fahrenheit for standard formula
    temp_f = temp_c * 9/5 + 32
    
    # Simple formula for most cases
    hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (rh * 0.094))
    
    # Use Rothfusz regression for high temperatures
    mask = hi >= 80
    if mask.any():
        hi_full = (
            -42.379 + 2.04901523 * temp_f
            + 10.14333127 * rh
            - 0.22475541 * temp_f * rh
            - 0.00683783 * temp_f**2
            - 0.05481717 * rh**2
            + 0.00122874 * temp_f**2 * rh
            + 0.00085282 * temp_f * rh**2
            - 0.00000199 * temp_f**2 * rh**2
        )
        hi = np.where(mask, hi_full, hi)
    
    # Convert back to Celsius
    return pd.Series((hi - 32) * 5/9, index=temp_c.index)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling/lagged features for temporal patterns."""
    df = df.sort_values('date').copy()
    
    # Rolling precipitation features (crucial for flood prediction)
    if 'precipitation' in df.columns:
        df['precip_3day_sum'] = df['precipitation'].rolling(window=3, min_periods=1).sum()
        df['precip_7day_sum'] = df['precipitation'].rolling(window=7, min_periods=1).sum()
        df['precip_14day_sum'] = df['precipitation'].rolling(window=14, min_periods=1).sum()
        df['precip_3day_avg'] = df['precipitation'].rolling(window=3, min_periods=1).mean()
        df['precip_7day_avg'] = df['precipitation'].rolling(window=7, min_periods=1).mean()
        df['precip_max_3day'] = df['precipitation'].rolling(window=3, min_periods=1).max()
        df['precip_max_7day'] = df['precipitation'].rolling(window=7, min_periods=1).max()
        
        # Lagged features (previous days)
        df['precip_lag1'] = df['precipitation'].shift(1)
        df['precip_lag2'] = df['precipitation'].shift(2)
        df['precip_lag3'] = df['precipitation'].shift(3)
        
        # Rain streak (consecutive rain days)
        df['is_rain'] = (df['precipitation'] > 0.1).astype(int)
        df['rain_streak'] = df['is_rain'].groupby(
            (df['is_rain'] != df['is_rain'].shift()).cumsum()
        ).cumcount() + 1
        df.loc[df['is_rain'] == 0, 'rain_streak'] = 0
    
    # Rolling humidity features
    if 'humidity' in df.columns:
        df['humidity_3day_avg'] = df['humidity'].rolling(window=3, min_periods=1).mean()
        df['humidity_7day_avg'] = df['humidity'].rolling(window=7, min_periods=1).mean()
        df['humidity_lag1'] = df['humidity'].shift(1)
    
    # Temperature features
    if 'temperature' in df.columns:
        df['temp_3day_avg'] = df['temperature'].rolling(window=3, min_periods=1).mean()
        df['temp_7day_avg'] = df['temperature'].rolling(window=7, min_periods=1).mean()
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between variables."""
    df = df.copy()
    
    # Temperature-Humidity interaction (affects evaporation/saturation)
    if all(c in df.columns for c in ['temperature', 'humidity']):
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    
    # Precipitation-Humidity interaction (soil saturation)
    if all(c in df.columns for c in ['humidity', 'precipitation']):
        df['humidity_precip_interaction'] = df['humidity'] * np.log1p(df['precipitation'])
    
    # Temperature-Precipitation interaction
    if all(c in df.columns for c in ['temperature', 'precipitation']):
        df['temp_precip_interaction'] = df['temperature'] * np.log1p(df['precipitation'])
    
    # Monsoon-Precipitation interaction
    if all(c in df.columns for c in ['is_monsoon_season', 'precipitation']):
        df['monsoon_precip_interaction'] = df['is_monsoon_season'] * df['precipitation']
    
    # Wind-Rain interaction (monsoon patterns)
    if all(c in df.columns for c in ['wind_speed', 'precipitation']):
        df['wind_rain_interaction'] = df['wind_speed'] * np.log1p(df['precipitation'])
    
    # Wind direction indicator (SW monsoon from 180-270°)
    if 'wind_direction' in df.columns:
        df['is_sw_monsoon_wind'] = (
            (df['wind_direction'] >= 180) & (df['wind_direction'] <= 270)
        ).astype(int)
    
    # High humidity + high precipitation = extreme flood risk
    if all(c in df.columns for c in ['humidity', 'precipitation']):
        df['saturation_risk'] = (
            (df['humidity'] > 85) & (df['precipitation'] > 20)
        ).astype(int)
    
    return df


def classify_flood_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add flood classification based on precipitation thresholds.
    
    Classification is based on:
    1. Daily precipitation thresholds
    2. Cumulative 3-day rainfall (ground saturation)
    3. Combined risk assessment
    
    Risk Levels:
    - 0 (LOW): Normal conditions
    - 1 (MODERATE): Elevated flood risk
    - 2 (HIGH): Significant flood risk
    """
    df = df.copy()
    
    # Daily precipitation-based risk
    # Thresholds based on PAGASA rainfall intensity and local flood patterns
    daily_conditions = [
        (df['precipitation'] < 20),                                # Low risk
        (df['precipitation'] >= 20) & (df['precipitation'] < 50),  # Moderate
        (df['precipitation'] >= 50)                                # High risk
    ]
    df['flood_risk_daily'] = np.select(daily_conditions, [0, 1, 2], default=0)
    
    # Cumulative 3-day rainfall risk (saturated ground)
    if 'precip_3day_sum' in df.columns:
        cum_conditions = [
            (df['precip_3day_sum'] < 40),
            (df['precip_3day_sum'] >= 40) & (df['precip_3day_sum'] < 80),
            (df['precip_3day_sum'] >= 80)
        ]
        df['flood_risk_cumulative'] = np.select(cum_conditions, [0, 1, 2], default=0)
    else:
        df['flood_risk_cumulative'] = df['flood_risk_daily']
    
    # Rain streak-based risk
    if 'rain_streak' in df.columns:
        streak_conditions = [
            (df['rain_streak'] < 3),
            (df['rain_streak'] >= 3) & (df['rain_streak'] < 5),
            (df['rain_streak'] >= 5)
        ]
        df['flood_risk_streak'] = np.select(streak_conditions, [0, 1, 2], default=0)
    else:
        df['flood_risk_streak'] = 0
    
    # Combined risk level (maximum of all indicators)
    risk_columns = ['flood_risk_daily', 'flood_risk_cumulative', 'flood_risk_streak']
    df['risk_level'] = df[risk_columns].max(axis=1)
    
    # Binary flood indicator (risk_level >= 1)
    df['flood'] = (df['risk_level'] >= 1).astype(int)
    
    # Add flood probability estimate based on multiple factors
    df['flood_probability'] = (
        0.4 * df['flood_risk_daily'] / 2 +
        0.4 * df['flood_risk_cumulative'] / 2 +
        0.2 * df['flood_risk_streak'] / 2
    )
    
    return df


def merge_with_flood_records(
    weather_df: pd.DataFrame,
    flood_records_dir: Path = PROCESSED_DIR
) -> pd.DataFrame:
    """
    Merge PAGASA weather data with official flood records.
    
    This updates flood labels based on actual flood events
    from Parañaque City official records.
    """
    weather_df = weather_df.copy()
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    # Load processed flood records
    flood_files = sorted(flood_records_dir.glob('processed_flood_records_*.csv'))
    
    if not flood_files:
        logger.warning("No processed flood records found. Using precipitation-based labels only.")
        return weather_df
    
    logger.info(f"Found {len(flood_files)} flood record files")
    
    flood_records = []
    for f in flood_files:
        try:
            df_flood = pd.read_csv(f)
            flood_records.append(df_flood)
            logger.info(f"  Loaded: {f.name} ({len(df_flood)} records)")
        except Exception as e:
            logger.warning(f"  Error loading {f}: {e}")
    
    if not flood_records:
        return weather_df
    
    all_floods = pd.concat(flood_records, ignore_index=True)
    
    # Extract confirmed flood dates
    if all(c in all_floods.columns for c in ['year', 'month']):
        # Create month-level flood indicators
        flood_months = set()
        for _, row in all_floods.iterrows():
            try:
                flood_months.add((int(row['year']), int(row['month'])))
            except (ValueError, KeyError):
                continue
        
        # Mark weather data with confirmed flood months
        weather_df['confirmed_flood_month'] = weather_df.apply(
            lambda x: 1 if (x['year'], x['month']) in flood_months else 0,
            axis=1
        )
        
        logger.info(f"Matched {weather_df['confirmed_flood_month'].sum()} days in flood months")
        
        # Update flood label for confirmed flood months
        # Only upgrade risk, don't downgrade
        weather_df.loc[
            (weather_df['confirmed_flood_month'] == 1) & 
            (weather_df['precipitation'] > 10),
            'flood'
        ] = 1
    
    return weather_df


def process_single_station(station_key: str, merge_flood_records: bool = False) -> pd.DataFrame:
    """Process a single PAGASA station completely."""
    logger.info(f"Processing {STATIONS[station_key]['name']} station...")
    
    df = load_pagasa_data(station_key)
    df = clean_pagasa_data(df, station_key)
    df = add_rolling_features(df)
    df = add_interaction_features(df)
    df = classify_flood_risk(df)
    
    if merge_flood_records:
        df = merge_with_flood_records(df)
    
    # Save processed file
    output_file = PROCESSED_DIR / f'pagasa_{station_key}_processed.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Saved: {output_file} ({len(df)} records)")
    
    # Print summary
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  Flood events: {df['flood'].sum()} ({df['flood'].mean()*100:.1f}%)")
    logger.info(f"  Risk distribution: {df['risk_level'].value_counts().to_dict()}")
    
    return df


def process_all_stations(merge_flood_records: bool = True) -> pd.DataFrame:
    """Process all PAGASA stations and create merged dataset."""
    all_data = []
    
    for station_key in STATIONS:
        try:
            df = process_single_station(station_key, merge_flood_records=False)
            all_data.append(df)
        except Exception as e:
            logger.error(f"Error processing {station_key}: {e}")
    
    if not all_data:
        raise ValueError("No station data could be processed")
    
    # Merge all stations
    merged = pd.concat(all_data, ignore_index=True)
    logger.info(f"Merged all stations: {len(merged)} total records")
    
    if merge_flood_records:
        merged = merge_with_flood_records(merged)
    
    # Save merged dataset
    output_file = PROCESSED_DIR / 'pagasa_all_stations_merged.csv'
    merged.to_csv(output_file, index=False)
    logger.info(f"Saved merged dataset: {output_file}")
    
    return merged


def create_training_dataset(
    use_naia_only: bool = True,
    include_flood_records: bool = True
) -> pd.DataFrame:
    """
    Create final training dataset optimized for flood prediction.
    
    Args:
        use_naia_only: Use only NAIA station (closest to Parañaque)
        include_flood_records: Merge with official flood records
    """
    logger.info("Creating training dataset...")
    
    if use_naia_only:
        df = process_single_station('naia', merge_flood_records=include_flood_records)
    else:
        df = process_all_stations(merge_flood_records=include_flood_records)
    
    # Select features for training
    # Core features (required)
    core_features = ['temperature', 'humidity', 'precipitation']
    
    # Target variables
    target_features = ['flood', 'risk_level']
    
    # Temporal features
    temporal_features = ['month', 'is_monsoon_season', 'year', 'season']
    
    # Rolling/lagged features (high importance for flood prediction)
    rolling_features = [
        'precip_3day_sum', 'precip_7day_sum', 'precip_3day_avg',
        'precip_7day_avg', 'precip_max_3day', 'precip_max_7day',
        'precip_lag1', 'precip_lag2', 'rain_streak',
        'humidity_3day_avg', 'humidity_lag1'
    ]
    
    # Derived features
    derived_features = [
        'temp_range', 'heat_index', 'elevation',
        'temp_humidity_interaction', 'humidity_precip_interaction',
        'monsoon_precip_interaction', 'saturation_risk'
    ]
    
    # Spatial features
    spatial_features = ['latitude', 'longitude', 'station']
    
    # Collect all available features
    all_training_features = (
        core_features + target_features + temporal_features +
        rolling_features + derived_features + spatial_features
    )
    
    # Filter to columns that exist
    available_features = [f for f in all_training_features if f in df.columns]
    missing_features = [f for f in all_training_features if f not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features (will be excluded): {missing_features}")
    
    # Create training dataset
    training_df = df[available_features].copy()
    
    # Drop rows with missing core features
    training_df = training_df.dropna(subset=core_features)
    
    # Fill remaining NaN with appropriate values
    numeric_cols = training_df.select_dtypes(include=[np.number]).columns
    training_df[numeric_cols] = training_df[numeric_cols].fillna(training_df[numeric_cols].median())
    
    # Save training dataset
    output_file = PROCESSED_DIR / 'pagasa_training_dataset.csv'
    training_df.to_csv(output_file, index=False)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING DATASET CREATED: {output_file}")
    logger.info(f"{'='*60}")
    logger.info(f"Total records: {len(training_df)}")
    logger.info(f"Features: {len(available_features)}")
    logger.info(f"Flood events: {training_df['flood'].sum()} ({training_df['flood'].mean()*100:.1f}%)")
    logger.info(f"Risk level distribution:")
    for level, count in training_df['risk_level'].value_counts().sort_index().items():
        logger.info(f"  Level {level}: {count} ({count/len(training_df)*100:.1f}%)")
    logger.info(f"Year range: {training_df['year'].min()} - {training_df['year'].max()}")
    logger.info(f"{'='*60}\n")
    
    return training_df


def print_data_summary(df: pd.DataFrame, title: str = "Data Summary"):
    """Print comprehensive data summary."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    print(f"\nNumeric statistics:")
    print(df.describe().round(2))
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    print(f"{'='*60}\n")


def main():
    """Main entry point for PAGASA data preprocessing."""
    parser = argparse.ArgumentParser(
        description='Preprocess PAGASA weather station data for flood prediction'
    )
    parser.add_argument(
        '--station',
        choices=['naia', 'port_area', 'science_garden', 'all'],
        default='all',
        help='Station to process (default: all)'
    )
    parser.add_argument(
        '--merge-flood-records',
        action='store_true',
        help='Merge with official Parañaque flood records'
    )
    parser.add_argument(
        '--create-training',
        action='store_true',
        help='Create final training dataset'
    )
    parser.add_argument(
        '--naia-only',
        action='store_true',
        help='Use only NAIA station for training (closest to Parañaque)'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print detailed data summary'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.create_training:
        df = create_training_dataset(
            use_naia_only=args.naia_only,
            include_flood_records=True
        )
        if args.summary:
            print_data_summary(df, "Training Dataset Summary")
    
    elif args.station == 'all':
        df = process_all_stations(merge_flood_records=args.merge_flood_records)
        if args.summary:
            print_data_summary(df, "All Stations Merged Summary")
    
    else:
        df = process_single_station(
            args.station,
            merge_flood_records=args.merge_flood_records
        )
        if args.summary:
            print_data_summary(df, f"{STATIONS[args.station]['name']} Station Summary")
    
    logger.info("Processing complete!")


if __name__ == '__main__':
    main()
