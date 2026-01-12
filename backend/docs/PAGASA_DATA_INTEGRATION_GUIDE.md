# PAGASA Climate Data Integration Guide for Model Training
## Floodingnaque Flood Prediction System Enhancement

**Generated:** January 12, 2026  
**Purpose:** Integrating DOST-PAGASA Weather Station Data to Enhance Flood Prediction Models

---

## Executive Summary

This guide documents how to integrate the **DOST-PAGASA Climate Data** from three Metro Manila weather stations into the Floodingnaque flood prediction model training pipeline. These datasets provide **high-quality, ground-truth meteorological observations** that can significantly improve model accuracy and real-world applicability.

### Available PAGASA Datasets

| Dataset File | Station | Coordinates | Elevation | Records |
|--------------|---------|-------------|-----------|---------|
| `Floodingnaque_CADS-S0126006_NAIA Daily Data.csv` | NAIA | 14.5047°N, 121.0048°E | 21m | ~1,829 days |
| `Floodingnaque_CADS-S0126006_Port Area Daily Data.csv` | Port Area | 14.5884°N, 120.9679°E | 15m | ~1,402 days |
| `Floodingnaque_CADS-S0126006_Science Garden Daily Data.csv` | Science Garden | 14.6451°N, 121.0443°E | 42m | ~1,829 days |

**Data Period:** 2020-2025 (daily observations)

---

## 1. Understanding the PAGASA Data Structure

### 1.1 Column Definitions

Based on the PAGASA ReadMe file (`Floodingnaque_CADS-S0126006_A.ReadMe.txt`):

| Column | Description | Unit | Notes |
|--------|-------------|------|-------|
| `YEAR` | Year of observation | - | 2020-2025 |
| `MONTH` | Month (1-12) | - | - |
| `DAY` | Day of month | - | - |
| `RAINFALL` | Daily precipitation | mm | `-1` = Trace (<0.1mm), `-999` = Missing |
| `TMAX` | Maximum temperature | °C | - |
| `TMIN` | Minimum temperature | °C | - |
| `RH` | Relative Humidity | % | - |
| `WIND_SPEED` | Wind speed | m/s | - |
| `WIND_DIRECTION` | Wind direction | degrees | 0-360° from North |

### 1.2 Special Values

```
-999.0 → Missing Value (exclude from training or impute)
-1.0   → Trace rainfall (less than 0.1mm, treat as ~0.05mm)
0      → No rainfall
```

### 1.3 Station Proximity to Parañaque City

```
Parañaque City Center: 14.4793°N, 121.0198°E

Distances (approximate):
- Port Area:      ~8.3 km NW  (closest coastal station)
- NAIA:           ~3.0 km NE  (closest station overall)
- Science Garden: ~18.5 km N  (higher elevation reference)
```

**Recommendation:** Prioritize **NAIA** station data for Parañaque predictions due to proximity.

---

## 2. Data Preprocessing Script

Create a preprocessing script to transform PAGASA data for model training:

### 2.1 Create `preprocess_pagasa_data.py`

```python
"""
Preprocess PAGASA Weather Station Data for Flood Model Training
================================================================

Transforms raw PAGASA climate data into ML-ready format compatible
with the existing Floodingnaque training pipeline.

Usage:
    python scripts/preprocess_pagasa_data.py
    python scripts/preprocess_pagasa_data.py --station naia
    python scripts/preprocess_pagasa_data.py --merge-all
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Station metadata
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
DATA_DIR = SCRIPT_DIR.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

def load_pagasa_data(station_key: str) -> pd.DataFrame:
    """Load and validate PAGASA data for a specific station."""
    station = STATIONS[station_key]
    file_path = DATA_DIR / station['file']
    
    if not file_path.exists():
        raise FileNotFoundError(f"Station data not found: {file_path}")
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} records from {station['name']}")
    
    return df

def clean_pagasa_data(df: pd.DataFrame, station_key: str) -> pd.DataFrame:
    """Clean and transform PAGASA data."""
    df = df.copy()
    station = STATIONS[station_key]
    
    # Create date column
    df['date'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    
    # Handle missing values (-999)
    for col in ['RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION']:
        if col in df.columns:
            df[col] = df[col].replace(-999, np.nan)
            df[col] = df[col].replace(-999.0, np.nan)
    
    # Handle trace rainfall (-1 → 0.05mm)
    df['RAINFALL'] = df['RAINFALL'].replace(-1, 0.05)
    df['RAINFALL'] = df['RAINFALL'].replace(-1.0, 0.05)
    
    # Calculate average temperature from TMAX/TMIN
    df['temperature'] = (df['TMAX'] + df['TMIN']) / 2
    
    # Convert temperature to Kelvin for consistency (optional)
    df['temperature_kelvin'] = df['temperature'] + 273.15
    
    # Rename columns to match training pipeline
    df = df.rename(columns={
        'RAINFALL': 'precipitation',
        'RH': 'humidity',
        'WIND_SPEED': 'wind_speed',
        'WIND_DIRECTION': 'wind_direction',
        'YEAR': 'year',
        'MONTH': 'month',
        'DAY': 'day'
    })
    
    # Add station metadata
    df['latitude'] = station['latitude']
    df['longitude'] = station['longitude']
    df['station'] = station['name']
    df['elevation'] = station['elevation']
    
    # Add seasonal features
    df['season'] = df['month'].map({
        1: 'dry', 2: 'dry', 3: 'dry', 4: 'dry', 5: 'dry',
        6: 'wet', 7: 'wet', 8: 'wet', 9: 'wet', 10: 'wet', 11: 'wet',
        12: 'dry'
    })
    
    # Monsoon season indicator (June-November)
    df['is_monsoon_season'] = df['month'].isin([6, 7, 8, 9, 10, 11]).astype(int)
    
    # Temperature range (diurnal variation)
    df['temp_range'] = df['TMAX'] - df['TMIN']
    
    # Apparent temperature / Heat Index approximation
    df['heat_index'] = calculate_heat_index(df['temperature'], df['humidity'])
    
    return df

def calculate_heat_index(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """Calculate heat index from temperature (°C) and relative humidity (%)."""
    # Simplified Steadman formula
    temp_f = temp_c * 9/5 + 32  # Convert to Fahrenheit
    
    hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (rh * 0.094))
    
    # For high temperatures, use more complex formula
    mask = hi > 80
    if mask.any():
        hi[mask] = (
            -42.379 + 2.04901523 * temp_f[mask] + 10.14333127 * rh[mask]
            - 0.22475541 * temp_f[mask] * rh[mask]
            - 0.00683783 * temp_f[mask]**2 - 0.05481717 * rh[mask]**2
            + 0.00122874 * temp_f[mask]**2 * rh[mask]
            + 0.00085282 * temp_f[mask] * rh[mask]**2
            - 0.00000199 * temp_f[mask]**2 * rh[mask]**2
        )
    
    # Convert back to Celsius
    return (hi - 32) * 5/9

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling/lagged features for temporal patterns."""
    df = df.sort_values('date').copy()
    
    # Rolling precipitation features (crucial for flood prediction)
    df['precip_3day_sum'] = df['precipitation'].rolling(window=3, min_periods=1).sum()
    df['precip_7day_sum'] = df['precipitation'].rolling(window=7, min_periods=1).sum()
    df['precip_3day_avg'] = df['precipitation'].rolling(window=3, min_periods=1).mean()
    df['precip_7day_avg'] = df['precipitation'].rolling(window=7, min_periods=1).mean()
    df['precip_max_3day'] = df['precipitation'].rolling(window=3, min_periods=1).max()
    
    # Rolling humidity features
    df['humidity_3day_avg'] = df['humidity'].rolling(window=3, min_periods=1).mean()
    
    # Lagged features (previous day)
    df['precip_lag1'] = df['precipitation'].shift(1)
    df['precip_lag2'] = df['precipitation'].shift(2)
    df['humidity_lag1'] = df['humidity'].shift(1)
    
    # Rain streak (consecutive rain days)
    df['is_rain'] = (df['precipitation'] > 0.1).astype(int)
    df['rain_streak'] = df['is_rain'].groupby(
        (df['is_rain'] != df['is_rain'].shift()).cumsum()
    ).cumcount() + 1
    df.loc[df['is_rain'] == 0, 'rain_streak'] = 0
    
    return df

def classify_flood_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add flood classification based on precipitation thresholds.
    
    These thresholds are based on PAGASA rainfall intensity classification:
    - Light rain:    0.1 - 2.5 mm/hr → No flood (0)
    - Moderate rain: 2.5 - 7.5 mm/hr → Low risk
    - Heavy rain:    7.5 - 15 mm/hr  → Moderate risk
    - Intense rain:  15 - 30 mm/hr   → High risk
    - Torrential:    > 30 mm/hr      → Very high risk
    
    For daily totals, we use accumulated thresholds.
    """
    df = df.copy()
    
    # Binary flood indicator based on daily precipitation thresholds
    # Adjusted for Parañaque's flood-prone areas
    conditions = [
        (df['precipitation'] < 20),                              # No flood likely
        (df['precipitation'] >= 20) & (df['precipitation'] < 50),  # Low-Moderate
        (df['precipitation'] >= 50)                              # High flood risk
    ]
    
    # More sophisticated classification considering cumulative rainfall
    df['flood_risk_precip'] = np.select(conditions, [0, 1, 2], default=0)
    
    # Consider 3-day cumulative rainfall for saturated ground
    cum_conditions = [
        (df['precip_3day_sum'] < 40),
        (df['precip_3day_sum'] >= 40) & (df['precip_3day_sum'] < 80),
        (df['precip_3day_sum'] >= 80)
    ]
    df['flood_risk_cumulative'] = np.select(cum_conditions, [0, 1, 2], default=0)
    
    # Combined risk level (max of both indicators)
    df['risk_level'] = df[['flood_risk_precip', 'flood_risk_cumulative']].max(axis=1)
    
    # Binary flood indicator
    df['flood'] = (df['risk_level'] >= 1).astype(int)
    
    return df

def merge_with_flood_records(
    weather_df: pd.DataFrame,
    flood_records_dir: Path = PROCESSED_DIR
) -> pd.DataFrame:
    """
    Merge PAGASA weather data with official flood records.
    
    This creates a supervised training dataset by matching
    weather conditions to actual flood events.
    """
    weather_df = weather_df.copy()
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    # Load processed flood records
    flood_files = list(flood_records_dir.glob('processed_flood_records_*.csv'))
    
    if not flood_files:
        logger.warning("No processed flood records found. Using precipitation-based labels.")
        return weather_df
    
    flood_records = []
    for f in flood_files:
        try:
            df = pd.read_csv(f)
            flood_records.append(df)
        except Exception as e:
            logger.warning(f"Error loading {f}: {e}")
    
    if flood_records:
        all_floods = pd.concat(flood_records, ignore_index=True)
        
        # Extract dates with confirmed floods
        if 'year' in all_floods.columns and 'month' in all_floods.columns:
            all_floods['flood_date'] = pd.to_datetime(
                all_floods[['year', 'month']].assign(day=15)
            )
            
            # Get unique flood months
            flood_months = set(all_floods['flood_date'].dt.to_period('M'))
            weather_df['period'] = weather_df['date'].dt.to_period('M')
            
            # Mark months with confirmed floods
            weather_df['confirmed_flood'] = weather_df['period'].isin(flood_months).astype(int)
            
            logger.info(f"Matched {weather_df['confirmed_flood'].sum()} days in flood months")
    
    return weather_df

def process_all_stations(merge_flood_records: bool = True) -> pd.DataFrame:
    """Process all PAGASA stations and optionally merge with flood records."""
    all_data = []
    
    for station_key in STATIONS:
        try:
            logger.info(f"Processing {STATIONS[station_key]['name']}...")
            
            df = load_pagasa_data(station_key)
            df = clean_pagasa_data(df, station_key)
            df = add_rolling_features(df)
            df = classify_flood_risk(df)
            
            all_data.append(df)
            
            # Save individual station file
            output_file = PROCESSED_DIR / f'pagasa_{station_key}_processed.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"Saved: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {station_key}: {e}")
    
    if all_data:
        merged = pd.concat(all_data, ignore_index=True)
        
        if merge_flood_records:
            merged = merge_with_flood_records(merged)
        
        # Save merged dataset
        output_file = PROCESSED_DIR / 'pagasa_all_stations_merged.csv'
        merged.to_csv(output_file, index=False)
        logger.info(f"Saved merged dataset: {output_file} ({len(merged)} records)")
        
        return merged
    
    return pd.DataFrame()

def create_training_dataset(
    use_naia_only: bool = True,
    include_synthetic_negative: bool = True
) -> pd.DataFrame:
    """
    Create final training dataset optimized for flood prediction.
    
    Args:
        use_naia_only: Use only NAIA station (closest to Parañaque)
        include_synthetic_negative: Add non-flood days to balance dataset
    """
    if use_naia_only:
        df = load_pagasa_data('naia')
        df = clean_pagasa_data(df, 'naia')
    else:
        df = process_all_stations(merge_flood_records=False)
    
    df = add_rolling_features(df)
    df = classify_flood_risk(df)
    
    # Merge with official flood records
    df = merge_with_flood_records(df)
    
    # Select features for training
    training_features = [
        'temperature', 'humidity', 'precipitation', 'wind_speed',
        'month', 'is_monsoon_season', 'year',
        'precip_3day_sum', 'precip_7day_sum', 'precip_3day_avg',
        'heat_index', 'temp_range', 'rain_streak',
        'flood', 'risk_level',
        'latitude', 'longitude', 'season'
    ]
    
    # Filter to available columns
    available_features = [f for f in training_features if f in df.columns]
    training_df = df[available_features].dropna(subset=['temperature', 'humidity', 'precipitation'])
    
    output_file = PROCESSED_DIR / 'pagasa_training_dataset.csv'
    training_df.to_csv(output_file, index=False)
    logger.info(f"Created training dataset: {output_file}")
    logger.info(f"  Records: {len(training_df)}")
    logger.info(f"  Flood events: {training_df['flood'].sum()}")
    logger.info(f"  Features: {available_features}")
    
    return training_df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess PAGASA weather data')
    parser.add_argument('--station', choices=['naia', 'port_area', 'science_garden', 'all'],
                        default='all', help='Station to process')
    parser.add_argument('--merge-flood-records', action='store_true',
                        help='Merge with official flood records')
    parser.add_argument('--create-training', action='store_true',
                        help='Create final training dataset')
    
    args = parser.parse_args()
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.create_training:
        create_training_dataset()
    elif args.station == 'all':
        process_all_stations(merge_flood_records=args.merge_flood_records)
    else:
        df = load_pagasa_data(args.station)
        df = clean_pagasa_data(df, args.station)
        df = add_rolling_features(df)
        df = classify_flood_risk(df)
        
        output_file = PROCESSED_DIR / f'pagasa_{args.station}_processed.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
```

---

## 3. Feature Engineering Enhancements

### 3.1 New Features from PAGASA Data

The PAGASA datasets enable several powerful features not available in synthetic data:

| Feature | Source | Importance | Description |
|---------|--------|------------|-------------|
| `temp_range` | TMAX - TMIN | Medium | Diurnal temperature variation |
| `heat_index` | Temp + Humidity | Medium | Apparent temperature |
| `precip_3day_sum` | Rolling RAINFALL | **High** | Cumulative rainfall (ground saturation) |
| `precip_7day_sum` | Rolling RAINFALL | **High** | Weekly rainfall accumulation |
| `rain_streak` | Consecutive days | Medium | Soil saturation indicator |
| `wind_direction` | WIND_DIRECTION | Low | Wind-driven rain patterns |
| `elevation` | Station metadata | Medium | Topographic context |

### 3.2 Interaction Features

Add these to `train_production.py`:

```python
# PAGASA-specific feature engineering
def engineer_pagasa_features(df: pd.DataFrame) -> pd.DataFrame:
    """Additional features specific to PAGASA data."""
    df = df.copy()
    
    # Temperature variability (larger range = unstable weather)
    if 'temp_range' in df.columns:
        df['temp_instability'] = df['temp_range'] / df['temperature'].clip(lower=1)
    
    # Precipitation intensity categories
    if 'precipitation' in df.columns:
        df['precip_category'] = pd.cut(
            df['precipitation'],
            bins=[-np.inf, 0.1, 7.5, 30, 75, np.inf],
            labels=['none', 'light', 'moderate', 'heavy', 'intense']
        )
    
    # Wind-rain interaction (monsoon indicator)
    if all(c in df.columns for c in ['wind_speed', 'precipitation']):
        df['wind_rain_interaction'] = df['wind_speed'] * np.log1p(df['precipitation'])
    
    # Humidity saturation risk
    if all(c in df.columns for c in ['humidity', 'precipitation']):
        df['saturation_risk'] = (df['humidity'] > 85) & (df['precipitation'] > 20)
        df['saturation_risk'] = df['saturation_risk'].astype(int)
    
    # Monsoon wind indicator (SW monsoon from 180-270°)
    if 'wind_direction' in df.columns:
        df['is_sw_monsoon_wind'] = (
            (df['wind_direction'] >= 180) & (df['wind_direction'] <= 270)
        ).astype(int)
    
    return df
```

---

## 4. Training Pipeline Integration

### 4.1 Update Training Configuration

Modify `train_production.py` to include PAGASA features:

```python
# Updated feature configuration for PAGASA data
NUMERIC_FEATURES = [
    # Core features
    'temperature', 'humidity', 'precipitation', 'wind_speed',
    # Temporal
    'month', 'is_monsoon_season', 'year',
    # PAGASA-specific (NEW)
    'temp_range', 'heat_index', 'elevation',
    # Rolling features (NEW)
    'precip_3day_sum', 'precip_7day_sum', 'precip_3day_avg',
    'humidity_3day_avg', 'rain_streak',
    # Lagged features (NEW)
    'precip_lag1', 'precip_lag2'
]

CATEGORICAL_FEATURES = [
    'weather_type', 'season', 
    'station'  # NEW - for multi-station models
]
```

### 4.2 Training Commands

```powershell
# Step 1: Preprocess PAGASA data
python scripts/preprocess_pagasa_data.py --create-training

# Step 2: Train with PAGASA-enhanced data
python scripts/train_production.py --data-path data/processed/pagasa_training_dataset.csv

# Step 3: Train with combined data (PAGASA + Official Records)
python scripts/train_production.py \
    --data-path data/processed/pagasa_training_dataset.csv \
    --additional-data data/processed/cumulative_up_to_2025.csv \
    --grid-search
```

---

## 5. Data Quality Considerations

### 5.1 Handling Missing Values

```python
# Missing value strategy for PAGASA data
MISSING_VALUE_STRATEGY = {
    'precipitation': 0,           # Missing rain = no rain
    'temperature': 'interpolate', # Linear interpolation
    'humidity': 'interpolate',    # Linear interpolation
    'wind_speed': 'median',       # Station median
    'wind_direction': 'mode',     # Most common direction
}
```

### 5.2 Outlier Detection

```python
# Physical bounds for validation
VALID_RANGES = {
    'temperature': (15, 45),      # °C - Metro Manila range
    'humidity': (30, 100),        # %
    'precipitation': (0, 500),    # mm/day (historical max ~450mm)
    'wind_speed': (0, 30),        # m/s
}

def validate_pagasa_data(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and optionally remove physically impossible values."""
    df = df.copy()
    
    for col, (min_val, max_val) in VALID_RANGES.items():
        if col in df.columns:
            invalid = (df[col] < min_val) | (df[col] > max_val)
            if invalid.any():
                logger.warning(f"{col}: {invalid.sum()} values outside valid range")
                df.loc[invalid, col] = np.nan
    
    return df
```

---

## 6. Multi-Station Analysis

### 6.1 Spatial Interpolation

For improved predictions at specific Parañaque locations:

```python
def interpolate_for_location(
    lat: float, lon: float,
    station_data: Dict[str, pd.DataFrame],
    method: str = 'idw'  # Inverse Distance Weighting
) -> pd.DataFrame:
    """
    Interpolate weather values for a specific location
    using data from multiple PAGASA stations.
    """
    distances = {}
    for station_key, station_info in STATIONS.items():
        dist = haversine(lat, lon, 
                        station_info['latitude'], 
                        station_info['longitude'])
        distances[station_key] = dist
    
    # Calculate IDW weights
    total_inv_dist = sum(1/d for d in distances.values())
    weights = {k: (1/v)/total_inv_dist for k, v in distances.items()}
    
    # Apply weighted average to each feature
    result_df = station_data[list(distances.keys())[0]].copy()
    
    for col in ['temperature', 'humidity', 'precipitation', 'wind_speed']:
        result_df[col] = sum(
            weights[k] * station_data[k][col] 
            for k in distances.keys()
        )
    
    return result_df
```

### 6.2 Station Comparison

```python
# Compare stations to identify local patterns
def compare_station_statistics():
    """Generate comparison statistics across stations."""
    stats = {}
    
    for station_key in STATIONS:
        df = load_pagasa_data(station_key)
        df = clean_pagasa_data(df, station_key)
        
        stats[station_key] = {
            'avg_temp': df['temperature'].mean(),
            'avg_humidity': df['humidity'].mean(),
            'total_precip': df['precipitation'].sum(),
            'rainy_days': (df['precipitation'] > 0.1).sum(),
            'heavy_rain_days': (df['precipitation'] > 50).sum(),
            'extreme_rain_days': (df['precipitation'] > 100).sum()
        }
    
    return pd.DataFrame(stats).T
```

---

## 7. Integration with Existing Flood Records

### 7.1 Date Matching Strategy

The PAGASA data can be matched with official flood records:

```python
def match_flood_events(
    weather_df: pd.DataFrame,
    flood_records: pd.DataFrame
) -> pd.DataFrame:
    """
    Match weather conditions to confirmed flood events.
    
    Strategy:
    1. For each flood event date, find corresponding weather data
    2. Update flood labels with confirmed events
    3. Keep weather-only days as potential negative samples
    """
    weather_df = weather_df.copy()
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date
    weather_df['confirmed_flood'] = 0
    
    # Parse flood record dates
    if 'year' in flood_records.columns and 'month' in flood_records.columns:
        for _, record in flood_records.iterrows():
            year = record['year']
            month = record['month']
            
            # Mark entire month as flood-prone (conservative approach)
            mask = (
                (weather_df['date'].apply(lambda x: x.year) == year) &
                (weather_df['date'].apply(lambda x: x.month) == month)
            )
            weather_df.loc[mask, 'confirmed_flood'] = 1
    
    return weather_df
```

### 7.2 Combined Dataset Creation

```powershell
# Create combined training dataset
python scripts/preprocess_pagasa_data.py --merge-flood-records

# Verify output
python -c "
import pandas as pd
df = pd.read_csv('data/processed/pagasa_all_stations_merged.csv')
print(f'Total records: {len(df)}')
print(f'Flood events: {df[\"flood\"].sum()}')
print(f'Features: {list(df.columns)}')
"
```

---

## 8. Model Enhancement Recommendations

### 8.1 Feature Importance Analysis

After training with PAGASA data, analyze feature importance:

```python
# Expected high-importance features
EXPECTED_TOP_FEATURES = [
    'precip_3day_sum',      # Cumulative rainfall - CRITICAL
    'precipitation',         # Daily rainfall
    'humidity',             # Moisture content
    'is_monsoon_season',    # Seasonal factor
    'precip_7day_sum',      # Weekly rainfall
    'rain_streak',          # Consecutive rain days
]
```

### 8.2 Recommended Model Configurations

```python
# Optimized for PAGASA-enhanced data
PRODUCTION_RF_PARAMS = {
    'n_estimators': 300,        # Increased for more features
    'max_depth': 25,            # Slightly deeper
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# Gradient Boosting alternative
GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'min_samples_split': 10,
    'subsample': 0.8
}
```

---

## 9. Validation Strategy

### 9.1 Temporal Cross-Validation

Important for weather data to prevent data leakage:

```python
from sklearn.model_selection import TimeSeriesSplit

def temporal_cross_validation(X, y, n_splits=5):
    """Time-based cross-validation for weather data."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = RandomForestClassifier(**PRODUCTION_RF_PARAMS)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)
```

### 9.2 Hold-Out Test Set

```python
# Reserve 2025 data for final validation
def create_temporal_split(df: pd.DataFrame):
    """Split data temporally: train on 2020-2024, test on 2025."""
    train_df = df[df['year'] < 2025]
    test_df = df[df['year'] == 2025]
    
    logger.info(f"Train: {len(train_df)} (2020-2024)")
    logger.info(f"Test: {len(test_df)} (2025)")
    
    return train_df, test_df
```

---

## 10. Quick Start Commands

```powershell
# Navigate to backend
cd d:\floodingnaque\backend

# Step 1: Preprocess all PAGASA stations
python scripts/preprocess_pagasa_data.py --station all

# Step 2: Create training dataset
python scripts/preprocess_pagasa_data.py --create-training

# Step 3: Verify data
python -c "
import pandas as pd
df = pd.read_csv('data/processed/pagasa_training_dataset.csv')
print('Dataset Statistics:')
print(df.describe())
print(f'\nClass Distribution:')
print(df['flood'].value_counts())
"

# Step 4: Train enhanced model
python scripts/train_production.py \
    --data-path data/processed/pagasa_training_dataset.csv \
    --grid-search \
    --generate-shap

# Step 5: Evaluate model
python scripts/evaluate_model.py \
    --model-path models/flood_rf_model.joblib \
    --data-path data/processed/pagasa_training_dataset.csv
```

---

## 11. Expected Improvements

| Metric | Before (Synthetic Only) | After (PAGASA Integration) |
|--------|------------------------|---------------------------|
| Data Volume | ~1,000 samples | ~5,000+ samples |
| Temporal Coverage | Limited | 2020-2025 (5 years) |
| Feature Count | 4-5 | 15-20+ |
| Feature Quality | Synthetic | Ground-truth observations |
| Spatial Resolution | Single point | 3 station network |
| Validation | Random split | Temporal validation |

---

## 12. Next Steps

1. **Execute preprocessing pipeline** to generate PAGASA training data
2. **Run comparative training** with and without PAGASA features
3. **Analyze feature importance** to validate PAGASA contributions
4. **Deploy updated model** with enhanced feature set
5. **Monitor real-time predictions** against actual flood events

---

## References

- PAGASA Climate Data Portal: http://bagong.pagasa.dost.gov.ph/climate/climate-data
- DOST-PAGASA ClimDatPh Paper: https://philjournalsci.dost.gov.ph/images/pdf/pjs_pdf/vol150no1/ClimDatPh_an_online_platform_for_data_acquisition_.pdf
- Floodingnaque Training Documentation: [MODEL_TRAINING_AUDIT_REPORT.md](MODEL_TRAINING_AUDIT_REPORT.md)
