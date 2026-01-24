# Floodingnaque Data Dictionary

This document provides comprehensive definitions for all data columns used in the
Floodingnaque flood prediction system, including PAGASA weather data, flood records,
and derived features.

> **Last Updated:** 2026-01-24  
> **Version:** 1.0

---

## Table of Contents

- [PAGASA Weather Station Data](#pagasa-weather-station-data)
  - [Column Definitions](#pagasa-column-definitions)
  - [Special Values](#special-values)
  - [Weather Stations](#weather-stations)
- [Official Flood Records](#official-flood-records)
  - [Raw Data Columns](#raw-data-columns)
  - [Cleaned Data Columns](#cleaned-data-columns)
- [Derived Features](#derived-features)
  - [Temporal Features](#temporal-features)
  - [Rainfall Features](#rainfall-features)
  - [Environmental Features](#environmental-features)
- [Target Variable](#target-variable)
- [Data Quality Codes](#data-quality-codes)

---

## PAGASA Weather Station Data

### Source Information

- **Provider:** DOST-PAGASA (Philippine Atmospheric, Geophysical and Astronomical Services Administration)
- **Data Portal:** http://bagong.pagasa.dost.gov.ph/climate/climate-data
- **Reference:** [ClimDatPh Paper](https://philjournalsci.dost.gov.ph/images/pdf/pjs_pdf/vol150no1/ClimDatPh_an_online_platform_for_data_acquisition_.pdf)

### PAGASA Column Definitions

| Column | Full Name | Data Type | Unit | Description |
|--------|-----------|-----------|------|-------------|
| `YEAR` | Year | Integer | - | Observation year (2020-2025) |
| `MONTH` | Month | Integer | - | Month of observation (1-12) |
| `DAY` | Day | Integer | - | Day of month (1-31) |
| `RAINFALL` | Daily Rainfall | Float | mm | Total precipitation for the day. Measured from 08:00 to 08:00 local time |
| `TMAX` | Maximum Temperature | Float | °C | Highest temperature recorded during the day |
| `TMIN` | Minimum Temperature | Float | °C | Lowest temperature recorded during the day |
| `RH` | Relative Humidity | Float | % | Daily mean relative humidity (0-100%) |
| `WIND_SPEED` | Wind Speed | Float | m/s | Daily average wind speed |
| `WIND_DIRECTION` | Wind Direction | Float | ° | Direction from which wind blows (0-360°, 0/360=North) |

### Special Values

| Value | Meaning | Column(s) | Handling Recommendation |
|-------|---------|-----------|-------------------------|
| `-999.0` | Missing Data | All numeric | Replace with `NaN` or impute using neighboring stations/dates |
| `-1.0` | Trace Rainfall | RAINFALL only | Indicates precipitation < 0.1mm; replace with `0.05` or `0` |
| `0.0` | No Rainfall | RAINFALL | Actual zero - no precipitation detected |
| `NaN` | Not Available | All | Impute or exclude from analysis |

#### Handling Missing Values (`-999.0`)

```python
# Recommended preprocessing
import pandas as pd
import numpy as np

df = pd.read_csv("NAIA_Daily_Data.csv")

# Replace -999.0 with NaN for all numeric columns
numeric_cols = ['RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION']
df[numeric_cols] = df[numeric_cols].replace(-999.0, np.nan)

# Handle trace rainfall
df['RAINFALL'] = df['RAINFALL'].replace(-1.0, 0.05)
```

### Weather Stations

| Station Code | Station Name | Latitude | Longitude | Elevation | Notes |
|--------------|--------------|----------|-----------|-----------|-------|
| `NAIA` | Ninoy Aquino Int'l Airport | 14.5047°N | 121.0048°E | 21 m | Primary station for Parañaque |
| `PORT_AREA` | Port Area Manila | 14.5884°N | 120.9679°E | 15 m | Coastal station, captures sea effects |
| `SCIENCE_GARDEN` | Science Garden Quezon City | 14.6451°N | 121.0443°E | 42 m | Higher elevation reference |

**Station Selection Notes:**
- NAIA is the primary station for Parañaque flood prediction due to proximity
- Port Area captures coastal weather effects relevant to tidal flooding
- Science Garden provides upland reference for spatial interpolation

---

## Official Flood Records

### Source Information

- **Provider:** City of Parañaque CDRRMO (City Disaster Risk Reduction and Management Office)
- **Coverage:** 2022-2025
- **Format:** CSV with irregular headers (rows 1-4 contain metadata)

### Raw Data Columns

| Column | Data Type | Description | Notes |
|--------|-----------|-------------|-------|
| `#` or `record_num` | Integer | Sequential record number | May restart per year |
| `DATE` | String | Date of flood event | Various formats: MM/DD/YYYY, M/D/YY |
| `MONTH` | String | Month name | "January", "February", etc. |
| `BARANGAY` | String | Barangay (village) name | Administrative unit in Parañaque |
| `LOCATION` | String | Specific address/landmark | Street name, intersection, etc. |
| `LATITUDE` | Float | GPS latitude (WGS84) | Decimal degrees |
| `LONGITUDE` | Float | GPS longitude (WGS84) | Decimal degrees |
| `HEIGHT` or `FLOOD DEPTH` | String | Water depth category | Categorical: gutter/ankle/knee/waist/chest |
| `TIME REPORTED` | String | Time flood was reported | 24-hour format, may be empty |
| `TIME SUBSIDED` | String | Time flood subsided | 24-hour format, may be empty |
| `WEATHER DISTURBANCE` | String | Cause of flooding | Typhoon name, monsoon, etc. |
| `REMARKS` | String | Additional notes | May include damage assessment |

### Cleaned Data Columns

After preprocessing with `scripts/clean_raw_flood_records.py`:

| Column | Data Type | Format | Description |
|--------|-----------|--------|-------------|
| `record_num` | Integer | - | Unique sequential identifier |
| `date` | String | YYYY-MM-DD | Standardized ISO date format |
| `year` | Integer | YYYY | Extracted year |
| `month` | Integer | 1-12 | Extracted month |
| `day` | Integer | 1-31 | Extracted day |
| `barangay` | String | Title Case | Normalized barangay name |
| `location` | String | - | Specific location |
| `latitude` | Float | -90 to 90 | WGS84 latitude |
| `longitude` | Float | -180 to 180 | WGS84 longitude |
| `flood_depth` | String | lowercase | Standardized: gutter/ankle/knee/waist/chest |
| `flood_depth_numeric` | Float | meters | Estimated depth: 0.05/0.15/0.45/0.90/1.20 |
| `weather_disturbance` | String | - | Weather event description |
| `remarks` | String | - | Additional notes |
| `time_reported` | Float | 0-24 | Hour as decimal (e.g., 14.5 = 2:30 PM) |
| `time_subsided` | Float | 0-24 | Hour flood subsided |
| `duration_hours` | Float | hours | Flood duration if both times available |

### Flood Depth Categories

| Category | Description | Estimated Depth | Numeric Value |
|----------|-------------|-----------------|---------------|
| `gutter` | Water in gutters/drains | < 10 cm | 0.05 m |
| `ankle` | Ankle-deep water | 10-30 cm | 0.15 m |
| `knee` | Knee-deep water | 30-60 cm | 0.45 m |
| `waist` | Waist-deep water | 60-120 cm | 0.90 m |
| `chest` | Chest-deep or higher | > 120 cm | 1.20 m |

---

## Derived Features

### Temporal Features

| Feature | Calculation | Description |
|---------|-------------|-------------|
| `day_of_year` | 1-366 | Sequential day number within year |
| `week_of_year` | 1-53 | ISO week number |
| `quarter` | 1-4 | Calendar quarter |
| `is_weekend` | 0/1 | Saturday or Sunday |
| `monsoon_season` | 0/1 | June-November (SW monsoon) |
| `typhoon_season` | 0/1 | July-December (peak typhoon months) |
| `dry_season` | 0/1 | December-May |

### Rainfall Features

| Feature | Calculation | Unit | Description |
|---------|-------------|------|-------------|
| `rainfall_1d` | Previous day rainfall | mm | Lag-1 rainfall |
| `rainfall_3d` | Sum of last 3 days | mm | Short-term accumulation |
| `rainfall_7d` | Sum of last 7 days | mm | Weekly accumulation |
| `rainfall_14d` | Sum of last 14 days | mm | Bi-weekly accumulation |
| `rainfall_30d` | Sum of last 30 days | mm | Monthly accumulation |
| `rainfall_rolling_mean_7d` | 7-day rolling average | mm/day | Smoothed recent rainfall |
| `rainfall_intensity` | Current / rolling_mean | ratio | Relative intensity |
| `heavy_rain_days_7d` | Count of days > 20mm | count | Recent heavy rain events |
| `dry_spell_days` | Consecutive days < 1mm | count | Antecedent dry period |

### Environmental Features

| Feature | Source | Unit | Description |
|---------|--------|------|-------------|
| `temp_range` | TMAX - TMIN | °C | Daily temperature range |
| `heat_index` | Derived | °C | Apparent temperature |
| `humidity_category` | From RH | category | low/moderate/high |
| `wind_category` | From WIND_SPEED | category | calm/light/moderate/strong |
| `tide_height` | WorldTides API | m | Tidal water level |
| `tide_phase` | Derived | category | high/rising/low/falling |
| `soil_moisture_proxy` | Rainfall accumulation | index | Soil saturation estimate |

### Aggregated Features

| Feature | Description |
|---------|-------------|
| `station_avg_rainfall` | Average across all 3 PAGASA stations |
| `station_max_rainfall` | Maximum across stations (conservative) |
| `spatial_rainfall_variance` | Variance across stations (storm localization) |

---

## Target Variable

| Column | Data Type | Values | Description |
|--------|-----------|--------|-------------|
| `flood` | Integer | 0 or 1 | Binary flood occurrence |
| `flood_occurred` | Integer | 0 or 1 | Alternative column name |

### Labeling Rules

- **flood = 1:** At least one flood record exists for this date in Parañaque
- **flood = 0:** No flood records for this date

**Note:** Absence of records may indicate:
1. No flooding occurred (true negative)
2. Flood occurred but was not reported (false negative)
3. Data collection gap

---

## Data Quality Codes

### Quality Flags in Processed Data

| Flag | Meaning | Action |
|------|---------|--------|
| `Q=0` | Good quality | Use as-is |
| `Q=1` | Suspect value | Review before use |
| `Q=2` | Estimated/Imputed | Use with caution |
| `Q=9` | Missing | Excluded or imputed |

### Data Lineage Tracking

See `data/data_lineage.json` for:
- Source file timestamps
- Processing steps applied
- Checksum verification
- Version history

---

## Appendix: Parañaque Barangays

Complete list of barangays (villages) in Parañaque City for reference:

| District 1 | District 2 |
|------------|------------|
| B.F. Homes | Baclaran |
| Don Bosco | Don Galo |
| La Huerta | La Huerta |
| Marcelo Green | Merville |
| Moonwalk | San Antonio |
| San Isidro | San Dionisio |
| San Martin de Porres | San Martin de Porres |
| Santo Niño | Tambo |
| Sun Valley | Vitalez |

---

## References

1. DOST-PAGASA ClimDatPh Platform Documentation
2. Parañaque CDRRMO Flood Monitoring Reports (2022-2025)
3. Philippine Geographic Names Database
4. WorldTides API Documentation

---

*For questions about this data dictionary, contact the Floodingnaque development team.*
