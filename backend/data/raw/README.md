# Raw Data Directory

This directory contains the original, unprocessed source data files for the
Floodingnaque flood prediction system.

## Directory Structure

`
raw/
├── pagasa/               # PAGASA weather station data
│   ├── *_NAIA Daily Data.csv
│   ├── *_Port Area Daily Data.csv
│   └── *_Science Garden Daily Data.csv
└── flood_records/        # Official flood records from Paranaque CDRRMO
    ├── *_2022.csv
    ├── *_2023.csv
    ├── *_2024.csv
    └── *_2025.csv
`

## Data Sources

### PAGASA Weather Data
- **Provider:** DOST-PAGASA
- **Portal:** http://bagong.pagasa.dost.gov.ph/climate/climate-data
- **Stations:** NAIA, Port Area, Science Garden

### Flood Records
- **Provider:** City of Paranaque CDRRMO
- **Format:** CSV with irregular headers (requires preprocessing)

## Processing Pipeline

1. **Raw data** (this directory) - Original source files
2. **Cleaned data** (../cleaned/) - Standardized format, fixed headers
3. **Processed data** (../processed/) - Training-ready datasets

---
*Last updated: 2026-01-24*
