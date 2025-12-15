"""
Preprocess Official Flood Records from Parañaque City (2022-2025)
Converts raw flood records into ML-ready format with extracted features.

This script:
1. Cleans and standardizes CSV formats across different years
2. Extracts flood depth as numerical values (target variable)
3. Extracts weather conditions as features
4. Converts location data into usable features
5. Handles missing values intelligently
6. Creates timestamp features (month, day, hour)
7. Produces clean, ML-ready CSV files for training
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Flood depth mapping (convert descriptive levels to numerical values)
FLOOD_DEPTH_MAP = {
    'gutter': 0.1,    # 10cm - Very Low
    'ankle': 0.15,    # 15cm - Low  
    'half-knee': 0.25, # 25cm - Low-Moderate
    'knee': 0.5,      # 50cm - Moderate
    'waist': 1.0,     # 100cm - High
    'chest': 1.5,     # 150cm - Very High
    'above chest': 2.0, # 200cm+ - Extreme
    # Common misspellings/variations
    'below knee': 0.4,
    'half waist': 0.75,
}

# Binary flood classification thresholds
FLOOD_THRESHOLD = 0.3  # Above 30cm is considered flood (1), below is no-flood (0)

# Weather pattern extraction
WEATHER_PATTERNS = {
    'thunderstorm': ['thunderstorm', 'localized thunderstorm', 'thunderstorms'],
    'monsoon': ['monsoon', 'southwest monsoon', 'habagat'],
    'typhoon': ['typhoon', 'storm', 'tropical storm', 'super typhoon'],
    'itcz': ['itcz', 'intertropical convergence zone', 'inter tropical'],
    'lpa': ['lpa', 'low pressure area', 'trough'],
    'easterlies': ['easterlies', 'easterly'],
    'clear': ['clear', 'fair'],
}


def extract_flood_depth_numeric(depth_str):
    """Convert flood depth description to numeric value (in meters)."""
    if pd.isna(depth_str) or depth_str == '':
        return None
    
    depth_str = str(depth_str).lower().strip()
    
    # Try exact match first
    for key, value in FLOOD_DEPTH_MAP.items():
        if key in depth_str:
            return value
    
    # Try to extract numerical measurements if present (e.g., "19 inches")
    # Some 2025 data has measurements like '19"' (inches)
    match = re.search(r'(\d+)\s*["\']', depth_str)
    if match:
        inches = float(match.group(1))
        return inches * 0.0254  # Convert inches to meters
    
    # Default for unknown
    return None


def extract_flood_binary(depth_value):
    """Convert numeric flood depth to binary classification."""
    if depth_value is None or pd.isna(depth_value):
        return None
    return 1 if depth_value >= FLOOD_THRESHOLD else 0


def extract_weather_type(weather_str):
    """Extract primary weather disturbance type."""
    if pd.isna(weather_str) or weather_str == '':
        return 'unknown'
    
    weather_str = str(weather_str).lower()
    
    # Check each pattern
    for weather_type, patterns in WEATHER_PATTERNS.items():
        for pattern in patterns:
            if pattern in weather_str:
                return weather_type
    
    return 'other'


def extract_temperature_from_str(temp_str):
    """Extract temperature value from string (handle different formats)."""
    if pd.isna(temp_str) or temp_str == '':
        return None
    
    temp_str = str(temp_str)
    
    # Try to extract numbers
    match = re.search(r'(\d+\.?\d*)', temp_str)
    if match:
        temp = float(match.group(1))
        # Assume Celsius, convert to reasonable range
        if temp > 50:  # Likely Fahrenheit
            temp = (temp - 32) * 5/9
        return temp
    
    return None


def parse_datetime_flexible(date_str, time_str=None):
    """Parse date and time with flexible format handling."""
    try:
        if pd.isna(date_str):
            return None
        
        date_str = str(date_str).strip()
        
        # Common date formats
        date_formats = [
            '%B %d, %Y',      # May 6, 2025
            '%d %B %Y',       # 6 May 2025
            '%d-%b-%Y',       # 6-May-2025
            '%Y-%m-%d',       # 2025-05-06
            '%d/%m/%Y',       # 06/05/2025
            '%m/%d/%Y',       # 05/06/2025
        ]
        
        for fmt in date_formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                
                # Add time if provided
                if time_str and not pd.isna(time_str):
                    time_str = str(time_str).strip().upper()
                    time_match = re.search(r'(\d{1,2})[:.]?(\d{2})?([AP]M)?', time_str)
                    if time_match:
                        hour = int(time_match.group(1))
                        minute = int(time_match.group(2)) if time_match.group(2) else 0
                        
                        # Handle 12/24 hour format
                        if time_match.group(3) == 'PM' and hour != 12:
                            hour += 12
                        elif time_match.group(3) == 'AM' and hour == 12:
                            hour = 0
                        
                        dt = dt.replace(hour=hour, minute=minute)
                
                return dt
            except:
                continue
        
        return None
    except:
        return None


def preprocess_flood_records(input_csv, output_csv, year):
    """
    Preprocess official flood records into ML-ready format.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output processed CSV
        year: Year of the data (for validation)
    """
    logger.info(f"="*80)
    logger.info(f"Processing {year} Flood Records")
    logger.info(f"="*80)
    logger.info(f"Input: {input_csv}")
    logger.info(f"Output: {output_csv}")
    
    try:
        # Read CSV (handle different encodings)
        try:
            df = pd.read_csv(input_csv, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(input_csv, encoding='latin-1')
            except:
                df = pd.read_csv(input_csv, encoding='cp1252')
        
        logger.info(f"Loaded {len(df)} rows")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Initialize output dataframe
        processed_data = []
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Skip header rows or invalid rows
                if idx < 5:  # Skip first few rows (headers)
                    continue
                
                # Extract basic fields (adjust column names based on year)
                record = {}
                
                # Try to extract date
                date_col = None
                for col in df.columns:
                    if any(keyword in str(col).lower() for keyword in ['date', 'tanggal']):
                        date_col = col
                        break
                
                if date_col:
                    record['date_str'] = str(row[date_col]) if not pd.isna(row[date_col]) else None
                
                # Extract flood depth
                depth_str = None
                for col in df.columns:
                    val = str(row[col]).lower() if not pd.isna(row[col]) else ''
                    if any(keyword in val for keyword in ['gutter', 'knee', 'waist', 'chest', 'ankle']):
                        depth_str = val
                        break
                    # Check for depth measurements
                    if re.search(r'\d+\s*["\']', val):
                        depth_str = val
                        break
                
                if depth_str:
                    flood_depth = extract_flood_depth_numeric(depth_str)
                    if flood_depth is not None:
                        record['flood_depth_m'] = flood_depth
                        record['flood'] = extract_flood_binary(flood_depth)
                        record['flood_depth_category'] = depth_str
                
                # Extract weather
                weather_str = None
                for col in df.columns:
                    val = str(row[col]).lower() if not pd.isna(row[col]) else ''
                    if any(keyword in val for keyword in ['monsoon', 'typhoon', 'thunderstorm', 'easterlies']):
                        weather_str = val
                        break
                
                if weather_str:
                    record['weather_type'] = extract_weather_type(weather_str)
                    record['weather_description'] = weather_str[:100]  # Limit length
                
                # Extract location
                location_str = None
                for col in df.columns:
                    if 'location' in str(col).lower() or 'barangay' in str(col).lower():
                        location_str = str(row[col]) if not pd.isna(row[col]) else None
                        break
                
                if location_str:
                    record['location'] = location_str[:100]
                
                # Extract coordinates
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'latit' in col_lower:
                        try:
                            record['latitude'] = float(row[col])
                        except:
                            pass
                    elif 'longit' in col_lower:
                        try:
                            record['longitude'] = float(row[col])
                        except:
                            pass
                
                # Extract temperature if available
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'temp' in col_lower:
                        temp = extract_temperature_from_str(row[col])
                        if temp:
                            record['temperature'] = temp
                
                # Add year
                record['year'] = year
                
                # Only add if we have flood depth (minimum requirement)
                if 'flood_depth_m' in record and 'flood' in record:
                    processed_data.append(record)
                
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {str(e)}")
                continue
        
        # Create DataFrame
        if not processed_data:
            logger.error("No valid data extracted!")
            return None
        
        result_df = pd.DataFrame(processed_data)
        
        # Fill missing weather with default
        if 'weather_type' not in result_df.columns:
            result_df['weather_type'] = 'unknown'
        
        # Add default temperature/humidity/precipitation for missing values
        # These will be estimated based on historical averages for Parañaque
        if 'temperature' not in result_df.columns:
            # Average temperature in Metro Manila: 27-28°C
            result_df['temperature'] = 27.5
        
        result_df['temperature'] = result_df['temperature'].fillna(27.5)
        
        # Estimate precipitation based on flood depth
        # This is a rough estimation: deeper floods likely had more rain
        if 'precipitation' not in result_df.columns:
            result_df['precipitation'] = result_df['flood_depth_m'] * 50  # Rough correlation
        
        # Estimate humidity based on weather type
        if 'humidity' not in result_df.columns:
            humidity_map = {
                'monsoon': 85,
                'typhoon': 90,
                'thunderstorm': 80,
                'clear': 65,
                'unknown': 75,
                'other': 75,
                'itcz': 80,
                'lpa': 82,
                'easterlies': 70
            }
            result_df['humidity'] = result_df['weather_type'].map(humidity_map).fillna(75)
        
        # Reorder columns for ML training
        ml_columns = ['temperature', 'humidity', 'precipitation', 'flood']
        optional_columns = ['flood_depth_m', 'weather_type', 'year', 'latitude', 'longitude', 
                          'location', 'flood_depth_category', 'weather_description']
        
        # Keep only columns that exist
        final_columns = ml_columns + [col for col in optional_columns if col in result_df.columns]
        result_df = result_df[final_columns]
        
        # Summary statistics
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total records extracted: {len(result_df)}")
        logger.info(f"\nFlood Classification Distribution:")
        logger.info(result_df['flood'].value_counts())
        logger.info(f"\nFlood Depth Statistics:")
        logger.info(result_df['flood_depth_m'].describe())
        logger.info(f"\nWeather Type Distribution:")
        logger.info(result_df['weather_type'].value_counts())
        logger.info(f"\nTemperature: {result_df['temperature'].mean():.1f}°C (avg)")
        logger.info(f"Humidity: {result_df['humidity'].mean():.1f}% (avg)")
        logger.info(f"Precipitation: {result_df['precipitation'].mean():.1f}mm (avg)")
        
        # Save to CSV
        result_df.to_csv(output_csv, index=False)
        logger.info(f"\n✓ Processed data saved to: {output_csv}")
        logger.info(f"{'='*80}\n")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing {input_csv}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def process_all_years(data_dir='data', output_dir='data/processed'):
    """Process all years of flood records."""
    logger.info("="*80)
    logger.info("PROCESSING ALL OFFICIAL FLOOD RECORDS")
    logger.info("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Years to process
    years = [2022, 2023, 2024, 2025]
    
    processed_files = {}
    
    for year in years:
        input_file = Path(data_dir) / f'Floodingnaque_Paranaque_Official_Flood_Records_{year}.csv'
        output_file = output_path / f'processed_flood_records_{year}.csv'
        
        if not input_file.exists():
            logger.warning(f"File not found: {input_file}")
            continue
        
        result = preprocess_flood_records(str(input_file), str(output_file), year)
        
        if result is not None:
            processed_files[year] = {
                'input': str(input_file),
                'output': str(output_file),
                'records': len(result)
            }
    
    # Summary report
    logger.info("\n" + "="*80)
    logger.info("PROCESSING COMPLETE - SUMMARY")
    logger.info("="*80)
    logger.info(f"Files processed: {len(processed_files)}/{len(years)}")
    for year, info in processed_files.items():
        logger.info(f"\n{year}:")
        logger.info(f"  Records: {info['records']}")
        logger.info(f"  Output: {info['output']}")
    
    logger.info("\n" + "="*80)
    logger.info("READY FOR MODEL TRAINING!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("1. Review processed files in: " + str(output_path))
    logger.info("2. Use progressive_train.py to train models incrementally")
    logger.info("3. Compare model performance across years")
    
    return processed_files


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess official flood records for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process all years:
    python preprocess_official_flood_records.py
  
  Process specific year:
    python preprocess_official_flood_records.py --year 2024
  
  Custom directories:
    python preprocess_official_flood_records.py --data-dir ../data --output-dir ../processed
        """
    )
    parser.add_argument('--year', type=int, choices=[2022, 2023, 2024, 2025],
                       help='Process specific year only')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing input CSV files')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Directory for processed output files')
    
    args = parser.parse_args()
    
    if args.year:
        # Process single year
        input_file = Path(args.data_dir) / f'Floodingnaque_Paranaque_Official_Flood_Records_{args.year}.csv'
        output_file = Path(args.output_dir) / f'processed_flood_records_{args.year}.csv'
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        preprocess_flood_records(str(input_file), str(output_file), args.year)
    else:
        # Process all years
        process_all_years(args.data_dir, args.output_dir)
