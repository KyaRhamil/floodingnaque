"""
Utility script to merge multiple CSV datasets for training.
Useful when you have collected data from different time periods or sources.
"""

import pandas as pd
import os
import glob
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_csv_files(input_pattern='data/*.csv', output_file='data/merged_dataset.csv', 
                    remove_duplicates=True, validate_columns=True):
    """
    Merge multiple CSV files into a single dataset.
    
    Args:
        input_pattern: Glob pattern for input CSV files (e.g., 'data/*.csv', 'data/flood_*.csv')
        output_file: Path for the merged output CSV file
        remove_duplicates: If True, remove duplicate rows
        validate_columns: If True, ensure all files have the same columns
    
    Returns:
        DataFrame: The merged dataset
    """
    logger.info("="*80)
    logger.info("CSV DATASET MERGER")
    logger.info("="*80)
    
    # Find all matching CSV files
    csv_files = glob.glob(input_pattern)
    
    if not csv_files:
        logger.error(f"No CSV files found matching pattern: {input_pattern}")
        return None
    
    logger.info(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        logger.info(f"  - {f}")
    
    # Load and validate files
    dataframes = []
    total_rows = 0
    first_columns = None
    
    for i, file_path in enumerate(csv_files, 1):
        logger.info(f"\n[{i}/{len(csv_files)}] Processing: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"  Rows: {len(df)}, Columns: {list(df.columns)}")
            
            # Validate columns
            if validate_columns:
                if first_columns is None:
                    first_columns = set(df.columns)
                else:
                    current_columns = set(df.columns)
                    if current_columns != first_columns:
                        logger.warning(f"  ⚠ Column mismatch detected!")
                        logger.warning(f"    Expected: {sorted(first_columns)}")
                        logger.warning(f"    Got:      {sorted(current_columns)}")
                        logger.warning(f"    Missing:  {sorted(first_columns - current_columns)}")
                        logger.warning(f"    Extra:    {sorted(current_columns - first_columns)}")
                        
                        if input("  Continue with this file? (y/n): ").lower() != 'y':
                            logger.info(f"  Skipping {file_path}")
                            continue
            
            dataframes.append(df)
            total_rows += len(df)
            
        except Exception as e:
            logger.error(f"  ✗ Error loading {file_path}: {str(e)}")
            continue
    
    if not dataframes:
        logger.error("No valid dataframes to merge!")
        return None
    
    # Merge dataframes
    logger.info(f"\nMerging {len(dataframes)} dataframes...")
    merged_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"  Total rows before merge: {total_rows}")
    logger.info(f"  Rows after merge: {len(merged_df)}")
    
    # Remove duplicates if requested
    if remove_duplicates:
        logger.info("\nRemoving duplicate rows...")
        before_count = len(merged_df)
        merged_df = merged_df.drop_duplicates()
        after_count = len(merged_df)
        duplicates_removed = before_count - after_count
        logger.info(f"  Duplicates removed: {duplicates_removed}")
        logger.info(f"  Final row count: {after_count}")
    
    # Display dataset statistics
    logger.info("\nMERGED DATASET STATISTICS")
    logger.info("-"*80)
    logger.info(f"Total rows: {len(merged_df)}")
    logger.info(f"Total columns: {len(merged_df.columns)}")
    logger.info(f"Columns: {list(merged_df.columns)}")
    
    logger.info("\nData types:")
    for col, dtype in merged_df.dtypes.items():
        logger.info(f"  {col}: {dtype}")
    
    logger.info("\nMissing values:")
    missing = merged_df.isnull().sum()
    if missing.sum() == 0:
        logger.info("  No missing values ✓")
    else:
        for col, count in missing[missing > 0].items():
            logger.info(f"  {col}: {count} ({count/len(merged_df)*100:.2f}%)")
    
    # Show target distribution if 'flood' column exists
    if 'flood' in merged_df.columns:
        logger.info("\nTarget distribution (flood):")
        distribution = merged_df['flood'].value_counts().sort_index()
        for value, count in distribution.items():
            percentage = count / len(merged_df) * 100
            logger.info(f"  Class {value}: {count} ({percentage:.2f}%)")
    
    # Save merged dataset
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    merged_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Merged dataset saved to: {output_file}")
    
    # Create metadata file
    metadata = {
        'created_at': datetime.now().isoformat(),
        'source_files': csv_files,
        'total_source_files': len(csv_files),
        'total_rows': len(merged_df),
        'total_columns': len(merged_df.columns),
        'columns': list(merged_df.columns),
        'duplicates_removed': duplicates_removed if remove_duplicates else 0,
        'missing_values': {col: int(count) for col, count in merged_df.isnull().sum().items()}
    }
    
    if 'flood' in merged_df.columns:
        metadata['target_distribution'] = {
            str(k): int(v) for k, v in merged_df['flood'].value_counts().to_dict().items()
        }
    
    metadata_file = output_path.with_suffix('.metadata.json')
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata saved to: {metadata_file}")
    
    logger.info("\n" + "="*80)
    logger.info("MERGE COMPLETE!")
    logger.info("="*80)
    
    return merged_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merge multiple CSV datasets into one',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Merge all CSV files in data folder:
    python merge_datasets.py
  
  Merge specific pattern:
    python merge_datasets.py --input "data/flood_*.csv"
  
  Custom output file:
    python merge_datasets.py --output data/combined_data.csv
  
  Keep duplicates:
    python merge_datasets.py --keep-duplicates
        """
    )
    parser.add_argument('--input', type=str, default='data/*.csv',
                       help='Input file pattern (use quotes for patterns)')
    parser.add_argument('--output', type=str, default='data/merged_dataset.csv',
                       help='Output file path for merged dataset')
    parser.add_argument('--keep-duplicates', action='store_true',
                       help='Keep duplicate rows (default: remove duplicates)')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip column validation')
    
    args = parser.parse_args()
    
    merge_csv_files(
        input_pattern=args.input,
        output_file=args.output,
        remove_duplicates=not args.keep_duplicates,
        validate_columns=not args.no_validation
    )
