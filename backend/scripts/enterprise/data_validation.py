"""
Data Validation Module
======================

This module provides Pandera-based data validation for the training pipeline.
Ensures data quality and consistency before model training.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pandera as pa
    from pandera import Check, Column, DataFrameSchema
    from pandera.errors import SchemaError, SchemaErrors

    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False
    pa = None  # type: ignore[assignment]
    Check = None  # type: ignore[assignment,misc]
    Column = None  # type: ignore[assignment,misc]
    DataFrameSchema = None  # type: ignore[assignment,misc]
    SchemaError = Exception  # type: ignore[assignment,misc]
    SchemaErrors = Exception  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""

    def __init__(self, message: str, errors: Optional[List[Dict]] = None):
        super().__init__(message)
        self.errors = errors or []


class FloodDataValidator:
    """
    Data validator for flood prediction training data.

    Validates:
    - Schema correctness (columns, types)
    - Value ranges (temperature, humidity, etc.)
    - Data quality (missing values, duplicates)
    - Target distribution
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize validator with configuration.

        Args:
            config: Validation configuration from YAML
        """
        self.config = config or {}
        self.feature_ranges = self.config.get("feature_ranges", {})
        self._schema = None

        if PANDERA_AVAILABLE:
            self._build_schema()
        else:
            logger.warning("Pandera not installed. Using basic validation only.")

    def _build_schema(self) -> None:
        """Build Pandera schema from configuration."""
        if not PANDERA_AVAILABLE or pa is None:
            return

        from pandera import Check, Column, DataFrameSchema

        columns = {}

        # Temperature (Kelvin)
        temp_range = self.feature_ranges.get("temperature", {"min": 250, "max": 330})
        columns["temperature"] = Column(
            pa.Float,
            Check.in_range(temp_range["min"], temp_range["max"]),
            nullable=True,
            required=True,
            description="Temperature in Kelvin",
        )

        # Humidity (%)
        humidity_range = self.feature_ranges.get("humidity", {"min": 0, "max": 100})
        columns["humidity"] = Column(
            pa.Float,
            Check.in_range(humidity_range["min"], humidity_range["max"]),
            nullable=True,
            required=True,
            description="Relative humidity percentage",
        )

        # Precipitation (mm)
        precip_range = self.feature_ranges.get("precipitation", {"min": 0, "max": 500})
        columns["precipitation"] = Column(
            pa.Float, Check.ge(precip_range["min"]), nullable=True, required=True, description="Precipitation in mm"
        )

        # Target variable
        columns["flood"] = Column(
            pa.Int, Check.isin([0, 1]), nullable=False, required=True, description="Flood occurrence (0=No, 1=Yes)"
        )

        # Optional columns with flexible validation
        optional_columns = {
            "is_monsoon_season": Column(pa.Int, Check.isin([0, 1]), nullable=True, required=False),
            "month": Column(pa.Int, Check.in_range(1, 12), nullable=True, required=False),
            "year": Column(pa.Int, Check.in_range(2000, 2100), nullable=True, required=False),
            "day": Column(pa.Int, Check.in_range(1, 31), nullable=True, required=False),
        }
        columns.update(optional_columns)

        self._schema = DataFrameSchema(
            columns=columns,
            strict=False,  # Allow extra columns
            coerce=True,  # Coerce types when possible
        )

    def validate(
        self, df: pd.DataFrame, raise_on_error: bool = True, fix_errors: bool = False
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Validate DataFrame against schema.

        Args:
            df: DataFrame to validate
            raise_on_error: If True, raise exception on validation failure
            fix_errors: If True, attempt to fix recoverable errors

        Returns:
            Tuple of (validated DataFrame, list of errors)
        """
        errors = []
        validated_df = df.copy()

        # Check required columns
        required_cols = ["temperature", "humidity", "precipitation", "flood"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error = {
                "type": "missing_columns",
                "columns": missing_cols,
                "message": f"Missing required columns: {missing_cols}",
            }
            errors.append(error)
            if raise_on_error:
                raise DataValidationError(error["message"], errors)

        # Run Pandera validation if available
        if PANDERA_AVAILABLE and self._schema:
            try:
                validated_df = self._schema.validate(validated_df, lazy=True)
            except Exception as e:
                # Handle SchemaErrors from pandera
                if hasattr(e, "failure_cases"):
                    for failure in e.failure_cases.to_dict("records"):  # type: ignore[attr-defined]
                        error = {
                            "type": "schema_violation",
                            "column": failure.get("column"),
                            "check": failure.get("check"),
                            "failure_case": failure.get("failure_case"),
                            "index": failure.get("index"),
                        }
                        errors.append(error)

                if raise_on_error and not fix_errors:
                    raise DataValidationError(f"Schema validation failed with {len(errors)} errors", errors)

        # Additional quality checks
        quality_errors = self._check_data_quality(validated_df)
        errors.extend(quality_errors)

        # Fix errors if requested
        if fix_errors and errors:
            validated_df, fix_report = self._fix_errors(validated_df, errors)
            logger.info(f"Fixed {len(fix_report)} data issues")

        if errors and raise_on_error and not fix_errors:
            raise DataValidationError(f"Data validation failed with {len(errors)} errors", errors)

        return validated_df, errors

    def _check_data_quality(self, df: pd.DataFrame) -> List[Dict]:
        """Check data quality metrics."""
        errors = []

        # Check for excessive missing values
        max_missing_ratio = self.config.get("max_missing_ratio", 0.1)
        for col in df.columns:
            missing_ratio = df[col].isna().mean()
            if missing_ratio > max_missing_ratio:
                errors.append(
                    {
                        "type": "excessive_missing",
                        "column": col,
                        "missing_ratio": float(missing_ratio),
                        "threshold": max_missing_ratio,
                        "message": f"Column '{col}' has {missing_ratio:.1%} missing values (threshold: {max_missing_ratio:.1%})",
                    }
                )

        # Check minimum sample size
        min_samples = self.config.get("min_samples", 100)
        if len(df) < min_samples:
            errors.append(
                {
                    "type": "insufficient_samples",
                    "count": len(df),
                    "threshold": min_samples,
                    "message": f"Dataset has only {len(df)} samples (minimum: {min_samples})",
                }
            )

        # Check target class balance
        if "flood" in df.columns:
            class_counts = df["flood"].value_counts()
            if len(class_counts) < 2:
                errors.append({"type": "single_class", "message": "Target variable has only one class"})
            else:
                minority_ratio = class_counts.min() / class_counts.sum()
                if minority_ratio < 0.05:
                    errors.append(
                        {
                            "type": "severe_imbalance",
                            "minority_ratio": float(minority_ratio),
                            "message": f"Severe class imbalance: minority class is {minority_ratio:.1%}",
                        }
                    )

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            duplicate_ratio = duplicate_count / len(df)
            if duplicate_ratio > 0.01:
                errors.append(
                    {
                        "type": "duplicates",
                        "count": int(duplicate_count),
                        "ratio": float(duplicate_ratio),
                        "message": f"Found {duplicate_count} duplicate rows ({duplicate_ratio:.1%})",
                    }
                )

        # Check for outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == "flood":
                continue

            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > len(df) * 0.01:
                errors.append(
                    {
                        "type": "outliers",
                        "column": col,
                        "count": len(outliers),
                        "ratio": float(len(outliers) / len(df)),
                        "message": f"Column '{col}' has {len(outliers)} outliers",
                    }
                )

        return errors

    def _fix_errors(self, df: pd.DataFrame, errors: List[Dict]) -> Tuple[pd.DataFrame, List[str]]:
        """Attempt to fix recoverable errors."""
        fixed_df = df.copy()
        fix_report = []

        for error in errors:
            error_type = error.get("type")

            if error_type == "excessive_missing":
                col = error["column"]
                if col in fixed_df.columns:
                    # Impute with median for numeric, mode for categorical
                    if fixed_df[col].dtype in [np.float64, np.int64]:
                        median_val = fixed_df[col].median()
                        fixed_df[col] = fixed_df[col].fillna(median_val)
                        fix_report.append(f"Imputed {col} with median: {median_val}")
                    else:
                        mode_val = fixed_df[col].mode().iloc[0] if not fixed_df[col].mode().empty else 0
                        fixed_df[col] = fixed_df[col].fillna(mode_val)
                        fix_report.append(f"Imputed {col} with mode: {mode_val}")

            elif error_type == "duplicates":
                before = len(fixed_df)
                fixed_df = fixed_df.drop_duplicates()
                after = len(fixed_df)
                fix_report.append(f"Removed {before - after} duplicate rows")

            elif error_type == "outliers":
                col = error["column"]
                if col in fixed_df.columns:
                    q1, q3 = fixed_df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 3 * iqr
                    upper_bound = q3 + 3 * iqr

                    # Clip outliers
                    fixed_df[col] = fixed_df[col].clip(lower_bound, upper_bound)
                    fix_report.append(f"Clipped outliers in {col} to [{lower_bound:.2f}, {upper_bound:.2f}]")

        return fixed_df, fix_report

    def get_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with validation results
        """
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
            },
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isna().sum().to_dict(),
            "missing_ratios": (df.isna().mean() * 100).round(2).to_dict(),
        }

        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report["numeric_stats"] = df[numeric_cols].describe().to_dict()

        # Target distribution
        if "flood" in df.columns:
            report["target_distribution"] = df["flood"].value_counts().to_dict()
            report["target_balance"] = {
                "flood_ratio": float(df["flood"].mean()),
                "no_flood_ratio": float(1 - df["flood"].mean()),
            }

        # Validation status
        try:
            _, errors = self.validate(df, raise_on_error=False)
            report["validation_passed"] = len(errors) == 0
            report["validation_errors"] = errors
        except Exception as e:
            report["validation_passed"] = False
            report["validation_errors"] = [{"type": "exception", "message": str(e)}]

        return report


def validate_training_data(df: pd.DataFrame, config: Optional[Dict] = None, fix_errors: bool = True) -> pd.DataFrame:
    """
    Convenience function to validate and optionally fix training data.

    Args:
        df: DataFrame to validate
        config: Validation configuration
        fix_errors: Whether to fix recoverable errors

    Returns:
        Validated (and optionally fixed) DataFrame
    """
    validator = FloodDataValidator(config)
    validated_df, errors = validator.validate(df, raise_on_error=False, fix_errors=fix_errors)

    if errors:
        logger.warning(f"Data validation found {len(errors)} issues")
        for error in errors[:5]:  # Log first 5 errors
            logger.warning(f"  - {error.get('message', error.get('type'))}")
        if len(errors) > 5:
            logger.warning(f"  ... and {len(errors) - 5} more")

    return validated_df


def create_feature_schema(features: List[str]) -> Any:
    """
    Create a dynamic schema for specific features.

    Args:
        features: List of feature column names

    Returns:
        Pandera DataFrameSchema or None if Pandera unavailable
    """
    if not PANDERA_AVAILABLE or pa is None:
        return None

    from pandera import Check, Column, DataFrameSchema

    columns = {}
    for feature in features:
        # Default to nullable float
        columns[feature] = Column(pa.Float, nullable=True, required=True)

    return DataFrameSchema(columns=columns, strict=False, coerce=True)
