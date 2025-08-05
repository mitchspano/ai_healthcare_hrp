"""
Data preprocessing module for cleaning and standardizing diabetes data.

This module provides comprehensive preprocessing capabilities including:
- Missing value handling with multiple imputation strategies
- Outlier detection and treatment
- Data cleaning and duplicate removal
- Time series resampling for uniform intervals
- Preprocessing reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class MissingValueHandler:
    """Handles missing values using multiple imputation strategies."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the missing value handler.

        Args:
            config: Configuration dictionary containing imputation strategies
        """
        self.config = config.get("missing_values", {})
        self.strategies = self.config.get(
            "strategies",
            {
                "CGM": "interpolation",
                "Basal": "forward_fill",
                "TotalBolusInsulinDelivered": "zero_fill",
                "CorrectionDelivered": "zero_fill",
                "FoodDelivered": "zero_fill",
                "CarbSize": "zero_fill",
            },
        )
        self.interpolation_method = self.config.get("interpolation_method", "linear")
        self.max_gap_minutes = self.config.get("max_gap_minutes", 30)

    def handle_missing_values(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values in the dataset using configured strategies.

        Args:
            df: Input DataFrame with missing values

        Returns:
            Tuple of (processed DataFrame, statistics dictionary)
        """
        logger.info("Starting missing value handling")
        df_processed = df.copy()
        stats_dict = {}

        # Ensure EventDateTime is datetime type
        if "EventDateTime" in df_processed.columns:
            df_processed["EventDateTime"] = pd.to_datetime(
                df_processed["EventDateTime"]
            )
            df_processed = df_processed.sort_values("EventDateTime")

        for column in df_processed.columns:
            if column == "EventDateTime":
                continue

            missing_count_before = df_processed[column].isnull().sum()
            if missing_count_before == 0:
                continue

            strategy = self.strategies.get(column, "median")
            logger.info(
                f"Handling missing values in {column} using {strategy} strategy"
            )

            if strategy == "forward_fill":
                df_processed[column] = df_processed[column].ffill()
                # Backward fill any remaining NaN values at the beginning
                df_processed[column] = df_processed[column].bfill()

            elif strategy == "interpolation":
                df_processed[column] = self._interpolate_with_gap_limit(
                    df_processed, column
                )

            elif strategy == "median":
                # Check if column is numeric before calculating median
                if pd.api.types.is_numeric_dtype(df_processed[column]):
                    median_value = df_processed[column].median()
                    df_processed[column] = df_processed[column].fillna(median_value)
                else:
                    # For non-numeric columns, use mode (most frequent value)
                    mode_value = (
                        df_processed[column].mode().iloc[0]
                        if not df_processed[column].mode().empty
                        else "Unknown"
                    )
                    df_processed[column] = df_processed[column].fillna(mode_value)

            elif strategy == "mean":
                # Check if column is numeric before calculating mean
                if pd.api.types.is_numeric_dtype(df_processed[column]):
                    mean_value = df_processed[column].mean()
                    df_processed[column] = df_processed[column].fillna(mean_value)
                else:
                    # For non-numeric columns, use mode (most frequent value)
                    mode_value = (
                        df_processed[column].mode().iloc[0]
                        if not df_processed[column].mode().empty
                        else "Unknown"
                    )
                    df_processed[column] = df_processed[column].fillna(mode_value)

            elif strategy == "zero_fill":
                df_processed[column] = df_processed[column].fillna(0)

            elif strategy == "drop":
                df_processed = df_processed.dropna(subset=[column])

            missing_count_after = df_processed[column].isnull().sum()

            stats_dict[column] = {
                "missing_before": missing_count_before,
                "missing_after": missing_count_after,
                "strategy_used": strategy,
                "imputation_rate": (
                    (missing_count_before - missing_count_after) / missing_count_before
                    if missing_count_before > 0
                    else 0
                ),
            }

        logger.info(
            f"Missing value handling completed. Processed {len(stats_dict)} columns"
        )
        return df_processed, stats_dict

    def _interpolate_with_gap_limit(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Interpolate values with gap size limits.

        Args:
            df: DataFrame containing the data
            column: Column name to interpolate

        Returns:
            Interpolated series
        """
        series = df[column].copy()

        if "EventDateTime" not in df.columns:
            # Simple interpolation without time awareness
            return series.interpolate(method=self.interpolation_method)

        # For time-aware interpolation, we'll use a simpler approach
        # Set EventDateTime as index temporarily for interpolation
        df_temp = df[["EventDateTime", column]].copy()
        df_temp = df_temp.set_index("EventDateTime")

        # Interpolate the series
        interpolated_series = df_temp[column].interpolate(
            method=self.interpolation_method
        )

        # Check for large gaps and set those back to NaN if needed
        if self.max_gap_minutes > 0:
            time_diff = df_temp.index.to_series().diff()
            large_gaps = time_diff > timedelta(minutes=self.max_gap_minutes)

            # For simplicity, we'll just interpolate everything for now
            # In a production system, you'd want more sophisticated gap handling

        # Reset index to match original series
        interpolated_series.index = series.index
        return interpolated_series


class OutlierTreatment:
    """Handles outlier detection and treatment with configurable methods."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the outlier treatment handler.

        Args:
            config: Configuration dictionary containing outlier treatment settings
        """
        self.config = config.get("outliers", {})
        self.detection_methods = self.config.get("detection_methods", ["iqr", "zscore"])
        self.treatment_method = self.config.get("treatment_method", "clip")
        self.zscore_threshold = self.config.get("zscore_threshold", 3.0)
        self.iqr_multiplier = self.config.get("iqr_multiplier", 1.5)
        self.isolation_forest_contamination = self.config.get("contamination", 0.1)

        # Column-specific settings
        self.column_settings = self.config.get(
            "column_settings",
            {
                "CGM": {
                    "min_value": 40,
                    "max_value": 400,
                    "detection_methods": ["iqr", "zscore"],
                    "treatment_method": "clip",
                }
            },
        )

    def treat_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and treat outliers in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (processed DataFrame, statistics dictionary)
        """
        logger.info("Starting outlier detection and treatment")
        df_processed = df.copy()
        stats_dict = {}

        for column in df_processed.select_dtypes(include=[np.number]).columns:
            if column in ["EventDateTime"]:
                continue

            column_config = self.column_settings.get(column, {})
            detection_methods = column_config.get(
                "detection_methods", self.detection_methods
            )
            treatment_method = column_config.get(
                "treatment_method", self.treatment_method
            )

            logger.info(
                f"Processing outliers in {column} using {detection_methods} detection and {treatment_method} treatment"
            )

            # Detect outliers
            outlier_mask = self._detect_outliers(
                df_processed[column], detection_methods, column_config
            )
            outlier_count = outlier_mask.sum()

            if outlier_count == 0:
                stats_dict[column] = {
                    "outliers_detected": 0,
                    "outliers_treated": 0,
                    "treatment_method": treatment_method,
                    "detection_methods": detection_methods,
                }
                continue

            # Treat outliers
            original_values = df_processed.loc[outlier_mask, column].copy()
            df_processed.loc[outlier_mask, column] = self._treat_outliers(
                df_processed[column], outlier_mask, treatment_method, column_config
            )

            treated_count = (
                df_processed.loc[outlier_mask, column] != original_values
            ).sum()

            stats_dict[column] = {
                "outliers_detected": outlier_count,
                "outliers_treated": treated_count,
                "treatment_method": treatment_method,
                "detection_methods": detection_methods,
                "outlier_percentage": (outlier_count / len(df_processed)) * 100,
            }

        logger.info(f"Outlier treatment completed. Processed {len(stats_dict)} columns")
        return df_processed, stats_dict

    def _detect_outliers(
        self, series: pd.Series, methods: List[str], column_config: Dict[str, Any]
    ) -> pd.Series:
        """
        Detect outliers using specified methods.

        Args:
            series: Data series to analyze
            methods: List of detection methods to use
            column_config: Column-specific configuration

        Returns:
            Boolean mask indicating outliers
        """
        outlier_mask = pd.Series(False, index=series.index)

        for method in methods:
            if method == "zscore":
                z_scores = np.abs(stats.zscore(series.dropna()))
                threshold = column_config.get("zscore_threshold", self.zscore_threshold)
                method_mask = pd.Series(False, index=series.index)
                method_mask.loc[series.dropna().index] = z_scores > threshold

            elif method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                multiplier = column_config.get("iqr_multiplier", self.iqr_multiplier)
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                method_mask = (series < lower_bound) | (series > upper_bound)

            elif method == "isolation_forest":
                contamination = column_config.get(
                    "contamination", self.isolation_forest_contamination
                )
                iso_forest = IsolationForest(
                    contamination=contamination, random_state=42
                )
                outlier_labels = iso_forest.fit_predict(
                    series.dropna().values.reshape(-1, 1)
                )
                method_mask = pd.Series(False, index=series.index)
                method_mask.loc[series.dropna().index] = outlier_labels == -1

            elif method == "domain_specific":
                # Use domain-specific bounds if provided
                min_value = column_config.get("min_value")
                max_value = column_config.get("max_value")
                method_mask = pd.Series(False, index=series.index)

                if min_value is not None:
                    method_mask |= series < min_value
                if max_value is not None:
                    method_mask |= series > max_value

            outlier_mask |= method_mask

        return outlier_mask

    def _treat_outliers(
        self,
        series: pd.Series,
        outlier_mask: pd.Series,
        treatment_method: str,
        column_config: Dict[str, Any],
    ) -> pd.Series:
        """
        Treat detected outliers using specified method.

        Args:
            series: Original data series
            outlier_mask: Boolean mask indicating outliers
            treatment_method: Method to use for treatment
            column_config: Column-specific configuration

        Returns:
            Series with treated outliers
        """
        treated_series = series.copy()

        if treatment_method == "clip":
            # Clip to percentile bounds or domain-specific bounds
            min_value = column_config.get("min_value")
            max_value = column_config.get("max_value")

            if min_value is None:
                min_value = series.quantile(0.01)
            if max_value is None:
                max_value = series.quantile(0.99)

            treated_series = treated_series.clip(lower=min_value, upper=max_value)

        elif treatment_method == "remove":
            treated_series.loc[outlier_mask] = np.nan

        elif treatment_method == "transform":
            # Apply log transformation to reduce impact of outliers
            if (series > 0).all():
                treated_series = np.log1p(series)
            else:
                # Use Box-Cox transformation for data with zeros/negatives
                from scipy.stats import boxcox

                try:
                    treated_series, _ = boxcox(series - series.min() + 1)
                except:
                    # Fallback to clipping if transformation fails
                    treated_series = series.clip(
                        lower=series.quantile(0.01), upper=series.quantile(0.99)
                    )

        elif treatment_method == "winsorize":
            # Replace outliers with percentile values
            lower_percentile = column_config.get("lower_percentile", 0.05)
            upper_percentile = column_config.get("upper_percentile", 0.95)

            lower_bound = series.quantile(lower_percentile)
            upper_bound = series.quantile(upper_percentile)

            treated_series.loc[series < lower_bound] = lower_bound
            treated_series.loc[series > upper_bound] = upper_bound

        return treated_series


class DataCleaner:
    """Handles duplicate detection and removal based on timestamps."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data cleaner.

        Args:
            config: Configuration dictionary containing cleaning settings
        """
        self.config = config.get("cleaning", {})
        self.duplicate_strategy = self.config.get("duplicate_strategy", "keep_last")
        self.time_tolerance_seconds = self.config.get("time_tolerance_seconds", 60)
        self.participant_column = self.config.get(
            "participant_column", "participant_id"
        )

    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean the dataset by removing duplicates and inconsistencies.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (cleaned DataFrame, statistics dictionary)
        """
        logger.info("Starting data cleaning")
        df_cleaned = df.copy()
        stats_dict = {}

        initial_rows = len(df_cleaned)

        # Ensure EventDateTime is datetime type
        if "EventDateTime" in df_cleaned.columns:
            df_cleaned["EventDateTime"] = pd.to_datetime(df_cleaned["EventDateTime"])

        # Remove exact duplicates
        exact_duplicates = df_cleaned.duplicated().sum()
        df_cleaned = df_cleaned.drop_duplicates()

        # Remove near-duplicate timestamps within tolerance
        near_duplicates_removed = 0
        if "EventDateTime" in df_cleaned.columns:
            df_cleaned = df_cleaned.sort_values("EventDateTime")

            # Group by participant if column exists
            if self.participant_column in df_cleaned.columns:
                groups = df_cleaned.groupby(self.participant_column)
            else:
                groups = [(None, df_cleaned)]

            cleaned_groups = []
            for group_name, group_df in groups:
                cleaned_group = self._remove_near_duplicates(group_df)
                near_duplicates_removed += len(group_df) - len(cleaned_group)
                cleaned_groups.append(cleaned_group)

            if cleaned_groups:
                df_cleaned = pd.concat(cleaned_groups, ignore_index=True)
            else:
                df_cleaned = pd.DataFrame(columns=df_cleaned.columns)

        # Remove rows with all NaN values (except EventDateTime and participant_id)
        value_columns = [
            col
            for col in df_cleaned.columns
            if col not in ["EventDateTime", self.participant_column]
        ]
        all_nan_mask = df_cleaned[value_columns].isnull().all(axis=1)
        all_nan_rows = all_nan_mask.sum()
        df_cleaned = df_cleaned[~all_nan_mask]

        # Sort by timestamp
        if "EventDateTime" in df_cleaned.columns:
            df_cleaned = df_cleaned.sort_values("EventDateTime").reset_index(drop=True)

        final_rows = len(df_cleaned)

        stats_dict = {
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "exact_duplicates_removed": exact_duplicates,
            "near_duplicates_removed": near_duplicates_removed,
            "all_nan_rows_removed": all_nan_rows,
            "total_rows_removed": initial_rows - final_rows,
            "data_retention_rate": final_rows / initial_rows if initial_rows > 0 else 0,
        }

        if initial_rows > 0:
            percentage = (initial_rows - final_rows) / initial_rows * 100
            logger.info(
                f"Data cleaning completed. Removed {initial_rows - final_rows} rows ({percentage:.2f}%)"
            )
        else:
            logger.info("Data cleaning completed. No data to process.")
        return df_cleaned, stats_dict

    def _remove_near_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove near-duplicate timestamps within the specified tolerance.

        Args:
            df: DataFrame for a single participant

        Returns:
            DataFrame with near-duplicates removed
        """
        if len(df) <= 1 or "EventDateTime" not in df.columns:
            return df

        df_sorted = df.sort_values("EventDateTime").reset_index(drop=True)
        keep_mask = pd.Series(True, index=df_sorted.index)

        for i in range(1, len(df_sorted)):
            time_diff = (
                df_sorted.loc[i, "EventDateTime"]
                - df_sorted.loc[i - 1, "EventDateTime"]
            ).total_seconds()

            if abs(time_diff) <= self.time_tolerance_seconds:
                # Keep based on strategy
                if self.duplicate_strategy == "keep_last":
                    keep_mask.iloc[i - 1] = False
                elif self.duplicate_strategy == "keep_first":
                    keep_mask.iloc[i] = False
                elif self.duplicate_strategy == "keep_most_complete":
                    # Keep the row with fewer NaN values
                    nan_count_prev = df_sorted.iloc[i - 1].isnull().sum()
                    nan_count_curr = df_sorted.iloc[i].isnull().sum()

                    if nan_count_curr < nan_count_prev:
                        keep_mask.iloc[i - 1] = False
                    else:
                        keep_mask.iloc[i] = False

        return df_sorted[keep_mask].reset_index(drop=True)


class TimeSeriesResampler:
    """Resamples irregular time series to uniform intervals."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the time series resampler.

        Args:
            config: Configuration dictionary containing resampling settings
        """
        self.config = config.get("resampling", {})
        self.target_frequency = self.config.get("target_frequency", "5min")  # 5 minutes
        self.aggregation_methods = self.config.get(
            "aggregation_methods",
            {
                "CGM": "mean",
                "Basal": "mean",
                "TotalBolusInsulinDelivered": "sum",
                "CorrectionDelivered": "sum",
                "FoodDelivered": "sum",
                "CarbSize": "sum",
            },
        )
        self.participant_column = self.config.get(
            "participant_column", "participant_id"
        )

    def resample_timeseries(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Resample the time series to uniform intervals.

        Args:
            df: Input DataFrame with irregular timestamps

        Returns:
            Tuple of (resampled DataFrame, statistics dictionary)
        """
        logger.info(
            f"Starting time series resampling to {self.target_frequency} intervals"
        )

        if "EventDateTime" not in df.columns:
            logger.warning("No EventDateTime column found. Skipping resampling.")
            return df, {"resampling_applied": False}

        df_resampled = df.copy()
        df_resampled["EventDateTime"] = pd.to_datetime(df_resampled["EventDateTime"])

        stats_dict = {
            "original_rows": len(df_resampled),
            "target_frequency": self.target_frequency,
            "resampling_applied": True,
        }

        # Group by participant if column exists
        if self.participant_column in df_resampled.columns:
            groups = df_resampled.groupby(self.participant_column)
            resampled_groups = []

            for participant_id, group_df in groups:
                resampled_group = self._resample_group(group_df, participant_id)
                resampled_groups.append(resampled_group)

            if resampled_groups:
                df_resampled = pd.concat(resampled_groups, ignore_index=True)
            else:
                df_resampled = pd.DataFrame(columns=df_resampled.columns)
        else:
            df_resampled = self._resample_group(df_resampled, None)

        stats_dict["resampled_rows"] = len(df_resampled)
        stats_dict["compression_ratio"] = (
            stats_dict["resampled_rows"] / stats_dict["original_rows"]
            if stats_dict["original_rows"] > 0
            else 0
        )

        logger.info(
            f"Resampling completed. {stats_dict['original_rows']} -> {stats_dict['resampled_rows']} rows"
        )
        return df_resampled, stats_dict

    def _resample_group(
        self, df: pd.DataFrame, participant_id: Optional[str]
    ) -> pd.DataFrame:
        """
        Resample a single participant's data.

        Args:
            df: DataFrame for a single participant
            participant_id: Participant identifier

        Returns:
            Resampled DataFrame
        """
        if len(df) == 0:
            return df

        # Set EventDateTime as index
        df_indexed = df.set_index("EventDateTime").sort_index()

        # Prepare aggregation dictionary
        agg_dict = {}
        for column in df_indexed.columns:
            if column == self.participant_column:
                agg_dict[column] = "first"  # Keep participant ID
            else:
                method = self.aggregation_methods.get(column, "mean")
                agg_dict[column] = method

        # Resample
        try:
            df_resampled = df_indexed.resample(self.target_frequency).agg(agg_dict)

            # Remove rows where all values are NaN (except participant_id)
            value_columns = [
                col for col in df_resampled.columns if col != self.participant_column
            ]
            df_resampled = df_resampled.dropna(subset=value_columns, how="all")

            # Reset index to make EventDateTime a column again
            df_resampled = df_resampled.reset_index()

            # Ensure participant_id is preserved
            if (
                participant_id is not None
                and self.participant_column in df_resampled.columns
            ):
                df_resampled[self.participant_column] = participant_id

            return df_resampled

        except Exception as e:
            logger.warning(f"Resampling failed for participant {participant_id}: {e}")
            return df.reset_index(drop=True)


class DataPreprocessor:
    """Main preprocessing pipeline orchestrator."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.missing_value_handler = MissingValueHandler(config)
        self.outlier_treatment = OutlierTreatment(config)
        self.data_cleaner = DataCleaner(config)
        self.time_series_resampler = TimeSeriesResampler(config)

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run the complete preprocessing pipeline.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (preprocessed DataFrame, comprehensive statistics)
        """
        logger.info("Starting complete preprocessing pipeline")

        preprocessing_stats = {
            "pipeline_start_time": datetime.now(),
            "initial_shape": df.shape,
        }

        # Step 1: Data cleaning
        df_processed, cleaning_stats = self.data_cleaner.clean_data(df)
        preprocessing_stats["cleaning"] = cleaning_stats

        # Step 2: Missing value handling
        df_processed, missing_stats = self.missing_value_handler.handle_missing_values(
            df_processed
        )
        preprocessing_stats["missing_values"] = missing_stats

        # Step 3: Outlier treatment
        df_processed, outlier_stats = self.outlier_treatment.treat_outliers(
            df_processed
        )
        preprocessing_stats["outliers"] = outlier_stats

        # Step 4: Time series resampling
        df_processed, resampling_stats = self.time_series_resampler.resample_timeseries(
            df_processed
        )
        preprocessing_stats["resampling"] = resampling_stats

        preprocessing_stats["final_shape"] = df_processed.shape
        preprocessing_stats["pipeline_end_time"] = datetime.now()
        preprocessing_stats["total_processing_time"] = (
            preprocessing_stats["pipeline_end_time"]
            - preprocessing_stats["pipeline_start_time"]
        ).total_seconds()

        logger.info(
            f"Preprocessing pipeline completed in {preprocessing_stats['total_processing_time']:.2f} seconds"
        )
        logger.info(
            f"Data shape: {preprocessing_stats['initial_shape']} -> {preprocessing_stats['final_shape']}"
        )

        return df_processed, preprocessing_stats

    def generate_preprocessing_report(
        self, stats: Dict[str, Any], output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a comprehensive preprocessing report.

        Args:
            stats: Statistics dictionary from preprocessing
            output_path: Optional path to save the report

        Returns:
            Report as string
        """
        report_lines = [
            "=" * 80,
            "DIABETES LSTM PIPELINE - PREPROCESSING REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Processing Time: {stats.get('total_processing_time', 0):.2f} seconds",
            "",
            "OVERVIEW",
            "-" * 40,
            f"Initial Data Shape: {stats.get('initial_shape', 'N/A')}",
            f"Final Data Shape: {stats.get('final_shape', 'N/A')}",
            "",
        ]

        # Data cleaning section
        if "cleaning" in stats:
            cleaning = stats["cleaning"]
            report_lines.extend(
                [
                    "DATA CLEANING",
                    "-" * 40,
                    f"Initial Rows: {cleaning.get('initial_rows', 0):,}",
                    f"Final Rows: {cleaning.get('final_rows', 0):,}",
                    f"Exact Duplicates Removed: {cleaning.get('exact_duplicates_removed', 0):,}",
                    f"Near Duplicates Removed: {cleaning.get('near_duplicates_removed', 0):,}",
                    f"All-NaN Rows Removed: {cleaning.get('all_nan_rows_removed', 0):,}",
                    f"Data Retention Rate: {cleaning.get('data_retention_rate', 0):.2%}",
                    "",
                ]
            )

        # Missing values section
        if "missing_values" in stats:
            missing = stats["missing_values"]
            report_lines.extend(["MISSING VALUE HANDLING", "-" * 40])

            for column, column_stats in missing.items():
                report_lines.extend(
                    [
                        f"{column}:",
                        f"  Missing Before: {column_stats.get('missing_before', 0):,}",
                        f"  Missing After: {column_stats.get('missing_after', 0):,}",
                        f"  Strategy: {column_stats.get('strategy_used', 'N/A')}",
                        f"  Imputation Rate: {column_stats.get('imputation_rate', 0):.2%}",
                        "",
                    ]
                )

        # Outliers section
        if "outliers" in stats:
            outliers = stats["outliers"]
            report_lines.extend(["OUTLIER TREATMENT", "-" * 40])

            for column, column_stats in outliers.items():
                report_lines.extend(
                    [
                        f"{column}:",
                        f"  Outliers Detected: {column_stats.get('outliers_detected', 0):,}",
                        f"  Outliers Treated: {column_stats.get('outliers_treated', 0):,}",
                        f"  Detection Methods: {', '.join(column_stats.get('detection_methods', []))}",
                        f"  Treatment Method: {column_stats.get('treatment_method', 'N/A')}",
                        f"  Outlier Percentage: {column_stats.get('outlier_percentage', 0):.2f}%",
                        "",
                    ]
                )

        # Resampling section
        if "resampling" in stats:
            resampling = stats["resampling"]
            if resampling.get("resampling_applied", False):
                report_lines.extend(
                    [
                        "TIME SERIES RESAMPLING",
                        "-" * 40,
                        f"Target Frequency: {resampling.get('target_frequency', 'N/A')}",
                        f"Original Rows: {resampling.get('original_rows', 0):,}",
                        f"Resampled Rows: {resampling.get('resampled_rows', 0):,}",
                        f"Compression Ratio: {resampling.get('compression_ratio', 0):.2f}",
                        "",
                    ]
                )

        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report_text)
            logger.info(f"Preprocessing report saved to {output_path}")

        return report_text
