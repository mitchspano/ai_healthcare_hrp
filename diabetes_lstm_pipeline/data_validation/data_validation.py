"""
Data validation module for diabetes LSTM pipeline.

This module provides comprehensive data validation and quality assessment
for the AZT1D diabetes dataset, including schema validation, quality metrics,
and outlier detection.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of schema validation."""

    is_valid: bool
    missing_columns: List[str] = field(default_factory=list)
    invalid_types: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Data quality assessment report."""

    total_records: int
    missing_value_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    data_type_stats: Dict[str, str] = field(default_factory=dict)
    value_range_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    duplicate_records: int = 0
    quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class OutlierReport:
    """Outlier detection report."""

    method: str
    outlier_indices: List[int] = field(default_factory=list)
    outlier_counts: Dict[str, int] = field(default_factory=dict)
    outlier_percentages: Dict[str, float] = field(default_factory=dict)
    summary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


class SchemaValidator:
    """Validates dataset schema and data types."""

    def __init__(self, schema_config: Dict[str, Any]):
        """
        Initialize schema validator.

        Args:
            schema_config: Configuration containing expected schema
        """
        self.schema_config = schema_config
        self.required_columns = self._get_required_columns()
        self.expected_types = self._get_expected_types()
        self.value_ranges = self._get_value_ranges()

    def _get_required_columns(self) -> List[str]:
        """Get list of required columns from config."""
        return [
            "EventDateTime",
            "DeviceMode",
            "BolusType",
            "Basal",
            "CorrectionDelivered",
            "TotalBolusInsulinDelivered",
            "FoodDelivered",
            "CarbSize",
            "CGM",
        ]

    def _get_expected_types(self) -> Dict[str, str]:
        """Get expected data types for each column."""
        return {
            "EventDateTime": "datetime64[ns]",
            "DeviceMode": "object",
            "BolusType": "object",
            "Basal": "float64",
            "CorrectionDelivered": "float64",
            "TotalBolusInsulinDelivered": "float64",
            "FoodDelivered": "float64",
            "CarbSize": "float64",
            "CGM": "float64",
        }

    def _get_value_ranges(self) -> Dict[str, Dict[str, float]]:
        """Get expected value ranges for numeric columns."""
        return {
            "Basal": {"min": 0.0, "max": 10.0},
            "CorrectionDelivered": {"min": 0.0, "max": 50.0},
            "TotalBolusInsulinDelivered": {"min": 0.0, "max": 50.0},
            "FoodDelivered": {"min": 0.0, "max": 200.0},
            "CarbSize": {"min": 0.0, "max": 200.0},
            "CGM": {"min": 20.0, "max": 600.0},  # mg/dL range
        }

    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate dataset schema.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(is_valid=True)

        # Check for missing columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            result.missing_columns = missing_cols
            result.is_valid = False
            result.errors.append(f"Missing required columns: {missing_cols}")

        # Check data types for existing columns
        for col in self.required_columns:
            if col in df.columns:
                expected_type = self.expected_types[col]
                actual_type = str(df[col].dtype)

                # Special handling for datetime
                if (
                    expected_type == "datetime64[ns]"
                    and not pd.api.types.is_datetime64_any_dtype(df[col])
                ):
                    result.invalid_types[col] = (
                        f"Expected {expected_type}, got {actual_type}"
                    )
                    result.warnings.append(f"Column {col} should be datetime type")
                elif expected_type != "datetime64[ns]" and expected_type != actual_type:
                    # Allow some flexibility for numeric types
                    if not (
                        expected_type == "float64"
                        and pd.api.types.is_numeric_dtype(df[col])
                    ):
                        result.invalid_types[col] = (
                            f"Expected {expected_type}, got {actual_type}"
                        )
                        result.warnings.append(
                            f"Column {col} has unexpected type: {actual_type}"
                        )

        # Check value ranges for numeric columns
        for col, ranges in self.value_ranges.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                valid_data = df[col].dropna()
                if len(valid_data) > 0:
                    min_val, max_val = valid_data.min(), valid_data.max()
                    if min_val < ranges["min"] or max_val > ranges["max"]:
                        result.warnings.append(
                            f"Column {col} has values outside expected range "
                            f"[{ranges['min']}, {ranges['max']}]: actual range [{min_val:.2f}, {max_val:.2f}]"
                        )

        logger.info(f"Schema validation completed. Valid: {result.is_valid}")
        return result


class QualityAssessor:
    """Assesses data quality and generates comprehensive reports."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize quality assessor.

        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}

    def assess_quality(self, df: pd.DataFrame) -> QualityReport:
        """
        Assess overall data quality.

        Args:
            df: DataFrame to assess

        Returns:
            QualityReport with comprehensive quality metrics
        """
        report = QualityReport(total_records=len(df))

        # Missing value analysis
        report.missing_value_stats = self._analyze_missing_values(df)

        # Data type analysis
        report.data_type_stats = self._analyze_data_types(df)

        # Value range analysis
        report.value_range_stats = self._analyze_value_ranges(df)

        # Duplicate analysis
        report.duplicate_records = self._count_duplicates(df)

        # Calculate overall quality score
        report.quality_score = self._calculate_quality_score(df, report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        logger.info(f"Quality assessment completed. Score: {report.quality_score:.2f}")
        return report

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze missing values in the dataset."""
        missing_stats = {}

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100

            missing_stats[col] = {
                "count": int(missing_count),
                "percentage": float(missing_percentage),
                "consecutive_missing": self._find_consecutive_missing(df[col]),
            }

        return missing_stats

    def _find_consecutive_missing(self, series: pd.Series) -> int:
        """Find maximum consecutive missing values."""
        if series.isnull().sum() == 0:
            return 0

        # Convert to boolean mask and find consecutive groups
        is_null = series.isnull()

        # Create groups of consecutive values (both null and non-null)
        groups = (is_null != is_null.shift()).cumsum()

        # Filter to only null groups and count their sizes
        null_group_sizes = is_null.groupby(groups).sum()
        null_group_sizes = null_group_sizes[null_group_sizes > 0]

        return int(null_group_sizes.max()) if len(null_group_sizes) > 0 else 0

    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze data types of columns."""
        return {col: str(df[col].dtype) for col in df.columns}

    def _analyze_value_ranges(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze value ranges for numeric columns."""
        range_stats = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                range_stats[col] = {
                    "min": float(valid_data.min()),
                    "max": float(valid_data.max()),
                    "mean": float(valid_data.mean()),
                    "std": float(valid_data.std()),
                    "median": float(valid_data.median()),
                    "q25": float(valid_data.quantile(0.25)),
                    "q75": float(valid_data.quantile(0.75)),
                }

        return range_stats

    def _count_duplicates(self, df: pd.DataFrame) -> int:
        """Count duplicate records."""
        return int(df.duplicated().sum())

    def _calculate_quality_score(
        self, df: pd.DataFrame, report: QualityReport
    ) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0

        # Handle empty DataFrame
        if len(df) == 0 or len(df.columns) == 0:
            return 0.0

        # Penalize for missing values
        total_missing_percentage = sum(
            stats["percentage"] for stats in report.missing_value_stats.values()
        ) / len(df.columns)
        score -= min(total_missing_percentage, 50)  # Max 50 point penalty

        # Penalize for duplicates
        duplicate_percentage = (report.duplicate_records / len(df)) * 100
        score -= min(duplicate_percentage * 2, 20)  # Max 20 point penalty

        return max(score, 0.0)

    def _generate_recommendations(self, report: QualityReport) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []

        # Missing value recommendations
        high_missing_cols = [
            col
            for col, stats in report.missing_value_stats.items()
            if stats["percentage"] > 20
        ]
        if high_missing_cols:
            recommendations.append(
                f"Consider imputation strategies for columns with high missing values: {high_missing_cols}"
            )

        # Duplicate recommendations
        if report.duplicate_records > 0:
            recommendations.append(
                f"Remove {report.duplicate_records} duplicate records to improve data quality"
            )

        # Quality score recommendations
        if report.quality_score < 70:
            recommendations.append(
                "Data quality is below acceptable threshold. Consider data cleaning."
            )

        return recommendations


class OutlierDetector:
    """Detects outliers using various statistical methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize outlier detector.

        Args:
            config: Configuration with detection parameters
        """
        self.config = config or {}
        self.methods = {
            "iqr": self._detect_iqr_outliers,
            "zscore": self._detect_zscore_outliers,
            "isolation_forest": self._detect_isolation_forest_outliers,
        }

    def detect_outliers(
        self, df: pd.DataFrame, method: str = "iqr", columns: Optional[List[str]] = None
    ) -> OutlierReport:
        """
        Detect outliers using specified method.

        Args:
            df: DataFrame to analyze
            method: Detection method ('iqr', 'zscore', 'isolation_forest')
            columns: Columns to analyze (default: all numeric columns)

        Returns:
            OutlierReport with detection results
        """
        if method not in self.methods:
            raise ValueError(
                f"Unknown method: {method}. Available: {list(self.methods.keys())}"
            )

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Filter to existing numeric columns
        columns = [
            col
            for col in columns
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

        report = OutlierReport(method=method)

        # Detect outliers using specified method
        outlier_indices = self.methods[method](df, columns)
        report.outlier_indices = outlier_indices

        # Calculate statistics per column
        for col in columns:
            col_outliers = self._get_column_outliers(df, col, outlier_indices)
            report.outlier_counts[col] = len(col_outliers)
            # Handle empty DataFrame
            if len(df) > 0:
                report.outlier_percentages[col] = (len(col_outliers) / len(df)) * 100
            else:
                report.outlier_percentages[col] = 0.0

            # Summary statistics
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                report.summary_stats[col] = {
                    "mean": float(valid_data.mean()),
                    "std": float(valid_data.std()),
                    "outlier_threshold_low": self._get_threshold(
                        valid_data, method, "low"
                    ),
                    "outlier_threshold_high": self._get_threshold(
                        valid_data, method, "high"
                    ),
                }

        logger.info(
            f"Outlier detection completed using {method}. Found {len(outlier_indices)} outliers."
        )
        return report

    def _detect_iqr_outliers(self, df: pd.DataFrame, columns: List[str]) -> List[int]:
        """Detect outliers using Interquartile Range method."""
        outlier_indices = set()

        for col in columns:
            data = df[col].dropna()
            if len(data) == 0:
                continue

            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outlier_indices.update(col_outliers)

        return list(outlier_indices)

    def _detect_zscore_outliers(
        self, df: pd.DataFrame, columns: List[str]
    ) -> List[int]:
        """Detect outliers using Z-score method."""
        threshold = self.config.get("zscore_threshold", 3.0)
        outlier_indices = set()

        for col in columns:
            data = df[col].dropna()
            if len(data) == 0:
                continue

            z_scores = np.abs(stats.zscore(data))
            col_outliers = data[z_scores > threshold].index
            outlier_indices.update(col_outliers)

        return list(outlier_indices)

    def _detect_isolation_forest_outliers(
        self, df: pd.DataFrame, columns: List[str]
    ) -> List[int]:
        """Detect outliers using Isolation Forest method."""
        contamination = self.config.get("contamination", 0.1)
        random_state = self.config.get("random_state", 42)

        # Prepare data for isolation forest
        data = df[columns].dropna()
        if len(data) == 0:
            return []

        # Fit isolation forest
        iso_forest = IsolationForest(
            contamination=contamination, random_state=random_state, n_jobs=-1
        )

        outlier_labels = iso_forest.fit_predict(data)
        outlier_indices = data[outlier_labels == -1].index.tolist()

        return outlier_indices

    def _get_column_outliers(
        self, df: pd.DataFrame, column: str, all_outlier_indices: List[int]
    ) -> List[int]:
        """Get outlier indices for a specific column."""
        return [
            idx
            for idx in all_outlier_indices
            if idx in df.index and pd.notna(df.loc[idx, column])
        ]

    def _get_threshold(self, data: pd.Series, method: str, bound: str) -> float:
        """Get outlier threshold for a given method and bound."""
        if method == "iqr":
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            if bound == "low":
                return float(Q1 - 1.5 * IQR)
            else:
                return float(Q3 + 1.5 * IQR)
        elif method == "zscore":
            threshold = self.config.get("zscore_threshold", 3.0)
            mean, std = data.mean(), data.std()
            if bound == "low":
                return float(mean - threshold * std)
            else:
                return float(mean + threshold * std)
        else:
            return float("nan")


class ValidationReportGenerator:
    """Generates comprehensive validation reports with visualizations."""

    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_comprehensive_report(
        self,
        df: pd.DataFrame,
        validation_result: ValidationResult,
        quality_report: QualityReport,
        outlier_report: OutlierReport,
        save_plots: bool = True,
    ) -> str:
        """
        Generate comprehensive validation report.

        Args:
            df: Original DataFrame
            validation_result: Schema validation results
            quality_report: Data quality assessment results
            outlier_report: Outlier detection results
            save_plots: Whether to save visualization plots

        Returns:
            Report as formatted string
        """
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("DIABETES LSTM PIPELINE - DATA VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"Dataset shape: {df.shape}")
        report_lines.append("")

        # Schema validation section
        report_lines.append("SCHEMA VALIDATION")
        report_lines.append("-" * 40)
        report_lines.append(
            f"Overall validation: {'PASSED' if validation_result.is_valid else 'FAILED'}"
        )

        if validation_result.missing_columns:
            report_lines.append(f"Missing columns: {validation_result.missing_columns}")

        if validation_result.invalid_types:
            report_lines.append("Type mismatches:")
            for col, issue in validation_result.invalid_types.items():
                report_lines.append(f"  - {col}: {issue}")

        if validation_result.errors:
            report_lines.append("Errors:")
            for error in validation_result.errors:
                report_lines.append(f"  - {error}")

        if validation_result.warnings:
            report_lines.append("Warnings:")
            for warning in validation_result.warnings:
                report_lines.append(f"  - {warning}")

        report_lines.append("")

        # Quality assessment section
        report_lines.append("DATA QUALITY ASSESSMENT")
        report_lines.append("-" * 40)
        report_lines.append(
            f"Overall quality score: {quality_report.quality_score:.2f}/100"
        )
        report_lines.append(f"Total records: {quality_report.total_records:,}")
        report_lines.append(f"Duplicate records: {quality_report.duplicate_records:,}")
        report_lines.append("")

        # Missing values summary
        report_lines.append("Missing Values Summary:")
        for col, stats in quality_report.missing_value_stats.items():
            if stats["count"] > 0:
                report_lines.append(
                    f"  - {col}: {stats['count']:,} ({stats['percentage']:.1f}%)"
                )

        report_lines.append("")

        # Outlier detection section
        report_lines.append("OUTLIER DETECTION")
        report_lines.append("-" * 40)
        report_lines.append(f"Detection method: {outlier_report.method}")
        report_lines.append(
            f"Total outlier records: {len(outlier_report.outlier_indices):,}"
        )
        report_lines.append("")

        report_lines.append("Outliers by column:")
        for col, count in outlier_report.outlier_counts.items():
            percentage = outlier_report.outlier_percentages[col]
            report_lines.append(f"  - {col}: {count:,} ({percentage:.1f}%)")

        report_lines.append("")

        # Recommendations
        if quality_report.recommendations:
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 40)
            for i, rec in enumerate(quality_report.recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")

        # Generate visualizations if requested
        if save_plots:
            self._generate_visualizations(df, quality_report, outlier_report)
            report_lines.append("Visualization plots saved to: " + str(self.output_dir))

        report_text = "\n".join(report_lines)

        # Save report to file
        report_file = (
            self.output_dir
            / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(report_file, "w") as f:
            f.write(report_text)

        logger.info(f"Comprehensive validation report saved to: {report_file}")
        return report_text

    def _generate_visualizations(
        self,
        df: pd.DataFrame,
        quality_report: QualityReport,
        outlier_report: OutlierReport,
    ):
        """Generate and save visualization plots."""
        plt.style.use("default")

        # Missing values heatmap
        if any(
            stats["count"] > 0 for stats in quality_report.missing_value_stats.values()
        ):
            fig, ax = plt.subplots(figsize=(12, 8))
            missing_data = df.isnull()
            sns.heatmap(missing_data, cbar=True, ax=ax, cmap="viridis")
            ax.set_title("Missing Values Heatmap")
            plt.tight_layout()
            plt.savefig(
                self.output_dir / "missing_values_heatmap.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # Outlier visualization for numeric columns
        numeric_cols = [
            col
            for col in df.select_dtypes(include=[np.number]).columns
            if col in outlier_report.outlier_counts
        ]

        if numeric_cols:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()

            for i, col in enumerate(numeric_cols):
                if i < len(axes):
                    data = df[col].dropna()
                    if len(data) > 0:
                        axes[i].boxplot(data)
                        axes[i].set_title(
                            f"{col} - Outliers: {outlier_report.outlier_counts[col]}"
                        )
                        axes[i].set_ylabel("Value")

            # Hide unused subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "outlier_boxplots.png", dpi=300, bbox_inches="tight"
            )
            plt.close()


class DataValidator:
    """Main data validation orchestrator."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.schema_validator = SchemaValidator(config)
        self.quality_assessor = QualityAssessor(config)
        self.outlier_detector = OutlierDetector(config)
        self.report_generator = ValidationReportGenerator(
            config.get("report_output_dir", "reports")
        )

    def validate_dataset(
        self,
        df: pd.DataFrame,
        outlier_method: str = "iqr",
        generate_report: bool = True,
    ) -> Tuple[ValidationResult, QualityReport, OutlierReport]:
        """
        Perform comprehensive dataset validation.

        Args:
            df: DataFrame to validate
            outlier_method: Method for outlier detection
            generate_report: Whether to generate comprehensive report

        Returns:
            Tuple of (ValidationResult, QualityReport, OutlierReport)
        """
        logger.info("Starting comprehensive dataset validation...")

        # Schema validation
        validation_result = self.schema_validator.validate_schema(df)

        # Quality assessment
        quality_report = self.quality_assessor.assess_quality(df)

        # Outlier detection
        outlier_report = self.outlier_detector.detect_outliers(
            df, method=outlier_method
        )

        # Generate comprehensive report
        if generate_report:
            report_text = self.report_generator.generate_comprehensive_report(
                df, validation_result, quality_report, outlier_report
            )
            logger.info("Comprehensive validation report generated")

        logger.info("Dataset validation completed successfully")
        return validation_result, quality_report, outlier_report
