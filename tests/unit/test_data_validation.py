"""
Unit tests for data validation module.

Tests cover schema validation, quality assessment, outlier detection,
and edge cases for the diabetes LSTM pipeline data validation system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from diabetes_lstm_pipeline.data_validation import (
    DataValidator,
    SchemaValidator,
    QualityAssessor,
    OutlierDetector,
    ValidationReportGenerator,
    ValidationResult,
    QualityReport,
    OutlierReport,
)


class TestSchemaValidator:
    """Test cases for SchemaValidator class."""

    @pytest.fixture
    def schema_config(self):
        """Sample schema configuration."""
        return {
            "required_columns": [
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
        }

    @pytest.fixture
    def validator(self, schema_config):
        """Create SchemaValidator instance."""
        return SchemaValidator(schema_config)

    @pytest.fixture
    def valid_dataframe(self):
        """Create a valid DataFrame for testing."""
        dates = pd.date_range("2023-01-01", periods=100, freq="5min")
        return pd.DataFrame(
            {
                "EventDateTime": dates,
                "DeviceMode": ["Auto"] * 100,
                "BolusType": ["Normal"] * 100,
                "Basal": np.random.uniform(0.5, 2.0, 100),
                "CorrectionDelivered": np.random.uniform(0, 5, 100),
                "TotalBolusInsulinDelivered": np.random.uniform(0, 10, 100),
                "FoodDelivered": np.random.uniform(0, 50, 100),
                "CarbSize": np.random.uniform(0, 100, 100),
                "CGM": np.random.uniform(70, 200, 100),
            }
        )

    def test_valid_schema(self, validator, valid_dataframe):
        """Test validation of valid schema."""
        result = validator.validate_schema(valid_dataframe)

        assert result.is_valid is True
        assert len(result.missing_columns) == 0
        assert len(result.invalid_types) == 0
        assert len(result.errors) == 0

    def test_missing_columns(self, validator, valid_dataframe):
        """Test detection of missing columns."""
        # Remove some required columns
        df_missing = valid_dataframe.drop(["CGM", "Basal"], axis=1)

        result = validator.validate_schema(df_missing)

        assert result.is_valid is False
        assert "CGM" in result.missing_columns
        assert "Basal" in result.missing_columns
        assert len(result.errors) > 0

    def test_invalid_data_types(self, validator, valid_dataframe):
        """Test detection of invalid data types."""
        # Convert numeric column to string
        df_invalid = valid_dataframe.copy()
        df_invalid["CGM"] = df_invalid["CGM"].astype(str)

        result = validator.validate_schema(df_invalid)

        # Should generate warnings for type mismatches
        assert len(result.warnings) > 0

    def test_value_range_validation(self, validator, valid_dataframe):
        """Test validation of value ranges."""
        # Create data with out-of-range values
        df_out_of_range = valid_dataframe.copy()
        df_out_of_range.loc[0, "CGM"] = 1000  # Way above normal range
        df_out_of_range.loc[1, "Basal"] = -1  # Below minimum

        result = validator.validate_schema(df_out_of_range)

        # Should generate warnings for out-of-range values
        assert len(result.warnings) > 0

    def test_datetime_validation(self, validator, valid_dataframe):
        """Test datetime column validation."""
        # Convert datetime to string
        df_invalid_datetime = valid_dataframe.copy()
        df_invalid_datetime["EventDateTime"] = df_invalid_datetime[
            "EventDateTime"
        ].astype(str)

        result = validator.validate_schema(df_invalid_datetime)

        # Should detect datetime type issue
        assert "EventDateTime" in result.invalid_types or len(result.warnings) > 0

    def test_empty_dataframe(self, validator):
        """Test validation of empty DataFrame."""
        empty_df = pd.DataFrame()

        result = validator.validate_schema(empty_df)

        assert result.is_valid is False
        assert len(result.missing_columns) > 0


class TestQualityAssessor:
    """Test cases for QualityAssessor class."""

    @pytest.fixture
    def assessor(self):
        """Create QualityAssessor instance."""
        return QualityAssessor()

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame with quality issues."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="5min")

        df = pd.DataFrame(
            {
                "EventDateTime": dates,
                "CGM": np.random.uniform(70, 200, 100),
                "Basal": np.random.uniform(0.5, 2.0, 100),
                "CarbSize": np.random.uniform(0, 100, 100),
            }
        )

        # Introduce missing values
        df.loc[10:15, "CGM"] = np.nan
        df.loc[20:22, "Basal"] = np.nan

        # Introduce duplicates
        df.loc[50] = df.loc[49]

        return df

    def test_missing_value_analysis(self, assessor, sample_dataframe):
        """Test missing value analysis."""
        report = assessor.assess_quality(sample_dataframe)

        assert "CGM" in report.missing_value_stats
        assert report.missing_value_stats["CGM"]["count"] == 6
        assert report.missing_value_stats["CGM"]["percentage"] == 6.0

        assert "Basal" in report.missing_value_stats
        assert report.missing_value_stats["Basal"]["count"] == 3

    def test_duplicate_detection(self, assessor, sample_dataframe):
        """Test duplicate record detection."""
        report = assessor.assess_quality(sample_dataframe)

        assert report.duplicate_records == 1

    def test_quality_score_calculation(self, assessor, sample_dataframe):
        """Test quality score calculation."""
        report = assessor.assess_quality(sample_dataframe)

        # Quality score should be less than 100 due to missing values and duplicates
        assert 0 <= report.quality_score <= 100
        assert report.quality_score < 100

    def test_value_range_statistics(self, assessor, sample_dataframe):
        """Test value range statistics calculation."""
        report = assessor.assess_quality(sample_dataframe)

        assert "CGM" in report.value_range_stats
        cgm_stats = report.value_range_stats["CGM"]

        assert "min" in cgm_stats
        assert "max" in cgm_stats
        assert "mean" in cgm_stats
        assert "std" in cgm_stats
        assert "median" in cgm_stats

    def test_recommendations_generation(self, assessor, sample_dataframe):
        """Test generation of quality recommendations."""
        report = assessor.assess_quality(sample_dataframe)

        assert len(report.recommendations) > 0
        # Should recommend handling duplicates
        duplicate_rec = any(
            "duplicate" in rec.lower() for rec in report.recommendations
        )
        assert duplicate_rec

    def test_consecutive_missing_detection(self, assessor):
        """Test detection of consecutive missing values."""
        # Create data with consecutive missing values
        df = pd.DataFrame(
            {"values": [1, 2, np.nan, np.nan, np.nan, 6, 7, np.nan, 9, 10]}
        )

        consecutive_missing = assessor._find_consecutive_missing(df["values"])
        assert consecutive_missing == 3  # Three consecutive NaN values

    def test_empty_dataframe_quality(self, assessor):
        """Test quality assessment of empty DataFrame."""
        empty_df = pd.DataFrame()

        report = assessor.assess_quality(empty_df)

        assert report.total_records == 0
        assert report.quality_score >= 0


class TestOutlierDetector:
    """Test cases for OutlierDetector class."""

    @pytest.fixture
    def detector(self):
        """Create OutlierDetector instance."""
        return OutlierDetector({"random_state": 42})

    @pytest.fixture
    def sample_data_with_outliers(self):
        """Create sample data with known outliers."""
        np.random.seed(42)

        # Normal data
        normal_data = np.random.normal(100, 10, 95)

        # Add clear outliers
        outliers = [200, 300, -50, 400, 500]

        df = pd.DataFrame(
            {
                "CGM": np.concatenate([normal_data, outliers]),
                "Basal": np.random.uniform(0.5, 2.0, 100),
                "CarbSize": np.random.uniform(0, 100, 100),
            }
        )

        return df

    def test_iqr_outlier_detection(self, detector, sample_data_with_outliers):
        """Test IQR-based outlier detection."""
        report = detector.detect_outliers(sample_data_with_outliers, method="iqr")

        assert report.method == "iqr"
        assert len(report.outlier_indices) > 0
        assert "CGM" in report.outlier_counts
        assert report.outlier_counts["CGM"] > 0

    def test_zscore_outlier_detection(self, detector, sample_data_with_outliers):
        """Test Z-score based outlier detection."""
        report = detector.detect_outliers(sample_data_with_outliers, method="zscore")

        assert report.method == "zscore"
        assert len(report.outlier_indices) > 0
        assert "CGM" in report.outlier_counts

    def test_isolation_forest_detection(self, detector, sample_data_with_outliers):
        """Test Isolation Forest outlier detection."""
        report = detector.detect_outliers(
            sample_data_with_outliers, method="isolation_forest"
        )

        assert report.method == "isolation_forest"
        assert (
            len(report.outlier_indices) >= 0
        )  # May not find outliers in small dataset

    def test_invalid_method(self, detector, sample_data_with_outliers):
        """Test handling of invalid detection method."""
        with pytest.raises(ValueError, match="Unknown method"):
            detector.detect_outliers(sample_data_with_outliers, method="invalid_method")

    def test_column_filtering(self, detector, sample_data_with_outliers):
        """Test outlier detection on specific columns."""
        report = detector.detect_outliers(
            sample_data_with_outliers, method="iqr", columns=["CGM"]
        )

        assert "CGM" in report.outlier_counts
        assert (
            "Basal" not in report.outlier_counts or report.outlier_counts["Basal"] == 0
        )

    def test_empty_data_handling(self, detector):
        """Test handling of empty data."""
        empty_df = pd.DataFrame({"values": []})

        report = detector.detect_outliers(empty_df, method="iqr")

        assert len(report.outlier_indices) == 0
        assert report.outlier_counts["values"] == 0
        assert report.outlier_percentages["values"] == 0.0

    def test_all_nan_column(self, detector):
        """Test handling of column with all NaN values."""
        df = pd.DataFrame({"all_nan": [np.nan] * 10, "normal": range(10)})

        report = detector.detect_outliers(df, method="iqr")

        # Should handle NaN column gracefully
        assert len(report.outlier_indices) >= 0

    def test_outlier_percentages(self, detector, sample_data_with_outliers):
        """Test calculation of outlier percentages."""
        report = detector.detect_outliers(sample_data_with_outliers, method="iqr")

        for col, percentage in report.outlier_percentages.items():
            assert 0 <= percentage <= 100
            expected_percentage = (
                report.outlier_counts[col] / len(sample_data_with_outliers)
            ) * 100
            assert abs(percentage - expected_percentage) < 0.01


class TestValidationReportGenerator:
    """Test cases for ValidationReportGenerator class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test reports."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def report_generator(self, temp_dir):
        """Create ValidationReportGenerator instance."""
        return ValidationReportGenerator(temp_dir)

    @pytest.fixture
    def sample_results(self):
        """Create sample validation results."""
        validation_result = ValidationResult(
            is_valid=True,
            missing_columns=[],
            invalid_types={},
            errors=[],
            warnings=["Sample warning"],
        )

        quality_report = QualityReport(
            total_records=100,
            missing_value_stats={
                "CGM": {"count": 5, "percentage": 5.0, "consecutive_missing": 2}
            },
            duplicate_records=2,
            quality_score=85.0,
            recommendations=["Remove duplicates"],
        )

        outlier_report = OutlierReport(
            method="iqr",
            outlier_indices=[1, 2, 3],
            outlier_counts={"CGM": 3},
            outlier_percentages={"CGM": 3.0},
        )

        return validation_result, quality_report, outlier_report

    def test_report_generation(self, report_generator, sample_results):
        """Test comprehensive report generation."""
        validation_result, quality_report, outlier_report = sample_results

        # Create sample DataFrame
        df = pd.DataFrame(
            {
                "CGM": np.random.uniform(70, 200, 100),
                "Basal": np.random.uniform(0.5, 2.0, 100),
            }
        )

        report_text = report_generator.generate_comprehensive_report(
            df, validation_result, quality_report, outlier_report, save_plots=False
        )

        assert "DATA VALIDATION REPORT" in report_text
        assert "SCHEMA VALIDATION" in report_text
        assert "DATA QUALITY ASSESSMENT" in report_text
        assert "OUTLIER DETECTION" in report_text
        assert "85.00/100" in report_text  # Quality score

    def test_report_file_creation(self, report_generator, sample_results):
        """Test that report files are created."""
        validation_result, quality_report, outlier_report = sample_results

        df = pd.DataFrame({"CGM": np.random.uniform(70, 200, 100)})

        report_generator.generate_comprehensive_report(
            df, validation_result, quality_report, outlier_report, save_plots=False
        )

        # Check that report file was created
        report_files = list(
            Path(report_generator.output_dir).glob("validation_report_*.txt")
        )
        assert len(report_files) > 0

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_visualization_generation(
        self, mock_close, mock_savefig, report_generator, sample_results
    ):
        """Test visualization generation."""
        validation_result, quality_report, outlier_report = sample_results

        # Create DataFrame with missing values for visualization
        df = pd.DataFrame(
            {
                "CGM": [100, np.nan, 120, 130, np.nan],
                "Basal": [1.0, 1.2, np.nan, 1.5, 1.1],
            }
        )

        report_generator.generate_comprehensive_report(
            df, validation_result, quality_report, outlier_report, save_plots=True
        )

        # Should have called savefig for plots
        assert mock_savefig.called
        assert mock_close.called


class TestDataValidator:
    """Test cases for main DataValidator orchestrator."""

    @pytest.fixture
    def config(self):
        """Sample configuration."""
        return {"report_output_dir": "test_reports", "random_state": 42}

    @pytest.fixture
    def validator(self, config):
        """Create DataValidator instance."""
        return DataValidator(config)

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=50, freq="5min")

        df = pd.DataFrame(
            {
                "EventDateTime": dates,
                "DeviceMode": ["Auto"] * 50,
                "BolusType": ["Normal"] * 50,
                "Basal": np.random.uniform(0.5, 2.0, 50),
                "CorrectionDelivered": np.random.uniform(0, 5, 50),
                "TotalBolusInsulinDelivered": np.random.uniform(0, 10, 50),
                "FoodDelivered": np.random.uniform(0, 50, 50),
                "CarbSize": np.random.uniform(0, 100, 50),
                "CGM": np.concatenate(
                    [np.random.uniform(70, 200, 45), [300, 400, 500, -50, 600]]
                ),  # Add outliers
            }
        )

        # Add some missing values
        df.loc[5:7, "CGM"] = np.nan

        return df

    def test_comprehensive_validation(self, validator, sample_dataframe):
        """Test comprehensive dataset validation."""
        validation_result, quality_report, outlier_report = validator.validate_dataset(
            sample_dataframe, outlier_method="iqr", generate_report=False
        )

        # Check that all components returned results
        assert isinstance(validation_result, ValidationResult)
        assert isinstance(quality_report, QualityReport)
        assert isinstance(outlier_report, OutlierReport)

        # Validation should pass (all required columns present)
        assert validation_result.is_valid is True

        # Quality report should have reasonable values
        assert quality_report.total_records == 50
        assert 0 <= quality_report.quality_score <= 100

        # Outlier detection should find the extreme values we added
        assert outlier_report.method == "iqr"
        assert len(outlier_report.outlier_indices) > 0

    def test_different_outlier_methods(self, validator, sample_dataframe):
        """Test validation with different outlier detection methods."""
        methods = ["iqr", "zscore", "isolation_forest"]

        for method in methods:
            validation_result, quality_report, outlier_report = (
                validator.validate_dataset(
                    sample_dataframe, outlier_method=method, generate_report=False
                )
            )

            assert outlier_report.method == method
            assert isinstance(outlier_report.outlier_indices, list)

    @patch(
        "diabetes_lstm_pipeline.data_validation.data_validation.ValidationReportGenerator.generate_comprehensive_report"
    )
    def test_report_generation_flag(self, mock_report_gen, validator, sample_dataframe):
        """Test that report generation can be controlled."""
        # Test with report generation disabled
        validator.validate_dataset(sample_dataframe, generate_report=False)
        assert not mock_report_gen.called

        # Test with report generation enabled
        mock_report_gen.return_value = "Mock report"
        validator.validate_dataset(sample_dataframe, generate_report=True)
        assert mock_report_gen.called


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_row_dataframe(self):
        """Test validation with single row DataFrame."""
        df = pd.DataFrame(
            {
                "EventDateTime": [pd.Timestamp("2023-01-01")],
                "DeviceMode": ["Auto"],
                "BolusType": ["Normal"],
                "Basal": [1.0],
                "CorrectionDelivered": [0.0],
                "TotalBolusInsulinDelivered": [0.0],
                "FoodDelivered": [0.0],
                "CarbSize": [0.0],
                "CGM": [100.0],
            }
        )

        validator = DataValidator({})
        validation_result, quality_report, outlier_report = validator.validate_dataset(
            df, generate_report=False
        )

        assert validation_result.is_valid is True
        assert quality_report.total_records == 1

    def test_all_missing_values_column(self):
        """Test handling of column with all missing values."""
        df = pd.DataFrame(
            {
                "EventDateTime": pd.date_range("2023-01-01", periods=10, freq="5min"),
                "DeviceMode": ["Auto"] * 10,
                "BolusType": ["Normal"] * 10,
                "Basal": [np.nan] * 10,  # All missing
                "CorrectionDelivered": [0.0] * 10,
                "TotalBolusInsulinDelivered": [0.0] * 10,
                "FoodDelivered": [0.0] * 10,
                "CarbSize": [0.0] * 10,
                "CGM": np.random.uniform(70, 200, 10),
            }
        )

        validator = DataValidator({})
        validation_result, quality_report, outlier_report = validator.validate_dataset(
            df, generate_report=False
        )

        # Should handle gracefully
        assert quality_report.missing_value_stats["Basal"]["percentage"] == 100.0

    def test_extreme_outliers(self):
        """Test handling of extreme outlier values."""
        df = pd.DataFrame(
            {
                "EventDateTime": pd.date_range("2023-01-01", periods=10, freq="5min"),
                "DeviceMode": ["Auto"] * 10,
                "BolusType": ["Normal"] * 10,
                "Basal": [1.0] * 10,
                "CorrectionDelivered": [0.0] * 10,
                "TotalBolusInsulinDelivered": [0.0] * 10,
                "FoodDelivered": [0.0] * 10,
                "CarbSize": [0.0] * 10,
                "CGM": [
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    100,
                    1e6,
                    -1e6,
                ],  # Extreme outliers
            }
        )

        validator = DataValidator({})
        validation_result, quality_report, outlier_report = validator.validate_dataset(
            df, outlier_method="iqr", generate_report=False
        )

        # Should detect extreme outliers
        assert len(outlier_report.outlier_indices) > 0
        assert outlier_report.outlier_counts["CGM"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
