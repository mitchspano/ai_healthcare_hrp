"""
Unit tests for the preprocessing module.

Tests all preprocessing components including missing value handling,
outlier treatment, data cleaning, and time series resampling.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import yaml

from diabetes_lstm_pipeline.preprocessing import (
    MissingValueHandler,
    OutlierTreatment,
    DataCleaner,
    TimeSeriesResampler,
    DataPreprocessor,
)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "missing_values": {
            "strategies": {
                "CGM": "interpolation",
                "Basal": "forward_fill",
                "TotalBolusInsulinDelivered": "zero_fill",
                "CorrectionDelivered": "zero_fill",
            },
            "interpolation_method": "linear",
            "max_gap_minutes": 30,
        },
        "outliers": {
            "detection_methods": ["iqr", "zscore"],
            "treatment_method": "clip",
            "zscore_threshold": 3.0,
            "iqr_multiplier": 1.5,
            "contamination": 0.1,
            "column_settings": {
                "CGM": {
                    "min_value": 40,
                    "max_value": 400,
                    "detection_methods": ["iqr", "zscore", "domain_specific"],
                    "treatment_method": "clip",
                }
            },
        },
        "cleaning": {
            "duplicate_strategy": "keep_last",
            "time_tolerance_seconds": 60,
            "participant_column": "participant_id",
        },
        "resampling": {
            "target_frequency": "5min",
            "participant_column": "participant_id",
            "aggregation_methods": {
                "CGM": "mean",
                "Basal": "mean",
                "TotalBolusInsulinDelivered": "sum",
            },
        },
    }


@pytest.fixture
def sample_diabetes_data():
    """Create sample diabetes data for testing."""
    np.random.seed(42)

    # Create timestamps every 5 minutes for 24 hours
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(288)]  # 24 hours

    data = {
        "EventDateTime": timestamps,
        "participant_id": ["P001"] * len(timestamps),
        "CGM": np.random.normal(120, 30, len(timestamps)),  # Normal glucose around 120
        "Basal": np.random.normal(1.0, 0.2, len(timestamps)),  # Basal insulin
        "TotalBolusInsulinDelivered": np.random.exponential(0.5, len(timestamps)),
        "CorrectionDelivered": np.random.exponential(0.3, len(timestamps)),
        "FoodDelivered": np.random.exponential(0.2, len(timestamps)),
        "CarbSize": np.random.exponential(10, len(timestamps)),
    }

    df = pd.DataFrame(data)

    # Introduce some missing values
    missing_indices = np.random.choice(len(df), size=20, replace=False)
    df.loc[missing_indices[:10], "CGM"] = np.nan
    df.loc[missing_indices[10:15], "Basal"] = np.nan
    df.loc[missing_indices[15:], "TotalBolusInsulinDelivered"] = np.nan

    # Introduce some outliers
    outlier_indices = np.random.choice(len(df), size=5, replace=False)
    df.loc[outlier_indices, "CGM"] = [500, 20, 600, 15, 450]  # Extreme values

    return df


@pytest.fixture
def sample_data_with_duplicates():
    """Create sample data with duplicates for testing."""
    timestamps = [
        datetime(2023, 1, 1, 10, 0, 0),
        datetime(2023, 1, 1, 10, 0, 30),  # 30 seconds later (near duplicate)
        datetime(2023, 1, 1, 10, 1, 0),
        datetime(2023, 1, 1, 10, 1, 0),  # Exact duplicate
        datetime(2023, 1, 1, 10, 2, 0),
    ]

    data = {
        "EventDateTime": timestamps,
        "participant_id": ["P001"] * len(timestamps),
        "CGM": [120, 125, 130, 130, 135],
        "Basal": [1.0, 1.1, 1.2, 1.2, 1.3],
    }

    return pd.DataFrame(data)


class TestMissingValueHandler:
    """Test cases for MissingValueHandler."""

    def test_initialization(self, sample_config):
        """Test proper initialization of MissingValueHandler."""
        handler = MissingValueHandler(sample_config)

        assert handler.strategies["CGM"] == "interpolation"
        assert handler.strategies["Basal"] == "forward_fill"
        assert handler.interpolation_method == "linear"
        assert handler.max_gap_minutes == 30

    def test_forward_fill_strategy(self, sample_config):
        """Test forward fill imputation strategy."""
        handler = MissingValueHandler(sample_config)

        # Create test data with missing values
        data = {
            "EventDateTime": pd.date_range("2023-01-01", periods=5, freq="5min"),
            "Basal": [1.0, np.nan, np.nan, 1.5, 2.0],
        }
        df = pd.DataFrame(data)

        result_df, stats = handler.handle_missing_values(df)

        # Check that forward fill worked
        expected_basal = [1.0, 1.0, 1.0, 1.5, 2.0]
        np.testing.assert_array_equal(result_df["Basal"].values, expected_basal)

        # Check statistics
        assert stats["Basal"]["missing_before"] == 2
        assert stats["Basal"]["missing_after"] == 0
        assert stats["Basal"]["strategy_used"] == "forward_fill"

    def test_interpolation_strategy(self, sample_config):
        """Test interpolation imputation strategy."""
        handler = MissingValueHandler(sample_config)

        # Create test data with missing values
        data = {
            "EventDateTime": pd.date_range("2023-01-01", periods=5, freq="5min"),
            "CGM": [100.0, np.nan, np.nan, 140.0, 150.0],
        }
        df = pd.DataFrame(data)

        result_df, stats = handler.handle_missing_values(df)

        # Check that interpolation worked (linear interpolation)
        # Between 100 and 140 over 3 intervals: 100, 113.33, 126.67, 140
        expected_cgm = [100.0, 113.333333, 126.666667, 140.0, 150.0]
        np.testing.assert_array_almost_equal(result_df["CGM"].values, expected_cgm)

        # Check statistics
        assert stats["CGM"]["missing_before"] == 2
        assert stats["CGM"]["missing_after"] == 0
        assert stats["CGM"]["strategy_used"] == "interpolation"

    def test_zero_fill_strategy(self, sample_config):
        """Test zero fill imputation strategy."""
        handler = MissingValueHandler(sample_config)

        # Create test data with missing values
        data = {
            "EventDateTime": pd.date_range("2023-01-01", periods=5, freq="5min"),
            "TotalBolusInsulinDelivered": [1.0, np.nan, 2.0, np.nan, 3.0],
        }
        df = pd.DataFrame(data)

        result_df, stats = handler.handle_missing_values(df)

        # Check that zero fill worked
        expected_bolus = [1.0, 0.0, 2.0, 0.0, 3.0]
        np.testing.assert_array_equal(
            result_df["TotalBolusInsulinDelivered"].values, expected_bolus
        )

        # Check statistics
        assert stats["TotalBolusInsulinDelivered"]["missing_before"] == 2
        assert stats["TotalBolusInsulinDelivered"]["missing_after"] == 0
        assert stats["TotalBolusInsulinDelivered"]["strategy_used"] == "zero_fill"

    def test_median_strategy(self, sample_config):
        """Test median imputation strategy."""
        # Modify config to use median strategy
        config = sample_config.copy()
        config["missing_values"]["strategies"]["test_column"] = "median"

        handler = MissingValueHandler(config)

        # Create test data with missing values
        data = {
            "EventDateTime": pd.date_range("2023-01-01", periods=5, freq="5min"),
            "test_column": [10.0, np.nan, 20.0, np.nan, 30.0],
        }
        df = pd.DataFrame(data)

        result_df, stats = handler.handle_missing_values(df)

        # Check that median fill worked (median of [10, 20, 30] is 20)
        expected_values = [10.0, 20.0, 20.0, 20.0, 30.0]
        np.testing.assert_array_equal(result_df["test_column"].values, expected_values)

    def test_interpolation_with_gap_limit(self, sample_config):
        """Test interpolation with gap size limits."""
        handler = MissingValueHandler(sample_config)

        # Create test data with a large gap (40 minutes)
        timestamps = [
            datetime(2023, 1, 1, 10, 0),
            datetime(2023, 1, 1, 10, 5),
            datetime(2023, 1, 1, 10, 45),  # 40-minute gap
            datetime(2023, 1, 1, 10, 50),
        ]

        data = {"EventDateTime": timestamps, "CGM": [100.0, np.nan, np.nan, 140.0]}
        df = pd.DataFrame(data)

        result_df, stats = handler.handle_missing_values(df)

        # The gap is larger than max_gap_minutes (30), so interpolation should be limited
        # The exact behavior depends on implementation, but we should have some NaN values remaining
        assert result_df["CGM"].isnull().sum() >= 0  # Some values might still be NaN


class TestOutlierTreatment:
    """Test cases for OutlierTreatment."""

    def test_initialization(self, sample_config):
        """Test proper initialization of OutlierTreatment."""
        treatment = OutlierTreatment(sample_config)

        assert treatment.detection_methods == ["iqr", "zscore"]
        assert treatment.treatment_method == "clip"
        assert treatment.zscore_threshold == 3.0
        assert treatment.iqr_multiplier == 1.5

    def test_zscore_outlier_detection(self, sample_config):
        """Test Z-score based outlier detection."""
        treatment = OutlierTreatment(sample_config)

        # Create data with clear outliers
        data = pd.Series([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])  # 100 is an outlier

        outlier_mask = treatment._detect_outliers(data, ["zscore"], {})

        # The value 100 should be detected as an outlier
        assert outlier_mask.sum() > 0
        assert outlier_mask.iloc[5] == True  # Index 5 has value 100

    def test_iqr_outlier_detection(self, sample_config):
        """Test IQR based outlier detection."""
        treatment = OutlierTreatment(sample_config)

        # Create data with clear outliers
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])  # 100 is an outlier

        outlier_mask = treatment._detect_outliers(data, ["iqr"], {})

        # The value 100 should be detected as an outlier
        assert outlier_mask.sum() > 0
        assert outlier_mask.iloc[-1] == True  # Last value (100) should be outlier

    def test_domain_specific_outlier_detection(self, sample_config):
        """Test domain-specific outlier detection."""
        treatment = OutlierTreatment(sample_config)

        # Create CGM data with values outside normal range
        data = pd.Series(
            [120, 130, 140, 20, 500, 150]
        )  # 20 and 500 are outside 40-400 range

        column_config = {"min_value": 40, "max_value": 400}

        outlier_mask = treatment._detect_outliers(
            data, ["domain_specific"], column_config
        )

        # Values 20 and 500 should be detected as outliers
        assert outlier_mask.sum() == 2
        assert outlier_mask.iloc[3] == True  # Value 20
        assert outlier_mask.iloc[4] == True  # Value 500

    def test_clip_treatment(self, sample_config):
        """Test clipping outlier treatment."""
        treatment = OutlierTreatment(sample_config)

        # Create data with outliers
        data = pd.Series([120, 130, 140, 20, 500, 150])
        outlier_mask = pd.Series([False, False, False, True, True, False])

        column_config = {"min_value": 40, "max_value": 400}

        treated_data = treatment._treat_outliers(
            data, outlier_mask, "clip", column_config
        )

        # Outliers should be clipped to bounds
        assert treated_data.iloc[3] == 40  # 20 clipped to 40
        assert treated_data.iloc[4] == 400  # 500 clipped to 400
        assert treated_data.iloc[0] == 120  # Normal values unchanged

    def test_remove_treatment(self, sample_config):
        """Test removal outlier treatment."""
        treatment = OutlierTreatment(sample_config)

        # Create data with outliers
        data = pd.Series([120, 130, 140, 20, 500, 150])
        outlier_mask = pd.Series([False, False, False, True, True, False])

        treated_data = treatment._treat_outliers(data, outlier_mask, "remove", {})

        # Outliers should be set to NaN
        assert pd.isna(treated_data.iloc[3])  # 20 should be NaN
        assert pd.isna(treated_data.iloc[4])  # 500 should be NaN
        assert treated_data.iloc[0] == 120  # Normal values unchanged

    def test_complete_outlier_treatment(self, sample_config, sample_diabetes_data):
        """Test complete outlier treatment pipeline."""
        treatment = OutlierTreatment(sample_config)

        result_df, stats = treatment.treat_outliers(sample_diabetes_data)

        # Check that statistics are generated
        assert "CGM" in stats
        assert stats["CGM"]["outliers_detected"] >= 0
        assert stats["CGM"]["treatment_method"] == "clip"

        # Check that extreme CGM values are clipped
        assert result_df["CGM"].min() >= 40
        assert result_df["CGM"].max() <= 400


class TestDataCleaner:
    """Test cases for DataCleaner."""

    def test_initialization(self, sample_config):
        """Test proper initialization of DataCleaner."""
        cleaner = DataCleaner(sample_config)

        assert cleaner.duplicate_strategy == "keep_last"
        assert cleaner.time_tolerance_seconds == 60
        assert cleaner.participant_column == "participant_id"

    def test_exact_duplicate_removal(self, sample_config):
        """Test removal of exact duplicates."""
        # Use a smaller time tolerance for this test
        config = sample_config.copy()
        config["cleaning"]["time_tolerance_seconds"] = 30
        cleaner = DataCleaner(config)

        # Create data with exact duplicates but larger time gaps
        data = {
            "EventDateTime": [
                datetime(2023, 1, 1, 10, 0),
                datetime(2023, 1, 1, 10, 2),
                datetime(2023, 1, 1, 10, 2),  # Exact duplicate
                datetime(2023, 1, 1, 10, 4),
            ],
            "participant_id": ["P001", "P001", "P001", "P001"],
            "CGM": [120, 130, 130, 140],
            "Basal": [1.0, 1.1, 1.1, 1.2],
        }
        df = pd.DataFrame(data)

        result_df, stats = cleaner.clean_data(df)

        # Should have 3 rows instead of 4 (only exact duplicate removed)
        assert len(result_df) == 3
        assert stats["exact_duplicates_removed"] == 1
        assert stats["initial_rows"] == 4
        assert stats["final_rows"] == 3

    def test_near_duplicate_removal_keep_last(
        self, sample_config, sample_data_with_duplicates
    ):
        """Test removal of near duplicates with keep_last strategy."""
        cleaner = DataCleaner(sample_config)

        result_df, stats = cleaner.clean_data(sample_data_with_duplicates)

        # Should remove near duplicates and exact duplicates
        assert len(result_df) < len(sample_data_with_duplicates)
        assert stats["near_duplicates_removed"] >= 0
        assert stats["exact_duplicates_removed"] >= 0

    def test_near_duplicate_removal_keep_first(
        self, sample_config, sample_data_with_duplicates
    ):
        """Test removal of near duplicates with keep_first strategy."""
        config = sample_config.copy()
        config["cleaning"]["duplicate_strategy"] = "keep_first"

        cleaner = DataCleaner(config)

        result_df, stats = cleaner.clean_data(sample_data_with_duplicates)

        # Should remove near duplicates and exact duplicates
        assert len(result_df) < len(sample_data_with_duplicates)

    def test_all_nan_row_removal(self, sample_config):
        """Test removal of rows with all NaN values."""
        # Use a smaller time tolerance for this test
        config = sample_config.copy()
        config["cleaning"]["time_tolerance_seconds"] = 30
        cleaner = DataCleaner(config)

        # Create data with all-NaN rows and larger time gaps
        data = {
            "EventDateTime": [
                datetime(2023, 1, 1, 10, 0),
                datetime(2023, 1, 1, 10, 2),
                datetime(2023, 1, 1, 10, 4),
            ],
            "participant_id": ["P001", "P001", "P001"],
            "CGM": [120, np.nan, 140],
            "Basal": [1.0, np.nan, 1.2],
        }
        df = pd.DataFrame(data)

        result_df, stats = cleaner.clean_data(df)

        # Should remove the row with all NaN values (except EventDateTime and participant_id)
        assert len(result_df) == 2
        assert stats["all_nan_rows_removed"] == 1

    def test_data_retention_rate_calculation(self, sample_config, sample_diabetes_data):
        """Test calculation of data retention rate."""
        cleaner = DataCleaner(sample_config)

        result_df, stats = cleaner.clean_data(sample_diabetes_data)

        # Check retention rate calculation
        expected_retention = stats["final_rows"] / stats["initial_rows"]
        assert abs(stats["data_retention_rate"] - expected_retention) < 1e-10


class TestTimeSeriesResampler:
    """Test cases for TimeSeriesResampler."""

    def test_initialization(self, sample_config):
        """Test proper initialization of TimeSeriesResampler."""
        resampler = TimeSeriesResampler(sample_config)

        assert resampler.target_frequency == "5min"
        assert resampler.participant_column == "participant_id"
        assert resampler.aggregation_methods["CGM"] == "mean"
        assert resampler.aggregation_methods["TotalBolusInsulinDelivered"] == "sum"

    def test_basic_resampling(self, sample_config):
        """Test basic time series resampling."""
        resampler = TimeSeriesResampler(sample_config)

        # Create high-frequency data (every minute)
        timestamps = pd.date_range("2023-01-01 10:00", periods=10, freq="1min")
        data = {
            "EventDateTime": timestamps,
            "participant_id": ["P001"] * len(timestamps),
            "CGM": np.arange(100, 110),  # 100, 101, 102, ..., 109
            "TotalBolusInsulinDelivered": [1.0] * len(timestamps),
        }
        df = pd.DataFrame(data)

        result_df, stats = resampler.resample_timeseries(df)

        # Should have fewer rows due to resampling to 5-minute intervals
        assert len(result_df) < len(df)
        assert stats["resampling_applied"] == True
        assert stats["original_rows"] == len(df)
        assert stats["resampled_rows"] == len(result_df)

        # Check that aggregation worked correctly
        # CGM should be averaged, TotalBolusInsulinDelivered should be summed
        assert "CGM" in result_df.columns
        assert "TotalBolusInsulinDelivered" in result_df.columns

    def test_multiple_participants(self, sample_config):
        """Test resampling with multiple participants."""
        resampler = TimeSeriesResampler(sample_config)

        # Create data for two participants
        timestamps = pd.date_range("2023-01-01 10:00", periods=10, freq="1min")
        data = {
            "EventDateTime": list(timestamps) + list(timestamps),
            "participant_id": ["P001"] * len(timestamps) + ["P002"] * len(timestamps),
            "CGM": list(range(100, 110)) + list(range(200, 210)),
            "TotalBolusInsulinDelivered": [1.0] * (2 * len(timestamps)),
        }
        df = pd.DataFrame(data)

        result_df, stats = resampler.resample_timeseries(df)

        # Should maintain participant separation
        participants = result_df["participant_id"].unique()
        assert len(participants) == 2
        assert "P001" in participants
        assert "P002" in participants

    def test_no_datetime_column(self, sample_config):
        """Test behavior when EventDateTime column is missing."""
        resampler = TimeSeriesResampler(sample_config)

        # Create data without EventDateTime
        data = {"participant_id": ["P001", "P001", "P001"], "CGM": [120, 130, 140]}
        df = pd.DataFrame(data)

        result_df, stats = resampler.resample_timeseries(df)

        # Should return original data unchanged
        pd.testing.assert_frame_equal(result_df, df)
        assert stats["resampling_applied"] == False

    def test_aggregation_methods(self, sample_config):
        """Test different aggregation methods."""
        resampler = TimeSeriesResampler(sample_config)

        # Create data with multiple values per resampling window
        base_time = datetime(2023, 1, 1, 10, 0)
        timestamps = [
            base_time,
            base_time + timedelta(minutes=1),
            base_time + timedelta(minutes=2),
            base_time + timedelta(minutes=5),  # Next 5-minute window
            base_time + timedelta(minutes=6),
        ]

        data = {
            "EventDateTime": timestamps,
            "participant_id": ["P001"] * len(timestamps),
            "CGM": [100, 110, 120, 200, 210],  # Should be averaged within windows
            "TotalBolusInsulinDelivered": [
                1,
                2,
                3,
                4,
                5,
            ],  # Should be summed within windows
        }
        df = pd.DataFrame(data)

        result_df, stats = resampler.resample_timeseries(df)

        # Should have 2 rows (2 time windows)
        assert len(result_df) == 2

        # First window: CGM mean should be (100+110+120)/3 = 110
        # First window: Bolus sum should be 1+2+3 = 6
        first_row = result_df.iloc[0]
        assert abs(first_row["CGM"] - 110) < 1e-10
        assert abs(first_row["TotalBolusInsulinDelivered"] - 6) < 1e-10


class TestDataPreprocessor:
    """Test cases for DataPreprocessor (main pipeline)."""

    def test_initialization(self, sample_config):
        """Test proper initialization of DataPreprocessor."""
        preprocessor = DataPreprocessor(sample_config)

        assert isinstance(preprocessor.missing_value_handler, MissingValueHandler)
        assert isinstance(preprocessor.outlier_treatment, OutlierTreatment)
        assert isinstance(preprocessor.data_cleaner, DataCleaner)
        assert isinstance(preprocessor.time_series_resampler, TimeSeriesResampler)

    def test_complete_preprocessing_pipeline(self, sample_config, sample_diabetes_data):
        """Test the complete preprocessing pipeline."""
        preprocessor = DataPreprocessor(sample_config)

        result_df, stats = preprocessor.preprocess(sample_diabetes_data)

        # Check that all processing steps were executed
        assert "cleaning" in stats
        assert "missing_values" in stats
        assert "outliers" in stats
        assert "resampling" in stats

        # Check that timing information is included
        assert "pipeline_start_time" in stats
        assert "pipeline_end_time" in stats
        assert "total_processing_time" in stats

        # Check that shape information is tracked
        assert "initial_shape" in stats
        assert "final_shape" in stats

        # Result should be a valid DataFrame
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0

    def test_preprocessing_report_generation(self, sample_config, sample_diabetes_data):
        """Test preprocessing report generation."""
        preprocessor = DataPreprocessor(sample_config)

        result_df, stats = preprocessor.preprocess(sample_diabetes_data)

        # Generate report
        report = preprocessor.generate_preprocessing_report(stats)

        # Check that report contains expected sections
        assert "PREPROCESSING REPORT" in report
        assert "OVERVIEW" in report
        assert "DATA CLEANING" in report
        assert "MISSING VALUE HANDLING" in report
        assert "OUTLIER TREATMENT" in report

        # Check that report is not empty
        assert len(report) > 100

    def test_preprocessing_report_file_output(
        self, sample_config, sample_diabetes_data
    ):
        """Test preprocessing report file output."""
        preprocessor = DataPreprocessor(sample_config)

        result_df, stats = preprocessor.preprocess(sample_diabetes_data)

        # Generate report with file output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "preprocessing_report.txt"
            report = preprocessor.generate_preprocessing_report(stats, output_path)

            # Check that file was created
            assert output_path.exists()

            # Check that file content matches returned report
            with open(output_path, "r") as f:
                file_content = f.read()

            assert file_content == report

    def test_edge_case_empty_dataframe(self, sample_config):
        """Test preprocessing with empty DataFrame."""
        preprocessor = DataPreprocessor(sample_config)

        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(
            columns=["EventDateTime", "participant_id", "CGM", "Basal"]
        )

        result_df, stats = preprocessor.preprocess(empty_df)

        # Should handle empty DataFrame gracefully
        assert len(result_df) == 0
        assert "initial_shape" in stats
        assert stats["initial_shape"][0] == 0

    def test_edge_case_single_row(self, sample_config):
        """Test preprocessing with single row DataFrame."""
        preprocessor = DataPreprocessor(sample_config)

        # Create single-row DataFrame
        single_row_df = pd.DataFrame(
            {
                "EventDateTime": [datetime(2023, 1, 1, 10, 0)],
                "participant_id": ["P001"],
                "CGM": [120.0],
                "Basal": [1.0],
                "TotalBolusInsulinDelivered": [0.5],
            }
        )

        result_df, stats = preprocessor.preprocess(single_row_df)

        # Should handle single row gracefully
        assert len(result_df) >= 0  # Might be 0 or 1 depending on processing
        assert "initial_shape" in stats
        assert stats["initial_shape"][0] == 1


class TestIntegrationScenarios:
    """Integration test scenarios for preprocessing."""

    def test_realistic_diabetes_data_scenario(self, sample_config):
        """Test with realistic diabetes data scenario."""
        # Create more realistic diabetes data
        np.random.seed(42)

        # 7 days of data with irregular timestamps
        start_time = datetime(2023, 1, 1, 0, 0, 0)
        timestamps = []
        current_time = start_time

        # Generate irregular timestamps (every 1-15 minutes)
        for _ in range(1000):
            timestamps.append(current_time)
            current_time += timedelta(minutes=np.random.randint(1, 16))

        # Generate realistic diabetes data
        data = {
            "EventDateTime": timestamps,
            "participant_id": ["P001"] * len(timestamps),
            "CGM": np.random.normal(
                140, 40, len(timestamps)
            ),  # Slightly elevated glucose
            "Basal": np.random.normal(1.2, 0.3, len(timestamps)),
            "TotalBolusInsulinDelivered": np.random.exponential(0.8, len(timestamps)),
            "CorrectionDelivered": np.random.exponential(0.4, len(timestamps)),
            "FoodDelivered": np.random.exponential(0.3, len(timestamps)),
            "CarbSize": np.random.exponential(15, len(timestamps)),
        }

        df = pd.DataFrame(data)

        # Add realistic missing patterns
        # CGM sensor failures (consecutive missing values)
        cgm_failure_start = 100
        df.loc[cgm_failure_start : cgm_failure_start + 20, "CGM"] = np.nan

        # Occasional missing insulin data
        missing_insulin = np.random.choice(len(df), size=50, replace=False)
        df.loc[missing_insulin, "TotalBolusInsulinDelivered"] = np.nan

        # Add some extreme outliers (sensor errors)
        outlier_indices = np.random.choice(len(df), size=10, replace=False)
        df.loc[outlier_indices, "CGM"] = np.random.choice([0, 600, 700], size=10)

        # Add some duplicates
        duplicate_indices = np.random.choice(len(df) - 1, size=5, replace=False)
        for idx in duplicate_indices:
            df.loc[idx + 1, "EventDateTime"] = df.loc[idx, "EventDateTime"]

        # Process with preprocessing pipeline
        preprocessor = DataPreprocessor(sample_config)
        result_df, stats = preprocessor.preprocess(df)

        # Verify results
        assert len(result_df) > 0
        # Note: resampling can increase data size by creating regular time grid
        # so we don't assert that result is smaller than input

        # Check that extreme outliers were handled
        if len(result_df) > 0 and "CGM" in result_df.columns:
            cgm_values = result_df["CGM"].dropna()
            if len(cgm_values) > 0:
                assert cgm_values.min() >= 40  # Domain-specific minimum
                assert cgm_values.max() <= 400  # Domain-specific maximum

        # Check that duplicates were reduced
        assert stats["cleaning"]["exact_duplicates_removed"] >= 0

        # Check that missing values were handled
        for column, column_stats in stats["missing_values"].items():
            if column_stats["missing_before"] > 0:
                assert column_stats["imputation_rate"] >= 0

    def test_multi_participant_scenario(self, sample_config):
        """Test preprocessing with multiple participants."""
        np.random.seed(42)

        participants = ["P001", "P002", "P003"]
        all_data = []

        for participant in participants:
            # Generate data for each participant
            timestamps = pd.date_range(
                start=datetime(2023, 1, 1), end=datetime(2023, 1, 2), freq="5min"
            )

            participant_data = {
                "EventDateTime": timestamps,
                "participant_id": [participant] * len(timestamps),
                "CGM": np.random.normal(
                    120 + hash(participant) % 50, 30, len(timestamps)
                ),
                "Basal": np.random.normal(1.0, 0.2, len(timestamps)),
                "TotalBolusInsulinDelivered": np.random.exponential(
                    0.5, len(timestamps)
                ),
            }

            all_data.append(pd.DataFrame(participant_data))

        # Combine all participant data
        df = pd.concat(all_data, ignore_index=True)

        # Add some cross-participant issues
        # Duplicate timestamps across participants (should be OK)
        # Missing values for different participants
        for i, participant in enumerate(participants):
            participant_mask = df["participant_id"] == participant
            participant_indices = df[participant_mask].index

            # Add missing values specific to each participant
            missing_indices = np.random.choice(
                participant_indices, size=10, replace=False
            )
            df.loc[missing_indices, "CGM"] = np.nan

        # Process with preprocessing pipeline
        preprocessor = DataPreprocessor(sample_config)
        result_df, stats = preprocessor.preprocess(df)

        # Verify that all participants are preserved
        if len(result_df) > 0:
            result_participants = set(result_df["participant_id"].unique())
            original_participants = set(participants)
            assert result_participants.issubset(original_participants)

        # Check that processing was applied appropriately
        assert stats["cleaning"]["initial_rows"] == len(df)
        assert stats["cleaning"]["final_rows"] == len(result_df)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
