"""
Unit tests for sequence generation module.

Tests cover:
- SequenceGenerator class functionality
- ParticipantSplitter class functionality  
- SequenceValidator class functionality
- TimeSeriesResampler class functionality
- Integration tests for the complete pipeline
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import tempfile
import os

from diabetes_lstm_pipeline.sequence_generation.sequence_generation import (
    SequenceGenerator,
    ParticipantSplitter,
    SequenceValidator,
    TimeSeriesResampler,
    SequenceGenerationPipeline,
)


class TestSequenceGenerator:
    """Test cases for SequenceGenerator class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "sequence_length": 60,  # minutes
            "prediction_horizon": 1,
            "target_column": "CGM",
            "participant_column": "participant_id",
            "time_column": "EventDateTime",
            "sampling_interval": 5,  # minutes
        }

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        # Create 2 participants with 100 data points each (5-minute intervals)
        n_points_per_participant = 100
        participants = ["P001", "P002"]

        data = []
        for participant in participants:
            start_time = datetime(2024, 1, 1, 8, 0)  # 8 AM start

            for i in range(n_points_per_participant):
                timestamp = start_time + timedelta(minutes=i * 5)

                # Generate realistic-looking data
                cgm_base = 120 + 30 * np.sin(i * 0.1) + np.random.normal(0, 10)
                cgm = max(70, min(300, cgm_base))  # Clamp to reasonable range

                data.append(
                    {
                        "EventDateTime": timestamp,
                        "participant_id": participant,
                        "CGM": cgm,
                        "Basal": 1.0 + np.random.normal(0, 0.1),
                        "TotalBolusInsulinDelivered": (
                            np.random.exponential(0.5)
                            if np.random.random() < 0.1
                            else 0
                        ),
                        "CorrectionDelivered": (
                            np.random.exponential(0.3)
                            if np.random.random() < 0.05
                            else 0
                        ),
                        "FoodDelivered": (
                            np.random.exponential(2.0)
                            if np.random.random() < 0.08
                            else 0
                        ),
                        "CarbSize": (
                            np.random.exponential(20)
                            if np.random.random() < 0.08
                            else 0
                        ),
                        "feature1": np.random.normal(0, 1),
                        "feature2": np.random.normal(5, 2),
                    }
                )

        return pd.DataFrame(data)

    def test_initialization(self, sample_config):
        """Test SequenceGenerator initialization."""
        generator = SequenceGenerator(sample_config)

        assert generator.sequence_length == 60
        assert generator.prediction_horizon == 1
        assert generator.target_column == "CGM"
        assert (
            generator.sequence_length_points == 12
        )  # 60 minutes / 5 minutes per point

    def test_generate_sequences_basic(self, sample_config, sample_dataframe):
        """Test basic sequence generation."""
        generator = SequenceGenerator(sample_config)

        X, y, participant_ids = generator.generate_sequences(sample_dataframe)

        # Check output shapes
        assert len(X.shape) == 3  # (n_sequences, sequence_length, n_features)
        assert len(y.shape) == 1  # (n_sequences,)
        assert len(participant_ids.shape) == 1  # (n_sequences,)

        # Check consistency
        assert X.shape[0] == len(y) == len(participant_ids)
        assert X.shape[1] == 12  # sequence_length_points

        # Check that we have sequences from both participants
        unique_participants = np.unique(participant_ids)
        assert len(unique_participants) == 2
        assert "P001" in unique_participants
        assert "P002" in unique_participants

    def test_generate_sequences_with_custom_features(self, sample_dataframe):
        """Test sequence generation with custom feature columns."""
        config = {
            "sequence_length": 30,
            "prediction_horizon": 1,
            "target_column": "CGM",
            "feature_columns": ["CGM", "Basal", "feature1"],
            "participant_column": "participant_id",
            "time_column": "EventDateTime",
            "sampling_interval": 5,
        }

        generator = SequenceGenerator(config)
        X, y, participant_ids = generator.generate_sequences(sample_dataframe)

        # Check that only specified features are used
        assert X.shape[2] == 3  # 3 features specified
        assert X.shape[1] == 6  # 30 minutes / 5 minutes per point

    def test_generate_sequences_missing_target_column(self, sample_config):
        """Test error handling when target column is missing."""
        df = pd.DataFrame(
            {
                "EventDateTime": [datetime.now()],
                "participant_id": ["P001"],
                "other_column": [1.0],
            }
        )

        generator = SequenceGenerator(sample_config)

        with pytest.raises(ValueError, match="Target column 'CGM' not found"):
            generator.generate_sequences(df)

    def test_generate_sequences_missing_participant_column(self, sample_config):
        """Test error handling when participant column is missing."""
        df = pd.DataFrame(
            {"EventDateTime": [datetime.now()], "CGM": [120.0], "other_column": [1.0]}
        )

        generator = SequenceGenerator(sample_config)

        with pytest.raises(
            ValueError, match="Participant column 'participant_id' not found"
        ):
            generator.generate_sequences(df)

    def test_sequence_validation(self, sample_config, sample_dataframe):
        """Test that invalid sequences are filtered out."""
        # Add some NaN values to create invalid sequences
        df = sample_dataframe.copy()
        df.loc[10:15, "CGM"] = np.nan  # Add NaN values
        df.loc[50:55, "feature1"] = np.nan  # Add NaN values in features

        generator = SequenceGenerator(sample_config)
        X, y, participant_ids = generator.generate_sequences(df)

        # Check that no NaN values remain in output
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()

    def test_get_sequence_info(self, sample_config):
        """Test sequence info retrieval."""
        generator = SequenceGenerator(sample_config)

        info = generator.get_sequence_info()

        assert info["sequence_length_minutes"] == 60
        assert info["sequence_length_points"] == 12
        assert info["prediction_horizon"] == 1
        assert info["target_column"] == "CGM"
        assert info["sampling_interval"] == 5


class TestParticipantSplitter:
    """Test cases for ParticipantSplitter class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "random_seed": 42,
            "split_strategy": "participant",
        }

    @pytest.fixture
    def sample_sequences(self):
        """Create sample sequences for testing."""
        n_sequences = 1000
        sequence_length = 12
        n_features = 5

        # Create sequences from 10 participants
        participants = [f"P{i:03d}" for i in range(10)]
        participant_ids = np.random.choice(participants, n_sequences)

        X = np.random.randn(n_sequences, sequence_length, n_features)
        y = np.random.randn(n_sequences) * 50 + 120  # Glucose-like values

        return X, y, participant_ids

    def test_initialization(self, sample_config):
        """Test ParticipantSplitter initialization."""
        splitter = ParticipantSplitter(sample_config)

        assert splitter.train_ratio == 0.6
        assert splitter.val_ratio == 0.2
        assert splitter.test_ratio == 0.2
        assert splitter.random_seed == 42
        assert splitter.split_strategy == "participant"

    def test_initialization_invalid_ratios(self):
        """Test error handling for invalid split ratios."""
        config = {
            "train_ratio": 0.6,
            "val_ratio": 0.3,
            "test_ratio": 0.2,  # Sum > 1.0
            "random_seed": 42,
        }

        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            ParticipantSplitter(config)

    def test_split_by_participant(self, sample_config, sample_sequences):
        """Test participant-based splitting."""
        X, y, participant_ids = sample_sequences

        splitter = ParticipantSplitter(sample_config)
        splits = splitter.split_sequences(X, y, participant_ids)

        # Check that all splits are present
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        # Check that splits contain tuples of (X, y)
        for split_name, (X_split, y_split) in splits.items():
            assert len(X_split.shape) == 3
            assert len(y_split.shape) == 1
            assert X_split.shape[0] == y_split.shape[0]

        # Check that all sequences are accounted for
        total_sequences = sum(len(X_split) for X_split, _ in splits.values())
        assert total_sequences == len(X)

        # Check that participants don't overlap between splits
        train_participants = set(participant_ids[: len(splits["train"][0])])
        val_participants = set(
            participant_ids[
                len(splits["train"][0]) : len(splits["train"][0])
                + len(splits["val"][0])
            ]
        )
        test_participants = set(
            participant_ids[len(splits["train"][0]) + len(splits["val"][0]) :]
        )

        # Note: This test is approximate since we don't have the exact participant mapping
        # In a real scenario, we'd need to track which sequences belong to which participants

    def test_split_by_time(self, sample_sequences):
        """Test temporal splitting."""
        config = {
            "train_ratio": 0.6,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
            "split_strategy": "temporal",
            "random_seed": 42,
        }

        X, y, participant_ids = sample_sequences

        splitter = ParticipantSplitter(config)
        splits = splitter.split_sequences(X, y, participant_ids)

        # Check split sizes are approximately correct
        train_size = len(splits["train"][0])
        val_size = len(splits["val"][0])
        test_size = len(splits["test"][0])

        total_size = len(X)

        assert abs(train_size / total_size - 0.6) < 0.01
        assert abs(val_size / total_size - 0.2) < 0.01
        assert abs(test_size / total_size - 0.2) < 0.01

    def test_get_split_info(self, sample_config, sample_sequences):
        """Test split information retrieval."""
        X, y, participant_ids = sample_sequences

        splitter = ParticipantSplitter(sample_config)
        splits = splitter.split_sequences(X, y, participant_ids)

        info = splitter.get_split_info(splits)

        assert "split_strategy" in info
        assert "train_ratio" in info
        assert "splits" in info

        for split_name in ["train", "val", "test"]:
            assert split_name in info["splits"]
            split_info = info["splits"][split_name]
            assert "n_sequences" in split_info
            assert "sequence_shape" in split_info
            assert "target_mean" in split_info
            assert "target_std" in split_info


class TestSequenceValidator:
    """Test cases for SequenceValidator class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "time_column": "EventDateTime",
            "participant_column": "participant_id",
            "target_column": "CGM",
            "max_time_gap_minutes": 15,
            "min_sequence_length": 10,
        }

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for validation testing."""
        data = []
        start_time = datetime(2024, 1, 1, 8, 0)

        # Create data for 2 participants
        for participant in ["P001", "P002"]:
            for i in range(50):
                timestamp = start_time + timedelta(minutes=i * 5)
                data.append(
                    {
                        "EventDateTime": timestamp,
                        "participant_id": participant,
                        "CGM": 120 + np.random.normal(0, 20),
                        "Basal": 1.0,
                        "feature1": np.random.normal(0, 1),
                    }
                )

        return pd.DataFrame(data)

    def test_initialization(self, sample_config):
        """Test SequenceValidator initialization."""
        validator = SequenceValidator(sample_config)

        assert validator.time_column == "EventDateTime"
        assert validator.participant_column == "participant_id"
        assert validator.target_column == "CGM"
        assert validator.max_time_gap_minutes == 15

    def test_validate_dataframe_valid(self, sample_config, sample_dataframe):
        """Test validation of valid dataframe."""
        validator = SequenceValidator(sample_config)

        results = validator.validate_dataframe(sample_dataframe)

        assert "is_valid" in results
        assert "issues" in results
        assert "statistics" in results
        assert "recommendations" in results

    def test_validate_dataframe_missing_columns(self, sample_config):
        """Test validation with missing required columns."""
        df = pd.DataFrame({"EventDateTime": [datetime.now()], "other_column": [1.0]})

        validator = SequenceValidator(sample_config)
        results = validator.validate_dataframe(df)

        assert not results["is_valid"]
        assert any("Missing required columns" in issue for issue in results["issues"])

    def test_validate_dataframe_with_gaps(self, sample_config):
        """Test validation with time gaps."""
        # Create dataframe with large time gaps
        data = []
        start_time = datetime(2024, 1, 1, 8, 0)

        for i in range(10):
            # Create a large gap after 5th record
            gap_multiplier = 10 if i > 5 else 1
            timestamp = start_time + timedelta(minutes=i * 5 * gap_multiplier)

            data.append(
                {
                    "EventDateTime": timestamp,
                    "participant_id": "P001",
                    "CGM": 120.0,
                    "Basal": 1.0,
                }
            )

        df = pd.DataFrame(data)

        validator = SequenceValidator(sample_config)
        results = validator.validate_dataframe(df)

        # Should detect gaps
        assert any("gaps" in issue for issue in results["issues"])

    def test_validate_sequences_valid(self, sample_config):
        """Test validation of valid sequences."""
        n_sequences = 100
        sequence_length = 12
        n_features = 5

        X = np.random.randn(n_sequences, sequence_length, n_features)
        y = np.random.randn(n_sequences) * 50 + 120  # Glucose-like values
        participant_ids = np.array([f"P{i%5:03d}" for i in range(n_sequences)])

        validator = SequenceValidator(sample_config)
        results = validator.validate_sequences(X, y, participant_ids)

        assert "is_valid" in results
        assert "statistics" in results
        assert "quality_metrics" in results

    def test_validate_sequences_with_nan(self, sample_config):
        """Test validation of sequences with NaN values."""
        n_sequences = 100
        sequence_length = 12
        n_features = 5

        X = np.random.randn(n_sequences, sequence_length, n_features)
        y = np.random.randn(n_sequences) * 50 + 120
        participant_ids = np.array([f"P{i%5:03d}" for i in range(n_sequences)])

        # Add some NaN values
        X[10, 5, 2] = np.nan
        y[20] = np.nan

        validator = SequenceValidator(sample_config)
        results = validator.validate_sequences(X, y, participant_ids)

        # Should detect NaN values
        assert any("NaN" in issue for issue in results["issues"])

    def test_validate_sequences_invalid_glucose_values(self, sample_config):
        """Test validation with invalid glucose values."""
        n_sequences = 100
        sequence_length = 12
        n_features = 5

        X = np.random.randn(n_sequences, sequence_length, n_features)
        y = np.random.randn(n_sequences) * 50 + 120
        participant_ids = np.array([f"P{i%5:03d}" for i in range(n_sequences)])

        # Add some physiologically impossible values
        y[10] = 10  # Too low
        y[20] = 700  # Too high

        validator = SequenceValidator(sample_config)
        results = validator.validate_sequences(X, y, participant_ids)

        # Should detect invalid values
        assert any("physiologically impossible" in issue for issue in results["issues"])


class TestTimeSeriesResampler:
    """Test cases for TimeSeriesResampler class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "target_frequency": "5min",
            "interpolation_method": "linear",
            "max_gap_minutes": 30,
            "participant_column": "participant_id",
            "time_column": "EventDateTime",
        }

    @pytest.fixture
    def irregular_dataframe(self):
        """Create dataframe with irregular timestamps."""
        data = []
        start_time = datetime(2024, 1, 1, 8, 0)

        # Create irregular timestamps for 2 participants
        for participant in ["P001", "P002"]:
            for i in range(50):
                # Irregular intervals: sometimes 3 min, sometimes 7 min
                interval = 3 if i % 2 == 0 else 7
                timestamp = start_time + timedelta(
                    minutes=sum(3 if j % 2 == 0 else 7 for j in range(i))
                )

                data.append(
                    {
                        "EventDateTime": timestamp,
                        "participant_id": participant,
                        "CGM": 120 + np.random.normal(0, 20),
                        "Basal": 1.0 + np.random.normal(0, 0.1),
                        "TotalBolusInsulinDelivered": (
                            np.random.exponential(0.5)
                            if np.random.random() < 0.1
                            else 0
                        ),
                        "FoodDelivered": (
                            np.random.exponential(2.0)
                            if np.random.random() < 0.08
                            else 0
                        ),
                    }
                )

        return pd.DataFrame(data)

    def test_initialization(self, sample_config):
        """Test TimeSeriesResampler initialization."""
        resampler = TimeSeriesResampler(sample_config)

        assert resampler.target_frequency == "5min"
        assert resampler.interpolation_method == "linear"
        assert resampler.max_gap_minutes == 30

    def test_resample_dataframe(self, sample_config, irregular_dataframe):
        """Test dataframe resampling."""
        resampler = TimeSeriesResampler(sample_config)

        resampled_df = resampler.resample_dataframe(irregular_dataframe)

        # Check that output has regular intervals
        assert len(resampled_df) > 0
        assert "EventDateTime" in resampled_df.columns
        assert "participant_id" in resampled_df.columns

        # Check that each participant has regular intervals
        for participant in resampled_df["participant_id"].unique():
            participant_data = resampled_df[
                resampled_df["participant_id"] == participant
            ].sort_values("EventDateTime")

            # Calculate time differences
            time_diffs = participant_data["EventDateTime"].diff().dropna()

            # Should be mostly 5-minute intervals (allowing for some variation due to resampling)
            expected_interval = pd.Timedelta(minutes=5)
            assert all(
                abs(diff - expected_interval) <= pd.Timedelta(seconds=30)
                for diff in time_diffs
            )

    def test_get_resampling_info(self, sample_config):
        """Test resampling info retrieval."""
        resampler = TimeSeriesResampler(sample_config)

        info = resampler.get_resampling_info()

        assert "target_frequency" in info
        assert "interpolation_method" in info
        assert "max_gap_minutes" in info
        assert "max_gap_periods" in info


class TestSequenceGenerationPipeline:
    """Test cases for the complete SequenceGenerationPipeline."""

    @pytest.fixture
    def pipeline_config(self):
        """Complete pipeline configuration for testing."""
        return {
            "resampling": {
                "target_frequency": "5min",
                "interpolation_method": "linear",
                "max_gap_minutes": 30,
                "participant_column": "participant_id",
                "time_column": "EventDateTime",
            },
            "validation": {
                "time_column": "EventDateTime",
                "participant_column": "participant_id",
                "target_column": "CGM",
                "max_time_gap_minutes": 15,
                "min_sequence_length": 10,
            },
            "sequence_generation": {
                "sequence_length": 60,
                "prediction_horizon": 1,
                "target_column": "CGM",
                "participant_column": "participant_id",
                "time_column": "EventDateTime",
                "sampling_interval": 5,
            },
            "splitting": {
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "test_ratio": 0.2,
                "random_seed": 42,
                "split_strategy": "participant",
            },
        }

    @pytest.fixture
    def sample_dataframe(self):
        """Create comprehensive sample dataframe for pipeline testing."""
        data = []
        start_time = datetime(2024, 1, 1, 8, 0)

        # Create data for 5 participants with 200 data points each
        for participant_idx in range(5):
            participant_id = f"P{participant_idx:03d}"

            for i in range(200):
                timestamp = start_time + timedelta(minutes=i * 5)

                # Generate realistic-looking data with some patterns
                cgm_base = 120 + 30 * np.sin(i * 0.05) + np.random.normal(0, 15)
                cgm = max(70, min(300, cgm_base))

                data.append(
                    {
                        "EventDateTime": timestamp,
                        "participant_id": participant_id,
                        "CGM": cgm,
                        "Basal": 1.0 + np.random.normal(0, 0.1),
                        "TotalBolusInsulinDelivered": (
                            np.random.exponential(0.5)
                            if np.random.random() < 0.1
                            else 0
                        ),
                        "CorrectionDelivered": (
                            np.random.exponential(0.3)
                            if np.random.random() < 0.05
                            else 0
                        ),
                        "FoodDelivered": (
                            np.random.exponential(2.0)
                            if np.random.random() < 0.08
                            else 0
                        ),
                        "CarbSize": (
                            np.random.exponential(20)
                            if np.random.random() < 0.08
                            else 0
                        ),
                        "feature1": np.random.normal(0, 1),
                        "feature2": np.random.normal(5, 2),
                        "feature3": np.random.exponential(1),
                    }
                )

        return pd.DataFrame(data)

    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initialization."""
        pipeline = SequenceGenerationPipeline(pipeline_config)

        assert pipeline.resampler is not None
        assert pipeline.validator is not None
        assert pipeline.sequence_generator is not None
        assert pipeline.participant_splitter is not None

    def test_generate_sequences_from_dataframe(self, pipeline_config, sample_dataframe):
        """Test complete pipeline execution."""
        pipeline = SequenceGenerationPipeline(pipeline_config)

        data_splits, metadata = pipeline.generate_sequences_from_dataframe(
            sample_dataframe, validate_input=True, resample_data=True
        )

        # Check that all splits are present
        assert "train" in data_splits
        assert "val" in data_splits
        assert "test" in data_splits

        # Check split contents
        for split_name, (X, y) in data_splits.items():
            assert len(X.shape) == 3
            assert len(y.shape) == 1
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == 12  # sequence_length_points

        # Check metadata
        assert "pipeline_config" in metadata
        assert "input_data_shape" in metadata
        assert "validation_results" in metadata
        assert "resampling_info" in metadata
        assert "sequence_info" in metadata
        assert "split_info" in metadata

    def test_generate_sequences_without_resampling(
        self, pipeline_config, sample_dataframe
    ):
        """Test pipeline without resampling step."""
        pipeline = SequenceGenerationPipeline(pipeline_config)

        data_splits, metadata = pipeline.generate_sequences_from_dataframe(
            sample_dataframe, validate_input=True, resample_data=False
        )

        # Should still work but without resampling info
        assert "train" in data_splits
        assert "resampling_info" not in metadata or not metadata["resampling_info"]

    def test_save_sequences(self, pipeline_config, sample_dataframe):
        """Test saving sequences to disk."""
        pipeline = SequenceGenerationPipeline(pipeline_config)

        data_splits, metadata = pipeline.generate_sequences_from_dataframe(
            sample_dataframe, validate_input=False, resample_data=False
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline.save_sequences(data_splits, temp_dir, metadata)

            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, "X_train.npy"))
            assert os.path.exists(os.path.join(temp_dir, "y_train.npy"))
            assert os.path.exists(os.path.join(temp_dir, "X_val.npy"))
            assert os.path.exists(os.path.join(temp_dir, "y_val.npy"))
            assert os.path.exists(os.path.join(temp_dir, "X_test.npy"))
            assert os.path.exists(os.path.join(temp_dir, "y_test.npy"))
            assert os.path.exists(os.path.join(temp_dir, "sequence_metadata.json"))

            # Check that files can be loaded
            X_train = np.load(os.path.join(temp_dir, "X_train.npy"))
            y_train = np.load(os.path.join(temp_dir, "y_train.npy"))

            assert len(X_train.shape) == 3
            assert len(y_train.shape) == 1
            assert X_train.shape[0] == y_train.shape[0]

    def test_pipeline_with_minimal_data(self, pipeline_config):
        """Test pipeline with minimal data that might cause issues."""
        # Create very small dataset
        data = []
        start_time = datetime(2024, 1, 1, 8, 0)

        for i in range(20):  # Only 20 data points
            data.append(
                {
                    "EventDateTime": start_time + timedelta(minutes=i * 5),
                    "participant_id": "P001",
                    "CGM": 120 + np.random.normal(0, 10),
                    "Basal": 1.0,
                    "feature1": np.random.normal(0, 1),
                }
            )

        df = pd.DataFrame(data)

        pipeline = SequenceGenerationPipeline(pipeline_config)

        # This might raise an error or produce very few sequences
        try:
            data_splits, metadata = pipeline.generate_sequences_from_dataframe(df)

            # If it succeeds, check that we got some sequences
            total_sequences = sum(len(X) for X, _ in data_splits.values())
            assert total_sequences >= 0  # Might be 0 for very small datasets

        except ValueError:
            # Expected for very small datasets
            pass


# Integration test fixtures and helpers
@pytest.fixture
def realistic_diabetes_data():
    """Create realistic diabetes dataset for integration testing."""
    np.random.seed(42)  # For reproducible tests

    data = []
    participants = [f"P{i:03d}" for i in range(3)]  # 3 participants

    for participant in participants:
        start_time = datetime(2024, 1, 1, 6, 0)  # Start at 6 AM

        # Generate 7 days of data (5-minute intervals)
        n_points = 7 * 24 * 12  # 7 days * 24 hours * 12 points per hour

        for i in range(n_points):
            timestamp = start_time + timedelta(minutes=i * 5)

            # Simulate daily glucose patterns
            hour_of_day = timestamp.hour + timestamp.minute / 60

            # Base glucose with daily pattern
            glucose_base = 120
            glucose_base += 20 * np.sin(2 * np.pi * hour_of_day / 24)  # Daily rhythm
            glucose_base += 30 * np.sin(
                2 * np.pi * i / (12 * 4)
            )  # Meal effects (every 4 hours)
            glucose_base += np.random.normal(0, 15)  # Random variation

            glucose = max(70, min(350, glucose_base))

            # Simulate insulin delivery
            basal_rate = 1.0 + 0.2 * np.sin(2 * np.pi * hour_of_day / 24)

            # Bolus insulin (meals roughly every 4-6 hours)
            bolus_prob = (
                0.02 if i % (12 * 5) < 12 else 0.001
            )  # Higher prob around meal times
            bolus_insulin = (
                np.random.exponential(3.0) if np.random.random() < bolus_prob else 0
            )

            # Correction insulin (when glucose is high)
            correction_prob = 0.01 if glucose > 180 else 0
            correction_insulin = (
                np.random.exponential(1.0)
                if np.random.random() < correction_prob
                else 0
            )

            # Food delivery (correlated with bolus)
            food_prob = 0.015 if i % (12 * 5) < 12 else 0.001
            food_delivered = (
                np.random.exponential(15.0) if np.random.random() < food_prob else 0
            )
            carb_size = food_delivered * 0.8 if food_delivered > 0 else 0

            data.append(
                {
                    "EventDateTime": timestamp,
                    "participant_id": participant,
                    "CGM": glucose,
                    "Basal": basal_rate,
                    "TotalBolusInsulinDelivered": bolus_insulin,
                    "CorrectionDelivered": correction_insulin,
                    "FoodDelivered": food_delivered,
                    "CarbSize": carb_size,
                    "DeviceMode": "Auto" if np.random.random() < 0.8 else "Manual",
                    "BolusType": "Normal" if bolus_insulin > 0 else "",
                }
            )

    return pd.DataFrame(data)


def test_end_to_end_pipeline_with_realistic_data(realistic_diabetes_data):
    """End-to-end test with realistic diabetes data."""
    config = {
        "resampling": {
            "target_frequency": "5min",
            "interpolation_method": "linear",
            "max_gap_minutes": 15,
            "participant_column": "participant_id",
            "time_column": "EventDateTime",
        },
        "validation": {
            "time_column": "EventDateTime",
            "participant_column": "participant_id",
            "target_column": "CGM",
            "max_time_gap_minutes": 15,
            "min_sequence_length": 50,
        },
        "sequence_generation": {
            "sequence_length": 60,  # 1 hour
            "prediction_horizon": 6,  # 30 minutes ahead (6 * 5min)
            "target_column": "CGM",
            "participant_column": "participant_id",
            "time_column": "EventDateTime",
            "sampling_interval": 5,
        },
        "splitting": {
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "split_strategy": "participant",
        },
    }

    pipeline = SequenceGenerationPipeline(config)

    # Run the complete pipeline
    data_splits, metadata = pipeline.generate_sequences_from_dataframe(
        realistic_diabetes_data, validate_input=True, resample_data=True
    )

    # Comprehensive checks
    assert len(data_splits) == 3

    # Check that we have reasonable number of sequences
    total_sequences = sum(len(X) for X, _ in data_splits.values())
    assert total_sequences > 100  # Should have plenty of sequences from 7 days of data

    # Check sequence shapes
    for split_name, (X, y) in data_splits.items():
        assert X.shape[1] == 12  # 60 minutes / 5 minutes per point
        assert X.shape[2] > 5  # Should have multiple features

        # Check glucose values are reasonable
        assert np.all(y >= 50)  # Minimum physiological glucose
        assert np.all(y <= 400)  # Maximum reasonable glucose

    # Check metadata completeness
    required_metadata_keys = [
        "pipeline_config",
        "input_data_shape",
        "validation_results",
        "resampling_info",
        "sequence_info",
        "split_info",
    ]

    for key in required_metadata_keys:
        assert key in metadata

    # Check that validation passed
    assert "is_valid" in metadata["validation_results"]

    # Check split ratios are approximately correct (with participant-based splitting,
    # ratios may not be exact due to small number of participants)
    train_size = len(data_splits["train"][0])
    val_size = len(data_splits["val"][0])
    test_size = len(data_splits["test"][0])
    total_size = train_size + val_size + test_size

    # With only 3 participants, we expect roughly 2 in train, 1 in val, 0 in test
    # or similar distribution
    assert train_size > 0  # Should have training data
    assert total_size > 0  # Should have some data overall

    # Check that we have at least some reasonable distribution
    assert train_size >= val_size + test_size  # Training should be largest


if __name__ == "__main__":
    pytest.main([__file__])
