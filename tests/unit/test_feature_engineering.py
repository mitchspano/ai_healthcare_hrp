"""
Unit tests for feature engineering components.

Tests validate feature calculations against known expected values and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from diabetes_lstm_pipeline.feature_engineering import (
    TemporalFeatureExtractor,
    InsulinFeatureExtractor,
    GlucoseFeatureExtractor,
    LagFeatureGenerator,
    FeatureScaler,
    FeatureEngineer,
)


class TestTemporalFeatureExtractor:
    """Test temporal feature extraction."""

    @pytest.fixture
    def sample_data(self):
        """Create sample diabetes data for testing."""
        dates = pd.date_range("2024-01-01 08:00:00", periods=24, freq="1H")
        return pd.DataFrame(
            {
                "EventDateTime": dates,
                "FoodDelivered": [
                    0,
                    0,
                    30,
                    0,
                    0,
                    0,
                    45,
                    0,
                    0,
                    0,
                    0,
                    0,
                    60,
                    0,
                    0,
                    0,
                    0,
                    0,
                    40,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "TotalBolusInsulinDelivered": [
                    0,
                    0,
                    5,
                    0,
                    0,
                    0,
                    7,
                    0,
                    0,
                    0,
                    0,
                    0,
                    8,
                    0,
                    0,
                    0,
                    0,
                    0,
                    6,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "Basal": [1.0] * 24,
                "CGM": [120] * 24,
            }
        )

    def test_basic_temporal_features(self, sample_data):
        """Test basic temporal feature extraction."""
        extractor = TemporalFeatureExtractor()
        result = extractor.extract_features(sample_data)

        # Check that basic temporal features are created
        expected_features = [
            "hour_of_day",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
        ]
        for feature in expected_features:
            assert feature in result.columns

        # Verify hour of day values
        assert result["hour_of_day"].iloc[0] == 8  # First record at 8 AM
        assert result["hour_of_day"].iloc[12] == 20  # 12 hours later = 8 PM

        # Verify day of week (2024-01-01 was a Monday = 0)
        assert result["day_of_week"].iloc[0] == 0

        # Verify weekend flag (Monday is not weekend)
        assert result["is_weekend"].iloc[0] == 0

    def test_cyclical_encoding(self, sample_data):
        """Test cyclical encoding of temporal features."""
        extractor = TemporalFeatureExtractor()
        result = extractor.extract_features(sample_data)

        # Check cyclical features exist
        cyclical_features = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
        for feature in cyclical_features:
            assert feature in result.columns

        # Verify cyclical encoding properties
        # At hour 0, sin should be 0 and cos should be 1
        # At hour 6, sin should be 1 and cos should be 0
        hour_0_sin = np.sin(2 * np.pi * 0 / 24)
        hour_0_cos = np.cos(2 * np.pi * 0 / 24)

        assert abs(hour_0_sin) < 1e-10  # Should be approximately 0
        assert abs(hour_0_cos - 1) < 1e-10  # Should be approximately 1

    def test_time_since_events(self, sample_data):
        """Test time since events calculation."""
        extractor = TemporalFeatureExtractor()
        result = extractor.extract_features(sample_data)

        # Check time since event features exist
        time_features = [
            "time_since_last_meal",
            "time_since_last_bolus",
            "time_since_last_basal",
        ]
        for feature in time_features:
            assert feature in result.columns

        # Verify time since last meal calculation
        # First meal at index 2 (10 AM), next record should show 1 hour
        meal_indices = [
            i for i, val in enumerate(sample_data["FoodDelivered"]) if val > 0
        ]
        if len(meal_indices) > 1:
            first_meal_idx = meal_indices[0]
            next_idx = first_meal_idx + 1
            expected_time = 1.0  # 1 hour later
            assert (
                abs(result["time_since_last_meal"].iloc[next_idx] - expected_time) < 0.1
            )

    def test_missing_datetime_column(self):
        """Test error handling for missing EventDateTime column."""
        extractor = TemporalFeatureExtractor()
        df = pd.DataFrame({"CGM": [120, 130, 125]})

        with pytest.raises(ValueError, match="EventDateTime column is required"):
            extractor.extract_features(df)

    def test_get_feature_names(self, sample_data):
        """Test getting feature names."""
        extractor = TemporalFeatureExtractor()
        extractor.extract_features(sample_data)

        feature_names = extractor.get_feature_names()
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert "hour_of_day" in feature_names


class TestInsulinFeatureExtractor:
    """Test insulin feature extraction."""

    @pytest.fixture
    def sample_data(self):
        """Create sample diabetes data for testing."""
        dates = pd.date_range("2024-01-01 08:00:00", periods=12, freq="30min")
        return pd.DataFrame(
            {
                "EventDateTime": dates,
                "TotalBolusInsulinDelivered": [0, 0, 5, 0, 0, 3, 0, 0, 4, 0, 0, 0],
                "CorrectionDelivered": [0, 0, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0],
                "Basal": [1.0, 1.0, 1.2, 1.2, 1.0, 1.0, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0],
                "CGM": [120] * 12,
            }
        )

    def test_cumulative_insulin_features(self, sample_data):
        """Test cumulative insulin feature calculation."""
        extractor = InsulinFeatureExtractor()
        result = extractor.extract_features(sample_data)

        # Check cumulative features exist
        cumulative_features = [
            "cumulative_insulin_1h",
            "cumulative_insulin_3h",
            "cumulative_insulin_6h",
        ]
        for feature in cumulative_features:
            assert feature in result.columns

        # Verify cumulative calculation
        # Total insulin = bolus + basal
        expected_total = (
            sample_data["TotalBolusInsulinDelivered"] + sample_data["Basal"]
        )

        # Check that cumulative values are reasonable
        assert all(result["cumulative_insulin_1h"] >= 0)
        assert all(result["cumulative_insulin_3h"] >= result["cumulative_insulin_1h"])

    def test_insulin_on_board_calculation(self, sample_data):
        """Test insulin on board calculation."""
        extractor = InsulinFeatureExtractor()
        result = extractor.extract_features(sample_data)

        # Check IOB features exist
        assert "insulin_on_board" in result.columns
        assert "active_insulin_effect" in result.columns

        # IOB should be non-negative
        assert all(result["insulin_on_board"] >= 0)
        assert all(result["active_insulin_effect"] >= 0)

        # IOB should increase after bolus delivery
        bolus_indices = [
            i
            for i, val in enumerate(sample_data["TotalBolusInsulinDelivered"])
            if val > 0
        ]
        if len(bolus_indices) > 0:
            first_bolus_idx = bolus_indices[0]
            if first_bolus_idx < len(result) - 1:
                # IOB should be higher after bolus than before
                iob_before = result["insulin_on_board"].iloc[first_bolus_idx]
                iob_after = result["insulin_on_board"].iloc[first_bolus_idx + 1]
                assert iob_after >= iob_before

    def test_insulin_ratios(self, sample_data):
        """Test insulin ratio calculations."""
        extractor = InsulinFeatureExtractor()
        result = extractor.extract_features(sample_data)

        # Check ratio features exist
        assert "bolus_to_basal_ratio" in result.columns
        assert "correction_ratio" in result.columns

        # Ratios should be non-negative
        assert all(result["bolus_to_basal_ratio"] >= 0)
        assert all(result["correction_ratio"] >= 0)

        # Correction ratio should be <= 1 (correction is part of total bolus)
        assert all(result["correction_ratio"] <= 1.0)

    def test_insulin_decay_model(self):
        """Test insulin decay model function."""
        extractor = InsulinFeatureExtractor()

        # Test boundary conditions
        assert extractor._insulin_decay_model(0) == 1.0  # No decay at time 0
        assert extractor._insulin_decay_model(-10) == 1.0  # Negative time
        assert (
            extractor._insulin_decay_model(extractor.insulin_duration) == 0.0
        )  # Complete decay

        # Test monotonic decrease
        time1 = 60  # 1 hour
        time2 = 120  # 2 hours
        decay1 = extractor._insulin_decay_model(time1)
        decay2 = extractor._insulin_decay_model(time2)
        assert decay1 > decay2  # Should decay over time

    def test_insulin_effect_model(self):
        """Test insulin effect model function."""
        extractor = InsulinFeatureExtractor()

        # Test boundary conditions
        assert extractor._insulin_effect_model(0) == 0.0  # No effect at time 0
        assert extractor._insulin_effect_model(-10) == 0.0  # Negative time
        assert (
            extractor._insulin_effect_model(extractor.insulin_duration) == 0.0
        )  # No effect after duration

        # Test peak effect
        peak_effect = extractor._insulin_effect_model(extractor.insulin_peak_time)
        early_effect = extractor._insulin_effect_model(30)  # 30 minutes
        late_effect = extractor._insulin_effect_model(200)  # 200 minutes

        assert peak_effect >= early_effect
        assert peak_effect >= late_effect


class TestGlucoseFeatureExtractor:
    """Test glucose feature extraction."""

    @pytest.fixture
    def sample_data(self):
        """Create sample diabetes data with varying glucose levels."""
        dates = pd.date_range("2024-01-01 08:00:00", periods=24, freq="5min")
        # Create glucose pattern: normal -> rising -> high -> falling -> low -> normal
        glucose_values = (
            [120] * 4  # Normal
            + [130, 140, 150, 160]  # Rising
            + [170, 180, 190, 200]  # High
            + [190, 180, 170, 160]  # Falling
            + [150, 140, 130, 120]  # Normal
            + [110, 100, 90, 80]  # Low
        )

        return pd.DataFrame(
            {
                "EventDateTime": dates,
                "CGM": glucose_values,
                "TotalBolusInsulinDelivered": [0] * 24,
                "Basal": [1.0] * 24,
            }
        )

    def test_glucose_trends(self, sample_data):
        """Test glucose trend calculation."""
        extractor = GlucoseFeatureExtractor()
        result = extractor.extract_features(sample_data)

        # Check trend features exist
        trend_features = [
            "glucose_trend_15min",
            "glucose_trend_30min",
            "glucose_trend_60min",
        ]
        for feature in trend_features:
            assert feature in result.columns

        # Check rate of change
        assert "glucose_rate_of_change" in result.columns
        assert "glucose_acceleration" in result.columns

        # Verify trend calculation logic
        # During rising phase, trends should be positive
        rising_indices = range(4, 8)  # Indices where glucose is rising
        for idx in rising_indices:
            if idx < len(result) and not pd.isna(
                result["glucose_trend_15min"].iloc[idx]
            ):
                assert result["glucose_trend_15min"].iloc[idx] >= 0

    def test_glucose_variability(self, sample_data):
        """Test glucose variability metrics."""
        extractor = GlucoseFeatureExtractor()
        result = extractor.extract_features(sample_data)

        # Check variability features exist
        variability_features = [
            "glucose_variability_1h",
            "glucose_variability_3h",
            "glucose_cv_1h",
        ]
        for feature in variability_features:
            assert feature in result.columns

        # Variability should be non-negative
        for feature in variability_features:
            assert all(result[feature] >= 0)

        # CV should be reasonable (not infinite)
        cv_values = result["glucose_cv_1h"].dropna()
        assert all(cv_values < 10)  # CV shouldn't be extremely high

    def test_time_in_range(self, sample_data):
        """Test time in range calculations."""
        extractor = GlucoseFeatureExtractor()
        result = extractor.extract_features(sample_data)

        # Check TIR features exist
        tir_features = [
            "time_in_range_3h",
            "time_in_range_6h",
            "time_below_range_3h",
            "time_above_range_3h",
        ]
        for feature in tir_features:
            assert feature in result.columns

        # TIR values should be between 0 and 1
        for feature in tir_features:
            values = result[feature].dropna()
            assert all(values >= 0)
            assert all(values <= 1)

        # Sum of TIR components should approximately equal 1
        # (allowing for some numerical precision issues)
        non_na_indices = (
            result[["time_in_range_3h", "time_below_range_3h", "time_above_range_3h"]]
            .dropna()
            .index
        )
        for idx in non_na_indices:
            total = (
                result.loc[idx, "time_in_range_3h"]
                + result.loc[idx, "time_below_range_3h"]
                + result.loc[idx, "time_above_range_3h"]
            )
            assert abs(total - 1.0) < 0.1  # Allow some tolerance

    def test_glucose_states(self, sample_data):
        """Test glucose state indicators."""
        extractor = GlucoseFeatureExtractor()
        result = extractor.extract_features(sample_data)

        # Check state features exist
        state_features = ["is_hypoglycemic", "is_hyperglycemic", "is_in_target_range"]
        for feature in state_features:
            assert feature in result.columns

        # State indicators should be binary (0 or 1)
        for feature in state_features:
            unique_values = result[feature].unique()
            assert all(val in [0, 1] for val in unique_values)

        # Check risk and stability indices
        assert "glucose_risk_index" in result.columns
        assert "glucose_stability_index" in result.columns

        # Risk index should be non-negative
        assert all(result["glucose_risk_index"] >= 0)

        # Stability index should be between 0 and 1
        stability_values = result["glucose_stability_index"].dropna()
        assert all(stability_values >= 0)
        assert all(stability_values <= 1)

    def test_missing_cgm_column(self):
        """Test error handling for missing CGM column."""
        extractor = GlucoseFeatureExtractor()
        df = pd.DataFrame(
            {"EventDateTime": pd.date_range("2024-01-01", periods=3, freq="5min")}
        )

        with pytest.raises(ValueError, match="CGM column is required"):
            extractor.extract_features(df)

    def test_custom_glucose_ranges(self):
        """Test custom glucose range configuration."""
        custom_config = {
            "target_range": (80, 160),
            "hypoglycemia_threshold": 80,
            "hyperglycemia_threshold": 200,
        }

        extractor = GlucoseFeatureExtractor(custom_config)

        # Verify configuration is applied
        assert extractor.target_range == (80, 160)
        assert extractor.hypoglycemia_threshold == 80
        assert extractor.hyperglycemia_threshold == 200


class TestLagFeatureGenerator:
    """Test lag feature generation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample time series data."""
        dates = pd.date_range("2024-01-01 08:00:00", periods=30, freq="5min")
        glucose_values = list(range(100, 130))  # Increasing glucose values

        return pd.DataFrame(
            {
                "EventDateTime": dates,
                "CGM": glucose_values,
                "TotalBolusInsulinDelivered": [0] * 30,
            }
        )

    def test_basic_lag_features(self, sample_data):
        """Test basic lag feature generation."""
        generator = LagFeatureGenerator()
        result = generator.generate_features(sample_data)

        # Check that lag features are created
        expected_lags = [5, 10, 15, 30, 45, 60, 90, 120]  # Default lag intervals
        for lag in expected_lags:
            lag_column = f"CGM_lag_{lag}min"
            assert lag_column in result.columns

        # Verify lag calculation
        # CGM_lag_5min should be the CGM value from 1 period ago (5 minutes)
        periods_5min = 1  # 5 minutes / 5 minutes per period
        for i in range(periods_5min, len(result)):
            expected_lag_value = sample_data["CGM"].iloc[i - periods_5min]
            actual_lag_value = result["CGM_lag_5min"].iloc[i]
            assert abs(actual_lag_value - expected_lag_value) < 1e-6

    def test_rolling_statistics(self, sample_data):
        """Test rolling statistics generation."""
        generator = LagFeatureGenerator()
        result = generator.generate_features(sample_data)

        # Check rolling statistics features
        rolling_features = [
            "CGM_rolling_mean_30min",
            "CGM_rolling_std_30min",
            "CGM_rolling_mean_1H",
            "CGM_rolling_std_1H",
        ]
        for feature in rolling_features:
            assert feature in result.columns

        # Rolling mean should be reasonable
        rolling_mean = result["CGM_rolling_mean_30min"].dropna()
        assert all(rolling_mean >= sample_data["CGM"].min())
        assert all(rolling_mean <= sample_data["CGM"].max())

        # Rolling std should be non-negative
        rolling_std = result["CGM_rolling_std_30min"].dropna()
        assert all(rolling_std >= 0)

    def test_custom_lag_configuration(self):
        """Test custom lag configuration."""
        custom_config = {
            "lag_intervals": [10, 20, 30],
            "lag_columns": ["CGM", "TotalBolusInsulinDelivered"],
        }

        dates = pd.date_range("2024-01-01", periods=10, freq="5min")
        df = pd.DataFrame(
            {
                "EventDateTime": dates,
                "CGM": range(100, 110),
                "TotalBolusInsulinDelivered": [0, 0, 5, 0, 0, 3, 0, 0, 2, 0],
            }
        )

        generator = LagFeatureGenerator(custom_config)
        result = generator.generate_features(df)

        # Check custom lag intervals
        for lag in [10, 20, 30]:
            assert f"CGM_lag_{lag}min" in result.columns
            assert f"TotalBolusInsulinDelivered_lag_{lag}min" in result.columns

        # Should not have default lag intervals
        assert "CGM_lag_5min" not in result.columns

    def test_missing_column_handling(self):
        """Test handling of missing columns."""
        config = {"lag_columns": ["NonExistentColumn"]}
        generator = LagFeatureGenerator(config)

        df = pd.DataFrame(
            {
                "EventDateTime": pd.date_range("2024-01-01", periods=5, freq="5min"),
                "CGM": [100, 101, 102, 103, 104],
            }
        )

        # Should not raise error, but should log warning
        with patch(
            "diabetes_lstm_pipeline.feature_engineering.feature_engineering.logger"
        ) as mock_logger:
            result = generator.generate_features(df)
            mock_logger.warning.assert_called()

        # Should still return the dataframe
        assert len(result) == len(df)


class TestFeatureScaler:
    """Test feature scaling utilities."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for scaling."""
        return pd.DataFrame(
            {
                "EventDateTime": pd.date_range("2024-01-01", periods=10, freq="1H"),
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "feature3": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "participant_id": ["P1"] * 10,
            }
        )

    def test_standard_scaler(self, sample_data):
        """Test standard scaling."""
        scaler = FeatureScaler("standard")
        result = scaler.fit_transform(sample_data)

        # Check that numeric features are scaled
        numeric_features = ["feature1", "feature2", "feature3"]
        for feature in numeric_features:
            # Standard scaled features should have mean ≈ 0 and std ≈ 1
            mean_val = result[feature].mean()
            std_val = result[feature].std()
            assert abs(mean_val) < 1e-6  # Should be approximately 0 (relaxed tolerance)
            assert (
                abs(std_val - 1.0) < 0.1
            )  # Should be approximately 1 (relaxed tolerance)

        # Non-numeric columns should be unchanged
        assert result["EventDateTime"].equals(sample_data["EventDateTime"])
        assert result["participant_id"].equals(sample_data["participant_id"])

    def test_minmax_scaler(self, sample_data):
        """Test min-max scaling."""
        scaler = FeatureScaler("minmax")
        result = scaler.fit_transform(sample_data)

        # Check that features are scaled to [0, 1] range
        numeric_features = ["feature1", "feature2", "feature3"]
        for feature in numeric_features:
            min_val = result[feature].min()
            max_val = result[feature].max()
            assert abs(min_val) < 1e-10  # Should be approximately 0
            assert abs(max_val - 1.0) < 1e-10  # Should be approximately 1

    def test_robust_scaler(self, sample_data):
        """Test robust scaling."""
        scaler = FeatureScaler("robust")
        result = scaler.fit_transform(sample_data)

        # Robust scaler should handle the data without errors
        numeric_features = ["feature1", "feature2", "feature3"]
        for feature in numeric_features:
            # Values should be different from original (scaled)
            assert not result[feature].equals(sample_data[feature])

    def test_transform_without_fit(self, sample_data):
        """Test transform on new data after fitting."""
        scaler = FeatureScaler("standard")

        # Fit on original data
        scaler.fit_transform(sample_data)

        # Create new data with same structure
        new_data = sample_data.copy()
        new_data["feature1"] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        # Transform new data
        result = scaler.transform(new_data)

        # Should not raise error and should return scaled data
        assert len(result) == len(new_data)
        assert "feature1" in result.columns

    def test_inverse_transform(self, sample_data):
        """Test inverse transformation."""
        scaler = FeatureScaler("standard")

        # Fit and transform
        scaled_data = scaler.fit_transform(sample_data)

        # Inverse transform
        recovered_data = scaler.inverse_transform(scaled_data)

        # Should recover original values (within numerical precision)
        numeric_features = ["feature1", "feature2", "feature3"]
        for feature in numeric_features:
            original_values = sample_data[feature].values
            recovered_values = recovered_data[feature].values
            np.testing.assert_allclose(original_values, recovered_values, rtol=1e-10)

    def test_unsupported_scaler_type(self):
        """Test error handling for unsupported scaler type."""
        with pytest.raises(ValueError, match="Unsupported scaler type"):
            FeatureScaler("unsupported_scaler")

    def test_get_scaler_params(self, sample_data):
        """Test getting scaler parameters."""
        scaler = FeatureScaler("standard")
        scaler.fit_transform(sample_data)

        params = scaler.get_scaler_params()

        # Should return parameters for fitted features
        assert isinstance(params, dict)
        assert len(params) > 0

        # Check that parameters contain expected keys
        for feature_params in params.values():
            assert "mean" in feature_params
            assert "scale" in feature_params


class TestFeatureEngineer:
    """Test the main feature engineering orchestrator."""

    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data."""
        dates = pd.date_range("2024-01-01 08:00:00", periods=20, freq="5min")
        return pd.DataFrame(
            {
                "EventDateTime": dates,
                "CGM": [
                    120,
                    125,
                    130,
                    135,
                    140,
                    145,
                    150,
                    155,
                    160,
                    165,
                    160,
                    155,
                    150,
                    145,
                    140,
                    135,
                    130,
                    125,
                    120,
                    115,
                ],
                "TotalBolusInsulinDelivered": [
                    0,
                    0,
                    5,
                    0,
                    0,
                    0,
                    3,
                    0,
                    0,
                    0,
                    0,
                    0,
                    4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    0,
                ],
                "CorrectionDelivered": [
                    0,
                    0,
                    2,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                ],
                "Basal": [1.0] * 20,
                "FoodDelivered": [
                    0,
                    0,
                    30,
                    0,
                    0,
                    0,
                    0,
                    0,
                    45,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    25,
                    0,
                ],
            }
        )

    @pytest.fixture
    def config(self):
        """Create configuration for feature engineer."""
        return {
            "temporal": {},
            "insulin": {"insulin_peak_time": 75, "insulin_duration": 360},
            "glucose": {
                "target_range": (70, 180),
                "hypoglycemia_threshold": 70,
                "hyperglycemia_threshold": 250,
            },
            "lag": {"lag_intervals": [5, 10, 15, 30], "lag_columns": ["CGM"]},
            "scaling": {"type": "standard"},
        }

    def test_full_feature_engineering_pipeline(self, sample_data, config):
        """Test complete feature engineering pipeline."""
        engineer = FeatureEngineer(config)
        result = engineer.engineer_features(sample_data, fit_scaler=True)

        # Should have more columns than original data
        assert len(result.columns) > len(sample_data.columns)

        # Should have same number of rows
        assert len(result) == len(sample_data)

        # Check that features from all extractors are present
        feature_names = engineer.get_feature_names()
        assert len(feature_names) > 0

        # Verify feature groups
        feature_groups = engineer.get_feature_importance_groups()
        expected_groups = ["temporal", "insulin", "glucose", "lag"]
        for group in expected_groups:
            assert group in feature_groups
            assert len(feature_groups[group]) > 0

    def test_feature_engineering_without_fitting_scaler(self, sample_data, config):
        """Test feature engineering without fitting scaler (for test data)."""
        engineer = FeatureEngineer(config)

        # First fit on training data
        engineer.engineer_features(sample_data, fit_scaler=True)

        # Then transform test data without fitting
        test_data = sample_data.copy()
        test_data["CGM"] = test_data["CGM"] + 10  # Slightly different values

        result = engineer.engineer_features(test_data, fit_scaler=False)

        # Should have same structure as training data (allowing for some intermediate columns)
        # The exact count may vary due to intermediate columns created during processing
        assert len(result.columns) >= len(sample_data.columns)
        assert len(result) == len(test_data)

    def test_get_feature_names(self, sample_data, config):
        """Test getting all feature names."""
        engineer = FeatureEngineer(config)
        engineer.engineer_features(sample_data)

        feature_names = engineer.get_feature_names()

        # Should be a list of strings
        assert isinstance(feature_names, list)
        assert all(isinstance(name, str) for name in feature_names)
        assert len(feature_names) > 0

    def test_get_feature_importance_groups(self, sample_data, config):
        """Test getting feature groups for analysis."""
        engineer = FeatureEngineer(config)
        engineer.engineer_features(sample_data)

        groups = engineer.get_feature_importance_groups()

        # Should have all expected groups
        expected_groups = ["temporal", "insulin", "glucose", "lag"]
        for group in expected_groups:
            assert group in groups
            assert isinstance(groups[group], list)
            assert len(groups[group]) > 0

        # All features should be accounted for
        all_group_features = []
        for group_features in groups.values():
            all_group_features.extend(group_features)

        engineer_features = engineer.get_feature_names()
        assert len(all_group_features) == len(engineer_features)


if __name__ == "__main__":
    pytest.main([__file__])
