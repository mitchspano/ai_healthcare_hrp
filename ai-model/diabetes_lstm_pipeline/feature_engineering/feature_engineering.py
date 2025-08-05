"""
Feature engineering module for diabetes LSTM pipeline.

This module contains classes for extracting various types of features from diabetes data:
- Temporal features (time-based patterns)
- Insulin features (dosing and on-board calculations)
- Glucose features (trends and variability)
- Lag features (historical values)
- Scaling and normalization utilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor:
    """Extracts time-based features from diabetes data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize temporal feature extractor.

        Args:
            config: Configuration dictionary with temporal feature settings
        """
        self.config = config or {}
        self.feature_names = []

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from the dataframe.

        Args:
            df: DataFrame with EventDateTime column

        Returns:
            DataFrame with added temporal features
        """
        logger.info("Extracting temporal features")

        # Ensure EventDateTime is datetime type
        if "EventDateTime" not in df.columns:
            raise ValueError("EventDateTime column is required for temporal features")

        df = df.copy()
        df["EventDateTime"] = pd.to_datetime(df["EventDateTime"])

        # Remove rows with invalid EventDateTime values
        initial_count = len(df)
        df = df.dropna(subset=["EventDateTime"])
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.warning(
                f"Removed {removed_count} rows with invalid EventDateTime values"
            )

        # Basic time features
        df["hour_of_day"] = df["EventDateTime"].dt.hour
        df["day_of_week"] = df["EventDateTime"].dt.dayofweek
        df["day_of_month"] = df["EventDateTime"].dt.day
        df["month"] = df["EventDateTime"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Cyclical encoding for temporal features
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Time since events
        df = self._calculate_time_since_events(df)

        # Update feature names
        self.feature_names = [
            "hour_of_day",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "time_since_last_meal",
            "time_since_last_bolus",
            "time_since_last_basal",
        ]

        logger.info(f"Extracted {len(self.feature_names)} temporal features")
        return df

    def _calculate_time_since_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time since various diabetes management events."""
        df = df.sort_values("EventDateTime")

        # Time since last meal (when FoodDelivered > 0)
        meal_mask = df["FoodDelivered"].fillna(0) > 0
        df["last_meal_time"] = df.loc[meal_mask, "EventDateTime"]
        df["last_meal_time"] = df["last_meal_time"].ffill()
        df["time_since_last_meal"] = (
            df["EventDateTime"] - df["last_meal_time"]
        ).dt.total_seconds() / 3600
        df["time_since_last_meal"] = df["time_since_last_meal"].fillna(
            24
        )  # Default to 24 hours if no previous meal

        # Time since last bolus (when TotalBolusInsulinDelivered > 0)
        bolus_mask = df["TotalBolusInsulinDelivered"].fillna(0) > 0
        df["last_bolus_time"] = df.loc[bolus_mask, "EventDateTime"]
        df["last_bolus_time"] = df["last_bolus_time"].ffill()
        df["time_since_last_bolus"] = (
            df["EventDateTime"] - df["last_bolus_time"]
        ).dt.total_seconds() / 3600
        df["time_since_last_bolus"] = df["time_since_last_bolus"].fillna(
            12
        )  # Default to 12 hours

        # Time since last basal change (when Basal changes)
        df["basal_change"] = df["Basal"].diff().abs() > 0.01
        basal_mask = df["basal_change"].fillna(False)
        df["last_basal_time"] = df.loc[basal_mask, "EventDateTime"]
        df["last_basal_time"] = df["last_basal_time"].ffill()
        df["time_since_last_basal"] = (
            df["EventDateTime"] - df["last_basal_time"]
        ).dt.total_seconds() / 3600
        df["time_since_last_basal"] = df["time_since_last_basal"].fillna(
            6
        )  # Default to 6 hours

        # Clean up temporary columns
        df = df.drop(
            ["last_meal_time", "last_bolus_time", "last_basal_time", "basal_change"],
            axis=1,
        )

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of temporal feature names."""
        return self.feature_names.copy()


class InsulinFeatureExtractor:
    """Extracts insulin-related features from diabetes data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize insulin feature extractor.

        Args:
            config: Configuration dictionary with insulin feature settings
        """
        self.config = config or {}
        self.feature_names = []

        # Insulin action parameters (configurable)
        self.insulin_peak_time = self.config.get("insulin_peak_time", 75)  # minutes
        self.insulin_duration = self.config.get("insulin_duration", 360)  # minutes
        self.insulin_sensitivity = self.config.get(
            "insulin_sensitivity", 50
        )  # mg/dL per unit

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract insulin-related features from the dataframe.

        Args:
            df: DataFrame with insulin delivery columns

        Returns:
            DataFrame with added insulin features
        """
        logger.info("Extracting insulin features")

        df = df.copy()
        df["EventDateTime"] = pd.to_datetime(df["EventDateTime"])

        # Remove rows with invalid EventDateTime values
        initial_count = len(df)
        df = df.dropna(subset=["EventDateTime"])
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.warning(
                f"Removed {removed_count} rows with invalid EventDateTime values"
            )

        df = df.sort_values("EventDateTime")

        # Fill missing insulin values with 0
        insulin_columns = ["TotalBolusInsulinDelivered", "CorrectionDelivered", "Basal"]
        for col in insulin_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Cumulative insulin features
        df = self._calculate_cumulative_insulin(df)

        # Insulin on board calculation
        df = self._calculate_insulin_on_board(df)

        # Insulin ratios and rates
        df = self._calculate_insulin_ratios(df)

        # Update feature names
        self.feature_names = [
            "cumulative_insulin_1h",
            "cumulative_insulin_3h",
            "cumulative_insulin_6h",
            "cumulative_bolus_1h",
            "cumulative_bolus_3h",
            "cumulative_correction_1h",
            "insulin_on_board",
            "active_insulin_effect",
            "bolus_to_basal_ratio",
            "correction_ratio",
            "avg_basal_rate_1h",
        ]

        logger.info(f"Extracted {len(self.feature_names)} insulin features")
        return df

    def _calculate_cumulative_insulin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative insulin delivery over various time windows."""
        # Create time-based rolling windows
        df = df.set_index("EventDateTime")

        # Total insulin (bolus + basal)
        df["total_insulin"] = df["TotalBolusInsulinDelivered"] + df["Basal"]

        # Cumulative insulin over different time windows
        df["cumulative_insulin_1h"] = df["total_insulin"].rolling("1h").sum()
        df["cumulative_insulin_3h"] = df["total_insulin"].rolling("3h").sum()
        df["cumulative_insulin_6h"] = df["total_insulin"].rolling("6h").sum()

        # Cumulative bolus insulin
        df["cumulative_bolus_1h"] = df["TotalBolusInsulinDelivered"].rolling("1h").sum()
        df["cumulative_bolus_3h"] = df["TotalBolusInsulinDelivered"].rolling("3h").sum()

        # Cumulative correction insulin
        df["cumulative_correction_1h"] = df["CorrectionDelivered"].rolling("1h").sum()

        # Average basal rate
        df["avg_basal_rate_1h"] = df["Basal"].rolling("1h").mean()

        # Clean up intermediate columns
        df = df.drop(["total_insulin"], axis=1)

        df = df.reset_index()
        return df

    def _calculate_insulin_on_board(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate insulin on board using pharmacokinetic model."""
        df["insulin_on_board"] = 0.0
        df["active_insulin_effect"] = 0.0

        # Convert to numpy for faster computation
        times = df["EventDateTime"].values
        bolus_insulin = df["TotalBolusInsulinDelivered"].values
        iob = np.zeros(len(df))
        insulin_effect = np.zeros(len(df))

        for i in range(len(df)):
            current_time = times[i]

            # Look back at previous bolus deliveries
            for j in range(max(0, i - 100), i):  # Look back up to 100 records
                if bolus_insulin[j] > 0:
                    time_diff = (current_time - times[j]) / np.timedelta64(
                        1, "m"
                    )  # minutes

                    if time_diff <= self.insulin_duration:
                        # Calculate remaining insulin using exponential decay model
                        remaining_fraction = self._insulin_decay_model(time_diff)
                        iob[i] += bolus_insulin[j] * remaining_fraction

                        # Calculate insulin effect (peak at insulin_peak_time)
                        effect_fraction = self._insulin_effect_model(time_diff)
                        insulin_effect[i] += bolus_insulin[j] * effect_fraction

        df["insulin_on_board"] = iob
        df["active_insulin_effect"] = insulin_effect

        return df

    def _insulin_decay_model(self, time_minutes: float) -> float:
        """Model insulin decay over time."""
        if time_minutes <= 0:
            return 1.0
        if time_minutes >= self.insulin_duration:
            return 0.0

        # Exponential decay model
        decay_rate = 4.0 / self.insulin_duration  # Adjust for desired half-life
        return np.exp(-decay_rate * time_minutes / 60)  # Convert to hours

    def _insulin_effect_model(self, time_minutes: float) -> float:
        """Model insulin effect over time (peaks then decays)."""
        if time_minutes <= 0 or time_minutes >= self.insulin_duration:
            return 0.0

        # Gamma-like distribution for insulin effect
        peak_time = self.insulin_peak_time
        if time_minutes <= peak_time:
            return time_minutes / peak_time
        else:
            remaining_time = self.insulin_duration - time_minutes
            total_decay_time = self.insulin_duration - peak_time
            return remaining_time / total_decay_time

    def _calculate_insulin_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate insulin ratios and derived metrics."""
        # Bolus to basal ratio
        df["bolus_to_basal_ratio"] = np.where(
            df["Basal"] > 0, df["TotalBolusInsulinDelivered"] / df["Basal"], 0
        )

        # Correction ratio (correction to total bolus)
        df["correction_ratio"] = np.where(
            df["TotalBolusInsulinDelivered"] > 0,
            df["CorrectionDelivered"] / df["TotalBolusInsulinDelivered"],
            0,
        )

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of insulin feature names."""
        return self.feature_names.copy()


class GlucoseFeatureExtractor:
    """Extracts glucose-related features from CGM data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize glucose feature extractor.

        Args:
            config: Configuration dictionary with glucose feature settings
        """
        self.config = config or {}
        self.feature_names = []

        # Glucose range definitions (configurable)
        self.target_range = self.config.get("target_range", (70, 180))  # mg/dL
        self.hypoglycemia_threshold = self.config.get("hypoglycemia_threshold", 70)
        self.hyperglycemia_threshold = self.config.get("hyperglycemia_threshold", 250)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract glucose-related features from the dataframe.

        Args:
            df: DataFrame with CGM column

        Returns:
            DataFrame with added glucose features
        """
        logger.info("Extracting glucose features")

        if "CGM" not in df.columns:
            raise ValueError("CGM column is required for glucose features")

        df = df.copy()
        df["EventDateTime"] = pd.to_datetime(df["EventDateTime"])

        # Remove rows with invalid EventDateTime values
        initial_count = len(df)
        df = df.dropna(subset=["EventDateTime"])
        removed_count = initial_count - len(df)
        if removed_count > 0:
            logger.warning(
                f"Removed {removed_count} rows with invalid EventDateTime values"
            )

        df = df.sort_values("EventDateTime")

        # Fill missing CGM values using forward fill
        df["CGM"] = df["CGM"].ffill()

        # Glucose trends and rates of change
        df = self._calculate_glucose_trends(df)

        # Glucose variability metrics
        df = self._calculate_glucose_variability(df)

        # Time in range metrics
        df = self._calculate_time_in_range(df)

        # Glucose state indicators
        df = self._calculate_glucose_states(df)

        # Update feature names
        self.feature_names = [
            "glucose_trend_15min",
            "glucose_trend_30min",
            "glucose_trend_60min",
            "glucose_rate_of_change",
            "glucose_acceleration",
            "glucose_variability_1h",
            "glucose_variability_3h",
            "glucose_cv_1h",
            "time_in_range_3h",
            "time_in_range_6h",
            "time_below_range_3h",
            "time_above_range_3h",
            "is_hypoglycemic",
            "is_hyperglycemic",
            "is_in_target_range",
            "glucose_risk_index",
            "glucose_stability_index",
        ]

        logger.info(f"Extracted {len(self.feature_names)} glucose features")
        return df

    def _calculate_glucose_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate glucose trends and rates of change."""
        # Glucose trends over different time windows
        df["glucose_trend_15min"] = df["CGM"].diff(
            periods=3
        )  # Assuming 5-min intervals
        df["glucose_trend_30min"] = df["CGM"].diff(periods=6)
        df["glucose_trend_60min"] = df["CGM"].diff(periods=12)

        # Rate of change (mg/dL per minute)
        df["glucose_rate_of_change"] = (
            df["CGM"].diff() / df["EventDateTime"].diff().dt.total_seconds() * 60
        )

        # Glucose acceleration (second derivative)
        df["glucose_acceleration"] = df["glucose_rate_of_change"].diff()

        # Fill NaN values with 0
        trend_columns = [
            "glucose_trend_15min",
            "glucose_trend_30min",
            "glucose_trend_60min",
            "glucose_rate_of_change",
            "glucose_acceleration",
        ]
        for col in trend_columns:
            df[col] = df[col].fillna(0)

        return df

    def _calculate_glucose_variability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate glucose variability metrics."""
        df = df.set_index("EventDateTime")

        # Standard deviation over time windows
        df["glucose_variability_1h"] = df["CGM"].rolling("1h").std()
        df["glucose_variability_3h"] = df["CGM"].rolling("3h").std()

        # Coefficient of variation
        df["glucose_cv_1h"] = (
            df["glucose_variability_1h"] / df["CGM"].rolling("1h").mean()
        )

        # Fill NaN values
        variability_columns = [
            "glucose_variability_1h",
            "glucose_variability_3h",
            "glucose_cv_1h",
        ]
        for col in variability_columns:
            df[col] = df[col].fillna(0)

        df = df.reset_index()
        return df

    def _calculate_time_in_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time in range metrics."""
        df = df.set_index("EventDateTime")

        # Define range indicators
        in_range = (df["CGM"] >= self.target_range[0]) & (
            df["CGM"] <= self.target_range[1]
        )
        below_range = df["CGM"] < self.target_range[0]
        above_range = df["CGM"] > self.target_range[1]

        # Calculate time in range over different windows
        df["time_in_range_3h"] = in_range.rolling("3h").mean()
        df["time_in_range_6h"] = in_range.rolling("6h").mean()
        df["time_below_range_3h"] = below_range.rolling("3h").mean()
        df["time_above_range_3h"] = above_range.rolling("3h").mean()

        # Fill NaN values
        tir_columns = [
            "time_in_range_3h",
            "time_in_range_6h",
            "time_below_range_3h",
            "time_above_range_3h",
        ]
        for col in tir_columns:
            df[col] = df[col].fillna(0)

        df = df.reset_index()
        return df

    def _calculate_glucose_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate glucose state indicators and risk indices."""
        # Binary state indicators
        df["is_hypoglycemic"] = (df["CGM"] < self.hypoglycemia_threshold).astype(int)
        df["is_hyperglycemic"] = (df["CGM"] > self.hyperglycemia_threshold).astype(int)
        df["is_in_target_range"] = (
            (df["CGM"] >= self.target_range[0]) & (df["CGM"] <= self.target_range[1])
        ).astype(int)

        # Glucose risk index (higher for extreme values)
        target_center = (self.target_range[0] + self.target_range[1]) / 2
        df["glucose_risk_index"] = np.abs(df["CGM"] - target_center) / target_center

        # Glucose stability index (based on recent variability)
        df = df.set_index("EventDateTime")
        recent_std = df["CGM"].rolling("1h").std().fillna(0)
        df["glucose_stability_index"] = 1 / (1 + recent_std)  # Higher = more stable
        df = df.reset_index()

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of glucose feature names."""
        return self.feature_names.copy()


class LagFeatureGenerator:
    """Generates lag features (historical values) for time series modeling."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize lag feature generator.

        Args:
            config: Configuration dictionary with lag feature settings
        """
        self.config = config or {}
        self.feature_names = []

        # Default lag intervals (in minutes)
        self.lag_intervals = self.config.get(
            "lag_intervals", [5, 10, 15, 30, 45, 60, 90, 120]
        )
        self.lag_columns = self.config.get("lag_columns", ["CGM"])

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lag features from the dataframe.

        Args:
            df: DataFrame with time series data

        Returns:
            DataFrame with added lag features
        """
        logger.info("Generating lag features")

        df = df.copy()
        df["EventDateTime"] = pd.to_datetime(df["EventDateTime"])
        df = df.sort_values("EventDateTime")

        # Generate lag features for each specified column
        for column in self.lag_columns:
            if column not in df.columns:
                logger.warning(f"Column {column} not found, skipping lag features")
                continue

            df = self._generate_column_lags(df, column)

        # Generate rolling statistics
        df = self._generate_rolling_statistics(df)

        logger.info(f"Generated {len(self.feature_names)} lag features")
        return df

    def _generate_column_lags(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Generate lag features for a specific column."""
        # Assuming 5-minute intervals, calculate periods for each lag
        base_interval = 5  # minutes

        for lag_minutes in self.lag_intervals:
            periods = lag_minutes // base_interval
            lag_column_name = f"{column}_lag_{lag_minutes}min"

            df[lag_column_name] = df[column].shift(periods)
            self.feature_names.append(lag_column_name)

        # Fill NaN values with forward fill for initial periods
        lag_columns = [f"{column}_lag_{lag}min" for lag in self.lag_intervals]
        for lag_col in lag_columns:
            df[lag_col] = df[lag_col].ffill().fillna(df[column].mean())

        return df

    def _generate_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling statistics as additional lag-based features."""
        df = df.set_index("EventDateTime")

        for column in self.lag_columns:
            if column not in df.columns:
                continue

            # Rolling mean and std over different windows
            for window in ["30min", "1h", "2h"]:
                mean_col = f"{column}_rolling_mean_{window}"
                std_col = f"{column}_rolling_std_{window}"

                df[mean_col] = df[column].rolling(window).mean()
                df[std_col] = df[column].rolling(window).std()

                # Fill NaN values
                df[mean_col] = df[mean_col].fillna(df[column].mean())
                df[std_col] = df[std_col].fillna(0)

                self.feature_names.extend([mean_col, std_col])

        df = df.reset_index()
        return df

    def get_feature_names(self) -> List[str]:
        """Get list of lag feature names."""
        return self.feature_names.copy()


class FeatureScaler:
    """Utility class for scaling and normalizing features."""

    def __init__(
        self, scaler_type: str = "standard", config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize feature scaler.

        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            config: Configuration dictionary
        """
        self.scaler_type = scaler_type
        self.config = config or {}
        self.scalers = {}
        self.feature_columns = []

        # Initialize scaler based on type
        if scaler_type == "standard":
            self.base_scaler = StandardScaler()
        elif scaler_type == "minmax":
            feature_range = self.config.get("feature_range", (0, 1))
            self.base_scaler = MinMaxScaler(feature_range=feature_range)
        elif scaler_type == "robust":
            self.base_scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")

    def fit_transform(
        self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit scaler and transform features.

        Args:
            df: DataFrame with features to scale
            feature_columns: List of columns to scale (if None, scale all numeric columns)

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        if feature_columns is None:
            # Scale all numeric columns except datetime and identifier columns
            exclude_columns = ["EventDateTime", "participant_id"]
            feature_columns = [
                col
                for col in df.select_dtypes(include=[np.number]).columns
                if col not in exclude_columns
            ]

        self.feature_columns = feature_columns

        # Clean data before scaling - handle infinite and extreme values
        df = self._clean_data_for_scaling(df, feature_columns)

        # Fit and transform each feature column
        for column in feature_columns:
            if column in df.columns:
                scaler = self.base_scaler.__class__(**self.base_scaler.get_params())
                df[column] = scaler.fit_transform(df[[column]])
                self.scalers[column] = scaler

        logger.info(
            f"Fitted and transformed {len(feature_columns)} features using {self.scaler_type} scaler"
        )
        return df

    def _clean_data_for_scaling(
        self, df: pd.DataFrame, feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Clean data by handling infinite and extreme values before scaling.

        Args:
            df: DataFrame to clean
            feature_columns: List of feature columns to clean

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        for column in feature_columns:
            if column in df.columns:
                # Replace infinite values with NaN
                df[column] = df[column].replace([np.inf, -np.inf], np.nan)

                # Calculate robust statistics for outlier detection
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1

                # Define bounds for outlier detection (using 3x IQR for more tolerance)
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                # Replace extreme outliers with bounds
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

                # Fill remaining NaN values with median
                median_val = df[column].median()
                df[column] = df[column].fillna(median_val)

                # Final check for any remaining infinite values
                if np.any(np.isinf(df[column])):
                    logger.warning(
                        f"Replacing remaining infinite values in {column} with median"
                    )
                    df[column] = df[column].replace([np.inf, -np.inf], median_val)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scalers.

        Args:
            df: DataFrame with features to scale

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        # Clean data before scaling - handle infinite and extreme values
        df = self._clean_data_for_scaling(df, self.feature_columns)

        for column in self.feature_columns:
            if column in df.columns and column in self.scalers:
                df[column] = self.scalers[column].transform(df[[column]])

        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled features back to original scale.

        Args:
            df: DataFrame with scaled features

        Returns:
            DataFrame with features in original scale
        """
        df = df.copy()

        for column in self.feature_columns:
            if column in df.columns and column in self.scalers:
                df[column] = self.scalers[column].inverse_transform(df[[column]])

        return df

    def get_scaler_params(self) -> Dict[str, Any]:
        """Get parameters of fitted scalers."""
        params = {}
        for column, scaler in self.scalers.items():
            if hasattr(scaler, "mean_"):
                params[column] = {
                    "mean": scaler.mean_[0] if hasattr(scaler, "mean_") else None,
                    "scale": scaler.scale_[0] if hasattr(scaler, "scale_") else None,
                }
            elif hasattr(scaler, "data_min_"):
                params[column] = {
                    "data_min": scaler.data_min_[0],
                    "data_max": scaler.data_max_[0],
                    "data_range": scaler.data_range_[0],
                }
        return params


class FeatureEngineer:
    """Main feature engineering orchestrator class."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize feature engineer with configuration.

        Args:
            config: Configuration dictionary for all feature extractors
        """
        self.config = config

        # Initialize feature extractors
        self.temporal_extractor = TemporalFeatureExtractor(config.get("temporal", {}))
        self.insulin_extractor = InsulinFeatureExtractor(config.get("insulin", {}))
        self.glucose_extractor = GlucoseFeatureExtractor(config.get("glucose", {}))
        self.lag_generator = LagFeatureGenerator(config.get("lag", {}))

        # Initialize scaler
        scaler_config = config.get("scaling", {})
        scaler_type = scaler_config.get("type", "standard")
        self.scaler = FeatureScaler(scaler_type, scaler_config)

        self.all_feature_names = []

    def engineer_features(
        self, df: pd.DataFrame, fit_scaler: bool = True
    ) -> pd.DataFrame:
        """
        Apply all feature engineering steps to the dataframe.

        Args:
            df: Input dataframe with raw diabetes data
            fit_scaler: Whether to fit the scaler (True for training data)

        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline")

        # Extract temporal features
        df = self.temporal_extractor.extract_features(df)

        # Extract insulin features
        df = self.insulin_extractor.extract_features(df)

        # Extract glucose features
        df = self.glucose_extractor.extract_features(df)

        # Generate lag features
        df = self.lag_generator.generate_features(df)

        # Collect all feature names
        self.all_feature_names = (
            self.temporal_extractor.get_feature_names()
            + self.insulin_extractor.get_feature_names()
            + self.glucose_extractor.get_feature_names()
            + self.lag_generator.get_feature_names()
        )

        # Apply scaling
        if fit_scaler:
            df = self.scaler.fit_transform(df, self.all_feature_names)
        else:
            df = self.scaler.transform(df)

        logger.info(
            f"Feature engineering complete. Generated {len(self.all_feature_names)} features"
        )
        return df

    def get_feature_names(self) -> List[str]:
        """Get list of all engineered feature names."""
        return self.all_feature_names.copy()

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get feature names grouped by type for analysis."""
        return {
            "temporal": self.temporal_extractor.get_feature_names(),
            "insulin": self.insulin_extractor.get_feature_names(),
            "glucose": self.glucose_extractor.get_feature_names(),
            "lag": self.lag_generator.get_feature_names(),
        }
