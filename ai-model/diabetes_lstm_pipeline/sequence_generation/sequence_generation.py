"""
Sequence generation module for creating time-series sequences for LSTM training.

This module contains classes for:
- Creating input-output sequence pairs with configurable sequence length
- Maintaining participant boundaries during data splitting
- Ensuring temporal ordering and sequence integrity
- Handling irregular timestamps through interpolation and resampling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class SequenceGenerator:
    """Creates input-output sequence pairs for LSTM training."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sequence generator.

        Args:
            config: Configuration dictionary with sequence generation settings
        """
        self.config = config
        self.sequence_length = config.get("sequence_length", 60)  # minutes
        self.prediction_horizon = config.get("prediction_horizon", 1)  # steps ahead
        self.target_column = config.get("target_column", "CGM")
        self.feature_columns = config.get("feature_columns", [])
        self.participant_column = config.get("participant_column", "participant_id")
        self.time_column = config.get("time_column", "EventDateTime")
        self.sampling_interval = config.get("sampling_interval", 5)  # minutes

        # Calculate sequence length in data points
        self.sequence_length_points = self.sequence_length // self.sampling_interval

        logger.info(
            f"Initialized SequenceGenerator with sequence_length={self.sequence_length} minutes "
            f"({self.sequence_length_points} points), prediction_horizon={self.prediction_horizon}"
        )

    def generate_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate input-output sequence pairs from the dataframe.

        Args:
            df: DataFrame with time series data

        Returns:
            Tuple of (X, y, participant_ids) where:
            - X: Input sequences of shape (n_sequences, sequence_length, n_features)
            - y: Target values of shape (n_sequences,)
            - participant_ids: Participant IDs for each sequence
        """
        logger.info("Generating sequences from dataframe")

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in dataframe"
            )

        if self.participant_column not in df.columns:
            raise ValueError(
                f"Participant column '{self.participant_column}' not found in dataframe"
            )

        # Ensure data is sorted by participant and time
        df = df.sort_values([self.participant_column, self.time_column])

        # Auto-detect feature columns if not specified
        if not self.feature_columns:
            exclude_columns = [self.time_column, self.participant_column]
            # Only include numeric columns for features
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            self.feature_columns = [
                col for col in numeric_columns if col not in exclude_columns
            ]

        logger.info(
            f"Using {len(self.feature_columns)} feature columns for sequence generation"
        )

        sequences_X = []
        sequences_y = []
        sequence_participant_ids = []

        # Generate sequences for each participant separately
        participants = df[self.participant_column].unique()

        for participant_id in participants:
            participant_data = df[df[self.participant_column] == participant_id].copy()
            participant_data = participant_data.reset_index(drop=True)

            # Generate sequences for this participant
            X_part, y_part, ids_part = self._generate_participant_sequences(
                participant_data, participant_id
            )

            if len(X_part) > 0:
                sequences_X.extend(X_part)
                sequences_y.extend(y_part)
                sequence_participant_ids.extend(ids_part)

        if len(sequences_X) == 0:
            raise ValueError("No valid sequences could be generated from the data")

        # Convert to numpy arrays
        X = np.array(sequences_X)
        y = np.array(sequences_y)
        participant_ids = np.array(sequence_participant_ids)

        logger.info(f"Generated {len(X)} sequences with shape {X.shape}")

        return X, y, participant_ids

    def _generate_participant_sequences(
        self, participant_data: pd.DataFrame, participant_id: str
    ) -> Tuple[List[np.ndarray], List[float], List[str]]:
        """Generate sequences for a single participant."""
        sequences_X = []
        sequences_y = []
        sequence_ids = []

        # Extract feature matrix and target vector
        feature_matrix = participant_data[self.feature_columns].values
        target_vector = participant_data[self.target_column].values

        # Generate sequences with sliding window
        for i in range(
            len(participant_data)
            - self.sequence_length_points
            - self.prediction_horizon
            + 1
        ):
            # Input sequence
            X_seq = feature_matrix[i : i + self.sequence_length_points]

            # Target value (prediction_horizon steps ahead)
            y_val = target_vector[
                i + self.sequence_length_points + self.prediction_horizon - 1
            ]

            # Check for missing values in sequence
            if not self._is_valid_sequence(X_seq, y_val):
                continue

            sequences_X.append(X_seq)
            sequences_y.append(y_val)
            sequence_ids.append(participant_id)

        return sequences_X, sequences_y, sequence_ids

    def _is_valid_sequence(self, X_seq: np.ndarray, y_val: float) -> bool:
        """Check if a sequence is valid (no missing values, reasonable target)."""
        # Check for NaN values in input sequence (only for numeric data)
        try:
            if X_seq.dtype.kind in "biufc":  # numeric types
                if np.isnan(X_seq).any():
                    return False
            else:
                # For mixed or object arrays, check for None/NaN differently
                if pd.isna(X_seq).any():
                    return False
        except (TypeError, ValueError):
            # If we can't check for NaN, assume it's valid
            pass

        # Check for NaN in target
        if np.isnan(y_val):
            return False

        # Check for reasonable glucose values (if target is CGM)
        if self.target_column == "CGM":
            if y_val < 20 or y_val > 600:  # Physiologically impossible values
                return False

        return True

    def get_sequence_info(self) -> Dict[str, Any]:
        """Get information about sequence generation configuration."""
        return {
            "sequence_length_minutes": self.sequence_length,
            "sequence_length_points": self.sequence_length_points,
            "prediction_horizon": self.prediction_horizon,
            "target_column": self.target_column,
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns.copy(),
            "sampling_interval": self.sampling_interval,
        }


class ParticipantSplitter:
    """Maintains participant boundaries during data splitting."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize participant splitter.

        Args:
            config: Configuration dictionary with splitting settings
        """
        self.config = config
        self.train_ratio = config.get("train_ratio", 0.7)
        self.val_ratio = config.get("val_ratio", 0.15)
        self.test_ratio = config.get("test_ratio", 0.15)
        self.random_seed = config.get("random_seed", 42)
        self.split_strategy = config.get(
            "split_strategy", "participant"
        )  # 'participant' or 'temporal'

        # Validate ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

        logger.info(
            f"Initialized ParticipantSplitter with ratios: "
            f"train={self.train_ratio}, val={self.val_ratio}, test={self.test_ratio}"
        )

    def split_sequences(
        self, X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split sequences into train/validation/test sets while maintaining participant boundaries.

        Args:
            X: Input sequences of shape (n_sequences, sequence_length, n_features)
            y: Target values of shape (n_sequences,)
            participant_ids: Participant IDs for each sequence

        Returns:
            Dictionary with keys 'train', 'val', 'test', each containing (X, y) tuples
        """
        logger.info(
            f"Splitting {len(X)} sequences using {self.split_strategy} strategy"
        )

        if self.split_strategy == "participant":
            return self._split_by_participant(X, y, participant_ids)
        elif self.split_strategy == "temporal":
            return self._split_by_time(X, y, participant_ids)
        else:
            raise ValueError(f"Unknown split strategy: {self.split_strategy}")

    def _split_by_participant(
        self, X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data by participant to avoid data leakage."""
        unique_participants = np.unique(participant_ids)
        n_participants = len(unique_participants)

        logger.info(f"Splitting {n_participants} participants")

        # Calculate number of participants for each split
        n_train = int(n_participants * self.train_ratio)
        n_val = int(n_participants * self.val_ratio)
        n_test = n_participants - n_train - n_val

        # Randomly shuffle participants
        np.random.seed(self.random_seed)
        shuffled_participants = np.random.permutation(unique_participants)

        # Assign participants to splits
        train_participants = set(shuffled_participants[:n_train])
        val_participants = set(shuffled_participants[n_train : n_train + n_val])
        test_participants = set(shuffled_participants[n_train + n_val :])

        # Create masks for each split
        train_mask = np.array([pid in train_participants for pid in participant_ids])
        val_mask = np.array([pid in val_participants for pid in participant_ids])
        test_mask = np.array([pid in test_participants for pid in participant_ids])

        # Split the data
        splits = {
            "train": (X[train_mask], y[train_mask]),
            "val": (X[val_mask], y[val_mask]),
            "test": (X[test_mask], y[test_mask]),
        }

        # Log split statistics
        for split_name, (X_split, y_split) in splits.items():
            unique_pids = len(
                np.unique(
                    participant_ids[
                        (
                            train_mask
                            if split_name == "train"
                            else val_mask if split_name == "val" else test_mask
                        )
                    ]
                )
            )
            logger.info(
                f"{split_name}: {len(X_split)} sequences from {unique_pids} participants"
            )

        return splits

    def _split_by_time(
        self, X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data temporally (earlier data for training, later for testing)."""
        # For temporal splitting, we assume sequences are already ordered by time
        n_sequences = len(X)

        # Calculate split indices
        train_end = int(n_sequences * self.train_ratio)
        val_end = int(n_sequences * (self.train_ratio + self.val_ratio))

        splits = {
            "train": (X[:train_end], y[:train_end]),
            "val": (X[train_end:val_end], y[train_end:val_end]),
            "test": (X[val_end:], y[val_end:]),
        }

        # Log split statistics
        for split_name, (X_split, y_split) in splits.items():
            logger.info(f"{split_name}: {len(X_split)} sequences")

        return splits

    def get_split_info(
        self, splits: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Any]:
        """Get information about the data splits."""
        info = {
            "split_strategy": self.split_strategy,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "splits": {},
        }

        for split_name, (X_split, y_split) in splits.items():
            if len(y_split) > 0:
                info["splits"][split_name] = {
                    "n_sequences": len(X_split),
                    "sequence_shape": X_split.shape,
                    "target_mean": float(np.mean(y_split)),
                    "target_std": float(np.std(y_split)),
                    "target_min": float(np.min(y_split)),
                    "target_max": float(np.max(y_split)),
                }
            else:
                info["splits"][split_name] = {
                    "n_sequences": 0,
                    "sequence_shape": X_split.shape,
                    "target_mean": 0.0,
                    "target_std": 0.0,
                    "target_min": 0.0,
                    "target_max": 0.0,
                }

        return info


class SequenceValidator:
    """Ensures temporal ordering and sequence integrity."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sequence validator.

        Args:
            config: Configuration dictionary with validation settings
        """
        self.config = config
        self.time_column = config.get("time_column", "EventDateTime")
        self.participant_column = config.get("participant_column", "participant_id")
        self.target_column = config.get("target_column", "CGM")
        self.max_time_gap_minutes = config.get("max_time_gap_minutes", 15)
        self.min_sequence_length = config.get("min_sequence_length", 10)

        logger.info("Initialized SequenceValidator")

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate dataframe before sequence generation.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results and statistics
        """
        logger.info("Validating dataframe for sequence generation")

        validation_results = {
            "is_valid": True,
            "issues": [],
            "statistics": {},
            "recommendations": [],
        }

        # Check required columns
        required_columns = [
            self.time_column,
            self.participant_column,
            self.target_column,
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            validation_results["is_valid"] = False
            validation_results["issues"].append(
                f"Missing required columns: {missing_columns}"
            )

        # Check temporal ordering
        temporal_issues = self._check_temporal_ordering(df)
        if temporal_issues:
            validation_results["issues"].extend(temporal_issues)

        # Check data gaps
        gap_issues = self._check_data_gaps(df)
        if gap_issues:
            validation_results["issues"].extend(gap_issues)

        # Check data quality
        quality_issues = self._check_data_quality(df)
        if quality_issues:
            validation_results["issues"].extend(quality_issues)

        # Calculate statistics
        validation_results["statistics"] = self._calculate_statistics(df)

        # Generate recommendations
        validation_results["recommendations"] = self._generate_recommendations(
            validation_results["issues"], validation_results["statistics"]
        )

        logger.info(
            f"Validation complete. Found {len(validation_results['issues'])} issues"
        )

        return validation_results

    def validate_sequences(
        self, X: np.ndarray, y: np.ndarray, participant_ids: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate generated sequences.

        Args:
            X: Input sequences
            y: Target values
            participant_ids: Participant IDs

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {len(X)} generated sequences")

        validation_results = {
            "is_valid": True,
            "issues": [],
            "statistics": {},
            "quality_metrics": {},
        }

        # Check array shapes and consistency
        if len(X) != len(y) or len(X) != len(participant_ids):
            validation_results["is_valid"] = False
            validation_results["issues"].append("Inconsistent array lengths")

        # Check for NaN values
        nan_sequences = np.isnan(X).any(axis=(1, 2))
        nan_targets = np.isnan(y)

        if nan_sequences.any():
            n_nan_seq = np.sum(nan_sequences)
            validation_results["issues"].append(
                f"{n_nan_seq} sequences contain NaN values"
            )

        if nan_targets.any():
            n_nan_targets = np.sum(nan_targets)
            validation_results["issues"].append(
                f"{n_nan_targets} targets contain NaN values"
            )

        # Check target value ranges (for glucose)
        if self.target_column == "CGM":
            invalid_targets = (y < 20) | (y > 600)
            if invalid_targets.any():
                n_invalid = np.sum(invalid_targets)
                validation_results["issues"].append(
                    f"{n_invalid} targets have physiologically impossible values"
                )

        # Calculate sequence statistics
        validation_results["statistics"] = {
            "n_sequences": len(X),
            "sequence_shape": X.shape,
            "n_participants": len(np.unique(participant_ids)),
            "target_stats": {
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "median": float(np.median(y)),
            },
        }

        # Calculate quality metrics
        validation_results["quality_metrics"] = (
            self._calculate_sequence_quality_metrics(X, y)
        )

        if len(validation_results["issues"]) == 0:
            logger.info("All sequences passed validation")
        else:
            logger.warning(
                f"Found {len(validation_results['issues'])} validation issues"
            )

        return validation_results

    def _check_temporal_ordering(self, df: pd.DataFrame) -> List[str]:
        """Check if data is properly ordered by time within each participant."""
        issues = []

        if self.time_column not in df.columns:
            return issues

        if self.participant_column not in df.columns:
            return issues

        # Convert to datetime if not already
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])

        # Check ordering within each participant
        for participant_id in df[self.participant_column].unique():
            participant_data = df[df[self.participant_column] == participant_id]

            # Check if timestamps are monotonically increasing
            timestamps = participant_data[self.time_column].values
            if not np.all(timestamps[:-1] <= timestamps[1:]):
                issues.append(
                    f"Participant {participant_id} has non-monotonic timestamps"
                )

        return issues

    def _check_data_gaps(self, df: pd.DataFrame) -> List[str]:
        """Check for large gaps in the time series data."""
        issues = []

        if self.time_column not in df.columns:
            return issues

        if self.participant_column not in df.columns:
            return issues

        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])

        # Check gaps within each participant
        for participant_id in df[self.participant_column].unique():
            participant_data = df[
                df[self.participant_column] == participant_id
            ].sort_values(self.time_column)

            # Calculate time differences
            time_diffs = participant_data[self.time_column].diff()
            large_gaps = time_diffs > pd.Timedelta(minutes=self.max_time_gap_minutes)

            if large_gaps.any():
                n_gaps = large_gaps.sum()
                max_gap = time_diffs.max()
                issues.append(
                    f"Participant {participant_id} has {n_gaps} gaps > {self.max_time_gap_minutes} minutes "
                    f"(max gap: {max_gap})"
                )

        return issues

    def _check_data_quality(self, df: pd.DataFrame) -> List[str]:
        """Check data quality issues that might affect sequence generation."""
        issues = []

        # Check for missing target values
        if self.target_column in df.columns:
            missing_targets = df[self.target_column].isna().sum()
            if missing_targets > 0:
                missing_pct = (missing_targets / len(df)) * 100
                issues.append(
                    f"{missing_targets} ({missing_pct:.1f}%) missing target values"
                )

        # Check for participants with insufficient data
        if self.participant_column in df.columns:
            participant_counts = df[self.participant_column].value_counts()
            insufficient_data = participant_counts < self.min_sequence_length

            if insufficient_data.any():
                n_insufficient = insufficient_data.sum()
                issues.append(
                    f"{n_insufficient} participants have < {self.min_sequence_length} data points"
                )

        return issues

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics about the dataframe."""
        stats = {
            "n_records": len(df),
            "n_participants": (
                df[self.participant_column].nunique()
                if self.participant_column in df.columns
                else 0
            ),
            "date_range": {},
            "participant_stats": {},
        }

        # Date range statistics
        if self.time_column in df.columns:
            df_time = df.copy()
            df_time[self.time_column] = pd.to_datetime(df_time[self.time_column])

            stats["date_range"] = {
                "start": df_time[self.time_column].min().isoformat(),
                "end": df_time[self.time_column].max().isoformat(),
                "duration_days": (
                    df_time[self.time_column].max() - df_time[self.time_column].min()
                ).days,
            }

        # Participant statistics
        if self.participant_column in df.columns:
            participant_counts = df[self.participant_column].value_counts()
            stats["participant_stats"] = {
                "mean_records_per_participant": float(participant_counts.mean()),
                "std_records_per_participant": float(participant_counts.std()),
                "min_records_per_participant": int(participant_counts.min()),
                "max_records_per_participant": int(participant_counts.max()),
            }

        return stats

    def _calculate_sequence_quality_metrics(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, float]:
        """Calculate quality metrics for generated sequences."""
        metrics = {}

        # Feature completeness (percentage of non-NaN values)
        metrics["feature_completeness"] = float(1 - np.isnan(X).mean())

        # Target completeness
        metrics["target_completeness"] = float(1 - np.isnan(y).mean())

        # Feature variability (mean standard deviation across features)
        feature_stds = np.nanstd(X.reshape(-1, X.shape[-1]), axis=0)
        metrics["mean_feature_variability"] = float(np.mean(feature_stds))

        # Target variability
        metrics["target_variability"] = float(np.nanstd(y))

        return metrics

    def _generate_recommendations(
        self, issues: List[str], statistics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation issues and statistics."""
        recommendations = []

        # Recommendations based on issues
        if any("missing target values" in issue for issue in issues):
            recommendations.append(
                "Consider imputing missing target values or filtering out incomplete records"
            )

        if any("gaps" in issue for issue in issues):
            recommendations.append(
                "Consider interpolating data to fill time gaps or adjusting max_time_gap_minutes"
            )

        if any("insufficient data" in issue for issue in issues):
            recommendations.append(
                "Consider removing participants with insufficient data or reducing min_sequence_length"
            )

        # Recommendations based on statistics
        if "participant_stats" in statistics:
            min_records = statistics["participant_stats"].get(
                "min_records_per_participant", 0
            )
            if min_records < 50:
                recommendations.append(
                    "Some participants have very few records, consider data quality filtering"
                )

        return recommendations


class TimeSeriesResampler:
    """Handles irregular timestamps through interpolation and resampling."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize time series resampler.

        Args:
            config: Configuration dictionary with resampling settings
        """
        self.config = config
        self.target_frequency = config.get("target_frequency", "5min")
        self.interpolation_method = config.get("interpolation_method", "linear")
        self.max_gap_minutes = config.get("max_gap_minutes", 30)
        self.participant_column = config.get("participant_column", "participant_id")
        self.time_column = config.get("time_column", "EventDateTime")

        logger.info(
            f"Initialized TimeSeriesResampler with frequency={self.target_frequency}"
        )

    def resample_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample dataframe to uniform time intervals.

        Args:
            df: DataFrame with irregular timestamps

        Returns:
            DataFrame with uniform time intervals
        """
        logger.info(f"Resampling dataframe to {self.target_frequency} intervals")

        if self.time_column not in df.columns:
            raise ValueError(f"Time column '{self.time_column}' not found in dataframe")

        # Convert time column to datetime
        df = df.copy()
        df[self.time_column] = pd.to_datetime(df[self.time_column])

        resampled_dfs = []

        # Resample each participant separately
        for participant_id in df[self.participant_column].unique():
            participant_data = df[df[self.participant_column] == participant_id].copy()

            # Resample this participant's data
            resampled_participant = self._resample_participant_data(
                participant_data, participant_id
            )

            if len(resampled_participant) > 0:
                resampled_dfs.append(resampled_participant)

        if not resampled_dfs:
            raise ValueError("No data could be resampled")

        # Combine all resampled data
        resampled_df = pd.concat(resampled_dfs, ignore_index=True)

        logger.info(
            f"Resampling complete. Original: {len(df)} records, Resampled: {len(resampled_df)} records"
        )

        return resampled_df

    def _resample_participant_data(
        self, participant_data: pd.DataFrame, participant_id: str
    ) -> pd.DataFrame:
        """Resample data for a single participant."""
        # Set time as index for resampling
        participant_data = participant_data.set_index(self.time_column).sort_index()

        # Define aggregation methods for different column types
        agg_methods = self._get_aggregation_methods(participant_data.columns)

        # Filter aggregation methods to only include columns that exist
        agg_methods = {
            col: method
            for col, method in agg_methods.items()
            if col in participant_data.columns
        }

        # Resample using the target frequency
        resampled = participant_data.resample(self.target_frequency).agg(agg_methods)

        # Interpolate missing values
        resampled = self._interpolate_missing_values(resampled)

        # Reset index and add participant column back
        resampled = resampled.reset_index()
        resampled[self.participant_column] = participant_id

        return resampled

    def _get_aggregation_methods(self, columns: List[str]) -> Dict[str, str]:
        """Get appropriate aggregation methods for different column types."""
        agg_methods = {}

        for col in columns:
            if col == self.participant_column:
                continue
            elif "CGM" in col or "glucose" in col.lower():
                agg_methods[col] = "mean"
            elif (
                "insulin" in col.lower()
                or "bolus" in col.lower()
                or "correction" in col.lower()
            ):
                agg_methods[col] = "sum"
            elif "basal" in col.lower():
                agg_methods[col] = "mean"
            elif "food" in col.lower() or "carb" in col.lower():
                agg_methods[col] = "sum"
            elif col.startswith("time_since"):
                agg_methods[col] = "mean"
            elif col.startswith("is_") or col.endswith("_binary"):
                agg_methods[col] = "max"  # Binary flags
            elif (
                col.lower() in ["devicemode", "bolustype"]
                or "mode" in col.lower()
                or "type" in col.lower()
                or "description" in col.lower()
                or "serial" in col.lower()
                or "device" in col.lower()
                or "tandem" in col.lower()
            ):
                agg_methods[col] = "first"  # String columns - take first value
            elif col.startswith("Unnamed:"):
                agg_methods[col] = "first"  # Handle unnamed columns
            else:
                # Default to first for unknown columns to avoid aggregation errors
                agg_methods[col] = "first"

        return agg_methods

    def _interpolate_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values in the resampled data."""
        df = df.copy()

        # Interpolate based on column type
        for col in df.columns:
            if col == self.participant_column:
                continue

            # Calculate maximum gap size for interpolation
            max_gap_periods = self._calculate_max_gap_periods()

            if self.interpolation_method == "linear":
                df[col] = df[col].interpolate(method="linear", limit=max_gap_periods)
            elif self.interpolation_method == "forward_fill":
                df[col] = df[col].ffill(limit=max_gap_periods)
            elif self.interpolation_method == "backward_fill":
                df[col] = df[col].bfill(limit=max_gap_periods)
            else:
                # Default to linear interpolation
                df[col] = df[col].interpolate(method="linear", limit=max_gap_periods)

        return df

    def _calculate_max_gap_periods(self) -> int:
        """Calculate maximum number of periods to interpolate based on max_gap_minutes."""
        # Parse target frequency to get minutes per period
        freq_str = self.target_frequency.lower()

        if freq_str.endswith("min"):
            minutes_per_period = int(freq_str[:-3])
        elif freq_str.endswith("h"):
            minutes_per_period = int(freq_str[:-1]) * 60
        else:
            minutes_per_period = 5  # Default assumption

        max_gap_periods = self.max_gap_minutes // minutes_per_period
        return max(1, max_gap_periods)

    def get_resampling_info(self) -> Dict[str, Any]:
        """Get information about resampling configuration."""
        return {
            "target_frequency": self.target_frequency,
            "interpolation_method": self.interpolation_method,
            "max_gap_minutes": self.max_gap_minutes,
            "max_gap_periods": self._calculate_max_gap_periods(),
        }


class SequenceGenerationPipeline:
    """Main orchestrator for the sequence generation pipeline."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize sequence generation pipeline.

        Args:
            config: Configuration dictionary for all components
        """
        self.config = config

        # Initialize components
        self.resampler = TimeSeriesResampler(config.get("resampling", {}))
        self.validator = SequenceValidator(config.get("validation", {}))
        self.sequence_generator = SequenceGenerator(
            config.get("sequence_generation", {})
        )
        self.participant_splitter = ParticipantSplitter(config.get("splitting", {}))

        logger.info("Initialized SequenceGenerationPipeline")

    def generate_sequences_from_dataframe(
        self, df: pd.DataFrame, validate_input: bool = True, resample_data: bool = True
    ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Any]]:
        """
        Generate train/val/test sequences from a dataframe.

        Args:
            df: Input dataframe with time series data
            validate_input: Whether to validate input data
            resample_data: Whether to resample data to uniform intervals

        Returns:
            Tuple of (data_splits, metadata) where:
            - data_splits: Dictionary with 'train', 'val', 'test' keys
            - metadata: Dictionary with pipeline information and statistics
        """
        logger.info("Starting sequence generation pipeline")

        metadata = {
            "pipeline_config": self.config,
            "input_data_shape": df.shape,
            "validation_results": {},
            "resampling_info": {},
            "sequence_info": {},
            "split_info": {},
        }

        # Step 1: Validate input data
        if validate_input:
            logger.info("Step 1: Validating input data")
            validation_results = self.validator.validate_dataframe(df)
            metadata["validation_results"] = validation_results

            if not validation_results["is_valid"]:
                logger.warning(
                    "Input data validation failed, but continuing with processing"
                )

        # Step 2: Resample data to uniform intervals
        processed_df = df
        if resample_data:
            logger.info("Step 2: Resampling data to uniform intervals")
            processed_df = self.resampler.resample_dataframe(df)
            metadata["resampling_info"] = self.resampler.get_resampling_info()
            metadata["resampled_data_shape"] = processed_df.shape

        # Step 3: Generate sequences
        logger.info("Step 3: Generating sequences")
        X, y, participant_ids = self.sequence_generator.generate_sequences(processed_df)
        metadata["sequence_info"] = self.sequence_generator.get_sequence_info()

        # Step 4: Validate generated sequences
        logger.info("Step 4: Validating generated sequences")
        sequence_validation = self.validator.validate_sequences(X, y, participant_ids)
        metadata["sequence_validation"] = sequence_validation

        # Step 5: Split sequences into train/val/test
        logger.info("Step 5: Splitting sequences into train/val/test")
        data_splits = self.participant_splitter.split_sequences(X, y, participant_ids)
        metadata["split_info"] = self.participant_splitter.get_split_info(data_splits)

        logger.info("Sequence generation pipeline complete")

        return data_splits, metadata

    def save_sequences(
        self,
        data_splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
        output_dir: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Save generated sequences to disk.

        Args:
            data_splits: Dictionary with train/val/test splits
            output_dir: Directory to save sequences
            metadata: Pipeline metadata to save
        """
        import os
        import json

        os.makedirs(output_dir, exist_ok=True)

        # Save sequences
        for split_name, (X, y) in data_splits.items():
            np.save(os.path.join(output_dir, f"X_{split_name}.npy"), X)
            np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y)

        # Save metadata
        with open(os.path.join(output_dir, "sequence_metadata.json"), "w") as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_metadata = self._make_json_serializable(metadata)
            json.dump(serializable_metadata, f, indent=2)

        logger.info(f"Sequences and metadata saved to {output_dir}")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
