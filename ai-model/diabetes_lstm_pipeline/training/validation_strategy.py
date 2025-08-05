"""Validation strategies for time-series specific cross-validation."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ValidationStrategy:
    """Implements time-series specific validation approaches for LSTM models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize validation strategy.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.validation_config = config.get("validation", {})
        self.training_config = config.get("training", {})

        # Validation parameters
        self.n_splits = self.validation_config.get("cv_splits", 5)
        self.test_size = self.validation_config.get("test_size", 0.2)
        self.gap = self.validation_config.get(
            "gap", 0
        )  # Gap between train and validation
        self.max_train_size = self.validation_config.get("max_train_size", None)

        # Time-series specific parameters
        self.time_aware = self.validation_config.get("time_aware", True)
        self.participant_aware = self.validation_config.get("participant_aware", True)
        self.min_train_samples = self.validation_config.get("min_train_samples", 1000)

    def train_validation_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        participant_ids: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets with time-series considerations.

        Args:
            X: Input sequences
            y: Target values
            validation_split: Fraction of data to use for validation
            participant_ids: Optional participant identifiers
            timestamps: Optional timestamps for temporal splitting

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        logger.info(f"Splitting data with validation_split={validation_split}")

        n_samples = len(X)

        if self.time_aware and timestamps is not None:
            # Time-aware splitting: use latest data for validation
            split_idx = int(n_samples * (1 - validation_split))

            X_train = X[:split_idx]
            X_val = X[split_idx:]
            y_train = y[:split_idx]
            y_val = y[split_idx:]

            logger.info("Used time-aware splitting")

        elif self.participant_aware and participant_ids is not None:
            # Participant-aware splitting: ensure participants don't overlap
            unique_participants = np.unique(participant_ids)
            n_val_participants = max(
                1, int(len(unique_participants) * validation_split)
            )

            # Select validation participants (use last participants to maintain temporal order)
            val_participants = unique_participants[-n_val_participants:]
            val_mask = np.isin(participant_ids, val_participants)

            X_train = X[~val_mask]
            X_val = X[val_mask]
            y_train = y[~val_mask]
            y_val = y[val_mask]

            logger.info(
                f"Used participant-aware splitting with {n_val_participants} validation participants"
            )

        else:
            # Simple temporal splitting (no shuffling to maintain order)
            split_idx = int(n_samples * (1 - validation_split))

            X_train = X[:split_idx]
            X_val = X[split_idx:]
            y_train = y[:split_idx]
            y_val = y[split_idx:]

            logger.info("Used simple temporal splitting")

        logger.info(f"Split sizes - Train: {len(X_train)}, Validation: {len(X_val)}")

        return X_train, X_val, y_train, y_val

    def time_series_cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_builder_func: callable,
        participant_ids: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation.

        Args:
            X: Input sequences
            y: Target values
            model_builder_func: Function that returns a compiled model
            participant_ids: Optional participant identifiers
            timestamps: Optional timestamps

        Returns:
            Dictionary of cross-validation results
        """
        logger.info(
            f"Starting time-series cross-validation with {self.n_splits} splits"
        )

        # Initialize results storage
        cv_results = {
            "fold_results": [],
            "mean_metrics": {},
            "std_metrics": {},
            "fold_predictions": [],
            "fold_true_values": [],
        }

        if self.participant_aware and participant_ids is not None:
            # Participant-aware cross-validation
            cv_results.update(
                self._participant_aware_cv(X, y, model_builder_func, participant_ids)
            )
        else:
            # Time-series split cross-validation
            cv_results.update(self._time_series_split_cv(X, y, model_builder_func))

        # Calculate summary statistics
        self._calculate_cv_summary(cv_results)

        logger.info("Cross-validation completed")
        return cv_results

    def _time_series_split_cv(
        self, X: np.ndarray, y: np.ndarray, model_builder_func: callable
    ) -> Dict[str, Any]:
        """
        Perform time-series split cross-validation.

        Args:
            X: Input sequences
            y: Target values
            model_builder_func: Function that returns a compiled model

        Returns:
            Dictionary of CV results
        """
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=(
                int(len(X) * self.test_size)
                if self.test_size < 1
                else int(self.test_size)
            ),
            gap=self.gap,
            max_train_size=self.max_train_size,
        )

        fold_results = []
        fold_predictions = []
        fold_true_values = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{self.n_splits}")

            # Split data
            X_train_fold = X[train_idx]
            X_val_fold = X[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]

            # Check minimum training samples
            if len(X_train_fold) < self.min_train_samples:
                logger.warning(
                    f"Fold {fold + 1} has insufficient training samples ({len(X_train_fold)}), skipping"
                )
                continue

            # Build and train model
            model = model_builder_func()

            try:
                # Train model
                history = model.fit(
                    X_train_fold,
                    y_train_fold,
                    batch_size=self.training_config.get("batch_size", 32),
                    epochs=self.training_config.get("cv_epochs", 50),
                    validation_data=(X_val_fold, y_val_fold),
                    verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=10, restore_best_weights=True
                        )
                    ],
                )

                # Make predictions
                y_pred_fold = model.predict(X_val_fold, verbose=0).flatten()
                y_val_fold_flat = y_val_fold.flatten()

                # Calculate fold metrics
                fold_metrics = self._calculate_fold_metrics(
                    y_val_fold_flat, y_pred_fold
                )
                fold_metrics["fold"] = fold + 1
                fold_metrics["train_samples"] = len(X_train_fold)
                fold_metrics["val_samples"] = len(X_val_fold)
                fold_metrics["epochs_trained"] = len(history.history["loss"])

                fold_results.append(fold_metrics)
                fold_predictions.append(y_pred_fold)
                fold_true_values.append(y_val_fold_flat)

                logger.info(
                    f"Fold {fold + 1} completed - Val MAE: {fold_metrics['mae']:.4f}"
                )

            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {str(e)}")
                continue

        return {
            "fold_results": fold_results,
            "fold_predictions": fold_predictions,
            "fold_true_values": fold_true_values,
        }

    def _participant_aware_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_builder_func: callable,
        participant_ids: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Perform participant-aware cross-validation.

        Args:
            X: Input sequences
            y: Target values
            model_builder_func: Function that returns a compiled model
            participant_ids: Participant identifiers

        Returns:
            Dictionary of CV results
        """
        unique_participants = np.unique(participant_ids)
        n_participants = len(unique_participants)

        if n_participants < self.n_splits:
            logger.warning(
                f"Not enough participants ({n_participants}) for {self.n_splits} splits"
            )
            self.n_splits = n_participants

        # Create participant folds
        participants_per_fold = n_participants // self.n_splits
        fold_results = []
        fold_predictions = []
        fold_true_values = []

        for fold in range(self.n_splits):
            logger.info(f"Training participant-aware fold {fold + 1}/{self.n_splits}")

            # Select validation participants for this fold
            start_idx = fold * participants_per_fold
            end_idx = start_idx + participants_per_fold
            if fold == self.n_splits - 1:  # Last fold gets remaining participants
                end_idx = n_participants

            val_participants = unique_participants[start_idx:end_idx]
            val_mask = np.isin(participant_ids, val_participants)

            # Split data
            X_train_fold = X[~val_mask]
            X_val_fold = X[val_mask]
            y_train_fold = y[~val_mask]
            y_val_fold = y[val_mask]

            # Check minimum training samples
            if len(X_train_fold) < self.min_train_samples:
                logger.warning(
                    f"Fold {fold + 1} has insufficient training samples, skipping"
                )
                continue

            # Build and train model
            model = model_builder_func()

            try:
                # Train model
                history = model.fit(
                    X_train_fold,
                    y_train_fold,
                    batch_size=self.training_config.get("batch_size", 32),
                    epochs=self.training_config.get("cv_epochs", 50),
                    validation_data=(X_val_fold, y_val_fold),
                    verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=10, restore_best_weights=True
                        )
                    ],
                )

                # Make predictions
                y_pred_fold = model.predict(X_val_fold, verbose=0).flatten()
                y_val_fold_flat = y_val_fold.flatten()

                # Calculate fold metrics
                fold_metrics = self._calculate_fold_metrics(
                    y_val_fold_flat, y_pred_fold
                )
                fold_metrics["fold"] = fold + 1
                fold_metrics["train_samples"] = len(X_train_fold)
                fold_metrics["val_samples"] = len(X_val_fold)
                fold_metrics["val_participants"] = len(val_participants)
                fold_metrics["epochs_trained"] = len(history.history["loss"])

                fold_results.append(fold_metrics)
                fold_predictions.append(y_pred_fold)
                fold_true_values.append(y_val_fold_flat)

                logger.info(
                    f"Participant fold {fold + 1} completed - Val MAE: {fold_metrics['mae']:.4f}"
                )

            except Exception as e:
                logger.error(f"Error in participant fold {fold + 1}: {str(e)}")
                continue

        return {
            "fold_results": fold_results,
            "fold_predictions": fold_predictions,
            "fold_true_values": fold_true_values,
        }

    def _calculate_fold_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate metrics for a single fold.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of fold metrics
        """
        # Remove invalid values
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]

        if len(y_true_clean) == 0:
            return {
                "mae": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "mard": np.nan,
                "valid_predictions": 0,
            }

        # Basic regression metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)

        # MARD (Mean Absolute Relative Difference)
        mard = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mard": mard,
            "valid_predictions": len(y_true_clean),
        }

    def _calculate_cv_summary(self, cv_results: Dict[str, Any]) -> None:
        """
        Calculate summary statistics for cross-validation results.

        Args:
            cv_results: Cross-validation results dictionary (modified in-place)
        """
        fold_results = cv_results["fold_results"]

        if not fold_results:
            logger.warning("No successful folds to summarize")
            return

        # Extract metrics from all folds
        metrics_by_fold = {}
        for metric in fold_results[0].keys():
            if isinstance(fold_results[0][metric], (int, float)) and not np.isnan(
                fold_results[0][metric]
            ):
                metrics_by_fold[metric] = [
                    fold[metric] for fold in fold_results if not np.isnan(fold[metric])
                ]

        # Calculate mean and std for each metric
        mean_metrics = {}
        std_metrics = {}

        for metric, values in metrics_by_fold.items():
            if values:  # Only if we have valid values
                mean_metrics[metric] = np.mean(values)
                std_metrics[metric] = np.std(values)

        cv_results["mean_metrics"] = mean_metrics
        cv_results["std_metrics"] = std_metrics
        cv_results["n_successful_folds"] = len(fold_results)

    def cross_validate(
        self,
        model: keras.Model,
        X: np.ndarray,
        y: np.ndarray,
        participant_ids: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on a pre-trained model.

        Args:
            model: Pre-trained Keras model
            X: Input sequences
            y: Target values
            participant_ids: Optional participant identifiers
            timestamps: Optional timestamps

        Returns:
            Dictionary of cross-validation results
        """
        logger.info("Performing cross-validation on pre-trained model")

        def model_builder_func():
            # Clone the existing model architecture
            model_config = model.get_config()
            new_model = keras.Model.from_config(model_config)
            new_model.compile(
                optimizer=model.optimizer, loss=model.loss, metrics=model.metrics
            )
            return new_model

        return self.time_series_cross_validate(
            X, y, model_builder_func, participant_ids, timestamps
        )

    def walk_forward_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_builder_func: callable,
        initial_train_size: float = 0.5,
        step_size: int = 100,
        prediction_horizon: int = 1,
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation for time-series data.

        Args:
            X: Input sequences
            y: Target values
            model_builder_func: Function that returns a compiled model
            initial_train_size: Initial fraction of data to use for training
            step_size: Number of samples to add in each step
            prediction_horizon: Number of steps ahead to predict

        Returns:
            Dictionary of walk-forward validation results
        """
        logger.info("Starting walk-forward validation")

        n_samples = len(X)
        initial_train_samples = int(n_samples * initial_train_size)

        results = {
            "step_results": [],
            "predictions": [],
            "true_values": [],
            "step_metrics": [],
        }

        current_train_end = initial_train_samples

        while current_train_end + prediction_horizon < n_samples:
            # Define training and prediction windows
            train_start = 0
            train_end = current_train_end
            pred_start = train_end
            pred_end = min(train_end + prediction_horizon, n_samples)

            # Extract data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_pred = X[pred_start:pred_end]
            y_true = y[pred_start:pred_end]

            # Build and train model
            model = model_builder_func()

            try:
                # Train model
                model.fit(
                    X_train,
                    y_train,
                    batch_size=self.training_config.get("batch_size", 32),
                    epochs=self.training_config.get("wf_epochs", 20),
                    verbose=0,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor="loss", patience=5, restore_best_weights=True
                        )
                    ],
                )

                # Make predictions
                y_pred = model.predict(X_pred, verbose=0).flatten()
                y_true_flat = y_true.flatten()

                # Calculate step metrics
                step_metrics = self._calculate_fold_metrics(y_true_flat, y_pred)
                step_metrics["step"] = len(results["step_results"]) + 1
                step_metrics["train_end"] = train_end
                step_metrics["pred_start"] = pred_start
                step_metrics["pred_end"] = pred_end

                results["step_results"].append(step_metrics)
                results["predictions"].extend(y_pred)
                results["true_values"].extend(y_true_flat)
                results["step_metrics"].append(step_metrics)

                logger.info(
                    f"Walk-forward step {step_metrics['step']} - MAE: {step_metrics['mae']:.4f}"
                )

            except Exception as e:
                logger.error(f"Error in walk-forward step: {str(e)}")

            # Move to next step
            current_train_end += step_size

        # Calculate overall metrics
        if results["predictions"] and results["true_values"]:
            overall_metrics = self._calculate_fold_metrics(
                np.array(results["true_values"]), np.array(results["predictions"])
            )
            results["overall_metrics"] = overall_metrics

        logger.info("Walk-forward validation completed")
        return results
