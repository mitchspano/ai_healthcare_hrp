"""Clinical metrics for diabetes glucose prediction evaluation."""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ClinicalMetrics:
    """
    Implements clinical evaluation metrics for glucose prediction models.

    This class provides methods to calculate clinically relevant metrics
    including MARD, time-in-range accuracy, and hypoglycemia/hyperglycemia detection.
    """

    def __init__(
        self,
        target_range: Tuple[float, float] = (70.0, 180.0),
        hypoglycemia_threshold: float = 70.0,
        hyperglycemia_threshold: float = 250.0,
    ):
        """
        Initialize clinical metrics calculator.

        Args:
            target_range: Target glucose range (mg/dL) for time-in-range calculations
            hypoglycemia_threshold: Threshold for hypoglycemia detection (mg/dL)
            hyperglycemia_threshold: Threshold for hyperglycemia detection (mg/dL)
        """
        self.target_range = target_range
        self.hypoglycemia_threshold = hypoglycemia_threshold
        self.hyperglycemia_threshold = hyperglycemia_threshold

    def calculate_mard(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Relative Difference (MARD).

        MARD is the gold standard metric for CGM accuracy evaluation.

        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)

        Returns:
            MARD percentage
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")

        # Remove any zero or negative reference values to avoid division by zero
        valid_mask = y_true > 0
        if not valid_mask.any():
            raise ValueError("No valid reference values (all values <= 0)")

        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        # Calculate absolute relative differences
        relative_differences = np.abs(y_pred_valid - y_true_valid) / y_true_valid

        # MARD is the mean of relative differences, expressed as percentage
        mard = np.mean(relative_differences) * 100

        logger.info(f"MARD calculated: {mard:.2f}% (n={len(y_true_valid)} valid pairs)")

        return mard

    def calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y_pred - y_true))

    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return np.sqrt(np.mean((y_pred - y_true) ** 2))

    def calculate_time_in_range_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate time-in-range prediction accuracy.

        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)

        Returns:
            Dictionary with TIR accuracy metrics
        """
        print("DEBUG: calculate_time_in_range_accuracy called")
        print("DEBUG: self.target_range:", self.target_range)
        print("DEBUG: self.target_range type:", type(self.target_range))
        try:
            # Calculate actual time-in-range
            print("DEBUG: About to calculate true_in_range")
            true_in_range = (y_true >= self.target_range[0]) & (
                y_true <= self.target_range[1]
            )
            print("DEBUG: true_in_range calculated")

            print("DEBUG: About to calculate pred_in_range")
            pred_in_range = (y_pred >= self.target_range[0]) & (
                y_pred <= self.target_range[1]
            )
            print("DEBUG: pred_in_range calculated")

            # Calculate TIR percentages
            print("DEBUG: About to calculate TIR percentages")
            true_tir_percent = np.mean(true_in_range) * 100
            pred_tir_percent = np.mean(pred_in_range) * 100
            print("DEBUG: TIR percentages calculated")

            # Calculate TIR prediction accuracy (how well we predict in-range vs out-of-range)
            print("DEBUG: About to calculate TIR accuracy")
            tir_accuracy = np.mean((true_in_range == pred_in_range).astype(float)) * 100
            print("DEBUG: TIR accuracy calculated")

            # Calculate sensitivity and specificity for TIR prediction
            print("DEBUG: About to calculate confusion matrix elements")
            true_positives = float(np.sum(true_in_range & pred_in_range))
            false_positives = float(np.sum(~true_in_range & pred_in_range))
            true_negatives = float(np.sum(~true_in_range & ~pred_in_range))
            false_negatives = float(np.sum(true_in_range & ~pred_in_range))
            print("DEBUG: Confusion matrix elements calculated")

            print("DEBUG: About to calculate sensitivity and specificity")
            sensitivity = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )
            specificity = (
                true_negatives / (true_negatives + false_positives)
                if (true_negatives + false_positives) > 0
                else 0
            )
            print("DEBUG: Sensitivity and specificity calculated")

            print("DEBUG: About to create return dictionary")
            result = {
                "true_tir_percent": true_tir_percent,
                "predicted_tir_percent": pred_tir_percent,
                "tir_prediction_accuracy": tir_accuracy,
                "tir_sensitivity": sensitivity * 100,
                "tir_specificity": specificity * 100,
                "tir_absolute_error": abs(true_tir_percent - pred_tir_percent),
            }
            print("DEBUG: Return dictionary created:", result)
            return result
        except Exception as e:
            print("DEBUG: Exception in calculate_time_in_range_accuracy:", e)
            return 0

    def detect_hypoglycemia_events(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate hypoglycemia detection accuracy.

        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)

        Returns:
            Dictionary with hypoglycemia detection metrics
        """
        true_hypo = y_true < self.hypoglycemia_threshold
        pred_hypo = y_pred < self.hypoglycemia_threshold

        # Calculate confusion matrix elements
        true_positives = float(np.sum(true_hypo & pred_hypo))
        false_positives = float(np.sum(~true_hypo & pred_hypo))
        true_negatives = float(np.sum(~true_hypo & ~pred_hypo))
        false_negatives = float(np.sum(true_hypo & ~pred_hypo))

        # Calculate metrics
        sensitivity = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        specificity = (
            true_negatives / (true_negatives + false_positives)
            if (true_negatives + false_positives) > 0
            else 0
        )
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        accuracy = (true_positives + true_negatives) / len(y_true)

        # F1 score
        f1_score = (
            2 * (precision * sensitivity) / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0
        )

        return {
            "hypoglycemia_sensitivity": sensitivity * 100,
            "hypoglycemia_specificity": specificity * 100,
            "hypoglycemia_precision": precision * 100,
            "hypoglycemia_accuracy": accuracy * 100,
            "hypoglycemia_f1_score": f1_score * 100,
            "true_hypoglycemia_rate": np.mean(true_hypo) * 100,
            "predicted_hypoglycemia_rate": np.mean(pred_hypo) * 100,
        }

    def detect_hyperglycemia_events(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate hyperglycemia detection accuracy.

        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)

        Returns:
            Dictionary with hyperglycemia detection metrics
        """
        true_hyper = y_true > self.hyperglycemia_threshold
        pred_hyper = y_pred > self.hyperglycemia_threshold

        # Calculate confusion matrix elements
        true_positives = float(np.sum(true_hyper & pred_hyper))
        false_positives = float(np.sum(~true_hyper & pred_hyper))
        true_negatives = float(np.sum(~true_hyper & ~pred_hyper))
        false_negatives = float(np.sum(true_hyper & ~pred_hyper))

        # Calculate metrics
        sensitivity = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        specificity = (
            true_negatives / (true_negatives + false_positives)
            if (true_negatives + false_positives) > 0
            else 0
        )
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        accuracy = (true_positives + true_negatives) / len(y_true)

        # F1 score
        f1_score = (
            2 * (precision * sensitivity) / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0
        )

        return {
            "hyperglycemia_sensitivity": sensitivity * 100,
            "hyperglycemia_specificity": specificity * 100,
            "hyperglycemia_precision": precision * 100,
            "hyperglycemia_accuracy": accuracy * 100,
            "hyperglycemia_f1_score": f1_score * 100,
            "true_hyperglycemia_rate": np.mean(true_hyper) * 100,
            "predicted_hyperglycemia_rate": np.mean(pred_hyper) * 100,
        }

    def calculate_all_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all clinical metrics.

        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)

        Returns:
            Dictionary containing all clinical metrics
        """
        print("DEBUG: calculate_all_metrics called")
        metrics = {}

        # Basic metrics
        print("DEBUG: About to calculate mard")
        metrics["mard"] = self.calculate_mard(y_true, y_pred)
        print("DEBUG: mard calculated:", metrics["mard"])

        print("DEBUG: About to calculate mae")
        metrics["mae"] = self.calculate_mae(y_true, y_pred)
        print("DEBUG: mae calculated:", metrics["mae"])

        print("DEBUG: About to calculate rmse")
        metrics["rmse"] = self.calculate_rmse(y_true, y_pred)
        print("DEBUG: rmse calculated:", metrics["rmse"])

        # Time-in-range metrics
        print("DEBUG: About to calculate time_in_range_accuracy")
        tir_metrics = self.calculate_time_in_range_accuracy(y_true, y_pred)
        print("DEBUG: tir_metrics calculated:", tir_metrics)
        metrics.update(tir_metrics)

        # Hypoglycemia detection metrics
        print("DEBUG: About to calculate detect_hypoglycemia_events")
        hypo_metrics = self.detect_hypoglycemia_events(y_true, y_pred)
        print("DEBUG: hypo_metrics calculated:", hypo_metrics)
        metrics.update(hypo_metrics)

        # Hyperglycemia detection metrics
        print("DEBUG: About to calculate detect_hyperglycemia_events")
        hyper_metrics = self.detect_hyperglycemia_events(y_true, y_pred)
        print("DEBUG: hyper_metrics calculated:", hyper_metrics)
        metrics.update(hyper_metrics)

        print("DEBUG: Final metrics:", metrics)
        return metrics

    def evaluate_model(self, model, test_data):
        """
        Evaluate a trained model on test data.
        """
        print("DEBUG: evaluate_model called")
        print("DEBUG: model type:", type(model))
        print("DEBUG: test_data type:", type(test_data))
        print("DEBUG: test_data value:", test_data)

        if model is None:
            logger.debug("Model is None")
            print("DEBUG: Model is None")
            raise ValueError("Model cannot be None")

        if "X" not in test_data or "y" not in test_data:
            logger.debug(f"Test data keys: {list(test_data.keys())}")
            print("DEBUG: Test data keys:", list(test_data.keys()))
            raise ValueError("Test data must contain 'X' and 'y' keys")

        X_test = test_data["X"]
        y_test = test_data["y"]

        logger.info(f"Evaluating model on test set with {len(X_test)} samples")
        logger.debug(
            f"X_test type: {type(X_test)}, shape: {getattr(X_test, 'shape', None)}"
        )
        logger.debug(
            f"y_test type: {type(y_test)}, shape: {getattr(y_test, 'shape', None)}"
        )
        print(
            "DEBUG: X_test type:",
            type(X_test),
            "shape:",
            getattr(X_test, "shape", None),
        )
        print(
            "DEBUG: y_test type:",
            type(y_test),
            "shape:",
            getattr(y_test, "shape", None),
        )

        # Make predictions
        y_pred = model.predict(X_test, batch_size=32, verbose=0)
        y_pred = y_pred.flatten()
        y_test = y_test.flatten()

        logger.debug(
            f"y_pred type: {type(y_pred)}, shape: {getattr(y_pred, 'shape', None)}"
        )
        logger.debug(
            f"y_test type: {type(y_test)}, shape: {getattr(y_test, 'shape', None)}"
        )
        print(
            "DEBUG: y_pred type:",
            type(y_pred),
            "shape:",
            getattr(y_pred, "shape", None),
        )
        print(
            "DEBUG: y_test type:",
            type(y_test),
            "shape:",
            getattr(y_test, "shape", None),
        )

        # Calculate all clinical metrics
        print("DEBUG: About to call calculate_all_metrics")
        metrics = self.calculate_all_metrics(y_true=y_test, y_pred=y_pred)
        print("DEBUG: calculate_all_metrics completed")
        print("DEBUG: metrics type:", type(metrics))
        print("DEBUG: metrics value:", metrics)
        print(
            "DEBUG: metrics keys:",
            list(metrics.keys()) if isinstance(metrics, dict) else "Not a dict",
        )

        return metrics
