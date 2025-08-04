"""Clinical metrics for diabetes glucose prediction evaluation."""

import numpy as np
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


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
        if not np.any(valid_mask):
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
        # Calculate actual time-in-range
        true_in_range = (y_true >= self.target_range[0]) & (
            y_true <= self.target_range[1]
        )
        pred_in_range = (y_pred >= self.target_range[0]) & (
            y_pred <= self.target_range[1]
        )

        # Calculate TIR percentages
        true_tir_percent = np.mean(true_in_range) * 100
        pred_tir_percent = np.mean(pred_in_range) * 100

        # Calculate TIR prediction accuracy (how well we predict in-range vs out-of-range)
        tir_accuracy = np.mean(true_in_range == pred_in_range) * 100

        # Calculate sensitivity and specificity for TIR prediction
        true_positives = np.sum(true_in_range & pred_in_range)
        false_positives = np.sum(~true_in_range & pred_in_range)
        true_negatives = np.sum(~true_in_range & ~pred_in_range)
        false_negatives = np.sum(true_in_range & ~pred_in_range)

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

        return {
            "true_tir_percent": true_tir_percent,
            "predicted_tir_percent": pred_tir_percent,
            "tir_prediction_accuracy": tir_accuracy,
            "tir_sensitivity": sensitivity * 100,
            "tir_specificity": specificity * 100,
            "tir_absolute_error": abs(true_tir_percent - pred_tir_percent),
        }

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
        true_positives = np.sum(true_hypo & pred_hypo)
        false_positives = np.sum(~true_hypo & pred_hypo)
        true_negatives = np.sum(~true_hypo & ~pred_hypo)
        false_negatives = np.sum(true_hypo & ~pred_hypo)

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
        true_positives = np.sum(true_hyper & pred_hyper)
        false_positives = np.sum(~true_hyper & pred_hyper)
        true_negatives = np.sum(~true_hyper & ~pred_hyper)
        false_negatives = np.sum(true_hyper & ~pred_hyper)

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
        metrics = {}

        # Basic metrics
        metrics["mard"] = self.calculate_mard(y_true, y_pred)
        metrics["mae"] = self.calculate_mae(y_true, y_pred)
        metrics["rmse"] = self.calculate_rmse(y_true, y_pred)

        # Time-in-range metrics
        tir_metrics = self.calculate_time_in_range_accuracy(y_true, y_pred)
        metrics.update(tir_metrics)

        # Hypoglycemia detection metrics
        hypo_metrics = self.detect_hypoglycemia_events(y_true, y_pred)
        metrics.update(hypo_metrics)

        # Hyperglycemia detection metrics
        hyper_metrics = self.detect_hyperglycemia_events(y_true, y_pred)
        metrics.update(hyper_metrics)

        return metrics
