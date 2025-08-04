"""Metrics calculator for comprehensive model evaluation during training."""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculator for comprehensive training and validation metrics."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics calculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.target_range = config.get("evaluation", {}).get(
            "target_glucose_range", (70, 180)
        )
        self.hypoglycemia_threshold = config.get("evaluation", {}).get(
            "hypoglycemia_threshold", 70
        )
        self.hyperglycemia_threshold = config.get("evaluation", {}).get(
            "hyperglycemia_threshold", 180
        )

    def calculate_regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate standard regression metrics.

        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values

        Returns:
            Dictionary of regression metrics
        """
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]

        if len(y_true_clean) == 0:
            logger.warning("No valid predictions found for metric calculation")
            return {
                "mae": np.nan,
                "mse": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "valid_predictions": 0,
            }

        metrics = {
            "mae": mean_absolute_error(y_true_clean, y_pred_clean),
            "mse": mean_squared_error(y_true_clean, y_pred_clean),
            "rmse": np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            "r2": r2_score(y_true_clean, y_pred_clean),
            "valid_predictions": len(y_true_clean),
        }

        return metrics

    def calculate_glucose_specific_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate glucose-specific clinical metrics.

        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values

        Returns:
            Dictionary of glucose-specific metrics
        """
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0)
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]

        if len(y_true_clean) == 0:
            logger.warning("No valid predictions found for glucose metric calculation")
            return {
                "mard": np.nan,
                "time_in_range_accuracy": np.nan,
                "hypoglycemia_detection_rate": np.nan,
                "hyperglycemia_detection_rate": np.nan,
                "glucose_bias": np.nan,
                "glucose_precision": np.nan,
            }

        # Mean Absolute Relative Difference (MARD)
        mard = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100

        # Time-in-range accuracy
        true_in_range = (y_true_clean >= self.target_range[0]) & (
            y_true_clean <= self.target_range[1]
        )
        pred_in_range = (y_pred_clean >= self.target_range[0]) & (
            y_pred_clean <= self.target_range[1]
        )
        tir_accuracy = np.mean(true_in_range == pred_in_range) * 100

        # Hypoglycemia detection
        true_hypo = y_true_clean < self.hypoglycemia_threshold
        pred_hypo = y_pred_clean < self.hypoglycemia_threshold
        if np.sum(true_hypo) > 0:
            hypo_detection_rate = (
                np.sum(true_hypo & pred_hypo) / np.sum(true_hypo) * 100
            )
        else:
            hypo_detection_rate = np.nan

        # Hyperglycemia detection
        true_hyper = y_true_clean > self.hyperglycemia_threshold
        pred_hyper = y_pred_clean > self.hyperglycemia_threshold
        if np.sum(true_hyper) > 0:
            hyper_detection_rate = (
                np.sum(true_hyper & pred_hyper) / np.sum(true_hyper) * 100
            )
        else:
            hyper_detection_rate = np.nan

        # Bias and precision
        glucose_bias = np.mean(y_pred_clean - y_true_clean)
        glucose_precision = np.std(y_pred_clean - y_true_clean)

        metrics = {
            "mard": mard,
            "time_in_range_accuracy": tir_accuracy,
            "hypoglycemia_detection_rate": hypo_detection_rate,
            "hyperglycemia_detection_rate": hyper_detection_rate,
            "glucose_bias": glucose_bias,
            "glucose_precision": glucose_precision,
        }

        return metrics

    def calculate_training_metrics(
        self, history: tf.keras.callbacks.History
    ) -> Dict[str, Any]:
        """
        Calculate metrics from training history.

        Args:
            history: Keras training history object

        Returns:
            Dictionary of training metrics and statistics
        """
        if not hasattr(history, "history") or not history.history:
            logger.warning("No training history available")
            return {}

        hist = history.history
        epochs = len(hist.get("loss", []))

        metrics = {
            "total_epochs": epochs,
            "final_train_loss": hist.get("loss", [np.nan])[-1],
            "final_val_loss": hist.get("val_loss", [np.nan])[-1],
            "best_val_loss": min(hist.get("val_loss", [np.inf])),
            "best_epoch": (
                np.argmin(hist.get("val_loss", [np.inf])) + 1
                if hist.get("val_loss")
                else 0
            ),
        }

        # Add final metric values
        for metric_name in hist.keys():
            if not metric_name.startswith("val_") and metric_name != "loss":
                metrics[f"final_train_{metric_name}"] = hist[metric_name][-1]
                val_metric_name = f"val_{metric_name}"
                if val_metric_name in hist:
                    metrics[f"final_val_{metric_name}"] = hist[val_metric_name][-1]

        # Calculate convergence metrics
        if len(hist.get("loss", [])) > 10:
            # Check if loss is still decreasing (last 10 epochs)
            recent_losses = hist["loss"][-10:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            metrics["loss_trend"] = loss_trend
            metrics["converged"] = (
                abs(loss_trend) < 0.001
            )  # Small trend indicates convergence

        return metrics

    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        history: Optional[tf.keras.callbacks.History] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics combining regression, glucose-specific, and training metrics.

        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values
            history: Optional training history

        Returns:
            Dictionary of all calculated metrics
        """
        logger.info("Calculating comprehensive metrics")

        all_metrics = {}

        # Regression metrics
        regression_metrics = self.calculate_regression_metrics(y_true, y_pred)
        all_metrics.update(
            {f"regression_{k}": v for k, v in regression_metrics.items()}
        )

        # Glucose-specific metrics
        glucose_metrics = self.calculate_glucose_specific_metrics(y_true, y_pred)
        all_metrics.update({f"glucose_{k}": v for k, v in glucose_metrics.items()})

        # Training metrics
        if history is not None:
            training_metrics = self.calculate_training_metrics(history)
            all_metrics.update(
                {f"training_{k}": v for k, v in training_metrics.items()}
            )

        # Overall assessment
        all_metrics["overall_score"] = self._calculate_overall_score(all_metrics)

        return all_metrics

    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate an overall model performance score.

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Overall performance score (0-100)
        """
        score_components = []

        # MARD component (lower is better, target < 15%)
        mard = metrics.get("glucose_mard", np.nan)
        if not np.isnan(mard):
            mard_score = max(0, 100 - (mard / 15) * 50)  # 50 points max
            score_components.append(mard_score)

        # Time-in-range accuracy component
        tir_acc = metrics.get("glucose_time_in_range_accuracy", np.nan)
        if not np.isnan(tir_acc):
            score_components.append(tir_acc * 0.3)  # 30 points max

        # R² component
        r2 = metrics.get("regression_r2", np.nan)
        if not np.isnan(r2):
            r2_score = max(0, r2 * 20)  # 20 points max
            score_components.append(r2_score)

        if score_components:
            return np.mean(score_components)
        else:
            return 0.0

    def generate_metrics_report(
        self, metrics: Dict[str, Any], save_path: Optional[Path] = None
    ) -> str:
        """
        Generate a formatted metrics report.

        Args:
            metrics: Dictionary of calculated metrics
            save_path: Optional path to save the report

        Returns:
            Formatted metrics report string
        """
        report_lines = ["=" * 60, "LSTM MODEL PERFORMANCE REPORT", "=" * 60, ""]

        # Regression metrics
        if any(k.startswith("regression_") for k in metrics.keys()):
            report_lines.extend(["REGRESSION METRICS:", "-" * 20])

            for key, value in metrics.items():
                if key.startswith("regression_"):
                    metric_name = key.replace("regression_", "").upper()
                    if isinstance(value, float):
                        report_lines.append(f"{metric_name:.<20} {value:.4f}")
                    else:
                        report_lines.append(f"{metric_name:.<20} {value}")
            report_lines.append("")

        # Glucose-specific metrics
        if any(k.startswith("glucose_") for k in metrics.keys()):
            report_lines.extend(["GLUCOSE-SPECIFIC METRICS:", "-" * 25])

            for key, value in metrics.items():
                if key.startswith("glucose_"):
                    metric_name = key.replace("glucose_", "").replace("_", " ").title()
                    if isinstance(value, float):
                        if "rate" in key or "accuracy" in key:
                            report_lines.append(f"{metric_name:.<25} {value:.2f}%")
                        else:
                            report_lines.append(f"{metric_name:.<25} {value:.4f}")
                    else:
                        report_lines.append(f"{metric_name:.<25} {value}")
            report_lines.append("")

        # Training metrics
        if any(k.startswith("training_") for k in metrics.keys()):
            report_lines.extend(["TRAINING METRICS:", "-" * 17])

            for key, value in metrics.items():
                if key.startswith("training_"):
                    metric_name = key.replace("training_", "").replace("_", " ").title()
                    if isinstance(value, float):
                        report_lines.append(f"{metric_name:.<25} {value:.4f}")
                    else:
                        report_lines.append(f"{metric_name:.<25} {value}")
            report_lines.append("")

        # Overall score
        if "overall_score" in metrics:
            report_lines.extend(
                [
                    "OVERALL ASSESSMENT:",
                    "-" * 19,
                    f"Overall Score: {metrics['overall_score']:.2f}/100",
                    "",
                ]
            )

        # Clinical interpretation
        mard = metrics.get("glucose_mard", np.nan)
        if not np.isnan(mard):
            report_lines.extend(["CLINICAL INTERPRETATION:", "-" * 23])

            if mard < 10:
                interpretation = "Excellent clinical accuracy"
            elif mard < 15:
                interpretation = "Good clinical accuracy"
            elif mard < 20:
                interpretation = "Acceptable clinical accuracy"
            else:
                interpretation = "Poor clinical accuracy - needs improvement"

            report_lines.append(f"MARD Assessment: {interpretation}")
            report_lines.append("")

        report_lines.append("=" * 60)

        report_text = "\n".join(report_lines)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                f.write(report_text)
            logger.info(f"Metrics report saved to {save_path}")

        return report_text

    def plot_training_history(
        self, history: tf.keras.callbacks.History, save_path: Optional[Path] = None
    ) -> None:
        """
        Plot training history metrics.

        Args:
            history: Keras training history
            save_path: Optional path to save the plot
        """
        if not hasattr(history, "history") or not history.history:
            logger.warning("No training history available for plotting")
            return

        hist = history.history
        epochs = range(1, len(hist["loss"]) + 1)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Training History", fontsize=16)

        # Loss plot
        axes[0, 0].plot(epochs, hist["loss"], "b-", label="Training Loss")
        if "val_loss" in hist:
            axes[0, 0].plot(epochs, hist["val_loss"], "r-", label="Validation Loss")
        axes[0, 0].set_title("Model Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MAE plot
        if "mae" in hist:
            axes[0, 1].plot(epochs, hist["mae"], "b-", label="Training MAE")
            if "val_mae" in hist:
                axes[0, 1].plot(epochs, hist["val_mae"], "r-", label="Validation MAE")
            axes[0, 1].set_title("Mean Absolute Error")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("MAE")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # MARD plot
        if "mard" in hist:
            axes[1, 0].plot(epochs, hist["mard"], "b-", label="Training MARD")
            if "val_mard" in hist:
                axes[1, 0].plot(epochs, hist["val_mard"], "r-", label="Validation MARD")
            axes[1, 0].set_title("Mean Absolute Relative Difference")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("MARD (%)")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Time-in-range accuracy plot
        if "tir_accuracy" in hist:
            axes[1, 1].plot(
                epochs, hist["tir_accuracy"], "b-", label="Training TIR Accuracy"
            )
            if "val_tir_accuracy" in hist:
                axes[1, 1].plot(
                    epochs,
                    hist["val_tir_accuracy"],
                    "r-",
                    label="Validation TIR Accuracy",
                )
            axes[1, 1].set_title("Time-in-Range Accuracy")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Accuracy")
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Training history plot saved to {save_path}")

        plt.show()
