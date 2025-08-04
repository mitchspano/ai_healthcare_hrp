"""Visualization generator for clinical evaluation metrics and charts."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """
    Generates visualizations for clinical evaluation metrics and error grid analyses.

    This class creates comprehensive plots and charts for interpreting glucose
    prediction model performance from a clinical perspective.
    """

    def __init__(self, output_dir: str = "reports", figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize visualization generator.

        Args:
            output_dir: Directory to save generated plots
            figsize: Default figure size for plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize = figsize

        # Set style for clinical plots
        plt.style.use("default")
        sns.set_palette("husl")

    def plot_clarke_error_grid(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
    ) -> str:
        """
        Create Clarke Error Grid visualization.

        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            save_path: Optional path to save the plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Remove invalid values
        valid_mask = (y_true > 0) & (y_pred > 0)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        # Plot data points
        ax.scatter(y_true_valid, y_pred_valid, alpha=0.6, s=20, color="blue")

        # Draw zone boundaries
        self._draw_clarke_zones(ax)

        # Perfect prediction line
        max_val = max(np.max(y_true_valid), np.max(y_pred_valid))
        ax.plot(
            [0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect Prediction"
        )

        # Formatting
        ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=12)
        ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=12)
        ax.set_title("Clarke Error Grid Analysis", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set equal aspect ratio and limits
        ax.set_aspect("equal")
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "clarke_error_grid.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Clarke Error Grid plot saved to {save_path}")
        return str(save_path)

    def _draw_clarke_zones(self, ax):
        """Draw Clarke Error Grid zone boundaries."""
        # Zone boundaries (simplified version for visualization)
        # Zone A boundaries
        ax.axline((0, 0), slope=1.2, color="green", alpha=0.3, linestyle="-")
        ax.axline((0, 0), slope=0.8, color="green", alpha=0.3, linestyle="-")

        # Add zone labels
        ax.text(50, 60, "A", fontsize=16, fontweight="bold", color="green")
        ax.text(150, 200, "B", fontsize=16, fontweight="bold", color="orange")
        ax.text(300, 150, "B", fontsize=16, fontweight="bold", color="orange")
        ax.text(100, 300, "C", fontsize=16, fontweight="bold", color="red")
        ax.text(300, 100, "C", fontsize=16, fontweight="bold", color="red")

    def plot_parkes_error_grid(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        grid_type: str = "type1",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create Parkes Error Grid visualization.

        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            grid_type: Type of diabetes ('type1' or 'type2')
            save_path: Optional path to save the plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Remove invalid values
        valid_mask = (y_true > 0) & (y_pred > 0)
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        # Plot data points
        ax.scatter(y_true_valid, y_pred_valid, alpha=0.6, s=20, color="blue")

        # Draw zone boundaries
        self._draw_parkes_zones(ax, grid_type)

        # Perfect prediction line
        max_val = max(np.max(y_true_valid), np.max(y_pred_valid))
        ax.plot(
            [0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect Prediction"
        )

        # Formatting
        ax.set_xlabel("Reference Glucose (mg/dL)", fontsize=12)
        ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=12)
        ax.set_title(
            f"Parkes Error Grid Analysis ({grid_type.upper()})",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set equal aspect ratio and limits
        ax.set_aspect("equal")
        ax.set_xlim(0, max_val * 1.05)
        ax.set_ylim(0, max_val * 1.05)

        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"parkes_error_grid_{grid_type}.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Parkes Error Grid plot saved to {save_path}")
        return str(save_path)

    def _draw_parkes_zones(self, ax, grid_type: str):
        """Draw Parkes Error Grid zone boundaries."""
        # Simplified zone boundaries for visualization
        # More stringent than Clarke
        ax.axline((0, 0), slope=1.15, color="green", alpha=0.3, linestyle="-")
        ax.axline((0, 0), slope=0.85, color="green", alpha=0.3, linestyle="-")

        # Add zone labels
        ax.text(50, 55, "A", fontsize=16, fontweight="bold", color="green")
        ax.text(150, 190, "B", fontsize=16, fontweight="bold", color="orange")
        ax.text(300, 160, "B", fontsize=16, fontweight="bold", color="orange")

    def plot_prediction_scatter(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
    ) -> str:
        """
        Create scatter plot of predictions vs true values.

        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            save_path: Optional path to save the plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Create scatter plot with density coloring
        ax.scatter(y_true, y_pred, alpha=0.6, s=20)

        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            alpha=0.8,
            label="Perfect Prediction",
        )

        # Add clinical range indicators
        ax.axhline(
            y=70,
            color="orange",
            linestyle=":",
            alpha=0.7,
            label="Hypoglycemia Threshold",
        )
        ax.axhline(
            y=180, color="orange", linestyle=":", alpha=0.7, label="Target Range Upper"
        )
        ax.axvline(x=70, color="orange", linestyle=":", alpha=0.7)
        ax.axvline(x=180, color="orange", linestyle=":", alpha=0.7)

        # Formatting
        ax.set_xlabel("True Glucose (mg/dL)", fontsize=12)
        ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=12)
        ax.set_title("Glucose Prediction Accuracy", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "prediction_scatter.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Prediction scatter plot saved to {save_path}")
        return str(save_path)

    def plot_residuals(
        self, y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
    ) -> str:
        """
        Create residual plot for prediction errors.

        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            save_path: Optional path to save the plot

        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        residuals = y_pred - y_true

        # Residuals vs predicted values
        ax1.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax1.axhline(y=0, color="r", linestyle="--", alpha=0.8)
        ax1.set_xlabel("Predicted Glucose (mg/dL)", fontsize=12)
        ax1.set_ylabel("Residuals (mg/dL)", fontsize=12)
        ax1.set_title("Residuals vs Predicted Values", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Histogram of residuals
        ax2.hist(residuals, bins=50, alpha=0.7, edgecolor="black")
        ax2.axvline(x=0, color="r", linestyle="--", alpha=0.8)
        ax2.set_xlabel("Residuals (mg/dL)", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.set_title("Distribution of Residuals", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "residual_analysis.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Residual analysis plot saved to {save_path}")
        return str(save_path)

    def plot_clinical_metrics_summary(
        self, metrics: Dict[str, float], save_path: Optional[str] = None
    ) -> str:
        """
        Create summary visualization of clinical metrics.

        Args:
            metrics: Dictionary of clinical metrics
            save_path: Optional path to save the plot

        Returns:
            Path to saved plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # MARD and basic metrics
        basic_metrics = ["mard", "mae", "rmse"]
        basic_values = [metrics.get(m, 0) for m in basic_metrics]
        basic_labels = ["MARD (%)", "MAE (mg/dL)", "RMSE (mg/dL)"]

        bars1 = ax1.bar(basic_labels, basic_values, color=["red", "blue", "green"])
        ax1.set_title("Basic Prediction Metrics", fontsize=14, fontweight="bold")
        ax1.set_ylabel("Value")
        for bar, value in zip(bars1, basic_values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(basic_values) * 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )

        # Clarke Error Grid zones
        clarke_zones = [
            "clarke_zone_a_percent",
            "clarke_zone_b_percent",
            "clarke_zone_c_percent",
            "clarke_zone_d_percent",
            "clarke_zone_e_percent",
        ]
        clarke_values = [metrics.get(z, 0) for z in clarke_zones]
        clarke_labels = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"]
        colors = ["green", "yellow", "orange", "red", "darkred"]

        bars2 = ax2.bar(clarke_labels, clarke_values, color=colors)
        ax2.set_title("Clarke Error Grid Distribution", fontsize=14, fontweight="bold")
        ax2.set_ylabel("Percentage (%)")
        for bar, value in zip(bars2, clarke_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
            )

        # Hypoglycemia detection metrics
        hypo_metrics = [
            "hypoglycemia_sensitivity",
            "hypoglycemia_specificity",
            "hypoglycemia_precision",
        ]
        hypo_values = [metrics.get(m, 0) for m in hypo_metrics]
        hypo_labels = ["Sensitivity", "Specificity", "Precision"]

        bars3 = ax3.bar(hypo_labels, hypo_values, color=["purple", "orange", "cyan"])
        ax3.set_title(
            "Hypoglycemia Detection Performance", fontsize=14, fontweight="bold"
        )
        ax3.set_ylabel("Percentage (%)")
        ax3.set_ylim(0, 100)
        for bar, value in zip(bars3, hypo_values):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
            )

        # Time-in-range metrics
        tir_actual = metrics.get("true_tir_percent", 0)
        tir_predicted = metrics.get("predicted_tir_percent", 0)
        tir_accuracy = metrics.get("tir_prediction_accuracy", 0)

        ax4.bar(
            ["Actual TIR", "Predicted TIR"],
            [tir_actual, tir_predicted],
            color=["blue", "red"],
            alpha=0.7,
            label="TIR %",
        )
        ax4_twin = ax4.twinx()
        ax4_twin.bar(
            ["TIR Accuracy"], [tir_accuracy], color="green", alpha=0.7, width=0.5
        )

        ax4.set_title("Time-in-Range Analysis", fontsize=14, fontweight="bold")
        ax4.set_ylabel("TIR Percentage (%)", color="blue")
        ax4_twin.set_ylabel("Prediction Accuracy (%)", color="green")

        # Save plot
        if save_path is None:
            save_path = self.output_dir / "clinical_metrics_summary.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Clinical metrics summary plot saved to {save_path}")
        return str(save_path)

    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        clinical_metrics: Dict[str, float],
        clarke_results: Dict[str, float],
        parkes_results: Dict[str, float],
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate comprehensive evaluation report with all visualizations.

        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            clinical_metrics: Clinical metrics dictionary
            clarke_results: Clarke Error Grid results
            parkes_results: Parkes Error Grid results
            save_path: Optional path to save the report

        Returns:
            Path to saved report
        """
        # Create all visualizations
        plots = {}
        plots["scatter"] = self.plot_prediction_scatter(y_true, y_pred)
        plots["residuals"] = self.plot_residuals(y_true, y_pred)
        plots["clarke"] = self.plot_clarke_error_grid(y_true, y_pred)
        plots["parkes"] = self.plot_parkes_error_grid(y_true, y_pred)

        # Combine all metrics
        all_metrics = {**clinical_metrics, **clarke_results, **parkes_results}
        plots["summary"] = self.plot_clinical_metrics_summary(all_metrics)

        # Generate text report
        if save_path is None:
            save_path = self.output_dir / "evaluation_report.txt"
        else:
            save_path = Path(save_path)

        with open(save_path, "w") as f:
            f.write("GLUCOSE PREDICTION MODEL - CLINICAL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write("BASIC METRICS:\n")
            f.write(f"  MARD: {clinical_metrics.get('mard', 0):.2f}%\n")
            f.write(f"  MAE: {clinical_metrics.get('mae', 0):.2f} mg/dL\n")
            f.write(f"  RMSE: {clinical_metrics.get('rmse', 0):.2f} mg/dL\n\n")

            f.write("CLARKE ERROR GRID ANALYSIS:\n")
            f.write(
                f"  Zone A (Clinically Accurate): {clarke_results.get('clarke_zone_a_percent', 0):.1f}%\n"
            )
            f.write(
                f"  Zone B (Benign Errors): {clarke_results.get('clarke_zone_b_percent', 0):.1f}%\n"
            )
            f.write(
                f"  Clinically Acceptable (A+B): {clarke_results.get('clarke_clinically_acceptable_percent', 0):.1f}%\n"
            )
            f.write(
                f"  Dangerous Errors (D+E): {clarke_results.get('clarke_dangerous_errors_percent', 0):.1f}%\n\n"
            )

            f.write("PARKES ERROR GRID ANALYSIS:\n")
            f.write(
                f"  Zone A (Clinically Accurate): {parkes_results.get('parkes_zone_a_percent', 0):.1f}%\n"
            )
            f.write(
                f"  Zone B (Benign Errors): {parkes_results.get('parkes_zone_b_percent', 0):.1f}%\n"
            )
            f.write(
                f"  Clinically Acceptable (A+B): {parkes_results.get('parkes_clinically_acceptable_percent', 0):.1f}%\n"
            )
            f.write(
                f"  Dangerous Errors (D+E): {parkes_results.get('parkes_dangerous_errors_percent', 0):.1f}%\n\n"
            )

            f.write("GENERATED VISUALIZATIONS:\n")
            for plot_type, plot_path in plots.items():
                f.write(f"  {plot_type.title()}: {plot_path}\n")

        logger.info(f"Comprehensive evaluation report saved to {save_path}")
        return str(save_path)
