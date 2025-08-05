"""Training utilities for history logging and visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path
import json
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


class TrainingHistoryLogger:
    """Logs and manages training history with persistence and analysis."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training history logger.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # History storage
        self.training_runs = []
        self.current_run = None

        # Logging parameters
        self.save_frequency = config.get("logging", {}).get("save_frequency", 10)
        self.auto_save = config.get("logging", {}).get("auto_save", True)

    def start_run(
        self,
        run_id: str,
        config: Dict[str, Any],
        model_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Start logging a new training run.

        Args:
            run_id: Unique identifier for the run
            config: Training configuration
            model_summary: Optional model summary information
        """
        self.current_run = {
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "config": config.copy(),
            "model_summary": model_summary,
            "history": [],
            "metrics": {},
            "status": "running",
        }

        logger.info(f"Started logging training run: {run_id}")

    def log_epoch(
        self,
        epoch: int,
        logs: Dict[str, float],
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log data for a single epoch.

        Args:
            epoch: Epoch number
            logs: Training logs dictionary
            additional_data: Optional additional data to log
        """
        if self.current_run is None:
            logger.warning("No active run to log to")
            return

        epoch_data = {"epoch": epoch, "timestamp": datetime.now().isoformat(), **logs}

        if additional_data:
            epoch_data.update(additional_data)

        self.current_run["history"].append(epoch_data)

        # Auto-save periodically
        if self.auto_save and epoch % self.save_frequency == 0:
            self.save_current_run()

    def end_run(
        self, final_metrics: Optional[Dict[str, Any]] = None, status: str = "completed"
    ) -> None:
        """
        End the current training run.

        Args:
            final_metrics: Optional final metrics
            status: Run status ('completed', 'failed', 'stopped')
        """
        if self.current_run is None:
            logger.warning("No active run to end")
            return

        self.current_run["end_time"] = datetime.now().isoformat()
        self.current_run["status"] = status

        if final_metrics:
            self.current_run["final_metrics"] = final_metrics

        # Calculate run statistics
        self._calculate_run_statistics()

        # Save final run
        self.save_current_run()

        # Add to runs history
        self.training_runs.append(self.current_run.copy())

        logger.info(
            f"Ended training run: {self.current_run['run_id']} with status: {status}"
        )
        self.current_run = None

    def save_current_run(self) -> Optional[Path]:
        """
        Save the current run to disk.

        Returns:
            Path to saved file or None
        """
        if self.current_run is None:
            return None

        run_id = self.current_run["run_id"]
        save_path = self.log_dir / f"training_run_{run_id}.json"

        try:
            with open(save_path, "w") as f:
                json.dump(self.current_run, f, indent=2, default=str)

            logger.debug(f"Saved training run to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to save training run: {str(e)}")
            return None

    def load_run(self, run_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a training run from disk.

        Args:
            run_path: Path to the run file

        Returns:
            Training run dictionary
        """
        run_path = Path(run_path)

        if not run_path.exists():
            raise FileNotFoundError(f"Run file not found: {run_path}")

        with open(run_path, "r") as f:
            run_data = json.load(f)

        logger.info(f"Loaded training run: {run_data.get('run_id', 'unknown')}")
        return run_data

    def _calculate_run_statistics(self) -> None:
        """Calculate statistics for the current run."""
        if not self.current_run or not self.current_run["history"]:
            return

        history = self.current_run["history"]

        # Extract metrics
        metrics = {}
        for key in history[0].keys():
            if key not in ["epoch", "timestamp"] and isinstance(
                history[0][key], (int, float)
            ):
                values = [h[key] for h in history if key in h and not np.isnan(h[key])]
                if values:
                    metrics[f"{key}_final"] = values[-1]
                    metrics[f"{key}_best"] = (
                        min(values) if "loss" in key else max(values)
                    )
                    metrics[f"{key}_mean"] = np.mean(values)
                    metrics[f"{key}_std"] = np.std(values)

        # Training duration
        if len(history) > 1:
            start_time = datetime.fromisoformat(history[0]["timestamp"])
            end_time = datetime.fromisoformat(history[-1]["timestamp"])
            duration = (end_time - start_time).total_seconds()
            metrics["training_duration_seconds"] = duration
            metrics["epochs_completed"] = len(history)
            metrics["avg_epoch_time"] = duration / len(history)

        self.current_run["statistics"] = metrics

    def get_run_summary(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of a training run.

        Args:
            run_id: Run ID to summarize (current run if None)

        Returns:
            Run summary dictionary
        """
        if run_id is None:
            run_data = self.current_run
        else:
            run_data = next(
                (r for r in self.training_runs if r["run_id"] == run_id), None
            )

        if run_data is None:
            raise ValueError(f"Run not found: {run_id}")

        summary = {
            "run_id": run_data["run_id"],
            "status": run_data.get("status", "unknown"),
            "start_time": run_data.get("start_time"),
            "end_time": run_data.get("end_time"),
            "epochs": len(run_data.get("history", [])),
            "statistics": run_data.get("statistics", {}),
            "config": run_data.get("config", {}),
        }

        return summary


class TrainingVisualizer:
    """Creates visualizations for training history and analysis."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.plots_dir = Path(config.get("plots_dir", "plots"))
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Plotting parameters
        self.style = config.get("plotting", {}).get("style", "seaborn-v0_8")
        self.figsize = config.get("plotting", {}).get("figsize", (12, 8))
        self.dpi = config.get("plotting", {}).get("dpi", 300)

        # Set style
        plt.style.use(self.style)
        sns.set_palette("husl")

    def plot_training_history(
        self,
        history: Union[Dict[str, List], keras.callbacks.History],
        save_path: Optional[Path] = None,
        title: str = "Training History",
    ) -> Path:
        """
        Plot comprehensive training history.

        Args:
            history: Training history (dict or Keras History object)
            save_path: Optional path to save plot
            title: Plot title

        Returns:
            Path to saved plot
        """
        # Extract history data
        if hasattr(history, "history"):
            hist_data = history.history
        else:
            hist_data = history

        # Determine number of subplots needed
        metrics = [k for k in hist_data.keys() if not k.startswith("val_")]
        n_metrics = len(metrics)

        if n_metrics == 0:
            logger.warning("No metrics found in history")
            return None

        # Create subplots
        cols = 2
        rows = (n_metrics + 1) // 2
        fig, axes = plt.subplots(
            rows, cols, figsize=(self.figsize[0], self.figsize[1] * rows / 2)
        )
        fig.suptitle(title, fontsize=16)

        if rows == 1:
            axes = [axes] if n_metrics == 1 else axes
        else:
            axes = axes.flatten()

        epochs = range(1, len(hist_data[metrics[0]]) + 1)

        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Plot training metric
            ax.plot(
                epochs, hist_data[metric], "b-", label=f"Training {metric}", linewidth=2
            )

            # Plot validation metric if available
            val_metric = f"val_{metric}"
            if val_metric in hist_data:
                ax.plot(
                    epochs,
                    hist_data[val_metric],
                    "r-",
                    label=f"Validation {metric}",
                    linewidth=2,
                )

            ax.set_title(f"{metric.upper()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add best value annotation
            if val_metric in hist_data:
                best_val = (
                    min(hist_data[val_metric])
                    if "loss" in metric
                    else max(hist_data[val_metric])
                )
                best_epoch = hist_data[val_metric].index(best_val) + 1
                ax.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7)
                ax.annotate(
                    f"Best: {best_val:.4f}",
                    xy=(best_epoch, best_val),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                )

        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.plots_dir / f"training_history_{timestamp}.png"

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Training history plot saved to {save_path}")
        return save_path

    def plot_metrics_comparison(
        self,
        runs_data: List[Dict[str, Any]],
        metrics: List[str] = ["loss", "mae", "mard"],
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Compare metrics across multiple training runs.

        Args:
            runs_data: List of training run dictionaries
            metrics: List of metrics to compare
            save_path: Optional path to save plot

        Returns:
            Path to saved plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(
            1, n_metrics, figsize=(self.figsize[0], self.figsize[1] // 2)
        )
        fig.suptitle("Training Runs Comparison", fontsize=16)

        if n_metrics == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, len(runs_data)))

        for i, metric in enumerate(metrics):
            ax = axes[i]

            for j, run_data in enumerate(runs_data):
                history = run_data.get("history", [])
                if not history:
                    continue

                run_id = run_data.get("run_id", f"Run {j+1}")

                # Extract metric values
                epochs = [h["epoch"] for h in history if metric in h]
                values = [h[metric] for h in history if metric in h]

                if epochs and values:
                    ax.plot(epochs, values, color=colors[j], label=run_id, linewidth=2)

            ax.set_title(f"{metric.upper()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.plots_dir / f"runs_comparison_{timestamp}.png"

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Runs comparison plot saved to {save_path}")
        return save_path

    def plot_learning_curves(
        self,
        train_sizes: List[int],
        train_scores: List[float],
        val_scores: List[float],
        metric_name: str = "Loss",
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Plot learning curves showing performance vs training set size.

        Args:
            train_sizes: List of training set sizes
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            metric_name: Name of the metric being plotted
            save_path: Optional path to save plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(
            train_sizes,
            train_scores,
            "b-o",
            label=f"Training {metric_name}",
            linewidth=2,
        )
        ax.plot(
            train_sizes,
            val_scores,
            "r-o",
            label=f"Validation {metric_name}",
            linewidth=2,
        )

        ax.set_title(f"Learning Curves - {metric_name}")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add annotations for final values
        ax.annotate(
            f"Final Train: {train_scores[-1]:.4f}",
            xy=(train_sizes[-1], train_scores[-1]),
            xytext=(-50, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )

        ax.annotate(
            f"Final Val: {val_scores[-1]:.4f}",
            xy=(train_sizes[-1], val_scores[-1]),
            xytext=(-50, -20),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
        )

        plt.tight_layout()

        # Save plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.plots_dir / f"learning_curves_{timestamp}.png"

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Learning curves plot saved to {save_path}")
        return save_path

    def create_training_dashboard(
        self, run_data: Dict[str, Any], save_path: Optional[Path] = None
    ) -> Path:
        """
        Create a comprehensive training dashboard.

        Args:
            run_data: Training run data
            save_path: Optional path to save dashboard

        Returns:
            Path to saved dashboard
        """
        history = run_data.get("history", [])
        if not history:
            logger.warning("No history data available for dashboard")
            return None

        # Create dashboard with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Extract data
        epochs = [h["epoch"] for h in history]

        # Loss plot
        ax1 = fig.add_subplot(gs[0, 0])
        if "loss" in history[0]:
            train_loss = [h["loss"] for h in history]
            ax1.plot(epochs, train_loss, "b-", label="Training Loss")
            if "val_loss" in history[0]:
                val_loss = [h["val_loss"] for h in history]
                ax1.plot(epochs, val_loss, "r-", label="Validation Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MAE plot
        ax2 = fig.add_subplot(gs[0, 1])
        if "mae" in history[0]:
            train_mae = [h["mae"] for h in history]
            ax2.plot(epochs, train_mae, "b-", label="Training MAE")
            if "val_mae" in history[0]:
                val_mae = [h["val_mae"] for h in history]
                ax2.plot(epochs, val_mae, "r-", label="Validation MAE")
        ax2.set_title("Mean Absolute Error")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # MARD plot
        ax3 = fig.add_subplot(gs[0, 2])
        if "mard" in history[0]:
            train_mard = [h["mard"] for h in history]
            ax3.plot(epochs, train_mard, "b-", label="Training MARD")
            if "val_mard" in history[0]:
                val_mard = [h["val_mard"] for h in history]
                ax3.plot(epochs, val_mard, "r-", label="Validation MARD")
        ax3.set_title("MARD (%)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Learning rate plot
        ax4 = fig.add_subplot(gs[1, 0])
        if "lr" in history[0]:
            learning_rates = [h["lr"] for h in history]
            ax4.plot(epochs, learning_rates, "g-")
            ax4.set_yscale("log")
        ax4.set_title("Learning Rate")
        ax4.grid(True, alpha=0.3)

        # Epoch time plot
        ax5 = fig.add_subplot(gs[1, 1])
        if "epoch_time" in history[0]:
            epoch_times = [h["epoch_time"] for h in history]
            ax5.plot(epochs, epoch_times, "m-")
        ax5.set_title("Epoch Time (seconds)")
        ax5.grid(True, alpha=0.3)

        # Statistics text
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis("off")
        stats = run_data.get("statistics", {})
        stats_text = "Training Statistics:\n\n"
        for key, value in stats.items():
            if isinstance(value, float):
                stats_text += f"{key}: {value:.4f}\n"
            else:
                stats_text += f"{key}: {value}\n"
        ax6.text(
            0.1,
            0.9,
            stats_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        # Configuration text
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis("off")
        config = run_data.get("config", {})
        config_text = "Configuration:\n"
        config_text += f"Model: {config.get('model', {})}\n"
        config_text += f"Training: {config.get('training', {})}\n"
        ax7.text(
            0.05,
            0.9,
            config_text,
            transform=ax7.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.suptitle(
            f"Training Dashboard - {run_data.get('run_id', 'Unknown')}", fontsize=16
        )

        # Save dashboard
        if save_path is None:
            run_id = run_data.get("run_id", "unknown")
            save_path = self.plots_dir / f"training_dashboard_{run_id}.png"

        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Training dashboard saved to {save_path}")
        return save_path
