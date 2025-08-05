"""Training monitor for progress tracking and early stopping logic."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, List, Optional, Callable
import logging
import time
from datetime import datetime, timedelta
import threading
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitors training progress with advanced early stopping and progress tracking."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.training_config = config.get("training", {})
        self.monitor_config = config.get("monitor", {})

        # Early stopping parameters
        self.patience = self.training_config.get("early_stopping_patience", 15)
        self.min_delta = self.monitor_config.get("min_delta", 0.001)
        self.restore_best_weights = self.monitor_config.get(
            "restore_best_weights", True
        )
        self.monitor_metric = self.monitor_config.get("monitor_metric", "val_loss")
        self.mode = self.monitor_config.get("mode", "min")

        # Progress tracking
        self.log_frequency = self.monitor_config.get(
            "log_frequency", 10
        )  # Log every N epochs
        self.save_frequency = self.monitor_config.get(
            "save_frequency", 50
        )  # Save every N epochs
        self.plot_frequency = self.monitor_config.get(
            "plot_frequency", 25
        )  # Plot every N epochs

        # Monitoring state
        self.training_start_time = None
        self.epoch_times = []
        self.training_history = []
        self.best_metric_value = None
        self.best_epoch = 0
        self.wait_count = 0
        self.stopped_epoch = 0

        # Progress tracking
        self.total_epochs = 0
        self.current_epoch = 0
        self.train_samples = 0
        self.val_samples = 0

        # Callbacks and hooks
        self.progress_callbacks = []
        self.early_stop_callbacks = []

        # Thread safety
        self._lock = threading.Lock()

    def start_training(
        self, total_epochs: int, train_samples: int, val_samples: int = 0
    ) -> None:
        """
        Initialize training monitoring.

        Args:
            total_epochs: Total number of epochs to train
            train_samples: Number of training samples
            val_samples: Number of validation samples
        """
        with self._lock:
            self.training_start_time = datetime.now()
            self.total_epochs = total_epochs
            self.train_samples = train_samples
            self.val_samples = val_samples
            self.current_epoch = 0

            # Reset monitoring state
            self.epoch_times = []
            self.training_history = []
            self.best_metric_value = None
            self.best_epoch = 0
            self.wait_count = 0
            self.stopped_epoch = 0

            logger.info(
                f"Training monitor started - {total_epochs} epochs, {train_samples} train samples"
            )

    def stop_training(self) -> None:
        """Stop training monitoring and generate final report."""
        with self._lock:
            if self.training_start_time:
                total_time = datetime.now() - self.training_start_time
                logger.info(f"Training completed in {total_time}")

                # Generate final training report
                self._generate_final_report()

    def on_epoch_begin(
        self, epoch: int, logs: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number
            logs: Optional logs dictionary
        """
        with self._lock:
            self.current_epoch = epoch
            self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Logs dictionary containing metrics
        """
        with self._lock:
            # Record epoch time
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)

            # Store training history
            if logs:
                epoch_data = logs.copy()
                epoch_data["epoch"] = epoch
                epoch_data["epoch_time"] = epoch_time
                self.training_history.append(epoch_data)

            # Check for improvement
            should_stop = self._check_early_stopping(epoch, logs)

            # Log progress
            if epoch % self.log_frequency == 0 or should_stop:
                self._log_progress(epoch, logs)

            # Save checkpoint
            if epoch % self.save_frequency == 0:
                self._save_checkpoint(epoch, logs)

            # Plot progress
            if epoch % self.plot_frequency == 0:
                self._plot_progress()

            # Call progress callbacks
            for callback in self.progress_callbacks:
                callback(epoch, logs, self.get_training_stats())

            return should_stop

    def _check_early_stopping(
        self, epoch: int, logs: Optional[Dict[str, float]]
    ) -> bool:
        """
        Check if training should be stopped early.

        Args:
            epoch: Current epoch number
            logs: Logs dictionary

        Returns:
            True if training should stop
        """
        if not logs or self.monitor_metric not in logs:
            return False

        current_value = logs[self.monitor_metric]

        # Initialize best value on first epoch
        if self.best_metric_value is None:
            self.best_metric_value = current_value
            self.best_epoch = epoch
            return False

        # Check for improvement
        if self.mode == "min":
            improved = current_value < (self.best_metric_value - self.min_delta)
        else:  # mode == "max"
            improved = current_value > (self.best_metric_value + self.min_delta)

        if improved:
            self.best_metric_value = current_value
            self.best_epoch = epoch
            self.wait_count = 0
            logger.debug(
                f"New best {self.monitor_metric}: {current_value:.6f} at epoch {epoch}"
            )
        else:
            self.wait_count += 1

            if self.wait_count >= self.patience:
                self.stopped_epoch = epoch
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(
                    f"Best {self.monitor_metric}: {self.best_metric_value:.6f} at epoch {self.best_epoch}"
                )

                # Call early stop callbacks
                for callback in self.early_stop_callbacks:
                    callback(epoch, logs, self.best_epoch, self.best_metric_value)

                return True

        return False

    def _log_progress(self, epoch: int, logs: Optional[Dict[str, float]]) -> None:
        """
        Log training progress.

        Args:
            epoch: Current epoch number
            logs: Logs dictionary
        """
        if not logs:
            return

        # Calculate progress statistics
        progress_pct = (epoch / self.total_epochs) * 100
        avg_epoch_time = np.mean(self.epoch_times[-10:]) if self.epoch_times else 0
        eta = timedelta(seconds=avg_epoch_time * (self.total_epochs - epoch))

        # Format log message
        log_parts = [f"Epoch {epoch}/{self.total_epochs} ({progress_pct:.1f}%)"]

        # Add key metrics
        if "loss" in logs:
            log_parts.append(f"loss: {logs['loss']:.6f}")
        if "val_loss" in logs:
            log_parts.append(f"val_loss: {logs['val_loss']:.6f}")
        if "mae" in logs:
            log_parts.append(f"mae: {logs['mae']:.4f}")
        if "val_mae" in logs:
            log_parts.append(f"val_mae: {logs['val_mae']:.4f}")

        # Add timing info
        log_parts.append(f"time: {avg_epoch_time:.2f}s")
        log_parts.append(f"ETA: {eta}")

        # Add early stopping info
        if self.wait_count > 0:
            log_parts.append(f"patience: {self.wait_count}/{self.patience}")

        logger.info(" - ".join(log_parts))

    def _save_checkpoint(self, epoch: int, logs: Optional[Dict[str, float]]) -> None:
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch number
            logs: Logs dictionary
        """
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_data = {
            "epoch": epoch,
            "training_history": self.training_history,
            "best_metric_value": self.best_metric_value,
            "best_epoch": self.best_epoch,
            "wait_count": self.wait_count,
            "logs": logs,
        }

        checkpoint_path = checkpoint_dir / f"training_checkpoint_epoch_{epoch}.json"

        try:
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)
            logger.debug(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {str(e)}")

    def _plot_progress(self) -> None:
        """Plot training progress."""
        if len(self.training_history) < 2:
            return

        try:
            # Create progress plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Training Progress - Epoch {self.current_epoch}")

            # Extract data
            epochs = [h["epoch"] for h in self.training_history]

            # Loss plot
            if "loss" in self.training_history[0]:
                train_loss = [h["loss"] for h in self.training_history]
                axes[0, 0].plot(epochs, train_loss, "b-", label="Training Loss")

                if "val_loss" in self.training_history[0]:
                    val_loss = [h["val_loss"] for h in self.training_history]
                    axes[0, 0].plot(epochs, val_loss, "r-", label="Validation Loss")

                axes[0, 0].set_title("Loss")
                axes[0, 0].set_xlabel("Epoch")
                axes[0, 0].set_ylabel("Loss")
                axes[0, 0].legend()
                axes[0, 0].grid(True)

            # MAE plot
            if "mae" in self.training_history[0]:
                train_mae = [h["mae"] for h in self.training_history]
                axes[0, 1].plot(epochs, train_mae, "b-", label="Training MAE")

                if "val_mae" in self.training_history[0]:
                    val_mae = [h["val_mae"] for h in self.training_history]
                    axes[0, 1].plot(epochs, val_mae, "r-", label="Validation MAE")

                axes[0, 1].set_title("Mean Absolute Error")
                axes[0, 1].set_xlabel("Epoch")
                axes[0, 1].set_ylabel("MAE")
                axes[0, 1].legend()
                axes[0, 1].grid(True)

            # Epoch time plot
            if self.epoch_times:
                axes[1, 0].plot(
                    epochs[-len(self.epoch_times) :], self.epoch_times, "g-"
                )
                axes[1, 0].set_title("Epoch Time")
                axes[1, 0].set_xlabel("Epoch")
                axes[1, 0].set_ylabel("Time (seconds)")
                axes[1, 0].grid(True)

            # Learning rate plot (if available)
            if "lr" in self.training_history[0]:
                learning_rates = [h["lr"] for h in self.training_history]
                axes[1, 1].plot(epochs, learning_rates, "m-")
                axes[1, 1].set_title("Learning Rate")
                axes[1, 1].set_xlabel("Epoch")
                axes[1, 1].set_ylabel("Learning Rate")
                axes[1, 1].set_yscale("log")
                axes[1, 1].grid(True)

            plt.tight_layout()

            # Save plot
            plots_dir = Path("plots")
            plots_dir.mkdir(exist_ok=True)
            plot_path = plots_dir / f"training_progress_epoch_{self.current_epoch}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.debug(f"Progress plot saved to {plot_path}")

        except Exception as e:
            logger.warning(f"Failed to create progress plot: {str(e)}")

    def _generate_final_report(self) -> None:
        """Generate final training report."""
        if not self.training_history:
            return

        try:
            report_lines = [
                "=" * 60,
                "TRAINING SUMMARY REPORT",
                "=" * 60,
                "",
                f"Training Duration: {datetime.now() - self.training_start_time}",
                f"Total Epochs: {len(self.training_history)}",
                f"Training Samples: {self.train_samples}",
                f"Validation Samples: {self.val_samples}",
                "",
            ]

            # Best metrics
            if self.best_metric_value is not None:
                report_lines.extend(
                    [
                        "BEST PERFORMANCE:",
                        f"Best {self.monitor_metric}: {self.best_metric_value:.6f}",
                        f"Best Epoch: {self.best_epoch}",
                        "",
                    ]
                )

            # Final metrics
            final_logs = self.training_history[-1]
            report_lines.append("FINAL METRICS:")
            for key, value in final_logs.items():
                if key not in ["epoch", "epoch_time"] and isinstance(
                    value, (int, float)
                ):
                    report_lines.append(f"{key}: {value:.6f}")
            report_lines.append("")

            # Timing statistics
            if self.epoch_times:
                avg_epoch_time = np.mean(self.epoch_times)
                total_training_time = sum(self.epoch_times)
                report_lines.extend(
                    [
                        "TIMING STATISTICS:",
                        f"Average Epoch Time: {avg_epoch_time:.2f} seconds",
                        f"Total Training Time: {total_training_time:.2f} seconds",
                        f"Samples per Second: {self.train_samples / avg_epoch_time:.0f}",
                        "",
                    ]
                )

            # Early stopping info
            if self.stopped_epoch > 0:
                report_lines.extend(
                    [
                        "EARLY STOPPING:",
                        f"Stopped at Epoch: {self.stopped_epoch}",
                        f"Patience: {self.patience}",
                        f"Final Wait Count: {self.wait_count}",
                        "",
                    ]
                )

            report_lines.append("=" * 60)

            # Save report
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            report_path = (
                reports_dir
                / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

            with open(report_path, "w") as f:
                f.write("\n".join(report_lines))

            logger.info(f"Training report saved to {report_path}")

        except Exception as e:
            logger.warning(f"Failed to generate final report: {str(e)}")

    def get_callback(self) -> Optional[keras.callbacks.Callback]:
        """
        Get a Keras callback for integration with model training.

        Returns:
            Keras callback or None
        """
        return TrainingMonitorCallback(self)

    def add_progress_callback(self, callback: Callable) -> None:
        """
        Add a progress callback function.

        Args:
            callback: Function to call on progress updates
        """
        self.progress_callbacks.append(callback)

    def add_early_stop_callback(self, callback: Callable) -> None:
        """
        Add an early stopping callback function.

        Args:
            callback: Function to call when early stopping is triggered
        """
        self.early_stop_callbacks.append(callback)

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get current training statistics.

        Returns:
            Dictionary of training statistics
        """
        with self._lock:
            stats = {
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "progress_pct": (
                    (self.current_epoch / self.total_epochs) * 100
                    if self.total_epochs > 0
                    else 0
                ),
                "best_metric_value": self.best_metric_value,
                "best_epoch": self.best_epoch,
                "wait_count": self.wait_count,
                "patience": self.patience,
                "train_samples": self.train_samples,
                "val_samples": self.val_samples,
            }

            if self.epoch_times:
                stats.update(
                    {
                        "avg_epoch_time": np.mean(self.epoch_times[-10:]),
                        "total_training_time": sum(self.epoch_times),
                        "eta_seconds": np.mean(self.epoch_times[-10:])
                        * (self.total_epochs - self.current_epoch),
                    }
                )

            if self.training_start_time:
                stats["elapsed_time"] = (
                    datetime.now() - self.training_start_time
                ).total_seconds()

            return stats

    def reset(self) -> None:
        """Reset the training monitor state."""
        with self._lock:
            self.training_start_time = None
            self.epoch_times = []
            self.training_history = []
            self.best_metric_value = None
            self.best_epoch = 0
            self.wait_count = 0
            self.stopped_epoch = 0
            self.current_epoch = 0


class TrainingMonitorCallback(keras.callbacks.Callback):
    """Keras callback that integrates with TrainingMonitor."""

    def __init__(self, monitor: TrainingMonitor):
        """
        Initialize callback.

        Args:
            monitor: TrainingMonitor instance
        """
        super().__init__()
        self.monitor = monitor

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        self.monitor.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        should_stop = self.monitor.on_epoch_end(epoch, logs)
        if should_stop:
            self.model.stop_training = True
