"""Integration tests for the training pipeline with small datasets."""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import shutil
import yaml
from unittest.mock import Mock, patch

from diabetes_lstm_pipeline.training.model_trainer import ModelTrainer
from diabetes_lstm_pipeline.training.validation_strategy import ValidationStrategy
from diabetes_lstm_pipeline.training.training_monitor import TrainingMonitor
from diabetes_lstm_pipeline.training.checkpoint_manager import CheckpointManager
from diabetes_lstm_pipeline.training.training_utils import (
    TrainingHistoryLogger,
    TrainingVisualizer,
)


@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "model": {
            "sequence_length": 10,
            "lstm_units": [32, 16],
            "dropout_rate": 0.2,
            "recurrent_dropout": 0.1,
            "dense_units": [16, 8],
            "activation": "relu",
            "output_activation": "linear",
            "learning_rate": 0.01,
            "l1_regularization": 0.0,
            "l2_regularization": 0.001,
            "loss_function": "mse",
        },
        "training": {
            "batch_size": 8,
            "epochs": 5,
            "validation_split": 0.2,
            "early_stopping_patience": 3,
            "lr_patience": 2,
            "verbose": 0,
        },
        "validation": {
            "cv_splits": 3,
            "test_size": 0.2,
            "gap": 0,
            "time_aware": True,
            "participant_aware": False,
            "min_train_samples": 10,
        },
        "evaluation": {
            "target_glucose_range": [70, 180],
            "hypoglycemia_threshold": 70,
            "hyperglycemia_threshold": 180,
        },
        "checkpointing": {
            "save_best_only": True,
            "save_frequency": 2,
            "max_checkpoints": 3,
            "monitor_metric": "val_loss",
            "mode": "min",
        },
        "monitor": {
            "min_delta": 0.001,
            "log_frequency": 1,
            "save_frequency": 2,
            "plot_frequency": 5,
        },
    }


@pytest.fixture
def small_dataset():
    """Create a small synthetic dataset for testing."""
    np.random.seed(42)

    # Generate synthetic glucose-like time series data
    n_samples = 100
    sequence_length = 10
    n_features = 5

    # Create sequences
    X = np.random.normal(100, 20, (n_samples, sequence_length, n_features))
    X = np.clip(X, 40, 400)  # Realistic glucose range

    # Create targets (next glucose value)
    y = np.random.normal(120, 25, (n_samples, 1))
    y = np.clip(y, 40, 400)

    return X, y


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestModelTrainer:
    """Test ModelTrainer functionality."""

    def test_trainer_initialization(self, test_config):
        """Test trainer initialization."""
        trainer = ModelTrainer(test_config)

        assert trainer.config == test_config
        assert trainer.batch_size == test_config["training"]["batch_size"]
        assert trainer.epochs == test_config["training"]["epochs"]
        assert trainer.model is None
        assert trainer.training_history is None

    def test_data_preparation(self, test_config, small_dataset):
        """Test data preparation and splitting."""
        trainer = ModelTrainer(test_config)
        X, y = small_dataset

        X_train, y_train, validation_data = trainer.prepare_data(X, y)

        # Check that data was split correctly
        expected_train_size = int(
            len(X) * (1 - test_config["training"]["validation_split"])
        )
        assert len(X_train) == expected_train_size
        assert len(y_train) == expected_train_size
        assert validation_data is not None
        assert len(validation_data[0]) == len(X) - expected_train_size

    def test_model_building(self, test_config, small_dataset):
        """Test model building and compilation."""
        trainer = ModelTrainer(test_config)
        X, y = small_dataset

        input_shape = (X.shape[1], X.shape[2])
        model = trainer.build_and_compile_model(input_shape)

        assert model is not None
        assert trainer.model is not None
        assert model.input_shape[1:] == input_shape
        assert model.output_shape[1] == 1  # Single output for glucose prediction

    def test_training_pipeline(self, test_config, small_dataset, temp_dir):
        """Test complete training pipeline."""
        # Update config with temp directory
        test_config["model_save_dir"] = str(temp_dir / "models")

        trainer = ModelTrainer(test_config)
        X, y = small_dataset

        # Train model
        model = trainer.train(X, y)

        assert model is not None
        assert trainer.training_history is not None
        assert (
            len(trainer.training_history.history["loss"])
            <= test_config["training"]["epochs"]
        )

        # Check that training metadata was recorded
        assert trainer.training_metadata["train_samples"] > 0
        assert trainer.training_metadata["val_samples"] > 0
        assert "start_time" in trainer.training_metadata
        assert "end_time" in trainer.training_metadata

    def test_model_validation(self, test_config, small_dataset):
        """Test model validation."""
        trainer = ModelTrainer(test_config)
        X, y = small_dataset

        # Train model first
        trainer.train(X, y)

        # Validate model
        X_val, y_val = X[-20:], y[-20:]  # Use last 20 samples for validation
        metrics = trainer.validate(X_val, y_val)

        assert isinstance(metrics, dict)
        assert "regression_mae" in metrics
        assert "regression_mse" in metrics
        assert "glucose_mard" in metrics

    def test_model_saving_and_loading(self, test_config, small_dataset, temp_dir):
        """Test model saving and loading."""
        test_config["model_save_dir"] = str(temp_dir / "models")

        trainer = ModelTrainer(test_config)
        X, y = small_dataset

        # Train and save model
        trainer.train(X, y)
        saved_paths = trainer.save_model("test_model")

        assert "model" in saved_paths
        assert Path(saved_paths["model"]).exists()
        assert "metadata" in saved_paths
        assert Path(saved_paths["metadata"]).exists()

        # Load model
        new_trainer = ModelTrainer(test_config)
        loaded_model = new_trainer.load_model(saved_paths["model"])

        assert loaded_model is not None
        assert new_trainer.model is not None


class TestValidationStrategy:
    """Test ValidationStrategy functionality."""

    def test_validation_strategy_initialization(self, test_config):
        """Test validation strategy initialization."""
        strategy = ValidationStrategy(test_config)

        assert strategy.n_splits == test_config["validation"]["cv_splits"]
        assert strategy.test_size == test_config["validation"]["test_size"]
        assert strategy.time_aware == test_config["validation"]["time_aware"]

    def test_train_validation_split(self, test_config, small_dataset):
        """Test train-validation splitting."""
        strategy = ValidationStrategy(test_config)
        X, y = small_dataset

        X_train, X_val, y_train, y_val = strategy.train_validation_split(X, y, 0.2)

        assert len(X_train) + len(X_val) == len(X)
        assert len(y_train) + len(y_val) == len(y)
        assert len(X_val) == int(len(X) * 0.2)

    def test_time_series_cross_validation(self, test_config, small_dataset):
        """Test time-series cross-validation."""
        strategy = ValidationStrategy(test_config)
        X, y = small_dataset

        def mock_model_builder():
            """Mock model builder for testing."""
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(16, input_shape=(X.shape[1], X.shape[2])),
                    tf.keras.layers.Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            return model

        # Reduce splits for faster testing
        strategy.n_splits = 2

        cv_results = strategy.time_series_cross_validate(X, y, mock_model_builder)

        assert "fold_results" in cv_results
        assert "mean_metrics" in cv_results
        assert len(cv_results["fold_results"]) <= strategy.n_splits


class TestTrainingMonitor:
    """Test TrainingMonitor functionality."""

    def test_monitor_initialization(self, test_config):
        """Test monitor initialization."""
        monitor = TrainingMonitor(test_config)

        assert monitor.patience == test_config["training"]["early_stopping_patience"]
        assert monitor.monitor_metric == test_config["monitor"].get(
            "monitor_metric", "val_loss"
        )
        assert monitor.training_start_time is None

    def test_training_monitoring(self, test_config):
        """Test training monitoring functionality."""
        monitor = TrainingMonitor(test_config)

        # Start monitoring
        monitor.start_training(total_epochs=5, train_samples=80, val_samples=20)

        assert monitor.training_start_time is not None
        assert monitor.total_epochs == 5
        assert monitor.train_samples == 80
        assert monitor.val_samples == 20

    def test_early_stopping_logic(self, test_config):
        """Test early stopping logic."""
        monitor = TrainingMonitor(test_config)
        monitor.start_training(total_epochs=10, train_samples=80, val_samples=20)

        # Simulate improving then worsening validation loss
        logs_improving = {"val_loss": 1.0, "loss": 1.2}
        logs_worsening = {"val_loss": 1.5, "loss": 1.0}

        # First epoch - should not stop
        monitor.on_epoch_begin(0)
        should_stop = monitor.on_epoch_end(0, logs_improving)
        assert not should_stop

        # Simulate several epochs of worsening performance
        for epoch in range(1, monitor.patience + 2):
            monitor.on_epoch_begin(epoch)
            should_stop = monitor.on_epoch_end(epoch, logs_worsening)
            if epoch >= monitor.patience:
                assert should_stop
                break

    def test_keras_callback_integration(self, test_config):
        """Test Keras callback integration."""
        monitor = TrainingMonitor(test_config)
        callback = monitor.get_callback()

        assert callback is not None
        assert hasattr(callback, "on_epoch_begin")
        assert hasattr(callback, "on_epoch_end")


class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    def test_checkpoint_manager_initialization(self, test_config, temp_dir):
        """Test checkpoint manager initialization."""
        test_config["checkpoint_dir"] = str(temp_dir / "checkpoints")

        manager = CheckpointManager(test_config)

        assert manager.checkpoint_dir.exists()
        assert manager.best_models_dir.exists()
        assert manager.periodic_dir.exists()
        assert manager.resume_dir.exists()

    def test_checkpoint_creation_and_loading(
        self, test_config, small_dataset, temp_dir
    ):
        """Test checkpoint creation and loading."""
        test_config["checkpoint_dir"] = str(temp_dir / "checkpoints")

        manager = CheckpointManager(test_config)
        X, y = small_dataset

        # Create a simple model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(16, input_shape=(X.shape[1], X.shape[2])),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Create checkpoint
        logs = {"val_loss": 0.5, "loss": 0.6, "mae": 0.3}
        checkpoint_path = manager.create_checkpoint(
            model, epoch=1, logs=logs, run_id="test_run"
        )

        assert checkpoint_path is not None
        assert checkpoint_path.exists()

        # Load checkpoint
        loaded_model, metadata, optimizer_state = manager.load_checkpoint(
            checkpoint_path
        )

        assert loaded_model is not None
        assert metadata["epoch"] == 1
        assert metadata["run_id"] == "test_run"

    def test_best_model_tracking(self, test_config, small_dataset, temp_dir):
        """Test best model tracking."""
        test_config["checkpoint_dir"] = str(temp_dir / "checkpoints")

        manager = CheckpointManager(test_config)
        X, y = small_dataset

        # Create a simple model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(16, input_shape=(X.shape[1], X.shape[2])),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Create checkpoints with improving then worsening loss
        logs1 = {"val_loss": 1.0, "loss": 1.2}
        logs2 = {"val_loss": 0.8, "loss": 1.0}  # Better
        logs3 = {"val_loss": 0.9, "loss": 0.9}  # Worse

        manager.create_checkpoint(model, epoch=1, logs=logs1, run_id="test_run")
        assert manager.best_metric_value == 1.0

        manager.create_checkpoint(model, epoch=2, logs=logs2, run_id="test_run")
        assert manager.best_metric_value == 0.8

        manager.create_checkpoint(model, epoch=3, logs=logs3, run_id="test_run")
        assert manager.best_metric_value == 0.8  # Should remain the best


class TestTrainingUtils:
    """Test training utilities."""

    def test_history_logger(self, test_config, temp_dir):
        """Test training history logger."""
        test_config["log_dir"] = str(temp_dir / "logs")

        logger = TrainingHistoryLogger(test_config)

        # Start a run
        logger.start_run("test_run", test_config)

        # Log some epochs
        logger.log_epoch(0, {"loss": 1.0, "val_loss": 1.2, "mae": 0.5})
        logger.log_epoch(1, {"loss": 0.8, "val_loss": 1.0, "mae": 0.4})

        # End run
        logger.end_run({"final_mae": 0.4})

        assert logger.current_run is None
        assert len(logger.training_runs) == 1
        assert logger.training_runs[0]["run_id"] == "test_run"

    def test_visualizer(self, test_config, temp_dir):
        """Test training visualizer."""
        test_config["plots_dir"] = str(temp_dir / "plots")

        visualizer = TrainingVisualizer(test_config)

        # Create mock history
        history = {
            "loss": [1.0, 0.8, 0.6, 0.5],
            "val_loss": [1.2, 1.0, 0.8, 0.7],
            "mae": [0.5, 0.4, 0.3, 0.25],
            "val_mae": [0.6, 0.5, 0.4, 0.35],
        }

        # Create plot
        plot_path = visualizer.plot_training_history(history, title="Test Training")

        assert plot_path is not None
        assert plot_path.exists()


class TestIntegrationWorkflow:
    """Test complete integration workflow."""

    def test_complete_training_workflow(self, test_config, small_dataset, temp_dir):
        """Test complete training workflow with all components."""
        # Setup directories
        test_config["model_save_dir"] = str(temp_dir / "models")
        test_config["checkpoint_dir"] = str(temp_dir / "checkpoints")
        test_config["log_dir"] = str(temp_dir / "logs")
        test_config["plots_dir"] = str(temp_dir / "plots")

        # Reduce epochs for faster testing
        test_config["training"]["epochs"] = 3

        X, y = small_dataset

        # Initialize trainer
        trainer = ModelTrainer(test_config)

        # Train model
        model = trainer.train(X, y)

        # Validate model
        X_val, y_val = X[-20:], y[-20:]
        validation_metrics = trainer.validate(X_val, y_val)

        # Evaluate on test set
        X_test, y_test = X[-10:], y[-10:]
        test_results = trainer.evaluate_on_test_set(X_test, y_test)

        # Save model
        saved_paths = trainer.save_model("integration_test_model")

        # Verify all components worked
        assert model is not None
        assert validation_metrics is not None
        assert test_results is not None
        assert "metrics" in test_results
        assert "predictions" in test_results
        assert Path(saved_paths["model"]).exists()

        # Verify training artifacts were created
        assert (temp_dir / "models").exists()
        # Checkpoints are created within the models directory by default
        assert (temp_dir / "models" / "checkpoints").exists()

    def test_resume_training_workflow(self, test_config, small_dataset, temp_dir):
        """Test training resume functionality."""
        # Setup directories
        test_config["model_save_dir"] = str(temp_dir / "models")
        test_config["checkpoint_dir"] = str(temp_dir / "checkpoints")

        # Reduce epochs for testing
        test_config["training"]["epochs"] = 2

        X, y = small_dataset

        # Initial training
        trainer1 = ModelTrainer(test_config)
        model1 = trainer1.train(X, y)
        saved_paths = trainer1.save_model("resume_test_model")

        # Resume training
        trainer2 = ModelTrainer(test_config)
        model2 = trainer2.resume_training(
            X, y, checkpoint_path=saved_paths["model"], additional_epochs=2
        )

        assert model1 is not None
        assert model2 is not None
        # Models should be different instances but similar architecture
        assert model1.count_params() == model2.count_params()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
