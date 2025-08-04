"""Model trainer for orchestrating the complete LSTM training process."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from pathlib import Path
import pickle
import json
from datetime import datetime
import os

from ..model_architecture.model_builder import LSTMModelBuilder
from ..model_architecture.metrics_calculator import MetricsCalculator
from .validation_strategy import ValidationStrategy
from .training_monitor import TrainingMonitor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrates the complete LSTM model training process."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model trainer.

        Args:
            config: Configuration dictionary containing training parameters
        """
        self.config = config
        self.training_config = config.get("training", {})
        self.model_config = config.get("model", {})

        # Initialize components
        self.model_builder = LSTMModelBuilder(config)
        self.metrics_calculator = MetricsCalculator(config)
        self.validation_strategy = ValidationStrategy(config)
        self.training_monitor = TrainingMonitor(config)

        # Training parameters
        self.batch_size = self.training_config.get("batch_size", 32)
        self.epochs = self.training_config.get("epochs", 100)
        self.validation_split = self.training_config.get("validation_split", 0.2)
        self.verbose = self.training_config.get("verbose", 1)

        # Model persistence
        self.model_save_dir = Path(config.get("model_save_dir", "models"))
        self.checkpoint_dir = self.model_save_dir / "checkpoints"
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.model = None
        self.training_history = None
        self.best_model_path = None
        self.training_metadata = {}

    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Prepare training data with validation splitting if needed.

        Args:
            X: Input sequences
            y: Target values
            validation_data: Optional validation data tuple

        Returns:
            Tuple of (X_train, y_train, validation_data)
        """
        logger.info(f"Preparing training data with shape X: {X.shape}, y: {y.shape}")

        # If no validation data provided, use validation strategy to split
        if validation_data is None:
            X_train, X_val, y_train, y_val = (
                self.validation_strategy.train_validation_split(
                    X, y, validation_split=self.validation_split
                )
            )
            validation_data = (X_val, y_val)
            logger.info(
                f"Split data - Train: {X_train.shape}, Validation: {X_val.shape}"
            )
        else:
            X_train, y_train = X, y
            logger.info(
                f"Using provided validation data - Val: {validation_data[0].shape}"
            )

        return X_train, y_train, validation_data

    def build_and_compile_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build and compile the LSTM model.

        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)

        Returns:
            Compiled Keras model
        """
        logger.info(f"Building model with input shape: {input_shape}")

        # Build model
        model = self.model_builder.build_model(input_shape)

        # Compile model
        loss_function = self.model_config.get("loss_function", "mse")
        model = self.model_builder.compile_model(model, loss_function)

        # Log model summary
        model_summary = self.model_builder.get_model_summary(model)
        logger.info(f"Model built with {model_summary['total_params']} parameters")

        self.model = model
        return model

    def setup_callbacks(self, run_id: str) -> List[keras.callbacks.Callback]:
        """
        Set up training callbacks including checkpointing and monitoring.

        Args:
            run_id: Unique identifier for this training run

        Returns:
            List of Keras callbacks
        """
        # Model checkpoint path
        checkpoint_path = self.checkpoint_dir / f"model_{run_id}_best.keras"

        # Get callbacks from model builder
        callbacks = self.model_builder.get_callbacks(checkpoint_path)

        # Add training monitor callback
        monitor_callback = self.training_monitor.get_callback()
        if monitor_callback:
            callbacks.append(monitor_callback)

        # Add custom CSV logger with run-specific filename
        csv_path = Path("logs") / f"training_history_{run_id}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_logger = keras.callbacks.CSVLogger(str(csv_path), append=False)
        callbacks.append(csv_logger)

        self.best_model_path = checkpoint_path
        return callbacks

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> keras.Model:
        """
        Train the LSTM model with proper validation and monitoring.

        Args:
            X_train: Training input sequences
            y_train: Training target values
            validation_data: Optional validation data tuple
            resume_from_checkpoint: Optional path to checkpoint to resume from

        Returns:
            Trained Keras model
        """
        logger.info("Starting model training")

        # Generate unique run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare data
        X_train, y_train, validation_data = self.prepare_data(
            X_train, y_train, validation_data
        )

        # Build model if not already built or if resuming from checkpoint
        if self.model is None or resume_from_checkpoint:
            if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
                logger.info(f"Loading model from checkpoint: {resume_from_checkpoint}")
                self.model = keras.models.load_model(resume_from_checkpoint)
            else:
                input_shape = (X_train.shape[1], X_train.shape[2])
                self.model = self.build_and_compile_model(input_shape)

        # Setup callbacks
        callbacks = self.setup_callbacks(run_id)

        # Initialize training monitor
        self.training_monitor.start_training(
            total_epochs=self.epochs,
            train_samples=len(X_train),
            val_samples=len(validation_data[0]) if validation_data else 0,
        )

        # Store training metadata
        self.training_metadata = {
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "train_samples": len(X_train),
            "val_samples": len(validation_data[0]) if validation_data else 0,
            "input_shape": X_train.shape,
            "target_shape": y_train.shape,
            "config": self.config.copy(),
        }

        try:
            # Train model
            logger.info(
                f"Training model for {self.epochs} epochs with batch size {self.batch_size}"
            )
            history = self.model.fit(
                X_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=self.verbose,
                shuffle=True,
            )

            self.training_history = history

            # Update metadata
            self.training_metadata.update(
                {
                    "end_time": datetime.now().isoformat(),
                    "epochs_completed": len(history.history["loss"]),
                    "best_val_loss": min(
                        history.history.get("val_loss", [float("inf")])
                    ),
                    "final_train_loss": history.history["loss"][-1],
                    "final_val_loss": (
                        history.history.get("val_loss", [None])[-1]
                        if history.history.get("val_loss")
                        else None
                    ),
                }
            )

            logger.info(
                f"Training completed successfully in {len(history.history['loss'])} epochs"
            )

            # Load best model if checkpoint was saved
            if self.best_model_path and self.best_model_path.exists():
                logger.info(f"Loading best model from {self.best_model_path}")
                self.model = keras.models.load_model(self.best_model_path)

            return self.model

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.training_metadata["error"] = str(e)
            self.training_metadata["end_time"] = datetime.now().isoformat()
            raise

        finally:
            # Stop training monitor
            self.training_monitor.stop_training()

    def validate(
        self, X_val: np.ndarray, y_val: np.ndarray, use_cross_validation: bool = False
    ) -> Dict[str, float]:
        """
        Validate the trained model using various strategies.

        Args:
            X_val: Validation input sequences
            y_val: Validation target values
            use_cross_validation: Whether to use cross-validation

        Returns:
            Dictionary of validation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before validation")

        logger.info(f"Validating model on {len(X_val)} samples")

        if use_cross_validation:
            # Use cross-validation strategy
            cv_results = self.validation_strategy.cross_validate(
                self.model, X_val, y_val
            )
            return cv_results
        else:
            # Simple validation
            y_pred = self.model.predict(X_val, batch_size=self.batch_size, verbose=0)
            y_pred = y_pred.flatten()
            y_val = y_val.flatten()

            # Calculate comprehensive metrics
            metrics = self.metrics_calculator.calculate_comprehensive_metrics(
                y_val, y_pred, self.training_history
            )

            return metrics

    def evaluate_on_test_set(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate the trained model on test set.

        Args:
            X_test: Test input sequences
            y_test: Test target values

        Returns:
            Dictionary of test metrics and predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")

        logger.info(f"Evaluating model on test set with {len(X_test)} samples")

        # Make predictions
        y_pred = self.model.predict(X_test, batch_size=self.batch_size, verbose=0)
        y_pred = y_pred.flatten()
        y_test = y_test.flatten()

        # Calculate comprehensive metrics
        test_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            y_test, y_pred
        )

        # Generate evaluation report
        report_path = (
            Path("reports")
            / f"test_evaluation_{self.training_metadata.get('run_id', 'unknown')}.txt"
        )
        report = self.metrics_calculator.generate_metrics_report(
            test_metrics, report_path
        )

        return {
            "metrics": test_metrics,
            "predictions": y_pred,
            "true_values": y_test,
            "report": report,
            "report_path": str(report_path),
        }

    def save_model(
        self, model_name: Optional[str] = None, include_preprocessing: bool = True
    ) -> Dict[str, str]:
        """
        Save the trained model and associated metadata.

        Args:
            model_name: Optional custom model name
            include_preprocessing: Whether to save preprocessing components

        Returns:
            Dictionary of saved file paths
        """
        if self.model is None:
            raise ValueError("No model to save")

        run_id = self.training_metadata.get(
            "run_id", datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        if model_name is None:
            model_name = f"lstm_glucose_predictor_{run_id}"

        # Save paths
        model_path = self.model_save_dir / f"{model_name}.keras"
        metadata_path = self.model_save_dir / f"{model_name}_metadata.json"
        config_path = self.model_save_dir / f"{model_name}_config.json"

        saved_paths = {}

        # Save model
        self.model.save(model_path)
        saved_paths["model"] = str(model_path)
        logger.info(f"Model saved to {model_path}")

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(self.training_metadata, f, indent=2, default=str)
        saved_paths["metadata"] = str(metadata_path)

        # Save configuration
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        saved_paths["config"] = str(config_path)

        logger.info(f"Model artifacts saved with prefix: {model_name}")
        return saved_paths

    def load_model(
        self, model_path: Union[str, Path], load_metadata: bool = True
    ) -> keras.Model:
        """
        Load a previously saved model.

        Args:
            model_path: Path to the saved model
            load_metadata: Whether to load associated metadata

        Returns:
            Loaded Keras model
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from {model_path}")
        self.model = keras.models.load_model(model_path)

        if load_metadata:
            # Try to load metadata
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.training_metadata = json.load(f)
                logger.info("Loaded training metadata")

            # Try to load config
            config_path = model_path.parent / f"{model_path.stem}_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
                    # Update current config with loaded config
                    self.config.update(loaded_config)
                logger.info("Loaded model configuration")

        return self.model

    def get_training_history(self) -> Optional[Dict[str, List[float]]]:
        """
        Get training history if available.

        Returns:
            Training history dictionary or None
        """
        if self.training_history and hasattr(self.training_history, "history"):
            return self.training_history.history
        return None

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training process.

        Returns:
            Dictionary containing training summary
        """
        summary = {
            "metadata": self.training_metadata.copy(),
            "model_info": None,
            "training_metrics": None,
        }

        if self.model:
            model_summary = self.model_builder.get_model_summary(self.model)
            summary["model_info"] = model_summary

        if self.training_history:
            training_metrics = self.metrics_calculator.calculate_training_metrics(
                self.training_history
            )
            summary["training_metrics"] = training_metrics

        return summary

    def resume_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        checkpoint_path: str,
        additional_epochs: int = 50,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> keras.Model:
        """
        Resume training from a checkpoint.

        Args:
            X_train: Training input sequences
            y_train: Training target values
            checkpoint_path: Path to checkpoint to resume from
            additional_epochs: Number of additional epochs to train
            validation_data: Optional validation data

        Returns:
            Trained model
        """
        logger.info(
            f"Resuming training from {checkpoint_path} for {additional_epochs} epochs"
        )

        # Update epochs for additional training
        original_epochs = self.epochs
        self.epochs = additional_epochs

        try:
            # Train with checkpoint
            model = self.train(
                X_train,
                y_train,
                validation_data,
                resume_from_checkpoint=checkpoint_path,
            )
            return model
        finally:
            # Restore original epochs setting
            self.epochs = original_epochs
