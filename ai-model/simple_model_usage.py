#!/usr/bin/env python3
"""
Simple example showing how to use the generated diabetes LSTM models.

This script demonstrates:
1. How to load a saved model (without custom metrics)
2. How to prepare input data
3. How to make predictions
4. How to interpret the results
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDiabetesPredictor:
    """Simple class to load and use the diabetes LSTM model for predictions."""

    def __init__(self, model_path: str):
        """
        Initialize the predictor with a saved model.

        Args:
            model_path: Path to the model directory
        """
        self.model_path = Path(model_path)
        self.model = None
        self.config = None
        self.metadata = None

        # Load components
        self._load_model()
        self._load_config()
        self._load_metadata()

        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Input shape: {self.metadata['input_shape']}")
        logger.info(f"Output shape: {self.metadata['output_shape']}")

    def _load_model(self):
        """Load the Keras model without custom metrics."""
        model_file = self.model_path / "model.keras"
        if model_file.exists():
            # Load model with custom_objects to handle custom metrics
            try:
                # First try to load with custom objects
                from diabetes_lstm_pipeline.model_architecture.model_builder import (
                    GlucoseMARD,
                    TimeInRangeAccuracy,
                    GlucoseAwareLoss,
                )

                custom_objects = {
                    "GlucoseMARD": GlucoseMARD,
                    "TimeInRangeAccuracy": TimeInRangeAccuracy,
                    "GlucoseAwareLoss": GlucoseAwareLoss,
                }

                self.model = keras.models.load_model(
                    model_file, custom_objects=custom_objects
                )
                logger.info("Keras model loaded successfully with custom objects")

            except ImportError:
                # If custom objects can't be imported, load without them
                logger.warning(
                    "Custom metrics not available, loading model without them"
                )
                self.model = keras.models.load_model(model_file, compile=False)
                # Recompile with standard metrics
                self.model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
                logger.info("Keras model loaded and recompiled with standard metrics")
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")

    def _load_config(self):
        """Load model configuration."""
        config_file = self.model_path / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                self.config = json.load(f)
            logger.info("Configuration loaded successfully")
        else:
            logger.warning("Config file not found")
            self.config = {}

    def _load_metadata(self):
        """Load model metadata."""
        metadata_file = self.model_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
            logger.info("Metadata loaded successfully")
        else:
            logger.warning("Metadata file not found")
            self.metadata = {}

    def prepare_input_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare input data for the model.

        Args:
            data: DataFrame with time series data

        Returns:
            Preprocessed input array with shape (n_samples, sequence_length, n_features)
        """
        # Get model configuration - use the actual input shape from metadata
        input_shape = self.metadata.get("input_shape", [12, 62])
        sequence_length = input_shape[0]  # Should be 12
        n_features = input_shape[1]  # Should be 62

        logger.info(f"Model expects input shape: ({sequence_length}, {n_features})")

        # Ensure data is sorted by time
        if "EventDateTime" in data.columns:
            data = data.sort_values("EventDateTime")

        # For demonstration, we'll create synthetic data that matches the expected shape
        # In a real scenario, you would need to preprocess your data to match the training format

        # Create synthetic sequences that match the expected input shape
        n_samples = max(1, len(data) - sequence_length + 1)

        # Generate synthetic data with the correct shape
        # This is a simplified approach - in practice, you'd need to match the exact preprocessing
        synthetic_sequences = []

        for i in range(n_samples):
            # Create a sequence with the expected shape (12, 62)
            sequence = np.random.normal(0, 1, (sequence_length, n_features))

            # If we have real data, we can use it for the first few features
            if i < len(data) - sequence_length + 1:
                # Use real CGM data for the first feature
                cgm_data = data.iloc[i : i + sequence_length]["CGM"].values
                if len(cgm_data) == sequence_length:
                    sequence[:, 0] = cgm_data  # First feature is CGM

            synthetic_sequences.append(sequence)

        X = np.array(synthetic_sequences)

        logger.info(f"Prepared input data with shape: {X.shape}")
        return X

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model.

        Args:
            data: Input DataFrame with time series data

        Returns:
            Predicted glucose values
        """
        # Prepare input data
        X = self.prepare_input_data(data)

        # Make predictions
        predictions = self.model.predict(X, verbose=0)

        logger.info(f"Made predictions for {len(predictions)} samples")
        return predictions.flatten()

    def predict_single_sequence(self, sequence: np.ndarray) -> float:
        """
        Make a prediction for a single sequence.

        Args:
            sequence: Input sequence with shape (sequence_length, n_features)

        Returns:
            Single predicted glucose value
        """
        # Ensure correct shape
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, *sequence.shape)

        # Make prediction
        prediction = self.model.predict(sequence, verbose=0)

        return float(prediction[0, 0])

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.metadata.get("model_name", "Unknown"),
            "version": self.metadata.get("version", "Unknown"),
            "input_shape": self.metadata.get("input_shape", []),
            "output_shape": self.metadata.get("output_shape", []),
            "total_parameters": self.metadata.get("total_parameters", 0),
            "target_name": self.metadata.get("target_name", "glucose"),
            "created_at": self.metadata.get("created_at", "Unknown"),
        }


def create_sample_data() -> pd.DataFrame:
    """Create sample data for demonstration purposes."""
    # Generate sample time series data
    # np.random.seed(42)
    n_samples = 100

    # Create timestamps (5-minute intervals)
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=n_samples, freq="5min")

    # Generate realistic sample data
    data = pd.DataFrame(
        {
            "EventDateTime": timestamps,
            "CGM": np.random.normal(150, 30, n_samples).clip(70, 300),  # Glucose values
            "Basal": np.random.uniform(0.5, 2.0, n_samples),  # Basal insulin rates
            "TotalBolusInsulinDelivered": np.random.exponential(5, n_samples).clip(
                0, 20
            ),  # Bolus insulin
            "FoodDelivered": np.random.exponential(30, n_samples).clip(
                0, 100
            ),  # Food intake
            "CarbSize": np.random.exponential(25, n_samples).clip(
                0, 80
            ),  # Carbohydrates
            "CorrectionDelivered": np.random.exponential(2, n_samples).clip(
                0, 10
            ),  # Correction insulin
        }
    )

    return data


def main():
    """Main function demonstrating model usage."""
    print("=== Simple Diabetes LSTM Model Usage Example ===\n")

    # Find the latest model
    models_dir = Path("models/versions")
    if not models_dir.exists():
        print("No models found. Please run the pipeline first to generate a model.")
        return

    # Find the most recent model
    model_dirs = list(models_dir.glob("diabetes_lstm_*"))
    if not model_dirs:
        print("No diabetes LSTM models found. Please run the pipeline first.")
        return

    # Get the most recent model
    latest_model_dir = max(model_dirs, key=lambda x: x.name)
    version_dirs = list(latest_model_dir.glob("*"))
    if not version_dirs:
        print(f"No version directories found in {latest_model_dir}")
        return

    latest_version_dir = max(version_dirs, key=lambda x: x.name)
    model_path = str(latest_version_dir)

    print(f"Loading model from: {model_path}\n")

    try:
        # Load the model
        predictor = SimpleDiabetesPredictor(model_path)

        # Display model information
        model_info = predictor.get_model_info()
        print("Model Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        print()

        # Create sample data
        print("Creating sample data...")
        sample_data = create_sample_data()
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample data columns: {list(sample_data.columns)}")
        print()

        # Make predictions
        print("Making predictions...")
        predictions = predictor.predict(sample_data)
        print(f"Predictions shape: {predictions.shape}")
        print(
            f"Prediction range: {predictions.min():.1f} - {predictions.max():.1f} mg/dL"
        )
        print()

        # Show some sample predictions
        print("Sample Predictions:")
        sequence_length = model_info["input_shape"][0]
        for i in range(min(5, len(predictions))):
            actual_cgm = sample_data.iloc[i + sequence_length][
                "CGM"
            ]  # Skip first points (sequence length)
            predicted = predictions[i]
            print(
                f"  Sample {i+1}: Actual CGM = {actual_cgm:.1f} mg/dL, Predicted = {predicted:.1f} mg/dL"
            )
        print()

        # Demonstrate single sequence prediction
        print("Single Sequence Prediction:")
        # Create a single sequence with the correct shape (12, 62)
        single_sequence = np.random.normal(0, 1, (12, 62))
        single_prediction = predictor.predict_single_sequence(single_sequence)
        print(f"  Input sequence shape: {single_sequence.shape}")
        print(f"  Predicted glucose: {single_prediction:.1f} mg/dL")
        print()

        print("=== Model Usage Example Completed Successfully ===")

    except Exception as e:
        print(f"Error loading or using model: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
