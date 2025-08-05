# src/services/model_service.py

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

# Add the ai-model directory to the path so we can import the pipeline modules
ai_model_path = Path(__file__).parent.parent.parent / "ai-model"
sys.path.insert(0, str(ai_model_path))

logger = logging.getLogger(__name__)


class DiabetesModelService:
    """Service for loading and using the diabetes LSTM model."""

    def __init__(self):
        self.model = None
        self.preprocessing = None
        self.metadata = None
        self.config = None
        self.model_path = None
        self._load_latest_model()

    def _load_latest_model(self):
        """Load the latest trained model from the versions directory."""
        try:
            # Find the latest model
            models_dir = ai_model_path / "models" / "versions"
            if not models_dir.exists():
                logger.error(f"Models directory not found: {models_dir}")
                return

            # Find all diabetes LSTM model directories
            model_dirs = list(models_dir.glob("diabetes_lstm_*"))
            if not model_dirs:
                logger.error("No diabetes LSTM models found")
                return

            # Get the most recent model directory
            latest_model_dir = max(model_dirs, key=lambda x: x.name)
            version_dirs = list(latest_model_dir.glob("*"))
            if not version_dirs:
                logger.error(f"No version directories found in {latest_model_dir}")
                return

            # Get the most recent version directory
            latest_version_dir = max(version_dirs, key=lambda x: x.name)
            self.model_path = latest_version_dir

            logger.info(f"Loading model from: {self.model_path}")

            # Load components - continue even if some fail
            self._load_model()
            self._load_metadata()
            self._load_config()

            # Try to load preprocessing, but don't fail if it doesn't work
            try:
                self._load_preprocessing()
            except Exception as e:
                logger.warning(
                    f"Preprocessing loading failed, continuing without it: {e}"
                )
                self.preprocessing = {}

            if self.model is not None:
                logger.info(
                    f"Successfully loaded model: {self.metadata.get('model_name', 'Unknown')}"
                )
            else:
                logger.error("Model loading failed")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_model(self):
        """Load the Keras model."""
        model_file = self.model_path / "model.keras"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        try:
            # Try to load with custom objects first
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
            logger.info("Model loaded with custom objects")

        except ImportError:
            # Fallback to loading without custom objects
            logger.warning("Custom metrics not available, loading model without them")
            self.model = keras.models.load_model(model_file, compile=False)
            # Recompile with standard metrics
            self.model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
            logger.info("Model loaded and recompiled with standard metrics")

    def _load_metadata(self):
        """Load model metadata."""
        metadata_file = self.model_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            logger.warning("Metadata file not found")
            self.metadata = {}

    def _load_config(self):
        """Load model configuration."""
        config_file = self.model_path / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                self.config = json.load(f)
        else:
            logger.warning("Config file not found")
            self.config = {}

    def _load_preprocessing(self):
        """Load preprocessing components."""
        preprocessing_file = self.model_path / "preprocessing.pkl"
        if preprocessing_file.exists():
            try:
                with open(preprocessing_file, "rb") as f:
                    self.preprocessing = pickle.load(f)
                logger.info("Preprocessing components loaded")
            except Exception as e:
                logger.warning(f"Failed to load preprocessing file: {e}")
                self.preprocessing = {}
        else:
            logger.warning("Preprocessing file not found")
            self.preprocessing = {}

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded():
            return {"error": "Model not loaded"}

        return {
            "model_name": self.metadata.get("model_name", "Unknown"),
            "version": self.metadata.get("version", "Unknown"),
            "created_at": self.metadata.get("created_at", "Unknown"),
            "input_shape": self.metadata.get("input_shape", "Unknown"),
            "output_shape": self.metadata.get("output_shape", "Unknown"),
            "total_parameters": self.metadata.get("total_parameters", "Unknown"),
            "model_path": str(self.model_path) if self.model_path else "Unknown",
        }

    def predict_glucose(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make glucose predictions using the loaded model.

        Args:
            input_data: Input data with shape (batch_size, sequence_length, n_features)

        Returns:
            Predicted glucose values
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        try:
            predictions = self.model.predict(input_data, verbose=0)
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def preprocess_data(self, raw_data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess raw data for model input.

        Args:
            raw_data: Raw input data as DataFrame

        Returns:
            Preprocessed data ready for model input
        """
        if not self.preprocessing:
            raise RuntimeError("Preprocessing components not loaded")

        try:
            # This is a simplified preprocessing - you may need to adapt based on your actual preprocessing pipeline
            if "scaler" in self.preprocessing:
                scaler = self.preprocessing["scaler"]
                scaled_data = scaler.transform(raw_data)
                return scaled_data
            else:
                # Fallback to basic preprocessing
                return raw_data.values
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise


# Global instance
diabetes_model_service = DiabetesModelService()
