"""
Unit tests for model persistence and versioning system.

Tests cover:
- Model saving and loading functionality
- Preprocessing component serialization
- Metadata tracking and versioning
- Model comparison and information retrieval
- Error handling and edge cases
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from unittest.mock import Mock, patch
from datetime import datetime

from diabetes_lstm_pipeline.model_persistence.model_persistence import (
    ModelPersistence,
    ModelMetadata,
)


class TestModelPersistence:
    """Test cases for ModelPersistence class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        return {
            "model_persistence": {
                "base_dir": str(temp_dir),
                "versioning_strategy": "timestamp",
                "max_versions": 5,
                "compress_preprocessing": True,
                "compression_level": 6,
            }
        }

    @pytest.fixture
    def sample_model(self):
        """Create a simple test model."""
        model = keras.Sequential(
            [
                keras.layers.LSTM(64, input_shape=(60, 10), return_sequences=True),
                keras.layers.LSTM(32),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    @pytest.fixture
    def preprocessing_components(self):
        """Create sample preprocessing components."""
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, 10))

        return {
            "scaler": scaler,
            "feature_names": [f"feature_{i}" for i in range(10)],
            "target_name": "glucose",
            "feature_engineering_config": {
                "sequence_length": 60,
                "prediction_horizon": 1,
            },
            "preprocessing_stats": {
                "mean": np.random.randn(10).tolist(),
                "std": np.random.randn(10).tolist(),
            },
        }

    @pytest.fixture
    def training_metadata(self):
        """Create sample training metadata."""
        return {
            "run_id": "test_run_20240101_120000",
            "start_time": "2024-01-01T12:00:00",
            "end_time": "2024-01-01T13:00:00",
            "training_duration_seconds": 3600.0,
            "epochs_completed": 50,
            "train_samples": 1000,
            "val_samples": 200,
            "config": {
                "model": {
                    "lstm_units": [64, 32],
                    "dropout_rate": 0.2,
                    "learning_rate": 0.001,
                },
                "training": {
                    "batch_size": 32,
                    "epochs": 100,
                    "early_stopping_patience": 15,
                },
            },
        }

    @pytest.fixture
    def performance_metrics(self):
        """Create sample performance metrics."""
        return {
            "training": {"loss": 0.025, "mae": 0.12, "mard": 8.5},
            "validation": {"loss": 0.032, "mae": 0.15, "mard": 9.2},
            "test": {"loss": 0.030, "mae": 0.14, "mard": 8.8},
        }

    @pytest.fixture
    def model_persistence(self, config):
        """Create ModelPersistence instance."""
        return ModelPersistence(config)

    def test_initialization(self, model_persistence, temp_dir):
        """Test ModelPersistence initialization."""
        assert model_persistence.base_dir == temp_dir
        assert model_persistence.versioning_strategy == "timestamp"
        assert model_persistence.max_versions == 5

        # Check that directories are created
        assert model_persistence.models_dir.exists()
        assert model_persistence.preprocessing_dir.exists()
        assert model_persistence.metadata_dir.exists()
        assert model_persistence.versions_dir.exists()

    def test_save_model_basic(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test basic model saving functionality."""
        saved_paths = model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="test_model",
            description="Test model for unit testing",
        )

        # Check that all paths are returned
        assert "model" in saved_paths
        assert "preprocessing" in saved_paths
        assert "metadata" in saved_paths
        assert "config" in saved_paths
        assert "version_dir" in saved_paths

        # Check that files exist
        for path_key, path_value in saved_paths.items():
            if path_key != "version_dir":
                assert Path(path_value).exists()

    def test_save_model_with_custom_parameters(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test model saving with custom parameters."""
        saved_paths = model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="custom_model",
            description="Custom test model",
            tags=["test", "lstm", "diabetes"],
            author="test_user",
        )

        # Load metadata and verify custom parameters
        metadata_path = Path(saved_paths["metadata"])
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["model_name"] == "custom_model"
        assert metadata["description"] == "Custom test model"
        assert metadata["tags"] == ["test", "lstm", "diabetes"]
        assert metadata["author"] == "test_user"

    def test_load_model_basic(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test basic model loading functionality."""
        # First save a model
        model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="test_load_model",
        )

        # Load the model
        loaded_components = model_persistence.load_model("test_load_model")

        # Verify loaded components
        assert "model" in loaded_components
        assert "metadata" in loaded_components
        assert "preprocessing" in loaded_components
        assert "config" in loaded_components

        # Check model architecture
        loaded_model = loaded_components["model"]
        assert len(loaded_model.layers) == len(sample_model.layers)
        assert loaded_model.input_shape == sample_model.input_shape
        assert loaded_model.output_shape == sample_model.output_shape

        # Check preprocessing components
        loaded_preprocessing = loaded_components["preprocessing"]
        assert "scaler" in loaded_preprocessing
        assert "feature_names" in loaded_preprocessing
        assert (
            loaded_preprocessing["feature_names"]
            == preprocessing_components["feature_names"]
        )

    def test_load_model_without_preprocessing(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test loading model without preprocessing components."""
        # Save model
        model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="test_no_preprocessing",
        )

        # Load without preprocessing
        loaded_components = model_persistence.load_model(
            "test_no_preprocessing", load_preprocessing=False
        )

        assert "model" in loaded_components
        assert "metadata" in loaded_components
        assert "preprocessing" not in loaded_components

    def test_load_nonexistent_model(self, model_persistence):
        """Test loading a model that doesn't exist."""
        with pytest.raises(ValueError):
            model_persistence.load_model("nonexistent_model")

    def test_load_specific_version(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test loading a specific model version."""
        # Save first version
        saved_paths1 = model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="versioned_model",
        )

        # Extract version from path
        version1 = Path(saved_paths1["version_dir"]).name

        # Save second version (modify performance slightly)
        modified_metrics = performance_metrics.copy()
        modified_metrics["validation"]["loss"] = 0.025

        saved_paths2 = model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=modified_metrics,
            model_name="versioned_model",
        )

        version2 = Path(saved_paths2["version_dir"]).name

        # Load specific version
        loaded_v1 = model_persistence.load_model("versioned_model", version=version1)
        loaded_v2 = model_persistence.load_model("versioned_model", version=version2)

        assert loaded_v1["version"] == version1
        assert loaded_v2["version"] == version2

        # Check that metrics are different
        assert (
            loaded_v1["metadata"]["validation_metrics"]["loss"]
            != loaded_v2["metadata"]["validation_metrics"]["loss"]
        )

    def test_list_models(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test listing available models."""
        # Initially no models
        models = model_persistence.list_models()
        assert len(models) == 0

        # Save a few models
        model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="model_1",
        )

        model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="model_2",
        )

        # List models
        models = model_persistence.list_models()
        assert len(models) == 2

        model_names = [model["model_name"] for model in models]
        assert "model_1" in model_names
        assert "model_2" in model_names

        # Check model structure
        for model in models:
            assert "model_name" in model
            assert "versions" in model
            assert "latest_version" in model
            assert len(model["versions"]) > 0

    def test_delete_model_version(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test deleting a specific model version."""
        # Save two versions
        saved_paths1 = model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="delete_test_model",
        )

        version1 = Path(saved_paths1["version_dir"]).name

        saved_paths2 = model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="delete_test_model",
        )

        version2 = Path(saved_paths2["version_dir"]).name

        # Verify both versions exist
        models = model_persistence.list_models()
        delete_model = next(m for m in models if m["model_name"] == "delete_test_model")
        assert len(delete_model["versions"]) == 2

        # Delete one version
        success = model_persistence.delete_model("delete_test_model", version=version1)
        assert success

        # Verify only one version remains
        models = model_persistence.list_models()
        delete_model = next(m for m in models if m["model_name"] == "delete_test_model")
        assert len(delete_model["versions"]) == 1
        assert delete_model["versions"][0]["version"] == version2

    def test_delete_entire_model(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test deleting an entire model (all versions)."""
        # Save model
        model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="delete_entire_model",
        )

        # Verify model exists
        models = model_persistence.list_models()
        assert any(m["model_name"] == "delete_entire_model" for m in models)

        # Delete entire model
        success = model_persistence.delete_model("delete_entire_model")
        assert success

        # Verify model is gone
        models = model_persistence.list_models()
        assert not any(m["model_name"] == "delete_entire_model" for m in models)

    def test_get_model_info(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test getting detailed model information."""
        # Save model
        model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="info_test_model",
            description="Model for info testing",
        )

        # Get model info
        info = model_persistence.get_model_info("info_test_model")

        # Verify information structure
        assert "model_name" in info
        assert "version" in info
        assert "created_at" in info
        assert "training_metrics" in info
        assert "validation_metrics" in info
        assert "model_summary" in info
        assert "total_parameters" in info

        assert info["model_name"] == "info_test_model"
        assert info["description"] == "Model for info testing"

    def test_compare_models(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test comparing two model versions."""
        # Save first model
        saved_paths1 = model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="compare_model_1",
        )

        version1 = Path(saved_paths1["version_dir"]).name

        # Save second model with different performance
        modified_metrics = performance_metrics.copy()
        modified_metrics["validation"]["loss"] = 0.020  # Better performance
        modified_metrics["validation"]["mae"] = 0.10

        saved_paths2 = model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=modified_metrics,
            model_name="compare_model_2",
        )

        version2 = Path(saved_paths2["version_dir"]).name

        # Compare models
        comparison = model_persistence.compare_models(
            ("compare_model_1", version1), ("compare_model_2", version2)
        )

        # Verify comparison structure
        assert "model1" in comparison
        assert "model2" in comparison
        assert "performance_comparison" in comparison
        assert "architecture_comparison" in comparison
        assert "training_comparison" in comparison

        # Check performance comparison
        perf_comp = comparison["performance_comparison"]
        assert "loss" in perf_comp
        assert "mae" in perf_comp

        # Verify improvement calculation
        loss_comp = perf_comp["loss"]
        assert loss_comp["model1"] == 0.032  # Original validation loss
        assert loss_comp["model2"] == 0.020  # Improved validation loss
        assert (
            loss_comp["improvement_percent"] < 0
        )  # Negative because loss decreased (improved)

    def test_version_generation_timestamp(self, model_persistence):
        """Test timestamp-based version generation."""
        with patch(
            "diabetes_lstm_pipeline.model_persistence.model_persistence.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            version = model_persistence._generate_version("test_model", {})
            assert version == "20240101_120000"

    def test_version_generation_performance(self, temp_dir):
        """Test performance-based version generation."""
        config = {
            "model_persistence": {
                "base_dir": str(temp_dir),
                "versioning_strategy": "performance",
            }
        }

        model_persistence = ModelPersistence(config)

        performance_metrics = {"validation": {"loss": 0.0234}}

        with patch(
            "diabetes_lstm_pipeline.model_persistence.model_persistence.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            version = model_persistence._generate_version(
                "test_model", performance_metrics
            )
            assert version.startswith("0.0234_")
            assert version.endswith("20240101_120000")

    def test_checksum_calculation(self, model_persistence, temp_dir):
        """Test checksum calculation for files."""
        # Create a test file
        test_file = temp_dir / "test_file.txt"
        test_content = "This is test content for checksum calculation"
        test_file.write_text(test_content)

        # Calculate checksum
        checksum = model_persistence._calculate_checksum(test_file)

        # Verify checksum is a valid SHA256 hash
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

        # Verify checksum is consistent
        checksum2 = model_persistence._calculate_checksum(test_file)
        assert checksum == checksum2

        # Verify checksum changes with content
        test_file.write_text("Different content")
        checksum3 = model_persistence._calculate_checksum(test_file)
        assert checksum != checksum3

    def test_cleanup_old_versions(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test cleanup of old model versions."""
        # Set max_versions to 3 for testing
        model_persistence.max_versions = 3

        # Save 5 versions
        for i in range(5):
            model_persistence.save_model(
                model=sample_model,
                preprocessing_components=preprocessing_components,
                training_metadata=training_metadata,
                performance_metrics=performance_metrics,
                model_name="cleanup_test_model",
            )

        # Check that only 3 versions remain
        models = model_persistence.list_models()
        cleanup_model = next(
            m for m in models if m["model_name"] == "cleanup_test_model"
        )
        assert len(cleanup_model["versions"]) == 3

    def test_preprocessing_serialization_with_sklearn_objects(
        self, model_persistence, sample_model, training_metadata, performance_metrics
    ):
        """Test serialization of various sklearn preprocessing objects."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
        from sklearn.decomposition import PCA

        # Create complex preprocessing components
        preprocessing_components = {
            "standard_scaler": StandardScaler().fit(np.random.randn(100, 10)),
            "minmax_scaler": MinMaxScaler().fit(np.random.randn(100, 5)),
            "pca": PCA(n_components=5).fit(np.random.randn(100, 10)),
            "feature_names": [f"feature_{i}" for i in range(10)],
            "custom_dict": {"param1": 1.5, "param2": "test"},
            "numpy_array": np.random.randn(5, 3),
        }

        # Save and load
        model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="sklearn_test_model",
        )

        loaded_components = model_persistence.load_model("sklearn_test_model")
        loaded_preprocessing = loaded_components["preprocessing"]

        # Verify all components are loaded correctly
        assert "standard_scaler" in loaded_preprocessing
        assert "minmax_scaler" in loaded_preprocessing
        assert "pca" in loaded_preprocessing
        assert "feature_names" in loaded_preprocessing
        assert "custom_dict" in loaded_preprocessing
        assert "numpy_array" in loaded_preprocessing

        # Test that scalers work correctly
        test_data = np.random.randn(10, 10)
        original_scaled = preprocessing_components["standard_scaler"].transform(
            test_data
        )
        loaded_scaled = loaded_preprocessing["standard_scaler"].transform(test_data)

        np.testing.assert_array_almost_equal(original_scaled, loaded_scaled)

    def test_model_reproducibility(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test that saved and loaded models produce identical predictions."""
        # Generate test data
        test_data = np.random.randn(10, 60, 10)

        # Get predictions from original model
        original_predictions = sample_model.predict(test_data, verbose=0)

        # Save and load model
        model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="reproducibility_test_model",
        )

        loaded_components = model_persistence.load_model("reproducibility_test_model")
        loaded_model = loaded_components["model"]

        # Get predictions from loaded model
        loaded_predictions = loaded_model.predict(test_data, verbose=0)

        # Verify predictions are identical
        np.testing.assert_array_almost_equal(
            original_predictions, loaded_predictions, decimal=6
        )

    def test_error_handling_corrupted_metadata(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test error handling when metadata is corrupted."""
        # Save model
        saved_paths = model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="corruption_test_model",
        )

        # Corrupt metadata file
        metadata_path = Path(saved_paths["metadata"])
        metadata_path.write_text("corrupted json content")

        # Attempt to load should handle the error gracefully
        with pytest.raises(json.JSONDecodeError):
            model_persistence.load_model("corruption_test_model")

    def test_metadata_structure_completeness(
        self,
        model_persistence,
        sample_model,
        preprocessing_components,
        training_metadata,
        performance_metrics,
    ):
        """Test that saved metadata contains all required fields."""
        saved_paths = model_persistence.save_model(
            model=sample_model,
            preprocessing_components=preprocessing_components,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            model_name="metadata_test_model",
            description="Test description",
            tags=["test", "metadata"],
            author="test_author",
        )

        # Load and verify metadata structure
        metadata_path = Path(saved_paths["metadata"])
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Check all required fields are present
        required_fields = [
            "model_name",
            "version",
            "created_at",
            "model_type",
            "framework",
            "training_config",
            "model_config",
            "training_duration_seconds",
            "epochs_trained",
            "training_metrics",
            "validation_metrics",
            "input_shape",
            "output_shape",
            "feature_names",
            "target_name",
            "total_parameters",
            "trainable_parameters",
            "model_summary",
            "preprocessing_components",
            "feature_engineering_config",
            "model_path",
            "preprocessing_path",
            "config_path",
            "model_checksum",
            "preprocessing_checksum",
            "tags",
            "description",
            "author",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"

        # Verify specific values
        assert metadata["model_name"] == "metadata_test_model"
        assert metadata["description"] == "Test description"
        assert metadata["tags"] == ["test", "metadata"]
        assert metadata["author"] == "test_author"
        assert metadata["model_type"] == "LSTM"
        assert metadata["framework"] == "TensorFlow"


if __name__ == "__main__":
    pytest.main([__file__])
