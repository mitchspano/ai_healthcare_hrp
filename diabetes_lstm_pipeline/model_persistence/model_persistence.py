"""
Model persistence and versioning system for diabetes LSTM pipeline.

This module provides comprehensive model persistence capabilities including:
- Saving trained models in TensorFlow SavedModel format
- Serializing preprocessing components (scalers, encoders, feature pipelines)
- Model metadata tracking with training parameters and performance metrics
- Model loading utilities for inference and continued training
- Model versioning with timestamp and performance-based naming
"""

import tensorflow as tf
from tensorflow import keras
import pickle
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
import logging
import shutil
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata structure for saved models."""

    model_name: str
    version: str
    created_at: str
    model_type: str
    framework: str

    # Training information
    training_config: Dict[str, Any]
    model_config: Dict[str, Any]
    training_duration_seconds: float
    epochs_trained: int

    # Performance metrics
    training_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]]

    # Data information
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    feature_names: List[str]
    target_name: str

    # Model architecture
    total_parameters: int
    trainable_parameters: int
    model_summary: Dict[str, Any]

    # Preprocessing information
    preprocessing_components: List[str]
    feature_engineering_config: Dict[str, Any]

    # File paths
    model_path: str
    preprocessing_path: str
    config_path: str

    # Checksums for integrity
    model_checksum: str
    preprocessing_checksum: str

    # Additional metadata
    tags: List[str]
    description: str
    author: str


class ModelPersistence:
    """Handles saving and loading of trained models with preprocessing components."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model persistence system.

        Args:
            config: Configuration dictionary containing persistence settings
        """
        self.config = config
        self.persistence_config = config.get("model_persistence", {})

        # Base directories
        self.base_dir = Path(self.persistence_config.get("base_dir", "models"))
        self.models_dir = self.base_dir / "saved_models"
        self.preprocessing_dir = self.base_dir / "preprocessing"
        self.metadata_dir = self.base_dir / "metadata"
        self.versions_dir = self.base_dir / "versions"

        # Create directories
        for directory in [
            self.models_dir,
            self.preprocessing_dir,
            self.metadata_dir,
            self.versions_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Versioning settings
        self.versioning_strategy = self.persistence_config.get(
            "versioning_strategy", "timestamp"
        )
        self.performance_threshold = self.persistence_config.get(
            "performance_threshold", 0.01
        )
        self.max_versions = self.persistence_config.get("max_versions", 10)

        # Compression settings
        self.compress_preprocessing = self.persistence_config.get(
            "compress_preprocessing", True
        )
        self.compression_level = self.persistence_config.get("compression_level", 6)

        logger.info(
            f"Model persistence initialized with base directory: {self.base_dir}"
        )

    def save_model(
        self,
        model: keras.Model,
        preprocessing_components: Dict[str, Any],
        training_metadata: Dict[str, Any],
        performance_metrics: Dict[str, Dict[str, float]],
        model_name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        author: str = "system",
    ) -> Dict[str, str]:
        """
        Save a trained model with all associated components and metadata.

        Args:
            model: Trained Keras model
            preprocessing_components: Dictionary of preprocessing components to serialize
            training_metadata: Metadata from training process
            performance_metrics: Dictionary with training, validation, and test metrics
            model_name: Optional custom model name
            description: Model description
            tags: Optional list of tags for categorization
            author: Model author/creator

        Returns:
            Dictionary containing paths to saved components
        """
        logger.info("Starting model save process")

        # Generate model name and version
        if model_name is None:
            model_name = f"diabetes_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        version = self._generate_version(model_name, performance_metrics)

        # Create version-specific directories
        version_dir = self.versions_dir / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = self._save_model_artifacts(model, version_dir)

        # Save preprocessing components
        preprocessing_path = self._save_preprocessing_components(
            preprocessing_components, version_dir
        )

        # Save configuration
        config_path = self._save_configuration(version_dir)

        # Calculate checksums
        model_checksum = self._calculate_checksum(model_path)
        preprocessing_checksum = self._calculate_checksum(preprocessing_path)

        # Create metadata
        metadata = self._create_metadata(
            model_name=model_name,
            version=version,
            model=model,
            training_metadata=training_metadata,
            performance_metrics=performance_metrics,
            preprocessing_components=preprocessing_components,
            model_path=str(model_path),
            preprocessing_path=str(preprocessing_path),
            config_path=str(config_path),
            model_checksum=model_checksum,
            preprocessing_checksum=preprocessing_checksum,
            description=description,
            tags=tags or [],
            author=author,
        )

        # Save metadata
        metadata_path = self._save_metadata(metadata, version_dir)

        # Update version registry
        self._update_version_registry(model_name, version, metadata)

        # Clean up old versions if needed
        self._cleanup_old_versions(model_name)

        saved_paths = {
            "model": str(model_path),
            "preprocessing": str(preprocessing_path),
            "metadata": str(metadata_path),
            "config": str(config_path),
            "version_dir": str(version_dir),
        }

        logger.info(f"Model saved successfully: {model_name} v{version}")
        return saved_paths

    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        load_preprocessing: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a saved model with its preprocessing components.

        Args:
            model_name: Name of the model to load
            version: Specific version to load (latest if None)
            load_preprocessing: Whether to load preprocessing components

        Returns:
            Dictionary containing loaded model and components
        """
        logger.info(f"Loading model: {model_name}, version: {version}")

        # Get version to load
        if version is None:
            version = self._get_latest_version(model_name)

        version_dir = self.versions_dir / model_name / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version not found: {model_name} v{version}")

        # Load metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Verify checksums
        self._verify_checksums(metadata, version_dir)

        # Load model
        model_path = version_dir / "model"
        model = keras.models.load_model(model_path)

        loaded_components = {
            "model": model,
            "metadata": metadata,
            "version": version,
            "model_path": str(model_path),
        }

        # Load preprocessing components if requested
        if load_preprocessing:
            preprocessing_components = self._load_preprocessing_components(version_dir)
            loaded_components["preprocessing"] = preprocessing_components

        # Load configuration
        config_path = version_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                loaded_components["config"] = json.load(f)

        logger.info(f"Model loaded successfully: {model_name} v{version}")
        return loaded_components

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models with their versions and metadata.

        Returns:
            List of model information dictionaries
        """
        models = []

        for model_dir in self.versions_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            versions = []

            for version_dir in model_dir.iterdir():
                if not version_dir.is_dir():
                    continue

                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                        versions.append(
                            {
                                "version": version_dir.name,
                                "created_at": metadata.get("created_at"),
                                "performance": metadata.get("validation_metrics", {}),
                                "description": metadata.get("description", ""),
                                "tags": metadata.get("tags", []),
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load metadata for {model_name}/{version_dir.name}: {e}"
                        )

            if versions:
                # Sort versions by creation time
                versions.sort(key=lambda x: x["created_at"], reverse=True)
                models.append(
                    {
                        "model_name": model_name,
                        "versions": versions,
                        "latest_version": versions[0]["version"] if versions else None,
                    }
                )

        return models

    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """
        Delete a model version or entire model.

        Args:
            model_name: Name of the model to delete
            version: Specific version to delete (all versions if None)

        Returns:
            True if deletion was successful
        """
        try:
            if version is None:
                # Delete entire model
                model_dir = self.versions_dir / model_name
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    logger.info(f"Deleted entire model: {model_name}")
                else:
                    logger.warning(f"Model not found: {model_name}")
                    return False
            else:
                # Delete specific version
                version_dir = self.versions_dir / model_name / version
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                    logger.info(f"Deleted model version: {model_name} v{version}")

                    # Remove model directory if no versions left
                    model_dir = self.versions_dir / model_name
                    if model_dir.exists() and not any(model_dir.iterdir()):
                        model_dir.rmdir()
                else:
                    logger.warning(f"Model version not found: {model_name} v{version}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            return False

    def _save_model_artifacts(self, model: keras.Model, version_dir: Path) -> Path:
        """Save the Keras model in SavedModel format."""
        model_path = version_dir / "model.keras"
        model.save(model_path)
        logger.debug(f"Model saved to: {model_path}")
        return model_path

    def _save_preprocessing_components(
        self, preprocessing_components: Dict[str, Any], version_dir: Path
    ) -> Path:
        """Save preprocessing components using joblib."""
        preprocessing_path = version_dir / "preprocessing.pkl"

        # Use joblib for better scikit-learn compatibility
        if self.compress_preprocessing:
            joblib.dump(
                preprocessing_components,
                preprocessing_path,
                compress=self.compression_level,
            )
        else:
            joblib.dump(preprocessing_components, preprocessing_path)

        logger.debug(f"Preprocessing components saved to: {preprocessing_path}")
        return preprocessing_path

    def _save_configuration(self, version_dir: Path) -> Path:
        """Save the current configuration."""
        config_path = version_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        logger.debug(f"Configuration saved to: {config_path}")
        return config_path

    def _save_metadata(self, metadata: ModelMetadata, version_dir: Path) -> Path:
        """Save model metadata."""
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        logger.debug(f"Metadata saved to: {metadata_path}")
        return metadata_path

    def _load_preprocessing_components(self, version_dir: Path) -> Dict[str, Any]:
        """Load preprocessing components."""
        preprocessing_path = version_dir / "preprocessing.pkl"
        if not preprocessing_path.exists():
            logger.warning("No preprocessing components found")
            return {}

        preprocessing_components = joblib.load(preprocessing_path)
        logger.debug("Preprocessing components loaded successfully")
        return preprocessing_components

    def _create_metadata(
        self,
        model_name: str,
        version: str,
        model: keras.Model,
        training_metadata: Dict[str, Any],
        performance_metrics: Dict[str, Dict[str, float]],
        preprocessing_components: Dict[str, Any],
        model_path: str,
        preprocessing_path: str,
        config_path: str,
        model_checksum: str,
        preprocessing_checksum: str,
        description: str,
        tags: List[str],
        author: str,
    ) -> ModelMetadata:
        """Create comprehensive model metadata."""

        # Extract model information
        model_summary = self._get_model_summary(model)

        return ModelMetadata(
            model_name=model_name,
            version=version,
            created_at=datetime.now().isoformat(),
            model_type="LSTM",
            framework="TensorFlow",
            training_config=training_metadata.get("config", {}).get("training", {}),
            model_config=training_metadata.get("config", {}).get("model", {}),
            training_duration_seconds=training_metadata.get(
                "training_duration_seconds", 0.0
            ),
            epochs_trained=training_metadata.get("epochs_completed", 0),
            training_metrics=performance_metrics.get("training", {}),
            validation_metrics=performance_metrics.get("validation", {}),
            test_metrics=performance_metrics.get("test"),
            input_shape=(
                tuple(model.input_shape[1:])
                if hasattr(model, "input_shape") and model.input_shape
                else ()
            ),
            output_shape=(
                tuple(model.output_shape[1:])
                if hasattr(model, "output_shape") and model.output_shape
                else ()
            ),
            feature_names=preprocessing_components.get("feature_names", []),
            target_name=preprocessing_components.get("target_name", "glucose"),
            total_parameters=model.count_params(),
            trainable_parameters=sum(
                [tf.keras.backend.count_params(w) for w in model.trainable_weights]
            ),
            model_summary=model_summary,
            preprocessing_components=list(preprocessing_components.keys()),
            feature_engineering_config=preprocessing_components.get(
                "feature_engineering_config", {}
            ),
            model_path=model_path,
            preprocessing_path=preprocessing_path,
            config_path=config_path,
            model_checksum=model_checksum,
            preprocessing_checksum=preprocessing_checksum,
            tags=tags,
            description=description,
            author=author,
        )

    def _get_model_summary(self, model: keras.Model) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        summary = {"name": model.name, "layers": []}

        for layer in model.layers:
            layer_info = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": (
                    layer.output_shape if hasattr(layer, "output_shape") else None
                ),
                "params": layer.count_params(),
            }

            # Add layer-specific information
            if hasattr(layer, "units"):
                layer_info["units"] = layer.units
            if hasattr(layer, "activation"):
                layer_info["activation"] = layer.activation.__name__
            if hasattr(layer, "dropout"):
                layer_info["dropout"] = layer.dropout

            summary["layers"].append(layer_info)

        return summary

    def _generate_version(
        self, model_name: str, performance_metrics: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate version string based on versioning strategy."""

        if self.versioning_strategy == "timestamp":
            return datetime.now().strftime("%Y%m%d_%H%M%S")

        elif self.versioning_strategy == "performance":
            # Use validation loss as primary metric
            val_loss = performance_metrics.get("validation", {}).get(
                "loss", float("inf")
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{val_loss:.4f}_{timestamp}"

        elif self.versioning_strategy == "semantic":
            # Simple semantic versioning based on existing versions
            existing_versions = self._get_existing_versions(model_name)
            if not existing_versions:
                return "1.0.0"

            # Parse latest version and increment
            latest = max(existing_versions, key=lambda x: tuple(map(int, x.split("."))))
            major, minor, patch = map(int, latest.split("."))

            # Increment based on performance improvement
            val_loss = performance_metrics.get("validation", {}).get(
                "loss", float("inf")
            )
            if self._is_significant_improvement(model_name, val_loss):
                minor += 1
                patch = 0
            else:
                patch += 1

            return f"{major}.{minor}.{patch}"

        else:
            # Default to timestamp
            return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_existing_versions(self, model_name: str) -> List[str]:
        """Get list of existing versions for a model."""
        model_dir = self.versions_dir / model_name
        if not model_dir.exists():
            return []

        versions = []
        for version_dir in model_dir.iterdir():
            if version_dir.is_dir():
                versions.append(version_dir.name)

        return versions

    def _get_latest_version(self, model_name: str) -> str:
        """Get the latest version of a model."""
        versions = self._get_existing_versions(model_name)
        if not versions:
            raise ValueError(f"No versions found for model: {model_name}")

        if self.versioning_strategy == "semantic":
            return max(versions, key=lambda x: tuple(map(int, x.split("."))))
        else:
            return max(versions)

    def _is_significant_improvement(self, model_name: str, val_loss: float) -> bool:
        """Check if performance improvement is significant."""
        try:
            latest_version = self._get_latest_version(model_name)
            metadata_path = (
                self.versions_dir / model_name / latest_version / "metadata.json"
            )

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            previous_loss = metadata.get("validation_metrics", {}).get(
                "loss", float("inf")
            )
            improvement = (previous_loss - val_loss) / previous_loss

            return improvement > self.performance_threshold

        except Exception:
            return True  # Assume significant if can't compare

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file or directory."""
        if file_path.is_file():
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        elif file_path.is_dir():
            # For directories, calculate checksum of all files
            checksums = []
            for file in sorted(file_path.rglob("*")):
                if file.is_file():
                    with open(file, "rb") as f:
                        checksums.append(hashlib.sha256(f.read()).hexdigest())

            combined = "".join(checksums)
            return hashlib.sha256(combined.encode()).hexdigest()
        else:
            return ""

    def _verify_checksums(self, metadata: Dict[str, Any], version_dir: Path) -> None:
        """Verify file integrity using checksums."""
        model_path = version_dir / "model"
        preprocessing_path = version_dir / "preprocessing.pkl"

        # Verify model checksum
        if model_path.exists():
            current_checksum = self._calculate_checksum(model_path)
            expected_checksum = metadata.get("model_checksum", "")

            if current_checksum != expected_checksum:
                logger.warning("Model checksum mismatch - file may be corrupted")

        # Verify preprocessing checksum
        if preprocessing_path.exists():
            current_checksum = self._calculate_checksum(preprocessing_path)
            expected_checksum = metadata.get("preprocessing_checksum", "")

            if current_checksum != expected_checksum:
                logger.warning(
                    "Preprocessing checksum mismatch - file may be corrupted"
                )

    def _update_version_registry(
        self, model_name: str, version: str, metadata: ModelMetadata
    ) -> None:
        """Update the version registry with new model information."""
        registry_path = self.base_dir / "version_registry.json"

        # Load existing registry
        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)
        else:
            registry = {}

        # Update registry
        if model_name not in registry:
            registry[model_name] = {}

        registry[model_name][version] = {
            "created_at": metadata.created_at,
            "performance": metadata.validation_metrics,
            "description": metadata.description,
            "tags": metadata.tags,
        }

        # Save updated registry
        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, default=str)

    def _cleanup_old_versions(self, model_name: str) -> None:
        """Clean up old versions if max_versions limit is exceeded."""
        versions = self._get_existing_versions(model_name)

        if len(versions) <= self.max_versions:
            return

        # Sort versions by creation time and keep only the latest ones
        version_info = []
        for version in versions:
            metadata_path = self.versions_dir / model_name / version / "metadata.json"
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                version_info.append((version, metadata.get("created_at", "")))
            except Exception:
                version_info.append((version, ""))

        # Sort by creation time (newest first)
        version_info.sort(key=lambda x: x[1], reverse=True)

        # Remove old versions
        versions_to_remove = version_info[self.max_versions :]
        for version, _ in versions_to_remove:
            version_dir = self.versions_dir / model_name / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
                logger.info(f"Removed old version: {model_name} v{version}")

    def get_model_info(
        self, model_name: str, version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific model version.

        Args:
            model_name: Name of the model
            version: Specific version (latest if None)

        Returns:
            Dictionary containing model information
        """
        if version is None:
            version = self._get_latest_version(model_name)

        metadata_path = self.versions_dir / model_name / version / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {model_name} v{version}"
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return metadata

    def compare_models(
        self, model1: Tuple[str, str], model2: Tuple[str, str]
    ) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            model1: Tuple of (model_name, version)
            model2: Tuple of (model_name, version)

        Returns:
            Dictionary containing comparison results
        """
        info1 = self.get_model_info(model1[0], model1[1])
        info2 = self.get_model_info(model2[0], model2[1])

        comparison = {
            "model1": f"{model1[0]} v{model1[1]}",
            "model2": f"{model2[0]} v{model2[1]}",
            "performance_comparison": {},
            "architecture_comparison": {},
            "training_comparison": {},
        }

        # Compare performance metrics
        metrics1 = info1.get("validation_metrics", {})
        metrics2 = info2.get("validation_metrics", {})

        for metric in set(metrics1.keys()) | set(metrics2.keys()):
            val1 = metrics1.get(metric, "N/A")
            val2 = metrics2.get(metric, "N/A")

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                improvement = ((val2 - val1) / val1 * 100) if val1 != 0 else 0
                comparison["performance_comparison"][metric] = {
                    "model1": val1,
                    "model2": val2,
                    "improvement_percent": improvement,
                }
            else:
                comparison["performance_comparison"][metric] = {
                    "model1": val1,
                    "model2": val2,
                }

        # Compare architecture
        comparison["architecture_comparison"] = {
            "total_parameters": {
                "model1": info1.get("total_parameters", 0),
                "model2": info2.get("total_parameters", 0),
            },
            "trainable_parameters": {
                "model1": info1.get("trainable_parameters", 0),
                "model2": info2.get("trainable_parameters", 0),
            },
        }

        # Compare training
        comparison["training_comparison"] = {
            "epochs_trained": {
                "model1": info1.get("epochs_trained", 0),
                "model2": info2.get("epochs_trained", 0),
            },
            "training_duration": {
                "model1": info1.get("training_duration_seconds", 0),
                "model2": info2.get("training_duration_seconds", 0),
            },
        }

        return comparison
