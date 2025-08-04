"""
Demonstration of model persistence and versioning system.

This example shows how to:
1. Train a model using the existing pipeline
2. Save the model with preprocessing components
3. Load and use the saved model for inference
4. Compare different model versions
5. Manage model versions and metadata
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Import pipeline components
from diabetes_lstm_pipeline.model_architecture.model_builder import LSTMModelBuilder
from diabetes_lstm_pipeline.training.model_trainer import ModelTrainer
from diabetes_lstm_pipeline.model_persistence.model_persistence import ModelPersistence
from diabetes_lstm_pipeline.preprocessing.preprocessing import DataPreprocessor
from diabetes_lstm_pipeline.feature_engineering.feature_engineering import (
    FeatureEngineer,
)
from diabetes_lstm_pipeline.sequence_generation.sequence_generation import (
    SequenceGenerator,
)
from diabetes_lstm_pipeline.utils.config_manager import ConfigManager
from diabetes_lstm_pipeline.utils.logger import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample diabetes data for demonstration."""
    logger.info(f"Generating {n_samples} samples of synthetic diabetes data")

    # Create time series data
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    time_index = pd.date_range(start=start_time, periods=n_samples, freq="5min")

    # Generate synthetic glucose readings with realistic patterns
    np.random.seed(42)
    base_glucose = 120 + 30 * np.sin(
        np.arange(n_samples) * 2 * np.pi / 288
    )  # Daily pattern
    noise = np.random.normal(0, 10, n_samples)
    glucose = np.clip(base_glucose + noise, 40, 400)

    # Generate insulin and meal data
    basal_rate = np.random.uniform(0.5, 2.0, n_samples)
    bolus_insulin = np.random.exponential(2, n_samples) * (
        np.random.random(n_samples) < 0.1
    )
    carb_intake = np.random.exponential(30, n_samples) * (
        np.random.random(n_samples) < 0.08
    )

    # Create DataFrame
    df = pd.DataFrame(
        {
            "EventDateTime": time_index,
            "CGM": glucose,
            "Basal": basal_rate,
            "TotalBolusInsulinDelivered": bolus_insulin,
            "CorrectionDelivered": bolus_insulin * 0.3,
            "FoodDelivered": carb_intake > 0,
            "CarbSize": carb_intake,
            "DeviceMode": "Auto",
            "BolusType": "Normal",
        }
    )

    logger.info("Sample data generation completed")
    return df


def create_demo_config() -> dict:
    """Create configuration for the demonstration."""
    return {
        "model": {
            "sequence_length": 60,
            "lstm_units": [64, 32],
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "loss_function": "mse",
        },
        "training": {
            "batch_size": 32,
            "epochs": 10,  # Reduced for demo
            "validation_split": 0.2,
            "early_stopping_patience": 5,
        },
        "preprocessing": {
            "missing_values": {
                "strategies": {
                    "CGM": "interpolation",
                    "Basal": "forward_fill",
                    "TotalBolusInsulinDelivered": "zero_fill",
                    "CorrectionDelivered": "zero_fill",
                    "CarbSize": "zero_fill",
                }
            },
            "outliers": {"detection_methods": ["iqr"], "treatment_method": "clip"},
        },
        "feature_engineering": {
            "temporal_features": True,
            "insulin_features": True,
            "glucose_features": True,
            "lag_features": True,
        },
        "model_persistence": {
            "base_dir": "models_demo",
            "versioning_strategy": "timestamp",
            "max_versions": 5,
            "compress_preprocessing": True,
        },
    }


def train_and_save_model(config: dict, data: pd.DataFrame, model_name: str) -> dict:
    """Train a model and save it with persistence system."""
    logger.info(f"Training and saving model: {model_name}")

    # Initialize components
    preprocessor = DataPreprocessor(config)
    feature_engineer = FeatureEngineer(config)
    sequence_generator = SequenceGenerator(
        sequence_length=config["model"]["sequence_length"], prediction_horizon=1
    )
    model_trainer = ModelTrainer(config)
    model_persistence = ModelPersistence(config)

    # Preprocess data
    logger.info("Preprocessing data...")
    processed_data, preprocessing_stats = preprocessor.preprocess(data)

    # Feature engineering
    logger.info("Engineering features...")
    featured_data = feature_engineer.extract_all_features(processed_data)

    # Generate sequences
    logger.info("Generating sequences...")
    sequences, targets = sequence_generator.generate_sequences(featured_data)

    # Split data
    split_data = sequence_generator.split_sequences(sequences, targets)
    X_train, y_train = split_data["train"]
    X_val, y_val = split_data["validation"]
    X_test, y_test = split_data["test"]

    logger.info(f"Training data shape: {X_train.shape}, {y_train.shape}")

    # Train model
    logger.info("Training model...")
    trained_model = model_trainer.train(X_train, y_train, (X_val, y_val))

    # Evaluate model
    logger.info("Evaluating model...")
    test_results = model_trainer.evaluate_on_test_set(X_test, y_test)

    # Prepare preprocessing components for saving
    preprocessing_components = {
        "preprocessor": preprocessor,
        "feature_engineer": feature_engineer,
        "sequence_generator": sequence_generator,
        "feature_names": feature_engineer.get_feature_names(),
        "target_name": "glucose",
        "preprocessing_stats": preprocessing_stats,
        "feature_engineering_config": config["feature_engineering"],
        "sequence_config": {
            "sequence_length": config["model"]["sequence_length"],
            "prediction_horizon": 1,
        },
    }

    # Prepare performance metrics
    training_history = model_trainer.get_training_history()
    performance_metrics = {
        "training": {
            "loss": training_history["loss"][-1] if training_history else 0,
            "mae": training_history.get("mae", [0])[-1] if training_history else 0,
        },
        "validation": {
            "loss": training_history["val_loss"][-1] if training_history else 0,
            "mae": training_history.get("val_mae", [0])[-1] if training_history else 0,
        },
        "test": test_results["metrics"],
    }

    # Save model
    logger.info("Saving model with persistence system...")
    saved_paths = model_persistence.save_model(
        model=trained_model,
        preprocessing_components=preprocessing_components,
        training_metadata=model_trainer.training_metadata,
        performance_metrics=performance_metrics,
        model_name=model_name,
        description=f"Demo model trained on {len(data)} samples",
        tags=["demo", "lstm", "diabetes"],
        author="demo_user",
    )

    logger.info(f"Model saved successfully: {model_name}")
    logger.info(f"Saved paths: {saved_paths}")

    return {
        "model_persistence": model_persistence,
        "saved_paths": saved_paths,
        "test_results": test_results,
        "test_data": (X_test, y_test),
    }


def load_and_test_model(
    model_persistence: ModelPersistence, model_name: str, test_data: tuple
) -> dict:
    """Load a saved model and test it."""
    logger.info(f"Loading and testing model: {model_name}")

    # Load model
    loaded_components = model_persistence.load_model(model_name)
    loaded_model = loaded_components["model"]
    loaded_preprocessing = loaded_components["preprocessing"]
    metadata = loaded_components["metadata"]

    logger.info(f"Loaded model version: {loaded_components['version']}")
    logger.info(f"Model created at: {metadata['created_at']}")
    logger.info(f"Model description: {metadata['description']}")

    # Test predictions
    X_test, y_test = test_data
    predictions = loaded_model.predict(X_test, verbose=0)

    # Calculate simple metrics
    mae = np.mean(np.abs(predictions.flatten() - y_test.flatten()))
    rmse = np.sqrt(np.mean((predictions.flatten() - y_test.flatten()) ** 2))

    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")

    return {
        "loaded_model": loaded_model,
        "preprocessing": loaded_preprocessing,
        "metadata": metadata,
        "predictions": predictions,
        "test_mae": mae,
        "test_rmse": rmse,
    }


def demonstrate_model_management(model_persistence: ModelPersistence):
    """Demonstrate model management features."""
    logger.info("Demonstrating model management features...")

    # List all models
    models = model_persistence.list_models()
    logger.info(f"Available models: {len(models)}")

    for model in models:
        logger.info(f"  Model: {model['model_name']}")
        logger.info(f"    Latest version: {model['latest_version']}")
        logger.info(f"    Total versions: {len(model['versions'])}")

        # Get detailed info for latest version
        model_info = model_persistence.get_model_info(model["model_name"])
        logger.info(f"    Performance: {model_info['validation_metrics']}")
        logger.info(f"    Parameters: {model_info['total_parameters']:,}")

    # Compare models if we have multiple
    if len(models) >= 2:
        model1 = models[0]
        model2 = models[1]

        comparison = model_persistence.compare_models(
            (model1["model_name"], model1["latest_version"]),
            (model2["model_name"], model2["latest_version"]),
        )

        logger.info("Model comparison:")
        logger.info(f"  {comparison['model1']} vs {comparison['model2']}")

        perf_comp = comparison["performance_comparison"]
        for metric, values in perf_comp.items():
            if isinstance(values, dict) and "improvement_percent" in values:
                logger.info(
                    f"    {metric}: {values['improvement_percent']:.2f}% improvement"
                )


def main():
    """Main demonstration function."""
    logger.info("Starting Model Persistence Demonstration")

    # Create configuration
    config = create_demo_config()

    # Generate sample data
    data = generate_sample_data(n_samples=2000)

    # Train and save first model
    logger.info("=" * 60)
    logger.info("TRAINING FIRST MODEL")
    logger.info("=" * 60)

    result1 = train_and_save_model(config, data, "demo_model_v1")
    model_persistence = result1["model_persistence"]

    # Train and save second model with different configuration
    logger.info("=" * 60)
    logger.info("TRAINING SECOND MODEL")
    logger.info("=" * 60)

    # Modify config for second model
    config2 = config.copy()
    config2["model"]["lstm_units"] = [128, 64, 32]  # Larger model
    config2["model"]["dropout_rate"] = 0.3

    result2 = train_and_save_model(config2, data, "demo_model_v2")

    # Load and test models
    logger.info("=" * 60)
    logger.info("LOADING AND TESTING MODELS")
    logger.info("=" * 60)

    # Test first model
    test_result1 = load_and_test_model(
        model_persistence, "demo_model_v1", result1["test_data"]
    )

    # Test second model
    test_result2 = load_and_test_model(
        model_persistence, "demo_model_v2", result2["test_data"]
    )

    # Demonstrate model management
    logger.info("=" * 60)
    logger.info("MODEL MANAGEMENT FEATURES")
    logger.info("=" * 60)

    demonstrate_model_management(model_persistence)

    # Demonstrate inference pipeline
    logger.info("=" * 60)
    logger.info("INFERENCE PIPELINE DEMONSTRATION")
    logger.info("=" * 60)

    # Load best performing model
    models = model_persistence.list_models()
    best_model_name = models[0]["model_name"]  # Assume first is best for demo

    loaded_components = model_persistence.load_model(best_model_name)
    model = loaded_components["model"]
    preprocessing = loaded_components["preprocessing"]

    logger.info(f"Using model: {best_model_name}")

    # Generate new sample data for inference
    new_data = generate_sample_data(n_samples=100)

    # Process new data using saved preprocessing components
    preprocessor = preprocessing["preprocessor"]
    feature_engineer = preprocessing["feature_engineer"]
    sequence_generator = preprocessing["sequence_generator"]

    # Apply same preprocessing pipeline
    processed_new_data, _ = preprocessor.preprocess(new_data)
    featured_new_data = feature_engineer.extract_all_features(processed_new_data)
    new_sequences, new_targets = sequence_generator.generate_sequences(
        featured_new_data
    )

    # Make predictions
    if len(new_sequences) > 0:
        predictions = model.predict(new_sequences[:10], verbose=0)  # Predict first 10

        logger.info("Sample predictions:")
        for i, (pred, actual) in enumerate(zip(predictions[:5], new_targets[:5])):
            logger.info(f"  Sample {i+1}: Predicted={pred[0]:.1f}, Actual={actual:.1f}")

    logger.info("Model Persistence Demonstration completed successfully!")


if __name__ == "__main__":
    main()
