#!/usr/bin/env python3
"""
Demonstration script for LSTM model architecture and configuration system.

This script shows how to use the LSTMModelBuilder and MetricsCalculator classes
to create, configure, and evaluate LSTM models for glucose prediction.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import yaml

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from diabetes_lstm_pipeline.model_architecture import (
    LSTMModelBuilder,
    MetricsCalculator,
    GlucoseAwareLoss,
    GlucoseMARD,
    TimeInRangeAccuracy,
)
from diabetes_lstm_pipeline.utils.config_manager import ConfigManager


def load_config():
    """Load configuration from default config file."""
    config_path = Path(__file__).parent.parent / "configs" / "default_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_sample_data(n_samples=1000, sequence_length=60, n_features=10):
    """
    Generate sample time-series data for demonstration.

    Args:
        n_samples: Number of samples to generate
        sequence_length: Length of each sequence
        n_features: Number of features per timestep

    Returns:
        Tuple of (X, y) where X is input sequences and y is target glucose values
    """
    print(f"Generating {n_samples} sample sequences...")

    # Generate synthetic glucose-like data
    np.random.seed(42)

    # Create sequences with some temporal patterns
    X = np.random.randn(n_samples, sequence_length, n_features)

    # Add some temporal correlation
    for i in range(1, sequence_length):
        X[:, i, :] = 0.7 * X[:, i - 1, :] + 0.3 * X[:, i, :]

    # Generate target glucose values (70-300 mg/dL range)
    # Make targets somewhat correlated with the last few timesteps
    last_values = X[:, -5:, 0].mean(axis=1)  # Use first feature
    y = 120 + 30 * last_values + 10 * np.random.randn(n_samples)
    y = np.clip(y, 40, 400)  # Clip to reasonable glucose range

    return X.astype(np.float32), y.astype(np.float32)


def demonstrate_model_building():
    """Demonstrate LSTM model building with different configurations."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING LSTM MODEL BUILDING")
    print("=" * 60)

    config = load_config()

    # Test different model configurations
    configurations = [
        {
            "name": "Simple LSTM",
            "config": {
                "model": {
                    "lstm_units": 64,
                    "dense_units": 32,
                    "dropout_rate": 0.2,
                    "learning_rate": 0.001,
                }
            },
        },
        {
            "name": "Multi-layer LSTM",
            "config": {
                "model": {
                    "lstm_units": [128, 64],
                    "dense_units": [64, 32],
                    "dropout_rate": 0.3,
                    "learning_rate": 0.001,
                }
            },
        },
        {
            "name": "Deep LSTM",
            "config": {
                "model": {
                    "lstm_units": [128, 64, 32],
                    "dense_units": [64, 32, 16],
                    "dropout_rate": 0.25,
                    "learning_rate": 0.0005,
                }
            },
        },
    ]

    input_shape = (60, 10)  # 60 timesteps, 10 features

    for config_info in configurations:
        print(f"\nBuilding {config_info['name']}...")

        # Update base config with specific configuration
        model_config = config.copy()
        model_config.update(config_info["config"])

        # Create model builder
        builder = LSTMModelBuilder(model_config)

        # Build and compile model
        model = builder.build_model(input_shape)
        compiled_model = builder.compile_model(model, "glucose_aware")

        # Get model summary
        summary = builder.get_model_summary(compiled_model)

        print(f"  - Total parameters: {summary['total_params']:,}")
        print(f"  - Trainable parameters: {summary['trainable_params']:,}")
        print(f"  - Number of layers: {len(summary['layers'])}")
        print(f"  - Input shape: {summary['input_shape']}")
        print(f"  - Output shape: {summary['output_shape']}")


def demonstrate_custom_metrics():
    """Demonstrate custom glucose-specific metrics."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING CUSTOM GLUCOSE METRICS")
    print("=" * 60)

    # Generate sample predictions
    np.random.seed(42)
    y_true = np.array([80, 120, 150, 200, 60, 180, 90, 160])
    y_pred = np.array([85, 115, 145, 190, 65, 175, 95, 155])

    print(f"True values:      {y_true}")
    print(f"Predicted values: {y_pred}")

    # Test MARD metric
    print("\nTesting MARD (Mean Absolute Relative Difference):")
    mard_metric = GlucoseMARD()
    mard_metric.update_state(
        tf.constant(y_true, dtype=tf.float32), tf.constant(y_pred, dtype=tf.float32)
    )
    mard_result = mard_metric.result().numpy()
    print(f"MARD: {mard_result:.2f}%")

    # Test Time-in-Range Accuracy
    print("\nTesting Time-in-Range Accuracy:")
    tir_metric = TimeInRangeAccuracy(target_range=(70, 180))
    tir_metric.update_state(
        tf.constant(y_true, dtype=tf.float32), tf.constant(y_pred, dtype=tf.float32)
    )
    tir_result = tir_metric.result().numpy()
    print(f"Time-in-Range Accuracy: {tir_result:.2f}")

    # Test Glucose-Aware Loss
    print("\nTesting Glucose-Aware Loss:")
    glucose_loss = GlucoseAwareLoss()
    loss_value = glucose_loss(
        tf.constant(y_true, dtype=tf.float32), tf.constant(y_pred, dtype=tf.float32)
    )
    print(f"Glucose-Aware Loss: {loss_value.numpy():.4f}")


def demonstrate_metrics_calculator():
    """Demonstrate comprehensive metrics calculation."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING METRICS CALCULATOR")
    print("=" * 60)

    config = load_config()
    calculator = MetricsCalculator(config)

    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    y_true = 120 + 30 * np.random.randn(n_samples)
    y_true = np.clip(y_true, 40, 400)

    # Add some prediction error
    y_pred = y_true + 10 * np.random.randn(n_samples)
    y_pred = np.clip(y_pred, 40, 400)

    print(f"Calculating metrics for {n_samples} predictions...")

    # Calculate comprehensive metrics
    metrics = calculator.calculate_comprehensive_metrics(y_true, y_pred)

    # Generate and display report
    report = calculator.generate_metrics_report(metrics)
    print("\n" + report)


def demonstrate_model_training_simulation():
    """Demonstrate model training with callbacks and metrics."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING MODEL TRAINING SIMULATION")
    print("=" * 60)

    config = load_config()
    builder = LSTMModelBuilder(config)

    # Generate sample data
    X, y = generate_sample_data(n_samples=500, sequence_length=60, n_features=8)

    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Build and compile model
    model = builder.build_model(X_train.shape[1:])
    compiled_model = builder.compile_model(model, "glucose_aware")

    # Get callbacks
    callbacks = builder.get_callbacks()

    print(f"\nConfigured {len(callbacks)} training callbacks:")
    for callback in callbacks:
        print(f"  - {callback.__class__.__name__}")

    # Simulate a few training epochs (for demonstration)
    print("\nRunning short training simulation...")
    history = compiled_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=3,  # Just a few epochs for demo
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    # Calculate metrics on validation set
    y_pred = compiled_model.predict(X_val, verbose=0).flatten()

    calculator = MetricsCalculator(config)
    metrics = calculator.calculate_comprehensive_metrics(y_val, y_pred, history)

    print("\nTraining Results:")
    print(f"Final training loss: {metrics.get('training_final_train_loss', 'N/A'):.4f}")
    print(f"Final validation loss: {metrics.get('training_final_val_loss', 'N/A'):.4f}")
    print(f"Validation MARD: {metrics.get('glucose_mard', 'N/A'):.2f}%")
    print(f"Overall Score: {metrics.get('overall_score', 'N/A'):.2f}/100")


def main():
    """Run all demonstrations."""
    print("LSTM Model Architecture and Configuration System Demo")
    print("=" * 60)

    # Set TensorFlow to use CPU only for demo
    tf.config.set_visible_devices([], "GPU")

    try:
        demonstrate_model_building()
        demonstrate_custom_metrics()
        demonstrate_metrics_calculator()
        demonstrate_model_training_simulation()

        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
