"""Unit tests for LSTM model architecture components."""

import unittest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from diabetes_lstm_pipeline.model_architecture import (
    LSTMModelBuilder,
    GlucoseAwareLoss,
    GlucoseMARD,
    TimeInRangeAccuracy,
    MetricsCalculator,
)


class TestLSTMModelBuilder(unittest.TestCase):
    """Test cases for LSTMModelBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "model": {
                "sequence_length": 60,
                "lstm_units": [128, 64],
                "dropout_rate": 0.2,
                "recurrent_dropout": 0.1,
                "dense_units": [64, 32],
                "activation": "relu",
                "output_activation": "linear",
                "learning_rate": 0.001,
                "l1_regularization": 0.0,
                "l2_regularization": 0.001,
            },
            "training": {
                "batch_size": 32,
                "epochs": 100,
                "early_stopping_patience": 15,
                "lr_patience": 7,
            },
        }
        self.builder = LSTMModelBuilder(self.config)

    def test_initialization(self):
        """Test model builder initialization."""
        self.assertEqual(self.builder.sequence_length, 60)
        self.assertEqual(self.builder.lstm_units, [128, 64])
        self.assertEqual(self.builder.dropout_rate, 0.2)
        self.assertEqual(self.builder.learning_rate, 0.001)

    def test_initialization_with_single_lstm_unit(self):
        """Test initialization with single LSTM unit value."""
        config = self.config.copy()
        config["model"]["lstm_units"] = 128
        builder = LSTMModelBuilder(config)
        self.assertEqual(builder.lstm_units, 128)

    def test_build_model_basic(self):
        """Test basic model building."""
        input_shape = (60, 10)  # 60 timesteps, 10 features
        model = self.builder.build_model(input_shape)

        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, 60, 10))
        self.assertEqual(model.output_shape, (None, 1))
        self.assertEqual(model.name, "glucose_lstm_predictor")

    def test_build_model_with_single_lstm_layer(self):
        """Test model building with single LSTM layer."""
        config = self.config.copy()
        config["model"]["lstm_units"] = 128
        config["model"]["dense_units"] = 64
        builder = LSTMModelBuilder(config)

        input_shape = (60, 5)
        model = builder.build_model(input_shape)

        self.assertIsInstance(model, tf.keras.Model)
        self.assertTrue(model.count_params() > 0)

    def test_build_model_different_configurations(self):
        """Test model building with different configurations."""
        configurations = [
            {"lstm_units": [64], "dense_units": [32]},
            {"lstm_units": [128, 64, 32], "dense_units": [64, 32, 16]},
            {"lstm_units": 256, "dense_units": 128},
        ]

        for config_update in configurations:
            config = self.config.copy()
            config["model"].update(config_update)
            builder = LSTMModelBuilder(config)

            model = builder.build_model((60, 8))
            self.assertIsInstance(model, tf.keras.Model)
            self.assertTrue(model.count_params() > 0)

    def test_compile_model_mse_loss(self):
        """Test model compilation with MSE loss."""
        model = self.builder.build_model((60, 5))
        compiled_model = self.builder.compile_model(model, "mse")

        self.assertTrue(hasattr(compiled_model, "optimizer"))
        self.assertTrue(hasattr(compiled_model, "loss"))
        # Check that metrics were added during compilation
        self.assertTrue(hasattr(compiled_model, "compiled_metrics"))

    def test_compile_model_different_losses(self):
        """Test model compilation with different loss functions."""
        model = self.builder.build_model((60, 5))

        loss_functions = ["mse", "mae", "huber", "glucose_aware"]
        for loss_func in loss_functions:
            compiled_model = self.builder.compile_model(model, loss_func)
            self.assertTrue(hasattr(compiled_model, "loss"))

    def test_compile_model_invalid_loss(self):
        """Test model compilation with invalid loss function."""
        model = self.builder.build_model((60, 5))

        with self.assertRaises(ValueError):
            self.builder.compile_model(model, "invalid_loss")

    def test_get_callbacks(self):
        """Test callback creation."""
        callbacks = self.builder.get_callbacks()

        self.assertTrue(len(callbacks) > 0)

        # Check for specific callback types
        callback_types = [type(cb).__name__ for cb in callbacks]
        self.assertIn("EarlyStopping", callback_types)
        self.assertIn("ReduceLROnPlateau", callback_types)
        self.assertIn("CSVLogger", callback_types)
        self.assertIn("TerminateOnNaN", callback_types)

    def test_get_callbacks_with_checkpoint(self):
        """Test callback creation with model checkpointing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model_checkpoint.h5"
            callbacks = self.builder.get_callbacks(checkpoint_path)

            callback_types = [type(cb).__name__ for cb in callbacks]
            self.assertIn("ModelCheckpoint", callback_types)

    def test_get_model_summary(self):
        """Test model summary generation."""
        model = self.builder.build_model((60, 8))
        compiled_model = self.builder.compile_model(model, "mse")

        summary = self.builder.get_model_summary(compiled_model)

        self.assertIsInstance(summary, dict)
        self.assertIn("model_name", summary)
        self.assertIn("total_params", summary)
        self.assertIn("trainable_params", summary)
        self.assertIn("layers", summary)
        self.assertIn("input_shape", summary)
        self.assertIn("output_shape", summary)

        self.assertEqual(summary["model_name"], "glucose_lstm_predictor")
        self.assertTrue(summary["total_params"] > 0)
        self.assertTrue(len(summary["layers"]) > 0)


class TestGlucoseAwareLoss(unittest.TestCase):
    """Test cases for GlucoseAwareLoss class."""

    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = GlucoseAwareLoss()

    def test_initialization(self):
        """Test loss function initialization."""
        self.assertEqual(self.loss_fn.hypoglycemia_threshold, 70)
        self.assertEqual(self.loss_fn.hyperglycemia_threshold, 180)
        self.assertEqual(self.loss_fn.name, "glucose_aware_loss")

    def test_custom_thresholds(self):
        """Test loss function with custom thresholds."""
        loss_fn = GlucoseAwareLoss(
            hypoglycemia_threshold=60, hyperglycemia_threshold=200
        )
        self.assertEqual(loss_fn.hypoglycemia_threshold, 60)
        self.assertEqual(loss_fn.hyperglycemia_threshold, 200)

    def test_loss_calculation_normal_range(self):
        """Test loss calculation for normal glucose range."""
        y_true = tf.constant([100.0, 120.0, 150.0])
        y_pred = tf.constant([105.0, 115.0, 145.0])

        loss = self.loss_fn(y_true, y_pred)
        self.assertIsInstance(loss, tf.Tensor)
        self.assertTrue(loss.numpy() > 0)

    def test_loss_calculation_hypoglycemia(self):
        """Test loss calculation for hypoglycemic values."""
        y_true = tf.constant([60.0, 65.0])  # Below threshold
        y_pred = tf.constant([70.0, 75.0])

        loss_hypo = self.loss_fn(y_true, y_pred)

        # Compare with normal range
        y_true_normal = tf.constant([120.0, 125.0])
        y_pred_normal = tf.constant([130.0, 135.0])
        loss_normal = self.loss_fn(y_true_normal, y_pred_normal)

        # Hypoglycemia should have higher penalty
        self.assertTrue(loss_hypo.numpy() > loss_normal.numpy())

    def test_loss_calculation_hyperglycemia(self):
        """Test loss calculation for hyperglycemic values."""
        y_true = tf.constant([200.0, 250.0])  # Above threshold
        y_pred = tf.constant([190.0, 240.0])

        loss_hyper = self.loss_fn(y_true, y_pred)

        # Compare with normal range
        y_true_normal = tf.constant([120.0, 125.0])
        y_pred_normal = tf.constant([130.0, 135.0])
        loss_normal = self.loss_fn(y_true_normal, y_pred_normal)

        # Hyperglycemia should have higher penalty
        self.assertTrue(loss_hyper.numpy() > loss_normal.numpy())


class TestGlucoseMARD(unittest.TestCase):
    """Test cases for GlucoseMARD metric."""

    def setUp(self):
        """Set up test fixtures."""
        self.metric = GlucoseMARD()

    def test_initialization(self):
        """Test metric initialization."""
        self.assertEqual(self.metric.name, "mard")

    def test_mard_calculation_perfect_prediction(self):
        """Test MARD calculation with perfect predictions."""
        y_true = tf.constant([100.0, 150.0, 200.0])
        y_pred = tf.constant([100.0, 150.0, 200.0])

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        self.assertAlmostEqual(result.numpy(), 0.0, places=5)

    def test_mard_calculation_with_error(self):
        """Test MARD calculation with prediction errors."""
        y_true = tf.constant([100.0, 150.0, 200.0])
        y_pred = tf.constant([110.0, 135.0, 180.0])  # 10%, 10%, 10% errors

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        self.assertAlmostEqual(result.numpy(), 10.0, places=1)

    def test_mard_calculation_with_zero_values(self):
        """Test MARD calculation handling zero values."""
        y_true = tf.constant([0.0, 100.0, 150.0])
        y_pred = tf.constant([10.0, 110.0, 135.0])

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Should ignore the zero true value
        expected_mard = (10.0 + 10.0) / 2  # Only valid predictions
        self.assertAlmostEqual(result.numpy(), expected_mard, places=1)

    def test_reset_state(self):
        """Test metric state reset."""
        y_true = tf.constant([100.0, 150.0])
        y_pred = tf.constant([110.0, 135.0])

        self.metric.update_state(y_true, y_pred)
        self.assertTrue(self.metric.result().numpy() > 0)

        self.metric.reset_state()
        self.assertEqual(self.metric.result().numpy(), 0.0)


class TestTimeInRangeAccuracy(unittest.TestCase):
    """Test cases for TimeInRangeAccuracy metric."""

    def setUp(self):
        """Set up test fixtures."""
        self.metric = TimeInRangeAccuracy()

    def test_initialization(self):
        """Test metric initialization."""
        self.assertEqual(self.metric.name, "tir_accuracy")
        self.assertEqual(self.metric.target_range, (70, 180))

    def test_custom_range(self):
        """Test metric with custom target range."""
        metric = TimeInRangeAccuracy(target_range=(80, 160))
        self.assertEqual(metric.target_range, (80, 160))

    def test_perfect_range_prediction(self):
        """Test perfect time-in-range prediction."""
        y_true = tf.constant([100.0, 150.0, 200.0, 60.0])  # 2 in range, 2 out
        y_pred = tf.constant([110.0, 140.0, 190.0, 65.0])  # 2 in range, 2 out

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        self.assertAlmostEqual(result.numpy(), 1.0, places=5)  # 100% accuracy

    def test_partial_range_prediction(self):
        """Test partial time-in-range prediction accuracy."""
        y_true = tf.constant([100.0, 150.0, 200.0, 60.0])  # 2 in range, 2 out
        y_pred = tf.constant([110.0, 200.0, 190.0, 65.0])  # 1 in range, 3 out

        self.metric.update_state(y_true, y_pred)
        result = self.metric.result()

        # Correct predictions: (in,in), (out,out), (out,out) = 3 out of 4
        self.assertAlmostEqual(result.numpy(), 0.75, places=5)

    def test_reset_state(self):
        """Test metric state reset."""
        y_true = tf.constant([100.0, 150.0])
        y_pred = tf.constant([110.0, 140.0])

        self.metric.update_state(y_true, y_pred)
        self.assertTrue(self.metric.result().numpy() > 0)

        self.metric.reset_state()
        self.assertEqual(self.metric.result().numpy(), 0.0)


class TestMetricsCalculator(unittest.TestCase):
    """Test cases for MetricsCalculator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "evaluation": {
                "target_glucose_range": (70, 180),
                "hypoglycemia_threshold": 70,
                "hyperglycemia_threshold": 180,
            }
        }
        self.calculator = MetricsCalculator(self.config)

        # Sample data
        self.y_true = np.array([100, 120, 150, 80, 200, 60, 180])
        self.y_pred = np.array([105, 115, 145, 85, 190, 65, 175])

    def test_initialization(self):
        """Test metrics calculator initialization."""
        self.assertEqual(self.calculator.target_range, (70, 180))
        self.assertEqual(self.calculator.hypoglycemia_threshold, 70)
        self.assertEqual(self.calculator.hyperglycemia_threshold, 180)

    def test_calculate_regression_metrics(self):
        """Test regression metrics calculation."""
        metrics = self.calculator.calculate_regression_metrics(self.y_true, self.y_pred)

        self.assertIn("mae", metrics)
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertIn("valid_predictions", metrics)

        self.assertTrue(metrics["mae"] > 0)
        self.assertTrue(metrics["mse"] > 0)
        self.assertTrue(metrics["rmse"] > 0)
        self.assertEqual(metrics["valid_predictions"], len(self.y_true))

    def test_calculate_regression_metrics_with_nan(self):
        """Test regression metrics with NaN values."""
        y_true_nan = np.array([100, np.nan, 150, 80])
        y_pred_nan = np.array([105, 115, np.inf, 85])

        metrics = self.calculator.calculate_regression_metrics(y_true_nan, y_pred_nan)

        self.assertEqual(metrics["valid_predictions"], 2)  # Only 2 valid pairs
        self.assertTrue(np.isfinite(metrics["mae"]))

    def test_calculate_glucose_specific_metrics(self):
        """Test glucose-specific metrics calculation."""
        metrics = self.calculator.calculate_glucose_specific_metrics(
            self.y_true, self.y_pred
        )

        self.assertIn("mard", metrics)
        self.assertIn("time_in_range_accuracy", metrics)
        self.assertIn("hypoglycemia_detection_rate", metrics)
        self.assertIn("hyperglycemia_detection_rate", metrics)
        self.assertIn("glucose_bias", metrics)
        self.assertIn("glucose_precision", metrics)

        self.assertTrue(metrics["mard"] > 0)
        self.assertTrue(0 <= metrics["time_in_range_accuracy"] <= 100)

    def test_calculate_glucose_metrics_no_hypoglycemia(self):
        """Test glucose metrics when no hypoglycemia present."""
        y_true_no_hypo = np.array([100, 120, 150, 180])
        y_pred_no_hypo = np.array([105, 115, 145, 175])

        metrics = self.calculator.calculate_glucose_specific_metrics(
            y_true_no_hypo, y_pred_no_hypo
        )

        self.assertTrue(np.isnan(metrics["hypoglycemia_detection_rate"]))

    def test_calculate_training_metrics(self):
        """Test training metrics calculation."""
        # Mock training history
        mock_history = Mock()
        mock_history.history = {
            "loss": [1.0, 0.8, 0.6, 0.5, 0.4],
            "val_loss": [1.2, 0.9, 0.7, 0.6, 0.5],
            "mae": [10.0, 8.0, 6.0, 5.0, 4.0],
            "val_mae": [12.0, 9.0, 7.0, 6.0, 5.0],
        }

        metrics = self.calculator.calculate_training_metrics(mock_history)

        self.assertIn("total_epochs", metrics)
        self.assertIn("final_train_loss", metrics)
        self.assertIn("final_val_loss", metrics)
        self.assertIn("best_val_loss", metrics)
        self.assertIn("best_epoch", metrics)

        self.assertEqual(metrics["total_epochs"], 5)
        self.assertEqual(metrics["final_train_loss"], 0.4)
        self.assertEqual(metrics["best_val_loss"], 0.5)

    def test_calculate_comprehensive_metrics(self):
        """Test comprehensive metrics calculation."""
        mock_history = Mock()
        mock_history.history = {"loss": [1.0, 0.5], "val_loss": [1.2, 0.6]}

        metrics = self.calculator.calculate_comprehensive_metrics(
            self.y_true, self.y_pred, mock_history
        )

        # Check that all metric categories are present
        regression_keys = [k for k in metrics.keys() if k.startswith("regression_")]
        glucose_keys = [k for k in metrics.keys() if k.startswith("glucose_")]
        training_keys = [k for k in metrics.keys() if k.startswith("training_")]

        self.assertTrue(len(regression_keys) > 0)
        self.assertTrue(len(glucose_keys) > 0)
        self.assertTrue(len(training_keys) > 0)
        self.assertIn("overall_score", metrics)

    def test_generate_metrics_report(self):
        """Test metrics report generation."""
        metrics = {
            "regression_mae": 5.2,
            "regression_mse": 35.4,
            "glucose_mard": 12.5,
            "glucose_time_in_range_accuracy": 85.3,
            "training_total_epochs": 50,
            "overall_score": 78.5,
        }

        report = self.calculator.generate_metrics_report(metrics)

        self.assertIsInstance(report, str)
        self.assertIn("REGRESSION METRICS", report)
        self.assertIn("GLUCOSE-SPECIFIC METRICS", report)
        self.assertIn("TRAINING METRICS", report)
        self.assertIn("OVERALL ASSESSMENT", report)
        self.assertIn("78.50/100", report)

    def test_generate_metrics_report_with_save(self):
        """Test metrics report generation with file saving."""
        metrics = {"regression_mae": 5.2, "overall_score": 78.5}

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_report.txt"
            report = self.calculator.generate_metrics_report(metrics, save_path)

            self.assertTrue(save_path.exists())
            with open(save_path, "r") as f:
                saved_content = f.read()
            self.assertEqual(report, saved_content)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_training_history(self, mock_savefig, mock_show):
        """Test training history plotting."""
        mock_history = Mock()
        mock_history.history = {
            "loss": [1.0, 0.8, 0.6],
            "val_loss": [1.2, 0.9, 0.7],
            "mae": [10.0, 8.0, 6.0],
            "val_mae": [12.0, 9.0, 7.0],
        }

        # Test without saving
        self.calculator.plot_training_history(mock_history)
        mock_show.assert_called_once()

        # Test with saving
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "training_plot.png"
            self.calculator.plot_training_history(mock_history, save_path)
            mock_savefig.assert_called()


if __name__ == "__main__":
    # Set up TensorFlow for testing
    tf.config.set_visible_devices([], "GPU")  # Use CPU only for tests

    unittest.main()
