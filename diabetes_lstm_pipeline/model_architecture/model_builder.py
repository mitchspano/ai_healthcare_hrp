"""LSTM model builder for glucose prediction."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LSTMModelBuilder:
    """Builder class for creating configurable LSTM architectures for glucose prediction."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LSTM model builder.

        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})

        # Model architecture parameters
        self.sequence_length = self.model_config.get("sequence_length", 60)
        self.lstm_units = self.model_config.get("lstm_units", [128, 64])
        self.dropout_rate = self.model_config.get("dropout_rate", 0.2)
        self.recurrent_dropout = self.model_config.get("recurrent_dropout", 0.1)
        self.dense_units = self.model_config.get("dense_units", [64, 32])
        self.activation = self.model_config.get("activation", "relu")
        self.output_activation = self.model_config.get("output_activation", "linear")

        # Training parameters
        self.learning_rate = self.model_config.get("learning_rate", 0.001)
        self.batch_size = self.training_config.get("batch_size", 32)
        self.epochs = self.training_config.get("epochs", 100)

        # Regularization
        self.l1_reg = self.model_config.get("l1_regularization", 0.0)
        self.l2_reg = self.model_config.get("l2_regularization", 0.001)

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build the LSTM model with the specified architecture.

        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)

        Returns:
            Compiled Keras model
        """
        logger.info(f"Building LSTM model with input shape: {input_shape}")

        # Ensure lstm_units is a list
        if isinstance(self.lstm_units, int):
            lstm_units = [self.lstm_units]
        else:
            lstm_units = self.lstm_units

        # Ensure dense_units is a list
        if isinstance(self.dense_units, int):
            dense_units = [self.dense_units]
        else:
            dense_units = self.dense_units

        # Input layer
        inputs = keras.Input(shape=input_shape, name="glucose_sequence_input")
        x = inputs

        # Add batch normalization to input
        x = layers.BatchNormalization(name="input_batch_norm")(x)

        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = (
                i < len(lstm_units) - 1
            )  # Return sequences for all but last LSTM layer

            x = layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=keras.regularizers.L1L2(
                    l1=self.l1_reg, l2=self.l2_reg
                ),
                recurrent_regularizer=keras.regularizers.L1L2(
                    l1=self.l1_reg, l2=self.l2_reg
                ),
                name=f"lstm_layer_{i+1}",
            )(x)

            # Add batch normalization after each LSTM layer
            x = layers.BatchNormalization(name=f"lstm_batch_norm_{i+1}")(x)

        # Dense layers
        for i, units in enumerate(dense_units):
            x = layers.Dense(
                units=units,
                activation=self.activation,
                kernel_regularizer=keras.regularizers.L1L2(
                    l1=self.l1_reg, l2=self.l2_reg
                ),
                name=f"dense_layer_{i+1}",
            )(x)

            x = layers.Dropout(self.dropout_rate, name=f"dense_dropout_{i+1}")(x)
            x = layers.BatchNormalization(name=f"dense_batch_norm_{i+1}")(x)

        # Output layer for glucose prediction
        outputs = layers.Dense(
            units=1, activation=self.output_activation, name="glucose_prediction"
        )(x)

        # Create model
        model = keras.Model(
            inputs=inputs, outputs=outputs, name="glucose_lstm_predictor"
        )

        logger.info(f"Model built successfully with {model.count_params()} parameters")
        return model

    def compile_model(
        self, model: keras.Model, loss_function: str = "mse"
    ) -> keras.Model:
        """
        Compile the model with optimizer, loss function, and metrics.

        Args:
            model: Keras model to compile
            loss_function: Loss function to use ('mse', 'mae', 'huber', 'glucose_aware')

        Returns:
            Compiled Keras model
        """
        logger.info(f"Compiling model with loss function: {loss_function}")

        # Configure optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7
        )

        # Select loss function
        if loss_function == "mse":
            loss = keras.losses.MeanSquaredError()
        elif loss_function == "mae":
            loss = keras.losses.MeanAbsoluteError()
        elif loss_function == "huber":
            loss = keras.losses.Huber(delta=1.0)
        elif loss_function == "glucose_aware":
            loss = GlucoseAwareLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

        # Define metrics
        metrics = [
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.MeanSquaredError(name="mse"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
            GlucoseMARD(name="mard"),
            TimeInRangeAccuracy(name="tir_accuracy"),
        ]

        # Compile model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        logger.info("Model compiled successfully")
        return model

    def get_callbacks(
        self, model_save_path: Optional[Path] = None
    ) -> List[keras.callbacks.Callback]:
        """
        Create training callbacks for early stopping, learning rate scheduling, and checkpointing.

        Args:
            model_save_path: Path to save model checkpoints

        Returns:
            List of Keras callbacks
        """
        callbacks_list = []

        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.training_config.get("early_stopping_patience", 15),
            restore_best_weights=True,
            verbose=1,
            mode="min",
        )
        callbacks_list.append(early_stopping)

        # Learning rate reduction
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=self.training_config.get("lr_patience", 7),
            min_lr=1e-7,
            verbose=1,
            mode="min",
        )
        callbacks_list.append(lr_scheduler)

        # Model checkpointing
        if model_save_path:
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = callbacks.ModelCheckpoint(
                filepath=str(model_save_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                mode="min",
                verbose=1,
            )
            callbacks_list.append(checkpoint)

        # CSV logger for training history
        csv_logger = callbacks.CSVLogger(
            filename="logs/training_history.csv", append=True
        )
        callbacks_list.append(csv_logger)

        # Terminate on NaN
        terminate_on_nan = callbacks.TerminateOnNaN()
        callbacks_list.append(terminate_on_nan)

        return callbacks_list

    def get_model_summary(self, model: keras.Model) -> Dict[str, Any]:
        """
        Generate comprehensive model summary and parameter information.

        Args:
            model: Keras model to summarize

        Returns:
            Dictionary containing model summary information
        """
        summary_info = {
            "model_name": model.name,
            "total_params": model.count_params(),
            "trainable_params": sum(
                [tf.keras.backend.count_params(w) for w in model.trainable_weights]
            ),
            "non_trainable_params": sum(
                [tf.keras.backend.count_params(w) for w in model.non_trainable_weights]
            ),
            "layers": [],
            "input_shape": model.input_shape,
            "output_shape": getattr(model, "output_shape", None),
            "optimizer": (
                model.optimizer.__class__.__name__
                if hasattr(model, "optimizer")
                else None
            ),
            "loss_function": (
                model.loss.__class__.__name__ if hasattr(model, "loss") else None
            ),
        }

        # Layer information
        for layer in model.layers:
            layer_info = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": getattr(layer, "output_shape", None),
                "params": layer.count_params(),
            }

            # Add layer-specific information
            if hasattr(layer, "units"):
                layer_info["units"] = layer.units
            if hasattr(layer, "activation"):
                layer_info["activation"] = layer.activation.__name__
            if hasattr(layer, "dropout"):
                layer_info["dropout"] = layer.dropout
            if hasattr(layer, "recurrent_dropout"):
                layer_info["recurrent_dropout"] = layer.recurrent_dropout

            summary_info["layers"].append(layer_info)

        return summary_info


class GlucoseAwareLoss(keras.losses.Loss):
    """Custom loss function that penalizes clinically dangerous glucose prediction errors more heavily."""

    def __init__(
        self,
        hypoglycemia_threshold=70,
        hyperglycemia_threshold=180,
        name="glucose_aware_loss",
    ):
        """
        Initialize glucose-aware loss function.

        Args:
            hypoglycemia_threshold: Glucose level below which predictions are heavily penalized
            hyperglycemia_threshold: Glucose level above which predictions are heavily penalized
            name: Name of the loss function
        """
        super().__init__(name=name)
        self.hypoglycemia_threshold = hypoglycemia_threshold
        self.hyperglycemia_threshold = hyperglycemia_threshold

    def call(self, y_true, y_pred):
        """
        Compute glucose-aware loss.

        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values

        Returns:
            Loss value
        """
        # Base MSE loss
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)

        # Additional penalty for dangerous ranges
        hypoglycemia_penalty = tf.where(
            y_true < self.hypoglycemia_threshold,
            tf.square(y_true - y_pred) * 2.0,  # Double penalty for hypoglycemia
            0.0,
        )

        hyperglycemia_penalty = tf.where(
            y_true > self.hyperglycemia_threshold,
            tf.square(y_true - y_pred) * 1.5,  # 1.5x penalty for hyperglycemia
            0.0,
        )

        # Penalty for predicting normal when actual is dangerous
        false_normal_penalty = tf.where(
            tf.logical_and(
                tf.logical_or(
                    y_true < self.hypoglycemia_threshold,
                    y_true > self.hyperglycemia_threshold,
                ),
                tf.logical_and(
                    y_pred >= self.hypoglycemia_threshold,
                    y_pred <= self.hyperglycemia_threshold,
                ),
            ),
            tf.square(y_true - y_pred)
            * 3.0,  # Heavy penalty for missing dangerous values
            0.0,
        )

        total_loss = (
            mse_loss
            + hypoglycemia_penalty
            + hyperglycemia_penalty
            + false_normal_penalty
        )
        return tf.reduce_mean(total_loss)


class GlucoseMARD(keras.metrics.Metric):
    """Mean Absolute Relative Difference (MARD) metric for glucose predictions."""

    def __init__(self, name="mard", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total_ard = self.add_weight(name="total_ard", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update MARD metric state."""
        # Calculate absolute relative difference
        ard = tf.abs((y_true - y_pred) / y_true) * 100.0

        # Filter out invalid values (division by zero, etc.)
        valid_mask = tf.logical_and(tf.math.is_finite(ard), y_true > 0)
        valid_ard = tf.boolean_mask(ard, valid_mask)

        if sample_weight is not None:
            valid_sample_weight = tf.boolean_mask(sample_weight, valid_mask)
            valid_ard = valid_ard * valid_sample_weight
            self.count.assign_add(tf.reduce_sum(valid_sample_weight))
        else:
            self.count.assign_add(tf.cast(tf.shape(valid_ard)[0], tf.float32))

        self.total_ard.assign_add(tf.reduce_sum(valid_ard))

    def result(self):
        """Return MARD result."""
        return tf.math.divide_no_nan(self.total_ard, self.count)

    def reset_state(self):
        """Reset metric state."""
        self.total_ard.assign(0.0)
        self.count.assign(0.0)


class TimeInRangeAccuracy(keras.metrics.Metric):
    """Time-in-range prediction accuracy metric."""

    def __init__(self, target_range=(70, 180), name="tir_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.target_range = target_range
        self.correct_predictions = self.add_weight(
            name="correct_predictions", initializer="zeros"
        )
        self.total_predictions = self.add_weight(
            name="total_predictions", initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update time-in-range accuracy state."""
        # Determine if true values are in range
        true_in_range = tf.logical_and(
            y_true >= self.target_range[0], y_true <= self.target_range[1]
        )

        # Determine if predicted values are in range
        pred_in_range = tf.logical_and(
            y_pred >= self.target_range[0], y_pred <= self.target_range[1]
        )

        # Check if predictions match true range status
        correct = tf.equal(true_in_range, pred_in_range)

        if sample_weight is not None:
            correct = tf.cast(correct, tf.float32) * sample_weight
            self.total_predictions.assign_add(tf.reduce_sum(sample_weight))
        else:
            correct = tf.cast(correct, tf.float32)
            self.total_predictions.assign_add(tf.cast(tf.shape(correct)[0], tf.float32))

        self.correct_predictions.assign_add(tf.reduce_sum(correct))

    def result(self):
        """Return time-in-range accuracy result."""
        return tf.math.divide_no_nan(self.correct_predictions, self.total_predictions)

    def reset_state(self):
        """Reset metric state."""
        self.correct_predictions.assign(0.0)
        self.total_predictions.assign(0.0)
