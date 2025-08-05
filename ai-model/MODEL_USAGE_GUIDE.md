# Diabetes LSTM Model Usage Guide

This guide explains how to use the generated diabetes LSTM models for glucose prediction.

## 📋 **Model Overview**

The diabetes LSTM model is designed to predict glucose levels based on historical time series data from continuous glucose monitors (CGM) and insulin pumps.

### **Model Specifications**

- **Architecture**: LSTM (Long Short-Term Memory) neural network
- **Input Shape**: `(batch_size, 12, 62)` - 12 time steps with 62 features each
- **Output Shape**: `(batch_size, 1)` - Single glucose prediction value
- **Target**: Glucose levels in mg/dL
- **Total Parameters**: ~154,873

## 🗂️ **Model Files Structure**

Each saved model is stored in a versioned directory structure:

```
models/versions/diabetes_lstm_YYYYMMDD_HHMMSS/YYYYMMDD_HHMMSS/
├── model.keras          # The trained Keras model
├── metadata.json        # Model metadata and performance metrics
├── config.json          # Training configuration and parameters
└── preprocessing.pkl    # Preprocessing components (scalers, encoders)
```

## 🚀 **How to Use the Models**

### **1. Basic Model Loading**

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path

class DiabetesModelPredictor:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.metadata = None

        # Load the model
        self.model = keras.models.load_model(self.model_path / "model.keras")

        # Load metadata
        with open(self.model_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)

        print(f"Model loaded: {self.metadata['model_name']}")
        print(f"Input shape: {self.metadata['input_shape']}")
        print(f"Output shape: {self.metadata['output_shape']}")

# Usage
predictor = DiabetesModelPredictor("models/versions/diabetes_lstm_20250804_173243/20250804_173243")
```

### **2. Making Predictions**

#### **Batch Predictions**

```python
def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
    """
    Make predictions for multiple sequences.

    Args:
        sequences: Input array with shape (n_samples, 12, 62)

    Returns:
        Predicted glucose values with shape (n_samples,)
    """
    predictions = self.model.predict(sequences, verbose=0)
    return predictions.flatten()

# Example usage
sequences = np.random.normal(0, 1, (10, 12, 62))  # 10 sequences
predictions = predictor.predict_batch(sequences)
print(f"Predictions: {predictions}")
```

#### **Single Prediction**

```python
def predict_single(self, sequence: np.ndarray) -> float:
    """
    Make a prediction for a single sequence.

    Args:
        sequence: Input array with shape (12, 62)

    Returns:
        Single predicted glucose value
    """
    # Reshape for single prediction
    sequence = sequence.reshape(1, 12, 62)
    prediction = self.model.predict(sequence, verbose=0)
    return float(prediction[0, 0])

# Example usage
sequence = np.random.normal(0, 1, (12, 62))
prediction = predictor.predict_single(sequence)
print(f"Predicted glucose: {prediction:.1f} mg/dL")
```

## 📊 **Input Data Requirements**

### **Expected Input Format**

The model expects input data with the following characteristics:

- **Shape**: `(batch_size, 12, 62)`
- **Time Steps**: 12 consecutive time points (typically 5-minute intervals)
- **Features**: 62 engineered features including:
  - CGM (Continuous Glucose Monitoring) values
  - Insulin delivery rates (Basal, Bolus, Correction)
  - Food intake and carbohydrate amounts
  - Time-based features (hour of day, day of week)
  - Statistical features (rolling means, standard deviations)
  - Lagged features (previous glucose values)

### **Data Preprocessing**

To use your own data, you need to:

1. **Match the preprocessing pipeline** used during training
2. **Engineer the same 62 features** from your raw data
3. **Create sequences of length 12** with proper time alignment
4. **Apply the same scaling/normalization** as used during training

## 🔧 **Real-World Usage Example**

```python
import pandas as pd
import numpy as np
from pathlib import Path

def load_latest_model():
    """Load the most recent trained model."""
    models_dir = Path("models/versions")
    model_dirs = list(models_dir.glob("diabetes_lstm_*"))

    if not model_dirs:
        raise FileNotFoundError("No models found. Run the pipeline first.")

    latest_model_dir = max(model_dirs, key=lambda x: x.name)
    version_dirs = list(latest_model_dir.glob("*"))
    latest_version_dir = max(version_dirs, key=lambda x: x.name)

    return str(latest_version_dir)

def prepare_real_data(data: pd.DataFrame) -> np.ndarray:
    """
    Prepare real data for prediction.

    Note: This is a simplified example. In practice, you need to:
    1. Apply the exact same preprocessing as training
    2. Engineer the same 62 features
    3. Use the same scaling/normalization
    """
    # This is a placeholder - you need to implement the actual preprocessing
    # that matches what was used during training

    # For demonstration, create synthetic data with correct shape
    n_samples = max(1, len(data) - 12 + 1)
    sequences = []

    for i in range(n_samples):
        # Create sequence with shape (12, 62)
        sequence = np.random.normal(0, 1, (12, 62))

        # Use real CGM data if available
        if 'CGM' in data.columns and i < len(data) - 12 + 1:
            cgm_data = data.iloc[i:i+12]['CGM'].values
            if len(cgm_data) == 12:
                sequence[:, 0] = cgm_data  # First feature is CGM

        sequences.append(sequence)

    return np.array(sequences)

# Complete usage example
def main():
    # Load the model
    model_path = load_latest_model()
    predictor = DiabetesModelPredictor(model_path)

    # Load your data
    # data = pd.read_csv("your_glucose_data.csv")

    # For demonstration, create sample data
    data = pd.DataFrame({
        'CGM': np.random.normal(150, 30, 100),
        'Basal': np.random.uniform(0.5, 2.0, 100),
        'Bolus': np.random.exponential(5, 100),
        'Food': np.random.exponential(30, 100),
        'Carbs': np.random.exponential(25, 100),
    })

    # Prepare data for prediction
    sequences = prepare_real_data(data)

    # Make predictions
    predictions = predictor.predict_batch(sequences)

    # Interpret results
    print(f"Made {len(predictions)} predictions")
    print(f"Prediction range: {predictions.min():.1f} - {predictions.max():.1f} mg/dL")

    # Clinical interpretation
    for i, pred in enumerate(predictions[:5]):
        status = "Normal" if 70 <= pred <= 180 else "High" if pred > 180 else "Low"
        print(f"Sample {i+1}: {pred:.1f} mg/dL ({status})")

if __name__ == "__main__":
    main()
```

## 📈 **Model Performance**

Based on the training results, the model achieves:

- **MARD**: ~100% (Mean Absolute Relative Difference)
- **MAE**: ~146 mg/dL (Mean Absolute Error)
- **RMSE**: ~154 mg/dL (Root Mean Square Error)
- **Time-in-Range Accuracy**: ~22%

### **Clinical Interpretation**

- **Normal Range**: 70-180 mg/dL
- **Hypoglycemia**: <70 mg/dL
- **Hyperglycemia**: >180 mg/dL

## ⚠️ **Important Notes**

1. **Data Preprocessing**: The model requires the exact same preprocessing as used during training. The 62 features are engineered from raw data through a complex pipeline.

2. **Model Limitations**:

   - Predictions are based on historical patterns
   - Accuracy depends on data quality and completeness
   - Not a substitute for medical advice

3. **Production Use**: For production deployment, consider:
   - Model versioning and rollback strategies
   - Input validation and error handling
   - Performance monitoring and drift detection
   - Regulatory compliance requirements

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **Shape Mismatch Error**: Ensure input has shape `(batch_size, 12, 62)`
2. **Custom Metrics Error**: Load model with `compile=False` if custom metrics aren't available
3. **Memory Issues**: Use smaller batch sizes for large datasets

### **Getting Help**

- Check the model metadata for configuration details
- Review the training logs for preprocessing information
- Use the provided example scripts as templates

## 📚 **Additional Resources**

- `simple_model_usage.py`: Complete working example
- `example_model_usage.py`: Advanced usage with custom metrics
- Model metadata files for detailed configuration information
- Training logs for preprocessing pipeline details
