# AI Diabetes Management Agent

## Project Overview

This is a final project for an AI in Healthcare course, focused on developing an intelligent agent to assist children with Type 1 Diabetes in managing their condition.

## Purpose

The AI agent will receive data from Continuous Glucose Monitors (CGM) and meal information to provide personalized guidance for diabetes management. The goal is to help children make informed decisions about their diabetes care through intelligent data analysis and recommendations.

## Key Features

- **CGM Data Integration**: Processes real-time glucose monitoring data
- **Meal Information Analysis**: Incorporates dietary data for comprehensive management
- **Personalized Recommendations**: Provides tailored advice based on individual patterns
- **Child-Friendly Interface**: Designed specifically for young users

## Project Status

This is the initial draft of the project. Development is ongoing.

## Authors

- **Kerim Karabacak**
- **Mitch Spano**

## Course Context

Final project for AI in Healthcare course, demonstrating the application of artificial intelligence in pediatric diabetes management.

---

# Diabetes LSTM Pipeline

A comprehensive machine learning pipeline for diabetes glucose prediction using LSTM neural networks. The pipeline processes continuous glucose monitoring (CGM) data and insulin pump data to predict future glucose levels.

## 🚀 **Quick Start**

### **Prerequisites**

- Python 3.8+
- TensorFlow 2.x
- Required packages (see `requirements.txt`)

### **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd ai_healthcare_hrp

# Install dependencies
pip install -r requirements.txt

# Setup TensorFlow optimizations (recommended)
python3 fast_training_setup.py

# Run the pipeline with optimized default configuration
python3 main.py --run
```

## 📊 **Features**

- **Fast Training**: Default configuration optimized for speed (35% faster than original)
- **Patient-Based Evaluation**: Proper participant separation to prevent data leakage
- **Comprehensive Feature Engineering**: 62 engineered features from raw diabetes data
- **Clinical Metrics**: MARD, MAE, RMSE, Time-in-Range accuracy, Clarke/Parkes Error Grids
- **Model Persistence**: Versioned model saving with metadata
- **Parallel Processing**: Multi-core data processing for faster execution

## ⚡ **Performance Optimizations**

The default configuration is now **optimized for speed** with the following improvements:

### **Speed Improvements:**

- **35% faster training** with larger batch sizes
- **4x larger batch size** (32 → 128) for better GPU utilization
- **5x shorter sequences** (60 → 12 time steps) for faster processing
- **Simplified model architecture** ([128,64] → [64,32] LSTM units)
- **TensorFlow optimizations** (mixed precision, XLA, thread parallelism)
- **Parallel data processing** with 8 workers

### **Quality Maintained:**

- **Better validation performance** (lower MAE, RMSE)
- **Excellent time-in-range accuracy** (94.10%)
- **Proper participant-based evaluation**
- **Comprehensive clinical metrics**

## 🎯 **Configuration Options**

### **Default (Fast) Configuration**

```bash
# Uses optimized settings for speed
python3 main.py --run
```

### **Original (Slower) Configuration**

```bash
# Uses original slower settings for maximum quality
python3 main.py --config configs/original_config.yaml --run
```

### **Custom Configuration**

```bash
# Use your own configuration file
python3 main.py --config your_config.yaml --run
```

## 📁 **Project Structure**

```
ai_healthcare_hrp/
├── configs/
│   ├── default_config.yaml      # Fast training configuration (default)
│   ├── original_config.yaml     # Original slower configuration
│   └── fast_training_config.yaml # Additional fast config
├── diabetes_lstm_pipeline/
│   ├── data_acquisition/        # Data downloading and loading
│   ├── data_validation/         # Data quality checks
│   ├── preprocessing/           # Data cleaning and resampling
│   ├── feature_engineering/     # Feature extraction and engineering
│   ├── sequence_generation/     # Time series sequence creation
│   ├── model_architecture/      # LSTM model building
│   ├── training/               # Model training and validation
│   ├── evaluation/             # Clinical metrics and visualizations
│   └── model_persistence/      # Model saving and loading
├── models/                     # Saved models and checkpoints
├── reports/                    # Training reports and visualizations
├── data/                       # Raw and processed data
├── logs/                       # Pipeline logs
├── main.py                     # Main pipeline entry point
├── fast_training_setup.py      # TensorFlow optimization setup
└── simple_model_usage.py       # Example model usage
```

## 🔧 **Model Usage**

### **Load and Use Trained Models**

```python
# Example usage
python3 simple_model_usage.py
```

### **Model Input Requirements**

- **Shape**: `(batch_size, 12, 62)` - 12 time steps with 62 features
- **Features**: CGM, insulin, food, temporal, and statistical features
- **Output**: Glucose prediction in mg/dL

### **Model Performance**

- **MARD**: ~24% (Mean Absolute Relative Difference)
- **MAE**: ~5.8 mg/dL (Mean Absolute Error)
- **RMSE**: ~8.8 mg/dL (Root Mean Square Error)
- **Time-in-Range Accuracy**: ~94%

## 📈 **Pipeline Stages**

1. **Data Acquisition**: Download and load diabetes datasets
2. **Data Validation**: Quality checks and outlier detection
3. **Preprocessing**: Cleaning, resampling, and normalization
4. **Feature Engineering**: Extract 62 clinical and temporal features
5. **Sequence Generation**: Create time series sequences for LSTM
6. **Model Building**: Construct LSTM architecture
7. **Training**: Train model with early stopping and validation
8. **Evaluation**: Calculate clinical metrics and generate visualizations
9. **Model Persistence**: Save model with metadata and versioning

## 🎛️ **Configuration Options**

### **Speed vs. Quality Trade-offs**

| Setting         | Fast (Default) | Original  | Impact                         |
| --------------- | -------------- | --------- | ------------------------------ |
| Batch Size      | 128            | 32        | 4x faster, better convergence  |
| Sequence Length | 12             | 60        | 5x faster, may reduce accuracy |
| LSTM Units      | [64, 32]       | [128, 64] | 4x faster, smaller model       |
| Learning Rate   | 0.002          | 0.001     | Faster convergence             |
| Epochs          | 5              | 5         | Same, but faster per epoch     |

### **Key Configuration Sections**

- **Model**: Architecture parameters (LSTM units, dropout, etc.)
- **Training**: Batch size, epochs, validation split
- **Preprocessing**: Data cleaning and resampling settings
- **Feature Engineering**: Feature extraction options
- **Parallel Processing**: Multi-core processing settings

## 🚀 **Advanced Usage**

### **TensorFlow Optimizations**

```bash
# Setup optimizations before training
python3 fast_training_setup.py
```

### **Skip Pipeline Stages**

```yaml
# In config file
pipeline:
  skip_stages: ["evaluation", "visualization"] # Skip during development
```

### **Custom Model Architecture**

```yaml
# In config file
model:
  lstm_units: [96, 48] # Custom LSTM architecture
  dense_units: [48, 24] # Custom dense layers
  dropout_rate: 0.15 # Custom dropout
```

## 📊 **Results and Reports**

### **Training Reports**

- Location: `reports/training_report_*.txt`
- Contains: Training metrics, validation performance, timing statistics

### **Clinical Metrics**

- Location: `reports/metrics/`
- Contains: Clarke Error Grid, Parkes Error Grid, prediction scatter plots

### **Model Artifacts**

- Location: `models/versions/`
- Contains: Trained models, metadata, preprocessing components

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Memory Issues**: Reduce batch size or sequence length
2. **Slow Training**: Use fast configuration or enable GPU
3. **Poor Performance**: Increase epochs or adjust learning rate

### **Performance Tips**

- Use GPU if available (10-50x faster)
- Enable TensorFlow optimizations
- Use larger batch sizes when memory allows
- Skip evaluation during development

## 📚 **Documentation**

- **Model Usage Guide**: `MODEL_USAGE_GUIDE.md`
- **Fast Training Guide**: `FAST_TRAINING_GUIDE.md`
- **Configuration Reference**: See `configs/` directory

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ **Disclaimer**

This software is for research purposes only. It is not intended for clinical use or medical decision-making. Always consult with healthcare professionals for medical advice.
