# AI Diabetes Management Agent

A comprehensive AI system for Type 1 Diabetes management, featuring an LSTM neural network for glucose prediction and an interactive chat interface.

## Project Overview

This project consists of three main components:

- **🤖 AI Model** (`ai-model/`): LSTM neural network for glucose prediction
- **🔧 Backend Server** (`server/`): FastAPI server with chat functionality
- **🎨 Frontend UI** (`ui/`): React-based chat interface

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r ai-model/requirements.txt

# Install frontend dependencies
cd ui && npm install && cd ..
```

### 2. Train the AI Model

```bash
# Navigate to AI model directory
cd ai-model

# Train the model (fast configuration)
python3 main.py --run

# Or use original configuration for better quality
python3 main.py --config configs/original_config.yaml --run
```

### 3. Run the Application

```bash
# Start both servers (backend + frontend)
python3 start_clean.py
```

### 4. Access the Application

- **Frontend UI**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 5. Stop the Application

```bash
# Kill all server processes
python3 kill_servers.py
```

## Project Structure

```
ai_healthcare_hrp/
├── ai-model/                    # LSTM glucose prediction model
│   ├── main.py                 # Model training pipeline
│   ├── diabetes_lstm_pipeline/ # Core ML pipeline
│   ├── models/                 # Trained models
│   └── data/                   # Diabetes datasets
├── server/                     # FastAPI backend
│   ├── main_agent.py          # Server entry point
│   ├── routers/               # API endpoints
│   └── services/              # Business logic
├── ui/                        # React frontend
│   ├── src/                   # React components
│   └── package.json           # Frontend dependencies
├── start_clean.py             # Application startup script
└── kill_servers.py            # Application shutdown script
```

## Model Performance

- **MARD**: 18.60% (Mean Absolute Relative Difference)
- **MAE**: 10.87 mg/dL (Mean Absolute Error)
- **RMSE**: 16.70 mg/dL (Root Mean Square Error)
- **Time-in-Range Accuracy**: 51.13%

## Key Features

- **Glucose Prediction**: LSTM neural network predicts future glucose levels
- **Interactive Chat**: AI assistant provides personalized diabetes guidance
- **Real-time Analysis**: Processes CGM data and meal information
- **Child-Friendly Interface**: Designed for young diabetes patients

## Configuration Options

### Fast Training (Default)

```bash
python3 main.py --run
```

### High Quality Training

```bash
python3 main.py --config configs/original_config.yaml --run
```

## Troubleshooting

- **Port Conflicts**: The startup script automatically handles port conflicts
- **Memory Issues**: Reduce batch size in config if training fails
- **Slow Training**: Use GPU if available for 10-50x speed improvement

## Authors

- **Kerim Karabacak**
- **Mitch Spano**

## Disclaimer

This software is for research purposes only. It is not intended for clinical use or medical decision-making. Always consult with healthcare professionals for medical advice.

## Development Note

This project was generated with the assistance of Cursor, an AI-powered code editor.
