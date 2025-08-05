# src/services/tools.py

from typing import Dict, Any
import numpy as np
import pandas as pd
from .model_service import diabetes_model_service


def get_latest_metrics(subject_id: str, window_mins: int = 60) -> Dict[str, Any]:
    """
    Stub for fetching BG metrics.
    Later: hook into PatientSim.latest_hour() or your metrics API.
    """
    return {
        "subject_id": subject_id,
        "window_mins": window_mins,
        "avg_bg": 140.2,
        "time_in_range_pct": 72.5,
        "sensor_on_pct": 99.1,
    }


def send_alert(subject_id: str, message: str, severity: str = "info") -> Dict[str, Any]:
    """
    Stub for pushing a proactive alert.
    Later: broadcast via SSE/WebSocket to the front-end.
    """
    print(f"[ALERT::{severity.upper()}] {subject_id} → {message}")
    return {"status": "sent", "subject_id": subject_id, "severity": severity}


def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded diabetes LSTM model.
    """
    return diabetes_model_service.get_model_info()


def predict_glucose_levels(
    subject_id: str, historical_data: str = None
) -> Dict[str, Any]:
    """
    Predict glucose levels using the diabetes LSTM model.

    Args:
        subject_id: The subject ID to predict for
        historical_data: Optional historical data (for now, uses mock data)
    """
    try:
        if not diabetes_model_service.is_loaded():
            return {
                "error": "Model not loaded",
                "subject_id": subject_id,
                "prediction": None,
            }

        # For now, generate mock input data based on the model's expected input shape
        # In a real implementation, you would load actual historical data for the subject
        model_info = diabetes_model_service.get_model_info()
        input_shape = model_info.get("input_shape", "Unknown")

        if input_shape == "Unknown":
            # Default shape based on the model architecture
            mock_input = np.random.randn(
                1, 12, 62
            )  # batch_size=1, sequence_length=12, features=62
        else:
            # Use the actual input shape from the model
            # input_shape is [12, 62], so we need to add batch dimension
            mock_input = np.random.randn(
                1, *input_shape
            )  # batch_size=1, sequence_length=12, features=62

        # Make prediction
        prediction = diabetes_model_service.predict_glucose(mock_input)

        return {
            "subject_id": subject_id,
            "prediction": float(prediction[0][0]),  # Extract single prediction value
            "model_info": model_info,
            "input_shape_used": mock_input.shape,
            "note": "Using mock input data - replace with actual historical data",
        }

    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "subject_id": subject_id,
            "prediction": None,
        }
