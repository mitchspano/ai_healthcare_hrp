# src/services/tools.py

from typing import Dict, Any

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