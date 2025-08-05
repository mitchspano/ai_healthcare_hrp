# server/routers/chat.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from server.models.schemas import ChatRequest, ChatResponse
from server.services.agent import run_agent_async
from server.services.model_service import diabetes_model_service

router = APIRouter()


class StructuredPredictionRequest(BaseModel):
    subject_id: str
    glucoseReadings: List[float]
    carbohydrates: List[Dict[str, Any]]
    insulinBolus: List[Dict[str, Any]]


class StructuredPredictionResponse(BaseModel):
    prediction: Optional[float]
    prediction_text: str
    confidence: Optional[float] = None
    model_info: Dict[str, Any]


@router.post("/", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    # Log for debugging
    print("▶️ /chat hit with:", req.dict())
    try:
        result = await run_agent_async(req.subject_id, req.text)
        print("👈 run_agent returned:", result)
        return ChatResponse(**result)
    except Exception as e:
        # Full traceback in console
        import traceback

        traceback.print_exc()
        raise HTTPException(500, f"Agent error: {e}")


@router.post("/predict", response_model=StructuredPredictionResponse)
async def structured_prediction_endpoint(req: StructuredPredictionRequest):
    """Make a structured prediction using the diabetes LSTM model."""
    print("▶️ /predict hit with:", req.dict())
    try:
        result = diabetes_model_service.predict_structured(req.dict())
        print("👈 structured prediction returned:", result)
        return StructuredPredictionResponse(**result)
    except Exception as e:
        # Full traceback in console
        import traceback

        traceback.print_exc()
        raise HTTPException(500, f"Prediction error: {e}")


@router.get("/model/status")
async def get_model_status():
    """Get the status of the diabetes LSTM model."""
    try:
        model_info = diabetes_model_service.get_model_info()
        return {"loaded": diabetes_model_service.is_loaded(), "model_info": model_info}
    except Exception as e:
        raise HTTPException(500, f"Model status error: {e}")


@router.get("/model/info")
async def get_model_info():
    """Get detailed information about the loaded model."""
    try:
        return diabetes_model_service.get_model_info()
    except Exception as e:
        raise HTTPException(500, f"Model info error: {e}")
