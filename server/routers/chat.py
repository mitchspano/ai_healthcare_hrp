# server/routers/chat.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from server.models.schemas import ChatRequest, ChatResponse
from server.services.agent import run_agent_async
from server.services.model_service import diabetes_model_service
from server.services.conversation_service import conversation_service

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
        result = await run_agent_async(req.subject_id, req.text, req.conversation_id)
        print("👈 run_agent returned:", result)
        return ChatResponse(**result)
    except Exception as e:
        # Full traceback in console
        import traceback

        traceback.print_exc()
        raise HTTPException(500, f"Agent error: {e}")


@router.get("/conversations/{subject_id}")
async def list_conversations(subject_id: str):
    """List all conversations for a subject."""
    try:
        conversations = conversation_service.list_conversations(subject_id)
        return {
            "subject_id": subject_id,
            "conversations": [conv.to_dict() for conv in conversations],
        }
    except Exception as e:
        raise HTTPException(500, f"Error listing conversations: {e}")


@router.get("/conversations/{subject_id}/{conversation_id}")
async def get_conversation(subject_id: str, conversation_id: str):
    """Get a specific conversation."""
    try:
        conversation = conversation_service.get_conversation(conversation_id)
        if not conversation or conversation.subject_id != subject_id:
            raise HTTPException(404, "Conversation not found")
        return conversation.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error getting conversation: {e}")


@router.delete("/conversations/{subject_id}/{conversation_id}")
async def delete_conversation(subject_id: str, conversation_id: str):
    """Delete a conversation."""
    try:
        conversation = conversation_service.get_conversation(conversation_id)
        if not conversation or conversation.subject_id != subject_id:
            raise HTTPException(404, "Conversation not found")

        success = conversation_service.delete_conversation(conversation_id)
        if success:
            return {"message": "Conversation deleted successfully"}
        else:
            raise HTTPException(500, "Failed to delete conversation")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error deleting conversation: {e}")


@router.get("/conversations/stats")
async def get_conversation_stats():
    """Get conversation service statistics."""
    try:
        return conversation_service.get_stats()
    except Exception as e:
        raise HTTPException(500, f"Error getting stats: {e}")


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
