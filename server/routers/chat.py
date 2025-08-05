# server/routers/chat.py

from fastapi import APIRouter, HTTPException
from server.models.schemas import ChatRequest, ChatResponse
from server.services.agent import run_agent_async
from server.services.model_service import diabetes_model_service

router = APIRouter()


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
