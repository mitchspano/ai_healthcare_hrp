# src/routers/chat.py

from fastapi import APIRouter, HTTPException
from src.models.schemas import ChatRequest, ChatResponse
from src.services.agent import run_agent_async

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
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Agent error: {e}")