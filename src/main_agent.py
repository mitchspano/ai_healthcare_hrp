# src/main_agent.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.routers.chat import router as chat_router

app = FastAPI(title="T1D Chat MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

# mount at /chat/
app.include_router(chat_router, prefix="/chat")

@app.get("/ping")
async def ping():
    return {"ping": "pong"}

# mount the /chat router
app.include_router(chat_router, prefix="/chat")