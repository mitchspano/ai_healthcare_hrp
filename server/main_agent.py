# server/main_agent.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.config import settings
from server.routers.chat import router as chat_router

app = FastAPI(title="T1D Chat MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "file://",  # Allow file:// protocol for testing
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# mount at /chat/
app.include_router(chat_router, prefix="/chat")


@app.get("/ping")
async def ping():
    return {"ping": "pong"}


# mount the /chat router
app.include_router(chat_router, prefix="/chat")
