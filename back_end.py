# back_end.py  (rename if you like)
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="T1D Chat MVP")

# CORS so localhost:5173 can call localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    subject_id: str
    text: str

@app.post("/chat")
async def chat(req: ChatRequest):
    return {"reply": f"(echo) You said: {req.text}"}
