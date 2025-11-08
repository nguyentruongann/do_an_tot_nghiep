from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from ..llm import chat_completion
from ..redis_store import log_event

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatTurn(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    messages: List[ChatTurn]

class ChatResponse(BaseModel):
    reply: str

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    log_event(req.session_id, "chat_request", {"messages": [m.model_dump() for m in req.messages]})
    reply = chat_completion([m.model_dump() for m in req.messages])
    log_event(req.session_id, "chat_reply", {"reply": reply})
    return ChatResponse(reply=reply)
