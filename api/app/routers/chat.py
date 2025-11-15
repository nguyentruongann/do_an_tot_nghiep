from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import json
from ..llm import chat_completion
from ..redis_store import log_event
from ..data_index import get_index

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "session_id": "s1",
            "question": "Tổng kết giúp mình các lead tiềm năng cao",
            "rows": [
                {"lead_id":"L-10","order_count":2,"revenue":500,"currency":"USD","notes":"quan tâm gói năm"}
            ]
        }
    })
    session_id: str = Field(default="s1")
    question: str = Field(default="xin chào")
    rows: Optional[List[Dict[str, Any]]] = Field(default=None)

class ChatResponse(BaseModel):
    reply: str
    sources: list
    priority_chain: list

@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest):
    idx = get_index()
    q = req.question or ""
    if req.rows:
        try:
            q = json.dumps(req.rows, ensure_ascii=False) + " || " + q
        except Exception:
            pass
    rows, grounded = idx.retrieve(req.session_id, q, k_each=3, k_total=9)
    priority = [layer.name for layer in idx.get_layers(req.session_id)]
    system_extra = "Chỉ trả lời dựa trên dữ liệu CSV theo thứ tự ưu tiên: " + " > ".join(priority) + ". Nếu thiếu dữ liệu, hãy nói rõ."
    preface = "Ngữ cảnh CSV (JSON):\n" + json.dumps(rows, ensure_ascii=False) + "\n\nHãy bám sát ngữ cảnh để trả lời."
    messages = [{"role":"user","content":preface},{"role":"user","content":req.question}]
    log_event(req.session_id, "chat_request", {"priority": priority, "rows_count": len(rows)})
    reply = chat_completion(messages, temperature=0.2, system_extra=system_extra)
    log_event(req.session_id, "chat_reply", {"reply": reply})
    return ChatResponse(reply=reply, sources=rows[:5], priority_chain=priority)
